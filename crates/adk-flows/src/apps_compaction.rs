//! Capabilities C0288 (partial)/C0289: pure dedup and token-estimation
//! logic ported from `apps/compaction.py` — a distinct source module
//! from this crate's sibling [`crate::compaction`]
//! (`flows/llm_flows/_content_compaction.py`, C0185), which happens to
//! need the same subsumption-detection shape for a different purpose
//! (deciding which raw events to filter out of a request's contents,
//! not which sliding-window range to summarize next).
//!
//! **Scope, batch 1 (C0288 partial/C0289)**: the pure dedup logic
//! (`_valid_compactions`/`_is_compaction_subsumed`/
//! `_latest_compaction_event`) and token-count estimation
//! (`_estimate_prompt_token_count`/`_latest_prompt_token_count`). C0288's
//! OTel-traced summarization wrapper (`_summarize_events_with_trace`)
//! needs span/tracer machinery this port hasn't adopted (see
//! `adk-agents::telemetry_context`'s own disclosed scope) — nothing here
//! wraps the summarizer call in a span.
//!
//! **Scope, batch 2 (C0290/C0291/C0292)**: `_ensure_compaction_summarizer`
//! (C0290), `_events_to_compact_for_token_threshold`/
//! `_longest_self_contained_prefix` (C0291), and
//! `_safe_token_compaction_split_index` (C0292). `agent.canonical_model`
//! maps to [`crate::llm_flow::LlmFlow::model`] — already resolved once
//! at construction (the established memoization this crate's own
//! `LlmFlow`/`canonical_model.rs` disclose) — reached via the same
//! `agent.as_any().downcast_ref::<LlmFlow>()` pattern
//! `instructions.rs`'s `resolve_root_global_instruction` already
//! established for recovering a concrete `LlmAgent`-backed behavior
//! from a type-erased `BaseAgent`. Disclosed: the source mutates
//! `config.summarizer` in place; this port's `EventsCompactionConfig`
//! has no interior mutability, so [`ensure_compaction_summarizer`]
//! resolves and returns instead, leaving in-place caching (if wanted)
//! to whatever wires C0293.
//!
//! **Scope, batch 3 (C0293/C0871/C0872)**: `_run_compaction_for_token_threshold`/
//! `_run_compaction_for_token_threshold_config` and
//! `_run_compaction_for_sliding_window` — the two trigger entrypoints,
//! narrowed to take `agent`/`config`/raw session events directly (no
//! `App`, matching this file's other functions) and to return
//! `Option<Event>` (the source's `AsyncGenerator[Event, None]` never
//! yields more than one event, so there's nothing a stream buys here).
//! Neither performs the actual `session_service.append_event` call
//! itself — the caller (`Runner::run_async_with_config`) does, the same
//! "compute, caller applies" split `Runner::rewind_async` (C0891)
//! already established. `_summarize_events_with_trace`'s OTel span is
//! stripped for the same reason batch 1 disclosed for C0288 — what's
//! left is just `config.summarizer.maybe_summarize_events(..)`.
//!
//! **Adaptation, disclosed**: the source's `_count_chars_in_content`
//! `json.dumps`s a function call's `args`/a function response's
//! `response`, falling back to `str()` on a serialization failure. This
//! port's `args`/`response` are already-typed `BTreeMap<String, Value>`
//! (always JSON-serializable), so only the `json.dumps` path applies —
//! there's no failure case to fall back from.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use adk_agents::app_configs::{BaseEventsSummarizer, EventsCompactionConfig};
use adk_agents::base_agent::BaseAgent;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_genai::content::Content;
use adk_tools::llm_event_summarizer::LlmEventSummarizer;
use rusty_serde::value::Value;

use crate::contents::{get_contents, ContentsError};

/// `_count_chars_in_content` — character count across a content's parts:
/// text length, plus a function call's name + serialized args, plus a
/// function response's name + serialized response.
fn count_chars_in_content(content: Option<&Content>) -> usize {
    let Some(content) = content else {
        return 0;
    };
    let mut total = 0;
    for part in &content.parts {
        if let Some(text) = &part.text {
            total += text.chars().count();
        }
        if let Some(function_call) = &part.function_call {
            total += function_call
                .name
                .as_deref()
                .unwrap_or_default()
                .chars()
                .count();
            if let Some(args) = &function_call.args {
                total += rusty_serde::json::to_string(args)
                    .map(|s| s.chars().count())
                    .unwrap_or(0);
            }
        }
        if let Some(function_response) = &part.function_response {
            total += function_response
                .name
                .as_deref()
                .unwrap_or_default()
                .chars()
                .count();
            if let Some(response) = &function_response.response {
                total += rusty_serde::json::to_string(response)
                    .map(|s| s.chars().count())
                    .unwrap_or(0);
            }
        }
    }
    total
}

/// `_valid_compactions` — `(index, start_timestamp, end_timestamp)` for
/// every event carrying a compaction range. `EventCompaction`'s fields
/// are all non-optional in this port (see `crate::compaction`'s own
/// module doc for the same already-disclosed narrowing), so unlike the
/// source there's no per-field `is None` check to perform — every
/// `actions.compaction: Some(_)` is already valid.
fn valid_compactions(events: &[Event]) -> Vec<(usize, f64, f64)> {
    events
        .iter()
        .enumerate()
        .filter_map(|(i, event)| {
            event
                .actions
                .compaction
                .as_ref()
                .map(|c| (i, c.start_timestamp, c.end_timestamp))
        })
        .collect()
}

/// `_is_compaction_subsumed` — true if a compaction range is fully
/// contained by another. If two compactions have identical ranges, the
/// earlier event is treated as subsumed by the later one.
fn is_compaction_subsumed(
    start_timestamp: f64,
    end_timestamp: f64,
    event_index: usize,
    compactions: &[(usize, f64, f64)],
) -> bool {
    compactions
        .iter()
        .any(|&(other_index, other_start, other_end)| {
            other_index != event_index
                && other_start <= start_timestamp
                && other_end >= end_timestamp
                && (other_start < start_timestamp
                    || other_end > end_timestamp
                    || other_index > event_index)
        })
}

/// C0288: `_latest_compaction_event` — the latest non-subsumed
/// compaction event by stream order.
pub fn latest_compaction_event(events: &[Event]) -> Option<&Event> {
    let compactions = valid_compactions(events);
    let mut latest: Option<usize> = None;
    for &(event_index, start_ts, end_ts) in &compactions {
        if is_compaction_subsumed(start_ts, end_ts, event_index, &compactions) {
            continue;
        }
        if latest.is_none_or(|latest_index| event_index > latest_index) {
            latest = Some(event_index);
        }
    }
    latest.map(|index| &events[index])
}

/// `_latest_compaction_end_timestamp` — the end timestamp of the most
/// recent (non-subsumed) compaction event, or `0.0` if there isn't one.
/// Needed by [`events_to_compact_for_token_threshold`] (C0291); not its
/// own manifest row (a small helper built directly on
/// [`latest_compaction_event`], C0288, already DONE).
fn latest_compaction_end_timestamp(events: &[Event]) -> f64 {
    latest_compaction_event(events)
        .and_then(|event| event.actions.compaction.as_ref())
        .map(|compaction| compaction.end_timestamp)
        .unwrap_or(0.0)
}

#[derive(Debug, rusty_err::Error)]
pub enum EnsureCompactionSummarizerError {
    #[error("No LlmAgent model available for event compaction summarizer.")]
    NotAnLlmAgent,
}

/// C0290: `_ensure_compaction_summarizer` — resolves an
/// `EventsCompactionConfig`'s summarizer: the existing one if already
/// set, otherwise a new `LlmEventSummarizer` built from `agent`'s
/// already-resolved canonical model. See the module doc for the
/// disclosed resolve-and-return adaptation (no in-place mutation).
pub fn ensure_compaction_summarizer(
    config: &EventsCompactionConfig,
    agent: &BaseAgent,
) -> Result<Arc<dyn BaseEventsSummarizer>, EnsureCompactionSummarizerError> {
    if let Some(summarizer) = &config.summarizer {
        return Ok(summarizer.clone());
    }
    let llm_flow = agent
        .as_any()
        .downcast_ref::<crate::llm_flow::LlmFlow>()
        .ok_or(EnsureCompactionSummarizerError::NotAnLlmAgent)?;
    Ok(Arc::new(LlmEventSummarizer::new(llm_flow.model.clone())))
}

/// `_event_function_call_ids` — the ids of function calls carried by an
/// event.
fn event_function_call_ids(event: &Event) -> HashSet<String> {
    event
        .get_function_calls()
        .into_iter()
        .filter_map(|call| call.id.clone())
        .collect()
}

/// `_event_function_response_ids` — the ids of function responses
/// carried by an event.
fn event_function_response_ids(event: &Event) -> HashSet<String> {
    event
        .get_function_responses()
        .into_iter()
        .filter_map(|response| response.id.clone())
        .collect()
}

/// C0291: `_longest_self_contained_prefix` — the longest prefix of
/// `events` safe to compact. A single left-to-right pass tracks "open"
/// obligations keyed by call id: a function call, a requested tool
/// confirmation, or a requested auth config opens one; a function
/// response with the same id closes it. Responses are applied before
/// opens within each event so a response only closes an obligation
/// opened by an earlier event. The prefix is safe to summarize only at
/// points where no obligation is open, so the longest prefix ending at
/// such a balanced point is returned (empty if the window never reaches
/// a balanced point).
pub fn longest_self_contained_prefix(events: &[Event]) -> Vec<Event> {
    let mut open_ids: HashSet<String> = HashSet::new();
    let mut safe_length = 0;
    for (index, event) in events.iter().enumerate() {
        for id in event_function_response_ids(event) {
            open_ids.remove(&id);
        }
        open_ids.extend(event_function_call_ids(event));
        open_ids.extend(event.actions.requested_tool_confirmations.keys().cloned());
        open_ids.extend(event.actions.requested_auth_configs.keys().cloned());
        if open_ids.is_empty() {
            safe_length = index + 1;
        }
    }
    events[..safe_length].to_vec()
}

/// C0292: `_safe_token_compaction_split_index` — a split index that
/// avoids orphaning retained tool responses. Retained events (the tail
/// of `candidate_events`) may contain function responses; if their
/// matching function call events would fall in the compacted prefix,
/// contents assembly can fail — so the split shifts earlier so matching
/// call/response pairs stay together. Iterates backwards once,
/// maintaining a running set of unmatched response ids; the latest
/// valid split point where no unmatched responses remain is returned.
pub fn safe_token_compaction_split_index(
    candidate_events: &[Event],
    event_retention_size: i64,
) -> usize {
    let initial_split = candidate_events.len() as i64 - event_retention_size;
    if initial_split <= 0 {
        return 0;
    }
    let initial_split = initial_split as usize;

    let mut unmatched_response_ids: HashSet<String> = HashSet::new();
    let mut best_split = 0;

    for i in (0..candidate_events.len()).rev() {
        let event = &candidate_events[i];
        unmatched_response_ids.extend(event_function_response_ids(event));
        for call_id in event_function_call_ids(event) {
            unmatched_response_ids.remove(&call_id);
        }
        if unmatched_response_ids.is_empty() && i <= initial_split {
            best_split = i;
            break;
        }
    }
    best_split
}

/// C0291: `_events_to_compact_for_token_threshold` — collects
/// token-threshold compaction candidates: events since the last
/// compaction, safely split to retain `event_retention_size` trailing
/// events (via [`safe_token_compaction_split_index`]) and trimmed to
/// the longest self-contained prefix (via
/// [`longest_self_contained_prefix`]). If a previous compaction exists,
/// its summary is prepended as a synthetic leading event so the next
/// summary can supersede it.
pub fn events_to_compact_for_token_threshold(
    events: &[Event],
    event_retention_size: i64,
) -> Vec<Event> {
    let latest_compaction = latest_compaction_event(events).cloned();
    let last_compacted_end_timestamp = latest_compaction_end_timestamp(events);

    let candidate_events: Vec<Event> = events
        .iter()
        .filter(|event| {
            event.actions.compaction.is_none() && event.timestamp > last_compacted_end_timestamp
        })
        .cloned()
        .collect();

    if candidate_events.len() as i64 <= event_retention_size {
        return Vec::new();
    }

    let events_to_compact = if event_retention_size == 0 {
        candidate_events
    } else {
        let split_index =
            safe_token_compaction_split_index(&candidate_events, event_retention_size);
        candidate_events[..split_index].to_vec()
    };

    let events_to_compact = longest_self_contained_prefix(&events_to_compact);
    if events_to_compact.is_empty() {
        return Vec::new();
    }

    if let Some(compaction) = latest_compaction
        .as_ref()
        .and_then(|e| e.actions.compaction.as_ref())
    {
        let mut seed_event = Event::new(Event::new_id(), "model", NodeInfo::new(""));
        seed_event.timestamp = compaction.start_timestamp;
        seed_event.content = Some(compaction.compacted_content.clone());
        seed_event.branch = latest_compaction.as_ref().and_then(|e| e.branch.clone());
        let mut result = vec![seed_event];
        result.extend(events_to_compact);
        return result;
    }

    events_to_compact
}

/// C0289: `_estimate_prompt_token_count` — an approximate prompt token
/// count from session events, mirroring the effective content-building
/// path used by the contents request processor. Roughly 4 characters
/// per token; `None` when there's nothing to estimate from.
pub fn estimate_prompt_token_count(
    events: &[Event],
    current_branch: Option<&str>,
    agent_name: &str,
) -> Result<Option<i64>, ContentsError> {
    let effective_contents = get_contents(
        current_branch,
        events,
        agent_name,
        false,
        None,
        false,
        None,
        false,
    )?;
    let total_chars: usize = effective_contents
        .iter()
        .map(|content| count_chars_in_content(Some(content)))
        .sum();
    if total_chars == 0 {
        return Ok(None);
    }
    Ok(Some((total_chars / 4) as i64))
}

/// C0289: `_latest_prompt_token_count` — the most recently observed
/// prompt token count, preferring a real `usage_metadata.promptTokenCount`
/// (read out of the opaque `Value`, the same `"promptTokenCount"`-key
/// convention `cache_performance_analyzer.rs`/`context_cache.rs`
/// already established) over the estimate.
pub fn latest_prompt_token_count(
    events: &[Event],
    current_branch: Option<&str>,
    agent_name: &str,
) -> Result<Option<i64>, ContentsError> {
    for event in events.iter().rev() {
        if let Some(usage) = &event.usage_metadata {
            if let Some(prompt_token_count) = usage.get("promptTokenCount").and_then(Value::as_i64)
            {
                return Ok(Some(prompt_token_count));
            }
        }
    }
    estimate_prompt_token_count(events, current_branch, agent_name)
}

#[derive(Debug, rusty_err::Error)]
pub enum CompactionTriggerError {
    #[error("{0}")]
    Contents(ContentsError),
    #[error("{0}")]
    Summarizer(EnsureCompactionSummarizerError),
}

impl From<ContentsError> for CompactionTriggerError {
    fn from(err: ContentsError) -> Self {
        CompactionTriggerError::Contents(err)
    }
}

impl From<EnsureCompactionSummarizerError> for CompactionTriggerError {
    fn from(err: EnsureCompactionSummarizerError) -> Self {
        CompactionTriggerError::Summarizer(err)
    }
}

/// C0293: `_run_compaction_for_token_threshold`/
/// `_run_compaction_for_token_threshold_config` — checks whether
/// token-threshold compaction is fully configured and triggered by the
/// latest observed/estimated prompt token count, and if so, generates a
/// compaction event summarizing the retention-window candidates.
/// `agent_name`/`current_branch` are hardcoded to `""`/`None`, matching
/// the source's own `App`-wrapper call site.
pub async fn run_compaction_for_token_threshold(
    config: &EventsCompactionConfig,
    agent: &BaseAgent,
    session_events: &[Event],
) -> Result<Option<Event>, CompactionTriggerError> {
    let (Some(token_threshold), Some(event_retention_size)) =
        (config.token_threshold, config.event_retention_size)
    else {
        return Ok(None);
    };

    let events = adk_events::rewind::apply_rewinds(session_events);

    let Some(prompt_token_count) = latest_prompt_token_count(&events, None, "")? else {
        return Ok(None);
    };
    if prompt_token_count < token_threshold {
        return Ok(None);
    }

    let events_to_compact = events_to_compact_for_token_threshold(&events, event_retention_size);
    if events_to_compact.is_empty() {
        return Ok(None);
    }

    let summarizer = ensure_compaction_summarizer(config, agent)?;
    Ok(summarizer.maybe_summarize_events(&events_to_compact).await)
}

/// C0293: `_run_compaction_for_sliding_window` — the interval-based
/// trigger. Prefers token-threshold compaction if configured and
/// triggered (unless `skip_token_compaction`); otherwise checks whether
/// enough new invocations have completed since the last compaction
/// and, if so, selects the invocation-id range to compact (from
/// `overlap_size` invocations before the new block, through the last of
/// the new block) and generates a compaction event.
pub async fn run_compaction_for_sliding_window(
    config: &EventsCompactionConfig,
    agent: &BaseAgent,
    session_events: &[Event],
    skip_token_compaction: bool,
) -> Result<Option<Event>, CompactionTriggerError> {
    let events = adk_events::rewind::apply_rewinds(session_events);
    if events.is_empty() {
        return Ok(None);
    }

    let has_token_threshold_config =
        config.token_threshold.is_some() && config.event_retention_size.is_some();
    if !skip_token_compaction && has_token_threshold_config {
        if let Some(event) =
            run_compaction_for_token_threshold(config, agent, session_events).await?
        {
            return Ok(Some(event));
        }
    }

    let (Some(compaction_interval), Some(overlap_size)) =
        (config.compaction_interval, config.overlap_size)
    else {
        return Ok(None);
    };

    let last_compacted_end_timestamp = events
        .iter()
        .rev()
        .find_map(|event| event.actions.compaction.as_ref().map(|c| c.end_timestamp))
        .unwrap_or(0.0);

    let mut unique_invocation_ids: Vec<String> = Vec::new();
    let mut invocation_latest_timestamps: HashMap<String, f64> = HashMap::new();
    for event in &events {
        if event.invocation_id.is_empty() || event.actions.compaction.is_some() {
            continue;
        }
        let latest = invocation_latest_timestamps
            .entry(event.invocation_id.clone())
            .or_insert_with(|| {
                unique_invocation_ids.push(event.invocation_id.clone());
                0.0
            });
        if event.timestamp > *latest {
            *latest = event.timestamp;
        }
    }

    let new_invocation_ids: Vec<&String> = unique_invocation_ids
        .iter()
        .filter(|id| invocation_latest_timestamps[*id] > last_compacted_end_timestamp)
        .collect();

    if (new_invocation_ids.len() as i64) < compaction_interval {
        return Ok(None);
    }

    let end_inv_id = *new_invocation_ids.last().unwrap();
    let first_new_inv_id = new_invocation_ids[0];
    let first_new_inv_idx = unique_invocation_ids
        .iter()
        .position(|id| id == first_new_inv_id)
        .unwrap();

    let start_idx = (first_new_inv_idx as i64 - overlap_size).max(0) as usize;
    let start_inv_id = &unique_invocation_ids[start_idx];

    let last_event_idx = events
        .iter()
        .rposition(|event| &event.invocation_id == end_inv_id);

    let mut events_to_compact: Vec<Event> = Vec::new();
    if let Some(last_event_idx) = last_event_idx {
        if let Some(first_event_start_inv_idx) = events
            .iter()
            .position(|event| &event.invocation_id == start_inv_id)
        {
            events_to_compact = events[first_event_start_inv_idx..=last_event_idx]
                .iter()
                .filter(|event| event.actions.compaction.is_none())
                .cloned()
                .collect();
            events_to_compact = longest_self_contained_prefix(&events_to_compact);
        }
    }

    if events_to_compact.is_empty() {
        return Ok(None);
    }

    let summarizer = ensure_compaction_summarizer(config, agent)?;
    Ok(summarizer.maybe_summarize_events(&events_to_compact).await)
}

/// The real [`adk_runners::runner::CompactionTrigger`] implementation —
/// wires [`run_compaction_for_sliding_window`] into a `Runner`. Errors
/// (a `ContentsError` from prompt-token estimation, or a
/// `NotAnLlmAgent` summarizer-resolution error) are swallowed as "no
/// compaction this round," matching this operation's best-effort
/// nature and this port's established silent-degradation posture where
/// no error/logging channel exists to report through.
pub struct RealCompactionTrigger;

impl adk_runners::runner::CompactionTrigger for RealCompactionTrigger {
    fn run<'a>(
        &'a self,
        config: &'a EventsCompactionConfig,
        agent: &'a BaseAgent,
        session_events: &'a [Event],
        skip_token_compaction: bool,
    ) -> adk_agents::services::BoxFuture<'a, Option<Event>> {
        Box::pin(async move {
            run_compaction_for_sliding_window(config, agent, session_events, skip_token_compaction)
                .await
                .ok()
                .flatten()
        })
    }
}

/// Attaches [`RealCompactionTrigger`] to `runner` — the natural call
/// site is right after [`adk_runners::runner::Runner::from_app`],
/// since only an `App`-sourced `Runner` ever has an
/// `events_compaction_config` to act on.
pub fn with_real_compaction_trigger(
    runner: adk_runners::runner::Runner,
) -> adk_runners::runner::Runner {
    runner.with_compaction_trigger(Arc::new(RealCompactionTrigger))
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::event_compaction::EventCompaction;
    use adk_events::node_info::NodeInfo;
    use adk_events::EventActions;
    use adk_genai::content::{FunctionCall, FunctionResponse, Part};

    fn event_with_compaction(index: usize, start: f64, end: f64) -> Event {
        let mut event = Event::new(format!("inv-{index}"), "user", NodeInfo::new(""));
        event.actions = EventActions {
            compaction: Some(EventCompaction {
                start_timestamp: start,
                end_timestamp: end,
                compacted_content: Content::user_text("summary"),
            }),
            ..Default::default()
        };
        event
    }

    #[test]
    fn latest_compaction_event_returns_none_without_any_compaction() {
        let events = vec![Event::new("inv-1", "user", NodeInfo::new(""))];
        assert!(latest_compaction_event(&events).is_none());
    }

    #[test]
    fn latest_compaction_event_picks_the_only_compaction() {
        let events = vec![event_with_compaction(0, 1.0, 2.0)];
        let latest = latest_compaction_event(&events).unwrap();
        assert_eq!(
            latest.actions.compaction.as_ref().unwrap().start_timestamp,
            1.0
        );
    }

    #[test]
    fn latest_compaction_event_ignores_a_range_fully_subsumed_by_another() {
        // A narrower, earlier range (1..2) contained by a wider, later
        // one (0..4) is subsumed and skipped.
        let events = vec![
            event_with_compaction(0, 1.0, 2.0),
            event_with_compaction(1, 0.0, 4.0),
        ];
        let latest = latest_compaction_event(&events).unwrap();
        assert_eq!(
            latest.actions.compaction.as_ref().unwrap().start_timestamp,
            0.0
        );
    }

    #[test]
    fn latest_compaction_event_breaks_an_identical_range_tie_toward_the_later_event() {
        let events = vec![
            event_with_compaction(0, 1.0, 2.0),
            event_with_compaction(1, 1.0, 2.0),
        ];
        let latest = latest_compaction_event(&events).unwrap();
        // Both cover 1..2; the later event (index 1) wins the tie, so
        // it's the one NOT treated as subsumed.
        assert_eq!(latest.invocation_id, "inv-1");
    }

    #[test]
    fn latest_compaction_event_picks_the_later_of_two_non_overlapping_ranges() {
        let events = vec![
            event_with_compaction(0, 0.0, 1.0),
            event_with_compaction(1, 2.0, 3.0),
        ];
        let latest = latest_compaction_event(&events).unwrap();
        assert_eq!(
            latest.actions.compaction.as_ref().unwrap().start_timestamp,
            2.0
        );
    }

    fn text_event(author: &str, text: &str) -> Event {
        let mut event = Event::new("inv-1", author, NodeInfo::new(""));
        event.content = Some(Content::new(author, vec![Part::text(text)]));
        event
    }

    #[test]
    fn estimate_prompt_token_count_is_none_for_no_events() {
        assert_eq!(estimate_prompt_token_count(&[], None, "").unwrap(), None);
    }

    #[test]
    fn estimate_prompt_token_count_divides_total_chars_by_four() {
        // 16 chars ("a".repeat(16)) / 4 = 4 tokens.
        let events = vec![text_event("user", &"a".repeat(16))];
        assert_eq!(
            estimate_prompt_token_count(&events, None, "").unwrap(),
            Some(4)
        );
    }

    #[test]
    fn estimate_prompt_token_count_counts_function_call_and_response_text() {
        let mut call_event = Event::new("inv-1", "model", NodeInfo::new(""));
        call_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));
        let mut response_event = Event::new("inv-1", "user", NodeInfo::new(""));
        response_event.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));
        let count = estimate_prompt_token_count(&[call_event, response_event], None, "").unwrap();
        assert!(count.unwrap() > 0);
    }

    #[test]
    fn latest_prompt_token_count_prefers_real_usage_metadata_over_the_estimate() {
        let mut event = text_event("user", &"a".repeat(4000));
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(42),
        )]));
        assert_eq!(
            latest_prompt_token_count(&[event], None, "").unwrap(),
            Some(42)
        );
    }

    #[test]
    fn latest_prompt_token_count_falls_back_to_the_estimate_without_usage_metadata() {
        let events = vec![text_event("user", &"a".repeat(16))];
        assert_eq!(
            latest_prompt_token_count(&events, None, "").unwrap(),
            Some(4)
        );
    }

    #[test]
    fn latest_prompt_token_count_scans_from_the_most_recent_event_backward() {
        let mut older = text_event("user", "hi");
        older.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(10),
        )]));
        let mut newer = text_event("user", "hi again");
        newer.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(20),
        )]));
        assert_eq!(
            latest_prompt_token_count(&[older, newer], None, "").unwrap(),
            Some(20)
        );
    }

    // --- ensure_compaction_summarizer ---

    fn llm_flow_agent() -> BaseAgent {
        let llm_agent = adk_agents::llm_agent::LlmAgent::new(
            adk_agents::llm_agent::ModelRef::Name("gemini-2.0-flash".to_string()),
        );
        let llm_flow = crate::llm_flow::LlmFlow::with_model(
            llm_agent,
            Arc::new(adk_models::gemini::Gemini::new("gemini-2.0-flash")),
        );
        BaseAgent::new("root", llm_flow).unwrap()
    }

    #[test]
    fn ensure_compaction_summarizer_returns_the_existing_summarizer_unchanged() {
        struct StubSummarizer;
        impl BaseEventsSummarizer for StubSummarizer {
            fn maybe_summarize_events<'a>(
                &'a self,
                _events: &'a [Event],
            ) -> adk_agents::services::BoxFuture<'a, Option<Event>> {
                Box::pin(async { None })
            }
        }
        let config = EventsCompactionConfig {
            summarizer: Some(Arc::new(StubSummarizer)),
            compaction_interval: Some(1),
            overlap_size: Some(0),
            ..Default::default()
        };
        let agent = llm_flow_agent();
        assert!(ensure_compaction_summarizer(&config, &agent).is_ok());
    }

    #[test]
    fn ensure_compaction_summarizer_builds_one_from_the_agents_canonical_model() {
        let config = EventsCompactionConfig::default();
        let agent = llm_flow_agent();
        assert!(ensure_compaction_summarizer(&config, &agent).is_ok());
    }

    #[test]
    fn ensure_compaction_summarizer_errors_when_the_agent_isnt_llm_backed() {
        let config = EventsCompactionConfig::default();
        let agent = BaseAgent::new("root", adk_agents::base_agent::NoopBehavior).unwrap();
        match ensure_compaction_summarizer(&config, &agent) {
            Err(EnsureCompactionSummarizerError::NotAnLlmAgent) => {}
            Ok(_) => panic!("expected a NotAnLlmAgent error"),
        }
    }

    // --- longest_self_contained_prefix ---

    fn call_event(index: usize, call_id: &str) -> Event {
        let mut event = Event::new(format!("inv-{index}"), "model", NodeInfo::new(""));
        event.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                id: Some(call_id.to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));
        event
    }

    fn response_event(index: usize, call_id: &str) -> Event {
        let mut event = Event::new(format!("inv-{index}"), "user", NodeInfo::new(""));
        event.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some(call_id.to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));
        event
    }

    #[test]
    fn longest_self_contained_prefix_keeps_a_fully_matched_call_and_response() {
        let events = vec![call_event(0, "c1"), response_event(1, "c1")];
        let prefix = longest_self_contained_prefix(&events);
        assert_eq!(prefix.len(), 2);
    }

    #[test]
    fn longest_self_contained_prefix_drops_a_trailing_unmatched_call() {
        let events = vec![
            call_event(0, "c1"),
            response_event(1, "c1"),
            call_event(2, "c2"),
        ];
        let prefix = longest_self_contained_prefix(&events);
        assert_eq!(prefix.len(), 2);
    }

    #[test]
    fn longest_self_contained_prefix_is_empty_when_never_balanced() {
        let events = vec![call_event(0, "c1")];
        assert!(longest_self_contained_prefix(&events).is_empty());
    }

    #[test]
    fn longest_self_contained_prefix_treats_a_requested_tool_confirmation_as_open() {
        let mut event = Event::new("inv-1", "model", NodeInfo::new(""));
        event.actions.requested_tool_confirmations = [("fc-1".to_string(), Value::Bool(true))]
            .into_iter()
            .collect();
        assert!(longest_self_contained_prefix(&[event]).is_empty());
    }

    // --- safe_token_compaction_split_index ---

    #[test]
    fn safe_token_compaction_split_index_is_zero_when_retention_covers_everything() {
        let events = vec![text_event("user", "a"), text_event("user", "b")];
        assert_eq!(safe_token_compaction_split_index(&events, 5), 0);
    }

    #[test]
    fn safe_token_compaction_split_index_uses_the_initial_split_with_no_dangling_response() {
        let events: Vec<Event> = (0..5)
            .map(|i| text_event("user", &format!("e{i}")))
            .collect();
        // len=5, retention=2 -> initial split at 3.
        assert_eq!(safe_token_compaction_split_index(&events, 2), 3);
    }

    #[test]
    fn safe_token_compaction_split_index_shifts_earlier_to_keep_a_call_with_its_response() {
        // Retained tail (retention=2) would be [response(c1), text] --
        // the matching call is just before it, so the split must shift
        // earlier to keep call+response together.
        let events = vec![
            text_event("user", "e0"),
            call_event(1, "c1"),
            response_event(2, "c1"),
            text_event("user", "e3"),
        ];
        let split = safe_token_compaction_split_index(&events, 2);
        // The response at index 2 must not be orphaned from its call at
        // index 1, so the split must land at or before index 1.
        assert!(split <= 1);
    }

    // --- events_to_compact_for_token_threshold ---

    #[test]
    fn events_to_compact_for_token_threshold_is_empty_when_too_few_candidates() {
        let events = vec![text_event("user", "a"), text_event("user", "b")];
        assert!(events_to_compact_for_token_threshold(&events, 5).is_empty());
    }

    #[test]
    fn events_to_compact_for_token_threshold_takes_all_candidates_with_zero_retention() {
        let events = vec![text_event("user", "a"), text_event("user", "b")];
        let compacted = events_to_compact_for_token_threshold(&events, 0);
        assert_eq!(compacted.len(), 2);
    }

    #[test]
    fn events_to_compact_for_token_threshold_seeds_with_the_prior_compaction_summary() {
        let mut events = vec![event_with_compaction(0, 0.0, 1.0)];
        for i in 1..5 {
            let mut event = text_event("user", &format!("e{i}"));
            event.timestamp = i as f64;
            events.push(event);
        }
        let compacted = events_to_compact_for_token_threshold(&events, 0);
        // First event is the synthetic seed carrying the prior
        // compaction's summary content, authored 'model'.
        assert_eq!(compacted[0].author, "model");
        assert_eq!(
            compacted[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("summary")
        );
    }

    #[test]
    fn events_to_compact_for_token_threshold_only_considers_events_after_the_last_compaction() {
        let mut compaction_event = event_with_compaction(0, 0.0, 5.0);
        compaction_event.timestamp = 5.0;
        let mut before = text_event("user", "before");
        before.timestamp = 1.0;
        let mut after1 = text_event("user", "after1");
        after1.timestamp = 6.0;
        let mut after2 = text_event("user", "after2");
        after2.timestamp = 7.0;

        let events = vec![before, compaction_event, after1, after2];
        let compacted = events_to_compact_for_token_threshold(&events, 0);
        // The existing compaction's summary is seeded onto the front,
        // plus the two candidate events after its end_timestamp (5.0);
        // `before` (ts=1.0) is excluded as a candidate.
        assert_eq!(compacted.len(), 3);
        assert_eq!(compacted[0].author, "model");
        assert_eq!(
            compacted[1].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("after1")
        );
        assert_eq!(
            compacted[2].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("after2")
        );
    }

    // --- run_compaction_for_token_threshold / run_compaction_for_sliding_window ---

    struct StubLlm {
        response: adk_models::llm_response::LlmResponse,
    }

    impl adk_models::base_llm::BaseLlm for StubLlm {
        fn model(&self) -> &str {
            "stub-model"
        }

        fn type_name(&self) -> &'static str {
            "StubLlm"
        }

        fn generate_content_async<'a>(
            &'a self,
            _llm_request: &'a adk_models::llm_request::LlmRequest,
            _stream: bool,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<
                            Vec<adk_models::llm_response::LlmResponse>,
                            adk_models::base_llm::BaseLlmError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            let response = self.response.clone();
            Box::pin(async move { Ok(vec![response]) })
        }
    }

    fn llm_flow_agent_with_summary(summary_text: &str) -> BaseAgent {
        let llm_agent = adk_agents::llm_agent::LlmAgent::new(
            adk_agents::llm_agent::ModelRef::Name("gemini-2.0-flash".to_string()),
        );
        let response = adk_models::llm_response::LlmResponse {
            content: Some(Content::new("model", vec![Part::text(summary_text)])),
            ..Default::default()
        };
        let llm_flow =
            crate::llm_flow::LlmFlow::with_model(llm_agent, Arc::new(StubLlm { response }));
        BaseAgent::new("root", llm_flow).unwrap()
    }

    fn invocation_event(invocation_id: &str, timestamp: f64) -> Event {
        let mut event = Event::new(invocation_id, "user", NodeInfo::new(""));
        event.content = Some(Content::user_text("hi"));
        event.timestamp = timestamp;
        event
    }

    fn token_threshold_config() -> EventsCompactionConfig {
        EventsCompactionConfig {
            token_threshold: Some(1),
            event_retention_size: Some(0),
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn token_threshold_returns_none_when_neither_trigger_is_configured() {
        let config = EventsCompactionConfig::default();
        let agent = llm_flow_agent_with_summary("summary");
        let events = vec![invocation_event("inv-1", 1.0)];
        assert!(run_compaction_for_token_threshold(&config, &agent, &events)
            .await
            .unwrap()
            .is_none());
    }

    #[rusty_tokio::test]
    async fn token_threshold_returns_none_when_below_the_threshold() {
        let config = EventsCompactionConfig {
            token_threshold: Some(1_000_000),
            event_retention_size: Some(0),
            ..Default::default()
        };
        let agent = llm_flow_agent_with_summary("summary");
        let events = vec![invocation_event("inv-1", 1.0)];
        assert!(run_compaction_for_token_threshold(&config, &agent, &events)
            .await
            .unwrap()
            .is_none());
    }

    #[rusty_tokio::test]
    async fn token_threshold_triggers_a_compaction_event_when_above_the_threshold() {
        let mut event = invocation_event("inv-1", 1.0);
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(1_000_000),
        )]));
        let config = token_threshold_config();
        let agent = llm_flow_agent_with_summary("the summary");

        let compaction_event = run_compaction_for_token_threshold(&config, &agent, &[event])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            compaction_event
                .actions
                .compaction
                .unwrap()
                .compacted_content
                .parts[0]
                .text
                .as_deref(),
            Some("the summary")
        );
    }

    #[rusty_tokio::test]
    async fn token_threshold_returns_none_when_there_are_no_candidates_to_compact() {
        let mut event = invocation_event("inv-1", 1.0);
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(1_000_000),
        )]));
        // retention_size >= candidate count -> nothing left to compact.
        let config = EventsCompactionConfig {
            token_threshold: Some(1),
            event_retention_size: Some(5),
            ..Default::default()
        };
        let agent = llm_flow_agent_with_summary("summary");
        assert!(
            run_compaction_for_token_threshold(&config, &agent, &[event])
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rusty_tokio::test]
    async fn token_threshold_errors_when_the_agent_isnt_llm_backed() {
        let mut event = invocation_event("inv-1", 1.0);
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(1_000_000),
        )]));
        let config = token_threshold_config();
        let agent = BaseAgent::new("root", adk_agents::base_agent::NoopBehavior).unwrap();

        match run_compaction_for_token_threshold(&config, &agent, &[event]).await {
            Err(CompactionTriggerError::Summarizer(
                EnsureCompactionSummarizerError::NotAnLlmAgent,
            )) => {}
            other => panic!("expected a Summarizer(NotAnLlmAgent) error, got {other:?}"),
        }
    }

    fn sliding_window_config(
        compaction_interval: i64,
        overlap_size: i64,
    ) -> EventsCompactionConfig {
        EventsCompactionConfig {
            compaction_interval: Some(compaction_interval),
            overlap_size: Some(overlap_size),
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn sliding_window_returns_none_for_no_events() {
        let config = sliding_window_config(1, 0);
        let agent = llm_flow_agent_with_summary("summary");
        assert!(
            run_compaction_for_sliding_window(&config, &agent, &[], false)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rusty_tokio::test]
    async fn sliding_window_returns_none_when_neither_trigger_mode_is_configured() {
        let config = EventsCompactionConfig::default();
        let agent = llm_flow_agent_with_summary("summary");
        let events = vec![invocation_event("inv-1", 1.0)];
        assert!(
            run_compaction_for_sliding_window(&config, &agent, &events, false)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rusty_tokio::test]
    async fn sliding_window_returns_none_when_not_enough_new_invocations() {
        let config = sliding_window_config(3, 0);
        let agent = llm_flow_agent_with_summary("summary");
        let events = vec![
            invocation_event("inv-1", 1.0),
            invocation_event("inv-2", 2.0),
        ];
        assert!(
            run_compaction_for_sliding_window(&config, &agent, &events, false)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rusty_tokio::test]
    async fn sliding_window_triggers_when_enough_new_invocations_have_completed() {
        let config = sliding_window_config(2, 0);
        let agent = llm_flow_agent_with_summary("window summary");
        let events = vec![
            invocation_event("inv-1", 1.0),
            invocation_event("inv-2", 2.0),
        ];

        let compaction_event = run_compaction_for_sliding_window(&config, &agent, &events, false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            compaction_event
                .actions
                .compaction
                .unwrap()
                .compacted_content
                .parts[0]
                .text
                .as_deref(),
            Some("window summary")
        );
    }

    #[rusty_tokio::test]
    async fn sliding_window_prefers_token_threshold_when_configured_and_triggered() {
        let mut event = invocation_event("inv-1", 1.0);
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(1_000_000),
        )]));
        let mut config = sliding_window_config(1, 0);
        config.token_threshold = Some(1);
        config.event_retention_size = Some(0);
        let agent = llm_flow_agent_with_summary("token summary");

        let compaction_event = run_compaction_for_sliding_window(&config, &agent, &[event], false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            compaction_event
                .actions
                .compaction
                .unwrap()
                .compacted_content
                .parts[0]
                .text
                .as_deref(),
            Some("token summary")
        );
    }

    #[rusty_tokio::test]
    async fn sliding_window_skip_token_compaction_bypasses_the_token_threshold_check() {
        // Token-threshold IS configured and would trigger, but
        // skip_token_compaction=true bypasses it; sliding window isn't
        // configured either, so nothing triggers.
        let mut event = invocation_event("inv-1", 1.0);
        event.usage_metadata = Some(Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(1_000_000),
        )]));
        let config = EventsCompactionConfig {
            token_threshold: Some(1),
            event_retention_size: Some(0),
            ..Default::default()
        };
        let agent = llm_flow_agent_with_summary("summary");

        assert!(
            run_compaction_for_sliding_window(&config, &agent, &[event], true)
                .await
                .unwrap()
                .is_none()
        );
    }

    // --- with_real_compaction_trigger (end-to-end Runner wiring) ---

    fn compacting_app(runner_agent: BaseAgent) -> adk_agents::app::App {
        adk_agents::app::App::new("test-app", runner_agent)
            .unwrap()
            .with_events_compaction_config(EventsCompactionConfig {
                compaction_interval: Some(1),
                overlap_size: Some(0),
                ..Default::default()
            })
    }

    #[rusty_tokio::test]
    async fn with_real_compaction_trigger_appends_a_compaction_event_after_a_turn() {
        use adk_agents::services::SessionService as _;
        let session_service = Arc::new(adk_agents::services::InMemorySessionService::new());
        let agent = llm_flow_agent_with_summary("the summary");
        let app = compacting_app(agent);
        let runner =
            adk_runners::runner::Runner::from_app(app, None, session_service.clone()).unwrap();
        let runner = with_real_compaction_trigger(runner).with_auto_create_session(true);

        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let session = session_service
            .get_session("test-app", "user", "s1")
            .await
            .unwrap();
        let compaction_event = session
            .events
            .iter()
            .find(|event| event.actions.compaction.is_some())
            .expect("expected a compaction event to have been appended");
        assert_eq!(
            compaction_event
                .actions
                .compaction
                .as_ref()
                .unwrap()
                .compacted_content
                .parts[0]
                .text
                .as_deref(),
            Some("the summary")
        );
    }

    #[rusty_tokio::test]
    async fn without_the_trigger_wired_compaction_never_runs_even_if_configured() {
        use adk_agents::services::SessionService as _;
        let session_service = Arc::new(adk_agents::services::InMemorySessionService::new());
        let agent = llm_flow_agent_with_summary("the summary");
        let app = compacting_app(agent);
        // No `with_real_compaction_trigger` call this time.
        let runner = adk_runners::runner::Runner::from_app(app, None, session_service.clone())
            .unwrap()
            .with_auto_create_session(true);

        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let session = session_service
            .get_session("test-app", "user", "s1")
            .await
            .unwrap();
        assert!(!session
            .events
            .iter()
            .any(|event| event.actions.compaction.is_some()));
    }
}
