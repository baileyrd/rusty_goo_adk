//! Capabilities C0288 (partial)/C0289: pure dedup and token-estimation
//! logic ported from `apps/compaction.py` — a distinct source module
//! from this crate's sibling [`crate::compaction`]
//! (`flows/llm_flows/_content_compaction.py`, C0185), which happens to
//! need the same subsumption-detection shape for a different purpose
//! (deciding which raw events to filter out of a request's contents,
//! not which sliding-window range to summarize next).
//!
//! **Scope, this batch**: only the pure, self-contained pieces — C0288's
//! dedup logic (`_valid_compactions`/`_is_compaction_subsumed`/
//! `_latest_compaction_event`) and C0289's token-count estimation
//! (`_estimate_prompt_token_count`/`_latest_prompt_token_count`). C0288's
//! OTel-traced summarization wrapper (`_summarize_events_with_trace`)
//! needs span/tracer machinery this port hasn't adopted (see
//! `adk-agents::telemetry_context`'s own disclosed scope) — nothing here
//! wraps the summarizer call in a span. Deliberately left for a
//! follow-up batch: C0290 (`_ensure_compaction_summarizer`), C0291
//! (`_events_to_compact_for_token_threshold`/
//! `_longest_self_contained_prefix`), C0292
//! (`_safe_token_compaction_split_index`), and C0293 (the two
//! `Runner`-facing trigger entrypoints, which need `App`/`Runner`
//! wiring this batch doesn't touch).
//!
//! **Adaptation, disclosed**: the source's `_count_chars_in_content`
//! `json.dumps`s a function call's `args`/a function response's
//! `response`, falling back to `str()` on a serialization failure. This
//! port's `args`/`response` are already-typed `BTreeMap<String, Value>`
//! (always JSON-serializable), so only the `json.dumps` path applies —
//! there's no failure case to fall back from.

use adk_events::Event;
use adk_genai::content::Content;
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
}
