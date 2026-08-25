//! Capability C0185: compaction-aware history reconstruction, ported from
//! `google.adk.flows.llm_flows._content_compaction`.
//!
//! **Adaptation, disclosed**: the source's defensive `is None` checks on
//! `compaction.start_timestamp`/`end_timestamp`/`compacted_content` are
//! omitted — `EventCompaction`'s fields are all non-optional (matching the
//! source's own pydantic model, which declares them required, not
//! `Optional`), so Rust's type system already guarantees they're present
//! whenever `actions.compaction` is `Some`.
//!
//! Landing this required narrowing `EventCompaction.compacted_content`
//! from a placeholder JSON `Value` to a real `adk_genai::content::Content`
//! (see `event_compaction.rs`'s own module doc) — `process_compaction_events`
//! needs a genuine `Content` to build the synthetic summary event's
//! `content` field.

use adk_events::node_info::NodeInfo;
use adk_events::Event;
use std::collections::{HashMap, HashSet};

/// C0185: `_process_compaction_events`.
///
/// Identifies compacted ranges and filters out events covered by a kept
/// compaction summary. Example:
/// `[e1(ts=1), e2(ts=2), compaction_1(1-2), e3(ts=4), compaction_2(2-4), e4(ts=6)]`
/// — overlaps are resolved by keeping only non-subsumed compaction
/// summaries; a summary event is materialized at its compaction end
/// timestamp, and raw events inside any kept compaction range are filtered
/// out.
///
/// `agent_name` attributes the materialized summary event so the agent
/// reads its own compacted history as its own prior turns; an empty
/// string falls back to `"model"`, matching the source's `agent_name or
/// 'model'`.
pub fn process_compaction_events(events: &[Event], agent_name: &str) -> Vec<Event> {
    let mut compaction_infos: Vec<(usize, f64, f64)> = Vec::new();
    for (i, event) in events.iter().enumerate() {
        if let Some(compaction) = &event.actions.compaction {
            compaction_infos.push((i, compaction.start_timestamp, compaction.end_timestamp));
        }
    }

    let mut subsumed: HashSet<usize> = HashSet::new();
    for &(event_index, start_ts, end_ts) in &compaction_infos {
        for &(other_index, other_start, other_end) in &compaction_infos {
            if other_index == event_index {
                continue;
            }
            if other_start <= start_ts
                && other_end >= end_ts
                && (other_start < start_ts || other_end > end_ts || other_index > event_index)
            {
                subsumed.insert(event_index);
                break;
            }
        }
    }

    let mut compaction_ranges: Vec<(f64, f64)> = Vec::new();
    let mut processed_items: Vec<(f64, usize, Event)> = Vec::new();

    for (i, event) in events.iter().enumerate() {
        let Some(compaction) = &event.actions.compaction else {
            continue;
        };
        if subsumed.contains(&i) {
            continue;
        }
        compaction_ranges.push((compaction.start_timestamp, compaction.end_timestamp));

        let author = if agent_name.is_empty() {
            "model"
        } else {
            agent_name
        };
        let mut summary = Event::new(event.invocation_id.clone(), author, NodeInfo::new(""));
        summary.timestamp = compaction.end_timestamp;
        summary.content = Some(compaction.compacted_content.clone());
        summary.branch = event.branch.clone();
        summary.actions = event.actions.clone();
        processed_items.push((compaction.end_timestamp, i, summary));
    }

    let is_timestamp_compacted = |ts: f64| -> bool {
        compaction_ranges
            .iter()
            .any(|&(start_ts, end_ts)| start_ts <= ts && ts <= end_ts)
    };

    for (i, event) in events.iter().enumerate() {
        if event.actions.compaction.is_some() {
            continue;
        }
        if is_timestamp_compacted(event.timestamp) {
            continue;
        }
        processed_items.push((event.timestamp, i, event.clone()));
    }

    // Keep chronological order and a stable tie-breaker for equal timestamps.
    processed_items.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    processed_items
        .into_iter()
        .map(|(_, _, event)| event)
        .collect()
}

/// C0185: `_recover_compacted_function_calls`.
///
/// Compaction can summarize away a `function_call` while a matching
/// `function_response` survives outside the compacted range — the clearest
/// case is a long-running tool call: the call is compacted along with its
/// intermediate placeholder response, then the real result arrives on
/// resume (a later event not covered by the summary). That surviving
/// response would be orphaned, breaking call/response pairing during
/// prompt assembly.
///
/// For each response whose call is no longer present, this restores the
/// original call event from `source_events` (the pre-compaction list),
/// inserting it immediately before the first surviving response that
/// references it. The whole call event is re-injected verbatim (rather
/// than trimmed to the resumed call) so parallel-call thought signatures,
/// which only the first part carries, are preserved. Any sibling responses
/// that compaction removed are re-injected too, so a sibling doesn't
/// surface as a phantom pending call.
pub fn recover_compacted_function_calls(events: Vec<Event>, source_events: &[Event]) -> Vec<Event> {
    let mut call_ids_present: HashSet<String> = HashSet::new();
    let mut response_ids_present: HashSet<String> = HashSet::new();
    for event in &events {
        for fc in event.get_function_calls() {
            if let Some(id) = &fc.id {
                call_ids_present.insert(id.clone());
            }
        }
        for fr in event.get_function_responses() {
            if let Some(id) = &fr.id {
                response_ids_present.insert(id.clone());
            }
        }
    }

    let orphaned_ids: HashSet<String> = response_ids_present
        .iter()
        .filter(|id| !call_ids_present.contains(*id))
        .cloned()
        .collect();
    if orphaned_ids.is_empty() {
        return events;
    }

    let mut call_event_by_id: HashMap<String, Event> = HashMap::new();
    for event in source_events {
        for fc in event.get_function_calls() {
            if let Some(id) = &fc.id {
                if orphaned_ids.contains(id) {
                    call_event_by_id
                        .entry(id.clone())
                        .or_insert_with(|| event.clone());
                }
            }
        }
    }

    if call_event_by_id.is_empty() {
        return events;
    }

    // Keep the highest-timestamp response per id so a sibling that
    // completed before being compacted contributes its real result, not
    // its stale placeholder; ties fall back to source order.
    let mut response_event_by_id: HashMap<String, Event> = HashMap::new();
    for event in source_events {
        for fr in event.get_function_responses() {
            let Some(id) = &fr.id else { continue };
            let should_replace = response_event_by_id
                .get(id)
                .is_none_or(|existing| event.timestamp >= existing.timestamp);
            if should_replace {
                response_event_by_id.insert(id.clone(), event.clone());
            }
        }
    }

    let mut result: Vec<Event> = Vec::with_capacity(events.len());
    let mut reinjected_ids: HashSet<String> = HashSet::new();
    for event in &events {
        for fr in event.get_function_responses() {
            let Some(fr_id) = &fr.id else { continue };
            if reinjected_ids.contains(fr_id) {
                continue;
            }
            let Some(call_event) = call_event_by_id.get(fr_id) else {
                continue;
            };
            result.push(call_event.clone());
            let sibling_ids: Vec<String> = call_event
                .get_function_calls()
                .into_iter()
                .filter_map(|fc| fc.id.clone())
                .collect();
            reinjected_ids.extend(sibling_ids.iter().cloned());
            // Recover sibling responses that compaction removed so a
            // parallel sibling isn't left looking like a pending call.
            for sibling_id in &sibling_ids {
                if !response_ids_present.contains(sibling_id) {
                    if let Some(sibling_response) = response_event_by_id.get(sibling_id) {
                        result.push(sibling_response.clone());
                    }
                }
            }
        }
        result.push(event.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::event_compaction::EventCompaction;
    use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};

    fn event_at(author: &str, ts: f64) -> Event {
        let mut e = Event::new("inv-1", author, NodeInfo::new("root"));
        e.timestamp = ts;
        e
    }

    fn compaction_event(ts: f64, start: f64, end: f64, summary_text: &str) -> Event {
        let mut e = event_at("model", ts);
        e.actions.compaction = Some(EventCompaction {
            start_timestamp: start,
            end_timestamp: end,
            compacted_content: Content::user_text(summary_text),
        });
        e
    }

    fn fc_event(ts: f64, id: &str) -> Event {
        let mut e = event_at("model", ts);
        e.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                partial_args: None,
                id: Some(id.to_string()),
                name: Some("tool".to_string()),
                args: None,
                will_continue: None,
            })],
        ));
        e
    }

    fn fr_event(ts: f64, id: &str) -> Event {
        let mut e = event_at("user", ts);
        e.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some(id.to_string()),
                name: Some("tool".to_string()),
                response: None,
                ..Default::default()
            })],
        ));
        e
    }

    // --- process_compaction_events ---

    #[test]
    fn events_outside_any_compaction_range_pass_through_unchanged() {
        let events = vec![event_at("user", 1.0), event_at("model", 2.0)];
        let result = process_compaction_events(&events, "");
        assert_eq!(result, events);
    }

    #[test]
    fn events_inside_a_compaction_range_are_replaced_by_the_summary() {
        let e1 = event_at("user", 1.0);
        let e2 = event_at("model", 2.0);
        let summary = compaction_event(2.5, 1.0, 2.0, "summary of e1/e2");
        let e3 = event_at("user", 4.0);
        let events = vec![e1, e2, summary, e3];

        let result = process_compaction_events(&events, "");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].author, "model");
        assert_eq!(
            result[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("summary of e1/e2")
        );
        assert_eq!(result[0].timestamp, 2.0);
        assert_eq!(result[1].timestamp, 4.0);
    }

    #[test]
    fn the_summary_event_is_attributed_to_the_given_agent_name() {
        let e1 = event_at("user", 1.0);
        let summary = compaction_event(2.0, 1.0, 1.0, "summary");
        let result = process_compaction_events(&[e1, summary], "my_agent");
        assert_eq!(result[0].author, "my_agent");
    }

    #[test]
    fn a_wider_compaction_range_subsumes_a_narrower_overlapping_one() {
        // compaction_1 covers 1-2, compaction_2 covers 1-4 (wider, subsumes it).
        let compaction_1 = compaction_event(2.5, 1.0, 2.0, "narrow");
        let compaction_2 = compaction_event(4.5, 1.0, 4.0, "wide");
        let events = vec![compaction_1, compaction_2];

        let result = process_compaction_events(&events, "");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("wide")
        );
    }

    #[test]
    fn identical_compaction_ranges_keep_only_the_later_indexed_one() {
        let compaction_1 = compaction_event(2.5, 1.0, 2.0, "first");
        let compaction_2 = compaction_event(2.6, 1.0, 2.0, "second");
        let events = vec![compaction_1, compaction_2];

        let result = process_compaction_events(&events, "");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("second")
        );
    }

    // --- recover_compacted_function_calls ---

    #[test]
    fn is_a_no_op_when_every_response_has_a_present_call() {
        let events = vec![fc_event(1.0, "id1"), fr_event(2.0, "id1")];
        let result = recover_compacted_function_calls(events.clone(), &events);
        assert_eq!(result, events);
    }

    #[test]
    fn reinserts_a_compacted_call_ahead_of_its_surviving_response() {
        let call = fc_event(1.0, "id1");
        let source_events = vec![call.clone(), fr_event(2.0, "id1")];
        // Post-compaction: the call was summarized away, only the response survived.
        let events = vec![fr_event(2.0, "id1")];

        let result = recover_compacted_function_calls(events, &source_events);
        assert_eq!(result.len(), 2);
        assert!(!result[0].get_function_calls().is_empty());
        assert!(!result[1].get_function_responses().is_empty());
    }

    #[test]
    fn reinserts_a_compacted_sibling_response_alongside_its_recovered_call() {
        // A parallel call with two ids; only id2's response survived compaction.
        let mut call = event_at("model", 1.0);
        call.content = Some(Content::new(
            "model",
            vec![
                Part::function_call(FunctionCall {
                    partial_args: None,
                    id: Some("id1".to_string()),
                    name: Some("tool".to_string()),
                    args: None,
                    will_continue: None,
                }),
                Part::function_call(FunctionCall {
                    partial_args: None,
                    id: Some("id2".to_string()),
                    name: Some("tool".to_string()),
                    args: None,
                    will_continue: None,
                }),
            ],
        ));
        let sibling_response = fr_event(1.5, "id1");
        let surviving_response = fr_event(2.0, "id2");
        let source_events = vec![
            call.clone(),
            sibling_response.clone(),
            surviving_response.clone(),
        ];
        let events = vec![surviving_response];

        let result = recover_compacted_function_calls(events, &source_events);
        assert_eq!(result.len(), 3);
        assert!(!result[0].get_function_calls().is_empty());
        assert_eq!(
            result[1].get_function_responses()[0].id.as_deref(),
            Some("id1")
        );
        assert_eq!(
            result[2].get_function_responses()[0].id.as_deref(),
            Some("id2")
        );
    }

    #[test]
    fn is_a_no_op_when_the_orphaned_calls_source_no_longer_exists() {
        let events = vec![fr_event(2.0, "id1")];
        let result = recover_compacted_function_calls(events.clone(), &[]);
        assert_eq!(result, events);
    }
}
