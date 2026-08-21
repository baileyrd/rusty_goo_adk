//! Capability C0033: `_apply_rewinds`, ported from
//! `google.adk.events._rewind_events`.
//!
//! **Flagged for extra scrutiny**: the source itself has no dedicated unit
//! test for this function (confirmed absent from
//! `tests/unittests/events/` in the source repo) despite being described
//! as "the single source of truth" shared between LLM prompt building and
//! context compaction — a correctness-critical function the source ships
//! without direct coverage. This port's test suite below is therefore
//! derived from the *description* of the intended behavior, not from a
//! ported source test, and should be treated as the primary parity
//! evidence to scrutinize if a bug ever surfaces here.

use crate::Event;

/// The single source of truth for "which events are live after rewinds":
/// walks backward through `events`, and whenever an event carries
/// `actions.rewind_before_invocation_id == Some(x)`, drops that event plus
/// everything back to (and including) the earliest event of invocation
/// `x`, then resumes scanning from there. Returns the surviving events in
/// their original relative order.
pub fn apply_rewinds(events: &[Event]) -> Vec<Event> {
    // Walk backward, deciding per-index whether to keep it, then reverse
    // back to original order at the end.
    let mut keep = vec![true; events.len()];
    let mut i = events.len();
    while i > 0 {
        i -= 1;
        if !keep[i] {
            continue;
        }
        if let Some(target_invocation) = &events[i].actions.rewind_before_invocation_id {
            // Drop this event...
            keep[i] = false;
            // ...and walk back further, dropping everything through the
            // earliest event of `target_invocation` (inclusive).
            let mut j = i;
            let mut found_target = false;
            while j > 0 {
                j -= 1;
                keep[j] = false;
                if &events[j].invocation_id == target_invocation {
                    found_target = true;
                    // Keep walking back while still inside the target
                    // invocation, so the *earliest* matching event is the
                    // one dropped, not just the first one encountered
                    // scanning backward.
                    while j > 0 && &events[j - 1].invocation_id == target_invocation {
                        j -= 1;
                        keep[j] = false;
                    }
                    break;
                }
            }
            // Resume scanning from just before whatever we dropped.
            i = if found_target { j } else { 0 };
            if i == 0 {
                break;
            }
        }
    }

    events
        .iter()
        .zip(keep)
        .filter_map(|(e, k)| if k { Some(e.clone()) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_info::NodeInfo;

    fn event(invocation_id: &str) -> Event {
        Event::new(invocation_id, "agent", NodeInfo::new("root"))
    }

    fn rewind_event(invocation_id: &str, rewind_before: &str) -> Event {
        let mut e = event(invocation_id);
        e.actions.rewind_before_invocation_id = Some(rewind_before.to_string());
        e
    }

    #[test]
    fn no_rewinds_keeps_every_event() {
        let events = vec![event("i1"), event("i2"), event("i3")];
        let result = apply_rewinds(&events);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn a_rewind_drops_back_through_and_including_the_target_invocation() {
        // i1, i2 (target), i3, rewind(back to i2)
        let events = vec![
            event("i1"),
            event("i2"),
            event("i3"),
            rewind_event("i4", "i2"),
        ];
        let result = apply_rewinds(&events);
        let ids: Vec<&str> = result.iter().map(|e| e.invocation_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["i1"],
            "i2, i3, and the rewind event itself should all be dropped"
        );
    }

    #[test]
    fn a_rewind_drops_every_event_of_a_multi_event_target_invocation() {
        // i1, i2a, i2b, i2c (all invocation i2), i3, rewind(back to i2)
        let events = vec![
            event("i1"),
            event("i2"),
            event("i2"),
            event("i2"),
            event("i3"),
            rewind_event("i4", "i2"),
        ];
        let result = apply_rewinds(&events);
        let ids: Vec<&str> = result.iter().map(|e| e.invocation_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["i1"],
            "all three i2 events should be dropped, not just the first found"
        );
    }

    #[test]
    fn scanning_resumes_before_the_dropped_range_for_earlier_rewinds() {
        // i1, i2, rewind_a(back to i2), i3, rewind_b(back to i1)
        // rewind_b should still see i1 as a candidate target even though
        // i2/rewind_a were already dropped.
        let events = vec![
            event("i1"),
            event("i2"),
            rewind_event("i3", "i2"),
            event("i4"),
            rewind_event("i5", "i1"),
        ];
        let result = apply_rewinds(&events);
        assert!(
            result.is_empty(),
            "the second rewind should drop everything back through i1"
        );
    }

    #[test]
    fn a_rewind_with_no_matching_target_invocation_drops_everything_before_it() {
        let events = vec![
            event("i1"),
            event("i2"),
            rewind_event("i3", "does-not-exist"),
        ];
        let result = apply_rewinds(&events);
        assert!(result.is_empty());
    }

    #[test]
    fn returns_surviving_events_in_original_relative_order() {
        let events = vec![event("i1"), event("i2"), event("i3")];
        let result = apply_rewinds(&events);
        let ids: Vec<&str> = result.iter().map(|e| e.invocation_id.as_str()).collect();
        assert_eq!(ids, vec!["i1", "i2", "i3"]);
    }
}
