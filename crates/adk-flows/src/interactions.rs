//! Capability C0174: the `interactions_processor` request processor,
//! ported from `google.adk.flows.llm_flows.interactions_processor`.
//!
//! Extracts the previous `interaction_id` from session events, to enable
//! stateful conversation chaining via the Gemini Interactions API. The
//! actual content filtering (retaining only the latest user messages) is
//! done by the content request processor (`crate::contents`) afterward.
//!
//! **Scope, disclosed**: this is the free-function core logic
//! ([`is_event_in_branch`], [`find_previous_interaction_state`]), not yet
//! a real `BaseLlmRequestProcessor` reading through `InvocationContext` —
//! same scope note as every other Phase 4 processor in this crate. The
//! processor's own gating (the resolved model is a Gemini with
//! `use_interactions_api`) is now wired: `crate::llm_flow::LlmFlow::preprocess`
//! calls [`find_previous_interaction_state`] directly, gated by
//! downcasting its already-resolved `self.model` onto `Gemini` via the
//! `AsAny` mechanism `adk-models::base_llm` provides.

use adk_events::Event;

/// `_is_event_in_branch`: whether `event` belongs to `current_branch` (or
/// the root). A falsy `current_branch` (`None` or empty) means the root:
/// only events with no branch (or an empty one) belong there. Otherwise
/// an event belongs if its branch matches exactly, or it has no branch at
/// all (root-level events are visible from any branch).
pub fn is_event_in_branch(current_branch: Option<&str>, event: &Event) -> bool {
    fn is_falsy(branch: Option<&str>) -> bool {
        branch.is_none_or(str::is_empty)
    }

    if is_falsy(current_branch) {
        return is_falsy(event.branch.as_deref());
    }
    event.branch.as_deref() == current_branch || is_falsy(event.branch.as_deref())
}

/// `_find_previous_interaction_state`: scans `events` in reverse, skipping
/// events outside `current_branch`, and returns the `(interaction_id,
/// environment_id)` from the first event authored by `agent_name` that
/// carries a non-empty `interaction_id`. Returns `(None, None)` if none is
/// found.
pub fn find_previous_interaction_state(
    events: &[Event],
    agent_name: &str,
    current_branch: Option<&str>,
) -> (Option<String>, Option<String>) {
    for event in events.iter().rev() {
        if !is_event_in_branch(current_branch, event) {
            continue;
        }
        if event.author == agent_name
            && event
                .interaction_id
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        {
            return (event.interaction_id.clone(), event.environment_id.clone());
        }
    }
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;

    fn event(author: &str) -> Event {
        Event::new("inv-1", author, NodeInfo::new("root"))
    }

    #[test]
    fn a_branchless_event_belongs_to_the_root() {
        assert!(is_event_in_branch(None, &event("agent_a")));
    }

    #[test]
    fn a_branched_event_does_not_belong_to_the_root() {
        let mut e = event("agent_a");
        e.branch = Some("root.child".to_string());
        assert!(!is_event_in_branch(None, &e));
    }

    #[test]
    fn an_event_on_a_matching_branch_belongs() {
        let mut e = event("agent_a");
        e.branch = Some("root.child".to_string());
        assert!(is_event_in_branch(Some("root.child"), &e));
    }

    #[test]
    fn a_root_level_event_is_visible_from_any_branch() {
        assert!(is_event_in_branch(Some("root.child"), &event("agent_a")));
    }

    #[test]
    fn an_event_on_a_different_branch_does_not_belong() {
        let mut e = event("agent_a");
        e.branch = Some("root.sibling".to_string());
        assert!(!is_event_in_branch(Some("root.child"), &e));
    }

    #[test]
    fn finds_nothing_when_no_event_carries_an_interaction_id() {
        let events = vec![event("agent_a"), event("agent_a")];
        assert_eq!(
            find_previous_interaction_state(&events, "agent_a", None),
            (None, None)
        );
    }

    #[test]
    fn finds_the_most_recent_interaction_id_for_the_given_agent() {
        let mut older = event("agent_a");
        older.interaction_id = Some("interaction-1".to_string());
        older.environment_id = Some("env-1".to_string());
        let mut newer = event("agent_a");
        newer.interaction_id = Some("interaction-2".to_string());
        newer.environment_id = Some("env-2".to_string());
        let events = vec![older, newer];

        assert_eq!(
            find_previous_interaction_state(&events, "agent_a", None),
            (Some("interaction-2".to_string()), Some("env-2".to_string()))
        );
    }

    #[test]
    fn ignores_interaction_ids_from_a_different_author() {
        let mut e = event("agent_b");
        e.interaction_id = Some("interaction-1".to_string());
        assert_eq!(
            find_previous_interaction_state(&[e], "agent_a", None),
            (None, None)
        );
    }

    #[test]
    fn ignores_events_outside_the_current_branch() {
        let mut e = event("agent_a");
        e.interaction_id = Some("interaction-1".to_string());
        e.branch = Some("root.sibling".to_string());
        assert_eq!(
            find_previous_interaction_state(&[e], "agent_a", Some("root.child")),
            (None, None)
        );
    }

    #[test]
    fn treats_an_empty_interaction_id_as_absent() {
        let mut e = event("agent_a");
        e.interaction_id = Some(String::new());
        assert_eq!(
            find_previous_interaction_state(&[e], "agent_a", None),
            (None, None)
        );
    }
}
