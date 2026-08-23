//! Capability C0172 (partial): the `request_confirmation` request
//! processor, ported from `google.adk.flows.llm_flows.request_confirmation`.
//!
//! Handles the round-trip for tools that require human confirmation
//! before running: a tool asks for confirmation via an
//! `adk_request_confirmation` function call, the user's approval/denial
//! comes back as a function response, and once approved the original
//! tool call is re-executed.
//!
//! **Scope, disclosed**: only the small, pure, tool-infrastructure-free
//! slice is ported — [`get_original_function_call_args`] (extracting the
//! `originalFunctionCall` payload out of an `adk_request_confirmation`
//! call's args) and [`map_confirmation_to_original_fc_ids`] (the cheap
//! validation-free dedup pre-pass that maps a confirmation call's id back
//! to the original function-call id it confirms, so already-consumed
//! confirmations can be dropped before expensive re-validation). **Not**
//! ported: parsing a `ToolConfirmation` out of a confirmation response,
//! resolving/validating the confirmed tool against session history
//! (`_resolve_confirmation_targets`), and re-executing it
//! (`functions.handle_function_call_list_async`) — all of these need
//! `BaseTool`/`ToolConfirmation`/`ToolContext` (Phase 8/9), which don't
//! exist in this port yet, the same blocker `agent_transfer.rs`/
//! `output_schema.rs` already disclose for their own Phase 8 gaps.

use adk_events::Event;
use adk_genai::content::FunctionCall;
use rusty_serde::value::Value;
use std::collections::{HashMap, HashSet};

/// The key an `adk_request_confirmation` function call's args carry the
/// original (pre-confirmation) function call payload under.
pub const ORIGINAL_FUNCTION_CALL_KEY: &str = "originalFunctionCall";

/// `_get_original_function_call_args`: the raw `originalFunctionCall`
/// payload of a confirmation call, or `None` if absent or not itself a
/// map (malformed).
pub fn get_original_function_call_args(
    function_call: &FunctionCall,
) -> Option<&Vec<(String, Value)>> {
    let args = function_call.args.as_ref()?;
    match args.get(ORIGINAL_FUNCTION_CALL_KEY)? {
        Value::Map(entries) => Some(entries),
        _ => None,
    }
}

fn find_id(entries: &[(String, Value)]) -> Option<&str> {
    entries
        .iter()
        .find(|(key, _)| key == "id")
        .and_then(|(_, value)| value.as_str())
        .filter(|id| !id.is_empty())
}

/// `_map_confirmation_to_original_fc_ids`: a cheap, validation-free
/// pre-pass mapping each confirmation function-call id (one of
/// `confirmation_fc_ids`) to the original function-call id it confirms,
/// scanning every function call in `events`. Confirmations whose original
/// function call id can't be determined are omitted.
pub fn map_confirmation_to_original_fc_ids(
    events: &[Event],
    confirmation_fc_ids: &HashSet<String>,
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    for event in events {
        for function_call in event.get_function_calls() {
            let Some(id) = &function_call.id else {
                continue;
            };
            if !confirmation_fc_ids.contains(id) {
                continue;
            }
            let Some(original_args) = get_original_function_call_args(function_call) else {
                continue;
            };
            if let Some(original_id) = find_id(original_args) {
                mapping.insert(id.clone(), original_id.to_string());
            }
        }
    }
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{Content, Part};
    use std::collections::BTreeMap;

    fn event_with_calls(author: &str, calls: Vec<Part>) -> Event {
        let mut e = Event::new("inv-1", author, NodeInfo::new("root"));
        e.content = Some(Content::new("model", calls));
        e
    }

    fn confirmation_call(id: &str, original_fc_id: Option<&str>) -> FunctionCall {
        let mut args = BTreeMap::new();
        if let Some(original_id) = original_fc_id {
            args.insert(
                ORIGINAL_FUNCTION_CALL_KEY.to_string(),
                Value::Map(vec![
                    ("id".to_string(), Value::String(original_id.to_string())),
                    ("name".to_string(), Value::String("some_tool".to_string())),
                ]),
            );
        }
        FunctionCall {
            id: Some(id.to_string()),
            name: Some("adk_request_confirmation".to_string()),
            args: Some(args),
            will_continue: None,
        }
    }

    #[test]
    fn extracts_the_original_function_call_map() {
        let call = confirmation_call("conf-1", Some("orig-1"));
        let entries = get_original_function_call_args(&call).unwrap();
        assert!(entries
            .iter()
            .any(|(k, v)| k == "id" && v.as_str() == Some("orig-1")));
    }

    #[test]
    fn returns_none_when_the_payload_is_absent() {
        let call = confirmation_call("conf-1", None);
        assert!(get_original_function_call_args(&call).is_none());
    }

    #[test]
    fn returns_none_when_the_payload_is_not_a_map() {
        let mut args = BTreeMap::new();
        args.insert(
            ORIGINAL_FUNCTION_CALL_KEY.to_string(),
            Value::String("not a map".to_string()),
        );
        let call = FunctionCall {
            id: Some("conf-1".to_string()),
            name: Some("adk_request_confirmation".to_string()),
            args: Some(args),
            will_continue: None,
        };
        assert!(get_original_function_call_args(&call).is_none());
    }

    #[test]
    fn maps_a_confirmation_id_to_its_original_function_call_id() {
        let call = confirmation_call("conf-1", Some("orig-1"));
        let events = vec![event_with_calls("model", vec![Part::function_call(call)])];
        let ids = HashSet::from(["conf-1".to_string()]);
        let mapping = map_confirmation_to_original_fc_ids(&events, &ids);
        assert_eq!(mapping.get("conf-1"), Some(&"orig-1".to_string()));
    }

    #[test]
    fn omits_confirmations_that_are_not_in_the_requested_id_set() {
        let call = confirmation_call("conf-1", Some("orig-1"));
        let events = vec![event_with_calls("model", vec![Part::function_call(call)])];
        let ids = HashSet::from(["some-other-id".to_string()]);
        let mapping = map_confirmation_to_original_fc_ids(&events, &ids);
        assert!(mapping.is_empty());
    }

    #[test]
    fn omits_confirmations_with_a_malformed_or_missing_original_id() {
        let call = confirmation_call("conf-1", None);
        let events = vec![event_with_calls("model", vec![Part::function_call(call)])];
        let ids = HashSet::from(["conf-1".to_string()]);
        assert!(map_confirmation_to_original_fc_ids(&events, &ids).is_empty());
    }
}
