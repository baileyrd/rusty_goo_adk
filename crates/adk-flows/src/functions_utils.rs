//! Capability C0196 (partial): a slice of `functions.py`'s helpers, ported
//! from `google.adk.flows.llm_flows.functions` — `merge_parallel_function_
//! response_events` and the client function-call-id lifecycle helpers.
//!
//! **Scope, disclosed**: `build_auth_request_event`/`generate_auth_event`/
//! `generate_request_confirmation_event` (the "auth/confirmation request
//! events" half of C0196) are **not** ported — they need `AuthConfig`
//! (Phase 9), which doesn't exist in this port yet.
//!
//! **Adaptation, disclosed**: `merge_parallel_function_response_events`'s
//! `EventActions` merge is ported by round-tripping through
//! [`rusty_serde::value::Value`] (`rusty_serde::json::to_value`/
//! `from_value`) and deep-merging the resulting maps, mirroring the
//! source's own approach exactly (`model_dump(exclude_none=True,
//! by_alias=True)` + `deep_merge_dicts` + `model_validate`) rather than
//! hand-writing bespoke per-field merge rules — a `None`/absent field
//! from a later event never overwrites an earlier value, matching
//! `exclude_none=True`; `render_ui_widgets` is popped out before the
//! generic merge and aggregated additively across every event (not
//! last-wins), then reattached, exactly as the source special-cases it.
//! `get_long_running_function_calls` takes an `is_long_running` callback
//! rather than a `tools_dict: dict[str, BaseTool]`, since `BaseTool`
//! (Phase 8) doesn't exist in this port yet.

use adk_events::{Event, EventActions};
use adk_genai::content::{Content, FunctionCall};
use rusty_serde::value::Value;

pub const AF_FUNCTION_CALL_ID_PREFIX: &str = "adk-";

#[derive(Debug, rusty_err::Error)]
pub enum FunctionsError {
    #[error("No function response events provided.")]
    NoFunctionResponseEvents,
    #[error("failed to serialize EventActions for merging: {0}")]
    ActionsSerialization(String),
    #[error("failed to deserialize merged EventActions: {0}")]
    ActionsDeserialization(String),
}

/// `generate_client_function_call_id`.
pub fn generate_client_function_call_id() -> String {
    format!("{AF_FUNCTION_CALL_ID_PREFIX}{}", Event::new_id())
}

/// `populate_client_function_call_id`: assigns a synthetic id to every
/// function call in `event` that doesn't already have one.
pub fn populate_client_function_call_id(event: &mut Event) {
    let Some(content) = &mut event.content else {
        return;
    };
    for part in &mut content.parts {
        if let Some(function_call) = &mut part.function_call {
            if function_call.id.is_none() {
                function_call.id = Some(generate_client_function_call_id());
            }
        }
    }
}

/// `remove_client_function_call_id`: strips `adk-`-prefixed function
/// call/response ids from `content` in place, so internal tracking ids
/// aren't sent to the model.
pub fn remove_client_function_call_id(content: Option<&mut Content>) {
    let Some(content) = content else {
        return;
    };
    for part in &mut content.parts {
        if let Some(function_call) = &mut part.function_call {
            if function_call
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with(AF_FUNCTION_CALL_ID_PREFIX))
            {
                function_call.id = None;
            }
        }
        if let Some(function_response) = &mut part.function_response {
            if function_response
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with(AF_FUNCTION_CALL_ID_PREFIX))
            {
                function_response.id = None;
            }
        }
    }
}

/// `get_long_running_function_calls`: the ids of every call in
/// `function_calls` whose named tool is long-running, per the caller-
/// supplied `is_long_running` lookup.
pub fn get_long_running_function_calls(
    function_calls: &[FunctionCall],
    is_long_running: &dyn Fn(&str) -> bool,
) -> std::collections::HashSet<String> {
    let mut long_running_tool_ids = std::collections::HashSet::new();
    for function_call in function_calls {
        let Some(name) = &function_call.name else {
            continue;
        };
        let Some(id) = &function_call.id else {
            continue;
        };
        if is_long_running(name) {
            long_running_tool_ids.insert(id.clone());
        }
    }
    long_running_tool_ids
}

/// `find_event_by_function_call_id`: the most recent event (scanning
/// backward) carrying a function call with the given id.
pub fn find_event_by_function_call_id<'a>(
    events: &'a [Event],
    function_call_id: &str,
) -> Option<&'a Event> {
    events.iter().rev().find(|event| {
        event
            .get_function_calls()
            .iter()
            .any(|fc| fc.id.as_deref() == Some(function_call_id))
    })
}

/// `find_matching_function_call`: the event carrying the function call
/// that `events`'s last event's (first) function response answers.
pub fn find_matching_function_call(events: &[Event]) -> Option<&Event> {
    let (last_event, earlier_events) = events.split_last()?;
    let function_responses = last_event.get_function_responses();
    let first_response = function_responses.first()?;
    let function_call_id = first_response.id.as_deref()?;
    find_event_by_function_call_id(earlier_events, function_call_id)
}

fn deep_merge_map_entries(
    mut merged: Vec<(String, Value)>,
    incoming: Vec<(String, Value)>,
) -> Vec<(String, Value)> {
    for (key, incoming_value) in incoming {
        // Mirrors the source's `exclude_none=True`: an absent (here, null)
        // field from the incoming side never overwrites an existing value.
        if matches!(incoming_value, Value::Null) {
            continue;
        }
        match merged.iter_mut().find(|(k, _)| *k == key) {
            Some(existing) => {
                let current = std::mem::replace(&mut existing.1, Value::Null);
                existing.1 = deep_merge_value(current, incoming_value);
            }
            None => merged.push((key, incoming_value)),
        }
    }
    merged
}

fn deep_merge_value(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Map(a_entries), Value::Map(b_entries)) => {
            Value::Map(deep_merge_map_entries(a_entries, b_entries))
        }
        (_, b) => b,
    }
}

const RENDER_UI_WIDGETS_KEY: &str = "renderUiWidgets";

/// `merge_parallel_function_response_events`: merges parallel tool-call
/// response events into one — concatenating their content parts and
/// deep-merging their `EventActions` (aggregating `render_ui_widgets`
/// additively rather than last-wins).
pub fn merge_parallel_function_response_events(events: &[Event]) -> Result<Event, FunctionsError> {
    let Some(first) = events.first() else {
        return Err(FunctionsError::NoFunctionResponseEvents);
    };
    if events.len() == 1 {
        return Ok(first.clone());
    }

    let mut merged_parts = Vec::new();
    for event in events {
        if let Some(content) = &event.content {
            merged_parts.extend(content.parts.iter().cloned());
        }
    }

    let mut merged_actions_value = Value::Map(Vec::new());
    let mut aggregated_ui_widgets: Vec<Value> = Vec::new();
    for event in events {
        let mut actions_value = rusty_serde::json::to_value(&event.actions)
            .map_err(|e| FunctionsError::ActionsSerialization(e.to_string()))?;
        if let Value::Map(entries) = &mut actions_value {
            if let Some(pos) = entries.iter().position(|(k, _)| k == RENDER_UI_WIDGETS_KEY) {
                let (_, widgets_value) = entries.remove(pos);
                if let Value::Seq(widgets) = widgets_value {
                    aggregated_ui_widgets.extend(widgets);
                }
            }
        }
        merged_actions_value = deep_merge_value(merged_actions_value, actions_value);
    }

    if !aggregated_ui_widgets.is_empty() {
        if let Value::Map(entries) = &mut merged_actions_value {
            entries.retain(|(k, _)| k != RENDER_UI_WIDGETS_KEY);
            entries.push((
                RENDER_UI_WIDGETS_KEY.to_string(),
                Value::Seq(aggregated_ui_widgets),
            ));
        }
    }

    let merged_actions: EventActions = rusty_serde::json::from_value(merged_actions_value)
        .map_err(|e| FunctionsError::ActionsDeserialization(e.to_string()))?;

    let mut merged_event = Event::new(
        first.invocation_id.clone(),
        first.author.clone(),
        adk_events::node_info::NodeInfo::new(""),
    );
    merged_event.branch = first.branch.clone();
    merged_event.content = Some(Content::new("user", merged_parts));
    merged_event.actions = merged_actions;
    merged_event.live_session_id = first.live_session_id.clone();
    merged_event.timestamp = first.timestamp;
    Ok(merged_event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_events::ui_widget::UiWidget;
    use adk_genai::content::{FunctionResponse, Part};

    fn event(author: &str) -> Event {
        Event::new("inv-1", author, NodeInfo::new("root"))
    }

    fn event_with_content(author: &str, content: Content) -> Event {
        let mut e = event(author);
        e.content = Some(content);
        e
    }

    fn fc_part(id: Option<&str>, name: &str) -> adk_genai::content::Part {
        adk_genai::content::Part::function_call(FunctionCall {
            id: id.map(str::to_string),
            name: Some(name.to_string()),
            args: None,
            will_continue: None,
        })
    }

    fn fr_part(id: &str, name: &str) -> adk_genai::content::Part {
        Part::function_response(FunctionResponse {
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            response: None,
        })
    }

    // --- client function-call-id lifecycle ---

    #[test]
    fn generated_ids_carry_the_adk_prefix() {
        assert!(generate_client_function_call_id().starts_with(AF_FUNCTION_CALL_ID_PREFIX));
    }

    #[test]
    fn populate_assigns_ids_only_where_missing() {
        let mut e = event_with_content(
            "model",
            Content::new(
                "model",
                vec![fc_part(None, "tool"), fc_part(Some("server-1"), "tool")],
            ),
        );
        populate_client_function_call_id(&mut e);
        let calls = e.get_function_calls();
        assert!(calls[0]
            .id
            .as_ref()
            .unwrap()
            .starts_with(AF_FUNCTION_CALL_ID_PREFIX));
        assert_eq!(calls[1].id.as_deref(), Some("server-1"));
    }

    #[test]
    fn remove_strips_only_adk_prefixed_ids() {
        let mut content = Content::new(
            "model",
            vec![
                fc_part(Some("adk-123"), "tool"),
                fc_part(Some("server-1"), "tool"),
            ],
        );
        remove_client_function_call_id(Some(&mut content));
        assert!(content.parts[0]
            .function_call
            .as_ref()
            .unwrap()
            .id
            .is_none());
        assert_eq!(
            content.parts[1]
                .function_call
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some("server-1")
        );
    }

    // --- get_long_running_function_calls ---

    #[test]
    fn collects_ids_of_long_running_tool_calls_only() {
        let calls = vec![
            FunctionCall {
                id: Some("id1".to_string()),
                name: Some("slow_tool".to_string()),
                args: None,
                will_continue: None,
            },
            FunctionCall {
                id: Some("id2".to_string()),
                name: Some("fast_tool".to_string()),
                args: None,
                will_continue: None,
            },
        ];
        let is_long_running = |name: &str| name == "slow_tool";
        let ids = get_long_running_function_calls(&calls, &is_long_running);
        assert_eq!(ids, std::collections::HashSet::from(["id1".to_string()]));
    }

    // --- find_event_by_function_call_id / find_matching_function_call ---

    #[test]
    fn finds_the_most_recent_event_carrying_the_call_id() {
        let older = event_with_content(
            "model",
            Content::new("model", vec![fc_part(Some("id1"), "tool")]),
        );
        let newer = event_with_content(
            "model",
            Content::new("model", vec![fc_part(Some("id1"), "tool")]),
        );
        let events = vec![older, newer.clone()];
        let found = find_event_by_function_call_id(&events, "id1").unwrap();
        assert_eq!(found.timestamp, newer.timestamp);
    }

    #[test]
    fn find_matching_function_call_locates_the_calls_own_event() {
        let call = event_with_content(
            "model",
            Content::new("model", vec![fc_part(Some("id1"), "tool")]),
        );
        let response =
            event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let events = vec![call.clone(), response];
        let found = find_matching_function_call(&events).unwrap();
        assert_eq!(found.timestamp, call.timestamp);
    }

    #[test]
    fn find_matching_function_call_is_none_without_a_trailing_response() {
        let events = vec![event("model")];
        assert!(find_matching_function_call(&events).is_none());
    }

    // --- merge_parallel_function_response_events ---

    #[test]
    fn errors_with_no_events() {
        let err = merge_parallel_function_response_events(&[]).unwrap_err();
        assert!(matches!(err, FunctionsError::NoFunctionResponseEvents));
    }

    #[test]
    fn a_single_event_is_returned_unchanged() {
        let e = event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let merged = merge_parallel_function_response_events(std::slice::from_ref(&e)).unwrap();
        assert_eq!(merged.timestamp, e.timestamp);
    }

    #[test]
    fn concatenates_parts_from_every_event() {
        let e1 = event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let e2 = event_with_content("user", Content::new("user", vec![fr_part("id2", "tool")]));
        let merged = merge_parallel_function_response_events(&[e1, e2]).unwrap();
        assert_eq!(merged.content.unwrap().parts.len(), 2);
    }

    #[test]
    fn merges_state_delta_across_events_without_dropping_earlier_keys() {
        let mut e1 = event("agent");
        e1.actions
            .state_delta
            .insert("a".to_string(), Value::Int(1));
        let mut e2 = event("agent");
        e2.actions
            .state_delta
            .insert("b".to_string(), Value::Int(2));

        let merged = merge_parallel_function_response_events(&[e1, e2]).unwrap();
        assert_eq!(merged.actions.state_delta.get("a"), Some(&Value::Int(1)));
        assert_eq!(merged.actions.state_delta.get("b"), Some(&Value::Int(2)));
    }

    #[test]
    fn a_later_events_action_field_wins_over_an_earlier_one() {
        let mut e1 = event("agent");
        e1.actions.escalate = false;
        let mut e2 = event("agent");
        e2.actions.escalate = true;

        let merged = merge_parallel_function_response_events(&[e1, e2]).unwrap();
        assert!(merged.actions.escalate);
    }

    #[test]
    fn aggregates_ui_widgets_from_every_event_rather_than_last_wins() {
        let mut e1 = event("agent");
        e1.actions.render_ui_widgets = Some(vec![UiWidget::new("w1", "mcp", Value::Null)]);
        let mut e2 = event("agent");
        e2.actions.render_ui_widgets = Some(vec![UiWidget::new("w2", "mcp", Value::Null)]);

        let merged = merge_parallel_function_response_events(&[e1, e2]).unwrap();
        let widgets = merged.actions.render_ui_widgets.unwrap();
        assert_eq!(widgets.len(), 2);
        assert_eq!(widgets[0].id, "w1");
        assert_eq!(widgets[1].id, "w2");
    }
}
