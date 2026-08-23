//! Capability C0178 (partial): the `_output_schema_processor` request
//! processor, ported from `google.adk.flows.llm_flows._output_schema_processor`.
//!
//! Handles output schema when tools are also present: some model
//! capabilities can't honor both `output_schema` and a tool list in the
//! same request, so instead a synthetic `set_model_response` tool call is
//! how the model reports its final structured answer.
//!
//! **Scope, disclosed**: only the *gating decision*
//! ([`should_apply_output_schema_processor`]), the instruction text
//! ([`OUTPUT_SCHEMA_TOOL_INSTRUCTION`]), and the two standalone helpers
//! that read back a completed structured response
//! ([`create_final_model_response_event`],
//! [`get_structured_model_response`]) are ported. **Not** ported: actually
//! injecting a `SetModelResponseTool` into the request
//! (`llm_request.append_tools`) — both `SetModelResponseTool` itself and
//! `LlmRequest::append_tools` (C0116) need `BaseTool` (Phase 8), which
//! doesn't exist in this port yet; `append_tools`'s own module doc in
//! `adk-models` already discloses this same blocker.

use adk_events::Event;
use adk_genai::content::{Content, Part};

/// The instruction appended alongside the injected `set_model_response`
/// tool, verbatim from the source.
pub const OUTPUT_SCHEMA_TOOL_INSTRUCTION: &str =
    "IMPORTANT: You have access to other tools, but you must provide your final response using \
     the set_model_response tool with the required structured format. After using any other \
     tools needed to complete the task, always call set_model_response with your final answer \
     in the specified schema format.";

/// Whether the processor would inject a `set_model_response` tool +
/// instruction for this request: an `output_schema` is set, `tools` is
/// non-empty, the resolved model's capabilities can't honor
/// `output_schema` and tools together, and the agent isn't in task mode
/// (task-mode agents report structured output through their return value,
/// not a tool call).
pub fn should_apply_output_schema_processor(
    has_output_schema: bool,
    has_tools: bool,
    model_supports_output_schema_and_tools: bool,
    is_task_mode: bool,
) -> bool {
    has_output_schema && has_tools && !model_supports_output_schema_and_tools && !is_task_mode
}

/// `create_final_model_response_event`: builds a model-response event that
/// looks like a normal (non-tool-call) response, carrying the
/// `set_model_response` tool's validated JSON as plain text.
pub fn create_final_model_response_event(
    invocation_id: impl Into<String>,
    author: impl Into<String>,
    branch: Option<&str>,
    json_response: impl Into<String>,
) -> Event {
    let mut event = Event::new(
        invocation_id.into(),
        author.into(),
        adk_events::node_info::NodeInfo::new(""),
    );
    event.branch = branch.map(str::to_string);
    event.content = Some(Content::new(
        "model",
        vec![Part::text(json_response.into())],
    ));
    event
}

/// `get_structured_model_response`: if `event` carries a
/// `set_model_response` function response, returns its validated result
/// (`event.actions.set_model_response`) as a JSON string — `None` if the
/// event has no function responses, none is named `set_model_response`,
/// or that action wasn't actually set.
pub fn get_structured_model_response(event: &Event) -> Option<String> {
    if event.get_function_responses().is_empty() {
        return None;
    }
    for function_response in event.get_function_responses() {
        if function_response.name.as_deref() == Some("set_model_response") {
            return event
                .actions
                .set_model_response
                .as_ref()
                .and_then(|response| rusty_serde::json::to_string(response).ok());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::FunctionResponse;
    use rusty_serde::value::Value;
    use std::collections::BTreeMap;

    #[test]
    fn applies_only_when_output_schema_and_tools_are_both_present() {
        assert!(should_apply_output_schema_processor(
            true, true, false, false
        ));
        assert!(!should_apply_output_schema_processor(
            false, true, false, false
        ));
        assert!(!should_apply_output_schema_processor(
            true, false, false, false
        ));
    }

    #[test]
    fn does_not_apply_when_the_model_already_supports_both() {
        assert!(!should_apply_output_schema_processor(
            true, true, true, false
        ));
    }

    #[test]
    fn does_not_apply_in_task_mode() {
        assert!(!should_apply_output_schema_processor(
            true, true, false, true
        ));
    }

    #[test]
    fn create_final_model_response_event_builds_a_plain_text_model_event() {
        let event = create_final_model_response_event(
            "inv-1",
            "my_agent",
            Some("root.child"),
            r#"{"answer":42}"#,
        );
        assert_eq!(event.invocation_id, "inv-1");
        assert_eq!(event.author, "my_agent");
        assert_eq!(event.branch.as_deref(), Some("root.child"));
        assert_eq!(
            event.content.unwrap().parts[0].text.as_deref(),
            Some(r#"{"answer":42}"#)
        );
    }

    fn event_with_set_model_response_call(
        response_name: &str,
        set_model_response: Option<Value>,
    ) -> Event {
        let mut e = Event::new("inv-1", "agent", NodeInfo::new("root"));
        e.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some("id1".to_string()),
                name: Some(response_name.to_string()),
                response: None,
            })],
        ));
        e.actions.set_model_response = set_model_response;
        e
    }

    #[test]
    fn returns_none_without_any_function_responses() {
        let e = Event::new("inv-1", "agent", NodeInfo::new("root"));
        assert!(get_structured_model_response(&e).is_none());
    }

    #[test]
    fn returns_none_when_the_function_response_is_for_a_different_tool() {
        let e = event_with_set_model_response_call("some_other_tool", Some(Value::Null));
        assert!(get_structured_model_response(&e).is_none());
    }

    #[test]
    fn returns_none_when_set_model_response_action_was_never_populated() {
        let e = event_with_set_model_response_call("set_model_response", None);
        assert!(get_structured_model_response(&e).is_none());
    }

    #[test]
    fn returns_the_json_encoded_action_when_present() {
        let mut response = BTreeMap::new();
        response.insert("answer".to_string(), Value::Int(42));
        let e = event_with_set_model_response_call(
            "set_model_response",
            Some(Value::Map(response.into_iter().collect())),
        );
        let json = get_structured_model_response(&e).unwrap();
        assert!(json.contains("42"));
    }
}
