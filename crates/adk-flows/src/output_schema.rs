//! Capability C0178: the `_output_schema_processor` request processor,
//! ported from `google.adk.flows.llm_flows._output_schema_processor`.
//!
//! Handles output schema when tools are also present: some model
//! capabilities can't honor both `output_schema` and a tool list in the
//! same request, so instead a synthetic `set_model_response` tool call is
//! how the model reports its final structured answer.
//!
//! **Now fully ported**: [`apply_output_schema_processor`] does the
//! actual injection — a free function taking `&LlmAgent` directly, the
//! same shape `basic::build_basic_request`/`identity::apply_identity`/
//! `instructions::build_instructions` already use (see those modules'
//! own docs), and — like them — called directly from
//! `LlmFlow::preprocess` rather than through a `BaseLlmRequestProcessor`
//! trait object. This closes the gap an earlier version of this module
//! left open: `SetModelResponseTool` (C0437) and the free-function
//! `append_tools`/`merge_declarations` (C0116, living in `adk-tools`, not
//! `adk-models` — the crate-cycle blocker only ever applied to a *method*
//! on `LlmRequest` itself, never to `adk-flows` calling a free function
//! from a crate it already depends on) were both already built; the
//! remaining gap was purely this wiring.
//!
//! [`should_apply_output_schema_processor`] (the gating decision),
//! [`OUTPUT_SCHEMA_TOOL_INSTRUCTION`], and the two standalone helpers that
//! read back a completed structured response
//! ([`create_final_model_response_event`],
//! [`get_structured_model_response`]) were already ported and are reused
//! by [`apply_output_schema_processor`] unchanged.

use adk_agents::llm_agent::{AgentMode, LlmAgent};
use adk_events::Event;
use adk_genai::content::{Content, Part};
use adk_models::llm_request::{Instructions, LlmRequest};
use adk_tools::append_tools::append_tools;
use adk_tools::set_model_response_tool::SetModelResponseTool;

use crate::canonical_model::{canonical_model, CanonicalModelError};

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

/// `_OutputSchemaRequestProcessor.run_async` — injects the synthetic
/// `set_model_response` tool + its instruction into `llm_request` when
/// [`should_apply_output_schema_processor`] gates true for `agent`. A
/// no-op (not an error) whenever it doesn't apply, matching the source's
/// own early `return` with nothing yielded. Resolves the model itself via
/// [`canonical_model`], the same independent-resolution shape
/// `basic::build_basic_request`'s own `output_schema` gating already
/// uses for the identical capability check — so the two processors'
/// gating decisions always agree with each other.
pub fn apply_output_schema_processor(
    agent: &LlmAgent,
    llm_request: &mut LlmRequest,
) -> Result<(), CanonicalModelError> {
    let Some(output_schema) = agent.output_schema.clone() else {
        return Ok(());
    };
    if agent.tools.is_empty() {
        return Ok(());
    }
    let model = canonical_model(agent)?;
    let is_task_mode = agent.mode == Some(AgentMode::Task);
    if !should_apply_output_schema_processor(
        true,
        true,
        model.capabilities().output_schema_and_tools,
        is_task_mode,
    ) {
        return Ok(());
    }

    let tool = SetModelResponseTool::new(output_schema);
    append_tools(llm_request, &[&tool]);
    llm_request.append_instructions(Instructions::Strings(vec![
        OUTPUT_SCHEMA_TOOL_INSTRUCTION.to_string()
    ]));
    Ok(())
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
                ..Default::default()
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

    use adk_agents::llm_agent::{ModelRef, ToolUnion};

    fn agent_with_schema_and_tools(model: &str) -> LlmAgent {
        let mut agent = LlmAgent::new(ModelRef::Name(model.to_string()));
        agent.output_schema = Some(Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]));
        agent.tools = vec![ToolUnion::Function(Value::Null)];
        agent
    }

    fn function_declaration_names(request: &LlmRequest) -> Vec<String> {
        let Some(Value::Seq(entries)) = &request.config.tools else {
            return Vec::new();
        };
        let mut names = Vec::new();
        for entry in entries {
            let Value::Map(fields) = entry else { continue };
            let Some((_, Value::Seq(declarations))) =
                fields.iter().find(|(k, _)| k == "functionDeclarations")
            else {
                continue;
            };
            for declaration in declarations {
                if let Value::Map(fields) = declaration {
                    if let Some((_, Value::String(name))) = fields.iter().find(|(k, _)| k == "name")
                    {
                        names.push(name.clone());
                    }
                }
            }
        }
        names
    }

    #[test]
    fn apply_output_schema_processor_is_a_no_op_without_an_output_schema() {
        let mut agent = agent_with_schema_and_tools("gemini-2.5-flash");
        agent.output_schema = None;
        let mut request = LlmRequest::new("placeholder");
        apply_output_schema_processor(&agent, &mut request).unwrap();
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn apply_output_schema_processor_is_a_no_op_without_tools() {
        let mut agent = agent_with_schema_and_tools("gemini-2.5-flash");
        agent.tools = Vec::new();
        let mut request = LlmRequest::new("placeholder");
        apply_output_schema_processor(&agent, &mut request).unwrap();
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn apply_output_schema_processor_is_a_no_op_in_task_mode() {
        let mut agent = agent_with_schema_and_tools("gemini-2.5-flash");
        agent.mode = Some(AgentMode::Task);
        let mut request = LlmRequest::new("placeholder");
        apply_output_schema_processor(&agent, &mut request).unwrap();
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn apply_output_schema_processor_injects_the_tool_and_instruction() {
        let agent = agent_with_schema_and_tools("gemini-2.5-flash");
        let mut request = LlmRequest::new("placeholder");
        apply_output_schema_processor(&agent, &mut request).unwrap();

        assert_eq!(
            function_declaration_names(&request),
            vec!["set_model_response".to_string()]
        );
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some(OUTPUT_SCHEMA_TOOL_INSTRUCTION)
        );
    }

    #[test]
    fn apply_output_schema_processor_errors_when_the_model_cannot_be_resolved() {
        let agent = agent_with_schema_and_tools("totally-unknown-model");
        let mut request = LlmRequest::new("placeholder");
        assert!(apply_output_schema_processor(&agent, &mut request).is_err());
    }
}
