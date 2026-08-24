//! Capability C0492: `request_input`/`_request_input_tool.py`, ported from
//! `google.adk.tools._request_input_tool`.
//!
//! A [`LongRunningFunctionTool`] that asks the user a question and
//! suspends the current turn to wait for their response — the same
//! long-running-interrupt mechanism [`crate::get_user_choice_tool`]
//! (C0421) uses, generalized to an arbitrary free-text/structured prompt
//! instead of a fixed options list.
//!
//! **Forward reference, disclosed**: the source imports
//! `REQUEST_INPUT_FUNCTION_CALL_NAME` from
//! `flows.llm_flows.functions` and renames the wrapped function to it
//! (`_request_input_func.__name__ = REQUEST_INPUT_FUNCTION_CALL_NAME`) so
//! the function-call name the model sees, and the name other modules
//! (`remote_a2a_agent.py`, `mcp_tool.py`, `cli/api_server.py`) match
//! against, are the same constant. `adk-flows::functions` (C0191/C0192,
//! already ported) hasn't defined this constant yet — its own HITL/
//! request-input wiring is a separate, not-yet-ported capability — and
//! `adk-tools` doesn't depend on `adk-flows` (see this crate's top-level
//! module doc for why: the dependency points the other way today). So
//! [`REQUEST_INPUT_FUNCTION_CALL_NAME`] is defined here instead, as the
//! single source of truth this port currently has for that string; a
//! follow-up batch wiring `adk-flows::functions`'s own request-input
//! interrupt handling should reuse this constant's value (`"adk_request_input"`)
//! rather than defining a second copy.
//!
//! **Not ported**: the source's `logging.info` call on each invocation —
//! no logging framework is adopted by this workspace yet (the same
//! disclosed omission as `preload_memory_tool.rs`'s dropped
//! `logging.warning`).

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::function_tool::FunctionTool;
use crate::long_running_tool::LongRunningFunctionTool;
use crate::tool_context::ToolContext;

/// The function-call name the model must use to invoke this tool — see
/// the module doc for why this port defines it here rather than in
/// `adk-flows::functions`.
pub const REQUEST_INPUT_FUNCTION_CALL_NAME: &str = "adk_request_input";

const DESCRIPTION: &str = "Ask the user a question and wait for their response.\n\nUse this when you need clarification or additional information before proceeding.\n\nReturns:\n  None. Long-running tools return None to signal that the execution should pause and wait for user input.";

/// C0492: asks the user a question. Always returns [`Value::Null`] —
/// returning `None`/`Null` is what triggers the long-running-tool
/// interruption mechanism, matching the source's `_request_input_func`
/// exactly (the `message`/`response_schema` arguments are only read by
/// the caller that suspends the turn and later resumes it with the
/// user's answer; this function itself does nothing with them).
pub fn request_input(_args: &BTreeMap<String, Value>, _tool_context: &mut ToolContext) -> Value {
    Value::Null
}

fn parameters_schema() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("object".to_string())),
        (
            "properties".to_string(),
            Value::Map(vec![
                (
                    "message".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String(
                                "The question or prompt to display to the user.".to_string(),
                            ),
                        ),
                    ]),
                ),
                (
                    "response_schema".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("object".to_string())),
                        (
                            "description".to_string(),
                            Value::String(
                                "JSON Schema describing the expected response format. Use \
                                 {\"type\": \"string\"} for free-text, {\"type\": \"boolean\"} \
                                 for yes/no, or a structured object schema for complex input."
                                    .to_string(),
                            ),
                        ),
                    ]),
                ),
            ]),
        ),
        (
            "required".to_string(),
            Value::Seq(vec![Value::String("message".to_string())]),
        ),
    ])
}

/// C0492: `request_input` — a [`LongRunningFunctionTool`] wrapping
/// [`request_input`](fn@request_input).
pub fn request_input_tool() -> LongRunningFunctionTool {
    LongRunningFunctionTool::new(FunctionTool::new(
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        DESCRIPTION,
        FunctionDeclaration {
            name: Some(REQUEST_INPUT_FUNCTION_CALL_NAME.to_string()),
            description: Some(DESCRIPTION.to_string()),
            parameters: Some(parameters_schema()),
            ..Default::default()
        },
        vec!["message".to_string()],
        Arc::new(|args, ctx| {
            let value = request_input(args, ctx);
            Box::pin(async move { value })
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_tool::BaseTool;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[test]
    fn request_input_returns_null() {
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "message".to_string(),
            Value::String("What is your name?".to_string()),
        );
        assert_eq!(request_input(&args, &mut context), Value::Null);
    }

    #[test]
    fn request_input_tool_has_the_expected_name_and_is_long_running() {
        let tool = request_input_tool();
        assert_eq!(tool.name(), "adk_request_input");
        assert!(tool.is_long_running());
        assert!(tool
            .description()
            .contains("Ask the user a question and wait for their response"));
    }

    #[test]
    fn get_declaration_exposes_message_and_response_schema_parameters() {
        let tool = request_input_tool();
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(declaration.name.as_deref(), Some("adk_request_input"));
        match declaration.parameters.unwrap() {
            Value::Map(fields) => {
                let properties = fields
                    .iter()
                    .find(|(k, _)| k == "properties")
                    .map(|(_, v)| v)
                    .unwrap();
                match properties {
                    Value::Map(props) => {
                        assert!(props.iter().any(|(k, _)| k == "message"));
                        assert!(props.iter().any(|(k, _)| k == "response_schema"));
                    }
                    other => panic!("expected a properties map, got {other:?}"),
                }
                let required = fields
                    .iter()
                    .find(|(k, _)| k == "required")
                    .map(|(_, v)| v)
                    .unwrap();
                assert_eq!(
                    required,
                    &Value::Seq(vec![Value::String("message".to_string())])
                );
            }
            other => panic!("expected a parameters map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_async_returns_null_when_message_is_present() {
        let tool = request_input_tool();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "message".to_string(),
            Value::String("What is your name?".to_string()),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Null);
    }

    #[rusty_tokio::test]
    async fn run_async_accepts_a_response_schema_argument() {
        let tool = request_input_tool();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "message".to_string(),
            Value::String("Enter your username:".to_string()),
        );
        args.insert(
            "response_schema".to_string(),
            Value::Map(vec![(
                "type".to_string(),
                Value::String("string".to_string()),
            )]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Null);
    }

    #[rusty_tokio::test]
    async fn run_async_reports_a_missing_mandatory_message_as_an_error() {
        let tool = request_input_tool();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "response_schema".to_string(),
            Value::Map(vec![(
                "type".to_string(),
                Value::String("string".to_string()),
            )]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields
                    .iter()
                    .find(|(k, _)| k == "error")
                    .map(|(_, v)| v)
                    .unwrap();
                match error {
                    Value::String(message) => {
                        assert!(message.contains("mandatory input parameters are not present"))
                    }
                    other => panic!("expected an error string, got {other:?}"),
                }
            }
            other => panic!("expected an error map, got {other:?}"),
        }
    }
}
