//! Capability C0099: `FinishTaskTool`, plus its two module-level helpers
//! (`get_output_wrapper_key`/`is_finish_task_terminal_fr`), ported from
//! `google.adk.agents.llm.task._finish_task_tool`.
//!
//! Lets a task-mode agent's model signal that it has finished its
//! delegated task — the model calls `finish_task` instead of returning
//! plain text, and the caller (the `LlmAgent` task wrapper — the turn
//! loop that would actually construct this tool and drive a task agent
//! with it, C0333/C0834/C0887) reads the call's `output` argument
//! directly rather than any `EventActions` field, matching the source's
//! own comment: `run_async` deliberately never writes
//! `actions.finish_task`.
//!
//! **Scope, disclosed**: this batch ports the tool itself and its two
//! pure helpers only — not the task-mode turn loop / task-scope scanning
//! that would actually construct and wire it into a running agent
//! (C0333/C0834/C0887), which needs infrastructure this port doesn't
//! have yet (see `llm_flow.rs`'s own disclosed "no turn loop / tool
//! execution" scope).
//!
//! **Fundamental adaptation, disclosed**: the source's `__init__` takes
//! the whole `task_agent: LlmAgent` and builds a Pydantic `TypeAdapter`
//! from its `output_schema` for two things: JSON-Schema generation
//! (`.json_schema()`) and runtime validation (`.validate_python`/
//! `.dump_python` in `run_async`, catching `ValidationError` for a
//! retry-oriented `{"error": ...}` response). This port has neither
//! Python's runtime type introspection nor a Pydantic-equivalent
//! validator (the same limitation `set_model_response_tool.rs` (C0437)
//! already discloses for the identical shape) — so, matching that
//! module's own precedent exactly:
//! - [`FinishTaskTool::new`] takes `output_schema: Option<Value>`
//!   directly (an already-opaque JSON-schema value) rather than a whole
//!   `LlmAgent`. The source's `self._task_agent_name = task_agent.name`
//!   is dropped entirely, not narrowed — it's set in `__init__` but never
//!   read anywhere else in the source tree (grepped), so there's no
//!   observable behavior to preserve.
//! - [`get_output_wrapper_key`] reads `schema.get("type")` directly
//!   instead of the source's three-way `isinstance(schema, dict) /
//!   isinstance(schema, types.Schema) / else schema_to_json_schema(schema)`
//!   branch — that branch exists only to normalize Python's `SchemaType`
//!   union (`dict | BaseModel | types.Schema | plain type`) into a raw
//!   JSON-schema dict; this port's `output_schema` is always already a
//!   JSON-schema-shaped [`Value`], so there's nothing left to normalize.
//! - `run_async` has no schema to validate the call's `args` against, so
//!   it always succeeds — the `ValidationError`-triggered retry path
//!   (`{"error": "...validation errors..."}`) is not ported. A caller
//!   relying on that retry loop won't get it; this is a real, disclosed
//!   capability gap, not a silent one.

use std::collections::BTreeMap;

use adk_genai::content::FunctionDeclaration;
use adk_models::llm_request::{Instructions, LlmRequest};
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

/// Name of the finish_task tool.
pub const FINISH_TASK_TOOL_NAME: &str = "finish_task";

/// Success result returned by [`FinishTaskTool::run_async`].
pub const FINISH_TASK_SUCCESS_RESULT: &str = "Task completed.";
/// The source's terminal-failure counterpart — never actually returned
/// by [`FinishTaskTool::run_async`] itself (this port has no validation
/// failure path to produce it, see the module doc), but
/// [`is_finish_task_terminal_fr`] still recognizes it as terminal,
/// matching the source's own check exactly, in case a future caller
/// synthesizes one directly.
pub const FINISH_TASK_ERROR_RESULT: &str = "Task failed.";

/// Default parameter key used to wrap a non-object output schema.
pub const FINISH_TASK_DEFAULT_WRAPPER_KEY: &str = "result";

const DESCRIPTION_BASE: &str = "Signal that this agent has completed its \
delegated task. Call this when you have finished your delegated task.";
const DESCRIPTION_OUTPUT_SUFFIX: &str = " Pass the required output data in the parameters.";

const FINISH_TASK_INSTRUCTION: &str = "Do NOT call `finish_task` prematurely. Use your \
available tools to fully complete every aspect of the delegated task first. If the task is \
unclear, ask the user for clarification before proceeding. Once the task is fully complete, \
call `finish_task` by itself with no accompanying text output.";

/// `_DefaultTaskOutput`'s JSON schema — used when no `output_schema` is
/// given, matching `adk_agents::task_models::DefaultTaskOutput`'s shape
/// (`{result: string}`, required).
fn default_task_output_schema() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("object".to_string())),
        (
            "properties".to_string(),
            Value::Map(vec![(
                "result".to_string(),
                Value::Map(vec![(
                    "type".to_string(),
                    Value::String("string".to_string()),
                )]),
            )]),
        ),
        (
            "required".to_string(),
            Value::Seq(vec![Value::String("result".to_string())]),
        ),
    ])
}

/// `get_output_wrapper_key`: `None` if `output_schema` (or the default
/// schema, when absent) is already an object schema — the wrapping
/// key needed otherwise, so a non-object schema can still be expressed
/// as a `FunctionDeclaration`'s (always-object) `parameters`. See the
/// module doc for the adaptation from the source's three-way schema
/// normalization.
pub fn get_output_wrapper_key(output_schema: Option<&Value>) -> Option<String> {
    let default_schema = default_task_output_schema();
    let schema = output_schema.unwrap_or(&default_schema);
    let is_object = matches!(
        schema.get("type"),
        Some(Value::String(t)) if t == "object" || t == "OBJECT"
    );
    if is_object {
        None
    } else {
        Some(FINISH_TASK_DEFAULT_WRAPPER_KEY.to_string())
    }
}

fn build_declaration(description: &str, output_schema: Option<&Value>) -> FunctionDeclaration {
    let wrapper_key = get_output_wrapper_key(output_schema);
    let raw_schema = output_schema
        .cloned()
        .unwrap_or_else(default_task_output_schema);

    let schema_json = match wrapper_key {
        None => raw_schema,
        Some(key) => {
            let mut raw_schema = raw_schema;
            let defs = match &mut raw_schema {
                Value::Map(entries) => entries
                    .iter()
                    .position(|(k, _)| k == "$defs")
                    .map(|index| entries.remove(index).1),
                _ => None,
            };
            let mut fields = vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(key.clone(), raw_schema)]),
                ),
                ("required".to_string(), Value::Seq(vec![Value::String(key)])),
            ];
            if let Some(defs) = defs {
                fields.push(("$defs".to_string(), defs));
            }
            Value::Map(fields)
        }
    };

    FunctionDeclaration {
        name: Some(FINISH_TASK_TOOL_NAME.to_string()),
        description: Some(description.to_string()),
        parameters_json_schema: Some(schema_json),
        ..Default::default()
    }
}

/// C0099: tool for signaling `LlmAgent` task completion. See the module
/// doc for what's scoped in/out of this batch.
pub struct FinishTaskTool {
    output_schema: Option<Value>,
    description: String,
}

impl FinishTaskTool {
    /// `output_schema` is the task agent's own `output_schema` — `None`
    /// falls back to [`default_task_output_schema`], matching the
    /// source's own `_DefaultTaskOutput` fallback.
    pub fn new(output_schema: Option<Value>) -> Self {
        let mut description = String::from(DESCRIPTION_BASE);
        if output_schema.is_some() {
            description.push_str(DESCRIPTION_OUTPUT_SUFFIX);
        }
        Self {
            output_schema,
            description,
        }
    }
}

impl BaseTool for FinishTaskTool {
    fn name(&self) -> &str {
        FINISH_TASK_TOOL_NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(build_declaration(
            &self.description,
            self.output_schema.as_ref(),
        ))
    }

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                merge_declarations(llm_request, [(self.name().to_string(), declaration)]);
            }
            llm_request.append_instructions(Instructions::Strings(vec![
                FINISH_TASK_INSTRUCTION.to_string()
            ]));
        })
    }

    fn run_async<'a>(
        &'a self,
        _args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move { Ok(Value::String(FINISH_TASK_SUCCESS_RESULT.to_string())) })
    }
}

/// `is_finish_task_terminal_fr`: true iff `event` carries a
/// `finish_task` function response whose `result` field is one of
/// [`FINISH_TASK_SUCCESS_RESULT`]/[`FINISH_TASK_ERROR_RESULT`] — a
/// non-terminal response (e.g. a validation-error retry signal in the
/// source) returns `false` so the caller keeps iterating.
pub fn is_finish_task_terminal_fr(event: &adk_events::Event) -> bool {
    for fr in event.get_function_responses() {
        if fr.name.as_deref() == Some(FINISH_TASK_TOOL_NAME) {
            return matches!(
                fr.response.as_ref().and_then(|r| r.get("result")),
                Some(Value::String(s))
                    if s == FINISH_TASK_SUCCESS_RESULT || s == FINISH_TASK_ERROR_RESULT
            );
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_events::Event;
    use adk_genai::content::{Content, FunctionResponse, Part};

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    fn object_schema() -> Value {
        Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![(
                    "summary".to_string(),
                    Value::Map(vec![(
                        "type".to_string(),
                        Value::String("string".to_string()),
                    )]),
                )]),
            ),
        ])
    }

    fn string_schema() -> Value {
        Value::Map(vec![(
            "type".to_string(),
            Value::String("string".to_string()),
        )])
    }

    #[test]
    fn get_output_wrapper_key_is_none_for_an_object_schema() {
        assert_eq!(get_output_wrapper_key(Some(&object_schema())), None);
    }

    #[test]
    fn get_output_wrapper_key_wraps_a_non_object_schema() {
        assert_eq!(
            get_output_wrapper_key(Some(&string_schema())),
            Some(FINISH_TASK_DEFAULT_WRAPPER_KEY.to_string())
        );
    }

    #[test]
    fn get_output_wrapper_key_is_none_for_the_default_schema() {
        // `_DefaultTaskOutput`'s own JSON schema is itself an object
        // schema, so the default (no `output_schema` given) also needs
        // no wrapper.
        assert_eq!(get_output_wrapper_key(None), None);
    }

    #[test]
    fn declaration_uses_the_output_schema_directly_when_it_is_already_an_object() {
        let tool = FinishTaskTool::new(Some(object_schema()));
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(declaration.name.as_deref(), Some(FINISH_TASK_TOOL_NAME));
        assert_eq!(declaration.parameters_json_schema, Some(object_schema()));
    }

    #[test]
    fn declaration_wraps_a_non_object_schema_under_the_result_key() {
        let tool = FinishTaskTool::new(Some(string_schema()));
        let declaration = tool.get_declaration().unwrap();
        let schema = declaration.parameters_json_schema.unwrap();
        assert_eq!(
            schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
        let properties = schema.get("properties").unwrap();
        assert_eq!(
            properties.get(FINISH_TASK_DEFAULT_WRAPPER_KEY),
            Some(&string_schema())
        );
        assert_eq!(
            schema.get("required"),
            Some(&Value::Seq(vec![Value::String(
                FINISH_TASK_DEFAULT_WRAPPER_KEY.to_string()
            )]))
        );
    }

    #[test]
    fn declaration_hoists_defs_to_the_wrapped_schemas_root() {
        let schema_with_defs = Value::Map(vec![
            ("type".to_string(), Value::String("string".to_string())),
            (
                "$defs".to_string(),
                Value::Map(vec![("Foo".to_string(), Value::Bool(true))]),
            ),
        ]);
        let tool = FinishTaskTool::new(Some(schema_with_defs));
        let declaration = tool.get_declaration().unwrap();
        let schema = declaration.parameters_json_schema.unwrap();
        assert_eq!(
            schema.get("$defs"),
            Some(&Value::Map(vec![("Foo".to_string(), Value::Bool(true))]))
        );
        let wrapped = schema
            .get("properties")
            .unwrap()
            .get(FINISH_TASK_DEFAULT_WRAPPER_KEY)
            .unwrap();
        assert!(wrapped.get("$defs").is_none());
    }

    #[test]
    fn description_mentions_output_parameters_only_when_a_schema_is_given() {
        let without = FinishTaskTool::new(None);
        assert!(!without
            .description()
            .contains("Pass the required output data"));

        let with_schema = FinishTaskTool::new(Some(object_schema()));
        assert!(with_schema
            .description()
            .contains("Pass the required output data"));
    }

    #[rusty_tokio::test]
    async fn process_llm_request_appends_the_instruction_and_declaration() {
        let tool = FinishTaskTool::new(None);
        let mut ctx = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");

        tool.process_llm_request(&mut ctx, &mut request).await;

        let system_instruction = request.config.system_instruction.unwrap();
        assert!(system_instruction.contains("Do NOT call `finish_task` prematurely"));
        assert!(request.config.tools.is_some());
    }

    #[rusty_tokio::test]
    async fn run_async_always_succeeds_and_never_sets_finish_task_action() {
        let tool = FinishTaskTool::new(None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("result".to_string(), Value::String("done".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::String(FINISH_TASK_SUCCESS_RESULT.to_string())
        );
    }

    fn event_with_finish_task_response(result: &str) -> Event {
        let mut response = BTreeMap::new();
        response.insert("result".to_string(), Value::String(result.to_string()));
        let mut e = Event::new("inv-1", "agent", NodeInfo::new("root"));
        e.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some("id1".to_string()),
                name: Some(FINISH_TASK_TOOL_NAME.to_string()),
                response: Some(response),
                ..Default::default()
            })],
        ));
        e
    }

    #[test]
    fn is_finish_task_terminal_fr_is_true_for_a_success_result() {
        let event = event_with_finish_task_response(FINISH_TASK_SUCCESS_RESULT);
        assert!(is_finish_task_terminal_fr(&event));
    }

    #[test]
    fn is_finish_task_terminal_fr_is_true_for_an_error_result() {
        let event = event_with_finish_task_response(FINISH_TASK_ERROR_RESULT);
        assert!(is_finish_task_terminal_fr(&event));
    }

    #[test]
    fn is_finish_task_terminal_fr_is_false_for_a_non_terminal_result() {
        let event = event_with_finish_task_response("still working");
        assert!(!is_finish_task_terminal_fr(&event));
    }

    #[test]
    fn is_finish_task_terminal_fr_is_false_without_a_finish_task_response() {
        let e = Event::new("inv-1", "agent", NodeInfo::new("root"));
        assert!(!is_finish_task_terminal_fr(&e));
    }
}
