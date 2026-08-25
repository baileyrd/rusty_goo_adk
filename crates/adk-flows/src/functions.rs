//! Capability C0191/C0192 (partial): the core of `functions.py`'s
//! function-call execution pipeline, ported from
//! `google.adk.flows.llm_flows.functions`.
//!
//! **Scope**: resolving a tool by name, building its `ToolContext`,
//! invoking it, and building the resulting function-response `Event` —
//! for both a single call and a parallel batch (real concurrency via
//! `rusty_tokio::spawn`, matching the precedent already established by
//! `adk_agents::parallel_agent::ParallelAgent`). Reuses
//! [`crate::functions_utils::merge_parallel_function_response_events`]
//! (already built) for the merge step rather than reimplementing it.
//!
//! **Not ported this batch** (all disclosed, not silently dropped):
//! - Tool-level plugin/canonical callback dispatch (`before_tool_callback`/
//!   `after_tool_callback`/`on_tool_error_callback`, both the
//!   `PluginManager` and `LlmAgent.canonical_*_tool_callbacks` halves) —
//!   `adk-agents::services::BasePlugin` deliberately excludes tool-level
//!   hooks (needs `BaseTool` from `adk-tools`, which already depends on
//!   `adk-agents` — the same crate-graph constraint disclosed in Phase 7
//!   batch 1's module doc), and `LlmAgent` has no canonical tool-callback
//!   resolution built yet either. A tool error here simply propagates as
//!   `Err`, rather than the source's callback-mediated recovery path.
//! - Auth-request/tool-confirmation-request event synthesis, and the
//!   long-running/`_defers_response` empty-response skip — the first
//!   needs `AuthConfig` (Phase 9, not built, the same gap
//!   `functions_utils.rs` already discloses); the second needs a way for
//!   [`adk_tools::base_tool::BaseTool::run_async`] to signal "no response
//!   yet" distinct from a real `Value` result, which its current
//!   `Result<Value, ToolError>` contract doesn't carry — flagged as a
//!   design gap to revisit, not silently narrowed.
//! - Computer-use image decoding (`_try_decode_computer_use_image`) —
//!   `ComputerUseTool` doesn't exist in this port yet, so there's nothing
//!   to special-case against. Multimodal-part extraction itself (C0195)
//!   is now built — see [`crate::functions_media`], wired into
//!   [`build_function_response_content`] below. The `AgentTool`
//!   skip-summarization display-text special case (source lines
//!   1449-1482, appending a displayable-text `Part` when
//!   `skip_summarization` is set and the tool is an `AgentTool`) is a
//!   separate capability, not yet picked up — `AgentTool` (C0406) does
//!   exist now, so this is no longer blocked, just unbuilt.
//! - `response_scheduling` forwarding onto the built `FunctionResponse` —
//!   `adk_genai::content::FunctionResponse` doesn't model a `scheduling`
//!   field yet (an already-established "opaque unless something reads
//!   it" narrowing for that type).
//! - OTel tracing/telemetry (`_instrumentation.record_tool_execution`,
//!   error-type detection) — Phase 12, not built anywhere in this port.
//! - Deep-copying `function_call.args` before use — this port's
//!   `BTreeMap<String, Value>` is already an owned clone by the time it
//!   reaches a tool (built fresh from the `FunctionCall`, never aliased
//!   with anything the caller can mutate concurrently), so there's
//!   nothing for a defensive copy to protect against here.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use adk_agents::invocation_context::InvocationContext;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};
use adk_tools::base_tool::{BaseTool, ToolError};
use adk_tools::tool_confirmation::ToolConfirmation;
use adk_tools::tool_context::ToolContext;
use rusty_serde::value::Value;

use crate::functions_media::extract_multimodal_parts;
use crate::functions_utils::merge_parallel_function_response_events;

/// `tools_dict: dict[str, BaseTool]` — resolves a function call's name to
/// the tool that answers it.
pub type ToolsDict = HashMap<String, Arc<dyn BaseTool>>;

#[derive(Debug, rusty_err::Error)]
pub enum FunctionExecutionError {
    #[error("Tool '{0}' not found.")]
    ToolNotFound(String),
    #[error("function call has no name")]
    MissingToolName,
    #[error("{0}")]
    ToolRun(ToolError),
    #[error("a function-call task panicked or was cancelled")]
    TaskFailed,
    #[error("{0}")]
    Merge(crate::functions_utils::FunctionsError),
}

/// `_get_tool`: resolves `function_call`'s named tool out of `tools_dict`.
pub fn get_tool(
    function_call: &FunctionCall,
    tools_dict: &ToolsDict,
) -> Result<Arc<dyn BaseTool>, FunctionExecutionError> {
    let name = function_call
        .name
        .as_deref()
        .ok_or(FunctionExecutionError::MissingToolName)?;
    tools_dict
        .get(name)
        .cloned()
        .ok_or_else(|| FunctionExecutionError::ToolNotFound(name.to_string()))
}

/// `_create_tool_context`: builds the [`ToolContext`] (`= Context`) a tool
/// runs against — stamped with the answering function call's id and any
/// confirmation the caller already resolved for it.
pub fn create_tool_context(
    invocation_context: &InvocationContext,
    function_call: &FunctionCall,
    tool_confirmation: Option<&ToolConfirmation>,
) -> ToolContext {
    let mut ctx = ToolContext::new(invocation_context.clone());
    ctx.set_function_call_id(function_call.id.clone());
    if let Some(confirmation) = tool_confirmation {
        // `rusty_serde::json::to_value` on an already-validated
        // `ToolConfirmation` cannot fail in practice; `Context` stores it
        // as an opaque `Value` (see the module doc on C0405's own field).
        if let Ok(value) = rusty_serde::json::to_value(confirmation) {
            ctx.set_tool_confirmation(Some(value));
        }
    }
    ctx
}

/// `_build_function_response_content` (narrowed — see the module doc):
/// wraps a tool's raw result as the `Content` carrying its
/// `FunctionResponse`. Media the result carried (C0195, see
/// `crate::functions_media`) is pulled out first and attached to
/// `FunctionResponse::parts` — only the remainder is coerced to a dict.
fn build_function_response_content(
    tool: &dyn BaseTool,
    function_result: Value,
    function_call_id: Option<&str>,
) -> Content {
    let (remaining_result, function_response_parts) = extract_multimodal_parts(function_result);

    // "Specs requires the result to be a dict" — a non-map result is
    // wrapped as `{"result": ...}`, matching the source exactly.
    let response = match remaining_result {
        Value::Map(fields) => fields.into_iter().collect::<BTreeMap<_, _>>(),
        other => BTreeMap::from([("result".to_string(), other)]),
    };
    let function_response = FunctionResponse {
        id: function_call_id.map(str::to_string),
        name: Some(tool.name().to_string()),
        response: Some(response),
        parts: (!function_response_parts.is_empty()).then_some(function_response_parts),
    };
    Content {
        role: Some("user".to_string()),
        parts: vec![Part {
            function_response: Some(function_response),
            ..Default::default()
        }],
    }
}

/// `_execute_single_function_call_async` (narrowed — see the module
/// doc): resolves the tool, runs it, and builds the resulting
/// function-response event.
pub async fn execute_single_function_call(
    invocation_context: &InvocationContext,
    function_call: &FunctionCall,
    tools_dict: &ToolsDict,
    agent_name: &str,
    tool_confirmation: Option<&ToolConfirmation>,
) -> Result<Event, FunctionExecutionError> {
    let tool = get_tool(function_call, tools_dict)?;
    let mut tool_context =
        create_tool_context(invocation_context, function_call, tool_confirmation);
    let args = function_call.args.clone().unwrap_or_default();

    let function_result = tool
        .run_async(&args, &mut tool_context)
        .await
        .map_err(FunctionExecutionError::ToolRun)?;

    let content = build_function_response_content(
        tool.as_ref(),
        function_result,
        function_call.id.as_deref(),
    );

    let mut event = Event::new(
        invocation_context.invocation_id.clone(),
        agent_name,
        NodeInfo::new(""),
    );
    event.branch = invocation_context.branch.clone();
    event.content = Some(content);
    event.actions = tool_context.into_actions();
    Ok(event)
}

/// `handle_function_call_list_async` (narrowed — see the module doc):
/// runs every (optionally filtered) call concurrently, merging the
/// resulting events into one via
/// [`merge_parallel_function_response_events`]. Returns `None` if
/// filtering leaves nothing to run — matching the source's own
/// `not filtered_calls -> return None`.
pub async fn execute_function_calls(
    invocation_context: &InvocationContext,
    function_calls: &[FunctionCall],
    tools_dict: &ToolsDict,
    agent_name: &str,
    filters: Option<&HashSet<String>>,
    tool_confirmations: Option<&HashMap<String, ToolConfirmation>>,
) -> Result<Option<Event>, FunctionExecutionError> {
    let filtered: Vec<FunctionCall> = function_calls
        .iter()
        .filter(|fc| match filters {
            None => true,
            Some(f) if f.is_empty() => true,
            Some(f) => fc.id.as_deref().is_some_and(|id| f.contains(id)),
        })
        .cloned()
        .collect();
    if filtered.is_empty() {
        return Ok(None);
    }

    let mut handles = Vec::with_capacity(filtered.len());
    for function_call in filtered {
        let invocation_context = invocation_context.clone();
        let tools_dict = tools_dict.clone();
        let agent_name = agent_name.to_string();
        let tool_confirmation = tool_confirmations.and_then(|confirmations| {
            function_call
                .id
                .as_deref()
                .and_then(|id| confirmations.get(id))
                .cloned()
        });
        handles.push(rusty_tokio::spawn(async move {
            execute_single_function_call(
                &invocation_context,
                &function_call,
                &tools_dict,
                &agent_name,
                tool_confirmation.as_ref(),
            )
            .await
        }));
    }

    let mut events = Vec::with_capacity(handles.len());
    for handle in handles {
        let event = handle
            .await
            .map_err(|_| FunctionExecutionError::TaskFailed)??;
        events.push(event);
    }

    let merged =
        merge_parallel_function_response_events(&events).map_err(FunctionExecutionError::Merge)?;
    Ok(Some(merged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_tools::base_tool::BoxFuture;
    use std::collections::BTreeMap as StdBTreeMap;

    struct EchoTool;
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes its `value` argument"
        }
        fn run_async<'a>(
            &'a self,
            args: &'a StdBTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> BoxFuture<'a, Result<Value, ToolError>> {
            let value = args.get("value").cloned().unwrap_or(Value::Null);
            Box::pin(async move { Ok(value) })
        }
    }

    struct MediaTool;
    impl BaseTool for MediaTool {
        fn name(&self) -> &str {
            "media"
        }
        fn description(&self) -> &str {
            "returns a result carrying an inline image"
        }
        fn run_async<'a>(
            &'a self,
            _args: &'a StdBTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> BoxFuture<'a, Result<Value, ToolError>> {
            let image = rusty_serde::json::to_value(&Part {
                inline_data: Some(adk_genai::content::MediaBlobStub {
                    mime_type: Some("image/png".to_string()),
                    rest: Some(Value::Map(vec![(
                        "data".to_string(),
                        Value::String("base64data".to_string()),
                    )])),
                }),
                ..Default::default()
            })
            .unwrap();
            let result = Value::Map(vec![
                ("caption".to_string(), Value::String("a chart".to_string())),
                ("image".to_string(), image),
            ]);
            Box::pin(async move { Ok(result) })
        }
    }

    struct FailingTool;
    impl BaseTool for FailingTool {
        fn name(&self) -> &str {
            "failing"
        }
        fn description(&self) -> &str {
            "always not-implemented"
        }
    }

    fn tools_dict() -> ToolsDict {
        let mut map: ToolsDict = HashMap::new();
        map.insert("echo".to_string(), Arc::new(EchoTool));
        map.insert("failing".to_string(), Arc::new(FailingTool));
        map.insert("media".to_string(), Arc::new(MediaTool));
        map
    }

    fn ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    fn call(id: &str, name: &str, args: Option<StdBTreeMap<String, Value>>) -> FunctionCall {
        FunctionCall {
            partial_args: None,
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            args,
            will_continue: None,
        }
    }

    #[test]
    fn get_tool_finds_a_registered_tool_by_name() {
        let fc = call("id1", "echo", None);
        let tool = get_tool(&fc, &tools_dict()).unwrap();
        assert_eq!(tool.name(), "echo");
    }

    #[test]
    fn get_tool_errors_on_an_unregistered_name() {
        let fc = call("id1", "nope", None);
        let err = get_tool(&fc, &tools_dict()).err().unwrap();
        assert!(matches!(err, FunctionExecutionError::ToolNotFound(name) if name == "nope"));
    }

    #[rusty_tokio::test]
    async fn execute_single_function_call_runs_the_tool_and_builds_a_response_event() {
        let mut args = StdBTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));
        let fc = call("fc-1", "echo", Some(args));

        let event = execute_single_function_call(&ctx(), &fc, &tools_dict(), "agent", None)
            .await
            .unwrap();

        let response = event.content.unwrap().parts[0]
            .function_response
            .clone()
            .unwrap();
        assert_eq!(response.id.as_deref(), Some("fc-1"));
        assert_eq!(response.name.as_deref(), Some("echo"));
        assert_eq!(
            response.response.unwrap().get("result"),
            Some(&Value::String("hi".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn execute_single_function_call_extracts_media_into_the_response_parts() {
        // C0195: a tool result carrying media (nested inside a plain
        // dict, here) ends up on `FunctionResponse::parts`, and the media
        // entry itself is removed from `response` — only `caption` (the
        // non-media sibling key) remains.
        let fc = call("fc-1", "media", None);

        let event = execute_single_function_call(&ctx(), &fc, &tools_dict(), "agent", None)
            .await
            .unwrap();

        let response = event.content.unwrap().parts[0]
            .function_response
            .clone()
            .unwrap();
        let parts = response.parts.unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].inline_data.as_ref().unwrap().mime_type.as_deref(),
            Some("image/png")
        );
        let remaining = response.response.unwrap();
        assert_eq!(
            remaining.get("caption"),
            Some(&Value::String("a chart".to_string()))
        );
        assert!(!remaining.contains_key("image"));
    }

    #[rusty_tokio::test]
    async fn execute_single_function_call_propagates_a_tool_error() {
        let fc = call("fc-1", "failing", None);
        let err = execute_single_function_call(&ctx(), &fc, &tools_dict(), "agent", None)
            .await
            .unwrap_err();
        assert!(matches!(err, FunctionExecutionError::ToolRun(_)));
    }

    #[rusty_tokio::test]
    async fn execute_function_calls_returns_none_for_an_empty_call_list() {
        let result = execute_function_calls(&ctx(), &[], &tools_dict(), "agent", None, None)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn execute_function_calls_filters_by_id() {
        let mut args1 = StdBTreeMap::new();
        args1.insert("value".to_string(), Value::String("a".to_string()));
        let mut args2 = StdBTreeMap::new();
        args2.insert("value".to_string(), Value::String("b".to_string()));
        let calls = vec![
            call("fc-1", "echo", Some(args1)),
            call("fc-2", "echo", Some(args2)),
        ];
        let filters: HashSet<String> = ["fc-1".to_string()].into_iter().collect();

        let merged =
            execute_function_calls(&ctx(), &calls, &tools_dict(), "agent", Some(&filters), None)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(merged.content.unwrap().parts.len(), 1);
    }

    #[rusty_tokio::test]
    async fn execute_function_calls_runs_every_call_and_merges_the_results() {
        let mut args1 = StdBTreeMap::new();
        args1.insert("value".to_string(), Value::String("a".to_string()));
        let mut args2 = StdBTreeMap::new();
        args2.insert("value".to_string(), Value::String("b".to_string()));
        let calls = vec![
            call("fc-1", "echo", Some(args1)),
            call("fc-2", "echo", Some(args2)),
        ];

        let merged = execute_function_calls(&ctx(), &calls, &tools_dict(), "agent", None, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(merged.content.unwrap().parts.len(), 2);
    }
}
