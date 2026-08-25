//! Capability C0172 (partial): the `request_confirmation` request
//! processor, ported from `google.adk.flows.llm_flows.request_confirmation`.
//!
//! Handles the round-trip for tools that require human confirmation
//! before running: a tool asks for confirmation via an
//! `adk_request_confirmation` function call, the user's approval/denial
//! comes back as a function response, and once approved the original
//! tool call is re-executed.
//!
//! **Now ported in full except `tools_dict` auto-resolution**:
//! [`resolve_confirmation_targets`] (`_resolve_confirmation_targets`) and
//! [`process_request_confirmations`] (the processor's own `run_async`,
//! Steps 1-4) both turned out to be implementable now — `BaseTool`/
//! `ToolConfirmation`/`ToolContext` all exist in `adk-tools` (which
//! `adk-flows` already depends on), and `functions::execute_function_calls`
//! is the already-built equivalent of `handle_function_call_list_async`
//! this row's own module doc previously claimed was missing (a stale
//! claim, the same pattern `output_schema.rs`'s C0178 correction already
//! found). Both new functions take `tools_dict: &ToolsDict` as a plain
//! caller-supplied parameter, the same "caller supplies the resolved
//! bits" adaptation `agent_transfer.rs::{get_transfer_targets,
//! get_agent_to_run}` already established for `llm_mode`/`current_agent`.
//!
//! **Still `Partial:`, and NOT wired into `LlmFlow::preprocess`**: the
//! source auto-builds `tools_dict` from `agent.canonical_tools()` (plus a
//! synthesized transfer tool) before calling `_resolve_confirmation_targets`.
//! `LlmAgent` has no `canonical_tools` resolution built yet (needs the
//! `BaseAgent`/`LlmAgent` tree fusion, C0092 — the same standing-blocked
//! row `agent_transfer.rs` is itself blocked on) — wiring
//! [`process_request_confirmations`] into `preprocess` today would only
//! ever see an empty `tools_dict`, which (correctly, per the ported
//! validation) would turn every confirmation attempt into a "tool is not
//! registered" error instead of the source's real behavior. Left
//! unwired, ready for a future C0092-unblocking batch to call directly
//! once it can build a real `tools_dict`.
//!
//! [`get_original_function_call_args`] (extracting the
//! `originalFunctionCall` payload out of an `adk_request_confirmation`
//! call's args) and [`map_confirmation_to_original_fc_ids`] (the cheap
//! validation-free dedup pre-pass) were already ported and are reused
//! unchanged.

use adk_agents::invocation_context::InvocationContext;
use adk_events::Event;
use adk_genai::content::FunctionCall;
use adk_tools::tool_confirmation::{ToolConfirmation, ToolConfirmationError};
use rusty_serde::value::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::contents::REQUEST_CONFIRMATION_FUNCTION_CALL_NAME;
use crate::functions::{
    create_tool_context, execute_function_calls, FunctionExecutionError, ToolsDict,
};

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

/// `_parse_tool_confirmation`.
pub fn parse_tool_confirmation(
    response: &BTreeMap<String, Value>,
) -> Result<ToolConfirmation, ToolConfirmationError> {
    ToolConfirmation::from_response_dict(response)
}

/// Raised by [`resolve_confirmation_targets`] — matches the source's
/// `raise ValueError(...)` messages verbatim.
#[derive(Debug, rusty_err::Error)]
pub enum ResolveConfirmationTargetsError {
    #[error("Original function call ID is missing.")]
    MissingOriginalFunctionCallId,
    #[error("Original function call name is missing.")]
    MissingOriginalFunctionCallName,
    #[error("could not parse the original function call payload: {0}")]
    MalformedOriginalFunctionCall(String),
    #[error("Original function call for ID '{0}' not found in session history.")]
    NotFoundInHistory(String),
    #[error("Tool '{0}' is not registered.")]
    ToolNotRegistered(String),
    #[error("Tool '{0}' does not require confirmation.")]
    ToolDoesNotRequireConfirmation(String),
    #[error(
        "Function call name mismatch for ID '{id}': history has '{history_name}', confirmation has '{confirmation_name}'."
    )]
    NameMismatch {
        id: String,
        history_name: String,
        confirmation_name: String,
    },
    #[error("Function call arguments mismatch for ID '{0}'.")]
    ArgumentsMismatch(String),
}

/// `_resolve_confirmation_targets` — see the module doc for the
/// `tools_dict`-as-a-parameter adaptation.
pub async fn resolve_confirmation_targets(
    invocation_context: &InvocationContext,
    events: &[Event],
    confirmation_fc_ids: &HashSet<String>,
    confirmations_by_fc_id: &HashMap<String, ToolConfirmation>,
    tools_dict: &ToolsDict,
) -> Result<
    (
        HashMap<String, ToolConfirmation>,
        HashMap<String, FunctionCall>,
    ),
    ResolveConfirmationTargetsError,
> {
    let mut tool_confirmation_dict = HashMap::new();
    let mut original_fcs_dict = HashMap::new();

    let mut history_fcs: HashMap<String, (FunctionCall, String)> = HashMap::new();
    for event in events {
        for function_call in event.get_function_calls() {
            let Some(id) = &function_call.id else {
                continue;
            };
            if function_call.name.as_deref() == Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME) {
                continue;
            }
            history_fcs.insert(id.clone(), (function_call.clone(), event.author.clone()));
        }
    }

    // IDs of function calls for which a tool dynamically requested
    // confirmation. Accumulates over ALL events (not one event per ID):
    // once the confirmed tool is re-executed it emits a second function
    // response with the same ID and no `requested_tool_confirmations`,
    // which would otherwise shadow the original request.
    let mut dynamically_requested_fc_ids: HashSet<String> = HashSet::new();
    for event in events {
        if event.actions.requested_tool_confirmations.is_empty() {
            continue;
        }
        for function_response in event.get_function_responses() {
            if let Some(id) = &function_response.id {
                if event.actions.requested_tool_confirmations.contains_key(id) {
                    dynamically_requested_fc_ids.insert(id.clone());
                }
            }
        }
    }

    for event in events {
        for function_call in event.get_function_calls() {
            let Some(fc_id) = &function_call.id else {
                continue;
            };
            if !confirmation_fc_ids.contains(fc_id) {
                continue;
            }
            if function_call.name.as_deref() != Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME) {
                continue;
            }

            let Some(original_args) = get_original_function_call_args(function_call) else {
                continue;
            };
            let original_function_call: FunctionCall =
                rusty_serde::json::from_value(Value::Map(original_args.clone())).map_err(|e| {
                    ResolveConfirmationTargetsError::MalformedOriginalFunctionCall(e.to_string())
                })?;
            let original_id = original_function_call
                .id
                .clone()
                .ok_or(ResolveConfirmationTargetsError::MissingOriginalFunctionCallId)?;
            let tool_name = original_function_call
                .name
                .clone()
                .ok_or(ResolveConfirmationTargetsError::MissingOriginalFunctionCallName)?;

            // Check 1: is the tool registered (in session history)?
            let Some((history_fc, history_author)) = history_fcs.get(&original_id) else {
                return Err(ResolveConfirmationTargetsError::NotFoundInHistory(
                    original_id,
                ));
            };

            // If this tool call was authored by another agent, skip it to
            // let that agent's own processor handle it.
            if let Some(agent) = &invocation_context.agent {
                if history_author != agent.name() {
                    continue;
                }
            }

            let Some(tool) = tools_dict.get(&tool_name) else {
                return Err(ResolveConfirmationTargetsError::ToolNotRegistered(
                    tool_name,
                ));
            };

            // Check 2: does the tool require confirmation for these
            // arguments — either statically, or because it was
            // dynamically requested earlier in the session?
            let args = original_function_call.args.clone().unwrap_or_default();
            let mut temp_tool_context =
                create_tool_context(invocation_context, &original_function_call, None);
            let requires_confirmation = tool
                .check_require_confirmation(&args, &mut temp_tool_context)
                .await;
            let requested_in_history = dynamically_requested_fc_ids.contains(&original_id);
            if !requires_confirmation && !requested_in_history {
                return Err(
                    ResolveConfirmationTargetsError::ToolDoesNotRequireConfirmation(tool_name),
                );
            }

            // Check 3: does the original function call match name and args?
            if history_fc.name.as_deref() != Some(tool_name.as_str()) {
                return Err(ResolveConfirmationTargetsError::NameMismatch {
                    id: original_id,
                    history_name: history_fc.name.clone().unwrap_or_default(),
                    confirmation_name: tool_name,
                });
            }
            let history_args = history_fc.args.clone().unwrap_or_default();
            if history_args != args {
                return Err(ResolveConfirmationTargetsError::ArgumentsMismatch(
                    original_id,
                ));
            }

            tool_confirmation_dict
                .insert(original_id.clone(), confirmations_by_fc_id[fc_id].clone());
            original_fcs_dict.insert(original_id, original_function_call);
        }
    }

    Ok((tool_confirmation_dict, original_fcs_dict))
}

/// Wraps every error [`process_request_confirmations`] can propagate.
#[derive(Debug, rusty_err::Error)]
pub enum ProcessRequestConfirmationsError {
    #[error("{0}")]
    ToolConfirmation(#[from] ToolConfirmationError),
    #[error("{0}")]
    ResolveTargets(#[from] ResolveConfirmationTargetsError),
    #[error("{0}")]
    Execution(#[from] FunctionExecutionError),
}

/// `_RequestConfirmationLlmRequestProcessor.run_async`, Steps 1-4 — see
/// the module doc for why `tools_dict` is a caller-supplied parameter and
/// this isn't wired into `LlmFlow::preprocess` yet.
pub async fn process_request_confirmations(
    invocation_context: &InvocationContext,
    events: &[Event],
    tools_dict: &ToolsDict,
) -> Result<Option<Event>, ProcessRequestConfirmationsError> {
    // Step 1: find the last user-authored event and parse confirmation
    // responses from it.
    let mut confirmations_by_fc_id: HashMap<String, ToolConfirmation> = HashMap::new();
    for event in events.iter().rev() {
        if event.author != "user" {
            continue;
        }
        let responses = event.get_function_responses();
        if responses.is_empty() {
            return Ok(None);
        }
        for function_response in responses {
            if function_response.name.as_deref() != Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME) {
                continue;
            }
            let (Some(id), Some(response)) = (&function_response.id, &function_response.response)
            else {
                continue;
            };
            confirmations_by_fc_id.insert(id.clone(), parse_tool_confirmation(response)?);
        }
        break;
    }
    if confirmations_by_fc_id.is_empty() {
        return Ok(None);
    }

    // Step 2: drop confirmations that have already been consumed — this
    // must happen BEFORE resolving targets. The processor re-runs on
    // every LLM step of the invocation, and the approval stays the last
    // user event for the rest of the turn, so a confirmation the
    // previous step already acted on is seen again here.
    let confirmation_ids: HashSet<String> = confirmations_by_fc_id.keys().cloned().collect();
    let confirmation_to_original_fc_id =
        map_confirmation_to_original_fc_ids(events, &confirmation_ids);
    let mut responded_fc_ids: HashSet<String> = HashSet::new();
    for event in events.iter().rev() {
        if event.author == "user" {
            break;
        }
        for function_response in event.get_function_responses() {
            if let Some(id) = &function_response.id {
                responded_fc_ids.insert(id.clone());
            }
        }
    }
    confirmations_by_fc_id.retain(|confirmation_fc_id, _| {
        match confirmation_to_original_fc_id.get(confirmation_fc_id) {
            Some(original_id) => !responded_fc_ids.contains(original_id),
            None => true,
        }
    });
    if confirmations_by_fc_id.is_empty() {
        return Ok(None);
    }

    // Step 3: resolve confirmation targets.
    let confirmation_fc_ids: HashSet<String> = confirmations_by_fc_id.keys().cloned().collect();
    let (tools_to_resume_with_confirmation, tools_to_resume_with_args) =
        resolve_confirmation_targets(
            invocation_context,
            events,
            &confirmation_fc_ids,
            &confirmations_by_fc_id,
            tools_dict,
        )
        .await?;

    if tools_to_resume_with_confirmation.is_empty() {
        return Ok(None);
    }

    // Step 4: re-execute the confirmed tools.
    let function_calls: Vec<FunctionCall> = tools_to_resume_with_args.into_values().collect();
    let filters: HashSet<String> = tools_to_resume_with_confirmation.keys().cloned().collect();
    let agent_name = invocation_context
        .agent
        .as_ref()
        .map(|a| a.name().to_string())
        .unwrap_or_default();

    Ok(execute_function_calls(
        invocation_context,
        &function_calls,
        tools_dict,
        &agent_name,
        Some(&filters),
        Some(&tools_to_resume_with_confirmation),
    )
    .await?)
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
            partial_args: None,
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
            partial_args: None,
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

    use adk_agents::base_agent::{BaseAgent, NoopBehavior};
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_genai::content::FunctionResponse;
    use adk_tools::base_tool::{BaseTool, BoxFuture, ToolError};
    use adk_tools::tool_context::ToolContext;
    use std::collections::BTreeMap as StdBTreeMap;
    use std::sync::Arc;

    struct RequiresConfirmationTool;
    impl BaseTool for RequiresConfirmationTool {
        fn name(&self) -> &str {
            "confirm_tool"
        }
        fn description(&self) -> &str {
            "a tool that always requires confirmation"
        }
        fn check_require_confirmation<'a>(
            &'a self,
            _args: &'a StdBTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> BoxFuture<'a, bool> {
            Box::pin(async { true })
        }
        fn run_async<'a>(
            &'a self,
            _args: &'a StdBTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> BoxFuture<'a, Result<Value, ToolError>> {
            Box::pin(async { Ok(Value::String("done".to_string())) })
        }
    }

    struct PlainTool;
    impl BaseTool for PlainTool {
        fn name(&self) -> &str {
            "plain_tool"
        }
        fn description(&self) -> &str {
            "a tool that never statically requires confirmation"
        }
        fn run_async<'a>(
            &'a self,
            _args: &'a StdBTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> BoxFuture<'a, Result<Value, ToolError>> {
            Box::pin(async { Ok(Value::Null) })
        }
    }

    fn agent_named(name: &str) -> BaseAgent {
        BaseAgent::new(name, NoopBehavior).unwrap()
    }

    fn ctx_with_agent(name: &str) -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(agent_named(name))
            .build()
    }

    fn history_call(id: &str, name: &str, args: Vec<(&str, Value)>) -> FunctionCall {
        FunctionCall {
            partial_args: None,
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            args: Some(args.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
            will_continue: None,
        }
    }

    fn confirmation_call_for(
        conf_id: &str,
        original_id: &str,
        original_name: &str,
        original_args: Vec<(&str, Value)>,
    ) -> FunctionCall {
        let original_args_map: Vec<(String, Value)> = original_args
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let mut args = BTreeMap::new();
        args.insert(
            ORIGINAL_FUNCTION_CALL_KEY.to_string(),
            Value::Map(vec![
                ("id".to_string(), Value::String(original_id.to_string())),
                ("name".to_string(), Value::String(original_name.to_string())),
                ("args".to_string(), Value::Map(original_args_map)),
            ]),
        );
        FunctionCall {
            partial_args: None,
            id: Some(conf_id.to_string()),
            name: Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.to_string()),
            args: Some(args),
            will_continue: None,
        }
    }

    fn confirmed() -> ToolConfirmation {
        ToolConfirmation {
            confirmed: true,
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_succeeds_when_everything_matches() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(1))],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(1))],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert(
            "confirm_tool".to_string(),
            Arc::new(RequiresConfirmationTool),
        );

        let (tool_confirmation_dict, original_fcs_dict) =
            resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
                .await
                .unwrap();

        assert_eq!(tool_confirmation_dict.get("orig-1"), Some(&confirmed()));
        assert!(original_fcs_dict.contains_key("orig-1"));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_when_the_original_id_is_missing() {
        let ctx = ctx_with_agent("agent1");
        let mut args = BTreeMap::new();
        args.insert(
            ORIGINAL_FUNCTION_CALL_KEY.to_string(),
            Value::Map(vec![(
                "name".to_string(),
                Value::String("confirm_tool".to_string()),
            )]),
        );
        let call = FunctionCall {
            partial_args: None,
            id: Some("conf-1".to_string()),
            name: Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.to_string()),
            args: Some(args),
            will_continue: None,
        };
        let events = vec![event_with_calls("agent1", vec![Part::function_call(call)])];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let tools_dict: ToolsDict = HashMap::new();

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::MissingOriginalFunctionCallId
        ));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_when_not_found_in_history() {
        let ctx = ctx_with_agent("agent1");
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let events = vec![confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let tools_dict: ToolsDict = HashMap::new();

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::NotFoundInHistory(id) if id == "orig-1"
        ));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_skips_a_call_authored_by_a_different_agent() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "some_other_agent",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let tools_dict: ToolsDict = HashMap::new();

        let (tool_confirmation_dict, original_fcs_dict) =
            resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
                .await
                .unwrap();
        assert!(tool_confirmation_dict.is_empty());
        assert!(original_fcs_dict.is_empty());
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_when_the_tool_is_not_registered() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let tools_dict: ToolsDict = HashMap::new();

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::ToolNotRegistered(name) if name == "confirm_tool"
        ));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_when_the_tool_does_not_require_confirmation() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "plain_tool",
                vec![],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "plain_tool",
                vec![],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert("plain_tool".to_string(), Arc::new(PlainTool));

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::ToolDoesNotRequireConfirmation(name) if name == "plain_tool"
        ));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_succeeds_when_dynamically_requested_in_history() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "plain_tool",
                vec![],
            ))],
        );
        let mut dynamic_request = Event::new("inv-1", "agent1", NodeInfo::new("root"));
        dynamic_request.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some("orig-1".to_string()),
                name: Some("plain_tool".to_string()),
                response: None,
                ..Default::default()
            })],
        ));
        dynamic_request.actions.requested_tool_confirmations =
            HashMap::from([("orig-1".to_string(), Value::Null)]);
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "plain_tool",
                vec![],
            ))],
        );
        let events = vec![history, dynamic_request, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert("plain_tool".to_string(), Arc::new(PlainTool));

        let (tool_confirmation_dict, _) =
            resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
                .await
                .unwrap();
        assert!(tool_confirmation_dict.contains_key("orig-1"));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_on_a_name_mismatch() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "other_tool",
                vec![],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert(
            "confirm_tool".to_string(),
            Arc::new(RequiresConfirmationTool),
        );

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::NameMismatch { .. }
        ));
    }

    #[rusty_tokio::test]
    async fn resolve_confirmation_targets_errors_on_an_arguments_mismatch() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(1))],
            ))],
        );
        let confirmation = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(2))],
            ))],
        );
        let events = vec![history, confirmation];
        let ids = HashSet::from(["conf-1".to_string()]);
        let confirmations = HashMap::from([("conf-1".to_string(), confirmed())]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert(
            "confirm_tool".to_string(),
            Arc::new(RequiresConfirmationTool),
        );

        let err = resolve_confirmation_targets(&ctx, &events, &ids, &confirmations, &tools_dict)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            ResolveConfirmationTargetsError::ArgumentsMismatch(id) if id == "orig-1"
        ));
    }

    fn user_confirmation_response(conf_id: &str, confirmed: bool) -> Event {
        let mut e = Event::new("inv-1", "user", NodeInfo::new("root"));
        e.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some(conf_id.to_string()),
                name: Some(REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.to_string()),
                response: Some(BTreeMap::from([(
                    "confirmed".to_string(),
                    Value::Bool(confirmed),
                )])),
                ..Default::default()
            })],
        ));
        e
    }

    #[rusty_tokio::test]
    async fn process_request_confirmations_returns_none_with_no_events() {
        let ctx = ctx_with_agent("agent1");
        let tools_dict: ToolsDict = HashMap::new();
        let result = process_request_confirmations(&ctx, &[], &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_request_confirmations_returns_none_without_a_user_authored_event() {
        let ctx = ctx_with_agent("agent1");
        let events = vec![event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        )];
        let tools_dict: ToolsDict = HashMap::new();
        let result = process_request_confirmations(&ctx, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_request_confirmations_returns_none_when_the_last_user_event_has_no_function_responses(
    ) {
        let ctx = ctx_with_agent("agent1");
        let mut plain_user_event = Event::new("inv-1", "user", NodeInfo::new("root"));
        plain_user_event.content = Some(Content::new("user", vec![Part::text("hello")]));
        let events = vec![plain_user_event];
        let tools_dict: ToolsDict = HashMap::new();
        let result = process_request_confirmations(&ctx, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_request_confirmations_drops_an_already_consumed_confirmation() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let confirmation_request = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![],
            ))],
        );
        let user_response = user_confirmation_response("conf-1", true);
        let mut already_resumed = Event::new("inv-1", "agent1", NodeInfo::new("root"));
        already_resumed.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some("orig-1".to_string()),
                name: Some("confirm_tool".to_string()),
                response: Some(BTreeMap::new()),
                ..Default::default()
            })],
        ));
        let events = vec![
            history,
            confirmation_request,
            user_response,
            already_resumed,
        ];
        let tools_dict: ToolsDict = HashMap::new();

        let result = process_request_confirmations(&ctx, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_request_confirmations_resumes_and_re_executes_the_confirmed_tool() {
        let ctx = ctx_with_agent("agent1");
        let history = event_with_calls(
            "agent1",
            vec![Part::function_call(history_call(
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(1))],
            ))],
        );
        let confirmation_request = event_with_calls(
            "agent1",
            vec![Part::function_call(confirmation_call_for(
                "conf-1",
                "orig-1",
                "confirm_tool",
                vec![("x", Value::Int(1))],
            ))],
        );
        let user_response = user_confirmation_response("conf-1", true);
        let events = vec![history, confirmation_request, user_response];

        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert(
            "confirm_tool".to_string(),
            Arc::new(RequiresConfirmationTool),
        );

        let event = process_request_confirmations(&ctx, &events, &tools_dict)
            .await
            .unwrap()
            .expect("expected a re-execution event");
        let responses = event.get_function_responses();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].name.as_deref(), Some("confirm_tool"));
    }
}
