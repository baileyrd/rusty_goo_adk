//! Capabilities C0511-C0515: the auth request/response processor, ported
//! from `google.adk.auth.auth_preprocessor`.
//!
//! Handles the round-trip for a tool that needs end-user credentials: a
//! tool asks for credentials via an `adk_request_credential` function
//! call, the user's browser round trip comes back as a function
//! response, the credential is stored, and the original tool call(s)
//! that needed it are re-executed.
//!
//! **Now ported in full except `tools_dict` auto-resolution**, the same
//! shape `request_confirmation.rs`'s C0172 batch already established for
//! the structurally identical confirmation round-trip:
//! [`store_auth_and_collect_resume_targets`] (`_store_auth_and_collect_resume_targets`)
//! and [`process_auth_responses`] (the processor's own `run_async`) both
//! take `tools_dict: &ToolsDict` as a plain caller-supplied parameter,
//! the same "caller supplies the resolved bits" adaptation
//! `agent_transfer.rs`/`request_confirmation.rs` already established —
//! `functions::execute_function_calls` (this port's `handle_function_calls_async`)
//! already takes exactly this shape, so the tool re-execution tail is
//! genuinely, fully ported here, not stubbed.
//!
//! **Still NOT wired into `LlmFlow::preprocess`**: the source auto-builds
//! `tools_dict` from `agent.canonical_tools()` before calling
//! `_store_auth_and_collect_resume_targets`. `LlmAgent` has no
//! `canonical_tools` resolution built yet (needs the `BaseAgent`/`LlmAgent`
//! tree fusion, C0092 — the same standing-blocked row
//! `request_confirmation.rs`/`agent_transfer.rs` are themselves blocked
//! on). Left unwired, ready for a future C0092-unblocking batch to call
//! directly once it can build a real `tools_dict`. The source's own
//! `if agent is None or not hasattr(agent, "canonical_tools"): return`
//! guard existed only to gate that dynamic resolution — moot here since
//! the caller already decided to supply a real `tools_dict`.
//!
//! **[`merge_credential_oauth2_fields`] (C0512), disclosed narrowing**:
//! the source additionally guards `token_endpoint_auth_method` behind
//! `model_fields_set` — only merging it when the target's value came
//! from the pydantic default, never overriding an explicit assignment.
//! This port has no equivalent "was this field explicitly set at
//! construction" tracking (plain public-field structs throughout this
//! port, no builder/fields-set concept), so `token_endpoint_auth_method`
//! is always adopted from `source_cred` here — matching the source's own
//! most common real-world path (`model_fields_set` doesn't contain the
//! field for a value round-tripped through `model_validate`, which is
//! how every credential reaching this function got here), not the rarer
//! explicit-assignment path.
//!
//! **`_store_auth_and_collect_resume_targets`'s in-place session-state
//! mutation, adapted**: the source passes `invocation_context.session.state`
//! (a live-reference `State` wrapper) straight through to
//! `AuthHandler.parse_and_store_auth_response`, which mutates it in
//! place. This port's `InvocationContext::session::state` is a plain
//! `BTreeMap<String, Value>`, not the `State` delta-tracking wrapper
//! `AuthHandler::parse_and_store_auth_response` needs — so
//! [`process_auth_responses`] builds a `State` from a clone of it,
//! passes that through, and writes the merged result back afterward,
//! the same round-trip [`crate` sibling] `workflow_hitl_utils::process_auth_resume`
//! already uses for the same `AuthHandler` method. Requires `&mut
//! InvocationContext` (the source's caller-supplied dict-reference
//! mutation needs nothing more; Rust's stricter mutability tracking
//! surfaces this explicitly) where `request_confirmation.rs`'s sibling
//! functions only ever needed `&InvocationContext`.
//!
//! **Disclosed, matching an established pattern**: a client's auth
//! response for a function-call id this session never requested is
//! silently ignored — the source logs a warning first; no logging
//! framework has been adopted in this port (the same disclosed
//! substitution used throughout this migration).

use std::collections::{BTreeMap, HashMap, HashSet};

use adk_agents::auth_credential::AuthCredential;
use adk_agents::auth_handler::{AuthHandler, AuthHandlerError};
use adk_agents::auth_tool::{AuthConfig, AuthToolArguments};
use adk_agents::invocation_context::InvocationContext;
use adk_agents::state::State;
use adk_events::Event;
use adk_genai::content::FunctionCall;
use rusty_serde::value::Value;

use crate::contents::REQUEST_EUC_FUNCTION_CALL_NAME;
use crate::functions::{execute_function_calls, FunctionExecutionError, ToolsDict};

/// C0513: marks a toolset-level (pre-tool-listing) auth request, distinct
/// from a per-tool-call auth request — such a request doesn't map back
/// to a resumable function call. The source duplicates this constant
/// independently in `flows/llm_flows/base_llm_flow.py`; this port keeps
/// one shared constant instead, per this manifest row's own note.
pub const TOOLSET_AUTH_CREDENTIAL_ID_PREFIX: &str = "_adk_toolset_auth_";

/// C0512: `_merge_credential_oauth2_fields` — merges OAuth2 fields from
/// `source_cred` into `target_cred` only where `target_cred`'s own field
/// is unset, preventing a client's auth response from overriding a
/// developer-configured secret. `None` on either side collapses to
/// whichever side is present. See the module doc for the disclosed
/// `token_endpoint_auth_method` narrowing.
pub fn merge_credential_oauth2_fields(
    target_cred: Option<AuthCredential>,
    source_cred: Option<AuthCredential>,
) -> Option<AuthCredential> {
    let Some(source_cred) = source_cred else {
        return target_cred;
    };
    let Some(mut target_cred) = target_cred else {
        return Some(source_cred);
    };

    match (target_cred.oauth2.as_mut(), source_cred.oauth2.as_ref()) {
        (None, Some(source_oauth2)) => {
            target_cred.oauth2 = Some(source_oauth2.clone());
        }
        (Some(target_oauth2), Some(source_oauth2)) => {
            if target_oauth2.client_id.is_none() {
                target_oauth2.client_id = source_oauth2.client_id.clone();
            }
            if target_oauth2.client_secret.is_none() {
                target_oauth2.client_secret = source_oauth2.client_secret.clone();
            }
            if target_oauth2.redirect_uri.is_none() {
                target_oauth2.redirect_uri = source_oauth2.redirect_uri.clone();
            }
            if target_oauth2.code_verifier.is_none() {
                target_oauth2.code_verifier = source_oauth2.code_verifier.clone();
            }
            if target_oauth2.code_challenge_method.is_none() {
                target_oauth2.code_challenge_method = source_oauth2.code_challenge_method.clone();
            }
            target_oauth2.token_endpoint_auth_method = source_oauth2.token_endpoint_auth_method;
        }
        (Some(_), None) | (None, None) => {}
    }

    Some(target_cred)
}

/// Parses a `adk_request_credential` function call's args into
/// [`AuthToolArguments`] — mirrors `AuthToolArguments.model_validate(...)`,
/// silently returning `None` (rather than propagating a validation
/// error) on malformed args, matching the source's own `except
/// TypeError: continue`.
fn parse_auth_tool_arguments(function_call: &FunctionCall) -> Option<AuthToolArguments> {
    let args = function_call.args.as_ref()?;
    let value = Value::Map(args.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    rusty_serde::json::from_value(value).ok()
}

/// Parses a client-supplied auth response body into [`AuthConfig`] —
/// mirrors `AuthConfig.model_validate(auth_responses[fc_id])`.
fn parse_auth_config(response: &BTreeMap<String, Value>) -> Option<AuthConfig> {
    let value = Value::Map(
        response
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    );
    rusty_serde::json::from_value(value).ok()
}

/// C0514: `_store_auth_and_collect_resume_targets` — a 3-step
/// reconciliation: (1) recovers the server-issued `credential_key`/
/// `auth_scheme` for each requested credential from session history,
/// (2) rehydrates and pins the client's auth response back onto those
/// server-issued values (rejecting an unrequested/unpinned response) and
/// stores each credential via [`AuthHandler::parse_and_store_auth_response`],
/// (3) resumes every pending tool call sharing a newly-authorized
/// `credential_key`. Returns the set of original function-call ids to
/// resume.
pub fn store_auth_and_collect_resume_targets(
    events: &[Event],
    auth_fc_ids: &HashSet<String>,
    auth_responses: &HashMap<String, BTreeMap<String, Value>>,
    state: &mut State,
) -> Result<HashSet<String>, AuthHandlerError> {
    // Step 1: scan events for matching adk_request_credential function
    // calls to extract AuthToolArguments (contains credential_key).
    let mut requested_auth_config_by_id: HashMap<String, AuthConfig> = HashMap::new();
    for event in events {
        for function_call in event.get_function_calls() {
            let Some(id) = &function_call.id else {
                continue;
            };
            if !auth_fc_ids.contains(id) {
                continue;
            }
            if function_call.name.as_deref() != Some(REQUEST_EUC_FUNCTION_CALL_NAME) {
                continue;
            }
            if let Some(args) = parse_auth_tool_arguments(function_call) {
                requested_auth_config_by_id.insert(id.clone(), args.auth_config);
            }
        }
    }

    // Step 2: store credentials. The client's response supplies the
    // result of the user's browser round trip; the auth scheme and the
    // credential key come from the request this server issued.
    let mut authorized_keys: HashSet<String> = HashSet::new();
    for fc_id in auth_fc_ids {
        let Some(response) = auth_responses.get(fc_id) else {
            continue;
        };
        let Some(requested_auth_config) = requested_auth_config_by_id.get(fc_id) else {
            // Nothing to pin against, so the response would get to choose
            // both the credential it is exchanged with and the endpoint
            // it goes to — ignored, matching the source's own guard.
            continue;
        };

        let Some(mut auth_config) = parse_auth_config(response) else {
            continue;
        };
        // The scheme names the token endpoint the developer's secret is
        // posted to.
        auth_config.auth_scheme = requested_auth_config.auth_scheme.clone();
        if requested_auth_config.credential_key.is_some() {
            auth_config.credential_key = requested_auth_config.credential_key.clone();
        }
        if requested_auth_config.raw_auth_credential.is_some() {
            auth_config.raw_auth_credential = merge_credential_oauth2_fields(
                auth_config.raw_auth_credential,
                requested_auth_config.raw_auth_credential.clone(),
            );
        }
        if requested_auth_config.exchanged_auth_credential.is_some() {
            auth_config.exchanged_auth_credential = merge_credential_oauth2_fields(
                auth_config.exchanged_auth_credential,
                requested_auth_config.exchanged_auth_credential.clone(),
            );
        }
        if let Some(key) = &auth_config.credential_key {
            authorized_keys.insert(key.clone());
        }

        AuthHandler::new(auth_config).parse_and_store_auth_response(state)?;
    }

    // Step 3: collect original function call IDs to resume, skipping
    // toolset auth entries which don't map to a resumable function call.
    let mut tools_to_resume: HashSet<String> = HashSet::new();
    for fc_id in auth_fc_ids {
        if !requested_auth_config_by_id.contains_key(fc_id) {
            continue;
        }
        // Re-parse to get function_call_id (AuthConfig doesn't carry it;
        // AuthToolArguments does).
        for event in events {
            for function_call in event.get_function_calls() {
                if function_call.id.as_deref() != Some(fc_id.as_str()) {
                    continue;
                }
                if function_call.name.as_deref() != Some(REQUEST_EUC_FUNCTION_CALL_NAME) {
                    continue;
                }
                let Some(args) = parse_auth_tool_arguments(function_call) else {
                    continue;
                };
                if args
                    .function_call_id
                    .starts_with(TOOLSET_AUTH_CREDENTIAL_ID_PREFIX)
                {
                    continue;
                }
                tools_to_resume.insert(args.function_call_id);
            }
        }
    }

    let matching_events: Vec<&Event> = events
        .iter()
        .filter(|event| {
            !event.actions.requested_auth_configs.is_empty()
                && tools_to_resume
                    .iter()
                    .any(|fc_id| event.actions.requested_auth_configs.contains_key(fc_id))
        })
        .collect();

    for event in matching_events {
        for (original_fc_id, config_value) in &event.actions.requested_auth_configs {
            let Ok(config) = rusty_serde::json::from_value::<AuthConfig>(config_value.clone())
            else {
                continue;
            };
            if config
                .credential_key
                .as_deref()
                .is_some_and(|key| authorized_keys.contains(key))
            {
                tools_to_resume.insert(original_fc_id.clone());
            }
        }
    }

    Ok(tools_to_resume)
}

/// Wraps every error [`process_auth_responses`] can propagate.
#[derive(Debug, rusty_err::Error)]
pub enum AuthPreprocessorError {
    #[error("{0}")]
    Store(#[from] AuthHandlerError),
    #[error("{0}")]
    Execution(#[from] FunctionExecutionError),
}

/// C0511/C0515: `_AuthLlmRequestProcessor.run_async` — see the module
/// doc for why `tools_dict` is a caller-supplied parameter, why this
/// isn't wired into `LlmFlow::preprocess` yet, and why this takes `&mut
/// InvocationContext`.
///
/// C0515: only the most recent user-authored event with non-`None`
/// content is inspected for `adk_request_credential` responses — walks
/// `events` backward from the end, matching the source's own scan.
pub async fn process_auth_responses(
    invocation_context: &mut InvocationContext,
    events: &[Event],
    tools_dict: &ToolsDict,
) -> Result<Option<Event>, AuthPreprocessorError> {
    let Some(last_event_with_content) = events.iter().rev().find(|e| e.content.is_some()) else {
        return Ok(None);
    };
    if last_event_with_content.author != "user" {
        return Ok(None);
    }

    let responses = last_event_with_content.get_function_responses();
    if responses.is_empty() {
        return Ok(None);
    }

    // Collect adk_request_credential function response IDs and their
    // response bodies.
    let mut auth_fc_ids: HashSet<String> = HashSet::new();
    let mut auth_responses: HashMap<String, BTreeMap<String, Value>> = HashMap::new();
    for function_response in &responses {
        if function_response.name.as_deref() != Some(REQUEST_EUC_FUNCTION_CALL_NAME) {
            continue;
        }
        let Some(id) = &function_response.id else {
            continue;
        };
        auth_fc_ids.insert(id.clone());
        if let Some(response) = &function_response.response {
            auth_responses.insert(id.clone(), response.clone());
        }
    }

    if auth_fc_ids.is_empty() {
        return Ok(None);
    }

    // Store credentials and collect tools to resume.
    let mut state = State::new(invocation_context.session.state.clone(), BTreeMap::new());
    let tools_to_resume =
        store_auth_and_collect_resume_targets(events, &auth_fc_ids, &auth_responses, &mut state)?;
    invocation_context.session.state = state.to_map();

    if tools_to_resume.is_empty() {
        return Ok(None);
    }

    // Find the original function call event and re-execute the tools
    // that needed auth. Deliberately skips the literal last event in
    // `events` (the just-processed auth-response event) unconditionally
    // by position, matching the source's own `range(len(events) - 2,
    // -1, -1)` — not "whichever event `last_event_with_content` was."
    let agent_name = invocation_context
        .agent
        .as_ref()
        .map(|agent| agent.name().to_string())
        .unwrap_or_default();

    let search_space = &events[..events.len().saturating_sub(1)];
    for event in search_space.iter().rev() {
        let function_calls = event.get_function_calls();
        if function_calls.is_empty() {
            continue;
        }
        let any_to_resume = function_calls.iter().any(|function_call| {
            function_call
                .id
                .as_deref()
                .is_some_and(|id| tools_to_resume.contains(id))
        });
        if !any_to_resume {
            continue;
        }
        let owned_calls: Vec<FunctionCall> = function_calls.into_iter().cloned().collect();
        return Ok(execute_function_calls(
            invocation_context,
            &owned_calls,
            tools_dict,
            &agent_name,
            Some(&tools_to_resume),
            None,
        )
        .await?);
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::auth_credential::{AuthCredentialTypes, OAuth2Auth};
    use adk_agents::auth_schemes::{AuthScheme, HttpScheme, SecurityScheme};
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{Content, Part};
    use adk_tools::base_tool::{BaseTool, BoxFuture as ToolBoxFuture, ToolError};
    use adk_tools::tool_context::ToolContext;
    use std::sync::Arc;

    fn http_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "bearer".to_string(),
            bearer_format: None,
        })))
    }

    fn oauth2_credential(client_id: Option<&str>, client_secret: Option<&str>) -> AuthCredential {
        let mut cred = AuthCredential::new(AuthCredentialTypes::OAuth2);
        cred.oauth2 = Some(OAuth2Auth {
            client_id: client_id.map(str::to_string),
            client_secret: client_secret.map(str::to_string),
            ..Default::default()
        });
        cred
    }

    // --- C0512: merge_credential_oauth2_fields ---

    #[test]
    fn merge_credential_oauth2_fields_returns_target_when_source_is_none() {
        let target = oauth2_credential(Some("id"), None);
        assert_eq!(
            merge_credential_oauth2_fields(Some(target.clone()), None),
            Some(target)
        );
    }

    #[test]
    fn merge_credential_oauth2_fields_returns_source_when_target_is_none() {
        let source = oauth2_credential(Some("id"), None);
        assert_eq!(
            merge_credential_oauth2_fields(None, Some(source.clone())),
            Some(source)
        );
    }

    #[test]
    fn merge_credential_oauth2_fields_never_overrides_an_existing_target_field() {
        let target = oauth2_credential(Some("developer-id"), Some("developer-secret"));
        let source = oauth2_credential(Some("client-id"), Some("client-secret"));
        let merged = merge_credential_oauth2_fields(Some(target), Some(source)).unwrap();
        let oauth2 = merged.oauth2.unwrap();
        assert_eq!(oauth2.client_id.as_deref(), Some("developer-id"));
        assert_eq!(oauth2.client_secret.as_deref(), Some("developer-secret"));
    }

    #[test]
    fn merge_credential_oauth2_fields_fills_in_a_missing_target_field() {
        let target = oauth2_credential(None, Some("developer-secret"));
        let source = oauth2_credential(Some("client-id"), Some("client-secret"));
        let merged = merge_credential_oauth2_fields(Some(target), Some(source)).unwrap();
        let oauth2 = merged.oauth2.unwrap();
        assert_eq!(oauth2.client_id.as_deref(), Some("client-id"));
        assert_eq!(oauth2.client_secret.as_deref(), Some("developer-secret"));
    }

    #[test]
    fn merge_credential_oauth2_fields_deep_copies_source_when_target_has_no_oauth2() {
        let mut target = AuthCredential::new(AuthCredentialTypes::OAuth2);
        target.oauth2 = None;
        let source = oauth2_credential(Some("client-id"), None);
        let merged = merge_credential_oauth2_fields(Some(target), Some(source)).unwrap();
        assert_eq!(
            merged.oauth2.unwrap().client_id.as_deref(),
            Some("client-id")
        );
    }

    // --- helpers for the processor-level tests ---

    fn request_credential_fc(
        id: &str,
        function_call_id: &str,
        credential_key: &str,
    ) -> FunctionCall {
        let auth_config =
            AuthConfig::new(http_scheme(), None, None, Some(credential_key.to_string()));
        let args = AuthToolArguments {
            function_call_id: function_call_id.to_string(),
            auth_config,
        };
        let value = rusty_serde::json::to_value(&args).unwrap();
        let args_map = match value {
            Value::Map(entries) => entries.into_iter().collect(),
            _ => BTreeMap::new(),
        };
        FunctionCall {
            id: Some(id.to_string()),
            name: Some(REQUEST_EUC_FUNCTION_CALL_NAME.to_string()),
            args: Some(args_map),
            partial_args: None,
            will_continue: None,
        }
    }

    fn original_tool_call(id: &str, name: &str) -> FunctionCall {
        FunctionCall {
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            args: Some(BTreeMap::new()),
            partial_args: None,
            will_continue: None,
        }
    }

    fn auth_response_value(credential_key: &str) -> BTreeMap<String, Value> {
        let auth_config =
            AuthConfig::new(http_scheme(), None, None, Some(credential_key.to_string()));
        match rusty_serde::json::to_value(&auth_config).unwrap() {
            Value::Map(entries) => entries.into_iter().collect(),
            _ => BTreeMap::new(),
        }
    }

    // --- C0514: store_auth_and_collect_resume_targets ---

    #[test]
    fn store_auth_and_collect_resume_targets_ignores_an_unrequested_response() {
        let events = vec![];
        let auth_fc_ids: HashSet<String> = ["fc-1".to_string()].into_iter().collect();
        let auth_responses: HashMap<String, BTreeMap<String, Value>> =
            [("fc-1".to_string(), auth_response_value("key-1"))]
                .into_iter()
                .collect();
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        let resumed = store_auth_and_collect_resume_targets(
            &events,
            &auth_fc_ids,
            &auth_responses,
            &mut state,
        )
        .unwrap();
        assert!(resumed.is_empty());
    }

    #[test]
    fn store_auth_and_collect_resume_targets_resumes_the_original_tool_call() {
        let mut request_event = Event::new("inv-1", "chat", NodeInfo::new(""));
        request_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(request_credential_fc(
                "fc-auth-1",
                "fc-tool-1",
                "key-1",
            ))],
        ));
        let events = vec![request_event];

        let auth_fc_ids: HashSet<String> = ["fc-auth-1".to_string()].into_iter().collect();
        let auth_responses: HashMap<String, BTreeMap<String, Value>> =
            [("fc-auth-1".to_string(), auth_response_value("key-1"))]
                .into_iter()
                .collect();
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        let resumed = store_auth_and_collect_resume_targets(
            &events,
            &auth_fc_ids,
            &auth_responses,
            &mut state,
        )
        .unwrap();
        assert_eq!(resumed, ["fc-tool-1".to_string()].into_iter().collect());
    }

    #[test]
    fn store_auth_and_collect_resume_targets_skips_toolset_auth_entries() {
        let toolset_fc_id = format!("{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}SomeToolset");
        let mut request_event = Event::new("inv-1", "chat", NodeInfo::new(""));
        request_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(request_credential_fc(
                "fc-auth-1",
                &toolset_fc_id,
                "key-1",
            ))],
        ));
        let events = vec![request_event];

        let auth_fc_ids: HashSet<String> = ["fc-auth-1".to_string()].into_iter().collect();
        let auth_responses: HashMap<String, BTreeMap<String, Value>> =
            [("fc-auth-1".to_string(), auth_response_value("key-1"))]
                .into_iter()
                .collect();
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        let resumed = store_auth_and_collect_resume_targets(
            &events,
            &auth_fc_ids,
            &auth_responses,
            &mut state,
        )
        .unwrap();
        assert!(resumed.is_empty());
    }

    // --- C0511/C0515: process_auth_responses ---

    fn test_invocation_context(events: Vec<Event>) -> InvocationContext {
        let mut session = Session::new("app", "user", "s1");
        session.events = events;
        InvocationContextBuilder::new("inv-1", session).build()
    }

    #[rusty_tokio::test]
    async fn process_auth_responses_is_none_without_a_recent_user_content_event() {
        let mut ic = test_invocation_context(vec![Event::new("inv-1", "chat", NodeInfo::new(""))]);
        let tools_dict: ToolsDict = HashMap::new();
        let events = ic.session.events.clone();
        let result = process_auth_responses(&mut ic, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_auth_responses_is_none_when_the_last_content_event_has_no_matching_response() {
        let mut user_event = Event::new("inv-1", "user", NodeInfo::new(""));
        user_event.content = Some(Content::new("user", vec![Part::text("hello")]));
        let mut ic = test_invocation_context(vec![user_event]);
        let tools_dict: ToolsDict = HashMap::new();
        let events = ic.session.events.clone();
        let result = process_auth_responses(&mut ic, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_auth_responses_is_none_when_no_tools_are_resumed() {
        let mut user_event = Event::new("inv-1", "user", NodeInfo::new(""));
        user_event.content = Some(Content::new(
            "user",
            vec![Part::function_response(
                adk_genai::content::FunctionResponse {
                    id: Some("fc-auth-1".to_string()),
                    name: Some(REQUEST_EUC_FUNCTION_CALL_NAME.to_string()),
                    response: Some(auth_response_value("key-1")),
                    parts: None,
                },
            )],
        ));
        let mut ic = test_invocation_context(vec![user_event]);
        let tools_dict: ToolsDict = HashMap::new();
        let events = ic.session.events.clone();
        // No prior request event for "fc-auth-1" exists in history, so
        // store_auth_and_collect_resume_targets resumes nothing.
        let result = process_auth_responses(&mut ic, &events, &tools_dict)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn process_auth_responses_stores_the_credential_even_when_nothing_resumes() {
        let mut request_event = Event::new("inv-1", "chat", NodeInfo::new(""));
        request_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(request_credential_fc(
                "fc-auth-1",
                "fc-tool-1",
                "key-1",
            ))],
        ));
        let mut user_event = Event::new("inv-1", "user", NodeInfo::new(""));
        user_event.content = Some(Content::new(
            "user",
            vec![Part::function_response(
                adk_genai::content::FunctionResponse {
                    id: Some("fc-auth-1".to_string()),
                    name: Some(REQUEST_EUC_FUNCTION_CALL_NAME.to_string()),
                    response: Some(auth_response_value("key-1")),
                    parts: None,
                },
            )],
        ));
        let mut ic = test_invocation_context(vec![request_event, user_event]);
        let tools_dict: ToolsDict = HashMap::new();
        let events = ic.session.events.clone();
        // The tool call "fc-tool-1" is never in the event history, so
        // nothing gets resumed — but the credential is still stored.
        process_auth_responses(&mut ic, &events, &tools_dict)
            .await
            .unwrap();
        assert!(!ic.session.state.is_empty());
    }

    struct EchoTool;
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }
        fn description(&self) -> &str {
            "echoes a fixed result"
        }
        fn run_async<'a>(
            &'a self,
            _args: &'a BTreeMap<String, Value>,
            _tool_context: &'a mut ToolContext,
        ) -> ToolBoxFuture<'a, Result<Value, ToolError>> {
            Box::pin(async move { Ok(Value::String("ok".to_string())) })
        }
    }

    #[rusty_tokio::test]
    async fn process_auth_responses_resumes_and_reexecutes_the_original_tool_call() {
        let mut original_event = Event::new("inv-1", "chat", NodeInfo::new(""));
        original_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(original_tool_call(
                "fc-tool-1",
                "echo_tool",
            ))],
        ));
        let mut request_event = Event::new("inv-1", "chat", NodeInfo::new(""));
        request_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(request_credential_fc(
                "fc-auth-1",
                "fc-tool-1",
                "key-1",
            ))],
        ));
        let mut user_event = Event::new("inv-1", "user", NodeInfo::new(""));
        user_event.content = Some(Content::new(
            "user",
            vec![Part::function_response(
                adk_genai::content::FunctionResponse {
                    id: Some("fc-auth-1".to_string()),
                    name: Some(REQUEST_EUC_FUNCTION_CALL_NAME.to_string()),
                    response: Some(auth_response_value("key-1")),
                    parts: None,
                },
            )],
        ));
        let mut ic = test_invocation_context(vec![original_event, request_event, user_event]);
        let mut tools_dict: ToolsDict = HashMap::new();
        tools_dict.insert("echo_tool".to_string(), Arc::new(EchoTool));
        let events = ic.session.events.clone();

        let result = process_auth_responses(&mut ic, &events, &tools_dict)
            .await
            .unwrap();
        let event = result.expect("expected a function-response event");
        let responses = event.get_function_responses();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].id.as_deref(), Some("fc-tool-1"));
    }
}
