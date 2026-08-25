//! Capability C0329: Human-in-the-Loop (HITL) workflow utilities, ported
//! from `google.adk.workflow.utils._workflow_hitl_utils`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc for
//! the standing crate-placement decision.
//!
//! **`schema_to_json_schema`, disclosed**: [`RequestInput::response_schema`]
//! is already `Option<Value>` in this port (JSON-Schema-shaped, per that
//! type's own doc) rather than the source's `SchemaType` union — so
//! [`create_request_input_event`] doesn't need a real schema→JSON-Schema
//! conversion step, just a pass-through of the field as-is.

use std::collections::BTreeMap;

use rusty_serde::value::Value;

use adk_events::Event;
use adk_events::RequestInput;
use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};

use crate::auth_credential::{AuthCredential, AuthCredentialTypes};
use crate::auth_handler::{AuthHandler, AuthHandlerError};
use crate::auth_schemes::{AuthScheme, SecurityScheme};
use crate::auth_tool::{AuthConfig, AuthToolArguments};
use crate::state::State;

pub const REQUEST_INPUT_FUNCTION_CALL_NAME: &str = "adk_request_input";
pub const REQUEST_CREDENTIAL_FUNCTION_CALL_NAME: &str = "adk_request_credential";

const OAUTH_STATE_KEY_PREFIX: &str = "adk_oauth_state:";

fn oauth_state_key(interrupt_id: &str) -> String {
    format!("{OAUTH_STATE_KEY_PREFIX}{interrupt_id}")
}

/// `create_request_input_event`: builds an `adk_request_input`
/// function-call event from a [`RequestInput`].
pub fn create_request_input_event(request_input: &RequestInput) -> Event {
    let mut args = BTreeMap::new();
    if let Some(payload) = &request_input.payload {
        args.insert("payload".to_string(), payload.clone());
    }
    if let Some(message) = &request_input.message {
        args.insert("message".to_string(), Value::String(message.clone()));
    }
    args.insert(
        "responseSchema".to_string(),
        request_input.response_schema.clone().unwrap_or(Value::Null),
    );

    let mut event = Event::new(
        String::new(),
        String::new(),
        adk_events::node_info::NodeInfo::new(""),
    );
    event.content = Some(Content {
        role: Some("model".to_string()),
        parts: vec![Part::function_call(FunctionCall {
            id: Some(request_input.interrupt_id.clone()),
            name: Some(REQUEST_INPUT_FUNCTION_CALL_NAME.to_string()),
            args: Some(args),
            ..Default::default()
        })],
    });
    event.set_long_running_tool_ids([request_input.interrupt_id.clone()]);
    event
}

/// `has_request_input_function_call`.
pub fn has_request_input_function_call(event: &Event) -> bool {
    event
        .get_function_calls()
        .iter()
        .any(|call| call.name.as_deref() == Some(REQUEST_INPUT_FUNCTION_CALL_NAME))
}

/// `has_auth_request_function_call`.
pub fn has_auth_request_function_call(event: &Event) -> bool {
    event
        .get_function_calls()
        .iter()
        .any(|call| call.name.as_deref() == Some(REQUEST_CREDENTIAL_FUNCTION_CALL_NAME))
}

/// `create_request_input_response`: builds the `FunctionResponse` part
/// answering a `request_input` function call.
pub fn create_request_input_response(
    interrupt_id: &str,
    response: BTreeMap<String, Value>,
) -> Part {
    Part::function_response(FunctionResponse {
        id: Some(interrupt_id.to_string()),
        name: Some(REQUEST_INPUT_FUNCTION_CALL_NAME.to_string()),
        response: Some(response),
        parts: None,
    })
}

/// `get_request_input_interrupt_ids`.
pub fn get_request_input_interrupt_ids(event: &Event) -> Vec<String> {
    event
        .get_function_calls()
        .iter()
        .filter(|call| call.name.as_deref() == Some(REQUEST_INPUT_FUNCTION_CALL_NAME))
        .filter_map(|call| call.id.clone())
        .collect()
}

/// `_build_auth_message`: a human-readable message describing what
/// credential is needed.
fn build_auth_message(auth_config: &AuthConfig) -> String {
    let Some(raw_cred) = &auth_config.raw_auth_credential else {
        return "Please provide your authentication credentials.".to_string();
    };
    match raw_cred.auth_type {
        AuthCredentialTypes::ApiKey => {
            let name = match &auth_config.auth_scheme {
                AuthScheme::Security(security) => match security.as_ref() {
                    SecurityScheme::ApiKey(api_key) => api_key.name.clone(),
                    _ => "API key".to_string(),
                },
                _ => "API key".to_string(),
            };
            format!("Please provide your API key for {name}.")
        }
        AuthCredentialTypes::OAuth2 | AuthCredentialTypes::OpenIdConnect => {
            "Please complete the authentication flow.".to_string()
        }
        _ => "Please provide your authentication credentials.".to_string(),
    }
}

/// `create_auth_request_event`: builds an event requesting user
/// authentication credentials, storing any generated OAuth state under
/// `state` so a resume response can be checked against it.
pub fn create_auth_request_event(
    auth_config: &AuthConfig,
    interrupt_id: &str,
    state: &mut State,
) -> Result<Event, AuthHandlerError> {
    let auth_handler = AuthHandler::new(auth_config.clone());
    let auth_request = auth_handler.generate_auth_request()?;
    if let Some(oauth2) = auth_request
        .exchanged_auth_credential
        .as_ref()
        .and_then(|credential| credential.oauth2.as_ref())
    {
        if let Some(oauth_state) = &oauth2.state {
            state.set(
                oauth_state_key(interrupt_id),
                Value::String(oauth_state.clone()),
            );
        }
    }

    let args_value = rusty_serde::json::to_value(&AuthToolArguments {
        function_call_id: interrupt_id.to_string(),
        auth_config: auth_request,
    })
    .unwrap_or(Value::Null);
    let mut args = match args_value {
        Value::Map(entries) => entries.into_iter().collect::<BTreeMap<_, _>>(),
        _ => BTreeMap::new(),
    };
    args.insert(
        "message".to_string(),
        Value::String(build_auth_message(auth_config)),
    );

    let mut event = Event::new(
        String::new(),
        String::new(),
        adk_events::node_info::NodeInfo::new(""),
    );
    event.content = Some(Content {
        role: Some("model".to_string()),
        parts: vec![Part::function_call(FunctionCall {
            id: Some(interrupt_id.to_string()),
            name: Some(REQUEST_CREDENTIAL_FUNCTION_CALL_NAME.to_string()),
            args: Some(args),
            ..Default::default()
        })],
    });
    event.set_long_running_tool_ids([interrupt_id.to_string()]);
    Ok(event)
}

/// `_build_credential_from_value`: builds an `AuthCredential` from a raw
/// user-provided value — for `API_KEY`, the value is used as the key
/// string directly; otherwise it's parsed as an `AuthCredential`.
fn build_credential_from_value(
    auth_config: &AuthConfig,
    value: &Value,
) -> Result<AuthCredential, String> {
    let Some(raw_cred) = &auth_config.raw_auth_credential else {
        return rusty_serde::json::from_value(value.clone()).map_err(|error| error.to_string());
    };
    if raw_cred.auth_type == AuthCredentialTypes::ApiKey {
        let api_key = value.as_str().map(str::to_string).unwrap_or_default();
        return Ok(AuthCredential {
            api_key: Some(api_key),
            ..AuthCredential::new(AuthCredentialTypes::ApiKey)
        });
    }
    rusty_serde::json::from_value(value.clone()).map_err(|error| error.to_string())
}

#[derive(Debug, rusty_err::Error)]
pub enum ProcessAuthResumeError {
    #[error(
        "The auth response does not carry back the state generated for this auth request. \
         Return the auth config from the credential request with the authorization result \
         filled in."
    )]
    OauthStateMismatch,
    #[error("{0}")]
    InvalidCredential(String),
    #[error("{0}")]
    AuthHandler(#[from] AuthHandlerError),
}

/// `process_auth_resume`: stores credentials from an auth resume
/// response into session state. The caller is responsible for
/// unwrapping `{"result": ...}` wrappers before calling this function
/// (matching the source's own documented contract).
pub fn process_auth_resume(
    response_data: &Value,
    auth_config: &AuthConfig,
    state: &mut State,
    interrupt_id: &str,
) -> Result<(), ProcessAuthResumeError> {
    let exchanged_credential: Option<AuthCredential> =
        match rusty_serde::json::from_value::<AuthConfig>(response_data.clone()) {
            Ok(config) => config.exchanged_auth_credential,
            Err(_) => build_credential_from_value(auth_config, response_data)
                .map(Some)
                .map_err(ProcessAuthResumeError::InvalidCredential)?,
        };

    if let Some(generated_state) = state.get(&oauth_state_key(interrupt_id)).cloned() {
        let matches = exchanged_credential
            .as_ref()
            .and_then(|credential| credential.oauth2.as_ref())
            .and_then(|oauth2| oauth2.state.as_ref())
            .is_some_and(|state_value| Value::String(state_value.clone()) == generated_state);
        if !matches {
            return Err(ProcessAuthResumeError::OauthStateMismatch);
        }
    }

    let mut resumed_config = auth_config.clone();
    resumed_config.exchanged_auth_credential = exchanged_credential;
    AuthHandler::new(resumed_config).parse_and_store_auth_response(state)?;
    Ok(())
}

/// `has_auth_credential`: whether a credential for `auth_config` already
/// exists in `state`.
pub fn has_auth_credential(auth_config: &AuthConfig, state: &State) -> bool {
    AuthHandler::new(auth_config.clone())
        .get_auth_response(state)
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_schemes::{ApiKeyIn, ApiKeyScheme};

    fn api_key_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })))
    }

    fn api_key_config() -> AuthConfig {
        AuthConfig::new(
            api_key_scheme(),
            Some(AuthCredential::api_key("placeholder")),
            None,
            Some("my_key".to_string()),
        )
    }

    // --- request_input events ---

    #[test]
    fn create_request_input_event_carries_the_interrupt_id_and_payload() {
        let request_input = RequestInput::new(
            Some("please confirm".to_string()),
            Some(Value::String("payload".to_string())),
            None,
        );
        let event = create_request_input_event(&request_input);
        assert_eq!(
            event.long_running_tool_ids,
            Some(vec![request_input.interrupt_id.clone()])
        );
        let calls = event.get_function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].name.as_deref(),
            Some(REQUEST_INPUT_FUNCTION_CALL_NAME)
        );
        assert_eq!(
            calls[0].id.as_deref(),
            Some(request_input.interrupt_id.as_str())
        );
    }

    #[test]
    fn has_request_input_function_call_detects_the_right_call() {
        let request_input = RequestInput::new(None, None, None);
        let event = create_request_input_event(&request_input);
        assert!(has_request_input_function_call(&event));
        assert!(!has_auth_request_function_call(&event));
    }

    #[test]
    fn get_request_input_interrupt_ids_extracts_the_id() {
        let request_input = RequestInput::new(None, None, None);
        let event = create_request_input_event(&request_input);
        assert_eq!(
            get_request_input_interrupt_ids(&event),
            vec![request_input.interrupt_id]
        );
    }

    #[test]
    fn create_request_input_response_builds_a_matching_function_response() {
        let mut response = BTreeMap::new();
        response.insert("result".to_string(), Value::String("ok".to_string()));
        let part = create_request_input_response("interrupt-1", response.clone());
        let function_response = part.function_response.unwrap();
        assert_eq!(function_response.id.as_deref(), Some("interrupt-1"));
        assert_eq!(function_response.response, Some(response));
    }

    // --- auth events ---

    #[test]
    fn create_auth_request_event_for_an_api_key_scheme() {
        let config = api_key_config();
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        let event = create_auth_request_event(&config, "interrupt-1", &mut state).unwrap();
        assert!(has_auth_request_function_call(&event));
        let calls = event.get_function_calls();
        let message = calls[0]
            .args
            .as_ref()
            .and_then(|args| args.get("message"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(message.contains("X-Api-Key"), "{message}");
    }

    #[test]
    fn has_auth_credential_is_false_without_a_stored_response() {
        let config = api_key_config();
        let state = State::new(BTreeMap::new(), BTreeMap::new());
        assert!(!has_auth_credential(&config, &state));
    }

    #[test]
    fn process_auth_resume_stores_an_api_key_value() {
        let config = api_key_config();
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        process_auth_resume(
            &Value::String("secret-key".to_string()),
            &config,
            &mut state,
            "interrupt-1",
        )
        .unwrap();
        assert!(has_auth_credential(&config, &state));
    }

    #[test]
    fn process_auth_resume_errors_on_an_oauth_state_mismatch() {
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::OAuth2(Box::new(
            crate::auth_schemes::OAuth2Scheme {
                description: None,
                flows: crate::auth_schemes::OAuthFlows::default(),
            },
        ))));
        let config = AuthConfig::new(scheme, None, None, Some("my_oauth".to_string()));
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        state.set(
            oauth_state_key("interrupt-1"),
            Value::String("expected-state".to_string()),
        );

        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(crate::auth_credential::OAuth2Auth {
            state: Some("different-state".to_string()),
            ..Default::default()
        });
        let response = rusty_serde::json::to_value(&AuthConfig {
            exchanged_auth_credential: Some(credential),
            ..config.clone()
        })
        .unwrap();

        let err = process_auth_resume(&response, &config, &mut state, "interrupt-1").unwrap_err();
        assert!(matches!(err, ProcessAuthResumeError::OauthStateMismatch));
    }
}
