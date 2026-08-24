//! Capability C0416: `ToolConfirmation`, ported from
//! `google.adk.tools.tool_confirmation`.
//!
//! Represents a tool's confirmation configuration — a tool asks for
//! confirmation via an `adk_request_confirmation` function call, the
//! user's approval/denial comes back as a function response whose
//! `response` dict is parsed into one of these
//! ([`ToolConfirmation::from_response_dict`]). This is what
//! `adk-flows::request_confirmation`'s own module doc discloses as
//! deferred ("parsing a `ToolConfirmation` out of a confirmation
//! response") — now closed.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, rusty_err::Error)]
pub enum ToolConfirmationError {
    #[error("failed to parse ToolConfirmation from a function response dict: {0}")]
    InvalidResponse(String),
}

/// Represents a tool confirmation configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ToolConfirmation {
    /// The hint text for why the input is needed.
    #[rusty_serde(default)]
    pub hint: String,
    /// Whether the tool execution is confirmed.
    #[rusty_serde(default)]
    pub confirmed: bool,
    /// The custom data payload needed from the user to continue the flow.
    /// Should be JSON-serializable.
    #[rusty_serde(default)]
    pub payload: Option<Value>,
}

impl ToolConfirmation {
    /// Parses a `ToolConfirmation` from a function response dict. Handles
    /// both the direct dict format and the ADK client's
    /// `{'response': json_string}` wrapper format.
    pub fn from_response_dict(
        response: &BTreeMap<String, Value>,
    ) -> Result<ToolConfirmation, ToolConfirmationError> {
        if response.len() == 1 {
            if let Some(Value::String(wrapped)) = response.get("response") {
                return rusty_serde::json::from_str(wrapped)
                    .map_err(|e| ToolConfirmationError::InvalidResponse(e.to_string()));
            }
        }
        let value = Value::Map(
            response
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        rusty_serde::json::from_value(value)
            .map_err(|e| ToolConfirmationError::InvalidResponse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_source() {
        let confirmation = ToolConfirmation::default();
        assert_eq!(confirmation.hint, "");
        assert!(!confirmation.confirmed);
        assert!(confirmation.payload.is_none());
    }

    #[test]
    fn parses_from_a_direct_dict() {
        let mut response = BTreeMap::new();
        response.insert("hint".to_string(), Value::String("why".to_string()));
        response.insert("confirmed".to_string(), Value::Bool(true));
        let confirmation = ToolConfirmation::from_response_dict(&response).unwrap();
        assert_eq!(confirmation.hint, "why");
        assert!(confirmation.confirmed);
    }

    #[test]
    fn parses_from_the_client_wrapped_json_string_format() {
        let mut response = BTreeMap::new();
        response.insert(
            "response".to_string(),
            Value::String(r#"{"confirmed": true, "hint": "wrapped"}"#.to_string()),
        );
        let confirmation = ToolConfirmation::from_response_dict(&response).unwrap();
        assert_eq!(confirmation.hint, "wrapped");
        assert!(confirmation.confirmed);
    }

    #[test]
    fn a_single_key_dict_not_named_response_is_treated_as_a_direct_dict() {
        let mut response = BTreeMap::new();
        response.insert("confirmed".to_string(), Value::Bool(true));
        let confirmation = ToolConfirmation::from_response_dict(&response).unwrap();
        assert!(confirmation.confirmed);
    }

    #[test]
    fn rejects_unknown_fields() {
        let mut response = BTreeMap::new();
        response.insert("unexpected_field".to_string(), Value::Bool(true));
        assert!(ToolConfirmation::from_response_dict(&response).is_err());
    }

    #[test]
    fn round_trips_with_camel_case_field_names() {
        let confirmation = ToolConfirmation {
            hint: "h".to_string(),
            confirmed: true,
            payload: Some(Value::String("p".to_string())),
        };
        let json = rusty_serde::json::to_string(&confirmation).unwrap();
        let back: ToolConfirmation = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(confirmation, back);
    }
}
