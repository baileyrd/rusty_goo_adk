//! Capability C0026: `EventActions`, ported from
//! `google.adk.events.event_actions`.

use crate::event_compaction::EventCompaction;
use crate::json_safe::make_json_serializable;
use crate::ui_widget::UiWidget;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Side-effects and routing decisions attached to an [`crate::Event`].
///
/// **Adaptation, permanent**: `requested_auth_configs`/
/// `requested_tool_confirmations` hold the source's `AuthConfig`/
/// `ToolConfirmation` types. Both now exist in this port
/// (`adk-agents::auth_tool::AuthConfig`, `adk-tools::tool_confirmation::
/// ToolConfirmation`) — but `adk-events` sits *beneath* both crates in the
/// dependency graph (`adk-agents`/`adk-tools` depend on `adk-events`, never
/// the reverse), so `EventActions` can't reference either type without a
/// crate cycle. These stay JSON [`Value`] placeholders for good, not
/// pending a later phase; callers that need the typed value (see
/// `adk-flows::functions_utils::{generate_auth_event,
/// generate_request_confirmation_event}`) round-trip through
/// `rusty_serde::json::from_value` instead. `set_model_response` is
/// likewise a placeholder for an arbitrary structured-output value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EventActions {
    #[rusty_serde(default)]
    pub skip_summarization: bool,
    #[rusty_serde(default)]
    pub state_delta: HashMap<String, Value>,
    #[rusty_serde(default)]
    pub artifact_delta: HashMap<String, i64>,
    #[rusty_serde(default)]
    pub transfer_to_agent: Option<String>,
    #[rusty_serde(default)]
    pub escalate: bool,
    #[rusty_serde(default)]
    pub requested_auth_configs: HashMap<String, Value>,
    #[rusty_serde(default)]
    pub requested_tool_confirmations: HashMap<String, Value>,
    #[rusty_serde(default)]
    pub compaction: Option<EventCompaction>,
    #[rusty_serde(default)]
    pub end_of_agent: bool,
    #[rusty_serde(default)]
    pub agent_state: Option<HashMap<String, Value>>,
    #[rusty_serde(default)]
    pub rewind_before_invocation_id: Option<String>,
    #[rusty_serde(default)]
    pub route: Option<Value>,
    #[rusty_serde(default)]
    pub render_ui_widgets: Option<Vec<UiWidget>>,
    #[rusty_serde(default)]
    pub set_model_response: Option<Value>,
}

impl EventActions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a state-delta entry, JSON-safe-coercing the value first —
    /// mirrors the source's wrap-serializer fallback for unserializable
    /// values (see [`crate::json_safe::make_json_serializable`]).
    pub fn set_state<T>(&mut self, key: impl Into<String>, value: &T)
    where
        T: rusty_serde::Serialize + std::fmt::Debug,
    {
        self.state_delta
            .insert(key.into(), make_json_serializable(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_no_op_actions_value() {
        let actions = EventActions::default();
        assert!(!actions.skip_summarization);
        assert!(!actions.escalate);
        assert!(actions.state_delta.is_empty());
    }

    #[test]
    fn set_state_json_safe_coerces_the_value() {
        let mut actions = EventActions::new();
        actions.set_state("count", &7i32);
        assert_eq!(actions.state_delta.get("count"), Some(&Value::Int(7)));
    }

    #[test]
    fn rejects_unknown_fields() {
        let json = r#"{"skipSummarization":false,"stateDelta":{},"artifactDelta":{},"escalate":false,"requestedAuthConfigs":{},"requestedToolConfirmations":{},"endOfAgent":false,"bogus":true}"#;
        assert!(rusty_serde::json::from_str::<EventActions>(json).is_err());
    }

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let mut actions = EventActions::new();
        actions.transfer_to_agent = Some("other_agent".to_string());
        let json = rusty_serde::json::to_string(&actions).unwrap();
        assert!(json.contains("\"transferToAgent\":\"other_agent\""));
        let back: EventActions = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(actions, back);
    }
}
