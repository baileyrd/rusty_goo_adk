//! Capability C0307 (partial): `Trigger`, ported from
//! `google.adk.workflow._trigger`.
//!
//! Represents a trigger for a downstream node in the P7 workflow/graph
//! engine. Pure data, no dependency on any other not-yet-built P7 piece —
//! see `workflow_node_state.rs`'s module doc for the rest of this row.
//!
//! **`model_config = ConfigDict(ser_json_bytes='base64')`, disclosed**:
//! this only changes how a `bytes`-typed field serializes; [`Trigger`]
//! has none (`input` is `Any`, opaque [`rusty_serde::value::Value`] in
//! this port) — nothing here actually exercises that config, so there's
//! nothing to port beyond noting it doesn't apply.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

fn default_value_null() -> Value {
    Value::Null
}

/// `workflow._trigger.Trigger`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Trigger {
    /// The input to pass to the triggered node.
    #[rusty_serde(default = "default_value_null")]
    pub input: Value,
    /// Whether this trigger should use a sub-branch.
    #[rusty_serde(default)]
    pub use_sub_branch: bool,
    /// The branch inherited from the predecessor node.
    #[rusty_serde(default)]
    pub branch: Option<String>,
    /// Scope tag explicitly propagated to this trigger.
    #[rusty_serde(default)]
    pub isolation_scope: Option<String>,
}

impl Default for Trigger {
    fn default() -> Self {
        Self {
            input: Value::Null,
            use_sub_branch: false,
            branch: None,
            isolation_scope: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_source() {
        let trigger = Trigger::default();
        assert_eq!(trigger.input, Value::Null);
        assert!(!trigger.use_sub_branch);
        assert_eq!(trigger.branch, None);
        assert_eq!(trigger.isolation_scope, None);
    }

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let trigger = Trigger {
            input: Value::String("hello".to_string()),
            use_sub_branch: true,
            branch: Some("main".to_string()),
            isolation_scope: Some("scope-1".to_string()),
        };
        let json = rusty_serde::json::to_string(&trigger).unwrap();
        assert!(json.contains("\"useSubBranch\":true"));
        assert!(json.contains("\"isolationScope\":\"scope-1\""));
        let back: Trigger = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(trigger, back);
    }

    #[test]
    fn deserializes_with_missing_optional_fields() {
        let trigger: Trigger = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(trigger, Trigger::default());
    }
}
