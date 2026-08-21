//! Capability C0029: `UiWidget`, ported from `google.adk.events.ui_widget`.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// A UI widget rendered by an event's actions. `provider`'s only known
/// value in the source is `"mcp"` (MCP Apps `ui://` resource extension),
/// but the field is a plain string, not a closed enum, so other providers
/// aren't rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct UiWidget {
    pub id: String,
    pub provider: String,
    /// Provider-specific payload — for `provider == "mcp"`: `resource_uri`,
    /// `tool`, `tool_args`.
    pub payload: Value,
}

impl UiWidget {
    pub fn new(id: impl Into<String>, provider: impl Into<String>, payload: Value) -> Self {
        Self {
            id: id.into(),
            provider: provider.into(),
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_serde::value::Value;

    #[test]
    fn round_trips_through_json() {
        let widget = UiWidget::new(
            "w1",
            "mcp",
            Value::Map(vec![(
                "tool".to_string(),
                Value::String("search".to_string()),
            )]),
        );
        let json = rusty_serde::json::to_string(&widget).unwrap();
        let back: UiWidget = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(widget, back);
    }

    #[test]
    fn rejects_unknown_fields() {
        let json = r#"{"id":"w1","provider":"mcp","payload":{},"extra":true}"#;
        assert!(rusty_serde::json::from_str::<UiWidget>(json).is_err());
    }
}
