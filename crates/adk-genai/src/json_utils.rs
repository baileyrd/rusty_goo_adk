//! Capability C0931: `safe_json_loads`, ported from
//! `google.adk.utils._json_utils`.
//!
//! **Adaptation**: the source's `json.loads` returns a dynamically-typed
//! `Any`, so callers narrow the result themselves after parsing. Rust has
//! no equivalent dynamic return; this port is generic over the target
//! type `T: Deserialize`, matching the "the boundary already deals in a
//! concrete type or `Value`" convention used everywhere else in this port
//! — a caller that genuinely wants the source's untyped behavior can
//! instantiate `T = rusty_serde::value::Value`.
//!
//! The source raises `ValueError` on malformed input; this port returns
//! `Result<T, String>`, the same "no domain-specific exception type for a
//! plain `ValueError`" convention already used for other small ported
//! helpers with a message-only failure (e.g. `adk-tools::bash_tool`'s
//! `shlex_split`, `adk-tools::load_web_page`'s `parse_request_target`).

use rusty_serde::Deserialize;

/// `_json_utils.safe_json_loads` — parses `text` as JSON, returning a
/// uniform `Err` (rather than a deserializer-specific error type) with
/// `context` folded into the message when provided, mirroring the
/// source's `f' in {context}'` suffix.
pub fn safe_json_loads<T>(text: &str, context: Option<&str>) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    rusty_serde::json::from_str(text).map_err(|e| match context {
        Some(context) => format!("Invalid JSON in {context}: {e}"),
        None => format!("Invalid JSON: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_serde::value::Value;

    #[test]
    fn safe_json_loads_parses_valid_json() {
        let result: Value = safe_json_loads(r#"{"a":1}"#, None).unwrap();
        assert_eq!(result, Value::Map(vec![("a".to_string(), Value::Int(1))]));
    }

    #[test]
    fn safe_json_loads_reports_malformed_input() {
        let result: Result<Value, String> = safe_json_loads("{not json", None);
        assert!(result.is_err());
    }

    #[test]
    fn safe_json_loads_includes_the_context_in_the_error_message() {
        let result: Result<Value, String> = safe_json_loads("{not json", Some("session state"));
        let message = result.unwrap_err();
        assert!(message.contains("session state"), "message was: {message}");
    }
}
