//! Capability C0028: `_make_json_serializable`, ported from
//! `google.adk.events.event_actions`.
//!
//! **Adaptation**: the source coerces an arbitrary Python object (which
//! might not be JSON-serializable — e.g. a callable stashed in session
//! state) into something JSON-safe, falling back to a string
//! representation with a logged warning on failure. Rust's type system
//! makes the failure mode far rarer (a value has to actually implement
//! [`rusty_serde::Serialize`] to be offered to this function at all), but
//! the *shape* of the capability — never hard-fail, always produce
//! something JSON-safe — is preserved: a serialization error still falls
//! back to a debug-formatted string [`Value`] rather than propagating.

use rusty_serde::value::Value;
use rusty_serde::Serialize;

/// Converts `value` into a JSON-safe [`Value`], falling back to a
/// debug-formatted string if serialization fails for any reason.
pub fn make_json_serializable<T>(value: &T) -> Value
where
    T: Serialize + std::fmt::Debug,
{
    match rusty_serde::json::to_value(value) {
        Ok(v) => v,
        Err(_) => Value::String(format!("{value:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializable_value_converts_normally() {
        let v = make_json_serializable(&42i32);
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn a_string_converts_to_a_value_string() {
        let v = make_json_serializable(&"hello".to_string());
        assert_eq!(v, Value::String("hello".to_string()));
    }
}
