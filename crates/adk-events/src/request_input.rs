//! Capability C0030: `RequestInput`, ported from
//! `google.adk.events.request_input`.

use adk_platform::uuid::new_uuid;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// A request for the user (or an out-of-band flow) to supply input mid-run
/// — the payload of an `adk_request_input`/HITL interrupt event.
///
/// **Adaptation**: the source's `response_schema` may be a Pydantic model
/// class, a generic type alias, or a raw JSON-Schema dict (defaulting to
/// `Any`) — a type reference, not just data. Rust has no direct analog for
/// carrying "a type" as a runtime value, so this is represented as an
/// optional JSON-Schema-shaped [`Value`] instead; a caller that needs to
/// validate a response against it does so by interpreting that JSON
/// Schema, not by holding a live Rust type handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RequestInput {
    pub interrupt_id: String,
    #[rusty_serde(default)]
    pub payload: Option<Value>,
    #[rusty_serde(default)]
    pub message: Option<String>,
    #[rusty_serde(default)]
    pub response_schema: Option<Value>,
}

impl RequestInput {
    /// Constructs a new `RequestInput` with a freshly generated
    /// `interrupt_id`. The source documents this id as reusable across
    /// retry-loop iterations for count-based function-call/response
    /// matching — callers that resume/retry should clone the original
    /// `interrupt_id` forward rather than calling this constructor again.
    pub fn new(
        message: Option<String>,
        payload: Option<Value>,
        response_schema: Option<Value>,
    ) -> Self {
        Self {
            interrupt_id: new_uuid(),
            payload,
            message,
            response_schema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test for capability C0030: `interrupt_id` is auto-assigned
    /// and unique per constructed `RequestInput`.
    #[test]
    fn interrupt_id_is_auto_assigned_and_unique() {
        let a = RequestInput::new(Some("pick one".to_string()), None, None);
        let b = RequestInput::new(Some("pick one".to_string()), None, None);
        assert_ne!(a.interrupt_id, b.interrupt_id);
        assert!(!a.interrupt_id.is_empty());
    }

    #[test]
    fn serializes_with_camel_case_field_names() {
        let req = RequestInput::new(Some("hi".to_string()), None, None);
        let json = rusty_serde::json::to_string(&req).unwrap();
        assert!(json.contains("\"interruptId\""));
    }
}
