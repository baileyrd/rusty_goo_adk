//! Part of capability C0078: `TranscriptionEntry`, ported from
//! `google.adk.agents.transcription_entry`.
//!
//! **Adaptation**: `data: Union[types.Blob, types.Content]` is an opaque
//! Gemini-API payload, represented as [`rusty_serde::value::Value`] (same
//! rationale as `run_config`).

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// Stores data usable for transcription.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct TranscriptionEntry {
    /// The role that created this data, typically `"user"` or `"model"`. For
    /// a function call, this is `None`.
    #[rusty_serde(default)]
    pub role: Option<String>,
    pub data: Value,
}

impl TranscriptionEntry {
    pub fn new(role: Option<String>, data: Value) -> Self {
        Self { role, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let entry =
            TranscriptionEntry::new(Some("user".to_string()), Value::String("hello".to_string()));
        let json = rusty_serde::json::to_string(&entry).unwrap();
        let back: TranscriptionEntry = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn role_is_none_for_function_call_data() {
        let entry = TranscriptionEntry::new(None, Value::Null);
        assert_eq!(entry.role, None);
    }
}
