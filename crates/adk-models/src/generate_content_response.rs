//! The wire shape of `google.genai.types.GenerateContentResponse` — the raw
//! REST response `LlmResponse::create` (C0121) maps from, and (via
//! [`GenerateContentResponse`]'s `Deserialize`) what `gemini.rs`'s real
//! `generate_content_async` parses a Gemini API HTTP response body into.
//!
//! **Adaptation**: same "minimal real subset" treatment as `adk_genai`'s
//! `Content` — only the fields `LlmResponse::create` actually reads
//! (`candidates[0].{content,finish_reason,finish_message,
//! grounding_metadata,citation_metadata,avg_logprobs,logprobs_result}`,
//! `prompt_feedback.{block_reason,block_reason_message}`, top-level
//! `usage_metadata`/`model_version`) are modeled. `finish_reason` and
//! `block_reason` are opaque [`rusty_serde::value::Value`] placeholders
//! (real enums on the wire, but `LlmResponse`'s own `finish_reason` field is
//! already an opaque placeholder — see `llm_response.rs` — so there's
//! nothing narrower to convert into); the source's direct enum-to-`str`
//! assignment (`error_code = candidate.finish_reason`) is reproduced by
//! [`value_to_string`], which assumes — true for every real Gemini
//! response — that these values are always plain strings on the wire.

use rusty_serde::value::Value;
use rusty_serde::Deserialize;

use adk_genai::content::Content;

/// `google.genai.types.Candidate` — see the module doc for the field
/// subset.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Candidate {
    #[rusty_serde(default)]
    pub content: Option<Content>,
    #[rusty_serde(default)]
    pub finish_reason: Option<Value>,
    #[rusty_serde(default)]
    pub finish_message: Option<String>,
    #[rusty_serde(default)]
    pub grounding_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub citation_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub avg_logprobs: Option<f64>,
    #[rusty_serde(default)]
    pub logprobs_result: Option<Value>,
}

/// `google.genai.types.GenerateContentResponsePromptFeedback`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[rusty_serde(default)]
    pub block_reason: Option<Value>,
    #[rusty_serde(default)]
    pub block_reason_message: Option<String>,
}

/// `google.genai.types.GenerateContentResponse`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[rusty_serde(default)]
    pub model_version: Option<String>,
    #[rusty_serde(default)]
    pub candidates: Option<Vec<Candidate>>,
    #[rusty_serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
    #[rusty_serde(default)]
    pub usage_metadata: Option<Value>,
}

/// Extracts a plain string from an opaque wire value — the source's
/// `FinishReason`/`BlockedReason` are `str` enums, so assigning one
/// directly to a `str`-typed field (as `LlmResponse.error_code` does) is
/// just reading the string underneath. See the module doc's adaptation
/// note.
pub fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_a_stop_finish_reason_candidate() {
        let json = r#"{
            "modelVersion": "gemini-2.5-flash-001",
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP"
            }]
        }"#;
        let response: GenerateContentResponse = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(
            response.model_version.as_deref(),
            Some("gemini-2.5-flash-001")
        );
        let candidate = &response.candidates.unwrap()[0];
        assert_eq!(
            candidate.finish_reason,
            Some(Value::String("STOP".to_string()))
        );
        assert_eq!(
            value_to_string(candidate.finish_reason.as_ref().unwrap()).as_deref(),
            Some("STOP")
        );
    }

    #[test]
    fn deserializes_prompt_feedback() {
        let json =
            r#"{"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "blocked"}}"#;
        let response: GenerateContentResponse = rusty_serde::json::from_str(json).unwrap();
        let feedback = response.prompt_feedback.unwrap();
        assert_eq!(
            feedback.block_reason,
            Some(Value::String("SAFETY".to_string()))
        );
        assert_eq!(feedback.block_reason_message.as_deref(), Some("blocked"));
    }

    #[test]
    fn value_to_string_returns_none_for_a_non_string_value() {
        assert_eq!(value_to_string(&Value::Bool(true)), None);
    }
}
