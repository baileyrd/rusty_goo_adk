//! Capabilities C0119-C0120: `LlmResponse`, ported from
//! `google.adk.models.llm_response`.
//!
//! **Deferred**: `LlmResponse.create(generate_content_response)` (C0121) —
//! the static factory mapping a raw Gemini SDK `GenerateContentResponse`
//! into an `LlmResponse` — needs that raw SDK response type, which only
//! exists once the native Gemini backend (Phase 3 batch 2, needs an HTTP
//! client decision) is built. Nothing to map from yet.
//!
//! **Adaptation**: fields typed as opaque `google.genai.types.*` shapes this
//! migration doesn't model (`GroundingMetadata`, `TurnCompleteReason`,
//! `FinishReason`, `GenerateContentResponseUsageMetadata`, live-session/
//! transcription/logprobs/citation types) stay [`rusty_serde::value::Value`]
//! placeholders — same rationale as `run_config.rs` in Phase 2. `content`
//! is the real `adk_genai::content::Content`.

use adk_genai::content::{Content, FunctionCall, FunctionResponse};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::cache_metadata::CacheMetadata;

/// LLM response class providing the first candidate response from the
/// model, if available; otherwise an error code and message.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LlmResponse {
    #[rusty_serde(default)]
    pub model_version: Option<String>,
    #[rusty_serde(default)]
    pub content: Option<Content>,
    #[rusty_serde(default)]
    pub grounding_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub partial: Option<bool>,
    #[rusty_serde(default)]
    pub turn_complete: Option<bool>,
    #[rusty_serde(default)]
    pub turn_complete_reason: Option<Value>,
    #[rusty_serde(default)]
    pub finish_reason: Option<Value>,
    #[rusty_serde(default)]
    pub error_code: Option<String>,
    #[rusty_serde(default)]
    pub error_message: Option<String>,
    #[rusty_serde(default)]
    pub interrupted: Option<bool>,
    #[rusty_serde(default)]
    pub custom_metadata: Option<BTreeMap<String, Value>>,
    #[rusty_serde(default)]
    pub usage_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub live_session_resumption_update: Option<Value>,
    #[rusty_serde(default)]
    pub live_session_id: Option<String>,
    #[rusty_serde(default)]
    pub go_away: Option<Value>,
    #[rusty_serde(default)]
    pub voice_activity: Option<Value>,
    #[rusty_serde(default)]
    pub input_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub output_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub avg_logprobs: Option<f64>,
    #[rusty_serde(default)]
    pub logprobs_result: Option<Value>,
    #[rusty_serde(default)]
    pub cache_metadata: Option<CacheMetadata>,
    #[rusty_serde(default)]
    pub citation_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub interaction_id: Option<String>,
    #[rusty_serde(default)]
    pub environment_id: Option<String>,
}

impl LlmResponse {
    /// C0120: the function calls in the response.
    pub fn get_function_calls(&self) -> Vec<&FunctionCall> {
        self.content
            .as_ref()
            .map(Content::get_function_calls)
            .unwrap_or_default()
    }

    /// C0120: the function responses in the response.
    pub fn get_function_responses(&self) -> Vec<&FunctionResponse> {
        self.content
            .as_ref()
            .map(Content::get_function_responses)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::Part;

    #[test]
    fn get_function_calls_and_responses_read_through_content() {
        let mut response = LlmResponse::default();
        assert!(response.get_function_calls().is_empty());

        response.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                name: Some("tool".to_string()),
                ..Default::default()
            })],
        ));
        assert_eq!(response.get_function_calls().len(), 1);
        assert!(response.get_function_responses().is_empty());
    }

    #[test]
    fn round_trips_with_camel_case_field_names() {
        let response = LlmResponse {
            content: Some(Content::user_text("hi")),
            turn_complete: Some(true),
            ..Default::default()
        };
        let json = rusty_serde::json::to_string(&response).unwrap();
        assert!(json.contains("\"turnComplete\""));
        let back: LlmResponse = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(response, back);
    }
}
