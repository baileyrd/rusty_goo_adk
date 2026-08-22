//! Capabilities C0119-C0121: `LlmResponse`, ported from
//! `google.adk.models.llm_response`.
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
use crate::generate_content_response::{value_to_string, GenerateContentResponse};

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

    /// C0121: maps a raw `GenerateContentResponse` (as parsed from a Gemini
    /// REST API response body) into an `LlmResponse` — normal/error/
    /// block-reason/empty-no-candidates branches, in the same order and
    /// with the same precedence as the source's static factory.
    pub fn create(response: GenerateContentResponse) -> LlmResponse {
        if let Some(candidate) = response.candidates.as_ref().and_then(|c| c.first()) {
            let has_parts = candidate
                .content
                .as_ref()
                .map(|c| !c.parts.is_empty())
                .unwrap_or(false);
            let is_stop = candidate.finish_reason == Some(Value::String("STOP".to_string()));

            if has_parts || is_stop {
                return LlmResponse {
                    content: candidate.content.clone(),
                    grounding_metadata: candidate.grounding_metadata.clone(),
                    usage_metadata: response.usage_metadata.clone(),
                    finish_reason: candidate.finish_reason.clone(),
                    citation_metadata: candidate.citation_metadata.clone(),
                    avg_logprobs: candidate.avg_logprobs,
                    logprobs_result: candidate.logprobs_result.clone(),
                    model_version: response.model_version.clone(),
                    ..Default::default()
                };
            }
            return LlmResponse {
                error_code: candidate.finish_reason.as_ref().and_then(value_to_string),
                error_message: candidate.finish_message.clone(),
                citation_metadata: candidate.citation_metadata.clone(),
                usage_metadata: response.usage_metadata.clone(),
                finish_reason: candidate.finish_reason.clone(),
                avg_logprobs: candidate.avg_logprobs,
                logprobs_result: candidate.logprobs_result.clone(),
                model_version: response.model_version.clone(),
                ..Default::default()
            };
        }

        if let Some(feedback) = &response.prompt_feedback {
            return LlmResponse {
                error_code: feedback.block_reason.as_ref().and_then(value_to_string),
                error_message: feedback.block_reason_message.clone(),
                usage_metadata: response.usage_metadata.clone(),
                model_version: response.model_version.clone(),
                ..Default::default()
            };
        }

        // Some model backends can legitimately complete a turn without
        // candidates (e.g. tool-driven UI turns with no text) — an empty
        // successful response, not an unknown error.
        LlmResponse {
            content: Some(Content::new("model", vec![])),
            usage_metadata: response.usage_metadata,
            model_version: response.model_version,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_content_response::{Candidate, PromptFeedback};
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

    #[test]
    fn create_maps_a_normal_candidate_with_parts() {
        let response = GenerateContentResponse {
            model_version: Some("gemini-2.5-flash".to_string()),
            candidates: Some(vec![Candidate {
                content: Some(Content::user_text("hi")),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let llm_response = LlmResponse::create(response);
        assert_eq!(
            llm_response.content.unwrap().parts[0].text.as_deref(),
            Some("hi")
        );
        assert_eq!(
            llm_response.model_version.as_deref(),
            Some("gemini-2.5-flash")
        );
        assert!(llm_response.error_code.is_none());
    }

    #[test]
    fn create_maps_a_stop_candidate_with_no_parts_as_a_normal_response() {
        let response = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![])),
                finish_reason: Some(Value::String("STOP".to_string())),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let llm_response = LlmResponse::create(response);
        assert!(llm_response.error_code.is_none());
        assert_eq!(
            llm_response.finish_reason,
            Some(Value::String("STOP".to_string()))
        );
    }

    #[test]
    fn create_maps_a_non_stop_candidate_with_no_parts_as_an_error() {
        let response = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: None,
                finish_reason: Some(Value::String("SAFETY".to_string())),
                finish_message: Some("blocked by safety filters".to_string()),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let llm_response = LlmResponse::create(response);
        assert_eq!(llm_response.error_code.as_deref(), Some("SAFETY"));
        assert_eq!(
            llm_response.error_message.as_deref(),
            Some("blocked by safety filters")
        );
    }

    #[test]
    fn create_maps_prompt_feedback_when_there_are_no_candidates() {
        let response = GenerateContentResponse {
            candidates: None,
            prompt_feedback: Some(PromptFeedback {
                block_reason: Some(Value::String("SAFETY".to_string())),
                block_reason_message: Some("blocked".to_string()),
            }),
            ..Default::default()
        };
        let llm_response = LlmResponse::create(response);
        assert_eq!(llm_response.error_code.as_deref(), Some("SAFETY"));
        assert_eq!(llm_response.error_message.as_deref(), Some("blocked"));
    }

    #[test]
    fn create_maps_an_empty_response_as_a_successful_empty_content() {
        let response = GenerateContentResponse::default();
        let llm_response = LlmResponse::create(response);
        assert!(llm_response.error_code.is_none());
        let content = llm_response.content.unwrap();
        assert_eq!(content.role.as_deref(), Some("model"));
        assert!(content.parts.is_empty());
    }

    #[test]
    fn create_treats_an_empty_candidates_list_the_same_as_none() {
        let response = GenerateContentResponse {
            candidates: Some(vec![]),
            prompt_feedback: Some(PromptFeedback {
                block_reason: Some(Value::String("OTHER".to_string())),
                ..Default::default()
            }),
            ..Default::default()
        };
        let llm_response = LlmResponse::create(response);
        assert_eq!(llm_response.error_code.as_deref(), Some("OTHER"));
    }
}
