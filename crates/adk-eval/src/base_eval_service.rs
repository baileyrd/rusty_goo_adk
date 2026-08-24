//! C0616: `evaluation.base_eval_service`, ported from
//! `google.adk.evaluation.base_eval_service`.
//!
//! **`BaseEvalService`, adaptation disclosed**: the source's
//! `perform_inference`/`evaluate` are `AsyncGenerator`s — they yield
//! results one at a time as they become available, rather than
//! returning a completed collection. This crate has no async story yet
//! (see the crate root doc: every `Evaluator`/`UserSimulator` built so
//! far is sync), so this port's [`BaseEvalService`] trait returns a
//! fully-materialized `Vec` instead of an item-at-a-time stream — the
//! same "collected `Vec<Event>`, not a live stream" adaptation already
//! disclosed for `BaseAgent`'s own `run_async_impl`. A real
//! streaming/async implementation is a larger, separate undertaking than
//! this batch's pure-data-model scope.

use rusty_serde::{Deserialize, Serialize};

use crate::eval_case::Invocation;
use crate::eval_metrics::EvalMetric;
use crate::eval_result::EvalCaseResult;

/// C0616: `base_eval_service.EvaluateConfig` — configurations needed to
/// run evaluations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvaluateConfig {
    pub eval_metrics: Vec<EvalMetric>,
    #[rusty_serde(default = "default_parallelism")]
    pub parallelism: i64,
}

fn default_parallelism() -> i64 {
    4
}

/// C0616: `base_eval_service.InferenceConfig` — configurations needed to
/// run inferences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    #[rusty_serde(default)]
    pub labels: Option<std::collections::HashMap<String, String>>,
    #[rusty_serde(default = "default_parallelism")]
    pub parallelism: i64,
    #[rusty_serde(default)]
    pub use_live: bool,
    #[rusty_serde(default = "default_live_timeout_seconds")]
    pub live_timeout_seconds: u64,
}

fn default_live_timeout_seconds() -> u64 {
    crate::constants::DEFAULT_LIVE_TIMEOUT_SECONDS
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            labels: None,
            parallelism: default_parallelism(),
            use_live: false,
            live_timeout_seconds: default_live_timeout_seconds(),
        }
    }
}

/// C0616: `base_eval_service.InferenceRequest` — a request to perform
/// inferences for the eval cases in an eval set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct InferenceRequest {
    pub app_name: String,
    pub eval_set_id: String,
    #[rusty_serde(default)]
    pub eval_case_ids: Option<Vec<String>>,
    pub inference_config: InferenceConfig,
}

/// `base_eval_service.InferenceStatus` — status of the inference.
///
/// **Wire format, disclosed**: the source is a plain (non-`str`) `Enum`
/// with int values (`UNKNOWN = 0`, ...), so under Pydantic v2's default
/// enum serialization the wire form is the bare integer. This port
/// serializes the variant name instead, the same disclosed, purely
/// cosmetic choice already made for `eval_metrics::EvalStatus` (no
/// cross-language consumer of this wire format exists yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InferenceStatus {
    #[default]
    Unknown,
    Success,
    Failure,
}

/// C0616: `base_eval_service.InferenceResult` — inference results for a
/// single eval case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct InferenceResult {
    pub app_name: String,
    pub eval_set_id: String,
    pub eval_case_id: String,
    #[rusty_serde(default)]
    pub inferences: Option<Vec<Invocation>>,
    pub session_id: Option<String>,
    #[rusty_serde(default)]
    pub status: InferenceStatus,
    #[rusty_serde(default)]
    pub error_message: Option<String>,
}

/// C0616: `base_eval_service.EvaluateRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvaluateRequest {
    pub inference_results: Vec<InferenceResult>,
    pub evaluate_config: EvaluateConfig,
}

/// C0616: `base_eval_service.BaseEvalService` — a service to run Evals
/// for an ADK agent. See this module's doc for the async-generator →
/// collected-`Vec` adaptation.
pub trait BaseEvalService {
    /// Returns the `InferenceResult`s obtained from the Agent.
    fn perform_inference(&self, inference_request: &InferenceRequest) -> Vec<InferenceResult>;

    /// Returns the `EvalCaseResult`s from performing metric evaluations
    /// on the given inferences.
    fn evaluate(&self, evaluate_request: &EvaluateRequest) -> Vec<EvalCaseResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_config_defaults_match_the_source() {
        let config = InferenceConfig::default();
        assert_eq!(config.labels, None);
        assert_eq!(config.parallelism, 4);
        assert!(!config.use_live);
        assert_eq!(config.live_timeout_seconds, 300);
    }

    #[test]
    fn inference_config_deserializes_with_defaults_from_an_empty_object() {
        let config: InferenceConfig = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(config, InferenceConfig::default());
    }

    #[test]
    fn evaluate_config_defaults_parallelism_to_four() {
        let config: EvaluateConfig = rusty_serde::json::from_str(r#"{"evalMetrics":[]}"#).unwrap();
        assert_eq!(config.parallelism, 4);
    }

    #[test]
    fn inference_status_defaults_to_unknown() {
        assert_eq!(InferenceStatus::default(), InferenceStatus::Unknown);
    }

    #[test]
    fn inference_request_round_trips_through_json_with_camel_case() {
        let request = InferenceRequest {
            app_name: "my_app".to_string(),
            eval_set_id: "set-1".to_string(),
            eval_case_ids: Some(vec!["case-1".to_string()]),
            inference_config: InferenceConfig::default(),
        };
        let json = rusty_serde::json::to_string(&request).unwrap();
        assert!(json.contains("\"appName\""));
        assert!(json.contains("\"evalSetId\""));
        assert!(json.contains("\"evalCaseIds\""));
        assert!(json.contains("\"inferenceConfig\""));
        let back: InferenceRequest = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(request, back);
    }

    #[test]
    fn inference_result_round_trips_through_json_with_camel_case() {
        let result = InferenceResult {
            app_name: "my_app".to_string(),
            eval_set_id: "set-1".to_string(),
            eval_case_id: "case-1".to_string(),
            inferences: None,
            session_id: Some("session-1".to_string()),
            status: InferenceStatus::Success,
            error_message: None,
        };
        let json = rusty_serde::json::to_string(&result).unwrap();
        assert!(json.contains("\"evalCaseId\""));
        let back: InferenceResult = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }
}
