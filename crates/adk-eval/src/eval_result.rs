//! C0609: `EvalCaseResult`/`EvalSetResult`, ported from
//! `google.adk.evaluation.eval_result`.
//!
//! **Disclosed narrowing**: `EvalCaseResult.session_details`'s real type
//! (`sessions.session.Session`) already exists in this workspace
//! (`adk_agents::session::Session`), but `adk-eval` is deliberately kept
//! at the bottom of the crate graph (only `adk-genai` + `rusty_serde`) —
//! nothing in this batch reads `session_details`' structure, it's purely
//! a passthrough projection carried in the result for the caller's own
//! inspection. Pulling in `adk-agents` (and its own dependency tree —
//! `adk-platform`/`adk-errors`/`adk-events`/`rusty_tokio`/`regex`) for one
//! opaque field would invert that design intentionally, so it stays an
//! opaque `Value` here, same "widen once a real consumer needs the
//! structure" convention used throughout this crate.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::eval_metrics::{EvalMetric, EvalMetricResult, EvalMetricResultPerInvocation};
use crate::evaluator::EvalStatus;

/// C0609: `eval_result.EvalCaseResult` — case-level evaluation results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalCaseResult {
    /// Deprecated; use `eval_set_id` instead.
    #[rusty_serde(default)]
    pub eval_set_file: Option<String>,
    #[rusty_serde(default)]
    pub eval_set_id: String,
    #[rusty_serde(default)]
    pub eval_id: String,
    pub final_eval_status: EvalStatus,
    /// Deprecated; use `overall_eval_metric_results` instead.
    #[rusty_serde(default)]
    pub eval_metric_results: Option<Vec<(EvalMetric, EvalMetricResult)>>,
    pub overall_eval_metric_results: Vec<EvalMetricResult>,
    pub eval_metric_result_per_invocation: Vec<EvalMetricResultPerInvocation>,
    pub session_id: String,
    /// See the module doc for why this stays an opaque `Value`.
    #[rusty_serde(default)]
    pub session_details: Option<Value>,
    #[rusty_serde(default)]
    pub user_id: Option<String>,
}

/// C0609: `eval_result.EvalSetResult` — eval-set-level evaluation
/// results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalSetResult {
    pub eval_set_result_id: String,
    #[rusty_serde(default)]
    pub eval_set_result_name: Option<String>,
    pub eval_set_id: String,
    #[rusty_serde(default)]
    pub eval_case_results: Vec<EvalCaseResult>,
    #[rusty_serde(default)]
    pub creation_timestamp: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_case_result() -> EvalCaseResult {
        EvalCaseResult {
            eval_set_file: None,
            eval_set_id: "set-1".to_string(),
            eval_id: "case-1".to_string(),
            final_eval_status: EvalStatus::Passed,
            eval_metric_results: None,
            overall_eval_metric_results: Vec::new(),
            eval_metric_result_per_invocation: Vec::new(),
            session_id: "session-1".to_string(),
            session_details: None,
            user_id: None,
        }
    }

    #[test]
    fn eval_case_result_round_trips_through_json_with_camel_case() {
        let result = eval_case_result();
        let json = rusty_serde::json::to_string(&result).unwrap();
        assert!(json.contains("\"evalSetId\""));
        assert!(json.contains("\"finalEvalStatus\""));
        assert!(json.contains("\"overallEvalMetricResults\""));
        let back: EvalCaseResult = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn eval_set_result_round_trips_through_json_with_camel_case() {
        let set_result = EvalSetResult {
            eval_set_result_id: "result-1".to_string(),
            eval_set_result_name: None,
            eval_set_id: "set-1".to_string(),
            eval_case_results: vec![eval_case_result()],
            creation_timestamp: 0.0,
        };
        let json = rusty_serde::json::to_string(&set_result).unwrap();
        assert!(json.contains("\"evalSetResultId\""));
        assert!(json.contains("\"evalCaseResults\""));
        let back: EvalSetResult = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(set_result, back);
    }
}
