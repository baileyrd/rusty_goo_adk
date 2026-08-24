//! Part of C0615: support functions ported from
//! `google.adk.evaluation._eval_set_results_manager_utils`, shared by
//! [`crate::local_eval_set_results_manager::LocalEvalSetResultsManager`]
//! (and would be shared by a future `GcsEvalSetResultsManager`, still
//! `REQUIRED`).

use crate::eval_result::{EvalCaseResult, EvalSetResult};

fn sanitize_eval_set_result_name(eval_set_result_name: &str) -> String {
    eval_set_result_name.replace('/', "_")
}

/// `_eval_set_results_manager_utils.create_eval_set_result`.
pub fn create_eval_set_result(
    app_name: &str,
    eval_set_id: &str,
    eval_case_results: Vec<EvalCaseResult>,
    timestamp: f64,
) -> EvalSetResult {
    let eval_set_result_id = format!("{app_name}_{eval_set_id}_{timestamp}");
    let eval_set_result_name = sanitize_eval_set_result_name(&eval_set_result_id);
    EvalSetResult {
        eval_set_result_id,
        eval_set_result_name: Some(eval_set_result_name),
        eval_set_id: eval_set_id.to_string(),
        eval_case_results,
        creation_timestamp: timestamp,
    }
}

/// `_eval_set_results_manager_utils.parse_eval_set_result_json` — parses
/// an `EvalSetResult` from JSON, back-compatible with legacy eval set
/// result files that were double-encoded (the outer JSON is a string
/// containing the inner JSON object).
pub fn parse_eval_set_result_json(eval_set_result_json: &str) -> Result<EvalSetResult, String> {
    if let Ok(result) = rusty_serde::json::from_str::<EvalSetResult>(eval_set_result_json) {
        return Ok(result);
    }
    // Fall back to the legacy double-encoded shape: the outer JSON is a
    // plain string containing the real (inner) JSON object.
    match rusty_serde::json::from_str::<String>(eval_set_result_json) {
        Ok(inner) => rusty_serde::json::from_str::<EvalSetResult>(&inner)
            .map_err(|error| format!("Failed to parse eval set result: {error}")),
        Err(_) => {
            // Not a JSON string either -- try treating it as a bare JSON
            // object one more time so the original (more useful) parse
            // error is what's reported.
            rusty_serde::json::from_str::<EvalSetResult>(eval_set_result_json)
                .map_err(|error| format!("Failed to parse eval set result: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::EvalStatus;

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
    fn create_eval_set_result_builds_the_id_and_sanitized_name() {
        let result = create_eval_set_result("app", "set/1", vec![eval_case_result()], 123.0);
        assert_eq!(result.eval_set_result_id, "app_set/1_123");
        assert_eq!(
            result.eval_set_result_name,
            Some("app_set_1_123".to_string())
        );
        assert_eq!(result.eval_case_results.len(), 1);
    }

    #[test]
    fn parse_eval_set_result_json_reads_the_current_format() {
        let result = create_eval_set_result("app", "set-1", vec![], 1.0);
        let json = rusty_serde::json::to_string(&result).unwrap();
        let parsed = parse_eval_set_result_json(&json).unwrap();
        assert_eq!(parsed, result);
    }

    #[test]
    fn parse_eval_set_result_json_reads_the_legacy_double_encoded_format() {
        let result = create_eval_set_result("app", "set-1", vec![], 1.0);
        let inner = rusty_serde::json::to_string(&result).unwrap();
        let double_encoded = rusty_serde::json::to_string(&inner).unwrap();
        let parsed = parse_eval_set_result_json(&double_encoded).unwrap();
        assert_eq!(parsed, result);
    }

    #[test]
    fn parse_eval_set_result_json_errors_on_garbage() {
        assert!(parse_eval_set_result_json("not json at all").is_err());
    }
}
