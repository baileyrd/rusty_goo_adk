//! C0617 (partial): `evaluation.local_eval_service`, ported from
//! `google.adk.evaluation.local_eval_service`'s `LocalEvalService`
//! (`@experimental`).
//!
//! Only [`generate_final_eval_status`] — the one synchronous,
//! dependency-free method on the source's 9-method class — is ported.
//! Every other method (`perform_inference`, `evaluate`,
//! `_evaluate_single_inference_result`, `_evaluate_metric_for_eval_case`,
//! `_evaluate_metric`, `_perform_inference_single_eval_item`) is `async`
//! and needs semaphore-bounded concurrency plus a real inference
//! generator driven by a `Runner` — that's C0621/C0622's still-open
//! scope, not this batch's. This row stays `Partial`, not `DONE`, until
//! that lands.

use crate::eval_metrics::{EvalMetricResult, EvalStatus};

/// `local_eval_service.LocalEvalService._generate_final_eval_status` —
/// rolls up a list of per-metric results into one overall `EvalStatus`.
///
/// A `Passed` result sets the running status to `Passed` but keeps
/// scanning (a later result can still override it); a `NotEvaluated`
/// result is skipped; a `Failed` result sets the running status to
/// `Failed` and stops immediately — even if some of the metrics that
/// were never reached would otherwise have passed. With no metric
/// results at all, or only `NotEvaluated` ones, the result is
/// `NotEvaluated`.
pub fn generate_final_eval_status(overall_eval_metric_results: &[EvalMetricResult]) -> EvalStatus {
    let mut final_eval_status = EvalStatus::NotEvaluated;

    for result in overall_eval_metric_results {
        match result.eval_status {
            EvalStatus::Passed => final_eval_status = EvalStatus::Passed,
            EvalStatus::NotEvaluated => continue,
            EvalStatus::Failed => {
                final_eval_status = EvalStatus::Failed;
                break;
            }
        }
    }

    final_eval_status
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(status: EvalStatus) -> EvalMetricResult {
        EvalMetricResult {
            metric_name: "some_metric".to_string(),
            threshold: None,
            criterion: None,
            custom_function_path: None,
            score: None,
            eval_status: status,
            details: Default::default(),
        }
    }

    #[test]
    fn empty_results_are_not_evaluated() {
        assert_eq!(generate_final_eval_status(&[]), EvalStatus::NotEvaluated);
    }

    #[test]
    fn all_not_evaluated_stays_not_evaluated() {
        let results = vec![
            result(EvalStatus::NotEvaluated),
            result(EvalStatus::NotEvaluated),
        ];
        assert_eq!(
            generate_final_eval_status(&results),
            EvalStatus::NotEvaluated
        );
    }

    #[test]
    fn a_single_passed_result_passes() {
        let results = vec![result(EvalStatus::NotEvaluated), result(EvalStatus::Passed)];
        assert_eq!(generate_final_eval_status(&results), EvalStatus::Passed);
    }

    #[test]
    fn a_failed_result_short_circuits_even_when_passed_results_follow() {
        let results = vec![
            result(EvalStatus::Passed),
            result(EvalStatus::Failed),
            result(EvalStatus::Passed),
        ];
        assert_eq!(generate_final_eval_status(&results), EvalStatus::Failed);
    }

    #[test]
    fn a_passed_result_after_not_evaluated_still_passes() {
        let results = vec![result(EvalStatus::NotEvaluated), result(EvalStatus::Passed)];
        assert_eq!(generate_final_eval_status(&results), EvalStatus::Passed);
    }

    #[test]
    fn all_passed_results_in_passes() {
        let results = vec![result(EvalStatus::Passed), result(EvalStatus::Passed)];
        assert_eq!(generate_final_eval_status(&results), EvalStatus::Passed);
    }
}
