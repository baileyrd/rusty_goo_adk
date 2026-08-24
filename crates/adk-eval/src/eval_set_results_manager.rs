//! C0615 (local part): `EvalSetResultsManager` trait, ported from
//! `google.adk.evaluation.eval_set_results_manager`. `GcsEvalSetResultsManager`
//! stays `REQUIRED` — no GCS SDK dependency is decided in this workspace
//! yet.

use crate::eval_result::{EvalCaseResult, EvalSetResult};
use crate::eval_sets_manager::EvalManagerError;

/// C0615 (local part): `eval_set_results_manager.EvalSetResultsManager`
/// — an interface to manage Eval Set Results.
pub trait EvalSetResultsManager {
    /// Creates and saves a new `EvalSetResult` given `eval_case_results`.
    fn save_eval_set_result(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_results: &[EvalCaseResult],
    ) -> Result<(), EvalManagerError>;

    /// Returns the `EvalSetResult` identified by `app_name`/
    /// `eval_set_result_id`.
    fn get_eval_set_result(
        &self,
        app_name: &str,
        eval_set_result_id: &str,
    ) -> Result<EvalSetResult, EvalManagerError>;

    /// Returns the eval result ids that belong to `app_name`.
    fn list_eval_set_results(&self, app_name: &str) -> Result<Vec<String>, EvalManagerError>;
}
