//! C0613 (local part): `EvalSetsManager` trait, ported from
//! `google.adk.evaluation.eval_sets_manager`. `GcsEvalSetsManager` stays
//! `REQUIRED` — no GCS SDK dependency is decided in this workspace yet.

use adk_errors::input_validation::InputValidationError;
use adk_errors::not_found::NotFoundError;

use crate::eval_case::EvalCase;
use crate::eval_set::EvalSet;

/// Failure modes shared by every `EvalSetsManager`/`EvalSetResultsManager`
/// implementor. The source raises a mix of plain `ValueError` (invalid
/// id, already exists) and the typed `NotFoundError`; this port keeps
/// that same split as two variants rather than collapsing everything
/// into one string, so a caller can match on "not found" specifically
/// the way the source's own `except NotFoundError` callers do.
#[derive(Debug, rusty_err::Error)]
pub enum EvalManagerError {
    #[error("{0}")]
    NotFound(NotFoundError),
    #[error("{0}")]
    InvalidPath(InputValidationError),
    #[error("{0}")]
    InvalidArgument(String),
    /// Wraps a real filesystem failure — the source lets Python's own
    /// `OSError` subclasses propagate uncaught for anything beyond the
    /// `FileNotFoundError` it explicitly handles (translated to `None`/
    /// `NotFoundError` at each call site instead).
    #[error("{0}")]
    Io(std::io::Error),
}

impl From<NotFoundError> for EvalManagerError {
    fn from(error: NotFoundError) -> Self {
        Self::NotFound(error)
    }
}

impl From<InputValidationError> for EvalManagerError {
    fn from(error: InputValidationError) -> Self {
        Self::InvalidPath(error)
    }
}

impl From<std::io::Error> for EvalManagerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// C0613 (local part): `eval_sets_manager.EvalSetsManager` — an
/// interface to manage Eval Sets.
pub trait EvalSetsManager {
    /// Returns an `EvalSet` identified by `app_name`/`eval_set_id`, or
    /// `None` if not found.
    fn get_eval_set(&self, app_name: &str, eval_set_id: &str) -> Option<EvalSet>;

    /// Creates and returns an empty `EvalSet`.
    ///
    /// Errors if `eval_set_id` is not valid (source: one or more of
    /// `[a-zA-Z0-9_]`) or an eval set already exists.
    fn create_eval_set(
        &self,
        app_name: &str,
        eval_set_id: &str,
    ) -> Result<EvalSet, EvalManagerError>;

    /// Returns the `EvalSet` ids that belong to `app_name`.
    fn list_eval_sets(&self, app_name: &str) -> Result<Vec<String>, EvalManagerError>;

    /// Returns an `EvalCase` if found; otherwise `None`.
    fn get_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Option<EvalCase>;

    /// Adds `eval_case` to an existing `EvalSet`.
    fn add_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case: EvalCase,
    ) -> Result<(), EvalManagerError>;

    /// Updates an existing `EvalCase`.
    fn update_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        updated_eval_case: EvalCase,
    ) -> Result<(), EvalManagerError>;

    /// Deletes the given `EvalCase`.
    fn delete_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Result<(), EvalManagerError>;
}
