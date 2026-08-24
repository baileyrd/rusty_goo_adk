//! C0615 (local part): `LocalEvalSetResultsManager`, ported from
//! `google.adk.evaluation.local_eval_set_results_manager`. Stores eval
//! set results as `.evalset_result.json` files under
//! `<agents_dir>/<app_name>/.adk/eval_history/`.

use std::fs;
use std::path::PathBuf;

use adk_errors::not_found::NotFoundError;

use crate::eval_result::{EvalCaseResult, EvalSetResult};
use crate::eval_set_results_manager::EvalSetResultsManager;
use crate::eval_set_results_manager_utils::{create_eval_set_result, parse_eval_set_result_json};
use crate::eval_sets_manager::EvalManagerError;
use crate::path_validation::validate_path_segment;

const ADK_EVAL_HISTORY_DIR: &str = ".adk/eval_history";
const EVAL_SET_RESULT_FILE_EXTENSION: &str = ".evalset_result.json";

/// C0615 (local part): `local_eval_set_results_manager.LocalEvalSetResultsManager`.
///
/// See [`crate::local_eval_sets_manager::LocalEvalSetsManager`]'s doc for
/// the same disclosed "no pretty-printer, always writes every field"
/// narrowing this manager's `model_dump_json(indent=2)` inherits too.
pub struct LocalEvalSetResultsManager {
    agents_dir: PathBuf,
}

impl LocalEvalSetResultsManager {
    pub fn new(agents_dir: impl Into<PathBuf>) -> Self {
        Self {
            agents_dir: agents_dir.into(),
        }
    }

    fn eval_history_dir(&self, app_name: &str) -> Result<PathBuf, EvalManagerError> {
        validate_path_segment(app_name, "app_name")?;
        Ok(self.agents_dir.join(app_name).join(ADK_EVAL_HISTORY_DIR))
    }
}

impl EvalSetResultsManager for LocalEvalSetResultsManager {
    fn save_eval_set_result(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_results: &[EvalCaseResult],
    ) -> Result<(), EvalManagerError> {
        validate_path_segment(app_name, "app_name")?;
        validate_path_segment(eval_set_id, "eval_set_id")?;

        let eval_set_result = create_eval_set_result(
            app_name,
            eval_set_id,
            eval_case_results.to_vec(),
            adk_platform::time::get_time(),
        );

        let app_eval_history_dir = self.eval_history_dir(app_name)?;
        fs::create_dir_all(&app_eval_history_dir)?;

        let eval_set_result_name =
            eval_set_result
                .eval_set_result_name
                .as_ref()
                .ok_or_else(|| {
                    EvalManagerError::InvalidArgument(
                        "A newly created eval set result must have a name.".to_string(),
                    )
                })?;
        let path = app_eval_history_dir.join(format!(
            "{eval_set_result_name}{EVAL_SET_RESULT_FILE_EXTENSION}"
        ));
        let json = rusty_serde::json::to_string(&eval_set_result)
            .map_err(|error| EvalManagerError::InvalidArgument(error.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    fn get_eval_set_result(
        &self,
        app_name: &str,
        eval_set_result_id: &str,
    ) -> Result<EvalSetResult, EvalManagerError> {
        validate_path_segment(eval_set_result_id, "eval_set_result_id")?;
        let path = self.eval_history_dir(app_name)?.join(format!(
            "{eval_set_result_id}{EVAL_SET_RESULT_FILE_EXTENSION}"
        ));
        if !path.exists() {
            return Err(EvalManagerError::NotFound(NotFoundError::new(format!(
                "Eval set result `{eval_set_result_id}` not found."
            ))));
        }
        let content = fs::read_to_string(path)?;
        parse_eval_set_result_json(&content).map_err(EvalManagerError::InvalidArgument)
    }

    fn list_eval_set_results(&self, app_name: &str) -> Result<Vec<String>, EvalManagerError> {
        let dir = self.eval_history_dir(app_name)?;
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&dir)?;
        let results: Vec<String> = entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let file_name = entry.file_name();
                let file_name = file_name.to_str()?.to_string();
                file_name
                    .strip_suffix(EVAL_SET_RESULT_FILE_EXTENSION)
                    .map(|stem| stem.to_string())
            })
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "adk_eval_local_eval_set_results_manager_{name}_{}",
            adk_platform::uuid::new_uuid()
        ));
        dir
    }

    #[test]
    fn save_then_list_then_get_round_trips() {
        let manager = LocalEvalSetResultsManager::new(temp_dir("crud"));
        manager.save_eval_set_result("app", "set-1", &[]).unwrap();

        let ids = manager.list_eval_set_results("app").unwrap();
        assert_eq!(ids.len(), 1);

        let fetched = manager.get_eval_set_result("app", &ids[0]).unwrap();
        assert_eq!(fetched.eval_set_id, "set-1");
    }

    #[test]
    fn list_eval_set_results_returns_empty_for_an_app_with_no_history() {
        let manager = LocalEvalSetResultsManager::new(temp_dir("empty"));
        assert_eq!(
            manager.list_eval_set_results("app").unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn get_eval_set_result_errors_when_missing() {
        let manager = LocalEvalSetResultsManager::new(temp_dir("missing"));
        assert!(manager.get_eval_set_result("app", "missing").is_err());
    }
}
