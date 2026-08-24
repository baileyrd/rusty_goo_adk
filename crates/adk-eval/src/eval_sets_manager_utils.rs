//! C0613 (local part) support functions, ported from
//! `google.adk.evaluation._eval_sets_manager_utils`. Shared by
//! [`crate::local_eval_sets_manager::LocalEvalSetsManager`] (and would be
//! shared by a future `GcsEvalSetsManager`, still `REQUIRED`).

use adk_errors::not_found::NotFoundError;

use crate::eval_case::EvalCase;
use crate::eval_set::EvalSet;
use crate::eval_sets_manager::{EvalManagerError, EvalSetsManager};

/// `_eval_sets_manager_utils.get_eval_set_from_app_and_id`.
pub fn get_eval_set_from_app_and_id(
    eval_sets_manager: &dyn EvalSetsManager,
    app_name: &str,
    eval_set_id: &str,
) -> Result<EvalSet, EvalManagerError> {
    eval_sets_manager
        .get_eval_set(app_name, eval_set_id)
        .ok_or_else(|| {
            EvalManagerError::NotFound(NotFoundError::new(format!(
                "Eval set `{eval_set_id}` not found."
            )))
        })
}

/// `_eval_sets_manager_utils.get_eval_case_from_eval_set`.
pub fn get_eval_case_from_eval_set(eval_set: &EvalSet, eval_case_id: &str) -> Option<EvalCase> {
    eval_set
        .eval_cases
        .iter()
        .find(|eval_case| eval_case.eval_id == eval_case_id)
        .cloned()
}

/// `_eval_sets_manager_utils.add_eval_case_to_eval_set`.
pub fn add_eval_case_to_eval_set(
    mut eval_set: EvalSet,
    eval_case: EvalCase,
) -> Result<EvalSet, EvalManagerError> {
    if eval_set
        .eval_cases
        .iter()
        .any(|existing| existing.eval_id == eval_case.eval_id)
    {
        return Err(EvalManagerError::InvalidArgument(format!(
            "Eval id `{}` already exists in `{}` eval set.",
            eval_case.eval_id, eval_set.eval_set_id
        )));
    }
    eval_set.eval_cases.push(eval_case);
    Ok(eval_set)
}

/// `_eval_sets_manager_utils.update_eval_case_in_eval_set`.
pub fn update_eval_case_in_eval_set(
    mut eval_set: EvalSet,
    updated_eval_case: EvalCase,
) -> Result<EvalSet, EvalManagerError> {
    let position = eval_set
        .eval_cases
        .iter()
        .position(|existing| existing.eval_id == updated_eval_case.eval_id)
        .ok_or_else(|| {
            EvalManagerError::NotFound(NotFoundError::new(format!(
                "Eval case `{}` not found in eval set `{}`.",
                updated_eval_case.eval_id, eval_set.eval_set_id
            )))
        })?;
    eval_set.eval_cases[position] = updated_eval_case;
    Ok(eval_set)
}

/// `_eval_sets_manager_utils.delete_eval_case_from_eval_set`.
pub fn delete_eval_case_from_eval_set(
    mut eval_set: EvalSet,
    eval_case_id: &str,
) -> Result<EvalSet, EvalManagerError> {
    let position = eval_set
        .eval_cases
        .iter()
        .position(|existing| existing.eval_id == eval_case_id)
        .ok_or_else(|| {
            EvalManagerError::NotFound(NotFoundError::new(format!(
                "Eval case `{eval_case_id}` not found in eval set `{}`.",
                eval_set.eval_set_id
            )))
        })?;
    eval_set.eval_cases.remove(position);
    Ok(eval_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_case(id: &str) -> EvalCase {
        EvalCase {
            eval_id: id.to_string(),
            conversation: Some(vec![]),
            ..Default::default()
        }
    }

    fn eval_set_with(cases: Vec<EvalCase>) -> EvalSet {
        EvalSet {
            eval_set_id: "set-1".to_string(),
            name: None,
            description: None,
            eval_cases: cases,
            creation_timestamp: 0.0,
        }
    }

    #[test]
    fn get_eval_case_from_eval_set_finds_by_id() {
        let eval_set = eval_set_with(vec![eval_case("c1"), eval_case("c2")]);
        let found = get_eval_case_from_eval_set(&eval_set, "c2");
        assert_eq!(found.map(|c| c.eval_id), Some("c2".to_string()));
    }

    #[test]
    fn get_eval_case_from_eval_set_returns_none_when_missing() {
        let eval_set = eval_set_with(vec![eval_case("c1")]);
        assert!(get_eval_case_from_eval_set(&eval_set, "missing").is_none());
    }

    #[test]
    fn add_eval_case_to_eval_set_appends() {
        let eval_set = eval_set_with(vec![]);
        let updated = add_eval_case_to_eval_set(eval_set, eval_case("c1")).unwrap();
        assert_eq!(updated.eval_cases.len(), 1);
    }

    #[test]
    fn add_eval_case_to_eval_set_rejects_a_duplicate_id() {
        let eval_set = eval_set_with(vec![eval_case("c1")]);
        assert!(add_eval_case_to_eval_set(eval_set, eval_case("c1")).is_err());
    }

    #[test]
    fn update_eval_case_in_eval_set_replaces_by_id() {
        let eval_set = eval_set_with(vec![eval_case("c1")]);
        let mut replacement = eval_case("c1");
        replacement.creation_timestamp = 42.0;
        let updated = update_eval_case_in_eval_set(eval_set, replacement).unwrap();
        assert_eq!(updated.eval_cases[0].creation_timestamp, 42.0);
    }

    #[test]
    fn update_eval_case_in_eval_set_errors_when_missing() {
        let eval_set = eval_set_with(vec![]);
        assert!(update_eval_case_in_eval_set(eval_set, eval_case("missing")).is_err());
    }

    #[test]
    fn delete_eval_case_from_eval_set_removes_by_id() {
        let eval_set = eval_set_with(vec![eval_case("c1"), eval_case("c2")]);
        let updated = delete_eval_case_from_eval_set(eval_set, "c1").unwrap();
        assert_eq!(updated.eval_cases.len(), 1);
        assert_eq!(updated.eval_cases[0].eval_id, "c2");
    }

    #[test]
    fn delete_eval_case_from_eval_set_errors_when_missing() {
        let eval_set = eval_set_with(vec![]);
        assert!(delete_eval_case_from_eval_set(eval_set, "missing").is_err());
    }
}
