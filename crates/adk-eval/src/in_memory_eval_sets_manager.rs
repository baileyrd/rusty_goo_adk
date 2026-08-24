//! C0613 (local part): `InMemoryEvalSetsManager`, ported from
//! `google.adk.evaluation.in_memory_eval_sets_manager`. Useful as a part
//! of a test case, or when a real `LocalEvalSetsManager` is too
//! expensive to use.

use std::collections::HashMap;
use std::sync::Mutex;

use adk_errors::not_found::NotFoundError;

use crate::eval_case::EvalCase;
use crate::eval_set::EvalSet;
use crate::eval_sets_manager::{EvalManagerError, EvalSetsManager};

/// `{app_name: {eval_set_id: {eval_case_id: EvalCase}}}`.
type EvalCasesByAppAndSet = HashMap<String, HashMap<String, HashMap<String, EvalCase>>>;

/// C0613 (local part): `in_memory_eval_sets_manager.InMemoryEvalSetsManager`.
///
/// **Adaptation**: the source mutates plain instance dicts (`self.
/// _eval_sets`/`self._eval_cases`) from methods that only ever run
/// single-threaded in Python. This port's [`EvalSetsManager`] trait takes
/// `&self` (to stay object-safe alongside
/// [`crate::local_eval_sets_manager::LocalEvalSetsManager`], which has no
/// need for `&mut self` either), so the two maps are `Mutex`-guarded —
/// the same interior-mutability pattern
/// `adk_agents::in_memory_memory_service::InMemoryMemoryService` already
/// uses for the identical shape of problem.
#[derive(Default)]
pub struct InMemoryEvalSetsManager {
    eval_sets: Mutex<HashMap<String, HashMap<String, EvalSet>>>,
    eval_cases: Mutex<EvalCasesByAppAndSet>,
}

impl InMemoryEvalSetsManager {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EvalSetsManager for InMemoryEvalSetsManager {
    fn get_eval_set(&self, app_name: &str, eval_set_id: &str) -> Option<EvalSet> {
        self.eval_sets
            .lock()
            .unwrap()
            .get(app_name)
            .and_then(|sets| sets.get(eval_set_id))
            .cloned()
    }

    fn create_eval_set(
        &self,
        app_name: &str,
        eval_set_id: &str,
    ) -> Result<EvalSet, EvalManagerError> {
        let mut eval_sets = self.eval_sets.lock().unwrap();
        let app_eval_sets = eval_sets.entry(app_name.to_string()).or_default();
        if app_eval_sets.contains_key(eval_set_id) {
            return Err(EvalManagerError::InvalidArgument(format!(
                "EvalSet {eval_set_id} already exists for app {app_name}."
            )));
        }
        let new_eval_set = EvalSet {
            eval_set_id: eval_set_id.to_string(),
            name: None,
            description: None,
            eval_cases: Vec::new(),
            creation_timestamp: adk_platform::time::get_time(),
        };
        app_eval_sets.insert(eval_set_id.to_string(), new_eval_set.clone());
        self.eval_cases
            .lock()
            .unwrap()
            .entry(app_name.to_string())
            .or_default()
            .insert(eval_set_id.to_string(), HashMap::new());
        Ok(new_eval_set)
    }

    fn list_eval_sets(&self, app_name: &str) -> Result<Vec<String>, EvalManagerError> {
        Ok(self
            .eval_sets
            .lock()
            .unwrap()
            .get(app_name)
            .map(|sets| sets.keys().cloned().collect())
            .unwrap_or_default())
    }

    fn get_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Option<EvalCase> {
        self.eval_cases
            .lock()
            .unwrap()
            .get(app_name)
            .and_then(|sets| sets.get(eval_set_id))
            .and_then(|cases| cases.get(eval_case_id))
            .cloned()
    }

    fn add_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case: EvalCase,
    ) -> Result<(), EvalManagerError> {
        let mut eval_sets = self.eval_sets.lock().unwrap();
        let app_eval_sets = eval_sets.entry(app_name.to_string()).or_default();
        if !app_eval_sets.contains_key(eval_set_id) {
            return Err(NotFoundError::new(format!(
                "EvalSet {eval_set_id} not found for app {app_name}."
            ))
            .into());
        }

        let mut eval_cases = self.eval_cases.lock().unwrap();
        let app_eval_cases = eval_cases.entry(app_name.to_string()).or_default();
        let set_eval_cases = app_eval_cases.entry(eval_set_id.to_string()).or_default();
        if set_eval_cases.contains_key(&eval_case.eval_id) {
            return Err(EvalManagerError::InvalidArgument(format!(
                "EvalCase {} already exists in EvalSet {eval_set_id} for app {app_name}.",
                eval_case.eval_id
            )));
        }

        set_eval_cases.insert(eval_case.eval_id.clone(), eval_case.clone());
        app_eval_sets
            .get_mut(eval_set_id)
            .unwrap()
            .eval_cases
            .push(eval_case);
        Ok(())
    }

    fn update_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        updated_eval_case: EvalCase,
    ) -> Result<(), EvalManagerError> {
        let mut eval_sets = self.eval_sets.lock().unwrap();
        let app_eval_sets = eval_sets.entry(app_name.to_string()).or_default();
        if !app_eval_sets.contains_key(eval_set_id) {
            return Err(NotFoundError::new(format!(
                "EvalSet {eval_set_id} not found for app {app_name}."
            ))
            .into());
        }

        let mut eval_cases = self.eval_cases.lock().unwrap();
        let app_eval_cases = eval_cases.entry(app_name.to_string()).or_default();
        let set_eval_cases = app_eval_cases.entry(eval_set_id.to_string()).or_default();
        if !set_eval_cases.contains_key(&updated_eval_case.eval_id) {
            return Err(NotFoundError::new(format!(
                "EvalCase {} not found in EvalSet {eval_set_id} for app {app_name}.",
                updated_eval_case.eval_id
            ))
            .into());
        }

        set_eval_cases.insert(updated_eval_case.eval_id.clone(), updated_eval_case.clone());
        let eval_set = app_eval_sets.get_mut(eval_set_id).unwrap();
        if let Some(existing) = eval_set
            .eval_cases
            .iter_mut()
            .find(|case| case.eval_id == updated_eval_case.eval_id)
        {
            *existing = updated_eval_case;
        }
        Ok(())
    }

    fn delete_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Result<(), EvalManagerError> {
        let mut eval_sets = self.eval_sets.lock().unwrap();
        let app_eval_sets = eval_sets.entry(app_name.to_string()).or_default();
        if !app_eval_sets.contains_key(eval_set_id) {
            return Err(NotFoundError::new(format!(
                "EvalSet {eval_set_id} not found for app {app_name}."
            ))
            .into());
        }

        let mut eval_cases = self.eval_cases.lock().unwrap();
        let app_eval_cases = eval_cases.entry(app_name.to_string()).or_default();
        let set_eval_cases = app_eval_cases.entry(eval_set_id.to_string()).or_default();
        if set_eval_cases.remove(eval_case_id).is_none() {
            return Err(NotFoundError::new(format!(
                "EvalCase {eval_case_id} not found in EvalSet {eval_set_id} for app {app_name}."
            ))
            .into());
        }

        let eval_set = app_eval_sets.get_mut(eval_set_id).unwrap();
        eval_set
            .eval_cases
            .retain(|case| case.eval_id != eval_case_id);
        Ok(())
    }
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

    #[test]
    fn get_eval_set_returns_none_for_an_unknown_set() {
        let manager = InMemoryEvalSetsManager::new();
        assert!(manager.get_eval_set("app", "missing").is_none());
    }

    #[test]
    fn create_eval_set_then_get_eval_set_round_trips() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        let found = manager.get_eval_set("app", "set-1").unwrap();
        assert_eq!(found.eval_set_id, "set-1");
    }

    #[test]
    fn create_eval_set_rejects_a_duplicate() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        assert!(manager.create_eval_set("app", "set-1").is_err());
    }

    #[test]
    fn list_eval_sets_returns_empty_for_an_unknown_app() {
        let manager = InMemoryEvalSetsManager::new();
        assert_eq!(
            manager.list_eval_sets("missing").unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn add_eval_case_requires_an_existing_eval_set() {
        let manager = InMemoryEvalSetsManager::new();
        assert!(manager
            .add_eval_case("app", "missing", eval_case("c1"))
            .is_err());
    }

    #[test]
    fn add_get_update_delete_eval_case_round_trips() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        manager
            .add_eval_case("app", "set-1", eval_case("c1"))
            .unwrap();
        assert!(manager.get_eval_case("app", "set-1", "c1").is_some());

        let mut updated = eval_case("c1");
        updated.creation_timestamp = 99.0;
        manager.update_eval_case("app", "set-1", updated).unwrap();
        assert_eq!(
            manager
                .get_eval_case("app", "set-1", "c1")
                .unwrap()
                .creation_timestamp,
            99.0
        );

        manager.delete_eval_case("app", "set-1", "c1").unwrap();
        assert!(manager.get_eval_case("app", "set-1", "c1").is_none());

        let eval_set = manager.get_eval_set("app", "set-1").unwrap();
        assert!(eval_set.eval_cases.is_empty());
    }

    #[test]
    fn add_eval_case_rejects_a_duplicate_id() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        manager
            .add_eval_case("app", "set-1", eval_case("c1"))
            .unwrap();
        assert!(manager
            .add_eval_case("app", "set-1", eval_case("c1"))
            .is_err());
    }

    #[test]
    fn update_eval_case_errors_when_the_case_is_missing() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        assert!(manager
            .update_eval_case("app", "set-1", eval_case("missing"))
            .is_err());
    }

    #[test]
    fn delete_eval_case_errors_when_the_case_is_missing() {
        let manager = InMemoryEvalSetsManager::new();
        manager.create_eval_set("app", "set-1").unwrap();
        assert!(manager.delete_eval_case("app", "set-1", "missing").is_err());
    }
}
