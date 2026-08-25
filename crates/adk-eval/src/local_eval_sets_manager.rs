//! C0613 (local part): `LocalEvalSetsManager`, ported from
//! `google.adk.evaluation.local_eval_sets_manager`. Stores eval sets as
//! `.evalset.json` files on disk, one per `(app_name, eval_set_id)`.
//!
//! **Disclosed narrowing**: the source's legacy-format conversion reads
//! required keys by direct indexing (`old_invocation["query"]`, raising
//! `KeyError` on a malformed file) and only a few keys via `.get(key,
//! default)`. This port's `value_str` helper defaults every missing key
//! to `""` uniformly — a malformed legacy file that's missing `"query"`
//! silently gets an empty query here instead of erroring the way the
//! source would.

use std::fs;
use std::path::{Path, PathBuf};

use adk_genai::content::{Content, FunctionCall, Part};
use rusty_serde::value::Value;

use crate::eval_case::{EvalCase, IntermediateData, Invocation, SessionInput};
use crate::eval_set::EvalSet;
use crate::eval_sets_manager::{EvalManagerError, EvalSetsManager};
use crate::eval_sets_manager_utils::{
    add_eval_case_to_eval_set, delete_eval_case_from_eval_set, get_eval_case_from_eval_set,
    get_eval_set_from_app_and_id, update_eval_case_in_eval_set,
};
use crate::path_validation::validate_path_segment;

const EVAL_SET_FILE_EXTENSION: &str = ".evalset.json";

fn value_str(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `local_eval_sets_manager._convert_invocation_to_pydantic_schema` —
/// converts an invocation from the legacy JSON format to [`Invocation`].
fn convert_invocation_to_pydantic_schema(old_invocation: &Value) -> Invocation {
    let query = value_str(old_invocation, "query");
    let reference = value_str(old_invocation, "reference");

    let mut tool_uses = Vec::new();
    if let Some(Value::Seq(items)) = old_invocation.get("expected_tool_use") {
        for item in items {
            let name = value_str(item, "tool_name");
            let args = match item.get("tool_input") {
                Some(Value::Map(entries)) => Some(entries.iter().cloned().collect()),
                _ => None,
            };
            tool_uses.push(FunctionCall {
                id: None,
                name: Some(name),
                args,
                partial_args: None,
                will_continue: None,
            });
        }
    }

    let mut intermediate_responses = Vec::new();
    if let Some(Value::Seq(items)) = old_invocation.get("expected_intermediate_agent_responses") {
        for item in items {
            let author = value_str(item, "author");
            let text = value_str(item, "text");
            intermediate_responses.push((author, vec![Part::text(text)]));
        }
    }

    Invocation {
        invocation_id: adk_platform::uuid::new_uuid(),
        user_content: Content::new("user", vec![Part::text(query)]),
        final_response: Some(Content::new("model", vec![Part::text(reference)])),
        intermediate_data: rusty_serde::json::to_value(&IntermediateData {
            tool_uses,
            tool_responses: Vec::new(),
            intermediate_responses,
        })
        .ok(),
        creation_timestamp: adk_platform::time::get_time(),
        rubrics: None,
        app_details: None,
    }
}

/// `local_eval_sets_manager.convert_eval_set_to_pydantic_schema`.
pub fn convert_eval_set_to_pydantic_schema(
    eval_set_id: &str,
    eval_set_in_json_format: &[Value],
) -> EvalSet {
    let eval_cases = eval_set_in_json_format
        .iter()
        .map(|old_eval_case| {
            let new_invocations: Vec<Invocation> = match old_eval_case.get("data") {
                Some(Value::Seq(items)) => items
                    .iter()
                    .map(convert_invocation_to_pydantic_schema)
                    .collect(),
                _ => Vec::new(),
            };

            let session_input = old_eval_case
                .get("initial_session")
                .filter(|value| matches!(value, Value::Map(entries) if !entries.is_empty()))
                .map(|initial_session| SessionInput {
                    app_name: value_str(initial_session, "app_name"),
                    user_id: value_str(initial_session, "user_id"),
                    session_id: None,
                    state: match initial_session.get("state") {
                        Some(Value::Map(entries)) => entries.iter().cloned().collect(),
                        _ => Default::default(),
                    },
                });

            EvalCase {
                eval_id: value_str(old_eval_case, "name"),
                conversation: Some(new_invocations),
                conversation_scenario: None,
                session_input,
                creation_timestamp: adk_platform::time::get_time(),
                rubrics: None,
                final_session_state: Default::default(),
            }
        })
        .collect();

    EvalSet {
        eval_set_id: eval_set_id.to_string(),
        name: Some(eval_set_id.to_string()),
        description: None,
        eval_cases,
        creation_timestamp: adk_platform::time::get_time(),
    }
}

/// `local_eval_sets_manager.load_eval_set_from_file` — tries the current
/// (typed) schema first, falling back to the legacy JSON-array format on
/// a parse failure.
pub fn load_eval_set_from_file(
    eval_set_file_path: &Path,
    eval_set_id: &str,
) -> Result<EvalSet, EvalManagerError> {
    let content = fs::read_to_string(eval_set_file_path)?;
    if let Ok(eval_set) = rusty_serde::json::from_str::<EvalSet>(&content) {
        return Ok(eval_set);
    }
    let legacy: Vec<Value> = rusty_serde::json::from_str(&content).map_err(|error| {
        EvalManagerError::InvalidArgument(format!(
            "Failed to parse eval set file `{}`: {error}",
            eval_set_file_path.display()
        ))
    })?;
    Ok(convert_eval_set_to_pydantic_schema(eval_set_id, &legacy))
}

fn validate_id(id_name: &str, id_value: &str) -> Result<(), EvalManagerError> {
    let valid = !id_value.is_empty()
        && id_value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(EvalManagerError::InvalidArgument(format!(
            "Invalid {id_name}. {id_name} should have the `^[a-zA-Z0-9_]+$` format",
        )));
    }
    Ok(())
}

/// C0613 (local part): `local_eval_sets_manager.LocalEvalSetsManager` —
/// an `EvalSetsManager` that stores eval sets locally on disk.
///
/// **Disclosed narrowing**: `_write_eval_set_to_path`'s
/// `model_dump_json(indent=2, exclude_unset=True, exclude_defaults=True,
/// exclude_none=True)` writes pretty-printed, sparse JSON (only the
/// fields that differ from a fresh model's defaults). `rusty_serde::json`
/// has no pretty-printer and this port's structs don't use
/// `#[rusty_serde(skip_serializing_if)]` anywhere, so this port always
/// writes a compact JSON object with every field present (including
/// `null`s and defaults). Files round-trip correctly through this same
/// port's own `Deserialize` either way — every optional field here is
/// `#[rusty_serde(default)]` — but the on-disk bytes are denser and less
/// human-diffable than the source's output.
pub struct LocalEvalSetsManager {
    agents_dir: PathBuf,
}

impl LocalEvalSetsManager {
    pub fn new(agents_dir: impl Into<PathBuf>) -> Self {
        Self {
            agents_dir: agents_dir.into(),
        }
    }

    fn eval_set_file_path(
        &self,
        app_name: &str,
        eval_set_id: &str,
    ) -> Result<PathBuf, EvalManagerError> {
        validate_path_segment(app_name, "app_name")?;
        validate_path_segment(eval_set_id, "eval_set_id")?;
        Ok(self
            .agents_dir
            .join(app_name)
            .join(format!("{eval_set_id}{EVAL_SET_FILE_EXTENSION}")))
    }

    fn write_eval_set_to_path(
        &self,
        eval_set_path: &Path,
        eval_set: &EvalSet,
    ) -> Result<(), EvalManagerError> {
        if let Some(parent) = eval_set_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = rusty_serde::json::to_string(eval_set)
            .map_err(|error| EvalManagerError::InvalidArgument(error.to_string()))?;
        fs::write(eval_set_path, json)?;
        Ok(())
    }

    fn save_eval_set(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_set: &EvalSet,
    ) -> Result<(), EvalManagerError> {
        let path = self.eval_set_file_path(app_name, eval_set_id)?;
        self.write_eval_set_to_path(&path, eval_set)
    }
}

impl EvalSetsManager for LocalEvalSetsManager {
    fn get_eval_set(&self, app_name: &str, eval_set_id: &str) -> Option<EvalSet> {
        let path = self.eval_set_file_path(app_name, eval_set_id).ok()?;
        load_eval_set_from_file(&path, eval_set_id).ok()
    }

    fn create_eval_set(
        &self,
        app_name: &str,
        eval_set_id: &str,
    ) -> Result<EvalSet, EvalManagerError> {
        validate_id("Eval Set ID", eval_set_id)?;
        let path = self.eval_set_file_path(app_name, eval_set_id)?;
        if path.exists() {
            return Err(EvalManagerError::InvalidArgument(format!(
                "EvalSet {eval_set_id} already exists for app {app_name}."
            )));
        }
        let new_eval_set = EvalSet {
            eval_set_id: eval_set_id.to_string(),
            name: Some(eval_set_id.to_string()),
            description: None,
            eval_cases: Vec::new(),
            creation_timestamp: adk_platform::time::get_time(),
        };
        self.write_eval_set_to_path(&path, &new_eval_set)?;
        Ok(new_eval_set)
    }

    fn list_eval_sets(&self, app_name: &str) -> Result<Vec<String>, EvalManagerError> {
        validate_path_segment(app_name, "app_name")?;
        let dir = self.agents_dir.join(app_name);
        let entries = fs::read_dir(&dir).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                EvalManagerError::NotFound(adk_errors::not_found::NotFoundError::new(format!(
                    "Eval directory for app `{app_name}` not found."
                )))
            } else {
                EvalManagerError::Io(error)
            }
        })?;
        let mut eval_sets: Vec<String> = entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let file_name = entry.file_name();
                let file_name = file_name.to_str()?.to_string();
                file_name
                    .strip_suffix(EVAL_SET_FILE_EXTENSION)
                    .map(|stem| stem.to_string())
            })
            .collect();
        eval_sets.sort();
        Ok(eval_sets)
    }

    fn get_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Option<EvalCase> {
        let eval_set = self.get_eval_set(app_name, eval_set_id)?;
        get_eval_case_from_eval_set(&eval_set, eval_case_id)
    }

    fn add_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case: EvalCase,
    ) -> Result<(), EvalManagerError> {
        let eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)?;
        let updated = add_eval_case_to_eval_set(eval_set, eval_case)?;
        self.save_eval_set(app_name, eval_set_id, &updated)
    }

    fn update_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        updated_eval_case: EvalCase,
    ) -> Result<(), EvalManagerError> {
        let eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)?;
        let updated = update_eval_case_in_eval_set(eval_set, updated_eval_case)?;
        self.save_eval_set(app_name, eval_set_id, &updated)
    }

    fn delete_eval_case(
        &self,
        app_name: &str,
        eval_set_id: &str,
        eval_case_id: &str,
    ) -> Result<(), EvalManagerError> {
        let eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)?;
        let updated = delete_eval_case_from_eval_set(eval_set, eval_case_id)?;
        self.save_eval_set(app_name, eval_set_id, &updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "adk_eval_local_eval_sets_manager_{name}_{}",
            adk_platform::uuid::new_uuid()
        ));
        dir
    }

    fn eval_case(id: &str) -> EvalCase {
        EvalCase {
            eval_id: id.to_string(),
            conversation: Some(vec![]),
            ..Default::default()
        }
    }

    #[test]
    fn get_eval_set_returns_none_when_the_file_is_missing() {
        let manager = LocalEvalSetsManager::new(temp_dir("get_missing"));
        assert!(manager.get_eval_set("app", "missing").is_none());
    }

    #[test]
    fn create_eval_set_then_get_eval_set_round_trips() {
        let manager = LocalEvalSetsManager::new(temp_dir("create_get"));
        manager.create_eval_set("app", "set_1").unwrap();
        let found = manager.get_eval_set("app", "set_1").unwrap();
        assert_eq!(found.eval_set_id, "set_1");
    }

    #[test]
    fn create_eval_set_rejects_an_invalid_id() {
        let manager = LocalEvalSetsManager::new(temp_dir("invalid_id"));
        assert!(manager.create_eval_set("app", "not valid!").is_err());
    }

    #[test]
    fn create_eval_set_rejects_a_duplicate() {
        let manager = LocalEvalSetsManager::new(temp_dir("duplicate"));
        manager.create_eval_set("app", "set_1").unwrap();
        assert!(manager.create_eval_set("app", "set_1").is_err());
    }

    #[test]
    fn list_eval_sets_errors_for_a_missing_app_directory() {
        let manager = LocalEvalSetsManager::new(temp_dir("list_missing"));
        assert!(manager.list_eval_sets("missing").is_err());
    }

    #[test]
    fn list_eval_sets_returns_sorted_ids() {
        let manager = LocalEvalSetsManager::new(temp_dir("list_sorted"));
        manager.create_eval_set("app", "b_set").unwrap();
        manager.create_eval_set("app", "a_set").unwrap();
        assert_eq!(
            manager.list_eval_sets("app").unwrap(),
            vec!["a_set", "b_set"]
        );
    }

    #[test]
    fn add_get_update_delete_eval_case_round_trips() {
        let manager = LocalEvalSetsManager::new(temp_dir("crud"));
        manager.create_eval_set("app", "set_1").unwrap();
        manager
            .add_eval_case("app", "set_1", eval_case("c1"))
            .unwrap();
        assert!(manager.get_eval_case("app", "set_1", "c1").is_some());

        let mut updated = eval_case("c1");
        updated.creation_timestamp = 7.0;
        manager.update_eval_case("app", "set_1", updated).unwrap();
        assert_eq!(
            manager
                .get_eval_case("app", "set_1", "c1")
                .unwrap()
                .creation_timestamp,
            7.0
        );

        manager.delete_eval_case("app", "set_1", "c1").unwrap();
        assert!(manager.get_eval_case("app", "set_1", "c1").is_none());
    }

    #[test]
    fn add_eval_case_requires_an_existing_eval_set() {
        let manager = LocalEvalSetsManager::new(temp_dir("add_missing_set"));
        assert!(manager
            .add_eval_case("app", "missing", eval_case("c1"))
            .is_err());
    }

    #[test]
    fn converts_a_legacy_format_eval_set_on_read() {
        let dir = temp_dir("legacy");
        fs::create_dir_all(dir.join("app")).unwrap();
        let legacy_json = r#"[
            {
                "name": "roll_dice_case",
                "data": [
                    {
                        "query": "What can you do?",
                        "expected_tool_use": [],
                        "expected_intermediate_agent_responses": [],
                        "reference": "I can roll dice."
                    },
                    {
                        "query": "Roll a 6 sided die",
                        "expected_tool_use": [
                            {"tool_name": "roll_die", "tool_input": {"sides": 6}}
                        ],
                        "expected_intermediate_agent_responses": [],
                        "reference": "You rolled a 4."
                    }
                ],
                "initial_session": {
                    "state": {},
                    "app_name": "hello_world",
                    "user_id": "user"
                }
            }
        ]"#;
        fs::write(dir.join("app").join("legacy_set.evalset.json"), legacy_json).unwrap();

        let manager = LocalEvalSetsManager::new(dir);
        let eval_set = manager.get_eval_set("app", "legacy_set").unwrap();
        assert_eq!(eval_set.eval_cases.len(), 1);
        let case = &eval_set.eval_cases[0];
        assert_eq!(case.eval_id, "roll_dice_case");
        let conversation = case.conversation.as_ref().unwrap();
        assert_eq!(conversation.len(), 2);
        assert_eq!(
            conversation[1].user_content.parts[0].text.as_deref(),
            Some("Roll a 6 sided die")
        );
        let session_input = case.session_input.as_ref().unwrap();
        assert_eq!(session_input.app_name, "hello_world");
    }
}
