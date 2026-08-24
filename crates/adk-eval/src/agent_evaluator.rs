//! C0619 (partial) / C0620: `evaluation.agent_evaluator`, ported from
//! `google.adk.evaluation.agent_evaluator`.
//!
//! **Partial**: only `find_config_for_test_file`, `load_eval_set_from_file`,
//! `get_eval_set_from_old_format`, `get_initial_session`, `load_dataset`,
//! and `validate_input` are ported (all pure file I/O + parsing, no
//! `Runner`/agent invocation). `AgentEvaluator.evaluate`/`evaluate_eval_set`
//! (the actual test-running entry points) stay unbuilt — they drive a
//! real `Runner` against a loaded agent module, orchestrate `EvaluateConfig`/
//! `InferenceConfig`, and print/persist results, none of which this
//! batch's pure-data-transform scope covers. `migrate_eval_data_to_new_schema`
//! (C0620) is fully ported — it's a thin wrapper over the above.
//!
//! **Naming note**: this module's [`load_eval_set_from_file`] ports
//! `AgentEvaluator._load_eval_set_from_file` — a distinct, differently-
//! shaped source function from `local_eval_sets_manager.load_eval_set_from_file`
//! (already ported as `local_eval_sets_manager::load_eval_set_from_file`).
//! The two happen to share a name in the source too; not a duplicate.
//!
//! **`_load_dataset`'s type hint vs. implementation, disclosed**: the
//! source types its `input_data` parameter
//! `str | List[str] | List[Dict[str, Any]] | List[List[Dict[str, Any]]]`,
//! but the actual `isinstance` dispatch only ever accepts a `str` (file
//! or directory path) or a `list[str]` where every entry is an existing
//! file path — a list of already-loaded dicts is never accepted at
//! runtime (`isinstance(i, str)` fails for a `dict`, so the function
//! raises `TypeError`). This port's [`DatasetInput`] enum models the
//! implementation's actual accepted shapes (`Path`/`Paths`), not the
//! type hint's aspirational, unreachable ones.
//!
//! **`_validate_input`'s dead check, not ported**: the source's
//! `if not isinstance(sample, list) and not isinstance(first_query, dict)`
//! is unreachable given how this function is only ever called — `sample`
//! is always a `list[dict]` by construction — so this port's static types
//! (`sample: &[Value]`) already guarantee it structurally; no runtime
//! check is needed to replicate a branch that can't fire.
//!
//! **`os.walk`, adapted**: [`load_dataset`]'s directory-recursion walk is
//! a small hand-rolled `std::fs::read_dir` recursion rather than a new
//! `walkdir`-style crate dependency — this workspace has no existing
//! directory-walk utility and the traversal itself is trivial (find every
//! `*.test.json` file under a root, no symlink/depth/filter options
//! needed). Enumeration order is OS-directory-listing-dependent either
//! way, matching `os.walk`'s own unspecified order.

use std::collections::HashMap;
use std::path::Path;

use rusty_serde::value::Value;

use crate::eval_case::SessionState;
use crate::eval_config::{get_evaluation_criteria_or_default, EvalConfig};
use crate::eval_metrics::PrebuiltMetrics;
use crate::eval_set::EvalSet;
use crate::local_eval_sets_manager::convert_eval_set_to_pydantic_schema;

fn tool_trajectory_score_key() -> &'static str {
    PrebuiltMetrics::ToolTrajectoryAvgScore.as_str()
}
fn response_evaluation_score_key() -> &'static str {
    PrebuiltMetrics::ResponseEvaluationScore.as_str()
}
fn response_match_score_key() -> &'static str {
    PrebuiltMetrics::ResponseMatchScore.as_str()
}
fn safety_v1_key() -> &'static str {
    PrebuiltMetrics::SafetyV1.as_str()
}

const QUERY_COLUMN: &str = "query";
const REFERENCE_COLUMN: &str = "reference";
const EXPECTED_TOOL_USE_COLUMN: &str = "expected_tool_use";

/// `agent_evaluator.load_json` — parses a whole file as JSON.
pub fn load_json(file_path: &str) -> Result<Value, String> {
    let content = std::fs::read_to_string(file_path).map_err(|error| error.to_string())?;
    adk_genai::json_utils::safe_json_loads(&content, Some(file_path))
}

/// C0619 (partial): `AgentEvaluator.find_config_for_test_file` — finds
/// the `test_config.json` file in the same folder as `test_file`.
pub fn find_config_for_test_file(test_file: &str) -> Result<EvalConfig, String> {
    let test_folder = Path::new(test_file)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let config_path = test_folder.join("test_config.json");
    get_evaluation_criteria_or_default(config_path.to_str())
}

/// `AgentEvaluator._get_initial_session` — the initial session state
/// declared in `initial_session_file`, or empty if none is given.
pub fn get_initial_session(initial_session_file: Option<&str>) -> Result<SessionState, String> {
    let Some(path) = initial_session_file.filter(|path| !path.is_empty()) else {
        return Ok(SessionState::new());
    };
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: Value = adk_genai::json_utils::safe_json_loads(
        &content,
        Some(&format!("initial session file {path}")),
    )?;
    match value {
        Value::Map(entries) => Ok(entries.into_iter().collect()),
        // Compile-time strengthening, disclosed: the source's return type
        // (`dict[str, Any]`) is unenforced at runtime -- a non-object JSON
        // file would silently mistype downstream. This port rejects it
        // explicitly instead.
        _ => Err(format!(
            "initial session file {path} must contain a JSON object."
        )),
    }
}

fn load_json_file(file_path: &str) -> Result<Vec<Value>, String> {
    match load_json(file_path)? {
        Value::Seq(items) if items.iter().all(|item| matches!(item, Value::Map(_))) => Ok(items),
        _ => Err(format!("{file_path} must contain a list of dictionaries.")),
    }
}

fn find_test_json_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            find_test_json_files(&path, out)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".test.json"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// `_load_dataset`'s accepted input shapes. See this module's doc for why
/// this narrows the source's type hint to what the implementation
/// actually accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetInput {
    /// A single file or directory path.
    Path(String),
    /// A list of file paths, every one of which must exist.
    Paths(Vec<String>),
}

/// C0619 (partial): `AgentEvaluator._load_dataset` — loads one or more
/// `*.test.json`-shaped datasets (each a list of query-record objects).
pub fn load_dataset(input: &DatasetInput) -> Result<Vec<Vec<Value>>, String> {
    match input {
        DatasetInput::Path(path) => {
            let path_ref = Path::new(path);
            if path_ref.is_dir() {
                let mut test_files = Vec::new();
                find_test_json_files(path_ref, &mut test_files)?;
                test_files
                    .iter()
                    .map(|file| load_json_file(&file.to_string_lossy()))
                    .collect()
            } else if path_ref.is_file() {
                Ok(vec![load_json_file(path)?])
            } else {
                Err(format!("Input path {path} is invalid."))
            }
        }
        DatasetInput::Paths(paths) => {
            if paths.iter().all(|path| Path::new(path).is_file()) {
                paths.iter().map(|path| load_json_file(path)).collect()
            } else {
                Err("Input list must contain valid file paths.".to_string())
            }
        }
    }
}

/// C0619 (partial): `AgentEvaluator._validate_input` — validates that
/// `criteria` aligns with `eval_dataset` (using only the first sample,
/// for efficiency).
pub fn validate_input(
    eval_dataset: &[Vec<Value>],
    criteria: &HashMap<String, Value>,
) -> Result<(), String> {
    if eval_dataset.is_empty() {
        return Err("The evaluation dataset is None or empty.".to_string());
    }

    let allowed_criteria = [
        tool_trajectory_score_key(),
        response_evaluation_score_key(),
        response_match_score_key(),
        safety_v1_key(),
    ];
    for key in criteria.keys() {
        if !allowed_criteria.contains(&key.as_str()) {
            return Err(format!(
                "Invalid criteria key: {key}. Expected one of {allowed_criteria:?}."
            ));
        }
    }

    let sample = &eval_dataset[0];
    let first_query = sample.first().ok_or_else(|| {
        format!(
            "Each evaluation dataset sample must be list of dictionary. \
             But it's {eval_dataset:?}"
        )
    })?;
    let has_key = |key: &str| matches!(first_query, Value::Map(entries) if entries.iter().any(|(k, _)| k == key));

    if criteria.contains_key(tool_trajectory_score_key())
        && (!has_key(QUERY_COLUMN) || !has_key(EXPECTED_TOOL_USE_COLUMN))
    {
        return Err(format!(
            "Samples for {} must include '{QUERY_COLUMN}' and '{EXPECTED_TOOL_USE_COLUMN}' \
             keys. The sample is {sample:?}.",
            tool_trajectory_score_key()
        ));
    }
    if criteria.contains_key(response_evaluation_score_key()) && !has_key(QUERY_COLUMN) {
        return Err(format!(
            "Samples for {} must include '{QUERY_COLUMN}' key. The sample is {sample:?}.",
            response_evaluation_score_key()
        ));
    }
    if criteria.contains_key(response_match_score_key())
        && (!has_key(QUERY_COLUMN) || !has_key(REFERENCE_COLUMN))
    {
        return Err(format!(
            "Samples for {} must include '{QUERY_COLUMN}' and '{REFERENCE_COLUMN}' keys. The \
             sample is {sample:?}.",
            response_match_score_key()
        ));
    }

    Ok(())
}

/// C0619 (partial): `AgentEvaluator._get_eval_set_from_old_format`.
pub fn get_eval_set_from_old_format(
    eval_set_file: &str,
    eval_config: &EvalConfig,
    initial_session: &SessionState,
) -> Result<EvalSet, String> {
    let data = load_dataset(&DatasetInput::Path(eval_set_file.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| format!("{eval_set_file} produced no datasets."))?;
    validate_input(std::slice::from_ref(&data), &eval_config.criteria)?;

    let initial_session_value = Value::Map(
        initial_session
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    let eval_data = Value::Map(vec![
        ("name".to_string(), Value::String(eval_set_file.to_string())),
        ("data".to_string(), Value::Seq(data)),
        ("initial_session".to_string(), initial_session_value),
    ]);

    Ok(convert_eval_set_to_pydantic_schema(
        &adk_platform::uuid::new_uuid(),
        &[eval_data],
    ))
}

/// C0619 (partial): `AgentEvaluator._load_eval_set_from_file` — see this
/// module's doc for why this isn't the same function as
/// `local_eval_sets_manager::load_eval_set_from_file`.
pub fn load_eval_set_from_file(
    eval_set_file: &str,
    eval_config: &EvalConfig,
    initial_session: &SessionState,
) -> Result<EvalSet, String> {
    if Path::new(eval_set_file).is_file() {
        let content = std::fs::read_to_string(eval_set_file).map_err(|error| error.to_string())?;
        if let Ok(eval_set) = rusty_serde::json::from_str::<EvalSet>(&content) {
            if !initial_session.is_empty() {
                return Err(
                    "Initial session should be specified as a part of EvalSet file. Explicit \
                     initial session is only needed, when specifying data in the older schema."
                        .to_string(),
                );
            }
            return Ok(eval_set);
        }
        // Parse failed: assume the old format and fall through. Disclosed:
        // the source logs a warning here; no logging framework is
        // adopted by this workspace yet (same disclosed omission as
        // elsewhere in this crate).
    }
    get_eval_set_from_old_format(eval_set_file, eval_config, initial_session)
}

/// C0620: `AgentEvaluator.migrate_eval_data_to_new_schema` — a utility
/// for migrating eval data to the new schema backed by `EvalSet`.
pub fn migrate_eval_data_to_new_schema(
    old_eval_data_file: &str,
    new_eval_data_file: &str,
    initial_session_file: Option<&str>,
) -> Result<(), String> {
    if old_eval_data_file.is_empty() || new_eval_data_file.is_empty() {
        return Err("One of old_eval_data_file or new_eval_data_file is empty.".to_string());
    }

    let eval_config = find_config_for_test_file(old_eval_data_file)?;
    let initial_session = get_initial_session(initial_session_file)?;
    let eval_set =
        get_eval_set_from_old_format(old_eval_data_file, &eval_config, &initial_session)?;

    // Disclosed narrowing: compact JSON, not `indent=2`-pretty -- same
    // gap already established for `local_eval_sets_manager`'s
    // `_write_eval_set_to_path` (C0613).
    let json = rusty_serde::json::to_string(&eval_set).map_err(|error| error.to_string())?;
    std::fs::write(new_eval_data_file, json).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "adk_agent_evaluator_test_{name}_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn find_config_for_test_file_falls_back_to_default_without_a_config_file() {
        let config = find_config_for_test_file("/no/such/dir/my.test.json").unwrap();
        assert_eq!(
            config.criteria.get("tool_trajectory_avg_score"),
            Some(&Value::Float(1.0))
        );
    }

    #[test]
    fn get_initial_session_is_empty_without_a_file() {
        assert_eq!(get_initial_session(None).unwrap(), SessionState::new());
        assert_eq!(get_initial_session(Some("")).unwrap(), SessionState::new());
    }

    #[test]
    fn get_initial_session_reads_a_real_file() {
        let dir = scratch_dir("initial_session");
        let path = dir.join("session.json");
        std::fs::write(&path, r#"{"app_name":"my_app"}"#).unwrap();
        let session = get_initial_session(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(
            session.get("app_name"),
            Some(&Value::String("my_app".to_string()))
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_initial_session_rejects_non_object_json() {
        let dir = scratch_dir("initial_session_bad");
        let path = dir.join("session.json");
        std::fs::write(&path, "[1, 2, 3]").unwrap();
        assert!(get_initial_session(Some(path.to_str().unwrap())).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_dataset_rejects_an_invalid_path() {
        assert!(load_dataset(&DatasetInput::Path("/no/such/path".to_string())).is_err());
    }

    #[test]
    fn load_dataset_loads_a_single_file() {
        let dir = scratch_dir("load_dataset_file");
        let path = dir.join("case.test.json");
        std::fs::write(&path, r#"[{"query":"hi"}]"#).unwrap();
        let dataset =
            load_dataset(&DatasetInput::Path(path.to_str().unwrap().to_string())).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_dataset_rejects_content_that_is_not_a_list_of_dicts() {
        let dir = scratch_dir("load_dataset_bad");
        let path = dir.join("case.test.json");
        std::fs::write(&path, r#"["not", "a", "dict"]"#).unwrap();
        assert!(load_dataset(&DatasetInput::Path(path.to_str().unwrap().to_string())).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_dataset_walks_a_directory_for_test_json_files() {
        let dir = scratch_dir("load_dataset_dir");
        std::fs::write(dir.join("a.test.json"), r#"[{"query":"a"}]"#).unwrap();
        let nested = dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("b.test.json"), r#"[{"query":"b"}]"#).unwrap();
        std::fs::write(dir.join("ignore.txt"), "not json").unwrap();

        let dataset = load_dataset(&DatasetInput::Path(dir.to_str().unwrap().to_string())).unwrap();
        assert_eq!(dataset.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_dataset_rejects_a_list_with_a_missing_file() {
        let result = load_dataset(&DatasetInput::Paths(vec!["/no/such/file.json".to_string()]));
        assert!(result.is_err());
    }

    fn dict(pairs: &[(&str, &str)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
                .collect(),
        )
    }

    #[test]
    fn validate_input_rejects_an_empty_dataset() {
        assert!(validate_input(&[], &HashMap::new()).is_err());
    }

    #[test]
    fn validate_input_rejects_an_unknown_criteria_key() {
        let dataset = vec![vec![dict(&[("query", "hi")])]];
        let mut criteria = HashMap::new();
        criteria.insert("not_a_real_metric".to_string(), Value::Float(1.0));
        assert!(validate_input(&dataset, &criteria).is_err());
    }

    #[test]
    fn validate_input_requires_query_and_expected_tool_use_for_trajectory_score() {
        let dataset = vec![vec![dict(&[("query", "hi")])]];
        let mut criteria = HashMap::new();
        criteria.insert(tool_trajectory_score_key().to_string(), Value::Float(1.0));
        assert!(validate_input(&dataset, &criteria).is_err());

        let dataset_ok = vec![vec![dict(&[("query", "hi"), ("expected_tool_use", "[]")])]];
        assert!(validate_input(&dataset_ok, &criteria).is_ok());
    }

    #[test]
    fn validate_input_requires_query_and_reference_for_response_match_score() {
        let mut criteria = HashMap::new();
        criteria.insert(response_match_score_key().to_string(), Value::Float(0.8));
        let dataset = vec![vec![dict(&[("query", "hi")])]];
        assert!(validate_input(&dataset, &criteria).is_err());
        let dataset_ok = vec![vec![dict(&[("query", "hi"), ("reference", "hello")])]];
        assert!(validate_input(&dataset_ok, &criteria).is_ok());
    }

    #[test]
    fn get_eval_set_from_old_format_builds_an_eval_set() {
        let dir = scratch_dir("old_format");
        let path = dir.join("case.test.json");
        std::fs::write(&path, r#"[{"query":"hi","expected_tool_use":[]}]"#).unwrap();
        let mut criteria = HashMap::new();
        criteria.insert(tool_trajectory_score_key().to_string(), Value::Float(1.0));
        let eval_config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };
        let eval_set = get_eval_set_from_old_format(
            path.to_str().unwrap(),
            &eval_config,
            &SessionState::new(),
        )
        .unwrap();
        assert_eq!(eval_set.eval_cases.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn migrate_eval_data_to_new_schema_rejects_empty_paths() {
        assert!(migrate_eval_data_to_new_schema("", "out.json", None).is_err());
        assert!(migrate_eval_data_to_new_schema("in.json", "", None).is_err());
    }

    #[test]
    fn migrate_eval_data_to_new_schema_writes_a_new_style_file() {
        let dir = scratch_dir("migrate");
        let old_path = dir.join("case.test.json");
        // `find_config_for_test_file` falls back to the default criteria
        // (tool_trajectory_avg_score + response_match_score) since no
        // `test_config.json` sits alongside this file, so the sample
        // needs both `expected_tool_use` and `reference` to validate.
        std::fs::write(
            &old_path,
            r#"[{"query":"hi","expected_tool_use":[],"reference":"hello"}]"#,
        )
        .unwrap();
        let new_path = dir.join("case.evalset.json");

        migrate_eval_data_to_new_schema(
            old_path.to_str().unwrap(),
            new_path.to_str().unwrap(),
            None,
        )
        .unwrap();

        let written = std::fs::read_to_string(&new_path).unwrap();
        let eval_set: EvalSet = rusty_serde::json::from_str(&written).unwrap();
        assert_eq!(eval_set.eval_cases.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_eval_set_from_file_falls_back_to_old_format_on_parse_failure() {
        let dir = scratch_dir("load_eval_set_old");
        let path = dir.join("case.test.json");
        std::fs::write(&path, r#"[{"query":"hi","expected_tool_use":[]}]"#).unwrap();
        let mut criteria = HashMap::new();
        criteria.insert(tool_trajectory_score_key().to_string(), Value::Float(1.0));
        let eval_config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };
        let eval_set =
            load_eval_set_from_file(path.to_str().unwrap(), &eval_config, &SessionState::new())
                .unwrap();
        assert_eq!(eval_set.eval_cases.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_eval_set_from_file_reads_a_new_style_file_directly() {
        let dir = scratch_dir("load_eval_set_new");
        let path = dir.join("case.evalset.json");
        let eval_set = EvalSet {
            eval_set_id: "set-1".to_string(),
            name: None,
            description: None,
            eval_cases: vec![],
            creation_timestamp: 0.0,
        };
        std::fs::write(&path, rusty_serde::json::to_string(&eval_set).unwrap()).unwrap();

        let loaded = load_eval_set_from_file(
            path.to_str().unwrap(),
            &EvalConfig::default(),
            &SessionState::new(),
        )
        .unwrap();
        assert_eq!(loaded.eval_set_id, "set-1");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_eval_set_from_file_rejects_an_explicit_initial_session_on_new_format() {
        let dir = scratch_dir("load_eval_set_conflict");
        let path = dir.join("case.evalset.json");
        let eval_set = EvalSet {
            eval_set_id: "set-1".to_string(),
            name: None,
            description: None,
            eval_cases: vec![],
            creation_timestamp: 0.0,
        };
        std::fs::write(&path, rusty_serde::json::to_string(&eval_set).unwrap()).unwrap();

        let mut initial_session = SessionState::new();
        initial_session.insert("app_name".to_string(), Value::String("my_app".to_string()));
        let result = load_eval_set_from_file(
            path.to_str().unwrap(),
            &EvalConfig::default(),
            &initial_session,
        );
        assert!(result.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
