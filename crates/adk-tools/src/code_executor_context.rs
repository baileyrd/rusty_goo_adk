//! Capability C0390: `CodeExecutorContext`, ported from
//! `google.adk.code_executors.code_executor_context`.
//!
//! **`_SessionState: State | dict[str, Any]` union, narrowed**: this
//! port's session state is always the concrete `BTreeMap<String, Value>`
//! that `adk_agents::session::Session::state` already is — there is no
//! separate `State`-vs-plain-`dict` distinction to preserve here.
//!
//! **The nested `_context` sub-dict vs. root `session_state`, preserved
//! as a real distinction**: the source stores `execution_session_id`/
//! `processed_input_files` inside a nested dict at
//! `session_state["_code_execution_context"]` (an in-memory snapshot
//! that only reaches the persistent session state when a caller applies
//! [`CodeExecutorContext::get_state_delta`]'s return value), while
//! `_code_executor_input_files`/`_code_executor_error_counts`/
//! `_code_execution_results` are written directly onto the root
//! `session_state` (already "live", no separate flush step). This port
//! keeps that same split: an owned `context: BTreeMap<String, Value>`
//! mirrors the nested sub-dict, and the borrowed `session_state` is
//! mutated directly for the other three keys.
//!
//! **`File`, round-tripped through base64 text**: the source's `File`
//! dataclass round-trips through `dataclasses.asdict`/`File(**dict)`
//! with its `content: str | bytes` field held as-is — a plain Python
//! dict can hold a raw `bytes` object with no serialization step. This
//! port's state is always `rusty_serde::value::Value` (JSON-shaped, no
//! raw-bytes variant), so `File.content` round-trips as base64 text
//! instead, reusing `code_execution_utils`'s own base64 codec — a real,
//! disclosed representational difference forced by this port's state
//! type, not a data-loss risk (the encoding round-trips exactly).
//!
//! **`update_code_execution_result`'s timestamp**: the source uses
//! `datetime.datetime.now().timestamp()`; this port uses
//! `adk_platform::time::get_time()`, the same runtime-abstracted clock
//! used throughout this port instead of a direct OS time call.

use std::collections::BTreeMap;

use rusty_serde::value::Value;

use crate::code_execution_utils::{base64_decode_strict, base64_encode, File};

const CONTEXT_KEY: &str = "_code_execution_context";
const SESSION_ID_KEY: &str = "execution_session_id";
const PROCESSED_FILE_NAMES_KEY: &str = "processed_input_files";
const INPUT_FILE_KEY: &str = "_code_executor_input_files";
const ERROR_COUNT_KEY: &str = "_code_executor_error_counts";
const CODE_EXECUTION_RESULTS_KEY: &str = "_code_execution_results";

fn value_as_map(value: &Value) -> BTreeMap<String, Value> {
    match value {
        Value::Map(entries) => entries.iter().cloned().collect(),
        _ => BTreeMap::new(),
    }
}

fn map_to_value(map: BTreeMap<String, Value>) -> Value {
    Value::Map(map.into_iter().collect())
}

fn file_to_value(file: &File) -> Value {
    Value::Map(vec![
        ("name".to_string(), Value::String(file.name.clone())),
        (
            "content".to_string(),
            Value::String(base64_encode(&file.content)),
        ),
        (
            "mime_type".to_string(),
            Value::String(file.mime_type.clone()),
        ),
    ])
}

fn file_from_value(value: &Value) -> Option<File> {
    let name = value.get("name").and_then(Value::as_str)?.to_string();
    let content = value
        .get("content")
        .and_then(Value::as_str)
        .and_then(|s| base64_decode_strict(s.as_bytes()))
        .unwrap_or_default();
    let mime_type = value
        .get("mime_type")
        .and_then(Value::as_str)
        .unwrap_or("text/plain")
        .to_string();
    Some(File::new(name, content, mime_type))
}

/// C0390: `CodeExecutorContext` — the persistent context used to
/// configure a code executor, backed by a session's state.
pub struct CodeExecutorContext<'a> {
    context: BTreeMap<String, Value>,
    session_state: &'a mut BTreeMap<String, Value>,
}

impl<'a> CodeExecutorContext<'a> {
    /// `CodeExecutorContext.__init__` — reads (or lazily creates) the
    /// nested context sub-dict from `session_state`.
    pub fn new(session_state: &'a mut BTreeMap<String, Value>) -> Self {
        let context = match session_state.get(CONTEXT_KEY) {
            Some(value) => value_as_map(value),
            None => BTreeMap::new(),
        };
        session_state
            .entry(CONTEXT_KEY.to_string())
            .or_insert_with(|| Value::Map(Vec::new()));
        Self {
            context,
            session_state,
        }
    }

    /// `get_state_delta` — the state delta to apply to the persistent
    /// session state for the in-memory `context` sub-dict changes made
    /// via [`Self::set_execution_id`]/[`Self::add_processed_file_names`].
    pub fn get_state_delta(&self) -> BTreeMap<String, Value> {
        let mut delta = BTreeMap::new();
        delta.insert(CONTEXT_KEY.to_string(), map_to_value(self.context.clone()));
        delta
    }

    pub fn get_execution_id(&self) -> Option<String> {
        self.context
            .get(SESSION_ID_KEY)
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    pub fn set_execution_id(&mut self, session_id: impl Into<String>) {
        self.context
            .insert(SESSION_ID_KEY.to_string(), Value::String(session_id.into()));
    }

    pub fn get_processed_file_names(&self) -> Vec<String> {
        match self.context.get(PROCESSED_FILE_NAMES_KEY) {
            Some(Value::Seq(items)) => items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn add_processed_file_names(&mut self, file_names: &[String]) {
        let mut names = self.get_processed_file_names();
        names.extend(file_names.iter().cloned());
        self.context.insert(
            PROCESSED_FILE_NAMES_KEY.to_string(),
            Value::Seq(names.into_iter().map(Value::String).collect()),
        );
    }

    pub fn get_input_files(&self) -> Vec<File> {
        match self.session_state.get(INPUT_FILE_KEY) {
            Some(Value::Seq(items)) => items.iter().filter_map(file_from_value).collect(),
            _ => Vec::new(),
        }
    }

    pub fn add_input_files(&mut self, input_files: &[File]) {
        let mut stored: Vec<Value> = match self.session_state.get(INPUT_FILE_KEY) {
            Some(Value::Seq(items)) => items.clone(),
            _ => Vec::new(),
        };
        stored.extend(input_files.iter().map(file_to_value));
        self.session_state
            .insert(INPUT_FILE_KEY.to_string(), Value::Seq(stored));
    }

    /// `clear_input_files` — removes the input files and processed file
    /// names from the code executor context.
    pub fn clear_input_files(&mut self) {
        if self.session_state.contains_key(INPUT_FILE_KEY) {
            self.session_state
                .insert(INPUT_FILE_KEY.to_string(), Value::Seq(Vec::new()));
        }
        if self.context.contains_key(PROCESSED_FILE_NAMES_KEY) {
            self.context
                .insert(PROCESSED_FILE_NAMES_KEY.to_string(), Value::Seq(Vec::new()));
        }
    }

    pub fn get_error_count(&self, invocation_id: &str) -> u32 {
        match self.session_state.get(ERROR_COUNT_KEY) {
            Some(value) => value
                .get(invocation_id)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0),
            None => 0,
        }
    }

    pub fn increment_error_count(&mut self, invocation_id: &str) {
        let mut counts = match self.session_state.get(ERROR_COUNT_KEY) {
            Some(value) => value_as_map(value),
            None => BTreeMap::new(),
        };
        let next = self.get_error_count(invocation_id) + 1;
        counts.insert(invocation_id.to_string(), Value::UInt(next as u64));
        self.session_state
            .insert(ERROR_COUNT_KEY.to_string(), map_to_value(counts));
    }

    pub fn reset_error_count(&mut self, invocation_id: &str) {
        let Some(value) = self.session_state.get(ERROR_COUNT_KEY) else {
            return;
        };
        let mut counts = value_as_map(value);
        counts.remove(invocation_id);
        self.session_state
            .insert(ERROR_COUNT_KEY.to_string(), map_to_value(counts));
    }

    /// `update_code_execution_result` — appends a code execution result
    /// entry for `invocation_id`.
    pub fn update_code_execution_result(
        &mut self,
        invocation_id: &str,
        code: &str,
        result_stdout: &str,
        result_stderr: &str,
    ) {
        let mut stored = match self.session_state.get(CODE_EXECUTION_RESULTS_KEY) {
            Some(value) => value_as_map(value),
            None => BTreeMap::new(),
        };
        let mut invocation_results = match stored.get(invocation_id) {
            Some(Value::Seq(items)) => items.clone(),
            _ => Vec::new(),
        };
        invocation_results.push(Value::Map(vec![
            ("code".to_string(), Value::String(code.to_string())),
            (
                "result_stdout".to_string(),
                Value::String(result_stdout.to_string()),
            ),
            (
                "result_stderr".to_string(),
                Value::String(result_stderr.to_string()),
            ),
            (
                "timestamp".to_string(),
                Value::Int(adk_platform::time::get_time() as i64),
            ),
        ]));
        stored.insert(invocation_id.to_string(), Value::Seq(invocation_results));
        self.session_state
            .insert(CODE_EXECUTION_RESULTS_KEY.to_string(), map_to_value(stored));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_id_round_trips_through_get_state_delta() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        assert_eq!(ctx.get_execution_id(), None);
        ctx.set_execution_id("exec-1");
        assert_eq!(ctx.get_execution_id().as_deref(), Some("exec-1"));

        let delta = ctx.get_state_delta();
        let stored_context = delta.get(CONTEXT_KEY).unwrap();
        assert_eq!(
            stored_context.get(SESSION_ID_KEY).and_then(Value::as_str),
            Some("exec-1")
        );
    }

    #[test]
    fn processed_file_names_accumulate() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        assert_eq!(ctx.get_processed_file_names(), Vec::<String>::new());
        ctx.add_processed_file_names(&["a.csv".to_string()]);
        ctx.add_processed_file_names(&["b.csv".to_string()]);
        assert_eq!(
            ctx.get_processed_file_names(),
            vec!["a.csv".to_string(), "b.csv".to_string()]
        );
    }

    #[test]
    fn input_files_round_trip_through_base64() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        let file = File::new("data.bin", vec![0, 1, 2, 255], "application/octet-stream");
        ctx.add_input_files(std::slice::from_ref(&file));
        assert_eq!(ctx.get_input_files(), vec![file]);
    }

    #[test]
    fn clear_input_files_resets_both_input_files_and_processed_names() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        ctx.add_input_files(&[File::with_default_mime_type("a.txt", vec![1])]);
        ctx.add_processed_file_names(&["a.txt".to_string()]);
        ctx.clear_input_files();
        assert!(ctx.get_input_files().is_empty());
        assert!(ctx.get_processed_file_names().is_empty());
    }

    #[test]
    fn error_count_increments_and_resets_per_invocation() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        assert_eq!(ctx.get_error_count("inv-1"), 0);
        ctx.increment_error_count("inv-1");
        ctx.increment_error_count("inv-1");
        assert_eq!(ctx.get_error_count("inv-1"), 2);
        assert_eq!(ctx.get_error_count("inv-2"), 0);
        ctx.reset_error_count("inv-1");
        assert_eq!(ctx.get_error_count("inv-1"), 0);
    }

    #[test]
    fn update_code_execution_result_appends_per_invocation() {
        let mut state = BTreeMap::new();
        let mut ctx = CodeExecutorContext::new(&mut state);
        ctx.update_code_execution_result("inv-1", "print(1)", "1", "");
        ctx.update_code_execution_result("inv-1", "print(2)", "2", "");

        let stored = state.get(CODE_EXECUTION_RESULTS_KEY).unwrap();
        let results = stored.get("inv-1").and_then(Value::as_seq).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[1].get("code").and_then(Value::as_str),
            Some("print(2)")
        );
    }

    #[test]
    fn constructing_a_context_seeds_the_session_state_with_an_empty_context_map() {
        let mut state = BTreeMap::new();
        let _ctx = CodeExecutorContext::new(&mut state);
        assert!(matches!(state.get(CONTEXT_KEY), Some(Value::Map(_))));
    }
}
