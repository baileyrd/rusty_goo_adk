//! Capability C0100: `TaskRequest`/`TaskResult`/`_DefaultTaskInput`/
//! `_DefaultTaskOutput`, ported from `google.adk.agents.llm.task._task_models`.
//!
//! Fully self-contained (no forward references) — data shapes used by
//! `FinishTaskTool`'s task-delegation payloads.
//!
//! **Not ported**: the source's `_as_task_request(value)` helper, which
//! normalizes either a live `TaskRequest` instance or a plain dict (after
//! session deserialization) to a `TaskRequest`. Rust has no equivalent
//! ambiguity — a caller always has a typed `TaskRequest` or deserializes
//! JSON straight into one via `rusty_serde::json::from_str` — so there's
//! nothing for a normalizing helper to do.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A request to delegate a task to a sub-agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TaskRequest {
    pub agent_name: String,
    pub input: BTreeMap<String, Value>,
}

/// The result returned by a task agent upon completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TaskResult {
    pub output: Value,
}

/// Default input schema when no custom `input_schema` is provided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DefaultTaskInput {
    #[rusty_serde(default)]
    pub goal: Option<String>,
    #[rusty_serde(default)]
    pub background: Option<String>,
}

/// Default output schema when no custom `output_schema` is provided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DefaultTaskOutput {
    pub result: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_request_round_trips_with_camel_case_field_names() {
        let mut input = BTreeMap::new();
        input.insert("goal".to_string(), Value::String("ship it".to_string()));
        let req = TaskRequest {
            agent_name: "worker".to_string(),
            input,
        };
        let json = rusty_serde::json::to_string(&req).unwrap();
        assert!(json.contains("\"agentName\""));
        let back: TaskRequest = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn task_result_rejects_unknown_fields() {
        let json = r#"{"output":"done","extra":true}"#;
        assert!(rusty_serde::json::from_str::<TaskResult>(json).is_err());
    }

    #[test]
    fn default_task_input_fields_are_optional() {
        let input: DefaultTaskInput = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(input.goal, None);
        assert_eq!(input.background, None);
    }

    #[test]
    fn default_task_output_requires_result() {
        let output = DefaultTaskOutput {
            result: "summary".to_string(),
        };
        let json = rusty_serde::json::to_string(&output).unwrap();
        assert!(json.contains("\"result\""));
    }
}
