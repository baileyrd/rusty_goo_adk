//! Capability C0441: `ExecuteTool`(`Execute`), ported from
//! `google.adk.tools.environment._execute_tool`.
//!
//! **Adaptation**: the source's `FunctionDeclaration.parameters_json_schema`
//! (a raw JSON-schema dict, not a `types.Schema`) maps to this port's
//! `FunctionDeclaration::parameters_json_schema` field directly — unlike
//! `load_memory_tool.rs`'s disclosed narrowing (which always uses
//! `parameters` because the source there branches on a feature flag this
//! port doesn't implement), the environment tools set
//! `parameters_json_schema` unconditionally in the source, so this port
//! matches that field choice exactly.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_environment::BaseEnvironment;
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

const DEFAULT_TIMEOUT_SECONDS: u64 = 30;
pub(crate) const MAX_OUTPUT_CHARS: usize = 30_000;

const DESCRIPTION: &str = "\nRun a shell command in the environment. For running programs, tests, and build\ncommands ONLY. WARNING: Do NOT use for file reading -- use the ReadFile tool\ninstead. Shell commands like 'cat, head, tail will produce inferior results.\nGood: Execute(\"python3 script.py\"), Execute(\"pytest\"), Execute(\"find ...\").\nBad: Execute(\"head ...\"), Execute(\"cat ...\").\n";

/// Truncates `text` to `limit` characters, appending a notice — matches
/// `tools/environment/_utils.py::truncate`.
pub(crate) fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let truncated: String = text.chars().take(limit).collect();
    format!(
        "{truncated}\n... (truncated, {} total chars)",
        text.chars().count()
    )
}

fn error_response(message: impl Into<String>) -> Value {
    Value::Map(vec![
        ("status".to_string(), Value::String("error".to_string())),
        ("error".to_string(), Value::String(message.into())),
    ])
}

/// C0441: runs a shell command via the injected [`BaseEnvironment`].
pub struct ExecuteTool {
    environment: Arc<dyn BaseEnvironment>,
    max_output_chars: usize,
}

impl ExecuteTool {
    pub fn new(environment: Arc<dyn BaseEnvironment>, max_output_chars: Option<usize>) -> Self {
        Self {
            environment,
            max_output_chars: max_output_chars.unwrap_or(MAX_OUTPUT_CHARS),
        }
    }
}

impl BaseTool for ExecuteTool {
    fn name(&self) -> &str {
        "Execute"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "command".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("string".to_string())),
                            (
                                "description".to_string(),
                                Value::String(
                                    "The shell command to execute. Chain dependent commands with &&."
                                        .to_string(),
                                ),
                            ),
                        ]),
                    )]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("command".to_string())]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let command = match args.get("command") {
                Some(Value::String(command)) if !command.is_empty() => command.clone(),
                _ => return Ok(error_response("`command` is required.")),
            };

            let result = match self
                .environment
                .execute(&command, Some(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS)))
                .await
            {
                Ok(result) => result,
                Err(error) => return Ok(error_response(error.to_string())),
            };

            let mut fields = vec![("status".to_string(), Value::String("ok".to_string()))];
            if !result.stdout.is_empty() {
                fields.push((
                    "stdout".to_string(),
                    Value::String(truncate(&result.stdout, self.max_output_chars)),
                ));
            }
            if !result.stderr.is_empty() {
                fields.push((
                    "stderr".to_string(),
                    Value::String(truncate(&result.stderr, self.max_output_chars)),
                ));
            }
            let mut status_is_error = false;
            if result.exit_code != 0 {
                status_is_error = true;
                fields.push(("exit_code".to_string(), Value::Int(result.exit_code as i64)));
            }
            if result.timed_out {
                status_is_error = true;
                fields.push((
                    "error".to_string(),
                    Value::String(format!(
                        "Command timed out after {DEFAULT_TIMEOUT_SECONDS}s."
                    )),
                ));
            }
            if status_is_error {
                if let Some(entry) = fields.iter_mut().find(|(key, _)| key == "status") {
                    entry.1 = Value::String("error".to_string());
                }
            }

            Ok(Value::Map(fields))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_environment::LocalEnvironment;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    async fn ready_environment() -> Arc<dyn BaseEnvironment> {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        Arc::new(env)
    }

    #[test]
    fn truncate_leaves_short_text_untouched() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_appends_a_notice_past_the_limit() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hello\n... (truncated, 11 total chars)");
    }

    #[rusty_tokio::test]
    async fn missing_command_returns_an_error() {
        let tool = ExecuteTool::new(ready_environment().await, None);
        let mut context = ctx();
        let result = tool
            .run_async(&BTreeMap::new(), &mut context)
            .await
            .unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "status").unwrap().1,
                    Value::String("error".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn executes_and_captures_stdout() {
        let tool = ExecuteTool::new(ready_environment().await, None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("echo hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "status").unwrap().1,
                    Value::String("ok".to_string())
                );
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                assert!(matches!(&stdout.1, Value::String(s) if s.trim() == "hi"));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn nonzero_exit_code_marks_status_error() {
        let tool = ExecuteTool::new(ready_environment().await, None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("exit 3".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "status").unwrap().1,
                    Value::String("error".to_string())
                );
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "exit_code").unwrap().1,
                    Value::Int(3)
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
