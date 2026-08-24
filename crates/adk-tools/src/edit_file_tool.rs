//! Capability C0443: `EditFileTool`(`EditFile`), ported from
//! `google.adk.tools.environment._edit_file_tool`.
//!
//! **Adaptation**: `re.sub(pattern, lambda m: new_string, content, count=1)`
//! uses a lambda specifically so `new_string` is inserted literally, not
//! `$`/backreference-expanded the way a plain string replacement would
//! be — this port's [`regex::Regex::replacen`] call passes a closure for
//! the same reason.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use regex::Regex;
use rusty_serde::value::Value;

use crate::base_environment::{BaseEnvironment, EnvironmentError};
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

const DESCRIPTION: &str = "Replace an exact substring in an existing file with new text. The old_string must appear exactly once in the file. To create new files, use the WriteFile tool.";

fn error_response(message: impl Into<String>) -> Value {
    Value::Map(vec![
        ("status".to_string(), Value::String("error".to_string())),
        ("error".to_string(), Value::String(message.into())),
    ])
}

/// Builds the CRLF-tolerant exact-match pattern — `re.escape(old).replace('\n', r'\r?\n')`.
fn crlf_tolerant_pattern(old_string: &str) -> String {
    let normalized_old = old_string.replace("\r\n", "\n");
    let escaped = regex::escape(&normalized_old);
    escaped.replace('\n', "\\r?\\n")
}

/// C0443: performs a surgical text replacement in an existing file via
/// the injected [`BaseEnvironment`].
pub struct EditFileTool {
    environment: Arc<dyn BaseEnvironment>,
}

impl EditFileTool {
    pub fn new(environment: Arc<dyn BaseEnvironment>) -> Self {
        Self { environment }
    }
}

impl BaseTool for EditFileTool {
    fn name(&self) -> &str {
        "EditFile"
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
                    Value::Map(vec![
                        (
                            "path".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "Path of the file to edit within the environment."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "old_string".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "The exact text to find and replace. Must not be empty."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "new_string".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String("The replacement text.".to_string()),
                                ),
                            ]),
                        ),
                    ]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![
                        Value::String("path".to_string()),
                        Value::String("old_string".to_string()),
                        Value::String("new_string".to_string()),
                    ]),
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
            let path = match args.get("path") {
                Some(Value::String(path)) if !path.is_empty() => path.clone(),
                _ => return Ok(error_response("`path` is required.")),
            };
            let old_string = match args.get("old_string") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                _ => return Ok(error_response(
                    "`old_string` cannot be empty. To create a new file, use the WriteFile tool.",
                )),
            };
            let new_string = match args.get("new_string") {
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };

            let data = match self.environment.read_file(&path).await {
                Ok(data) => data,
                Err(EnvironmentError::FileNotFound(_)) => {
                    return Ok(error_response(format!("File not found: {path}")))
                }
                Err(error) => return Ok(error_response(error.to_string())),
            };
            let content = String::from_utf8_lossy(&data).into_owned();

            let pattern = crlf_tolerant_pattern(&old_string);
            let regex = match Regex::new(&pattern) {
                Ok(regex) => regex,
                Err(error) => return Ok(error_response(error.to_string())),
            };

            let count = regex.find_iter(&content).count();
            if count == 0 {
                return Ok(error_response(
                    "`old_string` not found in file. Read the file first to verify contents.",
                ));
            }
            if count > 1 {
                return Ok(error_response(format!(
                    "`old_string` appears {count} times. Provide more surrounding context to make it unique."
                )));
            }

            let new_content = regex.replacen(&content, 1, |_: &regex::Captures| new_string.clone());
            if let Err(error) = self
                .environment
                .write_file(&path, new_content.as_bytes())
                .await
            {
                return Ok(error_response(error.to_string()));
            }

            Ok(Value::Map(vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "message".to_string(),
                    Value::String(format!("Edited {path}")),
                ),
            ]))
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

    async fn environment_with_file(path: &str, content: &str) -> Arc<dyn BaseEnvironment> {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        env.write_file(path, content.as_bytes()).await.unwrap();
        Arc::new(env)
    }

    fn args(path: &str, old: &str, new: &str) -> BTreeMap<String, Value> {
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String(path.to_string()));
        args.insert("old_string".to_string(), Value::String(old.to_string()));
        args.insert("new_string".to_string(), Value::String(new.to_string()));
        args
    }

    #[test]
    fn crlf_tolerant_pattern_escapes_regex_metacharacters() {
        let pattern = crlf_tolerant_pattern("a.b(c)");
        let regex = Regex::new(&pattern).unwrap();
        assert!(regex.is_match("a.b(c)"));
        assert!(!regex.is_match("aXb(c)"));
    }

    #[rusty_tokio::test]
    async fn replaces_a_unique_occurrence() {
        let env = environment_with_file("a.txt", "hello world").await;
        let tool = EditFileTool::new(env.clone());
        let mut context = ctx();
        let result = tool
            .run_async(&args("a.txt", "world", "there"), &mut context)
            .await
            .unwrap();
        assert_eq!(
            result,
            Value::Map(vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "message".to_string(),
                    Value::String("Edited a.txt".to_string())
                ),
            ])
        );
        let content = env.read_file("a.txt").await.unwrap();
        assert_eq!(content, b"hello there");
    }

    #[rusty_tokio::test]
    async fn errors_when_old_string_is_absent() {
        let env = environment_with_file("a.txt", "hello world").await;
        let tool = EditFileTool::new(env);
        let mut context = ctx();
        let result = tool
            .run_async(&args("a.txt", "missing", "x"), &mut context)
            .await
            .unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("not found in file")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn errors_when_old_string_is_ambiguous() {
        let env = environment_with_file("a.txt", "aa aa aa").await;
        let tool = EditFileTool::new(env);
        let mut context = ctx();
        let result = tool
            .run_async(&args("a.txt", "aa", "bb"), &mut context)
            .await
            .unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("appears 3 times")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn crlf_content_matches_an_lf_old_string() {
        let env = environment_with_file("a.txt", "line1\r\nline2\r\nline3").await;
        let tool = EditFileTool::new(env.clone());
        let mut context = ctx();
        let result = tool
            .run_async(&args("a.txt", "line1\nline2", "replaced"), &mut context)
            .await
            .unwrap();
        assert_eq!(
            result,
            Value::Map(vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "message".to_string(),
                    Value::String("Edited a.txt".to_string())
                ),
            ])
        );
        let content = env.read_file("a.txt").await.unwrap();
        assert_eq!(content, b"replaced\r\nline3");
    }

    #[rusty_tokio::test]
    async fn missing_file_is_reported() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let tool = EditFileTool::new(Arc::new(env));
        let mut context = ctx();
        let result = tool
            .run_async(&args("missing.txt", "a", "b"), &mut context)
            .await
            .unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("File not found")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
