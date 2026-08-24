//! Capability C0442: `ReadFileTool`(`ReadFile`), ported from
//! `google.adk.tools.environment._read_file_tool`.
//!
//! **`bytes.splitlines(keepends=True)`, narrowed**: this port's
//! [`split_lines_keepends`] recognizes `\n`, `\r\n`, and `\r` as line
//! boundaries — the overwhelming majority case for real text files.
//! Python's `bytes.splitlines()` also treats `\v`/`\f`/`\x1c`/`\x1d`/`\x1e`
//! as boundaries; those single-byte control characters essentially never
//! appear in real source/text files being read by this tool, and no
//! other module in this port needs a fuller boundary set, so they're not
//! reproduced here — a disclosed narrowing, not a silent one.
//!
//! **Adaptation**: `_is_valid_line_number`'s `isinstance(value, int) and
//! not isinstance(value, bool)` guard (needed in Python because `bool` is
//! an `int` subclass, so a raw `isinstance(v, int)` check would wrongly
//! accept `True`/`False`) has no Rust equivalent to guard against:
//! `rusty_serde::value::Value::Bool` and `Value::Int`/`Value::UInt` are
//! disjoint enum variants, so matching only the integer variants already
//! excludes booleans structurally.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_environment::{BaseEnvironment, EnvironmentError};
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::execute_tool::{truncate, MAX_OUTPUT_CHARS};
use crate::tool_context::ToolContext;

const DESCRIPTION: &str =
    "Read the contents of a file in the environment. Returns the file content with line numbers.";

fn error_response(message: impl Into<String>) -> Value {
    Value::Map(vec![
        ("status".to_string(), Value::String("error".to_string())),
        ("error".to_string(), Value::String(message.into())),
    ])
}

fn error_response_with_total(message: impl Into<String>, total: usize) -> Value {
    Value::Map(vec![
        ("status".to_string(), Value::String("error".to_string())),
        ("error".to_string(), Value::String(message.into())),
        ("total_lines".to_string(), Value::Int(total as i64)),
    ])
}

/// `bytes.splitlines(keepends=True)`, narrowed — see the module doc.
fn split_lines_keepends(data: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < data.len() {
        match data[i] {
            b'\n' => {
                lines.push(&data[start..=i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                let end = if i + 1 < data.len() && data[i + 1] == b'\n' {
                    i + 2
                } else {
                    i + 1
                };
                lines.push(&data[start..end]);
                i = end;
                start = end;
            }
            _ => i += 1,
        }
    }
    if start < data.len() {
        lines.push(&data[start..]);
    }
    lines
}

fn as_line_number(value: Option<&Value>) -> Result<Option<i64>, Value> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Int(n)) => Ok(Some(*n)),
        Some(Value::UInt(n)) => Ok(Some(*n as i64)),
        Some(_) => Err(Value::Null),
    }
}

/// C0442: reads a file from the injected [`BaseEnvironment`], returning
/// line-numbered, optionally-ranged, truncated content.
pub struct ReadFileTool {
    environment: Arc<dyn BaseEnvironment>,
    max_output_chars: usize,
}

impl ReadFileTool {
    pub fn new(environment: Arc<dyn BaseEnvironment>, max_output_chars: Option<usize>) -> Self {
        Self {
            environment,
            max_output_chars: max_output_chars.unwrap_or(MAX_OUTPUT_CHARS),
        }
    }
}

impl BaseTool for ReadFileTool {
    fn name(&self) -> &str {
        "ReadFile"
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
                                        "Path of the file to read within the environment."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "start_line".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("integer".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "First line to return (1-based, inclusive). Defaults to 1."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "end_line".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("integer".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "Last line to return (1-based, inclusive). Defaults to end of file."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                    ]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("path".to_string())]),
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

            let start_line = match as_line_number(args.get("start_line")) {
                Ok(value) => value,
                Err(_) => {
                    return Ok(error_response(
                        "`start_line` must be an integer if provided.",
                    ))
                }
            };
            let end_line = match as_line_number(args.get("end_line")) {
                Ok(value) => value,
                Err(_) => return Ok(error_response("`end_line` must be an integer if provided.")),
            };

            let data = match self.environment.read_file(&path).await {
                Ok(data) => data,
                Err(EnvironmentError::FileNotFound(_)) => {
                    return Ok(error_response(format!("File not found: {path}")))
                }
                Err(error) => return Ok(error_response(error.to_string())),
            };

            let lines_bytes = split_lines_keepends(&data);
            let total = lines_bytes.len() as i64;
            let start = start_line.unwrap_or(1).max(1);
            let end = end_line.unwrap_or(total).min(total);

            if start > total {
                return Ok(error_response_with_total(
                    format!("`start_line` {start} exceeds file length ({total} lines)."),
                    total as usize,
                ));
            }
            if start > end {
                return Ok(error_response_with_total(
                    format!("`start_line` ({start}) is after `end_line` ({end})."),
                    total as usize,
                ));
            }

            let selected = &lines_bytes[(start - 1) as usize..end as usize];
            let mut numbered = String::new();
            for (i, line_bytes) in selected.iter().enumerate() {
                let line_no = start + i as i64;
                numbered.push_str(&format!(
                    "{line_no:>6}\t{}",
                    String::from_utf8_lossy(line_bytes)
                ));
            }

            let mut fields = vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "content".to_string(),
                    Value::String(truncate(&numbered, self.max_output_chars)),
                ),
            ];
            if start > 1 || end < total {
                fields.push(("total_lines".to_string(), Value::Int(total)));
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

    async fn environment_with_file(path: &str, content: &str) -> Arc<dyn BaseEnvironment> {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        env.write_file(path, content.as_bytes()).await.unwrap();
        Arc::new(env)
    }

    #[test]
    fn split_lines_keepends_handles_lf_crlf_and_cr() {
        let lines = split_lines_keepends(b"a\nb\r\nc\rd");
        assert_eq!(
            lines,
            vec![
                b"a\n".as_slice(),
                b"b\r\n".as_slice(),
                b"c\r".as_slice(),
                b"d".as_slice()
            ]
        );
    }

    #[rusty_tokio::test]
    async fn reads_a_whole_file_with_line_numbers() {
        let tool = ReadFileTool::new(
            environment_with_file("a.txt", "one\ntwo\nthree").await,
            None,
        );
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("a.txt".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let content = fields.iter().find(|(k, _)| k == "content").unwrap();
                assert!(
                    matches!(&content.1, Value::String(s) if s.contains("     1\tone\n") && s.contains("     3\tthree"))
                );
                assert!(fields.iter().all(|(k, _)| k != "total_lines"));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn a_line_range_includes_total_lines() {
        let tool = ReadFileTool::new(
            environment_with_file("a.txt", "one\ntwo\nthree\nfour").await,
            None,
        );
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("a.txt".to_string()));
        args.insert("start_line".to_string(), Value::Int(2));
        args.insert("end_line".to_string(), Value::Int(3));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let content = fields.iter().find(|(k, _)| k == "content").unwrap();
                assert!(
                    matches!(&content.1, Value::String(s) if s.contains("two") && s.contains("three") && !s.contains("one") && !s.contains("four"))
                );
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "total_lines").unwrap().1,
                    Value::Int(4)
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn start_line_past_the_end_is_an_error() {
        let tool = ReadFileTool::new(environment_with_file("a.txt", "one\ntwo").await, None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("a.txt".to_string()));
        args.insert("start_line".to_string(), Value::Int(10));
        let result = tool.run_async(&args, &mut context).await.unwrap();
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
    async fn missing_file_is_reported() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let tool = ReadFileTool::new(Arc::new(env), None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("missing.txt".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("File not found")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn a_non_integer_start_line_is_rejected() {
        let tool = ReadFileTool::new(environment_with_file("a.txt", "one").await, None);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("a.txt".to_string()));
        args.insert("start_line".to_string(), Value::Bool(true));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("must be an integer")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
