//! Capability C0444: `WriteFileTool`(`WriteFile`), ported from
//! `google.adk.tools.environment._write_file_tool`.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_environment::BaseEnvironment;
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

const DESCRIPTION: &str = "Create or overwrite a file in the environment. Use for new files or full rewrites. For small changes to existing files, prefer EditFile.";

fn error_response(message: impl Into<String>) -> Value {
    Value::Map(vec![
        ("status".to_string(), Value::String("error".to_string())),
        ("error".to_string(), Value::String(message.into())),
    ])
}

/// C0444: creates or overwrites a file in the injected [`BaseEnvironment`].
pub struct WriteFileTool {
    environment: Arc<dyn BaseEnvironment>,
}

impl WriteFileTool {
    pub fn new(environment: Arc<dyn BaseEnvironment>) -> Self {
        Self { environment }
    }
}

impl BaseTool for WriteFileTool {
    fn name(&self) -> &str {
        "WriteFile"
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
                                        "Path to the file within the environment.".to_string(),
                                    ),
                                ),
                            ]),
                        ),
                        (
                            "content".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String("The full file content to write.".to_string()),
                                ),
                            ]),
                        ),
                    ]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![
                        Value::String("path".to_string()),
                        Value::String("content".to_string()),
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
            let content = match args.get("content") {
                Some(Value::String(content)) => content.clone(),
                _ => String::new(),
            };

            if let Err(error) = self.environment.write_file(&path, content.as_bytes()).await {
                return Ok(error_response(error.to_string()));
            }

            Ok(Value::Map(vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "message".to_string(),
                    Value::String(format!("Wrote {path}")),
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

    async fn ready_environment() -> Arc<dyn BaseEnvironment> {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        Arc::new(env)
    }

    #[rusty_tokio::test]
    async fn writes_a_new_file() {
        let env = ready_environment().await;
        let tool = WriteFileTool::new(env.clone());
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("new.txt".to_string()));
        args.insert("content".to_string(), Value::String("hello".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::Map(vec![
                ("status".to_string(), Value::String("ok".to_string())),
                (
                    "message".to_string(),
                    Value::String("Wrote new.txt".to_string())
                ),
            ])
        );
        assert_eq!(env.read_file("new.txt").await.unwrap(), b"hello");
    }

    #[rusty_tokio::test]
    async fn overwrites_an_existing_file() {
        let env = ready_environment().await;
        env.write_file("a.txt", b"old").await.unwrap();
        let tool = WriteFileTool::new(env.clone());
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), Value::String("a.txt".to_string()));
        args.insert("content".to_string(), Value::String("new".to_string()));
        tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(env.read_file("a.txt").await.unwrap(), b"new");
    }

    #[rusty_tokio::test]
    async fn missing_path_is_an_error() {
        let tool = WriteFileTool::new(ready_environment().await);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("content".to_string(), Value::String("x".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("required")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
