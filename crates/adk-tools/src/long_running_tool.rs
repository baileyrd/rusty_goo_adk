//! Capability C0422: `LongRunningFunctionTool`, ported from
//! `google.adk.tools.long_running_tool`.
//!
//! **Adaptation**: the source subclasses `FunctionTool` and overrides
//! `_get_declaration`. Rust has no struct inheritance, so this port wraps
//! a [`FunctionTool`] by composition instead, delegating every method
//! except `is_long_running` (always `true`) and `get_declaration` (appends
//! the "don't call again while pending" instruction) — the same
//! wrap-and-delegate shape this port already uses wherever the source
//! relies on subclassing a concrete class (not just an abstract
//! interface).

use std::collections::BTreeMap;

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::GoogleLlmVariant;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ResponseScheduling, ToolError};
use crate::function_tool::FunctionTool;
use crate::tool_context::ToolContext;

const LONG_RUNNING_INSTRUCTION: &str = "\n\nNOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status.";

/// C0422: a [`FunctionTool`] whose result is returned asynchronously —
/// the framework calls the function, and once it returns, the response is
/// matched back to the pending call by `function_call_id`.
pub struct LongRunningFunctionTool {
    inner: FunctionTool,
}

impl LongRunningFunctionTool {
    pub fn new(inner: FunctionTool) -> Self {
        Self { inner }
    }
}

impl BaseTool for LongRunningFunctionTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn is_long_running(&self) -> bool {
        true
    }

    fn custom_metadata(&self) -> Option<&BTreeMap<String, Value>> {
        self.inner.custom_metadata()
    }

    fn response_scheduling(&self) -> Option<ResponseScheduling> {
        self.inner.response_scheduling()
    }

    fn api_variant(&self) -> GoogleLlmVariant {
        self.inner.api_variant()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        let mut declaration = self.inner.get_declaration()?;
        declaration.description = Some(match declaration.description {
            Some(description) if !description.is_empty() => {
                format!("{description}{LONG_RUNNING_INSTRUCTION}")
            }
            _ => LONG_RUNNING_INSTRUCTION.trim_start().to_string(),
        });
        Some(declaration)
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        self.inner.run_async(args, tool_context)
    }

    fn check_require_confirmation<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        self.inner.check_require_confirmation(args, tool_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use std::sync::Arc;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    fn wrapped(description: &str) -> LongRunningFunctionTool {
        LongRunningFunctionTool::new(FunctionTool::new(
            "do_thing",
            "does a thing",
            FunctionDeclaration {
                name: Some("do_thing".to_string()),
                description: if description.is_empty() {
                    None
                } else {
                    Some(description.to_string())
                },
                ..Default::default()
            },
            vec![],
            Arc::new(|_args, _ctx| Box::pin(async move { Value::String("done".to_string()) })),
        ))
    }

    #[test]
    fn is_long_running_is_always_true() {
        assert!(wrapped("does a thing").is_long_running());
    }

    #[test]
    fn get_declaration_appends_the_pending_instruction() {
        let tool = wrapped("does a thing");
        let declaration = tool.get_declaration().unwrap();
        let description = declaration.description.unwrap();
        assert!(description.starts_with("does a thing"));
        assert!(description.contains("NOTE: This is a long-running operation"));
    }

    #[test]
    fn get_declaration_uses_the_bare_instruction_when_no_description_is_set() {
        let tool = wrapped("");
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(
            declaration.description.unwrap(),
            LONG_RUNNING_INSTRUCTION.trim_start()
        );
    }

    #[rusty_tokio::test]
    async fn run_async_delegates_to_the_wrapped_tool() {
        let tool = wrapped("does a thing");
        let mut context = ctx();
        let result = tool
            .run_async(&BTreeMap::new(), &mut context)
            .await
            .unwrap();
        assert_eq!(result, Value::String("done".to_string()));
    }
}
