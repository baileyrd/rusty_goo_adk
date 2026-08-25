//! Capability C0402 (partial): `BaseTool`, ported from
//! `google.adk.tools.base_tool`.
//!
//! **Adaptation**: the source's `BaseTool` is an abstract *class* — plain
//! instance attributes (`name`, `description`, `is_long_running`, ...) plus
//! overridable methods. Rust traits have no data fields, so every one of
//! those attributes becomes a trait method instead (matching
//! `adk_models::base_llm::BaseLlm`'s own `fn model(&self) -> &str` style,
//! the established precedent in this port for the same class-with-both-
//! data-and-behavior shape). A concrete tool struct stores its own
//! `name`/`description`/etc. however it likes and returns them from these
//! methods.
//!
//! **Not** ported: `from_config`. `ToolArgsConfig`/`ToolConfig` (C0417,
//! `crate::tool_configs`) — the declarative YAML/dict tool-reference
//! shape `from_config` would validate its `args` against — are real,
//! tested types now; what's still missing is the *dynamic-dispatch
//! resolution* itself (5 reference kinds: built-in name / instance path
//! / class+args / factory+args / function path), which needs Python's
//! `importlib` — genuinely inapplicable in this port, same
//! disclosed-inapplicable precedent already established for C0939's
//! `_lazy.accessors`, not a "not built yet" gap C0417 landing closes.
//! Also not ported: the `SelfTool` generic-return-type pattern
//! `from_config` uses (a Rust trait method can't return `Self` behind a
//! trait object the way Python's classmethod can return the concrete
//! subclass).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::{get_google_llm_variant, GoogleLlmVariant};
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::tool_context::ToolContext;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum ToolError {
    #[error("{0} is not implemented")]
    NotImplemented(String),
    /// A tool that drives a nested run (e.g. `AgentTool`, C0406) failed to
    /// create its session or complete the nested agent's turn.
    #[error("{0}")]
    NestedRunFailed(String),
    /// `BaseAuthenticatedTool`/`AuthenticatedFunctionTool` (C0412) failed
    /// to resolve or request an auth credential. The source lets the
    /// underlying exception propagate uncaught; this port surfaces it
    /// through `run_async`'s `Result` instead.
    #[error("{0}")]
    CredentialResolutionFailed(String),
}

/// `types.FunctionResponseScheduling` — controls when the model reacts to
/// the tool's response (Live API only). `None` (absent on [`BaseTool`])
/// preserves the default behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseScheduling {
    /// Feeds the response back without triggering a model turn.
    Silent,
    /// Defers the reaction until the model is idle.
    WhenIdle,
    /// Reacts immediately.
    Interrupt,
}

/// C0402: the base trait for all tools. See the module doc for why every
/// source instance attribute is a method here instead of a field.
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;

    /// Whether the tool is a long running operation, which typically
    /// returns a resource id first and finishes the operation later.
    fn is_long_running(&self) -> bool {
        false
    }

    /// Optional JSON-serializable key-value metadata for this tool.
    fn custom_metadata(&self) -> Option<&BTreeMap<String, Value>> {
        None
    }

    /// The tool-wide default for when the model reacts to this tool's
    /// response (Live API only, asynchronous function calling).
    fn response_scheduling(&self) -> Option<ResponseScheduling> {
        None
    }

    /// The API variant this tool is running against — `_api_variant`.
    fn api_variant(&self) -> GoogleLlmVariant {
        get_google_llm_variant()
    }

    /// Gets the `FunctionDeclaration` of this tool, or `None` if it
    /// doesn't need to be added to `LlmRequest.config` (e.g. a built-in
    /// Gemini tool like Google Search).
    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        None
    }

    /// Runs the tool with the given arguments and context. Required if
    /// this tool needs to run at the client side; otherwise can be
    /// skipped (e.g. for a built-in Gemini tool).
    fn run_async<'a>(
        &'a self,
        _args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        let name = self.name().to_string();
        Box::pin(async move { Err(ToolError::NotImplemented(name)) })
    }

    /// Processes the outgoing LLM request for this tool. Most tools just
    /// add themselves to the request via [`merge_declarations`]; some may
    /// only preprocess the request instead.
    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                merge_declarations(llm_request, [(self.name().to_string(), declaration)]);
            }
        })
    }

    /// Returns whether the tool requires confirmation for the given args.
    fn check_require_confirmation<'a>(
        &'a self,
        _args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        Box::pin(async { false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MinimalTool;

    impl BaseTool for MinimalTool {
        fn name(&self) -> &str {
            "minimal_tool"
        }
        fn description(&self) -> &str {
            "A minimal tool with only the required methods overridden."
        }
    }

    #[rusty_tokio::test]
    async fn default_run_async_is_not_implemented() {
        let tool = MinimalTool;
        let args = BTreeMap::new();
        let mut ctx = adk_agents::context::Context::new(
            adk_agents::invocation_context::InvocationContextBuilder::new(
                "inv-1",
                adk_agents::session::Session::new("app", "user", "s1"),
            )
            .build(),
        );
        match tool.run_async(&args, &mut ctx).await {
            Err(ToolError::NotImplemented(name)) => assert_eq!(name, "minimal_tool"),
            _ => panic!("expected NotImplemented"),
        }
    }

    #[rusty_tokio::test]
    async fn default_check_require_confirmation_is_false() {
        let tool = MinimalTool;
        let args = BTreeMap::new();
        let mut ctx = adk_agents::context::Context::new(
            adk_agents::invocation_context::InvocationContextBuilder::new(
                "inv-1",
                adk_agents::session::Session::new("app", "user", "s1"),
            )
            .build(),
        );
        assert!(!tool.check_require_confirmation(&args, &mut ctx).await);
    }

    #[rusty_tokio::test]
    async fn default_process_llm_request_appends_via_append_tools() {
        struct DeclaringTool;
        impl BaseTool for DeclaringTool {
            fn name(&self) -> &str {
                "declaring_tool"
            }
            fn description(&self) -> &str {
                "d"
            }
            fn get_declaration(&self) -> Option<FunctionDeclaration> {
                Some(FunctionDeclaration {
                    name: Some("declaring_tool".to_string()),
                    ..Default::default()
                })
            }
        }

        let tool = DeclaringTool;
        let mut ctx = adk_agents::context::Context::new(
            adk_agents::invocation_context::InvocationContextBuilder::new(
                "inv-1",
                adk_agents::session::Session::new("app", "user", "s1"),
            )
            .build(),
        );
        let mut request = LlmRequest::default();
        tool.process_llm_request(&mut ctx, &mut request).await;
        assert!(request.config.tools.is_some());
    }

    #[test]
    fn defaults_are_none_or_false() {
        let tool = MinimalTool;
        assert!(!tool.is_long_running());
        assert!(tool.custom_metadata().is_none());
        assert!(tool.response_scheduling().is_none());
        assert!(tool.get_declaration().is_none());
    }
}
