//! Capability C0428: `GoogleSearchTool`/`google_search`, ported from
//! `google.adk.tools.google_search_tool`.
//!
//! **Disclosed narrowing**: the source raises `ValueError` when the
//! request's model isn't Gemini-compatible. This port's
//! `BaseTool::process_llm_request` returns `BoxFuture<'a, ()>` — no
//! `Result`, so there is no way to propagate a hard failure through it
//! (the same structural constraint every other `process_llm_request`
//! override in this port is bound by). An unsupported model here simply
//! doesn't get `google_search` appended to `llm_request.config.tools`,
//! rather than the source's hard failure — a real, disclosed behavior
//! gap, not a silent no-op dressed up as success.
//!
//! `bypass_multi_tools_limit` is stored for API-shape parity but nothing
//! in this port enforces the "Gemini restricts `google_search` to
//! sole-tool use" limitation it would bypass — that enforcement
//! (`_get_incompatible_builtin_tool_error`) is deferred with the rest of
//! `agent_transfer.rs`'s own disclosed C0171 gap.
//!
//! `_is_managed_agent` isn't ported — see `model_name_utils.rs`.

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;

use crate::append_tools::append_built_in_tool_marker;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::model_name_utils::{is_gemini_model_id_check_disabled, is_managed_agent};
use crate::tool_context::ToolContext;

/// C0428: a built-in tool automatically invoked by Gemini models to
/// retrieve search results from Google Search. Operates entirely inside
/// the model — this port, like the source, performs no local search
/// execution.
pub struct GoogleSearchTool {
    pub bypass_multi_tools_limit: bool,
    pub model: Option<String>,
}

impl GoogleSearchTool {
    pub fn new() -> Self {
        Self {
            bypass_multi_tools_limit: false,
            model: None,
        }
    }
}

impl Default for GoogleSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for GoogleSearchTool {
    fn name(&self) -> &str {
        "google_search"
    }

    fn description(&self) -> &str {
        "google_search"
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        None
    }

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(model) = &self.model {
                llm_request.model = Some(model.clone());
            }
            if is_gemini_model(llm_request.model.as_deref())
                || is_gemini_model_id_check_disabled()
                || is_managed_agent()
            {
                append_built_in_tool_marker(llm_request, "googleSearch");
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use rusty_serde::value::Value;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn appends_the_marker_for_a_gemini_model() {
        let tool = GoogleSearchTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(
            request.config.tools,
            Some(Value::Seq(vec![Value::Map(vec![(
                "googleSearch".to_string(),
                Value::Map(vec![])
            )])]))
        );
    }

    #[rusty_tokio::test]
    async fn does_not_append_for_a_non_gemini_model() {
        let tool = GoogleSearchTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.config.tools, None);
    }

    #[rusty_tokio::test]
    async fn overrides_the_request_model_when_configured() {
        let tool = GoogleSearchTool {
            bypass_multi_tools_limit: false,
            model: Some("gemini-2.5-flash".to_string()),
        };
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
        assert!(request.config.tools.is_some());
    }
}
