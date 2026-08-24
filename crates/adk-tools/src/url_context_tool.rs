//! Capability C0432: `UrlContextTool`/`url_context`, ported from
//! `google.adk.tools.url_context_tool`.
//!
//! See `google_search_tool.rs`'s module doc for the two disclosed
//! narrowings shared by every built-in Gemini grounding tool in this
//! port: no `Result`-propagating failure path in `process_llm_request`
//! (an unsupported model simply doesn't get the marker appended, rather
//! than the source's `ValueError`), and `_is_managed_agent` always
//! reporting `false` (see `model_name_utils.rs`).

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;

use crate::append_tools::append_built_in_tool_marker;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::model_name_utils::{is_gemini_model_id_check_disabled, is_managed_agent};
use crate::tool_context::ToolContext;

/// C0432: a built-in tool automatically invoked by Gemini 2 models to
/// retrieve content from URLs and use it to inform the response.
pub struct UrlContextTool;

impl UrlContextTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UrlContextTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for UrlContextTool {
    fn name(&self) -> &str {
        "url_context"
    }

    fn description(&self) -> &str {
        "url_context"
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
            if is_gemini_model(llm_request.model.as_deref())
                || is_gemini_model_id_check_disabled()
                || is_managed_agent()
            {
                append_built_in_tool_marker(llm_request, "urlContext");
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
        let tool = UrlContextTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(
            request.config.tools,
            Some(Value::Seq(vec![Value::Map(vec![(
                "urlContext".to_string(),
                Value::Map(vec![])
            )])]))
        );
    }

    #[rusty_tokio::test]
    async fn does_not_append_for_a_non_gemini_model() {
        let tool = UrlContextTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.config.tools, None);
    }
}
