//! Capability C0431: `EnterpriseWebSearchTool`/`enterprise_web_search`,
//! ported from `google.adk.tools.enterprise_search_tool`.
//!
//! Distinct from Vertex AI Search (which is itself sometimes called
//! "Enterprise Search" — a naming overlap the source's own docstring
//! flags). See `google_search_tool.rs`'s module doc for the disclosed
//! narrowing shared by every built-in Gemini grounding tool in this
//! port: no `Result`-propagating failure path in `process_llm_request`
//! (an unsupported model simply doesn't get the marker appended, rather
//! than the source's `ValueError`). Unlike the sibling tools, the source
//! itself never checks `_is_managed_agent` here either — this port
//! matches that omission exactly, not a narrowing of its own.

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;

use crate::append_tools::append_built_in_tool_marker;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::model_name_utils::is_gemini_model_id_check_disabled;
use crate::tool_context::ToolContext;

/// C0431: a Gemini built-in tool using web grounding for Enterprise
/// compliance.
pub struct EnterpriseWebSearchTool;

impl EnterpriseWebSearchTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EnterpriseWebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for EnterpriseWebSearchTool {
    fn name(&self) -> &str {
        "enterprise_web_search"
    }

    fn description(&self) -> &str {
        "enterprise_web_search"
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
            if is_gemini_model(llm_request.model.as_deref()) || is_gemini_model_id_check_disabled()
            {
                append_built_in_tool_marker(llm_request, "enterpriseWebSearch");
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
        let tool = EnterpriseWebSearchTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(
            request.config.tools,
            Some(Value::Seq(vec![Value::Map(vec![(
                "enterpriseWebSearch".to_string(),
                Value::Map(vec![])
            )])]))
        );
    }

    #[rusty_tokio::test]
    async fn does_not_append_for_a_non_gemini_model() {
        let tool = EnterpriseWebSearchTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.config.tools, None);
    }
}
