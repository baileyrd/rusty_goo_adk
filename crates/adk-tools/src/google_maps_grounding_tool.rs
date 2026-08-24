//! Capability C0430: `GoogleMapsGroundingTool`/`google_maps_grounding`,
//! ported from `google.adk.tools.google_maps_grounding_tool`.
//!
//! Only available for use with the Vertex AI Gemini API (e.g.
//! `GOOGLE_GENAI_USE_ENTERPRISE=TRUE`) per the source's own docstring —
//! this port doesn't enforce that restriction either (neither does the
//! source's own `process_llm_request`; it's documentation, not a runtime
//! check). See `google_search_tool.rs`'s module doc for the disclosed
//! narrowing shared by every built-in Gemini grounding tool in this
//! port: no `Result`-propagating failure path in `process_llm_request`.
//! Like `enterprise_search_tool.rs`, the source itself never checks
//! `_is_managed_agent` here either.

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;

use crate::append_tools::append_built_in_tool_marker;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::model_name_utils::is_gemini_model_id_check_disabled;
use crate::tool_context::ToolContext;

/// C0430: a built-in tool automatically invoked by Gemini models to
/// ground query results with Google Maps.
pub struct GoogleMapsGroundingTool;

impl GoogleMapsGroundingTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GoogleMapsGroundingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for GoogleMapsGroundingTool {
    fn name(&self) -> &str {
        "google_maps"
    }

    fn description(&self) -> &str {
        "google_maps"
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
                append_built_in_tool_marker(llm_request, "googleMaps");
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
        let tool = GoogleMapsGroundingTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(
            request.config.tools,
            Some(Value::Seq(vec![Value::Map(vec![(
                "googleMaps".to_string(),
                Value::Map(vec![])
            )])]))
        );
    }

    #[rusty_tokio::test]
    async fn does_not_append_for_a_non_gemini_model() {
        let tool = GoogleMapsGroundingTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.config.tools, None);
    }
}
