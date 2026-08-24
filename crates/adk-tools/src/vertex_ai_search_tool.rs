//! Capability C0433: `VertexAiSearchTool`/`vertex_ai_search`, ported from
//! `google.adk.tools.vertex_ai_search_tool`.
//!
//! **Disclosed narrowing (shared with every sibling built-in grounding
//! tool)**: the source raises `ValueError` when the request's model isn't
//! Gemini-compatible. This port's `BaseTool::process_llm_request` returns
//! `BoxFuture<'a, ()>` — no `Result` — so an unsupported model simply
//! doesn't get the tool appended, rather than the source's hard failure.
//! See `google_search_tool.rs`'s own module doc for the same disclosure.
//!
//! **`bypass_multi_tools_limit`, stored but not enforced**: same as
//! `GoogleSearchTool` (`google_search_tool.rs`) — nothing in this port
//! yet enforces the "Gemini restricts certain built-in tools to sole-tool
//! use" limitation it would bypass; that enforcement is deferred with the
//! rest of `agent_transfer.rs`'s own disclosed C0171 gap.
//!
//! **`data_store_specs`, opaque**: `types.VertexAISearchDataStoreSpec` is
//! a third-party Gemini-SDK type this migration doesn't model structurally
//! (same "opaque `Value` placeholder for an SDK type" precedent used
//! throughout `run_config.rs`) — each entry round-trips through
//! `process_llm_request` as an opaque [`rusty_serde::value::Value`]
//! without this port ever inspecting its fields.
//!
//! **`_build_vertex_ai_search_config`, adapted**: the source's
//! customization point is subclassing (documented in its own docstring
//! example: override the method to set a per-request filter from session
//! state). This port has no subclassing, so it becomes an optional
//! closure field (`with_config_builder`) — the same "overridable Python
//! method → closure field" adaptation `base_agent.rs`'s `AgentCallback`
//! already established for `before_agent_callback`/`after_agent_callback`.
//!
//! **`logger.debug`, dropped**: no logging framework is adopted in this
//! crate (an already-established scope cut elsewhere in this migration).

use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture};
use crate::model_name_utils::is_gemini_model_id_check_disabled;
use crate::tool_context::ToolContext;

#[derive(Debug, rusty_err::Error)]
pub enum VertexAiSearchToolError {
    #[error("Either data_store_id or search_engine_id must be specified.")]
    MissingDataStoreOrSearchEngine,
    #[error("search_engine_id must be specified if data_store_specs is specified.")]
    DataStoreSpecsRequireSearchEngine,
}

/// The resolved `types.VertexAISearch` shape — either the tool's own
/// static fields, or whatever `with_config_builder`'s closure returns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VertexAiSearchConfig {
    pub data_store_id: Option<String>,
    pub data_store_specs: Option<Vec<Value>>,
    pub search_engine_id: Option<String>,
    pub filter: Option<String>,
    pub max_results: Option<u32>,
}

type ConfigBuilder = Arc<dyn Fn(&ToolContext) -> VertexAiSearchConfig + Send + Sync>;

/// C0433: a built-in Gemini tool using Vertex AI Search. Operates entirely
/// inside the model — this port, like the source, performs no local
/// search execution.
pub struct VertexAiSearchTool {
    pub data_store_id: Option<String>,
    pub data_store_specs: Option<Vec<Value>>,
    pub search_engine_id: Option<String>,
    pub filter: Option<String>,
    pub max_results: Option<u32>,
    pub bypass_multi_tools_limit: bool,
    config_builder: Option<ConfigBuilder>,
}

impl VertexAiSearchTool {
    /// Errors if `data_store_id`/`search_engine_id` are both set or both
    /// unset, or if `data_store_specs` is set without `search_engine_id`.
    pub fn new(
        data_store_id: Option<String>,
        data_store_specs: Option<Vec<Value>>,
        search_engine_id: Option<String>,
        filter: Option<String>,
        max_results: Option<u32>,
    ) -> Result<Self, VertexAiSearchToolError> {
        if data_store_id.is_none() == search_engine_id.is_none() {
            return Err(VertexAiSearchToolError::MissingDataStoreOrSearchEngine);
        }
        if data_store_specs.is_some() && search_engine_id.is_none() {
            return Err(VertexAiSearchToolError::DataStoreSpecsRequireSearchEngine);
        }
        Ok(Self {
            data_store_id,
            data_store_specs,
            search_engine_id,
            filter,
            max_results,
            bypass_multi_tools_limit: false,
            config_builder: None,
        })
    }

    /// See the module doc's `_build_vertex_ai_search_config` adaptation.
    pub fn with_config_builder(mut self, builder: ConfigBuilder) -> Self {
        self.config_builder = Some(builder);
        self
    }

    fn build_config(&self, ctx: &ToolContext) -> VertexAiSearchConfig {
        match &self.config_builder {
            Some(builder) => builder(ctx),
            None => VertexAiSearchConfig {
                data_store_id: self.data_store_id.clone(),
                data_store_specs: self.data_store_specs.clone(),
                search_engine_id: self.search_engine_id.clone(),
                filter: self.filter.clone(),
                max_results: self.max_results,
            },
        }
    }
}

impl BaseTool for VertexAiSearchTool {
    fn name(&self) -> &str {
        "vertex_ai_search"
    }

    fn description(&self) -> &str {
        "vertex_ai_search"
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        None
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if !(is_gemini_model(llm_request.model.as_deref())
                || is_gemini_model_id_check_disabled())
            {
                return;
            }

            let config = self.build_config(tool_context);
            let mut vertex_ai_search = Vec::new();
            if let Some(data_store) = config.data_store_id {
                vertex_ai_search.push(("datastore".to_string(), Value::String(data_store)));
            }
            if let Some(specs) = config.data_store_specs {
                vertex_ai_search.push(("dataStoreSpecs".to_string(), Value::Seq(specs)));
            }
            if let Some(engine) = config.search_engine_id {
                vertex_ai_search.push(("engine".to_string(), Value::String(engine)));
            }
            if let Some(filter) = config.filter {
                vertex_ai_search.push(("filter".to_string(), Value::String(filter)));
            }
            if let Some(max_results) = config.max_results {
                vertex_ai_search.push(("maxResults".to_string(), Value::from(max_results)));
            }

            if !matches!(llm_request.config.tools, Some(Value::Seq(_))) {
                llm_request.config.tools = Some(Value::Seq(Vec::new()));
            }
            let Some(Value::Seq(entries)) = &mut llm_request.config.tools else {
                unreachable!("just ensured config.tools is Some(Value::Seq(_))");
            };
            entries.push(Value::Map(vec![(
                "retrieval".to_string(),
                Value::Map(vec![(
                    "vertexAiSearch".to_string(),
                    Value::Map(vertex_ai_search),
                )]),
            )]));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    fn expected_tools_config(vertex_ai_search_fields: Vec<(String, Value)>) -> Option<Value> {
        let vertex_ai_search = Value::Map(vertex_ai_search_fields);
        let retrieval = Value::Map(vec![("vertexAiSearch".to_string(), vertex_ai_search)]);
        let tool_entry = Value::Map(vec![("retrieval".to_string(), retrieval)]);
        Some(Value::Seq(vec![tool_entry]))
    }

    #[test]
    fn new_errors_when_neither_data_store_nor_search_engine_is_set() {
        // `VertexAiSearchTool` doesn't derive `Debug` (its optional
        // closure field can't), so `unwrap_err()` isn't available here —
        // match directly instead, same as `canonical_model.rs`'s own
        // `Arc<dyn BaseLlm>`-returning tests.
        match VertexAiSearchTool::new(None, None, None, None, None) {
            Err(VertexAiSearchToolError::MissingDataStoreOrSearchEngine) => {}
            _ => panic!("expected MissingDataStoreOrSearchEngine"),
        }
    }

    #[test]
    fn new_errors_when_both_data_store_and_search_engine_are_set() {
        match VertexAiSearchTool::new(
            Some("ds".to_string()),
            None,
            Some("engine".to_string()),
            None,
            None,
        ) {
            Err(VertexAiSearchToolError::MissingDataStoreOrSearchEngine) => {}
            _ => panic!("expected MissingDataStoreOrSearchEngine"),
        }
    }

    #[test]
    fn new_errors_when_data_store_specs_is_set_without_search_engine() {
        match VertexAiSearchTool::new(
            Some("ds".to_string()),
            Some(vec![Value::Map(vec![])]),
            None,
            None,
            None,
        ) {
            Err(VertexAiSearchToolError::DataStoreSpecsRequireSearchEngine) => {}
            _ => panic!("expected DataStoreSpecsRequireSearchEngine"),
        }
    }

    #[test]
    fn new_accepts_a_data_store_id_alone() {
        assert!(VertexAiSearchTool::new(Some("ds".to_string()), None, None, None, None).is_ok());
    }

    #[test]
    fn new_accepts_a_search_engine_with_data_store_specs() {
        assert!(VertexAiSearchTool::new(
            None,
            Some(vec![Value::Map(vec![])]),
            Some("engine".to_string()),
            None,
            None
        )
        .is_ok());
    }

    #[rusty_tokio::test]
    async fn appends_the_populated_config_for_a_gemini_model() {
        let tool = VertexAiSearchTool::new(
            Some("ds".to_string()),
            None,
            None,
            Some("category = public".to_string()),
            Some(5),
        )
        .unwrap();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;

        assert_eq!(
            request.config.tools,
            expected_tools_config(vec![
                ("datastore".to_string(), Value::String("ds".to_string())),
                (
                    "filter".to_string(),
                    Value::String("category = public".to_string())
                ),
                ("maxResults".to_string(), Value::from(5u32)),
            ])
        );
    }

    #[rusty_tokio::test]
    async fn does_not_append_for_a_non_gemini_model() {
        let tool = VertexAiSearchTool::new(Some("ds".to_string()), None, None, None, None).unwrap();
        let mut context = ctx();
        let mut request = LlmRequest::new("gpt-4");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.config.tools, None);
    }

    #[rusty_tokio::test]
    async fn with_config_builder_overrides_the_static_fields() {
        let tool = VertexAiSearchTool::new(Some("ds".to_string()), None, None, None, None)
            .unwrap()
            .with_config_builder(Arc::new(|_ctx: &ToolContext| VertexAiSearchConfig {
                search_engine_id: Some("dynamic-engine".to_string()),
                ..Default::default()
            }));
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;

        assert_eq!(
            request.config.tools,
            expected_tools_config(vec![(
                "engine".to_string(),
                Value::String("dynamic-engine".to_string())
            )])
        );
    }
}
