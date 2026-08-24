//! Capability C0423: `load_memory`/`LoadMemoryTool`, ported from
//! `google.adk.tools.load_memory_tool`.
//!
//! **Adaptation**: the source's `load_memory` can raise (via
//! `tool_context.search_memory`, e.g. when no memory service is
//! configured). This port's `FunctionTool`'s wrapped-closure signature has
//! no `Result` to propagate an error through (see `function_tool.rs`'s own
//! module doc), so a missing memory service is reported the same way a
//! missing mandatory argument already is: an `{"error": ...}` response
//! value, not a panic or a silently swallowed failure.
//!
//! **Not** ported: the `is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)`
//! branch choosing between `parameters_json_schema` and `parameters` — no
//! feature-flag system exists in this port, so the declaration always
//! uses `parameters` (this port's `FunctionDeclaration` has both fields
//! available if a future batch adds the flag).

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_agents::services::SearchMemoryResponse;
use adk_genai::content::FunctionDeclaration;
use adk_models::llm_request::{Instructions, LlmRequest};
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::base_tool::{BaseTool, BoxFuture, ResponseScheduling, ToolError};
use crate::function_tool::FunctionTool;
use crate::tool_context::ToolContext;

const LOAD_MEMORY_INSTRUCTION: &str = "\nYou have memory. You can use it to answer questions. If any questions need\nyou to look up the memory, you should call load_memory function with a query.\n";

fn declaration() -> FunctionDeclaration {
    FunctionDeclaration {
        name: Some("load_memory".to_string()),
        description: Some("Loads the memory for the current user.".to_string()),
        parameters: Some(Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![(
                    "query".to_string(),
                    Value::Map(vec![(
                        "type".to_string(),
                        Value::String("string".to_string()),
                    )]),
                )]),
            ),
            (
                "required".to_string(),
                Value::Seq(vec![Value::String("query".to_string())]),
            ),
        ])),
        ..Default::default()
    }
}

fn search_memory_response_to_value(response: &SearchMemoryResponse) -> Value {
    let memories = response
        .memories
        .iter()
        .map(|memory| rusty_serde::json::to_value(memory).unwrap_or(Value::Null))
        .collect();
    Value::Map(vec![("memories".to_string(), Value::Seq(memories))])
}

/// C0423: loads the memory for the current user.
pub async fn load_memory(args: &BTreeMap<String, Value>, tool_context: &mut ToolContext) -> Value {
    let query = match args.get("query") {
        Some(Value::String(query)) => query.clone(),
        _ => String::new(),
    };
    match tool_context.search_memory(&query).await {
        Ok(response) => search_memory_response_to_value(&response),
        Err(err) => Value::Map(vec![("error".to_string(), Value::String(err.to_string()))]),
    }
}

/// C0423: a tool that loads the memory for the current user. Currently
/// only uses the text part from each memory entry (matches the source's
/// own documented limitation).
pub struct LoadMemoryTool {
    inner: FunctionTool,
}

impl LoadMemoryTool {
    pub fn new() -> Self {
        Self {
            inner: FunctionTool::new(
                "load_memory",
                "Loads the memory for the current user.",
                declaration(),
                vec!["query".to_string()],
                Arc::new(|args, tool_context| Box::pin(load_memory(args, tool_context))),
            ),
        }
    }
}

impl Default for LoadMemoryTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for LoadMemoryTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn custom_metadata(&self) -> Option<&BTreeMap<String, Value>> {
        self.inner.custom_metadata()
    }

    fn response_scheduling(&self) -> Option<ResponseScheduling> {
        self.inner.response_scheduling()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        self.inner.get_declaration()
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

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                merge_declarations(llm_request, [(self.name().to_string(), declaration)]);
            }
            llm_request.append_instructions(Instructions::Strings(vec![
                LOAD_MEMORY_INSTRUCTION.to_string()
            ]));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::services::{MemoryEntry, MemoryService};
    use adk_agents::session::Session;
    use adk_genai::content::{Content, Part};
    use std::sync::Arc as StdArc;

    struct StubMemoryService {
        memories: Vec<MemoryEntry>,
    }

    impl MemoryService for StubMemoryService {
        fn add_session_to_memory(&self, _session: &Session) {}
        fn add_events_to_memory(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _events: &[adk_events::Event],
            _custom_metadata: Option<&BTreeMap<String, Value>>,
        ) {
        }
        fn add_memory(
            &self,
            _app_name: &str,
            _user_id: &str,
            _memories: &[MemoryEntry],
            _custom_metadata: Option<&BTreeMap<String, Value>>,
        ) {
        }
        fn search_memory(
            &self,
            _app_name: &str,
            _user_id: &str,
            _query: &str,
        ) -> SearchMemoryResponse {
            SearchMemoryResponse {
                memories: self.memories.clone(),
            }
        }
    }

    fn ctx_with_memory(memories: Vec<MemoryEntry>) -> Context {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.memory_service = Some(StdArc::new(StubMemoryService { memories }));
        Context::new(ic)
    }

    fn ctx_without_memory_service() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn load_memory_returns_the_matched_memories() {
        let memory = MemoryEntry {
            content: Content::new("user", vec![Part::text("past conversation")]),
            custom_metadata: Default::default(),
            id: None,
            author: Some("user".to_string()),
            timestamp: None,
        };
        let mut ctx = ctx_with_memory(vec![memory]);
        let mut args = BTreeMap::new();
        args.insert("query".to_string(), Value::String("q".to_string()));

        let result = load_memory(&args, &mut ctx).await;
        match result {
            Value::Map(fields) => {
                let memories = fields.iter().find(|(k, _)| k == "memories").unwrap();
                match &memories.1 {
                    Value::Seq(items) => assert_eq!(items.len(), 1),
                    other => panic!("expected a seq, got {other:?}"),
                }
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_memory_reports_a_missing_memory_service_as_an_error_value() {
        let mut ctx = ctx_without_memory_service();
        let mut args = BTreeMap::new();
        args.insert("query".to_string(), Value::String("q".to_string()));

        let result = load_memory(&args, &mut ctx).await;
        match result {
            Value::Map(fields) => assert!(fields.iter().any(|(k, _)| k == "error")),
            other => panic!("expected an error map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn process_llm_request_appends_the_memory_instruction_and_declaration() {
        let tool = LoadMemoryTool::new();
        let mut ctx = ctx_without_memory_service();
        let mut request = LlmRequest::new("gemini-2.5-flash");

        tool.process_llm_request(&mut ctx, &mut request).await;

        let system_instruction = request.config.system_instruction.unwrap();
        assert!(system_instruction.contains("You have memory"));
        assert!(request.config.tools.is_some());
    }
}
