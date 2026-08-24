//! Capability C0424: `PreloadMemoryTool`, ported from
//! `google.adk.tools.preload_memory_tool`.
//!
//! **Adaptation**: `tool_context.user_content` reads this port's opaque
//! `InvocationContext.user_content: Option<Value>` placeholder, parsed
//! back into a typed `Content` via its own `Deserialize` impl — the same
//! pattern `ExampleTool` (C0419) already uses for the same field.
//!
//! **Not** ported: the source's `logging.warning` on a failed
//! `search_memory` call — no logging framework has been adopted by this
//! workspace yet (the same disclosed omission as `contents.rs`'s
//! `drop_orphaned_function_responses`). A failed lookup is simply treated
//! as "nothing to preload," matching the source's own control flow (it
//! also swallows the exception and returns without injecting anything).
//!
//! Currently this tool only uses the text part from each memory entry
//! (matches the source's own documented limitation).

use adk_genai::content::{Content, Part};
use adk_models::llm_request::LlmRequest;

use crate::base_tool::{BaseTool, BoxFuture};
use crate::memory_entry_utils::extract_text;
use crate::tool_context::ToolContext;

const PRELOAD_MEMORY_PREAMBLE: &str = "The following content is from your previous conversations with the user.\nThey may be useful for answering the user's current query.\n<PAST_CONVERSATIONS>\n";
const PRELOAD_MEMORY_SUFFIX: &str = "\n</PAST_CONVERSATIONS>\n";

/// C0424: preloads the memory for the current user. Automatically
/// executed for each `llm_request` — never called by the model itself.
pub struct PreloadMemoryTool;

impl PreloadMemoryTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PreloadMemoryTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for PreloadMemoryTool {
    fn name(&self) -> &str {
        "preload_memory"
    }

    fn description(&self) -> &str {
        "preload_memory"
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let Some(user_content_value) = tool_context.invocation_context().user_content.clone()
            else {
                return;
            };
            let Ok(user_content) = rusty_serde::json::from_value::<Content>(user_content_value)
            else {
                return;
            };
            let Some(user_query) = user_content
                .parts
                .first()
                .and_then(|part| part.text.clone())
                .filter(|text| !text.is_empty())
            else {
                return;
            };

            let Ok(response) = tool_context.search_memory(&user_query).await else {
                return;
            };
            if response.memories.is_empty() {
                return;
            }

            let mut lines: Vec<String> = Vec::new();
            for memory in &response.memories {
                if let Some(timestamp) = memory.timestamp.as_deref().filter(|t| !t.is_empty()) {
                    lines.push(format!("Time: {timestamp}"));
                }
                let memory_text = extract_text(memory);
                if !memory_text.is_empty() {
                    lines.push(match memory.author.as_deref().filter(|a| !a.is_empty()) {
                        Some(author) => format!("{author}: {memory_text}"),
                        None => memory_text,
                    });
                }
            }
            if lines.is_empty() {
                return;
            }

            let full_memory_text = lines.join("\n");
            let memory_context =
                format!("{PRELOAD_MEMORY_PREAMBLE}{full_memory_text}{PRELOAD_MEMORY_SUFFIX}");
            llm_request.insert_transient_user_content(vec![Content::new(
                "user",
                vec![Part::text(memory_context)],
            )]);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::services::{MemoryEntry, MemoryService, SearchMemoryResponse};
    use adk_agents::session::Session;
    use std::collections::BTreeMap;
    use std::sync::Arc;

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
            _custom_metadata: Option<&BTreeMap<String, rusty_serde::value::Value>>,
        ) {
        }
        fn add_memory(
            &self,
            _app_name: &str,
            _user_id: &str,
            _memories: &[MemoryEntry],
            _custom_metadata: Option<&BTreeMap<String, rusty_serde::value::Value>>,
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

    fn ctx_with(memories: Vec<MemoryEntry>, user_text: Option<&str>) -> Context {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.memory_service = Some(Arc::new(StubMemoryService { memories }));
        if let Some(text) = user_text {
            ic.user_content = Some(
                rusty_serde::json::to_value(&Content::new("user", vec![Part::text(text)])).unwrap(),
            );
        }
        Context::new(ic)
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_without_user_content() {
        let tool = PreloadMemoryTool::new();
        let mut ctx = ctx_with(vec![], None);
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut ctx, &mut request).await;
        assert!(request.contents.is_empty());
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_when_search_returns_no_memories() {
        let tool = PreloadMemoryTool::new();
        let mut ctx = ctx_with(vec![], Some("what did we discuss?"));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut ctx, &mut request).await;
        assert!(request.contents.is_empty());
    }

    #[rusty_tokio::test]
    async fn injects_transient_content_with_time_author_and_text() {
        let memory = MemoryEntry {
            content: Content::new("user", vec![Part::text("we talked about rust")]),
            custom_metadata: Default::default(),
            id: None,
            author: Some("alice".to_string()),
            timestamp: Some("2026-01-01T00:00:00Z".to_string()),
        };
        let tool = PreloadMemoryTool::new();
        let mut ctx = ctx_with(vec![memory], Some("what did we discuss?"));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut ctx, &mut request).await;

        assert_eq!(request.contents.len(), 1);
        let text = request.contents[0].parts[0].text.as_deref().unwrap();
        assert!(text.contains("<PAST_CONVERSATIONS>"));
        assert!(text.contains("Time: 2026-01-01T00:00:00Z"));
        assert!(text.contains("alice: we talked about rust"));
        assert!(text.contains("</PAST_CONVERSATIONS>"));
    }

    #[rusty_tokio::test]
    async fn omits_the_author_prefix_when_author_is_absent() {
        let memory = MemoryEntry {
            content: Content::new("user", vec![Part::text("just text")]),
            custom_metadata: Default::default(),
            id: None,
            author: None,
            timestamp: None,
        };
        let tool = PreloadMemoryTool::new();
        let mut ctx = ctx_with(vec![memory], Some("query"));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut ctx, &mut request).await;

        let text = request.contents[0].parts[0].text.as_deref().unwrap();
        assert!(text.contains("just text"));
        assert!(!text.contains("Time:"));
    }
}
