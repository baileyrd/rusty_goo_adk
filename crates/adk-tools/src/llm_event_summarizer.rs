//! Capabilities C0286/C0287: `LlmEventSummarizer`, ported from
//! `apps/llm_event_summarizer.py`.
//!
//! **Placement, disclosed**: the trait this implements
//! ([`adk_agents::app_configs::BaseEventsSummarizer`], C0285 DONE)
//! lives in `adk-agents`. But this summarizer also needs a real
//! [`adk_models::base_llm::BaseLlm`] — and `adk-models` already depends
//! on `adk-agents` (not the reverse), so `adk-agents` can't add
//! `adk-models` as a dependency without a crate-graph cycle. `adk-tools`
//! already depends on both `adk-agents` and `adk-models` (plus
//! `adk-events`), so it's the natural host — the same "a supporting
//! crate assembles the trait-object implementor" shape
//! `forwarding_artifact_service.rs` (C0489) already used for `AgentTool`.
//!
//! **Adaptation, disclosed**: the source formats a tool call's `args`
//! and a tool response's `response` with Python's `str()`. This port
//! uses `rusty_serde::json::to_string` instead — the same disclosed
//! compact-JSON-instead-of-`str()` divergence `adk-events::debug_output`
//! (C0933) already established for the same two fields, falling back to
//! `"<unserializable>"` if a value somehow doesn't serialize. Truncation
//! is likewise byte-based with char-boundary safety (matching
//! `debug_output::truncate`'s own disclosed divergence from Python's
//! character-based `text[:limit]` slicing) rather than reused directly,
//! since `debug_output`'s helpers are private to their own module.
//!
//! **Not ported**: the *decision* of when to compact and which events
//! form the sliding window — the source's own docs, and
//! `EventsCompactionConfig`'s own module doc, describe that as the
//! responsibility of an external component (a `Runner`); this type only
//! summarizes whatever events it's handed.

use std::sync::Arc;

use adk_agents::app_configs::BaseEventsSummarizer;
use adk_agents::services::BoxFuture;
use adk_events::event_compaction::EventCompaction;
use adk_events::node_info::NodeInfo;
use adk_events::{Event, EventActions};
use adk_genai::content::{Content, Part};
use adk_models::base_llm::BaseLlm;
use adk_models::llm_request::LlmRequest;

const DEFAULT_PROMPT_TEMPLATE: &str = "The following is a conversation history between a user \
     and an AI agent. It may or may not start from a compacted history. Please identify and \
     reiterate the user request, summarize the context so far, focusing on key decisions made \
     and information obtained, as well as any unresolved questions or tasks. CRITICAL \
     INSTRUCTIONS: 1. Explicitly identify and state the primary language used by the user at \
     the top of your summary (e.g., \"Conversation Language: English\"). 2. If the agent \
     called any tools, accurately list the exact tool names used to maintain tool grounding. \
     The rest of the summary should be concise and capture the essence of the \
     interaction.\n\n{conversation_history}";

/// Tool call args and responses can be large (e.g. search results). Cap
/// how much of each is rendered so compaction does not inflate the very
/// context it exists to shrink.
const MAX_TOOL_CONTENT_CHARS: usize = 2000;

fn truncate(text: &str) -> String {
    if text.len() <= MAX_TOOL_CONTENT_CHARS {
        return text.to_string();
    }
    let mut end = MAX_TOOL_CONTENT_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... [truncated {} chars]", &text[..end], text.len() - end)
}

fn display(value: &std::collections::BTreeMap<String, rusty_serde::value::Value>) -> String {
    rusty_serde::json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

/// C0286/C0287: an LLM-based event summarizer for sliding-window
/// compaction — formats events, generates a summary via `llm`, and
/// returns a new compacted `Event`.
pub struct LlmEventSummarizer {
    llm: Arc<dyn BaseLlm + Send + Sync>,
    prompt_template: String,
}

impl LlmEventSummarizer {
    pub fn new(llm: Arc<dyn BaseLlm + Send + Sync>) -> Self {
        Self {
            llm,
            prompt_template: DEFAULT_PROMPT_TEMPLATE.to_string(),
        }
    }

    /// Overrides the default prompt template. Must contain a
    /// `{conversation_history}` placeholder.
    pub fn with_prompt_template(mut self, prompt_template: impl Into<String>) -> Self {
        self.prompt_template = prompt_template.into();
        self
    }

    /// Formats `events` into prompt text, including thoughts and tool
    /// calls. Thoughts carry the agent's analysis of tool responses, and
    /// tool calls/responses carry the evidence retrieved so far, so all
    /// three are included. Thoughts emitted by a compaction event are
    /// skipped so a prior summary's reasoning doesn't leak into the next
    /// summary.
    fn format_events_for_prompt(events: &[Event]) -> String {
        let mut lines = Vec::new();
        for event in events {
            let Some(content) = &event.content else {
                continue;
            };
            let is_compaction = event.actions.compaction.is_some();
            for part in &content.parts {
                match (part.thought, &part.text) {
                    (Some(true), Some(text)) => {
                        if !is_compaction {
                            lines.push(format!("{} (thought): {text}", event.author));
                        }
                    }
                    (_, Some(text)) => lines.push(format!("{}: {text}", event.author)),
                    _ => {}
                }
                if let Some(function_call) = &part.function_call {
                    let args = function_call.args.as_ref().map(display).unwrap_or_default();
                    lines.push(format!(
                        "{} called tool: {}({})",
                        event.author,
                        function_call.name.as_deref().unwrap_or_default(),
                        truncate(&args)
                    ));
                }
                if let Some(function_response) = &part.function_response {
                    let response = function_response
                        .response
                        .as_ref()
                        .map(display)
                        .unwrap_or_default();
                    lines.push(format!(
                        "Tool response from {}: {}",
                        function_response.name.as_deref().unwrap_or_default(),
                        truncate(&response)
                    ));
                }
            }
        }
        lines.join("\n")
    }
}

impl BaseEventsSummarizer for LlmEventSummarizer {
    fn maybe_summarize_events<'a>(&'a self, events: &'a [Event]) -> BoxFuture<'a, Option<Event>> {
        Box::pin(async move {
            if events.is_empty() {
                return None;
            }

            let conversation_history = Self::format_events_for_prompt(events);
            let prompt = self
                .prompt_template
                .replace("{conversation_history}", &conversation_history);

            let mut llm_request = LlmRequest::new(self.llm.model());
            llm_request.contents = vec![Content::new("user", vec![Part::text(prompt)])];

            let responses = self
                .llm
                .generate_content_async(&llm_request, false)
                .await
                .ok()?;

            let mut summary_content = None;
            let mut summary_usage_metadata = None;
            for response in responses {
                if response.content.is_some() {
                    summary_content = response.content;
                    summary_usage_metadata = response.usage_metadata;
                    break;
                }
            }
            let mut summary_content = summary_content?;
            summary_content.role = Some("model".to_string());

            let start_timestamp = events[0].timestamp;
            let end_timestamp = events[events.len() - 1].timestamp;

            let actions = EventActions {
                compaction: Some(EventCompaction {
                    start_timestamp,
                    end_timestamp,
                    compacted_content: summary_content,
                }),
                ..Default::default()
            };

            let mut event = Event::new(Event::new_id(), "user", NodeInfo::new(""));
            event.actions = actions;
            event.usage_metadata = summary_usage_metadata;
            Some(event)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::{FunctionCall, FunctionResponse};
    use adk_models::base_llm::BaseLlmError;
    use adk_models::llm_response::LlmResponse;
    use std::sync::Mutex;

    struct StubLlm {
        responses: Mutex<Vec<Result<Vec<LlmResponse>, BaseLlmError>>>,
        last_request: Mutex<Option<LlmRequest>>,
    }

    impl StubLlm {
        fn returning(responses: Vec<Result<Vec<LlmResponse>, BaseLlmError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                last_request: Mutex::new(None),
            }
        }
    }

    impl BaseLlm for StubLlm {
        fn model(&self) -> &str {
            "stub-model"
        }

        fn type_name(&self) -> &'static str {
            "StubLlm"
        }

        fn generate_content_async<'a>(
            &'a self,
            llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<Vec<LlmResponse>, BaseLlmError>>
                    + Send
                    + 'a,
            >,
        > {
            *self.last_request.lock().unwrap() = Some(llm_request.clone());
            let response = self.responses.lock().unwrap().remove(0);
            Box::pin(async move { response })
        }
    }

    fn events_with_content(contents: Vec<Content>) -> Vec<Event> {
        contents
            .into_iter()
            .map(|content| {
                let mut event = Event::new("inv-1", "user", NodeInfo::new(""));
                event.content = Some(content);
                event
            })
            .collect()
    }

    #[rusty_tokio::test]
    async fn returns_none_for_no_events() {
        let llm = Arc::new(StubLlm::returning(vec![]));
        let summarizer = LlmEventSummarizer::new(llm);
        assert!(summarizer.maybe_summarize_events(&[]).await.is_none());
    }

    #[rusty_tokio::test]
    async fn returns_none_when_the_llm_call_fails() {
        let llm = Arc::new(StubLlm::returning(vec![Err(
            BaseLlmError::GenerationNotSupported("stub-model".to_string()),
        )]));
        let summarizer = LlmEventSummarizer::new(llm);
        let events = events_with_content(vec![Content::user_text("hi")]);
        assert!(summarizer.maybe_summarize_events(&events).await.is_none());
    }

    #[rusty_tokio::test]
    async fn returns_none_when_no_response_carries_content() {
        let llm = Arc::new(StubLlm::returning(vec![Ok(vec![LlmResponse::default()])]));
        let summarizer = LlmEventSummarizer::new(llm);
        let events = events_with_content(vec![Content::user_text("hi")]);
        assert!(summarizer.maybe_summarize_events(&events).await.is_none());
    }

    #[rusty_tokio::test]
    async fn builds_a_compaction_event_from_the_summary_response() {
        let llm = Arc::new(StubLlm::returning(vec![Ok(vec![LlmResponse {
            content: Some(Content::new("user", vec![Part::text("the summary")])),
            usage_metadata: Some(rusty_serde::value::Value::String("42 tokens".to_string())),
            ..Default::default()
        }])]));
        let summarizer = LlmEventSummarizer::new(llm);

        let mut first = Event::new("inv-1", "user", NodeInfo::new(""));
        first.content = Some(Content::user_text("hi"));
        first.timestamp = 1.0;
        let mut last = Event::new("inv-1", "model", NodeInfo::new(""));
        last.content = Some(Content::new("model", vec![Part::text("hello")]));
        last.timestamp = 2.0;

        let summary = summarizer
            .maybe_summarize_events(&[first, last])
            .await
            .unwrap();

        assert_eq!(summary.author, "user");
        assert_eq!(
            summary.usage_metadata,
            Some(rusty_serde::value::Value::String("42 tokens".to_string()))
        );
        let compaction = summary.actions.compaction.unwrap();
        assert_eq!(compaction.start_timestamp, 1.0);
        assert_eq!(compaction.end_timestamp, 2.0);
        assert_eq!(compaction.compacted_content.role.as_deref(), Some("model"));
        assert_eq!(
            compaction.compacted_content.parts[0].text.as_deref(),
            Some("the summary")
        );
    }

    #[rusty_tokio::test]
    async fn forces_the_summary_contents_role_to_model_even_if_the_llm_says_otherwise() {
        let llm = Arc::new(StubLlm::returning(vec![Ok(vec![LlmResponse {
            content: Some(Content::new("assistant", vec![Part::text("summary")])),
            ..Default::default()
        }])]));
        let summarizer = LlmEventSummarizer::new(llm);
        let events = events_with_content(vec![Content::user_text("hi")]);

        let summary = summarizer.maybe_summarize_events(&events).await.unwrap();
        assert_eq!(
            summary
                .actions
                .compaction
                .unwrap()
                .compacted_content
                .role
                .as_deref(),
            Some("model")
        );
    }

    #[test]
    fn format_events_for_prompt_includes_text_thoughts_and_tool_activity() {
        let mut text_event = Event::new("inv-1", "agent", NodeInfo::new(""));
        text_event.content = Some(Content::new("model", vec![Part::text("hello there")]));

        let mut thought_event = Event::new("inv-1", "agent", NodeInfo::new(""));
        thought_event.content = Some(Content::new(
            "model",
            vec![Part {
                text: Some("thinking...".to_string()),
                thought: Some(true),
                ..Default::default()
            }],
        ));

        let mut call_event = Event::new("inv-1", "agent", NodeInfo::new(""));
        call_event.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));

        let mut response_event = Event::new("inv-1", "agent", NodeInfo::new(""));
        response_event.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        ));

        let formatted = LlmEventSummarizer::format_events_for_prompt(&[
            text_event,
            thought_event,
            call_event,
            response_event,
        ]);

        assert!(formatted.contains("agent: hello there"));
        assert!(formatted.contains("agent (thought): thinking..."));
        assert!(formatted.contains("agent called tool: get_weather"));
        assert!(formatted.contains("Tool response from get_weather:"));
    }

    #[test]
    fn format_events_for_prompt_skips_thoughts_on_a_compaction_event() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new(""));
        event.content = Some(Content::new(
            "model",
            vec![Part {
                text: Some("a prior summary's reasoning".to_string()),
                thought: Some(true),
                ..Default::default()
            }],
        ));
        event.actions = EventActions {
            compaction: Some(EventCompaction {
                start_timestamp: 0.0,
                end_timestamp: 1.0,
                compacted_content: Content::user_text("old summary"),
            }),
            ..Default::default()
        };

        let formatted = LlmEventSummarizer::format_events_for_prompt(&[event]);
        assert!(!formatted.contains("a prior summary's reasoning"));
    }

    #[test]
    fn format_events_for_prompt_skips_events_with_no_content() {
        let event = Event::new("inv-1", "agent", NodeInfo::new(""));
        assert_eq!(LlmEventSummarizer::format_events_for_prompt(&[event]), "");
    }

    #[test]
    fn truncate_leaves_short_text_untouched() {
        assert_eq!(truncate("short"), "short");
    }

    #[test]
    fn truncate_caps_long_text_with_a_marker() {
        let long = "a".repeat(MAX_TOOL_CONTENT_CHARS + 100);
        let truncated = truncate(&long);
        assert!(truncated.starts_with(&"a".repeat(MAX_TOOL_CONTENT_CHARS)));
        assert!(truncated.contains("[truncated 100 chars]"));
    }
}
