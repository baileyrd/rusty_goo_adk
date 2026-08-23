//! Part of capability C0125 (the SSE-streaming half): `StreamingResponseAggregator`,
//! ported from `google.adk.utils.streaming_utils`.
//!
//! **Scope, disclosed**: the source has two aggregation modes, switched on
//! a feature flag (`FeatureName.PROGRESSIVE_SSE_STREAMING`): a newer
//! "progressive" mode that preserves part ordering and streams function-call
//! arguments incrementally via JSONPath-addressed partial args, and the
//! older "non-progressive" mode this module ports — text-only accumulation,
//! no partial function-call argument streaming. This workspace has adopted
//! no feature-flag registry (Phase 12's `features/` isn't built) and no
//! typed function-call/tool machinery to stream partial arguments into
//! (`config.tools`/`FunctionDeclaration` stay opaque, C0116, Phase 8's
//! `BaseTool`) — so the progressive branch has nothing to be built *on top
//! of* yet, and only the non-progressive/legacy behavior is ported here.
//! Every source code comment and branch below refers to that legacy path
//! unless noted otherwise.
//!
//! **Adaptation**: the source is an `async def process_response(...) ->
//! AsyncGenerator[LlmResponse, None]`; [`StreamingResponseAggregator::process_response`]
//! returns a plain `Vec<LlmResponse>` (0-2 entries, in yield order) instead —
//! this workspace's `BaseLlm::generate_content_async` already collects a
//! whole call's responses into one `Vec` rather than a true async stream
//! (see `base_llm.rs`'s module doc), so there's no incremental consumer for
//! a real generator to feed anyway.

use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;

use crate::generate_content_response::{value_to_string, GenerateContentResponse};
use crate::llm_response::LlmResponse;

fn is_stop(finish_reason: &Value) -> bool {
    matches!(finish_reason, Value::String(s) if s == "STOP")
}

fn has_inline_data(content: &Content) -> bool {
    content
        .parts
        .first()
        .map(|part| part.inline_data.is_some())
        .unwrap_or(false)
}

/// Aggregates partial streaming responses: yields an `LlmResponse` for each
/// individual (partial) chunk, plus one aggregated response covering the
/// whole call once the stream ends ([`StreamingResponseAggregator::close`]).
#[derive(Debug, Default)]
pub struct StreamingResponseAggregator {
    text: Vec<String>,
    thought_text: Vec<String>,
    usage_metadata: Option<Value>,
    grounding_metadata: Option<Value>,
    citation_metadata: Option<Value>,
    finish_reason: Option<Value>,
    response: Option<GenerateContentResponse>,
}

impl StreamingResponseAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes a single streamed chunk. Returns, in order: a flushed
    /// merged-text `LlmResponse` if buffered text needed to be flushed
    /// before this chunk, followed always by this chunk's own translated
    /// (and possibly `partial = true`-marked) `LlmResponse`.
    pub fn process_response(&mut self, response: GenerateContentResponse) -> Vec<LlmResponse> {
        self.response = Some(response.clone());
        let mut llm_response = LlmResponse::create(response);

        // Usage/grounding/citation are typically reported on a single
        // chunk; keep the last reported value rather than letting a
        // metadata-less trailing chunk erase it.
        if llm_response.usage_metadata.is_some() {
            self.usage_metadata = llm_response.usage_metadata.clone();
        }
        if llm_response.grounding_metadata.is_some() {
            self.grounding_metadata = llm_response.grounding_metadata.clone();
        }
        if llm_response.citation_metadata.is_some() {
            self.citation_metadata = llm_response.citation_metadata.clone();
        }
        if llm_response.finish_reason.is_some() {
            self.finish_reason = llm_response.finish_reason.clone();
        }

        let mut out = Vec::new();
        let first_part = llm_response
            .content
            .as_ref()
            .and_then(|content| content.parts.first());

        // Matches the source's `if ... parts[0].text:` — Python treats an
        // empty string as falsy, so an empty-but-present `text` field
        // falls through to the flush/passthrough branch below rather than
        // being treated as "this chunk carries text".
        let first_part_text = first_part
            .and_then(|part| part.text.as_deref())
            .filter(|text| !text.is_empty())
            .map(str::to_string);

        if let Some(text) = first_part_text {
            if first_part.and_then(|part| part.thought) == Some(true) {
                self.thought_text.push(text);
            } else {
                self.text.push(text);
            }
            llm_response.partial = Some(true);
        } else if !self.thought_text.is_empty() || !self.text.is_empty() {
            let should_flush = llm_response
                .content
                .as_ref()
                .map(|content| content.parts.is_empty() || !has_inline_data(content))
                .unwrap_or(true);
            if should_flush {
                out.push(self.flush_merged_text(&llm_response));
            }
        }

        out.push(llm_response);
        out
    }

    fn flush_merged_text(&mut self, llm_response: &LlmResponse) -> LlmResponse {
        let parts = self.take_buffered_parts();
        let flushed = LlmResponse {
            content: Some(Content::new("model", parts)),
            usage_metadata: llm_response.usage_metadata.clone(),
            grounding_metadata: llm_response.grounding_metadata.clone(),
            citation_metadata: llm_response.citation_metadata.clone(),
            finish_reason: llm_response.finish_reason.clone(),
            model_version: llm_response.model_version.clone(),
            ..Default::default()
        };
        self.thought_text.clear();
        self.text.clear();
        flushed
    }

    fn take_buffered_parts(&self) -> Vec<Part> {
        let mut parts = Vec::new();
        if !self.thought_text.is_empty() {
            parts.push(Part {
                text: Some(self.thought_text.join("")),
                thought: Some(true),
                ..Default::default()
            });
        }
        if !self.text.is_empty() {
            parts.push(Part::text(self.text.join("")));
        }
        parts
    }

    /// Produces the final aggregated `LlmResponse` covering the whole call,
    /// once every chunk has been processed. `None` if no chunk was ever
    /// processed.
    pub fn close(&mut self) -> Option<LlmResponse> {
        let response = self.response.clone()?;
        let candidate = response.candidates.as_ref().and_then(|c| c.first());

        let finish_reason = self
            .finish_reason
            .clone()
            .or_else(|| candidate.and_then(|c| c.finish_reason.clone()));

        let mut error_code = None;
        let mut error_message = None;
        match &finish_reason {
            Some(reason) if !is_stop(reason) => {
                error_code = value_to_string(reason);
                error_message = candidate.and_then(|c| c.finish_message.clone());
            }
            _ if candidate.is_none() => {
                if let Some(feedback) = &response.prompt_feedback {
                    error_code = feedback.block_reason.as_ref().and_then(value_to_string);
                    error_message = feedback.block_reason_message.clone();
                }
            }
            _ => {}
        }

        let parts = self.take_buffered_parts();
        let content = if parts.is_empty() {
            None
        } else {
            Some(Content::new("model", parts))
        };

        Some(LlmResponse {
            content,
            grounding_metadata: self.grounding_metadata.clone(),
            citation_metadata: self.citation_metadata.clone(),
            error_code,
            error_message,
            usage_metadata: self.usage_metadata.clone(),
            finish_reason,
            partial: Some(false),
            model_version: response.model_version,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_content_response::{Candidate, PromptFeedback};

    fn text_chunk(text: &str) -> GenerateContentResponse {
        GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![Part::text(text)])),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn stop_chunk() -> GenerateContentResponse {
        GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![])),
                finish_reason: Some(Value::String("STOP".to_string())),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    #[test]
    fn text_chunks_are_marked_partial_and_accumulated() {
        let mut aggregator = StreamingResponseAggregator::new();
        let out = aggregator.process_response(text_chunk("Hello"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].partial, Some(true));
        assert_eq!(
            out[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("Hello")
        );
    }

    #[test]
    fn a_terminal_stop_chunk_flushes_the_buffered_text_before_itself() {
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("Hello, "));
        aggregator.process_response(text_chunk("world!"));
        let out = aggregator.process_response(stop_chunk());

        assert_eq!(
            out.len(),
            2,
            "expected [flushed merged text, this chunk's own response]"
        );
        let flushed = &out[0];
        assert_eq!(
            flushed.content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("Hello, world!")
        );
        assert_eq!(
            flushed.partial, None,
            "the flushed merged-text event isn't itself partial"
        );
        assert_eq!(
            out[1].partial, None,
            "the stop chunk carries no text, so partial stays unset"
        );
    }

    #[test]
    fn an_empty_string_text_part_is_treated_as_no_text_matching_pythons_truthiness() {
        // The source's `if ... parts[0].text:` treats an empty string as
        // falsy — a STOP chunk carrying `text=""` (rather than an empty
        // `parts` list) must still flush buffered text, not be mistaken
        // for "this chunk has real text to accumulate".
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("Hello"));

        let empty_text_stop = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![Part::text("")])),
                finish_reason: Some(Value::String("STOP".to_string())),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let out = aggregator.process_response(empty_text_stop);
        assert_eq!(
            out.len(),
            2,
            "expected [flushed merged text, this chunk's own response]"
        );
        assert_eq!(
            out[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("Hello")
        );
        assert_eq!(out[1].partial, None);
    }

    #[test]
    fn thought_and_regular_text_are_tracked_separately_and_merged_on_flush() {
        let mut aggregator = StreamingResponseAggregator::new();
        let thought = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new(
                    "model",
                    vec![Part {
                        text: Some("thinking...".to_string()),
                        thought: Some(true),
                        ..Default::default()
                    }],
                )),
                ..Default::default()
            }]),
            ..Default::default()
        };
        aggregator.process_response(thought);
        aggregator.process_response(text_chunk("the answer"));
        let out = aggregator.process_response(stop_chunk());

        let flushed_parts = &out[0].content.as_ref().unwrap().parts;
        assert_eq!(flushed_parts.len(), 2);
        assert_eq!(flushed_parts[0].thought, Some(true));
        assert_eq!(flushed_parts[0].text.as_deref(), Some("thinking..."));
        assert_eq!(flushed_parts[1].thought, None);
        assert_eq!(flushed_parts[1].text.as_deref(), Some("the answer"));
    }

    #[test]
    fn an_inline_data_chunk_does_not_flush_buffered_text() {
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("partial caption"));

        let audio_chunk = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new(
                    "model",
                    vec![Part {
                        inline_data: Some(adk_genai::content::MediaBlobStub {
                            mime_type: Some("audio/pcm".to_string()),
                            rest: None,
                        }),
                        ..Default::default()
                    }],
                )),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let out = aggregator.process_response(audio_chunk);
        assert_eq!(
            out.len(),
            1,
            "an inline_data chunk must not trigger a flush of buffered text"
        );
    }

    #[test]
    fn close_returns_none_before_any_chunk_is_processed() {
        let mut aggregator = StreamingResponseAggregator::new();
        assert!(aggregator.close().is_none());
    }

    #[test]
    fn close_aggregates_the_final_response_as_non_partial() {
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("Hello, "));
        aggregator.process_response(text_chunk("world!"));
        aggregator.process_response(stop_chunk());

        let closed = aggregator.close().unwrap();
        assert_eq!(closed.partial, Some(false));
        // The buffer was already flushed by the STOP chunk above, so
        // close() has nothing left to merge — matches the source (once
        // `_text`/`_thought_text` are flushed mid-stream, `close()` only
        // rebuilds error/finish-reason bookkeeping, not the text again).
        assert!(closed.content.is_none());
        assert_eq!(
            closed.finish_reason,
            Some(Value::String("STOP".to_string()))
        );
    }

    #[test]
    fn close_surfaces_a_non_stop_finish_reason_as_an_error() {
        let mut aggregator = StreamingResponseAggregator::new();
        let safety_chunk = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: None,
                finish_reason: Some(Value::String("SAFETY".to_string())),
                finish_message: Some("blocked".to_string()),
                ..Default::default()
            }]),
            ..Default::default()
        };
        aggregator.process_response(safety_chunk);
        let closed = aggregator.close().unwrap();
        assert_eq!(closed.error_code.as_deref(), Some("SAFETY"));
        assert_eq!(closed.error_message.as_deref(), Some("blocked"));
    }

    #[test]
    fn close_surfaces_prompt_feedback_when_there_is_no_candidate() {
        let mut aggregator = StreamingResponseAggregator::new();
        let blocked = GenerateContentResponse {
            candidates: None,
            prompt_feedback: Some(PromptFeedback {
                block_reason: Some(Value::String("OTHER".to_string())),
                block_reason_message: Some("filtered".to_string()),
            }),
            ..Default::default()
        };
        aggregator.process_response(blocked);
        let closed = aggregator.close().unwrap();
        assert_eq!(closed.error_code.as_deref(), Some("OTHER"));
        assert_eq!(closed.error_message.as_deref(), Some("filtered"));
    }

    #[test]
    fn usage_metadata_survives_a_trailing_chunk_with_none() {
        let mut aggregator = StreamingResponseAggregator::new();
        let with_usage = GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![Part::text("hi")])),
                ..Default::default()
            }]),
            usage_metadata: Some(Value::String("5 tokens".to_string())),
            ..Default::default()
        };
        aggregator.process_response(with_usage);
        aggregator.process_response(stop_chunk());
        let closed = aggregator.close().unwrap();
        assert_eq!(
            closed.usage_metadata,
            Some(Value::String("5 tokens".to_string()))
        );
    }
}
