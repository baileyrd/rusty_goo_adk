//! Capability C0125: `StreamingResponseAggregator`, ported from
//! `google.adk.utils.streaming_utils`.
//!
//! **Two aggregation modes**, switched on a feature flag
//! (`FeatureName::ProgressiveSseStreaming`): a newer "progressive" mode
//! that preserves part ordering and streams function-call arguments
//! incrementally via JSONPath-addressed partial args, and an older
//! "non-progressive" mode — text-only accumulation, no partial
//! function-call argument streaming.
//!
//! **Stale-blocker correction**: an earlier version of this module doc
//! claimed the progressive branch was blocked on "no feature-flag registry
//! (Phase 12's `features/` isn't built)" and "no typed function-call/tool
//! machinery to stream partial arguments into ... (C0116, Phase 8's
//! `BaseTool`)". Both were false by the time this was revisited:
//! `adk-features` (a zero-dependency leaf crate, already a transitive
//! dependency of `adk-models` via `adk-agents`) has shipped
//! `FeatureName::ProgressiveSseStreaming`/`is_feature_enabled` for a while,
//! and progressive mode never actually touches `BaseTool` at all — the
//! source's own `_process_function_call_part` only reads/writes
//! [`adk_genai::content::FunctionCall`]'s own fields (`partial_args`,
//! `will_continue`, `id`, `name`) plus a synthesized id, none of which
//! involve a tool registry. The one genuine new dependency this needed —
//! `generate_client_function_call_id` — lives in `adk-flows`
//! (`functions_utils.rs`), which depends on `adk-models`, not the reverse;
//! the source dodges the identical circular import with a `from
//! ..flows.llm_flows.functions import generate_client_function_call_id`
//! *inside the function body*. This port can't do a lazy import, so
//! [`generate_client_function_call_id`] is reproduced locally here from
//! the one primitive it's actually built from (`adk_events::Event::new_id`,
//! already a transitive dependency via `adk-agents`) rather than pulled in
//! from `adk-flows` — same prefix (`"adk-"`), same shape, disclosed
//! duplication rather than a crate-cycle workaround.
//!
//! **Adaptation**: the source is an `async def process_response(...) ->
//! AsyncGenerator[LlmResponse, None]`; [`StreamingResponseAggregator::process_response`]
//! returns a plain `Vec<LlmResponse>` (0-1 entries in progressive mode,
//! 0-2 in non-progressive, in yield order) instead — this workspace's
//! `BaseLlm::generate_content_async` already collects a whole call's
//! responses into one `Vec` rather than a true async stream (see
//! `base_llm.rs`'s module doc), so there's no incremental consumer for a
//! real generator to feed anyway.

use std::collections::BTreeMap;

use adk_events::Event;
use adk_features::feature_registry::{is_feature_enabled, FeatureName};
use adk_genai::content::{Content, FunctionCall, Part, PartialArg};
use rusty_serde::value::Value;

use crate::generate_content_response::{value_to_string, GenerateContentResponse};
use crate::llm_response::LlmResponse;

/// Same synthetic-id shape as `adk-flows::functions_utils::AF_FUNCTION_CALL_ID_PREFIX`/
/// `generate_client_function_call_id` — see the module doc for why this is
/// a disclosed local duplicate rather than an `adk-flows` dependency.
const AF_FUNCTION_CALL_ID_PREFIX: &str = "adk-";

fn generate_client_function_call_id() -> String {
    format!("{AF_FUNCTION_CALL_ID_PREFIX}{}", Event::new_id())
}

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

fn is_truthy_str(value: Option<&str>) -> bool {
    value.is_some_and(|s| !s.is_empty())
}

/// Serializes tests across this crate that override
/// `FeatureName::ProgressiveSseStreaming` — `adk_features`'s override state
/// is process-global, so two tests concurrently overriding the same flag to
/// different values would race. Shared with `gemini.rs`'s own streaming
/// test, the same pattern `adk-tools::base_retrieval_tool`'s `TEST_LOCK`
/// already established for `FeatureName::JsonSchemaForFuncDecl`.
#[cfg(test)]
pub(crate) static PROGRESSIVE_SSE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Aggregates partial streaming responses: yields an `LlmResponse` for each
/// individual (partial) chunk, plus one aggregated response covering the
/// whole call once the stream ends ([`StreamingResponseAggregator::close`]).
#[derive(Debug)]
pub struct StreamingResponseAggregator {
    text: Vec<String>,
    thought_text: Vec<String>,
    usage_metadata: Option<Value>,
    grounding_metadata: Option<Value>,
    citation_metadata: Option<Value>,
    finish_reason: Option<Value>,
    response: Option<GenerateContentResponse>,

    // For progressive SSE streaming mode: accumulate parts in order.
    parts_sequence: Vec<Part>,
    current_text_buffer: Vec<String>,
    current_text_is_thought: Option<bool>,
    current_text_thought_signature: Option<Value>,

    // For streaming function-call arguments.
    current_fc_name: Option<String>,
    current_fc_args: Value,
    current_fc_id: Option<String>,
    current_thought_signature: Option<Value>,
}

impl Default for StreamingResponseAggregator {
    fn default() -> Self {
        Self {
            text: Vec::new(),
            thought_text: Vec::new(),
            usage_metadata: None,
            grounding_metadata: None,
            citation_metadata: None,
            finish_reason: None,
            response: None,
            parts_sequence: Vec::new(),
            current_text_buffer: Vec::new(),
            current_text_is_thought: None,
            current_text_thought_signature: None,
            current_fc_name: None,
            current_fc_args: Value::Map(Vec::new()),
            current_fc_id: None,
            current_thought_signature: None,
        }
    }
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

        if is_feature_enabled(FeatureName::ProgressiveSseStreaming) {
            return self.process_response_progressive(llm_response);
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

    // ===== Progressive SSE streaming mode =====

    /// The progressive-mode branch of [`Self::process_response`]: accumulates
    /// parts in order (only merging consecutive text parts of the same
    /// thought/non-thought type), marks every chunk `partial = true`, and
    /// yields exactly that one response — matches the source's own
    /// `yield llm_response; return`.
    fn process_response_progressive(&mut self, mut llm_response: LlmResponse) -> Vec<LlmResponse> {
        if let Some(content) = &llm_response.content {
            for part in &content.parts {
                if is_truthy_str(part.text.as_deref()) {
                    if !self.current_text_buffer.is_empty()
                        && part.thought != self.current_text_is_thought
                    {
                        self.flush_text_buffer_to_sequence();
                    }
                    if self.current_text_buffer.is_empty() {
                        self.current_text_is_thought = part.thought;
                    }
                    self.current_text_buffer.push(part.text.clone().unwrap());
                    if part.thought_signature.is_some()
                        && self.current_text_thought_signature.is_none()
                    {
                        self.current_text_thought_signature = part.thought_signature.clone();
                    }
                } else if part.function_call.is_some() {
                    self.process_function_call_part(part);
                } else {
                    self.flush_text_buffer_to_sequence();
                    self.parts_sequence.push(part.clone());
                }
            }
        }

        llm_response.partial = Some(true);
        vec![llm_response]
    }

    /// Flushes the buffered text run (if any) into [`Self::parts_sequence`]
    /// as one merged `Part`, carrying over the first thought-signature seen
    /// on the run.
    fn flush_text_buffer_to_sequence(&mut self) {
        if self.current_text_buffer.is_empty() {
            return;
        }
        let buffered_text = self.current_text_buffer.join("");
        let mut merged_part = if self.current_text_is_thought == Some(true) {
            Part {
                text: Some(buffered_text),
                thought: Some(true),
                ..Default::default()
            }
        } else {
            Part::text(buffered_text)
        };
        if self.current_text_thought_signature.is_some() {
            merged_part.thought_signature = self.current_text_thought_signature.clone();
        }
        self.parts_sequence.push(merged_part);
        self.current_text_buffer.clear();
        self.current_text_is_thought = None;
        self.current_text_thought_signature = None;
    }

    /// Extracts the value carried by one `PartialArg` chunk. `None` means
    /// the chunk carried no value at all (none of `string_value`/
    /// `number_value`/`bool_value`/`null_value` set) — distinct from
    /// `Some(Value::Null)`, which means the chunk explicitly set `null`.
    fn value_from_partial_arg(&self, partial_arg: &PartialArg, json_path: &str) -> Option<Value> {
        if let Some(string_chunk) = &partial_arg.string_value {
            let path_without_prefix = json_path.strip_prefix("$.").unwrap_or(json_path);
            let mut existing_value = &self.current_fc_args;
            for part in path_without_prefix.split('.') {
                match existing_value {
                    Value::Map(entries) => match entries.iter().find(|(k, _)| k == part) {
                        Some((_, value)) => existing_value = value,
                        None => break,
                    },
                    _ => break,
                }
            }
            let value = match existing_value {
                Value::String(existing) => format!("{existing}{string_chunk}"),
                _ => string_chunk.clone(),
            };
            Some(Value::String(value))
        } else if let Some(number_value) = partial_arg.number_value {
            Some(Value::Float(number_value))
        } else if let Some(bool_value) = partial_arg.bool_value {
            Some(Value::Bool(bool_value))
        } else if partial_arg.null_value.is_some() {
            Some(Value::Null)
        } else {
            None
        }
    }

    /// Sets a value in [`Self::current_fc_args`] using JSONPath notation
    /// (`"$.location"`/`"$.location.latitude"`), creating intermediate
    /// map levels as needed.
    fn set_value_by_json_path(&mut self, json_path: &str, value: Value) {
        let path = json_path.strip_prefix("$.").unwrap_or(json_path);
        let parts: Vec<&str> = path.split('.').collect();
        let Some((last, ancestors)) = parts.split_last() else {
            return;
        };

        let mut current = &mut self.current_fc_args;
        for part in ancestors {
            if !matches!(current, Value::Map(_)) {
                *current = Value::Map(Vec::new());
            }
            let Value::Map(entries) = current else {
                unreachable!()
            };
            if !entries.iter().any(|(k, _)| k == part) {
                entries.push(((*part).to_string(), Value::Map(Vec::new())));
            }
            let index = entries.iter().position(|(k, _)| k == part).unwrap();
            current = &mut entries[index].1;
        }

        if !matches!(current, Value::Map(_)) {
            *current = Value::Map(Vec::new());
        }
        let Value::Map(entries) = current else {
            unreachable!()
        };
        match entries.iter().position(|(k, _)| k == last) {
            Some(index) => entries[index].1 = value,
            None => entries.push(((*last).to_string(), value)),
        }
    }

    /// Flushes the accumulated function-call name/args/id/thought-signature
    /// into [`Self::parts_sequence`] as one complete `FunctionCall` part.
    fn flush_function_call_to_sequence(&mut self) {
        let Some(name) = self.current_fc_name.take().filter(|n| !n.is_empty()) else {
            return;
        };
        let args = match std::mem::replace(&mut self.current_fc_args, Value::Map(Vec::new())) {
            Value::Map(entries) => Some(entries.into_iter().collect::<BTreeMap<_, _>>()),
            _ => None,
        };
        let mut function_call = FunctionCall {
            name: Some(name),
            args,
            ..Default::default()
        };
        if let Some(id) = self.current_fc_id.take().filter(|i| !i.is_empty()) {
            function_call.id = Some(id);
        }
        let mut fc_part = Part::function_call(function_call);
        if self.current_thought_signature.is_some() {
            fc_part.thought_signature = self.current_thought_signature.take();
        }
        self.parts_sequence.push(fc_part);
    }

    /// Processes one streaming function-call chunk: records `name`/`id` on
    /// first sight, applies every `partial_args` entry via JSONPath, and —
    /// once `will_continue` isn't `true` — flushes the buffered text and
    /// the completed function call.
    fn process_streaming_function_call(&mut self, fc: &FunctionCall) {
        if is_truthy_str(fc.name.as_deref()) {
            self.current_fc_name = fc.name.clone();
        }
        if is_truthy_str(fc.id.as_deref()) {
            self.current_fc_id = fc.id.clone();
        }

        for partial_arg in fc.partial_args.iter().flatten() {
            let Some(json_path) = partial_arg.json_path.as_deref().filter(|p| !p.is_empty()) else {
                continue;
            };
            if let Some(value) = self.value_from_partial_arg(partial_arg, json_path) {
                self.set_value_by_json_path(json_path, value);
            }
        }

        if fc.will_continue != Some(true) {
            self.flush_text_buffer_to_sequence();
            self.flush_function_call_to_sequence();
        }
    }

    /// Processes one function-call-bearing part, streaming or not. A part
    /// carrying `partial_args` or `will_continue` is a streaming chunk
    /// (routed through [`Self::process_streaming_function_call`]); anything
    /// else is a complete, standard-format function call, appended directly
    /// after flushing any buffered text.
    fn process_function_call_part(&mut self, part: &Part) {
        let Some(fc) = &part.function_call else {
            return;
        };
        let has_partial_args = fc
            .partial_args
            .as_ref()
            .is_some_and(|args| !args.is_empty());

        if has_partial_args || fc.will_continue == Some(true) {
            let mut fc = fc.clone();
            if !is_truthy_str(fc.id.as_deref()) && self.current_fc_id.is_none() {
                fc.id = Some(generate_client_function_call_id());
            }
            if part.thought_signature.is_some() && self.current_thought_signature.is_none() {
                self.current_thought_signature = part.thought_signature.clone();
            }
            self.process_streaming_function_call(&fc);
        } else if is_truthy_str(fc.name.as_deref()) {
            let mut part = part.clone();
            let needs_id = !part
                .function_call
                .as_ref()
                .is_some_and(|fc| is_truthy_str(fc.id.as_deref()));
            if needs_id {
                if let Some(fc) = part.function_call.as_mut() {
                    fc.id = Some(generate_client_function_call_id());
                }
            }
            self.flush_text_buffer_to_sequence();
            self.parts_sequence.push(part);
        }
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

        if is_feature_enabled(FeatureName::ProgressiveSseStreaming) {
            self.flush_text_buffer_to_sequence();
            self.flush_function_call_to_sequence();
            let parts = std::mem::take(&mut self.parts_sequence);
            let content = if parts.is_empty() {
                None
            } else {
                Some(Content::new("model", parts))
            };
            return Some(LlmResponse {
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
            });
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
    use adk_features::feature_registry::TemporaryFeatureOverride;

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

    // --- Non-progressive (legacy) mode — every test below forces the flag
    // off, since `ProgressiveSseStreaming` defaults on and these all assert
    // the legacy per-call multi-response shape a progressive-mode caller
    // never produces (see the progressive-mode tests further down). ---

    #[test]
    fn text_chunks_are_marked_partial_and_accumulated() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, false);
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

    // --- Progressive SSE streaming mode ---

    fn function_call_chunk(fc: FunctionCall) -> GenerateContentResponse {
        GenerateContentResponse {
            candidates: Some(vec![Candidate {
                content: Some(Content::new("model", vec![Part::function_call(fc)])),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn partial_arg_string(json_path: &str, chunk: &str) -> PartialArg {
        PartialArg {
            json_path: Some(json_path.to_string()),
            string_value: Some(chunk.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn progressive_mode_marks_every_chunk_partial_and_returns_exactly_one_response() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        let out = aggregator.process_response(text_chunk("Hello"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].partial, Some(true));
        // The yielded response carries its own chunk's content unmodified —
        // aggregation happens internally, not on the returned value.
        assert_eq!(
            out[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("Hello")
        );
    }

    #[test]
    fn progressive_mode_merges_consecutive_text_chunks_of_the_same_type_on_close() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("Hello, "));
        aggregator.process_response(text_chunk("world!"));
        aggregator.process_response(stop_chunk());

        let closed = aggregator.close().unwrap();
        let parts = &closed.content.as_ref().unwrap().parts;
        assert_eq!(
            parts.len(),
            1,
            "consecutive text chunks merge into one part"
        );
        assert_eq!(parts[0].text.as_deref(), Some("Hello, world!"));
        assert_eq!(closed.partial, Some(false));
    }

    #[test]
    fn progressive_mode_keeps_thought_and_regular_text_as_separate_ordered_parts() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
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
        let closed = aggregator.close().unwrap();

        let parts = &closed.content.as_ref().unwrap().parts;
        assert_eq!(
            parts.len(),
            2,
            "a thought/regular type change flushes a new part rather than merging"
        );
        assert_eq!(parts[0].thought, Some(true));
        assert_eq!(parts[0].text.as_deref(), Some("thinking..."));
        assert_eq!(parts[1].thought, None);
        assert_eq!(parts[1].text.as_deref(), Some("the answer"));
    }

    #[test]
    fn progressive_mode_appends_a_complete_non_streaming_function_call_after_flushing_text() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("checking the weather..."));
        aggregator.process_response(function_call_chunk(FunctionCall {
            name: Some("get_weather".to_string()),
            ..Default::default()
        }));
        let closed = aggregator.close().unwrap();

        let parts = &closed.content.as_ref().unwrap().parts;
        assert_eq!(
            parts.len(),
            2,
            "flushed text part, then the function call part"
        );
        assert_eq!(parts[0].text.as_deref(), Some("checking the weather..."));
        let fc = parts[1].function_call.as_ref().unwrap();
        assert_eq!(fc.name.as_deref(), Some("get_weather"));
        assert!(
            fc.id.as_deref().is_some_and(|id| id.starts_with("adk-")),
            "a missing id on a complete function call gets synthesized"
        );
    }

    #[test]
    fn progressive_mode_preserves_an_explicit_function_call_id() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(function_call_chunk(FunctionCall {
            id: Some("server-1".to_string()),
            name: Some("get_weather".to_string()),
            ..Default::default()
        }));
        let closed = aggregator.close().unwrap();
        let fc = closed.content.as_ref().unwrap().parts[0]
            .function_call
            .as_ref()
            .unwrap();
        assert_eq!(fc.id.as_deref(), Some("server-1"));
    }

    #[test]
    fn progressive_mode_assembles_a_streamed_function_call_from_partial_args_via_json_path() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();

        // First chunk: name + id, plus the first half of a streamed string arg.
        aggregator.process_response(function_call_chunk(FunctionCall {
            id: Some("call-1".to_string()),
            name: Some("get_weather".to_string()),
            partial_args: Some(vec![partial_arg_string("$.location", "San ")]),
            will_continue: Some(true),
            ..Default::default()
        }));
        // Second chunk: the rest of the streamed string arg, plus a nested
        // numeric field — still not the last chunk.
        aggregator.process_response(function_call_chunk(FunctionCall {
            partial_args: Some(vec![
                partial_arg_string("$.location", "Francisco"),
                PartialArg {
                    json_path: Some("$.units.precision".to_string()),
                    number_value: Some(2.0),
                    ..Default::default()
                },
            ]),
            will_continue: Some(true),
            ..Default::default()
        }));
        // Final chunk: no more args, will_continue absent — flushes the call.
        aggregator.process_response(function_call_chunk(FunctionCall {
            will_continue: None,
            ..Default::default()
        }));

        let closed = aggregator.close().unwrap();
        let parts = &closed.content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 1);
        let fc = parts[0].function_call.as_ref().unwrap();
        assert_eq!(fc.id.as_deref(), Some("call-1"));
        assert_eq!(fc.name.as_deref(), Some("get_weather"));
        let args = fc.args.as_ref().unwrap();
        assert_eq!(
            args.get("location"),
            Some(&Value::String("San Francisco".to_string()))
        );
        let units = args.get("units").unwrap();
        assert_eq!(units.get("precision"), Some(&Value::Float(2.0)));
    }

    #[test]
    fn progressive_mode_synthesizes_an_id_on_the_first_streaming_chunk_when_missing() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(function_call_chunk(FunctionCall {
            name: Some("get_weather".to_string()),
            partial_args: Some(vec![partial_arg_string("$.location", "Tokyo")]),
            will_continue: Some(false),
            ..Default::default()
        }));
        let closed = aggregator.close().unwrap();
        let fc = closed.content.as_ref().unwrap().parts[0]
            .function_call
            .as_ref()
            .unwrap();
        assert!(fc.id.as_deref().is_some_and(|id| id.starts_with("adk-")));
    }

    #[test]
    fn progressive_mode_keeps_interleaved_text_and_function_call_parts_in_order() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        aggregator.process_response(text_chunk("Before. "));
        aggregator.process_response(function_call_chunk(FunctionCall {
            name: Some("tool_one".to_string()),
            ..Default::default()
        }));
        aggregator.process_response(text_chunk("After."));
        let closed = aggregator.close().unwrap();

        let parts = &closed.content.as_ref().unwrap().parts;
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].text.as_deref(), Some("Before. "));
        assert_eq!(
            parts[1].function_call.as_ref().unwrap().name.as_deref(),
            Some("tool_one")
        );
        assert_eq!(parts[2].text.as_deref(), Some("After."));
    }

    #[test]
    fn progressive_mode_close_returns_none_before_any_chunk_is_processed() {
        let _lock = PROGRESSIVE_SSE_TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ProgressiveSseStreaming, true);
        let mut aggregator = StreamingResponseAggregator::new();
        assert!(aggregator.close().is_none());
    }
}
