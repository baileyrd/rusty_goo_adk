//! Capabilities C0132, C0135-C0139: `GeminiLlmConnection`, ported from
//! `google.adk.models.gemini_llm_connection`.
//!
//! **Scope of batch 5**: the send-side methods —
//! `send_history`/`send_content`/`send_content_partial`/`send_realtime`
//! (C0135-C0138).
//!
//! **Scope of batch 6 (this batch)**: `receive()` (C0139) — the ~370-line
//! stateful message-translation engine: usage-metadata remap (see
//! `live_server_message.rs`), grounding-metadata accumulation across
//! messages, streamed text/thought aggregation (tracked here by part
//! *index* rather than the source's `id()` object identity — equivalent,
//! since Rust `Part`s in one `content.parts` list never alias each other
//! or move around mid-iteration), transcription streaming (persisted
//! across `receive()` calls, matching the source's `self._input_
//! transcription_text`/`self._output_transcription_text` instance
//! fields), Gemini-3.x-variant-dependent tool-call buffering, and
//! session-resumption/voice-activity/GoAway passthrough. [`process_message`]
//! is the pure, directly-testable core (one parsed message + mutable
//! aggregation state in, zero-or-more `LlmResponse`s + a stop signal out);
//! `receive()` itself is a thin loop reading real socket frames through it.
//!
//! **Omitted, deliberately**: the source's one `retrieval_queries`-without-
//! `grounding_chunks` branch at `turn_complete` only logs a warning — no
//! observable effect on any yielded response — so it's dropped rather than
//! plumbed through a logging framework this workspace hasn't adopted yet.
//!
//! **Adaptation, disclosed (confidence caveat)**: the exact Gemini Live API
//! WebSocket message envelopes (send-side: `BidiGenerateContentClientContent`/
//! `RealtimeInput`/`ToolResponse`; receive-side: `LiveServerMessage` and its
//! nested shapes) are Google's public Multimodal Live API wire protocol —
//! not part of `google/adk-python`'s own source (which only ever talks to
//! `google.genai.live.AsyncSession`, an opaque third-party object from this
//! migration's perspective, the same way `google.genai.Client` was for the
//! REST transport in `gemini.rs`). The envelope shapes below are this
//! migration's best-effort reconstruction of that public protocol, built
//! from the same "minimal real subset" discipline used throughout Phase 3
//! — modeling exactly the fields ADK's own dispatch logic in
//! `gemini_llm_connection.py` sends/reads — but unlike the REST
//! `generateContent` body (a simpler, extremely well-known shape), this
//! hasn't been validated against a live Gemini Live endpoint. Treat the
//! envelope field names as "best effort, unverified" until exercised
//! against the real service.
//!
//! **Scope note**: opening the actual connection (the WebSocket handshake
//! to Google's Live endpoint, and the initial `BidiGenerateContentSetup`
//! message) is still deferred, for the same confidence-caveat reason — see
//! `gemini.rs`'s module doc. [`GeminiLlmConnection::new`] takes an
//! already-open [`crate::live_connection::LiveWsConnection`], the same way
//! `GeminiApiClient` is constructed independently of any particular
//! `Gemini` instance.

use std::sync::{Arc, Mutex};

use adk_genai::content::{Content, FunctionResponse, Part};
use rusty_serde::value::Value;
use rusty_serde::Serialize;

use crate::base_llm_connection::{BaseLlmConnection, BoxFuture, ConnectionError, RealtimeInput};
use crate::capabilities::{is_gemini_3_5_live_translate, is_gemini_3_x_live};
use crate::live_connection::LiveWsConnection;
use crate::live_server_message::{
    merge_grounding_metadata, to_generate_content_usage_metadata, LiveServerMessage, ServerContent,
};
use crate::llm_response::LlmResponse;

/// Minimal placeholder text sent to Gemini 3.x Live to trigger a response
/// to history that was just replayed — see [`GeminiLlmConnection::send_history`].
const RESPONSE_TRIGGER_TEXT: &str = ".";

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct BlobBody {
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    #[rusty_serde(flatten)]
    rest: Option<rusty_serde::value::Value>,
}

impl From<adk_genai::content::MediaBlobStub> for BlobBody {
    fn from(blob: adk_genai::content::MediaBlobStub) -> Self {
        Self {
            mime_type: blob.mime_type,
            rest: blob.rest,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct RealtimeInputBody {
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    audio: Option<BlobBody>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    video: Option<BlobBody>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    media: Option<BlobBody>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    activity_start: Option<rusty_serde::value::Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    activity_end: Option<rusty_serde::value::Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    audio_stream_end: Option<bool>,
}

#[derive(Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct RealtimeInputEnvelope {
    realtime_input: RealtimeInputBody,
}

#[derive(Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ClientContentBody {
    turns: Vec<Content>,
    turn_complete: bool,
}

#[derive(Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ClientContentEnvelope {
    client_content: ClientContentBody,
}

#[derive(Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ToolResponseBody {
    function_responses: Vec<FunctionResponse>,
}

#[derive(Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ToolResponseEnvelope {
    tool_response: ToolResponseBody,
}

/// `utils/content_utils.py::filter_audio_parts` — now a shared,
/// single-source-of-truth port at `adk_genai::content_utils`
/// (consolidated there in the C0927-C0929 batch); this local wrapper
/// name is kept so this file's call sites don't need touching.
fn filter_audio_parts(content: &Content) -> Option<Content> {
    adk_genai::content_utils::filter_audio_parts(content)
}

/// The Gemini Live model connection. See the module doc for what's
/// implemented and what's deferred.
pub struct GeminiLlmConnection {
    socket: Arc<LiveWsConnection>,
    is_gemini_3_x_live: bool,
    is_gemini_3_5_live_translate: bool,
    model_version: Option<String>,
    /// Populated by the real WS handshake once that's implemented (see the
    /// module doc's scope note) — always `None` today.
    live_session_id: Option<String>,
    /// Persists across `receive()` calls, matching the source's own
    /// `self._input_transcription_text` instance field.
    input_transcription_text: Mutex<String>,
    /// Persists across `receive()` calls, matching the source's own
    /// `self._output_transcription_text` instance field.
    output_transcription_text: Mutex<String>,
}

impl GeminiLlmConnection {
    pub fn new(socket: LiveWsConnection, model_version: Option<&str>) -> Self {
        Self {
            socket: Arc::new(socket),
            is_gemini_3_x_live: is_gemini_3_x_live(model_version),
            is_gemini_3_5_live_translate: is_gemini_3_5_live_translate(model_version),
            model_version: model_version.map(|s| s.to_string()),
            live_session_id: None,
            input_transcription_text: Mutex::new(String::new()),
            output_transcription_text: Mutex::new(String::new()),
        }
    }

    async fn send_json(&self, json: String) -> Result<(), ConnectionError> {
        let socket = self.socket.clone();
        match rusty_tokio::spawn_blocking(move || socket.send_text(json)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(ConnectionError::Failed(e.to_string())),
            Err(join_error) => Err(ConnectionError::Failed(join_error.to_string())),
        }
    }

    async fn send_envelope<T: rusty_serde::Serialize>(
        &self,
        envelope: &T,
    ) -> Result<(), ConnectionError> {
        let json = rusty_serde::json::to_string(envelope)
            .map_err(|e| ConnectionError::Failed(e.to_string()))?;
        self.send_json(json).await
    }

    async fn send_client_content(
        &self,
        turns: Vec<Content>,
        turn_complete: bool,
    ) -> Result<(), ConnectionError> {
        self.send_envelope(&ClientContentEnvelope {
            client_content: ClientContentBody {
                turns,
                turn_complete,
            },
        })
        .await
    }

    async fn send_realtime_text(&self, text: &str) -> Result<(), ConnectionError> {
        self.send_envelope(&RealtimeInputEnvelope {
            realtime_input: RealtimeInputBody {
                text: Some(text.to_string()),
                ..Default::default()
            },
        })
        .await
    }

    /// C0136: sends conversation history, filtering out audio parts
    /// (already transcribed, and unsupported by the Live API over
    /// `send`/`send_client_content`) and triggering a response if the
    /// replayed history ends on a user turn — Gemini 3.x Live needs an
    /// extra placeholder nudge to actually start generating in that case.
    async fn send_history_impl(&self, history: Vec<Content>) -> Result<(), ConnectionError> {
        let contents: Vec<Content> = history.iter().filter_map(filter_audio_parts).collect();
        if contents.is_empty() {
            return Ok(());
        }
        let turn_complete = contents.last().and_then(|c| c.role.as_deref()) == Some("user");
        self.send_client_content(contents, turn_complete).await?;
        if turn_complete && self.is_gemini_3_x_live {
            self.send_realtime_text(RESPONSE_TRIGGER_TEXT).await?;
        }
        Ok(())
    }

    /// C0135/C0137: sends content, optionally as a partial (non-turn-
    /// completing) update. Function responses always route via the tool-
    /// response envelope; a single non-partial text part on Gemini 3.x
    /// Live routes via realtime input instead of client content.
    async fn send_content_partial_impl(
        &self,
        content: Content,
        partial: bool,
    ) -> Result<(), ConnectionError> {
        if content.parts.is_empty() {
            return Err(ConnectionError::Failed(
                "content.parts must not be empty".to_string(),
            ));
        }

        if content
            .parts
            .iter()
            .all(|part| part.function_response.is_some())
        {
            let function_responses: Vec<FunctionResponse> = content
                .parts
                .into_iter()
                .filter_map(|part| part.function_response)
                .collect();
            return self
                .send_envelope(&ToolResponseEnvelope {
                    tool_response: ToolResponseBody { function_responses },
                })
                .await;
        }

        if !partial && self.is_gemini_3_x_live && content.parts.len() == 1 {
            if let Some(text) = content.parts[0].text.clone() {
                return self.send_realtime_text(&text).await;
            }
        }

        self.send_client_content(vec![content], !partial).await
    }

    /// C0138: type-dispatches a realtime input chunk. Gemini 3.x
    /// Live/3.5-Live-Translate route audio/image blobs via dedicated
    /// realtime-input fields; other models use the generic `media` field.
    async fn send_realtime_impl(&self, input: RealtimeInput) -> Result<(), ConnectionError> {
        match input {
            RealtimeInput::Blob(blob) => {
                let is_audio = blob
                    .mime_type
                    .as_deref()
                    .map(|m| m.starts_with("audio/"))
                    .unwrap_or(false);
                let is_image = blob
                    .mime_type
                    .as_deref()
                    .map(|m| m.starts_with("image/"))
                    .unwrap_or(false);
                let variant_specific = self.is_gemini_3_x_live || self.is_gemini_3_5_live_translate;

                if variant_specific {
                    if is_audio {
                        self.send_envelope(&RealtimeInputEnvelope {
                            realtime_input: RealtimeInputBody {
                                audio: Some(blob.into()),
                                ..Default::default()
                            },
                        })
                        .await
                    } else if is_image {
                        self.send_envelope(&RealtimeInputEnvelope {
                            realtime_input: RealtimeInputBody {
                                video: Some(blob.into()),
                                ..Default::default()
                            },
                        })
                        .await
                    } else {
                        // Matches the source's "Blob not sent. Unknown or
                        // empty mime type" warn-and-skip.
                        Ok(())
                    }
                } else {
                    self.send_envelope(&RealtimeInputEnvelope {
                        realtime_input: RealtimeInputBody {
                            media: Some(blob.into()),
                            ..Default::default()
                        },
                    })
                    .await
                }
            }
            RealtimeInput::ActivityStart => {
                self.send_envelope(&RealtimeInputEnvelope {
                    realtime_input: RealtimeInputBody {
                        activity_start: Some(rusty_serde::value::Value::Map(vec![])),
                        ..Default::default()
                    },
                })
                .await
            }
            RealtimeInput::ActivityEnd => {
                self.send_envelope(&RealtimeInputEnvelope {
                    realtime_input: RealtimeInputBody {
                        activity_end: Some(rusty_serde::value::Value::Map(vec![])),
                        ..Default::default()
                    },
                })
                .await
            }
            RealtimeInput::AudioStreamEnd => {
                self.send_envelope(&RealtimeInputEnvelope {
                    realtime_input: RealtimeInputBody {
                        audio_stream_end: Some(true),
                        ..Default::default()
                    },
                })
                .await
            }
        }
    }

    fn build_full_text_response(
        &self,
        text: &str,
        is_thought: bool,
        grounding_metadata: Option<Value>,
        interrupted: bool,
    ) -> LlmResponse {
        let mut part = Part::text(text);
        if is_thought {
            part.thought = Some(true);
        }
        LlmResponse {
            content: Some(Content {
                role: Some("model".to_string()),
                parts: vec![part],
            }),
            grounding_metadata,
            interrupted: Some(interrupted),
            partial: Some(false),
            model_version: self.model_version.clone(),
            live_session_id: self.live_session_id.clone(),
            ..Default::default()
        }
    }

    /// C0139: translates one parsed `LiveServerMessage` into zero or more
    /// `LlmResponse`s, mutating `state` (the aggregation carried across
    /// messages within one `receive()` call). Returns `true` when the
    /// caller should stop reading further messages (the source's `break`
    /// on `turn_complete` — which also skips the source's `tool_call`/
    /// `session_resumption_update`/`voice_activity`/`go_away` checks for
    /// that same message, since they're siblings of `server_content` inside
    /// the same loop iteration, after the `break`). Pure and directly
    /// testable — no socket I/O.
    fn process_message(
        &self,
        message: LiveServerMessage,
        state: &mut ReceiveState,
    ) -> (Vec<LlmResponse>, bool) {
        let mut responses = Vec::new();
        let live_session_id = self.live_session_id.clone();
        let model_version = self.model_version.clone();
        let mut stop = false;

        if let Some(usage_metadata) = &message.usage_metadata {
            responses.push(LlmResponse {
                usage_metadata: Some(to_generate_content_usage_metadata(usage_metadata)),
                model_version: model_version.clone(),
                live_session_id: live_session_id.clone(),
                ..Default::default()
            });
        }

        if let Some(server_content) = &message.server_content {
            let grounding_metadata = server_content.grounding_metadata.clone();
            if grounding_metadata.is_some() {
                state.last_grounding_metadata = merge_grounding_metadata(
                    state.last_grounding_metadata.take(),
                    grounding_metadata.clone(),
                );
            }

            let has_content_parts = server_content
                .model_turn
                .as_ref()
                .map(|c| !c.parts.is_empty())
                .unwrap_or(false);

            // Standalone grounding_metadata event (when content is empty).
            if !has_content_parts
                && server_content.grounding_metadata.is_some()
                && !server_content.turn_complete.unwrap_or(false)
            {
                responses.push(LlmResponse {
                    grounding_metadata: server_content.grounding_metadata.clone(),
                    interrupted: server_content.interrupted,
                    model_version: model_version.clone(),
                    live_session_id: live_session_id.clone(),
                    turn_complete_reason: server_content.turn_complete_reason.clone(),
                    ..Default::default()
                });
            }

            if let Some(content) = &server_content.model_turn {
                if !content.parts.is_empty() {
                    self.process_model_turn(
                        content,
                        server_content,
                        message.tool_call.is_some(),
                        &grounding_metadata,
                        state,
                        &mut responses,
                    );
                }
            }

            // Note: in some cases tool_call may arrive before
            // generation_complete, causing transcription to appear after
            // tool_call in the session log.
            if let Some(input_transcription) = &server_content.input_transcription {
                self.process_input_transcription(input_transcription, &mut responses);
            }
            if let Some(output_transcription) = &server_content.output_transcription {
                self.process_output_transcription(output_transcription, &mut responses);
            }

            // The Gemini API or Vertex AI might not send a transcription-
            // finished signal — rely on generation_complete/turn_complete/
            // interrupted to flush any pending transcriptions instead.
            if server_content.interrupted.unwrap_or(false)
                || server_content.turn_complete.unwrap_or(false)
                || server_content.generation_complete.unwrap_or(false)
            {
                self.flush_pending_transcriptions(&mut responses);
            }

            if server_content.turn_complete.unwrap_or(false) {
                if !state.text.is_empty() {
                    responses.push(self.build_full_text_response(
                        &state.text,
                        state.is_thought,
                        state.last_grounding_metadata.take(),
                        server_content.interrupted.unwrap_or(false),
                    ));
                    state.text.clear();
                    state.is_thought = false;
                }
                if !state.tool_call_parts.is_empty() {
                    responses.push(LlmResponse {
                        content: Some(Content {
                            role: Some("model".to_string()),
                            parts: std::mem::take(&mut state.tool_call_parts),
                        }),
                        grounding_metadata: state.tool_call_metadata.take(),
                        model_version: model_version.clone(),
                        live_session_id: live_session_id.clone(),
                        ..Default::default()
                    });
                    state.last_grounding_metadata = None;
                }

                let final_grounding_metadata = grounding_metadata
                    .clone()
                    .or_else(|| state.last_grounding_metadata.clone())
                    .or_else(|| self.is_gemini_3_x_live.then(|| Value::Map(Vec::new())));
                responses.push(LlmResponse {
                    turn_complete: Some(true),
                    interrupted: server_content.interrupted,
                    grounding_metadata: final_grounding_metadata,
                    model_version: model_version.clone(),
                    live_session_id: live_session_id.clone(),
                    turn_complete_reason: server_content.turn_complete_reason.clone(),
                    ..Default::default()
                });
                state.last_grounding_metadata = None;
                stop = true;
            } else if server_content.interrupted.unwrap_or(false) {
                // In case of empty content or parts, still surface
                // interruption: merge the previous partial text if any;
                // otherwise don't (content can be absent when the model
                // safety threshold triggers).
                if !state.text.is_empty() {
                    responses.push(self.build_full_text_response(
                        &state.text,
                        state.is_thought,
                        state.last_grounding_metadata.take(),
                        true,
                    ));
                    state.text.clear();
                    state.is_thought = false;
                } else {
                    responses.push(LlmResponse {
                        interrupted: server_content.interrupted,
                        grounding_metadata: state.last_grounding_metadata.take(),
                        model_version: model_version.clone(),
                        live_session_id: live_session_id.clone(),
                        ..Default::default()
                    });
                }
            }
        }

        if !stop {
            if let Some(tool_call) = &message.tool_call {
                self.process_tool_call(tool_call, state, &mut responses);
            }
            if let Some(session_resumption_update) = &message.session_resumption_update {
                responses.push(LlmResponse {
                    live_session_resumption_update: Some(session_resumption_update.clone()),
                    model_version: model_version.clone(),
                    live_session_id: live_session_id.clone(),
                    ..Default::default()
                });
            }
            if let Some(voice_activity) = &message.voice_activity {
                responses.push(LlmResponse {
                    voice_activity: Some(voice_activity.clone()),
                    model_version: model_version.clone(),
                    live_session_id: live_session_id.clone(),
                    ..Default::default()
                });
            }
            if let Some(go_away) = &message.go_away {
                responses.push(LlmResponse {
                    go_away: Some(go_away.clone()),
                    model_version,
                    live_session_id,
                    ..Default::default()
                });
            }
        }

        (responses, stop)
    }

    /// The `if message.tool_call:` branch: buffers function-call parts,
    /// flushing accumulated text first; Gemini 3.x Live yields immediately
    /// (it doesn't emit `turn_complete` until it receives the tool
    /// response, so buffering would deadlock the conversation), other
    /// models buffer until `turn_complete`.
    fn process_tool_call(
        &self,
        tool_call: &crate::live_server_message::ToolCall,
        state: &mut ReceiveState,
        responses: &mut Vec<LlmResponse>,
    ) {
        if !state.text.is_empty() {
            responses.push(self.build_full_text_response(
                &state.text,
                state.is_thought,
                state.last_grounding_metadata.take(),
                false,
            ));
            state.text.clear();
            state.is_thought = false;
        }
        state.tool_call_parts.extend(
            crate::live_server_message::tool_call_function_calls(tool_call)
                .into_iter()
                .map(Part::function_call),
        );
        if !self.is_gemini_3_x_live && state.tool_call_metadata.is_none() {
            state.tool_call_metadata = state.last_grounding_metadata.clone();
        }
        if self.is_gemini_3_x_live && !state.tool_call_parts.is_empty() {
            responses.push(LlmResponse {
                content: Some(Content {
                    role: Some("model".to_string()),
                    parts: std::mem::take(&mut state.tool_call_parts),
                }),
                grounding_metadata: state.last_grounding_metadata.clone(),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
            state.last_grounding_metadata = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_model_turn(
        &self,
        content: &Content,
        server_content: &ServerContent,
        has_tool_call: bool,
        grounding_metadata: &Option<Value>,
        state: &mut ReceiveState,
        responses: &mut Vec<LlmResponse>,
    ) {
        let mut llm_response = LlmResponse {
            content: Some(content.clone()),
            interrupted: server_content.interrupted,
            model_version: self.model_version.clone(),
            live_session_id: self.live_session_id.clone(),
            turn_complete_reason: server_content.turn_complete_reason.clone(),
            ..Default::default()
        };
        // grounding_metadata is yielded again at turn_complete, so avoid
        // duplicating it here if turn_complete is true.
        if !server_content.turn_complete.unwrap_or(false) && grounding_metadata.is_some() {
            llm_response.grounding_metadata = grounding_metadata.clone();
        }

        let will_flush = server_content.turn_complete.unwrap_or(false)
            || server_content.interrupted.unwrap_or(false)
            || has_tool_call;

        let mut flushed_indices = std::collections::HashSet::new();
        let mut accumulated_indices = Vec::new();
        for (index, part) in content.parts.iter().enumerate() {
            if let Some(text) = &part.text {
                let current_is_thought = part.thought.unwrap_or(false);
                if !state.text.is_empty() && current_is_thought != state.is_thought {
                    responses.push(self.build_full_text_response(
                        &state.text,
                        state.is_thought,
                        None,
                        false,
                    ));
                    state.text.clear();
                    state.is_thought = false;
                    flushed_indices.extend(accumulated_indices.drain(..));
                }
                state.text.push_str(text);
                state.is_thought = current_is_thought;
                llm_response.partial = Some(true);
                accumulated_indices.push(index);
            } else if !state.text.is_empty() && part.inline_data.is_none() {
                // Don't yield the merged text event when receiving audio data.
                responses.push(self.build_full_text_response(
                    &state.text,
                    state.is_thought,
                    state.last_grounding_metadata.take(),
                    false,
                ));
                state.text.clear();
                state.is_thought = false;
                flushed_indices.extend(accumulated_indices.drain(..));
            }
        }
        if will_flush {
            flushed_indices.extend(accumulated_indices.drain(..));
        }
        if !flushed_indices.is_empty() {
            llm_response.content = Some(Content {
                role: content.role.clone(),
                parts: content
                    .parts
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !flushed_indices.contains(index))
                    .map(|(_, part)| part.clone())
                    .collect(),
            });
        }
        if llm_response
            .content
            .as_ref()
            .map(|c| !c.parts.is_empty())
            .unwrap_or(false)
        {
            responses.push(llm_response);
        }
    }

    fn process_input_transcription(
        &self,
        transcription: &crate::live_server_message::Transcription,
        responses: &mut Vec<LlmResponse>,
    ) {
        // Gemini 3.x Live only sends a single final input transcription.
        if self.is_gemini_3_x_live {
            if let Some(text) = &transcription.text {
                if !text.is_empty() {
                    responses.push(LlmResponse {
                        input_transcription: Some(adk_genai_transcription(text.clone(), true)),
                        partial: Some(false),
                        model_version: self.model_version.clone(),
                        live_session_id: self.live_session_id.clone(),
                        ..Default::default()
                    });
                }
            }
            return;
        }
        if let Some(text) = &transcription.text {
            if !text.is_empty() {
                let mut buffer = self.input_transcription_text.lock().unwrap();
                buffer.push_str(text);
                responses.push(LlmResponse {
                    input_transcription: Some(adk_genai_transcription(text.clone(), false)),
                    partial: Some(true),
                    model_version: self.model_version.clone(),
                    live_session_id: self.live_session_id.clone(),
                    ..Default::default()
                });
            }
        }
        // finished=true and partial transcription may happen in the same message.
        if transcription.finished.unwrap_or(false) {
            let mut buffer = self.input_transcription_text.lock().unwrap();
            responses.push(LlmResponse {
                input_transcription: Some(adk_genai_transcription(buffer.clone(), true)),
                partial: Some(false),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
            buffer.clear();
        }
    }

    fn process_output_transcription(
        &self,
        transcription: &crate::live_server_message::Transcription,
        responses: &mut Vec<LlmResponse>,
    ) {
        if let Some(text) = &transcription.text {
            if !text.is_empty() {
                let mut buffer = self.output_transcription_text.lock().unwrap();
                buffer.push_str(text);
                responses.push(LlmResponse {
                    output_transcription: Some(adk_genai_transcription(text.clone(), false)),
                    partial: Some(true),
                    model_version: self.model_version.clone(),
                    live_session_id: self.live_session_id.clone(),
                    ..Default::default()
                });
            }
        }
        if transcription.finished.unwrap_or(false) {
            let mut buffer = self.output_transcription_text.lock().unwrap();
            responses.push(LlmResponse {
                output_transcription: Some(adk_genai_transcription(buffer.clone(), true)),
                partial: Some(false),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
            buffer.clear();
        }
    }

    fn flush_pending_transcriptions(&self, responses: &mut Vec<LlmResponse>) {
        let mut input_buffer = self.input_transcription_text.lock().unwrap();
        if !input_buffer.is_empty() {
            responses.push(LlmResponse {
                input_transcription: Some(adk_genai_transcription(input_buffer.clone(), true)),
                partial: Some(false),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
            input_buffer.clear();
        }
        drop(input_buffer);

        let mut output_buffer = self.output_transcription_text.lock().unwrap();
        if !output_buffer.is_empty() {
            responses.push(LlmResponse {
                output_transcription: Some(adk_genai_transcription(output_buffer.clone(), true)),
                partial: Some(false),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
            output_buffer.clear();
        }
    }

    /// C0139: reads real socket frames through [`Self::process_message`]
    /// until a turn completes (`stop == true`) or the socket closes,
    /// collecting every yielded response into one `Vec` — matching the
    /// `BaseLlmConnection` trait's materialized-batch `receive()` contract
    /// (see `base_llm.rs`'s `generate_content_async` for the same
    /// stream-to-`Vec` collapse).
    async fn receive_impl(&self) -> Result<Vec<LlmResponse>, ConnectionError> {
        let mut state = ReceiveState::default();
        let mut all_responses = Vec::new();
        loop {
            let socket = self.socket.clone();
            let text = match rusty_tokio::spawn_blocking(move || socket.receive_text()).await {
                Ok(Ok(Some(text))) => text,
                Ok(Ok(None)) => break,
                Ok(Err(e)) => return Err(ConnectionError::Failed(e.to_string())),
                Err(join_error) => return Err(ConnectionError::Failed(join_error.to_string())),
            };
            let message: LiveServerMessage = rusty_serde::json::from_str(&text)
                .map_err(|e| ConnectionError::Failed(e.to_string()))?;
            let (responses, stop) = self.process_message(message, &mut state);
            all_responses.extend(responses);
            if stop {
                break;
            }
        }
        if !state.tool_call_parts.is_empty() {
            all_responses.push(LlmResponse {
                content: Some(Content {
                    role: Some("model".to_string()),
                    parts: state.tool_call_parts,
                }),
                model_version: self.model_version.clone(),
                live_session_id: self.live_session_id.clone(),
                ..Default::default()
            });
        }
        Ok(all_responses)
    }
}

fn adk_genai_transcription(text: String, finished: bool) -> Value {
    Value::Map(vec![
        ("text".to_string(), Value::String(text)),
        ("finished".to_string(), Value::Bool(finished)),
    ])
}

/// Aggregation state carried across messages within one `receive()` call —
/// local variables in the source, since it's a plain function-scoped
/// generator there.
#[derive(Default)]
struct ReceiveState {
    text: String,
    is_thought: bool,
    tool_call_parts: Vec<Part>,
    last_grounding_metadata: Option<Value>,
    tool_call_metadata: Option<Value>,
}

impl BaseLlmConnection for GeminiLlmConnection {
    fn send_history<'a>(
        &'a self,
        history: Vec<Content>,
    ) -> BoxFuture<'a, Result<(), ConnectionError>> {
        Box::pin(async move { self.send_history_impl(history).await })
    }

    fn send_content<'a>(&'a self, content: Content) -> BoxFuture<'a, Result<(), ConnectionError>> {
        Box::pin(async move { self.send_content_partial_impl(content, false).await })
    }

    fn send_content_partial<'a>(
        &'a self,
        content: Content,
        partial: bool,
    ) -> BoxFuture<'a, Result<(), ConnectionError>> {
        Box::pin(async move { self.send_content_partial_impl(content, partial).await })
    }

    fn send_realtime<'a>(
        &'a self,
        input: RealtimeInput,
    ) -> BoxFuture<'a, Result<(), ConnectionError>> {
        Box::pin(async move { self.send_realtime_impl(input).await })
    }

    /// C0139.
    fn receive<'a>(&'a self) -> BoxFuture<'a, Result<Vec<LlmResponse>, ConnectionError>> {
        Box::pin(async move { self.receive_impl().await })
    }

    fn close<'a>(&'a self) -> BoxFuture<'a, Result<(), ConnectionError>> {
        let socket = self.socket.clone();
        Box::pin(async move {
            match rusty_tokio::spawn_blocking(move || socket.close()).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(ConnectionError::Failed(e.to_string())),
                Err(join_error) => Err(ConnectionError::Failed(join_error.to_string())),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::{FunctionCall, MediaBlobStub};

    /// Spawns a one-shot local WebSocket server that records every text
    /// frame it receives (as parsed JSON) and never replies — enough to
    /// verify what `GeminiLlmConnection` sends, dependency-free, without a
    /// live Gemini endpoint.
    fn spawn_recording_server() -> (
        String,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        std::thread::JoinHandle<()>,
    ) {
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            loop {
                match socket.read() {
                    Ok(tungstenite::Message::Text(text)) => {
                        received_clone.lock().unwrap().push(text.to_string());
                    }
                    Ok(tungstenite::Message::Close(_)) => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
        });
        (format!("ws://{addr}"), received, handle)
    }

    fn connect_for_test() -> (
        GeminiLlmConnection,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        std::thread::JoinHandle<()>,
    ) {
        let (url, received, server) = spawn_recording_server();
        let socket = LiveWsConnection::connect(&url).unwrap();
        (GeminiLlmConnection::new(socket, None), received, server)
    }

    fn connect_for_test_with_model(
        model_version: &str,
    ) -> (
        GeminiLlmConnection,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        std::thread::JoinHandle<()>,
    ) {
        let (url, received, server) = spawn_recording_server();
        let socket = LiveWsConnection::connect(&url).unwrap();
        (
            GeminiLlmConnection::new(socket, Some(model_version)),
            received,
            server,
        )
    }

    fn finish(connection: &GeminiLlmConnection, server: std::thread::JoinHandle<()>) {
        let _ = connection.socket.close();
        server.join().unwrap();
    }

    #[rusty_tokio::test]
    async fn send_content_routes_a_plain_text_turn_as_client_content() {
        let (connection, received, server) = connect_for_test();
        connection
            .send_content(Content::user_text("hi"))
            .await
            .unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].contains("\"clientContent\""));
        assert!(sent[0].contains("\"turnComplete\":true"));
    }

    #[rusty_tokio::test]
    async fn send_content_partial_does_not_complete_the_turn() {
        let (connection, received, server) = connect_for_test();
        connection
            .send_content_partial(Content::user_text("partial"), true)
            .await
            .unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert!(sent[0].contains("\"turnComplete\":false"));
    }

    #[rusty_tokio::test]
    async fn send_content_routes_function_responses_as_a_tool_response() {
        let (connection, received, server) = connect_for_test();
        let content = Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        );
        connection.send_content(content).await.unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert!(sent[0].contains("\"toolResponse\""));
        assert!(sent[0].contains("get_weather"));
        assert!(!sent[0].contains("clientContent"));
    }

    #[rusty_tokio::test]
    async fn send_content_rejects_empty_parts() {
        let (connection, _received, server) = connect_for_test();
        let result = connection.send_content(Content::new("user", vec![])).await;
        finish(&connection, server);
        assert!(result.is_err());
    }

    #[rusty_tokio::test]
    async fn send_history_filters_audio_parts_and_completes_a_trailing_user_turn() {
        let (connection, received, server) = connect_for_test();
        let history = vec![Content::new(
            "user",
            vec![
                Part::text("hello"),
                Part {
                    inline_data: Some(MediaBlobStub {
                        mime_type: Some("audio/wav".to_string()),
                        rest: None,
                    }),
                    ..Default::default()
                },
            ],
        )];
        connection.send_history(history).await.unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].contains("hello"));
        assert!(!sent[0].contains("audio/wav"));
        assert!(sent[0].contains("\"turnComplete\":true"));
    }

    #[rusty_tokio::test]
    async fn send_history_is_a_noop_when_every_part_is_audio() {
        let (connection, received, server) = connect_for_test();
        let history = vec![Content::new(
            "user",
            vec![Part {
                inline_data: Some(MediaBlobStub {
                    mime_type: Some("audio/wav".to_string()),
                    rest: None,
                }),
                ..Default::default()
            }],
        )];
        connection.send_history(history).await.unwrap();
        finish(&connection, server);
        assert!(received.lock().unwrap().is_empty());
    }

    #[rusty_tokio::test]
    async fn send_realtime_routes_a_blob_through_the_generic_media_field_by_default() {
        let (connection, received, server) = connect_for_test();
        connection
            .send_realtime(RealtimeInput::Blob(MediaBlobStub {
                mime_type: Some("audio/pcm".to_string()),
                rest: None,
            }))
            .await
            .unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert!(sent[0].contains("\"media\""));
        assert!(!sent[0].contains("\"audio\""));
        // Unset RealtimeInputBody fields must be omitted, not sent as
        // `null` — a null-heavy body reads as ambiguous over-the-wire and
        // this assertion would have caught the very bug this batch found
        // and fixed (see the module doc's wire-fidelity note).
        assert!(!sent[0].contains("null"));
    }

    #[rusty_tokio::test]
    async fn send_realtime_routes_activity_boundaries() {
        let (connection, received, server) = connect_for_test();
        connection
            .send_realtime(RealtimeInput::ActivityStart)
            .await
            .unwrap();
        connection
            .send_realtime(RealtimeInput::ActivityEnd)
            .await
            .unwrap();
        connection
            .send_realtime(RealtimeInput::AudioStreamEnd)
            .await
            .unwrap();
        finish(&connection, server);

        let sent = received.lock().unwrap();
        assert!(sent[0].contains("\"activityStart\""));
        assert!(sent[1].contains("\"activityEnd\""));
        assert!(sent[2].contains("\"audioStreamEnd\":true"));
    }

    #[rusty_tokio::test]
    async fn receive_returns_an_empty_vec_when_the_socket_closes_with_no_messages() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            let _ = socket.close(None);
        });
        let socket = LiveWsConnection::connect(&format!("ws://{addr}")).unwrap();
        let connection = GeminiLlmConnection::new(socket, None);
        let responses = connection.receive().await.unwrap();
        server.join().unwrap();
        assert!(responses.is_empty());
    }

    #[rusty_tokio::test]
    async fn receive_reads_messages_over_the_real_socket_until_turn_complete() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            socket
                .send(tungstenite::Message::Text(
                    r#"{"serverContent":{"modelTurn":{"role":"model","parts":[{"text":"hi"}]}}}"#
                        .into(),
                ))
                .unwrap();
            socket
                .send(tungstenite::Message::Text(
                    r#"{"serverContent":{"turnComplete":true}}"#.into(),
                ))
                .unwrap();
        });
        let socket = LiveWsConnection::connect(&format!("ws://{addr}")).unwrap();
        let connection = GeminiLlmConnection::new(socket, None);
        let responses = connection.receive().await.unwrap();
        server.join().unwrap();
        assert!(responses.iter().any(|r| r.turn_complete == Some(true)));
        assert!(responses.iter().any(|r| r
            .content
            .as_ref()
            .and_then(|c| c.parts.first())
            .and_then(|p| p.text.as_deref())
            == Some("hi")));
    }

    #[rusty_tokio::test]
    async fn process_message_maps_usage_metadata() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let message = LiveServerMessage {
            usage_metadata: Some(crate::live_server_message::LiveUsageMetadata {
                prompt_token_count: Some(10),
                response_token_count: Some(5),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(message, &mut state);
        finish(&connection, server);
        assert!(!stop);
        assert_eq!(responses.len(), 1);
        let Value::Map(entries) = responses[0].usage_metadata.clone().unwrap() else {
            panic!("expected a map")
        };
        assert!(entries
            .iter()
            .any(|(k, v)| k == "candidatesTokenCount" && *v == Value::Int(5)));
    }

    #[rusty_tokio::test]
    async fn process_message_streams_partial_text_then_completes_the_turn() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();

        let first = LiveServerMessage {
            server_content: Some(ServerContent {
                model_turn: Some(Content::new("model", vec![Part::text("hello ")])),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses1, stop1) = connection.process_message(first, &mut state);
        assert!(!stop1);
        assert_eq!(responses1.len(), 1);
        assert_eq!(responses1[0].partial, Some(true));
        assert_eq!(
            responses1[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("hello ")
        );

        let second = LiveServerMessage {
            server_content: Some(ServerContent {
                turn_complete: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses2, stop2) = connection.process_message(second, &mut state);
        finish(&connection, server);
        assert!(stop2);
        assert_eq!(responses2.len(), 2);
        assert_eq!(
            responses2[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("hello ")
        );
        assert_eq!(responses2[0].partial, Some(false));
        assert_eq!(responses2[1].turn_complete, Some(true));
    }

    #[rusty_tokio::test]
    async fn process_message_buffers_tool_calls_until_turn_complete_on_non_gemini_3x() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let tool_call_message = LiveServerMessage {
            tool_call: Some(crate::live_server_message::ToolCall {
                function_calls: Some(vec![FunctionCall {
                    name: Some("get_weather".to_string()),
                    ..Default::default()
                }]),
            }),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(tool_call_message, &mut state);
        assert!(!stop);
        assert!(responses.is_empty());
        assert_eq!(state.tool_call_parts.len(), 1);

        let turn_complete_message = LiveServerMessage {
            server_content: Some(ServerContent {
                turn_complete: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses2, stop2) = connection.process_message(turn_complete_message, &mut state);
        finish(&connection, server);
        assert!(stop2);
        assert!(responses2.iter().any(|r| r
            .content
            .as_ref()
            .map(|c| c.get_function_calls().len() == 1)
            .unwrap_or(false)));
    }

    #[rusty_tokio::test]
    async fn process_message_yields_tool_calls_immediately_on_gemini_3x_live() {
        let (connection, _received, server) = connect_for_test_with_model("gemini-3.0-live");
        let mut state = ReceiveState::default();
        let message = LiveServerMessage {
            tool_call: Some(crate::live_server_message::ToolCall {
                function_calls: Some(vec![FunctionCall {
                    name: Some("get_weather".to_string()),
                    ..Default::default()
                }]),
            }),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(message, &mut state);
        finish(&connection, server);
        assert!(!stop);
        assert_eq!(responses.len(), 1);
        assert!(state.tool_call_parts.is_empty());
    }

    #[rusty_tokio::test]
    async fn process_message_streams_input_transcription_and_flushes_on_finished() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let msg1 = LiveServerMessage {
            server_content: Some(ServerContent {
                input_transcription: Some(crate::live_server_message::Transcription {
                    text: Some("hel".to_string()),
                    finished: Some(false),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses1, _) = connection.process_message(msg1, &mut state);
        assert_eq!(responses1.len(), 1);
        assert_eq!(responses1[0].partial, Some(true));

        let msg2 = LiveServerMessage {
            server_content: Some(ServerContent {
                input_transcription: Some(crate::live_server_message::Transcription {
                    text: Some("lo".to_string()),
                    finished: Some(true),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses2, _) = connection.process_message(msg2, &mut state);
        finish(&connection, server);
        assert_eq!(responses2.len(), 2);
        assert_eq!(responses2[1].partial, Some(false));
        let Value::Map(entries) = responses2[1].input_transcription.clone().unwrap() else {
            panic!("expected a map")
        };
        let text = entries.iter().find(|(k, _)| k == "text").unwrap().1.clone();
        assert_eq!(text, Value::String("hello".to_string()));
    }

    #[rusty_tokio::test]
    async fn process_message_sends_a_single_final_input_transcription_on_gemini_3x_live() {
        let (connection, _received, server) = connect_for_test_with_model("gemini-3.0-live");
        let mut state = ReceiveState::default();
        let msg = LiveServerMessage {
            server_content: Some(ServerContent {
                input_transcription: Some(crate::live_server_message::Transcription {
                    text: Some("hello".to_string()),
                    finished: None,
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses, _) = connection.process_message(msg, &mut state);
        finish(&connection, server);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].partial, Some(false));
    }

    #[rusty_tokio::test]
    async fn process_message_merges_grounding_metadata_and_surfaces_it_at_turn_complete() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let msg1 = LiveServerMessage {
            server_content: Some(ServerContent {
                grounding_metadata: Some(Value::Map(vec![(
                    "retrievalQueries".to_string(),
                    Value::Seq(vec![Value::String("q1".to_string())]),
                )])),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses1, _) = connection.process_message(msg1, &mut state);
        assert_eq!(responses1.len(), 1);
        assert!(responses1[0].grounding_metadata.is_some());

        let msg2 = LiveServerMessage {
            server_content: Some(ServerContent {
                turn_complete: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses2, stop2) = connection.process_message(msg2, &mut state);
        finish(&connection, server);
        assert!(stop2);
        let turn_complete_response = responses2
            .iter()
            .find(|r| r.turn_complete == Some(true))
            .unwrap();
        assert!(turn_complete_response.grounding_metadata.is_some());
    }

    #[rusty_tokio::test]
    async fn process_message_surfaces_interruption_without_accumulated_text() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let msg = LiveServerMessage {
            server_content: Some(ServerContent {
                interrupted: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(msg, &mut state);
        finish(&connection, server);
        assert!(!stop);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].interrupted, Some(true));
    }

    #[rusty_tokio::test]
    async fn process_message_passes_through_session_resumption_voice_activity_and_go_away() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let msg = LiveServerMessage {
            session_resumption_update: Some(Value::String("handle".to_string())),
            voice_activity: Some(Value::String("speaking".to_string())),
            go_away: Some(Value::String("bye".to_string())),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(msg, &mut state);
        finish(&connection, server);
        assert!(!stop);
        assert_eq!(responses.len(), 3);
        assert!(responses[0].live_session_resumption_update.is_some());
        assert!(responses[1].voice_activity.is_some());
        assert!(responses[2].go_away.is_some());
    }

    #[rusty_tokio::test]
    async fn process_message_skips_side_channels_when_turn_complete_fires_in_the_same_message() {
        let (connection, _received, server) = connect_for_test();
        let mut state = ReceiveState::default();
        let msg = LiveServerMessage {
            server_content: Some(ServerContent {
                turn_complete: Some(true),
                ..Default::default()
            }),
            voice_activity: Some(Value::String("speaking".to_string())),
            ..Default::default()
        };
        let (responses, stop) = connection.process_message(msg, &mut state);
        finish(&connection, server);
        assert!(stop);
        assert!(!responses.iter().any(|r| r.voice_activity.is_some()));
    }

    #[rusty_tokio::test]
    async fn function_call_is_unaffected_by_the_realtime_input_change() {
        // Sanity check that the retroactive `RealtimeInput` type change
        // didn't disturb unrelated Content/Part construction.
        let content = Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                name: Some("tool".to_string()),
                ..Default::default()
            })],
        );
        assert_eq!(content.get_function_calls().len(), 1);
    }
}
