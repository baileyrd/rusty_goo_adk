//! Capabilities C0132, C0135-C0138: `GeminiLlmConnection`, ported from
//! `google.adk.models.gemini_llm_connection`.
//!
//! **Scope of this batch**: the send-side methods only —
//! `send_history`/`send_content`/`send_content_partial`/`send_realtime`.
//! `receive()` (C0139) is deferred: it's a ~370-line stateful message-
//! translation engine (grounding-metadata accumulation with index-offset
//! merging, streamed text/thought aggregation tracked by part identity,
//! transcription streaming, Gemini-3.x-variant-dependent tool-call
//! buffering, session-resumption/voice-activity/GoAway passthrough) that
//! deserves its own dedicated batch — calling it here returns a named
//! "not implemented yet" [`ConnectionError`] rather than a wrong answer.
//! `close()` is implemented (it's simple: closes the socket).
//!
//! **Adaptation, disclosed**: the exact Gemini Live API WebSocket message
//! envelopes (`BidiGenerateContentClientContent`/`RealtimeInput`/
//! `ToolResponse`) are Google's public Multimodal Live API wire protocol —
//! not part of `google/adk-python`'s own source (which only ever talks to
//! `google.genai.live.AsyncSession`, an opaque third-party object from this
//! migration's perspective, the same way `google.genai.Client` was for the
//! REST transport in `gemini.rs`). The envelope shapes below are this
//! migration's best-effort reconstruction of that public protocol, built
//! from the same "minimal real subset" discipline used throughout Phase 3
//! — modeling exactly the fields ADK's own dispatch logic in
//! `gemini_llm_connection.py` sends — but unlike the REST `generateContent`
//! body (a simpler, extremely well-known shape), this hasn't been
//! validated against a live Gemini Live endpoint. Treat the envelope field
//! names as "best effort, unverified" until exercised against the real
//! service.
//!
//! **Scope note**: opening the actual connection (the WebSocket handshake
//! to Google's Live endpoint, and the initial `BidiGenerateContentSetup`
//! message) is also deferred, for the same reason — see `gemini.rs`'s
//! module doc. [`GeminiLlmConnection::new`] takes an already-open
//! [`crate::live_connection::LiveWsConnection`], the same way
//! `GeminiApiClient` is constructed independently of any particular
//! `Gemini` instance.

use std::sync::Arc;

use adk_genai::content::{Content, FunctionResponse, Part};
use rusty_serde::Serialize;

use crate::base_llm_connection::{BaseLlmConnection, BoxFuture, ConnectionError, RealtimeInput};
use crate::capabilities::{is_gemini_3_5_live_translate, is_gemini_3_x_live};
use crate::live_connection::LiveWsConnection;
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

/// `utils/content_utils.py::is_audio_part` — pulled forward for
/// [`filter_audio_parts`]. See `MediaBlobStub`'s doc for why `mime_type` is
/// the only inspectable field.
fn is_audio_part(part: &Part) -> bool {
    let mime_starts_with_audio = |blob: &adk_genai::content::MediaBlobStub| {
        blob.mime_type
            .as_deref()
            .map(|m| m.starts_with("audio/"))
            .unwrap_or(false)
    };
    part.inline_data
        .as_ref()
        .map(mime_starts_with_audio)
        .unwrap_or(false)
        || part
            .file_data
            .as_ref()
            .map(mime_starts_with_audio)
            .unwrap_or(false)
}

/// `utils/content_utils.py::filter_audio_parts`.
fn filter_audio_parts(content: &Content) -> Option<Content> {
    if content.parts.is_empty() {
        return None;
    }
    let filtered: Vec<Part> = content
        .parts
        .iter()
        .filter(|part| !is_audio_part(part))
        .cloned()
        .collect();
    if filtered.is_empty() {
        return None;
    }
    Some(Content {
        role: content.role.clone(),
        parts: filtered,
    })
}

/// The Gemini Live model connection. See the module doc for what's
/// implemented in this batch and what's deferred.
pub struct GeminiLlmConnection {
    socket: Arc<LiveWsConnection>,
    is_gemini_3_x_live: bool,
    is_gemini_3_5_live_translate: bool,
}

impl GeminiLlmConnection {
    pub fn new(socket: LiveWsConnection, model_version: Option<&str>) -> Self {
        Self {
            socket: Arc::new(socket),
            is_gemini_3_x_live: is_gemini_3_x_live(model_version),
            is_gemini_3_5_live_translate: is_gemini_3_5_live_translate(model_version),
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

    /// C0139 — deferred. See the module doc.
    fn receive<'a>(&'a self) -> BoxFuture<'a, Result<Vec<LlmResponse>, ConnectionError>> {
        Box::pin(async move {
            Err(ConnectionError::Failed(
                "GeminiLlmConnection::receive() isn't implemented yet — deferred to a later \
                 batch, see gemini_llm_connection.rs's module doc"
                    .to_string(),
            ))
        })
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
    async fn receive_reports_not_implemented_yet() {
        let (connection, _received, server) = connect_for_test();
        let result = connection.receive().await;
        finish(&connection, server);
        assert!(result.is_err());
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
