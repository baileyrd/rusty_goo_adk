//! Capability C0106: `BaseLlmConnection`, ported from
//! `google.adk.models.base_llm_connection`.
//!
//! **Adaptation, Phase 3 batch 5**: `send_realtime`'s `blob` parameter was
//! an opaque `Value` placeholder until `GeminiLlmConnection` (C0138) needed
//! to type-dispatch on it — now [`RealtimeInput`], the source's
//! `Union[types.Blob, types.ActivityStart, types.ActivityEnd,
//! types.LiveClientRealtimeInput]`. Only `LiveClientRealtimeInput`'s
//! `audio_stream_end` field is modeled for the fourth variant: the source
//! itself only handles that one field there too (`gemini_llm_connection.py`
//! logs "Unary LiveClientRealtimeInput not fully supported yet" for
//! anything else), so a dedicated `AudioStreamEnd` variant is a faithful
//! port of what's actually implemented, not a narrowing.

use std::future::Future;
use std::pin::Pin;

use adk_genai::content::{Content, MediaBlobStub};

use crate::llm_response::LlmResponse;

pub(crate) type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum ConnectionError {
    #[error("{0}")]
    Failed(String),
}

/// See the module doc's adaptation note.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeInput {
    /// `types.Blob` — inline binary media (audio/image), sent in realtime.
    Blob(MediaBlobStub),
    ActivityStart,
    ActivityEnd,
    /// The only `types.LiveClientRealtimeInput` field the source handles.
    AudioStreamEnd,
}

/// The base trait for a live model connection.
pub trait BaseLlmConnection: Send + Sync {
    fn send_history<'a>(
        &'a self,
        history: Vec<Content>,
    ) -> BoxFuture<'a, Result<(), ConnectionError>>;

    fn send_content<'a>(&'a self, content: Content) -> BoxFuture<'a, Result<(), ConnectionError>>;

    /// Sends content, optionally as a partial (non-turn-completing) update.
    /// The default implementation ignores `partial` and completes the turn
    /// — connections that support turn-based partial updates override this.
    fn send_content_partial<'a>(
        &'a self,
        content: Content,
        _partial: bool,
    ) -> BoxFuture<'a, Result<(), ConnectionError>> {
        self.send_content(content)
    }

    /// Sends a chunk of audio/video/activity-boundary input in realtime —
    /// C0138 (`GeminiLlmConnection::send_realtime`) type-dispatches on
    /// which [`RealtimeInput`] variant this is.
    fn send_realtime<'a>(
        &'a self,
        input: RealtimeInput,
    ) -> BoxFuture<'a, Result<(), ConnectionError>>;

    fn receive<'a>(&'a self) -> BoxFuture<'a, Result<Vec<LlmResponse>, ConnectionError>>;

    fn close<'a>(&'a self) -> BoxFuture<'a, Result<(), ConnectionError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingConnection {
        turn_completed: std::sync::Mutex<Vec<bool>>,
    }

    impl BaseLlmConnection for RecordingConnection {
        fn send_history<'a>(
            &'a self,
            _history: Vec<Content>,
        ) -> BoxFuture<'a, Result<(), ConnectionError>> {
            Box::pin(async { Ok(()) })
        }

        fn send_content<'a>(
            &'a self,
            _content: Content,
        ) -> BoxFuture<'a, Result<(), ConnectionError>> {
            Box::pin(async {
                self.turn_completed.lock().unwrap().push(true);
                Ok(())
            })
        }

        fn send_realtime<'a>(
            &'a self,
            _input: RealtimeInput,
        ) -> BoxFuture<'a, Result<(), ConnectionError>> {
            Box::pin(async { Ok(()) })
        }

        fn receive<'a>(&'a self) -> BoxFuture<'a, Result<Vec<LlmResponse>, ConnectionError>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn close<'a>(&'a self) -> BoxFuture<'a, Result<(), ConnectionError>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// Parity test for the default `send_content_partial`: it ignores
    /// `partial` and always delegates to `send_content` (turn-completing).
    #[rusty_tokio::test]
    async fn default_send_content_partial_delegates_to_send_content() {
        let connection = RecordingConnection {
            turn_completed: std::sync::Mutex::new(Vec::new()),
        };
        connection
            .send_content_partial(Content::user_text("hi"), true)
            .await
            .unwrap();
        assert_eq!(*connection.turn_completed.lock().unwrap(), vec![true]);
    }
}
