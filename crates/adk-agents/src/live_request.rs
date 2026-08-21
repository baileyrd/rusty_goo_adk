//! Capabilities C0076-C0077: `LiveRequest`/`LiveRequestQueue`, ported from
//! `google.adk.agents.live_request_queue`.
//!
//! **Adaptation**: `content`/`blob`/`activity_start`/`activity_end` are
//! `google.genai.types` payloads — opaque third-party shapes represented as
//! [`rusty_serde::value::Value`] placeholders (same rationale as
//! `run_config`).
//!
//! **First user of the async runtime**: this is the first capability that
//! needs one (Phase 1 explicitly deferred the choice). `rusty_tokio` was
//! chosen over `rustils_async` — see the root `Cargo.toml` comment — for its
//! general-purpose task/channel primitives, which `rustils_async` doesn't
//! provide. `asyncio.Queue`'s `put_nowait`/`get` maps directly onto
//! `rusty_tokio::sync::mpsc`'s unbounded channel: a non-blocking `send` and
//! an async `recv`.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use rusty_tokio::sync::mpsc;

/// Request sent to live agents. When multiple fields are set, they're
/// processed by priority (highest first): `activity_start > activity_end >
/// audio_stream_end > blob > content`. `state_delta`, if set, is always
/// applied regardless of the other fields.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LiveRequest {
    #[rusty_serde(default)]
    pub content: Option<Value>,
    #[rusty_serde(default)]
    pub blob: Option<Value>,
    #[rusty_serde(default)]
    pub activity_start: Option<Value>,
    #[rusty_serde(default)]
    pub activity_end: Option<Value>,
    pub audio_stream_end: bool,
    /// If set, closes the queue.
    pub close: bool,
    pub partial: bool,
    #[rusty_serde(default)]
    pub state_delta: Option<std::collections::BTreeMap<String, Value>>,
}

/// Queue used to send [`LiveRequest`]s in a live (bidirectional-streaming)
/// way. Backed by an unbounded `rusty_tokio` mpsc channel: `send*` methods
/// are non-blocking (mirroring `asyncio.Queue.put_nowait`), `get` is async.
pub struct LiveRequestQueue {
    sender: mpsc::UnboundedSender<LiveRequest>,
    receiver: mpsc::UnboundedReceiver<LiveRequest>,
}

impl Default for LiveRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveRequestQueue {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self { sender, receiver }
    }

    pub fn close(&self) {
        let _ = self.sender.send(LiveRequest {
            close: true,
            ..Default::default()
        });
    }

    pub fn send_content(&self, content: Value, partial: bool) {
        let _ = self.sender.send(LiveRequest {
            content: Some(content),
            partial,
            ..Default::default()
        });
    }

    pub fn send_realtime(&self, blob: Value) {
        let _ = self.sender.send(LiveRequest {
            blob: Some(blob),
            ..Default::default()
        });
    }

    pub fn send_activity_start(&self) {
        let _ = self.sender.send(LiveRequest {
            activity_start: Some(Value::Map(vec![])),
            ..Default::default()
        });
    }

    pub fn send_activity_end(&self) {
        let _ = self.sender.send(LiveRequest {
            activity_end: Some(Value::Map(vec![])),
            ..Default::default()
        });
    }

    pub fn send_audio_stream_end(&self) {
        let _ = self.sender.send(LiveRequest {
            audio_stream_end: true,
            ..Default::default()
        });
    }

    pub fn send(&self, req: LiveRequest) {
        let _ = self.sender.send(req);
    }

    pub async fn get(&mut self) -> Option<LiveRequest> {
        self.receiver.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rusty_tokio::test]
    async fn send_and_get_round_trip_in_fifo_order() {
        let mut queue = LiveRequestQueue::new();
        queue.send_realtime(Value::String("chunk1".to_string()));
        queue.send_audio_stream_end();

        let first = queue.get().await.unwrap();
        assert_eq!(first.blob, Some(Value::String("chunk1".to_string())));

        let second = queue.get().await.unwrap();
        assert!(second.audio_stream_end);
    }

    #[rusty_tokio::test]
    async fn close_sends_a_close_request() {
        let mut queue = LiveRequestQueue::new();
        queue.close();
        let req = queue.get().await.unwrap();
        assert!(req.close);
    }

    #[rusty_tokio::test]
    async fn send_content_carries_the_partial_flag() {
        let mut queue = LiveRequestQueue::new();
        queue.send_content(Value::String("hi".to_string()), true);
        let req = queue.get().await.unwrap();
        assert!(req.partial);
        assert_eq!(req.content, Some(Value::String("hi".to_string())));
    }
}
