//! Part of capability C0078: `ActiveStreamingTool`, ported from
//! `google.adk.agents.active_streaming_tool`.
//!
//! **Adaptation**: the source's `task: Optional[asyncio.Task]` is dropped —
//! a `rusty_tokio::task::JoinHandle` isn't `Clone`/inspectable the way this
//! struct's later `Session`-embedding usage needs, and nothing in this batch
//! consumes it yet. Revisit once a concrete consumer (live-mode tool
//! cancellation, P4) needs to observe or abort the handle.

use crate::live_request::LiveRequestQueue;

/// Manages streaming-tool-related resources during an invocation.
#[derive(Default)]
pub struct ActiveStreamingTool {
    pub stream: Option<LiveRequestQueue>,
}

impl ActiveStreamingTool {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_no_stream() {
        let tool = ActiveStreamingTool::new();
        assert!(tool.stream.is_none());
    }
}
