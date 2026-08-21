//! Capability C0098: `StreamingMode`, ported from
//! `google.adk.agents._streaming_mode`.

use rusty_serde::{Deserialize, Serialize};

/// Streaming behavior for how an agent returns events as model responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StreamingMode {
    /// Non-streaming (default): one aggregated event per turn.
    #[default]
    None,
    /// Server-Sent Events: partial (streaming-chunk) and aggregated events
    /// are both yielded.
    Sse,
    /// Bidirectional streaming. Not used by the standard `run_async` path —
    /// `run_live()` uses a separate code path that doesn't consult this
    /// field.
    Bidi,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_the_default() {
        assert_eq!(StreamingMode::default(), StreamingMode::None);
    }
}
