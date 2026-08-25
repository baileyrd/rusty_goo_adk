//! Capability C0307 (partial): `NodeStatus`, ported from
//! `google.adk.workflow._node_status`.
//!
//! The P7 workflow/graph engine's node-execution status enum. Pure data,
//! no dependency on any other not-yet-built P7 piece — see
//! `workflow_node_state.rs`'s module doc for the rest of this row and why
//! this port lands the whole `workflow/` subpackage as flat modules
//! inside `adk-agents` rather than a separate crate.
//!
//! **Wire format, disclosed**: the source's `NodeStatus` is a plain
//! (non-`str`) `Enum` with int values (`INACTIVE = 0`, ...); under
//! Pydantic v2's default enum serialization the wire form is the bare
//! integer, not a readable string. No cross-P7 consumer of this wire
//! format exists in this port yet (Chunk 1 has no persistence/replay
//! caller — that's Chunk 4/5), so this port serializes [`NodeStatus`] as
//! its variant name (`"INACTIVE"`/`"PENDING"`/...) instead — the same
//! disclosed, purely-cosmetic choice `adk-eval::eval_metrics::EvalStatus`
//! already established for the identical situation (a plain int-valued
//! source `Enum` with no wire consumer yet).

use rusty_serde::{Deserialize, Serialize};

/// `workflow._node_status.NodeStatus` — the status of a node in the
/// workflow graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeStatus {
    /// The node is not ready to be executed.
    #[default]
    Inactive,
    /// The node is ready to be executed.
    Pending,
    /// The node is being executed.
    Running,
    /// The node has been executed successfully.
    Completed,
    /// The node is waiting (e.g. for a user response or re-trigger).
    Waiting,
    /// The node has failed.
    Failed,
    /// The node has been cancelled.
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_inactive() {
        assert_eq!(NodeStatus::default(), NodeStatus::Inactive);
    }

    #[test]
    fn serializes_as_screaming_snake_case() {
        let json = rusty_serde::json::to_string(&NodeStatus::Running).unwrap();
        assert_eq!(json, "\"RUNNING\"");
    }

    #[test]
    fn round_trips_every_variant_through_json() {
        for status in [
            NodeStatus::Inactive,
            NodeStatus::Pending,
            NodeStatus::Running,
            NodeStatus::Completed,
            NodeStatus::Waiting,
            NodeStatus::Failed,
            NodeStatus::Cancelled,
        ] {
            let json = rusty_serde::json::to_string(&status).unwrap();
            let back: NodeStatus = rusty_serde::json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }
}
