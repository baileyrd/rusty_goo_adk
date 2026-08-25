//! Capability C0307: `NodeState`, ported from
//! `google.adk.workflow._node_state`.
//!
//! Per-node execution state — the first real piece of the P7 workflow/
//! graph engine (`BaseNode`/`Workflow`/`Graph`, manifest rows C0294-C0339,
//! ~6,300 Python lines total across `workflow/` + `workflow/utils/`)
//! this port builds. This module, [`crate::workflow_node_status`], and
//! [`crate::workflow_trigger`] together are the pure-data slice of that
//! engine (C0307) — no dependency on `BaseAgent`/`BaseTool`/`Context`/
//! `Event`, or on any other not-yet-built P7 piece.
//!
//! **Crate placement, disclosed**: the source's `workflow/` is its own
//! subpackage; this port flattens it into `adk-agents` directly rather
//! than a new `adk-workflow` crate, for the same reason
//! `auth_*.rs`'s files already flatten `agents/auth/` here — but also for
//! a structural one specific to this subsystem: `base_agent.rs`/
//! `context.rs`/`app.rs` already disclose that `BaseAgent._run_impl`/
//! `Context.run_node`/`App.root_agent` need `workflow::BaseNode` once it
//! exists, so `adk-agents` will eventually depend on wherever `BaseNode`
//! lives; and the workflow orchestrator itself (`_workflow.py`,
//! C0298-C0306, not built — genuinely blocked on C0092) needs
//! `agents.context.Context` directly. Two crates depending on each other
//! isn't expressible in Cargo — the same crate-cycle shape C0355/C0356
//! already hit and disclosed for `adk-tools`/`adk-agents`. Landing P7 in
//! `adk-agents` from the start avoids walking into a second instance of
//! that mistake once a later batch reaches the orchestrator.
//!
//! **Scope, this batch**: only the pure data/error/retry primitives
//! (C0307/C0308/C0309/C0324) — `BaseNode`/`Graph`/`Edge` (C0294/C0295/
//! C0297) and everything downstream of them are separate, larger,
//! independently-shippable follow-up batches.

use std::collections::BTreeMap;

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::workflow_node_status::NodeStatus;

fn default_attempt_count() -> i64 {
    1
}

fn default_value_null() -> Value {
    Value::Null
}

fn is_one(value: &i64) -> bool {
    *value == 1
}

fn is_zero(value: &i64) -> bool {
    *value == 0
}

/// `workflow._node_state.NodeState` — state of a node in the workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct NodeState {
    /// The run status of the node.
    #[rusty_serde(default)]
    pub status: NodeStatus,
    /// The input provided to the node.
    #[rusty_serde(default = "default_value_null")]
    pub input: Value,
    /// The attempt count for this node run (1-based). Excluded from
    /// serialization when it's still the default (`1`), matching the
    /// source's `exclude_if=lambda v: v == 1`.
    #[rusty_serde(default = "default_attempt_count", skip_serializing_if = "is_one")]
    pub attempt_count: i64,
    /// The interrupt ids that are pending to be resolved.
    #[rusty_serde(default)]
    pub interrupts: Vec<String>,
    /// The responses for resuming the node, keyed by interrupt id.
    #[rusty_serde(default)]
    pub resume_inputs: BTreeMap<String, Value>,
    /// Sequential counter incremented each time the node gets a fresh
    /// run. Preserving this count independently of `run_id` prevents
    /// path collisions if a node switches between custom string IDs and
    /// auto-generated numeric IDs. Excluded from serialization when
    /// it's still the default (`0`), matching the source's
    /// `exclude_if=lambda v: v == 0`.
    #[rusty_serde(default, skip_serializing_if = "is_zero")]
    pub run_counter: i64,
    /// The run ID of this node run.
    #[rusty_serde(default)]
    pub run_id: Option<String>,
    /// The run ID of the parent node which dynamically scheduled this
    /// node run.
    #[rusty_serde(default)]
    pub parent_run_id: Option<String>,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            status: NodeStatus::default(),
            input: Value::Null,
            attempt_count: 1,
            interrupts: Vec::new(),
            resume_inputs: BTreeMap::new(),
            run_counter: 0,
            run_id: None,
            parent_run_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_source() {
        let state = NodeState::default();
        assert_eq!(state.status, NodeStatus::Inactive);
        assert_eq!(state.attempt_count, 1);
        assert_eq!(state.run_counter, 0);
        assert!(state.interrupts.is_empty());
        assert!(state.resume_inputs.is_empty());
    }

    #[test]
    fn omits_attempt_count_and_run_counter_at_their_defaults() {
        let json = rusty_serde::json::to_string(&NodeState::default()).unwrap();
        assert!(!json.contains("attemptCount"), "{json}");
        assert!(!json.contains("runCounter"), "{json}");
    }

    #[test]
    fn includes_attempt_count_and_run_counter_once_non_default() {
        let state = NodeState {
            attempt_count: 2,
            run_counter: 3,
            ..NodeState::default()
        };
        let json = rusty_serde::json::to_string(&state).unwrap();
        assert!(json.contains("\"attemptCount\":2"), "{json}");
        assert!(json.contains("\"runCounter\":3"), "{json}");
    }

    #[test]
    fn deserializing_with_no_attempt_count_defaults_to_one() {
        let state: NodeState = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(state.attempt_count, 1);
        assert_eq!(state.run_counter, 0);
    }

    #[test]
    fn round_trips_a_fully_populated_state_through_json() {
        let mut resume_inputs = BTreeMap::new();
        resume_inputs.insert("interrupt-1".to_string(), Value::Bool(true));
        let state = NodeState {
            status: NodeStatus::Waiting,
            input: Value::String("hi".to_string()),
            attempt_count: 3,
            interrupts: vec!["interrupt-1".to_string()],
            resume_inputs,
            run_counter: 2,
            run_id: Some("run-1".to_string()),
            parent_run_id: Some("run-0".to_string()),
        };
        let json = rusty_serde::json::to_string(&state).unwrap();
        let back: NodeState = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(state, back);
    }
}
