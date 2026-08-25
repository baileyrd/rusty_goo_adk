//! Capability C0316 (partial — `JoinNode` half): `JoinNode`, ported from
//! `google.adk.workflow._join_node`. Part of the P7 workflow/graph
//! engine — see `workflow_node_state.rs`'s module doc for the standing
//! crate-placement decision.
//!
//! **`_ToolNode`, still out of scope, disclosed**: this file covers only
//! `JoinNode` — `_ToolNode` (wraps a `BaseTool` as a node) needs
//! `BaseTool`, and `adk-tools` (home of `BaseTool`) already depends on
//! `adk-agents`, the same crate-cycle shape already disclosed for
//! `workflow_graph.rs`'s own deferred `build_node`/`parse_edge_items`
//! (C0326/C0327) and C0355/C0356.
//!
//! **`_validate_input_data` override, subsumed**: the source overrides
//! this to route each value of a dict `node_input` through
//! `_validate_schema` against `input_schema`. `workflow_base_node.rs`'s
//! own module doc already discloses `input_schema` as an opaque `Value`
//! placeholder this port never interprets — `_validate_schema` is
//! already a pass-through there, so `JoinNode`'s override would reduce
//! to the exact same no-op `BaseNode::run` already performs. Nothing
//! distinct to port.
//!
//! **`_requires_all_predecessors`, has no reader yet**: only `Workflow`
//! (C0298-C0306, not built) reads this — see
//! `workflow_base_node.rs`'s own doc on the trait method this overrides.

use std::future::Future;
use std::pin::Pin;

use adk_events::branch_path::BranchPath;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use rusty_serde::value::Value;

use crate::context::Context;
use crate::workflow_base_node::{NodeBehavior, NodeRunError, NodeYield};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `_get_common_branch_prefix`: the common prefix of dot-separated
/// branch strings. Defined but never called anywhere in the source
/// (verified: no caller in `workflow/` or elsewhere) — ported anyway
/// per this migration's standing rule that "looks unused" isn't a
/// license to omit a capability, and it costs little: a thin wrapper
/// around the already-shipped [`BranchPath::common_prefix`].
pub fn common_branch_prefix(branches: &[String]) -> String {
    branches
        .iter()
        .map(|b| BranchPath::from_string(b))
        .reduce(|a, b| BranchPath::common_prefix(&a, &b))
        .map(|p| p.to_dotted_string())
        .unwrap_or_default()
}

/// `workflow._join_node.JoinNode`: a node that waits for all specified
/// predecessors to trigger it before outputting — see this module's own
/// doc for what's out of scope and why.
#[derive(Debug, Default)]
pub struct JoinNode;

impl NodeBehavior for JoinNode {
    fn requires_all_predecessors(&self) -> bool {
        true
    }

    fn run_impl<'a>(
        &'a self,
        ctx: &'a mut Context,
        node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
        Box::pin(async move {
            let mut event = Event::new(String::new(), String::new(), NodeInfo::new(""));
            event.output = Some(node_input);
            event.branch = ctx.invocation_context().branch.clone();
            Ok(vec![NodeYield::Event(Box::new(event))])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::BaseNode;

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    #[test]
    fn requires_all_predecessors_is_true() {
        let node = BaseNode::new("join", JoinNode).unwrap();
        assert!(node.requires_all_predecessors());
    }

    #[rusty_tokio::test]
    async fn run_passes_through_the_aggregated_input_as_output() {
        let node = BaseNode::new("join", JoinNode).unwrap();
        let mut ctx = ctx();
        let events = node
            .run(&mut ctx, Value::String("aggregated".to_string()))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].output,
            Some(Value::String("aggregated".to_string()))
        );
    }

    #[test]
    fn common_branch_prefix_of_a_single_branch_is_itself() {
        assert_eq!(common_branch_prefix(&["a.b".to_string()]), "a.b");
    }

    #[test]
    fn common_branch_prefix_narrows_across_divergent_branches() {
        assert_eq!(
            common_branch_prefix(&["a.b.c".to_string(), "a.b.d".to_string()]),
            "a.b"
        );
    }

    #[test]
    fn common_branch_prefix_of_no_branches_is_empty() {
        assert_eq!(common_branch_prefix(&[]), "");
    }
}
