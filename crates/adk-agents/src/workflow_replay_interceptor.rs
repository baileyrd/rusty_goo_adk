//! Capability C0321: replay interception, ported from
//! `google.adk.workflow.utils._replay_interceptor`. Part of the P7
//! workflow/graph engine — see `workflow_rehydration_utils.rs`'s module
//! doc for why this batch (P7 Chunk 4) has no caller yet and is still a
//! legitimate, independently-testable batch.
//!
//! **`current_run`, narrowed out entirely**: the source's `check_interception`
//! takes an optional `current_run: DynamicNodeRun` — populated only by
//! dynamic node dispatch (`Context.run_node()`, C0059/C0060). Confirmed
//! before writing this batch: `DynamicNodeScheduler`/`DynamicNodeRun`
//! (C0318/C0319) are themselves still blocked — their own `__call__`
//! needs `Context._run_node_standalone` to dispatch over an arbitrary
//! [`crate::workflow_base_node::BaseNode`] reference, and this port's
//! `BaseNode` is a concrete struct with no dynamic-dispatch seam. Since
//! nothing in this port can ever construct a same-turn dynamic run
//! record, [`check_interception`] simply omits the parameter rather
//! than threading through an always-`None` placeholder type invented
//! for a subsystem that doesn't exist yet — the source's "Case 1"
//! branch (same-turn completed/waiting interception) is dropped along
//! with it, and Case 5's `current_run is not None` fallback collapses
//! to its `else` arm (`should_run = False`) unconditionally.
//!
//! **`isinstance(node, Workflow)`, narrowed**: `Workflow` itself
//! (C0298-C0306) isn't built in this port, so no [`BaseNode`] value can
//! ever be a `Workflow` — this check is dropped from Case 5's condition
//! (leaving `wait_for_output`/`rerun_on_resume`), the same "the type this
//! checks for doesn't exist yet in this port" reasoning already applied
//! to `workflow_graph_validation::validate_chat_agent_wiring`'s
//! `LlmAgent` check.
//!
//! **`create_mock_context` needs no new `Context` surface**: every field
//! it populates (output, route, interrupt-ids, transfer-to-agent) is
//! reachable through [`crate::context::Context`]'s already-shipped,
//! guarded public API (`set_output`/`mark_output_emitted`/`set_route`/
//! `add_interrupt_ids`/`actions_mut`) — a freshly built
//! [`crate::context::Context::for_node`] starts with all of them unset,
//! so there is no "already set" conflict to bypass, unlike the source's
//! raw `_output_value =`/`_interrupt_ids =` field writes.

use std::collections::{BTreeMap, HashSet};

use rusty_serde::value::Value;

use crate::context::Context;
use crate::workflow_base_node::BaseNode;
use crate::workflow_rehydration_utils::{process_rehydrated_output, ChildScanState};

/// `InterceptionResult`: result of a replay interception check.
#[derive(Debug, Clone, Default)]
pub struct InterceptionResult {
    /// Whether the node should be executed natively.
    pub should_run: bool,
    /// The cached output to fast-forward or auto-complete with.
    pub output: Option<Value>,
    /// The cached route to fast-forward with.
    pub route: Option<Value>,
    /// Unresolved interrupts if the node should stay WAITING.
    pub interrupts: HashSet<String>,
    /// Resolved responses to feed into the node if it is rerun.
    pub resume_inputs: Option<BTreeMap<String, Value>>,
    /// Target agent name if fast-forwarding a same-turn transfer.
    pub transfer_to_agent: Option<String>,
}

/// `check_interception`: determines if a node execution should be
/// intercepted based on history. See this module's own doc for the
/// `current_run`/`Workflow` narrowing.
pub fn check_interception(
    node: &BaseNode,
    recovered: Option<&ChildScanState>,
) -> InterceptionResult {
    let Some(recovered) = recovered else {
        return InterceptionResult {
            should_run: true,
            ..Default::default()
        };
    };

    let unresolved: HashSet<String> = recovered
        .interrupt_ids
        .difference(&recovered.resolved_ids)
        .cloned()
        .collect();

    let mut should_run = false;
    let mut output = None;
    let mut route = None;
    let mut interrupts = HashSet::new();
    let mut resume_inputs = None;

    if !unresolved.is_empty() {
        // Case 2: cross-turn unresolved interrupts remain.
        if node.rerun_on_resume() && !recovered.resolved_ids.is_empty() {
            should_run = true;
            resume_inputs = Some(recovered.resolved_responses.clone());
        } else {
            interrupts = unresolved;
        }
    } else if recovered.route.is_some()
        || recovered.output.is_some()
        || recovered.transfer_to_agent.is_some()
    {
        // Case 3: cross-turn successfully completed in a prior turn (fast-forward).
        output = process_rehydrated_output(node, recovered.output.as_ref());
        route = recovered.route.clone();
    } else if !recovered.interrupt_ids.is_empty() {
        // Case 4: cross-turn all prior interrupts are resolved, but no output yet.
        if !node.rerun_on_resume() {
            output = if recovered.resolved_responses.len() == 1 {
                recovered.resolved_responses.values().next().cloned()
            } else {
                Some(Value::Map(
                    recovered
                        .resolved_responses
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ))
            };
        } else {
            should_run = true;
            resume_inputs = Some(recovered.resolved_responses.clone());
        }
    } else {
        // Case 5: cross-turn no events, or events contain no output, route,
        // or interrupts. Rerun `wait_for_output`/`rerun_on_resume` nodes
        // with no prior output so they can guide nested children or resume
        // execution; otherwise fall through to a fresh (fast-forwarded) run.
        if (node.wait_for_output() || node.rerun_on_resume()) && recovered.output.is_none() {
            should_run = true;
            resume_inputs = Some(recovered.resolved_responses.clone());
        } else {
            should_run = false;
        }
    }

    InterceptionResult {
        should_run,
        output,
        route,
        interrupts,
        resume_inputs,
        transfer_to_agent: recovered.transfer_to_agent.clone(),
    }
}

/// `create_mock_context`: builds a `Context` with cached results, with
/// no execution — see this module's own doc for why no new `Context`
/// surface was needed.
#[allow(clippy::too_many_arguments)]
pub fn create_mock_context(
    parent_ctx: &Context,
    node: &BaseNode,
    run_id: impl Into<String>,
    result: &InterceptionResult,
    ancestors: &[String],
    node_path: Option<&str>,
    branch: Option<String>,
) -> Context {
    let mut ic = parent_ctx.invocation_context().clone();
    if let Some(branch) = branch {
        ic.branch = Some(branch);
    }

    let mut mock_ctx = Context::for_node(
        ic,
        node_path.unwrap_or(parent_ctx.node_path()),
        ancestors,
        parent_ctx.isolation_scope().map(str::to_string),
        node.name(),
        run_id,
        BTreeMap::new(),
        1,
        false,
    );

    if let Some(output) = &result.output {
        let _ = mock_ctx.set_output(output.clone());
        mock_ctx.mark_output_emitted();
    }

    if let Some(transfer) = &result.transfer_to_agent {
        mock_ctx.actions_mut().transfer_to_agent = Some(transfer.clone());
    }

    if let Some(route) = &result.route {
        mock_ctx.set_route(route.clone());
    }

    mock_ctx.add_interrupt_ids(result.interrupts.iter().cloned());

    mock_ctx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::NoopNodeBehavior;
    use crate::workflow_rehydration_utils::ChildOutput;
    use crate::workflow_retry_config::RetryConfig;

    fn parent_ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    fn node(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    fn node_with_rerun_on_resume(name: &str) -> BaseNode {
        BaseNode::build(
            name,
            "",
            true,
            false,
            None::<RetryConfig>,
            None,
            None,
            None,
            None,
            NoopNodeBehavior,
        )
        .unwrap()
    }

    #[test]
    fn no_recovered_state_always_runs() {
        let result = check_interception(&node("n"), None);
        assert!(result.should_run);
    }

    #[test]
    fn unresolved_interrupts_with_no_progress_stay_waiting() {
        let mut recovered = ChildScanState::default();
        recovered.interrupt_ids.insert("i1".to_string());
        let result = check_interception(&node("n"), Some(&recovered));
        assert!(!result.should_run);
        assert!(result.interrupts.contains("i1"));
    }

    #[test]
    fn unresolved_interrupts_with_progress_reruns_with_resume_inputs() {
        let mut recovered = ChildScanState::default();
        recovered.interrupt_ids.insert("i1".to_string());
        recovered.interrupt_ids.insert("i2".to_string());
        recovered.resolved_ids.insert("i1".to_string());
        recovered
            .resolved_responses
            .insert("i1".to_string(), Value::String("answer".to_string()));
        let result = check_interception(&node_with_rerun_on_resume("n"), Some(&recovered));
        assert!(result.should_run);
        assert_eq!(
            result.resume_inputs.unwrap().get("i1"),
            Some(&Value::String("answer".to_string()))
        );
    }

    #[test]
    fn a_completed_prior_turn_fast_forwards_the_output() {
        let recovered = ChildScanState {
            output: Some(ChildOutput::Value(Value::String("done".to_string()))),
            ..Default::default()
        };
        let result = check_interception(&node("n"), Some(&recovered));
        assert!(!result.should_run);
        assert_eq!(result.output, Some(Value::String("done".to_string())));
    }

    #[test]
    fn all_interrupts_resolved_without_rerun_extracts_the_single_response() {
        let mut recovered = ChildScanState::default();
        recovered.interrupt_ids.insert("i1".to_string());
        recovered.resolved_ids.insert("i1".to_string());
        recovered
            .resolved_responses
            .insert("i1".to_string(), Value::Int(9));
        let result = check_interception(&node("n"), Some(&recovered));
        assert!(!result.should_run);
        assert_eq!(result.output, Some(Value::Int(9)));
    }

    #[test]
    fn all_interrupts_resolved_with_rerun_reruns_natively() {
        let mut recovered = ChildScanState::default();
        recovered.interrupt_ids.insert("i1".to_string());
        recovered.resolved_ids.insert("i1".to_string());
        recovered
            .resolved_responses
            .insert("i1".to_string(), Value::Int(9));
        let result = check_interception(&node_with_rerun_on_resume("n"), Some(&recovered));
        assert!(result.should_run);
    }

    #[test]
    fn no_prior_outcome_fast_forwards_a_plain_node() {
        let recovered = ChildScanState::default();
        let result = check_interception(&node("n"), Some(&recovered));
        assert!(!result.should_run);
    }

    #[test]
    fn no_prior_outcome_reruns_a_rerun_on_resume_node() {
        let recovered = ChildScanState::default();
        let result = check_interception(&node_with_rerun_on_resume("n"), Some(&recovered));
        assert!(result.should_run);
    }

    #[test]
    fn create_mock_context_populates_output_route_and_interrupts_without_running() {
        let parent = parent_ctx();
        let result = InterceptionResult {
            should_run: false,
            output: Some(Value::String("cached".to_string())),
            route: Some(Value::String("left".to_string())),
            interrupts: HashSet::from(["i1".to_string()]),
            resume_inputs: None,
            transfer_to_agent: Some("other_agent".to_string()),
        };
        let mock_ctx = create_mock_context(&parent, &node("n"), "1", &result, &[], None, None);

        assert_eq!(
            mock_ctx.output(),
            Some(&Value::String("cached".to_string()))
        );
        assert_eq!(mock_ctx.route(), Some(&Value::String("left".to_string())));
        assert!(mock_ctx.interrupt_ids().contains("i1"));
        assert_eq!(
            mock_ctx.actions().transfer_to_agent,
            Some("other_agent".to_string())
        );
    }
}
