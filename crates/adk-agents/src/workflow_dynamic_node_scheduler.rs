//! Capabilities C0318/C0319: `ScheduleDynamicNode`/`DynamicNodeScheduler`,
//! ported from `google.adk.workflow._schedule_dynamic_node`/
//! `_dynamic_node_scheduler`. Part of the P7 workflow/graph engine — see
//! `workflow_node_state.rs`'s module doc for the standing crate-placement
//! decision.
//!
//! **`ScheduleDynamicNode`, not ported as a formal trait**: the source
//! defines it as a `Protocol` purely so `Context._run_node_internal` can
//! type-hint "whatever object `Context._workflow_scheduler` holds" —
//! `DynamicNodeScheduler` is its only real implementor, and tests mock
//! the protocol rather than implementing a second one. This port's
//! [`Context::run_node`](crate::context::Context::run_node) calls
//! [`DynamicNodeScheduler::call`] directly (a concrete method, not a
//! trait object) — there is no second implementor to abstract over, and
//! introducing a trait here would just be ceremony around one impl.
//!
//! **`DynamicNodeState`, fused into `DynamicNodeScheduler` rather than
//! kept as a separate struct**: the source splits them so `Workflow`'s
//! own `_LoopState` (a *different* struct, not built yet — C0298) can
//! subclass/reuse the same tracking fields. Since nothing in this port
//! reuses this state a second way yet, [`DynamicNodeScheduler`] holds
//! `runs`/`interrupt_ids`/`replay_manager` directly. If `Workflow`'s own
//! loop state needs to share this exact shape later, split it back out
//! then — narrower than speculatively splitting it now for a consumer
//! that doesn't exist.
//!
//! **No `asyncio.Task`-based concurrent-call dedup**: the source's
//! `DynamicNodeRun.task` lets a *second*, concurrently-running call for
//! the same `node_path` `await` the *first* call's already-in-flight
//! task rather than starting a redundant run — meaningful only when two
//! separate async tasks can call `ctx.run_node()` for the same target at
//! the same time. Structurally unreachable in this port today: nothing
//! spawns concurrent tasks that could race on the same dynamic-node
//! path — [`crate::context::Context::run_node`] always executes and
//! `.await`s a call to completion inline (the same "eagerly collected"
//! adaptation used throughout this port's workflow engine), and
//! `Workflow`'s own concurrent LOOP phase (C0300-C0306, the only thing
//! that could ever drive two such tasks at once) isn't built. So
//! [`DynamicNodeRun`] carries no `task` field — same-turn dedup for a
//! node that already ran earlier *sequentially* in this turn is still
//! fully ported (via [`crate::workflow_replay_interceptor::
//! check_interception`]'s restored "Case 1", widened here to accept
//! this module's `current_run`), just not the genuinely-concurrent race.
//! Revisit once `Workflow`'s LOOP phase can drive real concurrent
//! dispatch.
//!
//! **`node_name` override, narrowed away**: the source lets a caller
//! dispatch the same [`crate::workflow_base_node::BaseNode`] object
//! under a different tracking name (`node.model_copy(update={'name':
//! name})`). This port's `BaseNode` has no post-construction rename —
//! every caller uses `node.name()` as the tracking name, matching every
//! actual call site in this port today (nothing yet needs to dispatch
//! one shared node template under varying names).
//!
//! **`skip_run_id_validation`, narrowed away**: only ever set `True` by
//! `Workflow` itself (`_workflow.py`), not built in this port — the
//! numeric-run-id-rejection check (see
//! [`crate::context::Context::run_node`]'s own doc) is therefore always
//! enforced here.
//!
//! **Input-schema validation, not re-checked here**: `workflow_base_node
//! ::BaseNode`'s own `validate_input_data`/`validate_output_data` are
//! already disclosed no-ops (opaque `Option<Value>` schema fields, see
//! that module's own doc) — nothing for this scheduler to validate
//! against either, so `node._validate_input_data`'s call is dropped.

use std::collections::{BTreeMap, HashSet};

use adk_events::node_path_builder::NodePathBuilder;
use adk_events::Event;
use rusty_serde::value::Value;

use crate::context::{Context, RunNodeError};
use crate::workflow_base_node::BaseNode;
use crate::workflow_node_runner::NodeRunner;
use crate::workflow_node_state::NodeState;
use crate::workflow_node_status::NodeStatus;
use crate::workflow_rehydration_utils::{reconstruct_node_states, ChildScanState};
use crate::workflow_replay_interceptor::{check_interception, create_mock_context};
use crate::workflow_replay_manager::ReplayManager;

/// `DynamicNodeRun`: tracking state, cached output, and (see this
/// module's own doc) no running-task handle for one dynamic node's
/// execution.
#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicNodeRun {
    pub state: NodeState,
    pub output: Option<Value>,
    pub transfer_to_agent: Option<String>,
    pub recovered_state: Option<ChildScanState>,
}

/// `workflow._dynamic_node_scheduler.DynamicNodeScheduler` (with its
/// `DynamicNodeState` fused in — see this module's own doc). Handles
/// [`crate::context::Context::run_node`]'s Mode 1 dispatch: fresh
/// execution, same-turn/cross-turn dedup, and interrupt-resume.
#[derive(Default)]
pub(crate) struct DynamicNodeScheduler {
    runs: BTreeMap<String, DynamicNodeRun>,
    interrupt_ids: HashSet<String>,
    replay_manager: ReplayManager,
}

impl DynamicNodeScheduler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Interrupt ids accumulated across every dynamic node this
    /// scheduler has dispatched — the source's own `_state.interrupt_ids`,
    /// read only by `Workflow._finalize` (C0306, not built) to propagate
    /// them to the `Workflow`'s own context once it completes.
    /// [`crate::context::Context::run_node`] doesn't read this: its own
    /// interrupt handling already works from each call's own returned
    /// `child_ctx.interrupt_ids()` (see that method's own doc), which is
    /// the per-request signal a *caller* of `run_node` needs — this
    /// accumulator is scheduler-wide bookkeeping for a still-unbuilt
    /// consumer. Kept correctly wired now (mirroring every `record_result`
    /// outcome, matching the source) so a future batch building `Workflow`
    /// doesn't also have to revisit this scheduler.
    #[allow(dead_code)]
    pub(crate) fn interrupt_ids(&self) -> &HashSet<String> {
        &self.interrupt_ids
    }

    /// `DynamicNodeScheduler.__call__`: schedule a dynamic node — dedup,
    /// resume, or fresh run. Returns the child context plus every event
    /// its execution produced (empty for a fast-forwarded/cached run) —
    /// see this module's own doc for why events are returned rather
    /// than enqueued.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn call(
        &mut self,
        ctx: &Context,
        node: BaseNode,
        node_input: Value,
        use_as_output: bool,
        run_id: String,
        use_sub_branch: bool,
        override_branch: Option<String>,
        override_isolation_scope: Option<String>,
    ) -> Result<(Context, Vec<Event>), RunNodeError> {
        let node_path = NodePathBuilder::from_string(ctx.node_path())
            .append(node.name(), Some(run_id.clone()))
            .to_slash_string();

        let parent_path = ctx.node_path().to_string();
        if !parent_path.is_empty() {
            self.replay_manager
                .prepare_parent_sequence_barrier(ctx, &parent_path);
        }

        if !self.runs.contains_key(&node_path) {
            self.rehydrate_from_events(ctx, &node_path);
        }

        let (child_ctx, events) = match self
            .check_existing_run(
                ctx,
                &node,
                &node_path,
                &run_id,
                node_input.clone(),
                use_as_output,
                use_sub_branch,
                override_branch.clone(),
                override_isolation_scope.clone(),
            )
            .await?
        {
            Some(result) => result,
            None => {
                self.run_node_internal(
                    ctx,
                    node.clone(),
                    &node_path,
                    &run_id,
                    node_input,
                    use_as_output,
                    true,
                    use_sub_branch,
                    override_branch,
                    override_isolation_scope,
                )
                .await?
            }
        };

        let key = format!("{}@{run_id}", node.name());
        self.replay_manager.advance_sequence(&parent_path, &key);

        Ok((child_ctx, events))
    }

    /// `DynamicNodeScheduler._check_existing_run`: `Ok(None)` means no
    /// tracked run exists yet (the caller should run fresh); `Ok(Some(
    /// ..))` covers both the fast-forwarded/cached path and a rerun
    /// dispatched back through [`Self::run_node_internal`].
    #[allow(clippy::too_many_arguments)]
    async fn check_existing_run(
        &mut self,
        curr_parent_ctx: &Context,
        curr_node: &BaseNode,
        node_path: &str,
        curr_run_id: &str,
        curr_input: Value,
        use_as_output: bool,
        use_sub_branch: bool,
        override_branch: Option<String>,
        override_isolation_scope: Option<String>,
    ) -> Result<Option<(Context, Vec<Event>)>, RunNodeError> {
        let Some(run) = self.runs.get(node_path) else {
            return Ok(None);
        };

        if let Some(recovered) = &run.recovered_state {
            let unresolved: HashSet<&String> = recovered
                .interrupt_ids
                .difference(&recovered.resolved_ids)
                .collect();
            if !recovered.interrupt_ids.is_empty()
                && unresolved.is_empty()
                && curr_node.wait_for_output()
                && !curr_node.rerun_on_resume()
            {
                return Err(RunNodeError::WaitingNodeRerunOnResumeDisabled(
                    node_path.to_string(),
                ));
            }
        }

        let current_run = DynamicNodeRunSnapshot {
            status: run.state.status,
            interrupts: run.state.interrupts.clone(),
            output: run.output.clone(),
            transfer_to_agent: run.transfer_to_agent.clone(),
        };
        let branch = run.recovered_state.as_ref().and_then(|r| r.branch.clone());
        let result =
            check_interception(curr_node, run.recovered_state.as_ref(), Some(&current_run));

        if !result.should_run {
            if !result.interrupts.is_empty() {
                self.interrupt_ids.extend(result.interrupts.clone());
            } else {
                let run = self.runs.get_mut(node_path).expect("checked above");
                run.output = result.output.clone();
                run.transfer_to_agent = result.transfer_to_agent.clone();
            }

            let mock_ctx = create_mock_context(
                curr_parent_ctx,
                curr_node,
                curr_run_id,
                &result,
                &[],
                Some(node_path),
                branch,
            );

            let parent_path = curr_parent_ctx.node_path();
            let key = format!("{}@{curr_run_id}", curr_node_tracking_name(curr_node));
            self.replay_manager
                .wait_sequence(parent_path, &key)
                .await
                .map_err(RunNodeError::SequenceBarrierWait)?;

            return Ok(Some((mock_ctx, Vec::new())));
        }

        let run = self.runs.get_mut(node_path).expect("checked above");
        run.state.resume_inputs = result.resume_inputs.unwrap_or_default();

        let outcome = self
            .run_node_internal(
                curr_parent_ctx,
                curr_node.clone(),
                node_path,
                curr_run_id,
                curr_input,
                use_as_output,
                false,
                use_sub_branch,
                override_branch,
                override_isolation_scope,
            )
            .await?;
        Ok(Some(outcome))
    }

    /// `DynamicNodeScheduler._rehydrate_from_events`: lazily scans
    /// session events for this dynamic node's prior state.
    fn rehydrate_from_events(&mut self, ctx: &Context, node_path: &str) {
        let filtered_events = self
            .replay_manager
            .get_events_for_rehydration(ctx, node_path);
        let Ok(results) = reconstruct_node_states(
            &filtered_events,
            node_path,
            &ctx.invocation_context().invocation_id,
            false,
        ) else {
            return;
        };

        if let Some(target_state) = results.get(node_path) {
            self.runs.insert(
                node_path.to_string(),
                DynamicNodeRun {
                    state: NodeState {
                        run_id: target_state.run_id.clone(),
                        ..Default::default()
                    },
                    recovered_state: Some(target_state.clone()),
                    ..Default::default()
                },
            );
        }
    }

    /// `DynamicNodeScheduler._run_node_internal`: unified runner for
    /// both fresh and resume executions, dispatching through
    /// [`NodeRunner`] directly — the same primitive
    /// [`crate::context::Context::run_node`]'s own Mode 2 (standalone)
    /// path already uses, matching the source's `ctx.
    /// _run_node_standalone` (this scheduler must never recurse back
    /// into the outer transfer loop, only ever run one node).
    #[allow(clippy::too_many_arguments)]
    async fn run_node_internal(
        &mut self,
        ctx: &Context,
        node: BaseNode,
        node_path: &str,
        run_id: &str,
        node_input: Value,
        use_as_output: bool,
        is_fresh: bool,
        use_sub_branch: bool,
        override_branch: Option<String>,
        override_isolation_scope: Option<String>,
    ) -> Result<(Context, Vec<Event>), RunNodeError> {
        let resume_inputs = if is_fresh {
            let state = NodeState {
                status: NodeStatus::Running,
                input: node_input.clone(),
                run_id: Some(run_id.to_string()),
                parent_run_id: Some(ctx.run_id().to_string()),
                ..Default::default()
            };
            self.runs.insert(
                node_path.to_string(),
                DynamicNodeRun {
                    state,
                    ..Default::default()
                },
            );
            BTreeMap::new()
        } else {
            let run = self
                .runs
                .get_mut(node_path)
                .expect("resume requires a tracked run");
            run.state.status = NodeStatus::Running;
            run.state.resume_inputs.clone()
        };

        let runner = NodeRunner::new(node.clone())
            .with_run_id(run_id)
            .with_use_as_output(use_as_output)
            .with_sub_branch(use_sub_branch)
            .with_override_branch(override_branch)
            .with_override_isolation_scope(override_isolation_scope);
        let (child_ctx, events) = runner.run(ctx, node_input, resume_inputs).await;
        self.record_result(node_path, &child_ctx, &node);
        Ok((child_ctx, events))
    }

    /// `DynamicNodeScheduler._record_result`: updates the tracked
    /// run's state after execution (5-way outcome classification:
    /// failed / waiting-with-interrupts / completed-via-transfer /
    /// waiting-for-output-with-none-yet / completed).
    fn record_result(&mut self, node_path: &str, child_ctx: &Context, node: &BaseNode) {
        let run = self
            .runs
            .get_mut(node_path)
            .expect("run_node_internal always tracks a run before calling this");

        if child_ctx.error_message().is_some() {
            run.state.status = NodeStatus::Failed;
        } else if !child_ctx.interrupt_ids().is_empty() {
            let interrupts = child_ctx.interrupt_ids();
            run.state.status = NodeStatus::Waiting;
            run.state.interrupts = interrupts.iter().cloned().collect();
            self.interrupt_ids.extend(interrupts);
        } else if let Some(transfer) = child_ctx.actions().transfer_to_agent.clone() {
            run.state.status = NodeStatus::Completed;
            run.transfer_to_agent = Some(transfer);
        } else if node.wait_for_output()
            && child_ctx.output().is_none()
            && child_ctx.route().is_none()
        {
            run.state.status = NodeStatus::Waiting;
        } else {
            run.state.status = NodeStatus::Completed;
            run.output = child_ctx.output().cloned();
        }
    }
}

/// The subset of [`DynamicNodeRun`] `check_interception`'s "Case 1"
/// (same-turn completed/waiting interception) needs — see
/// [`crate::workflow_replay_interceptor`]'s own doc for why it takes
/// this narrow snapshot rather than the full `DynamicNodeRun`.
pub(crate) struct DynamicNodeRunSnapshot {
    pub status: NodeStatus,
    pub interrupts: Vec<String>,
    pub output: Option<Value>,
    pub transfer_to_agent: Option<String>,
}

fn curr_node_tracking_name(node: &BaseNode) -> &str {
    node.name()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::{NodeBehavior, NodeRunError, NodeYield};
    use std::future::Future;
    use std::pin::Pin;

    type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    fn root_ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    struct Echo;
    impl NodeBehavior for Echo {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move { Ok(vec![NodeYield::Data(node_input)]) })
        }
    }

    #[rusty_tokio::test]
    async fn a_fresh_dynamic_node_runs_and_records_its_output() {
        let ctx = root_ctx();
        let node = BaseNode::new("child", Echo).unwrap();
        let mut scheduler = DynamicNodeScheduler::new();

        let (child_ctx, events) = scheduler
            .call(
                &ctx,
                node,
                Value::String("hi".to_string()),
                false,
                "1".to_string(),
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(child_ctx.output(), Some(&Value::String("hi".to_string())));
        assert_eq!(events.len(), 1);
        assert_eq!(
            scheduler.runs.get("child@1").map(|r| r.state.status),
            Some(NodeStatus::Completed)
        );
    }

    #[rusty_tokio::test]
    async fn a_second_call_for_the_same_path_fast_forwards_the_cached_output() {
        let ctx = root_ctx();
        let node = BaseNode::new("child", Echo).unwrap();
        let mut scheduler = DynamicNodeScheduler::new();

        let (_first, first_events) = scheduler
            .call(
                &ctx,
                node.clone(),
                Value::String("hi".to_string()),
                false,
                "1".to_string(),
                false,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(first_events.len(), 1);

        let (second_ctx, second_events) = scheduler
            .call(
                &ctx,
                node,
                Value::String("hi".to_string()),
                false,
                "1".to_string(),
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(second_ctx.output(), Some(&Value::String("hi".to_string())));
        assert!(
            second_events.is_empty(),
            "a fast-forwarded run should not re-run the node or emit new events"
        );
    }

    struct AsksForInput;
    impl NodeBehavior for AsksForInput {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move {
                Ok(vec![NodeYield::RequestInput(
                    adk_events::RequestInput::new(Some("please confirm".to_string()), None, None),
                )])
            })
        }
    }

    #[rusty_tokio::test]
    async fn a_waiting_dynamic_node_propagates_its_interrupt_ids() {
        let ctx = root_ctx();
        let node = BaseNode::new("waiter", AsksForInput).unwrap();
        let mut scheduler = DynamicNodeScheduler::new();

        let (child_ctx, events) = scheduler
            .call(
                &ctx,
                node,
                Value::Null,
                false,
                "1".to_string(),
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert!(!child_ctx.interrupt_ids().is_empty());
        assert_eq!(
            scheduler.runs.get("waiter@1").map(|r| r.state.status),
            Some(NodeStatus::Waiting)
        );
        assert!(!scheduler.interrupt_ids().is_empty());
    }

    #[rusty_tokio::test]
    async fn rejects_a_stale_waiting_node_with_rerun_on_resume_disabled() {
        // Build a scheduler whose tracked run already has an unresolvable
        // recovered wait state (interrupt_ids present, none unresolved --
        // i.e. all resolved -- matching the guard's own condition), then
        // call again for a node that is `wait_for_output` but not
        // `rerun_on_resume`.
        let ctx = root_ctx();
        let mut scheduler = DynamicNodeScheduler::new();
        let node = BaseNode::build(
            "waiter",
            "",
            false,
            true,
            None,
            None,
            None,
            None,
            None,
            AsksForInput,
        )
        .unwrap();

        let mut recovered = ChildScanState::default();
        recovered.interrupt_ids.insert("i1".to_string());
        recovered.resolved_ids.insert("i1".to_string());
        scheduler.runs.insert(
            "waiter@1".to_string(),
            DynamicNodeRun {
                recovered_state: Some(recovered),
                ..Default::default()
            },
        );

        let result = scheduler
            .call(
                &ctx,
                node,
                Value::Null,
                false,
                "1".to_string(),
                false,
                None,
                None,
            )
            .await;
        let Err(err) = result else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            RunNodeError::WaitingNodeRerunOnResumeDisabled(_)
        ));
    }
}
