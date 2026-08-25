//! Capabilities C0310/C0311/C0312: `NodeRunner`, ported from
//! `google.adk.workflow._node_runner`. Part of the P7 workflow/graph
//! engine — see `workflow_node_state.rs`'s module doc for the standing
//! crate-placement decision.
//!
//! `NodeRunner` drives one [`BaseNode::run`] call, with retry, timeout,
//! and result-tracking — the layer directly below the still-unbuilt
//! `Workflow` orchestrator (C0298-C0306) and above `BaseNode` itself
//! (C0294/C0295, `workflow_base_node.rs`). It needs nothing from
//! `Workflow`, `DynamicNodeScheduler` (C0318/C0319), the replay/
//! rehydration stack (C0320-C0323), or `Graph`/`Edge` — those stay
//! unbuilt without blocking this batch.
//!
//! **"Eagerly-collected `Vec`" adaptation, reused**: the source
//! streams events through a live per-invocation queue
//! (`ctx._invocation_context._enqueue_event`), draining and reacting to
//! partial output as it flows. This port's [`BaseNode::run`] already
//! collects a node's whole result into one `Vec<Event>` (the same
//! adaptation `AgentBehavior::run_async_impl` established), so
//! [`NodeRunner::run`] returns the child [`Context`] plus that same
//! kind of `Vec<Event>` — enriched, in emission order — instead of
//! pushing onto a queue. The source's per-event delta-flush inside
//! `_enqueue_event` (attaching any deltas still pending on `ctx` onto
//! the next *non-partial* event as it streams by) has no equivalent
//! here since nothing in this port emits partial events yet; any
//! deltas left on `ctx` after a node finishes are flushed once, in
//! [`flush_output_and_route`], mirroring the source's own unconditional
//! post-loop `_flush_output_and_deltas` call.
//!
//! **`NodeInterruptedError` catch, disclosed as unreachable today**:
//! the source's `_execute_node` catches `NodeInterruptedError` — raised
//! only by a *dynamic* child node scheduled via `ctx.run_node()`
//! (`context.rs`'s own module doc: dynamic dispatch is C0059/C0060,
//! still deferred). Nothing in this port can raise it yet, so this
//! batch doesn't port a corresponding catch — there is nothing to
//! catch it from. Revisit once `ctx.run_node()` lands.
//!
//! **`WorkflowNodeError::DynamicNodeFail` handling, kept even though
//! unreachable today**: mirrors `validate_chat_agent_wiring`'s
//! precedent in `workflow_graph_validation.rs` — a real, checked branch
//! that happens to never fire yet (nothing in this port currently
//! *produces* a `DynamicNodeFail`, for the same dynamic-dispatch reason
//! above), wired correctly now so a future batch that adds dynamic
//! dispatch doesn't also have to revisit this one.
//!
//! **Exception type name for retry matching, adaptation disclosed**:
//! `workflow_retry_utils.rs` already discloses that `should_retry_node`
//! takes a caller-supplied type name rather than deriving one from an
//! arbitrary error, since Rust has no generic way to recover a source-
//! matching class name from a `Box<dyn std::error::Error>`. `NodeRunner`
//! is exactly that caller: it can name the two node-error kinds this
//! port defines (`NodeTimeoutError`/`DynamicNodeFailError`, matching the
//! source's own class names) via a downcast, but anything else collapses
//! to a generic `"NodeError"` placeholder — `RetryConfig.exceptions`
//! allow-lists naming any other, more specific source exception type
//! can't be matched against a boxed `dyn Error` in this port.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use adk_events::branch_path::BranchPath;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use rusty_serde::value::Value;

use crate::context::Context;
use crate::workflow_base_node::{BaseNode, NodeRunError};
use crate::workflow_errors::WorkflowNodeError;
use crate::workflow_node_state::NodeState;
use crate::workflow_retry_utils::{get_retry_delay, should_retry_node};

/// The type name `NodeRunner` reports to [`should_retry_node`] for a
/// node error it can't identify more specifically — see this module's
/// own doc.
const GENERIC_NODE_ERROR_TYPE_NAME: &str = "NodeError";

/// Bridges [`WorkflowNodeError`] — this port's own sovereign
/// `rusty_err::Error` (see that crate's own module doc), not
/// `std::error::Error` — into [`NodeRunError`] (`Box<dyn
/// std::error::Error + Send + Sync>`). A manual `std::error::Error` impl
/// directly on `WorkflowNodeError` would conflict with the blanket
/// bridge `rusty_err` itself provides in the other direction (any
/// `std::error::Error` automatically becomes a `rusty_err::Error`), so
/// this wraps it in a local newtype instead of touching the
/// already-shipped `workflow_errors.rs`.
#[derive(Debug)]
struct BoxedWorkflowNodeError(WorkflowNodeError);

impl std::fmt::Display for BoxedWorkflowNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for BoxedWorkflowNodeError {}

fn workflow_error(error: WorkflowNodeError) -> NodeRunError {
    Box::new(BoxedWorkflowNodeError(error))
}

fn error_type_name(err: &NodeRunError) -> String {
    match err.downcast_ref::<BoxedWorkflowNodeError>().map(|e| &e.0) {
        Some(WorkflowNodeError::Timeout { .. }) => "NodeTimeoutError".to_string(),
        Some(WorkflowNodeError::DynamicNodeFail { .. }) => "DynamicNodeFailError".to_string(),
        None => GENERIC_NODE_ERROR_TYPE_NAME.to_string(),
    }
}

/// `workflow._node_runner.NodeRunner`: per-node executor. Drives
/// [`BaseNode::run`], enriches the resulting events, and returns the
/// child [`Context`] carrying output/route/interrupt-ids.
pub struct NodeRunner {
    node: BaseNode,
    run_id: String,
    use_as_output: bool,
    prior_output: Option<Value>,
    prior_interrupt_ids: HashSet<String>,
    use_sub_branch: bool,
    override_branch: Option<String>,
    override_isolation_scope: Option<String>,
}

impl NodeRunner {
    pub fn new(node: BaseNode) -> Self {
        Self {
            node,
            run_id: "1".to_string(),
            use_as_output: false,
            prior_output: None,
            prior_interrupt_ids: HashSet::new(),
            use_sub_branch: false,
            override_branch: None,
            override_isolation_scope: None,
        }
    }

    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    pub fn with_use_as_output(mut self, value: bool) -> Self {
        self.use_as_output = value;
        self
    }

    pub fn with_prior_output(mut self, value: Option<Value>) -> Self {
        self.prior_output = value;
        self
    }

    pub fn with_prior_interrupt_ids(mut self, value: HashSet<String>) -> Self {
        self.prior_interrupt_ids = value;
        self
    }

    pub fn with_sub_branch(mut self, value: bool) -> Self {
        self.use_sub_branch = value;
        self
    }

    pub fn with_override_branch(mut self, value: Option<String>) -> Self {
        self.override_branch = value;
        self
    }

    pub fn with_override_isolation_scope(mut self, value: Option<String>) -> Self {
        self.override_isolation_scope = value;
        self
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// `NodeRunner.run`: drives `node.run()`, retrying on failure per
    /// the node's `retry_config` and enforcing its `timeout`. The
    /// caller reads `ctx.output()`/`ctx.route()`/`ctx.interrupt_ids()`
    /// for the node's results, and the returned `Vec<Event>` for what to
    /// emit — see this module's own doc for the "eagerly-collected Vec"
    /// adaptation this replaces the source's live event queue with.
    pub async fn run(
        &self,
        parent_ctx: &Context,
        node_input: Value,
        resume_inputs: BTreeMap<String, Value>,
    ) -> (Context, Vec<Event>) {
        let mut attempt_count: u32 = 1;
        let mut events: Vec<Event> = Vec::new();
        loop {
            let mut ctx =
                self.create_child_context(parent_ctx, resume_inputs.clone(), attempt_count);
            match self
                .execute_node(&mut ctx, node_input.clone(), &mut events)
                .await
            {
                Ok(()) => {
                    self.flush_output_and_route(&mut ctx, &mut events);
                    return (ctx, events);
                }
                Err(err) => {
                    if let Some(WorkflowNodeError::DynamicNodeFail {
                        message,
                        error_node_path,
                        ..
                    }) = err.downcast_ref::<BoxedWorkflowNodeError>().map(|e| &e.0)
                    {
                        ctx.set_error(message.clone(), error_node_path.clone());
                        return (ctx, events);
                    }

                    let mut error_event =
                        Event::new(String::new(), String::new(), NodeInfo::new(""));
                    error_event.error_code = Some(error_type_name(&err));
                    error_event.error_message = Some(err.to_string());
                    self.enrich_event(&mut error_event, &mut ctx);
                    events.push(error_event);

                    if self.attempt_retry(&err, attempt_count).await {
                        attempt_count += 1;
                        continue;
                    }

                    let node_path = ctx.node_path().to_string();
                    ctx.set_error(err.to_string(), node_path);
                    return (ctx, events);
                }
            }
        }
    }

    /// `NodeRunner._attempt_retry`.
    async fn attempt_retry(&self, err: &NodeRunError, attempt_count: u32) -> bool {
        let node_state = NodeState {
            attempt_count: attempt_count as i64,
            ..Default::default()
        };
        let exception_type_name = error_type_name(err);
        if !should_retry_node(&exception_type_name, self.node.retry_config(), &node_state) {
            return false;
        }
        let delay = get_retry_delay(self.node.retry_config(), &node_state);
        rusty_tokio::time::sleep(Duration::from_secs_f64(delay.max(0.0))).await;
        true
    }

    /// `NodeRunner._create_child_context` (minus session-history
    /// rehydration, `workflow_graph.rs`-style disclosed deferral: this
    /// port's `resume_inputs` are only ever what the caller explicitly
    /// passes in via [`Self::run`], never reconstructed by scanning
    /// past session events — that needs `_reconstruct_node_states`,
    /// which itself needs the not-yet-built replay stack, C0320-C0323).
    fn create_child_context(
        &self,
        parent_ctx: &Context,
        resume_inputs: BTreeMap<String, Value>,
        attempt_count: u32,
    ) -> Context {
        let mut ic = parent_ctx.invocation_context().clone();
        let base_branch = self.override_branch.clone().or_else(|| ic.branch.clone());
        if self.use_sub_branch {
            let base = BranchPath::from_string(base_branch.as_deref().unwrap_or(""));
            ic.branch = Some(
                base.create_sub_branch(self.node.name(), Some(self.run_id.clone()))
                    .to_dotted_string(),
            );
        } else if let Some(branch) = &self.override_branch {
            ic.branch = Some(branch.clone());
        }

        let mut ctx = Context::for_node(
            ic,
            parent_ctx.node_path(),
            parent_ctx.output_for_ancestors(),
            parent_ctx.isolation_scope().map(str::to_string),
            self.node.name(),
            self.run_id.clone(),
            resume_inputs,
            attempt_count,
            self.use_as_output,
        );

        if let Some(scope) = &self.override_isolation_scope {
            ctx.set_isolation_scope(Some(scope.clone()));
        }

        if let Some(output) = &self.prior_output {
            let _ = ctx.set_output(output.clone());
            ctx.mark_output_emitted();
        }
        if !self.prior_interrupt_ids.is_empty() {
            ctx.add_interrupt_ids(self.prior_interrupt_ids.iter().cloned());
        }

        ctx
    }

    /// `NodeRunner._execute_node`.
    async fn execute_node(
        &self,
        ctx: &mut Context,
        node_input: Value,
        events: &mut Vec<Event>,
    ) -> Result<(), NodeRunError> {
        match self.node.timeout() {
            Some(timeout) => {
                let duration = Duration::from_secs_f64(timeout.max(0.0));
                match rusty_tokio::time::timeout(
                    duration,
                    self.run_node_loop(ctx, node_input, events),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_elapsed) => Err(workflow_error(WorkflowNodeError::Timeout {
                        node_name: self.node.name().to_string(),
                        timeout,
                    })),
                }
            }
            None => self.run_node_loop(ctx, node_input, events).await,
        }
    }

    /// `NodeRunner._run_node_loop`, folded together with `_enqueue_event`'s
    /// remaining (non-deferred) behavior — see this module's own doc for
    /// what that drops.
    async fn run_node_loop(
        &self,
        ctx: &mut Context,
        node_input: Value,
        events: &mut Vec<Event>,
    ) -> Result<(), NodeRunError> {
        let yielded = self.node.run(ctx, node_input).await?;
        for mut event in yielded {
            self.track_event_in_context(&event, ctx)?;
            self.enrich_event(&mut event, ctx);
            events.push(event);
        }
        Ok(())
    }

    /// `NodeRunner._track_event_in_context`. State-delta schema
    /// validation is not ported: `state.rs`'s own module doc already
    /// discloses this port has no per-key state-schema mechanism to
    /// validate against.
    fn track_event_in_context(&self, event: &Event, ctx: &mut Context) -> Result<(), NodeRunError> {
        // `ContextError` is this port's own sovereign `rusty_err::Error`
        // (see `rusty_err`'s own module doc), not `std::error::Error` —
        // converted to a plain string-backed `NodeRunError` here rather
        // than a dedicated bridge type, since nothing downstream needs
        // to downcast back to `ContextError` specifically.
        if let Some(output) = &event.output {
            ctx.set_output(output.clone())
                .map_err(|e| -> NodeRunError { e.to_string().into() })?;
            ctx.mark_output_emitted();
        } else if event.node_info.message_as_output.unwrap_or(false) {
            if let Some(content) = &event.content {
                // Matches `workflow_hitl_utils.rs`'s own precedent for this
                // exact `to_value` call: an already-well-formed `Content`
                // value serializing to JSON essentially can't fail, so a
                // failure here just falls back to `Value::Null` rather
                // than aborting the node run.
                let value = rusty_serde::json::to_value(content).unwrap_or(Value::Null);
                ctx.set_output(value)
                    .map_err(|e| -> NodeRunError { e.to_string().into() })?;
                ctx.mark_output_emitted();
            }
        }

        if let Some(ids) = &event.long_running_tool_ids {
            ctx.add_interrupt_ids(ids.iter().cloned());
        }

        let is_native_node_event = event.author.is_empty() || event.author == self.node.name();
        if is_native_node_event {
            if let Some(route) = &event.actions.route {
                ctx.set_route(route.clone());
                ctx.mark_route_emitted();
            }
            if let Some(transfer) = &event.actions.transfer_to_agent {
                ctx.actions_mut().transfer_to_agent = Some(transfer.clone());
            }
        }

        Ok(())
    }

    /// `NodeRunner._enrich_event`: sets author/invocation-id/node-path/
    /// branch/output-for/isolation-scope on an event about to be
    /// emitted.
    fn enrich_event(&self, event: &mut Event, ctx: &mut Context) {
        event.author = if !ctx.event_author().is_empty() {
            ctx.event_author().to_string()
        } else {
            self.node.name().to_string()
        };
        event.invocation_id = ctx.invocation_context().invocation_id.clone();
        event.node_info.path = ctx.node_path().to_string();

        match event.branch.take() {
            None => event.branch = ctx.invocation_context().branch.clone(),
            Some(branch) if branch.is_empty() => {
                event.branch = None;
                ctx.invocation_context_mut().branch = None;
            }
            Some(branch) => {
                ctx.invocation_context_mut().branch = Some(branch.clone());
                event.branch = Some(branch);
            }
        }

        if event.output.is_some() {
            let mut output_for = vec![ctx.node_path().to_string()];
            output_for.extend(ctx.output_for_ancestors().iter().cloned());
            event.node_info.output_for = Some(output_for);
        }

        if event.isolation_scope.is_none() {
            if let Some(scope) = ctx.isolation_scope() {
                event.isolation_scope = Some(scope.to_string());
            }
        }
    }

    /// `NodeRunner._flush_output_and_deltas` (called unconditionally on
    /// a successful run, matching the source) — see this module's own
    /// doc for the per-event-vs-trailing-flush adaptation.
    fn flush_output_and_route(&self, ctx: &mut Context, events: &mut Vec<Event>) {
        let output_value = ctx.output().cloned();
        let route_value = ctx.route().cloned();
        let has_deferred_output =
            output_value.is_some() && !ctx.output_emitted() && !ctx.output_delegated();
        let has_unflushed_route = route_value.is_some() && !ctx.route_emitted();
        let state_delta: std::collections::HashMap<String, Value> =
            ctx.state_mut().take_delta().into_iter().collect();
        let artifact_delta = std::mem::take(&mut ctx.actions_mut().artifact_delta);
        let has_deltas = !state_delta.is_empty() || !artifact_delta.is_empty();

        if !has_deferred_output && !has_deltas && !has_unflushed_route {
            return;
        }

        let mut event = Event::new(String::new(), String::new(), NodeInfo::new(""));
        if has_deferred_output {
            event.output = output_value;
        }
        if has_unflushed_route {
            event.actions.route = route_value;
        }
        if has_deltas {
            event.actions.state_delta = state_delta;
            event.actions.artifact_delta = artifact_delta;
        }

        self.enrich_event(&mut event, ctx);
        events.push(event);

        if has_deferred_output {
            ctx.mark_output_emitted();
        }
        if has_unflushed_route {
            ctx.mark_route_emitted();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::{NodeBehavior, NodeRunError as NRE, NodeYield};
    use crate::workflow_retry_config::RetryConfig;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    fn root_ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    struct YieldsData(Value);
    impl NodeBehavior for YieldsData {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NRE>> {
            let value = self.0.clone();
            Box::pin(async move { Ok(vec![NodeYield::Data(value)]) })
        }
    }

    #[rusty_tokio::test]
    async fn run_returns_the_nodes_output_and_an_enriched_event() {
        let node = BaseNode::new("greeter", YieldsData(Value::String("hi".to_string()))).unwrap();
        let runner = NodeRunner::new(node);
        let parent = root_ctx();
        let (ctx, events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        assert_eq!(ctx.output(), Some(&Value::String("hi".to_string())));
        assert_eq!(ctx.node_path(), "greeter@1");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "greeter");
        assert_eq!(events[0].node_info.path, "greeter@1");
        assert_eq!(events[0].output, Some(Value::String("hi".to_string())));
    }

    struct AlwaysFails;
    impl NodeBehavior for AlwaysFails {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NRE>> {
            Box::pin(async { Err("boom".into()) })
        }
    }

    #[rusty_tokio::test]
    async fn run_retries_per_the_retry_config_then_gives_up() {
        let node = BaseNode::build(
            "flaky",
            "",
            false,
            false,
            Some(RetryConfig {
                max_attempts: Some(2),
                initial_delay: Some(0.0),
                max_delay: Some(0.0),
                backoff_factor: Some(1.0),
                jitter: Some(0.0),
                exceptions: None,
            }),
            None,
            None,
            None,
            None,
            AlwaysFails,
        )
        .unwrap();
        let runner = NodeRunner::new(node);
        let parent = root_ctx();
        let (ctx, events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        // One error event per attempt (2 attempts total, matching
        // max_attempts=2), then the run gives up.
        assert_eq!(events.len(), 2);
        assert!(events
            .iter()
            .all(|e| e.error_message.as_deref() == Some("boom")));
        assert_eq!(ctx.error_message(), Some("boom"));
    }

    struct FailsThenSucceeds {
        attempts: Arc<AtomicU32>,
    }
    impl NodeBehavior for FailsThenSucceeds {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NRE>> {
            let attempts = self.attempts.clone();
            Box::pin(async move {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err("transient".into())
                } else {
                    Ok(vec![NodeYield::Data(Value::String(
                        "recovered".to_string(),
                    ))])
                }
            })
        }
    }

    #[rusty_tokio::test]
    async fn run_succeeds_after_a_retried_failure() {
        let attempts = Arc::new(AtomicU32::new(0));
        let node = BaseNode::build(
            "recovers",
            "",
            false,
            false,
            Some(RetryConfig {
                max_attempts: Some(5),
                initial_delay: Some(0.0),
                max_delay: Some(0.0),
                backoff_factor: Some(1.0),
                jitter: Some(0.0),
                exceptions: None,
            }),
            None,
            None,
            None,
            None,
            FailsThenSucceeds {
                attempts: attempts.clone(),
            },
        )
        .unwrap();
        let runner = NodeRunner::new(node);
        let parent = root_ctx();
        let (ctx, events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        assert_eq!(ctx.output(), Some(&Value::String("recovered".to_string())));
        assert_eq!(ctx.error_message(), None);
        // One error event from the first attempt, one output event from the second.
        assert_eq!(events.len(), 2);
        assert!(events[0].error_message.is_some());
        assert_eq!(
            events[1].output,
            Some(Value::String("recovered".to_string()))
        );
    }

    struct SetsRoute;
    impl NodeBehavior for SetsRoute {
        fn run_impl<'a>(
            &'a self,
            ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NRE>> {
            ctx.set_route(Value::String("left".to_string()));
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[rusty_tokio::test]
    async fn a_directly_set_route_is_flushed_as_a_trailing_event() {
        let node = BaseNode::new("router", SetsRoute).unwrap();
        let runner = NodeRunner::new(node);
        let parent = root_ctx();
        let (ctx, events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        assert_eq!(ctx.route(), Some(&Value::String("left".to_string())));
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].actions.route,
            Some(Value::String("left".to_string()))
        );
    }

    struct YieldsRequestInput;
    impl NodeBehavior for YieldsRequestInput {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NRE>> {
            Box::pin(async {
                Ok(vec![NodeYield::RequestInput(
                    adk_events::RequestInput::new(None, None, None),
                )])
            })
        }
    }

    #[rusty_tokio::test]
    async fn a_request_input_yield_populates_interrupt_ids() {
        let node = BaseNode::new("asks", YieldsRequestInput).unwrap();
        let runner = NodeRunner::new(node);
        let parent = root_ctx();
        let (ctx, events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        assert_eq!(ctx.interrupt_ids().len(), 1);
        assert_eq!(events.len(), 1);
    }

    #[rusty_tokio::test]
    async fn child_context_inherits_prior_output_and_interrupt_ids_on_resume() {
        let node = BaseNode::new("resumed", crate::workflow_base_node::NoopNodeBehavior).unwrap();
        let mut prior_ids = HashSet::new();
        prior_ids.insert("interrupt-1".to_string());
        let runner = NodeRunner::new(node)
            .with_prior_output(Some(Value::String("carried".to_string())))
            .with_prior_interrupt_ids(prior_ids);
        let parent = root_ctx();
        let (ctx, _events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;

        assert_eq!(ctx.output(), Some(&Value::String("carried".to_string())));
        assert!(ctx.interrupt_ids().contains("interrupt-1"));
    }

    #[rusty_tokio::test]
    async fn node_path_nests_under_the_parents_path() {
        let node = BaseNode::new("child", crate::workflow_base_node::NoopNodeBehavior).unwrap();
        let runner = NodeRunner::new(node).with_run_id("2");
        let mut parent = root_ctx();
        // Simulate a parent that is itself already a node context.
        parent = Context::for_node(
            parent.invocation_context().clone(),
            "",
            &[],
            None,
            "parent",
            "1",
            BTreeMap::new(),
            1,
            false,
        );
        let (ctx, _events) = runner.run(&parent, Value::Null, BTreeMap::new()).await;
        assert_eq!(ctx.node_path(), "parent@1/child@2");
    }
}
