//! Capabilities C0048, C0050-C0058, C0059/C0060, C0061-C0065: `Context`
//! (`CallbackContext` is now a unified alias for it), ported from
//! `google.adk.agents.context`.
//!
//! **Adaptation**: `telemetry_context` (Phase 12) is omitted — nothing in
//! this batch reads it.
//!
//! **C0310-C0312 (`NodeRunner`) additions**: `node_path`/`run_id`/
//! `attempt_count`/`resume_inputs`/`output_for_ancestors`/
//! `output_delegated`/the `_output_emitted`/`_route_emitted` flags/
//! `error`/`error_node_path`/`node_rerun_on_resume` are additive fields
//! (via [`Context::for_node`], used by `workflow_node_runner::NodeRunner`
//! and — for `node_rerun_on_resume` — [`Context::run_node`] itself)
//! mirroring the source's own `Context.__init__`'s workflow-execution
//! parameters. `Context::new` (every existing call site) is untouched —
//! these fields default to their root-context values (empty path, run_id
//! `"1"`, attempt 1, nothing delegated/emitted, `node_rerun_on_resume`
//! `true`) exactly as the source's own `_derive_node_path`/`node.
//! rerun_on_resume if node else True` fall back to for a node-less
//! context. `state_schema` inheritance (`node.state_schema` or
//! `parent_ctx.state._schema`) is not ported — `state.rs`'s own module
//! doc already discloses this port has no per-key state schema
//! mechanism at all, so there's nothing to inherit.
//!
//! **C0059/C0060 (`Context::run_node`), what's ported and what isn't**:
//! see [`Context::run_node`]'s own doc for the events-returned-not-
//! enqueued adaptation. `_workflow_scheduler`/`_child_run_counters`
//! (Mode 1's own dynamic-dispatch bookkeeping, C0318/C0319) are now real
//! fields — `workflow_scheduler` inherited-or-created via
//! [`Context::for_node`] (mirroring `_derive_scheduler`), so every node
//! context gets one; only a root (node-less) `Context::new` context has
//! `None`, taking Mode 2 (standalone) for any direct `ctx.run_node()`
//! call. `Context.node`/`Context.parent_ctx` (a permanent ancestry pair
//! on every `Context`) are still deliberately never added —
//! `workflow_transfer_utils.rs`'s own module doc explains the local,
//! per-call [`crate::workflow_transfer_utils::ChainFrame`] chain
//! `run_node`'s own loop builds instead; that adaptation doesn't depend
//! on which dispatch mode is active.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use adk_events::node_path_builder::NodePathBuilder;
use adk_events::ui_widget::UiWidget;
use adk_events::Event;
use adk_events::EventActions;
use rusty_serde::value::Value;

use crate::auth_handler::{AuthHandler, AuthHandlerError};
use crate::invocation_context::InvocationContext;
use crate::services::{self, AuthConfig, AuthCredential};
use crate::state::State;
use crate::workflow_agent_node::AgentNode;
use crate::workflow_base_node::{BaseNode, NodeRunError};
use crate::workflow_dynamic_node_scheduler::DynamicNodeScheduler;
use crate::workflow_errors::WorkflowNodeError;
use crate::workflow_node_runner::{workflow_error, NodeRunner};
use crate::workflow_transfer_utils::{
    resolve_and_derive_transfer_context, ChainFrame, TransferOutcome,
};

#[derive(Debug, rusty_err::Error)]
pub enum ContextError {
    #[error("Output already set. A node can produce at most one output.")]
    OutputAlreadySet,
    #[error("Artifact service is not initialized.")]
    ArtifactServiceUnset,
    #[error("Credential service is not initialized.")]
    CredentialServiceUnset,
    #[error("request_credential requires function_call_id. This method can only be used in a tool context, not a callback context. Consider using save_credential/load_credential instead.")]
    RequestCredentialNeedsFunctionCallId,
    #[error("{0}")]
    AuthHandler(#[from] AuthHandlerError),
    #[error("request_confirmation requires function_call_id. This method can only be used in a tool context.")]
    RequestConfirmationNeedsFunctionCallId,
    #[error("Cannot add session to memory: memory service is not available.")]
    MemoryServiceUnsetForSession,
    #[error("Cannot add events to memory: memory service is not available.")]
    MemoryServiceUnsetForEvents,
    #[error("Cannot add memory: memory service is not available.")]
    MemoryServiceUnsetForMemory,
    #[error("Memory service is not available.")]
    MemoryServiceUnsetForSearch,
    #[error("UI widget with ID '{0}' already exists in the current event actions.")]
    DuplicateUiWidget(String),
}

/// C0048: `CallbackContext` in the source is now a unified alias for
/// `Context` (no longer a distinct class) — mirrored directly here.
pub type CallbackContext = Context;

/// C0059/C0060: [`Context::run_node`]'s keyword-argument bundle — the
/// source's `use_as_output`/`run_id`/`use_sub_branch`/`override_branch`/
/// `override_isolation_scope`/`raise_on_wait` keyword parameters
/// (`node`/`node_input` stay positional method arguments; `Rust` has no
/// keyword arguments to mirror the rest with, so they collect here
/// instead — the same shape [`crate::workflow_node_runner::NodeRunner`]'s
/// own builder already established for an overlapping set of options).
#[derive(Debug, Clone, Default)]
pub struct RunNodeOptions {
    pub use_as_output: bool,
    pub run_id: Option<String>,
    pub use_sub_branch: bool,
    pub override_branch: Option<String>,
    pub override_isolation_scope: Option<String>,
    pub raise_on_wait: bool,
}

/// Events and output accumulated by one [`Context::run_node`] call —
/// see that method's own doc for why events are returned rather than
/// enqueued.
#[derive(Debug, Default)]
pub struct RunNodeOutput {
    pub output: Option<Value>,
    pub events: Vec<Event>,
}

/// [`Context::run_node`]'s two non-error outcomes — see that method's
/// own doc for why "interrupted" is a variant here rather than the
/// source's raised `NodeInterruptedError`.
#[derive(Debug)]
pub enum RunNodeOutcome {
    Completed(RunNodeOutput),
    Interrupted(RunNodeOutput),
}

/// [`Context::run_node`]'s own validation/routing errors — every one of
/// these is a plain, catchable `ValueError` in the source (unlike
/// `DynamicNodeFail`, reused directly from
/// [`crate::workflow_errors::WorkflowNodeError`] instead of duplicated
/// here, so it downcasts identically wherever the source's own
/// `DynamicNodeFailError` already does).
#[derive(Debug, rusty_err::Error)]
pub enum RunNodeError {
    #[error(
        "A node must have rerun_on_resume=true. Reason is that dynamically scheduled nodes might be interrupted, and the workflow wakes-up/re-runs the parent node, so it can get the child node response."
    )]
    RerunOnResumeRequired,
    #[error("Node {0} already has a use_as_output delegate.")]
    OutputAlreadyDelegated(String),
    #[error("Only agents can request an agent transfer.")]
    OnlyAgentsCanTransfer,
    #[error("Agent '{0}' cannot transfer to itself.")]
    SelfTransfer(String),
    #[error("Transfer target agent '{0}' not found.")]
    TransferTargetNotFound(String),
    #[error("Cannot transfer from '{0}' to unrelated agent '{1}'.")]
    UnrelatedTransfer(String, String),
    #[error(
        "Explicit run_id \"{1}\" for node \"{0}\" must contain non-numeric characters to prevent collision with auto-generated IDs."
    )]
    RunIdMustBeNonNumeric(String, String),
    #[error(
        "Node {0} is waiting for output but was called again with rerun_on_resume=False. This would cause it to auto-complete with empty output, which is likely a configuration error. Consider setting rerun_on_resume=True."
    )]
    WaitingNodeRerunOnResumeDisabled(String),
    #[error("{0}")]
    SequenceBarrierWait(String),
}

/// Bridges [`RunNodeError`] into a [`NodeRunError`] — the same
/// `BoxedWorkflowNodeError` workaround `workflow_node_runner.rs`'s own
/// doc explains (a manual `std::error::Error` impl directly on a
/// `rusty_err::Error`-derived type would conflict with the blanket
/// bridge `rusty_err` provides in the other direction).
#[derive(Debug)]
struct BoxedRunNodeError(RunNodeError);

impl std::fmt::Display for BoxedRunNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for BoxedRunNodeError {}

fn run_node_error(error: RunNodeError) -> NodeRunError {
    Box::new(BoxedRunNodeError(error))
}

/// The context within an agent run.
pub struct Context {
    invocation_context: InvocationContext,
    event_actions: EventActions,
    state: State,
    function_call_id: Option<String>,
    isolation_scope: Option<String>,
    output: Option<Value>,
    route: Option<Value>,
    interrupt_ids: HashSet<String>,
    event_author: String,
    tool_confirmation: Option<Value>,
    node_path: String,
    run_id: String,
    attempt_count: u32,
    resume_inputs: BTreeMap<String, Value>,
    output_for_ancestors: Vec<String>,
    output_delegated: bool,
    output_emitted: bool,
    route_emitted: bool,
    error_message: Option<String>,
    error_node_path: String,
    node_rerun_on_resume: bool,
    workflow_scheduler: Option<Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>>>,
    child_run_counters: HashMap<String, u32>,
}

impl Context {
    pub fn new(invocation_context: InvocationContext) -> Self {
        let state = State::new(invocation_context.session.state.clone(), Default::default());
        let isolation_scope = invocation_context.isolation_scope.clone();
        Self {
            state,
            function_call_id: None,
            isolation_scope,
            output: None,
            route: None,
            interrupt_ids: HashSet::new(),
            event_author: String::new(),
            event_actions: EventActions::default(),
            invocation_context,
            tool_confirmation: None,
            node_path: String::new(),
            run_id: "1".to_string(),
            attempt_count: 1,
            resume_inputs: BTreeMap::new(),
            output_for_ancestors: Vec::new(),
            output_delegated: false,
            output_emitted: false,
            route_emitted: false,
            error_message: None,
            error_node_path: String::new(),
            // `node.rerun_on_resume if node else True` — a node-less
            // (root) context defaults `True`.
            node_rerun_on_resume: true,
            // `_derive_scheduler(parent_ctx)`: a root (node-less)
            // context always has `None` — matches Mode 2/standalone
            // dispatch for a direct `ctx.run_node()` call off the root.
            workflow_scheduler: None,
            child_run_counters: HashMap::new(),
        }
    }

    /// C0310-C0312: builds a child `Context` for running one workflow
    /// node under `workflow_node_runner::NodeRunner` — mirrors the
    /// source's `Context(invocation_context, parent_ctx=.., node=..,
    /// run_id=.., resume_inputs=.., attempt_count=.., use_as_output=..)`
    /// path (see this module's own doc for what's omitted and why).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_node(
        invocation_context: InvocationContext,
        parent_node_path: &str,
        parent_output_for_ancestors: &[String],
        parent_isolation_scope: Option<String>,
        node_name: &str,
        run_id: impl Into<String>,
        resume_inputs: BTreeMap<String, Value>,
        attempt_count: u32,
        use_as_output: bool,
        node_rerun_on_resume: bool,
        parent_workflow_scheduler: Option<Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>>>,
    ) -> Self {
        let run_id = run_id.into();
        let node_path = NodePathBuilder::from_string(parent_node_path)
            .append(node_name, Some(run_id.clone()))
            .to_slash_string();
        let output_for_ancestors = if use_as_output {
            let mut ancestors = vec![parent_node_path.to_string()];
            ancestors.extend(parent_output_for_ancestors.iter().cloned());
            ancestors
        } else {
            Vec::new()
        };

        let mut ctx = Self::new(invocation_context);
        ctx.isolation_scope = parent_isolation_scope;
        ctx.node_path = node_path;
        ctx.run_id = run_id;
        ctx.resume_inputs = resume_inputs;
        ctx.attempt_count = attempt_count;
        ctx.output_for_ancestors = output_for_ancestors;
        ctx.node_rerun_on_resume = node_rerun_on_resume;
        // `_derive_scheduler`: inherit the parent's scheduler, or lazily
        // create one — a node context always ends up with `Some`.
        ctx.workflow_scheduler = Some(parent_workflow_scheduler.unwrap_or_else(|| {
            Arc::new(rusty_tokio::sync::Mutex::new(DynamicNodeScheduler::new()))
        }));
        ctx
    }

    pub fn invocation_context(&self) -> &InvocationContext {
        &self.invocation_context
    }

    pub(crate) fn invocation_context_mut(&mut self) -> &mut InvocationContext {
        &mut self.invocation_context
    }

    /// C0310-C0312: this node's path in the workflow graph (empty for a
    /// root, non-node context).
    pub fn node_path(&self) -> &str {
        &self.node_path
    }

    /// C0310-C0312: the execution id assigned to this node run.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// C0310-C0312: how many times this node has been attempted so far
    /// (1-based).
    pub fn attempt_count(&self) -> u32 {
        self.attempt_count
    }

    /// C0310-C0312: inputs for resuming an interrupted node, keyed by
    /// interrupt id.
    pub fn resume_inputs(&self) -> &BTreeMap<String, Value> {
        &self.resume_inputs
    }

    /// C0310-C0312: ancestor node paths this node's output also counts
    /// as the output of, when `use_as_output` was set.
    pub fn output_for_ancestors(&self) -> &[String] {
        &self.output_for_ancestors
    }

    pub(crate) fn output_delegated(&self) -> bool {
        self.output_delegated
    }

    /// C0059/C0060: `Context.run_node`'s own guard — once set, a second
    /// `use_as_output=true` dynamic dispatch on this same context is
    /// rejected (see [`Self::run_node`]).
    pub(crate) fn set_output_delegated(&mut self, value: bool) {
        self.output_delegated = value;
    }

    /// C0059/C0060: whether *this* context's own node (if any) may be
    /// re-run on resume — `Context.run_node`'s prerequisite for dynamic
    /// dispatch (the source's `self._node_rerun_on_resume`, checked
    /// because a dynamically scheduled child might interrupt, and only a
    /// re-runnable parent can wake up and retrieve the child's response).
    pub(crate) fn node_rerun_on_resume(&self) -> bool {
        self.node_rerun_on_resume
    }

    /// C0059/C0060/C0318/C0319: this context's dynamic-node scheduler —
    /// `None` for a root (node-less) context, `Some` (inherited from the
    /// parent, or freshly created) for any node context. Drives
    /// [`Self::run_node`]'s Mode-1-vs-Mode-2 dispatch.
    pub(crate) fn workflow_scheduler(
        &self,
    ) -> Option<Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>>> {
        self.workflow_scheduler.clone()
    }

    pub(crate) fn output_emitted(&self) -> bool {
        self.output_emitted
    }

    pub(crate) fn mark_output_emitted(&mut self) {
        self.output_emitted = true;
    }

    pub(crate) fn route_emitted(&self) -> bool {
        self.route_emitted
    }

    pub(crate) fn mark_route_emitted(&mut self) {
        self.route_emitted = true;
    }

    pub(crate) fn add_interrupt_ids(&mut self, ids: impl IntoIterator<Item = String>) {
        self.interrupt_ids.extend(ids);
    }

    /// C0310-C0312: the error (if any) this node's last run failed with,
    /// and the path of the node that actually raised it (may differ from
    /// [`Self::node_path`] once dynamic node dispatch propagates a
    /// failure up from a descendant — reachable since [`Self::run_node`]
    /// (C0059/C0060) landed: a `DynamicNodeFail` bubbling out of a
    /// dynamically-dispatched node is caught by a future outer
    /// `NodeRunner::run` call the same way any other node error is).
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    pub fn error_node_path(&self) -> &str {
        &self.error_node_path
    }

    pub(crate) fn set_error(&mut self, message: String, node_path: String) {
        self.error_message = Some(message);
        self.error_node_path = node_path;
    }

    pub fn branch(&self) -> Option<&str> {
        self.invocation_context.branch.as_deref()
    }

    pub fn custom_metadata(&self) -> &std::collections::BTreeMap<String, Value> {
        &self.invocation_context.custom_metadata
    }

    /// C0051.
    pub fn function_call_id(&self) -> Option<&str> {
        self.function_call_id.as_deref()
    }

    pub fn set_function_call_id(&mut self, value: Option<String>) {
        self.function_call_id = value;
    }

    /// The tool confirmation of the current tool call, if the inbound
    /// function response carried one. Stays a plain (opaque) `Value` here
    /// rather than a typed `ToolConfirmation` — that type lives in
    /// `adk-tools` (Phase 8), which depends on `adk-agents`, not the
    /// other way around, so this crate can't hold it as a typed field
    /// without a cycle. `adk_tools::function_tool::FunctionTool` narrows
    /// it via `ToolConfirmation`'s own (de)serialization.
    pub fn tool_confirmation(&self) -> Option<&Value> {
        self.tool_confirmation.as_ref()
    }

    pub fn set_tool_confirmation(&mut self, value: Option<Value>) {
        self.tool_confirmation = value;
    }

    /// C0052: internal mechanism — do not use directly outside the
    /// framework (see the source docstring).
    pub fn isolation_scope(&self) -> Option<&str> {
        self.isolation_scope.as_deref()
    }

    pub fn set_isolation_scope(&mut self, value: Option<String>) {
        self.isolation_scope = value;
    }

    /// C0053: the delta-aware state of the current session.
    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    pub fn actions(&self) -> &EventActions {
        &self.event_actions
    }

    pub fn actions_mut(&mut self) -> &mut EventActions {
        &mut self.event_actions
    }

    /// Consumes this `Context`, returning its accumulated event actions —
    /// used by `BaseAgent`'s callback wrapping to build the resulting
    /// `Event`. Syncs `state`'s pending delta into `state_delta` first: the
    /// source's `State._delta` *is* `EventActions.state_delta` (the same
    /// dict object, by reference), so a direct `ctx.state[key] = value`
    /// mutation is automatically visible on `event_actions.state_delta`
    /// there. This port's `State` owns its delta rather than sharing it by
    /// reference, so this sync step reproduces the same end result at the
    /// one point it's actually observed.
    pub fn into_actions(mut self) -> EventActions {
        self.event_actions.state_delta = self.state.delta_map().into_iter().collect();
        self.event_actions
    }

    /// C0054: at most one output per execution.
    pub fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    pub fn set_output(&mut self, value: Value) -> Result<(), ContextError> {
        if self.output.is_some() {
            return Err(ContextError::OutputAlreadySet);
        }
        self.output = Some(value);
        Ok(())
    }

    /// C0055: routing value for conditional edges, independent of output.
    pub fn route(&self) -> Option<&Value> {
        self.route.as_ref()
    }

    pub fn set_route(&mut self, value: Value) {
        self.route = Some(value);
    }

    /// C0056: interrupt IDs accumulated during this execution. Read-only —
    /// returns a copy, matching the source's `set(self._interrupt_ids)`.
    pub fn interrupt_ids(&self) -> HashSet<String> {
        self.interrupt_ids.clone()
    }

    /// C0057.
    pub fn event_author(&self) -> &str {
        &self.event_author
    }

    pub fn set_event_author(&mut self, value: impl Into<String>) {
        self.event_author = value.into();
    }

    /// C0058: a copy of the invocation context with the proxy session and
    /// isolation scope applied.
    pub fn get_invocation_context(&self) -> InvocationContext {
        let mut ctx = self.invocation_context.clone();
        ctx.isolation_scope = self.isolation_scope.clone();
        ctx
    }

    // ------------------------------------------------------------------
    // Dynamic node dispatch (C0059/C0060)
    // ------------------------------------------------------------------

    /// `Context.run_node`: executes `node` dynamically as a child run of
    /// this context, in-place resolving any agent transfer the node (or
    /// a node it transfers to) requests, and returns once the resulting
    /// branch finishes.
    ///
    /// **Mode 1 (`Workflow`-scheduled dispatch) vs Mode 2 (standalone)**:
    /// mirrors the source's `if self._workflow_scheduler:` check — since
    /// every node context now carries a scheduler (inherited or
    /// freshly created, see [`Self::for_node`]), Mode 1 fires for any
    /// dynamic dispatch from a *node's own* context; a direct call on a
    /// root (node-less) `Context::new` context has no scheduler and
    /// takes Mode 2 (standalone, via [`NodeRunner`] directly — the same
    /// primitive [`crate::workflow_dynamic_node_scheduler::
    /// DynamicNodeScheduler`] itself dispatches through internally, so
    /// this never recurses back into Mode 1). The explicit `run_id`
    /// digit-collision check and the `_child_run_counters` auto-increment
    /// are Mode-1-only in the source and stay that way here.
    ///
    /// **Events, returned instead of enqueued**: the source streams
    /// every event straight onto the shared invocation event queue as
    /// it's produced. This port's "eagerly collected `Vec`" adaptation
    /// (`workflow_node_runner.rs`'s own doc) has no shared queue to push
    /// onto, so [`RunNodeOutput::events`] carries everything produced
    /// across every node this call ran (including transfer hops) — the
    /// caller (a [`crate::workflow_base_node::NodeBehavior::run_impl`]
    /// that itself calls `run_node`) is responsible for folding these
    /// into its own returned yields.
    ///
    /// **`NodeInterruptedError`, not raised**: the source raises it (a
    /// deliberately non-catchable `BaseException`, `workflow_errors.rs`'s
    /// own doc) to signal a WAITING child. Since this method's `Result`
    /// is the ordinary catchable kind, that signal is instead a distinct
    /// `Ok` variant, [`RunNodeOutcome::Interrupted`] — a caller cannot
    /// accidentally swallow it via `?`/`.map_err` the way the source
    /// guards against, since it never reaches the `Err` side at all.
    pub async fn run_node(
        &mut self,
        node: BaseNode,
        node_input: Value,
        options: RunNodeOptions,
    ) -> Result<RunNodeOutcome, NodeRunError> {
        if !self.node_rerun_on_resume() {
            return Err(run_node_error(RunNodeError::RerunOnResumeRequired));
        }

        if options.use_as_output {
            if self.output_delegated {
                return Err(run_node_error(RunNodeError::OutputAlreadyDelegated(
                    self.node_path.clone(),
                )));
            }
            self.set_output_delegated(true);
        }

        let scheduler = self.workflow_scheduler();

        let mut chain: Vec<ChainFrame> = Vec::new();
        let mut ctxs: Vec<Context> = Vec::new();
        let mut all_events: Vec<Event> = Vec::new();

        let mut curr_parent_index: Option<usize> = None;
        let mut curr_node = node;
        let mut curr_run_id = options.run_id.clone();
        let mut curr_input = node_input;

        loop {
            let curr_use_as_output = curr_parent_index.is_none() && options.use_as_output;

            let (child_ctx, child_events) = if let Some(scheduler) = &scheduler {
                // Mode 1: dispatch through the (inherited or freshly
                // created) `DynamicNodeScheduler` — resolves the run id
                // first (validated if caller-supplied, auto-incremented
                // on `curr_parent_ctx`'s own counters otherwise).
                let run_id = match &curr_run_id {
                    Some(rid) => {
                        if !rid.is_empty() && rid.chars().all(|c| c.is_ascii_digit()) {
                            return Err(run_node_error(RunNodeError::RunIdMustBeNonNumeric(
                                curr_node.name().to_string(),
                                rid.clone(),
                            )));
                        }
                        rid.clone()
                    }
                    None => {
                        let counters = match curr_parent_index {
                            None => &mut self.child_run_counters,
                            Some(i) => &mut ctxs[i].child_run_counters,
                        };
                        let counter = counters.entry(curr_node.name().to_string()).or_insert(0);
                        *counter += 1;
                        counter.to_string()
                    }
                };

                let parent_ctx: &Context = match curr_parent_index {
                    None => &*self,
                    Some(i) => &ctxs[i],
                };
                scheduler
                    .lock()
                    .await
                    .call(
                        parent_ctx,
                        curr_node.clone(),
                        curr_input.clone(),
                        curr_use_as_output,
                        run_id,
                        options.use_sub_branch,
                        options.override_branch.clone(),
                        options.override_isolation_scope.clone(),
                    )
                    .await
                    .map_err(run_node_error)?
            } else {
                // Mode 2: standalone, via `NodeRunner` directly.
                let mut runner = NodeRunner::new(curr_node.clone())
                    .with_use_as_output(curr_use_as_output)
                    .with_sub_branch(options.use_sub_branch)
                    .with_override_branch(options.override_branch.clone())
                    .with_override_isolation_scope(options.override_isolation_scope.clone());
                if let Some(run_id) = &curr_run_id {
                    runner = runner.with_run_id(run_id.clone());
                }

                let parent_ctx: &Context = match curr_parent_index {
                    None => &*self,
                    Some(i) => &ctxs[i],
                };
                runner
                    .run(parent_ctx, curr_input.clone(), BTreeMap::new())
                    .await
            };
            all_events.extend(child_events);

            let transfer_to_agent = child_ctx.actions().transfer_to_agent.clone();

            if let Some(error_message) = child_ctx.error_message() {
                return Err(workflow_error(WorkflowNodeError::DynamicNodeFail {
                    message: format!("Dynamic node {} failed", curr_node.name()),
                    error: error_message.to_string().into(),
                    error_node_path: child_ctx.error_node_path().to_string(),
                }));
            }

            if !child_ctx.interrupt_ids().is_empty() {
                if curr_parent_index.is_none() {
                    self.add_interrupt_ids(child_ctx.interrupt_ids());
                }
                return Ok(RunNodeOutcome::Interrupted(RunNodeOutput {
                    output: None,
                    events: all_events,
                }));
            }
            if options.raise_on_wait
                && child_ctx.output().is_none()
                && transfer_to_agent.is_none()
                && curr_node.wait_for_output()
            {
                return Ok(RunNodeOutcome::Interrupted(RunNodeOutput {
                    output: None,
                    events: all_events,
                }));
            }

            let Some(target_name) = transfer_to_agent else {
                return Ok(RunNodeOutcome::Completed(RunNodeOutput {
                    output: child_ctx.output().cloned(),
                    events: all_events,
                }));
            };

            let Some(current_agent) = curr_node
                .as_any()
                .downcast_ref::<AgentNode>()
                .map(|n| n.agent().clone())
            else {
                return Err(run_node_error(RunNodeError::OnlyAgentsCanTransfer));
            };
            let root_agent = current_agent.root_agent();

            let curr_index = ctxs.len();
            chain.push(ChainFrame {
                node_name: curr_node.name().to_string(),
                parent: curr_parent_index,
            });
            ctxs.push(child_ctx);

            let outcome = resolve_and_derive_transfer_context(
                &target_name,
                &current_agent,
                &root_agent,
                &chain,
                curr_index,
                curr_parent_index,
            )
            .map_err(|e| run_node_error(RunNodeError::SelfTransfer(e.0)))?;

            match outcome {
                TransferOutcome::NotFound => {
                    return Err(run_node_error(RunNodeError::TransferTargetNotFound(
                        target_name,
                    )));
                }
                TransferOutcome::Unrelated { target_agent } => {
                    debug_assert_eq!(target_agent.name(), target_name);
                    return Err(run_node_error(RunNodeError::UnrelatedTransfer(
                        curr_node.name().to_string(),
                        target_name,
                    )));
                }
                TransferOutcome::Resolved {
                    target_agent,
                    next_parent,
                } => {
                    curr_parent_index = next_parent;
                    curr_node =
                        crate::workflow_agent_node::agent_node(target_agent, true, None, None)
                            .map_err(|e| -> NodeRunError { e.to_string().into() })?;
                    curr_run_id = None;
                    curr_input = Value::Null;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Artifact methods (C0061)
    // ------------------------------------------------------------------

    pub async fn load_artifact(
        &self,
        filename: &str,
        version: Option<i64>,
    ) -> Result<Option<Value>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.load_artifact(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            version,
        ))
    }

    pub async fn save_artifact(
        &mut self,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
    ) -> Result<i64, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        let version = service.save_artifact(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            artifact,
            custom_metadata,
        );
        self.event_actions
            .artifact_delta
            .insert(filename.to_string(), version);
        Ok(version)
    }

    pub async fn get_artifact_version(
        &self,
        filename: &str,
        version: Option<i64>,
    ) -> Result<Option<services::ArtifactVersion>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.get_artifact_version(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            version,
        ))
    }

    pub async fn list_artifacts(&self) -> Result<Vec<String>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.list_artifact_keys(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
        ))
    }

    // ------------------------------------------------------------------
    // Credential methods (C0062)
    // ------------------------------------------------------------------

    pub async fn save_credential(&mut self, auth_config: &AuthConfig) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .credential_service
            .clone()
            .ok_or(ContextError::CredentialServiceUnset)?;
        service.save_credential(auth_config, self).await;
        Ok(())
    }

    pub async fn load_credential(
        &self,
        auth_config: &AuthConfig,
    ) -> Result<Option<AuthCredential>, ContextError> {
        let service = self
            .invocation_context
            .credential_service
            .clone()
            .ok_or(ContextError::CredentialServiceUnset)?;
        Ok(service.load_credential(auth_config, self).await)
    }

    /// C0062: gets the auth response credential from session state —
    /// a previously-completed OAuth (or other) flow's stored credential.
    pub fn get_auth_response(&self, auth_config: &AuthConfig) -> Option<AuthCredential> {
        AuthHandler::new(auth_config.clone()).get_auth_response(self.state())
    }

    /// C0062: requests a credential for the current tool call. Requires
    /// `function_call_id` — for callback contexts, use
    /// `save_credential`/`load_credential` instead. Stores
    /// [`AuthHandler::generate_auth_request`]'s result, not `auth_config`
    /// verbatim: for an OAuth2/OIDC scheme this validates the raw
    /// credential and may substitute a freshly generated
    /// `exchanged_auth_credential`, matching the source's own
    /// `AuthHandler(auth_config).generate_auth_request()` call.
    pub fn request_credential(&mut self, auth_config: AuthConfig) -> Result<(), ContextError> {
        let function_call_id = self
            .function_call_id
            .clone()
            .ok_or(ContextError::RequestCredentialNeedsFunctionCallId)?;
        let auth_request = AuthHandler::new(auth_config).generate_auth_request()?;
        // `EventActions.requested_auth_configs` (`adk-events`) is `Value`-typed
        // and out of scope for this batch to widen — serialize the now-real
        // `AuthConfig` on the way in, same as this crate's other
        // real-struct-into-a-`Value`-typed-field sites.
        let auth_config_value = rusty_serde::json::to_value(&auth_request).unwrap_or(Value::Null);
        self.event_actions
            .requested_auth_configs
            .insert(function_call_id, auth_config_value);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Tool methods (C0063)
    // ------------------------------------------------------------------

    /// C0063: requests confirmation for the current tool call. Requires
    /// `function_call_id`.
    pub fn request_confirmation(
        &mut self,
        hint: Option<String>,
        payload: Option<Value>,
    ) -> Result<(), ContextError> {
        let function_call_id = self
            .function_call_id
            .clone()
            .ok_or(ContextError::RequestConfirmationNeedsFunctionCallId)?;
        let mut confirmation = std::collections::BTreeMap::new();
        if let Some(hint) = hint {
            confirmation.insert("hint".to_string(), Value::String(hint));
        }
        if let Some(payload) = payload {
            confirmation.insert("payload".to_string(), payload);
        }
        self.event_actions.requested_tool_confirmations.insert(
            function_call_id,
            Value::Map(confirmation.into_iter().collect()),
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Memory methods (C0064)
    // ------------------------------------------------------------------

    pub async fn add_session_to_memory(&self) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForSession)?;
        service.add_session_to_memory(&self.invocation_context.session);
        Ok(())
    }

    pub async fn add_events_to_memory(
        &self,
        events: &[adk_events::Event],
        custom_metadata: Option<&std::collections::BTreeMap<String, Value>>,
    ) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForEvents)?;
        service.add_events_to_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            events,
            custom_metadata,
        );
        Ok(())
    }

    pub async fn add_memory(
        &self,
        memories: &[services::MemoryEntry],
        custom_metadata: Option<&std::collections::BTreeMap<String, Value>>,
    ) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForMemory)?;
        service.add_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            memories,
            custom_metadata,
        );
        Ok(())
    }

    pub async fn search_memory(
        &self,
        query: &str,
    ) -> Result<services::SearchMemoryResponse, ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForSearch)?;
        Ok(service.search_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            query,
        ))
    }

    // ------------------------------------------------------------------
    // UI widget methods (C0065)
    // ------------------------------------------------------------------

    pub fn render_ui_widget(&mut self, ui_widget: UiWidget) -> Result<(), ContextError> {
        services::render_ui_widget(&mut self.event_actions.render_ui_widgets, ui_widget)
            .map_err(ContextError::DuplicateUiWidget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn context() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    #[test]
    fn output_can_only_be_set_once() {
        let mut ctx = context();
        ctx.set_output(Value::Int(1)).unwrap();
        let err = ctx.set_output(Value::Int(2)).unwrap_err();
        assert!(matches!(err, ContextError::OutputAlreadySet));
        assert_eq!(ctx.output(), Some(&Value::Int(1)));
    }

    #[test]
    fn event_author_defaults_to_empty_string() {
        let ctx = context();
        assert_eq!(ctx.event_author(), "");
    }

    #[test]
    fn interrupt_ids_returns_an_independent_copy() {
        let mut ctx = context();
        ctx.interrupt_ids.insert("i1".to_string());
        let mut copy = ctx.interrupt_ids();
        copy.insert("i2".to_string());
        assert_eq!(
            ctx.interrupt_ids().len(),
            1,
            "mutating the copy must not affect the original"
        );
    }

    fn test_auth_config() -> AuthConfig {
        use crate::auth_schemes::{AuthScheme, CustomAuthScheme};

        AuthConfig::new(
            AuthScheme::Custom(CustomAuthScheme {
                type_: "test".to_string(),
                extra: None,
            }),
            None,
            None,
            Some("key".to_string()),
        )
    }

    #[test]
    fn request_credential_requires_function_call_id() {
        let mut ctx = context();
        let err = ctx.request_credential(test_auth_config()).unwrap_err();
        assert!(matches!(
            err,
            ContextError::RequestCredentialNeedsFunctionCallId
        ));
    }

    #[test]
    fn request_credential_stores_it_keyed_by_function_call_id() {
        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        ctx.request_credential(test_auth_config()).unwrap();
        assert!(ctx.actions().requested_auth_configs.contains_key("fc-1"));
    }

    fn oauth2_auth_config(raw_auth_credential: Option<AuthCredential>) -> AuthConfig {
        use crate::auth_schemes::{
            AuthScheme, OAuth2Scheme, OAuthFlow, OAuthFlows, SecurityScheme,
        };

        AuthConfig::new(
            AuthScheme::Security(Box::new(SecurityScheme::OAuth2(Box::new(OAuth2Scheme {
                description: None,
                flows: OAuthFlows {
                    authorization_code: Some(OAuthFlow {
                        authorization_url: Some("https://example.com/authorize".to_string()),
                        token_url: Some("https://example.com/token".to_string()),
                        refresh_url: None,
                        scopes: Default::default(),
                    }),
                    ..Default::default()
                },
            })))),
            raw_auth_credential,
            None,
            Some("oauth2_key".to_string()),
        )
    }

    #[test]
    fn request_credential_routes_through_auth_handler_and_errors_without_a_raw_credential() {
        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        let err = ctx
            .request_credential(oauth2_auth_config(None))
            .unwrap_err();
        assert!(matches!(err, ContextError::AuthHandler(_)));
    }

    #[test]
    fn request_credential_stores_auth_handlers_generated_request_not_the_input_verbatim() {
        use crate::auth_credential::{AuthCredentialTypes, OAuth2Auth};

        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        let raw_credential = AuthCredential {
            oauth2: Some(OAuth2Auth {
                client_id: Some("id".to_string()),
                client_secret: Some("secret".to_string()),
                ..OAuth2Auth::default()
            }),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        };
        ctx.request_credential(oauth2_auth_config(Some(raw_credential.clone())))
            .unwrap();

        let stored = ctx.actions().requested_auth_configs.get("fc-1").unwrap();
        let stored: AuthConfig = rusty_serde::json::from_value(stored.clone()).unwrap();
        // `generate_auth_uri` (this port always takes the source's own
        // `not AUTHLIB_AVAILABLE` fallback, see `auth_handler.rs`) deep
        // copies the raw credential into `exchanged_auth_credential` --
        // proving `request_credential` stored `AuthHandler`'s output, not
        // `auth_config` verbatim (which had no `exchanged_auth_credential`
        // at all).
        assert_eq!(stored.exchanged_auth_credential, Some(raw_credential));
    }

    #[test]
    fn get_auth_response_returns_none_when_nothing_was_stored() {
        let ctx = context();
        assert_eq!(ctx.get_auth_response(&test_auth_config()), None);
    }

    #[test]
    fn get_auth_response_reads_a_credential_stored_under_the_temp_key() {
        let mut ctx = context();
        let auth_config = test_auth_config();
        let credential = AuthCredential::api_key("secret");
        ctx.state_mut().set(
            "temp:key",
            rusty_serde::json::to_value(&credential).unwrap(),
        );

        assert_eq!(ctx.get_auth_response(&auth_config), Some(credential));
    }

    #[test]
    fn request_confirmation_requires_function_call_id() {
        let mut ctx = context();
        let err = ctx.request_confirmation(None, None).unwrap_err();
        assert!(matches!(
            err,
            ContextError::RequestConfirmationNeedsFunctionCallId
        ));
    }

    #[rusty_tokio::test]
    async fn artifact_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.load_artifact("f", None).await.unwrap_err();
        assert!(matches!(err, ContextError::ArtifactServiceUnset));
    }

    #[rusty_tokio::test]
    async fn memory_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.add_session_to_memory().await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForSession));
        let err = ctx.search_memory("q").await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForSearch));
    }

    #[test]
    fn render_ui_widget_rejects_duplicate_ids() {
        let mut ctx = context();
        ctx.render_ui_widget(UiWidget::new("w1", "mcp", Value::Null))
            .unwrap();
        let err = ctx
            .render_ui_widget(UiWidget::new("w1", "mcp", Value::Null))
            .unwrap_err();
        assert!(matches!(err, ContextError::DuplicateUiWidget(id) if id == "w1"));
    }

    #[test]
    fn state_has_delta_is_false_for_a_freshly_built_context() {
        let ctx = context();
        assert!(!ctx.state().has_delta());
    }

    #[test]
    fn callback_context_is_the_same_type_as_context() {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let callback_ctx: CallbackContext = Context::new(ic);
        assert!(!callback_ctx.state().has_delta());
    }

    #[test]
    fn isolation_scope_can_be_read_and_overridden() {
        let mut ctx = context();
        assert_eq!(ctx.isolation_scope(), None);
        ctx.set_isolation_scope(Some("scope-1".to_string()));
        assert_eq!(ctx.isolation_scope(), Some("scope-1"));
    }

    #[test]
    fn route_is_independent_of_output() {
        let mut ctx = context();
        ctx.set_route(Value::String("branch-a".to_string()));
        ctx.set_output(Value::Int(1)).unwrap();
        assert_eq!(ctx.route(), Some(&Value::String("branch-a".to_string())));
        assert_eq!(ctx.output(), Some(&Value::Int(1)));
    }

    #[test]
    fn event_author_can_be_overridden() {
        let mut ctx = context();
        ctx.set_event_author("workflow");
        assert_eq!(ctx.event_author(), "workflow");
    }

    #[test]
    fn get_invocation_context_applies_the_current_isolation_scope() {
        let mut ctx = context();
        ctx.set_isolation_scope(Some("scope-1".to_string()));
        let copy = ctx.get_invocation_context();
        assert_eq!(copy.isolation_scope, Some("scope-1".to_string()));
    }

    #[rusty_tokio::test]
    async fn credential_methods_raise_when_service_unset() {
        let mut ctx = context();
        let auth_config = test_auth_config();
        let err = ctx.save_credential(&auth_config).await.unwrap_err();
        assert!(matches!(err, ContextError::CredentialServiceUnset));
        let err = ctx.load_credential(&auth_config).await.unwrap_err();
        assert!(matches!(err, ContextError::CredentialServiceUnset));
    }

    #[test]
    fn request_confirmation_stores_it_keyed_by_function_call_id() {
        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        ctx.request_confirmation(Some("pick one".to_string()), None)
            .unwrap();
        assert!(ctx
            .actions()
            .requested_tool_confirmations
            .contains_key("fc-1"));
    }

    #[rusty_tokio::test]
    async fn remaining_memory_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.add_events_to_memory(&[], None).await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForEvents));
        let err = ctx.add_memory(&[], None).await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForMemory));
    }

    struct FakeArtifactService;
    impl services::ArtifactService for FakeArtifactService {
        fn load_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            _version: Option<i64>,
        ) -> Option<Value> {
            Some(Value::String(format!("contents of {filename}")))
        }

        fn save_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _artifact: Value,
            _custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
        ) -> i64 {
            1
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<services::ArtifactVersion> {
            None
        }

        fn list_artifact_keys(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
        ) -> Vec<String> {
            vec!["f.txt".to_string()]
        }

        fn delete_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) {
        }

        fn list_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<i64> {
            vec![0]
        }

        fn list_artifact_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<services::ArtifactVersion> {
            Vec::new()
        }
    }

    #[rusty_tokio::test]
    async fn artifact_methods_delegate_to_a_configured_service() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.artifact_service = Some(std::sync::Arc::new(FakeArtifactService));
        let mut ctx = Context::new(ic);

        let loaded = ctx.load_artifact("f.txt", None).await.unwrap();
        assert_eq!(loaded, Some(Value::String("contents of f.txt".to_string())));

        let version = ctx
            .save_artifact("f.txt", Value::String("data".to_string()), None)
            .await
            .unwrap();
        assert_eq!(version, 1);
        assert_eq!(ctx.actions().artifact_delta.get("f.txt"), Some(&1));

        let keys = ctx.list_artifacts().await.unwrap();
        assert_eq!(keys, vec!["f.txt".to_string()]);
    }

    // ------------------------------------------------------------------
    // C0059/C0060: Context::run_node
    // ------------------------------------------------------------------

    mod run_node_tests {
        use super::*;
        use crate::base_agent::{AgentBehavior, AgentRunError, BaseAgent, NoopBehavior};
        use crate::workflow_base_node::{NodeYield, NoopNodeBehavior};
        use crate::workflow_function_node::{function_node, FunctionNodeBody};
        use adk_events::node_info::NodeInfo;
        use adk_events::RequestInput;
        use std::future::Future;
        use std::pin::Pin;

        type TestBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

        fn ctx_with_rerun(rerun: bool) -> Context {
            let ic =
                InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
            Context::for_node(
                ic,
                "",
                &[],
                None,
                "root",
                "1",
                BTreeMap::new(),
                1,
                false,
                rerun,
                None,
            )
        }

        #[rusty_tokio::test]
        async fn errors_when_this_contexts_own_node_is_not_rerun_on_resume() {
            let mut ctx = ctx_with_rerun(false);
            let node = BaseNode::new("child", NoopNodeBehavior).unwrap();
            let err = ctx
                .run_node(node, Value::Null, RunNodeOptions::default())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("rerun_on_resume=true"));
        }

        #[rusty_tokio::test]
        async fn rejects_a_second_use_as_output_delegate_on_the_same_context() {
            let mut ctx = context();
            ctx.set_output_delegated(true);
            let node = BaseNode::new("child", NoopNodeBehavior).unwrap();
            let options = RunNodeOptions {
                use_as_output: true,
                ..Default::default()
            };
            let err = ctx.run_node(node, Value::Null, options).await.unwrap_err();
            assert!(err
                .to_string()
                .contains("already has a use_as_output delegate"));
        }

        struct EchoBody;
        impl FunctionNodeBody for EchoBody {
            fn call<'a>(
                &'a self,
                _ctx: &'a mut Context,
                node_input: Value,
            ) -> TestBoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
                Box::pin(async move { Ok(vec![NodeYield::Data(node_input)]) })
            }
        }

        #[rusty_tokio::test]
        async fn completes_and_returns_the_nodes_output_and_events() {
            let mut ctx = context();
            let node = function_node("child", false, None, None, None, EchoBody).unwrap();
            let result = ctx
                .run_node(
                    node,
                    Value::String("hi".to_string()),
                    RunNodeOptions::default(),
                )
                .await
                .unwrap();
            match result {
                RunNodeOutcome::Completed(output) => {
                    assert_eq!(output.output, Some(Value::String("hi".to_string())));
                    assert_eq!(output.events.len(), 1);
                }
                RunNodeOutcome::Interrupted(_) => panic!("expected Completed"),
            }
        }

        struct AsksForInput;
        impl FunctionNodeBody for AsksForInput {
            fn call<'a>(
                &'a self,
                _ctx: &'a mut Context,
                _node_input: Value,
            ) -> TestBoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
                Box::pin(async move {
                    Ok(vec![NodeYield::RequestInput(RequestInput::new(
                        Some("please confirm".to_string()),
                        None,
                        None,
                    ))])
                })
            }
        }

        #[rusty_tokio::test]
        async fn reports_interrupted_and_propagates_interrupt_ids_onto_self() {
            let mut ctx = context();
            let node = function_node("waiter", true, None, None, None, AsksForInput).unwrap();
            let result = ctx
                .run_node(node, Value::Null, RunNodeOptions::default())
                .await
                .unwrap();
            match result {
                RunNodeOutcome::Interrupted(output) => {
                    assert!(output.output.is_none());
                    assert_eq!(output.events.len(), 1);
                }
                RunNodeOutcome::Completed(_) => panic!("expected Interrupted"),
            }
            assert_eq!(ctx.interrupt_ids().len(), 1);
        }

        /// A minimal agent double: transfers to `transfer_to` if set,
        /// otherwise yields a single event carrying `"<name>_output"`.
        struct ScriptedAgent {
            transfer_to: Option<String>,
        }

        impl AgentBehavior for ScriptedAgent {
            fn run_async_impl<'a>(
                &'a self,
                ctx: &'a mut InvocationContext,
            ) -> TestBoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
                let name = ctx.agent.as_ref().unwrap().name().to_string();
                let transfer_to = self.transfer_to.clone();
                let invocation_id = ctx.invocation_id.clone();
                Box::pin(async move {
                    let mut event = Event::new(invocation_id, name.clone(), NodeInfo::new(""));
                    match transfer_to {
                        Some(target) => event.actions.transfer_to_agent = Some(target),
                        None => event.output = Some(Value::String(format!("{name}_output"))),
                    }
                    Ok(vec![event])
                })
            }

            fn run_live_impl<'a>(
                &'a self,
                _ctx: &'a mut InvocationContext,
            ) -> TestBoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
                Box::pin(async { Ok(Vec::new()) })
            }
        }

        #[rusty_tokio::test]
        async fn resolves_a_sibling_transfer_and_returns_the_targets_output() {
            let agent_b = BaseAgent::new("agent_b", ScriptedAgent { transfer_to: None }).unwrap();
            let agent_a = BaseAgent::new(
                "agent_a",
                ScriptedAgent {
                    transfer_to: Some("agent_b".to_string()),
                },
            )
            .unwrap();
            let root = BaseAgent::build(
                "root",
                "",
                vec![agent_a.clone(), agent_b.clone()],
                Vec::new(),
                Vec::new(),
                NoopBehavior,
            )
            .unwrap();
            let agent_a = root.find_agent("agent_a").unwrap();

            let mut ctx = context();
            let node = crate::workflow_agent_node::agent_node(agent_a, true, None, None).unwrap();
            let result = ctx
                .run_node(node, Value::Null, RunNodeOptions::default())
                .await
                .unwrap();
            match result {
                RunNodeOutcome::Completed(output) => {
                    assert_eq!(
                        output.output,
                        Some(Value::String("agent_b_output".to_string()))
                    );
                    assert_eq!(output.events.len(), 2);
                }
                RunNodeOutcome::Interrupted(_) => panic!("expected Completed"),
            }
        }

        #[rusty_tokio::test]
        async fn surfaces_a_self_transfer_as_an_error() {
            let agent_a = BaseAgent::new(
                "agent_a",
                ScriptedAgent {
                    transfer_to: Some("agent_a".to_string()),
                },
            )
            .unwrap();
            let root = BaseAgent::build(
                "root",
                "",
                vec![agent_a.clone()],
                Vec::new(),
                Vec::new(),
                NoopBehavior,
            )
            .unwrap();
            let agent_a = root.find_agent("agent_a").unwrap();

            let mut ctx = context();
            let node = crate::workflow_agent_node::agent_node(agent_a, true, None, None).unwrap();
            let err = ctx
                .run_node(node, Value::Null, RunNodeOptions::default())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("cannot transfer to itself"));
        }

        struct DispatchesTwice;
        impl FunctionNodeBody for DispatchesTwice {
            fn call<'a>(
                &'a self,
                ctx: &'a mut Context,
                _node_input: Value,
            ) -> TestBoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
                Box::pin(async move {
                    let inner = function_node("inner", false, None, None, None, EchoBody).unwrap();
                    // Same explicit `run_id` both times: without one,
                    // each call auto-generates a fresh, distinct run id
                    // (a *different* dynamic node dispatch, not the same
                    // one called twice) — see `Context::run_node`'s own
                    // doc for the auto-increment `_child_run_counters`
                    // behavior this sidesteps.
                    let same_run_id = || RunNodeOptions {
                        run_id: Some("dyn-1".to_string()),
                        ..Default::default()
                    };
                    let first = ctx
                        .run_node(
                            inner.clone(),
                            Value::String("hi".to_string()),
                            same_run_id(),
                        )
                        .await?;
                    let second = ctx
                        .run_node(inner, Value::String("hi".to_string()), same_run_id())
                        .await?;
                    let (
                        RunNodeOutcome::Completed(first_out),
                        RunNodeOutcome::Completed(second_out),
                    ) = (first, second)
                    else {
                        panic!("expected both dispatches to complete");
                    };
                    assert_eq!(first_out.events.len(), 1);
                    assert!(
                        second_out.events.is_empty(),
                        "a second dynamic dispatch for the same node_path should fast-forward, not re-run"
                    );
                    assert_eq!(first_out.output, second_out.output);
                    Ok(vec![NodeYield::Data(
                        second_out.output.unwrap_or(Value::Null),
                    )])
                })
            }
        }

        #[rusty_tokio::test]
        async fn a_node_context_dispatches_dynamically_via_mode_1_and_dedups_the_second_call() {
            // Running `outer` via `NodeRunner` (not `BaseNode::run`
            // directly) is what gives its body a *node* context —
            // carrying an inherited-or-fresh `DynamicNodeScheduler` — so
            // `ctx.run_node()` inside it takes Mode 1, not Mode 2.
            let outer = function_node("outer", true, None, None, None, DispatchesTwice).unwrap();
            let root = context();
            let (_ctx, events) = crate::workflow_node_runner::NodeRunner::new(outer)
                .run(&root, Value::Null, BTreeMap::new())
                .await;
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].output, Some(Value::String("hi".to_string())));
        }
    }
}
