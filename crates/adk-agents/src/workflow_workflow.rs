//! Capabilities C0298-C0300, C0302, C0303: `Workflow`'s struct skeleton,
//! SETUP phase, and node-scheduling primitives, ported from
//! `google.adk.workflow._workflow`. Part of the P7 workflow/graph engine
//! — see `workflow_node_state.rs`'s module doc for the standing
//! crate-placement decision.
//!
//! **Scope, this batch (C0302/C0303 added to the earlier C0298-C0300)**:
//! [`Workflow::schedule_ready_nodes`]/[`Workflow::start_node_task`] and
//! their helpers, plus the resumability-checkpoint builders
//! ([`Workflow::node_checkpoint_event`]/[`Workflow::
//! maybe_reemit_replayed_output_event`]/[`Workflow::end_of_agent_event`]).
//! The LOOP driver itself (`_run_loop`, C0301), completion handling
//! (C0304), downstream-trigger buffering (C0305), and FINALIZE (C0306)
//! are **not** ported here — nothing yet *drives* [`WorkflowLoopState::
//! pending_tasks`] to completion, so `Workflow` still isn't wired into a
//! [`crate::workflow_base_node::BaseNode`]/`NodeBehavior` (the same
//! "build the layer, defer the caller" shape already used for
//! `NodeRunner`/`ReplayManager`/`check_interception`, each shipped and
//! unit-tested standalone before something called them — see those
//! modules' own docs). Revisit once the LOOP-phase batch lands.
//!
//! **`_start_node_task` dispatches via `NodeRunner` directly, not
//! `ctx.run_node()`/`DynamicNodeScheduler` — a deliberate, disclosed
//! divergence from the source's literal call chain**: the source's
//! `_start_node_task` calls `ctx._run_node_internal(node, ...,
//! skip_run_id_validation=True)` — the same internal dispatch
//! `Context.run_node()` itself calls — which, since SETUP installed this
//! Workflow's own `DynamicNodeScheduler` on `ctx`, routes every
//! graph-scheduled node through that one shared scheduler (Mode 1),
//! exactly like a dynamic `ctx.run_node()` call would. This port's
//! [`crate::context::Context::run_node`] (Mode 1) and
//! [`crate::workflow_dynamic_node_scheduler::DynamicNodeScheduler::call`]
//! both need `&mut Context`/hold an `Arc<Mutex<DynamicNodeScheduler>>`
//! guard for a node's *entire* execution — correct for `Context::
//! run_node`'s own contract ("always await this directly", genuinely
//! sequential by design), but structurally incompatible with the LOOP
//! phase's whole purpose: running multiple graph nodes *concurrently*.
//! Rust's borrow checker doesn't allow two overlapping `&mut Context`
//! borrows of the same context — reusing the scheduler path here would
//! serialize every graph node onto that one mutable/locked access,
//! silently defeating C0301's concurrent scheduling before it's even
//! built. [`NodeRunner::run`] needs only a *shared* `&Context` borrow,
//! which is what lets multiple node futures be constructed and polled
//! concurrently against the same underlying `ctx` — the only
//! Rust-shaped primitive this can correctly be built on. Fidelity isn't
//! lost by this: `_start_node_task` already does its own inline
//! replay-interception check against `loop_state.recovered_executions`
//! (independent of the scheduler's own dedup, which exists to protect
//! *concurrent dynamic* `ctx.run_node()` calls to the same node_path — a
//! case `_schedule_ready_nodes`'s own "skip if already RUNNING" check
//! already prevents for *static* graph nodes), and this port replicates
//! that inline check directly (see [`Workflow::start_node_task`]).
//!
//! **No `asyncio.Task`/`rusty_tokio::task::JoinSet` — a local, boxed,
//! non-`'static` future per pending node instead**: the source's
//! `loop_state.pending_tasks: dict[str, asyncio.Task[Context]]` spawns
//! each node's execution as a real OS-schedulable task. This port's
//! [`PendingNodeFuture`] is a boxed future instead — `JoinSet::spawn`
//! requires `Future: Send + 'static`, but `NodeRunner::run` borrows
//! `&Context` with a lifetime tied to the workflow's own `ctx` argument
//! (not `'static`), and `Context` has no `Clone`/`Arc` wrapping to make
//! it so without a breaking rework of already-shipped, tested code. The
//! LOOP phase (C0301, not built yet) will need a small local combinator
//! to poll several of these concurrently and return whichever complete
//! first — no `futures`/`indexmap` dependency needed for that either.
//!
//! **`edges`, not kept as a field**: the source keeps `self.edges: list
//! [EdgeItem]` as a real (Pydantic) field, but nothing in `_workflow.py`
//! reads it again after `_build_graph` consumes it once in
//! `model_post_init` — and this port's [`crate::workflow_graph_parser::
//! EdgeItem`] isn't `Clone` (its `Chain` variant holds non-`Copy`
//! `ChainElement`s), so keeping a reusable copy around would need one
//! anyway. [`Workflow::new`] just consumes `edges` once to build
//! [`Workflow::graph`], matching the only place the source ever reads
//! the field.
//!
//! **`_validate_state_schema`, narrowed to a no-op**: needs
//! `graph_node._sig.parameters`, reflective signature introspection
//! over a `FunctionNode`'s wrapped Python function. This port's
//! `FunctionNode` (`workflow_function_node.rs`'s own doc) already
//! discloses it has no such introspection — the wrapped body's
//! parameters aren't reflectable — and `BaseNode::state_schema` is
//! itself already an opaque `Value` placeholder
//! (`workflow_base_node.rs`'s own schema-fields disclosure), so there's
//! no typed `model_fields` to check parameter names against either.
//! [`Workflow::validate_state_schema`] therefore does nothing — kept as
//! a real, named method (matching the source's own method shape) so a
//! future revisit has an obvious place to land real validation if this
//! port ever grows typed schemas.
//!
//! **SETUP's shared `ReplayManager`, narrowed to a separate instance**:
//! the source's `_LoopState(DynamicNodeState)` inheritance means the
//! *same* `ReplayManager` instance backs both `Workflow`'s own
//! static-node event scan (`ReplayManager::scan_workflow_events`) and
//! the `DynamicNodeScheduler` it installs on `ctx`
//! (`DynamicNodeScheduler`'s own private `replay_manager` field).
//! `DynamicNodeScheduler::new` always builds its own fresh
//! `ReplayManager` (no constructor accepting an existing one), so
//! [`WorkflowLoopState::setup`] ends up with two separate `ReplayManager`
//! instances rather than one shared one. Both scan the same session
//! events and arrive at equivalent results — neither mutates session
//! state, and `ReplayManager::ensure_index`'s own event-count
//! dirty-check means a second, separate instance just rebuilds the same
//! index redundantly rather than diverging — so this is a real,
//! disclosed duplication of work, not a correctness gap. Revisit (a
//! `DynamicNodeScheduler::with_replay_manager` constructor) only if
//! this turns out to matter.
//!
//! **`trigger_buffer`, insertion-order preserved**: the source relies on
//! `dict[str, list[Trigger]]`'s insertion-order iteration for
//! deterministic scheduling (`_schedule_ready_nodes`'s own comment: "dicts
//! preserve insertion order... ensuring deterministic scheduling order
//! for parallel branches") — a plain `HashMap`/`BTreeMap` would silently
//! reorder this. [`WorkflowLoopState::trigger_buffer`] is a `Vec<(String,
//! Vec<Trigger>)>` instead, appended-to via [`WorkflowLoopState::
//! push_trigger`], which preserves first-seen key order the same way —
//! no `indexmap` dependency needed for a structure this small.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use adk_events::node_info::NodeInfo;
use adk_events::{Event, EventActions};
use rusty_serde::value::Value;

use crate::context::Context;
use crate::workflow_base_node::{start, BaseNode};
use crate::workflow_dynamic_node_scheduler::DynamicNodeScheduler;
use crate::workflow_graph::Graph;
use crate::workflow_graph_parser::EdgeItem;
use crate::workflow_node_runner::NodeRunner;
use crate::workflow_node_state::NodeState;
use crate::workflow_node_status::NodeStatus;
use crate::workflow_rehydration_utils::ChildScanState;
use crate::workflow_replay_interceptor::{check_interception, create_mock_context};
use crate::workflow_replay_manager::ReplayManager;
use crate::workflow_replay_sequence_barrier::ReplaySequenceBarrier;
use crate::workflow_trigger::Trigger;

/// One node's pending execution — either a real [`NodeRunner::run`] or a
/// fast-forwarded replay waiting on the sequence barrier (see
/// [`Workflow::start_node_task`]'s own doc). Borrows only `ctx: &'a
/// Context` (never `self`/`Workflow` — everything else this future
/// needs, e.g. the built [`NodeRunner`] or the already-constructed mock
/// [`Context`], is captured by value), and needs no `Send` bound: unlike
/// the source's `asyncio.create_task` (a real OS-schedulable task), this
/// port never spawns these onto the runtime — the LOOP phase (C0301,
/// not built yet) polls them all from within its own single task, the
/// same "eagerly collected" adaptation used throughout this port's
/// workflow engine. See this module's own doc for why that also means
/// `rusty_tokio::task::JoinSet` doesn't fit here (it requires `Send +
/// 'static`).
type PendingNodeFuture<'a> = Pin<Box<dyn Future<Output = (Context, Vec<Event>)> + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum WorkflowError {
    #[error("{0}")]
    Graph(String),
}

/// `workflow._workflow.Workflow`'s struct skeleton (C0298) — see this
/// module's own doc for why it isn't yet wrapped into a
/// [`crate::workflow_base_node::BaseNode`].
pub struct Workflow {
    name: String,
    rerun_on_resume: bool,
    max_concurrency: Option<usize>,
    graph: Option<Graph>,
}

impl Workflow {
    /// `Workflow.__init__` + `model_post_init`: builds and validates the
    /// graph from `edges` (skipped, leaving [`Self::graph`] `None`, when
    /// `edges` is empty — matching the source's own `if self.edges and
    /// self.graph is None`), then runs [`Self::validate_state_schema`].
    pub fn new(
        name: impl Into<String>,
        edges: Vec<EdgeItem>,
        max_concurrency: Option<usize>,
        rerun_on_resume: bool,
    ) -> Result<Self, WorkflowError> {
        let graph = if edges.is_empty() {
            None
        } else {
            let mut graph = Graph::from_edge_items(edges).map_err(WorkflowError::Graph)?;
            graph.validate().map_err(WorkflowError::Graph)?;
            Some(graph)
        };

        let workflow = Self {
            name: name.into(),
            rerun_on_resume,
            max_concurrency,
            graph,
        };
        workflow.validate_state_schema();
        Ok(workflow)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `Workflow.rerun_on_resume` — defaults `true`, unlike
    /// `BaseNode`'s own default of `false` (a `Workflow` node must be
    /// able to wake up and resume its own LOOP after an interrupt).
    pub fn rerun_on_resume(&self) -> bool {
        self.rerun_on_resume
    }

    pub fn max_concurrency(&self) -> Option<usize> {
        self.max_concurrency
    }

    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    /// C0299: narrowed to a no-op — see this module's own doc.
    fn validate_state_schema(&self) {}
}

/// `workflow._workflow._LoopState`, now including the static-node
/// scheduling fields C0302/C0303 read/write — see this module's own doc
/// for why `pending_tasks` borrows `'a` (tied to the `ctx: &'a Context`
/// [`Workflow::schedule_ready_nodes`]/[`Workflow::start_node_task`] are
/// called with), and for the fields still not here
/// (`node_outputs`/`node_branches`/`error_shut_down` — completion-
/// handling state, C0304).
pub struct WorkflowLoopState<'a> {
    pub nodes: HashMap<String, NodeState>,
    pub recovered_executions: BTreeMap<String, ChildScanState>,
    /// `Arc`-wrapped (not a bare `Option<ReplaySequenceBarrier>`):
    /// multiple pending node futures each need their own handle to wait
    /// on the same barrier, and `ReplaySequenceBarrier` isn't `Clone`
    /// (it owns `rusty_tokio::sync::Notify` values) — an `Arc` clone per
    /// future is the cheap, correct way to share one instance.
    pub sequence_barrier: Option<Arc<ReplaySequenceBarrier>>,
    /// See this module's own doc for why this is an order-preserving
    /// `Vec` of pairs rather than a `HashMap`/`BTreeMap`.
    pub trigger_buffer: Vec<(String, Vec<Trigger>)>,
    pub replayed_nodes: HashSet<String>,
    pending_tasks: Vec<(String, PendingNodeFuture<'a>)>,
    scheduler: Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>>,
}

impl<'a> WorkflowLoopState<'a> {
    /// Appends `trigger` to the buffer for `node_name`, creating the
    /// entry if this is the first trigger for that node — `dict.
    /// setdefault(node_name, []).append(trigger)`'s Rust shape over an
    /// order-preserving `Vec`.
    pub fn push_trigger(&mut self, node_name: String, trigger: Trigger) {
        if let Some((_, triggers)) = self
            .trigger_buffer
            .iter_mut()
            .find(|(n, _)| *n == node_name)
        {
            triggers.push(trigger);
        } else {
            self.trigger_buffer.push((node_name, vec![trigger]));
        }
    }

    /// `Workflow._pop_trigger`: pops the next trigger for `node_name`,
    /// or `None` if its buffer is empty — removing the buffer entry
    /// entirely once drained, matching the source's own `del
    /// loop_state.trigger_buffer[node_name]`.
    fn pop_trigger(&mut self, node_name: &str) -> Option<Trigger> {
        let idx = self
            .trigger_buffer
            .iter()
            .position(|(name, _)| name == node_name)?;
        let trigger = self.trigger_buffer[idx].1.remove(0);
        if self.trigger_buffer[idx].1.is_empty() {
            self.trigger_buffer.remove(idx);
        }
        Some(trigger)
    }

    /// The number of nodes with a pending (in-flight or fast-forwarded)
    /// execution — the LOOP phase's driver (C0301, not built yet) will
    /// poll these to completion.
    pub fn pending_task_count(&self) -> usize {
        self.pending_tasks.len()
    }

    /// The scheduler this loop state installed on `ctx` — dynamic
    /// `ctx.run_node()` calls from a running node's own body resolve
    /// through this instance (Mode 1). No reader in *this* batch's own
    /// code (`start_node_task` deliberately bypasses it — see this
    /// module's own doc); kept wired for the same reason
    /// `DynamicNodeScheduler::interrupt_ids` is.
    #[allow(dead_code)]
    pub(crate) fn scheduler(&self) -> &Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>> {
        &self.scheduler
    }
}

/// `Workflow._next_run_id`: increments and returns the next sequential
/// run id for a node.
fn next_run_id(node_state: &mut NodeState) -> String {
    node_state.run_counter += 1;
    node_state.run_counter.to_string()
}

/// `Workflow._compute_isolation_scope_for_node`, narrowed: the source's
/// Case 2 (a task-mode `LlmAgent` node gets its own full node_path as
/// isolation scope) needs `node.mode`, which — like every other
/// LlmAgent-attribute check already narrowed away in this port's P7
/// batches (`workflow_replay_interceptor.rs`'s `Case 5`,
/// `workflow_graph_validation.rs`'s `validate_chat_agent_wiring`) —
/// `BaseNode` has no equivalent for (the C0092 LlmAgent tree-fusion
/// gap). Case 1 (an explicit `trigger.isolation_scope`, set on resume)
/// is unaffected and still ported in full.
fn compute_isolation_scope_for_node(_node: &BaseNode, trigger: &Trigger) -> Option<String> {
    trigger.isolation_scope.clone()
}

/// `Workflow._create_node_state_for_new_run`: a fresh `NodeState` for a
/// new execution, preserving only the run counter (so a node that
/// switches between custom string ids and auto-generated numeric ones
/// doesn't collide on `node_path`).
fn create_node_state_for_new_run(old_state: &NodeState) -> NodeState {
    NodeState {
        run_counter: old_state.run_counter,
        ..Default::default()
    }
}

/// Builds the `{"status", "interrupts", "resumeInputs"}` subset of a
/// node's serialized state for [`Workflow::node_checkpoint_event`] — the
/// same three fields the source's own `model_dump(mode="json",
/// include={"status", "interrupts", "resume_inputs"})` keeps. Field
/// *names* in the resulting map follow [`NodeState`]'s own `rename_all =
/// "camelCase"` convention (`workflow_node_state.rs`) rather than the
/// source's raw snake_case — nothing in this port reads this checkpoint
/// back yet (the LOOP driver that would, C0301, isn't built), so there's
/// no existing wire format to stay compatible with; a future reader
/// should follow the same convention this writes.
fn node_state_checkpoint_value(node_state: &NodeState) -> Value {
    let full = rusty_serde::json::to_value(node_state).unwrap_or(Value::Null);
    match full {
        Value::Map(entries) => Value::Map(
            entries
                .into_iter()
                .filter(|(key, _)| matches!(key.as_str(), "status" | "interrupts" | "resumeInputs"))
                .collect(),
        ),
        other => other,
    }
}

impl Workflow {
    /// `Workflow._run_impl`'s SETUP phase: resumes from session events
    /// (or starts fresh), seeds START's successors as triggers, and
    /// installs this workflow's own dynamic-node scheduler on `ctx`.
    /// Returns `None` when [`Self::graph`] is `None` — the source's own
    /// `if self.graph is None: return` early exit, before SETUP even
    /// begins.
    pub async fn setup(
        &self,
        ctx: &mut Context,
        node_input: Value,
    ) -> Option<WorkflowLoopState<'_>> {
        let graph = self.graph.as_ref()?;

        // Set event_author so child events are attributed to this workflow.
        ctx.set_event_author(self.name.clone());

        // --- SETUP: resume from events or start fresh ---
        let mut replay_manager = ReplayManager::new();
        let (recovered_executions, _sequence) =
            replay_manager.scan_workflow_events(ctx).unwrap_or_default();
        let sequence_barrier = replay_manager.take_sequence_barrier().map(Arc::new);

        if !ctx.resume_inputs().is_empty() && recovered_executions.is_empty() {
            eprintln!(
                "Workflow {}: resume_inputs provided but no recovered executions found.",
                self.name
            );
        }

        let mut loop_state = WorkflowLoopState {
            nodes: HashMap::new(),
            recovered_executions,
            sequence_barrier,
            trigger_buffer: Vec::new(),
            replayed_nodes: HashSet::new(),
            pending_tasks: Vec::new(),
            scheduler: Arc::new(rusty_tokio::sync::Mutex::new(DynamicNodeScheduler::new())),
        };

        self.seed_start_triggers(graph, node_input, &mut loop_state);

        // Create the scheduler for dynamic node dispatch and install it
        // on `ctx` — Mode 1 in `Context::run_node`.
        ctx.set_workflow_scheduler(loop_state.scheduler.clone());

        Some(loop_state)
    }

    /// `Workflow._seed_start_triggers`: seeds triggers for `START`'s
    /// direct successors.
    fn seed_start_triggers(
        &self,
        graph: &Graph,
        node_input: Value,
        loop_state: &mut WorkflowLoopState,
    ) {
        let start_name = start().name().to_string();
        let start_edges: Vec<_> = graph
            .edges
            .iter()
            .filter(|edge| edge.from_node.name() == start_name)
            .collect();
        let use_sub_branch = start_edges.len() > 1;
        for edge in start_edges {
            let trigger = Trigger {
                input: node_input.clone(),
                use_sub_branch,
                ..Default::default()
            };
            loop_state.push_trigger(edge.to_node.name().to_string(), trigger);
        }
    }

    /// `Workflow._has_waiting_task_agent`, narrowed to always `false` —
    /// needs `node.mode == "task"`, the same LlmAgent-attribute gap
    /// [`compute_isolation_scope_for_node`]'s own doc discloses.
    /// Structurally correct today (this port has no node type that
    /// could ever make it `true`), kept as a real, called method rather
    /// than inlined away so a future batch that fuses `LlmAgent` into
    /// this port's node types doesn't also have to rediscover this call
    /// site.
    fn has_waiting_task_agent(&self, _loop_state: &WorkflowLoopState) -> bool {
        false
    }

    /// `Workflow._at_concurrency_limit`: `max_concurrency` of `0` counts
    /// as unset, matching the source's own `bool(self.max_concurrency)`
    /// falsy check (`bool(0) is False`).
    fn at_concurrency_limit(&self, loop_state: &WorkflowLoopState) -> bool {
        match self.max_concurrency {
            Some(limit) if limit != 0 => loop_state.pending_task_count() >= limit,
            _ => false,
        }
    }

    /// `Workflow._prepare_node_state_for_starting`: creates a node's
    /// first `NodeState`, or a fresh one (preserving only the run
    /// counter) for a new execution — never carrying over a prior run's
    /// input/status/interrupts.
    fn prepare_node_state_for_starting(
        loop_state: &mut WorkflowLoopState,
        node_name: &str,
        trigger: &Trigger,
    ) {
        let mut node_state = match loop_state.nodes.get(node_name) {
            Some(old) => create_node_state_for_new_run(old),
            None => NodeState::default(),
        };
        node_state.input = trigger.input.clone();
        node_state.status = NodeStatus::Running;
        loop_state.nodes.insert(node_name.to_string(), node_state);
    }

    /// `Workflow._schedule_ready_nodes`: pops triggers from the buffer
    /// and schedules ready nodes — deterministic scheduling order
    /// (processing `trigger_buffer` in trigger-arrival order), skipping
    /// nodes already `RUNNING` or `WAITING`-on-unresolved-interrupts,
    /// stopping once `max_concurrency` is reached.
    pub fn schedule_ready_nodes<'a>(
        &'a self,
        loop_state: &mut WorkflowLoopState<'a>,
        ctx: &'a Context,
    ) {
        if self.has_waiting_task_agent(loop_state) {
            return;
        }

        let node_names: Vec<String> = loop_state
            .trigger_buffer
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        for node_name in node_names {
            if loop_state
                .pending_tasks
                .iter()
                .any(|(name, _)| *name == node_name)
            {
                continue;
            }
            if let Some(node_state) = loop_state.nodes.get(&node_name) {
                if node_state.status == NodeStatus::Running {
                    continue;
                }
                if node_state.status == NodeStatus::Waiting && !node_state.interrupts.is_empty() {
                    continue;
                }
            }

            if self.at_concurrency_limit(loop_state) {
                break;
            }

            let Some(trigger) = loop_state.pop_trigger(&node_name) else {
                continue;
            };

            Self::prepare_node_state_for_starting(loop_state, &node_name, &trigger);
            self.start_node_task(loop_state, ctx, &node_name, trigger);
        }
    }

    /// `Workflow._start_node_task` — see this module's own doc for why
    /// this dispatches through [`NodeRunner`] directly rather than
    /// `ctx.run_node()`/`DynamicNodeScheduler`. Returns `true` if the
    /// node was scheduled for real execution, `false` if it was
    /// fast-forwarded from recovered history (a replayed no-op run).
    fn start_node_task<'a>(
        &'a self,
        loop_state: &mut WorkflowLoopState<'a>,
        ctx: &'a Context,
        node_name: &str,
        mut trigger: Trigger,
    ) -> bool {
        let graph = self
            .graph
            .as_ref()
            .expect("start_node_task requires a built graph");
        let node = graph
            .nodes
            .iter()
            .find(|candidate| candidate.name() == node_name)
            .cloned()
            .expect("start_node_task requires a known static node");
        let is_terminal = graph.terminal_node_names().contains(node_name);

        let run_id = {
            let node_state = loop_state
                .nodes
                .get_mut(node_name)
                .expect("prepare_node_state_for_starting runs before this");
            match node_state.run_id.clone() {
                Some(id) => id,
                None => {
                    let id = next_run_id(node_state);
                    node_state.run_id = Some(id.clone());
                    id
                }
            }
        };

        let key = format!("{node_name}@{run_id}");
        let recovered = loop_state.recovered_executions.get(&key).cloned();

        if let Some(recovered) = &recovered {
            let result = check_interception(&node, Some(recovered), None);
            if !result.should_run {
                let ancestors = if is_terminal {
                    let mut ancestors = vec![ctx.node_path().to_string()];
                    ancestors.extend(ctx.output_for_ancestors().iter().cloned());
                    ancestors
                } else {
                    ctx.output_for_ancestors().to_vec()
                };
                let mock_ctx = create_mock_context(
                    ctx,
                    &node,
                    run_id.clone(),
                    &result,
                    &ancestors,
                    None,
                    recovered.branch.clone(),
                );
                loop_state.replayed_nodes.insert(node_name.to_string());

                let barrier = loop_state.sequence_barrier.clone();
                let wait_key = key.clone();
                let future: PendingNodeFuture<'a> = Box::pin(async move {
                    if let Some(barrier) = &barrier {
                        let _ = barrier.wait(&wait_key).await;
                    }
                    (mock_ctx, Vec::new())
                });
                loop_state
                    .pending_tasks
                    .push((node_name.to_string(), future));
                return false;
            }

            {
                let node_state = loop_state.nodes.get_mut(node_name).expect("checked above");
                node_state.resume_inputs = result.resume_inputs.unwrap_or_default();
            }

            if trigger.isolation_scope.is_none() {
                if let Some(iso) = &recovered.isolation_scope {
                    trigger.isolation_scope = Some(iso.clone());
                }
            }
        }

        let resume_inputs = loop_state
            .nodes
            .get(node_name)
            .expect("checked above")
            .resume_inputs
            .clone();
        let override_isolation_scope = compute_isolation_scope_for_node(&node, &trigger);

        let runner = NodeRunner::new(node)
            .with_run_id(run_id)
            .with_use_as_output(is_terminal)
            .with_sub_branch(trigger.use_sub_branch)
            .with_override_branch(trigger.branch.clone())
            .with_override_isolation_scope(override_isolation_scope);
        let node_input = trigger.input.clone();
        let future: PendingNodeFuture<'a> =
            Box::pin(async move { runner.run(ctx, node_input, resume_inputs).await });
        loop_state
            .pending_tasks
            .push((node_name.to_string(), future));
        true
    }

    /// `Workflow._emit_node_checkpoint`, redesigned as a pure builder — the
    /// source enqueues onto `ic._enqueue_event`, a live per-invocation
    /// event queue this port has no equivalent for (see
    /// `workflow_node_state.rs`'s "eagerly collected `Vec`" precedent, and
    /// this crate's other checkpoint-shaped callers, e.g.
    /// `workflow_hitl_utils.rs`). This returns the built `Event` instead,
    /// for the (not-yet-built) LOOP driver (C0301) to push into its own
    /// accumulator. `None` when the session isn't resumable — a
    /// non-resumable session reconstructs the same state by replaying
    /// prior events instead of needing this snapshot.
    pub fn node_checkpoint_event(
        &self,
        loop_state: &WorkflowLoopState,
        ctx: &Context,
    ) -> Option<Event> {
        let ic = ctx.invocation_context();
        if !ic.is_resumable() {
            return None;
        }

        let mut names: Vec<&String> = loop_state.nodes.keys().collect();
        names.sort();
        let nodes: Vec<(String, Value)> = names
            .into_iter()
            .map(|name| {
                (
                    name.clone(),
                    node_state_checkpoint_value(&loop_state.nodes[name]),
                )
            })
            .collect();

        let mut agent_state = HashMap::new();
        agent_state.insert("nodes".to_string(), Value::Map(nodes));

        let mut event = Event::new(
            ic.invocation_id.clone(),
            self.name.clone(),
            NodeInfo::new(""),
        );
        event.branch = ic.branch.clone();
        event.actions = EventActions {
            agent_state: Some(agent_state),
            ..EventActions::default()
        };
        Some(event)
    }

    /// `Workflow._maybe_reemit_replayed_output` — see [`Self::
    /// node_checkpoint_event`]'s own doc for the enqueue-to-builder
    /// redesign. `None` when the session isn't resumable, or the
    /// fast-forwarded node produced no output.
    pub fn maybe_reemit_replayed_output_event(
        &self,
        child_ctx: &Context,
        ctx: &Context,
    ) -> Option<Event> {
        let ic = ctx.invocation_context();
        if !ic.is_resumable() {
            return None;
        }
        let output = child_ctx.output()?;

        let mut event = Event::new(
            ic.invocation_id.clone(),
            self.name.clone(),
            NodeInfo::new(child_ctx.node_path()),
        );
        event.branch = ic.branch.clone();
        event.output = Some(output.clone());
        Some(event)
    }

    /// `Workflow._emit_end_of_agent` — see [`Self::node_checkpoint_event`]'s
    /// own doc for the enqueue-to-builder redesign.
    pub fn end_of_agent_event(&self, ctx: &Context) -> Option<Event> {
        let ic = ctx.invocation_context();
        if !ic.is_resumable() {
            return None;
        }

        let mut event = Event::new(
            ic.invocation_id.clone(),
            self.name.clone(),
            NodeInfo::new(""),
        );
        event.branch = ic.branch.clone();
        event.actions = EventActions {
            end_of_agent: true,
            ..EventActions::default()
        };
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_configs::ResumabilityConfig;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::{start, BaseNode, NoopNodeBehavior};
    use crate::workflow_graph::Edge;
    use crate::workflow_graph_parser::EdgeItem;
    use crate::workflow_rehydration_utils::ChildOutput;

    fn node(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    fn resumable_ctx() -> Context {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.resumability_config = Some(ResumabilityConfig { is_resumable: true });
        Context::new(ic)
    }

    fn linear_edges(a: BaseNode, b: BaseNode) -> Vec<EdgeItem> {
        vec![
            EdgeItem::Edge(Edge::new(start(), a.clone(), None)),
            EdgeItem::Edge(Edge::new(a, b, None)),
        ]
    }

    #[test]
    fn a_workflow_with_no_edges_has_no_graph() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        assert!(workflow.graph().is_none());
        assert!(workflow.rerun_on_resume());
    }

    #[test]
    fn building_from_edges_produces_a_validated_graph() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let graph = workflow.graph().unwrap();
        assert_eq!(graph.nodes.len(), 3); // START, a, b
        assert!(graph.terminal_node_names().contains("b"));
    }

    #[test]
    fn an_invalid_graph_is_rejected_at_construction() {
        // A single edge with no START predecessor fails `validate_graph`.
        let edges = vec![EdgeItem::Edge(Edge::new(node("a"), node("b"), None))];
        let Err(err) = Workflow::new("wf", edges, None, true) else {
            panic!("expected an error");
        };
        assert!(matches!(err, WorkflowError::Graph(_)));
    }

    #[rusty_tokio::test]
    async fn setup_returns_none_without_a_graph() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let mut c = ctx();
        assert!(workflow.setup(&mut c, Value::Null).await.is_none());
    }

    #[rusty_tokio::test]
    async fn setup_seeds_a_trigger_for_starts_successor_and_installs_a_scheduler() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = ctx();
        let loop_state = workflow
            .setup(&mut c, Value::String("hi".to_string()))
            .await
            .unwrap();

        assert_eq!(loop_state.trigger_buffer.len(), 1);
        assert_eq!(loop_state.trigger_buffer[0].0, "a");
        assert_eq!(loop_state.trigger_buffer[0].1.len(), 1);
        assert_eq!(
            loop_state.trigger_buffer[0].1[0].input,
            Value::String("hi".to_string())
        );
        assert!(!loop_state.trigger_buffer[0].1[0].use_sub_branch);
        assert!(loop_state.recovered_executions.is_empty());
        assert_eq!(c.event_author(), "wf");
        assert!(c.workflow_scheduler().is_some());
    }

    #[rusty_tokio::test]
    async fn setup_uses_a_sub_branch_when_start_has_multiple_successors() {
        let a = node("a");
        let b = node("b");
        let edges = vec![
            EdgeItem::Edge(Edge::new(start(), a.clone(), None)),
            EdgeItem::Edge(Edge::new(start(), b.clone(), None)),
        ];
        let workflow = Workflow::new("wf", edges, None, true).unwrap();
        let mut c = ctx();
        let loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();

        assert_eq!(loop_state.trigger_buffer.len(), 2);
        for (_, triggers) in &loop_state.trigger_buffer {
            assert!(triggers[0].use_sub_branch);
        }
    }

    #[rusty_tokio::test]
    async fn push_trigger_appends_to_an_existing_entry_preserving_order() {
        let mut loop_state = WorkflowLoopState {
            nodes: HashMap::new(),
            recovered_executions: BTreeMap::new(),
            sequence_barrier: None,
            trigger_buffer: Vec::new(),
            replayed_nodes: HashSet::new(),
            pending_tasks: Vec::new(),
            scheduler: Arc::new(rusty_tokio::sync::Mutex::new(DynamicNodeScheduler::new())),
        };
        loop_state.push_trigger("b".to_string(), Trigger::default());
        loop_state.push_trigger("a".to_string(), Trigger::default());
        loop_state.push_trigger("b".to_string(), Trigger::default());

        assert_eq!(loop_state.trigger_buffer[0].0, "b");
        assert_eq!(loop_state.trigger_buffer[0].1.len(), 2);
        assert_eq!(loop_state.trigger_buffer[1].0, "a");
    }

    #[rusty_tokio::test]
    async fn schedule_ready_nodes_starts_a_ready_node_and_produces_a_pending_task() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = ctx();
        let mut loop_state = workflow
            .setup(&mut c, Value::String("hi".to_string()))
            .await
            .unwrap();

        workflow.schedule_ready_nodes(&mut loop_state, &c);

        assert_eq!(loop_state.pending_task_count(), 1);
        assert_eq!(
            loop_state.nodes.get("a").unwrap().status,
            NodeStatus::Running
        );
        assert!(loop_state.trigger_buffer.is_empty());

        let (name, future) = loop_state.pending_tasks.remove(0);
        assert_eq!(name, "a");
        let (_child_ctx, events) = future.await;
        assert!(events.is_empty());
    }

    #[rusty_tokio::test]
    async fn schedule_ready_nodes_respects_the_concurrency_limit() {
        let a = node("a");
        let b = node("b");
        let edges = vec![
            EdgeItem::Edge(Edge::new(start(), a.clone(), None)),
            EdgeItem::Edge(Edge::new(start(), b.clone(), None)),
        ];
        let workflow = Workflow::new("wf", edges, Some(1), true).unwrap();
        let mut c = ctx();
        let mut loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();
        assert_eq!(loop_state.trigger_buffer.len(), 2);

        workflow.schedule_ready_nodes(&mut loop_state, &c);

        assert_eq!(loop_state.pending_task_count(), 1);
        assert_eq!(loop_state.trigger_buffer.len(), 1);
    }

    #[rusty_tokio::test]
    async fn schedule_ready_nodes_skips_a_node_already_running() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = ctx();
        let mut loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();

        loop_state.nodes.insert(
            "a".to_string(),
            NodeState {
                status: NodeStatus::Running,
                ..Default::default()
            },
        );

        workflow.schedule_ready_nodes(&mut loop_state, &c);

        assert_eq!(loop_state.pending_task_count(), 0);
        assert_eq!(loop_state.trigger_buffer.len(), 1);
    }

    #[rusty_tokio::test]
    async fn schedule_ready_nodes_fast_forwards_a_recovered_completed_node() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = ctx();
        let mut loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();

        loop_state.recovered_executions.insert(
            "a@1".to_string(),
            ChildScanState {
                run_id: Some("1".to_string()),
                output: Some(ChildOutput::Value(Value::String("cached".to_string()))),
                route: None,
                branch: None,
                isolation_scope: None,
                transfer_to_agent: None,
                interrupt_ids: HashSet::new(),
                resolved_ids: HashSet::new(),
                resolved_responses: BTreeMap::new(),
            },
        );

        workflow.schedule_ready_nodes(&mut loop_state, &c);

        assert_eq!(loop_state.pending_task_count(), 1);
        assert!(loop_state.replayed_nodes.contains("a"));

        let (name, future) = loop_state.pending_tasks.remove(0);
        assert_eq!(name, "a");
        let (mock_ctx, events) = future.await;
        assert!(events.is_empty());
        assert_eq!(
            mock_ctx.output(),
            Some(&Value::String("cached".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn node_checkpoint_event_is_none_when_not_resumable() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = ctx();
        let loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();
        assert!(workflow.node_checkpoint_event(&loop_state, &c).is_none());
    }

    #[rusty_tokio::test]
    async fn node_checkpoint_event_snapshots_node_statuses_when_resumable() {
        let workflow = Workflow::new("wf", linear_edges(node("a"), node("b")), None, true).unwrap();
        let mut c = resumable_ctx();
        let mut loop_state = workflow.setup(&mut c, Value::Null).await.unwrap();
        workflow.schedule_ready_nodes(&mut loop_state, &c);

        let event = workflow.node_checkpoint_event(&loop_state, &c).unwrap();
        assert_eq!(event.author, "wf");
        let agent_state = event.actions.agent_state.expect("agent_state set");
        let nodes_value = agent_state.get("nodes").expect("nodes key present");
        let Value::Map(nodes) = nodes_value else {
            panic!("expected a map, got {nodes_value:?}");
        };
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, "a");
        let Value::Map(a_state) = &nodes[0].1 else {
            panic!("expected a map");
        };
        let status = a_state
            .iter()
            .find(|(k, _)| k == "status")
            .map(|(_, v)| v.clone());
        assert_eq!(status, Some(Value::String("RUNNING".to_string())));
    }

    #[test]
    fn maybe_reemit_replayed_output_event_is_none_without_output() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let c = resumable_ctx();
        let child = ctx();
        assert!(workflow
            .maybe_reemit_replayed_output_event(&child, &c)
            .is_none());
    }

    #[test]
    fn maybe_reemit_replayed_output_event_is_none_when_not_resumable() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let c = ctx();
        let mut child = ctx();
        child.set_output(Value::String("out".to_string())).unwrap();
        assert!(workflow
            .maybe_reemit_replayed_output_event(&child, &c)
            .is_none());
    }

    #[test]
    fn maybe_reemit_replayed_output_event_carries_the_output_when_resumable() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let c = resumable_ctx();
        let mut child = ctx();
        child.set_output(Value::String("out".to_string())).unwrap();
        let event = workflow
            .maybe_reemit_replayed_output_event(&child, &c)
            .unwrap();
        assert_eq!(event.output, Some(Value::String("out".to_string())));
    }

    #[test]
    fn end_of_agent_event_is_none_when_not_resumable() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let c = ctx();
        assert!(workflow.end_of_agent_event(&c).is_none());
    }

    #[test]
    fn end_of_agent_event_sets_the_flag_when_resumable() {
        let workflow = Workflow::new("wf", Vec::new(), None, true).unwrap();
        let c = resumable_ctx();
        let event = workflow.end_of_agent_event(&c).unwrap();
        assert!(event.actions.end_of_agent);
    }
}
