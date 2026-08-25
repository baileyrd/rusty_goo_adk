//! Capabilities C0298/C0299/C0300: `Workflow`'s struct skeleton and
//! SETUP phase, ported from `google.adk.workflow._workflow`. Part of
//! the P7 workflow/graph engine — see `workflow_node_state.rs`'s module
//! doc for the standing crate-placement decision.
//!
//! **Scope, this batch**: the [`Workflow`] struct (fields, graph
//! auto-build), [`Workflow::validate_state_schema`] (narrowed to a
//! no-op — see below), and the SETUP phase
//! ([`WorkflowLoopState::setup`]). The LOOP phase (C0301-C0305:
//! `_run_loop`/`_schedule_ready_nodes`/`_handle_completion`/
//! `_buffer_downstream_triggers`) and FINALIZE (C0306) are **not**
//! ported here — `_run_impl`'s orchestration loop can't meaningfully
//! run without them, so `Workflow` is not yet wired into a
//! [`crate::workflow_base_node::BaseNode`]/`NodeBehavior` (the same
//! "build the layer, defer the caller" shape already used for
//! `NodeRunner`/`ReplayManager`/`check_interception`, each shipped and
//! unit-tested standalone before something called them — see those
//! modules' own docs). Revisit once the LOOP-phase batch lands.
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

use std::collections::BTreeMap;
use std::sync::Arc;

use rusty_serde::value::Value;

use crate::context::Context;
use crate::workflow_base_node::start;
use crate::workflow_dynamic_node_scheduler::DynamicNodeScheduler;
use crate::workflow_graph::Graph;
use crate::workflow_graph_parser::EdgeItem;
use crate::workflow_rehydration_utils::ChildScanState;
use crate::workflow_replay_manager::ReplayManager;
use crate::workflow_replay_sequence_barrier::ReplaySequenceBarrier;
use crate::workflow_trigger::Trigger;

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

/// `workflow._workflow._LoopState`, narrowed to the fields SETUP itself
/// populates and reads — see this module's own doc for why the
/// LOOP-phase-only fields (`nodes`/`pending_tasks`/`replayed_nodes`/
/// `node_outputs`/`node_branches`/`error_shut_down`) aren't defined yet.
pub struct WorkflowLoopState {
    pub recovered_executions: BTreeMap<String, ChildScanState>,
    pub sequence_barrier: Option<ReplaySequenceBarrier>,
    /// See this module's own doc for why this is an order-preserving
    /// `Vec` of pairs rather than a `HashMap`/`BTreeMap`.
    pub trigger_buffer: Vec<(String, Vec<Trigger>)>,
    scheduler: Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>>,
}

impl WorkflowLoopState {
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

    /// The scheduler this loop state installed on `ctx` — the LOOP
    /// phase (C0301-C0305, not built yet) will read this to dispatch
    /// through the same instance `ctx.run_node()` calls resolve to. No
    /// reader yet in this batch; kept wired now for the same reason
    /// `DynamicNodeScheduler::interrupt_ids` is (see that method's own
    /// doc).
    #[allow(dead_code)]
    pub(crate) fn scheduler(&self) -> &Arc<rusty_tokio::sync::Mutex<DynamicNodeScheduler>> {
        &self.scheduler
    }
}

impl Workflow {
    /// `Workflow._run_impl`'s SETUP phase: resumes from session events
    /// (or starts fresh), seeds START's successors as triggers, and
    /// installs this workflow's own dynamic-node scheduler on `ctx`.
    /// Returns `None` when [`Self::graph`] is `None` — the source's own
    /// `if self.graph is None: return` early exit, before SETUP even
    /// begins.
    pub async fn setup(&self, ctx: &mut Context, node_input: Value) -> Option<WorkflowLoopState> {
        let graph = self.graph.as_ref()?;

        // Set event_author so child events are attributed to this workflow.
        ctx.set_event_author(self.name.clone());

        // --- SETUP: resume from events or start fresh ---
        let mut replay_manager = ReplayManager::new();
        let (recovered_executions, _sequence) =
            replay_manager.scan_workflow_events(ctx).unwrap_or_default();
        let sequence_barrier = replay_manager.take_sequence_barrier();

        if !ctx.resume_inputs().is_empty() && recovered_executions.is_empty() {
            eprintln!(
                "Workflow {}: resume_inputs provided but no recovered executions found.",
                self.name
            );
        }

        let mut loop_state = WorkflowLoopState {
            recovered_executions,
            sequence_barrier,
            trigger_buffer: Vec::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_base_node::{start, BaseNode, NoopNodeBehavior};
    use crate::workflow_graph::Edge;
    use crate::workflow_graph_parser::EdgeItem;

    fn node(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
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
            recovered_executions: BTreeMap::new(),
            sequence_barrier: None,
            trigger_buffer: Vec::new(),
            scheduler: Arc::new(rusty_tokio::sync::Mutex::new(DynamicNodeScheduler::new())),
        };
        loop_state.push_trigger("b".to_string(), Trigger::default());
        loop_state.push_trigger("a".to_string(), Trigger::default());
        loop_state.push_trigger("b".to_string(), Trigger::default());

        assert_eq!(loop_state.trigger_buffer[0].0, "b");
        assert_eq!(loop_state.trigger_buffer[0].1.len(), 2);
        assert_eq!(loop_state.trigger_buffer[1].0, "a");
    }
}
