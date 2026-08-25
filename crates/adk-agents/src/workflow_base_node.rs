//! Capabilities C0294/C0295: `BaseNode`, ported from
//! `google.adk.workflow._base_node`. Part of the P7 workflow/graph
//! engine — see `workflow_node_state.rs`'s module doc for the standing
//! crate-placement decision (flat `workflow_*.rs` modules inside
//! `adk-agents`, no separate crate).
//!
//! This batch's scope: `BaseNode` (this file), `Graph`/`Edge`
//! (`workflow_graph.rs`, narrowed — see its own module doc),
//! `validate_graph` (`workflow_graph_validation.rs`), and the HITL
//! utilities `BaseNode::run` itself calls (`workflow_hitl_utils.rs`).
//!
//! **Struct+trait split, the same shape `BaseAgent`/`AgentBehavior`
//! already established**: [`NodeBehavior`] is the override point
//! (`_run_impl`, an eagerly-collected `Vec` of yields rather than an
//! async generator — the same "materialize the whole result" adaptation
//! `base_agent.rs`'s own `AgentBehavior::run_async_impl` already makes);
//! [`BaseNode`] is the `Arc`-backed handle carrying the shared fields.
//! [`NoopNodeBehavior`] mirrors `base_agent.rs`'s own `NoopBehavior` —
//! used by [`start`] (the source's `START` sentinel, never actually run)
//! and as a test double.
//!
//! **Schema fields, disclosed narrowing**: `input_schema`/
//! `output_schema`/`state_schema` stay opaque `Option<Value>`
//! placeholders — the same `SchemaType` narrowing `llm_agent.rs`'s own
//! `input_schema`/`output_schema` fields already disclose (C0087).
//! `_validate_input_data`/`_validate_output_data`/`_validate_schema`
//! accordingly reduce to a pass-through: nothing in this port yet
//! interprets a schema value to coerce/validate data against it.
//!
//! **Node-name identifier check, disclosed**: ASCII-only, matching this
//! crate's own agent-name check (`base_agent.rs`'s private
//! `validate_name`) rather than Python's Unicode-aware
//! `str.isidentifier()` — the same narrowing already established there,
//! not newly introduced by this file.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rusty_serde::value::Value;

use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_events::RequestInput;

use crate::base_agent::AsAny;
use crate::context::Context;
use crate::workflow_hitl_utils::create_request_input_event;
use crate::workflow_retry_config::RetryConfig;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The error type a [`NodeBehavior`] run can fail with — same narrowing
/// as `AgentBehavior::run_async_impl`'s own `AgentRunError`: the source
/// allows raising any `Exception`, this just needs to propagate one.
pub type NodeRunError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, rusty_err::Error)]
pub enum BaseNodeError {
    #[error("Node name '{0}' must be a valid identifier.")]
    InvalidName(String),
}

fn validate_node_name(name: &str) -> Result<(), BaseNodeError> {
    let is_identifier = {
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        }
    };
    if !is_identifier {
        return Err(BaseNodeError::InvalidName(name.to_string()));
    }
    Ok(())
}

/// One item [`NodeBehavior::run_impl`] yields — the source's
/// `_run_impl` can yield an `Event`, a `RequestInput`, `None` (skipped —
/// simply omitted from the returned `Vec` here), or any other raw
/// value. See [`BaseNode::run`]'s own doc for how each variant
/// normalizes into an [`Event`].
pub enum NodeYield {
    Event(Box<Event>),
    RequestInput(RequestInput),
    Data(Value),
}

/// C0294/C0295: `BaseNode`'s override point — `_run_impl`.
pub trait NodeBehavior: AsAny + Send + Sync + 'static {
    fn run_impl<'a>(
        &'a self,
        ctx: &'a mut Context,
        node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>>;
}

/// A behavior that produces no yields — the source's own default
/// `_run_impl` (raises `NotImplementedError` when actually invoked, but
/// is never invoked for [`start`], the only user of this behavior in
/// this batch). See [`base_agent::NoopBehavior`](crate::base_agent::NoopBehavior)
/// for the identical precedent.
#[derive(Debug, Default)]
pub struct NoopNodeBehavior;

impl NodeBehavior for NoopNodeBehavior {
    fn run_impl<'a>(
        &'a self,
        _ctx: &'a mut Context,
        _node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

struct BaseNodeData {
    name: String,
    description: String,
    rerun_on_resume: bool,
    wait_for_output: bool,
    retry_config: Option<RetryConfig>,
    timeout: Option<f64>,
    input_schema: Option<Value>,
    output_schema: Option<Value>,
    state_schema: Option<Value>,
    behavior: Box<dyn NodeBehavior>,
}

/// C0294/C0295: `workflow._base_node.BaseNode` — a cheap-clone handle
/// sharing one underlying [`BaseNodeData`], the same ownership shape
/// `adk_agents::base_agent::BaseAgent` already established.
#[derive(Clone)]
pub struct BaseNode(Arc<BaseNodeData>);

// `BaseNodeData.behavior: Box<dyn NodeBehavior>` can't derive `Debug`, so
// this prints just the node's name — enough to distinguish nodes in a
// `Graph`/`Edge` debug dump or a failed test assertion.
impl std::fmt::Debug for BaseNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BaseNode").field(&self.0.name).finish()
    }
}

impl BaseNode {
    pub fn new(
        name: impl Into<String>,
        behavior: impl NodeBehavior,
    ) -> Result<Self, BaseNodeError> {
        Self::build(
            name,
            String::new(),
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            behavior,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        name: impl Into<String>,
        description: impl Into<String>,
        rerun_on_resume: bool,
        wait_for_output: bool,
        retry_config: Option<RetryConfig>,
        timeout: Option<f64>,
        input_schema: Option<Value>,
        output_schema: Option<Value>,
        state_schema: Option<Value>,
        behavior: impl NodeBehavior,
    ) -> Result<Self, BaseNodeError> {
        let name = name.into();
        validate_node_name(&name)?;
        Ok(Self(Arc::new(BaseNodeData {
            name,
            description: description.into(),
            rerun_on_resume,
            wait_for_output,
            retry_config,
            timeout,
            input_schema,
            output_schema,
            state_schema,
            behavior: Box::new(behavior),
        })))
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn description(&self) -> &str {
        &self.0.description
    }

    pub fn rerun_on_resume(&self) -> bool {
        self.0.rerun_on_resume
    }

    pub fn wait_for_output(&self) -> bool {
        self.0.wait_for_output
    }

    pub fn retry_config(&self) -> Option<&RetryConfig> {
        self.0.retry_config.as_ref()
    }

    pub fn timeout(&self) -> Option<f64> {
        self.0.timeout
    }

    pub fn input_schema(&self) -> Option<&Value> {
        self.0.input_schema.as_ref()
    }

    pub fn output_schema(&self) -> Option<&Value> {
        self.0.output_schema.as_ref()
    }

    pub fn state_schema(&self) -> Option<&Value> {
        self.0.state_schema.as_ref()
    }

    /// Downcast escape hatch onto this node's concrete [`NodeBehavior`],
    /// the same pattern `BaseAgent::as_any` already established.
    pub fn as_any(&self) -> &dyn std::any::Any {
        self.0.behavior.as_ref().as_any()
    }

    /// Object identity — Rust's equivalent of Python's `id()`, used by
    /// [`crate::workflow_graph::Graph::new`] to deduplicate nodes
    /// inferred from edges.
    pub fn ptr_eq(&self, other: &BaseNode) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn validate_input_data(&self, data: Value) -> Value {
        // `_validate_input_data` — see the module doc's schema-fields
        // disclosure: `input_schema` is opaque, so this is a pass-through.
        let _ = &self.0.input_schema;
        data
    }

    fn validate_output_data(&self, data: Value) -> Value {
        let _ = &self.0.output_schema;
        data
    }

    /// `BaseNode.run` — the `#[final]`-equivalent public entry point.
    /// Calls [`NodeBehavior::run_impl`] and normalizes every yielded
    /// [`NodeYield`] to an [`Event`]:
    /// - [`NodeYield::Event`] passes through (its `output`, if any, is
    ///   run through output-schema validation).
    /// - [`NodeYield::RequestInput`] converts to an interrupt event via
    ///   [`create_request_input_event`].
    /// - [`NodeYield::Data`] wraps the value as `Event { output: Some
    ///   (value), .. }`, output-schema-validated.
    ///
    /// (The source's fourth case, `None`, has no equivalent here: a
    /// behavior that wants to yield nothing simply omits an entry from
    /// its returned `Vec` — there's no separate "skip" value to model.)
    pub async fn run(
        &self,
        ctx: &mut Context,
        node_input: Value,
    ) -> Result<Vec<Event>, NodeRunError> {
        let node_input = self.validate_input_data(node_input);
        let items = self.0.behavior.run_impl(ctx, node_input).await?;
        let mut events = Vec::with_capacity(items.len());
        for item in items {
            match item {
                NodeYield::Event(boxed_event) => {
                    let mut event = *boxed_event;
                    if let Some(output) = event.output.take() {
                        event.output = Some(self.validate_output_data(output));
                    }
                    events.push(event);
                }
                NodeYield::RequestInput(request_input) => {
                    events.push(create_request_input_event(&request_input));
                }
                NodeYield::Data(value) => {
                    let validated = self.validate_output_data(value);
                    let mut event = Event::new(String::new(), String::new(), NodeInfo::new(""));
                    event.output = Some(validated);
                    events.push(event);
                }
            }
        }
        Ok(events)
    }
}

/// `workflow._base_node.START` — the sentinel node marking a workflow
/// graph's entry point. The source constructs this once, at module
/// import time (`START = BaseNode(name='__START__')`), and every
/// reference to `START` elsewhere is the *same* object — load-bearing
/// for `Graph::new`'s identity-based node deduplication (two different
/// `BaseNode` instances that happen to share the name `"__START__"`
/// would otherwise be treated as distinct nodes, and rejected later by
/// `validate_graph`'s duplicate-name check). [`start`] preserves that
/// same-instance guarantee via a process-wide [`std::sync::OnceLock`]
/// rather than constructing a fresh node on every call. Never actually
/// run (the future `Workflow` orchestrator bypasses it and seeds
/// triggers for its successors directly), so [`NoopNodeBehavior`] is a
/// correct, never-exercised stand-in for its `_run_impl`.
pub fn start() -> BaseNode {
    static START: std::sync::OnceLock<BaseNode> = std::sync::OnceLock::new();
    START
        .get_or_init(|| {
            BaseNode::new("__START__", NoopNodeBehavior).expect("__START__ is a valid identifier")
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn ctx_for(agent_name: &str) -> Context {
        let agent =
            crate::base_agent::BaseAgent::new(agent_name, crate::base_agent::NoopBehavior).unwrap();
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(agent)
            .build();
        Context::new(ic)
    }

    struct YieldingBehavior;
    impl NodeBehavior for YieldingBehavior {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move {
                Ok(vec![
                    NodeYield::Event(Box::new(Event::new(
                        "inv-1".to_string(),
                        "node".to_string(),
                        NodeInfo::new(""),
                    ))),
                    NodeYield::RequestInput(RequestInput::new(
                        Some("please confirm".to_string()),
                        None,
                        None,
                    )),
                    NodeYield::Data(node_input),
                ])
            })
        }
    }

    #[test]
    fn build_rejects_a_non_identifier_name() {
        let err = BaseNode::new("not an identifier", NoopNodeBehavior).unwrap_err();
        assert!(matches!(err, BaseNodeError::InvalidName(_)));
    }

    #[test]
    fn build_accepts_a_valid_identifier_name() {
        assert!(BaseNode::new("valid_name", NoopNodeBehavior).is_ok());
    }

    #[rusty_tokio::test]
    async fn run_normalizes_every_yield_kind() {
        let node = BaseNode::new("my_node", YieldingBehavior).unwrap();
        let mut ctx = ctx_for("agent");
        let events = node
            .run(&mut ctx, Value::String("hi".to_string()))
            .await
            .unwrap();

        assert_eq!(events.len(), 3, "expected {events:?}");
        // Event passthrough.
        assert_eq!(events[0].author, "node");
        // RequestInput -> interrupt event via create_request_input_event.
        assert!(!events[1].get_function_calls().is_empty());
        // Raw data -> Event { output: Some(value) }.
        assert_eq!(events[2].output, Some(Value::String("hi".to_string())));
    }

    #[rusty_tokio::test]
    async fn start_never_actually_runs_but_is_a_valid_node() {
        let node = start();
        assert_eq!(node.name(), "__START__");
        let mut ctx = ctx_for("agent");
        let events = node.run(&mut ctx, Value::Null).await.unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn ptr_eq_distinguishes_distinct_nodes_with_the_same_name() {
        let a = BaseNode::new("dup", NoopNodeBehavior).unwrap();
        let b = BaseNode::new("dup", NoopNodeBehavior).unwrap();
        assert!(!a.ptr_eq(&b));
        assert!(a.ptr_eq(&a.clone()));
    }
}
