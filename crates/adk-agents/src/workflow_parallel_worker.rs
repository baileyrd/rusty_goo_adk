//! Capability C0317: `_ParallelWorker`, ported from
//! `google.adk.workflow._parallel_worker`. Part of the P7 workflow/graph
//! engine — see `workflow_node_state.rs`'s module doc for the standing
//! crate-placement decision.
//!
//! **Sequential dispatch, not concurrent — a disclosed, structural
//! narrowing**: the source spawns one `asyncio.Task` per input-list item
//! (via `ctx.run_node()`, `asyncio.create_task`) and awaits them with
//! `asyncio.wait(..., FIRST_COMPLETED)`, genuinely running items in
//! parallel up to `max_parallel_workers`. This port's
//! [`crate::context::Context::run_node`] needs `&mut Context` for a
//! node's *entire* execution — the exact same constraint that already
//! forced `Workflow`'s own LOOP phase (C0301) to bypass `run_node`
//! entirely and build a bespoke concurrent-future combinator instead
//! (see `workflow_workflow.rs`'s own module doc). `ParallelWorker` has
//! no such bespoke machinery to reuse: it dispatches through the
//! already-shipped, already-tested `Context::run_node` directly rather
//! than duplicating `Workflow`'s LOOP-phase machinery for a single node,
//! so items run **one at a time**, each fully awaited before the next
//! starts. [`ParallelWorker::max_parallel_workers`] is preserved as a
//! field (validated `>= 1` exactly like the source) for API-shape
//! fidelity and so a future concurrent redesign has an obvious home for
//! it, but it has no effect on this port's behavior — every run is
//! effectively `max_parallel_workers = 1`.
//!
//! **Deterministic earliest-index failure, preserved — by construction,
//! not by replicating the source's sort**: the source's own comment
//! explains it sorts simultaneously-completed tasks by `_worker_index`
//! specifically because `asyncio.wait` returns completions unordered.
//! Sequential dispatch has no such nondeterminism to correct for: item
//! *i* is never even started until every item before it has completed
//! successfully, so the first failure encountered is always, trivially,
//! the earliest-index one — the same observable contract via a simpler
//! mechanism.
//!
//! **"Cancel remaining in-flight items"/5s drain timeout, not
//! applicable**: both exist in the source to stop still-running
//! concurrent tasks after a failure. Under sequential dispatch there are
//! never any in-flight items when a failure occurs — nothing has
//! started for any index past the failing one — so there is nothing to
//! cancel or drain.
//!
//! **An interrupted item stops the batch and yields nothing, a
//! disclosed narrowing**: the source doesn't special-case
//! `NodeInterruptedError` in `_run_impl` — it propagates through
//! `task.exception()` and is treated exactly like any other failure
//! (cancel the rest, raise). This port's [`crate::context::Context::
//! run_node`] instead surfaces an interrupt as a distinct
//! `RunNodeOutcome::Interrupted` (not an `Err` at all — see that
//! method's own doc for why), and already propagates the interrupt ids
//! onto the calling `ctx` internally. `ParallelWorker::run_impl` mirrors
//! that: on an interrupted item it stops dispatching further items and
//! returns no yields, the same shape any other node that goes WAITING
//! without producing output already has (e.g. `Workflow::
//! handle_completion`'s own WAITING branches never touch output
//! either) — the caller reads the interrupt off `ctx` afterward, not
//! off this method's return value.

use std::future::Future;
use std::pin::Pin;

use rusty_serde::value::Value;

use crate::context::{Context, RunNodeOptions, RunNodeOutcome};
use crate::workflow_base_node::{
    start, BaseNode, BaseNodeError, NodeBehavior, NodeRunError, NodeYield,
};
use crate::workflow_retry_config::RetryConfig;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum ParallelWorkerError {
    #[error("ParallelWorker cannot wrap a START node.")]
    CannotWrapStart,
    #[error("max_parallel_workers must be greater than or equal to 1.")]
    InvalidMaxParallelWorkers,
    #[error("{0}")]
    BaseNode(#[from] BaseNodeError),
}

/// `workflow._parallel_worker._ParallelWorker`'s behavior half — see
/// this module's own doc for the sequential-dispatch narrowing. Build
/// one via [`parallel_worker_node`], not this struct directly (matching
/// the `AgentNode`/`agent_node` precedent).
pub struct ParallelWorker {
    node: BaseNode,
    /// Preserved for API-shape fidelity — see this module's own doc for
    /// why it has no effect on dispatch under this port's sequential
    /// model.
    #[allow(dead_code)]
    max_parallel_workers: Option<usize>,
}

impl ParallelWorker {
    fn new(
        node: BaseNode,
        max_parallel_workers: Option<usize>,
    ) -> Result<Self, ParallelWorkerError> {
        if node.ptr_eq(&start()) {
            return Err(ParallelWorkerError::CannotWrapStart);
        }
        if max_parallel_workers == Some(0) {
            return Err(ParallelWorkerError::InvalidMaxParallelWorkers);
        }
        Ok(Self {
            node,
            max_parallel_workers,
        })
    }
}

impl NodeBehavior for ParallelWorker {
    /// `_ParallelWorker._run_impl`: runs the wrapped node once per
    /// list-input item — see this module's own doc for the
    /// sequential-dispatch narrowing this method embodies.
    fn run_impl<'a>(
        &'a self,
        ctx: &'a mut Context,
        node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
        Box::pin(async move {
            let items: Vec<Value> = match node_input {
                Value::Seq(items) => items,
                other => vec![other],
            };
            if items.is_empty() {
                return Ok(vec![NodeYield::Data(Value::Seq(Vec::new()))]);
            }

            let mut results = Vec::with_capacity(items.len());
            for item in items {
                let outcome = ctx
                    .run_node(
                        self.node.clone(),
                        item,
                        RunNodeOptions {
                            use_sub_branch: true,
                            ..Default::default()
                        },
                    )
                    .await?;
                match outcome {
                    RunNodeOutcome::Completed(output) => {
                        results.push(output.output.unwrap_or(Value::Null));
                    }
                    RunNodeOutcome::Interrupted(_) => {
                        return Ok(Vec::new());
                    }
                }
            }
            Ok(vec![NodeYield::Data(Value::Seq(results))])
        })
    }
}

/// Builds a [`BaseNode`] wrapping `node` as a `ParallelWorker` — the
/// `parallel_worker=True` case of the source's `node(...)`/`Node`
/// (C0296), narrowed to this standalone constructor since this port's
/// [`NodeBehavior`] trait-object design has no equivalent of `Node`'s
/// subclass-and-override pattern (any [`NodeBehavior`] implementor that
/// wants parallel-worker fan-out already has the tool it needs: wrap the
/// node it builds with this function directly, the same effect the
/// source's inheritance achieves — see this module's own doc). `name`/
/// `rerun_on_resume=true` come from `node` itself, matching
/// `_ParallelWorker.__init__`'s own `super().__init__(name=built_node
/// .name, rerun_on_resume=True, ...)` — `rerun_on_resume=true` is
/// required here, not just faithful: [`Context::run_node`] refuses to
/// dispatch dynamically from a node whose own `rerun_on_resume` is
/// `false`.
pub fn parallel_worker_node(
    node: BaseNode,
    max_parallel_workers: Option<usize>,
    retry_config: Option<RetryConfig>,
    timeout: Option<f64>,
) -> Result<BaseNode, ParallelWorkerError> {
    let name = node.name().to_string();
    let worker = ParallelWorker::new(node, max_parallel_workers)?;
    BaseNode::build(
        name,
        String::new(),
        true,
        false,
        retry_config,
        timeout,
        None,
        None,
        None,
        worker,
    )
    .map_err(ParallelWorkerError::BaseNode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use crate::workflow_node_runner::NodeRunner;

    fn root_ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    struct Double;
    impl NodeBehavior for Double {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move {
                let Value::Int(n) = node_input else {
                    return Err("expected an int".into());
                };
                Ok(vec![NodeYield::Data(Value::Int(n * 2))])
            })
        }
    }

    struct FailsOnThree;
    impl NodeBehavior for FailsOnThree {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move {
                if node_input == Value::Int(3) {
                    return Err("boom".into());
                }
                Ok(vec![NodeYield::Data(node_input)])
            })
        }
    }

    #[test]
    fn new_rejects_wrapping_start() {
        let Err(err) = ParallelWorker::new(start(), None) else {
            panic!("expected an error");
        };
        assert!(matches!(err, ParallelWorkerError::CannotWrapStart));
    }

    #[test]
    fn new_rejects_a_zero_max_parallel_workers() {
        let node = BaseNode::new("doubler", Double).unwrap();
        let Err(err) = ParallelWorker::new(node, Some(0)) else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            ParallelWorkerError::InvalidMaxParallelWorkers
        ));
    }

    #[test]
    fn parallel_worker_node_sets_name_and_rerun_on_resume_from_the_inner_node() {
        let inner = BaseNode::new("doubler", Double).unwrap();
        let worker = parallel_worker_node(inner, None, None, None).unwrap();
        assert_eq!(worker.name(), "doubler");
        assert!(worker.rerun_on_resume());
    }

    #[rusty_tokio::test]
    async fn runs_the_wrapped_node_once_per_list_item_in_order() {
        let inner = BaseNode::new("doubler", Double).unwrap();
        let worker = parallel_worker_node(inner, None, None, None).unwrap();
        let root = root_ctx();
        let (child_ctx, _events) = NodeRunner::new(worker)
            .run(
                &root,
                Value::Seq(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
                Default::default(),
            )
            .await;
        assert_eq!(
            child_ctx.output(),
            Some(&Value::Seq(vec![
                Value::Int(2),
                Value::Int(4),
                Value::Int(6)
            ]))
        );
    }

    #[rusty_tokio::test]
    async fn wraps_a_non_list_input_as_a_single_item() {
        let inner = BaseNode::new("doubler", Double).unwrap();
        let worker = parallel_worker_node(inner, None, None, None).unwrap();
        let root = root_ctx();
        let (child_ctx, _events) = NodeRunner::new(worker)
            .run(&root, Value::Int(5), Default::default())
            .await;
        assert_eq!(child_ctx.output(), Some(&Value::Seq(vec![Value::Int(10)])));
    }

    #[rusty_tokio::test]
    async fn an_empty_list_short_circuits_to_an_empty_result() {
        let inner = BaseNode::new("doubler", Double).unwrap();
        let worker = parallel_worker_node(inner, None, None, None).unwrap();
        let root = root_ctx();
        let (child_ctx, _events) = NodeRunner::new(worker)
            .run(&root, Value::Seq(Vec::new()), Default::default())
            .await;
        assert_eq!(child_ctx.output(), Some(&Value::Seq(Vec::new())));
    }

    #[rusty_tokio::test]
    async fn stops_at_the_earliest_failing_item_and_surfaces_its_error() {
        let inner = BaseNode::new("flaky", FailsOnThree).unwrap();
        let worker = parallel_worker_node(inner, None, None, None).unwrap();
        let root = root_ctx();
        let (child_ctx, _events) = NodeRunner::new(worker)
            .run(
                &root,
                Value::Seq(vec![Value::Int(1), Value::Int(3), Value::Int(2)]),
                Default::default(),
            )
            .await;
        // `Context::run_node`'s own dynamic-dispatch failure path wraps
        // the underlying error into a `DynamicNodeFail` message keyed on
        // the failed node's name (already-shipped, tested behavior —
        // see `context.rs`'s own `WorkflowNodeError::DynamicNodeFail`
        // construction) rather than propagating "boom" verbatim; the
        // item that actually failed (index 1, value 3) is what matters
        // here, not the exact wrapped text.
        assert_eq!(child_ctx.error_message(), Some("Dynamic node flaky failed"));
        assert!(child_ctx.output().is_none());
    }
}
