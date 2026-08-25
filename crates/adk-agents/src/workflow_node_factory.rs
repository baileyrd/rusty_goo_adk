//! Capabilities C0296/C0326: `node(...)`/`Node` and `build_node`/
//! `is_node_like`, ported from `google.adk.workflow._node` and
//! `google.adk.workflow.utils._workflow_graph_utils`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc for
//! the standing crate-placement decision.
//!
//! **`build_node`, narrowed to the `BaseNode`/`START` cases — see
//! `workflow_graph_parser.rs`'s own doc for why this port's [`NodeLike`]
//! already excludes everything else**: the source's `build_node`
//! dispatches on `isinstance(node_like, ...)` over `BaseNode | BaseAgent
//! | BaseTool | Callable`. The `BaseTool` branch (→ `_ToolNode`) and the
//! `LlmAgent`/task-mode-`RemoteA2aAgent` branch (auto-defaulting
//! `rerun_on_resume`/`wait_for_output`/`mode`, `parallel_worker`
//! auto-wrapping) both stay out of scope for the same two,
//! already-disclosed reasons the rest of P7 keeps citing them: `BaseTool`
//! lives in `adk-tools`, which already depends on this crate (the
//! `adk-tools`/`adk-agents` crate-cycle, C0355/C0356), and `LlmAgent`
//! attachment needs the C0092 tree-fusion gap this port hasn't closed.
//! The `callable` branch (→ `FunctionNode`) has no Rust equivalent to
//! *dispatch to* at this layer either — Rust has no runtime
//! `isinstance`/`callable()` check the way Python does, so a caller with
//! a function to wrap constructs a [`crate::workflow_function_node::
//! FunctionNode`] directly instead of routing through a dynamic
//! dispatcher (this is a language-shape difference, not a missing
//! capability: [`NodeLike`]'s closed enum *is* this port's compile-time
//! equivalent of the source's runtime type check).
//!
//! **`is_node_like`, narrowed to a disclosed constant `true`**: the
//! source's version is a real runtime predicate over `item: Any`,
//! needed because Python has no static type to lean on. This port's
//! [`NodeLike`] is already a closed, exhaustively-matched enum — any
//! value of that type is node-like by construction, so there is nothing
//! left to check. Kept as a real, callable function (not omitted) for
//! API-shape parity with the source, per this migration's own boundary
//! contract.
//!
//! **`build_node`'s override-a-`BaseNode` case, narrowed — a distinct,
//! smaller gap from the two above**: the source's `node_like.model_copy
//! (update=kwargs)` clones an existing `BaseNode` with some fields
//! overridden while preserving its behavior. This port's [`BaseNode`] is
//! `Arc<BaseNodeData>` wrapping a `Box<dyn NodeBehavior>` — `NodeBehavior`
//! has no `Clone` bound (adding one would be a breaking supertrait
//! change to every existing implementor: `JoinNode`, `AgentNode`,
//! `FunctionNode`'s behavior, etc. — itself the kind of already-shipped-
//! surface break this migration's standing rule reserves for a
//! stop-and-ask, not something to fold into this batch on a
//! judgment call), so there is no way to rebuild an equivalent node with
//! different metadata but the same dispatch behavior. [`build_node`]
//! therefore only supports the override-free case (`node_like` returned
//! unchanged, matching the source's own `if kwargs: ... else: return
//! node_like` else-branch) and returns [`BuildNodeError::
//! OverridesNotSupportedForBaseNode`] if any override is requested
//! against an already-built [`BaseNode`].
//!
//! **`node(...)`, narrowed to its "wrap an already-resolved `NodeLike`"
//! overload**: the source's other overload — used as a bare decorator
//! (`@node` / `@node()`) directly on a Python function — builds a fresh
//! [`crate::workflow_function_node::FunctionNode`] from the decorated
//! callable. Rust has no function-decorator mechanism; a caller with a
//! function to wrap already calls `FunctionNode::new`/`FunctionNode::
//! build` directly (see that module's own constructor) and then passes
//! the resulting [`BaseNode`] through [`node`] if it also wants
//! `parallel_worker` wrapping — the same two-step composition the
//! source's own `wrapper(func)` closure performs internally, just
//! without the decorator sugar. `auth_config`/`parameter_binding` are
//! therefore not parameters of this port's [`node`]: they belong to
//! `FunctionNode`'s own constructor, called before [`node`] ever sees
//! the result.
//!
//! **`Node` (the subclassable base class), not ported — a permanent
//! narrowing, not a deferred one**: the source's `Node` exists so a
//! Python subclass can override `run_node_impl` and opt into
//! `parallel_worker` templating for free via inheritance. This port's
//! [`crate::workflow_base_node::NodeBehavior`] trait-object design
//! already *is* the Rust equivalent of "subclass and override" — every
//! [`NodeBehavior`] implementor in this crate (`JoinNode`, `AgentNode`,
//! `ParallelWorker` itself) already works this way. A Rust
//! [`NodeBehavior`] implementor that wants parallel-worker fan-out
//! doesn't need to inherit anything: it builds its own [`BaseNode`] and
//! passes it through [`crate::workflow_parallel_worker::
//! parallel_worker_node`] directly, the same effect `Node.model_post_init`'s
//! self-wrapping achieves through `model_copy`/`_inner_node`. There is
//! no separate Rust type for `Node` to land on.

use crate::workflow_base_node::BaseNode;
use crate::workflow_graph_parser::NodeLike;
use crate::workflow_parallel_worker::{parallel_worker_node, ParallelWorkerError};
use crate::workflow_retry_config::RetryConfig;

#[derive(Debug, rusty_err::Error)]
pub enum BuildNodeError {
    /// `build_node`'s `BaseNode` branch, when any override was
    /// requested — see this module's own doc.
    #[error(
        "build_node cannot override name/rerun_on_resume/retry_config/timeout on an \
         already-built BaseNode: NodeBehavior has no Clone equivalent to rebuild it with."
    )]
    OverridesNotSupportedForBaseNode,
}

/// `workflow.utils._workflow_graph_utils.is_node_like`, narrowed to a
/// disclosed constant `true` — see this module's own doc for why
/// [`NodeLike`] makes the source's runtime check unnecessary.
pub fn is_node_like(_node_like: &NodeLike) -> bool {
    true
}

/// Overrides [`build_node`] may apply to an already-built [`BaseNode`]
/// — see this module's own doc for why only the all-`None` case is
/// actually supported today.
#[derive(Debug, Clone, Default)]
pub struct BuildNodeOverrides {
    pub name: Option<String>,
    pub rerun_on_resume: Option<bool>,
    pub retry_config: Option<RetryConfig>,
    pub timeout: Option<f64>,
}

impl BuildNodeOverrides {
    fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.rerun_on_resume.is_none()
            && self.retry_config.is_none()
            && self.timeout.is_none()
    }
}

/// `workflow.utils._workflow_graph_utils.build_node`, narrowed — see
/// this module's own doc for the `BaseTool`/`LlmAgent`/callable branches
/// this excludes, and for why the `BaseNode` branch only supports the
/// no-overrides case.
pub fn build_node(
    node_like: NodeLike,
    overrides: BuildNodeOverrides,
) -> Result<BaseNode, BuildNodeError> {
    match node_like {
        NodeLike::Start => Ok(crate::workflow_base_node::start()),
        NodeLike::Node(node) => {
            if overrides.is_empty() {
                Ok(node)
            } else {
                Err(BuildNodeError::OverridesNotSupportedForBaseNode)
            }
        }
    }
}

#[derive(Debug, rusty_err::Error)]
pub enum NodeFactoryError {
    /// `node(...)`'s own pre-flight check, raised before any building —
    /// matches the source's `if max_parallel_workers is not None: if
    /// not parallel_worker: raise ValueError(...)`.
    #[error("max_parallel_workers can only be set when parallel_worker is True.")]
    MaxParallelWorkersRequiresParallelWorker,
    #[error("{0}")]
    BuildNode(#[from] BuildNodeError),
    #[error("{0}")]
    ParallelWorker(#[from] ParallelWorkerError),
}

/// `workflow._node.node(...)`'s "wrap an already-resolved `NodeLike`"
/// overload — see this module's own doc for why the bare-decorator
/// overload doesn't port, and why `Node` (the class) has no separate
/// Rust equivalent to build.
pub fn node(
    node_like: NodeLike,
    overrides: BuildNodeOverrides,
    parallel_worker: bool,
    max_parallel_workers: Option<usize>,
) -> Result<BaseNode, NodeFactoryError> {
    if max_parallel_workers.is_some() && !parallel_worker {
        return Err(NodeFactoryError::MaxParallelWorkersRequiresParallelWorker);
    }

    let retry_config = overrides.retry_config.clone();
    let timeout = overrides.timeout;
    let built = build_node(node_like, overrides)?;

    if parallel_worker {
        Ok(parallel_worker_node(
            built,
            max_parallel_workers,
            retry_config,
            timeout,
        )?)
    } else {
        Ok(built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow_base_node::{start, NoopNodeBehavior};

    fn node_of(name: &str) -> BaseNode {
        BaseNode::new(name, NoopNodeBehavior).unwrap()
    }

    #[test]
    fn is_node_like_is_always_true() {
        assert!(is_node_like(&NodeLike::Start));
        assert!(is_node_like(&NodeLike::Node(node_of("a"))));
    }

    #[test]
    fn build_node_resolves_start_to_the_singleton() {
        let built = build_node(NodeLike::Start, BuildNodeOverrides::default()).unwrap();
        assert!(built.ptr_eq(&start()));
    }

    #[test]
    fn build_node_passes_through_a_base_node_unchanged_when_no_overrides_are_given() {
        let a = node_of("a");
        let built = build_node(NodeLike::Node(a.clone()), BuildNodeOverrides::default()).unwrap();
        assert!(built.ptr_eq(&a));
    }

    #[test]
    fn build_node_rejects_overrides_on_a_base_node() {
        let a = node_of("a");
        let overrides = BuildNodeOverrides {
            name: Some("renamed".to_string()),
            ..Default::default()
        };
        let Err(err) = build_node(NodeLike::Node(a), overrides) else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            BuildNodeError::OverridesNotSupportedForBaseNode
        ));
    }

    #[test]
    fn node_passes_through_a_plain_node_like_unchanged() {
        let a = node_of("a");
        let built = node(
            NodeLike::Node(a.clone()),
            BuildNodeOverrides::default(),
            false,
            None,
        )
        .unwrap();
        assert!(built.ptr_eq(&a));
    }

    #[test]
    fn node_wraps_in_a_parallel_worker_when_requested() {
        let a = node_of("a");
        let built = node(NodeLike::Node(a), BuildNodeOverrides::default(), true, None).unwrap();
        assert_eq!(built.name(), "a");
        assert!(built.rerun_on_resume());
    }

    #[test]
    fn node_rejects_max_parallel_workers_without_parallel_worker() {
        let a = node_of("a");
        let Err(err) = node(
            NodeLike::Node(a),
            BuildNodeOverrides::default(),
            false,
            Some(2),
        ) else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            NodeFactoryError::MaxParallelWorkersRequiresParallelWorker
        ));
    }

    #[test]
    fn node_propagates_an_invalid_max_parallel_workers() {
        let a = node_of("a");
        let Err(err) = node(
            NodeLike::Node(a),
            BuildNodeOverrides::default(),
            true,
            Some(0),
        ) else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            NodeFactoryError::ParallelWorker(ParallelWorkerError::InvalidMaxParallelWorkers)
        ));
    }

    #[test]
    fn node_cannot_wrap_start_in_a_parallel_worker() {
        let Err(err) = node(NodeLike::Start, BuildNodeOverrides::default(), true, None) else {
            panic!("expected an error");
        };
        assert!(matches!(
            err,
            NodeFactoryError::ParallelWorker(ParallelWorkerError::CannotWrapStart)
        ));
    }
}
