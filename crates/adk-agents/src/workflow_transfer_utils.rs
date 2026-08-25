//! Capability C0325: `resolve_and_derive_transfer_context`, ported from
//! `google.adk.workflow.utils._transfer_utils`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc for
//! the standing crate-placement decision.
//!
//! **`Context.node`/`Context.parent_ctx`, adapted to a local, per-call
//! chain instead of permanent `Context` fields**: the source walks
//! `curr_ctx.parent_ctx` (and `.node.name`) as live object references
//! stored on every `Context`. This port's `Context` (`context.rs`, own
//! module doc) deliberately never grew persistent `node`/`parent_ctx`
//! fields — nothing outside `Context::run_node`'s own in-place
//! agent-transfer loop (C0059/C0060) needs them, and adding them would
//! mean either an `Arc`-shared `Context` (a breaking change to every
//! existing `&mut Context` call site in this port) or an owned
//! `Context` tree that fights the borrow checker for no behavioral
//! gain. Instead, [`resolve_and_derive_transfer_context`] takes the
//! *local* ancestry [`ChainFrame`] list `Context::run_node`'s own loop
//! builds up as it runs (see that method's own doc) — every case below
//! (child/sibling/parent-climb/root-bypass-fallback) only ever needs
//! *this one call's* ancestry, since nothing in this port populates a
//! `Context`'s ancestry beyond one `run_node` invocation yet (`Workflow`,
//! C0298, and `DynamicNodeScheduler`, C0318/C0319, the only other
//! sources of a deeper chain, aren't built). `curr_index`/
//! `curr_parent_index` (`None` meaning "the context `run_node` was
//! originally called on") stand in for the source's `curr_ctx`/
//! `curr_parent_ctx` object references.
//!
//! **Name-based matching preserved**: every case compares agent *names*,
//! not object identity — matching the source's own
//! `target_agent.name == current_agent.name` (never `is`), which is
//! exactly what makes `test_resolve_and_derive_transfer_context_works_
//! with_cloned_agents` pass in the source (a cloned agent, a distinct
//! object with the same name, still resolves correctly).

use crate::base_agent::BaseAgent;

/// One frame of the local ancestry chain [`crate::context::Context::run_node`]'s
/// loop builds up as its in-place agent-transfer loop runs. `parent`
/// mirrors the source's `Context.parent_ctx`: `None` means "the context
/// `run_node` was originally called on" (the source's `self`), `Some(i)`
/// means an earlier frame in the same chain.
pub(crate) struct ChainFrame {
    pub node_name: String,
    pub parent: Option<usize>,
}

/// `target_agent.name == current_agent.name` — the source's own
/// `ValueError`, raised (not returned) since it signals a caller bug
/// rather than a resolvable routing outcome. Not itself a
/// `rusty_err::Error`/`std::error::Error` — its only caller
/// ([`crate::context::Context::run_node`]) immediately folds it into
/// its own `RunNodeError::SelfTransfer`, so it never needs to be boxed
/// on its own.
#[derive(Debug)]
pub(crate) struct SelfTransferError(pub String);

/// The three non-error outcomes `resolve_and_derive_transfer_context`
/// can reach — the source's own `(target_agent, next_parent_ctx) |
/// (None, None) | (target_agent, None)` tuple shape, split into a named
/// enum instead (a `BaseAgent` handle plus a bare `bool`/`Option` reads
/// less clearly than the source's tuple return, given how many of this
/// port's own `Option`s are already in play here).
pub(crate) enum TransferOutcome {
    /// `root_agent.find_agent(target_name)` found nothing.
    NotFound,
    /// Found, but no self/child/sibling/parent relationship to
    /// `current_agent` — the source's `(target_agent, None)`.
    Unrelated { target_agent: BaseAgent },
    /// Found and routed; `next_parent` is the chain index (or `None` for
    /// "self") the transferred node should run under next.
    Resolved {
        target_agent: BaseAgent,
        next_parent: Option<usize>,
    },
}

// `BaseAgent` has no `Debug` impl (`base_agent.rs`'s own type), so this
// is hand-written rather than derived — enough to distinguish variants
// in a failed test assertion.
impl std::fmt::Debug for TransferOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "NotFound"),
            Self::Unrelated { target_agent } => {
                write!(f, "Unrelated({})", target_agent.name())
            }
            Self::Resolved {
                target_agent,
                next_parent,
            } => write!(
                f,
                "Resolved({}, next_parent={next_parent:?})",
                target_agent.name()
            ),
        }
    }
}

/// `resolve_and_derive_transfer_context`: resolves an agent-transfer
/// target and the correct parent context to run it under, for the
/// self/child/sibling/parent-climb/unrelated relationship cases — see
/// this module's own doc for the `Context`-ancestry adaptation.
pub(crate) fn resolve_and_derive_transfer_context(
    target_name: &str,
    current_agent: &BaseAgent,
    root_agent: &BaseAgent,
    chain: &[ChainFrame],
    curr_index: usize,
    curr_parent_index: Option<usize>,
) -> Result<TransferOutcome, SelfTransferError> {
    let Some(target_agent) = root_agent.find_agent(target_name) else {
        return Ok(TransferOutcome::NotFound);
    };

    // Case 1: SELF (invalid transfer target).
    if target_agent.name() == current_agent.name() {
        return Err(SelfTransferError(target_name.to_string()));
    }

    // Case 2: direct CHILD (nests deeper under the current context).
    if let Some(target_parent) = target_agent.parent_agent() {
        if target_parent.name() == current_agent.name() {
            return Ok(TransferOutcome::Resolved {
                target_agent,
                next_parent: Some(curr_index),
            });
        }
    }

    // Case 3: SIBLING (runs under the same parent context).
    if let (Some(target_parent), Some(current_parent)) =
        (target_agent.parent_agent(), current_agent.parent_agent())
    {
        if target_parent.name() == current_parent.name() {
            return Ok(TransferOutcome::Resolved {
                target_agent,
                next_parent: curr_parent_index,
            });
        }
    }

    // Case 4: direct PARENT (climbs up the context chain to find the
    // parent's parent context).
    if let Some(current_parent) = current_agent.parent_agent() {
        if current_parent.name() == target_agent.name() {
            let mut cursor = Some(curr_index);
            while let Some(i) = cursor {
                if chain[i].node_name == target_name {
                    return Ok(TransferOutcome::Resolved {
                        target_agent,
                        next_parent: chain[i].parent,
                    });
                }
                cursor = chain[i].parent;
            }

            // Root Coordinator / Bypassed parent fallback: the outermost
            // ancestor still reachable from `curr_index`.
            let mut root_index = curr_index;
            while let Some(parent_index) = chain[root_index].parent {
                root_index = parent_index;
            }
            return Ok(TransferOutcome::Resolved {
                target_agent,
                next_parent: Some(root_index),
            });
        }
    }

    // Fallback: target found but has no direct routing relationship.
    Ok(TransferOutcome::Unrelated { target_agent })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_agent::NoopBehavior;

    fn agent(name: &str) -> BaseAgent {
        BaseAgent::new(name, NoopBehavior).unwrap()
    }

    fn agent_with_children(name: &str, children: Vec<BaseAgent>) -> BaseAgent {
        BaseAgent::build(name, "", children, Vec::new(), Vec::new(), NoopBehavior).unwrap()
    }

    #[test]
    fn raises_on_self_transfer() {
        let current = agent("current");
        let root = agent_with_children("root", vec![current.clone()]);
        let err = resolve_and_derive_transfer_context("current", &current, &root, &[], 0, None)
            .unwrap_err();
        assert_eq!(err.0, "current");
    }

    #[test]
    fn returns_current_context_for_a_direct_child_transfer() {
        let target = agent("target");
        let current = agent_with_children("current", vec![target.clone()]);
        let root = agent_with_children("root", vec![current.clone()]);
        // `current` (still a fresh handle with no parent set, since
        // `agent_with_children` builds it before wiring it under `root`)
        // is re-fetched from `root` so `.parent_agent()` resolves.
        let current = root.find_agent("current").unwrap();

        let chain = [ChainFrame {
            node_name: "current".to_string(),
            parent: None,
        }];
        let outcome =
            resolve_and_derive_transfer_context("target", &current, &root, &chain, 0, None)
                .unwrap();
        match outcome {
            TransferOutcome::Resolved {
                target_agent,
                next_parent,
            } => {
                assert_eq!(target_agent.name(), target.name());
                assert_eq!(next_parent, Some(0));
            }
            _ => panic!("expected Resolved"),
        }
    }

    #[test]
    fn returns_parent_context_for_a_sibling_transfer() {
        let current = agent("current");
        let target = agent("target");
        let root = agent_with_children("root", vec![current.clone(), target.clone()]);
        let current = root.find_agent("current").unwrap();

        let chain = [ChainFrame {
            node_name: "current".to_string(),
            parent: None,
        }];
        let outcome =
            resolve_and_derive_transfer_context("target", &current, &root, &chain, 0, Some(7))
                .unwrap();
        match outcome {
            TransferOutcome::Resolved { next_parent, .. } => assert_eq!(next_parent, Some(7)),
            _ => panic!("expected Resolved"),
        }
    }

    #[test]
    fn climbs_the_chain_to_find_the_target_parents_own_parent() {
        let child = agent("child");
        let root = agent_with_children("root", vec![child.clone()]);
        let child = root.find_agent("child").unwrap();

        // chain: [0]="root" (parent: self/None), [1]="child" (parent: 0).
        let chain = [
            ChainFrame {
                node_name: "root".to_string(),
                parent: None,
            },
            ChainFrame {
                node_name: "child".to_string(),
                parent: Some(0),
            },
        ];
        let outcome =
            resolve_and_derive_transfer_context("root", &child, &root, &chain, 1, None).unwrap();
        match outcome {
            TransferOutcome::Resolved {
                target_agent,
                next_parent,
            } => {
                assert_eq!(target_agent.name(), "root");
                // Found "root" at chain[0]; its own parent is self (None).
                assert_eq!(next_parent, None);
            }
            _ => panic!("expected Resolved"),
        }
    }

    #[test]
    fn falls_back_to_the_outermost_context_when_the_parent_was_bypassed() {
        let child = agent("child");
        let root = agent_with_children("root", vec![child.clone()]);
        let child = root.find_agent("child").unwrap();

        // No "root"-named frame in the chain (it was bypassed) — only
        // "child", whose own parent is self.
        let chain = [ChainFrame {
            node_name: "child".to_string(),
            parent: None,
        }];
        let outcome =
            resolve_and_derive_transfer_context("root", &child, &root, &chain, 0, None).unwrap();
        match outcome {
            TransferOutcome::Resolved { next_parent, .. } => {
                // Falls back to the outermost reachable frame: "child" itself.
                assert_eq!(next_parent, Some(0));
            }
            _ => panic!("expected Resolved"),
        }
    }

    #[test]
    fn returns_not_found_when_the_target_agent_is_missing() {
        let current = agent("current");
        let root = agent_with_children("root", vec![current.clone()]);
        let current = root.find_agent("current").unwrap();
        let outcome =
            resolve_and_derive_transfer_context("target", &current, &root, &[], 0, None).unwrap();
        assert!(matches!(outcome, TransferOutcome::NotFound));
    }

    #[test]
    fn returns_unrelated_when_there_is_no_routing_relationship() {
        let current = agent("current");
        let target = agent("target");
        let root1 = agent_with_children("root1", vec![current.clone()]);
        let root2 = agent_with_children("root2", vec![target.clone()]);
        let current = root1.find_agent("current").unwrap();

        let outcome =
            resolve_and_derive_transfer_context("target", &current, &root2, &[], 0, None).unwrap();
        match outcome {
            TransferOutcome::Unrelated { target_agent } => {
                assert_eq!(target_agent.name(), "target");
            }
            _ => panic!("expected Unrelated"),
        }
    }

    #[test]
    fn matches_by_name_so_a_cloned_current_agent_still_resolves() {
        let target = agent("target");
        let current = agent_with_children("current", vec![target.clone()]);
        let root = agent_with_children("root", vec![current.clone()]);
        let current = root.find_agent("current").unwrap();

        let cloned_current = current
            .clone_with(crate::base_agent::BaseAgentUpdate::default())
            .unwrap();
        assert_eq!(cloned_current.name(), current.name());

        let chain = [ChainFrame {
            node_name: "current".to_string(),
            parent: None,
        }];
        let outcome =
            resolve_and_derive_transfer_context("target", &cloned_current, &root, &chain, 0, None)
                .unwrap();
        match outcome {
            TransferOutcome::Resolved { target_agent, .. } => {
                assert_eq!(target_agent.name(), "target");
            }
            _ => panic!("expected Resolved"),
        }
    }
}
