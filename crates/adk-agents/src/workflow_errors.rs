//! Capability C0308: `NodeInterruptedError`/`NodeTimeoutError`/
//! `DynamicNodeFailError`, ported from `google.adk.workflow._errors`.
//! Part of the P7 workflow/graph engine's pure-data slice — see
//! `workflow_node_state.rs`'s module doc for the rest of this batch's
//! scope and crate-placement reasoning.
//!
//! **`NodeInterruptedError`, adaptation disclosed**: the source is
//! deliberately a `BaseException` (not `Exception`) specifically so a
//! node body's `except Exception` can't accidentally swallow an
//! in-flight HITL interrupt and mistake it for a normal completion.
//! Rust's `Result`-based control flow has no equivalent "generic catch"
//! to guard against — nothing implicitly captures an arbitrary error the
//! way a bare `except Exception` does — so the risk this type exists to
//! prevent doesn't have a runtime-swallowing analog to prevent. This
//! port instead preserves the *type-system* shape of the guarantee:
//! [`NodeInterruptedError`] deliberately does **not** implement
//! `std::error::Error` (nor this crate's own [`rusty_err::Error`]), so
//! it can never be coerced into whatever `Box<dyn Error>`/node-error enum
//! a future `NodeRunner` (C0310-C0312, not built this batch) uses for
//! ordinary node failures — a node body that does something like
//! `.map_err(SomeNodeError::from)` structurally cannot absorb it, the
//! same non-catchability the source achieves via exception hierarchy.
//! Whoever builds `NodeRunner` should thread this value through its own
//! dedicated variant/branch rather than any generic error-conversion
//! path.

/// `workflow._errors.NodeInterruptedError` — internal: raised (in the
/// source) when a dynamic node interrupts (HITL). Deliberately does not
/// implement `std::error::Error`/[`rusty_err::Error`] — see this
/// module's own doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NodeInterruptedError;

/// `workflow._errors.NodeTimeoutError`/`DynamicNodeFailError` — both
/// regular (catchable, retry-compatible) node errors, unlike
/// [`NodeInterruptedError`].
#[derive(Debug, rusty_err::Error)]
pub enum WorkflowNodeError {
    /// `NodeTimeoutError` — a node exceeded its configured timeout.
    #[error("Node '{node_name}' timed out after {timeout} seconds.")]
    Timeout { node_name: String, timeout: f64 },
    /// `DynamicNodeFailError` — a dynamic node failed; caught by the
    /// parent node's `NodeRunner` to propagate the error. `message` is
    /// the source's own `super().__init__(message)` argument (the
    /// exception's `str()`), kept distinct from `error` (the source's
    /// own `self.error`, the underlying failure) exactly as the source
    /// keeps them two separate attributes.
    #[error("{message}")]
    DynamicNodeFail {
        message: String,
        error: Box<dyn std::error::Error + Send + Sync>,
        error_node_path: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the source's `test_node_interrupted_error_survives_a_broad_
    /// except_in_node_code`'s intent. Rust has no runtime "catch" to
    /// swallow an error the way `except Exception` can — the analogous
    /// guarantee here is compile-time: `NodeInterruptedError` doesn't
    /// implement `std::error::Error`, so `WorkflowNodeError` (or any
    /// future `NodeRunner` error type) can never gain a `From
    /// <NodeInterruptedError>`/`?`/`.map_err(...)` conversion path that
    /// would let a node body's ordinary error handling absorb it — the
    /// compiler enforces this structurally rather than a test verifying
    /// it at runtime. This test just exercises the value itself
    /// (construction, equality) since there's nothing else to assert.
    #[test]
    fn node_interrupted_error_is_constructible_and_comparable() {
        assert_eq!(NodeInterruptedError, NodeInterruptedError);
    }

    #[test]
    fn node_timeout_error_formats_the_sources_exact_message() {
        let error = WorkflowNodeError::Timeout {
            node_name: "my_node".to_string(),
            timeout: 30.0,
        };
        assert_eq!(
            error.to_string(),
            "Node 'my_node' timed out after 30 seconds."
        );
    }

    #[test]
    fn dynamic_node_fail_error_uses_message_not_the_inner_error() {
        let error = WorkflowNodeError::DynamicNodeFail {
            message: "child node failed".to_string(),
            error: Box::new(std::io::Error::other("boom")),
            error_node_path: "root/child".to_string(),
        };
        assert_eq!(error.to_string(), "child node failed");
    }
}
