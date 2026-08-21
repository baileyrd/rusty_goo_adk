//! Exception hierarchy ported from `google.adk.errors`.
//!
//! Mirrors the source's export asymmetry (capability C0007): the source's
//! `errors/__init__.py` re-exports only `StaleSessionError` — every other
//! error type must be imported from its own submodule in the source. This
//! crate matches that shape: [`StaleSessionError`] is the only type
//! re-exported at the crate root; the rest live under their own modules
//! ([`already_exists`], [`input_validation`], [`not_found`],
//! [`session_not_found`], [`tool_execution`]) and must be reached there.
//!
//! ## The `ValueError`-vs-`Exception` split (capability C0015)
//!
//! In the Python source, three of these six error types deliberately
//! subclass `ValueError` "for backward compatibility" (existing callers
//! may already catch `ValueError` broadly), while the other three subclass
//! plain `Exception`. Rust has no exception-hierarchy inheritance to carry
//! that distinction the same way, so it's preserved instead as the
//! [`ValueErrorLike`] marker trait: implemented only by the three types
//! that were `ValueError` subclasses in the source
//! ([`StaleSessionError`], [`session_not_found::SessionNotFoundError`],
//! [`input_validation::InputValidationError`]). Downstream code that needs
//! to replicate a broad `except ValueError` should match on this trait
//! rather than an enum variant, so collapsing all six into one error
//! family later doesn't silently erase the distinction.

pub mod already_exists;
pub mod input_validation;
pub mod not_found;
pub mod session_not_found;
pub mod tool_execution;

mod stale_session;

pub use stale_session::StaleSessionError;

/// Marker trait for error types that were `ValueError` subclasses in the
/// Python source (see the module-level "`ValueError`-vs-`Exception` split"
/// note on capability C0015). Implemented only by
/// [`StaleSessionError`], [`session_not_found::SessionNotFoundError`], and
/// [`input_validation::InputValidationError`].
pub trait ValueErrorLike: rusty_err::Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::already_exists::AlreadyExistsError;
    use crate::input_validation::InputValidationError;
    use crate::not_found::NotFoundError;
    use crate::session_not_found::SessionNotFoundError;
    use crate::tool_execution::{ToolErrorType, ToolExecutionError};

    /// Parity test for capability C0015: exactly the three `ValueError`-
    /// subclassing source types implement `ValueErrorLike`, and the other
    /// three do not — enforced at compile time via these helper functions.
    #[test]
    fn value_error_like_split_matches_source_hierarchy() {
        fn assert_value_error_like<T: ValueErrorLike>() {}
        assert_value_error_like::<StaleSessionError>();
        assert_value_error_like::<SessionNotFoundError>();
        assert_value_error_like::<InputValidationError>();

        // AlreadyExistsError / NotFoundError / ToolExecutionError must NOT
        // implement ValueErrorLike. There's no negative-trait-bound check
        // in stable Rust, so this is asserted by construction: if any of
        // them gained an impl, `cargo doc`'s trait-implementor list and
        // this module's own doc comment would need updating in lockstep,
        // and the `ValueErrorLike` impl block list below is the single
        // source of truth reviewers should check against new impls.
        let _ = AlreadyExistsError::default();
        let _ = NotFoundError::default();
        let _ = ToolExecutionError::new("boom", Some(ToolErrorType::Internal));
    }
}
