//! Capability C0935: `missing_extra`, ported from
//! `google.adk.utils._dependency`.
//!
//! **Adaptation**: the source builds and returns a Python `ImportError` for
//! the caller to `raise`. `ImportError` is a plain builtin exception, not
//! one of `google.adk.errors`'s own six ADK-specific types (see this
//! crate's own module doc on the `ValueError`-vs-`Exception` split) — this
//! port therefore doesn't wrap it in a new error type of its own, and
//! doesn't implement [`crate::ValueErrorLike`]. It returns the formatted
//! message as a plain `String`, matching the "no domain-specific exception
//! type for a plain builtin exception" convention already used for
//! `ValueError`-raising helpers elsewhere in this port (e.g.
//! `adk-genai::json_utils::safe_json_loads`). No caller in this workspace
//! needs it yet — none of the source's optional-dependency-gated
//! subsystems that call it (`DatabaseSessionService`, `VertexAiSessionService`,
//! the A2A agent, the BigQuery analytics plugin) are built here — the same
//! "capability real and independently testable ahead of its own caller"
//! shape as `SKIP_THOUGHT_SIGNATURE_VALIDATOR` (C0929).

/// `_dependency.missing_extra` — the standard message for a missing
/// optional dependency.
pub fn missing_extra(package: &str, extra: &str) -> String {
    format!(
        "The '{package}' package is required to use this feature. \
         Please install it by running: pip install google-adk[{extra}]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_extra_names_the_package_and_extras_group() {
        let message = missing_extra("sqlalchemy", "db");
        assert!(message.contains("'sqlalchemy'"));
        assert!(message.contains("google-adk[db]"));
    }
}
