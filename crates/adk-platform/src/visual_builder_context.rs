//! Capability C0936: `_is_visual_builder`, ported from
//! `google.adk.utils._telemetry_context`.
//!
//! **Adaptation**: the source's `contextvars.ContextVar[bool]` is
//! async-task-scoped. This port uses a `thread_local!` `Cell<bool>`
//! instead — the same disclosed thread-vs-task-scoping narrowing already
//! used by `adk-models::google_client_headers::ClientLabelScope` (C0932);
//! see that module's doc for the full rationale. No caller in this
//! workspace sets this flag yet (the source's own callers — the BigQuery
//! integration, `cli/api_server.py`'s `/run`-family endpoints — aren't
//! built here), the same "capability real, caller not yet built" shape as
//! elsewhere in this port.

use std::cell::Cell;

thread_local! {
    static IS_VISUAL_BUILDER: Cell<bool> = const { Cell::new(false) };
}

/// `_telemetry_context._is_visual_builder.get()`.
pub fn is_visual_builder() -> bool {
    IS_VISUAL_BUILDER.with(Cell::get)
}

/// `_telemetry_context._is_visual_builder.set(value)`.
pub fn set_visual_builder(value: bool) {
    IS_VISUAL_BUILDER.with(|flag| flag.set(value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_false() {
        assert!(!is_visual_builder());
    }

    #[test]
    fn set_visual_builder_updates_the_flag() {
        set_visual_builder(true);
        assert!(is_visual_builder());
        set_visual_builder(false);
        assert!(!is_visual_builder());
    }
}
