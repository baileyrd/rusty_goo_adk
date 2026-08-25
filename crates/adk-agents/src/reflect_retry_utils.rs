//! Capability C0370: `plugins._reflect_retry_utils`, ported from
//! `google.adk.plugins._reflect_retry_utils`.
//!
//! **Built ahead of its own (still-blocked) callers**: the source's two
//! consumers, `ReflectAndRetryModelPlugin` (C0368) and
//! `ReflectAndRetryToolPlugin` (C0369), both route through
//! `before_model_callback`/`before_tool_callback`, deferred on the same
//! crate-cycle grounds as C0355/C0356. This module itself imports neither
//! `LlmRequest` nor `BaseTool`/`ToolContext` — it only operates on plain
//! strings and an in-memory counter map — so it has no dependency on that
//! blocked machinery and is safe to land now, ready for C0368/C0369 to
//! consume once C0355/C0356 unblock. Same "widen/build a placeholder ahead
//! of its still-blocked caller" precedent already used by
//! `runner::get_function_responses_from_content` and
//! `session_util.rs`/`artifact_util.rs`.
//!
//! **`asyncio.Lock` -> `std::sync::Mutex`**: the source's async lock only
//! ever guards a synchronous in-memory map mutation, with no `await`
//! inside the critical section — the same "shared counter/store guarded by
//! a lock" shape this crate already uses elsewhere (e.g.
//! `InMemoryMemoryService`, `InMemoryCredentialService`), both via
//! `std::sync::Mutex`. [`ScopedFailureTracker::increment`]/`reset` stay
//! `async fn` to mirror the source's own `async def` signatures, even
//! though nothing inside actually awaits.

use std::collections::HashMap;
use std::sync::Mutex;

/// The reflect-and-retry plugins' sentinel `LlmResponse.custom_metadata`
/// marker, used to recognize a response one of them already handled.
pub const REFLECT_AND_RETRY_RESPONSE_TYPE: &str = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN";

/// The fixed scope key used for [`TrackingScope::Global`].
pub const GLOBAL_SCOPE_KEY: &str = "__global_reflect_and_retry_scope__";

/// `plugins._reflect_retry_utils.TrackingScope` — the lifecycle scope for
/// tracking failure counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingScope {
    Invocation,
    Global,
}

/// Raised by [`resolve_scope_key`] for [`TrackingScope::Invocation`] with
/// no `invocation_id` — matches the source's `raise ValueError`.
#[derive(Debug, rusty_err::Error)]
pub enum MissingInvocationIdError {
    #[error("invocation_id must be provided for INVOCATION scope")]
    MissingInvocationId,
}

/// `plugins._reflect_retry_utils.resolve_scope_key` — the scope key for
/// failure tracking.
pub fn resolve_scope_key(
    scope: TrackingScope,
    invocation_id: Option<&str>,
) -> Result<String, MissingInvocationIdError> {
    match scope {
        TrackingScope::Invocation => invocation_id
            .filter(|id| !id.is_empty())
            .map(|id| id.to_string())
            .ok_or(MissingInvocationIdError::MissingInvocationId),
        TrackingScope::Global => Ok(GLOBAL_SCOPE_KEY.to_string()),
    }
}

/// A mapping from an item's (tool or model) name to its consecutive
/// failure count.
type PerItemFailuresCounter = HashMap<String, i64>;

/// `plugins._reflect_retry_utils.ScopedFailureTracker` — a thread-safe
/// failure counter scoped by invocation or global key.
#[derive(Default)]
pub struct ScopedFailureTracker {
    scoped_failure_counters: Mutex<HashMap<String, PerItemFailuresCounter>>,
}

impl ScopedFailureTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomically increments and returns the failure count for an item.
    pub async fn increment(&self, scope_key: &str, item_name: &str) -> i64 {
        let mut counters = self
            .scoped_failure_counters
            .lock()
            .expect("scoped failure tracker mutex poisoned");
        let failure_counter = counters.entry(scope_key.to_string()).or_default();
        let current = failure_counter.get(item_name).copied().unwrap_or(0) + 1;
        failure_counter.insert(item_name.to_string(), current);
        current
    }

    /// Atomically resets the failure count for an item and cleans up state.
    pub async fn reset(&self, scope_key: &str, item_name: &str) {
        let mut counters = self
            .scoped_failure_counters
            .lock()
            .expect("scoped failure tracker mutex poisoned");
        if let Some(counter) = counters.get_mut(scope_key) {
            counter.remove(item_name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_scope_key_uses_the_invocation_id_for_invocation_scope() {
        let key = resolve_scope_key(TrackingScope::Invocation, Some("inv-1")).unwrap();
        assert_eq!(key, "inv-1");
    }

    #[test]
    fn resolve_scope_key_errors_without_an_invocation_id_for_invocation_scope() {
        assert!(resolve_scope_key(TrackingScope::Invocation, None).is_err());
        assert!(resolve_scope_key(TrackingScope::Invocation, Some("")).is_err());
    }

    #[test]
    fn resolve_scope_key_uses_the_global_constant_for_global_scope() {
        let key = resolve_scope_key(TrackingScope::Global, None).unwrap();
        assert_eq!(key, GLOBAL_SCOPE_KEY);
    }

    #[rusty_tokio::test]
    async fn increment_starts_at_one_and_accumulates() {
        let tracker = ScopedFailureTracker::new();
        assert_eq!(tracker.increment("inv-1", "tool_a").await, 1);
        assert_eq!(tracker.increment("inv-1", "tool_a").await, 2);
        assert_eq!(tracker.increment("inv-1", "tool_a").await, 3);
    }

    #[rusty_tokio::test]
    async fn increment_tracks_items_independently_within_a_scope() {
        let tracker = ScopedFailureTracker::new();
        tracker.increment("inv-1", "tool_a").await;
        tracker.increment("inv-1", "tool_a").await;
        assert_eq!(tracker.increment("inv-1", "tool_b").await, 1);
    }

    #[rusty_tokio::test]
    async fn increment_tracks_scopes_independently() {
        let tracker = ScopedFailureTracker::new();
        tracker.increment("inv-1", "tool_a").await;
        tracker.increment("inv-1", "tool_a").await;
        assert_eq!(tracker.increment("inv-2", "tool_a").await, 1);
    }

    #[rusty_tokio::test]
    async fn reset_removes_only_the_targeted_item() {
        let tracker = ScopedFailureTracker::new();
        tracker.increment("inv-1", "tool_a").await;
        tracker.increment("inv-1", "tool_b").await;

        tracker.reset("inv-1", "tool_a").await;

        assert_eq!(tracker.increment("inv-1", "tool_a").await, 1);
        assert_eq!(tracker.increment("inv-1", "tool_b").await, 2);
    }

    #[rusty_tokio::test]
    async fn reset_is_a_no_op_for_an_unknown_scope_or_item() {
        let tracker = ScopedFailureTracker::new();
        tracker.reset("no-such-scope", "no-such-item").await;

        tracker.increment("inv-1", "tool_a").await;
        tracker.reset("inv-1", "no-such-item").await;
        assert_eq!(tracker.increment("inv-1", "tool_a").await, 2);
    }
}
