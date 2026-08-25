//! Capability C0323: `ReplaySequenceBarrier`, ported from
//! `google.adk.workflow.utils._replay_sequence_barrier`. Part of the P7
//! workflow/graph engine — see `workflow_rehydration_utils.rs`'s module
//! doc for why this batch (P7 Chunk 4) has no caller yet and is still a
//! legitimate, independently-testable batch.
//!
//! A chronological sequence barrier ensuring deterministic replay
//! ordering: each key in `sequence` unblocks only once every key before
//! it has been marked complete via [`ReplaySequenceBarrier::check_and_advance`].
//! Fully self-contained — no `Context`/`Event`/`BaseNode` coupling at
//! all, so this ports with no narrowing.
//!
//! **`asyncio.Event`, adaptation disclosed**: `rusty_tokio::sync::Notify`
//! is the closest already-adopted equivalent (already a dependency of
//! this crate, already used the same way by `oauth2_util.rs`'s polling
//! wait), but `Notify` has no persistent "is it already set" state the
//! way `asyncio.Event` does — a `notify_waiters()` call before a waiter
//! starts `.notified().await` is lost. This port instead tracks each
//! key's "already unblocked" state explicitly (a `HashSet<String>`
//! alongside the `Notify` map) and consults it before ever waiting, so
//! a key marked complete before a waiter arrives still fast-forwards
//! immediately — the same guarantee `asyncio.Event.is_set()` gives for
//! free.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use rusty_tokio::sync::Notify;

const DEFAULT_TIMEOUT_SEC: f64 = 15.0;

/// `ReplaySequenceBarrier`: unified chronological sequence barrier to
/// ensure deterministic replay ordering.
pub struct ReplaySequenceBarrier {
    sequence: Vec<String>,
    timeout: Duration,
    current_index: usize,
    notifies: HashMap<String, Notify>,
    unblocked: HashSet<String>,
}

impl ReplaySequenceBarrier {
    pub fn new(sequence: Vec<String>) -> Self {
        Self::with_timeout(sequence, Duration::from_secs_f64(DEFAULT_TIMEOUT_SEC))
    }

    pub fn with_timeout(sequence: Vec<String>, timeout: Duration) -> Self {
        let notifies = sequence
            .iter()
            .map(|key| (key.clone(), Notify::new()))
            .collect();
        let mut unblocked = HashSet::new();
        if let Some(first) = sequence.first() {
            unblocked.insert(first.clone());
        }
        Self {
            sequence,
            timeout,
            current_index: 0,
            notifies,
            unblocked,
        }
    }

    /// `ReplaySequenceBarrier.wait`: waits for the barrier if `key` is
    /// part of the expected chronological sequence. Only waits if the
    /// node had a terminal event (output, route, or interrupt) — a
    /// "silent" node that only yielded state updates isn't in the
    /// sequence, so it fast-forwards immediately (the `if key in
    /// self.events` guard in the source).
    pub async fn wait(&self, key: &str) -> Result<(), String> {
        let Some(notify) = self.notifies.get(key) else {
            return Ok(());
        };
        if self.unblocked.contains(key) {
            return Ok(());
        }
        match rusty_tokio::time::timeout(self.timeout, notify.notified()).await {
            Ok(()) => Ok(()),
            Err(_elapsed) => Err(format!(
                "Replay divergence detected: Timed out waiting for sequence key '{key}' to be unblocked."
            )),
        }
    }

    /// `ReplaySequenceBarrier.check_and_advance`: advances the sequence
    /// if `key` matches the current expected execution.
    pub fn check_and_advance(&mut self, key: &str) {
        let Some(expected_key) = self.sequence.get(self.current_index) else {
            return;
        };
        if expected_key != key {
            return;
        }
        self.current_index += 1;
        if let Some(next_key) = self.sequence.get(self.current_index) {
            self.unblocked.insert(next_key.clone());
            if let Some(notify) = self.notifies.get(next_key) {
                notify.notify_waiters();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rusty_tokio::test]
    async fn a_key_not_in_the_sequence_never_blocks() {
        let barrier = ReplaySequenceBarrier::new(vec!["a".to_string()]);
        barrier.wait("not-in-sequence").await.unwrap();
    }

    #[rusty_tokio::test]
    async fn the_first_key_in_the_sequence_is_unblocked_immediately() {
        let barrier = ReplaySequenceBarrier::new(vec!["a".to_string(), "b".to_string()]);
        barrier.wait("a").await.unwrap();
    }

    #[rusty_tokio::test]
    async fn advancing_past_a_key_unblocks_the_next_one() {
        let mut barrier = ReplaySequenceBarrier::new(vec!["a".to_string(), "b".to_string()]);
        barrier.check_and_advance("a");
        barrier.wait("b").await.unwrap();
    }

    #[rusty_tokio::test]
    async fn advancing_with_the_wrong_key_does_not_move_the_sequence() {
        let mut barrier = ReplaySequenceBarrier::with_timeout(
            vec!["a".to_string(), "b".to_string()],
            Duration::from_millis(10),
        );
        barrier.check_and_advance("b");
        let result = barrier.wait("b").await;
        assert!(result.is_err());
    }

    #[rusty_tokio::test]
    async fn waiting_on_an_unadvanced_key_times_out() {
        let barrier = ReplaySequenceBarrier::with_timeout(
            vec!["a".to_string(), "b".to_string()],
            Duration::from_millis(10),
        );
        let err = barrier.wait("b").await.unwrap_err();
        assert!(err.contains("Replay divergence detected"));
    }

    #[rusty_tokio::test]
    async fn advancing_past_every_key_leaves_check_and_advance_a_no_op() {
        let mut barrier = ReplaySequenceBarrier::new(vec!["a".to_string()]);
        barrier.check_and_advance("a");
        // No next key to unblock; a further advance call must not panic.
        barrier.check_and_advance("a");
    }
}
