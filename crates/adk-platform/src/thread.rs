//! Thread creation — capability C0005.
//!
//! Ports `google.adk.platform.thread.create_thread`: delegates to an
//! internal, platform-specific thread factory if one is installed, else
//! falls back to a plain spawn. The source's "internal factory" branch is
//! confirmed unreachable in the OSS tree itself (no such internal module
//! ships there — see the source inventory's suspected-dead-code note on
//! this exact branch); this crate accordingly starts with just the public
//! fallback path that's actually live, rather than inventing a factory
//! hook nothing in the source repo ever populates.

use std::thread::{self, JoinHandle};

/// Spawns a new OS thread running `f`, mirroring the source's
/// `create_thread` fallback behavior (a plain `threading.Thread` in the
/// source, `std::thread::spawn` here).
pub fn create_thread<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    thread::spawn(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test for capability C0005: `create_thread` runs the given
    /// closure on a separate thread and returns its result via the handle.
    #[test]
    fn create_thread_runs_closure_and_returns_result() {
        let handle = create_thread(|| 21 + 21);
        assert_eq!(handle.join().unwrap(), 42);
    }
}
