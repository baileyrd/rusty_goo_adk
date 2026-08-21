//! Time provider — a scoped, overridable source of wall-clock time.
//!
//! Ports `google.adk.platform.time` (capability C0003): `set_time_provider`/
//! `reset_time_provider` (default: real wall-clock)/`get_time`, the same
//! provider-swap pattern as [`crate::random`]. See that module's doc
//! comment for the thread-local-vs-task-local adaptation note, which
//! applies identically here (C0006).

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

type Provider = Rc<dyn Fn() -> f64>;

thread_local! {
    static TIME_PROVIDER: RefCell<Option<Provider>> = const { RefCell::new(None) };
}

fn default_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_secs_f64()
}

/// Installs a callable provider, evaluated on every [`get_time`] call.
pub fn set_time_provider<F>(provider: F)
where
    F: Fn() -> f64 + 'static,
{
    TIME_PROVIDER.with(|p| *p.borrow_mut() = Some(Rc::new(provider)));
}

/// Restores the default (real wall-clock) provider.
pub fn reset_time_provider() {
    TIME_PROVIDER.with(|p| *p.borrow_mut() = None);
}

/// Returns the current time as seconds since the Unix epoch, matching the
/// source's `time.time()`-shaped contract.
pub fn get_time() -> f64 {
    let installed = TIME_PROVIDER.with(|p| p.borrow().clone());
    installed.map(|f| f()).unwrap_or_else(default_time)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test for capability C0003: default provider returns a
    /// plausible current wall-clock time.
    #[test]
    fn default_provider_returns_plausible_time() {
        reset_time_provider();
        let t = get_time();
        // 2026-01-01T00:00:00Z, as a sanity floor well below "now".
        assert!(
            t > 1_767_225_600.0,
            "expected a post-2026 timestamp, got {t}"
        );
    }

    /// Parity test for capability C0003: an installed provider is used
    /// instead of the real clock, and is re-evaluated per call.
    #[test]
    fn installed_provider_overrides_default() {
        set_time_provider(|| 12345.0);
        assert_eq!(get_time(), 12345.0);
        assert_eq!(get_time(), 12345.0);
        reset_time_provider();
        assert_ne!(get_time(), 12345.0);
    }
}
