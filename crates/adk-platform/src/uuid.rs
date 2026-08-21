//! UUID provider — a scoped, overridable source of unique-id generation.
//!
//! Ports `google.adk.platform.uuid` (capability C0004): `set_id_provider`/
//! `reset_id_provider` (default: a random v4 UUID string)/`new_uuid`, the
//! same provider-swap pattern as [`crate::random`] and [`crate::time`].
//! Default generation is delegated to the sibling `rusty_uuid` crate
//! (zero-dependency, `std`-based v4 UUID generation) rather than
//! hand-rolled here, since RFC 4122 UUID generation is exactly the kind of
//! narrowly-scoped, already-solved capability the platform's sibling-check
//! step exists to catch before reimplementing it a second time.

use std::cell::RefCell;
use std::rc::Rc;

type Provider = Rc<dyn Fn() -> String>;

thread_local! {
    static ID_PROVIDER: RefCell<Option<Provider>> = const { RefCell::new(None) };
}

fn default_id() -> String {
    rusty_uuid::Uuid::new_v4().to_string()
}

/// Installs a callable provider, evaluated on every [`new_uuid`] call.
pub fn set_id_provider<F>(provider: F)
where
    F: Fn() -> String + 'static,
{
    ID_PROVIDER.with(|p| *p.borrow_mut() = Some(Rc::new(provider)));
}

/// Restores the default (random v4 UUID) provider.
pub fn reset_id_provider() {
    ID_PROVIDER.with(|p| *p.borrow_mut() = None);
}

/// Returns a new unique id string from the currently active provider.
pub fn new_uuid() -> String {
    let installed = ID_PROVIDER.with(|p| p.borrow().clone());
    installed.map(|f| f()).unwrap_or_else(default_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test for capability C0004: default provider returns a
    /// well-formed, unique v4 UUID string each call.
    #[test]
    fn default_provider_returns_unique_uuids() {
        reset_id_provider();
        let a = new_uuid();
        let b = new_uuid();
        assert_ne!(a, b);
        assert_eq!(a.len(), 36, "expected standard 8-4-4-4-12 hyphenated form");
    }

    /// Parity test for capability C0004: an installed provider overrides
    /// the default and is re-evaluated per call.
    #[test]
    fn installed_provider_overrides_default() {
        set_id_provider(|| "fixed-id".to_string());
        assert_eq!(new_uuid(), "fixed-id");
        assert_eq!(new_uuid(), "fixed-id");
        reset_id_provider();
        assert_ne!(new_uuid(), "fixed-id");
    }
}
