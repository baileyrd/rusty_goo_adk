//! `Session`, a placeholder for `google.adk.sessions.session.Session`
//! (Phase 5).
//!
//! **Disclosed forward-reference stub**: `InvocationContext`/`Context`/
//! `ReadonlyContext` all require a `Session` (its `state`, `id`, `app_name`,
//! `user_id`, `events`) to exist and compile — the same situation `State`
//! is in (see `state.rs`'s module doc). Unlike `State`, a faithful `Session`
//! needs the 4 real session backends and their schemas (Phase 5's actual
//! scope), so this is a genuine placeholder, not a forward-pull: only the
//! fields `agents/` code actually reads are present, with the minimum shape
//! to be useful in tests. It will be replaced (not extended) by the real
//! `adk-sessions` crate's `Session` when Phase 5 lands — every field here
//! has a same-named counterpart in the source's real `Session` model.

use adk_events::Event;
use rusty_serde::value::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct Session {
    pub id: String,
    pub app_name: String,
    pub user_id: String,
    pub state: BTreeMap<String, Value>,
    pub events: Vec<Event>,
}

impl Session {
    pub fn new(
        app_name: impl Into<String>,
        user_id: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            app_name: app_name.into(),
            user_id: user_id.into(),
            state: BTreeMap::new(),
            events: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_with_empty_state_and_events() {
        let session = Session::new("app", "user", "sess-1");
        assert!(session.state.is_empty());
        assert!(session.events.is_empty());
        assert_eq!(session.id, "sess-1");
    }
}
