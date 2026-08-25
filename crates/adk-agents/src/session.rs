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
//!
//! **`last_update_time` (C0204, closed)**: added once
//! `InMemorySessionService::list_sessions` (C0208) needed it to sort by.
//! The wire shape (`alias_generator=to_camel` + `extra='forbid'`) is fully
//! specified by `session.py` itself and doesn't actually require a real
//! backend to exist to encode correctly, so it's ported here too, via the
//! same `#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]`
//! pattern already used by `TaskRequest`/`ArtifactVersion`/`GetSessionConfig`.
//! The private `_storage_update_marker` optimistic-concurrency field is
//! also added, `#[rusty_serde(skip)]`-annotated (never serialized, matching
//! the source's `PrivateAttr`) and crate-private — it has no reader yet
//! (its real, near-term caller is `DatabaseSessionService`, C0221, the only
//! backend that will ever compare/bump it), the same "ahead of its own
//! caller" precedent already used by `GetSessionConfig`/`extract_state_delta`.
//! Adding these derives/fields is additive, not breaking: nothing in this
//! crate serializes or deserializes a `Session` today (only `Session::new`
//! constructor calls exist), so every existing call site stays valid.

use adk_events::Event;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Session {
    pub id: String,
    pub app_name: String,
    pub user_id: String,
    pub state: BTreeMap<String, Value>,
    pub events: Vec<Event>,
    /// C0204: Unix timestamp (seconds) of the most recent write to this
    /// session — set at creation and bumped on every `append_event`.
    /// `InMemorySessionService::list_sessions` sorts on this field
    /// (`ListSessionsResponse`, C0208).
    pub last_update_time: f64,
    /// C0204: optimistic-concurrency marker for a future persistent
    /// backend (`DatabaseSessionService`, C0221) — never serialized, and
    /// unread by anything in this crate today.
    #[rusty_serde(skip)]
    #[allow(dead_code)]
    pub(crate) storage_update_marker: Option<String>,
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
            last_update_time: adk_platform::time::get_time(),
            storage_update_marker: None,
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

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let session = Session::new("app", "user", "sess-1");
        let json = rusty_serde::json::to_string(&session).unwrap();
        assert!(json.contains("\"appName\":\"app\""));
        assert!(json.contains("\"userId\":\"user\""));
        assert!(json.contains("\"lastUpdateTime\":"));
        let back: Session = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(back.id, "sess-1");
    }

    #[test]
    fn rejects_an_unknown_field() {
        let json = r#"{"id":"s","appName":"a","userId":"u","state":{},"events":[],"lastUpdateTime":0.0,"bogus":true}"#;
        assert!(rusty_serde::json::from_str::<Session>(json).is_err());
    }

    #[test]
    fn storage_update_marker_is_never_serialized() {
        let mut session = Session::new("app", "user", "sess-1");
        session.storage_update_marker = Some("etag-1".to_string());
        let json = rusty_serde::json::to_string(&session).unwrap();
        assert!(!json.contains("storageUpdateMarker"));
        assert!(!json.contains("storage_update_marker"));
    }
}
