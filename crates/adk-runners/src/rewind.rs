//! Capabilities C0892/C0893: the two delta helpers behind
//! [`crate::runner::Runner::rewind_async`] (C0891/C0894), ported from
//! `runners.py`. Split into their own module as pure(-ish) functions
//! taking explicit parameters rather than `&Runner`, so they're
//! testable independent of a full `Runner` — the same "land the
//! self-contained pieces, wire them from the method" split this port's
//! `apps_compaction.rs` batches already established.

use std::collections::HashMap;

use adk_agents::services::ArtifactService;
use adk_agents::session::Session;
use adk_genai::content::{MediaBlobStub, Part};
use rusty_serde::value::Value;

/// C0892: `_compute_state_delta_for_rewind` — replays all state deltas
/// strictly before `rewind_event_index` (skipping `app:`/`user:`-scoped
/// keys entirely — those aren't session-scoped) to reconstruct the
/// state as of the rewind point, then diffs that against the session's
/// *current* state to build a delta that both restores/updates any
/// differing key and clears (sets [`Value::Null`]) every current
/// non-`app:`/`user:` key that's absent at the rewind point. A
/// `Value::Null` entry in a historical `state_delta` is itself a
/// tombstone — the same "explicit `None` deletes the key" convention
/// the source's own replay already relies on.
pub fn compute_state_delta_for_rewind(
    session: &Session,
    rewind_event_index: usize,
) -> HashMap<String, Value> {
    let mut state_at_rewind_point: HashMap<String, Value> = HashMap::new();
    for event in &session.events[..rewind_event_index] {
        for (key, value) in &event.actions.state_delta {
            if key.starts_with("app:") || key.starts_with("user:") {
                continue;
            }
            if matches!(value, Value::Null) {
                state_at_rewind_point.remove(key);
            } else {
                state_at_rewind_point.insert(key.clone(), value.clone());
            }
        }
    }

    let mut rewind_state_delta: HashMap<String, Value> = HashMap::new();

    for (key, value_at_rewind) in &state_at_rewind_point {
        let differs = session
            .state
            .get(key)
            .map(|current| current != value_at_rewind)
            .unwrap_or(true);
        if differs {
            rewind_state_delta.insert(key.clone(), value_at_rewind.clone());
        }
    }

    for key in session.state.keys() {
        if key.starts_with("app:") || key.starts_with("user:") {
            continue;
        }
        if !state_at_rewind_point.contains_key(key) {
            rewind_state_delta.insert(key.clone(), Value::Null);
        }
    }

    rewind_state_delta
}

/// A `Part` marking an artifact as inaccessible — the source's
/// `types.Part(inline_data=types.Blob(mime_type='application/octet-stream',
/// data=b''))`, represented the same way `save_files_as_artifacts_plugin.rs`
/// represents a stored artifact (`rusty_serde::json::to_value` of a
/// `Part`). Empty bytes base64-encode to the empty string, so no actual
/// base64 encoding is needed here.
fn inaccessible_artifact_placeholder() -> Value {
    let part = Part {
        inline_data: Some(MediaBlobStub {
            mime_type: Some("application/octet-stream".to_string()),
            rest: Some(Value::Map(vec![(
                "data".to_string(),
                Value::String(String::new()),
            )])),
        }),
        ..Default::default()
    };
    rusty_serde::json::to_value(&part).unwrap_or(Value::Null)
}

/// C0893: `_compute_artifact_delta_for_rewind` — `{}` if no artifact
/// service is configured. For each artifact filename whose version at
/// the rewind point differs from its current version (scanned across
/// *every* event, not just those before the rewind point — the current
/// version reflects the whole unrewound history), bumps the delta to
/// `current_version + 1` (rewind restores the old content as a NEW
/// version, never rewriting history) and re-persists the historical
/// content — or [`inaccessible_artifact_placeholder`] if the filename
/// didn't exist yet at the rewind point, or if the historical version
/// is unexpectedly missing from the backend. `user:`-scoped artifact
/// filenames are skipped entirely (never restored on rewind).
pub fn compute_artifact_delta_for_rewind(
    artifact_service: Option<&(dyn ArtifactService + Send + Sync)>,
    app_name: &str,
    session: &Session,
    rewind_event_index: usize,
) -> HashMap<String, i64> {
    let Some(artifact_service) = artifact_service else {
        return HashMap::new();
    };

    let mut versions_at_rewind_point: HashMap<String, i64> = HashMap::new();
    for event in &session.events[..rewind_event_index] {
        versions_at_rewind_point.extend(event.actions.artifact_delta.clone());
    }

    let mut current_versions: HashMap<String, i64> = HashMap::new();
    for event in &session.events {
        current_versions.extend(event.actions.artifact_delta.clone());
    }

    let mut rewind_artifact_delta: HashMap<String, i64> = HashMap::new();
    for (filename, current_version) in current_versions {
        if filename.starts_with("user:") {
            continue;
        }
        let version_at_rewind = versions_at_rewind_point.get(&filename).copied();
        if version_at_rewind == Some(current_version) {
            continue;
        }

        rewind_artifact_delta.insert(filename.clone(), current_version + 1);

        let artifact = match version_at_rewind {
            None => inaccessible_artifact_placeholder(),
            Some(version) => artifact_service
                .load_artifact(
                    app_name,
                    &session.user_id,
                    &session.id,
                    &filename,
                    Some(version),
                )
                .unwrap_or_else(inaccessible_artifact_placeholder),
        };

        artifact_service.save_artifact(
            app_name,
            &session.user_id,
            &session.id,
            &filename,
            artifact,
            None,
        );
    }

    rewind_artifact_delta
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_events::{Event, EventActions};
    use std::sync::Mutex;

    fn event_with_state_delta(delta: Vec<(&str, Value)>) -> Event {
        let mut event = Event::new("inv-1", "user", NodeInfo::new(""));
        event.actions = EventActions {
            state_delta: delta.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            ..Default::default()
        };
        event
    }

    fn session_with_events(state: Vec<(&str, Value)>, events: Vec<Event>) -> Session {
        let mut session = Session::new("app", "user", "s1");
        session.state = state.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        session.events = events;
        session
    }

    #[test]
    fn state_delta_restores_a_key_changed_after_the_rewind_point() {
        let events = vec![event_with_state_delta(vec![(
            "k",
            Value::String("old".to_string()),
        )])];
        let session = session_with_events(vec![("k", Value::String("new".to_string()))], events);
        let delta = compute_state_delta_for_rewind(&session, 1);
        assert_eq!(delta.get("k"), Some(&Value::String("old".to_string())));
    }

    #[test]
    fn state_delta_clears_a_key_set_only_after_the_rewind_point() {
        let events = vec![
            Event::new("inv-1", "user", NodeInfo::new("")),
            event_with_state_delta(vec![("k", Value::String("new".to_string()))]),
        ];
        let session = session_with_events(vec![("k", Value::String("new".to_string()))], events);
        // rewind_event_index=1: only the first (no-op) event is replayed.
        let delta = compute_state_delta_for_rewind(&session, 1);
        assert_eq!(delta.get("k"), Some(&Value::Null));
    }

    #[test]
    fn state_delta_ignores_app_and_user_scoped_keys() {
        let events = vec![event_with_state_delta(vec![
            ("app:k", Value::String("old".to_string())),
            ("user:k", Value::String("old".to_string())),
        ])];
        let session = session_with_events(
            vec![
                ("app:k", Value::String("new".to_string())),
                ("user:k", Value::String("new".to_string())),
            ],
            events,
        );
        let delta = compute_state_delta_for_rewind(&session, 1);
        assert!(delta.is_empty());
    }

    #[test]
    fn state_delta_treats_a_null_historical_entry_as_a_tombstone() {
        let events = vec![
            event_with_state_delta(vec![("k", Value::String("v".to_string()))]),
            event_with_state_delta(vec![("k", Value::Null)]),
        ];
        // Replaying both prior events: set then tombstoned -> absent at
        // the rewind point, so current "k" must be cleared.
        let session =
            session_with_events(vec![("k", Value::String("current".to_string()))], events);
        let delta = compute_state_delta_for_rewind(&session, 2);
        assert_eq!(delta.get("k"), Some(&Value::Null));
    }

    #[test]
    fn state_delta_leaves_an_unchanged_key_out_of_the_delta() {
        let events = vec![event_with_state_delta(vec![(
            "k",
            Value::String("same".to_string()),
        )])];
        let session = session_with_events(vec![("k", Value::String("same".to_string()))], events);
        let delta = compute_state_delta_for_rewind(&session, 1);
        assert!(delta.is_empty());
    }

    struct StubArtifactService {
        stored: Mutex<HashMap<String, (i64, Value)>>,
    }

    impl StubArtifactService {
        fn with_version(filename: &str, version: i64, content: Value) -> Self {
            let mut stored = HashMap::new();
            stored.insert(filename.to_string(), (version, content));
            Self {
                stored: Mutex::new(stored),
            }
        }
    }

    impl ArtifactService for StubArtifactService {
        fn load_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            version: Option<i64>,
        ) -> Option<Value> {
            let stored = self.stored.lock().unwrap();
            let (stored_version, content) = stored.get(filename)?;
            if Some(*stored_version) == version {
                Some(content.clone())
            } else {
                None
            }
        }

        fn save_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            artifact: Value,
            _custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
        ) -> i64 {
            let mut stored = self.stored.lock().unwrap();
            let next_version = stored.get(filename).map(|(v, _)| v + 1).unwrap_or(1);
            stored.insert(filename.to_string(), (next_version, artifact));
            next_version
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<adk_agents::services::ArtifactVersion> {
            None
        }

        fn list_artifact_keys(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
        ) -> Vec<String> {
            self.stored.lock().unwrap().keys().cloned().collect()
        }

        fn delete_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) {
        }

        fn list_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<i64> {
            Vec::new()
        }

        fn list_artifact_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<adk_agents::services::ArtifactVersion> {
            Vec::new()
        }
    }

    fn event_with_artifact_delta(filename: &str, version: i64) -> Event {
        let mut event = Event::new("inv-1", "user", NodeInfo::new(""));
        event.actions = EventActions {
            artifact_delta: [(filename.to_string(), version)].into_iter().collect(),
            ..Default::default()
        };
        event
    }

    #[test]
    fn artifact_delta_is_empty_without_an_artifact_service() {
        let session = session_with_events(vec![], vec![]);
        let delta = compute_artifact_delta_for_rewind(None, "app", &session, 0);
        assert!(delta.is_empty());
    }

    #[test]
    fn artifact_delta_restores_and_bumps_a_changed_artifacts_version() {
        let service =
            StubArtifactService::with_version("f.txt", 1, Value::String("v1 content".to_string()));
        let events = vec![
            event_with_artifact_delta("f.txt", 1),
            event_with_artifact_delta("f.txt", 2),
        ];
        let session = session_with_events(vec![], events);

        let delta = compute_artifact_delta_for_rewind(Some(&service), "app", &session, 1);
        // Current version is 2; rewind point saw version 1 -> restore
        // v1's content as a brand-new version, bumped to 3.
        assert_eq!(delta.get("f.txt"), Some(&3));
    }

    #[test]
    fn artifact_delta_marks_an_artifact_inaccessible_when_it_didnt_exist_at_the_rewind_point() {
        let service =
            StubArtifactService::with_version("f.txt", 1, Value::String("content".to_string()));
        // No events before the rewind point -> "f.txt" didn't exist yet.
        let events = vec![event_with_artifact_delta("f.txt", 1)];
        let session = session_with_events(vec![], events);

        let delta = compute_artifact_delta_for_rewind(Some(&service), "app", &session, 0);
        assert_eq!(delta.get("f.txt"), Some(&2));
    }

    #[test]
    fn artifact_delta_skips_user_scoped_filenames() {
        let service = StubArtifactService::with_version(
            "user:f.txt",
            1,
            Value::String("content".to_string()),
        );
        let events = vec![
            event_with_artifact_delta("user:f.txt", 1),
            event_with_artifact_delta("user:f.txt", 2),
        ];
        let session = session_with_events(vec![], events);

        let delta = compute_artifact_delta_for_rewind(Some(&service), "app", &session, 1);
        assert!(delta.is_empty());
    }

    #[test]
    fn artifact_delta_skips_an_artifact_unchanged_since_the_rewind_point() {
        let events = vec![
            event_with_artifact_delta("f.txt", 1),
            event_with_artifact_delta("g.txt", 1),
        ];
        let session = session_with_events(vec![], events);
        let service = StubArtifactService::with_version("f.txt", 1, Value::String("x".to_string()));

        // "f.txt" is at version 1 both before and after the rewind
        // point (only one delta entry ever recorded for it) -> skipped.
        let delta = compute_artifact_delta_for_rewind(Some(&service), "app", &session, 1);
        assert!(!delta.contains_key("f.txt"));
    }
}
