//! Capability C0265: `InMemoryArtifactService`, ported from
//! `google.adk.artifacts.in_memory_artifact_service`. Uses
//! `crate::artifact_util` (C0262-C0264) for its URI scheme,
//! cross-tenant-scope check, and path-segment validation — the first
//! real caller of that batch's utilities.
//!
//! **`ArtifactService` trait boundary, disclosed (predates this
//! batch)**: `session_id` is a required `&str` on every method, not
//! the source's `Optional[str]` — so the source's "no session in
//! play" branches (`_artifact_path`'s `InputValidationError` when a
//! session-scoped save is attempted with `session_id=None`;
//! `list_artifact_keys`'s "session_id is `None`" branch, which lists
//! *only* user-scoped filenames instead of the combined listing) have
//! no distinguishable path through this trait signature. This port's
//! `_artifact_path` still checks the `"user:"` filename-namespace
//! prefix first (matching the source exactly for that case) and
//! otherwise always treats `session_id` as present, validating it's
//! non-empty via `artifact_util::validate_path_segment` — a
//! structurally-guaranteed-non-`None` substitute for the source's
//! runtime `None`-check, not a behavior gap for any input this port's
//! trait signature can actually carry. `list_artifact_keys` always
//! returns the combined (session-scoped ∪ user-scoped) listing, since
//! "session absent" isn't representable.
//!
//! **`artifact`/return values, disclosed (predates this batch)**: the
//! trait's `artifact`/`load_artifact`'s return type are opaque
//! `Value`, not a typed `types.Part` — this service deserializes into
//! `adk_genai::content::Part` internally (via its own `Deserialize`
//! impl) to run the source's MIME-type-detection/artifact-reference
//! logic, then serializes back to `Value` at the boundary, the same
//! "parse the opaque `Value` via its own `Deserialize` impl" pattern
//! `ExampleTool`/`PreloadMemoryTool`/`LoadArtifactsTool` already use.
//!
//! **Version-lookup narrowing, disclosed**: the source resolves an
//! explicit negative `version` via Python list indexing (`versions[-2]`
//! means second-to-last). This port only supports `None` (→ latest)
//! or a non-negative index — `ArtifactService::save_artifact` always
//! returns non-negative version numbers, so no caller in this port
//! can construct a meaningful negative version to look up; the
//! narrowing has no reachable behavior difference for how this port's
//! own callers actually use `version`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use adk_genai::content::Part;
use rusty_serde::value::Value;

use crate::artifact_util;
use crate::services::{ArtifactService, ArtifactVersion};

#[derive(Clone)]
struct ArtifactEntry {
    data: Part,
    artifact_version: ArtifactVersion,
}

/// C0265: an in-memory implementation of the artifact service. Not
/// suitable for multi-threaded production environments — for testing
/// and development only, matching the source's own docstring.
pub struct InMemoryArtifactService {
    artifacts: Mutex<HashMap<String, Vec<ArtifactEntry>>>,
}

impl InMemoryArtifactService {
    pub fn new() -> Self {
        Self {
            artifacts: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryArtifactService {
    fn default() -> Self {
        Self::new()
    }
}

fn file_has_user_namespace(filename: &str) -> bool {
    filename.starts_with("user:")
}

fn artifact_path(app_name: &str, user_id: &str, filename: &str, session_id: &str) -> String {
    artifact_util::validate_path_segment(app_name, "app_name").unwrap_or_else(|e| panic!("{e}"));
    artifact_util::validate_path_segment(user_id, "user_id").unwrap_or_else(|e| panic!("{e}"));
    if file_has_user_namespace(filename) {
        return format!("{app_name}/{user_id}/user/{filename}");
    }
    artifact_util::validate_path_segment(session_id, "session_id")
        .unwrap_or_else(|e| panic!("{e}"));
    format!("{app_name}/{user_id}/{session_id}/{filename}")
}

/// The sentinel "reset"/empty-artifact shapes the source treats as
/// "no artifact" on load: a bare default `Part`, an empty-text `Part`,
/// or inline data with no bytes.
fn is_empty_artifact(part: &Part) -> bool {
    if *part == Part::default() || *part == Part::text("") {
        return true;
    }
    if let Some(inline_data) = &part.inline_data {
        let has_bytes = inline_data
            .rest
            .as_ref()
            .and_then(|rest| rest.get("data"))
            .and_then(|data| data.as_str())
            .is_some_and(|data| !data.is_empty());
        if !has_bytes {
            return true;
        }
    }
    false
}

impl ArtifactService for InMemoryArtifactService {
    fn load_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<Value> {
        let path = artifact_path(app_name, user_id, filename, session_id);
        let entry = {
            let artifacts = self.artifacts.lock().unwrap();
            let versions = artifacts.get(&path)?;
            if versions.is_empty() {
                return None;
            }
            let index = match version {
                Some(v) if v >= 0 && (v as usize) < versions.len() => v as usize,
                Some(_) => return None,
                None => versions.len() - 1,
            };
            versions[index].clone()
        };

        if artifact_util::is_artifact_ref(&entry.data) {
            let file_uri = entry
                .data
                .file_data
                .as_ref()
                .and_then(|file_data| file_data.rest.as_ref())
                .and_then(|rest| rest.get("fileUri"))
                .and_then(|value| value.as_str())
                .expect("is_artifact_ref confirmed file_data.fileUri is present");
            let parsed_uri = artifact_util::parse_artifact_uri(file_uri)
                .unwrap_or_else(|| panic!("Invalid artifact reference URI: {file_uri}"));
            artifact_util::validate_artifact_reference_scope(
                app_name,
                user_id,
                Some(session_id),
                &parsed_uri,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            return self.load_artifact(
                &parsed_uri.app_name,
                &parsed_uri.user_id,
                parsed_uri.session_id.as_deref().unwrap_or(session_id),
                &parsed_uri.filename,
                Some(parsed_uri.version as i64),
            );
        }

        if is_empty_artifact(&entry.data) {
            return None;
        }
        rusty_serde::json::to_value(&entry.data).ok()
    }

    fn save_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<BTreeMap<String, Value>>,
    ) -> i64 {
        let path = artifact_path(app_name, user_id, filename, session_id);
        let part: Part =
            rusty_serde::json::from_value(artifact).unwrap_or_else(|_| Part::default());

        let mut artifacts = self.artifacts.lock().unwrap();
        let versions = artifacts.entry(path).or_default();
        let version = versions.len() as i64;

        let canonical_uri = if file_has_user_namespace(filename) {
            format!(
                "memory://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"
            )
        } else {
            format!(
                "memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
            )
        };

        let mut mime_type = None;
        if let Some(inline_data) = &part.inline_data {
            mime_type = inline_data.mime_type.clone();
        } else if part.text.is_some() {
            mime_type = Some("text/plain".to_string());
        } else if let Some(file_data) = &part.file_data {
            if artifact_util::is_artifact_ref(&part) {
                if let Some(file_uri) = file_data
                    .rest
                    .as_ref()
                    .and_then(|rest| rest.get("fileUri"))
                    .and_then(|value| value.as_str())
                {
                    if let Some(parsed_uri) = artifact_util::parse_artifact_uri(file_uri) {
                        artifact_util::validate_artifact_reference_scope(
                            app_name,
                            user_id,
                            Some(session_id),
                            &parsed_uri,
                        )
                        .unwrap_or_else(|e| panic!("{e}"));
                    }
                }
                // If it's a valid artifact reference, the mime type is
                // unknown until it's loaded -- matching the source, which
                // leaves `mime_type` unset here too.
            } else {
                mime_type = file_data.mime_type.clone();
            }
        } else {
            panic!("Not supported artifact type.");
        }

        let artifact_version = ArtifactVersion {
            version,
            canonical_uri,
            custom_metadata: custom_metadata.unwrap_or_default(),
            create_time: adk_platform::time::get_time(),
            mime_type,
        };

        versions.push(ArtifactEntry {
            data: part,
            artifact_version,
        });
        version
    }

    fn get_artifact_version(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<ArtifactVersion> {
        let path = artifact_path(app_name, user_id, filename, session_id);
        let artifacts = self.artifacts.lock().unwrap();
        let versions = artifacts.get(&path)?;
        if versions.is_empty() {
            return None;
        }
        let index = match version {
            Some(v) if v >= 0 && (v as usize) < versions.len() => v as usize,
            Some(_) => return None,
            None => versions.len() - 1,
        };
        Some(versions[index].artifact_version.clone())
    }

    fn list_artifact_keys(&self, app_name: &str, user_id: &str, session_id: &str) -> Vec<String> {
        let user_prefix = format!("{app_name}/{user_id}/user/");
        let session_prefix = format!("{app_name}/{user_id}/{session_id}/");
        let artifacts = self.artifacts.lock().unwrap();
        let mut filenames: Vec<String> = artifacts
            .keys()
            .filter_map(|path| {
                path.strip_prefix(&session_prefix)
                    .or_else(|| path.strip_prefix(&user_prefix))
                    .map(str::to_string)
            })
            .collect();
        filenames.sort();
        filenames
    }

    fn delete_artifact(&self, app_name: &str, user_id: &str, session_id: &str, filename: &str) {
        let path = artifact_path(app_name, user_id, filename, session_id);
        self.artifacts.lock().unwrap().remove(&path);
    }

    fn list_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<i64> {
        let path = artifact_path(app_name, user_id, filename, session_id);
        let artifacts = self.artifacts.lock().unwrap();
        match artifacts.get(&path) {
            Some(versions) => (0..versions.len() as i64).collect(),
            None => Vec::new(),
        }
    }

    fn list_artifact_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<ArtifactVersion> {
        let path = artifact_path(app_name, user_id, filename, session_id);
        let artifacts = self.artifacts.lock().unwrap();
        match artifacts.get(&path) {
            Some(versions) => versions
                .iter()
                .map(|e| e.artifact_version.clone())
                .collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::MediaBlobStub;

    fn text_artifact(text: &str) -> Value {
        rusty_serde::json::to_value(&Part::text(text)).unwrap()
    }

    #[test]
    fn save_and_load_a_session_scoped_artifact() {
        let service = InMemoryArtifactService::new();
        let version = service.save_artifact(
            "app",
            "user1",
            "s1",
            "notes.txt",
            text_artifact("hello"),
            None,
        );
        assert_eq!(version, 0);

        let loaded = service
            .load_artifact("app", "user1", "s1", "notes.txt", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(loaded).unwrap();
        assert_eq!(part.text.as_deref(), Some("hello"));
    }

    #[test]
    fn save_increments_the_version_per_path() {
        let service = InMemoryArtifactService::new();
        assert_eq!(
            service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None),
            0
        );
        assert_eq!(
            service.save_artifact("app", "user1", "s1", "f", text_artifact("v1"), None),
            1
        );
        let loaded = service
            .load_artifact("app", "user1", "s1", "f", Some(0))
            .unwrap();
        let part: Part = rusty_serde::json::from_value(loaded).unwrap();
        assert_eq!(part.text.as_deref(), Some("v0"));
    }

    #[test]
    fn load_without_a_version_returns_the_latest() {
        let service = InMemoryArtifactService::new();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v1"), None);
        let loaded = service
            .load_artifact("app", "user1", "s1", "f", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(loaded).unwrap();
        assert_eq!(part.text.as_deref(), Some("v1"));
    }

    #[test]
    fn load_a_missing_artifact_returns_none() {
        let service = InMemoryArtifactService::new();
        assert!(service
            .load_artifact("app", "user1", "s1", "missing", None)
            .is_none());
    }

    #[test]
    fn user_namespaced_artifacts_are_shared_across_sessions() {
        let service = InMemoryArtifactService::new();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "user:profile.json",
            text_artifact("data"),
            None,
        );
        let loaded = service
            .load_artifact("app", "user1", "s2", "user:profile.json", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(loaded).unwrap();
        assert_eq!(part.text.as_deref(), Some("data"));
    }

    #[test]
    fn list_artifact_keys_combines_session_and_user_scoped_filenames() {
        let service = InMemoryArtifactService::new();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "session.txt",
            text_artifact("a"),
            None,
        );
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "user:shared.txt",
            text_artifact("b"),
            None,
        );
        let keys = service.list_artifact_keys("app", "user1", "s1");
        assert_eq!(
            keys,
            vec!["session.txt".to_string(), "user:shared.txt".to_string()]
        );
    }

    #[test]
    fn delete_artifact_removes_all_versions() {
        let service = InMemoryArtifactService::new();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        service.delete_artifact("app", "user1", "s1", "f");
        assert!(service
            .load_artifact("app", "user1", "s1", "f", None)
            .is_none());
    }

    #[test]
    fn list_versions_returns_every_version_number() {
        let service = InMemoryArtifactService::new();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v1"), None);
        assert_eq!(service.list_versions("app", "user1", "s1", "f"), vec![0, 1]);
    }

    #[test]
    fn list_artifact_versions_returns_metadata_for_every_version() {
        let service = InMemoryArtifactService::new();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        let versions = service.list_artifact_versions("app", "user1", "s1", "f");
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].version, 0);
        assert_eq!(versions[0].mime_type.as_deref(), Some("text/plain"));
    }

    #[test]
    fn get_artifact_version_returns_the_stored_metadata() {
        let service = InMemoryArtifactService::new();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        let version = service
            .get_artifact_version("app", "user1", "s1", "f", None)
            .unwrap();
        assert_eq!(version.version, 0);
        assert!(version
            .canonical_uri
            .starts_with("memory://apps/app/users/user1/sessions/s1"));
    }

    #[test]
    fn inline_data_mime_type_is_recorded() {
        let service = InMemoryArtifactService::new();
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("image/png".to_string()),
                rest: None,
            }),
            ..Default::default()
        };
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "img.png",
            rusty_serde::json::to_value(&part).unwrap(),
            None,
        );
        let version = service
            .get_artifact_version("app", "user1", "s1", "img.png", None)
            .unwrap();
        assert_eq!(version.mime_type.as_deref(), Some("image/png"));
    }

    #[test]
    fn custom_metadata_is_stored_on_the_version() {
        let service = InMemoryArtifactService::new();
        let metadata =
            BTreeMap::from([("source".to_string(), Value::String("upload".to_string()))]);
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "f",
            text_artifact("v0"),
            Some(metadata.clone()),
        );
        let version = service
            .get_artifact_version("app", "user1", "s1", "f", None)
            .unwrap();
        assert_eq!(version.custom_metadata, metadata);
    }
}
