//! Capability C0278: a cross-backend `ArtifactService` parity test suite,
//! ported from `tests/unittests/artifacts/test_artifact_service.py`'s
//! parametrized (`ArtifactServiceType.IN_MEMORY`/`FILE`/`GCS`) test
//! functions.
//!
//! **Scoped to the two backends this port has**: `GcsArtifactService`
//! stays its own blocked row (the P5/P6 GCP-SDK pocket) — this suite
//! covers [`InMemoryArtifactService`] (C0265) and [`FileArtifactService`]
//! (C0268-C0274) only, disclosed as `Partial:` in `capability-manifest.md`.
//!
//! Each behavior is a shared `assert_*` function taking `&dyn
//! ArtifactService`, called once per backend by a pair of `#[test]`
//! wrappers (`in_memory_*`/`file_*`) — Rust has no built-in test
//! parametrization equivalent to `pytest.mark.parametrize`, so this is
//! the idiomatic substitute: one assertion body, proven identical across
//! both concrete implementations through the same trait object.

use adk_agents::file_artifact_service::FileArtifactService;
use adk_agents::in_memory_artifact_service::InMemoryArtifactService;
use adk_agents::services::ArtifactService;
use adk_genai::content::Part;
use rusty_serde::value::Value;

fn text_artifact(text: &str) -> Value {
    rusty_serde::json::to_value(&Part::text(text)).unwrap()
}

fn loaded_text(value: Value) -> Option<String> {
    let part: Part = rusty_serde::json::from_value(value).unwrap();
    part.text
}

struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "adk-artifact-service-cross-backend-test-{}",
            adk_platform::uuid::new_uuid()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn file_service() -> (TempDir, FileArtifactService) {
    let dir = TempDir::new();
    let service = FileArtifactService::new(&dir.path).unwrap();
    (dir, service)
}

// ------------------------------------------------------------------
// Shared behaviors
// ------------------------------------------------------------------

fn assert_load_empty(service: &dyn ArtifactService) {
    assert!(service
        .load_artifact("test_app", "test_user", "session_id", "filename", None)
        .is_none());
}

fn assert_save_load_delete(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id, filename) = ("app0", "user0", "123", "file456");

    let version = service.save_artifact(
        app_name,
        user_id,
        session_id,
        filename,
        text_artifact("test_data"),
        None,
    );
    assert_eq!(version, 0);
    assert_eq!(
        loaded_text(
            service
                .load_artifact(app_name, user_id, session_id, filename, None)
                .unwrap()
        ),
        Some("test_data".to_string())
    );

    // A version that doesn't exist.
    assert!(service
        .load_artifact(app_name, user_id, session_id, filename, Some(3))
        .is_none());

    service.delete_artifact(app_name, user_id, session_id, filename);
    assert!(service
        .load_artifact(app_name, user_id, session_id, filename, None)
        .is_none());
}

fn assert_list_keys(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");
    let filenames: Vec<String> = (0..5).map(|i| format!("filename{i}")).collect();

    for filename in &filenames {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            filename,
            text_artifact("test_data"),
            None,
        );
    }

    assert_eq!(
        service.list_artifact_keys(app_name, user_id, session_id),
        filenames
    );
}

fn assert_list_versions(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id, filename) = ("app0", "user0", "123", "with/slash/filename");

    for i in 0..4 {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            filename,
            text_artifact(&format!("v{i}")),
            None,
        );
    }

    assert_eq!(
        service.list_versions(app_name, user_id, session_id, filename),
        vec![0, 1, 2, 3]
    );
}

/// A nested artifact ("doc/nested") must not leak versions into its
/// parent ("doc") — filenames may contain "/", so the two are distinct
/// artifacts even though "doc/nested"'s records live under the prefix a
/// flat keyspace would scan to list "doc"'s versions.
fn assert_nested_artifact_does_not_leak_versions_into_parent(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");

    service.save_artifact(
        app_name,
        user_id,
        session_id,
        "doc",
        text_artifact("parent v0"),
        None,
    );
    // Give the nested artifact more versions than the parent has, so a
    // leak would push max(versions) past any version "doc" actually has.
    for i in 0..3 {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            "doc/nested",
            text_artifact(&format!("nested v{i}")),
            None,
        );
    }

    assert_eq!(
        service.list_versions(app_name, user_id, session_id, "doc"),
        vec![0]
    );
    assert_eq!(
        loaded_text(
            service
                .load_artifact(app_name, user_id, session_id, "doc", None)
                .unwrap()
        ),
        Some("parent v0".to_string())
    );
    // The next version of "doc" must be 1, not 3.
    assert_eq!(
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            "doc",
            text_artifact("parent v1"),
            None,
        ),
        1
    );
    assert_eq!(
        service.list_versions(app_name, user_id, session_id, "doc/nested"),
        vec![0, 1, 2]
    );
}

fn assert_list_artifact_versions_excludes_nested_artifact(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");

    for filename in ["doc", "doc/nested"] {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            filename,
            text_artifact(filename),
            None,
        );
    }

    let versions = service.list_artifact_versions(app_name, user_id, session_id, "doc");
    assert_eq!(
        versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        vec![0]
    );
}

fn assert_delete_artifact_keeps_nested_artifact(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");

    service.save_artifact(
        app_name,
        user_id,
        session_id,
        "doc",
        text_artifact("parent v0"),
        None,
    );
    service.save_artifact(
        app_name,
        user_id,
        session_id,
        "doc/nested",
        text_artifact("nested v0"),
        None,
    );

    service.delete_artifact(app_name, user_id, session_id, "doc");

    assert!(service
        .list_versions(app_name, user_id, session_id, "doc")
        .is_empty());
    assert_eq!(
        loaded_text(
            service
                .load_artifact(app_name, user_id, session_id, "doc/nested", None)
                .unwrap()
        ),
        Some("nested v0".to_string())
    );
}

fn assert_list_keys_includes_nested_artifact(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");

    for filename in ["doc", "doc/nested"] {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            filename,
            text_artifact(filename),
            None,
        );
    }

    assert_eq!(
        service.list_artifact_keys(app_name, user_id, session_id),
        vec!["doc".to_string(), "doc/nested".to_string()]
    );
}

fn assert_list_keys_preserves_user_prefix(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id) = ("app0", "user0", "123");

    for filename in ["user:document.pdf", "user:image.png", "session_file.txt"] {
        service.save_artifact(
            app_name,
            user_id,
            session_id,
            filename,
            text_artifact("test_data"),
            None,
        );
    }

    let mut keys = service.list_artifact_keys(app_name, user_id, session_id);
    keys.sort();
    let mut expected = vec![
        "user:document.pdf".to_string(),
        "user:image.png".to_string(),
        "session_file.txt".to_string(),
    ];
    expected.sort();
    assert_eq!(keys, expected);
}

/// A "namespaced" `user_id` (containing its own `/`) must still work —
/// exercised because [`FileArtifactService`] maps `user_id` onto a
/// filesystem path segment, where a `/` would otherwise nest directories
/// unexpectedly if handled naively.
fn assert_save_and_load_namespaced_user_id_succeeds(service: &dyn ArtifactService) {
    let (app_name, user_id, session_id, filename) =
        ("myapp", "group/user123", "sess123", "safe.txt");

    service.save_artifact(
        app_name,
        user_id,
        session_id,
        filename,
        text_artifact("data"),
        None,
    );
    let loaded = service
        .load_artifact(app_name, user_id, session_id, filename, None)
        .unwrap();
    assert_eq!(loaded_text(loaded), Some("data".to_string()));
}

// ------------------------------------------------------------------
// Per-backend wrappers
// ------------------------------------------------------------------

#[test]
fn in_memory_load_empty() {
    assert_load_empty(&InMemoryArtifactService::new());
}

#[test]
fn file_load_empty() {
    let (_dir, service) = file_service();
    assert_load_empty(&service);
}

#[test]
fn in_memory_save_load_delete() {
    assert_save_load_delete(&InMemoryArtifactService::new());
}

#[test]
fn file_save_load_delete() {
    let (_dir, service) = file_service();
    assert_save_load_delete(&service);
}

#[test]
fn in_memory_list_keys() {
    assert_list_keys(&InMemoryArtifactService::new());
}

#[test]
fn file_list_keys() {
    let (_dir, service) = file_service();
    assert_list_keys(&service);
}

#[test]
fn in_memory_list_versions() {
    assert_list_versions(&InMemoryArtifactService::new());
}

#[test]
fn file_list_versions() {
    let (_dir, service) = file_service();
    assert_list_versions(&service);
}

#[test]
fn in_memory_nested_artifact_does_not_leak_versions_into_parent() {
    assert_nested_artifact_does_not_leak_versions_into_parent(&InMemoryArtifactService::new());
}

#[test]
fn file_nested_artifact_does_not_leak_versions_into_parent() {
    let (_dir, service) = file_service();
    assert_nested_artifact_does_not_leak_versions_into_parent(&service);
}

#[test]
fn in_memory_list_artifact_versions_excludes_nested_artifact() {
    assert_list_artifact_versions_excludes_nested_artifact(&InMemoryArtifactService::new());
}

#[test]
fn file_list_artifact_versions_excludes_nested_artifact() {
    let (_dir, service) = file_service();
    assert_list_artifact_versions_excludes_nested_artifact(&service);
}

#[test]
fn in_memory_delete_artifact_keeps_nested_artifact() {
    assert_delete_artifact_keeps_nested_artifact(&InMemoryArtifactService::new());
}

#[test]
fn file_delete_artifact_keeps_nested_artifact() {
    let (_dir, service) = file_service();
    assert_delete_artifact_keeps_nested_artifact(&service);
}

#[test]
fn in_memory_list_keys_includes_nested_artifact() {
    assert_list_keys_includes_nested_artifact(&InMemoryArtifactService::new());
}

#[test]
fn file_list_keys_includes_nested_artifact() {
    let (_dir, service) = file_service();
    assert_list_keys_includes_nested_artifact(&service);
}

#[test]
fn in_memory_list_keys_preserves_user_prefix() {
    assert_list_keys_preserves_user_prefix(&InMemoryArtifactService::new());
}

#[test]
fn file_list_keys_preserves_user_prefix() {
    let (_dir, service) = file_service();
    assert_list_keys_preserves_user_prefix(&service);
}

#[test]
fn in_memory_save_and_load_namespaced_user_id_succeeds() {
    assert_save_and_load_namespaced_user_id_succeeds(&InMemoryArtifactService::new());
}

#[test]
fn file_save_and_load_namespaced_user_id_succeeds() {
    let (_dir, service) = file_service();
    assert_save_and_load_namespaced_user_id_succeeds(&service);
}
