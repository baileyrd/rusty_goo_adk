//! Capabilities C0268-C0274: `FileArtifactService`, ported from
//! `google.adk.artifacts.file_artifact_service`.
//!
//! **`ArtifactService` trait boundary, disclosed (predates this
//! batch)**: same narrowing already disclosed in
//! `in_memory_artifact_service.rs`'s module doc — `session_id` is a
//! required `&str` on every method, not the source's `Optional[str]`.
//! Concretely here: `_is_user_scoped`'s "no session in play" branch, and
//! `list_artifact_keys`'s session-absent (user-scoped-only) branch, have
//! no distinguishable path through this trait — `list_artifact_keys`
//! always walks both the session-scoped and user-scoped trees, matching
//! the source's own `session_id is not None` branch exactly (this port
//! just never has the other branch reachable).
//!
//! **C0270, `_umask_derived_file_mode`, not needed**: the source samples
//! the process umask once because `tempfile.mkstemp` hardcodes new files
//! to mode `0600`, mismatching a normal `open()`'s umask-derived mode —
//! and then fixes the metadata document's mode up to match. This port's
//! atomic-write helper creates its temp file via `std::fs::File::create_new`
//! instead of an `mkstemp` equivalent, which already gets the OS's normal
//! umask-derived permissions for a new file — the mismatch this function
//! exists to correct doesn't arise here, so nothing needs porting.
//!
//! **C0268-C0269, path safety by construction, not by canonicalize**:
//! the source rejects rooted/drive-qualified/parent-traversal filenames
//! by string inspection, then *additionally* resolves the joined path
//! (`Path.resolve(strict=False)`, following symlinks, without requiring
//! the target to exist) and re-checks it's still under the scope root —
//! defense in depth against a symlinked scope-root component. Rust's
//! `std::fs::canonicalize` requires the full path to already exist,
//! which the final artifact-version path usually doesn't at the point
//! this needs computing. This port instead performs only a *lexical*
//! join (splitting on `/`, dropping empty/`.` segments, never touching
//! the filesystem) of a filename already proven to contain no `..`
//! segment — which makes escape impossible by construction, not merely
//! detected after the fact. Disclosed divergence: this port's join
//! never follows a symlink partway through the scope root the way the
//! source's `resolve()` would, so it doesn't replicate the source's
//! specific symlink-canonicalization behavior — only the traversal-
//! prevention property, which this construction guarantees at least as
//! strongly.
//!
//! **C0273, `canonical_uri`, hand-rolled `file://` URI, lexical**: for
//! the same "target path doesn't exist yet" reason, this port can't call
//! `std::fs::canonicalize` to build the URI either. Since `self.root_dir`
//! is canonicalized once at construction and every artifact path is
//! joined onto it, the final payload path is already absolute by
//! construction — no filesystem resolution is needed, just a `file://`
//! URI rendering of that already-absolute path (byte-wise percent-
//! encoding of each path segment, RFC 3986-style).
//!
//! **Base64 duplication, disclosed**: `adk-tools::load_artifacts_tool`
//! already hand-rolled a base64 codec (no `base64` crate is a workspace
//! dependency — see that module's own doc for why), but `adk-tools`
//! depends on `adk-agents`, not the reverse, so it can't be reused here
//! without a crate-graph cycle. This module hand-rolls its own minimal
//! encode/decode pair instead — the same "duplicate locally to avoid a
//! cycle" pattern already used by `adk-examples`'s
//! `value_to_display_string` (duplicated from `adk-flows`).
//!
//! **`create_time`, faithfully replicated as always-"now"**: the
//! source's `ArtifactVersion.create_time` defaults to the current time
//! at construction; `_build_artifact_version` never passes a stored
//! value even though `_write_metadata` *does* persist one to
//! `metadata.json` (via `FileArtifactVersion`'s own inherited
//! `create_time` field) — so every read gets a fresh "now" timestamp,
//! not the artifact's actual creation time, and the persisted value on
//! disk is dead weight nothing ever reads back. This port replicates the
//! observable behavior (`create_time` is always "now" on read) without
//! writing the never-read field to disk in the first place — an
//! invisible simplification, not a behavior change.
//!
//! **`expanduser`, narrowed**: only a leading `~/` or a bare `~` is
//! expanded via `$HOME`, the same narrowing already used by
//! `adk-platform::telemetry_config` (C0942) — no `~user`-other-home
//! syntax, no Windows `%USERPROFILE%` fallback.
//!
//! **Constructor, adapted**: the source's `__init__` isn't fallible in a
//! `Result` sense (an `OSError` from `mkdir`/`resolve` propagates
//! uncaught). This port's constructor returns `std::io::Result<Self>`
//! instead — Rust's idiomatic way to surface a boundary failure,
//! preserving the same "a construction failure is the caller's problem"
//! intent without a panic baked into every possible caller.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use adk_errors::input_validation::InputValidationError;
use adk_genai::content::{MediaBlobStub, Part};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::artifact_util;
use crate::services::{ArtifactService, ArtifactVersion};

const USER_NAMESPACE_PREFIX: &str = "user:";
const METADATA_FILENAME: &str = "metadata.json";

fn expand_home(path: &Path) -> PathBuf {
    let Some(path_str) = path.to_str() else {
        return path.to_path_buf();
    };
    if let Some(rest) = path_str.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if path_str == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    path.to_path_buf()
}

fn file_has_user_namespace(filename: &str) -> bool {
    filename.starts_with(USER_NAMESPACE_PREFIX)
}

fn strip_user_namespace(filename: &str) -> &str {
    filename
        .strip_prefix(USER_NAMESPACE_PREFIX)
        .unwrap_or(filename)
}

fn is_reserved_artifact_name(name: &str) -> bool {
    name.eq_ignore_ascii_case(METADATA_FILENAME)
}

fn is_rooted_or_drive_qualified(value: &str) -> bool {
    if value.starts_with('/') || value.starts_with('\\') {
        return true;
    }
    let mut chars = value.chars();
    matches!((chars.next(), chars.next()), (Some(l), Some(':')) if l.is_ascii_alphabetic())
}

fn has_parent_reference(value: &str) -> bool {
    value.replace('\\', "/").split('/').any(|seg| seg == "..")
}

/// C0268/C0271: builds the artifact directory (absolute) and its path
/// relative to `scope_root` for `filename` — see the module doc for why
/// this is a lexical join rather than a filesystem-resolving one.
fn resolve_scoped_artifact_path(
    scope_root: &Path,
    filename: &str,
) -> Result<(PathBuf, PathBuf), InputValidationError> {
    let stripped = strip_user_namespace(filename).trim();

    if is_rooted_or_drive_qualified(stripped) {
        return Err(InputValidationError::new(format!(
            "Rooted or drive-qualified artifact filename {filename:?} is not permitted; \
             provide a path relative to the storage scope."
        )));
    }
    if has_parent_reference(stripped) {
        return Err(InputValidationError::new(format!(
            "Artifact filename {filename:?} must not contain parent traversal."
        )));
    }

    let mut relative = PathBuf::new();
    for segment in stripped.replace('\\', "/").split('/') {
        if !segment.is_empty() && segment != "." {
            relative.push(segment);
        }
    }
    if relative.as_os_str().is_empty() {
        relative = PathBuf::from("artifact");
    }

    let candidate = scope_root.join(&relative);
    Ok((candidate, relative))
}

fn versions_dir(artifact_dir: &Path) -> PathBuf {
    artifact_dir.join("versions")
}

fn metadata_path(artifact_dir: &Path, version: i64) -> PathBuf {
    versions_dir(artifact_dir)
        .join(version.to_string())
        .join(METADATA_FILENAME)
}

fn path_to_file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for component in path.components() {
        if let Component::Normal(segment) = component {
            uri.push('/');
            for byte in segment.to_string_lossy().bytes() {
                match byte {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        uri.push(byte as char);
                    }
                    _ => uri.push_str(&format!("%{byte:02X}")),
                }
            }
        }
    }
    uri
}

/// C0273: builds the canonical `file://` URI for an artifact payload —
/// always recomputed from the storage layout, never trusted from
/// on-disk metadata (hardening against a tampered `metadata.json`
/// redirecting reads).
fn canonical_uri(artifact_dir: &Path, version: i64) -> String {
    let stored_filename = artifact_dir.file_name().unwrap_or_default();
    let payload_path = versions_dir(artifact_dir)
        .join(version.to_string())
        .join(stored_filename);
    path_to_file_uri(&payload_path)
}

fn prune_empty_dirs(leaf: &Path, stop_at: &Path) {
    let mut current = leaf.to_path_buf();
    while current != stop_at && current.starts_with(stop_at) {
        if std::fs::remove_dir(&current).is_err() {
            return;
        }
        match current.parent() {
            Some(parent) => current = parent.to_path_buf(),
            None => return,
        }
    }
}

/// C0268: artifact directory paths beneath `root` — a directory "is" an
/// artifact directory iff it has a `versions` child; walking continues
/// into every other child (an artifact directory doubles as the parent
/// of anything nested under it, e.g. `"doc"` and `"doc/nested"`).
fn iter_artifact_dirs(root: &Path) -> Vec<PathBuf> {
    if !root.exists() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("versions").is_dir() {
            result.push(dir.clone());
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.file_name().is_some_and(|n| n != "versions") {
                stack.push(path);
            }
        }
    }
    result
}

fn list_versions_on_disk(artifact_dir: &Path) -> Vec<i64> {
    let Ok(entries) = std::fs::read_dir(versions_dir(artifact_dir)) else {
        return Vec::new();
    };
    let mut versions: Vec<i64> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i64>().ok()))
        .collect();
    versions.sort_unstable();
    versions
}

/// C0269: atomically reserves the next version number, returning its
/// staging and final directory paths. A staging directory abandoned by
/// a killed save keeps its version number reserved (published versions
/// aren't guaranteed contiguous).
fn reserve_version_dir(artifact_dir: &Path) -> std::io::Result<(i64, PathBuf, PathBuf)> {
    let versions = versions_dir(artifact_dir);
    std::fs::create_dir_all(&versions)?;
    let existing = list_versions_on_disk(artifact_dir);
    let mut version = existing.last().map(|v| v + 1).unwrap_or(0);

    loop {
        let staging_dir = versions.join(format!(".{version}.pending"));
        match std::fs::create_dir(&staging_dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                version += 1;
                continue;
            }
            Err(e) => return Err(e),
        }
        let version_dir = versions.join(version.to_string());
        if !version_dir.exists() {
            return Ok((version, staging_dir, version_dir));
        }
        std::fs::remove_dir(&staging_dir)?;
        version += 1;
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied();
        let b2 = chunk.get(2).copied();
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as usize] as char);
        out.push(match b1 {
            Some(b1) => ALPHABET[(((b1 & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as usize] as char,
            None => '=',
        });
        out.push(match b2 {
            Some(b2) => ALPHABET[(b2 & 0x3f) as usize] as char,
            None => '=',
        });
    }
    out
}

fn base64_decode_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Standard-alphabet base64 decode. See the module doc for why this
/// duplicates (rather than reuses) `adk-tools::load_artifacts_tool`'s
/// own hand-rolled decoder.
fn base64_decode(data: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &byte in data.as_bytes() {
        if byte == b'=' {
            break;
        }
        let value = base64_decode_value(byte)?;
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    Some(bytes)
}

/// C0273 (persisted half): on-disk metadata for one artifact version.
/// Matches `FileArtifactVersion`'s wire shape, minus `create_time` (see
/// the module doc for why it's never read back and so isn't persisted
/// here).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
struct FileArtifactMetadata {
    file_name: String,
    #[rusty_serde(default)]
    display_name: Option<String>,
    #[rusty_serde(default)]
    mime_type: Option<String>,
    canonical_uri: String,
    version: i64,
    #[rusty_serde(default)]
    custom_metadata: BTreeMap<String, Value>,
}

fn write_metadata(path: &Path, metadata: &FileArtifactMetadata) -> std::io::Result<()> {
    let serialized = rusty_serde::json::to_string(metadata)
        .map_err(|e| std::io::Error::other(format!("failed to serialize metadata: {e}")))?;
    let parent = path.parent().expect("metadata path always has a parent");
    let tmp_path = parent.join(format!(".{}.tmp", adk_platform::uuid::new_uuid()));
    let write_result = std::fs::write(&tmp_path, serialized);
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return write_result;
    }
    std::fs::rename(&tmp_path, path)
}

/// Loads a metadata payload from disk, degrading to `None` for anything
/// that isn't a readable, well-formed metadata document — matching the
/// source's own tolerant `_read_metadata` (a caller-derived path can be
/// made to name a directory, or contain corrupted JSON, and that must
/// read as "no metadata" rather than raise).
fn read_metadata(path: &Path) -> Option<FileArtifactMetadata> {
    let raw = std::fs::read_to_string(path).ok()?;
    match rusty_serde::json::from_str(&raw) {
        Ok(metadata) => Some(metadata),
        Err(e) => {
            eprintln!("Failed to parse metadata at {}: {e}", path.display());
            None
        }
    }
}

/// C0268-C0274: `FileArtifactService` — stores filesystem-backed
/// artifacts beneath a configurable root directory. Layout:
/// `root/apps/{app}/users/{user}/[sessions/{session}/]artifacts/{path}/versions/{version}/{filename}`
/// + a sibling `metadata.json`.
pub struct FileArtifactService {
    root_dir: PathBuf,
}

impl FileArtifactService {
    pub fn new(root_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let expanded = expand_home(root_dir.as_ref());
        std::fs::create_dir_all(&expanded)?;
        let root_dir = std::fs::canonicalize(&expanded)?;
        Ok(Self { root_dir })
    }

    fn base_root(&self, app_name: &str, user_id: &str) -> Result<PathBuf, InputValidationError> {
        artifact_util::validate_path_segment(app_name, "app_name")?;
        artifact_util::validate_path_segment(user_id, "user_id")?;
        Ok(self
            .root_dir
            .join("apps")
            .join(app_name)
            .join("users")
            .join(user_id))
    }

    fn scope_root(
        &self,
        base_root: &Path,
        session_id: &str,
        filename: &str,
    ) -> Result<PathBuf, InputValidationError> {
        if file_has_user_namespace(filename) {
            return Ok(base_root.join("artifacts"));
        }
        artifact_util::validate_path_segment(session_id, "session_id")?;
        Ok(base_root
            .join("sessions")
            .join(session_id)
            .join("artifacts"))
    }

    fn artifact_dir(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Result<PathBuf, InputValidationError> {
        let base_root = self.base_root(app_name, user_id)?;
        let scope_root = self.scope_root(&base_root, session_id, filename)?;
        Ok(resolve_scoped_artifact_path(&scope_root, filename)?.0)
    }

    fn build_artifact_version(
        artifact_dir: &Path,
        version: i64,
        metadata: Option<&FileArtifactMetadata>,
    ) -> ArtifactVersion {
        ArtifactVersion {
            version,
            canonical_uri: canonical_uri(artifact_dir, version),
            custom_metadata: metadata
                .map(|m| m.custom_metadata.clone())
                .unwrap_or_default(),
            create_time: adk_platform::time::get_time(),
            mime_type: metadata.and_then(|m| m.mime_type.clone()),
        }
    }

    fn latest_metadata(artifact_dir: &Path) -> Option<FileArtifactMetadata> {
        let versions = list_versions_on_disk(artifact_dir);
        let latest = *versions.last()?;
        read_metadata(&metadata_path(artifact_dir, latest))
    }
}

impl ArtifactService for FileArtifactService {
    fn load_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<Value> {
        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        if !artifact_dir.exists() {
            return None;
        }

        let versions = list_versions_on_disk(&artifact_dir);
        let version_to_load = match version {
            Some(v) if versions.contains(&v) => v,
            Some(_) => return None,
            None => *versions.last()?,
        };

        let version_dir = versions_dir(&artifact_dir).join(version_to_load.to_string());
        let metadata = read_metadata(&metadata_path(&artifact_dir, version_to_load));
        let stored_filename = artifact_dir.file_name().unwrap_or_default();
        let content_path = version_dir.join(stored_filename);

        let part = if let Some(mime_type) = metadata.as_ref().and_then(|m| m.mime_type.clone()) {
            let data = std::fs::read(&content_path).ok()?;
            let mut rest = vec![("data".to_string(), Value::String(base64_encode(&data)))];
            if let Some(display_name) = metadata.as_ref().and_then(|m| m.display_name.clone()) {
                rest.push(("displayName".to_string(), Value::String(display_name)));
            }
            Part {
                inline_data: Some(MediaBlobStub {
                    mime_type: Some(mime_type),
                    rest: Some(Value::Map(rest)),
                }),
                ..Default::default()
            }
        } else {
            let text = std::fs::read_to_string(&content_path).ok()?;
            Part::text(text)
        };

        rusty_serde::json::to_value(&part).ok()
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
        let part: Part = rusty_serde::json::from_value(artifact)
            .unwrap_or_else(|e| panic!("Invalid artifact: {e}"));

        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        let stored_name = artifact_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        // Enforced here, not in `artifact_dir`, which reads and deletes share:
        // an artifact stored under this name before the name was rejected
        // must stay readable and, above all, deletable.
        if is_reserved_artifact_name(stored_name) {
            panic!(
                "Artifact filename {filename:?} is reserved: an artifact may not be named \
                 {METADATA_FILENAME:?} (in any casing) because its payload is stored under the \
                 artifact's own name and would overwrite the metadata document."
            );
        }
        std::fs::create_dir_all(&artifact_dir).unwrap_or_else(|e| panic!("{e}"));

        let (next_version, staging_dir, version_dir) =
            reserve_version_dir(&artifact_dir).unwrap_or_else(|e| panic!("{e}"));
        let content_path = staging_dir.join(stored_name);

        let result: std::io::Result<()> = (|| {
            let (mime_type, display_name) = if let Some(inline_data) = &part.inline_data {
                let raw = inline_data
                    .rest
                    .as_ref()
                    .and_then(|rest| rest.get("data"))
                    .and_then(Value::as_str)
                    .and_then(base64_decode)
                    .ok_or_else(|| {
                        std::io::Error::other("Artifact inline_data must contain data.")
                    })?;
                std::fs::write(&content_path, &raw)?;
                let mime_type = inline_data
                    .mime_type
                    .clone()
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                let display_name = inline_data
                    .rest
                    .as_ref()
                    .and_then(|rest| rest.get("displayName"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                (Some(mime_type), display_name)
            } else if let Some(text) = &part.text {
                std::fs::write(&content_path, text)?;
                (None, None)
            } else {
                return Err(std::io::Error::other(
                    "Artifact must have either inline_data or text content.",
                ));
            };

            let metadata = FileArtifactMetadata {
                file_name: filename.to_string(),
                display_name,
                mime_type,
                canonical_uri: canonical_uri(&artifact_dir, next_version),
                version: next_version,
                custom_metadata: custom_metadata.unwrap_or_default(),
            };
            write_metadata(&staging_dir.join(METADATA_FILENAME), &metadata)?;
            std::fs::rename(&staging_dir, &version_dir)
        })();

        if let Err(e) = result {
            let _ = std::fs::remove_dir_all(&staging_dir);
            panic!("{e}");
        }
        next_version
    }

    fn get_artifact_version(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<ArtifactVersion> {
        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        let versions = list_versions_on_disk(&artifact_dir);
        let version_to_read = match version {
            Some(v) if versions.contains(&v) => v,
            Some(_) => return None,
            None => *versions.last()?,
        };
        let metadata = read_metadata(&metadata_path(&artifact_dir, version_to_read));
        Some(Self::build_artifact_version(
            &artifact_dir,
            version_to_read,
            metadata.as_ref(),
        ))
    }

    fn list_artifact_keys(&self, app_name: &str, user_id: &str, session_id: &str) -> Vec<String> {
        let mut filenames = std::collections::BTreeSet::new();
        let Ok(base_root) = self.base_root(app_name, user_id) else {
            return Vec::new();
        };

        let session_root = base_root
            .join("sessions")
            .join(session_id)
            .join("artifacts");
        for artifact_dir in iter_artifact_dirs(&session_root) {
            match Self::latest_metadata(&artifact_dir) {
                Some(metadata) if !metadata.file_name.is_empty() => {
                    filenames.insert(metadata.file_name);
                }
                _ => {
                    if let Ok(rel) = artifact_dir.strip_prefix(&session_root) {
                        filenames.insert(rel.to_string_lossy().replace('\\', "/"));
                    }
                }
            }
        }

        let user_root = base_root.join("artifacts");
        for artifact_dir in iter_artifact_dirs(&user_root) {
            match Self::latest_metadata(&artifact_dir) {
                Some(metadata) if !metadata.file_name.is_empty() => {
                    filenames.insert(metadata.file_name);
                }
                _ => {
                    if let Ok(rel) = artifact_dir.strip_prefix(&user_root) {
                        filenames
                            .insert(format!("user:{}", rel.to_string_lossy().replace('\\', "/")));
                    }
                }
            }
        }

        filenames.into_iter().collect()
    }

    fn delete_artifact(&self, app_name: &str, user_id: &str, session_id: &str, filename: &str) {
        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        let versions = versions_dir(&artifact_dir);
        if !versions.exists() {
            return;
        }
        std::fs::remove_dir_all(&versions).unwrap_or_else(|e| panic!("{e}"));
        if let Ok(base_root) = self.base_root(app_name, user_id) {
            if let Ok(scope_root) = self.scope_root(&base_root, session_id, filename) {
                prune_empty_dirs(&artifact_dir, &scope_root);
            }
        }
    }

    fn list_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<i64> {
        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        list_versions_on_disk(&artifact_dir)
    }

    fn list_artifact_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<ArtifactVersion> {
        let artifact_dir = self
            .artifact_dir(app_name, user_id, session_id, filename)
            .unwrap_or_else(|e| panic!("{e}"));
        list_versions_on_disk(&artifact_dir)
            .into_iter()
            .map(|version| {
                let metadata = read_metadata(&metadata_path(&artifact_dir, version));
                Self::build_artifact_version(&artifact_dir, version, metadata.as_ref())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "adk-file-artifact-service-test-{}",
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

    fn text_artifact(text: &str) -> Value {
        rusty_serde::json::to_value(&Part::text(text)).unwrap()
    }

    fn service() -> (TempDir, FileArtifactService) {
        let dir = TempDir::new();
        let service = FileArtifactService::new(&dir.path).unwrap();
        (dir, service)
    }

    #[test]
    fn save_and_load_a_session_scoped_artifact() {
        let (_dir, service) = service();
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
    fn save_increments_the_version_per_path_and_load_selects_by_version() {
        let (_dir, service) = service();
        assert_eq!(
            service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None),
            0
        );
        assert_eq!(
            service.save_artifact("app", "user1", "s1", "f", text_artifact("v1"), None),
            1
        );

        let v0 = service
            .load_artifact("app", "user1", "s1", "f", Some(0))
            .unwrap();
        let part: Part = rusty_serde::json::from_value(v0).unwrap();
        assert_eq!(part.text.as_deref(), Some("v0"));

        let latest = service
            .load_artifact("app", "user1", "s1", "f", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(latest).unwrap();
        assert_eq!(part.text.as_deref(), Some("v1"));
    }

    #[test]
    fn load_a_missing_artifact_returns_none() {
        let (_dir, service) = service();
        assert!(service
            .load_artifact("app", "user1", "s1", "missing", None)
            .is_none());
    }

    #[test]
    fn nested_filenames_create_a_nested_directory_layout() {
        let (dir, service) = service();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "images/photo.png",
            text_artifact("bytes"),
            None,
        );
        assert!(dir
            .path
            .join("apps/app/users/user1/sessions/s1/artifacts/images/photo.png/versions/0")
            .is_dir());
    }

    #[test]
    fn user_namespaced_artifacts_are_shared_across_sessions() {
        let (_dir, service) = service();
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
        let (_dir, service) = service();
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
    fn delete_artifact_removes_all_versions_and_prunes_empty_dirs() {
        let (dir, service) = service();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        service.delete_artifact("app", "user1", "s1", "f");
        assert!(service
            .load_artifact("app", "user1", "s1", "f", None)
            .is_none());
        // Pruned all the way back up to (but not including) the session's
        // artifacts scope root.
        assert!(!dir
            .path
            .join("apps/app/users/user1/sessions/s1/artifacts/f")
            .exists());
    }

    #[test]
    fn deleting_a_nested_artifact_does_not_remove_its_sibling() {
        let (_dir, service) = service();
        service.save_artifact("app", "user1", "s1", "doc", text_artifact("parent"), None);
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "doc/nested",
            text_artifact("child"),
            None,
        );
        service.delete_artifact("app", "user1", "s1", "doc");
        assert!(service
            .load_artifact("app", "user1", "s1", "doc", None)
            .is_none());
        let nested = service
            .load_artifact("app", "user1", "s1", "doc/nested", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(nested).unwrap();
        assert_eq!(part.text.as_deref(), Some("child"));
    }

    #[test]
    fn list_versions_returns_every_version_number() {
        let (_dir, service) = service();
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v0"), None);
        service.save_artifact("app", "user1", "s1", "f", text_artifact("v1"), None);
        assert_eq!(service.list_versions("app", "user1", "s1", "f"), vec![0, 1]);
    }

    #[test]
    fn get_artifact_version_reports_a_file_uri_and_mime_type() {
        let (_dir, service) = service();
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("image/png".to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(base64_encode(b"bytes")),
                )])),
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
        assert_eq!(version.version, 0);
        assert_eq!(version.mime_type.as_deref(), Some("image/png"));
        assert!(version.canonical_uri.starts_with("file:///"));
        assert!(version.canonical_uri.ends_with("img.png"));
    }

    #[test]
    fn inline_binary_data_round_trips_through_base64() {
        let (_dir, service) = service();
        let raw_bytes = b"\x00\x01\xffhello binary";
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("application/octet-stream".to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(base64_encode(raw_bytes)),
                )])),
            }),
            ..Default::default()
        };
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "blob.bin",
            rusty_serde::json::to_value(&part).unwrap(),
            None,
        );
        let loaded = service
            .load_artifact("app", "user1", "s1", "blob.bin", None)
            .unwrap();
        let part: Part = rusty_serde::json::from_value(loaded).unwrap();
        let data = part
            .inline_data
            .unwrap()
            .rest
            .unwrap()
            .get("data")
            .and_then(Value::as_str)
            .and_then(base64_decode)
            .unwrap();
        assert_eq!(data, raw_bytes);
    }

    #[test]
    #[should_panic(expected = "reserved")]
    fn saving_a_metadata_json_named_artifact_panics() {
        let (_dir, service) = service();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "metadata.json",
            text_artifact("x"),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "traversal")]
    fn saving_a_traversal_filename_panics() {
        let (_dir, service) = service();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "../escape.txt",
            text_artifact("x"),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Rooted or drive-qualified")]
    fn saving_a_rooted_filename_panics() {
        let (_dir, service) = service();
        service.save_artifact(
            "app",
            "user1",
            "s1",
            "/etc/passwd",
            text_artifact("x"),
            None,
        );
    }

    #[test]
    fn custom_metadata_is_stored_on_the_version() {
        let (_dir, service) = service();
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

    #[test]
    fn base64_round_trips() {
        for input in [
            b"".as_slice(),
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
        ] {
            let encoded = base64_encode(input);
            assert_eq!(base64_decode(&encoded).unwrap(), input);
        }
    }
}
