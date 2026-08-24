//! Capability C0367: `plugins.save_files_as_artifacts_plugin
//! .SaveFilesAsArtifactsPlugin`, ported from
//! `google.adk.plugins.save_files_as_artifacts_plugin`.
//!
//! Saves files embedded in a user message's `inline_data` parts as
//! artifacts, replacing each with a placeholder text part (and,
//! optionally, a `file_data` reference part when the saved artifact's
//! `canonical_uri` is model-accessible) so the model knows where to
//! find the file.
//!
//! **`Blob.display_name`/`.data`, adapted**: the source reads
//! `inline_data.display_name`/`.data` directly (typed `types.Blob`
//! fields). This port's `MediaBlobStub` only models `mime_type` as a
//! first-class field — everything else lives in the flattened `rest`
//! map (`adk-genai::content`'s own documented narrowing) — so this
//! reads `"displayName"`/`"data"` (the camelCase wire keys) out of
//! `rest` instead, the same pattern `file_artifact_service.rs` already
//! established for the same fields; `data` is additionally base64-
//! decoded via that module's `base64_decode` (promoted `pub(crate)`
//! for this second call site rather than hand-rolling a third copy —
//! see its own doc for why it isn't shared with `adk-tools`).
//!
//! **Per-part error handling, narrowed**: the source wraps each part's
//! save in `try/except Exception`, keeping the original part unchanged
//! on failure. This port's `ArtifactService::save_artifact` is
//! infallible (returns a bare `i64` version, not a `Result` — an
//! already-established trait shape predating this batch), so there is
//! no per-part failure path to catch; only the explicit file-size-limit
//! check has an observable "leave a placeholder instead" branch.
//!
//! **`on_user_message_callback` → `before_agent_callback` state
//! bridge**: the source stashes `pending_delta` directly into
//! `invocation_context.session.state` (a plain shared dict — any later
//! read, including from a different hook, sees the mutation
//! immediately) and reads it back in `before_agent_callback` to flush
//! into `callback_context.actions.artifact_delta`. This port's
//! `Context` has no such reference semantics (see
//! `adk-runners::runner::merge_context_state_into_session`'s own doc,
//! added alongside this plugin to close that visibility gap generally
//! for every run-level hook) — with that fix in place, the same
//! stash-then-flush pattern works here unchanged.
//!
//! **`DeprecationWarning`/`logging`, dropped**: no warnings/logging
//! framework is adopted in this crate (an already-established scope
//! cut elsewhere in this migration) — the "artifact service unset"
//! warning and the per-file "saved"/"failed" info/error logs aren't
//! reproduced; the *behavior* they describe (returning the message
//! unchanged, keeping the original part) is preserved.
//!
//! **`_is_model_accessible_uri`, hand-rolled**: scheme extraction via
//! plain string splitting on `"://"`, not a URL-parsing crate — the
//! same "no new dependency for a one-off scheme check" precedent
//! `oauth2_util.rs` already established for endpoint-host detection.

use std::collections::BTreeMap;

use rusty_serde::value::Value;

use crate::base_agent::BaseAgent;
use crate::context::Context;
use crate::file_artifact_service::base64_decode;
use crate::services::{BasePlugin, BoxFuture};
use adk_genai::content::{Content, MediaBlobStub, Part};

/// Schemes supported by our current LLM connectors — matches the
/// source's `_MODEL_ACCESSIBLE_URI_SCHEMES`.
const MODEL_ACCESSIBLE_URI_SCHEMES: &[&str] = &["gs", "https", "http"];

/// 20 MB, matching the source's `_MAX_INLINE_DATA_SIZE_BYTES` (the
/// Gemini API's documented `inline_data` limit).
const MAX_INLINE_DATA_SIZE_BYTES: usize = 20 * 1024 * 1024;

/// `plugins.save_files_as_artifacts_plugin.SaveFilesAsArtifactsPlugin`.
pub struct SaveFilesAsArtifactsPlugin {
    name: String,
    attach_file_reference: bool,
}

impl SaveFilesAsArtifactsPlugin {
    pub fn new() -> Self {
        Self {
            name: "save_files_as_artifacts_plugin".to_string(),
            attach_file_reference: true,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// If `false`, files are saved as artifacts but no `file_data`
    /// reference part is attached — the model can't directly access
    /// them (they're still loadable via a `load_artifacts`-style tool).
    pub fn with_attach_file_reference(mut self, attach_file_reference: bool) -> Self {
        self.attach_file_reference = attach_file_reference;
        self
    }

    fn pending_delta_key(&self) -> String {
        format!("{}:pending_delta", self.name)
    }
}

impl Default for SaveFilesAsArtifactsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

fn is_model_accessible_uri(uri: &str) -> bool {
    let Some((scheme, _)) = uri.split_once("://") else {
        return false;
    };
    if scheme.is_empty() {
        return false;
    }
    MODEL_ACCESSIBLE_URI_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str())
}

impl BasePlugin for SaveFilesAsArtifactsPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_user_message_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
        user_message: &'a Content,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            let Some(artifact_service) = invocation_context
                .invocation_context()
                .artifact_service
                .clone()
            else {
                // No logging framework adopted (see the module doc) —
                // the source's warning is dropped, the short-circuiting
                // unchanged-message return is preserved.
                return Some(user_message.clone());
            };

            if user_message.parts.is_empty() {
                return None;
            }

            let session = invocation_context.invocation_context().session.clone();
            let invocation_id = invocation_context
                .invocation_context()
                .invocation_id
                .clone();

            let mut new_parts = Vec::with_capacity(user_message.parts.len());
            let mut pending_delta: BTreeMap<String, i64> = BTreeMap::new();
            let mut modified = false;

            for (index, part) in user_message.parts.iter().enumerate() {
                let Some(inline_data) = &part.inline_data else {
                    new_parts.push(part.clone());
                    continue;
                };

                let raw_data = inline_data
                    .rest
                    .as_ref()
                    .and_then(|rest| rest.get("data"))
                    .and_then(Value::as_str)
                    .and_then(base64_decode);
                let file_size = raw_data.as_ref().map(Vec::len).unwrap_or(0);

                let display_name = inline_data
                    .rest
                    .as_ref()
                    .and_then(|rest| rest.get("displayName"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("artifact_{invocation_id}_{index}"));

                if file_size > MAX_INLINE_DATA_SIZE_BYTES {
                    let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
                    let limit_mb = MAX_INLINE_DATA_SIZE_BYTES / (1024 * 1024);
                    let error_message = format!(
                        "File {display_name} ({file_size_mb:.2} MB) exceeds the maximum \
                         supported size of {limit_mb}MB. Please upload a smaller file."
                    );
                    new_parts.push(Part {
                        text: Some(format!("[Upload Error: {error_message}]")),
                        ..Default::default()
                    });
                    modified = true;
                    continue;
                }

                let version = artifact_service.save_artifact(
                    &session.app_name,
                    &session.user_id,
                    &session.id,
                    &display_name,
                    rusty_serde::json::to_value(part).unwrap_or(Value::Null),
                    None,
                );

                new_parts.push(Part {
                    text: Some(format!("[Uploaded Artifact: \"{display_name}\"]")),
                    ..Default::default()
                });

                if self.attach_file_reference {
                    if let Some(file_part) = build_file_reference_part(
                        artifact_service.as_ref(),
                        &session,
                        &display_name,
                        version,
                        inline_data.mime_type.as_deref(),
                        &display_name,
                    ) {
                        new_parts.push(file_part);
                    }
                }

                pending_delta.insert(display_name, version);
                modified = true;
            }

            if !modified {
                return None;
            }

            let key = self.pending_delta_key();
            let mut delta_value = invocation_context
                .state()
                .get(&key)
                .cloned()
                .unwrap_or_else(|| Value::Map(Vec::new()));
            if !matches!(delta_value, Value::Map(_)) {
                delta_value = Value::Map(Vec::new());
            }
            for (file_name, version) in pending_delta {
                delta_value.insert(file_name, Value::from(version));
            }
            invocation_context.state_mut().set(key, delta_value);

            Some(Content {
                role: user_message.role.clone(),
                parts: new_parts,
            })
        })
    }

    fn before_agent_callback<'a>(
        &'a self,
        _agent: &'a BaseAgent,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            let key = self.pending_delta_key();
            let Some(Value::Map(entries)) = callback_context.state().get(&key).cloned() else {
                return None;
            };
            if entries.is_empty() {
                return None;
            }
            for (file_name, version) in &entries {
                if let Some(version) = version.as_i64() {
                    callback_context
                        .actions_mut()
                        .artifact_delta
                        .insert(file_name.clone(), version);
                }
            }
            callback_context
                .state_mut()
                .set(key, Value::Map(Vec::new()));
            None
        })
    }
}

/// C0367's `_build_file_reference_part`: constructs a `file_data`
/// reference part if the saved artifact's `canonical_uri` is
/// model-accessible.
fn build_file_reference_part(
    artifact_service: &(dyn crate::services::ArtifactService + Send + Sync),
    session: &crate::session::Session,
    filename: &str,
    version: i64,
    mime_type: Option<&str>,
    display_name: &str,
) -> Option<Part> {
    let artifact_version = artifact_service.get_artifact_version(
        &session.app_name,
        &session.user_id,
        &session.id,
        filename,
        Some(version),
    )?;
    if artifact_version.canonical_uri.is_empty()
        || !is_model_accessible_uri(&artifact_version.canonical_uri)
    {
        return None;
    }
    let mime_type = mime_type.map(str::to_string).or(artifact_version.mime_type);
    let rest = vec![
        (
            "fileUri".to_string(),
            Value::String(artifact_version.canonical_uri),
        ),
        (
            "displayName".to_string(),
            Value::String(display_name.to_string()),
        ),
    ];
    Some(Part {
        file_data: Some(MediaBlobStub {
            mime_type,
            rest: Some(Value::Map(rest)),
        }),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::in_memory_artifact_service::InMemoryArtifactService;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use std::sync::Arc;

    fn context_with_artifact_service() -> Context {
        let mut invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        invocation_context.artifact_service = Some(Arc::new(InMemoryArtifactService::new()));
        Context::new(invocation_context)
    }

    fn inline_data_part(display_name: &str, data: &str) -> Part {
        Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("text/plain".to_string()),
                rest: Some(Value::Map(vec![
                    (
                        "displayName".to_string(),
                        Value::String(display_name.to_string()),
                    ),
                    (
                        "data".to_string(),
                        Value::String(base64_encode_for_test(data)),
                    ),
                ])),
            }),
            ..Default::default()
        }
    }

    fn base64_encode_for_test(data: &str) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = data.as_bytes();
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0];
            let b1 = chunk.get(1).copied();
            let b2 = chunk.get(2).copied();
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as usize] as char);
            out.push(match b1 {
                Some(b1) => {
                    ALPHABET[(((b1 & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as usize] as char
                }
                None => '=',
            });
            out.push(match b2 {
                Some(b2) => ALPHABET[(b2 & 0x3f) as usize] as char,
                None => '=',
            });
        }
        out
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_returns_none_when_no_inline_data() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let message = Content::user_text("hi");
        assert_eq!(
            plugin.on_user_message_callback(&mut ctx, &message).await,
            None
        );
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_returns_unchanged_message_when_no_artifact_service() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let mut ctx = Context::new(invocation_context);
        let message = Content::new("user", vec![inline_data_part("f.txt", "hello")]);
        let result = plugin
            .on_user_message_callback(&mut ctx, &message)
            .await
            .unwrap();
        assert!(result.parts[0].inline_data.is_some());
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_attaches_no_file_reference_for_a_non_model_accessible_backend(
    ) {
        // `InMemoryArtifactService`'s `memory://` canonical URIs aren't in
        // `MODEL_ACCESSIBLE_URI_SCHEMES`, so no `file_data` part should be
        // appended even with `attach_file_reference` at its default `true`.
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let message = Content::new("user", vec![inline_data_part("f.txt", "hello")]);

        let result = plugin
            .on_user_message_callback(&mut ctx, &message)
            .await
            .unwrap();

        assert_eq!(result.parts.len(), 1);
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_saves_the_file_and_replaces_it_with_a_placeholder() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let message = Content::new("user", vec![inline_data_part("f.txt", "hello world")]);

        let result = plugin
            .on_user_message_callback(&mut ctx, &message)
            .await
            .unwrap();

        assert!(result.parts[0].inline_data.is_none());
        assert_eq!(
            result.parts[0].text.as_deref(),
            Some("[Uploaded Artifact: \"f.txt\"]")
        );
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_generates_a_filename_when_display_name_is_absent() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("text/plain".to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(base64_encode_for_test("hello")),
                )])),
            }),
            ..Default::default()
        };
        let message = Content::new("user", vec![part]);

        let result = plugin
            .on_user_message_callback(&mut ctx, &message)
            .await
            .unwrap();

        assert_eq!(
            result.parts[0].text.as_deref(),
            Some("[Uploaded Artifact: \"artifact_inv-1_0\"]")
        );
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_flags_a_file_over_the_size_limit() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let big = "a".repeat(MAX_INLINE_DATA_SIZE_BYTES + 1);
        let message = Content::new("user", vec![inline_data_part("big.bin", &big)]);

        let result = plugin
            .on_user_message_callback(&mut ctx, &message)
            .await
            .unwrap();

        assert!(result.parts[0]
            .text
            .as_deref()
            .unwrap()
            .starts_with("[Upload Error:"));
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_flushes_the_pending_delta_into_artifact_delta() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let message = Content::new("user", vec![inline_data_part("f.txt", "hello")]);
        plugin.on_user_message_callback(&mut ctx, &message).await;

        let agent = BaseAgent::new("agent", crate::base_agent::NoopBehavior).unwrap();
        let result = plugin.before_agent_callback(&agent, &mut ctx).await;

        assert_eq!(result, None);
        assert_eq!(ctx.actions().artifact_delta.get("f.txt"), Some(&0));
        // The pending-delta stash is cleared after flushing.
        assert_eq!(
            ctx.state().get(&plugin.pending_delta_key()),
            Some(&Value::Map(Vec::new()))
        );
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_is_a_no_op_with_no_pending_delta() {
        let plugin = SaveFilesAsArtifactsPlugin::new();
        let mut ctx = context_with_artifact_service();
        let agent = BaseAgent::new("agent", crate::base_agent::NoopBehavior).unwrap();
        let result = plugin.before_agent_callback(&agent, &mut ctx).await;
        assert_eq!(result, None);
        assert!(ctx.actions().artifact_delta.is_empty());
    }

    #[test]
    fn is_model_accessible_uri_accepts_gs_https_http_only() {
        assert!(is_model_accessible_uri("gs://bucket/file"));
        assert!(is_model_accessible_uri("https://example.com/file"));
        assert!(is_model_accessible_uri("http://example.com/file"));
        assert!(!is_model_accessible_uri("file:///tmp/f.txt"));
        assert!(!is_model_accessible_uri("not-a-uri"));
    }
}
