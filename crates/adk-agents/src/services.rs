//! Placeholder service traits for capabilities C0061-C0064, forward-referencing
//! phases not yet built: `BaseArtifactService`/`ArtifactVersion` (Phase 6),
//! `BaseSessionService` (Phase 5), `BaseMemoryService` (Phase 6).
//!
//! **`SearchMemoryResponse`/`MemoryEntry`/`AuthCredential`/`AuthConfig`/
//! `CredentialService` are the exceptions**, pulled forward the same way
//! `SessionService` is (see below): they're real structs/a real trait,
//! not opaque placeholders — the first two because `adk-tools`'s
//! `LoadMemoryTool`/`PreloadMemoryTool` (C0423/C0424) need their actual
//! field shape, `AuthCredential`/`AuthConfig` because `crate::auth_credential`/
//! `crate::auth_tool` (C0494-C0497/C0499, C0504) port the real types
//! directly, and `CredentialService` itself (C0527) because
//! `InMemoryCredentialService`/`SessionStateCredentialService` (C0528/
//! C0529) need a real, working interface to implement — a synchronous
//! marker trait can't serve that, the same reasoning `SessionService`'s
//! own promotion already established. `BaseMemoryService` is still an
//! unimplemented placeholder trait below; only its return-type shapes
//! are real.
//!
//! **Disclosed adaptation**: the source's service methods are `async def`.
//! Since no concrete backend exists yet for the remaining placeholders
//! (nothing here performs real I/O for them), those placeholder traits
//! stay synchronous; `Context`'s own methods stay
//! `async fn` (preserving the `.await`-able call shape callers already use)
//! and simply call through. Revisit — trait methods become `async fn` too —
//! once a real backend (network/disk I/O) lands in its own phase.
//!
//! **`SessionService` is the one exception, pulled forward**: `Runner`
//! (`runners.py`, C0833-C0926) needs a real, working session backend to
//! fetch/create/persist sessions across a turn — a marker trait can't
//! serve that. So `SessionService` here is a genuine (if narrowed) port of
//! `sessions.base_session_service.BaseSessionService`, plus
//! [`InMemorySessionService`]. Both stay in this module, next to the
//! trait, for the same reason `Session`/`State` are pulled forward into
//! `adk-agents` rather than stubbed (see their own module docs): Phase 5's
//! real `adk-sessions` crate will *replace* this wholesale, not extend it.
//! Since `Arc<dyn SessionService>` is stored as a trait object
//! (`InvocationContext`), its methods return a boxed [`BoxFuture`] rather
//! than using native `async fn` (not object-safe) — the same pattern
//! `adk_tools::base_tool::BaseTool` already uses for the same reason.
//!
//! **Narrowed, disclosed**: no app:/user: state-prefix scoping
//! (`State.APP_PREFIX`/`USER_PREFIX`, `_session_util.extract_state_delta`,
//! the source's cross-session shared `app_state`/`user_state` maps) —
//! `Session`/`State` here have no prefix-scoping concept yet (a Phase 5
//! concern of their own), so `create_session`'s `state` is stored as flat,
//! session-scoped state only. No `get_user_state` (depends on the same
//! app:/user: architecture). No `last_update_time` field on the
//! placeholder `Session`, so `list_sessions` returns insertion order
//! (grouped by user id, then session id) rather than the source's
//! last-update-time sort. `append_event`'s `StaleSessionError` path is
//! N/A for this in-memory backend — the source's own `InMemorySessionService`
//! never raises it either (the docstring's warning is for a *persistent*
//! backend detecting a concurrent write, which a simple in-memory map
//! can't contend on). `GetSessionConfig` (`num_recent_events`/
//! `after_timestamp` event trimming, C0207) is now real — see
//! [`SessionService::get_session_with_config`] — but `RunConfig.get_session_config`
//! is still an opaque `Value` placeholder (its own disclosed scope cut,
//! C0875), so no call site in this crate or `adk-runners` threads a real
//! `GetSessionConfig` through yet; the trait method exists and is tested
//! ahead of its own caller, the same "widen once a real consumer needs
//! the structure" pattern this port applies elsewhere.

use adk_errors::already_exists::AlreadyExistsError;
use adk_events::ui_widget::UiWidget;
use adk_events::Event;
use adk_genai::content::Content;
use adk_platform::uuid::new_uuid;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::base_agent::{AgentRunError, BaseAgent};
use crate::context::Context;
use crate::session::Session;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `auth.auth_credential.AuthCredential` (C0493-C0497/C0499, real as of
/// this batch — see `auth_credential.rs`'s module doc for the "widen a
/// placeholder to a real type" precedent this follows).
pub use crate::auth_credential::AuthCredential;
/// `auth.auth_tool.AuthConfig` — widened from a bare `Value` placeholder
/// to the real type once `crate::auth_tool` (C0504) shipped it, the same
/// "widen a placeholder once a real consumer needs the structure"
/// precedent as `AuthCredential` above. Real consumer: this module's own
/// [`CredentialService`] trait (C0527), which needs `credential_key`/
/// `exchanged_auth_credential` to be actual fields, not opaque JSON.
pub use crate::auth_tool::AuthConfig;
/// `artifacts.base_artifact_service.ArtifactVersion` — metadata
/// describing a specific version of an artifact. Promoted from an
/// opaque `Value` placeholder to a real struct (Phase 6,
/// `InMemoryArtifactService`, C0265) — the same "widen a placeholder
/// to a real type once a real consumer needs its structure" precedent
/// already used for `MemoryEntry`/`SearchMemoryResponse` (C0423/C0424)
/// and `AuthCredential` (C0494-C0499). `alias_generator=to_camel` is
/// ported via `rename_all = "camelCase"`; `populate_by_name=True`'s
/// dual-name accept has no port, the same disclosed gap as
/// `auth_credential.rs`'s own camelCase fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ArtifactVersion {
    /// Monotonically increasing identifier for the artifact version.
    pub version: i64,
    /// Canonical URI referencing the persisted artifact payload.
    pub canonical_uri: String,
    /// Optional user-supplied metadata stored with the artifact.
    #[rusty_serde(default)]
    pub custom_metadata: BTreeMap<String, Value>,
    /// Unix timestamp (seconds) when the version record was created.
    pub create_time: f64,
    /// MIME type when the artifact payload is stored as binary data.
    #[rusty_serde(default)]
    pub mime_type: Option<String>,
}
/// `memory.memory_entry.MemoryEntry` — one memory entry. Promoted from an
/// opaque `Value` placeholder to a real struct (Phase 8,
/// `LoadMemoryTool`/`PreloadMemoryTool`, C0423/C0424) now that a concrete
/// consumer needs its fields — the same "widen a placeholder to a real
/// type once a real consumer needs its structure" precedent already used
/// for `EventCompaction.compacted_content` (Phase 4, C0185).
/// `InMemoryMemoryService` (`in_memory_memory_service.rs`, C0243/C0244/
/// C0247/C0248/C0249) already constructs real `MemoryEntry` values — the
/// remaining unbuilt backing implementation is `VertexAiMemoryBankService`
/// (GCP-backed, genuinely out of scope, SDK-blocked).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// The main content of the memory.
    pub content: Content,
    #[rusty_serde(default)]
    pub custom_metadata: BTreeMap<String, Value>,
    #[rusty_serde(default)]
    pub id: Option<String>,
    #[rusty_serde(default)]
    pub author: Option<String>,
    /// ISO 8601-preferred timestamp string, forwarded to the LLM as-is.
    #[rusty_serde(default)]
    pub timestamp: Option<String>,
}

/// `memory.base_memory_service.SearchMemoryResponse`. See [`MemoryEntry`]
/// for the same promoted-from-placeholder disclosure.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SearchMemoryResponse {
    #[rusty_serde(default)]
    pub memories: Vec<MemoryEntry>,
}

/// `sessions.base_session_service.GetSessionConfig` (C0207) — bounds how
/// much session history [`SessionService::get_session_with_config`]
/// returns, without truncating what's actually persisted. A shared
/// conformance contract every backend must honor identically — this port
/// applies it generically in that method's own default body rather than
/// per-backend, so every [`SessionService`] implementer gets it for free.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct GetSessionConfig {
    /// `None`: no limit. `Some(0)` (or negative): no events returned.
    /// `Some(n) if n > 0`: at most the `n` most recent events.
    #[rusty_serde(default)]
    pub num_recent_events: Option<i64>,
    /// `None`, or `Some(0.0)` (falsy in the source's own Python truthiness
    /// check — `if config.after_timestamp:` — replicated here rather than
    /// treated as "any `Some` value"): no limit. Otherwise, only events
    /// with `timestamp >= after_timestamp` are kept.
    #[rusty_serde(default)]
    pub after_timestamp: Option<f64>,
}

/// Applies [`GetSessionConfig`]'s trimming to `session.events` in place —
/// `num_recent_events` first (tail-slice), then `after_timestamp`
/// (drop everything older), matching the source's own application order.
/// Both filters compose: a session can be trimmed to its most recent `N`
/// events and then further cut to only those after a given time.
fn apply_get_session_config(session: &mut Session, config: &GetSessionConfig) {
    if let Some(num_recent_events) = config.num_recent_events {
        if num_recent_events <= 0 {
            session.events.clear();
        } else {
            let keep_from = session
                .events
                .len()
                .saturating_sub(num_recent_events as usize);
            session.events.drain(..keep_from);
        }
    }
    if let Some(after_timestamp) = config.after_timestamp {
        if after_timestamp != 0.0 {
            session
                .events
                .retain(|event| event.timestamp >= after_timestamp);
        }
    }
}

/// Real (narrowed) port of `sessions.base_session_service.BaseSessionService`
/// — see the module doc for why this one trait isn't a placeholder marker
/// like its siblings, and what's deliberately cut.
pub trait SessionService: Send + Sync {
    /// Creates a new session, storing `state` (flat, session-scoped — see
    /// the module doc) as its initial state. Errors if `session_id` is
    /// given and already in use; generates a fresh id otherwise (mirroring
    /// the source's `strip()`-then-fall-back-to-`new_uuid()` rule for a
    /// blank/absent id).
    fn create_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        state: Option<BTreeMap<String, Value>>,
        session_id: Option<String>,
    ) -> BoxFuture<'a, Result<Session, AlreadyExistsError>>;

    /// Gets a session by id, or `None` if it doesn't exist.
    fn get_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        session_id: &'a str,
    ) -> BoxFuture<'a, Option<Session>>;

    /// C0207: [`Self::get_session`], but additionally bounding the
    /// returned session's `events` per `config` (see [`GetSessionConfig`]
    /// and [`apply_get_session_config`]) — `None` behaves exactly like
    /// `get_session`. The default implementation defers to
    /// `get_session` and applies the trimming generically, since it's a
    /// conformance contract every backend must honor identically; a
    /// backend only needs to override this if it can apply the filter
    /// more efficiently at its own storage layer.
    fn get_session_with_config<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        session_id: &'a str,
        config: Option<&'a GetSessionConfig>,
    ) -> BoxFuture<'a, Option<Session>> {
        Box::pin(async move {
            let mut session = self.get_session(app_name, user_id, session_id).await?;
            if let Some(config) = config {
                apply_get_session_config(&mut session, config);
            }
            Some(session)
        })
    }

    /// Lists all sessions for a user (or, if `user_id` is `None`, every
    /// user under `app_name`) with their `events` cleared, matching the
    /// source's `ListSessionsResponse` contract.
    fn list_sessions<'a>(
        &'a self,
        app_name: &'a str,
        user_id: Option<&'a str>,
    ) -> BoxFuture<'a, Vec<Session>>;

    /// Deletes a session. A no-op if it doesn't exist.
    fn delete_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        session_id: &'a str,
    ) -> BoxFuture<'a, ()>;

    /// Appends an event to a session, applying/trimming its `temp:`-scoped
    /// state delta and merging the rest into session state — ported
    /// directly from the source's own (non-abstract, shared-by-every-
    /// backend) default implementation.
    fn append_event<'a>(&'a self, session: &'a mut Session, event: Event) -> BoxFuture<'a, Event> {
        Box::pin(async move { apply_session_event(session, event) })
    }

    /// Flushes any buffered events. A no-op for a non-buffering backend.
    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

fn apply_session_event(session: &mut Session, mut event: Event) -> Event {
    if event.partial.unwrap_or(false) {
        return event;
    }
    for (key, value) in event.actions.state_delta.iter() {
        if key.starts_with(crate::state::State::TEMP_PREFIX) {
            session.state.insert(key.clone(), value.clone());
        }
    }
    event
        .actions
        .state_delta
        .retain(|key, _| !key.starts_with(crate::state::State::TEMP_PREFIX));
    for (key, value) in event.actions.state_delta.iter() {
        session.state.insert(key.clone(), value.clone());
    }
    session.events.push(event.clone());
    event
}

/// `app_name -> user_id -> session_id -> Session`.
type SessionsByAppAndUser = BTreeMap<String, BTreeMap<String, BTreeMap<String, Session>>>;

/// An in-memory [`SessionService`] — for testing/development only, same
/// as the source's own `InMemorySessionService`. See the module doc for
/// what's narrowed relative to the source.
#[derive(Default)]
pub struct InMemorySessionService {
    sessions: Mutex<SessionsByAppAndUser>,
}

impl InMemorySessionService {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionService for InMemorySessionService {
    fn create_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        state: Option<BTreeMap<String, Value>>,
        session_id: Option<String>,
    ) -> BoxFuture<'a, Result<Session, AlreadyExistsError>> {
        Box::pin(async move {
            let mut sessions = self.sessions.lock().unwrap();
            let existing = sessions
                .get(app_name)
                .and_then(|by_user| by_user.get(user_id));
            let session_id = match session_id.as_deref().map(str::trim) {
                Some(id) if !id.is_empty() => {
                    if existing.is_some_and(|s| s.contains_key(id)) {
                        return Err(AlreadyExistsError::new(format!(
                            "Session with id {id} already exists."
                        )));
                    }
                    id.to_string()
                }
                _ => new_uuid().to_string(),
            };
            let session = Session {
                id: session_id.clone(),
                app_name: app_name.to_string(),
                user_id: user_id.to_string(),
                state: state.unwrap_or_default(),
                events: Vec::new(),
            };
            sessions
                .entry(app_name.to_string())
                .or_default()
                .entry(user_id.to_string())
                .or_default()
                .insert(session_id, session.clone());
            Ok(session)
        })
    }

    fn get_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        session_id: &'a str,
    ) -> BoxFuture<'a, Option<Session>> {
        Box::pin(async move {
            self.sessions
                .lock()
                .unwrap()
                .get(app_name)
                .and_then(|by_user| by_user.get(user_id))
                .and_then(|by_session| by_session.get(session_id))
                .cloned()
        })
    }

    fn list_sessions<'a>(
        &'a self,
        app_name: &'a str,
        user_id: Option<&'a str>,
    ) -> BoxFuture<'a, Vec<Session>> {
        Box::pin(async move {
            let sessions = self.sessions.lock().unwrap();
            let Some(by_user) = sessions.get(app_name) else {
                return Vec::new();
            };
            let mut result = Vec::new();
            for (uid, by_session) in by_user {
                if user_id.is_some_and(|wanted| wanted != uid) {
                    continue;
                }
                for session in by_session.values() {
                    let mut without_events = session.clone();
                    without_events.events.clear();
                    result.push(without_events);
                }
            }
            result
        })
    }

    fn delete_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        session_id: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(by_session) = self
                .sessions
                .lock()
                .unwrap()
                .get_mut(app_name)
                .and_then(|by_user| by_user.get_mut(user_id))
            {
                by_session.remove(session_id);
            }
        })
    }

    /// Overrides the shared default: `get_session`/`create_session` hand
    /// callers their own copy of a session (mirroring the source's
    /// `_copy_session`), so appending to that copy alone would never be
    /// visible to a later `get_session` call. This mirrors the resulting
    /// state/event onto the canonical stored session too — and, matching
    /// the source, dedupes against the STORED session's events (by id and
    /// full equality) so a re-delivered event isn't double-applied, and
    /// silently returns the event unstored (rather than raising) if the
    /// session's app/user/id isn't (or is no longer) in the store. The
    /// source logs a warning in that last case; this port has no logging
    /// framework adopted yet (the same disclosed substitution used
    /// elsewhere in this migration), so it's silent instead.
    fn append_event<'a>(&'a self, session: &'a mut Session, event: Event) -> BoxFuture<'a, Event> {
        Box::pin(async move {
            if event.partial.unwrap_or(false) {
                return event;
            }
            let mut sessions = self.sessions.lock().unwrap();
            let Some(storage_session) = sessions
                .get_mut(&session.app_name)
                .and_then(|by_user| by_user.get_mut(&session.user_id))
                .and_then(|by_session| by_session.get_mut(&session.id))
            else {
                return event;
            };
            if storage_session
                .events
                .iter()
                .any(|existing| existing.id == event.id && *existing == event)
            {
                return event;
            }

            let trimmed = apply_session_event(session, event);
            storage_session.state = session.state.clone();
            storage_session.events.push(trimmed.clone());
            trimmed
        })
    }
}

/// `artifacts.base_artifact_service.BaseArtifactService` — placeholder
/// since Phase 6, now with a first real implementor
/// (`InMemoryArtifactService`, C0265) exercising it. Disclosed
/// narrowings, predating this batch (not newly introduced by
/// `InMemoryArtifactService`): every method's `session_id` is a
/// required `&str`, not the source's `Optional[str]` — so the
/// source's "session_id is `None`" branch (list/scope user-only
/// artifacts without a session in play) isn't representable through
/// this trait signature at all; and `artifact`/the return values stay
/// as opaque `Value` rather than a typed `types.Part`, so an
/// implementor must serialize/deserialize at this boundary itself
/// (the same "parse the opaque `Value` via its own `Deserialize` impl"
/// pattern `ExampleTool`/`PreloadMemoryTool`/`LoadArtifactsTool`
/// already use). `delete_artifact`/`list_versions`/
/// `list_artifact_versions` are added in this batch to match the
/// source's full abstract interface — the pre-existing 4 methods'
/// signatures are otherwise unchanged.
pub trait ArtifactService {
    fn load_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<Value>;

    fn save_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<BTreeMap<String, Value>>,
    ) -> i64;

    fn get_artifact_version(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<ArtifactVersion>;

    fn list_artifact_keys(&self, app_name: &str, user_id: &str, session_id: &str) -> Vec<String>;

    /// C0265: deletes an artifact (all its versions). A no-op if the
    /// artifact doesn't exist, matching the source.
    fn delete_artifact(&self, app_name: &str, user_id: &str, session_id: &str, filename: &str);

    /// C0265: lists all version numbers of an artifact, oldest first.
    fn list_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<i64>;

    /// C0265: lists all versions and their metadata for an artifact.
    fn list_artifact_versions(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Vec<ArtifactVersion>;
}

/// Placeholder for `memory.base_memory_service.BaseMemoryService` (Phase 6).
pub trait MemoryService {
    fn add_session_to_memory(&self, session: &Session);

    fn add_events_to_memory(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        events: &[adk_events::Event],
        custom_metadata: Option<&BTreeMap<String, Value>>,
    );

    fn add_memory(
        &self,
        app_name: &str,
        user_id: &str,
        memories: &[MemoryEntry],
        custom_metadata: Option<&BTreeMap<String, Value>>,
    );

    fn search_memory(&self, app_name: &str, user_id: &str, query: &str) -> SearchMemoryResponse;
}

/// C0527: `auth.credential_service.base_credential_service.BaseCredentialService`
/// — abstract interface for loading/saving tool credentials to/from a
/// backend credential store. Widened from a synchronous, no-context
/// placeholder to the real async, `Context`-taking interface (no
/// implementor existed anywhere in this port yet, so this widening
/// breaks nothing) — the same "widen a placeholder once a real consumer
/// needs the structure" precedent as `AuthConfig` above.
/// `Arc<dyn CredentialService>` is stored as a trait object
/// (`InvocationContext::credential_service`), so methods return a boxed
/// [`BoxFuture`] rather than using native `async fn` (not object-safe) —
/// the same pattern `SessionService`/`adk_tools::base_tool::BaseTool`
/// already use for the same reason.
///
/// **`callback_context` mutability, adapted**: the source's
/// `callback_context: CallbackContext` parameter is the same reference
/// for both methods. This port narrows `load_credential` to `&Context`
/// (read-only — `SessionStateCredentialService` only reads state) and
/// widens `save_credential` to `&mut Context` (`SessionStateCredentialService`
/// needs `state_mut()` to write) — Rust's stricter mutability tracking
/// surfacing a distinction the source's dynamic typing doesn't need to
/// make explicit.
pub trait CredentialService: Send + Sync {
    fn load_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a Context,
    ) -> BoxFuture<'a, Option<AuthCredential>>;

    fn save_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, ()>;
}

/// C0353-C0354, C0357 (partial — see the module doc): `BasePlugin`, the
/// extension point every registered plugin implements. Every hook
/// defaults to a no-op, matching the source's own `BasePlugin` (a plain
/// `pass`-bodied base class) — a concrete plugin overrides only the hooks
/// it cares about.
///
/// **Not ported**: the model-level (`before_model_callback`/
/// `after_model_callback`/`on_model_error_callback`, C0355) and tool-level
/// (`before_tool_callback`/`after_tool_callback`/`on_tool_error_callback`,
/// C0356) hooks — those need `LlmRequest`/`LlmResponse` (`adk-models`) and
/// `BaseTool`/`ToolContext` (`adk-tools`) types, and `adk-tools` already
/// depends on `adk-agents` (for `BaseTool`'s own `ToolContext` alias), so
/// `adk-agents` can't depend back on either without the same crate-graph
/// cycle `LlmRequest::append_tools` (C0116) already disclosed. A unified
/// `BasePlugin` spanning all four hook levels needs its own home above
/// `adk-tools`/`adk-models` (mirroring `adk-tools`'s own placement above
/// `adk-models`) — deferred to a follow-up batch once model/tool call
/// sites exist to wire it into.
pub trait BasePlugin: Send + Sync {
    /// The plugin's registration name — must be unique within one
    /// [`PluginManager`].
    fn name(&self) -> &str;

    // ---- Run-level hooks (C0353) ----

    fn on_user_message_callback<'a>(
        &'a self,
        _invocation_context: &'a mut Context,
        _user_message: &'a Content,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async { None })
    }

    fn before_run_callback<'a>(
        &'a self,
        _invocation_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async { None })
    }

    fn on_event_callback<'a>(
        &'a self,
        _invocation_context: &'a mut Context,
        _event: &'a Event,
    ) -> BoxFuture<'a, Option<Event>> {
        Box::pin(async { None })
    }

    fn after_run_callback<'a>(&'a self, _invocation_context: &'a mut Context) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    // ---- Agent-level hooks (C0354) ----

    fn before_agent_callback<'a>(
        &'a self,
        _agent: &'a BaseAgent,
        _callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async { None })
    }

    fn after_agent_callback<'a>(
        &'a self,
        _agent: &'a BaseAgent,
        _callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async { None })
    }

    // ---- Notification-only hooks (C0357) — MUST NOT be relied on to
    // short-circuit; the triggering error is always re-raised by the
    // caller after every plugin is notified. ----

    fn on_agent_error_callback<'a>(
        &'a self,
        _agent: &'a BaseAgent,
        _callback_context: &'a mut Context,
        _error: &'a AgentRunError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    fn on_run_error_callback<'a>(
        &'a self,
        _invocation_context: &'a mut Context,
        _error: &'a AgentRunError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// Releases any resources the plugin holds.
    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

#[derive(Debug, rusty_err::Error)]
pub enum PluginManagerError {
    #[error("Plugin with name '{0}' already registered.")]
    DuplicateName(String),
}

/// C0359-C0361 (partial — see [`BasePlugin`]'s module doc for the hook
/// levels this doesn't cover yet): manages plugin registration and
/// dispatch. Runs registered plugins in registration order; for the
/// short-circuiting hooks, the first plugin to return `Some(..)` stops
/// the rest from running and its value is returned (C0358). The
/// notification-only hooks always run every plugin regardless (C0360).
///
/// **Adaptation**: the source's `_run_callbacks` wraps a plugin exception
/// in `RuntimeError` and re-raises, stopping iteration; this port's hooks
/// return a value rather than raising, so there's no exception channel to
/// intercept the same way — a panicking plugin propagates the panic
/// unchanged, the same "no unwind-safety net over an arbitrary callback"
/// posture `AgentCallback` closures already have in this port.
///
/// **Adaptation, disclosed**: [`PluginManager::close`] is sequential, not
/// concurrent, per plugin — matching the source's *actual* implementation
/// (a deliberate choice to avoid task-local-context issues with
/// anyio/MCP), not its docstring's claim of running plugins concurrently;
/// the manifest flags that inconsistency explicitly rather than
/// replicating it blindly.
#[derive(Clone, Default)]
pub struct PluginManager {
    plugins: Vec<Arc<dyn BasePlugin>>,
    skip_closing_plugins: bool,
}

impl PluginManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_plugin(
        &mut self,
        plugin: Arc<dyn BasePlugin>,
    ) -> Result<(), PluginManagerError> {
        if self.plugins.iter().any(|p| p.name() == plugin.name()) {
            return Err(PluginManagerError::DuplicateName(plugin.name().to_string()));
        }
        self.plugins.push(plugin);
        Ok(())
    }

    pub fn get_plugin(&self, name: &str) -> Option<&Arc<dyn BasePlugin>> {
        self.plugins.iter().find(|p| p.name() == name)
    }

    /// Controls whether [`PluginManager::close`] tears down the
    /// registered plugins — set when the plugins are owned by another
    /// component (e.g. a parent `Runner` sharing its plugin list).
    pub fn set_skip_closing_plugins(&mut self, value: bool) {
        self.skip_closing_plugins = value;
    }

    pub fn run_on_user_message_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
        user_message: &'a Content,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            for plugin in &self.plugins {
                if let Some(content) = plugin
                    .on_user_message_callback(invocation_context, user_message)
                    .await
                {
                    return Some(content);
                }
            }
            None
        })
    }

    pub fn run_before_run_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            for plugin in &self.plugins {
                if let Some(content) = plugin.before_run_callback(invocation_context).await {
                    return Some(content);
                }
            }
            None
        })
    }

    pub fn run_on_event_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
        event: &'a Event,
    ) -> BoxFuture<'a, Option<Event>> {
        Box::pin(async move {
            for plugin in &self.plugins {
                if let Some(event) = plugin.on_event_callback(invocation_context, event).await {
                    return Some(event);
                }
            }
            None
        })
    }

    pub fn run_after_run_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for plugin in &self.plugins {
                plugin.after_run_callback(invocation_context).await;
            }
        })
    }

    pub fn run_before_agent_callback<'a>(
        &'a self,
        agent: &'a BaseAgent,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            for plugin in &self.plugins {
                if let Some(content) = plugin.before_agent_callback(agent, callback_context).await {
                    return Some(content);
                }
            }
            None
        })
    }

    pub fn run_after_agent_callback<'a>(
        &'a self,
        agent: &'a BaseAgent,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            for plugin in &self.plugins {
                if let Some(content) = plugin.after_agent_callback(agent, callback_context).await {
                    return Some(content);
                }
            }
            None
        })
    }

    pub fn run_on_agent_error_callback<'a>(
        &'a self,
        agent: &'a BaseAgent,
        callback_context: &'a mut Context,
        error: &'a AgentRunError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for plugin in &self.plugins {
                plugin
                    .on_agent_error_callback(agent, callback_context, error)
                    .await;
            }
        })
    }

    pub fn run_on_run_error_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
        error: &'a AgentRunError,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for plugin in &self.plugins {
                plugin
                    .on_run_error_callback(invocation_context, error)
                    .await;
            }
        })
    }

    pub fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if self.skip_closing_plugins {
                return;
            }
            for plugin in &self.plugins {
                plugin.close().await;
            }
        })
    }
}

/// Renders a UI widget by appending it to the given actions list, raising
/// (as `Err`) on a duplicate widget id. Shared by `Context::render_ui_widget`
/// (C0065) — factored out here since it only needs `UiWidget`, not any
/// context state.
/// Returns `Err(widget_id)` on a duplicate id — the caller formats the
/// user-facing message (see `Context::render_ui_widget`'s `ContextError`).
pub fn render_ui_widget(
    widgets: &mut Option<Vec<UiWidget>>,
    widget: UiWidget,
) -> Result<(), String> {
    let list = widgets.get_or_insert_with(Vec::new);
    if list.iter().any(|existing| existing.id == widget.id) {
        return Err(widget.id);
    }
    list.push(widget);
    Ok(())
}

/// Generates a fresh invocation id, mirroring
/// `invocation_context.new_invocation_context_id`.
pub fn new_invocation_context_id() -> String {
    format!("e-{}", new_uuid())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingPlugin {
        name: String,
        before_agent_result: Option<Content>,
        closed: Arc<Mutex<bool>>,
        error_notifications: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingPlugin {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                before_agent_result: None,
                closed: Arc::new(Mutex::new(false)),
                error_notifications: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn returning(mut self, content: Content) -> Self {
            self.before_agent_result = Some(content);
            self
        }
    }

    impl BasePlugin for RecordingPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn before_agent_callback<'a>(
            &'a self,
            _agent: &'a BaseAgent,
            _callback_context: &'a mut Context,
        ) -> BoxFuture<'a, Option<Content>> {
            let result = self.before_agent_result.clone();
            Box::pin(async move { result })
        }

        fn on_run_error_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
            _error: &'a AgentRunError,
        ) -> BoxFuture<'a, ()> {
            let notifications = self.error_notifications.clone();
            let name = self.name.clone();
            Box::pin(async move {
                notifications.lock().unwrap().push(name);
            })
        }

        fn close(&self) -> BoxFuture<'_, ()> {
            let closed = self.closed.clone();
            Box::pin(async move {
                *closed.lock().unwrap() = true;
            })
        }
    }

    fn test_agent() -> BaseAgent {
        BaseAgent::new("test_agent", crate::base_agent::NoopBehavior).unwrap()
    }

    fn test_context() -> Context {
        Context::new(
            crate::invocation_context::InvocationContextBuilder::new(
                "inv-1",
                Session::new("app", "user", "s1"),
            )
            .build(),
        )
    }

    #[derive(Debug)]
    struct BoomError;
    impl std::fmt::Display for BoomError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("boom")
        }
    }
    impl std::error::Error for BoomError {}

    #[test]
    fn register_plugin_rejects_a_duplicate_name() {
        let mut manager = PluginManager::new();
        manager
            .register_plugin(Arc::new(RecordingPlugin::new("p1")))
            .unwrap();
        let err = manager
            .register_plugin(Arc::new(RecordingPlugin::new("p1")))
            .unwrap_err();
        assert!(matches!(err, PluginManagerError::DuplicateName(name) if name == "p1"));
    }

    #[test]
    fn get_plugin_finds_a_registered_plugin_by_name() {
        let mut manager = PluginManager::new();
        manager
            .register_plugin(Arc::new(RecordingPlugin::new("p1")))
            .unwrap();
        assert!(manager.get_plugin("p1").is_some());
        assert!(manager.get_plugin("missing").is_none());
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_short_circuits_at_the_first_non_none_result() {
        let mut manager = PluginManager::new();
        manager
            .register_plugin(Arc::new(
                RecordingPlugin::new("first").returning(Content::user_text("from first")),
            ))
            .unwrap();
        manager
            .register_plugin(Arc::new(
                RecordingPlugin::new("second").returning(Content::user_text("from second")),
            ))
            .unwrap();

        let agent = test_agent();
        let mut ctx = test_context();
        let result = manager.run_before_agent_callback(&agent, &mut ctx).await;
        assert_eq!(result, Some(Content::user_text("from first")));
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_returns_none_when_no_plugin_short_circuits() {
        let mut manager = PluginManager::new();
        manager
            .register_plugin(Arc::new(RecordingPlugin::new("silent")))
            .unwrap();

        let agent = test_agent();
        let mut ctx = test_context();
        assert_eq!(
            manager.run_before_agent_callback(&agent, &mut ctx).await,
            None
        );
    }

    #[rusty_tokio::test]
    async fn on_run_error_callback_notifies_every_plugin_regardless() {
        let mut manager = PluginManager::new();
        let tracker = Arc::new(Mutex::new(Vec::new()));
        let mut first = RecordingPlugin::new("first");
        first.error_notifications = tracker.clone();
        let mut second = RecordingPlugin::new("second");
        second.error_notifications = tracker.clone();
        manager.register_plugin(Arc::new(first)).unwrap();
        manager.register_plugin(Arc::new(second)).unwrap();

        let mut ctx = test_context();
        let error: AgentRunError = Box::new(BoomError);
        manager.run_on_run_error_callback(&mut ctx, &error).await;

        assert_eq!(*tracker.lock().unwrap(), vec!["first", "second"]);
    }

    #[rusty_tokio::test]
    async fn close_closes_every_registered_plugin() {
        let mut manager = PluginManager::new();
        let plugin = RecordingPlugin::new("p1");
        let closed = plugin.closed.clone();
        manager.register_plugin(Arc::new(plugin)).unwrap();

        manager.close().await;
        assert!(*closed.lock().unwrap());
    }

    #[rusty_tokio::test]
    async fn close_is_a_no_op_when_skip_closing_plugins_is_set() {
        let mut manager = PluginManager::new();
        let plugin = RecordingPlugin::new("p1");
        let closed = plugin.closed.clone();
        manager.register_plugin(Arc::new(plugin)).unwrap();
        manager.set_skip_closing_plugins(true);

        manager.close().await;
        assert!(!*closed.lock().unwrap());
    }

    #[test]
    fn render_ui_widget_rejects_duplicate_ids() {
        let mut widgets = None;
        render_ui_widget(&mut widgets, UiWidget::new("w1", "mcp", Value::Null)).unwrap();
        let err =
            render_ui_widget(&mut widgets, UiWidget::new("w1", "mcp", Value::Null)).unwrap_err();
        assert!(err.contains("w1"));
    }

    #[test]
    fn new_invocation_context_id_is_prefixed() {
        assert!(new_invocation_context_id().starts_with("e-"));
    }

    use adk_events::node_info::NodeInfo;
    use adk_events::Event;

    #[rusty_tokio::test]
    async fn create_session_generates_an_id_when_none_is_given() {
        let service = InMemorySessionService::new();
        let session = service
            .create_session("app", "user", None, None)
            .await
            .unwrap();
        assert!(!session.id.is_empty());
        assert_eq!(session.app_name, "app");
        assert_eq!(session.user_id, "user");
    }

    #[rusty_tokio::test]
    async fn create_session_rejects_a_duplicate_explicit_id() {
        let service = InMemorySessionService::new();
        service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();
        let err = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("s1"));
    }

    #[rusty_tokio::test]
    async fn get_session_returns_none_when_missing() {
        let service = InMemorySessionService::new();
        assert!(service
            .get_session("app", "user", "missing")
            .await
            .is_none());
    }

    #[rusty_tokio::test]
    async fn get_session_round_trips_a_created_session() {
        let service = InMemorySessionService::new();
        service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();
        let session = service.get_session("app", "user", "s1").await.unwrap();
        assert_eq!(session.id, "s1");
    }

    // --- GetSessionConfig (C0207) ---

    fn timestamped_event(timestamp: f64) -> Event {
        let mut event = Event::new("inv-1", "user", adk_events::node_info::NodeInfo::new(""));
        event.timestamp = timestamp;
        event
    }

    async fn session_with_timestamped_events(
        service: &InMemorySessionService,
        timestamps: &[f64],
    ) -> Session {
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();
        for &ts in timestamps {
            service
                .append_event(&mut session, timestamped_event(ts))
                .await;
        }
        session
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_behaves_like_get_session_without_a_config() {
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0, 3.0]).await;

        let session = service
            .get_session_with_config("app", "user", "s1", None)
            .await
            .unwrap();
        assert_eq!(session.events.len(), 3);
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_returns_none_when_missing() {
        let service = InMemorySessionService::new();
        assert!(service
            .get_session_with_config("app", "user", "missing", Some(&GetSessionConfig::default()))
            .await
            .is_none());
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_num_recent_events_keeps_only_the_tail() {
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0, 3.0, 4.0]).await;

        let config = GetSessionConfig {
            num_recent_events: Some(2),
            after_timestamp: None,
        };
        let session = service
            .get_session_with_config("app", "user", "s1", Some(&config))
            .await
            .unwrap();
        let timestamps: Vec<f64> = session.events.iter().map(|e| e.timestamp).collect();
        assert_eq!(timestamps, vec![3.0, 4.0]);
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_num_recent_events_zero_returns_no_events() {
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0]).await;

        let config = GetSessionConfig {
            num_recent_events: Some(0),
            after_timestamp: None,
        };
        let session = service
            .get_session_with_config("app", "user", "s1", Some(&config))
            .await
            .unwrap();
        assert!(session.events.is_empty());
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_after_timestamp_drops_older_events() {
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0, 3.0]).await;

        let config = GetSessionConfig {
            num_recent_events: None,
            after_timestamp: Some(2.0),
        };
        let session = service
            .get_session_with_config("app", "user", "s1", Some(&config))
            .await
            .unwrap();
        let timestamps: Vec<f64> = session.events.iter().map(|e| e.timestamp).collect();
        assert_eq!(timestamps, vec![2.0, 3.0]);
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_after_timestamp_zero_is_a_noop() {
        // Mirrors the source's own Python truthiness quirk: `if
        // config.after_timestamp:` treats `0.0` the same as unset.
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0]).await;

        let config = GetSessionConfig {
            num_recent_events: None,
            after_timestamp: Some(0.0),
        };
        let session = service
            .get_session_with_config("app", "user", "s1", Some(&config))
            .await
            .unwrap();
        assert_eq!(session.events.len(), 2);
    }

    #[rusty_tokio::test]
    async fn get_session_with_config_composes_both_filters() {
        let service = InMemorySessionService::new();
        session_with_timestamped_events(&service, &[1.0, 2.0, 3.0, 4.0, 5.0]).await;

        // Last 3 of the 5 events are [3.0, 4.0, 5.0]; then only those
        // >= 4.0 survive the second filter.
        let config = GetSessionConfig {
            num_recent_events: Some(3),
            after_timestamp: Some(4.0),
        };
        let session = service
            .get_session_with_config("app", "user", "s1", Some(&config))
            .await
            .unwrap();
        let timestamps: Vec<f64> = session.events.iter().map(|e| e.timestamp).collect();
        assert_eq!(timestamps, vec![4.0, 5.0]);
    }

    #[rusty_tokio::test]
    async fn list_sessions_filters_by_user_and_clears_events() {
        let service = InMemorySessionService::new();
        service
            .create_session("app", "alice", None, Some("s1".to_string()))
            .await
            .unwrap();
        service
            .create_session("app", "bob", None, Some("s2".to_string()))
            .await
            .unwrap();

        let all = service.list_sessions("app", None).await;
        assert_eq!(all.len(), 2);

        let alice_only = service.list_sessions("app", Some("alice")).await;
        assert_eq!(alice_only.len(), 1);
        assert_eq!(alice_only[0].id, "s1");
    }

    #[rusty_tokio::test]
    async fn delete_session_removes_it() {
        let service = InMemorySessionService::new();
        service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();
        service.delete_session("app", "user", "s1").await;
        assert!(service.get_session("app", "user", "s1").await.is_none());
    }

    #[rusty_tokio::test]
    async fn delete_session_is_a_no_op_when_missing() {
        let service = InMemorySessionService::new();
        service.delete_session("app", "user", "missing").await;
    }

    #[rusty_tokio::test]
    async fn append_event_persists_a_non_partial_event_and_updates_state() {
        let service = InMemorySessionService::new();
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();

        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        event
            .actions
            .state_delta
            .insert("k".to_string(), Value::String("v".to_string()));

        service.append_event(&mut session, event).await;

        assert_eq!(session.events.len(), 1);
        assert_eq!(
            session.state.get("k"),
            Some(&Value::String("v".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn append_event_skips_persistence_for_a_partial_event() {
        let service = InMemorySessionService::new();
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();

        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        event.partial = Some(true);

        service.append_event(&mut session, event).await;
        assert!(session.events.is_empty());
    }

    #[rusty_tokio::test]
    async fn append_event_applies_temp_state_but_trims_it_from_the_persisted_event() {
        let service = InMemorySessionService::new();
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();

        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        event.actions.state_delta.insert(
            "temp:scratch".to_string(),
            Value::String("ephemeral".to_string()),
        );
        event
            .actions
            .state_delta
            .insert("permanent".to_string(), Value::String("kept".to_string()));

        service.append_event(&mut session, event).await;

        assert_eq!(
            session.state.get("temp:scratch"),
            Some(&Value::String("ephemeral".to_string()))
        );
        assert_eq!(
            session.state.get("permanent"),
            Some(&Value::String("kept".to_string()))
        );
        assert!(!session.events[0]
            .actions
            .state_delta
            .contains_key("temp:scratch"));
        assert!(session.events[0]
            .actions
            .state_delta
            .contains_key("permanent"));
    }

    #[rusty_tokio::test]
    async fn flush_is_a_no_op() {
        let service = InMemorySessionService::new();
        service.flush().await;
    }

    #[rusty_tokio::test]
    async fn append_event_mirrors_onto_the_canonical_stored_session() {
        let service = InMemorySessionService::new();
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();

        let event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        service.append_event(&mut session, event).await;

        // A fresh get_session (a fresh copy) sees the appended event too —
        // proof the mutation reached the canonical stored session, not
        // just the caller's own copy.
        let refetched = service.get_session("app", "user", "s1").await.unwrap();
        assert_eq!(refetched.events.len(), 1);
    }

    #[rusty_tokio::test]
    async fn append_event_dedupes_a_redelivered_event() {
        let service = InMemorySessionService::new();
        let mut session = service
            .create_session("app", "user", None, Some("s1".to_string()))
            .await
            .unwrap();

        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        event
            .actions
            .state_delta
            .insert("k".to_string(), Value::String("first".to_string()));

        service.append_event(&mut session, event.clone()).await;
        // Re-deliver the exact same event (same id, same fields).
        service.append_event(&mut session, event).await;

        let refetched = service.get_session("app", "user", "s1").await.unwrap();
        assert_eq!(
            refetched.events.len(),
            1,
            "a re-delivered event must not be double-applied"
        );
    }

    #[rusty_tokio::test]
    async fn append_event_returns_the_event_unstored_for_an_unknown_session() {
        let service = InMemorySessionService::new();
        let mut session = Session::new("app", "user", "never-created");
        let event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        let returned = service.append_event(&mut session, event.clone()).await;
        assert_eq!(returned.id, event.id);
        // Nothing was stored: re-fetching finds nothing.
        assert!(service
            .get_session("app", "user", "never-created")
            .await
            .is_none());
    }
}
