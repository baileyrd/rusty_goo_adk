//! Capability C0833-C0926 (partial): `Runner`, ported from
//! `google.adk.runners`.
//!
//! **Scoping**: `runners.py` is 2609 lines / 94 capability rows, and most
//! of it depends on infrastructure this port doesn't have yet:
//! - `Runner.__init__`'s `app`/`agent`/`node` mutual-exclusivity contract
//!   and `App.model_construct` wrapping stay N/A: this port's `Runner`
//!   only ever wraps a concrete [`BaseAgent`] directly (`Runner::new`),
//!   so there's no app/agent/node union to validate against. An [`App`]
//!   type does exist now (`adk-agents::app`, C0279/C0280), so
//!   `app.root_agent`/`app.plugins`/`app.context_cache_config`/
//!   `app.resumability_config` resolution (C0846/C0849) is real, wired
//!   as an additive second constructor, [`Runner::from_app`], rather
//!   than a change to `Runner::new`'s already-shipped signature.
//! - The workflow/node/task-delegation engine (`BaseNode`,
//!   `Context.run_node`, `DynamicNodeScheduler`, task-scope tracking,
//!   Phase 7) — confirmed absent from this port. Every node/task-mode
//!   code path (`_run_node_async`/`_run_node_live`, task-scope resolution,
//!   resumability) is out of scope until that engine exists.
//! - Live mode (`run_live`), `rewind_async`, `run_debug` — each needs its
//!   own supporting infrastructure (live request queue wiring into
//!   `InvocationContext`, artifact-versioned rewind deltas, etc.) beyond
//!   this batch's scope.
//!
//! What's left, and what this batch builds, is the "legacy" (plain
//! `BaseAgent`, single always-non-resumable turn) execution path
//! (C0884-C0886): [`Runner::run_async`] fetches-or-creates a session,
//! appends the user message, drives `agent.run_async` (the real
//! orchestration primitive `adk_agents::base_agent::BaseAgent` already
//! provides), and persists the resulting events. Also built: the
//! constructor/field contract (C0840-C0845, narrowed — see above) and
//! [`Runner::close`] (C0924, partial — closes what actually exists today:
//! `session_service.flush()` and the registered plugins; toolset
//! collection (C0922/C0923) is deferred until `LlmAgent.tools` holds real
//! `BaseToolset` instances instead of the `ToolUnion` placeholder).
//!
//! **Run-level plugin wiring** (C0353/C0357/C0886's remaining gap,
//! C0895-C0899): `Runner` now stores a real `PluginManager`
//! (`Runner::with_plugin`) and wires its run-level hooks into
//! [`Runner::run_async`]/[`Runner::run_async_with_config`], the Rust
//! analog of `_exec_with_plugin`/`_handle_new_message`/
//! `_append_new_message_to_session`: `run_on_user_message_callback` (may
//! replace the incoming message), the deprecated
//! `save_input_blobs_as_artifacts` blob-saving path, appending the user
//! event, `run_before_run_callback` (a returned `Content` short-circuits
//! the whole turn into a single early-exit event), driving
//! `agent.run_async`, then per produced event: `_apply_run_config_custom_metadata`,
//! `run_on_event_callback` (see [`merge_output_event`], C0896), and
//! [`should_append_event`] (C0895) gating persistence — finally
//! `run_after_run_callback` on success. An `agent.run_async` failure
//! notifies `run_on_run_error_callback` (C0357) before propagating.
//!
//! **Disclosed narrowings**:
//! - [`merge_output_event`] (C0896): `on_event_callback` in this port
//!   already returns a full replacement `Event`, not a partial-update
//!   object gated by the source's `model_fields_set` — so "merge only the
//!   fields the plugin actually set" collapses to "use the plugin's full
//!   replacement event, except `id`/`invocation_id`/`timestamp` (never
//!   overridden, matching the source) and a blank `author` (falls back to
//!   the original, also matching the source)."
//! - [`should_append_event`] (C0895): the source's live-mode branch
//!   (suppressing raw inline-blob media events while still persisting
//!   ones referencing an artifact via `file_data`) is unreachable through
//!   this port today — `Runner` has no live-mode call path yet (see
//!   "Not ported this batch" below) — so only the always-`true`,
//!   non-live half is implemented; `is_live_call` is threaded through the
//!   signature for parity even though every call site here passes
//!   `false`.
//! - C0899 (the deprecated blob flag's read guard): the source reads
//!   `run_config.save_input_blobs_as_artifacts` only if that field name
//!   is literally in the pydantic model's `model_fields_set` — i.e. only
//!   if the caller's constructor call explicitly passed it, treating a
//!   value set by later attribute mutation as unset/`False`. This port's
//!   `RunConfig` is a plain public-field struct with no builder and no
//!   constructor-vs-mutation distinction to preserve (matching every
//!   other config type in this crate), so [`Runner::run_async_with_config`]
//!   reads the field directly — behaviorally identical for every
//!   construction path this crate's own call sites and tests use.
//! - This port's `agent.run_async` returns a fully-materialized
//!   `Vec<Event>`, not an async generator — so there is no "some events
//!   already streamed and persisted before a later one fails mid-stream"
//!   case to reproduce; a failure here means zero output events were
//!   produced (already an established scope cut for this whole module,
//!   see above).
//! - Plugin hooks in this port have no `Result`/exception channel to
//!   intercept (already established by `services::PluginManager`'s own
//!   module doc) — so unlike the source, an `after_run_callback` plugin
//!   "failure" can't be caught and re-notified through
//!   `on_run_error_callback`; a panicking plugin propagates the panic
//!   unchanged, same posture as every other callback in this port.
//!
//! **`InMemoryRunner`** (C0926): [`Runner::in_memory`] — a `Runner`
//! pre-wired with `InMemoryArtifactService`/`InMemorySessionService`/
//! `InMemoryMemoryService`, for testing and development. See its own
//! doc for the narrowing (a constructor, not a subclass) and the
//! `app_name` default.
//!
//! **`Runner::from_app`** (C0846/C0849): builds a `Runner` from a
//! resolved [`App`], deriving `context_cache_config`/
//! `resumability_config`/`plugins` from it rather than accepting them as
//! direct constructor arguments — additive, doesn't touch `Runner::new`'s
//! already-shipped signature. C0847's `_enforce_app_name_alignment`/
//! `_warn_uncached_agent_transfer` calls, and C0850's deprecated
//! `_validate_runner_params` back-compat wrapper, both depend on
//! `_infer_agent_origin` (C0851, already-disclosed N/A below — no Rust
//! module-path reflection) or on logging machinery not adopted in this
//! port, so neither has anything to port yet; C0848
//! (`_require_root_agent`) is also N/A — `Runner::agent` is always a
//! concrete `BaseAgent`, never a bare node, so there's no node-vs-agent
//! narrowing check to perform.
//!
//! **Not ported this batch**: `_find_agent_to_run` (C0907-C0910, picks up
//! a resumed multi-turn conversation's last-active agent — needs
//! resumability, always false here); `_resolve_invocation_id` (C0855,
//! same reason); `Runner.run()`'s sync thread-bridging wrapper
//! (C0877-C0880, a local-testing convenience — this port's whole call
//! surface is already async-native, so there's less need for it, and it
//! can be added later without disturbing anything here); compaction
//! (`_run_post_invocation_compaction`, C0871-C0872, needs
//! `events_compaction_config` wiring, Phase 7); agent-origin inference
//! and its warnings (C0851-C0854) — Rust has no runtime module-path
//! reflection to inspect, and no logging framework is adopted to warn
//! through even if it did.

use std::sync::Arc;

use adk_agents::app::App;
use adk_agents::app_configs::ResumabilityConfig;
use adk_agents::base_agent::BaseAgent;
use adk_agents::context::Context;
use adk_agents::context_cache_config::ContextCacheConfig;
use adk_agents::invocation_context::InvocationContextBuilder;
use adk_agents::run_config::RunConfig;
use adk_agents::services::{
    new_invocation_context_id, ArtifactService, BasePlugin, CredentialService, MemoryService,
    PluginManager, PluginManagerError, SessionService,
};
use adk_agents::session::Session;
use adk_errors::already_exists::AlreadyExistsError;
use adk_errors::session_not_found::SessionNotFoundError;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;

#[derive(Debug, rusty_err::Error)]
pub enum RunnerError {
    #[error("{0}")]
    SessionNotFound(SessionNotFoundError),
    #[error("{0}")]
    AlreadyExists(AlreadyExistsError),
    #[error("run_async rejects a user-authored new_message with no parts")]
    EmptyNewMessage,
    #[error("run_async rejects a user-authored new_message containing a function call")]
    NewMessageContainsFunctionCall,
    #[error("agent run failed: {0}")]
    AgentRun(String),
}

/// C0896: mirrors `_get_output_event`. If `modified` is `None`, the
/// original event is used verbatim; otherwise `modified` is used as the
/// output event, with `original`'s `id`/`invocation_id`/`timestamp`
/// restored (never overridden by a plugin) and `original`'s `author`
/// restored as a fallback if `modified`'s author is blank. See the
/// module doc for the disclosed narrowing relative to the source's
/// field-presence-tracked partial merge.
fn merge_output_event(original: Event, modified: Option<Event>) -> Event {
    let Some(mut output) = modified else {
        return original;
    };
    output.id = original.id;
    output.invocation_id = original.invocation_id;
    output.timestamp = original.timestamp;
    if output.author.is_empty() {
        output.author = original.author;
    }
    output
}

/// C0895: mirrors `_should_append_event`. See the module doc for the
/// disclosed narrowing — only the non-live (`is_live_call == false`)
/// half is implemented.
fn should_append_event(_event: &Event, is_live_call: bool) -> bool {
    !is_live_call
}

/// Bridges a run-level plugin hook's state mutations back onto the
/// session. In the source, run-level hooks (`on_user_message_callback`,
/// `before_run_callback`, `on_event_callback`, `after_run_callback`) take
/// the raw, shared `InvocationContext` and mutate `invocation_context
/// .session.state` as a plain dict in place — any later read (including
/// from a different hook) sees the mutation immediately, since it's the
/// same dict object. This port's `Context` wraps a *clone* of
/// `InvocationContext` (no reference semantics), so a hook's state
/// mutations are otherwise invisible outside the throwaway `Context` it
/// ran in. This applies the resulting state delta directly onto
/// `session.state` (bypassing the event/`state_delta` append path
/// entirely — matching the source's own no-event, immediate-mutation
/// semantics rather than the `CallbackContext`-based agent-level hooks'
/// synthesized-event pattern) and refreshes `invocation_context.session`
/// so every subsequent step (including a later hook, or the driven
/// agent itself) observes it — the same "widen once a real consumer
/// needs the structure" pattern this port applies elsewhere, here
/// applied to close a visibility gap `SaveFilesAsArtifactsPlugin`
/// (C0367) surfaced: its `on_user_message_callback` stashes a pending
/// delta that `before_agent_callback` must see.
fn merge_context_state_into_session(
    ctx: Context,
    invocation_context: &mut adk_agents::invocation_context::InvocationContext,
    session: &mut Session,
) {
    let state_delta = ctx.into_actions().state_delta;
    if state_delta.is_empty() {
        return;
    }
    session.state.extend(state_delta);
    invocation_context.session = session.clone();
}

/// Mirrors `_apply_run_config_custom_metadata`: merges
/// `run_config.custom_metadata` into `event.custom_metadata`, giving
/// priority to keys the event already carries.
fn apply_run_config_custom_metadata(event: &mut Event, run_config: &RunConfig) {
    let Some(config_metadata) = run_config.custom_metadata.as_ref() else {
        return;
    };
    if config_metadata.is_empty() {
        return;
    }
    let mut merged: std::collections::HashMap<String, Value> = config_metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(existing) = &event.custom_metadata {
        merged.extend(existing.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    event.custom_metadata = Some(merged);
}

/// C0840-C0845 (narrowed, see the module doc): the core execution engine.
/// Wraps exactly one [`BaseAgent`] — no `App`/bare-node union, since
/// neither exists in this port.
pub struct Runner {
    app_name: String,
    agent: BaseAgent,
    session_service: Arc<dyn SessionService + Send + Sync>,
    artifact_service: Option<Arc<dyn ArtifactService + Send + Sync>>,
    memory_service: Option<Arc<dyn MemoryService + Send + Sync>>,
    credential_service: Option<Arc<dyn CredentialService + Send + Sync>>,
    plugin_manager: PluginManager,
    /// C0844: stored for parity with the source's constructor contract.
    /// Not yet applied as an actual per-plugin timeout bound in
    /// [`Runner::close`] — this port's `PluginManager::close` has no
    /// timeout parameter of its own yet (see its own module doc).
    plugin_close_timeout: f64,
    /// C0845: the single switch controlling whether `run_async` creates
    /// a missing session or reports it as not found.
    auto_create_session: bool,
    /// C0846: derived from an [`App`] via [`Runner::from_app`] — never a
    /// direct constructor argument (`Runner::new` always leaves this
    /// `None`), matching the source exactly. Not yet read anywhere else
    /// in this port (compaction, C0871/C0872, isn't built).
    context_cache_config: Option<ContextCacheConfig>,
    /// C0846: same sourcing rule as `context_cache_config` above. Not yet
    /// read anywhere else in this port (resumable-run support isn't
    /// built).
    resumability_config: Option<ResumabilityConfig>,
}

impl Runner {
    /// C0841 (narrowed): `app_name` and `agent` are always both required
    /// here — there's no `App`/bare-node alternative to be mutually
    /// exclusive against.
    pub fn new(
        app_name: impl Into<String>,
        agent: BaseAgent,
        session_service: Arc<dyn SessionService + Send + Sync>,
    ) -> Self {
        Self {
            app_name: app_name.into(),
            agent,
            session_service,
            artifact_service: None,
            memory_service: None,
            credential_service: None,
            plugin_manager: PluginManager::new(),
            plugin_close_timeout: 5.0,
            auto_create_session: false,
            context_cache_config: None,
            resumability_config: None,
        }
    }

    /// C0846/C0849: the single normalization path from a resolved [`App`]
    /// to a `Runner` — the source's `_resolve_app` plus the constructor's
    /// `app.*`-extraction block, narrowed to the one shape this port's
    /// `App`/`Runner` pair can represent (no `app`/`agent`/`node` union;
    /// `App::root_agent` is a required, never-absent field — see both
    /// types' own module docs — so the source's "raises ValueError if
    /// app.root_agent is None" check has nothing to guard here).
    /// `app_name` defaults to `app.name`; pass `Some(..)` to override it,
    /// matching the source's `app_name or app.name`. `app.plugins` folds
    /// into the registered plugin set via [`Runner::with_plugin`], so a
    /// duplicate plugin name surfaces the same [`PluginManagerError`] it
    /// would through that method directly.
    pub fn from_app(
        app: App,
        app_name_override: Option<String>,
        session_service: Arc<dyn SessionService + Send + Sync>,
    ) -> Result<Self, PluginManagerError> {
        let app_name = app_name_override.unwrap_or(app.name);
        let mut runner = Self::new(app_name, app.root_agent, session_service);
        runner.context_cache_config = app.context_cache_config;
        runner.resumability_config = app.resumability_config;
        for plugin in app.plugins {
            runner = runner.with_plugin(plugin)?;
        }
        Ok(runner)
    }

    /// C0926: `InMemoryRunner` — a `Runner` pre-wired with in-memory
    /// session/artifact/memory services, for testing and development.
    /// The source is a `Runner` subclass; this port narrows that to a
    /// constructor (matching `Runner`'s own C0841 narrowing — no
    /// `App`/bare-node union to be a subclass alternative *of*).
    /// `app_name` defaults to the literal `"InMemoryRunner"`, matching
    /// the source's own default (there is no `App` here to make that
    /// default conditional on). `credential_service` stays unset,
    /// matching the source exactly. To use a different `app_name`, or
    /// to register plugins/a custom `plugin_close_timeout`, call
    /// `Runner::new`/`.with_artifact_service`/`.with_memory_service`
    /// directly instead — every argument this constructor would forward
    /// is already reachable through `Runner`'s existing builder methods.
    pub fn in_memory(agent: BaseAgent) -> Self {
        Self::new(
            "InMemoryRunner",
            agent,
            Arc::new(adk_agents::services::InMemorySessionService::new()),
        )
        .with_artifact_service(Arc::new(
            adk_agents::in_memory_artifact_service::InMemoryArtifactService::new(),
        ))
        .with_memory_service(Arc::new(
            adk_agents::in_memory_memory_service::InMemoryMemoryService::new(),
        ))
    }

    /// C0518-in-spirit for `Runner` itself (not the auth `CredentialManager`
    /// row of the same shape): registers a plugin, mirroring the source's
    /// `Runner.__init__(plugins=[...])`/`PluginManager.register_plugin`.
    /// Errors on a duplicate plugin name, exactly as the source raises
    /// `ValueError` for the same case.
    pub fn with_plugin(mut self, plugin: Arc<dyn BasePlugin>) -> Result<Self, PluginManagerError> {
        self.plugin_manager.register_plugin(plugin)?;
        Ok(self)
    }

    pub fn with_artifact_service(
        mut self,
        service: Arc<dyn ArtifactService + Send + Sync>,
    ) -> Self {
        self.artifact_service = Some(service);
        self
    }

    pub fn with_memory_service(mut self, service: Arc<dyn MemoryService + Send + Sync>) -> Self {
        self.memory_service = Some(service);
        self
    }

    pub fn with_credential_service(
        mut self,
        service: Arc<dyn CredentialService + Send + Sync>,
    ) -> Self {
        self.credential_service = Some(service);
        self
    }

    pub fn with_plugin_close_timeout(mut self, seconds: f64) -> Self {
        self.plugin_close_timeout = seconds;
        self
    }

    pub fn with_auto_create_session(mut self, auto_create_session: bool) -> Self {
        self.auto_create_session = auto_create_session;
        self
    }

    pub fn app_name(&self) -> &str {
        &self.app_name
    }

    pub fn agent(&self) -> &BaseAgent {
        &self.agent
    }

    /// C0844: seconds `close()` will eventually bound each toolset's
    /// `close()` call by, once toolset collection (C0922/C0923) is wired.
    pub fn plugin_close_timeout(&self) -> f64 {
        self.plugin_close_timeout
    }

    /// C0846: set only via [`Runner::from_app`]; always `None` after
    /// [`Runner::new`]/[`Runner::in_memory`].
    pub fn context_cache_config(&self) -> Option<&ContextCacheConfig> {
        self.context_cache_config.as_ref()
    }

    /// C0846: set only via [`Runner::from_app`]; always `None` after
    /// [`Runner::new`]/[`Runner::in_memory`].
    pub fn resumability_config(&self) -> Option<&ResumabilityConfig> {
        self.resumability_config.as_ref()
    }

    /// C0873 (narrowed — no `GetSessionConfig`, see `adk-agents::services`'
    /// own disclosed scope cut): gets the named session, or creates it
    /// (empty of history) if missing and [`Runner::with_auto_create_session`]
    /// is set; otherwise reports it as not found.
    async fn get_or_create_session(
        &self,
        user_id: &str,
        session_id: &str,
    ) -> Result<Session, RunnerError> {
        if let Some(session) = self
            .session_service
            .get_session(&self.app_name, user_id, session_id)
            .await
        {
            return Ok(session);
        }
        if !self.auto_create_session {
            return Err(RunnerError::SessionNotFound(SessionNotFoundError::new(
                format!("Session not found: {session_id}"),
            )));
        }
        self.session_service
            .create_session(&self.app_name, user_id, None, Some(session_id.to_string()))
            .await
            .map_err(RunnerError::AlreadyExists)
    }

    /// C0884-C0886 (narrowed "legacy" path, see the module doc): runs one
    /// turn of the wrapped agent against `new_message` with a default
    /// [`RunConfig`], returning the events it produced (the appended
    /// user-message event itself isn't included — the caller already
    /// knows what it sent, matching the source's own yielded-events
    /// contract). See [`Runner::run_async_with_config`] for the
    /// `RunConfig`-accepting form (custom metadata, the deprecated
    /// blob-saving flag).
    pub async fn run_async(
        &self,
        user_id: &str,
        session_id: &str,
        new_message: Content,
    ) -> Result<Vec<Event>, RunnerError> {
        self.run_async_with_config(user_id, session_id, new_message, RunConfig::default())
            .await
    }

    /// C0884-C0886/C0895-C0899: [`Runner::run_async`], but also accepting
    /// a per-call [`RunConfig`] — mirrors the source's optional
    /// `run_config` parameter (`RunConfig()` default). See the module doc
    /// for the full `_exec_with_plugin`/`_handle_new_message` mapping and
    /// its disclosed narrowings.
    pub async fn run_async_with_config(
        &self,
        user_id: &str,
        session_id: &str,
        mut new_message: Content,
        run_config: RunConfig,
    ) -> Result<Vec<Event>, RunnerError> {
        if new_message.parts.is_empty() {
            return Err(RunnerError::EmptyNewMessage);
        }
        if new_message
            .parts
            .iter()
            .any(|part| part.function_call.is_some())
        {
            return Err(RunnerError::NewMessageContainsFunctionCall);
        }

        let mut session = self.get_or_create_session(user_id, session_id).await?;

        let invocation_id = new_invocation_context_id();
        let mut invocation_context =
            InvocationContextBuilder::new(invocation_id.clone(), session.clone())
                .agent(self.agent.clone())
                .plugin_manager(self.plugin_manager.clone())
                .run_config(run_config.clone())
                .build();
        invocation_context.session_service = self.session_service.clone();
        invocation_context.artifact_service = self.artifact_service.clone();
        invocation_context.memory_service = self.memory_service.clone();
        invocation_context.credential_service = self.credential_service.clone();

        // C0897/`_handle_new_message`: on_user_message_callback may
        // replace the incoming message before anything else sees it.
        let mut user_message_ctx = Context::new(invocation_context.clone());
        if let Some(modified) = self
            .plugin_manager
            .run_on_user_message_callback(&mut user_message_ctx, &new_message)
            .await
        {
            new_message = modified;
        }
        merge_context_state_into_session(user_message_ctx, &mut invocation_context, &mut session);
        invocation_context.user_content = rusty_serde::json::to_value(&new_message).ok();

        // C0898/C0899: deprecated blob-saving path.
        self.maybe_save_input_blobs_as_artifacts(
            &invocation_id,
            user_id,
            session_id,
            &run_config,
            &mut new_message,
        );

        let mut user_event = Event::new(invocation_id.clone(), "user", NodeInfo::new("root"));
        user_event.content = Some(new_message);
        self.session_service
            .append_event(&mut session, user_event)
            .await;
        invocation_context.session = session.clone();

        // C0897 step 1: before_run_callback may short-circuit the whole
        // turn into a single early-exit event.
        let mut before_run_ctx = Context::new(invocation_context.clone());
        let early_exit = self
            .plugin_manager
            .run_before_run_callback(&mut before_run_ctx)
            .await;
        merge_context_state_into_session(before_run_ctx, &mut invocation_context, &mut session);

        let output_events = if let Some(content) = early_exit {
            let mut event = Event::new(invocation_id.clone(), "model", NodeInfo::new("root"));
            event.content = Some(content);
            apply_run_config_custom_metadata(&mut event, &run_config);
            if should_append_event(&event, false) {
                self.session_service
                    .append_event(&mut session, event.clone())
                    .await;
            }
            vec![event]
        } else {
            // C0897 step 2-3: drive the agent, then run on_event_callback
            // and persist each produced event before yielding it.
            match self.agent.run_async(&invocation_context).await {
                Ok(raw_events) => {
                    let mut output_events = Vec::with_capacity(raw_events.len());
                    for mut event in raw_events {
                        apply_run_config_custom_metadata(&mut event, &run_config);
                        let mut event_ctx = Context::new(invocation_context.clone());
                        let modified = self
                            .plugin_manager
                            .run_on_event_callback(&mut event_ctx, &event)
                            .await;
                        merge_context_state_into_session(
                            event_ctx,
                            &mut invocation_context,
                            &mut session,
                        );
                        let output_event = merge_output_event(event, modified);
                        if should_append_event(&output_event, false) {
                            self.session_service
                                .append_event(&mut session, output_event.clone())
                                .await;
                        }
                        output_events.push(output_event);
                    }
                    output_events
                }
                Err(error) => {
                    // C0357: notification-only, then re-raise.
                    let mut error_ctx = Context::new(invocation_context.clone());
                    self.plugin_manager
                        .run_on_run_error_callback(&mut error_ctx, &error)
                        .await;
                    return Err(RunnerError::AgentRun(error.to_string()));
                }
            }
        };

        // C0897 step 4: after_run_callback runs only on success.
        let mut after_run_ctx = Context::new(invocation_context.clone());
        self.plugin_manager
            .run_after_run_callback(&mut after_run_ctx)
            .await;
        merge_context_state_into_session(after_run_ctx, &mut invocation_context, &mut session);

        Ok(output_events)
    }

    /// C0898/C0899 (deprecated, see the module doc's disclosed narrowing):
    /// saves each `inline_data` part of `new_message` as an artifact and
    /// replaces it in place with a placeholder text part, only when an
    /// `ArtifactService` is configured and `run_config.save_input_blobs_as_artifacts`
    /// reads `true`.
    fn maybe_save_input_blobs_as_artifacts(
        &self,
        invocation_id: &str,
        user_id: &str,
        session_id: &str,
        run_config: &RunConfig,
        new_message: &mut Content,
    ) {
        if !run_config.save_input_blobs_as_artifacts {
            return;
        }
        let Some(artifact_service) = &self.artifact_service else {
            return;
        };
        for (index, part) in new_message.parts.iter_mut().enumerate() {
            if part.inline_data.is_none() {
                continue;
            }
            let file_name = format!("artifact_{invocation_id}_{index}");
            artifact_service.save_artifact(
                &self.app_name,
                user_id,
                session_id,
                &file_name,
                rusty_serde::json::to_value(&*part).unwrap_or(Value::Null),
                None,
            );
            *part = Part {
                text: Some(format!(
                    "Uploaded file: {file_name}. It is saved into artifacts"
                )),
                ..Default::default()
            };
        }
    }

    /// C0924 (partial, see the module doc): closes what actually exists
    /// today — `session_service.flush()` and every registered plugin.
    /// Toolset collection is deferred (see the module doc).
    pub async fn close(&self) {
        self.session_service.flush().await;
        self.plugin_manager.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::services::InMemorySessionService;
    use adk_genai::content::Part;

    struct EchoBehavior;

    impl adk_agents::base_agent::AgentBehavior for EchoBehavior {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let mut event = Event::new(
                    ctx.invocation_id.clone(),
                    "echo_agent",
                    NodeInfo::new("root"),
                );
                event.content = Some(Content::user_text("echo"));
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    fn runner(auto_create_session: bool) -> Runner {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        Runner::new("app", agent, Arc::new(InMemorySessionService::new()))
            .with_auto_create_session(auto_create_session)
    }

    #[rusty_tokio::test]
    async fn run_async_reports_a_missing_session_when_auto_create_is_off() {
        let runner = runner(false);
        let err = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, RunnerError::SessionNotFound(_)));
    }

    #[rusty_tokio::test]
    async fn run_async_auto_creates_a_missing_session_and_drives_the_agent() {
        let runner = runner(true);
        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");
    }

    #[rusty_tokio::test]
    async fn run_async_persists_both_the_user_message_and_the_agent_events() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        assert_eq!(session.events.len(), 2);
        assert_eq!(session.events[0].author, "user");
        assert_eq!(session.events[1].author, "echo_agent");
    }

    #[rusty_tokio::test]
    async fn run_async_rejects_a_new_message_containing_a_function_call() {
        let runner = runner(true);
        let message = Content::new(
            "user",
            vec![Part {
                function_call: Some(adk_genai::content::FunctionCall {
                    name: Some("do_thing".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        );
        let err = runner.run_async("user", "s1", message).await.unwrap_err();
        assert!(matches!(err, RunnerError::NewMessageContainsFunctionCall));
    }

    #[rusty_tokio::test]
    async fn close_flushes_the_session_service() {
        let runner = runner(true);
        runner.close().await;
    }

    #[test]
    fn in_memory_defaults_the_app_name_to_the_literal_in_memory_runner() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let runner = Runner::in_memory(agent);
        assert_eq!(runner.app_name(), "InMemoryRunner");
    }

    #[rusty_tokio::test]
    async fn in_memory_drives_a_turn_using_its_pre_wired_in_memory_services() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        // `Runner::in_memory` doesn't set `auto_create_session` (matching
        // the source's own constructor, which doesn't either) — so the
        // session must be created first, same as any other `Runner`.
        let runner = Runner::in_memory(agent).with_auto_create_session(true);

        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");
    }

    struct StubPlugin {
        stub_name: &'static str,
    }

    impl adk_agents::services::BasePlugin for StubPlugin {
        fn name(&self) -> &str {
            self.stub_name
        }
    }

    #[rusty_tokio::test]
    async fn from_app_folds_the_apps_plugins_into_the_registered_set() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let plugin = Arc::new(RecordingPlugin::new("p1"));
        let app = App::new("my-app", agent)
            .unwrap()
            .with_plugin(plugin.clone());

        let runner = Runner::from_app(app, None, Arc::new(InMemorySessionService::new()))
            .unwrap()
            .with_auto_create_session(true);
        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        assert!(plugin.calls.lock().unwrap().contains(&"on_user_message"));
    }

    #[test]
    fn from_app_surfaces_a_duplicate_plugin_name_error() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let app = App::new("my-app", agent)
            .unwrap()
            .with_plugin(Arc::new(StubPlugin { stub_name: "p1" }))
            .with_plugin(Arc::new(StubPlugin { stub_name: "p1" }));

        match Runner::from_app(app, None, Arc::new(InMemorySessionService::new())) {
            Err(PluginManagerError::DuplicateName(name)) => assert_eq!(name, "p1"),
            Ok(_) => panic!("expected a DuplicateName error"),
        }
    }

    #[test]
    fn from_app_derives_app_name_and_configs_from_the_app() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let app = App::new("my-app", agent)
            .unwrap()
            .with_context_cache_config(ContextCacheConfig::default())
            .with_resumability_config(ResumabilityConfig::new(true));

        let runner = Runner::from_app(app, None, Arc::new(InMemorySessionService::new())).unwrap();

        assert_eq!(runner.app_name(), "my-app");
        assert_eq!(runner.agent().name(), "echo_agent");
        assert!(runner.context_cache_config().is_some());
        assert!(runner.resumability_config().unwrap().is_resumable);
    }

    #[test]
    fn from_app_honors_an_app_name_override() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let app = App::new("my-app", agent).unwrap();

        let runner = Runner::from_app(
            app,
            Some("override-name".to_string()),
            Arc::new(InMemorySessionService::new()),
        )
        .unwrap();

        assert_eq!(runner.app_name(), "override-name");
    }

    #[test]
    fn from_app_leaves_configs_unset_when_the_app_has_none() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let app = App::new("my-app", agent).unwrap();

        let runner = Runner::from_app(app, None, Arc::new(InMemorySessionService::new())).unwrap();

        assert!(runner.context_cache_config().is_none());
        assert!(runner.resumability_config().is_none());
    }

    #[rusty_tokio::test]
    async fn from_app_drives_a_turn_like_any_other_runner() {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        let app = App::new("my-app", agent).unwrap();
        let runner = Runner::from_app(app, None, Arc::new(InMemorySessionService::new()))
            .unwrap()
            .with_auto_create_session(true);

        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");
    }

    #[rusty_tokio::test]
    async fn run_async_reuses_an_existing_session() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("first"))
            .await
            .unwrap();
        runner
            .run_async("user", "s1", Content::user_text("second"))
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        // 2 user events + 2 echoed agent events across both turns.
        assert_eq!(session.events.len(), 4);
    }

    #[rusty_tokio::test]
    async fn run_async_rejects_a_new_message_with_no_parts() {
        let runner = runner(true);
        let err = runner
            .run_async("user", "s1", Content::new("user", vec![]))
            .await
            .unwrap_err();
        assert!(matches!(err, RunnerError::EmptyNewMessage));
    }

    struct FailingBehavior;

    impl adk_agents::base_agent::AgentBehavior for FailingBehavior {
        fn run_async_impl<'a>(
            &'a self,
            _ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Err("boom".into()) })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    /// Records which run-level hooks fired, in order, and can be
    /// configured to rewrite the user message, short-circuit via
    /// `before_run_callback`, or rewrite an event via `on_event_callback`.
    struct RecordingPlugin {
        name: String,
        rewrite_user_message: bool,
        early_exit: bool,
        rewrite_event: bool,
        calls: std::sync::Mutex<Vec<&'static str>>,
    }

    impl RecordingPlugin {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                rewrite_user_message: false,
                early_exit: false,
                rewrite_event: false,
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl adk_agents::services::BasePlugin for RecordingPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn on_user_message_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
            _user_message: &'a Content,
        ) -> adk_agents::services::BoxFuture<'a, Option<Content>> {
            self.calls.lock().unwrap().push("on_user_message");
            Box::pin(async move {
                if self.rewrite_user_message {
                    Some(Content::user_text("rewritten"))
                } else {
                    None
                }
            })
        }

        fn before_run_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
        ) -> adk_agents::services::BoxFuture<'a, Option<Content>> {
            self.calls.lock().unwrap().push("before_run");
            Box::pin(async move {
                if self.early_exit {
                    Some(Content::user_text("short-circuited"))
                } else {
                    None
                }
            })
        }

        fn on_event_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
            event: &'a Event,
        ) -> adk_agents::services::BoxFuture<'a, Option<Event>> {
            self.calls.lock().unwrap().push("on_event");
            let mut replacement = event.clone();
            Box::pin(async move {
                if self.rewrite_event {
                    replacement.content = Some(Content::user_text("rewritten-event"));
                    replacement.id = "should-be-ignored".to_string();
                    Some(replacement)
                } else {
                    None
                }
            })
        }

        fn after_run_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
        ) -> adk_agents::services::BoxFuture<'a, ()> {
            self.calls.lock().unwrap().push("after_run");
            Box::pin(async {})
        }

        fn on_run_error_callback<'a>(
            &'a self,
            _invocation_context: &'a mut Context,
            _error: &'a adk_agents::base_agent::AgentRunError,
        ) -> adk_agents::services::BoxFuture<'a, ()> {
            self.calls.lock().unwrap().push("on_run_error");
            Box::pin(async {})
        }
    }

    #[rusty_tokio::test]
    async fn run_async_runs_the_full_run_level_plugin_sequence_on_success() {
        let plugin = Arc::new(RecordingPlugin::new("recorder"));
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_plugin(plugin.clone())
        .unwrap();

        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let calls = plugin.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec!["on_user_message", "before_run", "on_event", "after_run"]
        );
    }

    #[rusty_tokio::test]
    async fn run_async_honors_a_plugin_rewriting_the_user_message() {
        let mut plugin = RecordingPlugin::new("rewriter");
        plugin.rewrite_user_message = true;
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_plugin(Arc::new(plugin))
        .unwrap();

        runner
            .run_async("user", "s1", Content::user_text("original"))
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        assert_eq!(
            session.events[0].content.as_ref().unwrap().parts[0]
                .text
                .as_deref(),
            Some("rewritten")
        );
    }

    #[rusty_tokio::test]
    async fn run_async_short_circuits_on_a_before_run_early_exit() {
        let mut plugin = RecordingPlugin::new("early-exit");
        plugin.early_exit = true;
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_plugin(Arc::new(plugin))
        .unwrap();

        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "model");
        assert_eq!(
            events[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("short-circuited")
        );

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        // User event + the early-exit event, but never the (skipped) agent run.
        assert_eq!(session.events.len(), 2);
        assert_eq!(session.events[1].author, "model");
    }

    #[rusty_tokio::test]
    async fn run_async_merges_a_plugin_rewritten_event_but_keeps_the_original_id() {
        let mut plugin = RecordingPlugin::new("event-rewriter");
        plugin.rewrite_event = true;
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_plugin(Arc::new(plugin))
        .unwrap();

        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_ne!(events[0].id, "should-be-ignored");
        assert_eq!(
            events[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("rewritten-event")
        );
    }

    #[rusty_tokio::test]
    async fn run_async_notifies_on_run_error_and_still_propagates_the_failure() {
        let plugin = Arc::new(RecordingPlugin::new("error-recorder"));
        let runner = Runner::new(
            "app",
            BaseAgent::new("failing_agent", FailingBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_plugin(plugin.clone())
        .unwrap();

        let err = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, RunnerError::AgentRun(_)));

        let calls = plugin.calls.lock().unwrap().clone();
        assert!(calls.contains(&"on_run_error"));
        assert!(!calls.contains(&"after_run"));
    }

    #[rusty_tokio::test]
    async fn with_plugin_errors_on_a_duplicate_name() {
        let base = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_plugin(Arc::new(RecordingPlugin::new("dup")))
        .unwrap();

        let err = base.with_plugin(Arc::new(RecordingPlugin::new("dup")));
        assert!(err.is_err());
    }

    #[rusty_tokio::test]
    async fn close_closes_every_registered_plugin() {
        struct ClosingPlugin(Arc<std::sync::Mutex<bool>>);

        impl adk_agents::services::BasePlugin for ClosingPlugin {
            fn name(&self) -> &str {
                "closing"
            }

            fn close<'a>(&'a self) -> adk_agents::services::BoxFuture<'a, ()> {
                Box::pin(async move {
                    *self.0.lock().unwrap() = true;
                })
            }
        }

        let closed = Arc::new(std::sync::Mutex::new(false));
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_plugin(Arc::new(ClosingPlugin(closed.clone())))
        .unwrap();

        runner.close().await;
        assert!(*closed.lock().unwrap());
    }

    #[rusty_tokio::test]
    async fn run_async_with_config_saves_inline_data_parts_as_artifacts_when_deprecated_flag_is_set(
    ) {
        let artifact_service =
            Arc::new(adk_agents::in_memory_artifact_service::InMemoryArtifactService::new());
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_artifact_service(artifact_service.clone());

        let message = Content::new(
            "user",
            vec![Part {
                inline_data: Some(adk_genai::content::MediaBlobStub {
                    mime_type: Some("text/plain".to_string()),
                    rest: None,
                }),
                ..Default::default()
            }],
        );

        let run_config = RunConfig {
            save_input_blobs_as_artifacts: true,
            ..Default::default()
        };

        runner
            .run_async_with_config("user", "s1", message, run_config)
            .await
            .unwrap();

        let keys = artifact_service.list_artifact_keys("app", "user", "s1");
        assert_eq!(keys.len(), 1);

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        let saved_part = &session.events[0].content.as_ref().unwrap().parts[0];
        assert!(saved_part.inline_data.is_none());
        assert!(saved_part
            .text
            .as_deref()
            .unwrap()
            .starts_with("Uploaded file:"));
    }

    #[rusty_tokio::test]
    async fn run_async_bridges_on_user_message_state_into_before_agent_callback() {
        // End-to-end proof that `merge_context_state_into_session` closes
        // the visibility gap: `SaveFilesAsArtifactsPlugin` stashes a
        // pending delta from `on_user_message_callback` and only flushes
        // it into `artifact_delta` from `before_agent_callback` — a
        // different, later `Context`. Without the bridge, the stash would
        // never be visible there.
        let artifact_service =
            Arc::new(adk_agents::in_memory_artifact_service::InMemoryArtifactService::new());
        let plugin =
            Arc::new(adk_agents::save_files_as_artifacts_plugin::SaveFilesAsArtifactsPlugin::new());
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_artifact_service(artifact_service)
        .with_plugin(plugin)
        .unwrap();

        let message = Content::new(
            "user",
            vec![Part {
                inline_data: Some(adk_genai::content::MediaBlobStub {
                    mime_type: Some("text/plain".to_string()),
                    rest: Some(Value::Map(vec![
                        (
                            "displayName".to_string(),
                            Value::String("f.txt".to_string()),
                        ),
                        ("data".to_string(), Value::String("aGVsbG8=".to_string())),
                    ])),
                }),
                ..Default::default()
            }],
        );

        let events = runner.run_async("user", "s1", message).await.unwrap();

        assert!(events
            .iter()
            .any(|e| e.actions.artifact_delta.get("f.txt") == Some(&0)));

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        assert!(session.events[0].content.as_ref().unwrap().parts[0]
            .inline_data
            .is_none());
    }

    #[rusty_tokio::test]
    async fn run_async_with_config_leaves_inline_data_untouched_when_the_deprecated_flag_is_unset()
    {
        let artifact_service =
            Arc::new(adk_agents::in_memory_artifact_service::InMemoryArtifactService::new());
        let runner = Runner::new(
            "app",
            BaseAgent::new("echo_agent", EchoBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true)
        .with_artifact_service(artifact_service.clone());

        let message = Content::new(
            "user",
            vec![Part {
                inline_data: Some(adk_genai::content::MediaBlobStub {
                    mime_type: Some("text/plain".to_string()),
                    rest: None,
                }),
                ..Default::default()
            }],
        );

        runner
            .run_async_with_config("user", "s1", message, RunConfig::default())
            .await
            .unwrap();

        assert!(artifact_service
            .list_artifact_keys("app", "user", "s1")
            .is_empty());
    }
}
