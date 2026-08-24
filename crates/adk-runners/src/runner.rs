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
//! - Live mode (`run_live`) needs its own supporting infrastructure (live
//!   request queue wiring into `InvocationContext`, etc.) beyond this
//!   batch's scope. `rewind_async` (C0891-C0894) is now built too — see
//!   [`crate::rewind`]'s own module doc for its two delta helpers.
//!   `run_debug` (C0911-C0913) is now built too — see
//!   [`Runner::run_debug`]'s own doc.
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
//! **`Runner::run_debug`** (C0911-C0913): a debugging/experimentation
//! convenience — see [`Runner::run_debug`]/[`Runner::run_debug_with_config`]'s
//! own docs for the unconditional-session-creation (C0912) and
//! flat-event-collection (C0913) semantics, and their disclosed
//! narrowings (no logging framework adopted, matching C0851-C0854 below;
//! C0914's `run_config.get_session_config` forwarding is N/A — `RunConfig`
//! still has no `get_session_config` field to forward, even though
//! `GetSessionConfig` itself is now real, C0207 — see C0875).
//!
//! **Invocation-context factory** (C0918/C0919): [`InvocationContextBuilder`]
//! (`adk-agents`) already plays the role of the source's
//! `_create_invocation_context`/`_new_invocation_context` — a plain
//! builder rather than an overridable factory method, since this port has
//! no subclassing to override (C0918: `_create_invocation_context`'s only
//! purpose is being an override point). [`Runner::run_async_with_config`]
//! now also patches `context_cache_config`/`resumability_config`/
//! `events_compaction_config` onto the built context (previously missing
//! — these lived on `Runner` but never reached the `InvocationContext`
//! the agent/callbacks actually saw). **Disclosed narrowing**: the
//! source's `support_cfc` (Compositional Function Calling) branch —
//! validating the resolved model name and force-installing a
//! `BuiltInCodeExecutor` on the agent — has nothing to port onto yet:
//! `LlmAgent.code_executor` is still an opaque `Value` placeholder
//! (C0088), the same architecture-investment blocker as C0092/C0429.
//!
//! **`Runner::run`** (C0877-C0880): a synchronous wrapper around
//! [`Runner::run_async_with_config`] — see its own doc for the
//! thread+runtime bridging shape and its disclosed narrowings.
//!
//! **Not ported this batch**: `_find_agent_to_run` (C0907-C0910, picks up
//! a resumed multi-turn conversation's last-active agent — needs
//! resumability, always false here); `_resolve_invocation_id` (C0855,
//! same reason); the legacy
//! resumable-path context-setup helpers (`_setup_context_for_new_invocation`/
//! `_setup_context_for_resumed_invocation`/`_find_user_message_for_invocation`,
//! C0915-C0917, entangled with resumability wiring `Runner` doesn't have
//! yet); agent-origin inference and its warnings (C0851-C0854) — Rust
//! has no runtime module-path reflection to inspect, and no logging
//! framework is adopted to warn through even if it did.

use std::sync::Arc;

use adk_agents::app::App;
use adk_agents::app_configs::{EventsCompactionConfig, ResumabilityConfig};
use adk_agents::base_agent::BaseAgent;
use adk_agents::context::Context;
use adk_agents::context_cache_config::ContextCacheConfig;
use adk_agents::invocation_context::InvocationContextBuilder;
use adk_agents::run_config::RunConfig;
use adk_agents::services::{
    new_invocation_context_id, ArtifactService, BasePlugin, BoxFuture, CredentialService,
    MemoryService, PluginManager, PluginManagerError, SessionService,
};
use adk_agents::session::Session;
use adk_errors::already_exists::AlreadyExistsError;
use adk_errors::session_not_found::SessionNotFoundError;
use adk_events::debug_output::print_event;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_genai::content::{Content, FunctionResponse, Part};
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
    #[error("Invocation ID not found: {0}")]
    InvocationNotFound(String),
}

/// C0911: `run_debug`'s `user_messages: str | list[str]` parameter — a
/// bare string normalizes to a one-element list, matching the source's
/// `isinstance(user_messages, str)` check.
pub enum DebugMessages {
    Single(String),
    Many(Vec<String>),
}

impl DebugMessages {
    fn into_vec(self) -> Vec<String> {
        match self {
            DebugMessages::Single(message) => vec![message],
            DebugMessages::Many(messages) => messages,
        }
    }
}

impl From<&str> for DebugMessages {
    fn from(message: &str) -> Self {
        DebugMessages::Single(message.to_string())
    }
}

impl From<String> for DebugMessages {
    fn from(message: String) -> Self {
        DebugMessages::Single(message)
    }
}

impl From<Vec<String>> for DebugMessages {
    fn from(messages: Vec<String>) -> Self {
        DebugMessages::Many(messages)
    }
}

impl From<Vec<&str>> for DebugMessages {
    fn from(messages: Vec<&str>) -> Self {
        DebugMessages::Many(messages.into_iter().map(str::to_string).collect())
    }
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

/// C0835: `_get_function_responses_from_content` — extracts every
/// `FunctionResponse` from `content`'s parts; `[]` when `content` is
/// `None` or has no parts. Built ahead of its own caller
/// (`_resolve_invocation_id`, C0855) — needs resumability wiring
/// `Runner` doesn't have yet, the same "widen/build once a real
/// consumer needs it" precedent used elsewhere in this port.
pub fn get_function_responses_from_content(content: Option<&Content>) -> Vec<FunctionResponse> {
    content
        .map(|content| {
            content
                .get_function_responses()
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default()
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

/// C0871/C0872: the extension point for post-invocation compaction.
/// This port's crate layering (`adk-tools`/`adk-flows` both depend on
/// `adk-runners`, not the reverse — the same direction
/// `ForwardingArtifactService`, C0489, already relies on) means the
/// real decision logic
/// (`adk_flows::apps_compaction::run_compaction_for_sliding_window`,
/// C0293) can't be called directly from this crate. `Runner` instead
/// accepts one via this trait-object extension point, matching the
/// "overridable behavior → injected trait object" pattern this crate
/// already uses for `ArtifactService`/`SessionService`; a caller that
/// can see both crates (`adk-flows`, which already depends on
/// `adk-runners`) wires the real implementation in via
/// [`Runner::with_compaction_trigger`]. `Runner::from_app` alone
/// (without that wiring) leaves compaction configured but inert —
/// disclosed on [`Runner::events_compaction_config`].
pub trait CompactionTrigger: Send + Sync {
    fn run<'a>(
        &'a self,
        config: &'a EventsCompactionConfig,
        agent: &'a BaseAgent,
        session_events: &'a [Event],
        skip_token_compaction: bool,
    ) -> BoxFuture<'a, Option<Event>>;
}

/// C0840-C0845 (narrowed, see the module doc): the core execution engine.
/// Wraps exactly one [`BaseAgent`] — no `App`/bare-node union, since
/// neither exists in this port. `Clone` (every field is a cheap
/// `Arc`/value clone) exists so [`Runner::run`] (C0877-C0880) can move an
/// owned copy onto its background thread.
#[derive(Clone)]
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
    /// in this port (resumable-run support isn't built).
    context_cache_config: Option<ContextCacheConfig>,
    /// C0846: same sourcing rule as `context_cache_config` above. Not yet
    /// read anywhere else in this port (resumable-run support isn't
    /// built).
    resumability_config: Option<ResumabilityConfig>,
    /// C0846: derived from an [`App`] via [`Runner::from_app`], same
    /// sourcing rule as `context_cache_config`. Configuring this alone
    /// doesn't make compaction happen — see [`CompactionTrigger`]'s own
    /// doc for why the real decision logic must be injected separately
    /// via [`Runner::with_compaction_trigger`].
    events_compaction_config: Option<EventsCompactionConfig>,
    /// C0871/C0872: the injected real compaction decision logic — see
    /// [`CompactionTrigger`]'s own doc. `None` (the default after both
    /// [`Runner::new`] and [`Runner::from_app`]) means post-invocation
    /// compaction never runs, even if `events_compaction_config` is set.
    compaction_trigger: Option<Arc<dyn CompactionTrigger>>,
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
            events_compaction_config: None,
            compaction_trigger: None,
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
        runner.events_compaction_config = app.events_compaction_config;
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

    /// C0846: set only via [`Runner::from_app`]; always `None` after
    /// [`Runner::new`]/[`Runner::in_memory`]. Configuring this alone
    /// doesn't make compaction happen — see [`CompactionTrigger`]'s doc.
    pub fn events_compaction_config(&self) -> Option<&EventsCompactionConfig> {
        self.events_compaction_config.as_ref()
    }

    /// C0871/C0872: injects the real post-invocation compaction decision
    /// logic — see [`CompactionTrigger`]'s own doc for why this can't be
    /// wired automatically from this crate.
    pub fn with_compaction_trigger(mut self, trigger: Arc<dyn CompactionTrigger>) -> Self {
        self.compaction_trigger = Some(trigger);
        self
    }

    /// C0873 (narrowed — no `GetSessionConfig` threaded through yet;
    /// `GetSessionConfig`/`SessionService::get_session_with_config` are
    /// now real, C0207, but `RunConfig`/`Runner` don't carry one to
    /// forward, see C0875): gets the named session, or creates it
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
        self.run_async_with_config(user_id, session_id, new_message, None, RunConfig::default())
            .await
    }

    /// C0884-C0886/C0895-C0899: [`Runner::run_async`], but also accepting
    /// an optional `state_delta` (C0880: applied onto the appended user
    /// event's `actions.state_delta`, mirroring `_append_new_message_to_session`
    /// — only set when non-empty, matching the source's own `if
    /// state_delta:` truthiness check) and a per-call [`RunConfig`] —
    /// mirrors the source's optional `run_config` parameter (`RunConfig()`
    /// default). See the module doc for the full
    /// `_exec_with_plugin`/`_handle_new_message` mapping and its disclosed
    /// narrowings.
    pub async fn run_async_with_config(
        &self,
        user_id: &str,
        session_id: &str,
        mut new_message: Content,
        state_delta: Option<std::collections::HashMap<String, Value>>,
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
        // C0918/C0919: `_new_invocation_context` also wires
        // `context_cache_config`/`events_compaction_config`/
        // `resumability_config` onto the context it assembles —
        // previously missing here, so the agent/callbacks never saw
        // them even though `Runner` already held all three.
        invocation_context.context_cache_config = self.context_cache_config.clone();
        invocation_context.resumability_config = self.resumability_config;
        invocation_context.events_compaction_config = self.events_compaction_config.clone();

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
        if let Some(state_delta) = state_delta.filter(|delta| !delta.is_empty()) {
            user_event.actions.state_delta = state_delta;
        }
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

        // C0871/C0872: best-effort post-invocation compaction, run only
        // after all events are yielded from the agent. See
        // `CompactionTrigger`'s own doc for why the real decision logic
        // is injected rather than called directly from this crate.
        if let (Some(config), Some(trigger)) =
            (&self.events_compaction_config, &self.compaction_trigger)
        {
            if let Some(compaction_event) = trigger
                .run(
                    config,
                    &self.agent,
                    &session.events,
                    invocation_context.token_compaction_checked,
                )
                .await
            {
                self.session_service
                    .append_event(&mut session, compaction_event)
                    .await;
            }
        }

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

    /// C0891/C0894: rewinds the session to before the given invocation
    /// — gets-or-creates the session (honoring `auto_create_session`,
    /// C0894), linear-scans for the first event matching
    /// `rewind_before_invocation_id`, computes reversing state/artifact
    /// deltas ([`crate::rewind::compute_state_delta_for_rewind`]/
    /// [`crate::rewind::compute_artifact_delta_for_rewind`]), and
    /// appends a single new user-authored event carrying them. Rewind
    /// is a forward-only append of a reversing delta event, never a
    /// destructive truncation of `session.events` —
    /// `adk_events::rewind::apply_rewinds` (already DONE) interprets
    /// that delta downstream.
    pub async fn rewind_async(
        &self,
        user_id: &str,
        session_id: &str,
        rewind_before_invocation_id: &str,
    ) -> Result<(), RunnerError> {
        let mut session = self.get_or_create_session(user_id, session_id).await?;

        let rewind_event_index = session
            .events
            .iter()
            .position(|event| event.invocation_id == rewind_before_invocation_id)
            .ok_or_else(|| {
                RunnerError::InvocationNotFound(rewind_before_invocation_id.to_string())
            })?;

        let state_delta =
            crate::rewind::compute_state_delta_for_rewind(&session, rewind_event_index);
        let artifact_delta = crate::rewind::compute_artifact_delta_for_rewind(
            self.artifact_service.as_deref(),
            &self.app_name,
            &session,
            rewind_event_index,
        );

        let mut rewind_event = Event::new(new_invocation_context_id(), "user", NodeInfo::new(""));
        rewind_event.actions.rewind_before_invocation_id =
            Some(rewind_before_invocation_id.to_string());
        rewind_event.actions.state_delta = state_delta;
        rewind_event.actions.artifact_delta = artifact_delta;

        self.session_service
            .append_event(&mut session, rewind_event)
            .await;

        Ok(())
    }

    /// C0911-C0913: a debugging/experimentation convenience for quickly
    /// exercising the wrapped agent without dealing with session
    /// management, content formatting, or event streaming directly.
    /// Defaults `user_id`/`session_id` to the literal `"debug_user_id"`/
    /// `"debug_session_id"` (reusing the same defaults across calls
    /// continues the same conversation — intentional, matching the
    /// source's own documented behavior) and every other keyword
    /// argument to the source's own defaults. See
    /// [`Runner::run_debug_with_config`] for the full-control form.
    pub async fn run_debug(
        &self,
        user_messages: impl Into<DebugMessages>,
    ) -> Result<Vec<Event>, RunnerError> {
        self.run_debug_with_config(
            user_messages,
            "debug_user_id",
            "debug_session_id",
            RunConfig::default(),
            false,
            false,
        )
        .await
    }

    /// [`Runner::run_debug`], accepting the full set of keyword
    /// parameters the source exposes.
    ///
    /// **C0912**: session lookup here is *unconditional*
    /// get-or-create — it looks the session up directly and creates it
    /// if missing, regardless of [`Runner::with_auto_create_session`]
    /// (unlike [`Runner::run_async`], which honors that flag and errors
    /// instead of creating one). **Disclosed narrowing**: the source
    /// also logs `"Created new session"`/`"Continue session"` via
    /// Python's `logging` module at this point (unless `quiet`); this
    /// port has no logging framework adopted anywhere (see the module
    /// doc's C0851-C0854 note for the same posture), so those two log
    /// lines have no destination here.
    ///
    /// **C0913**: drives [`Runner::run_async_with_config`] once per
    /// message in `user_messages` (a bare string normalizes to one
    /// message), wrapping each as a user [`Content`] text turn; unless
    /// `quiet`, each produced event is printed via
    /// [`adk_events::debug_output::print_event`] with `verbose` forwarded.
    /// **Disclosed narrowing**: the source also logs `"User > %s"` for
    /// each message before driving it — same no-logging-framework
    /// narrowing as above. Returns the full flat list of events across
    /// *all* messages, not just the last (C0913).
    ///
    /// **Disclosed narrowing (C0914)**: the source forwards
    /// `run_config.get_session_config` into its initial `get_session`
    /// call; this port's [`SessionService::get_session`] takes no config
    /// parameter to forward it to (see `adk-agents::services`' own
    /// disclosed scope cut) — N/A here for the same reason.
    pub async fn run_debug_with_config(
        &self,
        user_messages: impl Into<DebugMessages>,
        user_id: &str,
        session_id: &str,
        run_config: RunConfig,
        quiet: bool,
        verbose: bool,
    ) -> Result<Vec<Event>, RunnerError> {
        let session = match self
            .session_service
            .get_session(&self.app_name, user_id, session_id)
            .await
        {
            Some(session) => session,
            None => self
                .session_service
                .create_session(&self.app_name, user_id, None, Some(session_id.to_string()))
                .await
                .map_err(RunnerError::AlreadyExists)?,
        };

        let mut collected_events = Vec::new();
        for message in user_messages.into().into_vec() {
            let events = self
                .run_async_with_config(
                    user_id,
                    &session.id,
                    Content::user_text(message),
                    None,
                    run_config.clone(),
                )
                .await?;
            for event in events {
                if !quiet {
                    print_event(&event, verbose);
                }
                collected_events.push(event);
            }
        }
        Ok(collected_events)
    }

    /// C0877-C0880: a synchronous wrapper around
    /// [`Runner::run_async_with_config`], documented in the source as a
    /// local-testing/convenience-only entrypoint ("Consider using
    /// `run_async` for production usage"). Spins a dedicated OS thread via
    /// [`adk_platform::thread::create_thread`] (C0005) running its own
    /// [`rusty_tokio::Runtime`], so this can be called safely from inside
    /// an already-running async context without nesting runtimes —
    /// mirroring the source's own reason for running `asyncio.run(...)` on
    /// a background thread rather than the calling one.
    ///
    /// **Disclosed narrowing (C0877/C0879)**: the source is a sync
    /// `Generator[Event, None, None]`, bridging events one at a time
    /// through a blocking `queue.Queue` so a caller can begin consuming
    /// events before the run finishes, and can stop iterating early to
    /// abandon the rest (nothing is raised in that case; the background
    /// thread is left to finish on its own). This port's own
    /// `run_async_with_config` already collapses to a single batched
    /// `Result<Vec<Event>, RunnerError>` rather than a stream — an
    /// already-established narrowing from the source's own async
    /// generator, see the module doc — so there is no incremental
    /// event-at-a-time bridging to reproduce here either; `run` collapses
    /// to a single background computation whose whole result becomes
    /// available at once. "Events produced before a failure are yielded
    /// before the exception is raised" (C0879) has no partial case to
    /// preserve for the same reason: a failure here means zero output
    /// events were produced, matching this crate's own established scope
    /// cut.
    ///
    /// **C0878**: the source distinguishes a plain `Exception` (re-raised
    /// directly on the calling thread) from any other `BaseException`
    /// (wrapped in a `RuntimeError` chained from the original — so a
    /// background-thread cancellation doesn't read as the *calling*
    /// thread's own task being cancelled, and a `SystemExit` doesn't kill
    /// the caller's process). Rust has no such exception hierarchy to
    /// preserve; the structural analog is `JoinHandle::join`'s own
    /// `Result` — an `Err(RunnerError::AgentRun(..))` returned normally by
    /// the background computation surfaces as-is, while a *panic* on the
    /// background thread (the only way this port's own call stack can
    /// terminate abnormally, with no cancellation/`SystemExit` concept to
    /// preserve) is caught by `join()` and re-wrapped into
    /// `RunnerError::AgentRun` naming the panic payload, rather than
    /// re-panicking the calling thread.
    ///
    /// **C0880**: `state_delta` forwards straight through to
    /// [`Runner::run_async_with_config`].
    pub fn run(
        &self,
        user_id: &str,
        session_id: &str,
        new_message: Content,
        state_delta: Option<std::collections::HashMap<String, Value>>,
        run_config: RunConfig,
    ) -> Result<Vec<Event>, RunnerError> {
        let runner = self.clone();
        let user_id = user_id.to_string();
        let session_id = session_id.to_string();
        let handle = adk_platform::thread::create_thread(move || {
            let runtime = rusty_tokio::Runtime::new()
                .expect("failed to start a background async runtime for Runner::run");
            runtime.block_on(runner.run_async_with_config(
                &user_id,
                &session_id,
                new_message,
                state_delta,
                run_config,
            ))
        });
        handle.join().unwrap_or_else(|panic| {
            Err(RunnerError::AgentRun(format!(
                "Agent run terminated by a panic on the background thread: {}",
                panic_message(&panic)
            )))
        })
    }
}

/// [`Runner::run`]'s panic-message extraction — mirrors the common
/// `Box<dyn Any + Send>` downcast dance for the two payload shapes
/// `panic!`/`.unwrap()`/`.expect()` actually produce.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        message.to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::services::InMemorySessionService;
    use adk_genai::content::Part;
    use std::sync::Mutex;

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

    /// C0918/C0919 test helper: captures the three configs a driven
    /// `InvocationContext` carries, so a test can assert `Runner`'s own
    /// configuration actually reached the context the agent ran with.
    #[derive(Default)]
    struct CapturedConfigs {
        context_cache_config: Option<ContextCacheConfig>,
        resumability_config: Option<ResumabilityConfig>,
        events_compaction_config: Option<EventsCompactionConfig>,
    }

    struct ConfigCapturingBehavior {
        captured: Arc<Mutex<CapturedConfigs>>,
    }

    impl adk_agents::base_agent::AgentBehavior for ConfigCapturingBehavior {
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
                let mut captured = self.captured.lock().unwrap();
                captured.context_cache_config = ctx.context_cache_config.clone();
                captured.resumability_config = ctx.resumability_config;
                captured.events_compaction_config = ctx.events_compaction_config.clone();
                Ok(Vec::new())
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
    async fn run_debug_creates_the_debug_session_even_when_auto_create_session_is_off() {
        // C0912: unconditional get-or-create, bypassing
        // `with_auto_create_session` entirely.
        let runner = runner(false);
        let events = runner.run_debug("hi").await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");

        let session = runner
            .session_service
            .get_session("app", "debug_user_id", "debug_session_id")
            .await
            .unwrap();
        assert_eq!(session.events.len(), 2);
    }

    #[rusty_tokio::test]
    async fn run_debug_reuses_the_same_default_session_across_calls() {
        let runner = runner(false);
        runner.run_debug("first").await.unwrap();
        runner.run_debug("second").await.unwrap();

        let session = runner
            .session_service
            .get_session("app", "debug_user_id", "debug_session_id")
            .await
            .unwrap();
        // 2 user events + 2 agent events across both calls, same session.
        assert_eq!(session.events.len(), 4);
    }

    #[rusty_tokio::test]
    async fn run_debug_normalizes_a_single_string_message_to_one_element() {
        let runner = runner(false);
        let events = runner.run_debug("only message").await.unwrap();
        assert_eq!(events.len(), 1);
    }

    #[rusty_tokio::test]
    async fn run_debug_drives_every_message_and_collects_the_full_flat_event_list() {
        let runner = runner(false);
        let events = runner
            .run_debug(vec!["first", "second", "third"])
            .await
            .unwrap();
        // One echo_agent event per message, in order, across all messages —
        // not just the last (C0913).
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|event| event.author == "echo_agent"));
    }

    #[rusty_tokio::test]
    async fn run_debug_with_config_honors_custom_user_and_session_ids() {
        let runner = runner(false);
        runner
            .run_debug_with_config("hi", "alice", "debug1", RunConfig::default(), false, false)
            .await
            .unwrap();

        assert!(runner
            .session_service
            .get_session("app", "debug_user_id", "debug_session_id")
            .await
            .is_none());
        let session = runner
            .session_service
            .get_session("app", "alice", "debug1")
            .await
            .unwrap();
        assert_eq!(session.events.len(), 2);
    }

    #[rusty_tokio::test]
    async fn run_debug_with_config_quiet_still_returns_events() {
        let runner = runner(false);
        let events = runner
            .run_debug_with_config("hi", "user", "s1", RunConfig::default(), true, false)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    #[rusty_tokio::test]
    async fn run_async_wires_the_runners_configs_onto_the_invocation_context() {
        // C0918/C0919: `context_cache_config`/`resumability_config`/
        // `events_compaction_config` must reach the `InvocationContext`
        // the agent actually runs with, not just live on `Runner` itself.
        let captured = Arc::new(Mutex::new(CapturedConfigs::default()));
        let agent = BaseAgent::new(
            "capturing_agent",
            ConfigCapturingBehavior {
                captured: captured.clone(),
            },
        )
        .unwrap();
        let app = App::new("my-app", agent)
            .unwrap()
            .with_context_cache_config(ContextCacheConfig::default())
            .with_resumability_config(ResumabilityConfig::new(true))
            .with_events_compaction_config(EventsCompactionConfig {
                token_threshold: Some(42),
                ..Default::default()
            });
        let runner = Runner::from_app(app, None, Arc::new(InMemorySessionService::new()))
            .unwrap()
            .with_auto_create_session(true);

        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let captured = captured.lock().unwrap();
        assert!(captured.context_cache_config.is_some());
        assert!(captured.resumability_config.unwrap().is_resumable);
        assert_eq!(
            captured
                .events_compaction_config
                .as_ref()
                .unwrap()
                .token_threshold,
            Some(42)
        );
    }

    #[rusty_tokio::test]
    async fn run_async_leaves_unset_configs_absent_on_the_invocation_context() {
        let captured = Arc::new(Mutex::new(CapturedConfigs::default()));
        let agent = BaseAgent::new(
            "capturing_agent",
            ConfigCapturingBehavior {
                captured: captured.clone(),
            },
        )
        .unwrap();
        let runner = Runner::new("app", agent, Arc::new(InMemorySessionService::new()))
            .with_auto_create_session(true);

        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let captured = captured.lock().unwrap();
        assert!(captured.context_cache_config.is_none());
        assert!(captured.resumability_config.is_none());
        assert!(captured.events_compaction_config.is_none());
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
    fn run_is_a_sync_wrapper_matching_run_async() {
        let runner = runner(true);
        let events = runner
            .run(
                "user",
                "s1",
                Content::user_text("hi"),
                None,
                RunConfig::default(),
            )
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");
    }

    #[test]
    fn run_forwards_state_delta_onto_the_appended_user_event() {
        // C0880: `state_delta` forwards straight through to
        // `run_async_with_config`, ending up on the appended user event's
        // `actions.state_delta`.
        let runner = runner(true);
        let mut state_delta = std::collections::HashMap::new();
        state_delta.insert("count".to_string(), Value::Int(1));

        runner
            .run(
                "user",
                "s1",
                Content::user_text("hi"),
                Some(state_delta),
                RunConfig::default(),
            )
            .unwrap();

        let rt = rusty_tokio::Runtime::new().unwrap();
        let session = rt
            .block_on(runner.session_service.get_session("app", "user", "s1"))
            .unwrap();
        assert_eq!(
            session.events[0].actions.state_delta.get("count"),
            Some(&Value::Int(1))
        );
    }

    #[test]
    fn run_propagates_an_agent_run_error() {
        let runner = Runner::new(
            "app",
            BaseAgent::new("failing_agent", FailingBehavior).unwrap(),
            Arc::new(InMemorySessionService::new()),
        )
        .with_auto_create_session(true);

        let err = runner
            .run(
                "user",
                "s1",
                Content::user_text("hi"),
                None,
                RunConfig::default(),
            )
            .unwrap_err();
        assert!(matches!(err, RunnerError::AgentRun(_)));
    }

    #[rusty_tokio::test]
    async fn run_is_callable_from_within_an_already_running_async_runtime() {
        // C0877's whole reason for existing: a caller already inside an
        // async context (like this test itself) must be able to call the
        // sync `run` without deadlocking or panicking from a nested
        // runtime — `run` spawns its own OS thread with its own runtime
        // rather than trying to `block_on` from here.
        let runner = runner(true);
        let events = runner
            .run(
                "user",
                "s1",
                Content::user_text("hi"),
                None,
                RunConfig::default(),
            )
            .unwrap();
        assert_eq!(events.len(), 1);
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
            .run_async_with_config("user", "s1", message, None, run_config)
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
            .run_async_with_config("user", "s1", message, None, RunConfig::default())
            .await
            .unwrap();

        assert!(artifact_service
            .list_artifact_keys("app", "user", "s1")
            .is_empty());
    }

    #[rusty_tokio::test]
    async fn rewind_async_appends_a_reversing_delta_event() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("first"))
            .await
            .unwrap();
        let target_invocation_id = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap()
            .events[0]
            .invocation_id
            .clone();
        runner
            .run_async("user", "s1", Content::user_text("second"))
            .await
            .unwrap();

        runner
            .rewind_async("user", "s1", &target_invocation_id)
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        let rewind_event = session.events.last().unwrap();
        assert_eq!(rewind_event.author, "user");
        assert_eq!(
            rewind_event.actions.rewind_before_invocation_id.as_deref(),
            Some(target_invocation_id.as_str())
        );
    }

    #[rusty_tokio::test]
    async fn rewind_async_errors_when_the_invocation_id_is_not_found() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        match runner
            .rewind_async("user", "s1", "no-such-invocation")
            .await
        {
            Err(RunnerError::InvocationNotFound(id)) => assert_eq!(id, "no-such-invocation"),
            other => panic!("expected InvocationNotFound, got {}", other.is_ok()),
        }
    }

    #[rusty_tokio::test]
    async fn rewind_async_auto_creates_a_missing_session_then_still_reports_invocation_not_found() {
        // C0894: proves auto-creation happened (no SessionNotFound) without
        // masking the real error (an empty, freshly-created session has no
        // invocation to match).
        let runner = runner(true);

        match runner
            .rewind_async("user", "s1", "no-such-invocation")
            .await
        {
            Err(RunnerError::InvocationNotFound(id)) => assert_eq!(id, "no-such-invocation"),
            other => panic!("expected InvocationNotFound, got {}", other.is_ok()),
        }
        assert!(runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .is_some());
    }

    #[rusty_tokio::test]
    async fn rewind_async_reports_a_missing_session_when_auto_create_is_off() {
        let runner = runner(false);
        let err = runner
            .rewind_async("user", "s1", "some-invocation")
            .await
            .unwrap_err();
        assert!(matches!(err, RunnerError::SessionNotFound(_)));
    }

    // C0835: `get_function_responses_from_content`.

    #[test]
    fn get_function_responses_from_content_is_empty_for_none() {
        assert!(get_function_responses_from_content(None).is_empty());
    }

    #[test]
    fn get_function_responses_from_content_is_empty_for_no_parts() {
        let content = Content::new("user", vec![]);
        assert!(get_function_responses_from_content(Some(&content)).is_empty());
    }

    #[test]
    fn get_function_responses_from_content_extracts_only_response_bearing_parts() {
        let content = Content::new(
            "user",
            vec![
                Part::text("not a response"),
                Part::function_response(FunctionResponse {
                    id: Some("fc-1".to_string()),
                    name: Some("tool_a".to_string()),
                    ..Default::default()
                }),
                Part::function_response(FunctionResponse {
                    id: Some("fc-2".to_string()),
                    name: Some("tool_b".to_string()),
                    ..Default::default()
                }),
            ],
        );
        let responses = get_function_responses_from_content(Some(&content));
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].id.as_deref(), Some("fc-1"));
        assert_eq!(responses[1].id.as_deref(), Some("fc-2"));
    }

    // C0836: `apply_run_config_custom_metadata` — exercised indirectly by
    // several `run_async` tests above, but tested directly here for the
    // three cases the manifest evidence cites by name.

    #[test]
    fn apply_run_config_custom_metadata_merges_config_metadata_onto_a_bare_event() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        let run_config = RunConfig {
            custom_metadata: Some(
                [(
                    "source".to_string(),
                    Value::String("run_config".to_string()),
                )]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };

        apply_run_config_custom_metadata(&mut event, &run_config);

        assert_eq!(
            event.custom_metadata.unwrap().get("source"),
            Some(&Value::String("run_config".to_string()))
        );
    }

    #[test]
    fn apply_run_config_custom_metadata_prefers_the_events_own_keys_on_conflict() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        event.custom_metadata = Some(
            [("source".to_string(), Value::String("event".to_string()))]
                .into_iter()
                .collect(),
        );
        let run_config = RunConfig {
            custom_metadata: Some(
                [(
                    "source".to_string(),
                    Value::String("run_config".to_string()),
                )]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };

        apply_run_config_custom_metadata(&mut event, &run_config);

        assert_eq!(
            event.custom_metadata.unwrap().get("source"),
            Some(&Value::String("event".to_string()))
        );
    }

    #[test]
    fn apply_run_config_custom_metadata_is_a_noop_without_config_metadata() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        apply_run_config_custom_metadata(&mut event, &RunConfig::default());
        assert!(event.custom_metadata.is_none());

        let run_config = RunConfig {
            custom_metadata: Some(std::collections::BTreeMap::new()),
            ..Default::default()
        };
        apply_run_config_custom_metadata(&mut event, &run_config);
        assert!(event.custom_metadata.is_none());
    }
}
