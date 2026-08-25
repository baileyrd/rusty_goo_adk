//! Agent core & context, ported from `google.adk.agents` (Phase 2).
//!
//! See each module's doc comment for capability IDs and disclosed
//! adaptations/deferrals relative to the source. Batch 1 covers `BaseAgent`
//! and the `Context`/`ReadonlyContext`/`InvocationContext`/`RunConfig`
//! family (C0035-C0078). Batch 2 covers `LlmAgent`'s config shape and
//! self-contained resolution helpers (`llm_agent.rs`) plus the
//! `TaskRequest`/`TaskResult` task-delegation payloads (`task_models.rs`,
//! C0100) — see `llm_agent.rs`'s module doc for exactly which C0079-C0099
//! rows stay deferred pending Phase 3/4/7/8.
//!
//! **Auth-scheme batch** (Phase 9, follows `auth_credential.rs`):
//! [`auth_schemes`] (`AuthScheme`/`SecurityScheme`/`OpenIdConnectWithConfig`/
//! `CustomAuthScheme`/`OAuthGrantType`/`AuthSchemeType`/`ExtendedOAuth2`,
//! C0503, plus C0498's wrap-up), [`auth_tool`] (`AuthConfig`/
//! `AuthToolArguments`/`stable_digest`, C0504), [`auth_headers`]
//! (`build_auth_headers`, C0522), and [`base_auth_provider`]/
//! [`auth_provider_registry`] (`BaseAuthProvider`/`AuthProviderRegistry`,
//! C0516) — closing out most of C0493's remaining unported names (only
//! `AuthHandler`, C0506, stays open). See each module's own doc for its
//! disclosed narrowing (`SecurityScheme`'s OpenAPI-shape modeling,
//! `CustomAuthScheme`'s flattened extensibility,
//! `AuthProviderRegistry`'s type→discriminant collapse). New dependency:
//! `sha2` (already a workspace dependency, `adk-models
//! ::gemini_context_cache_manager`, new usage site for `auth_tool
//! ::stable_digest`).
//!
//! **Optimization batch**: [`optimization_data_types`]
//! (`SamplingResult`/`BaseSamplingResult`/`UnstructuredSamplingResult`/
//! `AgentWithScores`/`BaseAgentWithScores`/`OptimizerResult`, C0637's
//! data-type half), [`sampler`] (`Sampler`/`ExampleSet`, C0637's
//! interface half), and [`agent_optimizer`] (`AgentOptimizer`, C0636) —
//! ported from `optimization/`, a distinct top-level source package
//! folded into this crate rather than a new one, the same placement
//! reasoning `app_configs.rs` already established for `apps/_configs.py`
//! (every type here reaches directly into `LlmAgent`, this crate's own
//! type, and needs nothing else new). The source's pydantic-generic
//! bounds (`SamplingResultT`/`AgentWithScoresT`) become traits
//! (`SamplingResult`/`AgentWithScores`) `Sampler`/`AgentOptimizer`'s
//! generic parameters are bounded by, rather than base structs callers
//! subclass — `UnstructuredSamplingResult`'s extra `data` field is
//! declared directly on its own struct instead (same "flatten inherited
//! fields" pattern as `ExtendedOAuth2`). `LlmAgent` has no `Clone`/
//! `Debug`, so `BaseAgentWithScores::optimized_agent` holds an `Arc`
//! handle rather than an owned value. No new dependency.
//!
//! **Telemetry per-request config batch**: [`telemetry_context`]
//! (`ContentCapturingMode`/`TelemetryConfig`/`SemconvStabilityOptIn`,
//! C0651/C0652, plus 5 of C0670's 6 env-var-name constants) and
//! [`schema_version`] (`resolve_schema_version`, C0679, plus its own
//! `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN` constant closing out C0670, and
//! `GOOGLE_CLOUD_AGENT_ENGINE_*`, C0671) — ported from `telemetry
//! /context.py`/`telemetry/_schema_version.py`. `RunConfig::telemetry`
//! widens from a `Value` placeholder to the real `TelemetryConfig` (same
//! "widen a placeholder once a real consumer needs the structure"
//! precedent already used repeatedly this port). Pure env-var-precedence
//! logic only — no OTel SDK/span/tracer machinery, that being a much
//! larger, still-unported surface (see `telemetry_context`'s own doc).
//! C0671's four Agent-Engine constants beyond `GOOGLE_CLOUD_AGENT_ENGINE_ID`
//! have no consumer yet either — declared for the next batch that needs
//! them. No new dependency.
//!
//! **Auth-service cluster batch**: [`base_credential_exchanger`]/
//! [`credential_exchanger_registry`] (`BaseCredentialExchanger`/
//! `CredentialExchangerRegistry`, C0523), [`base_credential_refresher`]/
//! [`credential_refresher_registry`] (`BaseCredentialRefresher`/
//! `CredentialRefresherRegistry`, C0525), [`in_memory_credential_service`]
//! (`InMemoryCredentialService`, C0528), and
//! [`session_state_credential_service`] (`SessionStateCredentialService`,
//! C0529) — ported from `auth/exchanger/`, `auth/refresher/`, and
//! `auth/credential_service/`. Both registries key directly on
//! `AuthCredentialTypes` (an already-closed enum, unlike
//! `AuthProviderRegistry`'s `type[AuthScheme]`), needing only new `Hash`/
//! `PartialOrd`/`Ord` derives on that enum to serve as a `HashMap` key.
//! Implementing this cluster required widening two long-stale
//! placeholders in [`services`]: `AuthConfig` (was `Value`) now re-exports
//! [`auth_tool::AuthConfig`] (same "widen a placeholder once a real
//! consumer needs the structure" precedent as `RunConfig::telemetry`),
//! and `CredentialService` grows from a synchronous, context-free trait
//! into the real async, `Context`-taking interface (via this crate's
//! `BoxFuture` convention) — safe because grep confirmed zero prior
//! implementors or call sites existed for either. That widening changed
//! `Context::save_credential`'s receiver from `&self` to `&mut self`
//! (again zero external call sites, so zero blast radius) and split the
//! trait's two methods on mutability — `load_credential` takes `&Context`,
//! `save_credential` takes `&mut Context` — a distinction the source's
//! single `callback_context: CallbackContext` parameter doesn't need to
//! make explicit but Rust's stricter tracking surfaces naturally; not a
//! narrowing. `Context::request_credential` now serializes `AuthConfig`
//! via `rusty_serde::json::to_value` before storing it in
//! `EventActions::requested_auth_configs` (still `Value`-typed there,
//! deliberately left out of scope to widen this batch), which is why
//! `auth_tool::AuthConfig` also grows `Serialize`/`Deserialize` derives.
//! No new dependency.
//!
//! **`SaveFilesAsArtifactsPlugin` batch**: [`save_files_as_artifacts_plugin`]
//! (`SaveFilesAsArtifactsPlugin`, C0367) — ported from `plugins
//! /save_files_as_artifacts_plugin.py`. Saves `inline_data` parts in a
//! user message as artifacts, replacing each with a placeholder (and,
//! optionally, a `file_data` reference part for a model-accessible
//! `canonical_uri`). Reads `MediaBlobStub`'s flattened `rest` map for
//! `displayName`/`data` (the same pattern `file_artifact_service.rs`
//! already established), whose `base64_decode` helper is promoted
//! `pub(crate)` for reuse here rather than a third hand-rolled copy.
//! Required fixing a real gap this plugin's own two-hook design
//! surfaced: `adk-runners::runner::merge_context_state_into_session`
//! (new) bridges a run-level plugin hook's state mutations back onto
//! the session — this port's `Context` clones `InvocationContext`
//! rather than sharing it by reference the way the source's raw
//! `InvocationContext.session.state` dict does, so without this fix a
//! mutation made in `on_user_message_callback` would never be visible
//! to a later hook (`before_agent_callback`, in this plugin's case).
//! Verified end-to-end in `adk-runners`'s own test suite. No new
//! dependency.
//!
//! **`LoggingPlugin` batch**: [`logging_plugin`] (`LoggingPlugin`,
//! C0362, partial) — ported from `plugins/logging_plugin.py`. 6 of its
//! 13 hooks (`on_user_message_callback`/`before_run_callback`/
//! `on_event_callback`/`after_run_callback`/`before_agent_callback`/
//! `after_agent_callback`) port in full, including the ANSI-grey
//! `println!`-based console output — faithful, not a substitution,
//! since the source itself calls bare `print()` rather than routing
//! through Python's `logging` module for this specific plugin. The
//! remaining 7 (model-level and tool-level hooks) stay N/A, blocked on
//! C0355/C0356's already-disclosed crate-cycle blocker.
//!
//! **`App` model batch**: [`app`] (`App`/`validate_app_name`, C0279/C0280)
//! — ported from `apps/app.py`. `root_agent` narrows from the source's
//! `Union[BaseAgent, BaseNode, None]` to `BaseAgent`-only (the workflow
//! graph engine, C0298-C0306, isn't built in this port — see the module's
//! own doc) and becomes a required constructor argument rather than an
//! `Option`, since the source's own `_validate` model-validator already
//! rejects a `None` root_agent. App-name validation is a new, distinct
//! validator from `base_agent::validate_name` — the source's app-name
//! regex additionally permits hyphens. `App` is deliberately not wired
//! into `Runner`'s constructor this batch (a follow-up, once `App` exists
//! and can be reviewed on its own). No new dependency.

pub mod active_streaming_tool;
pub mod agent_optimizer;
pub mod app;
pub mod app_configs;
pub mod artifact_util;
pub mod auth_credential;
pub mod auth_handler;
pub mod auth_headers;
pub mod auth_provider_registry;
pub mod auth_schemes;
pub mod auth_tool;
pub mod base_agent;
pub mod base_auth_provider;
pub mod base_credential_exchanger;
pub mod base_credential_refresher;
pub mod context;
pub mod context_cache_config;
pub mod credential_exchanger_registry;
pub mod credential_refresher_registry;
pub mod file_artifact_service;
pub mod in_memory_artifact_service;
pub mod in_memory_credential_service;
pub mod in_memory_memory_service;
pub mod invocation_context;
pub mod live_request;
pub mod llm_agent;
pub mod logging_plugin;
pub mod loop_agent;
pub mod oauth2_discovery;
pub mod oauth2_util;
pub mod optimization_data_types;
pub mod parallel_agent;
pub mod readonly_context;
pub mod reflect_retry_utils;
pub mod run_config;
pub mod sampler;
pub mod save_files_as_artifacts_plugin;
pub mod schema_version;
pub mod sequential_agent;
pub mod services;
pub mod session;
pub mod session_state_credential_service;
pub mod session_util;
pub mod state;
pub mod streaming_mode;
pub mod task_models;
pub mod telemetry_context;
pub mod transcription_entry;
