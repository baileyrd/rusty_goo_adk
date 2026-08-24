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

pub mod active_streaming_tool;
pub mod agent_optimizer;
pub mod app_configs;
pub mod artifact_util;
pub mod auth_credential;
pub mod auth_headers;
pub mod auth_provider_registry;
pub mod auth_schemes;
pub mod auth_tool;
pub mod base_agent;
pub mod base_auth_provider;
pub mod context;
pub mod context_cache_config;
pub mod file_artifact_service;
pub mod in_memory_artifact_service;
pub mod in_memory_memory_service;
pub mod invocation_context;
pub mod live_request;
pub mod llm_agent;
pub mod loop_agent;
pub mod oauth2_util;
pub mod optimization_data_types;
pub mod parallel_agent;
pub mod readonly_context;
pub mod run_config;
pub mod sampler;
pub mod sequential_agent;
pub mod services;
pub mod session;
pub mod session_util;
pub mod state;
pub mod streaming_mode;
pub mod task_models;
pub mod transcription_entry;
