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

pub mod active_streaming_tool;
pub mod app_configs;
pub mod artifact_util;
pub mod auth_credential;
pub mod base_agent;
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
pub mod parallel_agent;
pub mod readonly_context;
pub mod run_config;
pub mod sequential_agent;
pub mod services;
pub mod session;
pub mod session_util;
pub mod state;
pub mod streaming_mode;
pub mod task_models;
pub mod transcription_entry;
