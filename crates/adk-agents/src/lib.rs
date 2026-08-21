//! Agent core & context, ported from `google.adk.agents` (Phase 2).
//!
//! See each module's doc comment for capability IDs and disclosed
//! adaptations/deferrals relative to the source. Batch 1 covers `BaseAgent`
//! and the `Context`/`ReadonlyContext`/`InvocationContext`/`RunConfig`
//! family (C0035-C0078); `LlmAgent` (C0079-C0100) lands in a follow-up
//! batch.

pub mod active_streaming_tool;
pub mod base_agent;
pub mod context;
pub mod context_cache_config;
pub mod invocation_context;
pub mod live_request;
pub mod readonly_context;
pub mod run_config;
pub mod services;
pub mod session;
pub mod state;
pub mod streaming_mode;
pub mod transcription_entry;
