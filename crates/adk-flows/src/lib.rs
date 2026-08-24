//! LLM flow orchestration, ported from `google.adk.flows` (Phase 4).
//!
//! **Why a new crate, not `adk-agents` or `adk-models`**: `LlmAgent`'s real
//! model-resolution behavior (`canonical_model`/`canonical_live_model`,
//! C0080/C0090) needs `BaseLlm`/`LlmRegistry` (`adk-models`) — but
//! `adk-models` already depends on `adk-agents` (for
//! `ContextCacheConfig`, used by `LlmRequest.cache_config`). Having
//! `adk-agents` depend back on `adk-models` for these two methods would
//! make the two crates depend on each other, which Cargo doesn't allow.
//! This crate sits *above* both (depending on `adk-agents`, `adk-events`,
//! and `adk-models`) and hosts the capabilities that need all three
//! together — starting with model resolution and the
//! `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor` processor
//! interfaces `BaseLlmFlow`'s whole request/response pipeline is built
//! from. See `canonical_model.rs`'s module doc for the one real capability
//! gap this split creates (an agent constructed with a live `BaseLlm`
//! instance, rather than a model name, can't be resolved through it).

pub mod agent_transfer;
pub mod basic;
pub mod canonical_model;
pub mod compaction;
pub mod contents;
pub mod context_cache;
pub mod fencing;
pub mod functions;
pub mod functions_utils;
pub mod identity;
pub mod instructions;
pub mod instructions_utils;
pub mod interactions;
pub mod llm_flow;
pub mod output_schema;
pub mod planners;
pub mod processor;
pub mod request_confirmation;
