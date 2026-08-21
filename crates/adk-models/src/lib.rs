//! Model layer core, ported from `google.adk.models` (Phase 3).
//!
//! See each module's doc comment for capability IDs and disclosed
//! adaptations/deferrals. The native Gemini backend (`Gemini`,
//! `GeminiLlmConnection`, `GeminiContextCacheManager` — C0123-C0143) is
//! deferred to a follow-up batch: it needs a real HTTP/WebSocket client to
//! Google's API, a dependency decision on the same scale as Phase 2's
//! async-runtime choice, and deserves its own focused pass with a sibling-repo
//! check before hand-rolling.

pub mod base_llm;
pub mod base_llm_connection;
pub mod cache_metadata;
pub mod capabilities;
pub mod llm_request;
pub mod llm_response;
pub mod registry;
