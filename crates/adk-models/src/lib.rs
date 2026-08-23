//! Model layer core, ported from `google.adk.models` (Phase 3).
//!
//! See each module's doc comment for capability IDs and disclosed
//! adaptations/deferrals. Phase 3 batch 2 adopted `reqwest` (REST/SSE
//! transport, `rustls-tls`) after checking every sibling Rusty-Mill repo for
//! an HTTP/TLS candidate (none exists) and started the native Gemini
//! backend (`gemini` module: C0123, C0124, C0129, C0130). Later batches
//! landed real `generate_content_async` wire calls, the Live WebSocket
//! `connect()`/`GeminiLlmConnection`, and `GeminiContextCacheManager`
//! (C0125, C0127, C0129-C0143) — see `gemini.rs`'s and
//! `gemini_context_cache_manager.rs`'s module docs for exactly what's still
//! deferred (SSE streaming, the interactions API, wiring the cache manager
//! into `generate_content_async`) and why. `ollama` is a narrower,
//! independently-scoped addition (not a manifest capability of its own —
//! see its module doc).

pub mod base_llm;
pub mod base_llm_connection;
pub mod cache_metadata;
pub mod capabilities;
pub mod gemini;
pub mod gemini_context_cache_manager;
pub mod gemini_llm_connection;
pub mod generate_content_request;
pub mod generate_content_response;
pub mod google_client_headers;
pub mod live_connection;
pub mod live_server_message;
pub mod llm_request;
pub mod llm_response;
pub mod ollama;
pub mod registry;
