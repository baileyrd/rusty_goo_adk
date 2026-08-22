//! Model layer core, ported from `google.adk.models` (Phase 3).
//!
//! See each module's doc comment for capability IDs and disclosed
//! adaptations/deferrals. Phase 3 batch 2 adopted `reqwest` (REST/SSE
//! transport, `rustls-tls`) after checking every sibling Rusty-Mill repo for
//! an HTTP/TLS candidate (none exists) and started the native Gemini
//! backend (`gemini` module: C0123, C0124, C0129, C0130). The rest of the
//! Gemini backend — real `generate_content_async` wire calls, the Live
//! WebSocket `connect()`, `GeminiLlmConnection`, and
//! `GeminiContextCacheManager` (C0125-C0128, C0131-C0143) — is deferred to
//! further batches; see `gemini.rs`'s module doc for exactly what's left and
//! why.

pub mod base_llm;
pub mod base_llm_connection;
pub mod cache_metadata;
pub mod capabilities;
pub mod gemini;
pub mod google_client_headers;
pub mod llm_request;
pub mod llm_response;
pub mod registry;
