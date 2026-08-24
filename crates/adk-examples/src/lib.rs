//! Few-shot example extension point, ported from `google.adk.examples`
//! (Phase 17).
//!
//! **Why a new crate**: `Example`/`BaseExampleProvider`/`build_example_si`
//! need `adk-agents` (`Session`, for `get_latest_message_from_user`) and
//! `adk-genai` (`Content`). `adk-tools::ExampleTool` (C0419) needs all
//! three of these in turn, so this crate sits alongside `adk-tools` (same
//! dependency level: `adk-agents` + `adk-genai` + `adk-events`) rather
//! than inside it, keeping `adk-tools` free to depend on it without
//! `adk-examples` ever depending back on `adk-tools`.
//!
//! **Not** ported: `VertexAiExampleStore` (C0830) — a
//! `BaseExampleProvider` backed by the Vertex AI Example Store service.
//! Like this port's other Vertex-AI-auth-gated capabilities (see
//! `gemini_context_cache_manager.rs`'s own disclosed Vertex AI deferral),
//! it needs a real Vertex AI client/credentials this workspace doesn't
//! have; the `BaseExampleProvider` trait it would implement is fully
//! built and ready for it.

pub mod base_example_provider;
pub mod example;
pub mod example_util;
