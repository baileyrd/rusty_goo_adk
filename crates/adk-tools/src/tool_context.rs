//! Capability C0415 (partial): `ToolContext`, ported from
//! `google.adk.tools.tool_context`.
//!
//! The source's `ToolContext = Context` is a bare type alias (`Context`
//! already carries everything a tool needs — state, artifacts, the
//! invocation). This port already has a real `Context` in `adk-agents`
//! (Phase 2), so this alias is the whole capability.
//!
//! **Not** ported: the lazy `AuthCredential`/`AuthHandler`/`AuthConfig`
//! back-compat re-exports — those types belong to `auth/` (Phase 9),
//! which doesn't exist in this port yet.

/// `ToolContext = Context` — see the module doc.
pub type ToolContext = adk_agents::context::Context;
