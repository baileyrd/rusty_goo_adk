//! Tool/toolset infrastructure, ported from `google.adk.tools` (Phase 8).
//!
//! **Why a new crate, not `adk-flows`**: `BaseTool::process_llm_request`'s
//! default behavior needs to mutate an `LlmRequest` (`adk-models`), and
//! `BaseTool` itself needs `ToolContext` (`adk_agents::context::Context`,
//! `adk-agents`). `adk-flows` already depends on both, and *its* processors
//! (`output_schema`, `agent_transfer`, `request_confirmation`) are exactly
//! what still needs `BaseTool` to finish wiring — so this crate sits
//! alongside `adk-flows` (same dependency level: `adk-agents` +
//! `adk-genai` + `adk-models`), and `adk-flows` will depend on it once a
//! follow-up batch wires those processors' disclosed gaps closed.
//!
//! **Adaptation, disclosed**: `LlmRequest::append_tools` (C0116) can't
//! become a real *method* on `LlmRequest` — `adk-models` would then need
//! to depend on `adk-tools` for `BaseTool`, and `adk-tools` already
//! depends on `adk-models` for `LlmRequest`, a crate-graph cycle (the same
//! constraint `adk-flows`'s own top-level module doc discloses for
//! `canonical_model`). It's a free function here instead
//! ([`append_tools::append_tools`]), taking `&mut LlmRequest` directly —
//! the same "processor as a free function, not a method" pattern
//! `adk-flows` uses throughout.

pub mod agent_tool;
pub mod append_tools;
pub mod base_tool;
pub mod base_toolset;
pub mod bash_tool;
pub mod example_tool;
pub mod exit_loop_tool;
pub mod function_tool;
pub mod get_user_choice_tool;
pub mod load_memory_tool;
pub mod long_running_tool;
pub mod memory_entry_utils;
pub mod preload_memory_tool;
pub mod set_model_response_tool;
pub mod tool_confirmation;
pub mod tool_context;
pub mod transfer_to_agent_tool;
