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
//!
//! **Skills batch**: [`skill_registry`] (`SkillRegistry`, C0395),
//! [`skill_instructions_utils`] (a local duplicate of `adk_flows
//! ::instructions_utils::inject_session_state`, C0170 — needed for C0401,
//! since `adk-flows` already depends on `adk-tools`), and [`skill_toolset`]
//! (`SkillToolset`/`ListSkillsTool`/`SearchSkillsTool`/`LoadSkillTool`,
//! C0408; `LoadSkillResourceTool`, C0409; `RunSkillScriptTool` partial,
//! C0410 — the `environment`-configured path only, its `code_executor`
//! path needing `_SkillScriptCodeExecutor`'s from-scratch Python-wrapper-
//! generation design deferred to its own batch;
//! `DEFAULT_SKILL_SYSTEM_INSTRUCTION`, C0411; C0401's `adk_inject_state`
//! interpolation, exercised via `LoadSkillTool`). `skills_models::Resources`
//! widens from `String`-only to a real `ResourceContent` (`Text`/`Bytes`)
//! enum this batch, now that `LoadSkillResourceTool` is a real consumer
//! needing the binary branch — see that module's own doc. See
//! `skill_toolset`'s own module doc for its central architectural
//! adaptation (a shared `Arc<SkillCoreState>` every tool clones a handle
//! to, replacing the source's toolset-back-reference cycle — the same
//! pattern the environment batch's `EnvironmentToolset` already
//! established) and its other disclosed narrowings. New manifest row
//! **C0950** (a discovered gap: `SkillToolset.additional_tools`/
//! `_resolve_additional_tools_from_state`/`clone_with_updated_skills`,
//! left `REQUIRED` this batch) is closed out in the next one, below. No
//! new dependency.
//!
//! **Skills additional_tools batch**: closes **C0950** —
//! `skill_toolset::{AdditionalTool, SkillToolsetConfig::additional_tools,
//! SkillToolset::resolve_additional_tools_from_state,
//! SkillToolset::clone_with_updated_skills}`. Every real behavior ports:
//! provided-tool/provided-toolset candidate resolution once a skill
//! naming them in `adk_additional_tools` is activated, the core-tool-
//! name-collision skip, and `clone_with_updated_skills`'s exact field
//! carry-forward (faithfully including the source's own omission of
//! `tool_name_prefix`/`tool_filter` from the clone). Disclosed
//! narrowing: `ToolUnion`'s third member — a bare `Callable`, wrapped via
//! `FunctionTool(callable)`'s `inspect.signature` reflection in the
//! source — has no port, since `FunctionTool`'s own module doc already
//! discloses this port has no such runtime reflection; `AdditionalTool`
//! only models the two Rust-expressible branches (`Tool`/`Toolset`). No
//! new dependency.
//!
//! **`RunSkillScriptTool` code_executor batch**: closes out **C0410** in
//! full — `skill_toolset::SkillScriptCodeExecutor` (the
//! `_SkillScriptCodeExecutor` port: self-extracting Python wrapper
//! generation for `.py`/`.sh`/`.bash` skill scripts, executed against
//! `BaseCodeExecutor::execute_code` via `rusty_tokio::spawn_blocking`)
//! plus `python_str_literal`/`python_bytes_literal`/`python_list_literal`/
//! `python_dict_literal` (a `repr()`-equivalent, cross-verified round-
//! trip-correct against a real `python3` interpreter in this module's
//! own tests) and the `code_executor`/`environment` mutual-exclusivity
//! check the constructor was missing. Also verified end-to-end against
//! real `python3`/`bash` interpreters. Disclosed narrowing: the source's
//! `except SystemExit as e:` branch is dead code for this port's only
//! concrete `BaseCodeExecutor` (`UnsafeLocalCodeExecutor`, always
//! subprocess-based) — see `skill_toolset`'s own module doc. No new
//! dependency.
//!
//! **Environment batch**: [`base_environment`] (`BaseEnvironment`/
//! `ExecutionResult`, C0948 — a genuine inventory gap discovered this
//! batch, `environment/` having no manifest row at all despite four
//! existing rows already referencing it), [`local_environment`]
//! (`LocalEnvironment`, C0949), [`environment_toolset`]
//! (`EnvironmentToolset`, C0440), and [`execute_tool`]/[`read_file_tool`]/
//! [`edit_file_tool`]/[`write_file_tool`] (`Execute`/`ReadFile`/
//! `EditFile`/`WriteFile`, C0441-C0444). None need GCP, an LLM-invocation
//! path, or a new dependency — `regex` and `rusty_tokio` are both already
//! workspace dependencies of this crate. See `base_environment`'s and
//! `local_environment`'s own module docs for the interior-mutability and
//! lexical-path-resolution adaptations (the latter reusing the "path
//! safety by construction, not by canonicalize" pattern
//! `file_artifact_service.rs`, C0268-C0269, already established), and
//! `environment_toolset`'s for why an uncaught initialize-failure
//! translates to a panic rather than widening the already-shipped
//! `BaseToolset` trait's infallible signature.
//!
//! **Environment simulation batch**: [`environment_simulation_config`]
//! (`InjectedError`/`InjectionConfig`/`MockStrategy`/
//! `ToolSimulationConfig`/`EnvironmentSimulationConfig`, C0486 — also the
//! first real call site for `adk_features::feature_decorator
//! ::check_feature_enabled`, C0647's guard function, landed but unwired
//! last batch), [`environment_simulation_engine`]
//! (`EnvironmentSimulationEngine::simulate`'s injection-only path — the
//! probability roll, `match_args` filtering, latency, and injected
//! error/response, C0487 partial), and [`tool_connection_map`]/
//! [`environment_simulation_factory`] (`StatefulParameter`/
//! `ToolConnectionMap` as pure data, `EnvironmentSimulationFactory
//! ::create_callback` producing a real closure with the source's shape,
//! C0488 partial). Deferred, disclosed in each module's own doc: the
//! LLM-synthesized mock-strategy fallback (`ToolConnectionAnalyzer`,
//! `ToolSpecMockStrategy`, `agent.canonical_tools`) and
//! `EnvironmentSimulationPlugin`/`create_plugin` (needs a `BasePlugin`
//! tool-hook this port doesn't expose yet, same gap as the existing C0356
//! deferral) — this port has no LLM-invocation path to drive either, and
//! no tool-scoped `before_tool_callback` type (`adk_agents::llm_agent
//! ::LlmCallback` takes `&mut Context` only, no `tool`/`args`) to wire
//! `create_callback`'s output into regardless. No new dependency.
//!
//! **`ForwardingArtifactService` batch**: [`forwarding_artifact_service`]
//! (`ForwardingArtifactService`, C0489 partial) — ported from
//! `tools/_forwarding_artifact_service.py`, closing the disclosed gap
//! `agent_tool.rs` has carried since C0406: [`agent_tool::AgentTool
//! ::run_async`] now installs one on the nested `Runner` whenever the
//! parent tool context has a real artifact service, so a nested agent
//! can read/write real artifacts instead of running with none. See the
//! new module's own doc for the disclosed post-hoc-merge adaptation
//! (this port's synchronous `ArtifactService` trait can't hold a live
//! mutable borrow of the parent `Context` across the nested run, so
//! artifact-delta bookkeeping is deferred to a merge after the run
//! completes — the same idiom `agent_tool.rs` already uses for state
//! deltas). No new dependency.
//!
//! **`LlmEventSummarizer` batch**: [`llm_event_summarizer`]
//! (`LlmEventSummarizer`, C0286/C0287) — ported from
//! `apps/llm_event_summarizer.py`, implementing `adk-agents`'s
//! `BaseEventsSummarizer` trait (C0285, DONE there since it has no
//! `adk-models` dependency of its own). Lands here rather than in
//! `adk-agents` because this type needs a real `adk_models::BaseLlm`,
//! and `adk-models` already depends on `adk-agents` — the same
//! supporting-crate placement `forwarding_artifact_service.rs` (C0489)
//! already used. Formats a conversation history (including thoughts and
//! tool calls, skipping thoughts on a prior compaction event), drives
//! one non-streaming LLM call, and wraps the result into an `Event`
//! carrying an `EventCompaction` action. `args`/`response` formatting
//! reuses the same disclosed compact-JSON-instead-of-`str()` divergence
//! `adk-events::debug_output` (C0933) already established. No new
//! dependency.

pub mod agent_tool;
pub mod append_tools;
pub mod base_code_executor;
pub mod base_environment;
pub mod base_retrieval_tool;
pub mod base_tool;
pub mod base_toolset;
pub mod bash_tool;
pub mod built_in_code_executor;
pub mod code_execution_utils;
pub mod code_executor_context;
pub mod edit_file_tool;
pub mod enterprise_search_tool;
pub mod environment_simulation_config;
pub mod environment_simulation_engine;
pub mod environment_simulation_factory;
pub mod environment_toolset;
pub mod example_tool;
pub mod execute_tool;
pub mod exit_loop_tool;
pub mod finish_task_tool;
pub mod forwarding_artifact_service;
pub mod function_tool;
pub mod gemini_schema_util;
pub mod get_user_choice_tool;
pub mod google_maps_grounding_tool;
pub mod google_search_tool;
pub mod llm_event_summarizer;
pub mod load_artifacts_tool;
pub mod load_mcp_resource_tool;
pub mod load_memory_tool;
pub mod load_web_page;
pub mod local_environment;
pub mod long_running_tool;
pub mod mcp_conversion_utils;
pub mod memory_entry_utils;
pub mod model_name_utils;
pub mod preload_memory_tool;
pub mod read_file_tool;
pub mod remote_mcp_server;
pub mod request_input_tool;
pub mod set_model_response_tool;
pub mod skill_instructions_utils;
pub mod skill_registry;
pub mod skill_toolset;
pub mod skills_models;
pub mod skills_prompt;
pub mod tool_confirmation;
pub mod tool_connection_map;
pub mod tool_context;
pub mod transfer_to_agent_tool;
pub mod unsafe_local_code_executor;
pub mod url_context_tool;
pub mod vertex_ai_search_tool;
pub mod write_file_tool;
