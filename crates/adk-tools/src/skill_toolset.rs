//! Capabilities C0408 (core `SkillToolset`/`ListSkillsTool`/
//! `SearchSkillsTool`/`LoadSkillTool`), C0409 (`LoadSkillResourceTool`),
//! C0410 (`RunSkillScriptTool`, now in full — both the `environment` and
//! `code_executor` branches), C0411 (`DEFAULT_SKILL_SYSTEM_INSTRUCTION`),
//! C0401 (`adk_inject_state` interpolation, exercised via `LoadSkillTool`),
//! and C0950 (`SkillToolset.additional_tools`/`_resolve_additional_tools_from_state`/
//! `clone_with_updated_skills`), ported from `google.adk.tools.skill_toolset`.
//!
//! **Architectural adaptation, disclosed at length**: the source's five
//! tool classes (`ListSkillsTool`, `SearchSkillsTool`, `LoadSkillTool`,
//! `LoadSkillResourceTool`, `RunSkillScriptTool`) each hold a live
//! back-reference to their owning `SkillToolset` (`self._toolset`), and
//! reach into its "private" state directly (`toolset._registry`,
//! `toolset._get_or_fetch_skill(...)`, etc.) — an ordinary reference
//! cycle in a garbage-collected language. Rust has no such cycle: a tool
//! can't hold an `Arc<SkillToolset>` back to a toolset that itself owns
//! `Arc<dyn BaseTool>` handles to that same tool. This port instead pulls
//! the shared, mutable state every tool needs (`skills`, `registry`,
//! `environment`, the per-invocation fetched-skill cache, `skills_folder`,
//! `script_timeout`) into its own [`SkillCoreState`], owned behind one
//! `Arc` that `SkillToolset` and every tool clone a handle to — the same
//! "shared resource behind `Arc`, not a toolset back-reference" pattern
//! `environment_toolset.rs` (C0440) already established for its four
//! tools sharing one `Arc<dyn BaseEnvironment>`.
//!
//! **C0950, `ToolUnion`'s callable branch not ported**: the source's
//! `additional_tools` accepts `BaseTool | BaseToolset | Callable` and
//! wraps a bare callable in `FunctionTool(tool_union)` via
//! `inspect.signature` reflection. [`crate::function_tool::FunctionTool`]'s
//! own module doc already discloses that this port has no such runtime
//! reflection — `FunctionTool::new` requires an explicit, hand-built
//! `FunctionDeclaration`, it can't derive one from a bare closure. So
//! [`AdditionalTool`] only models the two branches Rust can actually
//! express (`Tool`/`Toolset`) — the callable branch was a convenience
//! overload in Python with no faithful equivalent here, not a capability
//! this port drops.
//!
//! **C0410, now DONE — `SkillScriptCodeExecutor`**: `RunSkillScriptTool`'s
//! `code_executor`-configured branch (`_SkillScriptCodeExecutor` in the
//! source) is ported in full: [`SkillScriptCodeExecutor::build_wrapper_code`]
//! generates the same self-extracting Python wrapper source (embedding
//! every skill resource as a Python literal, extracting to a temp dir,
//! then either `runpy.run_path`-ing a `.py` target or `subprocess.run`-ing
//! a `.sh`/`.bash` target through `bash` with a JSON-envelope result) and
//! hands it to [`BaseCodeExecutor::execute_code`] via
//! `rusty_tokio::spawn_blocking` (the `asyncio.to_thread` equivalent).
//! [`python_str_literal`]/[`python_bytes_literal`]/[`python_list_literal`]/
//! [`python_dict_literal`] are this port's `repr()`-equivalent — round-
//! trip-correct (verified against a real `python3` interpreter in this
//! module's own tests) but not byte-identical to CPython's adaptive
//! quote-selection, the same disclosed caveat already given to
//! `value_to_display_string` elsewhere in this port. This port's only
//! concrete `BaseCodeExecutor` (`UnsafeLocalCodeExecutor`, C0385) always
//! runs the wrapper as a real subprocess, so the source's defensive
//! `except SystemExit as e:` branch (meaningful only for an in-process-
//! `exec`-based executor) is dead code here — see
//! [`SkillScriptCodeExecutor`]'s own doc for why it isn't reproduced.
//! Also added: the mutual-exclusivity check
//! (`"Cannot have both code_executor and environment"`) the source's own
//! constructor enforces, which this port's constructor was missing until
//! now.
//!
//! **`_get_or_fetch_skill`'s cache, disclosed narrowing**: the source
//! stores an `asyncio.Future` in the per-invocation turn cache the
//! instant a registry fetch starts, so a second concurrent call for the
//! same not-yet-resolved skill within the same invocation awaits the
//! first call's in-flight future instead of firing a second registry
//! request. Replicating that coalescing exactly needs a cross-task-
//! shared, cloneable pending-future primitive Rust's stricter ownership
//! model doesn't hand you for free the way Python's shared-object-
//! identity futures do. This port's [`SkillCoreState::get_or_fetch_skill`]
//! keeps the 16-turn FIFO-eviction *caching* behavior exactly (a
//! resolved skill is cached and reused across calls within the same
//! invocation, oldest invocation evicted past 16), but two concurrent
//! calls for the same uncached skill each independently call
//! `SkillRegistry::get_skill` rather than one waiting on the other —
//! redundant work, not an observable correctness difference, since a
//! registry fetch is expected to be idempotent.
//!
//! **`has_list_skills`, dead code not reproduced**: the source's
//! `process_llm_request` conditionally appends a skills-XML instruction
//! only `if not has_list_skills` — but `_tools` unconditionally includes
//! `ListSkillsTool(self)` in `__init__`, with no configuration path that
//! omits it, so `has_list_skills` is always `True` and that branch is
//! unreachable given how this class is actually used. Not reproduced,
//! same "dead code, not reproduced" treatment already given to
//! `AgentEvaluator._validate_input`'s analogous always-true guard
//! (`adk_eval::agent_evaluator`, C0619).
//!
//! **Path types**: per [`crate::base_environment::BaseEnvironment`]'s own
//! disclosed adaptation, `read_file`/`write_file` take `&str`, not
//! `Path`/`PurePosixPath` — this port builds POSIX-style path strings
//! with plain `/`-joins instead of round-tripping through `PurePosixPath`.
//!
//! **Telemetry hooks not ported**: `_instrumentation.track_skill_load`/
//! `track_skill_resource_load` and every `_detect_error_in_response`
//! override are omitted — no telemetry pipeline exists anywhere in this
//! port yet (same standing gap already disclosed throughout this crate).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use adk_agents::readonly_context::ReadonlyContext;
use adk_genai::content::{Content, FunctionDeclaration, MediaBlobStub, Part};
use adk_models::llm_request::{Instructions, LlmRequest};
use rusty_serde::value::Value;

use crate::base_code_executor::BaseCodeExecutor;
use crate::base_environment::BaseEnvironment;
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::base_toolset::{BaseToolset, PrefixCache, ToolFilter};
use crate::code_execution_utils::{base64_encode, CodeExecutionInput};
use crate::skill_instructions_utils::inject_session_state;
use crate::skill_registry::SkillRegistry;
use crate::skills_models::{ResourceContent, Skill};
use crate::skills_prompt::{format_skills_as_xml, SkillSummary};
use crate::tool_context::ToolContext;

const DEFAULT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_CACHE_TURNS: usize = 16;
const MAX_SKILL_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const BINARY_FILE_DETECTED_MSG: &str = "Binary file detected. The content has been injected into the conversation history for you to analyze.";

// ---------------------------------------------------------------------
// C0411: DEFAULT_SKILL_SYSTEM_INSTRUCTION
// ---------------------------------------------------------------------

/// `_build_skill_system_instruction`.
pub fn build_skill_system_instruction(
    prefix: Option<&str>,
    skills_folder: Option<&Path>,
) -> String {
    let p = prefix.map(|p| format!("{p}_")).unwrap_or_default();
    let skills_folder_posix = skills_folder.map(|path| path.to_string_lossy().replace('\\', "/"));

    let env_note = match &skills_folder_posix {
        Some(folder) => format!(
            "8. NOTE ON ENVIRONMENT EXECUTION: When using `{p}run_skill_script` with the \
             `command` parameter, all skill resources (including scripts and assets) are \
             materialized in the execution environment under `{folder}/<skill_name>/`. Always \
             specify file and script paths relative to or starting with \
             `{folder}/<skill_name>/` (e.g., `{folder}/<skill_name>/scripts/<script_name>`).\n"
        ),
        None => String::new(),
    };

    format!(
        "You can use specialized 'skills' to help you with complex tasks. You MUST use the \
         skill tools to interact with these skills.\n\n\
         Skills are folders of instructions and resources that extend your capabilities for \
         specialized tasks. Each skill folder contains:\n\
         - **SKILL.md** (required): The main instruction file with skill metadata and detailed \
         markdown instructions.\n\
         - **references/** (Optional): Additional documentation or examples for skill usage.\n\
         - **assets/** (Optional): Templates, scripts or other resources used by the skill.\n\
         - **scripts/** (Optional): Executable scripts that can be run via bash.\n\n\
         This is very important:\n\n\
         1. If a skill seems relevant to the current user query, you MUST use the \
         `{p}load_skill` tool with `skill_name=\"<SKILL_NAME>\"` to read its full instructions \
         before proceeding.\n\
         2. Once you have read the instructions, follow them exactly as documented before \
         replying to the user. For example, If the instruction lists multiple steps, please \
         make sure you complete all of them in order.\n\
         3. The `{p}load_skill_resource` tool is for viewing files within a skill's directory \
         (e.g., `references/*`, `assets/*`, `scripts/*`). It is ONLY for skill-bundled files — \
         do NOT use it to access documents or files provided by the user at runtime. Do NOT use \
         other tools to access skill files.\n\
         4. Use `{p}run_skill_script` to run scripts from a skill's `scripts/` directory. Use \
         `{p}load_skill_resource` to view script content first if needed.\n\
         5. If `{p}load_skill_resource` returns any error, do not retry any path. Report the \
         error to the user and stop.\n\
         6. If `{p}run_skill_script` returns an error (for example `SCRIPT_NOT_FOUND`), do not \
         retry the same script or guess a different script path. Report the error to the user \
         and stop.\n\
         7. Loading a skill only retrieves its instructions; it does NOT complete your turn. \
         After a `{p}load_skill` call returns, continue in the SAME turn: call whatever tools \
         the skill's steps require (search, data retrieval, render), then write your reply. \
         Never end your turn with an empty response right after loading a skill.\n\
         {env_note}"
    )
}

/// C0411: the canonical, no-prefix/no-environment instance —
/// `DEFAULT_SKILL_SYSTEM_INSTRUCTION`.
pub fn default_skill_system_instruction() -> String {
    build_skill_system_instruction(None, None)
}

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

fn error_response(message: impl Into<String>, error_code: &str) -> Value {
    Value::Map(vec![
        ("error".to_string(), Value::String(message.into())),
        (
            "error_code".to_string(),
            Value::String(error_code.to_string()),
        ),
    ])
}

/// The name of the agent currently running the invocation, `"unknown"` if
/// none is set — matches `ReadonlyContext::agent_name`'s own fallback,
/// duplicated here since `ToolContext` (`= Context`) has no equivalent
/// method of its own.
fn agent_name(context: &ToolContext) -> &str {
    match &context.invocation_context().agent {
        Some(agent) => agent.name(),
        None => "unknown",
    }
}

/// `Context::get_invocation_context` clones the `InvocationContext` as it
/// stood at construction — it does *not* pick up subsequent
/// `Context::state_mut()` writes, since `Context`'s own `State` (base
/// value + delta) is a separate copy from `invocation_context.session.state`,
/// diverging the moment `Context::new` clones it. `inject_session_state`
/// needs the *current* merged state (matching Python's live
/// `tool_context.state` dict), so this builds a [`ReadonlyContext`] whose
/// `session.state` is overridden with `Context::state()::to_map()`'s
/// merged base+delta view first.
fn readonly_context_with_current_state(context: &ToolContext) -> ReadonlyContext {
    let mut invocation_context = context.get_invocation_context();
    invocation_context.session.state = context.state().to_map();
    ReadonlyContext::new(invocation_context)
}

/// Invocation-scoped failure counter shared by `LoadSkillResourceTool`/
/// `RunSkillScriptTool`'s "2-failure fatal-retry guard" — see each
/// tool's own `run_async` for why the `temp:` prefix and per-invocation
/// key matter.
fn increment_failure_counter(context: &mut ToolContext, counter_key: &str) -> i64 {
    let current = match context.state().get(counter_key) {
        Some(Value::Int(n)) => *n,
        Some(Value::UInt(n)) => *n as i64,
        _ => 0,
    };
    let next = current + 1;
    context
        .state_mut()
        .set(counter_key.to_string(), Value::Int(next));
    next
}

/// Small hand-rolled extension→MIME-type table, covering the file types
/// a skill's `references/`/`assets/` realistically hold. Narrower than
/// Python's `mimetypes.guess_type` (which reads a large built-in table
/// plus `/etc/mime.types`) — falls back to `application/octet-stream`
/// for anything not in this table, matching the source's own fallback.
fn guess_mime_type(file_path: &str) -> &'static str {
    let extension = file_path.rsplit('.').next().unwrap_or("").to_lowercase();
    match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "ico" => "image/vnd.microsoft.icon",
        "pdf" => "application/pdf",
        "zip" => "application/zip",
        "gz" => "application/gzip",
        "tar" => "application/x-tar",
        "json" => "application/json",
        "xml" => "application/xml",
        "csv" => "text/csv",
        "txt" | "md" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" => "text/javascript",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------
// SkillCoreState: shared resource every tool clones a handle to
// ---------------------------------------------------------------------

#[derive(Default)]
struct FetchedSkillCache {
    /// Invocation ids in first-seen order — oldest first, for FIFO
    /// eviction past `MAX_CACHE_TURNS`. Same "`HashMap` + parallel `Vec`
    /// for order" pattern already established in
    /// `adk_eval::evaluation_generator::collect_events_by_invocation_id`.
    order: Vec<String>,
    entries: HashMap<String, HashMap<String, Skill>>,
}

/// The shared state every `SkillToolset` tool needs — see this module's
/// top-level doc for why this exists instead of a toolset back-reference.
struct SkillCoreState {
    skills: HashMap<String, Skill>,
    registry: Option<Arc<dyn SkillRegistry>>,
    env: Option<Arc<dyn BaseEnvironment>>,
    code_executor: Option<Arc<dyn BaseCodeExecutor + Send + Sync>>,
    skills_folder_override: Option<PathBuf>,
    script_timeout: Duration,
    fetched_skill_cache: Mutex<FetchedSkillCache>,
}

impl SkillCoreState {
    /// `SkillToolset.skills_folder`.
    fn skills_folder(&self) -> Option<PathBuf> {
        if let Some(path) = &self.skills_folder_override {
            return Some(path.clone());
        }
        self.env
            .as_ref()
            .and_then(|env| env.working_dir().ok())
            .map(|working_dir| working_dir.join("skills"))
    }

    /// `SkillToolset._get_skill`.
    fn get_skill(&self, skill_name: &str) -> Option<Skill> {
        self.skills.get(skill_name).cloned()
    }

    /// `SkillToolset._get_or_fetch_skill` — see this module's doc for the
    /// disclosed narrowing versus the source's `asyncio.Future`-based
    /// in-flight-fetch deduplication.
    async fn get_or_fetch_skill(
        &self,
        skill_name: &str,
        invocation_id: &str,
    ) -> Result<Option<Skill>, String> {
        if let Some(skill) = self.get_skill(skill_name) {
            return Ok(Some(skill));
        }
        let Some(registry) = &self.registry else {
            return Ok(None);
        };

        {
            let cache = self.fetched_skill_cache.lock().unwrap();
            if let Some(cached) = cache
                .entries
                .get(invocation_id)
                .and_then(|turn| turn.get(skill_name))
            {
                return Ok(Some(cached.clone()));
            }
        }

        let skill = registry.get_skill(skill_name).await?;

        let mut cache = self.fetched_skill_cache.lock().unwrap();
        if !cache.entries.contains_key(invocation_id) {
            if cache.order.len() >= MAX_CACHE_TURNS {
                let oldest = cache.order.remove(0);
                cache.entries.remove(&oldest);
            }
            cache.order.push(invocation_id.to_string());
            cache
                .entries
                .insert(invocation_id.to_string(), HashMap::new());
        }
        cache
            .entries
            .get_mut(invocation_id)
            .unwrap()
            .insert(skill_name.to_string(), skill.clone());

        Ok(Some(skill))
    }

    /// `SkillToolset._list_skills`.
    fn list_skills(&self) -> Vec<Skill> {
        self.skills.values().cloned().collect()
    }
}

fn skill_summaries(skills: &[Skill]) -> Vec<SkillSummary<'_>> {
    skills
        .iter()
        .map(|skill| SkillSummary {
            name: skill.name(),
            description: skill.description(),
        })
        .collect()
}

// ---------------------------------------------------------------------
// ListSkillsTool
// ---------------------------------------------------------------------

/// C0408: `ListSkillsTool`(`list_skills`) — lists all available skills.
pub struct ListSkillsTool {
    core: Arc<SkillCoreState>,
}

impl ListSkillsTool {
    fn new(core: Arc<SkillCoreState>) -> Self {
        Self { core }
    }
}

impl BaseTool for ListSkillsTool {
    fn name(&self) -> &str {
        "list_skills"
    }

    fn description(&self) -> &str {
        "Lists all available skills with their names and descriptions."
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                ("properties".to_string(), Value::Map(vec![])),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        _args: &'a std::collections::BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let skills = self.core.list_skills();
            Ok(Value::String(format_skills_as_xml(&skill_summaries(
                &skills,
            ))))
        })
    }
}

// ---------------------------------------------------------------------
// SearchSkillsTool
// ---------------------------------------------------------------------

/// C0408: `SearchSkillsTool`(`search_skills`) — searches for relevant
/// skills in the registry.
pub struct SearchSkillsTool {
    core: Arc<SkillCoreState>,
    description: String,
}

impl SearchSkillsTool {
    /// `None` if `core.registry` isn't configured — matches the source's
    /// `raise ValueError("SearchSkillsTool requires a configured skill
    /// registry.")`, translated to a constructor that simply can't be
    /// called successfully without one.
    fn new(core: Arc<SkillCoreState>) -> Option<Self> {
        let description = core
            .registry
            .as_ref()?
            .search_tool_description()
            .unwrap_or_else(|| {
                "Searches for relevant skills in the registry based on a semantic or keyword \
                 query."
                    .to_string()
            });
        Some(Self { core, description })
    }
}

impl BaseTool for SearchSkillsTool {
    fn name(&self) -> &str {
        "search_skills"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "query".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("string".to_string())),
                            (
                                "description".to_string(),
                                Value::String("Semantic or keyword search query.".to_string()),
                            ),
                        ]),
                    )]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("query".to_string())]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a std::collections::BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let query = match args.get("query") {
                Some(Value::String(q)) if !q.is_empty() => q.clone(),
                _ => {
                    return Ok(error_response(
                        "Argument 'query' is required.",
                        "INVALID_ARGUMENTS",
                    ))
                }
            };

            let Some(registry) = &self.core.registry else {
                return Ok(error_response(
                    "Failed to search skills from registry: no registry configured.",
                    "REGISTRY_ERROR",
                ));
            };

            let results = registry.search_skills(&query).await;
            let formatted: Vec<Value> = results
                .into_iter()
                .filter(|frontmatter| !self.core.skills.contains_key(&frontmatter.name))
                .map(|frontmatter| rusty_serde::json::to_value(&frontmatter).unwrap_or(Value::Null))
                .collect();
            Ok(Value::Seq(formatted))
        })
    }
}

// ---------------------------------------------------------------------
// LoadSkillTool
// ---------------------------------------------------------------------

/// C0408 + C0401: `LoadSkillTool`(`load_skill`) — loads a skill's
/// `SKILL.md` instructions, tracks it as activated in agent state, and
/// (per C0401) interpolates session state into the instructions when
/// `Frontmatter.metadata["adk_inject_state"]` is set.
pub struct LoadSkillTool {
    core: Arc<SkillCoreState>,
}

impl LoadSkillTool {
    fn new(core: Arc<SkillCoreState>) -> Self {
        Self { core }
    }
}

impl BaseTool for LoadSkillTool {
    fn name(&self) -> &str {
        "load_skill"
    }

    fn description(&self) -> &str {
        "Loads the SKILL.md instructions for a given skill."
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "skill_name".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("string".to_string())),
                            (
                                "description".to_string(),
                                Value::String("The name of the skill to load.".to_string()),
                            ),
                        ]),
                    )]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("skill_name".to_string())]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a std::collections::BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let skill_name = match args.get("skill_name") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                _ => {
                    return Ok(error_response(
                        "Argument 'skill_name' is required.",
                        "INVALID_ARGUMENTS",
                    ))
                }
            };

            let invocation_id = tool_context.invocation_context().invocation_id.clone();
            let skill = match self
                .core
                .get_or_fetch_skill(&skill_name, &invocation_id)
                .await
            {
                Ok(skill) => skill,
                Err(e) => {
                    return Ok(error_response(
                        format!("Failed to fetch skill '{skill_name}' from registry: {e}"),
                        "REGISTRY_ERROR",
                    ))
                }
            };
            let Some(skill) = skill else {
                return Ok(error_response(
                    format!("Skill '{skill_name}' not found."),
                    "SKILL_NOT_FOUND",
                ));
            };

            // Record skill activation in agent state for tool resolution.
            let state_key = format!("_adk_activated_skill_{}", agent_name(tool_context));
            let mut activated: Vec<String> = match tool_context.state().get(&state_key) {
                Some(Value::Seq(items)) => items
                    .iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            if !activated.iter().any(|s| s == &skill_name) {
                activated.push(skill_name.clone());
                tool_context.state_mut().set(
                    state_key,
                    Value::Seq(activated.into_iter().map(Value::String).collect()),
                );
            }

            let mut instructions = skill.instructions.clone();
            if matches!(
                skill.frontmatter.metadata.get("adk_inject_state"),
                Some(Value::Bool(true))
            ) {
                let readonly = readonly_context_with_current_state(tool_context);
                instructions = match inject_session_state(&instructions, &readonly) {
                    Ok(rendered) => rendered,
                    Err(e) => {
                        return Ok(error_response(
                            format!("Failed to inject session state: {e}"),
                            "STATE_INJECTION_ERROR",
                        ))
                    }
                };
            }

            Ok(Value::Map(vec![
                ("skill_name".to_string(), Value::String(skill_name)),
                ("instructions".to_string(), Value::String(instructions)),
                (
                    "frontmatter".to_string(),
                    rusty_serde::json::to_value(&skill.frontmatter).unwrap_or(Value::Null),
                ),
            ]))
        })
    }
}

// ---------------------------------------------------------------------
// LoadSkillResourceTool
// ---------------------------------------------------------------------

/// C0409: `LoadSkillResourceTool`(`load_skill_resource`) — reads a
/// skill's `references/`/`assets/`/`scripts/` file.
pub struct LoadSkillResourceTool {
    core: Arc<SkillCoreState>,
}

impl LoadSkillResourceTool {
    fn new(core: Arc<SkillCoreState>) -> Self {
        Self { core }
    }

    fn resolve_content(skill: &Skill, file_path: &str) -> Result<Option<ResourceContent>, Value> {
        if let Some(name) = file_path.strip_prefix("references/") {
            Ok(skill.resources.get_reference(name).cloned())
        } else if let Some(name) = file_path.strip_prefix("assets/") {
            Ok(skill.resources.get_asset(name).cloned())
        } else if let Some(name) = file_path.strip_prefix("scripts/") {
            Ok(skill
                .resources
                .get_script(name)
                .map(|script| ResourceContent::Text(script.src.clone())))
        } else {
            Err(error_response(
                "Path must start with 'references/', 'assets/', or 'scripts/'.",
                "INVALID_RESOURCE_PATH",
            ))
        }
    }
}

impl BaseTool for LoadSkillResourceTool {
    fn name(&self) -> &str {
        "load_skill_resource"
    }

    fn description(&self) -> &str {
        "Loads a resource file (from references/, assets/, or scripts/) from within a skill."
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![
                        (
                            "skill_name".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String("The name of the skill.".to_string()),
                                ),
                            ]),
                        ),
                        (
                            "file_path".to_string(),
                            Value::Map(vec![
                                ("type".to_string(), Value::String("string".to_string())),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "The relative path to the resource (e.g., \
                                         'references/my_doc.md', 'assets/template.txt', or \
                                         'scripts/setup.sh')."
                                            .to_string(),
                                    ),
                                ),
                            ]),
                        ),
                    ]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![
                        Value::String("skill_name".to_string()),
                        Value::String("file_path".to_string()),
                    ]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a std::collections::BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let skill_name = args.get("skill_name").and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });
            let file_path = args.get("file_path").and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });
            let (skill_name, file_path) = match (skill_name, file_path) {
                (Some(s), Some(f)) => (s, f),
                (skill_name, file_path) => {
                    let mut errors = Vec::new();
                    if skill_name.is_none() {
                        errors.push("Argument 'skill_name' is required.");
                    }
                    if file_path.is_none() {
                        errors.push("Argument 'file_path' is required.");
                    }
                    return Ok(error_response(errors.join("\n"), "INVALID_ARGUMENTS"));
                }
            };

            let invocation_id = tool_context.invocation_context().invocation_id.clone();
            let skill = match self
                .core
                .get_or_fetch_skill(&skill_name, &invocation_id)
                .await
            {
                Ok(skill) => skill,
                Err(e) => {
                    return Ok(error_response(
                        format!("Failed to fetch skill '{skill_name}' from registry: {e}"),
                        "REGISTRY_ERROR",
                    ))
                }
            };
            let Some(skill) = skill else {
                return Ok(error_response(
                    format!("Skill '{skill_name}' not found."),
                    "SKILL_NOT_FOUND",
                ));
            };

            let content = match Self::resolve_content(&skill, &file_path) {
                Ok(content) => content,
                Err(response) => return Ok(response),
            };

            let Some(content) = content else {
                // Invocation-scoped failure counter. Counts RESOURCE_NOT_FOUND across
                // ALL paths so the guard fires even when the LLM hallucinates a
                // different path on each retry. The `temp:` prefix prevents
                // persistence to durable session storage; invocation_id isolates
                // in-memory backends.
                let counter_key =
                    format!("temp:_adk_skill_resource_not_found_count_{invocation_id}");
                let fail_count = increment_failure_counter(tool_context, &counter_key);
                if fail_count > 1 {
                    return Ok(error_response(
                        format!(
                            "Resource '{file_path}' not found in skill '{skill_name}'. This is \
                             resource lookup failure #{fail_count} this invocation. Do not \
                             retry any path — report the error to the user and stop."
                        ),
                        "RESOURCE_NOT_FOUND_FATAL",
                    ));
                }
                return Ok(error_response(
                    format!("Resource '{file_path}' not found in skill '{skill_name}'."),
                    "RESOURCE_NOT_FOUND",
                ));
            };

            match content {
                ResourceContent::Bytes(_) => Ok(Value::Map(vec![
                    ("skill_name".to_string(), Value::String(skill_name)),
                    ("file_path".to_string(), Value::String(file_path)),
                    (
                        "status".to_string(),
                        Value::String(BINARY_FILE_DETECTED_MSG.to_string()),
                    ),
                ])),
                ResourceContent::Text(text) => Ok(Value::Map(vec![
                    ("skill_name".to_string(), Value::String(skill_name)),
                    ("file_path".to_string(), Value::String(file_path)),
                    ("content".to_string(), Value::String(text)),
                ])),
            }
        })
    }

    /// Injects binary content into the LLM request if the model viewed
    /// it in the previous turn — see this module's own doc for the
    /// context on `ResourceContent::Bytes` support.
    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                crate::append_tools::merge_declarations(
                    llm_request,
                    [(self.name().to_string(), declaration)],
                );
            }

            let Some(last) = llm_request.contents.last() else {
                return;
            };

            let mut to_inject: Vec<(String, String, Vec<u8>)> = Vec::new();
            for part in &last.parts {
                let Some(function_response) = &part.function_response else {
                    continue;
                };
                if function_response.name.as_deref() != Some(self.name()) {
                    continue;
                }
                let Some(response) = &function_response.response else {
                    continue;
                };
                if response.get("status")
                    != Some(&Value::String(BINARY_FILE_DETECTED_MSG.to_string()))
                {
                    continue;
                }
                let (Some(Value::String(skill_name)), Some(Value::String(file_path))) =
                    (response.get("skill_name"), response.get("file_path"))
                else {
                    continue;
                };

                let invocation_id = tool_context.invocation_context().invocation_id.clone();
                let skill = match self
                    .core
                    .get_or_fetch_skill(skill_name, &invocation_id)
                    .await
                {
                    Ok(Some(skill)) => skill,
                    _ => continue,
                };
                let content = match Self::resolve_content(&skill, file_path) {
                    Ok(Some(ResourceContent::Bytes(bytes))) => bytes,
                    _ => continue,
                };
                to_inject.push((skill_name.clone(), file_path.clone(), content));
            }

            for (_, file_path, content) in to_inject {
                let mime_type = guess_mime_type(&file_path);
                llm_request.contents.push(Content::new(
                    "user",
                    vec![
                        Part::text(format!("The content of binary file '{file_path}' is:")),
                        Part {
                            inline_data: Some(MediaBlobStub {
                                mime_type: Some(mime_type.to_string()),
                                rest: Some(Value::Map(vec![(
                                    "data".to_string(),
                                    Value::String(base64_encode(&content)),
                                )])),
                            }),
                            ..Default::default()
                        },
                    ],
                ));
            }
        })
    }
}

// ---------------------------------------------------------------------
// SkillScriptCodeExecutor: `_SkillScriptCodeExecutor`'s Python-wrapper-
// generation path (the `code_executor`-configured branch of C0410).
// ---------------------------------------------------------------------

/// Python `str.__repr__`-equivalent — a single-quoted literal, escaping
/// backslash/quote/control characters so the result is valid Python
/// source that parses back to exactly this string. **Not** byte-identical
/// to CPython's `repr()` (which adaptively picks the quote character and
/// may render some characters differently) — only guaranteed
/// round-trip-correct, the same "reasonable, not byte-identical"
/// treatment already given to `value_to_display_string` elsewhere in
/// this port.
fn python_str_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// Python `bytes.__repr__`-equivalent — see [`python_str_literal`]'s doc
/// for the same round-trip-correct-not-byte-identical caveat.
fn python_bytes_literal(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 3);
    out.push_str("b'");
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push('\'');
    out
}

/// Python `list[str].__repr__`-equivalent.
fn python_list_literal(items: &[String]) -> String {
    let reprs: Vec<String> = items.iter().map(|s| python_str_literal(s)).collect();
    format!("[{}]", reprs.join(", "))
}

/// Python `dict[str, str | bytes].__repr__`-equivalent — `_build_wrapper_code`'s
/// `{files_dict!r}` embedding.
fn python_dict_literal(entries: &[(String, ResourceContent)]) -> String {
    let parts: Vec<String> = entries
        .iter()
        .map(|(key, value)| {
            let value_repr = match value {
                ResourceContent::Text(s) => python_str_literal(s),
                ResourceContent::Bytes(b) => python_bytes_literal(b),
            };
            format!("{}: {}", python_str_literal(key), value_repr)
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// `type(x).__name__`-equivalent, for the `script_args`/`short_options`/
/// `positional_args` type-validation error messages — approximate, not
/// distinguishing every Python type precisely (e.g. `tuple` vs `list`),
/// since `rusty_serde::value::Value`'s variants don't map onto Python's
/// type system 1:1.
fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::UInt(_) => "int",
        Value::Float(_) => "float",
        Value::String(_) => "str",
        Value::Seq(_) => "list",
        Value::Map(_) => "dict",
    }
}

/// Python `str(v)`-equivalent for a JSON-sourced value — used to render
/// `script_args`/`short_options` values into `--flag value` argv
/// entries. Same shape (and same disclosed non-byte-identical `float`/
/// `list`/`dict` rendering) as `skill_instructions_utils::value_to_display_string`,
/// duplicated locally rather than shared — this is the only other call
/// site in this crate, not worth a shared helper.
fn python_str_of_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::UInt(u) => u.to_string(),
        Value::Float(_) | Value::Seq(_) | Value::Map(_) => {
            rusty_serde::json::to_string(value).unwrap_or_default()
        }
    }
}

/// `script_args: dict[str, Any] | list[str] | None` — see the module's
/// use of this in `RunSkillScriptTool::run_async`'s validation.
enum ScriptArgsValue {
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
}

/// Builds the `--{k} {v}`/`-{k} {v}`/positional argv entries shared by
/// both the `.py` and `.sh`/`.bash` wrapper branches — `_build_wrapper_code`'s
/// `argv_list`/`arr` construction (identical shape in both branches; the
/// source's cosmetic `str(v) for v in positional_args` vs. bare
/// `positional_args` difference between the two branches has no
/// behavioral effect, since `positional_args` is always `list[str]` by
/// the point this runs — validated before either branch is reached).
fn build_argv(
    script_args: Option<&ScriptArgsValue>,
    short_options: Option<&[(String, Value)]>,
    positional_args: Option<&[String]>,
) -> Vec<String> {
    let mut argv = Vec::new();
    match script_args {
        Some(ScriptArgsValue::List(items)) => {
            argv.extend(items.iter().map(python_str_of_value));
        }
        _ => {
            if let Some(ScriptArgsValue::Map(entries)) = script_args {
                for (k, v) in entries {
                    argv.push(format!("--{k}"));
                    argv.push(python_str_of_value(v));
                }
            }
            if let Some(entries) = short_options {
                for (k, v) in entries {
                    argv.push(format!("-{k}"));
                    argv.push(python_str_of_value(v));
                }
            }
            if let Some(positional) = positional_args {
                if !positional.is_empty() {
                    argv.push("--".to_string());
                    argv.extend(positional.iter().cloned());
                }
            }
        }
    }
    argv
}

fn skill_files_dict(skill: &Skill) -> Vec<(String, ResourceContent)> {
    let mut files = Vec::new();
    for ref_name in skill.resources.list_references() {
        if let Some(content) = skill.resources.get_reference(ref_name) {
            files.push((format!("references/{ref_name}"), content.clone()));
        }
    }
    for asset_name in skill.resources.list_assets() {
        if let Some(content) = skill.resources.get_asset(asset_name) {
            files.push((format!("assets/{asset_name}"), content.clone()));
        }
    }
    for script_name in skill.resources.list_scripts() {
        if let Some(script) = skill.resources.get_script(script_name) {
            files.push((
                format!("scripts/{script_name}"),
                ResourceContent::Text(script.src.clone()),
            ));
        }
    }
    files
}

/// C0410: `_SkillScriptCodeExecutor` — materializes a skill's files and
/// executes a `.py`/`.sh`/`.bash` script against a [`BaseCodeExecutor`]
/// by generating a self-extracting Python wrapper script (embedding every
/// resource as a Python literal, extracting to a temp dir, then either
/// `runpy.run_path`-ing the target `.py` file or `subprocess.run`-ing it
/// through `bash` with a JSON-envelope result).
///
/// **`except SystemExit`, dead code for this port's only concrete
/// executor, disclosed**: the source's `execute_script_async` catches a
/// `SystemExit` the underlying executor might raise back into the
/// *calling* Python process — meaningful only for an in-process-`exec`-
/// based `BaseCodeExecutor`. This port's only concrete implementor,
/// `UnsafeLocalCodeExecutor` (C0385), always runs the wrapper as a real
/// subprocess — a subprocess's own uncaught `SystemExit` just becomes
/// its process exit status, never a Rust-visible exception, and
/// `CodeExecutionResult` (C0391) carries no exit-code field to inspect
/// either. Not reproduced: there is nothing this port's executors could
/// raise into that branch.
struct SkillScriptCodeExecutor {
    base_executor: Arc<dyn BaseCodeExecutor + Send + Sync>,
    script_timeout: Duration,
}

impl SkillScriptCodeExecutor {
    fn new(
        base_executor: Arc<dyn BaseCodeExecutor + Send + Sync>,
        script_timeout: Duration,
    ) -> Self {
        Self {
            base_executor,
            script_timeout,
        }
    }

    /// `_SkillScriptCodeExecutor._build_wrapper_code`. `None` for an
    /// unsupported extension (matches the source's `.py`/`.sh`/`.bash`-only
    /// support, surfaced by the caller as `UNSUPPORTED_SCRIPT_TYPE`).
    fn build_wrapper_code(
        &self,
        skill: &Skill,
        file_path: &str,
        script_args: Option<&ScriptArgsValue>,
        short_options: Option<&[(String, Value)]>,
        positional_args: Option<&[String]>,
    ) -> Option<String> {
        let ext = if file_path.contains('.') {
            file_path
                .rsplit('.')
                .next()
                .unwrap_or_default()
                .to_lowercase()
        } else {
            String::new()
        };

        let file_path = if file_path.starts_with("scripts/") {
            file_path.to_string()
        } else {
            format!("scripts/{file_path}")
        };

        let files_dict = skill_files_dict(skill);
        let total_size: usize = files_dict
            .iter()
            .map(|(_, content)| match content {
                ResourceContent::Text(s) => s.len(),
                ResourceContent::Bytes(b) => b.len(),
            })
            .sum();
        if total_size > MAX_SKILL_PAYLOAD_BYTES {
            eprintln!(
                "Skill '{}' resources total {total_size} bytes, exceeding the recommended \
                 limit of {MAX_SKILL_PAYLOAD_BYTES} bytes.",
                skill.name()
            );
        }

        let mut code_lines: Vec<String> = vec![
            "import os".to_string(),
            "import tempfile".to_string(),
            "import sys".to_string(),
            "import json as _json".to_string(),
            "import subprocess".to_string(),
            "import runpy".to_string(),
            format!("_files = {}", python_dict_literal(&files_dict)),
            "def _materialize_and_run():".to_string(),
            "  _orig_cwd = os.getcwd()".to_string(),
            "  with tempfile.TemporaryDirectory() as td:".to_string(),
            "    for rel_path, content in _files.items():".to_string(),
            "      norm_rel = os.path.normpath(rel_path)".to_string(),
            "      if norm_rel.startswith('..') or os.path.isabs(norm_rel):".to_string(),
            "        raise PermissionError('Path traversal blocked in skill file: ' + rel_path)"
                .to_string(),
            "      full_path = os.path.join(os.path.abspath(td), norm_rel)".to_string(),
            "      os.makedirs(os.path.dirname(full_path), exist_ok=True)".to_string(),
            "      mode = 'wb' if isinstance(content, bytes) else 'w'".to_string(),
            "      with open(full_path, mode, encoding='utf-8' if mode == 'w' else None) as f:"
                .to_string(),
            "        f.write(content)".to_string(),
            "    os.chdir(td)".to_string(),
            "    try:".to_string(),
        ];

        if ext == "py" {
            let mut argv_list = vec![file_path.clone()];
            argv_list.extend(build_argv(script_args, short_options, positional_args));
            code_lines.push(format!(
                "      sys.argv = {}",
                python_list_literal(&argv_list)
            ));
            code_lines.push(format!(
                "      sys.path.insert(0, os.path.dirname(os.path.abspath({})))",
                python_str_literal(&file_path)
            ));
            code_lines.push("      try:".to_string());
            code_lines.push(format!(
                "        runpy.run_path({}, run_name='__main__')",
                python_str_literal(&file_path)
            ));
            code_lines.push("      except SystemExit as e:".to_string());
            code_lines.push("        if e.code is not None and e.code != 0:".to_string());
            code_lines.push("          raise e".to_string());
        } else if ext == "sh" || ext == "bash" {
            let mut arr = vec!["bash".to_string(), file_path.clone()];
            arr.extend(build_argv(script_args, short_options, positional_args));
            let timeout_secs = self.script_timeout.as_secs();
            code_lines.push("      try:".to_string());
            code_lines.push("        _r = subprocess.run(".to_string());
            code_lines.push(format!("          {},", python_list_literal(&arr)));
            code_lines.push("          capture_output=True, text=True,".to_string());
            code_lines.push("          encoding='utf-8', errors='replace',".to_string());
            code_lines.push(format!("          timeout={timeout_secs}, cwd=td,"));
            code_lines.push("        )".to_string());
            code_lines.push("        print(_json.dumps({".to_string());
            code_lines.push("            '__shell_result__': True,".to_string());
            code_lines.push("            'stdout': _r.stdout,".to_string());
            code_lines.push("            'stderr': _r.stderr,".to_string());
            code_lines.push("            'returncode': _r.returncode,".to_string());
            code_lines.push("        }))".to_string());
            code_lines.push("      except subprocess.TimeoutExpired as _e:".to_string());
            code_lines.push("        print(_json.dumps({".to_string());
            code_lines.push("            '__shell_result__': True,".to_string());
            code_lines.push("            'stdout': _e.stdout or '',".to_string());
            code_lines.push(format!(
                "            'stderr': 'Timed out after {timeout_secs}s',"
            ));
            code_lines.push("            'returncode': -1,".to_string());
            code_lines.push("            'timeout': True,".to_string());
            code_lines.push("        }))".to_string());
        } else {
            return None;
        }

        code_lines.push("    finally:".to_string());
        code_lines.push("      os.chdir(_orig_cwd)".to_string());
        code_lines.push("_materialize_and_run()".to_string());
        Some(code_lines.join("\n"))
    }

    /// `_SkillScriptCodeExecutor.execute_script_async`.
    async fn execute_script_async(
        &self,
        invocation_context: &adk_agents::invocation_context::InvocationContext,
        skill: &Skill,
        file_path: &str,
        script_args: Option<&ScriptArgsValue>,
        short_options: Option<&[(String, Value)]>,
        positional_args: Option<&[String]>,
    ) -> Value {
        let Some(code) = self.build_wrapper_code(
            skill,
            file_path,
            script_args,
            short_options,
            positional_args,
        ) else {
            let ext_msg = if let Some((_, ext)) = file_path.rsplit_once('.') {
                format!("'.{ext}'")
            } else {
                "(no extension)".to_string()
            };
            return error_response(
                format!("Unsupported script type {ext_msg}. Supported types: .py, .sh, .bash"),
                "UNSUPPORTED_SCRIPT_TYPE",
            );
        };

        let input = CodeExecutionInput {
            code,
            input_files: Vec::new(),
            execution_id: None,
        };

        // `asyncio.to_thread`-equivalent: `execute_code` is synchronous.
        let executor = self.base_executor.clone();
        let invocation_context_owned = invocation_context.clone();
        let outcome = rusty_tokio::spawn_blocking(move || {
            executor.execute_code(&invocation_context_owned, &input)
        })
        .await;

        let result = match outcome {
            Ok(result) => result,
            Err(join_error) => {
                return error_response(
                    format!("Failed to execute script '{file_path}':\n{join_error}"),
                    "EXECUTION_ERROR",
                )
            }
        };

        let mut stdout = result.stdout;
        let mut stderr = result.stderr;
        let mut return_code: i64 = 0;

        let extension = file_path.rsplit('.').next().unwrap_or("").to_lowercase();
        let is_shell = file_path.contains('.') && (extension == "sh" || extension == "bash");
        if is_shell && !stdout.is_empty() {
            if let Ok(Value::Map(fields)) = rusty_serde::json::from_str::<Value>(&stdout) {
                let get = |key: &str| {
                    fields
                        .iter()
                        .find(|(k, _)| k == key)
                        .map(|(_, v)| v.clone())
                };
                let is_shell_result = matches!(get("__shell_result__"), Some(Value::Bool(true)));
                if is_shell_result {
                    stdout = match get("stdout") {
                        Some(Value::String(s)) => s,
                        _ => String::new(),
                    };
                    stderr = match get("stderr") {
                        Some(Value::String(s)) => s,
                        _ => String::new(),
                    };
                    return_code = match get("returncode") {
                        Some(Value::Int(n)) => n,
                        Some(Value::UInt(n)) => n as i64,
                        _ => 0,
                    };
                    let timed_out = matches!(get("timeout"), Some(Value::Bool(true)));
                    if return_code != 0 && !timed_out {
                        let exit_code_message = format!("Exit code {return_code}");
                        stderr = if stderr.is_empty() {
                            exit_code_message
                        } else {
                            format!("{}\n{exit_code_message}", stderr.trim_end())
                        };
                    }
                }
            }
        }

        let status = if return_code != 0 || (!stderr.is_empty() && stdout.is_empty()) {
            "error"
        } else if !stderr.is_empty() {
            "warning"
        } else {
            "success"
        };

        Value::Map(vec![
            (
                "skill_name".to_string(),
                Value::String(skill.name().to_string()),
            ),
            (
                "file_path".to_string(),
                Value::String(file_path.to_string()),
            ),
            ("stdout".to_string(), Value::String(stdout)),
            ("stderr".to_string(), Value::String(stderr)),
            ("status".to_string(), Value::String(status.to_string())),
        ])
    }
}

// ---------------------------------------------------------------------
// RunSkillScriptTool
// ---------------------------------------------------------------------

/// C0410: `RunSkillScriptTool` (`run_skill_script`) — executes a script
/// from a skill's `scripts/` directory, against either an `environment`
/// or a `code_executor`.
pub struct RunSkillScriptTool {
    core: Arc<SkillCoreState>,
}

impl RunSkillScriptTool {
    fn new(core: Arc<SkillCoreState>) -> Self {
        Self { core }
    }

    /// `RunSkillScriptTool._ensure_skill_materialized_in_env` — JIT
    /// materialization: writes every reference/asset/script into the
    /// environment under `<skills_folder>/<skill_name>/...` the first
    /// time this skill's script is run, if it isn't there already.
    async fn ensure_materialized(
        &self,
        skill: &Skill,
        file_path: &str,
        env: &Arc<dyn BaseEnvironment>,
    ) -> Result<(), String> {
        let Some(skills_folder) = self.core.skills_folder() else {
            return Err(
                "skills_folder is not set and no environment working_dir available.".to_string(),
            );
        };
        let skill_dir = format!("{}/{}", skills_folder.display(), skill.name());
        let rel_script = if file_path.starts_with("scripts/") {
            file_path.to_string()
        } else {
            format!("scripts/{file_path}")
        };
        let script_path = format!("{skill_dir}/{rel_script}");

        let script_exists = env.read_file(&script_path).await.is_ok();
        if script_exists {
            return Ok(());
        }

        for ref_name in skill.resources.list_references() {
            if let Some(ResourceContent::Text(content)) = skill.resources.get_reference(ref_name) {
                let path = format!("{skill_dir}/references/{ref_name}");
                env.write_file(&path, content.as_bytes())
                    .await
                    .map_err(|e| e.to_string())?;
            } else if let Some(ResourceContent::Bytes(content)) =
                skill.resources.get_reference(ref_name)
            {
                let path = format!("{skill_dir}/references/{ref_name}");
                env.write_file(&path, content)
                    .await
                    .map_err(|e| e.to_string())?;
            }
        }
        for asset_name in skill.resources.list_assets() {
            match skill.resources.get_asset(asset_name) {
                Some(ResourceContent::Text(content)) => {
                    let path = format!("{skill_dir}/assets/{asset_name}");
                    env.write_file(&path, content.as_bytes())
                        .await
                        .map_err(|e| e.to_string())?;
                }
                Some(ResourceContent::Bytes(content)) => {
                    let path = format!("{skill_dir}/assets/{asset_name}");
                    env.write_file(&path, content)
                        .await
                        .map_err(|e| e.to_string())?;
                }
                None => {}
            }
        }
        for script_name in skill.resources.list_scripts() {
            if let Some(script) = skill.resources.get_script(script_name) {
                let path = format!("{skill_dir}/scripts/{script_name}");
                env.write_file(&path, script.src.as_bytes())
                    .await
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

impl BaseTool for RunSkillScriptTool {
    fn name(&self) -> &str {
        "run_skill_script"
    }

    fn description(&self) -> &str {
        "Executes a script from a skill's scripts/ directory."
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        let properties = if self.core.env.is_some() {
            vec![
                (
                    "skill_name".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String("The name of the skill.".to_string()),
                        ),
                    ]),
                ),
                (
                    "file_path".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String(
                                "The relative path to the script (e.g., 'scripts/setup.py')."
                                    .to_string(),
                            ),
                        ),
                    ]),
                ),
                (
                    "command".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String("The command to execute in the environment.".to_string()),
                        ),
                    ]),
                ),
            ]
        } else {
            vec![
                (
                    "skill_name".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String("The name of the skill.".to_string()),
                        ),
                    ]),
                ),
                (
                    "file_path".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "description".to_string(),
                            Value::String(
                                "The relative path to the script (e.g., 'scripts/setup.py')."
                                    .to_string(),
                            ),
                        ),
                    ]),
                ),
            ]
        };

        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters_json_schema: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                ("properties".to_string(), Value::Map(properties)),
                (
                    "required".to_string(),
                    if self.core.env.is_some() {
                        Value::Seq(vec![
                            Value::String("skill_name".to_string()),
                            Value::String("file_path".to_string()),
                            Value::String("command".to_string()),
                        ])
                    } else {
                        Value::Seq(vec![
                            Value::String("skill_name".to_string()),
                            Value::String("file_path".to_string()),
                        ])
                    },
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a std::collections::BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let skill_name = args.get("skill_name").and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });
            let file_path = args.get("file_path").and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });
            let (skill_name, file_path) = match (skill_name, file_path) {
                (Some(s), Some(f)) => (s, f),
                (skill_name, file_path) => {
                    let mut errors = Vec::new();
                    if skill_name.is_none() {
                        errors.push("Argument 'skill_name' is required.");
                    }
                    if file_path.is_none() {
                        errors.push("Argument 'file_path' is required.");
                    }
                    return Ok(error_response(errors.join("\n"), "INVALID_ARGUMENTS"));
                }
            };

            let command = args.get("command").and_then(|v| match v {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            });

            let mut script_args: Option<ScriptArgsValue> = None;
            let mut short_options: Option<Vec<(String, Value)>> = None;
            let mut positional_args: Option<Vec<String>> = None;

            if self.core.env.is_some() {
                if command.is_none() {
                    return Ok(error_response(
                        "Argument 'command' is required and must be a string.",
                        "INVALID_ARGUMENTS",
                    ));
                }
            } else {
                let mut errors: Vec<String> = Vec::new();

                match args.get("args") {
                    None | Some(Value::Null) => {}
                    Some(Value::Seq(items)) => {
                        script_args = Some(ScriptArgsValue::List(items.clone()))
                    }
                    Some(Value::Map(entries)) => {
                        script_args = Some(ScriptArgsValue::Map(entries.clone()))
                    }
                    Some(other) => errors.push(format!(
                        "'args' must be a JSON object (dict) or a list of strings, got {}.",
                        value_type_name(other)
                    )),
                }

                match args.get("short_options") {
                    None | Some(Value::Null) => {}
                    Some(Value::Map(entries)) => short_options = Some(entries.clone()),
                    Some(other) => errors.push(format!(
                        "'short_options' must be a JSON object (dict), got {}.",
                        value_type_name(other)
                    )),
                }

                match args.get("positional_args") {
                    None | Some(Value::Null) => {}
                    Some(Value::Seq(items)) => {
                        positional_args = Some(items.iter().map(python_str_of_value).collect())
                    }
                    Some(other) => errors.push(format!(
                        "'positional_args' must be a list of strings, got {}.",
                        value_type_name(other)
                    )),
                }

                if matches!(script_args, Some(ScriptArgsValue::List(_)))
                    && (short_options.is_some() || positional_args.is_some())
                {
                    errors.push(
                        "Cannot specify 'short_options' or 'positional_args' when 'args' is a \
                         list."
                            .to_string(),
                    );
                }

                if !errors.is_empty() {
                    return Ok(error_response(errors.join("\n"), "INVALID_ARGUMENTS"));
                }
            }

            let invocation_id = tool_context.invocation_context().invocation_id.clone();
            let skill = match self
                .core
                .get_or_fetch_skill(&skill_name, &invocation_id)
                .await
            {
                Ok(skill) => skill,
                Err(e) => {
                    return Ok(error_response(
                        format!("Failed to fetch skill '{skill_name}' from registry: {e}"),
                        "REGISTRY_ERROR",
                    ))
                }
            };
            let Some(skill) = skill else {
                return Ok(error_response(
                    format!("Skill '{skill_name}' not found."),
                    "SKILL_NOT_FOUND",
                ));
            };

            let script_key = file_path.strip_prefix("scripts/").unwrap_or(&file_path);
            if skill.resources.get_script(script_key).is_none() {
                // Invocation-scoped failure counter -- see LoadSkillResourceTool's
                // identical guard for the `temp:` prefix / invocation_id rationale.
                let counter_key = format!("temp:_adk_skill_script_not_found_count_{invocation_id}");
                let fail_count = increment_failure_counter(tool_context, &counter_key);
                if fail_count > 1 {
                    return Ok(error_response(
                        format!(
                            "Script '{file_path}' not found in skill '{skill_name}'. This is \
                             script lookup failure #{fail_count} this invocation. Do not retry \
                             any script path — report the error to the user and stop."
                        ),
                        "SCRIPT_NOT_FOUND_FATAL",
                    ));
                }
                return Ok(error_response(
                    format!("Script '{file_path}' not found in skill '{skill_name}'."),
                    "SCRIPT_NOT_FOUND",
                ));
            }

            if let Some(env) = &self.core.env {
                if let Err(e) = self.ensure_materialized(&skill, &file_path, env).await {
                    return Ok(error_response(
                        format!("Failed to execute script '{file_path}' in environment:\n{e}"),
                        "EXECUTION_ERROR",
                    ));
                }

                return match env
                    .execute(&command.unwrap(), Some(self.core.script_timeout))
                    .await
                {
                    Ok(result) => Ok(Value::Map(vec![
                        ("stdout".to_string(), Value::String(result.stdout)),
                        ("stderr".to_string(), Value::String(result.stderr)),
                        ("exit_code".to_string(), Value::Int(result.exit_code as i64)),
                        ("timed_out".to_string(), Value::Bool(result.timed_out)),
                    ])),
                    Err(e) => Ok(error_response(
                        format!("Failed to execute script '{file_path}' in environment:\n{e}"),
                        "EXECUTION_ERROR",
                    )),
                };
            }

            let Some(code_executor) = self.core.code_executor.clone() else {
                return Ok(error_response(
                    "Neither Environment nor CodeExecutor is configured. An environment or \
                     code executor is required to run scripts.",
                    "NO_CODE_EXECUTOR",
                ));
            };

            let script_executor =
                SkillScriptCodeExecutor::new(code_executor, self.core.script_timeout);
            let invocation_context = tool_context.get_invocation_context();
            Ok(script_executor
                .execute_script_async(
                    &invocation_context,
                    &skill,
                    &file_path,
                    script_args.as_ref(),
                    short_options.as_deref(),
                    positional_args.as_deref(),
                )
                .await)
        })
    }
}

// ---------------------------------------------------------------------
// SkillToolset
// ---------------------------------------------------------------------

/// C0950: `ToolUnion`'s two Rust-expressible member kinds — see the
/// module doc for why the source's third (`callable`, wrapped via
/// `FunctionTool(callable)`) has no port here.
pub enum AdditionalTool {
    Tool(Arc<dyn BaseTool>),
    Toolset(Arc<dyn BaseToolset>),
}

/// Configuration for [`SkillToolset::new`] — Python's keyword-argument
/// constructor collapsed into one struct, per this port's usual
/// many-optional-params convention.
pub struct SkillToolsetConfig {
    pub skills: Vec<Skill>,
    pub registry: Option<Arc<dyn SkillRegistry>>,
    pub environment: Option<Arc<dyn BaseEnvironment>>,
    /// C0410: mutually exclusive with `environment` (checked in
    /// `SkillToolset::new`, matching the source's own `raise ValueError`).
    pub code_executor: Option<Arc<dyn BaseCodeExecutor + Send + Sync>>,
    pub skills_folder: Option<PathBuf>,
    pub script_timeout: Duration,
    /// C0950: tools/toolsets made available when an activated skill's
    /// `Frontmatter.metadata["adk_additional_tools"]` names them.
    pub additional_tools: Vec<AdditionalTool>,
    pub tool_name_prefix: Option<String>,
    pub tool_filter: Option<ToolFilter>,
}

impl Default for SkillToolsetConfig {
    fn default() -> Self {
        Self {
            skills: Vec::new(),
            registry: None,
            environment: None,
            code_executor: None,
            skills_folder: None,
            script_timeout: DEFAULT_SCRIPT_TIMEOUT,
            additional_tools: Vec::new(),
            tool_name_prefix: None,
            tool_filter: None,
        }
    }
}

/// C0408: a toolset for managing and interacting with agent skills.
pub struct SkillToolset {
    core: Arc<SkillCoreState>,
    tools: Vec<Arc<dyn BaseTool>>,
    provided_tools_by_name: HashMap<String, Arc<dyn BaseTool>>,
    provided_toolsets: Vec<Arc<dyn BaseToolset>>,
    tool_name_prefix: Option<String>,
    tool_filter: Option<ToolFilter>,
    prefix_cache: Mutex<PrefixCache>,
}

impl SkillToolset {
    pub fn new(config: SkillToolsetConfig) -> Result<Self, String> {
        let mut skills = HashMap::new();
        for skill in config.skills {
            if skills.contains_key(skill.name()) {
                return Err(format!("Duplicate skill name '{}'.", skill.name()));
            }
            skills.insert(skill.name().to_string(), skill);
        }

        if config.code_executor.is_some() && config.environment.is_some() {
            return Err("Cannot have both code_executor and environment".to_string());
        }

        if let Some(folder) = &config.skills_folder {
            if config.environment.is_none() {
                return Err("Cannot specify skills_folder without an environment".to_string());
            }
            // Disclosed narrowing: the source checks both `PurePosixPath`/
            // `PureWindowsPath` regardless of host OS; this port checks the
            // host-native `Path::is_absolute` only.
            if !folder.is_absolute() {
                return Err(format!(
                    "`skills_folder` must be an absolute path: '{}'",
                    folder.display()
                ));
            }
        }

        let mut provided_tools_by_name = HashMap::new();
        let mut provided_toolsets = Vec::new();
        for tool in config.additional_tools {
            match tool {
                AdditionalTool::Tool(tool) => {
                    provided_tools_by_name.insert(tool.name().to_string(), tool);
                }
                AdditionalTool::Toolset(toolset) => provided_toolsets.push(toolset),
            }
        }

        let core = Arc::new(SkillCoreState {
            skills,
            registry: config.registry,
            env: config.environment,
            code_executor: config.code_executor,
            skills_folder_override: config.skills_folder,
            script_timeout: config.script_timeout,
            fetched_skill_cache: Mutex::new(FetchedSkillCache::default()),
        });

        let mut tools: Vec<Arc<dyn BaseTool>> = vec![
            Arc::new(ListSkillsTool::new(core.clone())),
            Arc::new(LoadSkillTool::new(core.clone())),
            Arc::new(LoadSkillResourceTool::new(core.clone())),
            Arc::new(RunSkillScriptTool::new(core.clone())),
        ];
        if let Some(search_tool) = SearchSkillsTool::new(core.clone()) {
            tools.push(Arc::new(search_tool));
        }

        Ok(Self {
            core,
            tools,
            provided_tools_by_name,
            provided_toolsets,
            tool_name_prefix: config.tool_name_prefix,
            tool_filter: config.tool_filter,
            prefix_cache: Mutex::new(PrefixCache::new()),
        })
    }

    /// `SkillToolset.skills_folder`.
    pub fn skills_folder(&self) -> Option<PathBuf> {
        self.core.skills_folder()
    }

    /// `SkillToolset.skills`.
    pub fn skills(&self) -> Vec<Skill> {
        self.core.list_skills()
    }

    /// C0950: `SkillToolset.clone_with_updated_skills` — a new toolset
    /// with identical configuration but a different `skills` list. Note
    /// the source itself doesn't carry `tool_name_prefix`/`tool_filter`
    /// forward through this call (only `additional_tools`/`registry`/
    /// `code_executor`/`environment`/`skills_folder`/`script_timeout`),
    /// so neither does this port — a faithful port of that omission, not
    /// an oversight.
    pub fn clone_with_updated_skills(&self, skills: Vec<Skill>) -> Result<SkillToolset, String> {
        let mut additional_tools: Vec<AdditionalTool> = self
            .provided_tools_by_name
            .values()
            .cloned()
            .map(AdditionalTool::Tool)
            .collect();
        additional_tools.extend(
            self.provided_toolsets
                .iter()
                .cloned()
                .map(AdditionalTool::Toolset),
        );

        SkillToolset::new(SkillToolsetConfig {
            skills,
            registry: self.core.registry.clone(),
            environment: self.core.env.clone(),
            code_executor: self.core.code_executor.clone(),
            skills_folder: self.core.skills_folder_override.clone(),
            script_timeout: self.core.script_timeout,
            additional_tools,
            ..Default::default()
        })
    }

    /// `SkillToolset._resolve_additional_tools_from_state` — see the
    /// module doc for the `asyncio.gather(..., return_exceptions=True)`
    /// simplification: this port's `BaseToolset::get_tools_with_prefix`
    /// is already infallible, so there's no exception path to catch.
    async fn resolve_additional_tools_from_state(
        &self,
        readonly_context: Option<&ReadonlyContext>,
    ) -> Vec<Arc<dyn BaseTool>> {
        let Some(readonly_context) = readonly_context else {
            return Vec::new();
        };

        let state_key = format!("_adk_activated_skill_{}", readonly_context.agent_name());
        let activated_skills: Vec<String> = match readonly_context.state().get(&state_key) {
            Some(Value::Seq(items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        if activated_skills.is_empty() {
            return Vec::new();
        }

        // A `BTreeSet`, not Python's unordered `set()` -- deterministic
        // iteration order for tests; correctness doesn't depend on order
        // (every name is looked up independently below).
        let mut additional_tool_names: std::collections::BTreeSet<String> = Default::default();
        for skill_name in &activated_skills {
            if let Ok(Some(skill)) = self
                .core
                .get_or_fetch_skill(skill_name, readonly_context.invocation_id())
                .await
            {
                if let Some(Value::Seq(names)) =
                    skill.frontmatter.metadata.get("adk_additional_tools")
                {
                    for name in names {
                        if let Value::String(s) = name {
                            additional_tool_names.insert(s.clone());
                        }
                    }
                }
            }
        }
        if additional_tool_names.is_empty() {
            return Vec::new();
        }

        let mut candidate_tools: HashMap<String, Arc<dyn BaseTool>> =
            self.provided_tools_by_name.clone();
        for toolset in &self.provided_toolsets {
            for tool in toolset.get_tools_with_prefix(Some(readonly_context)).await {
                candidate_tools.insert(tool.name().to_string(), tool);
            }
        }

        let mut resolved = Vec::new();
        let mut existing_names: std::collections::HashSet<String> =
            self.tools.iter().map(|t| t.name().to_string()).collect();
        for name in &additional_tool_names {
            let Some(tool) = candidate_tools.get(name) else {
                continue;
            };
            if existing_names.contains(tool.name()) {
                // Name collision with a core tool -- skip, matching the
                // source's `logger.error(...); continue`.
                continue;
            }
            existing_names.insert(tool.name().to_string());
            resolved.push(tool.clone());
        }
        resolved
    }
}

impl BaseToolset for SkillToolset {
    fn get_tools<'a>(
        &'a self,
        readonly_context: Option<&'a ReadonlyContext>,
    ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
        Box::pin(async move {
            let dynamic_tools = self
                .resolve_additional_tools_from_state(readonly_context)
                .await;
            self.tools
                .iter()
                .cloned()
                .chain(dynamic_tools)
                .filter(|tool| self.is_tool_selected(tool.as_ref(), readonly_context))
                .collect()
        })
    }

    fn prefix_cache(&self) -> &Mutex<PrefixCache> {
        &self.prefix_cache
    }

    fn tool_filter(&self) -> Option<&ToolFilter> {
        self.tool_filter.as_ref()
    }

    fn tool_name_prefix(&self) -> Option<&str> {
        self.tool_name_prefix.as_deref()
    }

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(env) = &self.core.env {
                if !env.is_initialized() {
                    env.initialize()
                        .await
                        .expect("SkillToolset: failed to initialize environment");
                }
            }

            let mut instructions = vec![build_skill_system_instruction(
                self.tool_name_prefix(),
                self.core.skills_folder().as_deref(),
            )];

            if self.core.registry.is_some() {
                let p = self
                    .tool_name_prefix()
                    .map(|p| format!("{p}_"))
                    .unwrap_or_default();
                instructions.push(format!(
                    "\nIf the locally available skills are not sufficient to complete your \
                     task, you can use the `{p}search_skills` tool to discover additional \
                     skills from the registry."
                ));
            }

            llm_request.append_instructions(Instructions::Strings(instructions));
        })
    }

    fn close<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(env) = &self.core.env {
                if env.is_initialized() {
                    env.close().await;
                }
            }
            let mut cache = self.core.fetched_skill_cache.lock().unwrap();
            cache.order.clear();
            cache.entries.clear();
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_environment::LocalEnvironment;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    fn frontmatter(name: &str) -> crate::skills_models::Frontmatter {
        crate::skills_models::Frontmatter {
            name: name.to_string(),
            description: "A test skill.".to_string(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            metadata: HashMap::new(),
        }
    }

    fn skill(name: &str) -> Skill {
        Skill {
            frontmatter: frontmatter(name),
            instructions: format!("Instructions for {name}."),
            resources: Default::default(),
            uri: None,
        }
    }

    async fn run(
        tool: &dyn BaseTool,
        args: BTreeMap<String, Value>,
        context: &mut Context,
    ) -> Value {
        tool.run_async(&args, context).await.unwrap()
    }

    fn args_map(pairs: &[(&str, &str)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect()
    }

    // -------------------------------------------------------------
    // build_skill_system_instruction / DEFAULT_SKILL_SYSTEM_INSTRUCTION
    // -------------------------------------------------------------

    #[test]
    fn default_instruction_has_no_prefix_and_no_env_note() {
        let instruction = default_skill_system_instruction();
        assert!(instruction.contains("`load_skill`"));
        assert!(!instruction.contains("NOTE ON ENVIRONMENT EXECUTION"));
    }

    #[test]
    fn instruction_applies_the_prefix_and_env_note() {
        let instruction =
            build_skill_system_instruction(Some("ns"), Some(Path::new("/work/skills")));
        assert!(instruction.contains("`ns_load_skill`"));
        assert!(instruction.contains("NOTE ON ENVIRONMENT EXECUTION"));
        assert!(instruction.contains("/work/skills/<skill_name>/"));
    }

    // -------------------------------------------------------------
    // SkillToolset::new construction checks
    // -------------------------------------------------------------

    #[test]
    fn rejects_duplicate_skill_names() {
        let config = SkillToolsetConfig {
            skills: vec![skill("a"), skill("a")],
            ..Default::default()
        };
        let err = SkillToolset::new(config).err().unwrap();
        assert!(err.contains("Duplicate skill name"));
    }

    #[test]
    fn rejects_skills_folder_without_an_environment() {
        let config = SkillToolsetConfig {
            skills_folder: Some(PathBuf::from("/skills")),
            ..Default::default()
        };
        let err = SkillToolset::new(config).err().unwrap();
        assert!(err.contains("Cannot specify skills_folder without an environment"));
    }

    #[test]
    fn rejects_a_relative_skills_folder() {
        let config = SkillToolsetConfig {
            environment: Some(Arc::new(LocalEnvironment::new()) as Arc<dyn BaseEnvironment>),
            skills_folder: Some(PathBuf::from("relative/skills")),
            ..Default::default()
        };
        let err = SkillToolset::new(config).err().unwrap();
        assert!(err.contains("must be an absolute path"));
    }

    #[test]
    fn constructs_the_expected_tools_without_a_registry() {
        let config = SkillToolsetConfig {
            skills: vec![skill("a")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let names: Vec<&str> = toolset.tools.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            vec![
                "list_skills",
                "load_skill",
                "load_skill_resource",
                "run_skill_script"
            ]
        );
    }

    struct StubRegistry {
        skills: HashMap<String, Skill>,
        search_results: Vec<crate::skills_models::Frontmatter>,
        call_count: AtomicUsize,
    }

    impl SkillRegistry for StubRegistry {
        fn get_skill<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Skill, String>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                self.skills
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("no such skill: {name}"))
            })
        }

        fn search_skills<'a>(
            &'a self,
            _query: &'a str,
        ) -> BoxFuture<'a, Vec<crate::skills_models::Frontmatter>> {
            let results = self.search_results.clone();
            Box::pin(async move { results })
        }
    }

    #[test]
    fn constructs_search_skills_tool_when_a_registry_is_configured() {
        let config = SkillToolsetConfig {
            registry: Some(Arc::new(StubRegistry {
                skills: HashMap::new(),
                search_results: vec![],
                call_count: AtomicUsize::new(0),
            })),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        assert!(toolset.tools.iter().any(|t| t.name() == "search_skills"));
    }

    // -------------------------------------------------------------
    // ListSkillsTool
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn list_skills_returns_formatted_xml() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "list_skills")
            .unwrap();
        let result = run(tool.as_ref(), BTreeMap::new(), &mut ctx()).await;
        match result {
            Value::String(xml) => {
                assert!(xml.contains("alpha"));
                assert!(xml.contains("A test skill."));
            }
            other => panic!("expected a string, got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // SearchSkillsTool
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn search_skills_filters_out_locally_shadowed_skills() {
        let mut shadowed = frontmatter("alpha");
        shadowed.description = "Registry copy (shadowed).".to_string();
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            registry: Some(Arc::new(StubRegistry {
                skills: HashMap::new(),
                search_results: vec![shadowed, frontmatter("beta")],
                call_count: AtomicUsize::new(0),
            })),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "search_skills")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("query", "anything")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Seq(results) => assert_eq!(results.len(), 1),
            other => panic!("expected a seq, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn search_skills_requires_a_query() {
        let config = SkillToolsetConfig {
            registry: Some(Arc::new(StubRegistry {
                skills: HashMap::new(),
                search_results: vec![],
                call_count: AtomicUsize::new(0),
            })),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "search_skills")
            .unwrap();
        let result = run(tool.as_ref(), BTreeMap::new(), &mut ctx()).await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("INVALID_ARGUMENTS".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // LoadSkillTool
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn load_skill_requires_a_skill_name() {
        let toolset = SkillToolset::new(SkillToolsetConfig::default()).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill")
            .unwrap();
        let result = run(tool.as_ref(), BTreeMap::new(), &mut ctx()).await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("INVALID_ARGUMENTS".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_skill_reports_a_missing_skill() {
        let toolset = SkillToolset::new(SkillToolsetConfig::default()).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "missing")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("SKILL_NOT_FOUND".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_skill_returns_instructions_and_tracks_activation() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill")
            .unwrap();
        let mut context = ctx();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha")]),
            &mut context,
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "instructions").unwrap().1,
                    Value::String("Instructions for alpha.".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
        let state_key = format!("_adk_activated_skill_{}", agent_name(&context));
        assert_eq!(
            context.state().get(&state_key),
            Some(&Value::Seq(vec![Value::String("alpha".to_string())]))
        );
    }

    #[rusty_tokio::test]
    async fn load_skill_interpolates_state_when_adk_inject_state_is_set() {
        let mut injected = skill("alpha");
        injected.instructions = "Hello {user_name}!".to_string();
        injected
            .frontmatter
            .metadata
            .insert("adk_inject_state".to_string(), Value::Bool(true));
        let config = SkillToolsetConfig {
            skills: vec![injected],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill")
            .unwrap();
        let mut context = ctx();
        context
            .state_mut()
            .set("user_name", Value::String("Ada".to_string()));
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha")]),
            &mut context,
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "instructions").unwrap().1,
                    Value::String("Hello Ada!".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // LoadSkillResourceTool
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn load_skill_resource_requires_both_arguments() {
        let toolset = SkillToolset::new(SkillToolsetConfig::default()).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill_resource")
            .unwrap();
        let result = run(tool.as_ref(), BTreeMap::new(), &mut ctx()).await;
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(
                    matches!(&error.1, Value::String(s) if s.contains("skill_name") && s.contains("file_path"))
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_skill_resource_rejects_an_invalid_prefix() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill_resource")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha"), ("file_path", "bad/path.txt")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("INVALID_RESOURCE_PATH".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_skill_resource_returns_text_content() {
        let mut with_reference = skill("alpha");
        with_reference.resources.references.insert(
            "doc.md".to_string(),
            ResourceContent::Text("hello there".to_string()),
        );
        let config = SkillToolsetConfig {
            skills: vec![with_reference],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill_resource")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha"), ("file_path", "references/doc.md")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "content").unwrap().1,
                    Value::String("hello there".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn load_skill_resource_flags_binary_content_and_process_llm_request_injects_it() {
        let mut with_asset = skill("alpha");
        with_asset.resources.assets.insert(
            "logo.png".to_string(),
            ResourceContent::Bytes(vec![1, 2, 3]),
        );
        let config = SkillToolsetConfig {
            skills: vec![with_asset],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill_resource")
            .unwrap();
        let mut context = ctx();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha"), ("file_path", "assets/logo.png")]),
            &mut context,
        )
        .await;
        let response_map: BTreeMap<String, Value> = match &result {
            Value::Map(fields) => fields.iter().cloned().collect(),
            other => panic!("expected a map, got {other:?}"),
        };
        assert_eq!(
            response_map.get("status"),
            Some(&Value::String(BINARY_FILE_DETECTED_MSG.to_string()))
        );

        let mut request = LlmRequest::default();
        request.contents.push(Content::new(
            "model",
            vec![Part {
                function_response: Some(adk_genai::content::FunctionResponse {
                    id: None,
                    name: Some("load_skill_resource".to_string()),
                    response: Some(response_map),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        ));

        tool.process_llm_request(&mut context, &mut request).await;
        let injected = request.contents.last().unwrap();
        assert!(injected.parts.iter().any(|p| p.inline_data.is_some()));
    }

    #[rusty_tokio::test]
    async fn load_skill_resource_not_found_becomes_fatal_on_the_second_failure() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "load_skill_resource")
            .unwrap();
        let mut context = ctx();
        let args = args_map(&[
            ("skill_name", "alpha"),
            ("file_path", "references/missing.md"),
        ]);

        let first = run(tool.as_ref(), args.clone(), &mut context).await;
        match first {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("RESOURCE_NOT_FOUND".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }

        let second = run(tool.as_ref(), args, &mut context).await;
        match second {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("RESOURCE_NOT_FOUND_FATAL".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // RunSkillScriptTool
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn run_skill_script_reports_no_code_executor_without_an_environment() {
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.py".to_string(),
            crate::skills_models::Script {
                src: "print('hi')".to_string(),
            },
        );
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.py")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("NO_CODE_EXECUTOR".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_reports_script_not_found_and_escalates_to_fatal() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let mut context = ctx();
        let args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/missing.py")]);

        let first = run(tool.as_ref(), args.clone(), &mut context).await;
        match first {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("SCRIPT_NOT_FOUND".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
        let second = run(tool.as_ref(), args, &mut context).await;
        match second {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("SCRIPT_NOT_FOUND_FATAL".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_requires_a_command_when_an_environment_is_configured() {
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.sh".to_string(),
            crate::skills_models::Script {
                src: "echo hi".to_string(),
            },
        );
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            environment: Some(Arc::new(LocalEnvironment::new()) as Arc<dyn BaseEnvironment>),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let result = run(
            tool.as_ref(),
            args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.sh")]),
            &mut ctx(),
        )
        .await;
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields.iter().find(|(k, _)| k == "error_code").unwrap().1,
                    Value::String("INVALID_ARGUMENTS".to_string())
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_materializes_and_executes_against_the_environment() {
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.sh".to_string(),
            crate::skills_models::Script {
                src: "echo hi".to_string(),
            },
        );
        with_script.resources.references.insert(
            "notes.md".to_string(),
            ResourceContent::Text("notes content".to_string()),
        );
        let env = Arc::new(LocalEnvironment::new());
        env.initialize().await.unwrap();
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            environment: Some(env.clone() as Arc<dyn BaseEnvironment>),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let skills_folder = toolset.skills_folder().unwrap();
        let mut args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.sh")]);
        args.insert(
            "command".to_string(),
            Value::String(format!(
                "sh {}/alpha/scripts/run.sh",
                skills_folder.display()
            )),
        );
        let mut context = ctx();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                assert!(matches!(&stdout.1, Value::String(s) if s.trim() == "hi"));
            }
            other => panic!("expected a map, got {other:?}"),
        }

        // The referenced file was materialized alongside the script.
        let notes_path = format!("{}/alpha/references/notes.md", skills_folder.display());
        let notes = env.read_file(&notes_path).await.unwrap();
        assert_eq!(notes, b"notes content");
        env.close().await;
    }

    // -------------------------------------------------------------
    // SkillCoreState::get_or_fetch_skill
    // -------------------------------------------------------------

    #[rusty_tokio::test]
    async fn get_or_fetch_skill_caches_a_registry_result_within_the_same_invocation() {
        let registry = Arc::new(StubRegistry {
            skills: HashMap::from([("remote".to_string(), skill("remote"))]),
            search_results: vec![],
            call_count: AtomicUsize::new(0),
        });
        let core = SkillCoreState {
            skills: HashMap::new(),
            registry: Some(registry.clone()),
            env: None,
            code_executor: None,
            skills_folder_override: None,
            script_timeout: DEFAULT_SCRIPT_TIMEOUT,
            fetched_skill_cache: Mutex::new(FetchedSkillCache::default()),
        };
        let first = core.get_or_fetch_skill("remote", "inv-1").await.unwrap();
        let second = core.get_or_fetch_skill("remote", "inv-1").await.unwrap();
        assert!(first.is_some());
        assert!(second.is_some());
        assert_eq!(registry.call_count.load(Ordering::SeqCst), 1);
    }

    #[rusty_tokio::test]
    async fn get_or_fetch_skill_evicts_the_oldest_invocation_past_16_turns() {
        let registry = Arc::new(StubRegistry {
            skills: HashMap::from([("remote".to_string(), skill("remote"))]),
            search_results: vec![],
            call_count: AtomicUsize::new(0),
        });
        let core = SkillCoreState {
            skills: HashMap::new(),
            registry: Some(registry),
            env: None,
            code_executor: None,
            skills_folder_override: None,
            script_timeout: DEFAULT_SCRIPT_TIMEOUT,
            fetched_skill_cache: Mutex::new(FetchedSkillCache::default()),
        };
        for i in 0..MAX_CACHE_TURNS {
            core.get_or_fetch_skill("remote", &format!("inv-{i}"))
                .await
                .unwrap();
        }
        {
            let cache = core.fetched_skill_cache.lock().unwrap();
            assert_eq!(cache.order.len(), MAX_CACHE_TURNS);
            assert!(cache.entries.contains_key("inv-0"));
        }

        core.get_or_fetch_skill("remote", "inv-overflow")
            .await
            .unwrap();
        let cache = core.fetched_skill_cache.lock().unwrap();
        assert_eq!(cache.order.len(), MAX_CACHE_TURNS);
        assert!(!cache.entries.contains_key("inv-0"));
        assert!(cache.entries.contains_key("inv-overflow"));
    }

    #[test]
    fn guess_mime_type_falls_back_to_octet_stream() {
        assert_eq!(guess_mime_type("logo.png"), "image/png");
        assert_eq!(guess_mime_type("data.bin"), "application/octet-stream");
    }

    // -------------------------------------------------------------
    // C0950: additional_tools / _resolve_additional_tools_from_state /
    // clone_with_updated_skills
    // -------------------------------------------------------------

    struct NamedTool {
        name: String,
    }

    impl BaseTool for NamedTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "a test tool"
        }
    }

    struct StaticToolset {
        tools: Vec<Arc<dyn BaseTool>>,
        prefix: Option<String>,
        cache: Mutex<PrefixCache>,
    }

    impl BaseToolset for StaticToolset {
        fn get_tools<'a>(
            &'a self,
            _readonly_context: Option<&'a ReadonlyContext>,
        ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
            let tools = self.tools.clone();
            Box::pin(async move { tools })
        }

        fn prefix_cache(&self) -> &Mutex<PrefixCache> {
            &self.cache
        }

        fn tool_name_prefix(&self) -> Option<&str> {
            self.prefix.as_deref()
        }
    }

    fn readonly_context_with_activated_skill(skill_name: &str) -> ReadonlyContext {
        let mut session = Session::new("app", "user", "s1");
        session.state.insert(
            "_adk_activated_skill_unknown".to_string(),
            Value::Seq(vec![Value::String(skill_name.to_string())]),
        );
        let ic = InvocationContextBuilder::new("inv-1", session).build();
        ReadonlyContext::new(ic)
    }

    fn skill_with_additional_tools(name: &str, tool_names: &[&str]) -> Skill {
        let mut skill = skill(name);
        skill.frontmatter.metadata.insert(
            "adk_additional_tools".to_string(),
            Value::Seq(
                tool_names
                    .iter()
                    .map(|n| Value::String(n.to_string()))
                    .collect(),
            ),
        );
        skill
    }

    #[rusty_tokio::test]
    async fn get_tools_returns_only_core_tools_without_a_readonly_context() {
        let config = SkillToolsetConfig {
            additional_tools: vec![AdditionalTool::Tool(Arc::new(NamedTool {
                name: "extra_tool".to_string(),
            }))],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tools = toolset.get_tools(None).await;
        assert!(!tools.iter().any(|t| t.name() == "extra_tool"));
    }

    #[rusty_tokio::test]
    async fn get_tools_resolves_a_provided_tool_once_its_skill_is_activated() {
        let config = SkillToolsetConfig {
            skills: vec![skill_with_additional_tools("alpha", &["extra_tool"])],
            additional_tools: vec![AdditionalTool::Tool(Arc::new(NamedTool {
                name: "extra_tool".to_string(),
            }))],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let ctx = readonly_context_with_activated_skill("alpha");
        let tools = toolset.get_tools(Some(&ctx)).await;
        assert!(tools.iter().any(|t| t.name() == "extra_tool"));
    }

    #[rusty_tokio::test]
    async fn get_tools_resolves_a_provided_toolsets_tools() {
        let config = SkillToolsetConfig {
            skills: vec![skill_with_additional_tools("alpha", &["ns_extra_tool"])],
            additional_tools: vec![AdditionalTool::Toolset(Arc::new(StaticToolset {
                tools: vec![Arc::new(NamedTool {
                    name: "extra_tool".to_string(),
                })],
                prefix: Some("ns".to_string()),
                cache: Mutex::new(PrefixCache::new()),
            }))],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let ctx = readonly_context_with_activated_skill("alpha");
        let tools = toolset.get_tools(Some(&ctx)).await;
        assert!(tools.iter().any(|t| t.name() == "ns_extra_tool"));
    }

    #[rusty_tokio::test]
    async fn get_tools_skips_a_provided_tool_that_collides_with_a_core_tool_name() {
        let config = SkillToolsetConfig {
            skills: vec![skill_with_additional_tools("alpha", &["list_skills"])],
            additional_tools: vec![AdditionalTool::Tool(Arc::new(NamedTool {
                name: "list_skills".to_string(),
            }))],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let ctx = readonly_context_with_activated_skill("alpha");
        let tools = toolset.get_tools(Some(&ctx)).await;
        assert_eq!(
            tools.iter().filter(|t| t.name() == "list_skills").count(),
            1
        );
    }

    #[rusty_tokio::test]
    async fn get_tools_resolves_nothing_when_no_skill_is_activated() {
        let config = SkillToolsetConfig {
            skills: vec![skill_with_additional_tools("alpha", &["extra_tool"])],
            additional_tools: vec![AdditionalTool::Tool(Arc::new(NamedTool {
                name: "extra_tool".to_string(),
            }))],
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let ctx = readonly_context_with_activated_skill("beta"); // not "alpha"
        let tools = toolset.get_tools(Some(&ctx)).await;
        assert!(!tools.iter().any(|t| t.name() == "extra_tool"));
    }

    #[test]
    fn clone_with_updated_skills_preserves_additional_tools_but_resets_prefix() {
        let config = SkillToolsetConfig {
            skills: vec![skill("alpha")],
            additional_tools: vec![AdditionalTool::Tool(Arc::new(NamedTool {
                name: "extra_tool".to_string(),
            }))],
            tool_name_prefix: Some("ns".to_string()),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let cloned = toolset
            .clone_with_updated_skills(vec![skill("beta")])
            .unwrap();

        assert_eq!(
            cloned
                .skills()
                .iter()
                .map(|s| s.name().to_string())
                .collect::<Vec<_>>(),
            vec!["beta"]
        );
        assert!(cloned.provided_tools_by_name.contains_key("extra_tool"));
        assert_eq!(cloned.tool_name_prefix(), None);
    }

    // -------------------------------------------------------------
    // C0410: RunSkillScriptTool's `code_executor` path
    // -------------------------------------------------------------

    fn python_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn python_str_literal_round_trips_through_a_real_interpreter() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let tricky = "it's a \\test\\ with\nnewlines\tand\rcontrol\x01chars and 'quotes'";
        let literal = python_str_literal(tricky);
        let code = format!("import sys; sys.stdout.write({literal})");
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(&code)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout), tricky);
    }

    #[test]
    fn python_bytes_literal_round_trips_through_a_real_interpreter() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let tricky: &[u8] = b"raw \xffbytes\x00 with 'quotes' and \\backslash\\";
        let literal = python_bytes_literal(tricky);
        let code = format!("import sys; sys.stdout.buffer.write({literal})");
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(&code)
            .output()
            .unwrap();
        assert_eq!(output.stdout, tricky);
    }

    #[test]
    fn build_wrapper_code_returns_none_for_an_unsupported_extension() {
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let script_executor = SkillScriptCodeExecutor::new(base_executor, DEFAULT_SCRIPT_TIMEOUT);
        let skill = skill("alpha");
        assert!(script_executor
            .build_wrapper_code(&skill, "scripts/run.rb", None, None, None)
            .is_none());
    }

    #[rusty_tokio::test]
    async fn skill_toolset_new_rejects_both_code_executor_and_environment() {
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let config = SkillToolsetConfig {
            code_executor: Some(base_executor),
            environment: Some(Arc::new(LocalEnvironment::new()) as Arc<dyn BaseEnvironment>),
            ..Default::default()
        };
        let err = SkillToolset::new(config).err().unwrap();
        assert!(err.contains("Cannot have both code_executor and environment"));
    }

    #[rusty_tokio::test]
    async fn run_skill_script_executes_a_python_script_via_the_code_executor() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.py".to_string(),
            crate::skills_models::Script {
                src: "import sys\nprint('argv:', sys.argv[1:])".to_string(),
            },
        );
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            code_executor: Some(base_executor),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let mut args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.py")]);
        args.insert(
            "args".to_string(),
            Value::Map(vec![(
                "greeting".to_string(),
                Value::String("hi".to_string()),
            )]),
        );
        let mut context = ctx();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                assert!(
                    matches!(&stdout.1, Value::String(s) if s.contains("--greeting") && s.contains("hi"))
                );
                let status = fields.iter().find(|(k, _)| k == "status").unwrap();
                assert_eq!(status.1, Value::String("success".to_string()));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_executes_a_shell_script_via_the_code_executor() {
        if !python_available()
            || std::process::Command::new("bash")
                .arg("--version")
                .output()
                .is_err()
        {
            eprintln!("skipping: no python3/bash interpreter on PATH");
            return;
        }
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.sh".to_string(),
            crate::skills_models::Script {
                src: "echo hello-from-shell".to_string(),
            },
        );
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            code_executor: Some(base_executor),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.sh")]);
        let mut context = ctx();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                assert!(matches!(&stdout.1, Value::String(s) if s.trim() == "hello-from-shell"));
                let status = fields.iter().find(|(k, _)| k == "status").unwrap();
                assert_eq!(status.1, Value::String("success".to_string()));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_rejects_a_non_dict_non_list_args_value() {
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.py".to_string(),
            crate::skills_models::Script {
                src: "print('hi')".to_string(),
            },
        );
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            code_executor: Some(base_executor),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let mut args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.py")]);
        args.insert(
            "args".to_string(),
            Value::String("not-a-dict-or-list".to_string()),
        );
        let mut context = ctx();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("'args' must be")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn run_skill_script_rejects_positional_args_combined_with_a_list_args() {
        let mut with_script = skill("alpha");
        with_script.resources.scripts.insert(
            "run.py".to_string(),
            crate::skills_models::Script {
                src: "print('hi')".to_string(),
            },
        );
        let base_executor: Arc<dyn BaseCodeExecutor + Send + Sync> =
            Arc::new(crate::unsafe_local_code_executor::UnsafeLocalCodeExecutor::new());
        let config = SkillToolsetConfig {
            skills: vec![with_script],
            code_executor: Some(base_executor),
            ..Default::default()
        };
        let toolset = SkillToolset::new(config).unwrap();
        let tool = toolset
            .tools
            .iter()
            .find(|t| t.name() == "run_skill_script")
            .unwrap();
        let mut args = args_map(&[("skill_name", "alpha"), ("file_path", "scripts/run.py")]);
        args.insert(
            "args".to_string(),
            Value::Seq(vec![Value::String("a".to_string())]),
        );
        args.insert(
            "positional_args".to_string(),
            Value::Seq(vec![Value::String("b".to_string())]),
        );
        let mut context = ctx();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(matches!(&error.1, Value::String(s) if s.contains("Cannot specify")));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
