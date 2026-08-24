//! Capabilities C0408 (core `SkillToolset`/`ListSkillsTool`/
//! `SearchSkillsTool`/`LoadSkillTool`), C0409 (`LoadSkillResourceTool`),
//! C0410 (`RunSkillScriptTool`, partial), C0411
//! (`DEFAULT_SKILL_SYSTEM_INSTRUCTION`), and C0401 (`adk_inject_state`
//! interpolation, exercised via `LoadSkillTool`), ported from
//! `google.adk.tools.skill_toolset`.
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
//! **Scope narrowing, disclosed**: this batch does NOT port
//! `SkillToolset.additional_tools`/`_resolve_additional_tools_from_state`/
//! `clone_with_updated_skills` — a genuine inventory gap discovered while
//! reading this file for this batch (no manifest row covers it), added
//! as its own new row **C0950** (REQUIRED, not implemented here). Every
//! other observable behavior this file's rows describe is ported.
//!
//! **C0410, partial**: `RunSkillScriptTool`'s `environment`-configured
//! branch (JIT resource materialization + `env.execute(command)`) and its
//! "neither environment nor code executor configured" error branch are
//! both ported in full. The `code_executor`-configured branch — which
//! needs `_SkillScriptCodeExecutor`, generating literal Python wrapper
//! source (`runpy.run_path` for `.py`, a `subprocess.run`-plus-JSON-
//! envelope wrapper for `.sh`/`.bash`) to hand to `BaseCodeExecutor
//! ::execute_code` — is NOT ported: it's a from-scratch code-generation
//! design, not a mechanical translation, and is substantial enough to
//! warrant its own batch. Consequently this port's [`SkillCoreState`]
//! (and [`SkillToolsetConfig`]) has no `code_executor` field at all, only
//! `environment` — exposing a `code_executor` option that couldn't
//! actually run a script would be worse than not exposing it. When
//! `environment` is `None`, this port's `RunSkillScriptTool` always
//! returns `NO_CODE_EXECUTOR`, matching the source's own behavior for
//! that same "neither configured" case exactly (the source's own
//! agent-level `code_executor` fallback lookup, one indirection deeper
//! still, is likewise not reproduced for the same reason).
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

use crate::base_environment::BaseEnvironment;
use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::base_toolset::{BaseToolset, PrefixCache, ToolFilter};
use crate::code_execution_utils::base64_encode;
use crate::skill_instructions_utils::inject_session_state;
use crate::skill_registry::SkillRegistry;
use crate::skills_models::{ResourceContent, Skill};
use crate::skills_prompt::{format_skills_as_xml, SkillSummary};
use crate::tool_context::ToolContext;

const DEFAULT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_CACHE_TURNS: usize = 16;
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
// RunSkillScriptTool
// ---------------------------------------------------------------------

/// C0410 (partial — see this module's doc): `RunSkillScriptTool`
/// (`run_skill_script`) — executes a script from a skill's `scripts/`
/// directory. Only the `environment`-configured path (and the "neither
/// configured" error) are ported this batch.
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

            if self.core.env.is_some() && command.is_none() {
                return Ok(error_response(
                    "Argument 'command' is required and must be a string.",
                    "INVALID_ARGUMENTS",
                ));
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

            let Some(env) = &self.core.env else {
                return Ok(error_response(
                    "Neither Environment nor CodeExecutor is configured. An environment or \
                     code executor is required to run scripts.",
                    "NO_CODE_EXECUTOR",
                ));
            };

            if let Err(e) = self.ensure_materialized(&skill, &file_path, env).await {
                return Ok(error_response(
                    format!("Failed to execute script '{file_path}' in environment:\n{e}"),
                    "EXECUTION_ERROR",
                ));
            }

            match env
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
            }
        })
    }
}

// ---------------------------------------------------------------------
// SkillToolset
// ---------------------------------------------------------------------

/// Configuration for [`SkillToolset::new`] — Python's keyword-argument
/// constructor collapsed into one struct, per this port's usual
/// many-optional-params convention.
pub struct SkillToolsetConfig {
    pub skills: Vec<Skill>,
    pub registry: Option<Arc<dyn SkillRegistry>>,
    pub environment: Option<Arc<dyn BaseEnvironment>>,
    pub skills_folder: Option<PathBuf>,
    pub script_timeout: Duration,
    pub tool_name_prefix: Option<String>,
    pub tool_filter: Option<ToolFilter>,
}

impl Default for SkillToolsetConfig {
    fn default() -> Self {
        Self {
            skills: Vec::new(),
            registry: None,
            environment: None,
            skills_folder: None,
            script_timeout: DEFAULT_SCRIPT_TIMEOUT,
            tool_name_prefix: None,
            tool_filter: None,
        }
    }
}

/// C0408: a toolset for managing and interacting with agent skills.
pub struct SkillToolset {
    core: Arc<SkillCoreState>,
    tools: Vec<Arc<dyn BaseTool>>,
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

        let core = Arc::new(SkillCoreState {
            skills,
            registry: config.registry,
            env: config.environment,
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
}

impl BaseToolset for SkillToolset {
    fn get_tools<'a>(
        &'a self,
        readonly_context: Option<&'a ReadonlyContext>,
    ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
        Box::pin(async move {
            self.tools
                .iter()
                .filter(|tool| self.is_tool_selected(tool.as_ref(), readonly_context))
                .cloned()
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
}
