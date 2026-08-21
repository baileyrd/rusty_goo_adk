//! Capabilities C0079, C0081-C0089, C0091, C0093: `LlmAgent` config shape and
//! self-contained resolution helpers, ported from
//! `google.adk.agents.llm_agent`.
//!
//! **Scope note**: `LlmAgent` in the source extends `BaseAgent` and its real
//! behavior (`canonical_model`/`canonical_tools`/`_run_async_impl`) is
//! driven by types this migration hasn't built yet: `BaseLlm`/`LlmRequest`/
//! `LlmResponse`/`LLMRegistry` (Phase 3), `BaseTool`/`BaseToolset`/
//! `ToolContext` (Phase 8), `BaseLlmFlow`/`SingleFlow`/`AutoFlow`/planners
//! (Phase 4). Wiring `LlmAgent` into `BaseAgent`'s tree/`AgentBehavior` now
//! would mean giving it a `_run_async_impl` that can't actually run anything
//! — deferred rather than built as throwaway integration. This batch
//! implements `LlmAgent` as a standalone struct covering the config fields
//! and the handful of methods that don't need those forward phases
//! (`canonical_instruction`/`canonical_global_instruction`, the
//! `generate_content_config` validator, the `_llm_flow` single-vs-auto
//! *decision*). Once Phase 3/4/8 land, `LlmAgent` gets reworked to actually
//! implement `AgentBehavior` and become constructible via `BaseAgent`.
//!
//! **Deferred, blocked on forward phases** (left `REQUIRED` in the
//! manifest): C0080/C0090 (`canonical_model`/`canonical_live_model` — need
//! `LLMRegistry`), C0092 (`canonical_tools` — needs `BaseTool` resolution),
//! C0094 (`_get_subagent_to_resume` — needs `Event::get_function_responses`,
//! blocked on Phase 3's real `Content`/`Part`), C0095
//! (`__maybe_save_output_to_state`/`__maybe_accumulate_streaming_output` —
//! same `Content`/`Part` block), C0096 (`_pre_validate_tools`/
//! `model_post_init`'s tool-wrapping/sub-agent-wrapping — needs
//! `BaseNode`/`BaseTool`/`FinishTaskTool`), C0097 (deprecated YAML config
//! pipeline — needs the same design decision as `BaseAgent::from_config`,
//! C0047), C0099 (`FinishTaskTool` — needs `BaseTool`/`LlmRequest`/
//! `ToolContext`).
//!
//! **Adaptation**: every field typed as an opaque third-party/forward-phase
//! shape (`BaseLlm`, `LlmRequest`/`LlmResponse`, `BaseTool`/`BaseToolset`,
//! `BasePlanner`, `BaseCodeExecutor`, `google.genai.types.*`) is represented
//! as [`rusty_serde::value::Value`] or a placeholder enum, per the pattern
//! established in `run_config.rs`. The 6 model/tool callback fields (C0089)
//! are collapsed to one uniform closure signature (`Fn(&mut Context) ->
//! Option<Value>`) rather than the source's 6 distinct per-field-argument
//! signatures (request/response/tool/args/error) — nothing constructs a
//! real `LlmRequest`/`BaseTool` yet to pass through them, and the
//! observable capability under test (C0089's "stop at first non-`None`"
//! chain contract) doesn't depend on the closure's argument shape.

use std::sync::{Arc, OnceLock, RwLock};

use rusty_serde::value::Value;

use crate::context::Context;
use crate::readonly_context::ReadonlyContext;

const DEFAULT_MODEL_NAME: &str = "gemini-3.5-flash";
const DEFAULT_LIVE_MODEL_NAME: &str = "gemini-live-2.5-flash-native-audio";

/// Placeholder for `Union[str, BaseLlm]` (`BaseLlm` is Phase 3).
#[derive(Debug, Clone, PartialEq)]
pub enum ModelRef {
    Name(String),
    Instance(Value),
}

#[derive(Debug, rusty_err::Error)]
pub enum LlmAgentError {
    #[error("Default model must be a non-empty string.")]
    EmptyDefaultModel,
    #[error("Default live model must be a non-empty string.")]
    EmptyDefaultLiveModel,
    #[error("All tools must be set via LlmAgent.tools, not via generate_content_config.tools. Move your tools to the LlmAgent(tools=[...]) parameter.")]
    ToolsInGenerateContentConfig,
    #[error("System instruction must be set via LlmAgent.instruction, not via generate_content_config.system_instruction. Move your instruction to LlmAgent(instruction=\"...\").")]
    SystemInstructionInGenerateContentConfig,
    #[error("Response schema must be set via LlmAgent.output_schema, not via generate_content_config.response_schema. Move your schema to LlmAgent(output_schema=...).")]
    ResponseSchemaInGenerateContentConfig,
    #[error("Base URL is a transport setting and must be set on the model or its client, not via LlmAgent.generate_content_config.http_options.base_url.")]
    BaseUrlInGenerateContentConfig,
}

fn default_model_cell() -> &'static RwLock<ModelRef> {
    static CELL: OnceLock<RwLock<ModelRef>> = OnceLock::new();
    CELL.get_or_init(|| RwLock::new(ModelRef::Name(DEFAULT_MODEL_NAME.to_string())))
}

fn default_live_model_cell() -> &'static RwLock<ModelRef> {
    static CELL: OnceLock<RwLock<ModelRef>> = OnceLock::new();
    CELL.get_or_init(|| RwLock::new(ModelRef::Name(DEFAULT_LIVE_MODEL_NAME.to_string())))
}

/// C0079: overrides the process-wide default model used when an agent has
/// none set (mirrors the source's classmethod mutating a class attribute —
/// there is no per-instance equivalent, this is process-wide in both).
pub fn set_default_model(model: ModelRef) -> Result<(), LlmAgentError> {
    if matches!(&model, ModelRef::Name(name) if name.is_empty()) {
        return Err(LlmAgentError::EmptyDefaultModel);
    }
    *default_model_cell()
        .write()
        .expect("default model lock poisoned") = model;
    Ok(())
}

pub fn default_model() -> ModelRef {
    default_model_cell()
        .read()
        .expect("default model lock poisoned")
        .clone()
}

/// C0079: overrides the process-wide default live-mode model.
pub fn set_default_live_model(model: ModelRef) -> Result<(), LlmAgentError> {
    if matches!(&model, ModelRef::Name(name) if name.is_empty()) {
        return Err(LlmAgentError::EmptyDefaultLiveModel);
    }
    *default_live_model_cell()
        .write()
        .expect("default live model lock poisoned") = model;
    Ok(())
}

pub fn default_live_model() -> ModelRef {
    default_live_model_cell()
        .read()
        .expect("default live model lock poisoned")
        .clone()
}

/// Placeholder for `Union[str, InstructionProvider]`.
pub enum Instruction {
    Static(String),
    Provider(Arc<dyn Fn(&ReadonlyContext) -> String + Send + Sync>),
}

impl Default for Instruction {
    fn default() -> Self {
        Instruction::Static(String::new())
    }
}

impl Instruction {
    fn is_set(&self) -> bool {
        !matches!(self, Instruction::Static(s) if s.is_empty())
    }
}

/// C0091: resolves `instruction`, returning `(text, bypass_state_injection)`
/// — `bypass_state_injection` is true when the instruction came from a
/// provider (callable) rather than a plain string.
pub fn canonical_instruction(instruction: &Instruction, ctx: &ReadonlyContext) -> (String, bool) {
    match instruction {
        Instruction::Static(s) => (s.clone(), false),
        Instruction::Provider(f) => (f(ctx), true),
    }
}

/// C0082/C0091: resolves `global_instruction` (deprecated — logs a warning
/// whenever it's actually set, matching the source's `warnings.warn` on
/// every call).
pub fn canonical_global_instruction(
    instruction: &Instruction,
    ctx: &ReadonlyContext,
) -> (String, bool) {
    if instruction.is_set() {
        eprintln!(
            "DeprecationWarning: global_instruction field is deprecated and will be \
             removed in a future version. Use GlobalInstructionPlugin instead for the \
             same functionality at the App level."
        );
    }
    match instruction {
        Instruction::Static(s) => (s.clone(), false),
        Instruction::Provider(f) => (f(ctx), true),
    }
}

/// C0083: placeholder for `ToolUnion = Union[Callable, BaseTool,
/// BaseToolset]` — `BaseTool`/`BaseToolset` are Phase 8. Each variant is an
/// opaque `Value` since nothing resolves or calls a tool yet; the
/// three-way split is kept so the shape distinction survives to when a
/// real `canonical_tools` resolution (C0092) is implemented.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolUnion {
    Function(Value),
    Tool(Value),
    Toolset(Value),
}

/// C0085: the delegation mode for an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Chat,
    Task,
    SingleTurn,
}

/// C0087: controls content inclusion in model requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IncludeContents {
    #[default]
    Default,
    None,
}

/// C0093: which flow the agent would use — `SingleFlow` vs `AutoFlow` are
/// Phase 4; this is the *decision* the source's `_llm_flow` property makes,
/// without the flow objects themselves existing yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmFlowKind {
    Single,
    Auto,
}

/// C0093: selects [`LlmFlowKind::Single`] only when transfer to both parent
/// and peers is disallowed and the agent has no sub-agents; [`LlmFlowKind::Auto`]
/// otherwise.
pub fn llm_flow_kind(
    disallow_transfer_to_parent: bool,
    disallow_transfer_to_peers: bool,
    has_sub_agents: bool,
) -> LlmFlowKind {
    if disallow_transfer_to_parent && disallow_transfer_to_peers && !has_sub_agents {
        LlmFlowKind::Single
    } else {
        LlmFlowKind::Auto
    }
}

/// C0089: a model/tool callback — see the module doc for the signature
/// simplification.
pub type LlmCallback = Arc<dyn Fn(&mut Context) -> Option<Value> + Send + Sync>;

/// C0089: runs `callbacks` in order, stopping at (and returning) the first
/// non-`None` result.
pub fn run_first_non_none(callbacks: &[LlmCallback], ctx: &mut Context) -> Option<Value> {
    for callback in callbacks {
        if let Some(result) = callback(ctx) {
            return Some(result);
        }
    }
    None
}

/// C0084: validates `generate_content_config`, matching the source's
/// rejected-keys check — represented as an opaque [`Value`] map
/// (placeholder for `google.genai.types.GenerateContentConfig`, Phase 3)
/// since only specific keys need checking, not the whole schema.
pub fn validate_generate_content_config(config: Option<Value>) -> Result<Value, LlmAgentError> {
    let config = config.unwrap_or_else(|| Value::Map(vec![]));
    if let Value::Map(entries) = &config {
        for (key, value) in entries {
            match key.as_str() {
                "tools" if !is_empty_value(value) => {
                    return Err(LlmAgentError::ToolsInGenerateContentConfig);
                }
                "system_instruction" if !is_empty_value(value) => {
                    return Err(LlmAgentError::SystemInstructionInGenerateContentConfig);
                }
                "response_schema" if !is_empty_value(value) => {
                    return Err(LlmAgentError::ResponseSchemaInGenerateContentConfig);
                }
                "http_options" => {
                    if let Value::Map(http_entries) = value {
                        let has_base_url = http_entries
                            .iter()
                            .any(|(k, v)| k == "base_url" && !is_empty_value(v));
                        if has_base_url {
                            return Err(LlmAgentError::BaseUrlInGenerateContentConfig);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(config)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Seq(items) => items.is_empty(),
        Value::Map(entries) => entries.is_empty(),
        _ => false,
    }
}

/// Capabilities C0081-C0089, C0100 (config shape): `LlmAgent`'s config
/// fields — see the module doc for the standalone-struct scope decision.
pub struct LlmAgent {
    pub model: ModelRef,
    pub instruction: Instruction,
    pub global_instruction: Instruction,
    /// C0081: static instruction content — an opaque `types.ContentUnion`
    /// placeholder (Phase 3).
    pub static_instruction: Option<Value>,
    pub tools: Vec<ToolUnion>,
    pub generate_content_config: Value,
    pub mode: Option<AgentMode>,
    pub parallel_worker: Option<bool>,
    pub disallow_transfer_to_parent: bool,
    pub disallow_transfer_to_peers: bool,
    pub include_contents: IncludeContents,
    /// C0087: opaque `type[BaseModel]` placeholder.
    pub input_schema: Option<Value>,
    /// C0087: opaque `SchemaType` placeholder.
    pub output_schema: Option<Value>,
    pub output_key: Option<String>,
    /// C0088: opaque `BasePlanner` placeholder (Phase 4).
    pub planner: Option<Value>,
    /// C0088: opaque `BaseCodeExecutor` placeholder (Phase 8).
    pub code_executor: Option<Value>,
    pub before_model_callback: Vec<LlmCallback>,
    pub after_model_callback: Vec<LlmCallback>,
    pub on_model_error_callback: Vec<LlmCallback>,
    pub before_tool_callback: Vec<LlmCallback>,
    pub after_tool_callback: Vec<LlmCallback>,
    pub on_tool_error_callback: Vec<LlmCallback>,
}

impl LlmAgent {
    pub fn new(model: ModelRef) -> Self {
        Self {
            model,
            instruction: Instruction::default(),
            global_instruction: Instruction::default(),
            static_instruction: None,
            tools: Vec::new(),
            generate_content_config: Value::Map(vec![]),
            mode: None,
            parallel_worker: None,
            disallow_transfer_to_parent: false,
            disallow_transfer_to_peers: false,
            include_contents: IncludeContents::default(),
            input_schema: None,
            output_schema: None,
            output_key: None,
            planner: None,
            code_executor: None,
            before_model_callback: Vec::new(),
            after_model_callback: Vec::new(),
            on_model_error_callback: Vec::new(),
            before_tool_callback: Vec::new(),
            after_tool_callback: Vec::new(),
            on_tool_error_callback: Vec::new(),
        }
    }

    /// C0084: sets `generate_content_config`, validating it first.
    pub fn with_generate_content_config(mut self, config: Value) -> Result<Self, LlmAgentError> {
        self.generate_content_config = validate_generate_content_config(Some(config))?;
        Ok(self)
    }

    /// C0093: which flow this agent's configuration would select.
    pub fn llm_flow_kind(&self, has_sub_agents: bool) -> LlmFlowKind {
        llm_flow_kind(
            self.disallow_transfer_to_parent,
            self.disallow_transfer_to_peers,
            has_sub_agents,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    // set_default_model/set_default_live_model mutate process-wide state;
    // serialize the tests that touch it so they don't race each other.
    static DEFAULT_MODEL_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn set_default_model_rejects_an_empty_name() {
        let _guard = DEFAULT_MODEL_TEST_LOCK.lock().unwrap();
        let err = set_default_model(ModelRef::Name(String::new())).unwrap_err();
        assert!(matches!(err, LlmAgentError::EmptyDefaultModel));
    }

    #[test]
    fn set_default_model_overrides_the_process_wide_default() {
        let _guard = DEFAULT_MODEL_TEST_LOCK.lock().unwrap();
        set_default_model(ModelRef::Name("custom-model".to_string())).unwrap();
        assert_eq!(default_model(), ModelRef::Name("custom-model".to_string()));
        // Restore the built-in default so other tests aren't affected.
        set_default_model(ModelRef::Name(DEFAULT_MODEL_NAME.to_string())).unwrap();
    }

    #[test]
    fn set_default_live_model_rejects_an_empty_name() {
        let _guard = DEFAULT_MODEL_TEST_LOCK.lock().unwrap();
        let err = set_default_live_model(ModelRef::Name(String::new())).unwrap_err();
        assert!(matches!(err, LlmAgentError::EmptyDefaultLiveModel));
    }

    fn readonly_ctx() -> ReadonlyContext {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ReadonlyContext::new(ic)
    }

    #[test]
    fn canonical_instruction_returns_the_static_string_unchanged() {
        let instruction = Instruction::Static("be helpful".to_string());
        let (text, bypass) = canonical_instruction(&instruction, &readonly_ctx());
        assert_eq!(text, "be helpful");
        assert!(!bypass);
    }

    #[test]
    fn canonical_instruction_invokes_a_provider_and_flags_bypass() {
        let instruction = Instruction::Provider(Arc::new(|_ctx| "from provider".to_string()));
        let (text, bypass) = canonical_instruction(&instruction, &readonly_ctx());
        assert_eq!(text, "from provider");
        assert!(bypass);
    }

    #[test]
    fn canonical_global_instruction_resolves_like_instruction() {
        let instruction = Instruction::Static("shared identity".to_string());
        let (text, bypass) = canonical_global_instruction(&instruction, &readonly_ctx());
        assert_eq!(text, "shared identity");
        assert!(!bypass);
    }

    #[test]
    fn generate_content_config_rejects_tools() {
        let config = Value::Map(vec![("tools".to_string(), Value::Seq(vec![Value::Null]))]);
        let err = validate_generate_content_config(Some(config)).unwrap_err();
        assert!(matches!(err, LlmAgentError::ToolsInGenerateContentConfig));
    }

    #[test]
    fn generate_content_config_rejects_system_instruction() {
        let config = Value::Map(vec![(
            "system_instruction".to_string(),
            Value::String("hi".to_string()),
        )]);
        let err = validate_generate_content_config(Some(config)).unwrap_err();
        assert!(matches!(
            err,
            LlmAgentError::SystemInstructionInGenerateContentConfig
        ));
    }

    #[test]
    fn generate_content_config_rejects_response_schema() {
        let config = Value::Map(vec![(
            "response_schema".to_string(),
            Value::Map(vec![(
                "type".to_string(),
                Value::String("object".to_string()),
            )]),
        )]);
        let err = validate_generate_content_config(Some(config)).unwrap_err();
        assert!(matches!(
            err,
            LlmAgentError::ResponseSchemaInGenerateContentConfig
        ));
    }

    #[test]
    fn generate_content_config_rejects_http_options_base_url() {
        let config = Value::Map(vec![(
            "http_options".to_string(),
            Value::Map(vec![(
                "base_url".to_string(),
                Value::String("https://example.com".to_string()),
            )]),
        )]);
        let err = validate_generate_content_config(Some(config)).unwrap_err();
        assert!(matches!(err, LlmAgentError::BaseUrlInGenerateContentConfig));
    }

    #[test]
    fn generate_content_config_defaults_to_an_empty_map_when_unset() {
        let config = validate_generate_content_config(None).unwrap();
        assert_eq!(config, Value::Map(vec![]));
    }

    #[test]
    fn generate_content_config_allows_other_fields() {
        let config = Value::Map(vec![("temperature".to_string(), Value::Float(0.5))]);
        assert!(validate_generate_content_config(Some(config)).is_ok());
    }

    #[test]
    fn llm_flow_kind_is_single_only_when_fully_isolated() {
        assert_eq!(llm_flow_kind(true, true, false), LlmFlowKind::Single);
        assert_eq!(llm_flow_kind(true, false, false), LlmFlowKind::Auto);
        assert_eq!(llm_flow_kind(true, true, true), LlmFlowKind::Auto);
    }

    #[test]
    fn run_first_non_none_stops_at_the_first_hit() {
        let mut ctx = Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        let second_ran = Arc::new(AtomicUsize::new(0));
        let second_ran_clone = second_ran.clone();
        let callbacks: Vec<LlmCallback> = vec![
            Arc::new(|_ctx| Some(Value::String("first".to_string()))),
            Arc::new(move |_ctx| {
                second_ran_clone.fetch_add(1, Ordering::SeqCst);
                Some(Value::String("second".to_string()))
            }),
        ];
        let result = run_first_non_none(&callbacks, &mut ctx);
        assert_eq!(result, Some(Value::String("first".to_string())));
        assert_eq!(second_ran.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn run_first_non_none_returns_none_when_every_callback_declines() {
        let mut ctx = Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        let callbacks: Vec<LlmCallback> = vec![Arc::new(|_ctx| None), Arc::new(|_ctx| None)];
        assert_eq!(run_first_non_none(&callbacks, &mut ctx), None);
    }

    #[test]
    fn llm_agent_new_defaults_to_no_tools_or_callbacks() {
        let agent = LlmAgent::new(ModelRef::Name("gemini-3.5-flash".to_string()));
        assert!(agent.tools.is_empty());
        assert!(agent.before_model_callback.is_empty());
        assert_eq!(agent.include_contents, IncludeContents::Default);
    }

    #[test]
    fn with_generate_content_config_rejects_invalid_config_via_the_builder() {
        let config = Value::Map(vec![(
            "system_instruction".to_string(),
            Value::String("hi".to_string()),
        )]);
        let result =
            LlmAgent::new(ModelRef::Name("m".to_string())).with_generate_content_config(config);
        match result {
            Err(LlmAgentError::SystemInstructionInGenerateContentConfig) => {}
            _ => panic!("expected SystemInstructionInGenerateContentConfig"),
        }
    }
}
