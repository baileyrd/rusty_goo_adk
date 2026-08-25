//! C0628: `evaluation.simulation.llm_backed_user_simulator`/
//! `llm_backed_user_simulator_prompts`, ported from
//! `google.adk.evaluation.simulation.llm_backed_user_simulator`(`_prompts`).
//!
//! [`LlmBackedUserSimulator`] implements
//! [`crate::user_simulator::UserSimulator`] — see that trait's own doc
//! for the async/`Result` widening this implementor required.
//!
//! **Not auto-registered under `"llm_backed"`**: this port has no
//! module-load-time side effects the way a Python module's top-level
//! code does, so nothing in this crate calls
//! [`crate::user_simulator::register_user_simulator`] for any config
//! type — the same "no implicit registration" state every other
//! registry-driven type in this crate is already in (verified: grep
//! finds zero non-test call sites for `register_user_simulator` in this
//! crate). Wiring `LlmBackedUserSimulator` in under the `"llm_backed"`
//! discriminator is left to whatever future code composes the full
//! evaluation pipeline; a direct caller can construct it via
//! [`LlmBackedUserSimulator::new`] without going through the registry
//! at all.
//!
//! **Jinja2, narrowed — only the flat, no-persona template renders**: the
//! source's default (no-persona) prompt template
//! (`_DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE`) uses only 3 flat
//! `{{ var }}` substitutions, no loops/filters — trivially portable
//! without a real template engine, and ported verbatim (byte-for-byte
//! literal string) below. The persona-decorated template
//! (`_USER_SIMULATOR_INSTRUCTIONS_WITH_PERSONA_TEMPLATE`) uses a genuine
//! `{% for b in persona.behaviors %}` loop plus a custom
//! `render_string_filter` — this port has no Jinja2 equivalent (same
//! disclosed gap already established for `adk-flows/src/instructions_utils.rs`,
//! C0170), so [`get_llm_backed_user_simulator_prompt`] returns
//! [`LlmBackedUserSimulatorError::PersonaTemplateUnsupported`] whenever
//! a `ConversationScenario` carries a `user_persona` — a real narrowing,
//! not a silently-wrong render.
//!
//! **`is_valid_user_simulator_template`, narrowed to a regex presence
//! check**: the source uses `jinja2.meta.find_undeclared_variables` —
//! real Jinja AST inspection. This port instead checks each required
//! param's `{{ name }}`-shaped pattern (`\{\{\s*name\s*\}\}`) is present
//! literally in the string via [`regex`] (already a workspace
//! dependency, `rubric_based_evaluator.rs`'s own C0601 precedent) —
//! catches the common case this validator exists for (a user forgetting
//! a placeholder), not full Jinja syntax validation.
//!
//! **`add_default_retry_options_if_not_present`, not ported**: the
//! source itself flags this helper `NOTE: intended for eval systems
//! internal usage. Do not take direct dependency on it.` It would need
//! `adk_models::llm_request::HttpOptionsStub` to grow a `retry_options`
//! field it doesn't have — a cross-crate struct used at 3 other call
//! sites in `adk-models`/`adk-flows` — for a helper the source itself
//! marks as non-load-bearing. Left as a disclosed gap rather than
//! touching that shared type for this internal-only helper.
//!
//! **Python dict repr, compact JSON stand-in**: `_summarize_conversation`'s
//! `include_function_calls` branch formats a tool call's `args`/a tool
//! response's `response` via Python's `str(dict)`. This port uses
//! compact JSON instead — the same disclosed lower-fidelity stand-in
//! `adk-flows/src/fencing.rs`/`instructions_utils::value_to_display_string`
//! already establish.
//!
//! **`model_configuration`'s default `thinking_config`, opaque**: the
//! source's default is a real `ThinkingConfig(include_thoughts=True,
//! thinking_budget=10240)`. `GenerateContentConfigStub::thinking_config`
//! is `Option<Value>` (opaque, Phase 3) — the default is set as an
//! opaque JSON map with the same two keys, matching the wire's own
//! camelCase, since nothing in this port reads it back structurally yet.

use std::collections::BTreeMap;

use adk_events::Event;
use adk_genai::content::{Content, Part};
use adk_models::base_llm::{BaseLlm, BaseLlmError};
use adk_models::llm_request::{GenerateContentConfigStub, LlmRequest};
use adk_models::registry::{default_registry, RegistryError};
use regex::Regex;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::conversation_scenarios::ConversationScenario;
use crate::evaluator::Evaluator;
use crate::user_simulator::{
    parse_simulator_config, BoxFuture, NextUserMessage, Status, UserSimulator,
};

const AUTHOR_USER: &str = "user";
const STOP_SIGNAL: &str = "</finished>";

const DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE: &str = r#"You are a Simulated User designed to test an AI Agent.

Your single most important job is to react logically to the Agent's last message.
The Conversation Plan is your canonical grounding, not a script; your response MUST be dictated by what the Agent just said.

# Primary Operating Loop

You MUST follow this three-step process while thinking:

Step 1: Analyze what the Agent just said or did. Specifically, is the Agent asking you a question, reporting a successful or unsuccessful operation, or saying something incorrect or unexpected?

Step 2: Choose one action based on your analysis:
* ANSWER any questions the Agent asked.
* ADVANCE to the next request as per the Conversation Plan if the Agent succeeds in satisfying your current request.
* INTERVENE if the Agent is yet to complete your current request and the Conversation Plan requires you to modify it.
* CORRECT the Agent if it is making a mistake or failing.
* END the conversation if any of the below stopping conditions are met:
  - The Agent has completed all your requests from the Conversation Plan.
  - The Agent has failed to fulfill a request *more than once*.
  - The Agent has performed an incorrect operation and informs you that it is unable to correct it.
  - The Agent ends the conversation on its own by transferring you to a *human/live agent* (NOT another AI Agent).

Step 3: Formulate a response based on the chosen action and the below Action Protocols and output it.

# Action Protocols

**PROTOCOL: ANSWER**
* Only answer the Agent's questions using information from the Conversation Plan.
* Do NOT provide any additional information the Agent did not explicitly ask for.
* If you do not have the information requested by the Agent, inform the Agent. Do NOT make up information that is not in the Conversation Plan.
* Do NOT advance to the next request in the Conversation Plan.

**PROTOCOL: ADVANCE**
* Make the next request from the Conversation Plan.
* Skip redundant requests already fulfilled by the Agent.

**PROTOCOL: INTERVENE**
* Change your current request as directed by the Conversation Plan with natural phrasing.

**PROTOCOL: CORRECT**
* Challenge illogical or incorrect statements made by the Agent.
* If the Agent did an incorrect operation, ask the Agent to fix it.
* If this is the FIRST time the Agent failed to satisfy your request, ask the Agent to try again.

**PROTOCOL: END**
* End the conversation only when any of the stopping conditions are met; do NOT end prematurely.
* Output `{{ stop_signal }}` to indicate that the conversation with the AI Agents is over.

# Conversation Plan

{{ conversation_plan }}

# Conversation History

{{ conversation_history }}
"#;

/// Error type for [`get_llm_backed_user_simulator_prompt`],
/// [`LlmBackedUserSimulatorConfig::validate`], and
/// [`LlmBackedUserSimulator::new`].
#[derive(Debug, rusty_err::Error)]
pub enum LlmBackedUserSimulatorError {
    #[error(
        "custom_instructions must contain each of the following formatting placeholders using \
         Jinja syntax: {{{{ stop_signal }}}}, {{{{ conversation_plan }}}}, {{{{ conversation_history }}}}"
    )]
    InvalidCustomInstructions,
    #[error(
        "a ConversationScenario with a user_persona needs the persona-decorated prompt \
         template, which needs real Jinja2 template rendering (loops/filters) this port \
         doesn't have — see this module's own doc"
    )]
    PersonaTemplateUnsupported,
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Registry(#[from] RegistryError),
    #[error("{0}")]
    Generation(#[from] BaseLlmError),
    #[error("Failed to generate a user message: {0}")]
    GenerationFailed(String),
}

/// `llm_backed_user_simulator.LlmBackedUserSimulatorConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LlmBackedUserSimulatorConfig {
    #[rusty_serde(rename = "type", default = "default_simulator_type")]
    pub simulator_type: String,
    #[rusty_serde(default = "default_model")]
    pub model: String,
    #[rusty_serde(default = "default_model_configuration")]
    pub model_configuration: GenerateContentConfigStub,
    #[rusty_serde(default = "default_max_allowed_invocations")]
    pub max_allowed_invocations: i64,
    #[rusty_serde(default)]
    pub custom_instructions: Option<String>,
    #[rusty_serde(default)]
    pub include_function_calls: bool,
}

fn default_simulator_type() -> String {
    "llm_backed".to_string()
}

fn default_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_model_configuration() -> GenerateContentConfigStub {
    GenerateContentConfigStub {
        thinking_config: Some(Value::Map(vec![
            ("includeThoughts".to_string(), Value::Bool(true)),
            ("thinkingBudget".to_string(), Value::UInt(10240)),
        ])),
        ..Default::default()
    }
}

fn default_max_allowed_invocations() -> i64 {
    20
}

impl Default for LlmBackedUserSimulatorConfig {
    fn default() -> Self {
        Self {
            simulator_type: default_simulator_type(),
            model: default_model(),
            model_configuration: default_model_configuration(),
            max_allowed_invocations: default_max_allowed_invocations(),
            custom_instructions: None,
            include_function_calls: false,
        }
    }
}

impl LlmBackedUserSimulatorConfig {
    /// `@field_validator("custom_instructions")` — see the module doc
    /// for why this checks literal placeholder presence via regex
    /// rather than real Jinja AST inspection.
    pub fn validate(&self) -> Result<(), LlmBackedUserSimulatorError> {
        let Some(custom_instructions) = &self.custom_instructions else {
            return Ok(());
        };
        if is_valid_user_simulator_template(
            custom_instructions,
            &["stop_signal", "conversation_plan", "conversation_history"],
        ) {
            Ok(())
        } else {
            Err(LlmBackedUserSimulatorError::InvalidCustomInstructions)
        }
    }
}

/// `llm_backed_user_simulator_prompts.is_valid_user_simulator_template` —
/// see the module doc for the narrowing from real Jinja parsing to a
/// regex presence check.
pub fn is_valid_user_simulator_template(template_str: &str, required_params: &[&str]) -> bool {
    required_params.iter().all(|param| {
        let pattern = format!(r"\{{\{{\s*{param}\s*\}}\}}");
        Regex::new(&pattern)
            .map(|regex| regex.is_match(template_str))
            .unwrap_or(false)
    })
}

fn substitute_placeholders(template: &str, values: &[(&str, &str)]) -> String {
    let mut rendered = template.to_string();
    for (name, value) in values {
        let pattern = format!(r"\{{\{{\s*{name}\s*\}}\}}");
        if let Ok(regex) = Regex::new(&pattern) {
            rendered = regex
                .replace_all(&rendered, value.replace('$', "$$"))
                .into_owned();
        }
    }
    rendered
}

/// `llm_backed_user_simulator_prompts.get_llm_backed_user_simulator_prompt`
/// — see the module doc for the persona-template narrowing.
pub fn get_llm_backed_user_simulator_prompt(
    conversation_plan: &str,
    conversation_history: &str,
    stop_signal: &str,
    custom_instructions: Option<&str>,
    has_user_persona: bool,
) -> Result<String, LlmBackedUserSimulatorError> {
    if has_user_persona {
        return Err(LlmBackedUserSimulatorError::PersonaTemplateUnsupported);
    }
    let template = custom_instructions.unwrap_or(DEFAULT_USER_SIMULATOR_INSTRUCTIONS_TEMPLATE);
    Ok(substitute_placeholders(
        template,
        &[
            ("stop_signal", stop_signal),
            ("conversation_plan", conversation_plan),
            ("conversation_history", conversation_history),
        ],
    ))
}

fn display_args(args: &Option<BTreeMap<String, Value>>) -> String {
    args.as_ref()
        .map(|args| Value::Map(args.iter().map(|(k, v)| (k.clone(), v.clone())).collect()))
        .and_then(|value| rusty_serde::json::to_string(&value).ok())
        .unwrap_or_default()
}

/// `LlmBackedUserSimulator._summarize_conversation` — see the module
/// doc for the Python-dict-repr → compact-JSON stand-in.
pub fn summarize_conversation(events: &[Event], include_function_calls: bool) -> String {
    let mut rewritten_dialogue: Vec<String> = Vec::new();
    for event in events {
        let Some(content) = &event.content else {
            continue;
        };
        if content.parts.is_empty() {
            continue;
        }
        let author = &event.author;
        for part in &content.parts {
            if let Some(text) = part.text.as_deref().filter(|text| !text.is_empty()) {
                if part.thought != Some(true) {
                    rewritten_dialogue.push(format!("{author}: {text}"));
                    continue;
                }
            }
            if let Some(call) = part
                .function_call
                .as_ref()
                .filter(|_| include_function_calls)
            {
                rewritten_dialogue.push(format!(
                    "{author} called tool '{}' with args: {}",
                    call.name.as_deref().unwrap_or_default(),
                    display_args(&call.args)
                ));
            } else if let Some(response) = part
                .function_response
                .as_ref()
                .filter(|_| include_function_calls)
            {
                rewritten_dialogue.push(format!(
                    "Tool '{}' returned: {}",
                    response.name.as_deref().unwrap_or_default(),
                    display_args(&response.response)
                ));
            }
        }
    }
    rewritten_dialogue.join("\n\n")
}

/// C0628: `llm_backed_user_simulator.LlmBackedUserSimulator`.
pub struct LlmBackedUserSimulator {
    config: LlmBackedUserSimulatorConfig,
    conversation_scenario: ConversationScenario,
    invocation_count: i64,
    llm: Box<dyn BaseLlm>,
}

impl LlmBackedUserSimulator {
    /// `LlmBackedUserSimulator.__init__` — `config` is the opaque
    /// [`Value`] [`crate::user_simulator::UserSimulatorProvider`]
    /// already resolves; parsed and validated here via
    /// [`parse_simulator_config`], the same round-trip idiom every
    /// other config-driven type in this crate uses.
    pub fn new(
        config: &Value,
        conversation_scenario: ConversationScenario,
    ) -> Result<Self, LlmBackedUserSimulatorError> {
        let config: LlmBackedUserSimulatorConfig =
            parse_simulator_config(config).map_err(LlmBackedUserSimulatorError::Config)?;
        config.validate()?;
        let llm = default_registry()
            .read()
            .expect("llm registry lock poisoned")
            .new_llm(&config.model)?;
        Ok(Self {
            config,
            conversation_scenario,
            invocation_count: 0,
            llm,
        })
    }

    /// `LlmBackedUserSimulator._get_llm_response` — sends a user-message
    /// generation request to the LLM, returning the generated text and
    /// (if generation failed) a human-readable error reason.
    async fn get_llm_response(
        &self,
        rewritten_dialogue: &str,
    ) -> Result<(String, Option<String>), LlmBackedUserSimulatorError> {
        if self.invocation_count == 0 {
            return Ok((self.conversation_scenario.starting_prompt.clone(), None));
        }

        let has_user_persona = self.conversation_scenario.user_persona.is_some();
        let user_agent_instructions = get_llm_backed_user_simulator_prompt(
            &self.conversation_scenario.conversation_plan,
            rewritten_dialogue,
            STOP_SIGNAL,
            self.config.custom_instructions.as_deref(),
            has_user_persona,
        )?;

        let mut llm_request = LlmRequest::new(self.config.model.clone());
        llm_request.config = self.config.model_configuration.clone();
        llm_request.contents = vec![Content::new(
            AUTHOR_USER,
            vec![Part::text(user_agent_instructions)],
        )];

        let responses = self.llm.generate_content_async(&llm_request, false).await?;

        let mut response = String::new();
        let mut error_reason: Option<String> = None;
        let mut has_thought_tokens = false;
        for llm_response in &responses {
            if let Some(error_code) = &llm_response.error_code {
                error_reason = Some(format!("safety filters or other error (code={error_code})"));
                response.clear();
                break;
            }

            let Some(content) = &llm_response.content else {
                continue;
            };
            if content.parts.is_empty() {
                continue;
            }

            for part in &content.parts {
                if part.thought == Some(true) {
                    has_thought_tokens = true;
                } else if let Some(text) = &part.text {
                    response.push_str(text);
                }
            }
        }

        if response.is_empty() && error_reason.is_none() {
            error_reason = Some(if has_thought_tokens {
                "LLM returned only thinking tokens".to_string()
            } else {
                "LLM returned empty response".to_string()
            });
        }

        Ok((response, error_reason))
    }
}

impl UserSimulator for LlmBackedUserSimulator {
    fn get_next_user_message<'a>(
        &'a mut self,
        events: &'a [Event],
    ) -> BoxFuture<'a, Result<NextUserMessage, String>> {
        Box::pin(async move {
            let invocation_limit = self.config.max_allowed_invocations;
            if invocation_limit >= 0 && self.invocation_count >= invocation_limit {
                return Ok(NextUserMessage {
                    status: Status::TurnLimitReached,
                    user_message: None,
                });
            }

            let rewritten_dialogue =
                summarize_conversation(events, self.config.include_function_calls);

            let (response, error_reason) = self
                .get_llm_response(&rewritten_dialogue)
                .await
                .map_err(|error| error.to_string())?;
            self.invocation_count += 1;

            if !response.is_empty()
                && response
                    .to_lowercase()
                    .contains(&STOP_SIGNAL.to_lowercase())
            {
                return Ok(NextUserMessage {
                    status: Status::StopSignalDetected,
                    user_message: None,
                });
            }

            if !response.is_empty() {
                return Ok(NextUserMessage {
                    status: Status::Success,
                    user_message: Some(Content::new(AUTHOR_USER, vec![Part::text(response)])),
                });
            }

            Err(
                LlmBackedUserSimulatorError::GenerationFailed(error_reason.unwrap_or_default())
                    .to_string(),
            )
        })
    }

    /// `LlmBackedUserSimulator.get_simulation_evaluator` — the source
    /// itself raises `NotImplementedError()` here.
    fn get_simulation_evaluator(&self) -> Option<Box<dyn Evaluator>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::FunctionCall;
    use adk_models::llm_response::LlmResponse;
    use std::future::Future;
    use std::pin::Pin;

    // --- is_valid_user_simulator_template / get_llm_backed_user_simulator_prompt ---

    #[test]
    fn valid_template_requires_every_param() {
        let template = "{{ stop_signal }} {{ conversation_plan }} {{ conversation_history }}";
        assert!(is_valid_user_simulator_template(
            template,
            &["stop_signal", "conversation_plan", "conversation_history"]
        ));
    }

    #[test]
    fn invalid_template_is_missing_a_param() {
        let template = "{{ stop_signal }} {{ conversation_plan }}";
        assert!(!is_valid_user_simulator_template(
            template,
            &["stop_signal", "conversation_plan", "conversation_history"]
        ));
    }

    #[test]
    fn default_template_renders_all_three_placeholders() {
        let prompt = get_llm_backed_user_simulator_prompt(
            "the plan",
            "the history",
            "</finished>",
            None,
            false,
        )
        .unwrap();
        assert!(prompt.contains("the plan"));
        assert!(prompt.contains("the history"));
        assert!(prompt.contains("</finished>"));
        assert!(!prompt.contains("{{"));
    }

    #[test]
    fn custom_instructions_are_rendered_in_place_of_the_default_template() {
        let prompt = get_llm_backed_user_simulator_prompt(
            "the plan",
            "the history",
            "</finished>",
            Some("STOP={{ stop_signal }} PLAN={{ conversation_plan }} HIST={{ conversation_history }}"),
            false,
        )
        .unwrap();
        assert_eq!(prompt, "STOP=</finished> PLAN=the plan HIST=the history");
    }

    #[test]
    fn a_user_persona_is_unsupported() {
        let err = get_llm_backed_user_simulator_prompt("plan", "history", "stop", None, true)
            .unwrap_err();
        assert!(matches!(
            err,
            LlmBackedUserSimulatorError::PersonaTemplateUnsupported
        ));
    }

    // --- LlmBackedUserSimulatorConfig ---

    #[test]
    fn config_defaults_match_the_source() {
        let config = LlmBackedUserSimulatorConfig::default();
        assert_eq!(config.simulator_type, "llm_backed");
        assert_eq!(config.model, "gemini-2.5-flash");
        assert_eq!(config.max_allowed_invocations, 20);
        assert_eq!(config.custom_instructions, None);
        assert!(!config.include_function_calls);
    }

    #[test]
    fn config_validate_accepts_a_well_formed_custom_instructions() {
        let config = LlmBackedUserSimulatorConfig {
            custom_instructions: Some(
                "{{ stop_signal }} {{ conversation_plan }} {{ conversation_history }}".to_string(),
            ),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_rejects_a_custom_instructions_missing_a_placeholder() {
        let config = LlmBackedUserSimulatorConfig {
            custom_instructions: Some("{{ stop_signal }}".to_string()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // --- summarize_conversation ---

    fn event_with_text(author: &str, text: &str, thought: bool) -> Event {
        let mut event = Event::new("inv-1", author, NodeInfo::new("root"));
        let mut part = Part::text(text);
        part.thought = thought.then_some(true);
        event.content = Some(Content::new(author, vec![part]));
        event
    }

    #[test]
    fn summarize_conversation_drops_thoughts_and_empty_content() {
        let events = vec![
            event_with_text("user", "hi", false),
            event_with_text("agent", "thinking...", true),
            event_with_text("agent", "hello back", false),
            Event::new("inv-1", "agent", NodeInfo::new("root")),
        ];
        let summary = summarize_conversation(&events, false);
        assert_eq!(summary, "user: hi\n\nagent: hello back");
    }

    #[test]
    fn summarize_conversation_includes_tool_calls_and_responses_when_requested() {
        let mut call_event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        call_event.content = Some(Content::new(
            "agent",
            vec![Part::function_call(FunctionCall {
                id: None,
                name: Some("roll_die".to_string()),
                args: Some(BTreeMap::from([("sides".to_string(), Value::UInt(6))])),
                ..Default::default()
            })],
        ));
        let events = vec![call_event];

        let with_calls = summarize_conversation(&events, true);
        assert!(with_calls.contains("agent called tool 'roll_die' with args:"));

        let without_calls = summarize_conversation(&events, false);
        assert_eq!(without_calls, "");
    }

    // --- LlmBackedUserSimulator::get_next_user_message ---

    struct FixedResponseLlm {
        model: String,
        responses: Vec<LlmResponse>,
    }

    impl BaseLlm for FixedResponseLlm {
        fn model(&self) -> &str {
            &self.model
        }
        fn type_name(&self) -> &'static str {
            "FixedResponseLlm"
        }
        fn generate_content_async<'a>(
            &'a self,
            _llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let responses = self.responses.clone();
            Box::pin(async move { Ok(responses) })
        }
    }

    fn scenario() -> ConversationScenario {
        ConversationScenario::new("Hi, I'd like to book a flight.", "Book a one-way flight.")
    }

    fn simulator_with(responses: Vec<LlmResponse>) -> LlmBackedUserSimulator {
        LlmBackedUserSimulator {
            config: LlmBackedUserSimulatorConfig::default(),
            conversation_scenario: scenario(),
            invocation_count: 0,
            llm: Box::new(FixedResponseLlm {
                model: "gemini-2.5-flash".to_string(),
                responses,
            }),
        }
    }

    #[rusty_tokio::test]
    async fn the_first_invocation_returns_the_starting_prompt_unconditionally() {
        let mut simulator = simulator_with(vec![]);
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::Success);
        assert_eq!(
            result.user_message,
            Some(Content::new(
                AUTHOR_USER,
                vec![Part::text("Hi, I'd like to book a flight.")]
            ))
        );
    }

    #[rusty_tokio::test]
    async fn a_later_invocation_detects_the_stop_signal() {
        let mut simulator = simulator_with(vec![LlmResponse {
            content: Some(Content::new(
                "model",
                vec![Part::text(format!("Great, all done! {STOP_SIGNAL}"))],
            )),
            ..Default::default()
        }]);
        simulator.invocation_count = 1; // skip the first-invocation short circuit
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::StopSignalDetected);
    }

    #[rusty_tokio::test]
    async fn a_later_invocation_returns_the_generated_text_as_a_user_message() {
        let mut simulator = simulator_with(vec![LlmResponse {
            content: Some(Content::new("model", vec![Part::text("What's the price?")])),
            ..Default::default()
        }]);
        simulator.invocation_count = 1;
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::Success);
        assert_eq!(
            result.user_message,
            Some(Content::new(
                AUTHOR_USER,
                vec![Part::text("What's the price?")]
            ))
        );
    }

    #[rusty_tokio::test]
    async fn the_turn_limit_is_enforced_before_calling_the_llm() {
        let mut simulator = simulator_with(vec![]);
        simulator.config.max_allowed_invocations = 0;
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::TurnLimitReached);
    }

    #[rusty_tokio::test]
    async fn a_negative_turn_limit_means_unlimited() {
        let mut simulator = simulator_with(vec![]);
        simulator.config.max_allowed_invocations = -1;
        simulator.invocation_count = 1_000;
        let result = simulator.get_next_user_message(&[]).await;
        // Doesn't short-circuit on the turn limit; falls through to the
        // (non-first) LLM call, which returns no responses here, so this
        // should fail with a "generation failed" error, not TurnLimitReached.
        assert!(result.is_err());
    }

    #[rusty_tokio::test]
    async fn an_empty_response_with_only_thought_parts_fails_with_that_reason() {
        let mut simulator = simulator_with(vec![LlmResponse {
            content: Some(Content::new(
                "model",
                vec![{
                    let mut part = Part::text("thinking...");
                    part.thought = Some(true);
                    part
                }],
            )),
            ..Default::default()
        }]);
        simulator.invocation_count = 1;
        let error = simulator.get_next_user_message(&[]).await.unwrap_err();
        assert!(error.contains("LLM returned only thinking tokens"));
    }

    #[rusty_tokio::test]
    async fn a_genuinely_empty_response_fails_with_that_reason() {
        let mut simulator = simulator_with(vec![LlmResponse::default()]);
        simulator.invocation_count = 1;
        let error = simulator.get_next_user_message(&[]).await.unwrap_err();
        assert!(error.contains("LLM returned empty response"));
    }

    #[rusty_tokio::test]
    async fn an_error_code_response_fails_with_the_safety_filter_reason() {
        let mut simulator = simulator_with(vec![LlmResponse {
            error_code: Some("SAFETY".to_string()),
            ..Default::default()
        }]);
        simulator.invocation_count = 1;
        let error = simulator.get_next_user_message(&[]).await.unwrap_err();
        assert!(error.contains("safety filters"));
        assert!(error.contains("SAFETY"));
    }

    #[rusty_tokio::test]
    async fn get_simulation_evaluator_is_always_none() {
        let simulator = simulator_with(vec![]);
        assert!(simulator.get_simulation_evaluator().is_none());
    }
}
