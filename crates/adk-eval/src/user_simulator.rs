//! C0626: `evaluation.simulation.user_simulator`, ported from
//! `google.adk.evaluation.simulation.user_simulator`.
//!
//! **`UserSimulator`, adaptation disclosed**: the source's `ABC` marks
//! neither `get_next_user_message` nor `get_simulation_evaluator` with
//! `@abstractmethod` — both just `raise NotImplementedError()` in the
//! base, so a subclass that forgets to override one only fails at
//! runtime, on first call. This port's [`UserSimulator`] trait makes
//! both required methods instead — a compile-time strengthening (every
//! implementor must substantively provide both), not a narrowing of
//! behavior.
//!
//! **The config→simulator registry, adaptation disclosed**: the source
//! keys `_SIMULATOR_BY_CONFIG_TYPE` by the concrete
//! `BaseUserSimulatorConfig` *subclass itself* (a Python class object is
//! hashable and usable as a dict key). Rust has no equivalent — a type
//! isn't a runtime value. This port keys the registry by the
//! `type` discriminator *string* each config subclass already carries
//! (e.g. `"llm_backed"`) instead: that string is exactly what
//! `EvalConfig`'s own discriminated-union deserialization already
//! dispatches on, and what a caller holding raw JSON actually has in
//! hand — arguably a more direct implementation of the same purpose,
//! not a workaround. A registered entry is a constructor closure
//! (`Fn(&Value) -> Result<Box<dyn UserSimulator>, String>`) rather than a
//! class object, since Rust has no generic "instantiate this type from a
//! config" reflection either.
//!
//! **`UserSimulator.__init__`'s config round-trip**: the source's
//! `config_type.model_validate(config.model_dump())` — re-parsing a base
//! config into a concrete subclass shape — is ported as the free
//! function [`parse_simulator_config`], since Rust has no shared
//! constructor across trait implementors the way an ABC's `__init__`
//! provides; each concrete simulator's own constructor calls it
//! directly. Same "round-trip through JSON to narrow a base type" idiom
//! already used by `TrajectoryEvaluator`'s own criterion handling.
//!
//! **`NextUserMessage.user_message`, disclosed**: the source types this
//! `Optional[genai_types.Content]` — already a real, non-opaque type in
//! this port (`adk_genai::content::Content`), so no narrowing here.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};

use adk_events::Event;
use adk_genai::content::Content;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::conversation_scenarios::ConversationScenario;
use crate::evaluator::Evaluator;

/// Same shape as `adk-tools::base_tool::BoxFuture` — this crate's own
/// local alias, since `adk-eval` doesn't depend on `adk-tools` (the
/// dependency runs the other way).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `user_simulator.BaseUserSimulatorConfig` — base class for
/// user-simulator configuration. Concrete subclasses give `simulator_type`
/// (the source's `type`, renamed at the Rust level only — `type` is a
/// reserved word) a fixed, non-`None` discriminator value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct BaseUserSimulatorConfig {
    #[rusty_serde(rename = "type", default)]
    pub simulator_type: Option<String>,
}

/// `user_simulator.Status` — the resulting status of
/// `get_next_user_message()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "snake_case")]
pub enum Status {
    Success,
    TurnLimitReached,
    StopSignalDetected,
    NoMessageGenerated,
}

/// `user_simulator.NextUserMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct NextUserMessage {
    pub status: Status,
    #[rusty_serde(default)]
    pub user_message: Option<Content>,
}

impl NextUserMessage {
    /// `ensure_user_message_iff_success` — a `@model_validator(mode="after")`
    /// in the source, so it runs automatically on every construction. This
    /// port keeps the fields plainly `pub`/deserializable and exposes the
    /// same check explicitly, the same "plain fields + explicit `validate()`"
    /// pattern used throughout this crate.
    pub fn validate(&self) -> Result<(), String> {
        if (self.status == Status::Success) == self.user_message.is_none() {
            return Err(
                "A user_message should be provided if and only if the status is SUCCESS"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// C0626: `user_simulator.UserSimulator` — a user simulator for the
/// purposes of automating interaction with an Agent. See this module's
/// doc for why both methods are required rather than mirroring the
/// source's non-abstract, `NotImplementedError`-by-default shape.
pub trait UserSimulator {
    /// Returns the next user message to send to the agent, given the
    /// unaltered conversation history so far.
    ///
    /// **Adaptation**: `&mut self`, not `&self` — every implementor this
    /// port builds ([`crate::static_user_simulator::StaticUserSimulator`],
    /// [`crate::llm_backed_user_simulator::LlmBackedUserSimulator`])
    /// advances an internal cursor/counter on every call, matching the
    /// source's own `self.invocation_idx += 1`/`self._invocation_count += 1`
    /// mutations. **`async`, widened for C0628**: originally sync (no
    /// implementor needed to await anything); `LlmBackedUserSimulator`
    /// needs to await `BaseLlm::generate_content_async`, so this widens
    /// to the same `BoxFuture`-returning shape `adk-tools::base_tool::
    /// BaseTool::run_async` already established. **`Result`-wrapped,
    /// also widened for C0628**: the source's own docstring documents a
    /// real `raise RuntimeError(...)` path (the LLM genuinely fails to
    /// produce a usable message — distinct from any `Status` variant,
    /// which all describe a *successful* outcome of one kind or
    /// another), so a fallible `Result<NextUserMessage, String>` is the
    /// faithful shape, the same `Result<_, String>` convention
    /// `evaluator::Evaluator::evaluate_invocations` already uses for its
    /// own fallible trait method. Both widenings land together since
    /// zero external callers exist for either (verified at the time of
    /// the change) — an internal signature change, not a break to
    /// already-shipped public surface, same bar already used for
    /// `apply_code_execution_response`/`process_auth_responses`.
    fn get_next_user_message<'a>(
        &'a mut self,
        events: &'a [Event],
    ) -> BoxFuture<'a, Result<NextUserMessage, String>>;

    /// Returns an evaluator that evaluates whether the user simulation
    /// was successful, if this simulator has one.
    fn get_simulation_evaluator(&self) -> Option<Box<dyn Evaluator>>;
}

/// `UserSimulator.__init__`'s config round-trip. See this module's doc.
pub fn parse_simulator_config<T>(config: &Value) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let json = rusty_serde::json::to_string(config).map_err(|error| error.to_string())?;
    rusty_serde::json::from_str(&json)
        .map_err(|error| format!("Expect config of the given type: {error}"))
}

/// A registered [`UserSimulator`] constructor. See this module's doc for
/// why this replaces the source's config-*class*-keyed registry.
///
/// **Widened for C0627**: originally `Fn(&Value) -> ...`; a
/// scenario-driven simulator (e.g. [`crate::llm_backed_user_simulator::LlmBackedUserSimulator`])
/// needs the [`crate::conversation_scenarios::ConversationScenario`] too
/// — the source's own `_ScenarioUserSimulatorFactory` Protocol takes both
/// `config` and `conversation_scenario`. Zero external callers verified
/// at the time of the change (same bar already used for
/// `UserSimulator::get_next_user_message`'s own C0628 widening).
pub type SimulatorFactory = Box<
    dyn Fn(&Value, &ConversationScenario) -> Result<Box<dyn UserSimulator>, String> + Send + Sync,
>;

fn registry() -> &'static Mutex<HashMap<String, SimulatorFactory>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, SimulatorFactory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map: HashMap<String, SimulatorFactory> = HashMap::new();
        // `user_simulator_provider.py`'s own module-level
        // `register_user_simulator(LlmBackedUserSimulatorConfig,
        // LlmBackedUserSimulator)` — Rust has no module-import-time side
        // effects, so this port seeds both of the source's built-in
        // registrations here instead, in the registry's own lazy init,
        // the same "auto-register built-ins in the lazy static" shape
        // `metric_evaluator_registry::default_registry` already
        // established.
        map.insert(
            UserSimulatorProvider::LEGACY_DEFAULT_CONFIG_TYPE.to_string(),
            Box::new(
                |config: &Value, conversation_scenario: &ConversationScenario| {
                    crate::llm_backed_user_simulator::LlmBackedUserSimulator::new(
                        config,
                        conversation_scenario.clone(),
                    )
                    .map(|simulator| Box::new(simulator) as Box<dyn UserSimulator>)
                    .map_err(|error| error.to_string())
                },
            ) as SimulatorFactory,
        );
        // `user_simulator_provider.py`'s second module-level
        // `register_user_simulator(LlmAudioUserSimulatorConfig,
        // _LlmAudioUserSimulator)` (C0627, this batch). Builds the inner
        // `LlmBackedUserSimulator` from the audio config's own
        // text-generation fields, exactly matching the source's
        // `simulator_cls is _LlmAudioUserSimulator` scenario-branch special
        // case.
        map.insert(
            "llm_audio".to_string(),
            Box::new(
                |config: &Value, conversation_scenario: &ConversationScenario| {
                    let audio_config: crate::llm_audio_user_simulator::LlmAudioUserSimulatorConfig =
                        parse_simulator_config(config)?;
                    let text_config =
                        crate::llm_backed_user_simulator::LlmBackedUserSimulatorConfig {
                            simulator_type: UserSimulatorProvider::LEGACY_DEFAULT_CONFIG_TYPE
                                .to_string(),
                            model: audio_config.model.clone(),
                            model_configuration: audio_config.model_configuration.clone(),
                            max_allowed_invocations: audio_config.max_allowed_invocations,
                            custom_instructions: audio_config.custom_instructions.clone(),
                            include_function_calls: audio_config.include_function_calls,
                        };
                    let text_config_value = rusty_serde::json::to_value(&text_config)
                        .map_err(|error| error.to_string())?;
                    let text_simulator =
                        crate::llm_backed_user_simulator::LlmBackedUserSimulator::new(
                            &text_config_value,
                            conversation_scenario.clone(),
                        )
                        .map_err(|error| error.to_string())?;
                    crate::llm_audio_user_simulator::LlmAudioUserSimulator::new(
                        config,
                        Box::new(text_simulator) as Box<dyn UserSimulator + Send + Sync>,
                    )
                    .map(|simulator| Box::new(simulator) as Box<dyn UserSimulator>)
                    .map_err(|error| error.to_string())
                },
            ) as SimulatorFactory,
        );
        Mutex::new(map)
    })
}

/// C0626: `user_simulator.register_user_simulator` — the extension point
/// for new user-simulator types. A new simulator registers a constructor
/// under its config's `type` discriminator string once (typically at
/// startup); [`create_user_simulator`] (this port's stand-in for the
/// scenario branch of `UserSimulatorProvider`'s registry lookup, C0627,
/// now built — see [`UserSimulatorProvider`]) then dispatches to it
/// whenever an `EvalConfig` carries a config of that type. Overwrites any
/// existing registration for the same `config_type`, including the
/// built-in `"llm_backed"` one this module's own [`registry`] seeds — the
/// same override-friendly behavior the source's plain-dict registration
/// already has.
pub fn register_user_simulator(config_type: impl Into<String>, factory: SimulatorFactory) {
    registry()
        .lock()
        .expect("user simulator registry lock poisoned")
        .insert(config_type.into(), factory);
}

/// Looks up and invokes the constructor registered under `config_type`
/// via [`register_user_simulator`] (or the built-in `"llm_backed"`
/// registration [`registry`] seeds).
pub fn create_user_simulator(
    config_type: &str,
    config: &Value,
    conversation_scenario: &ConversationScenario,
) -> Result<Box<dyn UserSimulator>, String> {
    let registry = registry()
        .lock()
        .expect("user simulator registry lock poisoned");
    let factory = registry
        .get(config_type)
        .ok_or_else(|| format!("No user simulator registered for config type {config_type:?}."))?;
    factory(config, conversation_scenario)
}

/// C0627: `user_simulator_provider.UserSimulatorProvider` — provides a
/// [`UserSimulator`] per [`crate::eval_case::EvalCase`], mixing
/// `EvalConfig`-level simulator configuration with per-case conversation
/// data. Dispatch: a case carrying a static `conversation` gets a
/// [`crate::static_user_simulator::StaticUserSimulator`] (or, when the
/// configured type is `"llm_audio"`, that static simulator wrapped in
/// [`crate::llm_audio_user_simulator::LlmAudioUserSimulator`]); a case
/// carrying a `conversation_scenario` gets whatever
/// [`create_user_simulator`] resolves for the configured `type`
/// discriminator.
///
/// **Adaptation, disclosed**: the source's constructor takes a whole
/// `BaseUserSimulatorConfig` *instance* and later reads `type(config)` at
/// dispatch time; this port stores the config as an opaque [`Value`] and
/// reads its embedded `"type"` discriminator string instead — the same
/// registry-by-discriminator-string shape [`create_user_simulator`]
/// itself already established over the source's registry-by-class-object
/// shape. `None` (no config supplied) is stored as a bare
/// `{"type": "llm_backed"}` value, preserving the source's own
/// `_LEGACY_DEFAULT_CONFIG_TYPE = LlmBackedUserSimulatorConfig` fallback
/// (`"llm_backed"` is that config's own `type` discriminator literal).
/// The source's `isinstance(user_simulator_config, BaseUserSimulatorConfig)`
/// constructor-time check isn't ported: this port's config is already
/// untyped `Value` until [`create_user_simulator`] resolves and parses
/// it, so there's no stronger check to perform earlier — a malformed
/// config surfaces as a dispatch/parse error at `provide()` time instead,
/// not a construction-time one.
///
/// **`"llm_backed"` now wired, C0627**: [`registry`] seeds a built-in
/// `"llm_backed"` registration resolving to
/// [`crate::llm_backed_user_simulator::LlmBackedUserSimulator`] (C0628) —
/// a scenario case with no config, or an explicit `"llm_backed"` config,
/// now successfully dispatches through [`create_user_simulator`] instead
/// of hitting its "no simulator registered" error.
///
/// **`"llm_audio"` decorator composition now wired, C0627**: mirrors the
/// source's `if simulator_cls is _LlmAudioUserSimulator: ...` branches in
/// both the scenario and static paths. [`registry`] seeds a built-in
/// `"llm_audio"` registration that parses the resolved config as
/// [`crate::llm_audio_user_simulator::LlmAudioUserSimulatorConfig`],
/// builds an inner `LlmBackedUserSimulatorConfig` from that config's own
/// text-generation fields (`model`/`model_configuration`/
/// `max_allowed_invocations`/`custom_instructions`/`include_function_calls`),
/// constructs the inner `LlmBackedUserSimulator`, and wraps it via
/// `LlmAudioUserSimulator::new`. The static-conversation branch (below,
/// in [`Self::provide`]) checks the same `"type"` discriminator directly
/// and wraps `StaticUserSimulator` the same way when it reads
/// `"llm_audio"` — the source's own static-path special case, since that
/// path never goes through [`create_user_simulator`]'s registry lookup.
pub struct UserSimulatorProvider {
    config: Value,
}

impl UserSimulatorProvider {
    /// The source's `_LEGACY_DEFAULT_CONFIG_TYPE = LlmBackedUserSimulatorConfig`,
    /// named by that config's own `type` discriminator literal.
    pub const LEGACY_DEFAULT_CONFIG_TYPE: &'static str = "llm_backed";

    /// `UserSimulatorProvider.__init__`. `None` falls back to the legacy
    /// default config type — see this struct's own doc.
    pub fn new(user_simulator_config: Option<Value>) -> Self {
        let config = user_simulator_config.unwrap_or_else(|| {
            Value::Map(vec![(
                "type".to_string(),
                Value::String(Self::LEGACY_DEFAULT_CONFIG_TYPE.to_string()),
            )])
        });
        Self { config }
    }

    /// `UserSimulatorProvider.provide` — see this struct's own doc for
    /// the routing rules and the audio-decorator narrowing.
    pub fn provide(
        &self,
        eval_case: &crate::eval_case::EvalCase,
    ) -> Result<Box<dyn UserSimulator>, String> {
        match (&eval_case.conversation, &eval_case.conversation_scenario) {
            (None, None) => Err(
                "Neither static invocations nor conversation scenario provided in EvalCase. \
                 Provide exactly one."
                    .to_string(),
            ),
            (Some(_), Some(_)) => Err(
                "Both static invocations and conversation scenario provided in EvalCase. \
                 Provide exactly one."
                    .to_string(),
            ),
            (None, Some(conversation_scenario)) => {
                let config_type = self
                    .config
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or(Self::LEGACY_DEFAULT_CONFIG_TYPE);
                create_user_simulator(config_type, &self.config, conversation_scenario)
            }
            (Some(conversation), None) => {
                let static_simulator =
                    crate::static_user_simulator::StaticUserSimulator::new(conversation.clone());
                let config_type = self
                    .config
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or(Self::LEGACY_DEFAULT_CONFIG_TYPE);
                if config_type == "llm_audio" {
                    crate::llm_audio_user_simulator::LlmAudioUserSimulator::new(
                        &self.config,
                        Box::new(static_simulator) as Box<dyn UserSimulator + Send + Sync>,
                    )
                    .map(|simulator| Box::new(simulator) as Box<dyn UserSimulator>)
                    .map_err(|error| error.to_string())
                } else {
                    Ok(Box::new(static_simulator))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_user_message_requires_a_message_on_success() {
        let message = NextUserMessage {
            status: Status::Success,
            user_message: None,
        };
        assert!(message.validate().is_err());
    }

    #[test]
    fn next_user_message_forbids_a_message_without_success() {
        let message = NextUserMessage {
            status: Status::StopSignalDetected,
            user_message: Some(Content::user_text("hi")),
        };
        assert!(message.validate().is_err());
    }

    #[test]
    fn next_user_message_accepts_success_with_a_message() {
        let message = NextUserMessage {
            status: Status::Success,
            user_message: Some(Content::user_text("hi")),
        };
        assert!(message.validate().is_ok());
    }

    #[test]
    fn next_user_message_accepts_a_non_success_status_without_a_message() {
        let message = NextUserMessage {
            status: Status::TurnLimitReached,
            user_message: None,
        };
        assert!(message.validate().is_ok());
    }

    #[test]
    fn status_serializes_as_snake_case() {
        let json = rusty_serde::json::to_string(&Status::TurnLimitReached).unwrap();
        assert_eq!(json, "\"turn_limit_reached\"");
    }

    #[test]
    fn base_user_simulator_config_type_field_maps_from_the_type_key() {
        let config: BaseUserSimulatorConfig =
            rusty_serde::json::from_str(r#"{"type":"llm_backed"}"#).unwrap();
        assert_eq!(config.simulator_type, Some("llm_backed".to_string()));
    }

    #[test]
    fn parse_simulator_config_round_trips_through_json() {
        let config = Value::Map(vec![(
            "type".to_string(),
            Value::String("static".to_string()),
        )]);
        let parsed: BaseUserSimulatorConfig = parse_simulator_config(&config).unwrap();
        assert_eq!(parsed.simulator_type, Some("static".to_string()));
    }

    struct NoOpSimulator;
    impl UserSimulator for NoOpSimulator {
        fn get_next_user_message<'a>(
            &'a mut self,
            _events: &'a [Event],
        ) -> BoxFuture<'a, Result<NextUserMessage, String>> {
            Box::pin(async move {
                Ok(NextUserMessage {
                    status: Status::StopSignalDetected,
                    user_message: None,
                })
            })
        }
        fn get_simulation_evaluator(&self) -> Option<Box<dyn Evaluator>> {
            None
        }
    }

    #[test]
    fn register_and_create_dispatches_by_the_type_discriminator() {
        register_user_simulator(
            "no_op_test_simulator",
            Box::new(|_config, _conversation_scenario| Ok(Box::new(NoOpSimulator))),
        );
        let simulator = create_user_simulator(
            "no_op_test_simulator",
            &Value::Map(vec![(
                "type".to_string(),
                Value::String("no_op_test_simulator".to_string()),
            )]),
            &crate::conversation_scenarios::ConversationScenario::new("hi", "plan"),
        );
        assert!(simulator.is_ok());
    }

    #[test]
    fn create_user_simulator_errors_for_an_unregistered_type() {
        let result = create_user_simulator(
            "no_such_type_registered",
            &Value::Null,
            &crate::conversation_scenarios::ConversationScenario::new("hi", "plan"),
        );
        assert!(result.is_err());
    }

    // --- UserSimulatorProvider (C0627) ---

    use crate::eval_case::EvalCase;

    fn eval_case_with_conversation() -> EvalCase {
        EvalCase {
            eval_id: "case-1".to_string(),
            conversation: Some(Vec::new()),
            ..Default::default()
        }
    }

    fn eval_case_with_scenario() -> EvalCase {
        EvalCase {
            eval_id: "case-1".to_string(),
            conversation_scenario: Some(ConversationScenario::new("hi", "plan")),
            ..Default::default()
        }
    }

    #[test]
    fn provide_returns_a_static_simulator_for_a_conversation_case() {
        let provider = UserSimulatorProvider::new(None);
        let simulator = provider.provide(&eval_case_with_conversation()).unwrap();
        drop(simulator);
    }

    #[test]
    fn provide_dispatches_an_explicit_llm_backed_config_to_the_built_in_registration() {
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![(
            "type".to_string(),
            Value::String(UserSimulatorProvider::LEGACY_DEFAULT_CONFIG_TYPE.to_string()),
        )])));
        let simulator = provider.provide(&eval_case_with_scenario());
        assert!(simulator.is_ok());
    }

    #[test]
    fn provide_errors_for_a_case_with_neither_conversation_nor_scenario() {
        let provider = UserSimulatorProvider::new(None);
        let eval_case = EvalCase {
            eval_id: "case-1".to_string(),
            ..Default::default()
        };
        let err = provider.provide(&eval_case).err().unwrap();
        assert!(err.contains("Neither"));
    }

    #[test]
    fn provide_errors_for_a_case_with_both_conversation_and_scenario() {
        let provider = UserSimulatorProvider::new(None);
        let mut eval_case = eval_case_with_conversation();
        eval_case.conversation_scenario = Some(ConversationScenario::new("hi", "plan"));
        let err = provider.provide(&eval_case).err().unwrap();
        assert!(err.contains("Both"));
    }

    #[test]
    fn provide_dispatches_a_scenario_case_through_the_registry() {
        register_user_simulator(
            "provider_test_simulator",
            Box::new(|_config, _conversation_scenario| Ok(Box::new(NoOpSimulator))),
        );
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![(
            "type".to_string(),
            Value::String("provider_test_simulator".to_string()),
        )])));
        let simulator = provider.provide(&eval_case_with_scenario());
        assert!(simulator.is_ok());
    }

    #[test]
    fn provide_defaults_to_the_legacy_config_type_when_none_is_given() {
        // `registry()` now seeds a built-in `"llm_backed"` registration
        // (C0627, this batch) resolving to a real `LlmBackedUserSimulator`
        // — the legacy default dispatches successfully instead of hitting
        // "no simulator registered", matching the source's own
        // "preserves the pre-discriminator behavior of always
        // instantiating LlmBackedUserSimulator" intent.
        let provider = UserSimulatorProvider::new(None);
        let simulator = provider.provide(&eval_case_with_scenario());
        assert!(simulator.is_ok());
    }

    // --- "llm_audio" decorator composition (C0627) ---

    #[test]
    fn provide_dispatches_an_llm_audio_config_to_the_audio_decorator_for_a_scenario_case() {
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![
            ("type".to_string(), Value::String("llm_audio".to_string())),
            (
                "audioModel".to_string(),
                Value::String("gemini-2.5-flash".to_string()),
            ),
        ])));
        let simulator = provider.provide(&eval_case_with_scenario());
        assert!(simulator.is_ok());
    }

    #[test]
    fn provide_wraps_the_static_simulator_in_the_audio_decorator_for_an_llm_audio_config() {
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![
            ("type".to_string(), Value::String("llm_audio".to_string())),
            (
                "audioModel".to_string(),
                Value::String("gemini-2.5-flash".to_string()),
            ),
        ])));
        let simulator = provider.provide(&eval_case_with_conversation());
        assert!(simulator.is_ok());
    }

    #[test]
    fn provide_surfaces_the_unregistered_default_audio_model_as_a_dispatch_error() {
        // The source's default `audio_model` is `"cloud_tts"`, a live
        // Google Cloud TTS backend this port doesn't register (C0631,
        // disclosed). Dispatch itself succeeds (the "llm_audio" type
        // resolves correctly); construction then fails resolving that
        // unregistered model — not "no simulator registered for this
        // type", the same disclosed gap `llm_audio_user_simulator.rs`'s
        // own module doc already establishes.
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![(
            "type".to_string(),
            Value::String("llm_audio".to_string()),
        )])));
        let err = provider.provide(&eval_case_with_scenario()).err().unwrap();
        assert!(!err.contains("No user simulator registered"));
    }

    #[test]
    fn provide_errors_for_a_scenario_case_with_an_unregistered_config_type() {
        let provider = UserSimulatorProvider::new(Some(Value::Map(vec![(
            "type".to_string(),
            Value::String("no_such_type_registered_either".to_string()),
        )])));
        let err = provider.provide(&eval_case_with_scenario()).err().unwrap();
        assert!(err.contains("no_such_type_registered_either"));
    }
}
