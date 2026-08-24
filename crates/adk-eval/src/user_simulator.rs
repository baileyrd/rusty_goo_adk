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
use std::sync::{Mutex, OnceLock};

use adk_events::Event;
use adk_genai::content::Content;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::evaluator::Evaluator;

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
    /// **Adaptation**: `&mut self`, not `&self` — the one implementor
    /// this batch ports ([`crate::static_user_simulator::StaticUserSimulator`])
    /// advances an internal cursor on every call, matching the source's
    /// own `self.invocation_idx += 1` mutation. Sync, not `async` — no
    /// implementor built so far needs to await anything; a future
    /// LLM-backed implementor will need its own async story, not modeled
    /// here yet (same disclosed adaptation already made for
    /// `evaluator::Evaluator`).
    fn get_next_user_message(&mut self, events: &[Event]) -> NextUserMessage;

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
pub type SimulatorFactory =
    Box<dyn Fn(&Value) -> Result<Box<dyn UserSimulator>, String> + Send + Sync>;

fn registry() -> &'static Mutex<HashMap<String, SimulatorFactory>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, SimulatorFactory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// C0626: `user_simulator.register_user_simulator` — the extension point
/// for new user-simulator types. A new simulator registers a constructor
/// under its config's `type` discriminator string once (typically at
/// startup); [`create_user_simulator`] (this port's stand-in for
/// `UserSimulatorProvider`'s registry lookup, C0627, not built this
/// batch) then dispatches to it whenever an `EvalConfig` carries a
/// config of that type.
pub fn register_user_simulator(config_type: impl Into<String>, factory: SimulatorFactory) {
    registry()
        .lock()
        .expect("user simulator registry lock poisoned")
        .insert(config_type.into(), factory);
}

/// Looks up and invokes the constructor registered under `config_type`
/// via [`register_user_simulator`].
pub fn create_user_simulator(
    config_type: &str,
    config: &Value,
) -> Result<Box<dyn UserSimulator>, String> {
    let registry = registry()
        .lock()
        .expect("user simulator registry lock poisoned");
    let factory = registry
        .get(config_type)
        .ok_or_else(|| format!("No user simulator registered for config type {config_type:?}."))?;
    factory(config)
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
        fn get_next_user_message(&mut self, _events: &[Event]) -> NextUserMessage {
            NextUserMessage {
                status: Status::StopSignalDetected,
                user_message: None,
            }
        }
        fn get_simulation_evaluator(&self) -> Option<Box<dyn Evaluator>> {
            None
        }
    }

    #[test]
    fn register_and_create_dispatches_by_the_type_discriminator() {
        register_user_simulator(
            "no_op_test_simulator",
            Box::new(|_config| Ok(Box::new(NoOpSimulator))),
        );
        let simulator = create_user_simulator(
            "no_op_test_simulator",
            &Value::Map(vec![(
                "type".to_string(),
                Value::String("no_op_test_simulator".to_string()),
            )]),
        );
        assert!(simulator.is_ok());
    }

    #[test]
    fn create_user_simulator_errors_for_an_unregistered_type() {
        let result = create_user_simulator("no_such_type_registered", &Value::Null);
        assert!(result.is_err());
    }
}
