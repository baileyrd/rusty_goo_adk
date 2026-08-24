//! C0606 (`ConversationScenario`/`ConversationGenerationConfig` half),
//! ported from `google.adk.evaluation.conversation_scenarios`.
//!
//! **Disclosed narrowing**: `user_persona`'s real type
//! (`simulation.user_simulator_personas.UserPersona`) and the source's
//! `get_default_persona_registry().get_persona(str_id)` string-to-persona
//! resolution both belong to the persona system, its own still-`REQUIRED`
//! manifest row (C0632, `UserBehavior`/`UserPersona`/`UserPersonaRegistry`).
//! `TrajectoryEvaluator`/`RougeEvaluator` — the only `Evaluator`s built so
//! far — never read `user_persona` (the source `del`s the whole
//! `conversation_scenario` parameter in `TrajectoryEvaluator`), so this
//! stays an opaque `Value` here rather than pulling in C0632 unbuilt.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// C0606: `conversation_scenarios.ConversationScenario` — a scenario for
/// a conversation between a simulated user and the Agent under test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ConversationScenario {
    pub starting_prompt: String,
    pub conversation_plan: String,
    /// `Optional[UserPersona]` in the source, with a `field_validator`
    /// resolving a bare string id through the persona registry. See the
    /// module doc for why this stays opaque.
    #[rusty_serde(default)]
    pub user_persona: Option<Value>,
}

impl ConversationScenario {
    pub fn new(starting_prompt: impl Into<String>, conversation_plan: impl Into<String>) -> Self {
        Self {
            starting_prompt: starting_prompt.into(),
            conversation_plan: conversation_plan.into(),
            user_persona: None,
        }
    }
}

/// `conversation_scenarios.ConversationScenarios` — a container purely
/// for (de)serialization convenience.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ConversationScenarios {
    #[rusty_serde(default)]
    pub scenarios: Vec<ConversationScenario>,
}

/// C0606: `conversation_scenarios.ConversationGenerationConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ConversationGenerationConfig {
    pub count: i64,
    #[rusty_serde(default)]
    pub generation_instruction: Option<String>,
    #[rusty_serde(default)]
    pub environment_context: Option<String>,
    pub model_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_scenario_round_trips_through_json_with_camel_case() {
        let scenario = ConversationScenario::new("I need to book a flight.", "Book SFO to LAX.");
        let json = rusty_serde::json::to_string(&scenario).unwrap();
        assert!(json.contains("\"startingPrompt\""));
        assert!(json.contains("\"conversationPlan\""));
        let back: ConversationScenario = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(scenario, back);
    }

    #[test]
    fn conversation_generation_config_round_trips() {
        let config = ConversationGenerationConfig {
            count: 5,
            generation_instruction: None,
            environment_context: None,
            model_name: "gemini-2.5-flash".to_string(),
        };
        let json = rusty_serde::json::to_string(&config).unwrap();
        let back: ConversationGenerationConfig = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }
}
