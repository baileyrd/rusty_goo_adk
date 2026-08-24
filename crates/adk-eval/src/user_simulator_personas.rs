//! C0632 (part 1): `evaluation.simulation.user_simulator_personas`, ported
//! from `google.adk.evaluation.simulation.user_simulator_personas`. See
//! [`crate::pre_built_personas`] for the concrete behaviors/personas
//! built from these types.

use rusty_serde::{Deserialize, Serialize};

use adk_errors::not_found::NotFoundError;

/// `user_simulator_personas.UserBehavior` — container for the behavior of
/// a persona.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserBehavior {
    pub name: String,
    pub description: String,
    pub behavior_instructions: Vec<String>,
    pub violation_rubrics: Vec<String>,
}

impl UserBehavior {
    /// `UserBehavior.get_behavior_instructions_str`.
    pub fn get_behavior_instructions_str(&self) -> String {
        self.behavior_instructions
            .iter()
            .map(|instruction| format!("  * {instruction}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `UserBehavior.get_violation_rubrics_str`.
    pub fn get_violation_rubrics_str(&self) -> String {
        self.violation_rubrics
            .iter()
            .map(|rubric| format!("  * {rubric}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// `user_simulator_personas.UserPersona` — container for a persona.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserPersona {
    pub id: String,
    pub description: String,
    pub behaviors: Vec<UserBehavior>,
}

/// `user_simulator_personas.UserPersonaRegistry` — a registry for
/// `UserPersona` instances.
#[derive(Debug, Default)]
pub struct UserPersonaRegistry {
    registry: std::collections::HashMap<String, UserPersona>,
}

impl UserPersonaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// `UserPersonaRegistry.get_persona` — returns the persona registered
    /// under `persona_id`.
    pub fn get_persona(&self, persona_id: &str) -> Result<&UserPersona, NotFoundError> {
        self.registry
            .get(persona_id)
            .ok_or_else(|| NotFoundError::new(format!("{persona_id} not found in registry.")))
    }

    /// `UserPersonaRegistry.register_persona` — registers (or, if already
    /// present, overwrites) the persona under `persona_id`.
    pub fn register_persona(&mut self, persona_id: impl Into<String>, user_persona: UserPersona) {
        self.registry.insert(persona_id.into(), user_persona);
    }

    /// `UserPersonaRegistry.get_registered_personas`.
    pub fn get_registered_personas(&self) -> Vec<&UserPersona> {
        self.registry.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn behavior() -> UserBehavior {
        UserBehavior {
            name: "Test Behavior".to_string(),
            description: "A behavior for testing.".to_string(),
            behavior_instructions: vec!["Do the thing.".to_string(), "Do it well.".to_string()],
            violation_rubrics: vec!["Failed to do the thing.".to_string()],
        }
    }

    #[test]
    fn behavior_instructions_str_bullets_each_instruction() {
        assert_eq!(
            behavior().get_behavior_instructions_str(),
            "  * Do the thing.\n  * Do it well."
        );
    }

    #[test]
    fn violation_rubrics_str_bullets_each_rubric() {
        assert_eq!(
            behavior().get_violation_rubrics_str(),
            "  * Failed to do the thing."
        );
    }

    #[test]
    fn registry_returns_a_registered_persona() {
        let mut registry = UserPersonaRegistry::new();
        let persona = UserPersona {
            id: "TEST".to_string(),
            description: "A test persona.".to_string(),
            behaviors: vec![behavior()],
        };
        registry.register_persona("TEST", persona.clone());
        assert_eq!(registry.get_persona("TEST").unwrap(), &persona);
    }

    #[test]
    fn registry_errors_for_an_unregistered_persona() {
        let registry = UserPersonaRegistry::new();
        assert!(registry.get_persona("NO_SUCH_PERSONA").is_err());
    }

    #[test]
    fn registering_the_same_id_twice_overwrites() {
        let mut registry = UserPersonaRegistry::new();
        registry.register_persona(
            "TEST",
            UserPersona {
                id: "TEST".to_string(),
                description: "first".to_string(),
                behaviors: vec![],
            },
        );
        registry.register_persona(
            "TEST",
            UserPersona {
                id: "TEST".to_string(),
                description: "second".to_string(),
                behaviors: vec![],
            },
        );
        assert_eq!(registry.get_persona("TEST").unwrap().description, "second");
    }

    #[test]
    fn get_registered_personas_returns_every_registered_persona() {
        let mut registry = UserPersonaRegistry::new();
        registry.register_persona(
            "A",
            UserPersona {
                id: "A".to_string(),
                description: "a".to_string(),
                behaviors: vec![],
            },
        );
        registry.register_persona(
            "B",
            UserPersona {
                id: "B".to_string(),
                description: "b".to_string(),
                behaviors: vec![],
            },
        );
        assert_eq!(registry.get_registered_personas().len(), 2);
    }
}
