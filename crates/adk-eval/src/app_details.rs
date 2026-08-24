//! C0610: `AgentDetails`/`AppDetails`, ported from
//! `google.adk.evaluation.app_details`. A projection of the actual App
//! (agentic system) capturing only what's relevant to the Eval System.

use std::collections::HashMap;

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// `app_details.AgentDetails` — details about one agent in the App (the
/// root agent or a sub-agent in the Agent Tree).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct AgentDetails {
    pub name: String,
    #[rusty_serde(default)]
    pub instructions: String,
    /// `list[genai_types.ToolListUnion]` at runtime — the source itself
    /// types this `list[Any]` "for Pydantic schema generation
    /// compatibility", so this port keeps it an opaque `Value` list too;
    /// this narrowing is the source's own, not one this port introduces.
    #[rusty_serde(default)]
    pub tool_declarations: Vec<Value>,
}

impl AgentDetails {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: String::new(),
            tool_declarations: Vec::new(),
        }
    }
}

/// C0610: `app_details.AppDetails` — a projection of the App relevant to
/// the Eval System.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct AppDetails {
    #[rusty_serde(default)]
    pub agent_details: HashMap<String, AgentDetails>,
}

impl AppDetails {
    /// `AppDetails.get_developer_instructions`.
    pub fn get_developer_instructions(&self, agent_name: &str) -> Result<&str, String> {
        self.agent_details
            .get(agent_name)
            .map(|details| details.instructions.as_str())
            .ok_or_else(|| format!("`{agent_name}` not found in the agentic system."))
    }

    /// `AppDetails.get_tools_by_agent_name`.
    pub fn get_tools_by_agent_name(&self) -> HashMap<String, Vec<Value>> {
        self.agent_details
            .iter()
            .map(|(name, details)| (name.clone(), details.tool_declarations.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_one_agent() -> AppDetails {
        let mut agent_details = HashMap::new();
        agent_details.insert(
            "root_agent".to_string(),
            AgentDetails {
                name: "root_agent".to_string(),
                instructions: "Be helpful.".to_string(),
                tool_declarations: vec![Value::String("get_weather".to_string())],
            },
        );
        AppDetails { agent_details }
    }

    #[test]
    fn get_developer_instructions_returns_the_agents_instructions() {
        let app = app_with_one_agent();
        assert_eq!(
            app.get_developer_instructions("root_agent"),
            Ok("Be helpful.")
        );
    }

    #[test]
    fn get_developer_instructions_errors_for_an_unknown_agent() {
        let app = app_with_one_agent();
        assert!(app.get_developer_instructions("missing").is_err());
    }

    #[test]
    fn get_tools_by_agent_name_maps_each_agent_to_its_tools() {
        let app = app_with_one_agent();
        let tools = app.get_tools_by_agent_name();
        assert_eq!(
            tools.get("root_agent"),
            Some(&vec![Value::String("get_weather".to_string())])
        );
    }

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let app = app_with_one_agent();
        let json = rusty_serde::json::to_string(&app).unwrap();
        assert!(json.contains("\"agentDetails\""));
        assert!(json.contains("\"toolDeclarations\""));
        let back: AppDetails = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(app, back);
    }
}
