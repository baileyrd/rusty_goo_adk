//! C0488 (partial): `tools.environment_simulation.tool_connection_map` —
//! [`StatefulParameter`]/[`ToolConnectionMap`], the pure data types
//! describing which tools produce/consume a shared stateful parameter
//! (e.g. a `"ticket_id"` one tool creates and another later consumes).
//!
//! **Deferred, disclosed**: this port has no LLM-invocation path to build
//! one of these maps from — the source's only producer,
//! `ToolConnectionAnalyzer.analyze` (an LLM call over a tool list), and
//! its only consumer, `ToolSpecMockStrategy`'s mock-response synthesis,
//! both stay unported. Same LLM-blocked deferral this manifest's C0356
//! row already established for `agent_simulator`/mock-strategy machinery
//! elsewhere. These two structs still get ported on their own — they're
//! plain data, useful to any future caller that builds a
//! [`ToolConnectionMap`] by hand (e.g. a test, or a non-LLM analyzer) even
//! before `ToolConnectionAnalyzer` exists.

use rusty_serde::{Deserialize, Serialize};

/// Represents a stateful parameter and its connections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatefulParameter {
    /// The name of the shared parameter (e.g. `"ticket_id"`).
    pub parameter_name: String,
    /// Tools that generate this parameter.
    pub creating_tools: Vec<String>,
    /// Tools that use this parameter as input.
    pub consuming_tools: Vec<String>,
}

/// Represents the map of tool connections.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolConnectionMap {
    pub stateful_parameters: Vec<StatefulParameter>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_stateful_parameter_through_json() {
        let map = ToolConnectionMap {
            stateful_parameters: vec![StatefulParameter {
                parameter_name: "ticket_id".to_string(),
                creating_tools: vec!["create_ticket".to_string()],
                consuming_tools: vec!["update_ticket".to_string(), "close_ticket".to_string()],
            }],
        };
        let json = rusty_serde::json::to_string(&map).expect("serialize");
        let round_tripped: ToolConnectionMap =
            rusty_serde::json::from_str(&json).expect("deserialize");
        assert_eq!(map, round_tripped);
    }

    #[test]
    fn defaults_to_an_empty_map() {
        assert_eq!(ToolConnectionMap::default().stateful_parameters, Vec::new());
    }
}
