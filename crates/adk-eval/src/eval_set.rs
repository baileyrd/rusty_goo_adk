//! C0607 (`EvalSet` half), ported from `google.adk.evaluation.eval_set`.

use rusty_serde::{Deserialize, Serialize};

use crate::eval_case::EvalCase;

/// C0607: `eval_set.EvalSet` — a set of eval cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalSet {
    pub eval_set_id: String,
    #[rusty_serde(default)]
    pub name: Option<String>,
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub eval_cases: Vec<EvalCase>,
    #[rusty_serde(default)]
    pub creation_timestamp: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_set_round_trips_through_json_with_camel_case() {
        let eval_set = EvalSet {
            eval_set_id: "set-1".to_string(),
            name: Some("Weather queries".to_string()),
            description: None,
            eval_cases: Vec::new(),
            creation_timestamp: 0.0,
        };
        let json = rusty_serde::json::to_string(&eval_set).unwrap();
        assert!(json.contains("\"evalSetId\""));
        assert!(json.contains("\"evalCases\""));
        let back: EvalSet = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(eval_set, back);
    }
}
