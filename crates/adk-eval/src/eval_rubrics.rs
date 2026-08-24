//! C0607 (Rubric half): `Rubric`/`RubricContent`/`RubricScore`, ported
//! from `google.adk.evaluation.eval_rubrics`.

use rusty_serde::{Deserialize, Serialize};

/// `eval_rubrics.RubricContent`. `text_property` has no default in the
/// source (`Field(description=...)` with no `default=`), so — like
/// pydantic — it's required at construction even though its value may be
/// `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RubricContent {
    pub text_property: Option<String>,
}

/// C0607: `eval_rubrics.Rubric` — a single testable criterion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Rubric {
    pub rubric_id: String,
    pub rubric_content: RubricContent,
    #[rusty_serde(default)]
    pub description: Option<String>,
    /// `type` on the wire — `r#type` as a field name breaks this
    /// codebase's derive macro, so it's renamed at the Rust level only.
    #[rusty_serde(rename = "type", default)]
    pub rubric_type: Option<String>,
}

/// `eval_rubrics.RubricScore` — the score obtained after applying a
/// [`Rubric`] to the Agent's response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RubricScore {
    pub rubric_id: String,
    #[rusty_serde(default)]
    pub rationale: Option<String>,
    #[rusty_serde(default)]
    pub score: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rubric_round_trips_through_json_with_camel_case() {
        let rubric = Rubric {
            rubric_id: "r1".to_string(),
            rubric_content: RubricContent {
                text_property: Some("The response is polite.".to_string()),
            },
            description: Some("Politeness check.".to_string()),
            rubric_type: Some("FINAL_RESPONSE_QUALITY".to_string()),
        };
        let json = rusty_serde::json::to_string(&rubric).unwrap();
        assert!(json.contains("\"rubricId\""));
        assert!(json.contains("\"rubricContent\""));
        assert!(json.contains("\"textProperty\""));
        assert!(json.contains("\"type\":\"FINAL_RESPONSE_QUALITY\""));
        let back: Rubric = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(rubric, back);
    }

    #[test]
    fn rubric_content_requires_the_key_but_allows_a_null_value() {
        let json = r#"{"textProperty":null}"#;
        let content: RubricContent = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(content.text_property, None);
    }

    #[test]
    fn rubric_score_defaults_are_none() {
        let json = r#"{"rubricId":"r1"}"#;
        let score: RubricScore = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(score.rationale, None);
        assert_eq!(score.score, None);
    }
}
