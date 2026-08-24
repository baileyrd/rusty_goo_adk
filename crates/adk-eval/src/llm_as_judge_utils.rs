//! C0947: `evaluation.llm_as_judge_utils`, ported from
//! `google.adk.evaluation.llm_as_judge_utils`. Shared, pure-computation
//! helpers for LLM-judge-backed metrics — none of the functions here
//! call an LLM themselves, they only prepare text/JSON to send to one or
//! interpret a score once it comes back. Consumed by
//! `final_response_match_v2`/`hallucinations_v1`/`llm_as_judge`/
//! `rubric_based_evaluator`/`rubric_based_final_response_quality_v1`/
//! `rubric_based_tool_use_quality_v1`/`per_turn_user_simulator_quality_v1`
//! — all still `REQUIRED`, blocked on the still-deferred `LlmAsJudge`
//! harness (C0600) — so nothing in this crate calls these functions yet
//! either; they're ported ahead of their consumers the same way
//! `Rubric`/`RubricScore` (C0607) and the criterion types (C0612) were.
//!
//! **`get_text_from_content`, split by type**: the source overloads a
//! single function over `Union[Content, Invocation]`. Rust has no
//! function overloading, so this splits into [`get_text_from_content`]
//! (the `Content` case) and [`get_text_from_invocation`] (the
//! `Invocation` case, which calls the former internally, mirroring the
//! source's own recursive calls). Distinct from, and not a duplicate
//! of, `adk_genai::content_utils::extract_text_from_content` — that's a
//! different source function (`content_utils.py`) with different
//! behavior (drops `thought` parts, concatenates with no separator,
//! always returns a `String` never `None`); this one joins with `"\n"`
//! and returns `None` only when there's no `Content`/no parts at all —
//! note it can still return `Some("")` when parts exist but none carry
//! non-empty text, matching the source's own `"\n".join([...])` over an
//! empty list producing `""`, not `None` (verified against the real
//! source logic run standalone).
//!
//! **`Label`, disclosed improvement**: the source's `PARTIALLY_VALID`
//! member sets its `.value` to a 3-tuple of alternate strings (spread at
//! every call site as `*Label.PARTIALLY_VALID.value`), while every other
//! member's `.value` is a plain string — an inconsistent shape duck-typed
//! around at each call site. [`Label::value`] returns `&'static
//! [&'static str]` uniformly for every variant instead, a strict
//! improvement (one consistent shape) rather than a narrowing.
//!
//! **JSON serialization, disclosed narrowing**: the source calls
//! `model_dump_json(indent=2, exclude_unset=True, exclude_defaults=True,
//! exclude_none=True)` on three small internal models before sending the
//! result to an LLM prompt. `rusty_serde` has no pretty-printer and this
//! port's structs don't support `skip_serializing_if`, so the JSON this
//! port emits is compact with every field present — same disclosed
//! narrowing already established for `local_eval_sets_manager`'s
//! `_write_eval_set_to_path` (C0613).
//!
//! **`_ToolCallAndResponse.tool_response`, adapted**: the source types
//! this `Union[FunctionResponse, str]` (a `FunctionResponse`, or the
//! literal string `"None"` when the response is absent). This port
//! keeps the field an opaque [`Value`] holding either the serialized
//! `FunctionResponse` or `Value::String("None")` — the same "represent a
//! small closed union as `Value` rather than inventing a new enum for a
//! single call site" choice already made elsewhere in this crate.

use std::collections::HashMap;

use adk_genai::content::{Content, FunctionCall};
use rusty_serde::value::Value;
use rusty_serde::Serialize;

use crate::app_details::AppDetails;
use crate::eval_case::{get_all_tool_calls_with_responses, IntermediateDataType, Invocation};
use crate::eval_rubrics::RubricScore;
use crate::evaluator::EvalStatus;

/// `llm_as_judge_utils.Label` — labels for an auto-rater response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    True,
    Invalid,
    Valid,
    PartiallyValid,
    Almost,
    False,
    NotFound,
}

impl Label {
    /// The wire value(s) this label matches. Every variant but
    /// `PartiallyValid` returns a single-element slice; see this
    /// module's doc for why `PartiallyValid` alone carries three.
    pub fn value(&self) -> &'static [&'static str] {
        match self {
            Self::True => &["true"],
            Self::Invalid => &["invalid"],
            Self::Valid => &["valid"],
            Self::PartiallyValid => &["partially_valid", "partially valid", "partially"],
            Self::Almost => &["almost"],
            Self::False => &["false"],
            Self::NotFound => &["label field not found"],
        }
    }
}

/// `llm_as_judge_utils.get_text_from_content` — the `Content` half. See
/// this module's doc for the split from the source's overloaded
/// function and the `Some("")`-vs-`None` distinction.
pub fn get_text_from_content(content: Option<&Content>) -> Option<String> {
    match content {
        Some(content) if !content.parts.is_empty() => {
            let texts: Vec<&str> = content
                .parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .filter(|text| !text.is_empty())
                .collect();
            Some(texts.join("\n"))
        }
        _ => None,
    }
}

/// `llm_as_judge_utils.get_text_from_content` — the `Invocation` half.
/// Returns the text of `invocation`'s final response, optionally
/// prefixed with text from its intermediate events/responses when
/// `include_intermediate_responses_in_final` is set.
pub fn get_text_from_invocation(
    invocation: &Invocation,
    include_intermediate_responses_in_final: bool,
) -> Option<String> {
    if !include_intermediate_responses_in_final {
        return get_text_from_content(invocation.final_response.as_ref());
    }

    let mut parts: Vec<String> = Vec::new();
    match invocation.intermediate_data_type() {
        Some(IntermediateDataType::Events(events)) => {
            for event in &events.invocation_events {
                if let Some(text) = get_text_from_content(event.content.as_ref()) {
                    if !text.is_empty() {
                        parts.push(text);
                    }
                }
            }
        }
        Some(IntermediateDataType::Data(data)) => {
            for (_, response_parts) in &data.intermediate_responses {
                let synthetic = Content {
                    role: None,
                    parts: response_parts.clone(),
                };
                if let Some(text) = get_text_from_content(Some(&synthetic)) {
                    if !text.is_empty() {
                        parts.push(text);
                    }
                }
            }
        }
        None => {}
    }

    if let Some(final_text) = get_text_from_content(invocation.final_response.as_ref()) {
        if !final_text.is_empty() {
            parts.push(final_text);
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// `llm_as_judge_utils.get_eval_status`.
pub fn get_eval_status(score: Option<f64>, threshold: f64) -> EvalStatus {
    match score {
        None => EvalStatus::NotEvaluated,
        Some(score) if score >= threshold => EvalStatus::Passed,
        Some(_) => EvalStatus::Failed,
    }
}

/// `llm_as_judge_utils.get_average_rubric_score` — a single score value
/// from the given rubric scores, or `None` if none of them carry a
/// score.
pub fn get_average_rubric_score(rubric_scores: &[RubricScore]) -> Option<f64> {
    let scores: Vec<f64> = rubric_scores
        .iter()
        .filter_map(|score| score.score)
        .collect();
    if scores.is_empty() {
        None
    } else {
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

/// `llm_as_judge_utils._ToolDeclarations` — internal data model used for
/// serializing Tool declarations.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ToolDeclarations {
    tool_declarations: HashMap<String, Vec<Value>>,
}

/// `llm_as_judge_utils.get_tool_declarations_as_json_str` — a JSON
/// string representation of Tool declarations, intended to be sent to
/// the LLM.
pub fn get_tool_declarations_as_json_str(app_details: &AppDetails) -> Result<String, String> {
    let tool_declarations = ToolDeclarations {
        tool_declarations: app_details.get_tools_by_agent_name(),
    };
    rusty_serde::json::to_string(&tool_declarations).map_err(|error| error.to_string())
}

/// `llm_as_judge_utils._ToolCallAndResponse` — internal data model to
/// capture one single tool call and response. See this module's doc for
/// the `tool_response` adaptation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ToolCallAndResponse {
    step: i64,
    tool_call: FunctionCall,
    tool_response: Value,
}

/// `llm_as_judge_utils._ToolCallsAndResponses` — internal data model
/// used for serializing tool calls and responses.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct ToolCallsAndResponses {
    tool_calls_and_response: Vec<ToolCallAndResponse>,
}

/// `llm_as_judge_utils.get_tool_calls_and_responses_as_json_str` — a
/// JSON string representation of tool calls and their corresponding
/// responses, intended to be sent to the LLM.
pub fn get_tool_calls_and_responses_as_json_str(
    intermediate_data: Option<&IntermediateDataType>,
) -> Result<String, String> {
    let raw_tool_calls_and_responses = get_all_tool_calls_with_responses(intermediate_data);
    if raw_tool_calls_and_responses.is_empty() {
        return Ok("No intermediate steps were taken.".to_string());
    }

    let mut tool_calls_and_responses = Vec::with_capacity(raw_tool_calls_and_responses.len());
    for (step, (tool_call, tool_response)) in raw_tool_calls_and_responses.into_iter().enumerate() {
        let tool_response = match tool_response {
            Some(response) => rusty_serde::json::to_value(&response).map_err(|e| e.to_string())?,
            None => Value::String("None".to_string()),
        };
        tool_calls_and_responses.push(ToolCallAndResponse {
            step: step as i64,
            tool_call,
            tool_response,
        });
    }

    let wrapper = ToolCallsAndResponses {
        tool_calls_and_response: tool_calls_and_responses,
    };
    rusty_serde::json::to_string(&wrapper).map_err(|error| error.to_string())
}

/// `llm_as_judge_utils._GroundingMetadataEntry` — internal data model to
/// capture grounding metadata from an invocation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct GroundingMetadataEntry {
    step: i64,
    #[rusty_serde(default)]
    author: Option<String>,
    grounding_metadata: Value,
}

/// `llm_as_judge_utils._GroundingMetadataEntries` — internal data model
/// used for serializing grounding metadata.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct GroundingMetadataEntries {
    grounding_metadata: Vec<GroundingMetadataEntry>,
}

/// `llm_as_judge_utils.get_grounding_metadata_as_json_str` — a JSON
/// string representation of grounding metadata.
pub fn get_grounding_metadata_as_json_str(
    intermediate_data: Option<&IntermediateDataType>,
) -> Result<String, String> {
    let Some(IntermediateDataType::Events(events)) = intermediate_data else {
        return Ok("No grounding metadata was provided.".to_string());
    };

    let mut grounding_metadata = Vec::new();
    for (step, invocation_event) in events.invocation_events.iter().enumerate() {
        if let Some(metadata) = &invocation_event.grounding_metadata {
            grounding_metadata.push(GroundingMetadataEntry {
                step: step as i64,
                author: Some(invocation_event.author.clone()),
                grounding_metadata: metadata.clone(),
            });
        }
    }

    if grounding_metadata.is_empty() {
        return Ok("No grounding metadata was provided.".to_string());
    }

    let wrapper = GroundingMetadataEntries { grounding_metadata };
    rusty_serde::json::to_string(&wrapper).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_case::{IntermediateData, InvocationEvent, InvocationEvents};
    use adk_genai::content::{FunctionResponse, Part};

    #[test]
    fn label_value_is_uniformly_a_slice() {
        assert_eq!(Label::True.value(), &["true"]);
        assert_eq!(
            Label::PartiallyValid.value(),
            &["partially_valid", "partially valid", "partially"]
        );
    }

    #[test]
    fn get_text_from_content_returns_none_for_no_content() {
        assert_eq!(get_text_from_content(None), None);
    }

    #[test]
    fn get_text_from_content_returns_none_for_no_parts() {
        assert_eq!(
            get_text_from_content(Some(&Content {
                role: None,
                parts: vec![]
            })),
            None
        );
    }

    #[test]
    fn get_text_from_content_returns_empty_string_when_no_part_has_text() {
        let content = Content {
            role: None,
            parts: vec![Part::default(), Part::default()],
        };
        assert_eq!(get_text_from_content(Some(&content)), Some(String::new()));
    }

    #[test]
    fn get_text_from_content_joins_non_empty_texts_with_a_newline() {
        let content = Content {
            role: None,
            parts: vec![Part::text("a"), Part::default(), Part::text("b")],
        };
        assert_eq!(
            get_text_from_content(Some(&content)),
            Some("a\nb".to_string())
        );
    }

    fn invocation() -> Invocation {
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: Some(Content::user_text("final")),
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    #[test]
    fn get_text_from_invocation_returns_only_final_response_when_flag_is_off() {
        assert_eq!(
            get_text_from_invocation(&invocation(), false),
            Some("final".to_string())
        );
    }

    #[test]
    fn get_text_from_invocation_prepends_invocation_events_text() {
        let mut inv = invocation();
        inv.intermediate_data = Some(
            rusty_serde::json::to_value(&InvocationEvents {
                invocation_events: vec![InvocationEvent {
                    author: "agent".to_string(),
                    content: Some(Content::user_text("intermediate")),
                    grounding_metadata: None,
                }],
            })
            .unwrap(),
        );
        assert_eq!(
            get_text_from_invocation(&inv, true),
            Some("intermediate\nfinal".to_string())
        );
    }

    #[test]
    fn get_text_from_invocation_prepends_intermediate_data_responses() {
        let mut inv = invocation();
        inv.intermediate_data = Some(
            rusty_serde::json::to_value(&IntermediateData {
                tool_uses: vec![],
                tool_responses: vec![],
                intermediate_responses: vec![("agent".to_string(), vec![Part::text("mid")])],
            })
            .unwrap(),
        );
        assert_eq!(
            get_text_from_invocation(&inv, true),
            Some("mid\nfinal".to_string())
        );
    }

    #[test]
    fn get_eval_status_matches_the_source() {
        assert_eq!(get_eval_status(None, 0.5), EvalStatus::NotEvaluated);
        assert_eq!(get_eval_status(Some(0.5), 0.5), EvalStatus::Passed);
        assert_eq!(get_eval_status(Some(0.4), 0.5), EvalStatus::Failed);
    }

    #[test]
    fn get_average_rubric_score_returns_none_without_any_scores() {
        let scores = vec![RubricScore {
            rubric_id: "r1".to_string(),
            rationale: None,
            score: None,
        }];
        assert_eq!(get_average_rubric_score(&scores), None);
    }

    #[test]
    fn get_average_rubric_score_averages_present_scores() {
        let scores = vec![
            RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(1.0),
            },
            RubricScore {
                rubric_id: "r2".to_string(),
                rationale: None,
                score: None,
            },
            RubricScore {
                rubric_id: "r3".to_string(),
                rationale: None,
                score: Some(0.0),
            },
        ];
        assert_eq!(get_average_rubric_score(&scores), Some(0.5));
    }

    #[test]
    fn get_tool_declarations_as_json_str_uses_camel_case() {
        let app_details = AppDetails::default();
        let json = get_tool_declarations_as_json_str(&app_details).unwrap();
        assert!(json.contains("\"toolDeclarations\""));
    }

    #[test]
    fn get_tool_calls_and_responses_as_json_str_reports_no_steps() {
        assert_eq!(
            get_tool_calls_and_responses_as_json_str(None).unwrap(),
            "No intermediate steps were taken."
        );
    }

    #[test]
    fn get_tool_calls_and_responses_as_json_str_uses_none_for_a_missing_response() {
        let data = IntermediateDataType::Data(IntermediateData {
            tool_uses: vec![FunctionCall {
                id: Some("c1".to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            }],
            tool_responses: vec![],
            intermediate_responses: vec![],
        });
        let json = get_tool_calls_and_responses_as_json_str(Some(&data)).unwrap();
        assert!(json.contains("\"toolResponse\":\"None\""));
    }

    #[test]
    fn get_tool_calls_and_responses_as_json_str_serializes_a_real_response() {
        let data = IntermediateDataType::Data(IntermediateData {
            tool_uses: vec![FunctionCall {
                id: Some("c1".to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            }],
            tool_responses: vec![FunctionResponse {
                id: Some("c1".to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            }],
            intermediate_responses: vec![],
        });
        let json = get_tool_calls_and_responses_as_json_str(Some(&data)).unwrap();
        assert!(json.contains("\"step\":0"));
        assert!(!json.contains("\"toolResponse\":\"None\""));
    }

    #[test]
    fn get_grounding_metadata_as_json_str_reports_none_for_non_events_data() {
        assert_eq!(
            get_grounding_metadata_as_json_str(None).unwrap(),
            "No grounding metadata was provided."
        );
        let data = IntermediateDataType::Data(IntermediateData::default());
        assert_eq!(
            get_grounding_metadata_as_json_str(Some(&data)).unwrap(),
            "No grounding metadata was provided."
        );
    }

    #[test]
    fn get_grounding_metadata_as_json_str_reports_none_when_no_event_carries_it() {
        let events = IntermediateDataType::Events(InvocationEvents {
            invocation_events: vec![InvocationEvent {
                author: "agent".to_string(),
                content: None,
                grounding_metadata: None,
            }],
        });
        assert_eq!(
            get_grounding_metadata_as_json_str(Some(&events)).unwrap(),
            "No grounding metadata was provided."
        );
    }

    #[test]
    fn get_grounding_metadata_as_json_str_serializes_present_metadata() {
        let events = IntermediateDataType::Events(InvocationEvents {
            invocation_events: vec![InvocationEvent {
                author: "agent".to_string(),
                content: None,
                grounding_metadata: Some(Value::String("some grounding data".to_string())),
            }],
        });
        let json = get_grounding_metadata_as_json_str(Some(&events)).unwrap();
        assert!(json.contains("\"author\":\"agent\""));
        assert!(json.contains("some grounding data"));
    }
}
