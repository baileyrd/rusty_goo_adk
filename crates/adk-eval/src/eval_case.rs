//! Capability C0605: `Invocation`/`IntermediateData`/`InvocationEvents`
//! and their accessor helpers, ported from `google.adk.evaluation.eval_case`.
//! See the crate root doc for what's deliberately left as an opaque
//! placeholder in this batch.

use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// C0605: `eval_case.IntermediateData` — the legacy container for
/// intermediate data an agent generates en route to a final answer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct IntermediateData {
    #[rusty_serde(default)]
    pub tool_uses: Vec<FunctionCall>,
    #[rusty_serde(default)]
    pub tool_responses: Vec<FunctionResponse>,
    /// `(author, parts)` pairs — a sub-agent name plus the parts of its
    /// intermediate response.
    #[rusty_serde(default)]
    pub intermediate_responses: Vec<(String, Vec<Part>)>,
}

/// C0605: `eval_case.InvocationEvent` — a simple projection of the real
/// `Event` model, intended for the Eval System.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InvocationEvent {
    pub author: String,
    #[rusty_serde(default)]
    pub content: Option<Content>,
    /// `types.GroundingMetadata` — opaque placeholder; no consumer in
    /// this batch inspects it.
    #[rusty_serde(default)]
    pub grounding_metadata: Option<Value>,
}

/// C0605: `eval_case.InvocationEvents` — the modern event-projection
/// container.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InvocationEvents {
    #[rusty_serde(default)]
    pub invocation_events: Vec<InvocationEvent>,
}

/// C0605: `eval_case.IntermediateDataType` — the source's
/// `Union[IntermediateData, InvocationEvents]`, resolved at the Rust
/// level as an explicit enum (no runtime `isinstance` dispatch needed).
#[derive(Debug, Clone, PartialEq)]
pub enum IntermediateDataType {
    Data(IntermediateData),
    Events(InvocationEvents),
}

/// C0605: `eval_case.Invocation` — a single invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Invocation {
    #[rusty_serde(default)]
    pub invocation_id: String,
    pub user_content: Content,
    #[rusty_serde(default)]
    pub final_response: Option<Content>,
    /// `IntermediateDataType`'s two shapes don't share a wire tag in the
    /// source (plain `Union`), so this port keeps the field itself as an
    /// opaque `Value` at the serialization boundary and exposes the
    /// resolved [`IntermediateDataType`] only via
    /// [`Invocation::intermediate_data_type`] — a caller that already
    /// knows which shape it holds parses it explicitly.
    #[rusty_serde(default)]
    pub intermediate_data: Option<Value>,
    #[rusty_serde(default)]
    pub creation_timestamp: f64,
    /// `eval_rubrics.Rubric` — its own still-`REQUIRED` row, C0607.
    #[rusty_serde(default)]
    pub rubrics: Option<Value>,
    /// `app_details.AppDetails` — its own still-`REQUIRED` row, C0610.
    #[rusty_serde(default)]
    pub app_details: Option<Value>,
}

impl Invocation {
    /// Resolves [`Self::intermediate_data`] into a typed
    /// [`IntermediateDataType`], trying `IntermediateData` first (the
    /// more common legacy shape) then `InvocationEvents` — mirroring the
    /// source's `isinstance` checks in `get_all_tool_calls`/
    /// `get_all_tool_responses`, just performed once here instead of at
    /// every call site.
    pub fn intermediate_data_type(&self) -> Option<IntermediateDataType> {
        let value = self.intermediate_data.clone()?;
        if let Ok(data) = rusty_serde::json::from_value::<IntermediateData>(value.clone()) {
            return Some(IntermediateDataType::Data(data));
        }
        rusty_serde::json::from_value::<InvocationEvents>(value)
            .ok()
            .map(IntermediateDataType::Events)
    }
}

/// C0605: `eval_case.get_all_tool_calls`.
pub fn get_all_tool_calls(intermediate_data: Option<&IntermediateDataType>) -> Vec<FunctionCall> {
    match intermediate_data {
        None => Vec::new(),
        Some(IntermediateDataType::Data(data)) => data.tool_uses.clone(),
        Some(IntermediateDataType::Events(events)) => events
            .invocation_events
            .iter()
            .filter_map(|event| event.content.as_ref())
            .flat_map(|content| content.parts.iter())
            .filter_map(|part| part.function_call.clone())
            .collect(),
    }
}

/// C0605: `eval_case.get_all_tool_responses`.
pub fn get_all_tool_responses(
    intermediate_data: Option<&IntermediateDataType>,
) -> Vec<FunctionResponse> {
    match intermediate_data {
        None => Vec::new(),
        Some(IntermediateDataType::Data(data)) => data.tool_responses.clone(),
        Some(IntermediateDataType::Events(events)) => events
            .invocation_events
            .iter()
            .filter_map(|event| event.content.as_ref())
            .flat_map(|content| content.parts.iter())
            .filter_map(|part| part.function_response.clone())
            .collect(),
    }
}

/// C0605: `eval_case.ToolCallAndResponse`.
pub type ToolCallAndResponse = (FunctionCall, Option<FunctionResponse>);

/// C0605: `eval_case.get_all_tool_calls_with_responses`.
pub fn get_all_tool_calls_with_responses(
    intermediate_data: Option<&IntermediateDataType>,
) -> Vec<ToolCallAndResponse> {
    let responses = get_all_tool_responses(intermediate_data);
    get_all_tool_calls(intermediate_data)
        .into_iter()
        .map(|call| {
            let response = responses
                .iter()
                .find(|response| response.id == call.id)
                .cloned();
            (call, response)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function_call(id: &str, name: &str) -> FunctionCall {
        FunctionCall {
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            args: None,
            will_continue: None,
        }
    }

    #[test]
    fn invocation_round_trips_through_json_with_camel_case() {
        let invocation = Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: None,
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        };
        let json = rusty_serde::json::to_string(&invocation).unwrap();
        assert!(json.contains("\"invocationId\""));
        assert!(json.contains("\"userContent\""));
        let back: Invocation = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(invocation, back);
    }

    #[test]
    fn get_all_tool_calls_returns_empty_for_none() {
        assert_eq!(get_all_tool_calls(None), Vec::new());
    }

    #[test]
    fn get_all_tool_calls_reads_from_intermediate_data() {
        let data = IntermediateDataType::Data(IntermediateData {
            tool_uses: vec![function_call("c1", "get_weather")],
            ..Default::default()
        });
        let calls = get_all_tool_calls(Some(&data));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn get_all_tool_calls_reads_from_invocation_events() {
        let events = IntermediateDataType::Events(InvocationEvents {
            invocation_events: vec![InvocationEvent {
                author: "agent".to_string(),
                content: Some(Content::new(
                    "model",
                    vec![Part::function_call(function_call("c1", "get_weather"))],
                )),
                grounding_metadata: None,
            }],
        });
        let calls = get_all_tool_calls(Some(&events));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn get_all_tool_calls_with_responses_pairs_by_id() {
        let data = IntermediateDataType::Data(IntermediateData {
            tool_uses: vec![function_call("c1", "get_weather")],
            tool_responses: vec![FunctionResponse {
                id: Some("c1".to_string()),
                name: Some("get_weather".to_string()),
                response: None,
            }],
            ..Default::default()
        });
        let pairs = get_all_tool_calls_with_responses(Some(&data));
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].1.is_some());
    }

    #[test]
    fn get_all_tool_calls_with_responses_leaves_unmatched_calls_without_a_response() {
        let data = IntermediateDataType::Data(IntermediateData {
            tool_uses: vec![function_call("c1", "get_weather")],
            ..Default::default()
        });
        let pairs = get_all_tool_calls_with_responses(Some(&data));
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].1.is_none());
    }

    #[test]
    fn intermediate_data_type_resolves_the_data_shape() {
        let value = rusty_serde::json::to_value(&IntermediateData {
            tool_uses: vec![function_call("c1", "get_weather")],
            ..Default::default()
        })
        .unwrap();
        let invocation = Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: None,
            intermediate_data: Some(value),
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        };
        assert!(matches!(
            invocation.intermediate_data_type(),
            Some(IntermediateDataType::Data(_))
        ));
    }

    #[test]
    fn intermediate_data_type_resolves_the_events_shape() {
        let value = rusty_serde::json::to_value(&InvocationEvents {
            invocation_events: vec![InvocationEvent {
                author: "agent".to_string(),
                content: None,
                grounding_metadata: None,
            }],
        })
        .unwrap();
        let invocation = Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: None,
            intermediate_data: Some(value),
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        };
        assert!(matches!(
            invocation.intermediate_data_type(),
            Some(IntermediateDataType::Events(_))
        ));
    }
}
