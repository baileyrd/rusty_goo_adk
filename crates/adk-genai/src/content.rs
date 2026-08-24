//! A minimal, *real* (not opaque-`Value`) subset of `google.genai.types` —
//! `Content`, `Part`, `FunctionCall`, `FunctionResponse`.
//!
//! **Why this exists, and why it isn't an ADK capability of its own**: these
//! types belong to the third-party `google-genai` SDK package, not to
//! `google.adk` — this migration's inventory scoped to `src/google/adk/`, so
//! porting Google's entire Gemini SDK type system was never in scope (that
//! would be like also being asked to port `pydantic` or `requests`).
//! Phases 1 and 2 therefore represented `Content` as an opaque
//! [`rusty_serde::value::Value`] placeholder wherever it appeared (`Event`,
//! `LlmAgent`'s callback signatures, etc.) — correct for parity of *shape*,
//! but it meant capabilities that inspect `Content`'s actual structure
//! (`Event::is_final_response`, `get_function_calls`/`get_function_responses`,
//! `LlmAgent`'s output-saving helpers) stayed permanently blocked, since a
//! `Value` has no `.parts`/`.function_call` to inspect.
//!
//! Phase 3's own capabilities (`LlmRequest`/`LlmResponse`) genuinely need a
//! real `Content`/`Part` to have any behavior at all (`get_function_calls`
//! literally means "look at `content.parts` for a part with `function_call`
//! set"), so this crate builds the load-bearing subset — the fields
//! `google/adk-python`'s own code actually reads (confirmed by grepping
//! every `part.*`/`content.*`/`.function_call.*`/`.function_response.*`
//! access across the source tree) — as real, inspectable Rust types. Fields
//! ADK's own code never reaches into (`inline_data`'s internal shape,
//! `file_data`, `executable_code`, `code_execution_result`) stay opaque
//! [`rusty_serde::value::Value`] placeholders: real enough to round-trip,
//! not modeled deeper than anything here actually needs.
//!
//! Landing this unblocks work that was explicitly left `REQUIRED` in
//! earlier phases: `Event::is_final_response`/`has_trailing_code_execution_result`
//! (Phase 1, C0022/C0023), `LlmAgent`'s `_get_subagent_to_resume`/
//! `__maybe_save_output_to_state`/`__maybe_accumulate_streaming_output`
//! (Phase 2, C0094/C0095).

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// A function call part of a `Content`, e.g. requested by the model.
///
/// **Adaptation, fixed in Phase 3 batch 3**: the source's `google.genai`
/// pydantic models use `alias_generator=to_camel` (matching `LlmRequest`/
/// `LlmResponse`'s own camelCase wire format) — needed both for ADK's own
/// event/session JSON and for the real Gemini REST API body this type now
/// also serializes into (`gemini.rs`'s `generate_content_async`). This
/// crate's initial cut (Phase 3 batch 1) left these as bare field names;
/// `rename_all = "camelCase"` below closes that gap.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct FunctionCall {
    #[rusty_serde(default)]
    pub id: Option<String>,
    #[rusty_serde(default)]
    pub name: Option<String>,
    #[rusty_serde(default)]
    pub args: Option<std::collections::BTreeMap<String, Value>>,
    /// Streaming-only: whether this call's `args` are still being
    /// incrementally assembled.
    #[rusty_serde(default)]
    pub will_continue: Option<bool>,
}

/// A function response part of a `Content`, e.g. a tool's result sent back
/// to the model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    #[rusty_serde(default)]
    pub id: Option<String>,
    #[rusty_serde(default)]
    pub name: Option<String>,
    #[rusty_serde(default)]
    pub response: Option<std::collections::BTreeMap<String, Value>>,
}

/// A tool's declared calling contract, advertised to the model —
/// `types.FunctionDeclaration`. Added for Phase 8 (`tools/`):
/// `BaseTool::get_declaration` returns one of these, and
/// `adk-tools::append_tools` (C0116) assembles them into
/// `LlmRequest.config.tools`.
///
/// **Adaptation**: only `name`/`description` are real, inspectable fields —
/// ADK's own code reads them (for tool-name dedup in `append_tools`,
/// building instruction/error text). `parameters`/`parameters_json_schema`/
/// `response`/`response_json_schema` (all JSON-Schema-shaped) stay opaque
/// `Value` placeholders: ADK's own code never inspects their internal
/// shape, only forwards whatever a tool builds (e.g. from a Python
/// function's type hints, or Pydantic's own schema generation) straight to
/// the wire.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    #[rusty_serde(default)]
    pub name: Option<String>,
    #[rusty_serde(default)]
    pub description: Option<String>,
    #[rusty_serde(default)]
    pub parameters: Option<Value>,
    #[rusty_serde(default)]
    pub parameters_json_schema: Option<Value>,
    #[rusty_serde(default)]
    pub response: Option<Value>,
    #[rusty_serde(default)]
    pub response_json_schema: Option<Value>,
}

/// Narrowed placeholder for `types.Blob` (`inline_data`) and
/// `types.FileData` (`file_data`) — only `mime_type` is modeled, since
/// `utils/content_utils.py::is_audio_part` (needed by `GeminiLlmConnection`,
/// Phase 3 batch 5) branches on it. The rest of the payload (`data`/
/// `file_uri`/`display_name`) is flattened into `rest` rather than dropped,
/// so round-tripping through JSON doesn't lose it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct MediaBlobStub {
    #[rusty_serde(default)]
    pub mime_type: Option<String>,
    #[rusty_serde(flatten)]
    pub rest: Option<Value>,
}

/// One part of a `Content` — a `oneof`-style union in the source (only one
/// of `text`/`function_call`/`function_response`/... is meaningfully set at
/// a time), represented here as a flat struct of options for simplicity,
/// matching the source's own field layout.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Part {
    #[rusty_serde(default)]
    pub text: Option<String>,
    /// True when `text` is a "thinking" trace rather than the visible
    /// response.
    #[rusty_serde(default)]
    pub thought: Option<bool>,
    /// Opaque placeholder — an opaque signature blob, never inspected by
    /// ADK's own code beyond presence.
    #[rusty_serde(default)]
    pub thought_signature: Option<Value>,
    #[rusty_serde(default)]
    pub function_call: Option<FunctionCall>,
    #[rusty_serde(default)]
    pub function_response: Option<FunctionResponse>,
    /// See [`MediaBlobStub`].
    #[rusty_serde(default)]
    pub inline_data: Option<MediaBlobStub>,
    /// See [`MediaBlobStub`].
    #[rusty_serde(default)]
    pub file_data: Option<MediaBlobStub>,
    /// Opaque placeholder for `types.ExecutableCode`.
    #[rusty_serde(default)]
    pub executable_code: Option<Value>,
    /// Opaque placeholder for `types.CodeExecutionResult`.
    #[rusty_serde(default)]
    pub code_execution_result: Option<Value>,
    /// Opaque placeholder for `types.ToolCall` — a server-side (model-run)
    /// tool invocation, distinct from `function_call` (added in Phase 4
    /// batch 5, C0189: `flows/llm_flows/contents.py`'s `_is_part_invisible`
    /// treats a part carrying one as never invisible, since it must be
    /// echoed back to the model on the next request).
    #[rusty_serde(default)]
    pub tool_call: Option<Value>,
    /// Opaque placeholder for `types.ToolResponse` — the response half of
    /// `tool_call`. See that field's doc.
    #[rusty_serde(default)]
    pub tool_response: Option<Value>,
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Default::default()
        }
    }

    pub fn function_call(call: FunctionCall) -> Self {
        Self {
            function_call: Some(call),
            ..Default::default()
        }
    }

    pub fn function_response(response: FunctionResponse) -> Self {
        Self {
            function_response: Some(response),
            ..Default::default()
        }
    }
}

/// A single turn's content — a `role` plus its `parts`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Content {
    #[rusty_serde(default)]
    pub role: Option<String>,
    #[rusty_serde(default)]
    pub parts: Vec<Part>,
}

impl Content {
    pub fn new(role: impl Into<String>, parts: Vec<Part>) -> Self {
        Self {
            role: Some(role.into()),
            parts,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::new("user", vec![Part::text(text)])
    }

    pub fn get_function_calls(&self) -> Vec<&FunctionCall> {
        self.parts
            .iter()
            .filter_map(|part| part.function_call.as_ref())
            .collect()
    }

    pub fn get_function_responses(&self) -> Vec<&FunctionResponse> {
        self.parts
            .iter()
            .filter_map(|part| part.function_response.as_ref())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_function_calls_extracts_only_call_bearing_parts() {
        let content = Content::new(
            "model",
            vec![
                Part::text("checking..."),
                Part::function_call(FunctionCall {
                    name: Some("get_weather".to_string()),
                    ..Default::default()
                }),
            ],
        );
        let calls = content.get_function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn get_function_responses_extracts_only_response_bearing_parts() {
        let content = Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        );
        assert_eq!(content.get_function_responses().len(), 1);
    }

    #[test]
    fn media_blob_stub_round_trips_mime_type_and_flattens_the_rest() {
        let json = r#"{"mimeType":"audio/pcm","data":"base64data","displayName":"clip"}"#;
        let blob: MediaBlobStub = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(blob.mime_type.as_deref(), Some("audio/pcm"));
        assert!(blob.rest.is_some());

        let round_tripped = rusty_serde::json::to_string(&blob).unwrap();
        let blob_again: MediaBlobStub = rusty_serde::json::from_str(&round_tripped).unwrap();
        assert_eq!(blob, blob_again);
    }

    #[test]
    fn is_audio_part_is_reachable_through_inline_data_mime_type() {
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("audio/wav".to_string()),
                rest: None,
            }),
            ..Default::default()
        };
        assert_eq!(
            part.inline_data.as_ref().unwrap().mime_type.as_deref(),
            Some("audio/wav")
        );
    }

    #[test]
    fn user_text_builds_a_single_text_part_user_turn() {
        let content = Content::user_text("hello");
        assert_eq!(content.role.as_deref(), Some("user"));
        assert_eq!(content.parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn round_trips_through_json() {
        let content = Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                id: Some("fc-1".to_string()),
                name: Some("tool".to_string()),
                args: None,
                will_continue: None,
            })],
        );
        let json = rusty_serde::json::to_string(&content).unwrap();
        let back: Content = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(content, back);
    }

    #[test]
    fn multi_word_fields_serialize_as_camel_case_on_the_wire() {
        let content = Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                id: None,
                name: Some("tool".to_string()),
                args: None,
                will_continue: Some(true),
            })],
        );
        let json = rusty_serde::json::to_string(&content).unwrap();
        assert!(json.contains("\"functionCall\""));
        assert!(json.contains("\"willContinue\""));
        assert!(!json.contains("function_call"));
        assert!(!json.contains("will_continue"));
    }

    #[test]
    fn function_declaration_round_trips_with_camel_case_field_names() {
        let declaration = FunctionDeclaration {
            name: Some("get_weather".to_string()),
            description: Some("Looks up the weather".to_string()),
            parameters: Some(Value::Map(vec![(
                "type".to_string(),
                Value::String("object".to_string()),
            )])),
            parameters_json_schema: None,
            response: None,
            response_json_schema: None,
        };
        let json = rusty_serde::json::to_string(&declaration).unwrap();
        assert!(json.contains("\"parameters\""));
        assert!(!json.contains("parameters_json_schema"));

        let back: FunctionDeclaration = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(declaration, back);
    }

    #[test]
    fn function_declaration_camel_cases_its_multi_word_field_names() {
        let declaration = FunctionDeclaration {
            parameters_json_schema: Some(Value::Bool(true)),
            response_json_schema: Some(Value::Bool(true)),
            ..Default::default()
        };
        let json = rusty_serde::json::to_string(&declaration).unwrap();
        assert!(json.contains("\"parametersJsonSchema\""));
        assert!(json.contains("\"responseJsonSchema\""));
        assert!(!json.contains("parameters_json_schema"));
        assert!(!json.contains("response_json_schema"));
    }
}
