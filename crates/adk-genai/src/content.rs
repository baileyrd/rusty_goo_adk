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
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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
pub struct FunctionResponse {
    #[rusty_serde(default)]
    pub id: Option<String>,
    #[rusty_serde(default)]
    pub name: Option<String>,
    #[rusty_serde(default)]
    pub response: Option<std::collections::BTreeMap<String, Value>>,
}

/// One part of a `Content` — a `oneof`-style union in the source (only one
/// of `text`/`function_call`/`function_response`/... is meaningfully set at
/// a time), represented here as a flat struct of options for simplicity,
/// matching the source's own field layout.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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
    /// Opaque placeholder for `types.Blob` (inline binary data).
    #[rusty_serde(default)]
    pub inline_data: Option<Value>,
    /// Opaque placeholder for `types.FileData`.
    #[rusty_serde(default)]
    pub file_data: Option<Value>,
    /// Opaque placeholder for `types.ExecutableCode`.
    #[rusty_serde(default)]
    pub executable_code: Option<Value>,
    /// Opaque placeholder for `types.CodeExecutionResult`.
    #[rusty_serde(default)]
    pub code_execution_result: Option<Value>,
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
}
