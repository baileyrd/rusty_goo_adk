//! Capabilities C0136 (consolidated)/C0927-C0929: `Content`/`Part`
//! utility functions, ported from `google.adk.utils.content_utils`.
//!
//! **Consolidation**: `is_audio_part`/`filter_audio_parts` were
//! already ported, but as private duplicates local to
//! `adk-models::gemini_llm_connection` (C0136, Phase 3 batch 5) rather
//! than this shared module — `content_utils.py` itself wasn't in
//! scope yet when that batch needed just those two functions. This
//! batch moves them here as the single source of truth and updates
//! `gemini_llm_connection.rs` to import from here instead of keeping
//! its own copy, the same "reconcile a pre-existing duplicate now that
//! the shared module exists" cleanup this session already did for
//! `TEMP_STATE_PREFIX`/`REQUEST_INPUT_FUNCTION_CALL_NAME`.
//!
//! **New rows, previously uninventoried**: `extract_text_from_content`/
//! `to_user_content`/`SKIP_THOUGHT_SIGNATURE_VALIDATOR` had no
//! manifest rows at all before this batch — `content_utils.py`'s
//! `is_audio_part`/`filter_audio_parts` were the only two exports
//! ever inventoried (folded into C0136's evidence). Per the boundary
//! contract, a capability missing from the original inventory still
//! gets tracked once found, not silently ported without a row — added
//! as C0927 (`extract_text_from_content`), C0928 (`to_user_content`),
//! C0929 (`SKIP_THOUGHT_SIGNATURE_VALIDATOR`).
//!
//! **`to_user_content`, adapted**: the source dispatches on
//! `isinstance(value, ...)` at runtime across `types.Content`/`str`/
//! `BaseModel`/`dict`/`list`/anything else. Rust has no runtime
//! `isinstance`, and callers already know which shape they hold at
//! the call site, so this port takes an explicit [`ToUserContentInput`]
//! enum instead — `Content`/`Text`/`Value` variants. A `BaseModel` in
//! the source becomes, in this port, whatever the caller's own typed
//! struct serializes to via `rusty_serde::json::to_value` before
//! calling this function (the same "the boundary already deals in
//! `Value`" convention this port uses everywhere structs cross an
//! opaque-JSON boundary) — there is no separate "was this originally
//! a model or a plain dict" distinction to preserve on the Rust side.
//! The source's final "anything else -> `str(value)`" catch-all (for
//! an arbitrary Python object with no more specific branch) has no
//! Rust equivalent either — every non-`Content`/`str` input here is
//! already a `Value`, so [`ToUserContentInput::Value`] covers what
//! would otherwise need that generic fallback, formatted as compact
//! JSON rather than Python's `str()`/`repr()` (which, e.g., renders a
//! bool as `True`/`False` rather than `true`/`false`) — a disclosed,
//! low-severity cosmetic divergence for that specific case.
//!
//! **`SKIP_THOUGHT_SIGNATURE_VALIDATOR`, ahead of its own caller**:
//! the source's sole consumer, `_reflect_retry_model_plugin.py`
//! (`ReflectAndRetryToolCallsPlugin`), isn't ported in this workspace
//! yet. The constant is real (a `&'static [u8]` byte string, matching
//! the source's `bytes` value exactly) but nothing here assigns it to
//! a `Part.thought_signature` yet — `thought_signature` stays the
//! already-disclosed opaque `Option<Value>` placeholder
//! (`adk-genai::content`'s own module doc), and no byte-to-`Value`
//! encoding convention exists yet for a caller that would want to.

use rusty_serde::value::Value;

use crate::content::{Content, MediaBlobStub, Part};

/// C0929: placeholder `Part.thought_signature` bytes that bypass
/// backend validation for a part synthesized locally (a model turn or
/// tool call/response the model never produced) — see the module doc
/// for why nothing here assigns it to a real `Part` yet.
pub const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &[u8] = b"skip_thought_signature_validator";

fn mime_starts_with_audio(blob: &MediaBlobStub) -> bool {
    blob.mime_type
        .as_deref()
        .map(|m| m.starts_with("audio/"))
        .unwrap_or(false)
}

/// C0136: `content_utils.is_audio_part`.
pub fn is_audio_part(part: &Part) -> bool {
    part.inline_data
        .as_ref()
        .map(mime_starts_with_audio)
        .unwrap_or(false)
        || part
            .file_data
            .as_ref()
            .map(mime_starts_with_audio)
            .unwrap_or(false)
}

/// C0136: `content_utils.filter_audio_parts`.
pub fn filter_audio_parts(content: &Content) -> Option<Content> {
    if content.parts.is_empty() {
        return None;
    }
    let filtered: Vec<Part> = content
        .parts
        .iter()
        .filter(|part| !is_audio_part(part))
        .cloned()
        .collect();
    if filtered.is_empty() {
        return None;
    }
    Some(Content {
        role: content.role.clone(),
        parts: filtered,
    })
}

/// C0927: `content_utils.extract_text_from_content` — extracts text
/// from a `Content`, filtering out "thinking" parts (`thought:
/// true`).
pub fn extract_text_from_content(content: Option<&Content>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    content
        .parts
        .iter()
        .filter(|part| !part.thought.unwrap_or(false))
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .concat()
}

/// `content_utils.to_user_content`'s input shape — see the module doc
/// for why this replaces the source's runtime `isinstance` dispatch.
pub enum ToUserContentInput {
    /// Re-wrapped with `role: "user"` — parts are moved, not
    /// deep-copied, matching the source's shared-not-copied `parts`.
    Content(Content),
    Text(String),
    /// Covers the source's `BaseModel`/`dict`/`list`/anything-else
    /// branches — see the module doc for the disclosed formatting
    /// divergence for non-string, non-Content values.
    Value(Value),
}

/// C0928: `content_utils.to_user_content` — coerces an arbitrary
/// value into a user-role `Content`.
pub fn to_user_content(value: ToUserContentInput) -> Content {
    match value {
        ToUserContentInput::Content(content) => Content {
            role: Some("user".to_string()),
            parts: content.parts,
        },
        ToUserContentInput::Text(text) => Content::user_text(text),
        ToUserContentInput::Value(Value::String(text)) => Content::user_text(text),
        ToUserContentInput::Value(value) => Content::user_text(
            rusty_serde::json::to_string(&value).unwrap_or_else(|_| format!("{value:?}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::FunctionCall;

    fn audio_part() -> Part {
        Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("audio/wav".to_string()),
                rest: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn is_audio_part_detects_inline_audio() {
        assert!(is_audio_part(&audio_part()));
    }

    #[test]
    fn is_audio_part_detects_file_reference_audio() {
        let part = Part {
            file_data: Some(MediaBlobStub {
                mime_type: Some("audio/mpeg".to_string()),
                rest: None,
            }),
            ..Default::default()
        };
        assert!(is_audio_part(&part));
    }

    #[test]
    fn is_audio_part_is_false_for_text() {
        assert!(!is_audio_part(&Part::text("hello")));
    }

    #[test]
    fn filter_audio_parts_drops_only_audio_parts() {
        let content = Content::new("user", vec![audio_part(), Part::text("hello")]);
        let filtered = filter_audio_parts(&content).unwrap();
        assert_eq!(filtered.parts.len(), 1);
        assert_eq!(filtered.parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn filter_audio_parts_returns_none_when_every_part_is_audio() {
        let content = Content::new("user", vec![audio_part()]);
        assert_eq!(filter_audio_parts(&content), None);
    }

    #[test]
    fn extract_text_from_content_joins_text_parts() {
        let content = Content::new("model", vec![Part::text("hello "), Part::text("world")]);
        assert_eq!(extract_text_from_content(Some(&content)), "hello world");
    }

    #[test]
    fn extract_text_from_content_filters_out_thoughts() {
        let content = Content::new(
            "model",
            vec![
                Part {
                    text: Some("thinking...".to_string()),
                    thought: Some(true),
                    ..Default::default()
                },
                Part::text("the answer"),
            ],
        );
        assert_eq!(extract_text_from_content(Some(&content)), "the answer");
    }

    #[test]
    fn extract_text_from_content_returns_empty_for_none() {
        assert_eq!(extract_text_from_content(None), "");
    }

    #[test]
    fn extract_text_from_content_ignores_non_text_parts() {
        let content = Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                name: Some("tool".to_string()),
                ..Default::default()
            })],
        );
        assert_eq!(extract_text_from_content(Some(&content)), "");
    }

    #[test]
    fn to_user_content_rewraps_content_with_the_user_role() {
        let content = Content::new("model", vec![Part::text("hi")]);
        let result = to_user_content(ToUserContentInput::Content(content));
        assert_eq!(result.role.as_deref(), Some("user"));
        assert_eq!(result.parts[0].text.as_deref(), Some("hi"));
    }

    #[test]
    fn to_user_content_wraps_a_string_as_a_text_part() {
        let result = to_user_content(ToUserContentInput::Text("hello".to_string()));
        assert_eq!(result.role.as_deref(), Some("user"));
        assert_eq!(result.parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn to_user_content_wraps_a_value_string_as_a_text_part() {
        let result = to_user_content(ToUserContentInput::Value(Value::String("hi".to_string())));
        assert_eq!(result.parts[0].text.as_deref(), Some("hi"));
    }

    #[test]
    fn to_user_content_serializes_a_non_string_value_as_json() {
        let value = Value::Map(vec![("k".to_string(), Value::Int(1))]);
        let result = to_user_content(ToUserContentInput::Value(value));
        assert_eq!(result.parts[0].text.as_deref(), Some(r#"{"k":1}"#));
    }
}
