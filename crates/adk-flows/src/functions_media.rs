//! Capability C0195: `functions.py`'s multimodal tool-result extraction —
//! `_as_function_response_part`/`_extract_media_from_entry`/
//! `_extract_multimodal_parts`, ported from
//! `google.adk.flows.llm_flows.functions`.
//!
//! A tool's result is otherwise required to be JSON-serializable, which
//! leaves no way to hand back media except by encoding it into a string
//! the model reads as text. A tool that produces an image, audio clip, or
//! document instead returns a part carrying `inline_data`/`file_data`, on
//! its own or among the entries of a returned container (which may itself
//! hold a container of parts) — this module pulls those out (bounded to
//! one level of container nesting, `_MAX_MEDIA_CONTAINER_DEPTH`) before
//! the rest of the result is coerced to a plain JSON dict, in
//! [`crate::functions::execute_single_function_call`]'s response-content
//! step.
//!
//! **Adaptation, disclosed**: the source's `_as_function_response_part`
//! checks `isinstance(value, types.Part)` — a real Python object identity
//! check. This port's tools only ever return an already-JSON-shaped
//! [`Value`] (there's no way to embed a typed `Part` object inside an
//! arbitrary result tree the way a Python tool can), so the check here is
//! necessarily structural instead: a `Value::Map` that round-trips
//! through `rusty_serde::json::from_value::<Part>` (the same opaque-
//! payload round-trip convention already used elsewhere in this port,
//! e.g. `load_artifacts_tool.rs`'s own `Part` deserialization) and
//! carries a populated `inline_data`/`file_data` counts as media here.
//! This is looser than the source's identity check — a plain dict that
//! happens to carry `inlineData`/`fileData`-shaped keys would also match
//! here, where Python's `isinstance` check would reject a bare `dict` —
//! but it's the only representation a tool constructing media in this
//! port actually has to work with (build an
//! `adk_genai::content::Part { inline_data: Some(..), ..Default::default() }`,
//! convert it via `to_value`, and return that).
//!
//! Not ported: computer-use image decoding
//! (`_try_decode_computer_use_image`) — needs `ComputerUseTool`, not
//! built in this port.

use rusty_serde::value::Value;

use adk_genai::content::{FunctionResponsePart, MediaBlobStub, Part};

/// `_MAX_MEDIA_CONTAINER_DEPTH`: only one level of container nesting is
/// descended into looking for media.
const MAX_MEDIA_CONTAINER_DEPTH: usize = 1;

/// `_as_function_response_part`: converts `value` into a
/// [`FunctionResponsePart`] if it structurally looks like a `Part`
/// carrying usable inline or file-referenced media (mirrors the source's
/// own `blob.data is not None and blob.mime_type` / `file.file_uri and
/// file.mime_type` checks) — see the module doc for the disclosed
/// narrowing relative to the source's `isinstance` check.
pub fn as_function_response_part(value: &Value) -> Option<FunctionResponsePart> {
    let part = rusty_serde::json::from_value::<Part>(value.clone()).ok()?;

    if let Some(blob) = &part.inline_data {
        if blob.mime_type.is_some() && is_present(blob_field(blob, "data")) {
            return Some(FunctionResponsePart {
                inline_data: Some(blob.clone()),
                file_data: None,
            });
        }
    }
    if let Some(file) = &part.file_data {
        if file.mime_type.is_some() && is_present(blob_field(file, "fileUri")) {
            return Some(FunctionResponsePart {
                inline_data: None,
                file_data: Some(file.clone()),
            });
        }
    }
    None
}

/// Reads `key` out of a [`MediaBlobStub`]'s flattened `rest` payload.
fn blob_field<'a>(blob: &'a MediaBlobStub, key: &str) -> Option<&'a Value> {
    match &blob.rest {
        Some(Value::Map(entries)) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// Mirrors Python's `is not None` truthiness for a field that might be
/// absent from `rest` entirely, or present but explicitly `null`.
fn is_present(value: Option<&Value>) -> bool {
    !matches!(value, None | Some(Value::Null))
}

fn is_empty_container(value: &Value) -> bool {
    match value {
        Value::Map(entries) => entries.is_empty(),
        Value::Seq(items) => items.is_empty(),
        _ => false,
    }
}

/// `_extract_media_from_entry`: removes media from one entry of a tool
/// result. Any parts found are appended to `parts`. Only maps and
/// sequences are descended into, so an arbitrary scalar a tool returns is
/// left alone. Returns whether the entry should be kept, and what's left
/// of it — an entry that was media itself, or a container left empty once
/// its media was taken out, is not kept.
fn extract_media_from_entry(
    value: &Value,
    parts: &mut Vec<FunctionResponsePart>,
    depth: usize,
) -> (bool, Value) {
    if let Some(part) = as_function_response_part(value) {
        parts.push(part);
        return (false, Value::Null);
    }
    if depth >= MAX_MEDIA_CONTAINER_DEPTH || !matches!(value, Value::Map(_) | Value::Seq(_)) {
        return (true, value.clone());
    }
    let (remaining, nested_parts) = extract_multimodal_parts_at_depth(value.clone(), depth + 1);
    if nested_parts.is_empty() {
        return (true, value.clone());
    }
    parts.extend(nested_parts);
    let keep = !is_empty_container(&remaining);
    (keep, remaining)
}

/// `_extract_multimodal_parts`: moves media in a tool result into
/// function response parts. Returns the result with the media removed,
/// and the extracted parts (empty when the result carries no media, in
/// which case the result is returned unchanged).
pub fn extract_multimodal_parts(function_result: Value) -> (Value, Vec<FunctionResponsePart>) {
    extract_multimodal_parts_at_depth(function_result, 0)
}

fn extract_multimodal_parts_at_depth(
    function_result: Value,
    depth: usize,
) -> (Value, Vec<FunctionResponsePart>) {
    if let Some(single_part) = as_function_response_part(&function_result) {
        return (Value::Map(vec![]), vec![single_part]);
    }

    let mut parts = Vec::new();
    let remaining = match &function_result {
        Value::Map(entries) => {
            let mut kept_items = Vec::new();
            for (key, value) in entries {
                let (keep, kept) = extract_media_from_entry(value, &mut parts, depth);
                if keep {
                    kept_items.push((key.clone(), kept));
                }
            }
            Value::Map(kept_items)
        }
        Value::Seq(items) => {
            let mut kept_values = Vec::new();
            for value in items {
                let (keep, kept) = extract_media_from_entry(value, &mut parts, depth);
                if keep {
                    kept_values.push(kept);
                }
            }
            Value::Seq(kept_values)
        }
        _ => return (function_result, Vec::new()),
    };

    if parts.is_empty() {
        return (function_result, Vec::new());
    }
    let remaining = if is_empty_container(&remaining) {
        Value::Map(vec![])
    } else {
        remaining
    };
    (remaining, parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media_part(mime_type: &str, data: &str) -> Value {
        rusty_serde::json::to_value(&Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some(mime_type.to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(data.to_string()),
                )])),
            }),
            ..Default::default()
        })
        .unwrap()
    }

    fn file_part(mime_type: &str, uri: &str) -> Value {
        rusty_serde::json::to_value(&Part {
            file_data: Some(MediaBlobStub {
                mime_type: Some(mime_type.to_string()),
                rest: Some(Value::Map(vec![(
                    "fileUri".to_string(),
                    Value::String(uri.to_string()),
                )])),
            }),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn as_function_response_part_recognizes_inline_media() {
        let value = media_part("image/png", "base64data");
        let part = as_function_response_part(&value).unwrap();
        assert_eq!(
            part.inline_data.unwrap().mime_type.as_deref(),
            Some("image/png")
        );
        assert!(part.file_data.is_none());
    }

    #[test]
    fn as_function_response_part_recognizes_file_referenced_media() {
        let value = file_part("audio/wav", "gs://bucket/clip.wav");
        let part = as_function_response_part(&value).unwrap();
        assert_eq!(
            part.file_data.unwrap().mime_type.as_deref(),
            Some("audio/wav")
        );
        assert!(part.inline_data.is_none());
    }

    #[test]
    fn as_function_response_part_rejects_a_plain_text_part() {
        let value = rusty_serde::json::to_value(&Part::text("hello")).unwrap();
        assert!(as_function_response_part(&value).is_none());
    }

    #[test]
    fn as_function_response_part_rejects_inline_data_missing_mime_type() {
        let value = rusty_serde::json::to_value(&Part {
            inline_data: Some(MediaBlobStub {
                mime_type: None,
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String("x".to_string()),
                )])),
            }),
            ..Default::default()
        })
        .unwrap();
        assert!(as_function_response_part(&value).is_none());
    }

    #[test]
    fn as_function_response_part_rejects_a_bare_scalar() {
        assert!(as_function_response_part(&Value::String("hi".to_string())).is_none());
        assert!(as_function_response_part(&Value::Null).is_none());
    }

    #[test]
    fn extract_multimodal_parts_returns_the_result_unchanged_when_nothing_is_media() {
        let value = Value::Map(vec![("count".to_string(), Value::UInt(3))]);
        let (remaining, parts) = extract_multimodal_parts(value.clone());
        assert_eq!(remaining, value);
        assert!(parts.is_empty());
    }

    #[test]
    fn extract_multimodal_parts_lifts_a_bare_media_result() {
        let value = media_part("image/png", "data");
        let (remaining, parts) = extract_multimodal_parts(value);
        assert_eq!(remaining, Value::Map(vec![]));
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn extract_multimodal_parts_pulls_media_out_of_a_map_entry() {
        let value = Value::Map(vec![
            ("label".to_string(), Value::String("chart".to_string())),
            ("image".to_string(), media_part("image/png", "data")),
        ]);
        let (remaining, parts) = extract_multimodal_parts(value);
        assert_eq!(parts.len(), 1);
        assert_eq!(
            remaining,
            Value::Map(vec![(
                "label".to_string(),
                Value::String("chart".to_string())
            )])
        );
    }

    #[test]
    fn extract_multimodal_parts_pulls_media_out_of_a_list_entry() {
        let value = Value::Seq(vec![
            Value::String("first".to_string()),
            media_part("audio/wav", "data"),
        ]);
        let (remaining, parts) = extract_multimodal_parts(value);
        assert_eq!(parts.len(), 1);
        assert_eq!(
            remaining,
            Value::Seq(vec![Value::String("first".to_string())])
        );
    }

    #[test]
    fn extract_multimodal_parts_becomes_an_empty_map_when_everything_was_media() {
        let value = Value::Seq(vec![media_part("image/png", "a")]);
        let (remaining, parts) = extract_multimodal_parts(value);
        assert_eq!(remaining, Value::Map(vec![]));
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn extract_multimodal_parts_descends_one_level_of_nested_container() {
        let nested = Value::Map(vec![(
            "images".to_string(),
            Value::Seq(vec![
                media_part("image/png", "a"),
                media_part("image/png", "b"),
            ]),
        )]);
        let (remaining, parts) = extract_multimodal_parts(nested);
        assert_eq!(parts.len(), 2);
        // Both entries in the inner list were media, so the whole
        // "images" key is dropped rather than kept as an empty list.
        assert_eq!(remaining, Value::Map(vec![]));
    }

    #[test]
    fn extract_multimodal_parts_does_not_descend_past_the_depth_limit() {
        // A single level of container nesting is inspected (an entry that
        // is itself a container gets recursed into once, at depth 0 -> 1
        // per `_MAX_MEDIA_CONTAINER_DEPTH`); an entry found at depth 1
        // that is itself just another container (not a media part) is
        // kept as-is rather than being recursed into a second time, so
        // media three levels deep is never reached.
        let triply_nested = Value::Seq(vec![Value::Seq(vec![Value::Seq(vec![media_part(
            "image/png",
            "a",
        )])])]);
        let (remaining, parts) = extract_multimodal_parts(triply_nested.clone());
        assert!(parts.is_empty());
        assert_eq!(remaining, triply_nested);
    }
}
