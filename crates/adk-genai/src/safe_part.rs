//! `as_safe_part_for_llm`, ported from
//! `google.adk.tools.load_artifacts_tool` — see the crate-graph note
//! below for why this Rust port relocates it out of `adk-tools`.
//!
//! **Relocated across the crate graph, disclosed**: in the source,
//! `models/google_llm.py` imports this function from
//! `tools/load_artifacts_tool.py` — Python has no acyclic-crate-graph
//! constraint, so `models` depending on `tools` is unremarkable there.
//! This workspace's crate graph runs the other way: `adk-tools` already
//! depends on `adk-models` (for `BaseTool`/`ToolContext`'s LLM-facing
//! surface), so `adk-models` depending on `adk-tools` back would be a
//! cycle. `adk-genai` is the common ancestor both crates already depend
//! on, so this port moves `as_safe_part_for_llm` (and its MIME/base64
//! helpers) here — the same "move shared logic to the crate both sides
//! already depend on" fix already used for `is_audio_part`/
//! `filter_audio_parts` (see `content_utils.rs`'s own consolidation
//! note). `adk-tools::load_artifacts_tool` re-exports both public names
//! from here, so no caller's import path changes.
//!
//! **`maybe_base64_to_bytes`, hand-rolled**: standard alphabet strictly
//! first (rejecting any non-alphabet/non-padding byte, matching
//! `base64.b64decode(..., validate=True)`), then a lenient URL-safe
//! fallback (skipping unrecognized bytes) — not a vetted `base64`
//! crate, since none is a workspace dependency and this is a small,
//! well-defined, independently testable algorithm. A non-empty input
//! with no recognizable base64 characters at all decodes to an empty,
//! useless byte vector under the lenient pass; this is treated as a
//! decode failure (`None`) rather than silently returning zero bytes.

use crate::content::Part;
use rusty_serde::value::Value;

const GEMINI_SUPPORTED_INLINE_MIME_PREFIXES: [&str; 3] = ["image/", "audio/", "video/"];
const GEMINI_SUPPORTED_INLINE_MIME_TYPES: [&str; 1] = ["application/pdf"];
const GEMINI_UNSUPPORTED_INLINE_SUBTYPES: [&str; 3] = ["image/svg", "image/svg+xml", "image/xml"];
const TEXT_LIKE_MIME_TYPES: [&str; 6] = [
    "application/csv",
    "application/json",
    "application/svg+xml",
    "application/xml",
    "image/svg",
    "image/svg+xml",
];

/// Returns the normalized MIME type, without parameters like `charset`.
fn normalize_mime_type(mime_type: Option<&str>) -> Option<String> {
    let mime_type = mime_type?;
    let trimmed = mime_type.split(';').next().unwrap_or("").trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Returns `true` if Gemini accepts this MIME type as inline data.
fn is_inline_mime_type_supported(mime_type: Option<&str>) -> bool {
    let Some(normalized) = normalize_mime_type(mime_type) else {
        return false;
    };
    if GEMINI_UNSUPPORTED_INLINE_SUBTYPES.contains(&normalized.as_str()) {
        return false;
    }
    GEMINI_SUPPORTED_INLINE_MIME_PREFIXES
        .iter()
        .any(|prefix| normalized.starts_with(prefix))
        || GEMINI_SUPPORTED_INLINE_MIME_TYPES.contains(&normalized.as_str())
}

fn base64_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn base64_value_urlsafe(c: u8) -> Option<u8> {
    match c {
        b'-' => Some(62),
        b'_' => Some(63),
        other => base64_value(other),
    }
}

/// Decodes `data` from base64. See the module doc for the two-attempt
/// strict-then-lenient shape and its disclosed adaptation.
pub fn maybe_base64_to_bytes(data: &str) -> Option<Vec<u8>> {
    let result = decode_base64(data, false).or_else(|| decode_base64(data, true))?;
    if result.is_empty() && !data.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn decode_base64(data: &str, lenient_urlsafe: bool) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &byte in data.as_bytes() {
        if byte == b'=' {
            break;
        }
        if byte == b'\n' || byte == b'\r' {
            continue;
        }
        let value = if lenient_urlsafe {
            match base64_value_urlsafe(byte) {
                Some(v) => v,
                None => continue,
            }
        } else {
            base64_value(byte)?
        };
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    Some(bytes)
}

fn text_placeholder(text: impl Into<String>) -> Part {
    Part::text(text)
}

/// `load_artifacts_tool.as_safe_part_for_llm` — returns a `Part` safe to
/// send to an LLM. See the crate root/`adk-tools::load_artifacts_tool`'s
/// module docs for what this narrows relative to the source (no DOCX
/// text extraction, no spreadsheet parsing).
pub fn as_safe_part_for_llm(artifact: &Part, artifact_name: &str) -> Part {
    let Some(inline_data) = &artifact.inline_data else {
        return artifact.clone();
    };

    if is_inline_mime_type_supported(inline_data.mime_type.as_deref()) {
        return artifact.clone();
    }

    let mime_type = normalize_mime_type(inline_data.mime_type.as_deref())
        .unwrap_or_else(|| "application/octet-stream".to_string());

    let raw_data = inline_data.rest.as_ref().and_then(|rest| rest.get("data"));
    let Some(raw_data) = raw_data else {
        return text_placeholder(format!(
            "[Artifact: {artifact_name}, type: {mime_type}. No inline data was provided.]"
        ));
    };

    let data: Vec<u8> = match raw_data {
        Value::String(s) => match maybe_base64_to_bytes(s) {
            Some(bytes) => bytes,
            None => return text_placeholder(s.clone()),
        },
        Value::Seq(items) => items
            .iter()
            .filter_map(|v| match v {
                Value::Int(i) => Some(*i as u8),
                Value::UInt(u) => Some(*u as u8),
                _ => None,
            })
            .collect(),
        _ => return text_placeholder(format!(
            "[Binary artifact: {artifact_name}, type: {mime_type}. Content cannot be displayed inline.]"
        )),
    };

    let is_text_like = mime_type.starts_with("text/")
        || TEXT_LIKE_MIME_TYPES.contains(&mime_type.as_str())
        || ["csv", "txt", "json", "xml"]
            .iter()
            .any(|ext| artifact_name.to_lowercase().ends_with(&format!(".{ext}")));
    if is_text_like {
        return text_placeholder(String::from_utf8_lossy(&data).into_owned());
    }

    let size_kb = data.len() as f64 / 1024.0;
    text_placeholder(format!(
        "[Binary artifact: {artifact_name}, type: {mime_type}, size: {size_kb:.1} KB. Content cannot be displayed inline.]"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::MediaBlobStub;

    #[allow(dead_code)]
    fn media_blob(mime_type: &str, data: Value) -> MediaBlobStub {
        MediaBlobStub {
            mime_type: Some(mime_type.to_string()),
            rest: Some(Value::Map(vec![("data".to_string(), data)])),
        }
    }

    #[test]
    fn normalize_mime_type_strips_parameters() {
        assert_eq!(
            normalize_mime_type(Some("text/plain; charset=utf-8")),
            Some("text/plain".to_string())
        );
        assert_eq!(normalize_mime_type(None), None);
    }

    #[test]
    fn is_inline_mime_type_supported_accepts_images_audio_video_and_pdf() {
        assert!(is_inline_mime_type_supported(Some("image/png")));
        assert!(is_inline_mime_type_supported(Some("audio/mp3")));
        assert!(is_inline_mime_type_supported(Some("video/mp4")));
        assert!(is_inline_mime_type_supported(Some("application/pdf")));
    }

    #[test]
    fn is_inline_mime_type_supported_rejects_svg_and_unrelated_types() {
        assert!(!is_inline_mime_type_supported(Some("image/svg+xml")));
        assert!(!is_inline_mime_type_supported(Some(
            "application/octet-stream"
        )));
        assert!(!is_inline_mime_type_supported(None));
    }

    #[test]
    fn maybe_base64_to_bytes_decodes_standard_and_urlsafe() {
        assert_eq!(
            maybe_base64_to_bytes("aGVsbG8=").unwrap(),
            b"hello".to_vec()
        );
        assert_eq!(
            maybe_base64_to_bytes("aGVsbG8_d29ybGQ").unwrap(),
            b"hello?world".to_vec()
        );
    }

    #[test]
    fn as_safe_part_for_llm_passes_through_supported_inline_mime_types() {
        let artifact = Part {
            inline_data: Some(media_blob("image/png", Value::String("YWJj".to_string()))),
            ..Default::default()
        };
        let safe = as_safe_part_for_llm(&artifact, "photo.png");
        assert_eq!(safe, artifact);
    }

    #[test]
    fn as_safe_part_for_llm_decodes_text_like_data_to_text() {
        let artifact = Part {
            inline_data: Some(media_blob(
                "text/plain",
                Value::String("aGVsbG8gd29ybGQ=".to_string()),
            )),
            ..Default::default()
        };
        let safe = as_safe_part_for_llm(&artifact, "notes.txt");
        assert_eq!(safe.text.as_deref(), Some("hello world"));
    }

    #[test]
    fn as_safe_part_for_llm_falls_back_to_a_binary_placeholder() {
        let artifact = Part {
            inline_data: Some(media_blob(
                "application/octet-stream",
                Value::String("//79/A==".to_string()),
            )),
            ..Default::default()
        };
        let safe = as_safe_part_for_llm(&artifact, "blob.bin");
        let text = safe.text.unwrap();
        assert!(text.contains("Binary artifact: blob.bin"));
        assert!(text.contains("application/octet-stream"));
    }

    #[test]
    fn as_safe_part_for_llm_reports_missing_inline_data() {
        let artifact = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("application/octet-stream".to_string()),
                rest: None,
            }),
            ..Default::default()
        };
        let safe = as_safe_part_for_llm(&artifact, "missing.bin");
        assert!(safe.text.unwrap().contains("No inline data was provided"));
    }

    #[test]
    fn as_safe_part_for_llm_returns_non_inline_parts_unchanged() {
        let artifact = Part::text("plain text part");
        let safe = as_safe_part_for_llm(&artifact, "n/a");
        assert_eq!(safe, artifact);
    }
}
