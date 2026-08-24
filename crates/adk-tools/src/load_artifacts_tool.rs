//! Capability C0425 (partial): `LoadArtifactsTool`/`load_artifacts_tool`,
//! ported from `google.adk.tools.load_artifacts_tool`.
//!
//! **Scope, disclosed**: `as_safe_part_for_llm`'s MIME
//! normalization/classification, base64 decoding, text-like decoding,
//! and binary-placeholder fallback are ported. **Not** ported:
//! - DOCX text extraction (`_try_extract_docx_text`) — needs a zip
//!   reader; no zip-reading crate is a workspace dependency, and adding
//!   one for this single narrow use wasn't judged worth it for this
//!   batch. A `.docx` artifact falls through to the generic
//!   binary-placeholder response instead of extracted text.
//! - Spreadsheet parsing (`_parse_spreadsheet`) — needs a `pandas`
//!   equivalent this port has none of; disabled by default upstream too
//!   (`enable_spreadsheet_parsing=False`), so this is the same
//!   optional-dependency treatment the source itself gives it, not a
//!   narrowing unique to this port.
//! - `process_artifact` (the custom sync/async override callback) — not
//!   exposed; every artifact goes through the built-in safety conversion.
//! - `_maybe_base64_to_bytes` is a hand-rolled decoder (standard alphabet,
//!   strict; URL-safe alphabet, lenient — the same two-attempt shape the
//!   source's own `b64decode`-then-`urlsafe_b64decode` fallback uses), not
//!   a vetted `base64` crate — no such crate is a workspace dependency,
//!   and this is a small, well-defined, independently testable algorithm
//!   (unlike, say, a wire protocol or a security-critical parser).
//!
//! `tool_context.load_artifact`/`list_artifacts` return an opaque
//! `Value` (`adk-agents`'s own disclosed placeholder shape for the
//! not-yet-built Phase 6 artifact backend) — parsed back into a typed
//! [`Part`] via its own `Deserialize` impl, the same pattern
//! `ExampleTool`/`PreloadMemoryTool` already use for `user_content`.

use std::collections::BTreeMap;

use adk_genai::content::{Content, FunctionDeclaration, MediaBlobStub, Part};
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::tool_context::ToolContext;

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

/// Decodes `data` from base64. Tries the standard alphabet strictly
/// first (rejecting any non-alphabet/non-padding byte, matching
/// `base64.b64decode(..., validate=True)`), then falls back to a lenient
/// URL-safe decode (skipping unrecognized bytes) — see the module doc.
fn maybe_base64_to_bytes(data: &str) -> Option<Vec<u8>> {
    decode_base64(data, false).or_else(|| decode_base64(data, true))
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

/// C0425 (partial): returns a `Part` safe to send to an LLM. See the
/// module doc for what this narrows relative to the source.
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

/// C0425 (partial): loads artifacts and adds them to the session.
pub struct LoadArtifactsTool;

impl LoadArtifactsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoadArtifactsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for LoadArtifactsTool {
    fn name(&self) -> &str {
        "load_artifacts"
    }

    fn description(&self) -> &str {
        "Loads artifacts into the session for this request.\n\nNOTE: Call when you need access to artifacts (for example, uploads saved by the\nweb UI)."
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "artifact_names".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("array".to_string())),
                            (
                                "items".to_string(),
                                Value::Map(vec![(
                                    "type".to_string(),
                                    Value::String("string".to_string()),
                                )]),
                            ),
                        ]),
                    )]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, crate::base_tool::ToolError>> {
        Box::pin(async move {
            let artifact_names = args
                .get("artifact_names")
                .cloned()
                .unwrap_or(Value::Seq(Vec::new()));
            Ok(Value::Map(vec![
                ("artifact_names".to_string(), artifact_names),
                (
                    "status".to_string(),
                    Value::String(
                        "artifact contents temporarily inserted and removed. to access these artifacts, call load_artifacts tool again."
                            .to_string(),
                    ),
                ),
            ]))
        })
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                merge_declarations(llm_request, [(self.name().to_string(), declaration)]);
            }
            self.append_artifacts_to_llm_request(tool_context, llm_request)
                .await;
        })
    }
}

impl LoadArtifactsTool {
    async fn append_artifacts_to_llm_request(
        &self,
        tool_context: &mut ToolContext,
        llm_request: &mut LlmRequest,
    ) {
        let Ok(artifact_names) = tool_context.list_artifacts().await else {
            return;
        };
        if artifact_names.is_empty() {
            return;
        }

        let names_json = rusty_serde::json::to_string(&Value::Seq(
            artifact_names.iter().cloned().map(Value::String).collect(),
        ))
        .unwrap_or_default();
        let instruction_text = format!(
            "You have a list of artifacts:\n  {names_json}\n\n  When the user asks questions about any of the artifacts, you should call the\n  `load_artifacts` function to load the artifact. Always call load_artifacts\n  before answering questions related to the artifacts, regardless of whether the\n  artifacts have been loaded before. Do not depend on prior answers about the\n  artifacts.\n  "
        );
        llm_request.append_dynamic_instructions(&[instruction_text]);

        let Some(last_content) = llm_request.contents.last() else {
            return;
        };
        let Some(first_part) = last_content.parts.first() else {
            return;
        };
        let Some(function_response) = &first_part.function_response else {
            return;
        };
        if function_response.name.as_deref() != Some("load_artifacts") {
            return;
        }
        let requested_names: Vec<String> = function_response
            .response
            .as_ref()
            .and_then(|response| response.get("artifact_names"))
            .and_then(|value| match value {
                Value::Seq(items) => Some(
                    items
                        .iter()
                        .filter_map(|v| match v {
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        for artifact_name in requested_names {
            let mut artifact_value = tool_context
                .load_artifact(&artifact_name, None)
                .await
                .ok()
                .flatten();
            if artifact_value.is_none() && !artifact_name.starts_with("user:") {
                let prefixed_name = format!("user:{artifact_name}");
                artifact_value = tool_context
                    .load_artifact(&prefixed_name, None)
                    .await
                    .ok()
                    .flatten();
            }
            let Some(artifact_value) = artifact_value else {
                continue;
            };
            let Ok(artifact) = rusty_serde::json::from_value::<Part>(artifact_value) else {
                continue;
            };

            let artifact_part = as_safe_part_for_llm(&artifact, &artifact_name);

            llm_request.contents.push(Content::new(
                "user",
                vec![
                    Part::text(format!("Artifact {artifact_name} is:")),
                    artifact_part,
                ],
            ));
        }
    }
}

#[allow(dead_code)]
fn media_blob(mime_type: &str, data: Value) -> MediaBlobStub {
    MediaBlobStub {
        mime_type: Some(mime_type.to_string()),
        rest: Some(Value::Map(vec![("data".to_string(), data)])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
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

    #[rusty_tokio::test]
    async fn run_async_echoes_requested_artifact_names_with_a_status() {
        let tool = LoadArtifactsTool::new();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "artifact_names".to_string(),
            Value::Seq(vec![Value::String("a.txt".to_string())]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert!(fields.iter().any(|(k, _)| k == "status"));
                let names = fields.iter().find(|(k, _)| k == "artifact_names").unwrap();
                assert_eq!(
                    names.1,
                    Value::Seq(vec![Value::String("a.txt".to_string())])
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn process_llm_request_is_a_no_op_without_an_artifact_service() {
        let tool = LoadArtifactsTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert!(request.dynamic_instructions().is_empty());
    }
}
