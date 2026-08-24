//! Capability C0391: `File`/`CodeExecutionInput`/`CodeExecutionResult`/
//! `CodeExecutionUtils`, ported from
//! `google.adk.code_executors.code_execution_utils`.
//!
//! **`File.content`, narrowed**: the source's `content: str | bytes`
//! union (either the base64-encoded text or the file's raw bytes) always
//! normalizes here to raw `Vec<u8>` — this port's own code decides when
//! to base64-encode/decode at the I/O boundary (`get_encoded_file_content`)
//! rather than a caller being able to stash either shape in the same
//! field.
//!
//! **`executable_code`/`code_execution_result`, opaque `Value` fields
//! read/written by known wire keys**: `adk_genai::content::Part`'s own
//! module doc discloses these two fields as opaque placeholders ADK's
//! own code never previously reached into. This capability is the first
//! caller that needs to — the same "read/write an opaque `Value` by its
//! known Gemini wire key" pattern already used by
//! `adk-tools::append_tools::append_built_in_tool_marker` (for
//! `LlmRequest.config.tools`) and `adk-events::debug_output` (C0933),
//! not a widening of `Part`'s field types.
//!
//! **Base64, duplicated**: no `base64` crate is a workspace dependency
//! (see `adk-tools::load_artifacts_tool`'s own module doc). This module
//! hand-rolls its own small encode/decode pair rather than reusing
//! `load_artifacts_tool`'s private one (that one is `pub(crate)`, this
//! module needs a byte-in/byte-out shape it doesn't have) or
//! `adk-agents::file_artifact_service`'s (a lower crate this one already
//! depends on, but its decoder takes `&str` not `&[u8]`, and a strict,
//! not lenient, decode is what `get_encoded_file_content`'s exact
//! round-trip check needs) — the same "duplicate locally, small and
//! well-defined" precedent used throughout this port.

use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;

/// C0391: `code_execution_utils.File` — a file name plus its content.
#[derive(Debug, Clone, PartialEq)]
pub struct File {
    pub name: String,
    pub content: Vec<u8>,
    pub mime_type: String,
}

impl File {
    pub fn new(name: impl Into<String>, content: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content,
            mime_type: mime_type.into(),
        }
    }

    /// Builds a `File` with the source's default `mime_type` ("text/plain").
    pub fn with_default_mime_type(name: impl Into<String>, content: Vec<u8>) -> Self {
        Self::new(name, content, "text/plain")
    }
}

/// C0391: `code_execution_utils.CodeExecutionInput`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CodeExecutionInput {
    pub code: String,
    pub input_files: Vec<File>,
    pub execution_id: Option<String>,
}

/// C0391: `code_execution_utils.CodeExecutionResult`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CodeExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub output_files: Vec<File>,
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied();
        let b2 = chunk.get(2).copied();
        out.push(BASE64_ALPHABET[(b0 >> 2) as usize] as char);
        out.push(BASE64_ALPHABET[(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as usize] as char);
        out.push(match b1 {
            Some(b1) => {
                BASE64_ALPHABET[(((b1 & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as usize] as char
            }
            None => '=',
        });
        out.push(match b2 {
            Some(b2) => BASE64_ALPHABET[(b2 & 0x3f) as usize] as char,
            None => '=',
        });
    }
    out
}

fn base64_decode_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Strict standard-alphabet base64 decode: rejects any byte outside the
/// alphabet/padding, matching `base64.b64decode(..., validate=True)`'s
/// strictness (needed for `get_encoded_file_content`'s exact round-trip
/// check — the source's own un-validated `b64decode` is more lenient
/// about stray characters, a disclosed, low-severity divergence for
/// input containing them).
fn base64_decode_strict(data: &[u8]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        if byte == b'=' {
            return if data[i..].iter().all(|&b| b == b'=') {
                Some(bytes)
            } else {
                None
            };
        }
        let value = base64_decode_value(byte)?;
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    Some(bytes)
}

/// C0391: `CodeExecutionUtils.get_encoded_file_content` — returns `data`
/// as base64-encoded bytes, passing it through unchanged if it's already
/// valid base64.
pub fn get_encoded_file_content(data: &[u8]) -> Vec<u8> {
    let already_encoded = base64_decode_strict(data)
        .map(|decoded| base64_encode(&decoded).into_bytes() == data)
        .unwrap_or(false);
    if already_encoded {
        data.to_vec()
    } else {
        base64_encode(data).into_bytes()
    }
}

/// C0391: `CodeExecutionUtils.build_executable_code_part`.
pub fn build_executable_code_part(code: &str) -> Part {
    Part {
        executable_code: Some(Value::Map(vec![
            ("code".to_string(), Value::String(code.to_string())),
            ("language".to_string(), Value::String("PYTHON".to_string())),
        ])),
        ..Default::default()
    }
}

/// C0391: `CodeExecutionUtils.build_code_execution_result_part`.
pub fn build_code_execution_result_part(result: &CodeExecutionResult) -> Part {
    if !result.stderr.is_empty() {
        return Part {
            code_execution_result: Some(Value::Map(vec![
                (
                    "outcome".to_string(),
                    Value::String("OUTCOME_FAILED".to_string()),
                ),
                ("output".to_string(), Value::String(result.stderr.clone())),
            ])),
            ..Default::default()
        };
    }

    let mut final_result = Vec::new();
    if !result.stdout.is_empty() || result.output_files.is_empty() {
        final_result.push(format!("Code execution result:\n{}\n", result.stdout));
    }
    if !result.output_files.is_empty() {
        let names = result
            .output_files
            .iter()
            .map(|f| format!("`{}`", f.name))
            .collect::<Vec<_>>()
            .join(",");
        final_result.push(format!("Saved artifacts:\n{names}"));
    }

    Part {
        code_execution_result: Some(Value::Map(vec![
            (
                "outcome".to_string(),
                Value::String("OUTCOME_OK".to_string()),
            ),
            (
                "output".to_string(),
                Value::String(final_result.join("\n\n")),
            ),
        ])),
        ..Default::default()
    }
}

/// C0391: `CodeExecutionUtils.extract_code_and_truncate_content` —
/// extracts the first code block from `content` and truncates everything
/// after it, mutating `content` in place.
pub fn extract_code_and_truncate_content(
    content: &mut Content,
    code_block_delimiters: &[(String, String)],
) -> Option<String> {
    if content.parts.is_empty() {
        return None;
    }

    for idx in 0..content.parts.len() {
        if content.parts[idx].executable_code.is_none() {
            continue;
        }
        let next_has_result = content
            .parts
            .get(idx + 1)
            .is_some_and(|p| p.code_execution_result.is_some());
        if idx != content.parts.len() - 1 && next_has_result {
            continue;
        }
        let code = content.parts[idx]
            .executable_code
            .as_ref()
            .and_then(|v| v.get("code"))
            .and_then(Value::as_str)
            .map(str::to_string);
        content.parts.truncate(idx + 1);
        return code;
    }

    // Python's `if p.text` is a truthy check: an empty-string text part
    // doesn't count as a text part either.
    let text_parts: Vec<Part> = content
        .parts
        .iter()
        .filter(|p| p.text.as_deref().is_some_and(|t| !t.is_empty()))
        .cloned()
        .collect();
    if text_parts.is_empty() {
        return None;
    }

    let first_text_part = text_parts[0].clone();
    let response_text = text_parts
        .iter()
        .map(|p| p.text.as_deref().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");

    let mut best: Option<(usize, usize, usize)> = None;
    for (lead, trail) in code_block_delimiters {
        let Some(start_idx) = response_text.find(lead.as_str()) else {
            continue;
        };
        let code_start = start_idx + lead.len();
        let Some(end_idx) = response_text[code_start..]
            .find(trail.as_str())
            .map(|i| i + code_start)
        else {
            continue;
        };
        if best.is_none_or(|(best_start, _, _)| start_idx < best_start) {
            best = Some((start_idx, end_idx, lead.len()));
        }
    }
    let (best_start, best_end, best_lead_len) = best?;
    let code_str = &response_text[best_start + best_lead_len..best_end];
    if code_str.is_empty() {
        return None;
    }
    let code_str = code_str.to_string();

    content.parts.clear();
    let prefix_text = &response_text[..best_start];
    if !prefix_text.is_empty() {
        let mut part = first_text_part;
        part.text = Some(prefix_text.to_string());
        content.parts.push(part);
    }
    content.parts.push(build_executable_code_part(&code_str));
    Some(code_str)
}

/// C0391: `CodeExecutionUtils.convert_code_execution_parts` — converts
/// trailing executable-code/code-execution-result parts of `content`
/// into plain text parts, mutating `content` in place.
pub fn convert_code_execution_parts(
    content: &mut Content,
    code_block_delimiter: &(String, String),
    execution_result_delimiters: &(String, String),
) {
    let Some(last_idx) = content.parts.len().checked_sub(1) else {
        return;
    };

    if let Some(code_value) = content.parts[last_idx].executable_code.clone() {
        let code = code_value.get("code").and_then(Value::as_str).unwrap_or("");
        let text = format!(
            "{}{}{}",
            code_block_delimiter.0, code, code_block_delimiter.1
        );
        content.parts[last_idx] = Part::text(text);
    } else if content.parts.len() == 1 {
        if let Some(result_value) = content.parts[last_idx].code_execution_result.clone() {
            match result_value.get("output").and_then(Value::as_str) {
                Some(output) => {
                    let text = format!(
                        "{}{}{}",
                        execution_result_delimiters.0, output, execution_result_delimiters.1
                    );
                    content.parts[last_idx] = Part::text(text);
                }
                None => {
                    content.parts[last_idx] = Part::text("");
                }
            }
            content.role = Some("user".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        for input in [
            b"".as_slice(),
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
        ] {
            let encoded = base64_encode(input);
            assert_eq!(base64_decode_strict(encoded.as_bytes()).unwrap(), input);
        }
    }

    #[test]
    fn get_encoded_file_content_encodes_raw_bytes() {
        let raw = b"hello world";
        let encoded = get_encoded_file_content(raw);
        assert_eq!(encoded, base64_encode(raw).into_bytes());
    }

    #[test]
    fn get_encoded_file_content_passes_through_already_encoded_data() {
        let raw = b"hello world";
        let already_encoded = base64_encode(raw).into_bytes();
        assert_eq!(get_encoded_file_content(&already_encoded), already_encoded);
    }

    #[test]
    fn build_executable_code_part_sets_code_and_python_language() {
        let part = build_executable_code_part("print(1)");
        let value = part.executable_code.unwrap();
        assert_eq!(value.get("code").and_then(Value::as_str), Some("print(1)"));
        assert_eq!(
            value.get("language").and_then(Value::as_str),
            Some("PYTHON")
        );
    }

    #[test]
    fn build_code_execution_result_part_reports_stderr_as_failed() {
        let result = CodeExecutionResult {
            stderr: "boom".to_string(),
            ..Default::default()
        };
        let part = build_code_execution_result_part(&result);
        let value = part.code_execution_result.unwrap();
        assert_eq!(
            value.get("outcome").and_then(Value::as_str),
            Some("OUTCOME_FAILED")
        );
        assert_eq!(value.get("output").and_then(Value::as_str), Some("boom"));
    }

    #[test]
    fn build_code_execution_result_part_lists_saved_artifacts() {
        let result = CodeExecutionResult {
            stdout: String::new(),
            output_files: vec![File::with_default_mime_type("out.png", vec![1, 2, 3])],
            ..Default::default()
        };
        let part = build_code_execution_result_part(&result);
        let value = part.code_execution_result.unwrap();
        let output = value.get("output").and_then(Value::as_str).unwrap();
        assert!(output.contains("Saved artifacts:"));
        assert!(output.contains("`out.png`"));
    }

    #[test]
    fn extract_code_and_truncate_content_finds_a_python_fenced_block() {
        let mut content = Content::new(
            "model",
            vec![Part::text(
                "Let's compute it:\n```python\nprint(1)\n```\ndone",
            )],
        );
        let delimiters = vec![
            ("```tool_code\n".to_string(), "\n```".to_string()),
            ("```python\n".to_string(), "\n```".to_string()),
        ];
        let code = extract_code_and_truncate_content(&mut content, &delimiters);
        assert_eq!(code.as_deref(), Some("print(1)"));
        assert_eq!(content.parts.len(), 2);
        assert_eq!(
            content.parts[0].text.as_deref(),
            Some("Let's compute it:\n")
        );
        assert_eq!(
            content.parts[1]
                .executable_code
                .as_ref()
                .and_then(|v| v.get("code"))
                .and_then(Value::as_str),
            Some("print(1)")
        );
    }

    #[test]
    fn extract_code_and_truncate_content_returns_none_without_a_code_block() {
        let mut content = Content::new("model", vec![Part::text("just text")]);
        let delimiters = vec![("```python\n".to_string(), "\n```".to_string())];
        assert_eq!(
            extract_code_and_truncate_content(&mut content, &delimiters),
            None
        );
    }

    #[test]
    fn extract_code_and_truncate_content_reuses_an_unconsumed_executable_code_part() {
        let mut content = Content::new("model", vec![build_executable_code_part("print(2)")]);
        let code = extract_code_and_truncate_content(&mut content, &[]);
        assert_eq!(code.as_deref(), Some("print(2)"));
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn convert_code_execution_parts_formats_trailing_executable_code() {
        let mut content = Content::new("model", vec![build_executable_code_part("print(3)")]);
        let code_delim = ("```tool_code\n".to_string(), "\n```".to_string());
        let result_delim = ("```tool_output\n".to_string(), "\n```".to_string());
        convert_code_execution_parts(&mut content, &code_delim, &result_delim);
        assert_eq!(
            content.parts[0].text.as_deref(),
            Some("```tool_code\nprint(3)\n```")
        );
    }

    #[test]
    fn convert_code_execution_parts_formats_a_sole_result_part_and_sets_user_role() {
        let mut content = Content::new(
            "model",
            vec![build_code_execution_result_part(&CodeExecutionResult {
                stdout: "42".to_string(),
                ..Default::default()
            })],
        );
        let code_delim = ("```tool_code\n".to_string(), "\n```".to_string());
        let result_delim = ("```tool_output\n".to_string(), "\n```".to_string());
        convert_code_execution_parts(&mut content, &code_delim, &result_delim);
        assert!(content.parts[0]
            .text
            .as_deref()
            .unwrap()
            .starts_with("```tool_output\n"));
        assert_eq!(content.role.as_deref(), Some("user"));
    }

    #[test]
    fn convert_code_execution_parts_skips_a_result_part_when_content_has_multiple_parts() {
        let mut content = Content::new(
            "model",
            vec![
                Part::text("intro"),
                build_code_execution_result_part(&CodeExecutionResult {
                    stdout: "42".to_string(),
                    ..Default::default()
                }),
            ],
        );
        let code_delim = ("```tool_code\n".to_string(), "\n```".to_string());
        let result_delim = ("```tool_output\n".to_string(), "\n```".to_string());
        convert_code_execution_parts(&mut content, &code_delim, &result_delim);
        // Unmodified: the multi-part guard skipped the conversion.
        assert!(content.parts[1].code_execution_result.is_some());
    }
}
