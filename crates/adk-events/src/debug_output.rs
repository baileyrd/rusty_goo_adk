//! Capability C0933: `print_event`, ported from
//! `google.adk.utils._debug_output`.
//!
//! **Adaptation**: the source formats a tool call's `args` (a `dict`) and a
//! tool response's `response` (an arbitrary value) with Python's `str()`.
//! This port uses `rusty_serde::json::to_string` instead — the same
//! disclosed compact-JSON-instead-of-`str()`/`repr()` divergence already
//! used by `adk-genai::content_utils::to_user_content` (C0928) — falling
//! back to `{value:?}` if a value somehow doesn't serialize.
//!
//! **Adaptation**: `executable_code`'s `language` and `file_data`'s
//! `file_uri` (read by the source directly, `part.executable_code.language`
//! / `part.file_data.file_uri`) stay behind this port's already-disclosed
//! opaque `Value` placeholders (`adk-genai::content`'s module doc) rather
//! than real typed fields — this port reads them back out of the opaque
//! `Value`/flattened `rest` map by key instead.

use rusty_serde::value::Value;
use rusty_serde::Serialize;

use adk_genai::content::Part;

use crate::event::Event;

const ARGS_MAX_LEN: usize = 50;
const RESPONSE_MAX_LEN: usize = 100;
const CODE_OUTPUT_MAX_LEN: usize = 100;

/// `_debug_output._truncate`. Truncates on byte length, matching the
/// source's `text[:max_len]` slicing (a disclosed divergence for
/// non-ASCII text, where Python's `str` slicing is by code point — this
/// port instead truncates to a `char`-boundary-safe prefix no longer than
/// `max_len` bytes, rather than panicking on a mid-character split).
fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

fn display<T: Serialize>(value: &T) -> String {
    rusty_serde::json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn map_field(value: &Value, key: &str) -> Option<String> {
    match value {
        Value::Map(entries) => entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| display(v)),
        _ => None,
    }
}

fn print_part(author: &str, part: &Part) {
    if let Some(call) = &part.function_call {
        let args = call.args.as_ref().map(display).unwrap_or_default();
        println!(
            "{author} > [Calling tool: {}({})]",
            call.name.as_deref().unwrap_or(""),
            truncate(&args, ARGS_MAX_LEN)
        );
    } else if let Some(response) = &part.function_response {
        let rendered = response.response.as_ref().map(display).unwrap_or_default();
        println!(
            "{author} > [Tool result: {}]",
            truncate(&rendered, RESPONSE_MAX_LEN)
        );
    } else if let Some(code) = &part.executable_code {
        let lang = map_field(code, "language").unwrap_or_else(|| "code".to_string());
        println!("{author} > [Executing {lang} code...]");
    } else if let Some(result) = &part.code_execution_result {
        let output = map_field(result, "output").unwrap_or_else(|| display(result));
        println!(
            "{author} > [Code output: {}]",
            truncate(&output, CODE_OUTPUT_MAX_LEN)
        );
    } else if let Some(inline) = &part.inline_data {
        let mime_type = inline.mime_type.as_deref().unwrap_or("data");
        println!("{author} > [Inline data: {mime_type}]");
    } else if let Some(file) = &part.file_data {
        let uri = file
            .rest
            .as_ref()
            .and_then(|rest| map_field(rest, "fileUri"))
            .unwrap_or_else(|| "file".to_string());
        println!("{author} > [File: {uri}]");
    }
}

/// `_debug_output.print_event` — prints `event` to stdout in a
/// user-friendly format. Text parts are always shown; non-text parts
/// (tool calls, code execution, inline/file data) are shown only when
/// `verbose` is set.
pub fn print_event(event: &Event, verbose: bool) {
    let Some(content) = &event.content else {
        return;
    };
    if content.parts.is_empty() {
        return;
    }

    let mut text_buffer = String::new();
    let flush_text = |buffer: &mut String| {
        if !buffer.is_empty() {
            println!("{} > {buffer}", event.author);
            buffer.clear();
        }
    };

    for part in &content.parts {
        if let Some(text) = &part.text {
            text_buffer.push_str(text);
        } else {
            flush_text(&mut text_buffer);
            if verbose {
                print_part(&event.author, part);
            }
        }
    }
    flush_text(&mut text_buffer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_info::NodeInfo;
    use adk_genai::content::{Content, FunctionCall, FunctionResponse, MediaBlobStub};

    fn sample_event(content: Content) -> Event {
        Event::new("inv-1", "agent", NodeInfo::new("root")).with_message(content)
    }

    #[test]
    fn print_event_is_a_no_op_for_no_content() {
        // Just verifying this doesn't panic — stdout isn't captured here.
        let event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        print_event(&event, true);
    }

    #[test]
    fn truncate_appends_ellipsis_only_when_over_the_limit() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long string", 7), "this is...");
    }

    #[test]
    fn truncate_does_not_split_a_multi_byte_character() {
        let text = "a".repeat(9) + "é"; // 'é' is 2 bytes in UTF-8.
        let truncated = truncate(&text, 10);
        assert!(truncated.starts_with(&"a".repeat(9)));
    }

    #[test]
    fn print_event_prints_text_and_tool_calls_when_verbose() {
        let content = Content::new(
            "model",
            vec![
                adk_genai::content::Part::text("hello"),
                adk_genai::content::Part::function_call(FunctionCall {
                    name: Some("get_weather".to_string()),
                    ..Default::default()
                }),
                adk_genai::content::Part::function_response(FunctionResponse {
                    name: Some("get_weather".to_string()),
                    ..Default::default()
                }),
            ],
        );
        print_event(&sample_event(content), true);
    }

    #[test]
    fn print_event_hides_non_text_parts_when_not_verbose() {
        let content = Content::new(
            "model",
            vec![adk_genai::content::Part::function_call(FunctionCall {
                name: Some("get_weather".to_string()),
                ..Default::default()
            })],
        );
        print_event(&sample_event(content), false);
    }

    #[test]
    fn print_event_handles_inline_and_file_data_parts() {
        let content = Content::new(
            "model",
            vec![
                adk_genai::content::Part {
                    inline_data: Some(MediaBlobStub {
                        mime_type: Some("image/png".to_string()),
                        rest: None,
                    }),
                    ..Default::default()
                },
                adk_genai::content::Part {
                    file_data: Some(MediaBlobStub {
                        mime_type: None,
                        rest: Some(Value::Map(vec![(
                            "fileUri".to_string(),
                            Value::String("gs://bucket/file".to_string()),
                        )])),
                    }),
                    ..Default::default()
                },
            ],
        );
        print_event(&sample_event(content), true);
    }
}
