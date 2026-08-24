//! Capabilities C0831/C0832: `example_util.py`, ported from
//! `google.adk.examples.example_util`.
//!
//! **Adaptation, disclosed**: the source's function-call-arg rendering
//! (`f"{k}={v}"` for a non-string `v`) and function-response rendering
//! (`part.function_response.__dict__`) both rely on Python's dict/object
//! string reprs. This port has no equivalent — the same "compact JSON
//! stand-in for Python's `str()`/`repr()`" adaptation already disclosed by
//! `adk-flows::instructions_utils::value_to_display_string` and
//! `adk-flows::fencing` is duplicated locally here rather than reused
//! directly: `adk-examples` sits below `adk-tools` in the crate graph
//! (`adk-tools::ExampleTool`, C0419, depends on it), so depending the
//! other way on `adk-flows` would create a cycle. Also:
//! `FunctionCall.args` is a `BTreeMap` (key-sorted), not an
//! insertion-ordered map like Python's `dict` — so a multi-argument
//! function call renders its arguments in sorted-key order here, not
//! call-site order (a pre-existing adaptation of `FunctionCall.args`'s own
//! type, not new to this module). Neither of these is byte-for-byte
//! parity with the source; every other rendering path (prefixes, role
//! clumping, per-part text joining, the gemini-2 fence-style switch)
//! matches exactly, verified against the source's own
//! `test_example_util.py`.

use std::collections::BTreeMap;

use adk_agents::session::Session;
use adk_genai::content::FunctionResponse;
use rusty_serde::value::Value;

use crate::base_example_provider::BaseExampleProvider;
use crate::example::Example;

const EXAMPLES_INTRO: &str = "<EXAMPLES>\nBegin few-shot\nThe following are examples of user queries and model responses using the available tools.\n\n";
const EXAMPLES_END: &str = "End few-shot\n<EXAMPLES>";
const EXAMPLE_END: &str = "End example\n\n";
const USER_PREFIX: &str = "[user]\n";
const MODEL_PREFIX: &str = "[model]\n";
const FUNCTION_PREFIX: &str = "```\n";
const FUNCTION_CALL_PREFIX: &str = "```tool_code\n";
const FUNCTION_CALL_SUFFIX: &str = "\n```\n";
const FUNCTION_RESPONSE_PREFIX: &str = "```tool_outputs\n";
const FUNCTION_RESPONSE_SUFFIX: &str = "\n```\n";

fn example_start(example_number: usize) -> String {
    format!("EXAMPLE {example_number}:\nBegin example\n")
}

/// See the module doc's disclosed adaptation for what this doesn't
/// reproduce byte-for-byte against Python's `str()`.
fn value_to_python_like_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::UInt(u) => u.to_string(),
        Value::Float(_) | Value::Seq(_) | Value::Map(_) => {
            rusty_serde::json::to_string(value).unwrap_or_default()
        }
    }
}

fn format_function_call_args(args: Option<&BTreeMap<String, Value>>) -> String {
    let Some(args) = args else {
        return String::new();
    };
    args.iter()
        .map(|(key, value)| match value {
            Value::String(s) => format!("{key}='{s}'"),
            other => format!("{key}={}", value_to_python_like_string(other)),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Compact-JSON stand-in for Python's `part.function_response.__dict__` —
/// see the module doc.
fn format_function_response(function_response: &FunctionResponse) -> String {
    let mut fields = Vec::new();
    if let Some(id) = &function_response.id {
        fields.push(("id".to_string(), Value::String(id.clone())));
    }
    if let Some(name) = &function_response.name {
        fields.push(("name".to_string(), Value::String(name.clone())));
    }
    if let Some(response) = &function_response.response {
        fields.push((
            "response".to_string(),
            Value::Map(
                response
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
        ));
    }
    rusty_serde::json::to_string(&Value::Map(fields)).unwrap_or_default()
}

/// C0831: converts a list of examples to a string usable in a system
/// instruction.
pub fn convert_examples_to_text(examples: &[Example], model: Option<&str>) -> String {
    let mut examples_str = String::new();
    for (index, example) in examples.iter().enumerate() {
        let mut output = format!("{}{USER_PREFIX}", example_start(index + 1));

        let input_texts: Vec<&str> = example
            .input
            .parts
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect();
        if !input_texts.is_empty() {
            output.push_str(&input_texts.join("\n"));
            output.push('\n');
        }

        let gemini2 = model.map(|m| m.contains("gemini-2")).unwrap_or(true);
        let mut previous_role: Option<&str> = None;
        for content in &example.output {
            let role = if content.role.as_deref() == Some("model") {
                MODEL_PREFIX
            } else {
                USER_PREFIX
            };
            if previous_role != Some(role) {
                output.push_str(role);
            }
            previous_role = Some(role);

            for part in &content.parts {
                if let Some(function_call) = &part.function_call {
                    let args = format_function_call_args(function_call.args.as_ref());
                    let prefix = if gemini2 {
                        FUNCTION_PREFIX
                    } else {
                        FUNCTION_CALL_PREFIX
                    };
                    let name = function_call.name.as_deref().unwrap_or_default();
                    output.push_str(&format!("{prefix}{name}({args}){FUNCTION_CALL_SUFFIX}"));
                } else if let Some(function_response) = &part.function_response {
                    let prefix = if gemini2 {
                        FUNCTION_PREFIX
                    } else {
                        FUNCTION_RESPONSE_PREFIX
                    };
                    output.push_str(&format!(
                        "{prefix}{}{FUNCTION_RESPONSE_SUFFIX}",
                        format_function_response(function_response)
                    ));
                } else if let Some(text) = &part.text {
                    output.push_str(text);
                    output.push('\n');
                }
            }
        }

        output.push_str(EXAMPLE_END);
        examples_str.push_str(&output);
    }

    format!("{EXAMPLES_INTRO}{examples_str}{EXAMPLES_END}")
}

/// C0832: gets the latest message from the user — the most recent
/// user-authored, non-function-response event's first text part. Returns
/// an empty string if not found.
///
/// Adaptation: the source logs a warning when the latest user event
/// carries no usable text; no logging framework has been adopted by this
/// workspace yet (the same disclosed omission as `contents.rs`'s
/// `drop_orphaned_function_responses`).
pub fn get_latest_message_from_user(session: &Session) -> String {
    let Some(event) = session.events.last() else {
        return String::new();
    };
    if event.author == "user" && event.get_function_responses().is_empty() {
        if let Some(text) = event
            .content
            .as_ref()
            .and_then(|content| content.parts.first())
            .and_then(|part| part.text.as_deref())
        {
            return text.to_string();
        }
    }
    String::new()
}

/// `Union[list[Example], BaseExampleProvider]` for [`build_example_si`].
pub enum ExamplesSource<'a> {
    List(&'a [Example]),
    Provider(&'a dyn BaseExampleProvider),
}

/// C0831: builds the examples system instruction from either a fixed list
/// or a [`BaseExampleProvider`] queried live.
pub fn build_example_si(examples: ExamplesSource, query: &str, model: Option<&str>) -> String {
    match examples {
        ExamplesSource::List(list) => convert_examples_to_text(list, model),
        ExamplesSource::Provider(provider) => {
            convert_examples_to_text(&provider.get_examples(query), model)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::{Content, FunctionCall, Part};

    fn basic_input() -> Content {
        Content::new("user", vec![Part::text("test_input")])
    }

    fn basic_output() -> Vec<Content> {
        vec![Content::new("model", vec![Part::text("test_output")])]
    }

    fn basic_example() -> Example {
        Example::new(basic_input(), basic_output())
    }

    const MODELS: [Option<&str>; 3] = [Some("gemini-2.5-flash"), Some("llama3_vertex_agent"), None];

    #[test]
    fn converts_examples_handles_content_without_parts() {
        let sample = Example::new(
            Content::new("user", vec![]),
            vec![Content::new("model", vec![])],
        );
        let expected = format!(
            "{EXAMPLES_INTRO}{}{USER_PREFIX}{MODEL_PREFIX}{EXAMPLE_END}{EXAMPLES_END}",
            example_start(1)
        );
        assert_eq!(convert_examples_to_text(&[sample], None), expected);
    }

    #[test]
    fn text_only_example_conversion() {
        for model in MODELS {
            let expected = format!(
                "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output\n{EXAMPLE_END}{EXAMPLES_END}",
                example_start(1)
            );
            assert_eq!(
                convert_examples_to_text(&[basic_example()], model),
                expected,
                "model={model:?}"
            );
        }
    }

    #[test]
    fn multi_part_text_example_conversion() {
        let output = vec![Content::new(
            "model",
            vec![
                Part::text("test_output_1"),
                Part::text("test_output_2"),
                Part::text("test_output_3"),
            ],
        )];
        let example = Example::new(basic_input(), output);
        for model in MODELS {
            let expected = format!(
                "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output_1\ntest_output_2\ntest_output_3\n{EXAMPLE_END}{EXAMPLES_END}",
                example_start(1)
            );
            assert_eq!(
                convert_examples_to_text(std::slice::from_ref(&example), model),
                expected,
                "model={model:?}"
            );
        }
    }

    #[test]
    fn example_conversion_prefix_insertion() {
        let output = vec![
            Content::new("model", vec![Part::text("test_output_1")]),
            Content::new("user", vec![Part::text("test_output_2")]),
            Content::new("model", vec![Part::text("test_output_3")]),
        ];
        let example = Example::new(basic_input(), output);
        for model in MODELS {
            let expected = format!(
                "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output_1\n{USER_PREFIX}test_output_2\n{MODEL_PREFIX}test_output_3\n{EXAMPLE_END}{EXAMPLES_END}",
                example_start(1)
            );
            assert_eq!(
                convert_examples_to_text(std::slice::from_ref(&example), model),
                expected,
                "model={model:?}"
            );
        }
    }

    #[test]
    fn example_conversion_output_clumping() {
        let output = vec![
            Content::new("model", vec![Part::text("test_output_1")]),
            Content::new("model", vec![Part::text("test_output_2")]),
            Content::new("user", vec![Part::text("test_output_3")]),
            Content::new("user", vec![Part::text("test_output_4")]),
        ];
        let example = Example::new(basic_input(), output);
        for model in MODELS {
            let expected = format!(
                "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output_1\ntest_output_2\n{USER_PREFIX}test_output_3\ntest_output_4\n{EXAMPLE_END}{EXAMPLES_END}",
                example_start(1)
            );
            assert_eq!(
                convert_examples_to_text(std::slice::from_ref(&example), model),
                expected,
                "model={model:?}"
            );
        }
    }

    #[test]
    fn empty_examples_list_conversion() {
        for model in MODELS {
            let expected = format!("{EXAMPLES_INTRO}{EXAMPLES_END}");
            assert_eq!(
                convert_examples_to_text(&[], model),
                expected,
                "model={model:?}"
            );
        }
    }

    fn function_call_args() -> BTreeMap<String, Value> {
        let mut args = BTreeMap::new();
        args.insert(
            "test_string_argument".to_string(),
            Value::String("test_value".to_string()),
        );
        args.insert("test_int_argument".to_string(), Value::Int(1));
        args
    }

    #[test]
    fn example_conversion_with_function_call_renders_sorted_args() {
        // Adaptation: BTreeMap key order (test_int_argument before
        // test_string_argument), not the source's dict insertion order —
        // see the module doc.
        let output = vec![Content::new(
            "model",
            vec![Part {
                function_call: Some(FunctionCall {
                    id: None,
                    name: Some("test_function".to_string()),
                    args: Some(function_call_args()),
                    will_continue: None,
                }),
                ..Default::default()
            }],
        )];
        let example = Example::new(basic_input(), output);
        for model in MODELS {
            let gemini2 = model.map(|m| m.contains("gemini-2")).unwrap_or(true);
            let prefix = if gemini2 {
                FUNCTION_PREFIX
            } else {
                FUNCTION_CALL_PREFIX
            };
            let expected = format!(
                "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}{prefix}test_function(test_int_argument=1, test_string_argument='test_value'){FUNCTION_CALL_SUFFIX}{EXAMPLE_END}{EXAMPLES_END}",
                example_start(1)
            );
            assert_eq!(
                convert_examples_to_text(std::slice::from_ref(&example), model),
                expected,
                "model={model:?}"
            );
        }
    }

    #[test]
    fn example_conversion_with_function_response_uses_the_gemini2_fence_switch() {
        let mut response = BTreeMap::new();
        response.insert(
            "test_string_argument".to_string(),
            Value::String("test_value".to_string()),
        );
        let output = vec![Content::new(
            "model",
            vec![Part {
                function_response: Some(FunctionResponse {
                    id: None,
                    name: Some("test_function".to_string()),
                    response: Some(response),
                }),
                ..Default::default()
            }],
        )];
        let example = Example::new(basic_input(), output);
        for model in MODELS {
            let gemini2 = model.map(|m| m.contains("gemini-2")).unwrap_or(true);
            let prefix = if gemini2 {
                FUNCTION_PREFIX
            } else {
                FUNCTION_RESPONSE_PREFIX
            };
            let text = convert_examples_to_text(std::slice::from_ref(&example), model);
            assert!(text.contains(prefix), "model={model:?} text={text}");
            assert!(text.contains(FUNCTION_RESPONSE_SUFFIX), "model={model:?}");
            assert!(text.contains("test_function"), "model={model:?}");
        }
    }

    #[test]
    fn building_si_from_list() {
        let expected = format!(
            "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output\n{EXAMPLE_END}{EXAMPLES_END}",
            example_start(1)
        );
        let examples = [basic_example()];
        assert_eq!(
            build_example_si(ExamplesSource::List(&examples), "", None),
            expected
        );
    }

    struct MockExampleProvider {
        examples: Vec<Example>,
        query: String,
    }

    impl BaseExampleProvider for MockExampleProvider {
        fn get_examples(&self, query: &str) -> Vec<Example> {
            if query == self.query {
                self.examples.clone()
            } else {
                Vec::new()
            }
        }
    }

    #[test]
    fn building_si_from_base_example_provider() {
        let expected = format!(
            "{EXAMPLES_INTRO}{}{USER_PREFIX}test_input\n{MODEL_PREFIX}test_output\n{EXAMPLE_END}{EXAMPLES_END}",
            example_start(1)
        );
        let provider = MockExampleProvider {
            examples: vec![basic_example()],
            query: "test_query".to_string(),
        };
        assert_eq!(
            build_example_si(ExamplesSource::Provider(&provider), "test_query", None),
            expected
        );
    }

    #[test]
    fn get_latest_message_from_user_returns_empty_for_no_events() {
        let session = Session::new("app", "user", "s1");
        assert_eq!(get_latest_message_from_user(&session), "");
    }

    #[test]
    fn get_latest_message_from_user_returns_the_text_of_the_latest_user_event() {
        use adk_events::node_info::NodeInfo;
        use adk_events::Event;

        let mut session = Session::new("app", "user", "s1");
        let mut event = Event::new("inv-1", "user", NodeInfo::new("root"));
        event.content = Some(Content::new("user", vec![Part::text("hello there")]));
        session.events.push(event);

        assert_eq!(get_latest_message_from_user(&session), "hello there");
    }

    #[test]
    fn get_latest_message_from_user_ignores_a_non_user_authored_latest_event() {
        use adk_events::node_info::NodeInfo;
        use adk_events::Event;

        let mut session = Session::new("app", "user", "s1");
        let mut event = Event::new("inv-1", "some_agent", NodeInfo::new("root"));
        event.content = Some(Content::new("model", vec![Part::text("hi")]));
        session.events.push(event);

        assert_eq!(get_latest_message_from_user(&session), "");
    }

    #[test]
    fn get_latest_message_from_user_ignores_a_function_response_event() {
        use adk_events::node_info::NodeInfo;
        use adk_events::Event;

        let mut session = Session::new("app", "user", "s1");
        let mut event = Event::new("inv-1", "user", NodeInfo::new("root"));
        event.content = Some(Content::new(
            "user",
            vec![
                Part::text("would otherwise be returned"),
                Part {
                    function_response: Some(FunctionResponse {
                        id: None,
                        name: Some("f".to_string()),
                        response: None,
                    }),
                    ..Default::default()
                },
            ],
        ));
        session.events.push(event);

        assert_eq!(get_latest_message_from_user(&session), "");
    }
}
