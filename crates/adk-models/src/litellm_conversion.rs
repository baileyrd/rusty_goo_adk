//! P10 LiteLLM batch (C0561/C0562/C0569/C0571): pure, HTTP-free helper
//! functions from `models/lite_llm.py` — the universal LiteLLM-backed
//! provider wrapper (C0557). No new dependency: every function here
//! operates on plain strings or [`rusty_serde::value::Value`], the same
//! "declare pure conversion helpers ahead of the real HTTP-calling
//! backend" precedent `anthropic_conversion.rs` already established for
//! the sibling Anthropic backend (C0540/C0542/C0538/C0539). No
//! `litellm` crate, no HTTP client, no `LiteLlm(BaseLlm)` struct itself
//! (C0557/C0558, still `REQUIRED`) needed for any of these.
//!
//! **Scope of this batch, disclosed**: [`strip_proxy_prefix`]/
//! [`get_provider_from_model`] (C0561), [`is_anthropic_provider`]/
//! [`is_anthropic_route`] (C0562, plus their shared
//! [`is_anthropic_model`] dependency — used again elsewhere in the
//! source at a call site this port hasn't reached yet, but with no
//! manifest row of its own; C0562's own description already covers its
//! exact behavior: "Anthropic formatting only applied when the model
//! name itself identifies Claude"), [`quote_unquoted_json_object_keys`]/
//! [`parse_tool_call_arguments`] (C0569), and
//! [`enforce_strict_openai_schema`] (C0571) are ported here. The rest of
//! P10's LiteLLM rows (C0557-C0560, C0563-C0570, C0572-C0574) stay
//! `REQUIRED` — genuinely separable, larger work (the actual
//! `LiteLlm(BaseLlm)` struct + HTTP wiring, streaming assembly, media
//! conversion, and the several other provider-specific behaviors this
//! 774-line source file covers) left for follow-up batches.
//!
//! **`parse_tool_call_arguments`'s `ast.literal_eval` step, narrowed —
//! disclosed**: the source tries strict `json.loads`, then Python's
//! `ast.literal_eval` (parses a Python dict/list *literal*, distinct
//! from JSON — e.g. single-quoted strings, `True`/`False`/`None`), then
//! falls through to the character-level unquoted-key repair. Rust has
//! no equivalent to `ast.literal_eval` (no Python-literal parser in this
//! workspace, and adding one for this single call site would be a new
//! dependency for a narrow repair path), so this port's version skips
//! straight from strict JSON to the character-level repair — the
//! `ast.literal_eval` step exists in the source only to catch a Python
//! `dict`-literal shape (single-quoted keys/values, no LLM actually
//! emits that on the wire) that the following character-level repair
//! doesn't already handle; the common "unquoted identifier-shaped keys"
//! case this whole function exists for is still repaired identically.
//! The source's final `raise json_error` (re-raising the *original*
//! `json.JSONDecodeError`) becomes a fresh `Err(String)` here — this
//! port has no equivalent exception type to re-raise, and the original
//! parse error's message is preserved in the returned string instead.
//!
//! **`_parse_tool_call_arguments`'s `Any`/non-`str` branches, dropped**:
//! the source's `arguments: Any` parameter tolerates a caller passing an
//! already-parsed value (`if not isinstance(arguments, str): return
//! arguments`) — this port's [`parse_tool_call_arguments`] is already
//! `&str`-typed at its call boundary (LiteLLM tool-call arguments are
//! always a JSON string on the wire), so there is no "was never a
//! string" case left to handle, the same boundary-typed narrowing
//! `anthropic_conversion.rs::to_claude_role` already uses for its own
//! `Option<&str>` parameter.
//!
//! **`_UNQUOTED_KEY_RE`, hand-rolled not `regex`-backed**: the source's
//! compiled pattern is the fixed identifier shape `[A-Za-z_][A-Za-z0-9_]*`
//! — simple enough to match with a plain character scan
//! ([`match_unquoted_key`]) rather than pulling in a `Regex::new` call
//! for a single fixed pattern, matching the "hand-roll a simple fixed
//! pattern" precedent already used elsewhere in this port (e.g.
//! `is_valid_tool_id` in `anthropic_conversion.rs`).

use rusty_serde::value::Value;

const PROXY_PROVIDER: &str = "litellm_proxy";
const ANTHROPIC_PROVIDERS: &[&str] = &["anthropic", "bedrock", "vertex_ai"];

/// `lite_llm._strip_proxy_prefix` — removes a leading `litellm_proxy/`
/// routing prefix from a model string. `litellm_proxy` selects the
/// transport, not the model family; the segment after it identifies the
/// provider that actually serves the request. A bare
/// `litellm_proxy/<deployment>` has no nested provider, so the prefix is
/// left untouched.
pub fn strip_proxy_prefix(model: &str) -> String {
    if model.is_empty() {
        return model.to_string();
    }
    let prefix = format!("{PROXY_PROVIDER}/");
    if model.to_lowercase().starts_with(&prefix) {
        let remaining = &model[prefix.len()..];
        if remaining.contains('/') {
            return remaining.to_string();
        }
    }
    model.to_string()
}

/// `lite_llm._get_provider_from_model` — extracts the provider name from
/// a LiteLLM model string (`"provider/model"`), falling back to a couple
/// of naming-convention heuristics (`azure`, `gpt-`/`o1` → `openai`) when
/// there's no `/`. Returns an empty string when the provider can't be
/// determined, matching the source's own fallback.
pub fn get_provider_from_model(model: &str) -> String {
    if model.is_empty() {
        return String::new();
    }
    let stripped = strip_proxy_prefix(model);
    if let Some((provider, _)) = stripped.split_once('/') {
        return provider.to_lowercase();
    }
    let model_lower = stripped.to_lowercase();
    if model_lower.contains("azure") {
        return "azure".to_string();
    }
    if model_lower.starts_with("gpt-") || model_lower.starts_with("o1") {
        return "openai".to_string();
    }
    String::new()
}

/// `lite_llm._is_anthropic_provider` — true for any provider that *can*
/// route to an Anthropic model endpoint (`bedrock`/`vertex_ai` are
/// multi-model platforms, so this alone doesn't mean the request
/// actually reaches Claude — see [`is_anthropic_route`]).
pub fn is_anthropic_provider(provider: &str) -> bool {
    if provider.is_empty() {
        return false;
    }
    ANTHROPIC_PROVIDERS.contains(&provider.to_lowercase().as_str())
}

/// `lite_llm._is_anthropic_model` — true for an `anthropic/` model, or a
/// `bedrock/`/`vertex_ai/` model whose own name identifies Claude.
pub fn is_anthropic_model(model: &str) -> bool {
    let lower = strip_proxy_prefix(&model.to_lowercase());
    if lower.starts_with("anthropic/") {
        return true;
    }
    if let Some(model_part) = lower.strip_prefix("bedrock/") {
        return model_part.contains("anthropic") || model_part.contains("claude");
    }
    if let Some(model_part) = lower.strip_prefix("vertex_ai/") {
        return model_part.contains("claude");
    }
    false
}

/// `lite_llm._is_anthropic_route` — true only when a request actually
/// reaches an Anthropic Claude model: `bedrock`/`vertex_ai` also host
/// non-Anthropic models, so for those platforms the model name must
/// identify Claude too (formatting thinking blocks for a non-Claude
/// model on those platforms triggers API validation errors).
pub fn is_anthropic_route(provider: &str, model: &str) -> bool {
    if !is_anthropic_provider(provider) {
        return false;
    }
    let provider_lower = provider.to_lowercase();
    if provider_lower == "bedrock" || provider_lower == "vertex_ai" {
        return is_anthropic_model(model);
    }
    true
}

/// `lite_llm._UNQUOTED_KEY_RE`'s pattern (`[A-Za-z_][A-Za-z0-9_]*`),
/// hand-matched from `start` — see this module's own doc for why this
/// isn't a `regex::Regex`. Returns the index just past the match, or
/// `None` if `start` isn't the start of a valid identifier.
fn match_unquoted_key(chars: &[char], start: usize) -> Option<usize> {
    let mut i = start;
    let first = *chars.get(i)?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    i += 1;
    while let Some(&c) = chars.get(i) {
        if c.is_ascii_alphanumeric() || c == '_' {
            i += 1;
        } else {
            break;
        }
    }
    Some(i)
}

/// `lite_llm._quote_unquoted_json_object_keys` — quotes simple unquoted
/// object keys (immediately following `{`/`,`, immediately preceding
/// `:`) without touching string contents. Ported as a direct,
/// character-by-character transcription of the source's own scanner.
pub fn quote_unquoted_json_object_keys(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut result = String::new();
    let mut i = 0usize;
    let mut in_string = false;
    let mut string_quote = '\0';
    let mut escaped = false;

    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            result.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == string_quote {
                in_string = false;
                string_quote = '\0';
            }
            i += 1;
            continue;
        }

        if ch == '"' || ch == '\'' {
            in_string = true;
            string_quote = ch;
            result.push(ch);
            i += 1;
            continue;
        }

        if ch == '{' || ch == ',' {
            result.push(ch);
            i += 1;
            let whitespace_start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            result.extend(chars[whitespace_start..i].iter().copied());

            if let Some(key_end) = match_unquoted_key(&chars, i) {
                let mut colon_index = key_end;
                while colon_index < chars.len() && chars[colon_index].is_whitespace() {
                    colon_index += 1;
                }
                if colon_index < chars.len() && chars[colon_index] == ':' {
                    result.push('"');
                    result.extend(chars[i..key_end].iter().copied());
                    result.push('"');
                    result.extend(chars[key_end..colon_index].iter().copied());
                    i = colon_index;
                    continue;
                }
            }
            continue;
        }

        result.push(ch);
        i += 1;
    }

    result
}

/// `lite_llm._parse_tool_call_arguments` — see this module's own doc for
/// the `ast.literal_eval`/`Any`-parameter narrowings.
pub fn parse_tool_call_arguments(arguments: &str) -> Result<Value, String> {
    if arguments.is_empty() {
        return Ok(Value::Map(Vec::new()));
    }

    if let Ok(parsed) = rusty_serde::json::from_str::<Value>(arguments) {
        return Ok(parsed);
    }

    let repaired = quote_unquoted_json_object_keys(arguments);
    if repaired != arguments {
        if let Ok(parsed) = rusty_serde::json::from_str::<Value>(&repaired) {
            return Ok(parsed);
        }
    }

    Err(format!(
        "Failed to parse LiteLLM tool call arguments as JSON: {arguments:?}"
    ))
}

fn map_entries_mut(value: &mut Value) -> Option<&mut Vec<(String, Value)>> {
    match value {
        Value::Map(entries) => Some(entries),
        _ => None,
    }
}

fn set_entry(entries: &mut Vec<(String, Value)>, key: &str, value: Value) {
    match entries.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => entries.push((key.to_string(), value)),
    }
}

/// `lite_llm._enforce_strict_openai_schema` — recursively transforms a
/// JSON schema for OpenAI strict structured outputs: `$ref` nodes lose
/// every sibling keyword, every object schema with a `properties` key
/// gets `additionalProperties: false` plus every property listed as
/// `required` (sorted, matching the source's own `sorted(...)`), and
/// `$defs`/`properties`/`anyOf`/`oneOf`/`allOf`/`items` all recurse.
/// Mutates `schema` in place, exactly like the source; a no-op on
/// anything that isn't a `Value::Map` (the source's own `if not
/// isinstance(schema, dict): return`).
pub fn enforce_strict_openai_schema(schema: &mut Value) {
    let Some(entries) = map_entries_mut(schema) else {
        return;
    };

    if entries.iter().any(|(key, _)| key == "$ref") {
        entries.retain(|(key, _)| key == "$ref");
        return;
    }

    let is_object = entries
        .iter()
        .any(|(key, value)| key == "type" && value.as_str() == Some("object"));
    let has_properties = entries.iter().any(|(key, _)| key == "properties");
    if is_object && has_properties {
        let mut required: Vec<String> = entries
            .iter()
            .find(|(key, _)| key == "properties")
            .and_then(|(_, value)| match value {
                Value::Map(props) => Some(props.iter().map(|(k, _)| k.clone()).collect()),
                _ => None,
            })
            .unwrap_or_default();
        required.sort();
        set_entry(entries, "additionalProperties", Value::Bool(false));
        set_entry(
            entries,
            "required",
            Value::Seq(required.into_iter().map(Value::String).collect()),
        );
    }

    if let Some((_, Value::Map(defs_entries))) = entries.iter_mut().find(|(key, _)| key == "$defs")
    {
        for (_, def) in defs_entries.iter_mut() {
            enforce_strict_openai_schema(def);
        }
    }

    if let Some((_, Value::Map(prop_entries))) =
        entries.iter_mut().find(|(key, _)| key == "properties")
    {
        for (_, prop) in prop_entries.iter_mut() {
            enforce_strict_openai_schema(prop);
        }
    }

    for combinator in ["anyOf", "oneOf", "allOf"] {
        if let Some((_, Value::Seq(seq))) = entries.iter_mut().find(|(key, _)| key == combinator) {
            for item in seq.iter_mut() {
                enforce_strict_openai_schema(item);
            }
        }
    }

    if let Some((_, items)) = entries.iter_mut().find(|(key, _)| key == "items") {
        if matches!(items, Value::Map(_)) {
            enforce_strict_openai_schema(items);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- C0561: strip_proxy_prefix / get_provider_from_model ---

    #[test]
    fn strip_proxy_prefix_unwraps_a_nested_provider() {
        assert_eq!(
            strip_proxy_prefix("litellm_proxy/azure/gpt-4"),
            "azure/gpt-4"
        );
    }

    #[test]
    fn strip_proxy_prefix_is_case_insensitive_on_the_prefix() {
        assert_eq!(
            strip_proxy_prefix("LiteLLM_Proxy/azure/gpt-4"),
            "azure/gpt-4"
        );
    }

    #[test]
    fn strip_proxy_prefix_leaves_a_bare_proxy_deployment_untouched() {
        assert_eq!(
            strip_proxy_prefix("litellm_proxy/my-deployment"),
            "litellm_proxy/my-deployment"
        );
    }

    #[test]
    fn strip_proxy_prefix_leaves_a_non_proxy_model_untouched() {
        assert_eq!(strip_proxy_prefix("openai/gpt-4o"), "openai/gpt-4o");
    }

    #[test]
    fn strip_proxy_prefix_handles_an_empty_model() {
        assert_eq!(strip_proxy_prefix(""), "");
    }

    #[test]
    fn get_provider_from_model_reads_the_provider_segment() {
        assert_eq!(get_provider_from_model("openai/gpt-4o"), "openai");
        assert_eq!(get_provider_from_model("Azure/gpt-4"), "azure");
    }

    #[test]
    fn get_provider_from_model_unwraps_the_proxy_prefix_first() {
        assert_eq!(
            get_provider_from_model("litellm_proxy/anthropic/claude-4"),
            "anthropic"
        );
    }

    #[test]
    fn get_provider_from_model_falls_back_to_azure_heuristic() {
        assert_eq!(get_provider_from_model("my-azure-deployment"), "azure");
    }

    #[test]
    fn get_provider_from_model_falls_back_to_openai_heuristic() {
        assert_eq!(get_provider_from_model("gpt-4o"), "openai");
        assert_eq!(get_provider_from_model("o1-preview"), "openai");
    }

    #[test]
    fn get_provider_from_model_returns_empty_when_undeterminable() {
        assert_eq!(get_provider_from_model("some-custom-model"), "");
        assert_eq!(get_provider_from_model(""), "");
    }

    // --- C0562: is_anthropic_provider / is_anthropic_model / is_anthropic_route ---

    #[test]
    fn is_anthropic_provider_recognizes_all_three_platforms() {
        assert!(is_anthropic_provider("anthropic"));
        assert!(is_anthropic_provider("Bedrock"));
        assert!(is_anthropic_provider("vertex_ai"));
        assert!(!is_anthropic_provider("openai"));
        assert!(!is_anthropic_provider(""));
    }

    #[test]
    fn is_anthropic_model_recognizes_the_anthropic_prefix() {
        assert!(is_anthropic_model("anthropic/claude-4-sonnet"));
    }

    #[test]
    fn is_anthropic_model_checks_bedrock_model_names() {
        assert!(is_anthropic_model("bedrock/anthropic.claude-3-5-sonnet"));
        assert!(is_anthropic_model("bedrock/claude-instant"));
        assert!(!is_anthropic_model("bedrock/llama-3"));
    }

    #[test]
    fn is_anthropic_model_checks_vertex_ai_model_names() {
        assert!(is_anthropic_model("vertex_ai/claude-4-sonnet"));
        assert!(!is_anthropic_model("vertex_ai/gemini-2.5-flash"));
    }

    #[test]
    fn is_anthropic_model_unwraps_the_proxy_prefix() {
        assert!(is_anthropic_model(
            "litellm_proxy/anthropic/claude-4-sonnet"
        ));
    }

    #[test]
    fn is_anthropic_model_rejects_unrelated_providers() {
        assert!(!is_anthropic_model("openai/gpt-4o"));
    }

    #[test]
    fn is_anthropic_route_is_unconditional_for_the_anthropic_provider() {
        assert!(is_anthropic_route("anthropic", "claude-4-sonnet"));
        // Even a model name that wouldn't itself pass is_anthropic_model.
        assert!(is_anthropic_route("anthropic", "anything"));
    }

    #[test]
    fn is_anthropic_route_checks_the_model_name_for_bedrock_and_vertex() {
        assert!(is_anthropic_route(
            "bedrock",
            "bedrock/anthropic.claude-3-5-sonnet"
        ));
        assert!(!is_anthropic_route("bedrock", "bedrock/llama-3"));
        assert!(is_anthropic_route("vertex_ai", "vertex_ai/claude-4-sonnet"));
        assert!(!is_anthropic_route(
            "vertex_ai",
            "vertex_ai/gemini-2.5-flash"
        ));
    }

    #[test]
    fn is_anthropic_route_rejects_a_non_anthropic_provider() {
        assert!(!is_anthropic_route("openai", "gpt-4o"));
    }

    // --- C0569: quote_unquoted_json_object_keys / parse_tool_call_arguments ---

    #[test]
    fn quote_unquoted_json_object_keys_quotes_a_simple_key() {
        assert_eq!(quote_unquoted_json_object_keys("{foo: 1}"), r#"{"foo": 1}"#);
    }

    #[test]
    fn quote_unquoted_json_object_keys_quotes_every_key_after_a_comma() {
        assert_eq!(
            quote_unquoted_json_object_keys("{foo: 1, bar: 2}"),
            r#"{"foo": 1, "bar": 2}"#
        );
    }

    #[test]
    fn quote_unquoted_json_object_keys_leaves_already_quoted_keys_alone() {
        assert_eq!(
            quote_unquoted_json_object_keys(r#"{"foo": 1}"#),
            r#"{"foo": 1}"#
        );
    }

    #[test]
    fn quote_unquoted_json_object_keys_does_not_touch_string_contents() {
        assert_eq!(
            quote_unquoted_json_object_keys(r#"{foo: "a, bar: b"}"#),
            r#"{"foo": "a, bar: b"}"#
        );
    }

    #[test]
    fn quote_unquoted_json_object_keys_handles_escaped_quotes_in_strings() {
        assert_eq!(
            quote_unquoted_json_object_keys(r#"{foo: "a\"b"}"#),
            r#"{"foo": "a\"b"}"#
        );
    }

    #[test]
    fn quote_unquoted_json_object_keys_ignores_a_non_identifier_after_brace() {
        assert_eq!(quote_unquoted_json_object_keys("{1: 2}"), "{1: 2}");
    }

    #[test]
    fn parse_tool_call_arguments_returns_an_empty_map_for_empty_input() {
        assert_eq!(
            parse_tool_call_arguments("").unwrap(),
            Value::Map(Vec::new())
        );
    }

    #[test]
    fn parse_tool_call_arguments_parses_strict_json() {
        let result = parse_tool_call_arguments(r#"{"a": 1}"#).unwrap();
        assert_eq!(result.get("a").and_then(Value::as_i64), Some(1));
    }

    #[test]
    fn parse_tool_call_arguments_repairs_unquoted_keys() {
        let result = parse_tool_call_arguments("{a: 1, b: 2}").unwrap();
        assert_eq!(result.get("a").and_then(Value::as_i64), Some(1));
        assert_eq!(result.get("b").and_then(Value::as_i64), Some(2));
    }

    #[test]
    fn parse_tool_call_arguments_errors_when_nothing_repairs_it() {
        assert!(parse_tool_call_arguments("not json at all").is_err());
    }

    // --- C0571: enforce_strict_openai_schema ---

    #[test]
    fn enforce_strict_openai_schema_ignores_a_non_map_value() {
        let mut value = Value::String("not a schema".to_string());
        enforce_strict_openai_schema(&mut value);
        assert_eq!(value, Value::String("not a schema".to_string()));
    }

    #[test]
    fn enforce_strict_openai_schema_strips_ref_siblings() {
        let mut schema = Value::Map(vec![
            ("$ref".to_string(), Value::String("#/$defs/Foo".to_string())),
            (
                "description".to_string(),
                Value::String("ignored".to_string()),
            ),
        ]);
        enforce_strict_openai_schema(&mut schema);
        assert_eq!(
            schema,
            Value::Map(vec![(
                "$ref".to_string(),
                Value::String("#/$defs/Foo".to_string())
            )])
        );
    }

    #[test]
    fn enforce_strict_openai_schema_sets_additional_properties_and_sorted_required() {
        let mut schema = Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![
                    ("zeta".to_string(), Value::Map(vec![])),
                    ("alpha".to_string(), Value::Map(vec![])),
                ]),
            ),
        ]);
        enforce_strict_openai_schema(&mut schema);
        assert_eq!(
            schema.get("additionalProperties"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            schema.get("required"),
            Some(&Value::Seq(vec![
                Value::String("alpha".to_string()),
                Value::String("zeta".to_string()),
            ]))
        );
    }

    #[test]
    fn enforce_strict_openai_schema_leaves_non_object_schemas_alone() {
        let mut schema = Value::Map(vec![(
            "type".to_string(),
            Value::String("string".to_string()),
        )]);
        enforce_strict_openai_schema(&mut schema);
        assert_eq!(schema.get("additionalProperties"), None);
    }

    #[test]
    fn enforce_strict_openai_schema_recurses_into_defs() {
        let mut schema = Value::Map(vec![(
            "$defs".to_string(),
            Value::Map(vec![(
                "Foo".to_string(),
                Value::Map(vec![
                    ("type".to_string(), Value::String("object".to_string())),
                    ("properties".to_string(), Value::Map(vec![])),
                ]),
            )]),
        )]);
        enforce_strict_openai_schema(&mut schema);
        let foo = schema.get("$defs").unwrap().get("Foo").unwrap();
        assert_eq!(foo.get("additionalProperties"), Some(&Value::Bool(false)));
    }

    #[test]
    fn enforce_strict_openai_schema_recurses_into_properties() {
        let mut schema = Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![(
                    "nested".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("object".to_string())),
                        ("properties".to_string(), Value::Map(vec![])),
                    ]),
                )]),
            ),
        ]);
        enforce_strict_openai_schema(&mut schema);
        let nested = schema.get("properties").unwrap().get("nested").unwrap();
        assert_eq!(
            nested.get("additionalProperties"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn enforce_strict_openai_schema_recurses_into_combinators() {
        for key in ["anyOf", "oneOf", "allOf"] {
            let mut schema = Value::Map(vec![(
                key.to_string(),
                Value::Seq(vec![Value::Map(vec![
                    ("type".to_string(), Value::String("object".to_string())),
                    ("properties".to_string(), Value::Map(vec![])),
                ])]),
            )]);
            enforce_strict_openai_schema(&mut schema);
            let item = &schema.get(key).unwrap().as_seq().unwrap()[0];
            assert_eq!(
                item.get("additionalProperties"),
                Some(&Value::Bool(false)),
                "combinator {key} did not recurse"
            );
        }
    }

    #[test]
    fn enforce_strict_openai_schema_recurses_into_array_items() {
        let mut schema = Value::Map(vec![
            ("type".to_string(), Value::String("array".to_string())),
            (
                "items".to_string(),
                Value::Map(vec![
                    ("type".to_string(), Value::String("object".to_string())),
                    ("properties".to_string(), Value::Map(vec![])),
                ]),
            ),
        ]);
        enforce_strict_openai_schema(&mut schema);
        let items = schema.get("items").unwrap();
        assert_eq!(items.get("additionalProperties"), Some(&Value::Bool(false)));
    }

    #[test]
    fn enforce_strict_openai_schema_ignores_a_non_map_items_value() {
        let mut schema = Value::Map(vec![
            ("type".to_string(), Value::String("array".to_string())),
            ("items".to_string(), Value::Bool(true)),
        ]);
        enforce_strict_openai_schema(&mut schema);
        assert_eq!(schema.get("items"), Some(&Value::Bool(true)));
    }
}
