//! C0540/C0542 (partial P10 slice): pure, HTTP-free conversion helpers
//! from `models/anthropic_llm.py` — the Anthropic (Claude) model
//! backend. No new dependency: this port talks to Anthropic's Messages
//! API the same way `gemini.rs` already talks to Gemini's REST API
//! (`reqwest::blocking::Client`, since `adk-models` already depends on
//! `reqwest`/`regex` — see that file's own module doc for why the
//! blocking client, not the async one, is required under this
//! workspace's `rusty_tokio` runtime), not via the Python source's
//! `anthropic` SDK — there is no Rust equivalent of that SDK to add as a
//! dependency, and none is needed for a plain HTTPS JSON API.
//!
//! **Scope of this batch, disclosed**: [`ToolUseIdSanitizer`] (C0540),
//! the finish-reason mapping + token-usage extraction/reconciliation
//! functions (C0542), and now [`update_type_string`]/
//! [`function_declaration_to_tool_param`] (C0541) are ported here — all
//! pure, self-contained, and testable without any wire-format type
//! beyond the minimal [`AnthropicUsage`]/[`AnthropicToolParam`] structs
//! declared alongside them (both ahead of their own real caller, same
//! "widen/declare ahead of a consumer" precedent used throughout this
//! port — their real consumer is the still-deferred `AnthropicLlm`
//! backend). The rest of P10 (C0536-C0539, C0543, C0544 — the actual
//! `AnthropicLlm` `BaseLlm` backend, credential resolution, extended-
//! thinking mapping, the full content↔block conversion including
//! media/tool-result handling, and SSE streaming) is real, substantial
//! additional work deliberately left for a follow-up batch: each of
//! those needs either a non-trivial new wire-shape enum
//! (`_MessageBlockParam`'s 7 variants, with real image/PDF/tool-result
//! branching this port has no way to verify without a live Anthropic
//! endpoint to test against) or new fields on
//! [`crate::llm_request::GenerateContentConfigStub`]
//! (`temperature`/`top_p`/`top_k`/`stop_sequences`/`max_output_tokens`/a
//! real `thinking_config`, none of which exist there yet) — real,
//! separable units of work, not something to fold into this small
//! slice. C0541 turned out **not** to need any of those — it only
//! touches [`adk_genai::content::FunctionDeclaration`], which already
//! has everything required.
//!
//! **`to_google_genai_finish_reason`, wire string not enum**: the source
//! maps to a `types.FinishReason` enum member; this port's
//! `LlmResponse::finish_reason` is already an opaque
//! [`rusty_serde::value::Value`] holding the raw wire string (e.g.
//! `Value::String("STOP")`, per `llm_response.rs`'s own tests) — so this
//! returns that same wire string directly rather than a typed enum,
//! consistent with the "no enum to normalize away" precedent already
//! established in `stable_semconv.rs`.

use rusty_serde::value::Value;
use std::collections::HashMap;

/// `anthropic_llm._ToolUseIdSanitizer` — maps invalid `tool_use` ids to
/// deterministic fallbacks. Reuse one instance per conversation so a
/// `tool_use` and its paired `tool_result` with the same invalid source
/// id get matching outputs.
#[derive(Debug, Default)]
pub struct ToolUseIdSanitizer {
    mapping: HashMap<String, String>,
    next_fallback: u64,
}

fn is_valid_tool_id(tool_id: &str) -> bool {
    !tool_id.is_empty()
        && tool_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl ToolUseIdSanitizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `_ToolUseIdSanitizer.sanitize`.
    pub fn sanitize(&mut self, tool_id: Option<&str>) -> String {
        if let Some(tool_id) = tool_id {
            if is_valid_tool_id(tool_id) {
                return tool_id.to_string();
            }
        }
        let key = tool_id.unwrap_or("").to_string();
        if let Some(existing) = self.mapping.get(&key) {
            return existing.clone();
        }
        let fallback = format!("toolu_fallback_{}", self.next_fallback);
        self.next_fallback += 1;
        self.mapping.insert(key, fallback.clone());
        fallback
    }
}

/// Anthropic stop-reason wire strings → the GenAI `FinishReason` wire
/// string this port's [`crate::llm_response::LlmResponse::finish_reason`]
/// already holds directly (see the module doc for why no typed enum is
/// introduced here).
///
/// `anthropic_llm._STOP_REASON_MAPPING`/`to_google_genai_finish_reason`.
pub fn to_google_genai_finish_reason(anthropic_stop_reason: Option<&str>) -> Option<Value> {
    let stop_reason = anthropic_stop_reason?;
    let mapped = match stop_reason {
        "end_turn" | "stop_sequence" | "tool_use" | "pause_turn" => "STOP",
        "max_tokens" => "MAX_TOKENS",
        "refusal" => "SAFETY",
        _ => "FINISH_REASON_UNSPECIFIED",
    };
    Some(Value::String(mapped.to_string()))
}

/// `anthropic_types.Usage`/`MessageDeltaUsage` — only the fields this
/// port's token-usage reconciliation reads. Declared ahead of its real
/// caller (`message_to_generate_content_response`, deferred — see the
/// module doc), the same precedent used throughout this port.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    /// `usage.output_tokens_details.thinking_tokens`.
    pub thinking_tokens: Option<i64>,
}

/// `_extract_prompt_token_count` — every input token billed for the
/// turn. Anthropic reports tokens served from the prompt cache and
/// tokens written to it in their own fields, disjoint from
/// `input_tokens`; the GenAI shape instead expects a single prompt count
/// with the cached portion folded in (`cached_content_token_count` is a
/// breakdown of it, not an addition to it), so this sums all three.
pub fn extract_prompt_token_count(usage: &AnthropicUsage) -> i64 {
    usage.input_tokens.unwrap_or(0)
        + usage.cache_read_input_tokens.unwrap_or(0)
        + usage.cache_creation_input_tokens.unwrap_or(0)
}

/// `_extract_cached_token_count` — Anthropic cache-read tokens, the
/// analog of `cached_content_token_count`.
pub fn extract_cached_token_count(usage: &AnthropicUsage) -> Option<i64> {
    usage.cache_read_input_tokens
}

/// `_extract_cache_creation_token_count` — Anthropic cache-write tokens,
/// the analog of `cache_creation_input_tokens`.
pub fn extract_cache_creation_token_count(usage: &AnthropicUsage) -> Option<i64> {
    usage.cache_creation_input_tokens
}

/// `_extract_thinking_token_count` — Anthropic counts extended-thinking
/// tokens inside `output_tokens`, whereas the GenAI shape keeps the
/// candidate and thought counts disjoint and sums them downstream;
/// callers subtract this from `output_tokens` to get the candidate
/// count. Clamped to `output_tokens` so that subtraction stays
/// non-negative even if the two counters ever disagree.
pub fn extract_thinking_token_count(usage: &AnthropicUsage) -> Option<i64> {
    let thinking = usage.thinking_tokens?;
    match usage.output_tokens {
        Some(output_tokens) => Some(thinking.min(output_tokens)),
        None => Some(thinking),
    }
}

const DICT_KEYS_TO_RECURSE: &[&str] = &[
    "$defs",
    "defs",
    "dependentSchemas",
    "patternProperties",
    "properties",
];

const SINGLE_KEYS_TO_RECURSE: &[&str] = &[
    "additionalProperties",
    "additional_properties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedProperties",
];

const LIST_KEYS_TO_RECURSE: &[&str] = &[
    "allOf",
    "all_of",
    "anyOf",
    "any_of",
    "oneOf",
    "one_of",
    "prefixItems",
];

/// `anthropic_llm._update_type_string` — lowercases nested JSON-Schema
/// `"type"` strings for Anthropic compatibility, recursing into every
/// key a JSON Schema commonly nests a sub-schema under.
pub fn update_type_string(value: &mut Value) {
    match value {
        Value::Seq(items) => {
            for item in items {
                update_type_string(item);
            }
        }
        Value::Map(entries) => {
            if let Some((_, Value::String(type_name))) =
                entries.iter_mut().find(|(k, _)| k == "type")
            {
                *type_name = type_name.to_lowercase();
            }
            for key in DICT_KEYS_TO_RECURSE {
                if let Some((_, Value::Map(child_entries))) =
                    entries.iter_mut().find(|(k, _)| k == key)
                {
                    for (_, child_value) in child_entries {
                        update_type_string(child_value);
                    }
                }
            }
            for key in SINGLE_KEYS_TO_RECURSE {
                if let Some((_, child)) = entries.iter_mut().find(|(k, _)| k == key) {
                    if matches!(child, Value::Map(_) | Value::Seq(_)) {
                        update_type_string(child);
                    }
                }
            }
            for key in LIST_KEYS_TO_RECURSE {
                if let Some((_, child @ Value::Seq(_))) = entries.iter_mut().find(|(k, _)| k == key)
                {
                    update_type_string(child);
                }
            }
        }
        _ => {}
    }
}

/// `anthropic_types.ToolParam` — only the three fields the source ever
/// sets when constructing one (`anthropic_llm.py:801-805`). Declared
/// ahead of its real HTTP-transport caller, same precedent as
/// [`AnthropicUsage`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicToolParam {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// `anthropic_llm.function_declaration_to_tool_param` — converts a
/// function declaration to an Anthropic tool param.
///
/// **`parameters`-fallback, simplified — disclosed**: the source's
/// `else` branch reads `function_declaration.parameters.properties`, a
/// dict of typed `Schema` objects, and calls `.model_dump(by_alias=True,
/// exclude_none=True)` on each one to flatten it to a plain dict.
/// [`adk_genai::content::FunctionDeclaration::parameters`] is already an
/// opaque, already-flattened [`Value`] in this port (no typed `Schema`
/// object per property to begin with — same "already opaque, nothing
/// left to strip" situation this module already discloses for
/// `usage_metadata`/token extraction), so this reads `parameters`'s own
/// `"properties"`/`"required"` keys directly instead of rebuilding them
/// key-by-key.
///
/// Panics if `function_declaration.name` is `None` or empty, mirroring
/// the source's own `assert function_declaration.name` — a caller
/// invariant, not a user-reachable error path (this function has no
/// real caller yet to receive a `Result` through — see the module doc).
pub fn function_declaration_to_tool_param(
    function_declaration: &adk_genai::content::FunctionDeclaration,
) -> AnthropicToolParam {
    let name = function_declaration
        .name
        .clone()
        .filter(|n| !n.is_empty())
        .expect("function_declaration.name is required");

    let mut input_schema = if let Some(schema) = &function_declaration.parameters_json_schema {
        schema.clone()
    } else {
        let properties = function_declaration
            .parameters
            .as_ref()
            .and_then(|p| p.get("properties"))
            .cloned()
            .unwrap_or_else(|| Value::Map(Vec::new()));
        let required = function_declaration
            .parameters
            .as_ref()
            .and_then(|p| p.get("required"))
            .cloned();

        let mut schema = Value::Map(Vec::new());
        schema.insert("type", Value::String("object".to_string()));
        schema.insert("properties", properties);
        if let Some(required) = required {
            let is_empty = matches!(&required, Value::Seq(items) if items.is_empty());
            if !is_empty {
                schema.insert("required", required);
            }
        }
        schema
    };
    update_type_string(&mut input_schema);

    AnthropicToolParam {
        name,
        description: function_declaration.description.clone().unwrap_or_default(),
        input_schema,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ToolUseIdSanitizer ---

    #[test]
    fn sanitize_passes_through_a_valid_id() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        assert_eq!(sanitizer.sanitize(Some("toolu_abc123")), "toolu_abc123");
    }

    #[test]
    fn sanitize_replaces_an_invalid_id_with_a_deterministic_fallback() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        assert_eq!(sanitizer.sanitize(Some("bad id!")), "toolu_fallback_0");
    }

    #[test]
    fn sanitize_maps_the_same_invalid_id_to_the_same_fallback() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(Some("bad id!"));
        let second = sanitizer.sanitize(Some("bad id!"));
        assert_eq!(first, second);
    }

    #[test]
    fn sanitize_gives_distinct_ids_distinct_fallbacks() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(Some("bad id 1"));
        let second = sanitizer.sanitize(Some("bad id 2"));
        assert_ne!(first, second);
    }

    #[test]
    fn sanitize_treats_a_missing_id_as_its_own_key() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(None);
        let second = sanitizer.sanitize(None);
        assert_eq!(first, second);
        assert_eq!(first, "toolu_fallback_0");
    }

    // --- to_google_genai_finish_reason ---

    #[test]
    fn finish_reason_maps_none_to_none() {
        assert_eq!(to_google_genai_finish_reason(None), None);
    }

    #[test]
    fn finish_reason_maps_normal_stops_to_stop() {
        for reason in ["end_turn", "stop_sequence", "tool_use", "pause_turn"] {
            assert_eq!(
                to_google_genai_finish_reason(Some(reason)),
                Some(Value::String("STOP".to_string())),
                "reason {reason}"
            );
        }
    }

    #[test]
    fn finish_reason_maps_max_tokens() {
        assert_eq!(
            to_google_genai_finish_reason(Some("max_tokens")),
            Some(Value::String("MAX_TOKENS".to_string()))
        );
    }

    #[test]
    fn finish_reason_maps_refusal_to_safety() {
        assert_eq!(
            to_google_genai_finish_reason(Some("refusal")),
            Some(Value::String("SAFETY".to_string()))
        );
    }

    #[test]
    fn finish_reason_maps_unknown_to_unspecified() {
        assert_eq!(
            to_google_genai_finish_reason(Some("something_new")),
            Some(Value::String("FINISH_REASON_UNSPECIFIED".to_string()))
        );
    }

    // --- token usage extraction ---

    #[test]
    fn extract_prompt_token_count_sums_input_and_cache_fields() {
        let usage = AnthropicUsage {
            input_tokens: Some(10),
            cache_read_input_tokens: Some(5),
            cache_creation_input_tokens: Some(2),
            ..Default::default()
        };
        assert_eq!(extract_prompt_token_count(&usage), 17);
    }

    #[test]
    fn extract_prompt_token_count_treats_missing_fields_as_zero() {
        let usage = AnthropicUsage {
            input_tokens: Some(10),
            ..Default::default()
        };
        assert_eq!(extract_prompt_token_count(&usage), 10);
    }

    #[test]
    fn extract_cached_token_count_reads_cache_read_input_tokens() {
        let usage = AnthropicUsage {
            cache_read_input_tokens: Some(4),
            ..Default::default()
        };
        assert_eq!(extract_cached_token_count(&usage), Some(4));
    }

    #[test]
    fn extract_cache_creation_token_count_reads_cache_creation_input_tokens() {
        let usage = AnthropicUsage {
            cache_creation_input_tokens: Some(3),
            ..Default::default()
        };
        assert_eq!(extract_cache_creation_token_count(&usage), Some(3));
    }

    #[test]
    fn extract_thinking_token_count_is_none_without_thinking_tokens() {
        let usage = AnthropicUsage {
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), None);
    }

    #[test]
    fn extract_thinking_token_count_returns_the_raw_value_without_output_tokens() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(30),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(30));
    }

    #[test]
    fn extract_thinking_token_count_clamps_to_output_tokens() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(150),
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(100));
    }

    #[test]
    fn extract_thinking_token_count_does_not_clamp_when_within_bounds() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(20),
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(20));
    }

    // --- update_type_string ---

    fn object_map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    #[test]
    fn update_type_string_lowercases_a_top_level_type() {
        let mut schema = object_map(vec![("type", Value::String("OBJECT".to_string()))]);
        update_type_string(&mut schema);
        assert_eq!(
            schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
    }

    #[test]
    fn update_type_string_recurses_into_properties() {
        let mut schema = object_map(vec![(
            "properties",
            object_map(vec![(
                "x",
                object_map(vec![("type", Value::String("STRING".to_string()))]),
            )]),
        )]);
        update_type_string(&mut schema);
        let x_type = schema
            .get("properties")
            .and_then(|p| p.get("x"))
            .and_then(|x| x.get("type"));
        assert_eq!(x_type, Some(&Value::String("string".to_string())));
    }

    #[test]
    fn update_type_string_recurses_into_items_and_array_type_lists() {
        let mut schema = object_map(vec![
            (
                "items",
                object_map(vec![("type", Value::String("NUMBER".to_string()))]),
            ),
            (
                "anyOf",
                Value::Seq(vec![object_map(vec![(
                    "type",
                    Value::String("INTEGER".to_string()),
                )])]),
            ),
        ]);
        update_type_string(&mut schema);
        assert_eq!(
            schema.get("items").and_then(|i| i.get("type")),
            Some(&Value::String("number".to_string()))
        );
        let Some(Value::Seq(any_of)) = schema.get("anyOf") else {
            panic!("expected anyOf to remain a Seq");
        };
        assert_eq!(
            any_of[0].get("type"),
            Some(&Value::String("integer".to_string()))
        );
    }

    #[test]
    fn update_type_string_recurses_into_defs_and_pattern_properties() {
        let mut schema = object_map(vec![(
            "$defs",
            object_map(vec![(
                "Foo",
                object_map(vec![("type", Value::String("BOOLEAN".to_string()))]),
            )]),
        )]);
        update_type_string(&mut schema);
        let foo_type = schema
            .get("$defs")
            .and_then(|d| d.get("Foo"))
            .and_then(|f| f.get("type"));
        assert_eq!(foo_type, Some(&Value::String("boolean".to_string())));
    }

    #[test]
    fn update_type_string_ignores_non_dict_non_list_values() {
        let mut value = Value::String("OBJECT".to_string());
        update_type_string(&mut value);
        assert_eq!(value, Value::String("OBJECT".to_string()));
    }

    // --- function_declaration_to_tool_param ---

    use adk_genai::content::FunctionDeclaration;

    #[test]
    fn function_declaration_to_tool_param_prefers_parameters_json_schema() {
        let decl = FunctionDeclaration {
            name: Some("get_weather".to_string()),
            description: Some("Gets the weather".to_string()),
            parameters_json_schema: Some(object_map(vec![(
                "type",
                Value::String("OBJECT".to_string()),
            )])),
            parameters: Some(object_map(vec![(
                "type",
                Value::String("ignored".to_string()),
            )])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.name, "get_weather");
        assert_eq!(tool_param.description, "Gets the weather");
        assert_eq!(
            tool_param.input_schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
    }

    #[test]
    fn function_declaration_to_tool_param_builds_object_schema_from_parameters_when_json_schema_absent(
    ) {
        let decl = FunctionDeclaration {
            name: Some("get_weather".to_string()),
            parameters: Some(object_map(vec![
                (
                    "properties",
                    object_map(vec![(
                        "city",
                        object_map(vec![("type", Value::String("string".to_string()))]),
                    )]),
                ),
                (
                    "required",
                    Value::Seq(vec![Value::String("city".to_string())]),
                ),
            ])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(
            tool_param.input_schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
        assert_eq!(
            tool_param.input_schema.get("required"),
            Some(&Value::Seq(vec![Value::String("city".to_string())]))
        );
        assert!(tool_param
            .input_schema
            .get("properties")
            .and_then(|p| p.get("city"))
            .is_some());
    }

    #[test]
    fn function_declaration_to_tool_param_omits_required_when_empty() {
        let decl = FunctionDeclaration {
            name: Some("noop".to_string()),
            parameters: Some(object_map(vec![("required", Value::Seq(Vec::new()))])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.input_schema.get("required"), None);
    }

    #[test]
    fn function_declaration_to_tool_param_defaults_description_to_empty_string() {
        let decl = FunctionDeclaration {
            name: Some("noop".to_string()),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.description, "");
    }

    #[test]
    #[should_panic(expected = "function_declaration.name is required")]
    fn function_declaration_to_tool_param_panics_without_a_name() {
        let decl = FunctionDeclaration::default();
        let _ = function_declaration_to_tool_param(&decl);
    }
}
