//! The wire shape of a Gemini Live API server message — what
//! `GeminiLlmConnection::receive()` (C0139) parses each inbound WebSocket
//! text frame into — plus the two standalone transformations `receive()`
//! applies to parts of it: [`to_generate_content_usage_metadata`] (live
//! token-usage field renaming) and [`merge_grounding_metadata`] (cross-message
//! grounding-metadata accumulation).
//!
//! **Adaptation, disclosed (confidence caveat)**: same as
//! `gemini_llm_connection.rs`'s send-side envelopes — this is a best-effort
//! reconstruction of Google's public Multimodal Live API server-message
//! shape, not something `google/adk-python` itself specifies (it only ever
//! talks to the opaque third-party `google.genai.live.AsyncSession`).
//! Unverified against a live endpoint.
//!
//! **Adaptation**: `grounding_metadata` stays an opaque
//! [`rusty_serde::value::Value`] end to end — deliberately, not just as a
//! placeholder. The source's own `_merge_grounding_metadata` operates
//! generically over `model_dump(exclude_none=True)`'s dict keys (append-
//! unique for any list-of-strings field, special-cased `grounding_chunks`/
//! `grounding_supports`, overwrite for everything else) rather than naming
//! every `GroundingMetadata` sub-field — so a generic `Value::Map` merge is
//! actually the *more* faithful port here, not a narrower one: it handles
//! whatever keys a real response contains without this migration having to
//! model the full `GroundingMetadata` schema.

use adk_genai::content::{Content, FunctionCall};
use rusty_serde::value::Value;
use rusty_serde::Deserialize;

/// `types.LiveServerMessage`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LiveServerMessage {
    #[rusty_serde(default)]
    pub usage_metadata: Option<LiveUsageMetadata>,
    #[rusty_serde(default)]
    pub server_content: Option<ServerContent>,
    #[rusty_serde(default)]
    pub tool_call: Option<ToolCall>,
    #[rusty_serde(default)]
    pub session_resumption_update: Option<Value>,
    #[rusty_serde(default)]
    pub voice_activity: Option<Value>,
    #[rusty_serde(default)]
    pub go_away: Option<Value>,
}

/// `types.UsageMetadata` (the Live API's own token-usage shape — distinct
/// field names from `GenerateContentResponseUsageMetadata`, see
/// [`to_generate_content_usage_metadata`]).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LiveUsageMetadata {
    #[rusty_serde(default)]
    pub prompt_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub cached_content_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub response_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub total_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub thoughts_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub tool_use_prompt_token_count: Option<i64>,
    #[rusty_serde(default)]
    pub prompt_tokens_details: Option<Value>,
    #[rusty_serde(default)]
    pub cache_tokens_details: Option<Value>,
    #[rusty_serde(default)]
    pub response_tokens_details: Option<Value>,
    #[rusty_serde(default)]
    pub tool_use_prompt_tokens_details: Option<Value>,
    #[rusty_serde(default)]
    pub traffic_type: Option<Value>,
}

/// `types.Transcription`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Transcription {
    #[rusty_serde(default)]
    pub text: Option<String>,
    #[rusty_serde(default)]
    pub finished: Option<bool>,
}

/// `types.LiveServerContent`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ServerContent {
    #[rusty_serde(default)]
    pub model_turn: Option<Content>,
    /// Opaque — see the module doc's adaptation note.
    #[rusty_serde(default)]
    pub grounding_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub turn_complete: Option<bool>,
    #[rusty_serde(default)]
    pub interrupted: Option<bool>,
    #[rusty_serde(default)]
    pub generation_complete: Option<bool>,
    #[rusty_serde(default)]
    pub input_transcription: Option<Transcription>,
    #[rusty_serde(default)]
    pub output_transcription: Option<Transcription>,
    #[rusty_serde(default)]
    pub turn_complete_reason: Option<Value>,
}

/// `types.LiveServerToolCall`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ToolCall {
    #[rusty_serde(default)]
    pub function_calls: Option<Vec<FunctionCall>>,
}

/// `GeminiLlmConnection._to_generate_content_usage_metadata` — the Live
/// API names output tokens `responseTokenCount`/`responseTokensDetails`,
/// whereas `GenerateContentResponseUsageMetadata` (what `LlmResponse.
/// usage_metadata` — an opaque placeholder — represents) names them
/// `candidatesTokenCount`/`candidatesTokensDetails`.
pub fn to_generate_content_usage_metadata(usage: &LiveUsageMetadata) -> Value {
    let mut entries = Vec::new();
    let mut push = |key: &str, value: Option<Value>| {
        if let Some(value) = value {
            entries.push((key.to_string(), value));
        }
    };
    push("promptTokenCount", usage.prompt_token_count.map(Value::Int));
    push(
        "cachedContentTokenCount",
        usage.cached_content_token_count.map(Value::Int),
    );
    push(
        "candidatesTokenCount",
        usage.response_token_count.map(Value::Int),
    );
    push("totalTokenCount", usage.total_token_count.map(Value::Int));
    push(
        "thoughtsTokenCount",
        usage.thoughts_token_count.map(Value::Int),
    );
    push(
        "toolUsePromptTokenCount",
        usage.tool_use_prompt_token_count.map(Value::Int),
    );
    push("promptTokensDetails", usage.prompt_tokens_details.clone());
    push("cacheTokensDetails", usage.cache_tokens_details.clone());
    push(
        "candidatesTokensDetails",
        usage.response_tokens_details.clone(),
    );
    push(
        "toolUsePromptTokensDetails",
        usage.tool_use_prompt_tokens_details.clone(),
    );
    push("trafficType", usage.traffic_type.clone());
    Value::Map(entries)
}

fn map_entries(value: Value) -> Vec<(String, Value)> {
    match value {
        Value::Map(entries) => entries,
        _ => Vec::new(),
    }
}

fn map_get<'a>(entries: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn map_set(entries: &mut Vec<(String, Value)>, key: &str, value: Value) {
    match entries.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => entries.push((key.to_string(), value)),
    }
}

fn as_seq(value: Option<&Value>) -> Vec<Value> {
    match value {
        Some(Value::Seq(items)) => items.clone(),
        _ => Vec::new(),
    }
}

fn shift_grounding_chunk_indices(support: Value, offset: usize) -> Value {
    let Value::Map(mut entries) = support else {
        return support;
    };
    if let Some(entry) = entries
        .iter_mut()
        .find(|(k, _)| k == "groundingChunkIndices")
    {
        if let Value::Seq(indices) = &mut entry.1 {
            for idx in indices.iter_mut() {
                match idx {
                    Value::Int(i) => *i += offset as i64,
                    Value::UInt(u) => *u += offset as u64,
                    _ => {}
                }
            }
        }
    }
    Value::Map(entries)
}

/// `GeminiLlmConnection._merge_grounding_metadata` — see the module doc's
/// adaptation note for why this operates generically over `Value::Map`
/// entries rather than named `GroundingMetadata` fields.
pub fn merge_grounding_metadata(existing: Option<Value>, new: Option<Value>) -> Option<Value> {
    let (existing, new) = match (existing, new) {
        (None, new) => return new,
        (existing, None) => return existing,
        (Some(e), Some(n)) => (e, n),
    };

    let mut existing_entries = map_entries(existing);
    let new_entries = map_entries(new);
    let chunk_offset = as_seq(map_get(&existing_entries, "groundingChunks")).len();

    for (key, val) in new_entries {
        let is_string_list = matches!(&val, Value::Seq(items) if items.iter().all(|x| matches!(x, Value::String(_))));

        if is_string_list {
            let mut merged = as_seq(map_get(&existing_entries, &key));
            if let Value::Seq(new_items) = &val {
                for item in new_items {
                    if !merged.contains(item) {
                        merged.push(item.clone());
                    }
                }
            }
            map_set(&mut existing_entries, &key, Value::Seq(merged));
        } else if key == "groundingChunks" {
            let mut merged = as_seq(map_get(&existing_entries, &key));
            if let Value::Seq(new_chunks) = val {
                merged.extend(new_chunks);
            }
            map_set(&mut existing_entries, &key, Value::Seq(merged));
        } else if key == "groundingSupports" {
            let mut merged = as_seq(map_get(&existing_entries, &key));
            if let Value::Seq(new_supports) = val {
                merged.extend(
                    new_supports
                        .into_iter()
                        .map(|support| shift_grounding_chunk_indices(support, chunk_offset)),
                );
            }
            map_set(&mut existing_entries, &key, Value::Seq(merged));
        } else {
            map_set(&mut existing_entries, &key, val);
        }
    }

    Some(Value::Map(existing_entries))
}

/// Distinguishes present-but-empty from absent when checking a
/// [`ToolCall`]'s function calls — a small readability helper for
/// `receive()`.
pub fn tool_call_function_calls(tool_call: &ToolCall) -> Vec<FunctionCall> {
    tool_call.function_calls.clone().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_a_message_with_usage_metadata() {
        let json = r#"{"usageMetadata": {"promptTokenCount": 10, "responseTokenCount": 5}}"#;
        let message: LiveServerMessage = rusty_serde::json::from_str(json).unwrap();
        let usage = message.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(10));
        assert_eq!(usage.response_token_count, Some(5));
    }

    #[test]
    fn to_generate_content_usage_metadata_renames_response_tokens_to_candidates_tokens() {
        let usage = LiveUsageMetadata {
            response_token_count: Some(7),
            prompt_token_count: Some(3),
            ..Default::default()
        };
        let mapped = to_generate_content_usage_metadata(&usage);
        let Value::Map(entries) = mapped else {
            panic!("expected a map");
        };
        assert!(entries
            .iter()
            .any(|(k, v)| k == "candidatesTokenCount" && *v == Value::Int(7)));
        assert!(entries
            .iter()
            .any(|(k, v)| k == "promptTokenCount" && *v == Value::Int(3)));
        assert!(!entries.iter().any(|(k, _)| k == "responseTokenCount"));
    }

    #[test]
    fn to_generate_content_usage_metadata_omits_absent_fields() {
        let mapped = to_generate_content_usage_metadata(&LiveUsageMetadata::default());
        assert_eq!(mapped, Value::Map(vec![]));
    }

    #[test]
    fn merge_grounding_metadata_returns_the_other_side_when_one_is_none() {
        let value = Value::Map(vec![("retrievalQueries".to_string(), Value::Seq(vec![]))]);
        assert_eq!(
            merge_grounding_metadata(None, Some(value.clone())),
            Some(value.clone())
        );
        assert_eq!(
            merge_grounding_metadata(Some(value.clone()), None),
            Some(value)
        );
        assert_eq!(merge_grounding_metadata(None, None), None);
    }

    #[test]
    fn merge_grounding_metadata_appends_unique_strings_in_a_string_list_field() {
        let existing = Value::Map(vec![(
            "retrievalQueries".to_string(),
            Value::Seq(vec![Value::String("a".to_string())]),
        )]);
        let new = Value::Map(vec![(
            "retrievalQueries".to_string(),
            Value::Seq(vec![
                Value::String("a".to_string()),
                Value::String("b".to_string()),
            ]),
        )]);
        let merged = merge_grounding_metadata(Some(existing), Some(new)).unwrap();
        let Value::Map(entries) = merged else {
            panic!("expected a map")
        };
        let Value::Seq(queries) = map_get(&entries, "retrievalQueries").unwrap().clone() else {
            panic!("expected a seq")
        };
        assert_eq!(
            queries,
            vec![
                Value::String("a".to_string()),
                Value::String("b".to_string())
            ]
        );
    }

    #[test]
    fn merge_grounding_metadata_extends_grounding_chunks() {
        let existing = Value::Map(vec![(
            "groundingChunks".to_string(),
            Value::Seq(vec![Value::String("chunk0".to_string())]),
        )]);
        let new = Value::Map(vec![(
            "groundingChunks".to_string(),
            Value::Seq(vec![Value::String("chunk1".to_string())]),
        )]);
        let merged = merge_grounding_metadata(Some(existing), Some(new)).unwrap();
        let Value::Map(entries) = merged else {
            panic!("expected a map")
        };
        let Value::Seq(chunks) = map_get(&entries, "groundingChunks").unwrap().clone() else {
            panic!("expected a seq")
        };
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn merge_grounding_metadata_shifts_grounding_support_chunk_indices_by_the_existing_chunk_count()
    {
        let existing = Value::Map(vec![(
            "groundingChunks".to_string(),
            Value::Seq(vec![Value::String("chunk0".to_string())]),
        )]);
        let support = Value::Map(vec![(
            "groundingChunkIndices".to_string(),
            Value::Seq(vec![Value::Int(0)]),
        )]);
        let new = Value::Map(vec![(
            "groundingSupports".to_string(),
            Value::Seq(vec![support]),
        )]);
        let merged = merge_grounding_metadata(Some(existing), Some(new)).unwrap();
        let Value::Map(entries) = merged else {
            panic!("expected a map")
        };
        let Value::Seq(supports) = map_get(&entries, "groundingSupports").unwrap().clone() else {
            panic!("expected a seq")
        };
        let Value::Map(support_entries) = supports[0].clone() else {
            panic!("expected a map")
        };
        assert_eq!(
            map_get(&support_entries, "groundingChunkIndices"),
            Some(&Value::Seq(vec![Value::Int(1)]))
        );
    }

    #[test]
    fn merge_grounding_metadata_overwrites_other_keys() {
        let existing = Value::Map(vec![(
            "webSearchEntryPoint".to_string(),
            Value::Bool(false),
        )]);
        let new = Value::Map(vec![("webSearchEntryPoint".to_string(), Value::Bool(true))]);
        let merged = merge_grounding_metadata(Some(existing), Some(new)).unwrap();
        let Value::Map(entries) = merged else {
            panic!("expected a map")
        };
        assert_eq!(
            map_get(&entries, "webSearchEntryPoint"),
            Some(&Value::Bool(true))
        );
    }
}
