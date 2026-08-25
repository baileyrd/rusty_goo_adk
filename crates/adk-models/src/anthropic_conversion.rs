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
//! **Scope of this batch, disclosed**: only [`ToolUseIdSanitizer`]
//! (C0540) and the finish-reason mapping + token-usage extraction/
//! reconciliation functions (C0542) are ported here — both pure,
//! self-contained, and testable without any wire-format type beyond the
//! minimal [`AnthropicUsage`] struct declared alongside them (itself
//! ahead of its own real caller, same "widen/declare ahead of a
//! consumer" precedent used throughout this port — its real consumer is
//! the still-deferred `message_to_generate_content_response`). The rest
//! of P10 (C0536-C0539, C0541, C0543, C0544 — the actual `AnthropicLlm`
//! `BaseLlm` backend, credential resolution, extended-thinking mapping,
//! the full content↔block conversion including media/tool-result
//! handling, tool-schema conversion, and SSE streaming) is real,
//! substantial additional work deliberately left for a follow-up batch:
//! each of those needs either a non-trivial new wire-shape enum
//! (`_MessageBlockParam`'s 7 variants, with real image/PDF/tool-result
//! branching this port has no way to verify without a live Anthropic
//! endpoint to test against) or new fields on
//! [`crate::llm_request::GenerateContentConfigStub`]
//! (`temperature`/`top_p`/`top_k`/`stop_sequences`/`max_output_tokens`/a
//! real `thinking_config`, none of which exist there yet) — real,
//! separable units of work, not something to fold into this small
//! slice.
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
}
