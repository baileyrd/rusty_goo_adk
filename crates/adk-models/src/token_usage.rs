//! C0668: `telemetry._token_usage`, ported from
//! `google.adk.telemetry._token_usage`.
//!
//! **`usage_metadata`, kept as the opaque `Value` this port already
//! uses**: the source's `TokenUsage` wraps a typed
//! `types.GenerateContentResponseUsageMetadata`. [`crate::llm_response::LlmResponse::usage_metadata`]
//! is an opaque `Value` here (raw camelCase wire keys —
//! `promptTokenCount`/`candidatesTokenCount`/`cachedContentTokenCount`/
//! `toolUsePromptTokenCount`/`thoughtsTokenCount`), the same convention
//! `adk-flows::cache_performance_analyzer` already reads directly. This
//! port's [`TokenUsage`] reads those same keys instead of typed struct
//! fields — no new typed model is introduced for a single consumer.
//!
//! **`cache_creation_input_tokens`/`system_instruction_tokens`, disclosed
//! as never-present**: the source reads both via `getattr(...,
//! default=None)` since neither is a real field on the upstream
//! `GenerateContentResponseUsageMetadata` type yet (a forward-compatible
//! read for a future API addition). This port's `usage_metadata` is raw
//! wire JSON, so the equivalent is simply checking whether the
//! (currently never-emitted) camelCase key is present — ported the same
//! way, so this becomes real automatically the day the wire format adds
//! either key, with no code change needed here.

use rusty_serde::value::Value;
use std::collections::BTreeMap;

/// `_token_usage.GEN_AI_USAGE_INPUT_TOKENS`/`GEN_AI_USAGE_OUTPUT_TOKENS`
/// (imported from the OTel semconv package in the source; inlined here
/// since this port has no OTel-semconv dependency).
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
/// Not yet in a released OTel semconv version as of the source's own
/// comment — hand-written there too, ported verbatim.
pub const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";
pub const GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: &str =
    "gen_ai.usage.cache_creation.input_tokens";
pub const GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";
pub const GEN_AI_USAGE_EXPERIMENTAL_SYSTEM_INSTRUCTION_TOKENS: &str =
    "gen_ai.usage.experimental.system_instruction_tokens";

/// `_token_usage.TokenUsage` — centralized representation and processing
/// of GenAI token usage metadata.
pub struct TokenUsage<'a> {
    pub usage_metadata: Option<&'a Value>,
}

impl<'a> TokenUsage<'a> {
    pub fn new(usage_metadata: Option<&'a Value>) -> Self {
        Self { usage_metadata }
    }

    fn field(&self, key: &str) -> Option<i64> {
        self.usage_metadata?.get(key)?.as_i64()
    }

    /// `TokenUsage.input_token_count` — aggregates prompt and tool-use
    /// tokens, per the OTel semconv for `gen_ai.client.token.usage`.
    pub fn input_token_count(&self) -> Option<i64> {
        let prompt = self.field("promptTokenCount");
        let tool_use = self.field("toolUsePromptTokenCount");
        if prompt.is_none() && tool_use.is_none() {
            return None;
        }
        Some(prompt.unwrap_or(0) + tool_use.unwrap_or(0))
    }

    /// `TokenUsage.output_token_count` — `gen_ai.usage.reasoning.output_tokens`
    /// (`thoughts_token_count`) is included in the total per the OTel
    /// semconv, not reported only under its own separate attribute.
    pub fn output_token_count(&self) -> Option<i64> {
        let candidates = self.field("candidatesTokenCount");
        let thoughts = self.field("thoughtsTokenCount");
        if candidates.is_none() && thoughts.is_none() {
            return None;
        }
        Some(candidates.unwrap_or(0) + thoughts.unwrap_or(0))
    }

    /// `TokenUsage.to_attributes` — a map of OpenTelemetry token-usage
    /// attributes.
    pub fn to_attributes(&self) -> BTreeMap<String, Value> {
        let mut attrs = BTreeMap::new();
        if let Some(input) = self.input_token_count() {
            attrs.insert(GEN_AI_USAGE_INPUT_TOKENS.to_string(), Value::Int(input));
        }
        if let Some(output) = self.output_token_count() {
            attrs.insert(GEN_AI_USAGE_OUTPUT_TOKENS.to_string(), Value::Int(output));
        }
        if let Some(cached) = self.field("cachedContentTokenCount") {
            attrs.insert(
                GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS.to_string(),
                Value::Int(cached),
            );
        }
        if let Some(cache_creation) = self.field("cacheCreationInputTokens") {
            attrs.insert(
                GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS.to_string(),
                Value::Int(cache_creation),
            );
        }
        if let Some(thoughts) = self.field("thoughtsTokenCount") {
            attrs.insert(
                GEN_AI_USAGE_REASONING_OUTPUT_TOKENS.to_string(),
                Value::Int(thoughts),
            );
        }
        if let Some(system_instruction) = self.field("systemInstructionTokens") {
            attrs.insert(
                GEN_AI_USAGE_EXPERIMENTAL_SYSTEM_INSTRUCTION_TOKENS.to_string(),
                Value::Int(system_instruction),
            );
        }
        attrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(entries: Vec<(&str, i64)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), Value::Int(v)))
                .collect(),
        )
    }

    #[test]
    fn input_token_count_is_none_without_a_usage_metadata() {
        assert_eq!(TokenUsage::new(None).input_token_count(), None);
    }

    #[test]
    fn input_token_count_aggregates_prompt_and_tool_use() {
        let metadata = usage(vec![
            ("promptTokenCount", 10),
            ("toolUsePromptTokenCount", 5),
        ]);
        assert_eq!(
            TokenUsage::new(Some(&metadata)).input_token_count(),
            Some(15)
        );
    }

    #[test]
    fn input_token_count_treats_a_missing_field_as_zero() {
        let metadata = usage(vec![("promptTokenCount", 10)]);
        assert_eq!(
            TokenUsage::new(Some(&metadata)).input_token_count(),
            Some(10)
        );
    }

    #[test]
    fn output_token_count_aggregates_candidates_and_thoughts() {
        let metadata = usage(vec![("candidatesTokenCount", 7), ("thoughtsTokenCount", 3)]);
        assert_eq!(
            TokenUsage::new(Some(&metadata)).output_token_count(),
            Some(10)
        );
    }

    #[test]
    fn to_attributes_includes_only_present_fields() {
        let metadata = usage(vec![
            ("promptTokenCount", 10),
            ("candidatesTokenCount", 7),
            ("cachedContentTokenCount", 4),
        ]);
        let attrs = TokenUsage::new(Some(&metadata)).to_attributes();
        assert_eq!(attrs.get(GEN_AI_USAGE_INPUT_TOKENS), Some(&Value::Int(10)));
        assert_eq!(attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS), Some(&Value::Int(7)));
        assert_eq!(
            attrs.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS),
            Some(&Value::Int(4))
        );
        assert!(!attrs.contains_key(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS));
        assert!(!attrs.contains_key(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS));
        assert!(!attrs.contains_key(GEN_AI_USAGE_EXPERIMENTAL_SYSTEM_INSTRUCTION_TOKENS));
    }

    #[test]
    fn to_attributes_is_empty_without_a_usage_metadata() {
        assert!(TokenUsage::new(None).to_attributes().is_empty());
    }
}
