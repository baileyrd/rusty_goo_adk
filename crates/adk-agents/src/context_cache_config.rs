//! Part of capability C0078: `ContextCacheConfig`, ported from
//! `google.adk.agents.context_cache_config`.
//!
//! Marked experimental in the source (`FeatureName.AGENT_CONFIG`) — Rust has
//! no decorator-based feature-flag registry equivalent, so the experimental
//! status is documented here rather than runtime-enforced (see
//! `features/` in the P12 phase for whether that registry gets ported).

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// Configuration for context caching across all agents in an app.
///
/// Caching begins on the second turn of a session at the earliest and
/// requires the cacheable prefix to reach the model-specific minimum (2048
/// tokens for Gemini 2.5, 4096 for Gemini 3) — that floor always applies
/// regardless of `min_tokens`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct ContextCacheConfig {
    pub cache_intervals: u32,
    pub ttl_seconds: u32,
    pub min_tokens: u32,
    /// Opaque `google.genai.types.HttpOptions` placeholder — see
    /// `run_config`'s module doc for why.
    #[rusty_serde(default)]
    pub create_http_options: Option<Value>,
}

impl Default for ContextCacheConfig {
    fn default() -> Self {
        Self {
            cache_intervals: 10,
            ttl_seconds: 1800,
            min_tokens: 0,
            create_http_options: None,
        }
    }
}

impl ContextCacheConfig {
    pub fn ttl_string(&self) -> String {
        format!("{}s", self.ttl_seconds)
    }
}

impl std::fmt::Display for ContextCacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ContextCacheConfig(cache_intervals={}, ttl={}s, min_tokens={}, create_http_options={:?})",
            self.cache_intervals, self.ttl_seconds, self.min_tokens, self.create_http_options
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_source() {
        let config = ContextCacheConfig::default();
        assert_eq!(config.cache_intervals, 10);
        assert_eq!(config.ttl_seconds, 1800);
        assert_eq!(config.min_tokens, 0);
    }

    #[test]
    fn ttl_string_appends_seconds_suffix() {
        assert_eq!(ContextCacheConfig::default().ttl_string(), "1800s");
    }
}
