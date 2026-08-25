//! Capability C0309: `RetryConfig`, ported from
//! `google.adk.workflow._retry_config`. Part of the P7 workflow/graph
//! engine's pure-data slice — see `workflow_node_state.rs`'s module doc
//! for the rest of this batch's scope and crate-placement reasoning.
//!
//! **`exceptions`, adaptation disclosed**: the source's field type is
//! `list[str | type[BaseException]] | None`, normalized by a
//! `@field_validator` into a uniform `list[str]` (an exception class is
//! replaced by its `__name__`) — needed because Python callers can pass
//! either a string or an actual exception class object interchangeably.
//! Rust has no such duality: there's no way to pass "a type" as an
//! ordinary value the way Python does, so every caller already has a
//! plain `Vec<String>` (or nothing) to supply — this port accepts
//! `Option<Vec<String>>` directly, with no normalization step to port,
//! since the thing being normalized *away from* isn't representable
//! here in the first place.

use rusty_serde::{Deserialize, Serialize};

/// `workflow._retry_config.RetryConfig` — configuration for retrying a
/// node. Every field is `Option`-with-a-runtime-default (not a
/// `#[rusty_serde(default = ...)]`-baked-in value) so an explicit `None`
/// round-trips distinctly from "the field was never set" — the same
/// shape the source's own `int | None = None` fields carry, read back
/// by [`crate::workflow_retry_utils`] via an `unwrap_or` at the point of
/// use rather than here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RetryConfig {
    /// Maximum number of attempts, including the original request. `0`
    /// or `1` means no retries. Defaults to `5` when unset.
    #[rusty_serde(default)]
    pub max_attempts: Option<i64>,
    /// Initial delay before the first retry, in seconds. Defaults to
    /// `1.0` when unset.
    #[rusty_serde(default)]
    pub initial_delay: Option<f64>,
    /// Maximum delay between retries, in seconds. Defaults to `60.0`
    /// when unset.
    #[rusty_serde(default)]
    pub max_delay: Option<f64>,
    /// Multiplier by which the delay increases after each attempt.
    /// Defaults to `2.0` when unset.
    #[rusty_serde(default)]
    pub backoff_factor: Option<f64>,
    /// Randomness factor for the delay. Defaults to `1.0` when unset;
    /// `0.0` removes randomness.
    #[rusty_serde(default)]
    pub jitter: Option<f64>,
    /// Exception type names to retry on. `None` means retry on all
    /// exceptions. See this module's own doc for the disclosed
    /// narrowing from the source's `list[str | type[BaseException]]`.
    #[rusty_serde(default)]
    pub exceptions: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_leaves_every_field_unset() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, None);
        assert_eq!(config.initial_delay, None);
        assert_eq!(config.max_delay, None);
        assert_eq!(config.backoff_factor, None);
        assert_eq!(config.jitter, None);
        assert_eq!(config.exceptions, None);
    }

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let config = RetryConfig {
            max_attempts: Some(3),
            initial_delay: Some(2.0),
            max_delay: Some(15.0),
            backoff_factor: Some(2.0),
            jitter: Some(0.5),
            exceptions: Some(vec!["ValueError".to_string()]),
        };
        let json = rusty_serde::json::to_string(&config).unwrap();
        assert!(json.contains("\"maxAttempts\":3"), "{json}");
        assert!(json.contains("\"backoffFactor\":2.0"), "{json}");
        let back: RetryConfig = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn deserializes_from_an_empty_object() {
        let config: RetryConfig = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(config, RetryConfig::default());
    }
}
