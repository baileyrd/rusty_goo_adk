//! Capability C0175: the `context_cache_processor` request processor,
//! ported from `google.adk.flows.llm_flows.context_cache_processor`.
//!
//! Enables context caching for agents that have it configured, and finds
//! the most recent cache metadata / prompt token count for the current
//! agent from session history — the actual cache management (creating/
//! refreshing the cache itself) is handled by the model-specific cache
//! manager (`GeminiContextCacheManager`, already ported).
//!
//! **Adaptation, disclosed**: `Event.cache_metadata`/`usage_metadata` stay
//! opaque `Value` placeholders (see `adk-events`'s own module doc) —
//! `find_cache_info_from_events` parses `cache_metadata` back into a real
//! `adk_models::cache_metadata::CacheMetadata` via its own `Deserialize`
//! impl (`rusty_serde::json::from_value`) rather than requiring `Event`
//! itself to hold a typed field, since `adk-events` sits below `adk-models`
//! in the crate graph and can't depend on it without a cycle (the same
//! constraint disclosed in this crate's own top-level module doc). Reads
//! `usage_metadata`'s `promptTokenCount` key directly rather than through
//! a typed `UsageMetadata`, since no such type exists in this port yet.
//!
//! **Scope, disclosed**: this is the free-function core logic
//! (`find_cache_info_from_events`, `apply_context_cache`), not yet a real
//! `BaseLlmRequestProcessor` reading through `InvocationContext` — same
//! scope note as every other Phase 4 processor in this crate.

use adk_agents::context_cache_config::ContextCacheConfig;
use adk_events::Event;
use adk_models::cache_metadata::CacheMetadata;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

#[derive(Debug, rusty_err::Error)]
pub enum ContextCacheError {
    #[error("event.cache_metadata failed to parse as CacheMetadata: {0}")]
    InvalidCacheMetadata(String),
    #[error("Active cache metadata must include invocations_used.")]
    MissingInvocationsUsed,
}

/// `_find_cache_info_from_events`: scans `events` from most recent to
/// oldest for the given agent's most recent cache metadata and prompt
/// token count.
///
/// A cache metadata found on a *different* invocation than
/// `current_invocation_id`, with an active cache (`cache_name` set), has
/// its `invocations_used` incremented by one — it's about to be used
/// again by this new invocation. Same-invocation or fingerprint-only
/// metadata is returned as-is.
pub fn find_cache_info_from_events(
    events: &[Event],
    agent_name: &str,
    current_invocation_id: &str,
) -> Result<(Option<CacheMetadata>, Option<i64>), ContextCacheError> {
    let mut cache_metadata: Option<CacheMetadata> = None;
    let mut previous_token_count: Option<i64> = None;

    for event in events.iter().rev() {
        if event.author != agent_name {
            continue;
        }

        if cache_metadata.is_none() {
            if let Some(raw) = &event.cache_metadata {
                let parsed: CacheMetadata = rusty_serde::json::from_value(raw.clone())
                    .map_err(|e| ContextCacheError::InvalidCacheMetadata(e.to_string()))?;
                let is_active_cache_from_prior_invocation = !event.invocation_id.is_empty()
                    && event.invocation_id != current_invocation_id
                    && parsed.cache_name.is_some();
                cache_metadata = Some(if is_active_cache_from_prior_invocation {
                    let invocations_used = parsed
                        .invocations_used
                        .ok_or(ContextCacheError::MissingInvocationsUsed)?;
                    CacheMetadata {
                        invocations_used: Some(invocations_used + 1),
                        ..parsed
                    }
                } else {
                    parsed
                });
            }
        }

        if previous_token_count.is_none() {
            if let Some(usage) = &event.usage_metadata {
                previous_token_count = usage.get("promptTokenCount").and_then(Value::as_i64);
            }
        }

        if cache_metadata.is_some() && previous_token_count.is_some() {
            break;
        }
    }

    Ok((cache_metadata, previous_token_count))
}

/// `ContextCacheRequestProcessor.run_async`'s core assembly: sets
/// `llm_request.cache_config` when the agent has context caching
/// configured, and fills in the most recent `cache_metadata`/
/// `cacheable_contents_token_count` found via
/// [`find_cache_info_from_events`]. A no-op when `context_cache_config`
/// is `None`.
pub fn apply_context_cache(
    llm_request: &mut LlmRequest,
    context_cache_config: Option<&ContextCacheConfig>,
    events: &[Event],
    agent_name: &str,
    current_invocation_id: &str,
) -> Result<(), ContextCacheError> {
    let Some(context_cache_config) = context_cache_config else {
        return Ok(());
    };
    llm_request.cache_config = Some(context_cache_config.clone());

    let (cache_metadata, previous_token_count) =
        find_cache_info_from_events(events, agent_name, current_invocation_id)?;
    if let Some(cache_metadata) = cache_metadata {
        llm_request.cache_metadata = Some(cache_metadata);
    }
    if let Some(previous_token_count) = previous_token_count {
        llm_request.cacheable_contents_token_count = Some(previous_token_count);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;

    fn event(author: &str, invocation_id: &str) -> Event {
        Event::new(invocation_id, author, NodeInfo::new("root"))
    }

    fn active_cache_value(cache_name: &str, invocations_used: u32) -> Value {
        Value::Map(vec![
            (
                "cache_name".to_string(),
                Value::String(cache_name.to_string()),
            ),
            ("expire_time".to_string(), Value::Float(1_000_000.0)),
            ("fingerprint".to_string(), Value::String("fp".to_string())),
            (
                "invocations_used".to_string(),
                Value::UInt(invocations_used as u64),
            ),
            ("contents_count".to_string(), Value::UInt(3)),
            ("created_at".to_string(), Value::Null),
        ])
    }

    fn fingerprint_only_value() -> Value {
        Value::Map(vec![
            ("cache_name".to_string(), Value::Null),
            ("expire_time".to_string(), Value::Null),
            ("fingerprint".to_string(), Value::String("fp".to_string())),
            ("invocations_used".to_string(), Value::Null),
            ("contents_count".to_string(), Value::UInt(3)),
            ("created_at".to_string(), Value::Null),
        ])
    }

    fn usage_metadata_value(prompt_token_count: i64) -> Value {
        Value::Map(vec![(
            "promptTokenCount".to_string(),
            Value::Int(prompt_token_count),
        )])
    }

    #[test]
    fn finds_nothing_when_no_events_carry_cache_or_usage_metadata() {
        let events = vec![event("agent_a", "inv-1")];
        let (cache_metadata, token_count) =
            find_cache_info_from_events(&events, "agent_a", "inv-2").unwrap();
        assert!(cache_metadata.is_none());
        assert!(token_count.is_none());
    }

    #[test]
    fn ignores_events_from_a_different_author() {
        let mut e = event("agent_b", "inv-1");
        e.cache_metadata = Some(fingerprint_only_value());
        let (cache_metadata, _) = find_cache_info_from_events(&[e], "agent_a", "inv-2").unwrap();
        assert!(cache_metadata.is_none());
    }

    #[test]
    fn returns_fingerprint_only_metadata_as_is() {
        let mut e = event("agent_a", "inv-1");
        e.cache_metadata = Some(fingerprint_only_value());
        let (cache_metadata, _) = find_cache_info_from_events(&[e], "agent_a", "inv-1").unwrap();
        let cache_metadata = cache_metadata.unwrap();
        assert!(cache_metadata.cache_name.is_none());
    }

    #[test]
    fn increments_invocations_used_for_an_active_cache_from_a_prior_invocation() {
        let mut e = event("agent_a", "inv-1");
        e.cache_metadata = Some(active_cache_value("projects/p/cachedContents/1", 3));
        let (cache_metadata, _) = find_cache_info_from_events(&[e], "agent_a", "inv-2").unwrap();
        assert_eq!(cache_metadata.unwrap().invocations_used, Some(4));
    }

    #[test]
    fn leaves_invocations_used_unchanged_within_the_same_invocation() {
        let mut e = event("agent_a", "inv-1");
        e.cache_metadata = Some(active_cache_value("projects/p/cachedContents/1", 3));
        let (cache_metadata, _) = find_cache_info_from_events(&[e], "agent_a", "inv-1").unwrap();
        assert_eq!(cache_metadata.unwrap().invocations_used, Some(3));
    }

    #[test]
    fn finds_the_most_recent_prompt_token_count() {
        let mut older = event("agent_a", "inv-0");
        older.usage_metadata = Some(usage_metadata_value(10));
        let mut newer = event("agent_a", "inv-1");
        newer.usage_metadata = Some(usage_metadata_value(42));
        let (_, token_count) =
            find_cache_info_from_events(&[older, newer], "agent_a", "inv-2").unwrap();
        assert_eq!(token_count, Some(42));
    }

    #[test]
    fn apply_context_cache_is_a_no_op_without_a_configured_context_cache() {
        let mut request = LlmRequest::default();
        apply_context_cache(&mut request, None, &[], "agent_a", "inv-1").unwrap();
        assert!(request.cache_config.is_none());
        assert!(request.cache_metadata.is_none());
    }

    #[test]
    fn apply_context_cache_sets_config_and_recovered_metadata() {
        let config = ContextCacheConfig::default();
        let mut e = event("agent_a", "inv-1");
        e.cache_metadata = Some(fingerprint_only_value());
        e.usage_metadata = Some(usage_metadata_value(99));

        let mut request = LlmRequest::default();
        apply_context_cache(&mut request, Some(&config), &[e], "agent_a", "inv-2").unwrap();
        assert!(request.cache_config.is_some());
        assert!(request.cache_metadata.is_some());
        assert_eq!(request.cacheable_contents_token_count, Some(99));
    }
}
