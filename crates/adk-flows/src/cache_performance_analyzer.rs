//! Capability C0946: `CachePerformanceAnalyzer`, ported from
//! `google.adk.utils.cache_performance_analyzer`.
//!
//! **Adaptation, disclosed**: `Event.cache_metadata`/`usage_metadata` stay
//! opaque `Value` placeholders (see `adk-events`'s own module doc), the
//! same constraint already disclosed in this crate's own
//! `context_cache.rs` (C0175) — `adk-events` sits below `adk-models` in
//! the crate graph and can't depend on it for a typed `CacheMetadata`
//! field without a cycle. This port parses `cache_metadata` back into a
//! real `adk_models::cache_metadata::CacheMetadata` on demand
//! (`rusty_serde::json::from_value`), the same idiom
//! `find_cache_info_from_events` already uses, and reads
//! `usage_metadata`'s `promptTokenCount`/`cachedContentTokenCount` keys
//! directly rather than through a typed `UsageMetadata` (no such type
//! exists in this port yet).
//!
//! **Adaptation**: the source returns an untyped `Dict[str, Any]` (either
//! `{"status": "no_cache_data"}` or a big flat dict of stats). This port
//! returns a [`CachePerformanceReport`] enum (`NoCacheData` /
//! `Active(CachePerformanceStats)`) instead — a closed, inspectable shape
//! is a strict improvement over a stringly-keyed dict a caller would have
//! to know the exact key set of, not a narrowing. No consumer in this
//! workspace needs a serialized wire form yet, so `CachePerformanceStats`
//! has no `Serialize` derive; add one if/when a caller needs it.
//!
//! **Adaptation, compile-time strengthening**: the source calls
//! `self.session_service.get_session(...)` and immediately reads
//! `session.events` with no `None` check — an implicit assumption the
//! session exists (would raise `AttributeError` otherwise). This port's
//! `get_session` returns `Option<Session>` (already the case before this
//! batch), so a missing session becomes an explicit
//! `Err(CachePerformanceError::SessionNotFound)` rather than a possible
//! panic — preserving the source's *intent* (a missing session is a
//! caller error) more safely than its literal implementation.
//!
//! **Not represented**: the source decorates the class `@experimental`
//! (`utils/feature_decorator.py`) — that decorator's own manifest row,
//! C0797, is still unresolved (possibly a second, parallel feature-gating
//! mechanism to `features/_feature_decorator.py`, flagged for
//! verification), so this port doesn't invent a representation for it
//! here rather than guess at C0797's still-open resolution.
//!
//! **Preserved deliberately**: `analyze_agent_cache_performance` fetches
//! the session twice — once inside `get_agent_cache_history`, once again
//! directly — matching the source exactly rather than "fixing" the
//! redundancy, since a real session backend could reasonably reload
//! fresher data on a second call.

use std::collections::HashSet;

use adk_agents::services::SessionService;
use adk_models::cache_metadata::CacheMetadata;
use rusty_serde::value::Value;

#[derive(Debug, rusty_err::Error)]
pub enum CachePerformanceError {
    #[error("session {session_id:?} not found (app {app_name:?}, user {user_id:?})")]
    SessionNotFound {
        session_id: String,
        app_name: String,
        user_id: String,
    },
    #[error("event.cache_metadata failed to parse as CacheMetadata: {0}")]
    InvalidCacheMetadata(String),
}

/// `analyze_agent_cache_performance`'s return shape — see the module doc
/// for why this replaces the source's untyped `Dict[str, Any]`.
#[derive(Debug, Clone, PartialEq)]
pub enum CachePerformanceReport {
    /// `{"status": "no_cache_data"}` — no cache metadata found for this
    /// agent in the session.
    NoCacheData,
    Active(CachePerformanceStats),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CachePerformanceStats {
    pub requests_with_cache: usize,
    pub avg_invocations_used: f64,
    pub latest_cache: Option<String>,
    pub cache_refreshes: usize,
    pub total_invocations: u64,
    pub total_prompt_tokens: i64,
    pub total_cached_tokens: i64,
    pub cache_hit_ratio_percent: f64,
    pub cache_utilization_ratio_percent: f64,
    pub avg_cached_tokens_per_request: f64,
    pub total_requests: usize,
    pub requests_with_cache_hits: usize,
}

/// `cache_performance_analyzer.CachePerformanceAnalyzer` — analyzes cache
/// performance through a session's event history.
pub struct CachePerformanceAnalyzer<'a> {
    session_service: &'a dyn SessionService,
}

impl<'a> CachePerformanceAnalyzer<'a> {
    pub fn new(session_service: &'a dyn SessionService) -> Self {
        Self { session_service }
    }

    async fn session_or_error(
        &self,
        session_id: &str,
        user_id: &str,
        app_name: &str,
    ) -> Result<adk_agents::session::Session, CachePerformanceError> {
        self.session_service
            .get_session(app_name, user_id, session_id)
            .await
            .ok_or_else(|| CachePerformanceError::SessionNotFound {
                session_id: session_id.to_string(),
                app_name: app_name.to_string(),
                user_id: user_id.to_string(),
            })
    }

    /// `_get_agent_cache_history` — cache usage history for an agent, in
    /// chronological order. `agent_name: None` returns every cache event
    /// regardless of author.
    async fn get_agent_cache_history(
        &self,
        session_id: &str,
        user_id: &str,
        app_name: &str,
        agent_name: Option<&str>,
    ) -> Result<Vec<CacheMetadata>, CachePerformanceError> {
        let session = self.session_or_error(session_id, user_id, app_name).await?;

        let mut history = Vec::new();
        for event in &session.events {
            let Some(raw) = &event.cache_metadata else {
                continue;
            };
            if agent_name.is_some_and(|name| event.author != name) {
                continue;
            }
            let parsed: CacheMetadata = rusty_serde::json::from_value(raw.clone())
                .map_err(|e| CachePerformanceError::InvalidCacheMetadata(e.to_string()))?;
            history.push(parsed);
        }
        Ok(history)
    }

    /// `analyze_agent_cache_performance` — analyzes cache performance for
    /// `agent_name` within the given session.
    pub async fn analyze_agent_cache_performance(
        &self,
        session_id: &str,
        user_id: &str,
        app_name: &str,
        agent_name: &str,
    ) -> Result<CachePerformanceReport, CachePerformanceError> {
        let cache_history = self
            .get_agent_cache_history(session_id, user_id, app_name, Some(agent_name))
            .await?;
        if cache_history.is_empty() {
            return Ok(CachePerformanceReport::NoCacheData);
        }

        let session = self.session_or_error(session_id, user_id, app_name).await?;

        let mut total_prompt_tokens = 0i64;
        let mut total_cached_tokens = 0i64;
        let mut requests_with_cache_hits = 0usize;
        let mut total_requests = 0usize;

        for event in &session.events {
            if event.author != agent_name {
                continue;
            }
            let Some(usage) = &event.usage_metadata else {
                continue;
            };
            total_requests += 1;
            if let Some(prompt) = usage.get("promptTokenCount").and_then(Value::as_i64) {
                if prompt != 0 {
                    total_prompt_tokens += prompt;
                }
            }
            if let Some(cached) = usage.get("cachedContentTokenCount").and_then(Value::as_i64) {
                if cached != 0 {
                    total_cached_tokens += cached;
                    requests_with_cache_hits += 1;
                }
            }
        }

        let cache_hit_ratio_percent = if total_prompt_tokens > 0 {
            (total_cached_tokens as f64 / total_prompt_tokens as f64) * 100.0
        } else {
            0.0
        };
        let cache_utilization_ratio_percent = if total_requests > 0 {
            (requests_with_cache_hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        let avg_cached_tokens_per_request = if total_requests > 0 {
            total_cached_tokens as f64 / total_requests as f64
        } else {
            0.0
        };

        let invocations_used: Vec<u32> = cache_history
            .iter()
            .filter_map(|c| c.invocations_used)
            .collect();
        let total_invocations: u64 = invocations_used.iter().map(|&v| u64::from(v)).sum();
        let avg_invocations_used = if invocations_used.is_empty() {
            0.0
        } else {
            total_invocations as f64 / invocations_used.len() as f64
        };

        let latest_cache = cache_history.last().and_then(|c| c.cache_name.clone());
        let cache_refreshes = cache_history
            .iter()
            .filter_map(|c| c.cache_name.as_deref())
            .collect::<HashSet<_>>()
            .len();

        Ok(CachePerformanceReport::Active(CachePerformanceStats {
            requests_with_cache: cache_history.len(),
            avg_invocations_used,
            latest_cache,
            cache_refreshes,
            total_invocations,
            total_prompt_tokens,
            total_cached_tokens,
            cache_hit_ratio_percent,
            cache_utilization_ratio_percent,
            avg_cached_tokens_per_request,
            total_requests,
            requests_with_cache_hits,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::services::BoxFuture;
    use adk_agents::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_events::Event;
    use std::collections::BTreeMap;

    struct FakeSessionService {
        session: Option<Session>,
    }

    impl SessionService for FakeSessionService {
        fn create_session<'a>(
            &'a self,
            _app_name: &'a str,
            _user_id: &'a str,
            _state: Option<BTreeMap<String, Value>>,
            _session_id: Option<String>,
        ) -> BoxFuture<'a, Result<Session, adk_errors::already_exists::AlreadyExistsError>>
        {
            unimplemented!("not needed for these tests")
        }

        fn get_session<'a>(
            &'a self,
            _app_name: &'a str,
            _user_id: &'a str,
            _session_id: &'a str,
        ) -> BoxFuture<'a, Option<Session>> {
            let session = self.session.clone();
            Box::pin(async move { session })
        }

        fn list_sessions<'a>(
            &'a self,
            _app_name: &'a str,
            _user_id: Option<&'a str>,
        ) -> BoxFuture<'a, Vec<Session>> {
            Box::pin(async { Vec::new() })
        }

        fn delete_session<'a>(
            &'a self,
            _app_name: &'a str,
            _user_id: &'a str,
            _session_id: &'a str,
        ) -> BoxFuture<'a, ()> {
            Box::pin(async {})
        }
    }

    fn cache_event(author: &str, cache_name: &str, invocations_used: u32) -> Event {
        let mut event = Event::new("inv-1", author, NodeInfo::new("root"));
        let metadata = CacheMetadata {
            cache_name: Some(cache_name.to_string()),
            expire_time: Some(0.0),
            fingerprint: "fp".to_string(),
            invocations_used: Some(invocations_used),
            contents_count: 1,
            created_at: None,
        };
        event.cache_metadata = Some(rusty_serde::json::to_value(&metadata).unwrap());
        event
    }

    fn usage_event(author: &str, prompt_tokens: i64, cached_tokens: i64) -> Event {
        let mut event = Event::new("inv-1", author, NodeInfo::new("root"));
        event.usage_metadata = Some(Value::Map(vec![
            ("promptTokenCount".to_string(), Value::Int(prompt_tokens)),
            (
                "cachedContentTokenCount".to_string(),
                Value::Int(cached_tokens),
            ),
        ]));
        event
    }

    #[rusty_tokio::test]
    async fn analyze_agent_cache_performance_reports_no_cache_data_without_history() {
        let session = Session::new("app", "user", "s1");
        let service = FakeSessionService {
            session: Some(session),
        };
        let analyzer = CachePerformanceAnalyzer::new(&service);
        let report = analyzer
            .analyze_agent_cache_performance("s1", "user", "app", "agent")
            .await
            .unwrap();
        assert_eq!(report, CachePerformanceReport::NoCacheData);
    }

    #[rusty_tokio::test]
    async fn analyze_agent_cache_performance_errors_for_a_missing_session() {
        let service = FakeSessionService { session: None };
        let analyzer = CachePerformanceAnalyzer::new(&service);
        let result = analyzer
            .analyze_agent_cache_performance("missing", "user", "app", "agent")
            .await;
        assert!(matches!(
            result,
            Err(CachePerformanceError::SessionNotFound { .. })
        ));
    }

    #[rusty_tokio::test]
    async fn analyze_agent_cache_performance_aggregates_token_and_invocation_metrics() {
        let mut session = Session::new("app", "user", "s1");
        session.events = vec![
            cache_event("agent", "projects/p/cachedContents/1", 2),
            usage_event("agent", 100, 40),
            usage_event("agent", 100, 0),
            cache_event("agent", "projects/p/cachedContents/2", 5),
        ];
        let service = FakeSessionService {
            session: Some(session),
        };
        let analyzer = CachePerformanceAnalyzer::new(&service);
        let report = analyzer
            .analyze_agent_cache_performance("s1", "user", "app", "agent")
            .await
            .unwrap();

        let CachePerformanceReport::Active(stats) = report else {
            panic!("expected an active report");
        };
        assert_eq!(stats.requests_with_cache, 2);
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.requests_with_cache_hits, 1);
        assert_eq!(stats.total_prompt_tokens, 200);
        assert_eq!(stats.total_cached_tokens, 40);
        assert_eq!(stats.cache_hit_ratio_percent, 20.0);
        assert_eq!(stats.cache_utilization_ratio_percent, 50.0);
        assert_eq!(stats.avg_cached_tokens_per_request, 20.0);
        assert_eq!(
            stats.latest_cache.as_deref(),
            Some("projects/p/cachedContents/2")
        );
        assert_eq!(stats.cache_refreshes, 2);
        assert_eq!(stats.total_invocations, 7);
        assert_eq!(stats.avg_invocations_used, 3.5);
    }

    #[rusty_tokio::test]
    async fn analyze_agent_cache_performance_ignores_other_agents_events() {
        let mut session = Session::new("app", "user", "s1");
        session.events = vec![
            cache_event("agent", "projects/p/cachedContents/1", 1),
            usage_event("other-agent", 500, 500),
        ];
        let service = FakeSessionService {
            session: Some(session),
        };
        let analyzer = CachePerformanceAnalyzer::new(&service);
        let report = analyzer
            .analyze_agent_cache_performance("s1", "user", "app", "agent")
            .await
            .unwrap();

        let CachePerformanceReport::Active(stats) = report else {
            panic!("expected an active report");
        };
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_prompt_tokens, 0);
    }
}
