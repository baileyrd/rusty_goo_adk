//! Capabilities C0283-C0285: `apps._configs`/`apps.base_events_summarizer`
//! — [`ResumabilityConfig`], [`EventsCompactionConfig`], and the
//! [`BaseEventsSummarizer`] extension point.
//!
//! **Placement, disclosed**: the source's `apps/` is a distinct top-level
//! module from `agents/`, but no `adk-apps` crate exists in this port —
//! these three types have exactly one consumer so far
//! (`InvocationContext::resumability_config`/`::events_compaction_config`,
//! already in this crate), so they land here directly rather than
//! standing up a new crate for three types. Same reasoning already
//! applied to `context_cache_config` living alongside its own consumer.
//!
//! **`EventsCompactionConfig`, adapted**: the source's `summarizer:
//! Optional[BaseEventsSummarizer]` is an arbitrary (non-pydantic) object
//! field — the source's own `model_config = ConfigDict(arbitrary_types_allowed
//! =True)` exists for exactly this reason. This port models it as
//! `Option<Arc<dyn BaseEventsSummarizer>>`, which can't derive
//! `Serialize`/`Deserialize` (a trait object has no wire representation) —
//! this struct has no such derive at all, matching the source's own
//! "arbitrary, not a serializable config value" treatment of this field.
//! `Debug`/`Clone` are implemented by hand instead (`Clone` is free —
//! `Arc<dyn _>` clones the handle; `Debug` prints an opaque placeholder for
//! `summarizer` rather than requiring `BaseEventsSummarizer: Debug`).
//!
//! **`@experimental`, wired**: both config types carry the source's bare
//! `@experimental` decorator (`utils.feature_decorator.experimental`, no
//! `FeatureName` — the *other* experimental-marking mechanism this port
//! already ported as `adk_features::legacy_feature_decorator::warn_experimental`,
//! C0797, landed but unwired until now). `EventsCompactionConfig::validate`
//! is the natural call site (it already exists for the trigger-pair
//! checks); `ResumabilityConfig` has no other validation, so
//! [`ResumabilityConfig::new`] exists solely to give the warning a
//! construction-equivalent hook to fire from, mirroring the source's
//! decorator wrapping `__init__`.
//!
//! **`Field(gt=0)`/`Field(ge=0)`, ported**: pydantic enforces these at
//! construction, before the source's own `_validate_trigger_params`
//! model-validator runs; this port checks them first inside
//! [`EventsCompactionConfig::validate`] too, same "plain fields +
//! explicit `validate()`" pattern used throughout this port.

use std::fmt;
use std::sync::Arc;

use adk_events::Event;
use adk_features::legacy_feature_decorator::warn_experimental;
use rusty_serde::{Deserialize, Serialize};

use crate::services::BoxFuture;

/// C0285: `apps.base_events_summarizer.BaseEventsSummarizer` — extension
/// point for compacting a list of events into a single summary event.
/// Returns `None` if compaction didn't happen.
pub trait BaseEventsSummarizer: Send + Sync {
    fn maybe_summarize_events<'a>(&'a self, events: &'a [Event]) -> BoxFuture<'a, Option<Event>>;
}

/// C0283: `apps._configs.ResumabilityConfig` — whether an app supports
/// pausing an invocation on a long-running function call and resuming it
/// from the last event. Best-effort: resumed tool calls need to be
/// idempotent (only at-least-once is guaranteed), and any temporary/
/// in-memory state is lost on resumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResumabilityConfig {
    #[rusty_serde(default)]
    pub is_resumable: bool,
}

impl ResumabilityConfig {
    /// Constructs the config, emitting the source's `@experimental`
    /// warning — see the module doc.
    pub fn new(is_resumable: bool) -> Self {
        warn_experimental("ResumabilityConfig", None);
        ResumabilityConfig { is_resumable }
    }
}

/// C0284: `apps._configs.EventsCompactionConfig` — event compaction
/// configuration for an application.
#[derive(Clone, Default)]
pub struct EventsCompactionConfig {
    pub summarizer: Option<Arc<dyn BaseEventsSummarizer>>,
    /// Sliding-window trigger: number of *new* user-initiated invocations
    /// that, once fully represented in the session's events, triggers a
    /// compaction. Must be set together with `overlap_size`. `> 0` if set.
    pub compaction_interval: Option<i64>,
    /// Number of preceding invocations to include from the end of the
    /// last compacted range, for overlap between consecutive summaries.
    /// Must be set together with `compaction_interval`. `>= 0` if set.
    pub overlap_size: Option<i64>,
    /// Post-invocation token-budget trigger: if the most recently observed
    /// prompt token count meets or exceeds this, a compaction is
    /// attempted. Must be set together with `event_retention_size`. `> 0`
    /// if set.
    pub token_threshold: Option<i64>,
    /// Post-invocation raw event retention size once a token-based
    /// compaction triggers. Must be set together with `token_threshold`.
    /// `>= 0` if set.
    pub event_retention_size: Option<i64>,
}

impl fmt::Debug for EventsCompactionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventsCompactionConfig")
            .field(
                "summarizer",
                &self.summarizer.as_ref().map(|_| "<BaseEventsSummarizer>"),
            )
            .field("compaction_interval", &self.compaction_interval)
            .field("overlap_size", &self.overlap_size)
            .field("token_threshold", &self.token_threshold)
            .field("event_retention_size", &self.event_retention_size)
            .finish()
    }
}

/// The source's `ValueError` from `_validate_trigger_params` (and the
/// `Field(gt=0)`/`Field(ge=0)` constraints checked ahead of it here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventsCompactionConfigError(pub String);

impl fmt::Display for EventsCompactionConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EventsCompactionConfigError {}

impl EventsCompactionConfig {
    /// The source's `Field(gt=0)`/`Field(ge=0)` constraints plus
    /// `_validate_trigger_params` — see the module doc.
    pub fn validate(&self) -> Result<(), EventsCompactionConfigError> {
        warn_experimental("EventsCompactionConfig", None);

        if matches!(self.compaction_interval, Some(v) if v <= 0) {
            return Err(EventsCompactionConfigError(
                "compaction_interval must be > 0.".to_string(),
            ));
        }
        if matches!(self.overlap_size, Some(v) if v < 0) {
            return Err(EventsCompactionConfigError(
                "overlap_size must be >= 0.".to_string(),
            ));
        }
        if matches!(self.token_threshold, Some(v) if v <= 0) {
            return Err(EventsCompactionConfigError(
                "token_threshold must be > 0.".to_string(),
            ));
        }
        if matches!(self.event_retention_size, Some(v) if v < 0) {
            return Err(EventsCompactionConfigError(
                "event_retention_size must be >= 0.".to_string(),
            ));
        }

        let token_threshold_set = self.token_threshold.is_some();
        let retention_size_set = self.event_retention_size.is_some();
        if token_threshold_set != retention_size_set {
            return Err(EventsCompactionConfigError(
                "token_threshold and event_retention_size must be set together.".to_string(),
            ));
        }
        let compaction_interval_set = self.compaction_interval.is_some();
        let overlap_size_set = self.overlap_size.is_some();
        if compaction_interval_set != overlap_size_set {
            return Err(EventsCompactionConfigError(
                "compaction_interval and overlap_size must be set together.".to_string(),
            ));
        }
        if !(token_threshold_set || compaction_interval_set) {
            return Err(EventsCompactionConfigError(
                "At least one compaction trigger must be configured: the token-threshold pair \
                 or the sliding-window pair."
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resumability_config_defaults_to_false() {
        assert!(!ResumabilityConfig::default().is_resumable);
    }

    #[test]
    fn resumability_config_new_carries_the_given_value() {
        assert!(ResumabilityConfig::new(true).is_resumable);
    }

    fn sliding_window_config() -> EventsCompactionConfig {
        EventsCompactionConfig {
            compaction_interval: Some(5),
            overlap_size: Some(1),
            ..Default::default()
        }
    }

    fn token_budget_config() -> EventsCompactionConfig {
        EventsCompactionConfig {
            token_threshold: Some(1000),
            event_retention_size: Some(2),
            ..Default::default()
        }
    }

    #[test]
    fn rejects_no_trigger_configured() {
        assert!(EventsCompactionConfig::default().validate().is_err());
    }

    #[test]
    fn accepts_sliding_window_trigger_alone() {
        assert!(sliding_window_config().validate().is_ok());
    }

    #[test]
    fn accepts_token_budget_trigger_alone() {
        assert!(token_budget_config().validate().is_ok());
    }

    #[test]
    fn accepts_both_triggers_configured() {
        let config = EventsCompactionConfig {
            compaction_interval: Some(5),
            overlap_size: Some(1),
            token_threshold: Some(1000),
            event_retention_size: Some(2),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_compaction_interval_without_overlap_size() {
        let config = EventsCompactionConfig {
            compaction_interval: Some(5),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_overlap_size_without_compaction_interval() {
        let config = EventsCompactionConfig {
            overlap_size: Some(1),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_token_threshold_without_event_retention_size() {
        let config = EventsCompactionConfig {
            token_threshold: Some(1000),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_event_retention_size_without_token_threshold() {
        let config = EventsCompactionConfig {
            event_retention_size: Some(2),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_non_positive_compaction_interval() {
        let mut config = sliding_window_config();
        config.compaction_interval = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_negative_overlap_size() {
        let mut config = sliding_window_config();
        config.overlap_size = Some(-1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn accepts_zero_overlap_size() {
        let mut config = sliding_window_config();
        config.overlap_size = Some(0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_non_positive_token_threshold() {
        let mut config = token_budget_config();
        config.token_threshold = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_negative_event_retention_size() {
        let mut config = token_budget_config();
        config.event_retention_size = Some(-1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn accepts_zero_event_retention_size() {
        let mut config = token_budget_config();
        config.event_retention_size = Some(0);
        assert!(config.validate().is_ok());
    }

    struct StubSummarizer;

    impl BaseEventsSummarizer for StubSummarizer {
        fn maybe_summarize_events<'a>(
            &'a self,
            _events: &'a [Event],
        ) -> BoxFuture<'a, Option<Event>> {
            Box::pin(async { None })
        }
    }

    #[test]
    fn events_compaction_config_clones_a_shared_summarizer_handle() {
        let config = EventsCompactionConfig {
            summarizer: Some(Arc::new(StubSummarizer)),
            ..sliding_window_config()
        };
        let cloned = config.clone();
        assert!(cloned.summarizer.is_some());
        assert!(format!("{cloned:?}").contains("BaseEventsSummarizer"));
    }

    #[rusty_tokio::test]
    async fn base_events_summarizer_stub_returns_none() {
        let summarizer = StubSummarizer;
        let result = summarizer.maybe_summarize_events(&[]).await;
        assert!(result.is_none());
    }
}
