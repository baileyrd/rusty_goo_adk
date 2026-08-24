//! Capabilities C0651/C0652/C0670: `telemetry.context`, ported from
//! `google.adk.telemetry.context` — per-request OpenTelemetry
//! configuration.
//!
//! [`TelemetryConfig`] (attached to `RunConfig.telemetry`) is the single
//! source of truth for how each telemetry knob resolves. Its
//! `resolved_*`/`should_*` methods own the precedence ladder (admin lock,
//! then per-request field, then the `OTEL_*` env var, then a default) —
//! the same env-var-precedence-ladder shape `adk_features::feature_registry`
//! already established for feature-flag resolution
//! (`is_env_enabled`/`override_feature_enabled`), applied here to
//! telemetry knobs instead of feature flags.
//!
//! **`RunConfig.telemetry`, now a real type**: was a bare `Option<Value>`
//! placeholder in `run_config.rs` (explicitly marked "P12 placeholder")
//! — this batch widens it to `Option<TelemetryConfig>`, the same
//! "widen a placeholder once a real consumer needs the structure"
//! precedent as `EventCompaction.compacted_content`/`services::
//! {MemoryEntry, SearchMemoryResponse}`.
//!
//! **Frozen/shared config, adapted**: the source's `frozen=True` lets
//! one `TelemetryConfig` be shared safely across concurrent invocations
//! (Python immutability). This port's `TelemetryConfig` has no interior
//! mutability and no setters — the same effect via ordinary Rust
//! immutable-by-default semantics, no `frozen` marker needed.
//!
//! **No OTel SDK/span/tracer machinery here, disclosed**: this module
//! ports the *resolution logic* only — what a caller's `RunConfig`
//! carries and how it resolves against env vars. The consumers of these
//! resolved values (`_experimental_semconv`/`tracing`'s span/attribute
//! emission) need a real OTel SDK integration this port doesn't have
//! yet; that's its own much larger, still-unported surface.

use std::env;

use rusty_serde::{Deserialize, Serialize};

/// `telemetry.context.ADK_TELEMETRY_IGNORE_RUN_CONFIG` and its siblings —
/// C0670's env-var name constants.
pub const ADK_TELEMETRY_IGNORE_RUN_CONFIG: &str = "ADK_TELEMETRY_IGNORE_RUN_CONFIG";
pub const OTEL_SEMCONV_STABILITY_OPT_IN: &str = "OTEL_SEMCONV_STABILITY_OPT_IN";
pub const OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: &str =
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";
/// Legacy ADK span-content knob; unlike the OTel env var above, it
/// defaults on.
pub const ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS: &str = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS";
pub const ADK_EXPERIMENTAL_TELEMETRY: &str = "ADK_EXPERIMENTAL_TELEMETRY";

/// Token in `OTEL_SEMCONV_STABILITY_OPT_IN` that selects experimental
/// GenAI semconv.
const GENAI_EXPERIMENTAL_OPT_IN: &str = "gen_ai_latest_experimental";

fn is_truthy_env(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "1" | "true")
}

fn is_falsy_env(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "0" | "false")
}

/// `telemetry.context.ContentCapturingMode` — mirror of
/// `opentelemetry.util.genai.types.ContentCapturingMode`, defined
/// locally rather than imported since `opentelemetry-util-genai` is an
/// optional, in-development dependency neither the source nor this port
/// takes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentCapturingMode {
    /// No content captured (matches env value `""`).
    #[rusty_serde(rename = "NO_CONTENT")]
    NoContent,
    /// Content on the emitted LogRecord only.
    #[rusty_serde(rename = "EVENT_ONLY")]
    EventOnly,
    /// Content on the active span only.
    #[rusty_serde(rename = "SPAN_ONLY")]
    SpanOnly,
    /// Content on both the LogRecord and the active span.
    #[rusty_serde(rename = "SPAN_AND_EVENT")]
    SpanAndEvent,
}

impl ContentCapturingMode {
    /// The canonical uppercase string this mode's env-var value uses
    /// (`resolved_content_capturing_mode`'s `ContentCapturingMode(stripped
    /// .upper())` parse target).
    fn from_env_value(value: &str) -> Option<Self> {
        match value {
            "NO_CONTENT" => Some(ContentCapturingMode::NoContent),
            "EVENT_ONLY" => Some(ContentCapturingMode::EventOnly),
            "SPAN_ONLY" => Some(ContentCapturingMode::SpanOnly),
            "SPAN_AND_EVENT" => Some(ContentCapturingMode::SpanAndEvent),
            _ => None,
        }
    }

    /// `content_capturing_mode_value` — `""` for `NoContent`, the
    /// member's canonical string otherwise.
    pub fn value(&self) -> &'static str {
        match self {
            ContentCapturingMode::NoContent => "",
            ContentCapturingMode::EventOnly => "EVENT_ONLY",
            ContentCapturingMode::SpanOnly => "SPAN_ONLY",
            ContentCapturingMode::SpanAndEvent => "SPAN_AND_EVENT",
        }
    }

    /// `_is_span_bearing` — whether this mode routes content onto the
    /// span (`SpanOnly`/`SpanAndEvent`).
    fn is_span_bearing(&self) -> bool {
        matches!(
            self,
            ContentCapturingMode::SpanOnly | ContentCapturingMode::SpanAndEvent
        )
    }
}

/// `genai_semconv_stability_opt_in`'s `Literal['stable', 'experimental']`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "lowercase")]
pub enum SemconvStabilityOptIn {
    Stable,
    Experimental,
}

/// `telemetry.context.TelemetryConfig` — per-request OpenTelemetry
/// configuration, attached to an invocation via `RunConfig.telemetry`.
/// Any field left `None` falls back to its corresponding env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// Override for `OTEL_SEMCONV_STABILITY_OPT_IN`.
    #[rusty_serde(default)]
    pub genai_semconv_stability_opt_in: Option<SemconvStabilityOptIn>,
    /// Override for `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
    #[rusty_serde(default)]
    pub capture_message_content: Option<ContentCapturingMode>,
    /// Override for `ADK_EXPERIMENTAL_TELEMETRY`.
    #[rusty_serde(default)]
    pub adk_experimental_telemetry_opt_in: Option<bool>,
}

impl TelemetryConfig {
    /// Whether the admin lock (`ADK_TELEMETRY_IGNORE_RUN_CONFIG`) is
    /// set — when set, per-request fields are ignored and resolution
    /// falls back to the `OTEL_*` env vars.
    fn ignore_per_request(&self) -> bool {
        env::var(ADK_TELEMETRY_IGNORE_RUN_CONFIG)
            .map(|v| is_truthy_env(&v))
            .unwrap_or(false)
    }

    /// Whether to emit experimental GenAI semconv attributes.
    ///
    /// Precedence: admin lock, then `genai_semconv_stability_opt_in`, then
    /// the `OTEL_SEMCONV_STABILITY_OPT_IN` env var, then `false`.
    pub fn should_use_experimental_genai_semconv(&self) -> bool {
        if !self.ignore_per_request() {
            if let Some(opt_in) = self.genai_semconv_stability_opt_in {
                return opt_in == SemconvStabilityOptIn::Experimental;
            }
        }
        let Ok(opt_ins) = env::var(OTEL_SEMCONV_STABILITY_OPT_IN) else {
            return false;
        };
        if opt_ins.is_empty() {
            return false;
        }
        opt_ins
            .split(',')
            .any(|token| token.trim() == GENAI_EXPERIMENTAL_OPT_IN)
    }

    /// The effective GenAI content-capturing mode.
    ///
    /// Precedence: admin lock, then `capture_message_content`, then the
    /// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var
    /// (legacy `"true"`/`"1"` coerce to `EventOnly`), then `NoContent`.
    /// Env values outside the four-state set fall back to `NoContent`.
    pub fn resolved_content_capturing_mode(&self) -> ContentCapturingMode {
        if !self.ignore_per_request() {
            if let Some(mode) = self.capture_message_content {
                return mode;
            }
        }
        let stripped =
            env::var(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT).unwrap_or_default();
        let stripped = stripped.trim();
        if is_truthy_env(stripped) {
            return ContentCapturingMode::EventOnly;
        }
        ContentCapturingMode::from_env_value(&stripped.to_uppercase())
            .unwrap_or(ContentCapturingMode::NoContent)
    }

    /// `resolved_content_capturing_mode` as the canonical string —
    /// `""` for `NoContent`, the member's value otherwise.
    pub fn content_capturing_mode_value(&self) -> &'static str {
        self.resolved_content_capturing_mode().value()
    }

    /// Whether content goes on emitted LogRecords (`EventOnly`/
    /// `SpanAndEvent`).
    pub fn should_add_content_to_logs(&self) -> bool {
        matches!(
            self.resolved_content_capturing_mode(),
            ContentCapturingMode::EventOnly | ContentCapturingMode::SpanAndEvent
        )
    }

    /// Whether content goes on the experimental inference span
    /// (OTel-spec routing: the span-bearing modes).
    pub fn should_add_content_to_experimental_spans(&self) -> bool {
        self.resolved_content_capturing_mode().is_span_bearing()
    }

    /// Whether content goes on ADK-owned (legacy) spans. Separate knob
    /// from the OTel content env var: a per-request
    /// `capture_message_content` uses the OTel-spec span routing;
    /// otherwise this falls back to `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`,
    /// which defaults on.
    pub fn should_add_content_to_legacy_spans(&self) -> bool {
        if !self.ignore_per_request() {
            if let Some(mode) = self.capture_message_content {
                return mode.is_span_bearing();
            }
        }
        let env_value =
            env::var(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS).unwrap_or_else(|_| "true".to_string());
        !is_falsy_env(env_value.trim())
    }

    /// Whether to emit experimental telemetry.
    ///
    /// Precedence: admin lock, then `adk_experimental_telemetry_opt_in`,
    /// then the `ADK_EXPERIMENTAL_TELEMETRY` env var, then `false`.
    pub fn should_emit_experimental_telemetry(&self) -> bool {
        if !self.ignore_per_request() {
            if let Some(opt_in) = self.adk_experimental_telemetry_opt_in {
                return opt_in;
            }
        }
        let env_value =
            env::var(ADK_EXPERIMENTAL_TELEMETRY).unwrap_or_else(|_| "false".to_string());
        is_truthy_env(env_value.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Environment variables are process-global state; serialize the
    // tests that touch them so they don't race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        for var in [
            ADK_TELEMETRY_IGNORE_RUN_CONFIG,
            OTEL_SEMCONV_STABILITY_OPT_IN,
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
            ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS,
            ADK_EXPERIMENTAL_TELEMETRY,
        ] {
            unsafe {
                env::remove_var(var);
            }
        }
    }

    #[test]
    fn defaults_to_stable_semconv_with_no_config_or_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        assert!(!TelemetryConfig::default().should_use_experimental_genai_semconv());
    }

    #[test]
    fn per_request_opt_in_selects_experimental_semconv() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = TelemetryConfig {
            genai_semconv_stability_opt_in: Some(SemconvStabilityOptIn::Experimental),
            ..Default::default()
        };
        assert!(config.should_use_experimental_genai_semconv());
    }

    #[test]
    fn env_var_selects_experimental_semconv_absent_per_request_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(
                OTEL_SEMCONV_STABILITY_OPT_IN,
                "foo,gen_ai_latest_experimental",
            );
        }
        assert!(TelemetryConfig::default().should_use_experimental_genai_semconv());
        clear_env();
    }

    #[test]
    fn admin_lock_ignores_the_per_request_semconv_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_TELEMETRY_IGNORE_RUN_CONFIG, "true");
        }
        let config = TelemetryConfig {
            genai_semconv_stability_opt_in: Some(SemconvStabilityOptIn::Experimental),
            ..Default::default()
        };
        assert!(!config.should_use_experimental_genai_semconv());
        clear_env();
    }

    #[test]
    fn content_capturing_mode_defaults_to_no_content() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(
            TelemetryConfig::default().resolved_content_capturing_mode(),
            ContentCapturingMode::NoContent
        );
        assert_eq!(
            TelemetryConfig::default().content_capturing_mode_value(),
            ""
        );
    }

    #[test]
    fn content_capturing_mode_per_request_override_wins() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        let config = TelemetryConfig {
            capture_message_content: Some(ContentCapturingMode::SpanOnly),
            ..Default::default()
        };
        assert_eq!(
            config.resolved_content_capturing_mode(),
            ContentCapturingMode::SpanOnly
        );
    }

    #[test]
    fn content_capturing_mode_legacy_truthy_env_coerces_to_event_only() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "1");
        }
        assert_eq!(
            TelemetryConfig::default().resolved_content_capturing_mode(),
            ContentCapturingMode::EventOnly
        );
        clear_env();
    }

    #[test]
    fn content_capturing_mode_reads_the_env_vars_four_state_value() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(
                OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
                "span_and_event",
            );
        }
        assert_eq!(
            TelemetryConfig::default().resolved_content_capturing_mode(),
            ContentCapturingMode::SpanAndEvent
        );
        clear_env();
    }

    #[test]
    fn content_capturing_mode_falls_back_to_no_content_for_unrecognized_env_value() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "bogus");
        }
        assert_eq!(
            TelemetryConfig::default().resolved_content_capturing_mode(),
            ContentCapturingMode::NoContent
        );
        clear_env();
    }

    #[test]
    fn should_add_content_to_logs_true_for_event_only_and_span_and_event() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        let event_only = TelemetryConfig {
            capture_message_content: Some(ContentCapturingMode::EventOnly),
            ..Default::default()
        };
        let span_only = TelemetryConfig {
            capture_message_content: Some(ContentCapturingMode::SpanOnly),
            ..Default::default()
        };
        assert!(event_only.should_add_content_to_logs());
        assert!(!span_only.should_add_content_to_logs());
    }

    #[test]
    fn should_add_content_to_experimental_spans_true_for_span_bearing_modes() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        let span_and_event = TelemetryConfig {
            capture_message_content: Some(ContentCapturingMode::SpanAndEvent),
            ..Default::default()
        };
        assert!(span_and_event.should_add_content_to_experimental_spans());
    }

    #[test]
    fn should_add_content_to_legacy_spans_defaults_on() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        assert!(TelemetryConfig::default().should_add_content_to_legacy_spans());
    }

    #[test]
    fn should_add_content_to_legacy_spans_env_var_can_turn_it_off() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, "false");
        }
        assert!(!TelemetryConfig::default().should_add_content_to_legacy_spans());
        clear_env();
    }

    #[test]
    fn should_add_content_to_legacy_spans_uses_per_request_span_routing() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, "false");
        }
        let config = TelemetryConfig {
            capture_message_content: Some(ContentCapturingMode::SpanOnly),
            ..Default::default()
        };
        assert!(config.should_add_content_to_legacy_spans());
        clear_env();
    }

    #[test]
    fn should_emit_experimental_telemetry_defaults_off() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        assert!(!TelemetryConfig::default().should_emit_experimental_telemetry());
    }

    #[test]
    fn should_emit_experimental_telemetry_per_request_override_wins() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_EXPERIMENTAL_TELEMETRY, "true");
        }
        let config = TelemetryConfig {
            adk_experimental_telemetry_opt_in: Some(false),
            ..Default::default()
        };
        assert!(!config.should_emit_experimental_telemetry());
        clear_env();
    }

    #[test]
    fn should_emit_experimental_telemetry_env_var_turns_it_on() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_EXPERIMENTAL_TELEMETRY, "true");
        }
        assert!(TelemetryConfig::default().should_emit_experimental_telemetry());
        clear_env();
    }

    #[test]
    fn admin_lock_ignores_the_per_request_experimental_telemetry_override() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_TELEMETRY_IGNORE_RUN_CONFIG, "true");
        }
        let config = TelemetryConfig {
            adk_experimental_telemetry_opt_in: Some(true),
            ..Default::default()
        };
        assert!(!config.should_emit_experimental_telemetry());
        clear_env();
    }
}
