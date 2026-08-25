//! Capability C0679: `telemetry._schema_version`, ported from
//! `google.adk.telemetry._schema_version` — the opt-in for the ADK
//! telemetry schema version.
//!
//! `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN` lets a deployment pin which
//! version of the ADK telemetry format (span names, span/log
//! attributes, metrics) it emits — a staged-migration knob, expected to
//! be phased out once ADK is fully OTel-semconv compliant.
//!
//! **C0671, bundled here**: the Agent-Engine env-var name constants
//! (`GOOGLE_CLOUD_AGENT_ENGINE_{ID,LOCATION,RUNTIME_REVISION_ID,
//! ENABLE_TELEMETRY}`, the metrics-collection-interval floor) are
//! declared here as plain constants — `GOOGLE_CLOUD_AGENT_ENGINE_ID` is
//! this module's own `resolve_schema_version` detection signal, and the
//! other four have no consumer in this port yet (the OTel-SDK/FastAPI-
//! middleware/GCP-exporter machinery in the source's `_agent_engine.py`/
//! `_agent_engine_metric_exporter.py`/`google_cloud.py` is a much larger,
//! still-unported surface — this row is scoped to just the env-var
//! *names* themselves, per its own manifest description). Declaring them
//! now means the next batch that needs one doesn't have to rediscover
//! the exact string.
//!
//! **C0672/C0673, bundled here for the same reason**: the GCP exporter's
//! `GOOGLE_API_USE_MTLS_ENDPOINT`/`GOOGLE_API_USE_CLIENT_CERTIFICATE`/
//! `GOOGLE_CLOUD_DEFAULT_LOG_NAME`(+ its `"adk-otel"` default)/
//! `GCP_DEFAULT_LOG_NAME`(+ its distinct `"adk-on-agent-engine"`
//! default) from `telemetry/google_cloud.py`, and the generic OTel
//! exporter's `OTEL_EXPORTER_OTLP_ENDPOINT`(+`_TRACES`/`_METRICS`/
//! `_LOGS_ENDPOINT` variants)/`OTEL_METRIC_EXPORT_INTERVAL`(+ its
//! `60000` ms default)/`OTEL_METRIC_EXPORT_TIMEOUT`(+ its `30000` ms
//! default) from `telemetry/setup.py`/`_agent_engine_metric_exporter.py`
//! — again just names/defaults, no consumer yet, same as C0671.

use std::env;

/// Env var users set to pin the ADK telemetry schema version (`"1"` or
/// `"2"`).
pub const ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN: &str = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN";

/// Presence of this env var indicates the process runs on Vertex Agent
/// Engine.
pub const GOOGLE_CLOUD_AGENT_ENGINE_ID: &str = "GOOGLE_CLOUD_AGENT_ENGINE_ID";

/// C0671: Agent-Engine env var name constants — no consumer in this
/// port yet, see the module doc.
pub const GOOGLE_CLOUD_AGENT_ENGINE_LOCATION: &str = "GOOGLE_CLOUD_AGENT_ENGINE_LOCATION";
pub const GOOGLE_CLOUD_AGENT_ENGINE_RUNTIME_REVISION_ID: &str =
    "GOOGLE_CLOUD_AGENT_ENGINE_RUNTIME_REVISION_ID";
pub const GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY: &str =
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY";
pub const GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS: &str =
    "GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS";
/// The default floor value (milliseconds) when that env var is unset —
/// the source's `MIN_EXPORT_INTERVAL_MS`.
pub const GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS_DEFAULT: f64 = 5000.0;

/// C0672: GCP exporter env var name constants — no consumer in this
/// port yet, see the module doc.
pub const GOOGLE_API_USE_MTLS_ENDPOINT: &str = "GOOGLE_API_USE_MTLS_ENDPOINT";
pub const GOOGLE_API_USE_CLIENT_CERTIFICATE: &str = "GOOGLE_API_USE_CLIENT_CERTIFICATE";
pub const GOOGLE_CLOUD_DEFAULT_LOG_NAME: &str = "GOOGLE_CLOUD_DEFAULT_LOG_NAME";
/// Default log name for the `GOOGLE_CLOUD_DEFAULT_LOG_NAME`-driven
/// exporter path — the source's `_DEFAULT_LOG_NAME`.
pub const GOOGLE_CLOUD_DEFAULT_LOG_NAME_DEFAULT: &str = "adk-otel";
pub const GCP_DEFAULT_LOG_NAME: &str = "GCP_DEFAULT_LOG_NAME";
/// Default log name for the Agent-Engine-specific exporter path — a
/// deliberately different default than [`GOOGLE_CLOUD_DEFAULT_LOG_NAME_DEFAULT`].
pub const GCP_DEFAULT_LOG_NAME_DEFAULT: &str = "adk-on-agent-engine";

/// C0673: generic OTel exporter env var name constants — no consumer in
/// this port yet, see the module doc.
pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
pub const OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";
pub const OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
pub const OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT";
pub const OTEL_METRIC_EXPORT_INTERVAL: &str = "OTEL_METRIC_EXPORT_INTERVAL";
/// Default export interval (milliseconds) when that env var is unset.
pub const OTEL_METRIC_EXPORT_INTERVAL_DEFAULT_MS: f64 = 60000.0;
pub const OTEL_METRIC_EXPORT_TIMEOUT: &str = "OTEL_METRIC_EXPORT_TIMEOUT";
/// Default export timeout (milliseconds) when that env var is unset.
pub const OTEL_METRIC_EXPORT_TIMEOUT_DEFAULT_MS: f64 = 30000.0;

/// Legacy telemetry format: top-level `invocation` span, no entrypoint
/// `invoke_workflow` span/metric.
pub const SCHEMA_VERSION_LEGACY: u32 = 1;

/// OTel-semconv-aligned telemetry format: the `invocation` span is
/// replaced by an entrypoint `invoke_workflow {entrypoint}` span +
/// duration metric.
pub const SCHEMA_VERSION_SEMCONV_ALIGNED: u32 = 2;

/// C0679: `resolve_schema_version` — resolves the active ADK telemetry
/// schema version.
///
/// Precedence: `ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN` (if set to a
/// recognized value) > `2` on Agent Engine > `1`.
pub fn resolve_schema_version() -> u32 {
    let opt_in = env::var(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN).unwrap_or_default();
    match opt_in.trim() {
        "1" => return SCHEMA_VERSION_LEGACY,
        "2" => return SCHEMA_VERSION_SEMCONV_ALIGNED,
        _ => {}
    }

    if env::var(GOOGLE_CLOUD_AGENT_ENGINE_ID).is_ok_and(|v| !v.is_empty()) {
        return SCHEMA_VERSION_SEMCONV_ALIGNED;
    }
    SCHEMA_VERSION_LEGACY
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        unsafe {
            env::remove_var(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN);
            env::remove_var(GOOGLE_CLOUD_AGENT_ENGINE_ID);
        }
    }

    #[test]
    fn defaults_to_legacy_off_agent_engine() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(resolve_schema_version(), SCHEMA_VERSION_LEGACY);
    }

    #[test]
    fn defaults_to_semconv_aligned_on_agent_engine() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(GOOGLE_CLOUD_AGENT_ENGINE_ID, "some-engine-id");
        }
        assert_eq!(resolve_schema_version(), SCHEMA_VERSION_SEMCONV_ALIGNED);
        clear_env();
    }

    #[test]
    fn explicit_opt_in_1_wins_even_on_agent_engine() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(GOOGLE_CLOUD_AGENT_ENGINE_ID, "some-engine-id");
            env::set_var(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, "1");
        }
        assert_eq!(resolve_schema_version(), SCHEMA_VERSION_LEGACY);
        clear_env();
    }

    #[test]
    fn explicit_opt_in_2_wins_off_agent_engine() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, "2");
        }
        assert_eq!(resolve_schema_version(), SCHEMA_VERSION_SEMCONV_ALIGNED);
        clear_env();
    }

    #[test]
    fn unrecognized_opt_in_falls_back_to_agent_engine_detection() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_env();
        unsafe {
            env::set_var(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, "bogus");
        }
        assert_eq!(resolve_schema_version(), SCHEMA_VERSION_LEGACY);
        clear_env();
    }

    #[test]
    fn gcp_log_name_defaults_differ_by_exporter_path() {
        assert_ne!(
            GOOGLE_CLOUD_DEFAULT_LOG_NAME_DEFAULT,
            GCP_DEFAULT_LOG_NAME_DEFAULT
        );
        assert_eq!(GOOGLE_CLOUD_DEFAULT_LOG_NAME_DEFAULT, "adk-otel");
        assert_eq!(GCP_DEFAULT_LOG_NAME_DEFAULT, "adk-on-agent-engine");
    }
}
