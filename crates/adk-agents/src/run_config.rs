//! Capabilities C0072-C0075: `RunConfig`/`ToolThreadPoolConfig`, ported from
//! `google.adk.agents.run_config`.
//!
//! **Adaptation**: every field typed `google.genai.types.X` in the source
//! (`SpeechConfig`, `HttpOptions`, `Modality`, `AvatarConfig`,
//! `AudioTranscriptionConfig`, `RealtimeInputConfig`, `TranslationConfig`,
//! `ProactivityConfig`, `SessionResumptionConfig`, `HistoryConfig`,
//! `ContextWindowCompressionConfig`, `Content`) is an opaque third-party
//! Gemini-API request/response shape, not an ADK capability of its own —
//! porting Google's entire Gemini SDK type system is out of scope for this
//! migration. Each is represented here as an untyped [`rusty_serde::value::Value`]
//! placeholder, preserving field presence/optionality/defaults (what parity
//! actually requires) without modeling the opaque schema.
//!
//! `GetSessionConfig` (P5) is likewise a placeholder pending its own
//! phase. `telemetry` was a `TelemetryConfig` (P12) placeholder until
//! C0651/C0652 landed a real type (`crate::telemetry_context
//! ::TelemetryConfig`) — see that module's own doc.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::env;

use crate::streaming_mode::StreamingMode;
use crate::telemetry_context::TelemetryConfig;

const DEFAULT_MAX_LLM_CALLS: i64 = 500;

fn default_max_llm_calls() -> i64 {
    if let Ok(env_val) = env::var("ADK_MAX_LLM_CALLS") {
        match env_val.parse::<i64>() {
            Ok(v) => return v,
            Err(_) => {
                eprintln!(
                    "Invalid value for ADK_MAX_LLM_CALLS env var: {env_val}. Using default {DEFAULT_MAX_LLM_CALLS}."
                );
            }
        }
    }
    DEFAULT_MAX_LLM_CALLS
}

/// Configuration for the tool thread pool executor (live-mode blocking
/// tools). Helps I/O-bound work (GIL released), not CPU-bound work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct ToolThreadPoolConfig {
    pub max_workers: u32,
}

impl Default for ToolThreadPoolConfig {
    fn default() -> Self {
        Self { max_workers: 4 }
    }
}

/// Runtime behavior config for agents. Overridden by agent-specific
/// configuration where applicable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct RunConfig {
    #[rusty_serde(default)]
    pub speech_config: Option<Value>,
    #[rusty_serde(default)]
    pub http_options: Option<Value>,
    #[rusty_serde(default)]
    pub labels: Option<std::collections::BTreeMap<String, String>>,
    #[rusty_serde(default)]
    pub response_modalities: Option<Vec<Value>>,
    #[rusty_serde(default)]
    pub avatar_config: Option<Value>,
    /// DEPRECATED: use a `SaveFilesAsArtifactsPlugin` instead.
    #[rusty_serde(default)]
    pub save_input_blobs_as_artifacts: bool,
    pub support_cfc: bool,
    pub streaming_mode: StreamingMode,
    #[rusty_serde(default)]
    pub output_audio_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub input_audio_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub realtime_input_config: Option<Value>,
    #[rusty_serde(default)]
    pub explicit_vad_signal: Option<bool>,
    #[rusty_serde(default)]
    pub translation_config: Option<Value>,
    #[rusty_serde(default)]
    pub enable_affective_dialog: Option<bool>,
    #[rusty_serde(default)]
    pub proactivity: Option<Value>,
    #[rusty_serde(default)]
    pub session_resumption: Option<Value>,
    #[rusty_serde(default)]
    pub history_config: Option<Value>,
    #[rusty_serde(default)]
    pub context_window_compression: Option<Value>,
    pub save_live_blob: bool,
    #[rusty_serde(default)]
    pub tool_thread_pool_config: Option<ToolThreadPoolConfig>,
    /// DEPRECATED: use `save_live_blob` instead.
    pub max_llm_calls: i64,
    #[rusty_serde(default)]
    pub custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// Per-request OpenTelemetry config override.
    #[rusty_serde(default)]
    pub telemetry: Option<TelemetryConfig>,
    /// Passed to the session service's `get_session` (P5 placeholder).
    #[rusty_serde(default)]
    pub get_session_config: Option<Value>,
    #[rusty_serde(default)]
    pub model_input_context: Option<Vec<Value>>,
    pub include_thoughts_from_other_agents: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            speech_config: None,
            http_options: None,
            labels: None,
            response_modalities: None,
            avatar_config: None,
            save_input_blobs_as_artifacts: false,
            support_cfc: false,
            streaming_mode: StreamingMode::None,
            output_audio_transcription: None,
            input_audio_transcription: None,
            realtime_input_config: None,
            explicit_vad_signal: None,
            translation_config: None,
            enable_affective_dialog: None,
            proactivity: None,
            session_resumption: None,
            history_config: None,
            context_window_compression: None,
            save_live_blob: false,
            tool_thread_pool_config: None,
            max_llm_calls: default_max_llm_calls(),
            custom_metadata: None,
            telemetry: None,
            get_session_config: None,
            model_input_context: None,
            include_thoughts_from_other_agents: false,
        }
    }
}

impl RunConfig {
    /// Mirrors the source's `save_live_audio` (deprecated) -> `save_live_blob`
    /// migration: a `RunConfig` built with the deprecated flag set behaves as
    /// if `save_live_blob` had been set instead.
    pub fn with_deprecated_save_live_audio(mut self, save_live_audio: bool) -> Self {
        if save_live_audio {
            self.save_live_blob = true;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_llm_calls_is_500_without_env_override() {
        // SAFETY: single-threaded test process; no other test reads this var
        // concurrently in a way that would race with this removal.
        unsafe {
            env::remove_var("ADK_MAX_LLM_CALLS");
        }
        assert_eq!(RunConfig::default().max_llm_calls, 500);
    }

    #[test]
    fn tool_thread_pool_config_defaults_to_4_workers() {
        assert_eq!(ToolThreadPoolConfig::default().max_workers, 4);
    }

    #[test]
    fn deprecated_save_live_audio_sets_save_live_blob() {
        let config = RunConfig::default().with_deprecated_save_live_audio(true);
        assert!(config.save_live_blob);
    }

    #[test]
    fn rejects_unknown_fields() {
        let json = r#"{"streaming_mode":"None","support_cfc":false,"save_live_blob":false,"max_llm_calls":500,"include_thoughts_from_other_agents":false,"extra":true}"#;
        assert!(rusty_serde::json::from_str::<RunConfig>(json).is_err());
    }
}
