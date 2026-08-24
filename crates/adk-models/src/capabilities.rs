//! Capability C0105: `LlmCapabilities`/`gemini_output_schema_and_tools`,
//! ported from `google.adk.models._capabilities`.
//!
//! **Forward-pull**: `is_gemini_model`/`extract_model_name`
//! (`utils/model_name_utils.py`) and `get_google_llm_variant`/
//! `is_enterprise_mode_enabled`/`is_env_enabled` (`utils/variant_utils.py`,
//! `utils/env_utils.py`, C0796) aren't inventoried under a `utils/`
//! phase of their own, but `BaseLlm.capabilities`'s deprecated
//! name-based fallback needs them, so they're pulled forward here —
//! small, self-contained, env-var/regex-based utilities, same
//! rationale as `sessions.state.State` in Phase 2. (`is_env_enabled`/
//! `is_enterprise_mode_enabled` do have their own manifest row,
//! C0796 — this doc originally said otherwise; corrected once that
//! row's evidence was filled in.)

use rusty_serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// `utils/env_utils.py::is_env_enabled`.
pub fn is_env_enabled(value: Option<&str>, default: &str) -> bool {
    let value = value.unwrap_or(default);
    matches!(value.to_lowercase().as_str(), "true" | "1")
}

/// `utils/variant_utils.py::is_enterprise_mode_enabled` — reads
/// `GOOGLE_GENAI_USE_ENTERPRISE` (preferred) or the deprecated
/// `GOOGLE_GENAI_USE_VERTEXAI` env var.
pub fn is_enterprise_mode_enabled() -> bool {
    if let Ok(value) = std::env::var("GOOGLE_GENAI_USE_ENTERPRISE") {
        return is_env_enabled(Some(&value), "0");
    }
    if let Ok(value) = std::env::var("GOOGLE_GENAI_USE_VERTEXAI") {
        eprintln!(
            "DeprecationWarning: GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use \
             GOOGLE_GENAI_USE_ENTERPRISE instead"
        );
        return is_env_enabled(Some(&value), "0");
    }
    false
}

/// `utils/variant_utils.py::GoogleLLMVariant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleLlmVariant {
    VertexAi,
    GeminiApi,
}

/// `utils/variant_utils.py::get_google_llm_variant`.
pub fn get_google_llm_variant() -> GoogleLlmVariant {
    if is_enterprise_mode_enabled() {
        GoogleLlmVariant::VertexAi
    } else {
        GoogleLlmVariant::GeminiApi
    }
}

/// `utils/model_name_utils.py::extract_model_name` — strips a path-based
/// (`projects/.../models/NAME`) or provider-prefixed (`provider/NAME`)
/// wrapper down to the bare model name.
pub fn extract_model_name(model_string: &str) -> &str {
    static PATH_PREFIXES: &[&str] = &["/models/", "/publisherModels/"];
    for prefix in PATH_PREFIXES {
        if let Some(index) = model_string.rfind(prefix) {
            return &model_string[index + prefix.len()..];
        }
    }
    model_string
        .rsplit_once('/')
        .map(|(_, name)| name)
        .unwrap_or(model_string)
}

/// `utils/model_name_utils.py::is_gemini_model`.
pub fn is_gemini_model(model_string: Option<&str>) -> bool {
    match model_string {
        Some(s) if !s.is_empty() => extract_model_name(s).starts_with("gemini-"),
        _ => false,
    }
}

/// `models._capabilities.gemini_output_schema_and_tools`.
pub fn gemini_output_schema_and_tools(model_name: &str) -> bool {
    get_google_llm_variant() == GoogleLlmVariant::VertexAi && is_gemini_model(Some(model_name))
}

/// `utils/model_name_utils.py::is_gemini_3_5_live_translate` — forward-pulled
/// for `GeminiLlmConnection` (Phase 3 batch 5), same rationale as the module
/// doc's other forward-pulls.
pub fn is_gemini_3_5_live_translate(model_string: Option<&str>) -> bool {
    let Some(model_string) = model_string.filter(|s| !s.is_empty()) else {
        return false;
    };
    extract_model_name(model_string).starts_with("gemini-3.5-live-translate")
}

/// `utils/model_name_utils.py::_is_gemini_3_x_live`.
pub fn is_gemini_3_x_live(model_string: Option<&str>) -> bool {
    let Some(model_string) = model_string.filter(|s| !s.is_empty()) else {
        return false;
    };
    let model_name = extract_model_name(model_string);
    model_name.starts_with("gemini-3.")
        && model_name.contains("-live")
        && !is_gemini_3_5_live_translate(Some(model_string))
}

/// Resolved capabilities for an LLM instance — an immutable snapshot a
/// model reports via `BaseLlm::capabilities`, rather than a caller
/// re-deriving support from the model name/type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct LlmCapabilities {
    pub output_schema_and_tools: bool,
}

/// One-shot `FutureWarning` per process for the deprecated name-based
/// capability fallback, mirroring the source's warning firing on every call
/// that grants the capability — this port logs once instead of on every
/// call, since a per-call `eprintln!` in a hot capability-check path would
/// be far noisier than Python's `warnings` module (which itself
/// deduplicates identical warnings by default).
static FALLBACK_WARNED: OnceLock<()> = OnceLock::new();

/// The deprecated name-based fallback used by `BaseLlm::capabilities`'s
/// default implementation for a model that doesn't override it.
pub fn legacy_output_schema_and_tools(model_name: &str, type_name: &str) -> bool {
    if !gemini_output_schema_and_tools(model_name) {
        return false;
    }
    FALLBACK_WARNED.get_or_init(|| {
        eprintln!(
            "FutureWarning: {type_name} relies on name-based detection of \
             output_schema_and_tools. Override BaseLlm.capabilities to declare \
             it explicitly; this fallback will be removed in a future release."
        );
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_model_name_strips_a_models_path_prefix() {
        assert_eq!(
            extract_model_name("projects/p/locations/l/models/gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn extract_model_name_strips_a_provider_prefix() {
        assert_eq!(
            extract_model_name("gemini/gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn extract_model_name_passes_through_a_bare_name() {
        assert_eq!(extract_model_name("gemini-2.5-flash"), "gemini-2.5-flash");
    }

    #[test]
    fn is_gemini_model_matches_the_gemini_prefix() {
        assert!(is_gemini_model(Some("gemini-2.5-flash")));
        assert!(!is_gemini_model(Some("gpt-4")));
        assert!(!is_gemini_model(None));
        assert!(!is_gemini_model(Some("")));
    }

    #[test]
    fn is_gemini_3_x_live_matches_a_gemini_3_live_model() {
        assert!(is_gemini_3_x_live(Some("gemini-3.0-live")));
        assert!(!is_gemini_3_x_live(Some("gemini-2.5-live")));
        assert!(!is_gemini_3_x_live(Some("gemini-3.0-flash")));
        assert!(!is_gemini_3_x_live(None));
    }

    #[test]
    fn is_gemini_3_x_live_excludes_the_3_5_live_translate_variant() {
        assert!(!is_gemini_3_x_live(Some("gemini-3.5-live-translate")));
        assert!(is_gemini_3_5_live_translate(Some(
            "gemini-3.5-live-translate"
        )));
    }

    #[test]
    fn is_gemini_3_5_live_translate_requires_the_exact_prefix() {
        assert!(!is_gemini_3_5_live_translate(Some("gemini-3.0-live")));
        assert!(!is_gemini_3_5_live_translate(None));
        assert!(!is_gemini_3_5_live_translate(Some("")));
    }

    #[test]
    fn is_env_enabled_accepts_true_and_1_case_insensitively() {
        assert!(is_env_enabled(Some("true"), "0"));
        assert!(is_env_enabled(Some("TRUE"), "0"));
        assert!(is_env_enabled(Some("1"), "0"));
        assert!(!is_env_enabled(Some("false"), "0"));
        assert!(!is_env_enabled(None, "0"));
    }

    // Serializes tests that mutate GOOGLE_GENAI_USE_ENTERPRISE/
    // GOOGLE_GENAI_USE_VERTEXAI, process-wide env state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn is_enterprise_mode_enabled_reads_the_preferred_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
            std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        }
        let result = is_enterprise_mode_enabled();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
        }
        assert!(result);
    }

    #[test]
    fn is_enterprise_mode_enabled_prefers_enterprise_over_the_deprecated_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "false");
            std::env::set_var("GOOGLE_GENAI_USE_VERTEXAI", "true");
        }
        let result = is_enterprise_mode_enabled();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        assert!(!result);
    }

    #[test]
    fn is_enterprise_mode_enabled_falls_back_to_the_deprecated_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::set_var("GOOGLE_GENAI_USE_VERTEXAI", "1");
        }
        let result = is_enterprise_mode_enabled();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        assert!(result);
    }

    #[test]
    fn is_enterprise_mode_enabled_defaults_to_false() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        assert!(!is_enterprise_mode_enabled());
    }

    #[test]
    fn llm_capabilities_default_is_false() {
        assert!(!LlmCapabilities::default().output_schema_and_tools);
    }

    #[test]
    fn llm_capabilities_rejects_unknown_fields() {
        let json = r#"{"output_schema_and_tools":true,"extra":1}"#;
        assert!(rusty_serde::json::from_str::<LlmCapabilities>(json).is_err());
    }
}
