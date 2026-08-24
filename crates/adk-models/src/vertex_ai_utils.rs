//! Capability C0934: `get_express_mode_api_key`, ported from
//! `google.adk.utils.vertex_ai_utils`.

use std::env;

use crate::capabilities::is_enterprise_mode_enabled;

/// `vertex_ai_utils.get_express_mode_api_key` — validates and returns the
/// API key for Vertex AI Express Mode.
///
/// The source raises `ValueError` when both a `project`/`location` and an
/// `express_mode_api_key` are given; this port returns `Err(String)`
/// instead, the same "no domain-specific exception type for a plain
/// `ValueError`" convention used by `adk-genai::json_utils::safe_json_loads`
/// (C0931).
pub fn get_express_mode_api_key(
    project: Option<&str>,
    location: Option<&str>,
    express_mode_api_key: Option<&str>,
) -> Result<Option<String>, String> {
    if (project.is_some() || location.is_some()) && express_mode_api_key.is_some() {
        return Err(
            "Cannot specify project or location and express_mode_api_key. \
             Either use project and location, or just the express_mode_api_key."
                .to_string(),
        );
    }
    if !is_enterprise_mode_enabled() {
        return Ok(None);
    }
    Ok(express_mode_api_key
        .map(str::to_string)
        .or_else(|| env::var("GOOGLE_API_KEY").ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::capabilities::ENV_LOCK;

    #[test]
    fn rejects_both_project_or_location_and_an_explicit_api_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let result = get_express_mode_api_key(Some("proj"), None, Some("key"));
        assert!(result.is_err());
    }

    #[test]
    fn returns_none_when_enterprise_mode_is_disabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        assert_eq!(get_express_mode_api_key(None, None, Some("key")), Ok(None));
    }

    #[test]
    fn returns_the_explicit_api_key_when_enterprise_mode_is_enabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        }
        let result = get_express_mode_api_key(None, None, Some("explicit-key"));
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
        }
        assert_eq!(result, Ok(Some("explicit-key".to_string())));
    }

    #[test]
    fn falls_back_to_the_env_var_when_enterprise_mode_is_enabled_and_no_key_given() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
            std::env::set_var("GOOGLE_API_KEY", "env-key");
        }
        let result = get_express_mode_api_key(None, None, None);
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        assert_eq!(result, Ok(Some("env-key".to_string())));
    }
}
