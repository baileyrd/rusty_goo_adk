//! Part of capabilities C0428/C0430-C0432: `model_name_utils.py`'s
//! `is_gemini_model_id_check_disabled`/`_is_managed_agent`, ported from
//! `google.adk.utils.model_name_utils`. Pulled forward the same way
//! `is_gemini_model` already was in `adk-models::capabilities` — small,
//! self-contained, env-var-based utilities the built-in Gemini
//! grounding tools all share.

use adk_models::capabilities::is_env_enabled;

const DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR: &str = "ADK_DISABLE_GEMINI_MODEL_ID_CHECK";

/// Returns `true` when Gemini model-id validation should be bypassed —
/// an opt-in environment variable for internal usage where model ids may
/// not follow the public `gemini-*` naming convention.
pub fn is_gemini_model_id_check_disabled() -> bool {
    let value = std::env::var(DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR).ok();
    is_env_enabled(value.as_deref(), "0")
}

/// `_is_managed_agent`: whether the request was built by a "Managed
/// Agent". Not ported — `LlmRequest` has no `_is_managed_agent` field in
/// this port (the Managed Agents feature itself isn't built), so this
/// always reports `false`. Disclosed narrowing, not a silent one: a
/// managed-agent request that the source would treat as always-Gemini-
/// compatible regardless of model name gets no such exemption here.
pub fn is_managed_agent() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_gemini_model_id_check_disabled_defaults_to_false() {
        // SAFETY: single-threaded test, no other test in this crate reads
        // or writes this specific env var.
        unsafe {
            std::env::remove_var(DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR);
        }
        assert!(!is_gemini_model_id_check_disabled());
    }

    #[test]
    fn is_gemini_model_id_check_disabled_reads_the_env_var_case_insensitively() {
        // SAFETY: single-threaded test, no other test in this crate reads
        // or writes this specific env var.
        unsafe {
            std::env::set_var(DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR, "TRUE");
        }
        assert!(is_gemini_model_id_check_disabled());
        unsafe {
            std::env::remove_var(DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR);
        }
    }

    #[test]
    fn is_managed_agent_is_always_false() {
        assert!(!is_managed_agent());
    }
}
