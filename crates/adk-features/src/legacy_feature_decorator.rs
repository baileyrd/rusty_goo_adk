//! C0797: `utils.feature_decorator`'s `working_in_progress`/
//! `experimental` — a second, genuinely independent feature-gating
//! mechanism, confirmed (not assumed) to be a real duplicate of
//! [`crate::feature_decorator`], not the same system reached two ways.
//! Unlike that module's decorators, this one has no [`crate::feature_registry`]
//! involvement at all: no `FeatureName`, no registered stage, no
//! `is_feature_enabled` check — just a message, a label, and an
//! environment-variable escape hatch, checked *at call time* (matching
//! the source's own `new_init`/`wrapper` closures re-reading the env var
//! on every call, not once at decoration time).
//!
//! **Adaptation**: ported as plain guard functions rather than
//! decorators, same reasoning as [`crate::feature_decorator`]. The
//! source's message includes the decorated object's `__name__` via
//! runtime reflection Rust has no equivalent for; this port takes
//! `item_name` as an explicit caller-supplied argument instead — the
//! caller already knows what it's guarding, unlike a generic decorator
//! that has to introspect its target.

/// `utils.feature_decorator._is_truthy_env` — `'1'`/`'true'`/`'yes'`/
/// `'on'`, case-insensitive, surrounding whitespace trimmed.
fn is_truthy_env(var_name: &str) -> bool {
    std::env::var(var_name)
        .map(|value| {
            matches!(
                value.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

const WIP_BYPASS_ENV_VAR: &str = "ADK_ALLOW_WIP_FEATURES";
const WIP_DEFAULT_MESSAGE: &str = "This feature is a work in progress and is not working \
                                    completely. ADK users are not supposed to use it.";

const EXPERIMENTAL_SUPPRESS_ENV_VAR: &str = "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS";
const EXPERIMENTAL_DEFAULT_MESSAGE: &str = "This feature is experimental and may change or be \
                                             removed in future versions without notice. It may \
                                             introduce breaking changes at any time.";

/// C0797: `utils.feature_decorator.working_in_progress` — blocks usage
/// (an `Err`, matching the source's `RuntimeError`) unless
/// `ADK_ALLOW_WIP_FEATURES` is truthy, in which case it's bypassed
/// entirely (no error).
pub fn check_wip_or_bypass(item_name: &str, message: Option<&str>) -> Result<(), String> {
    if is_truthy_env(WIP_BYPASS_ENV_VAR) {
        return Ok(());
    }
    let message = message.unwrap_or(WIP_DEFAULT_MESSAGE);
    Err(format!("[WIP] {item_name}: {message}"))
}

/// C0797: `utils.feature_decorator.experimental` — never blocks usage;
/// emits a warning (via `eprintln!`, matching the disclosed
/// `warnings.warn`-has-no-Rust-equivalent adaptation
/// `feature_registry::emit_non_stable_warning_once` already
/// established) unless `ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS` is
/// truthy. Returns whether the warning was actually emitted, so a
/// caller (and this module's own tests) can observe the suppression
/// without a real logging/warning framework to capture against.
pub fn warn_experimental(item_name: &str, message: Option<&str>) -> bool {
    if is_truthy_env(EXPERIMENTAL_SUPPRESS_ENV_VAR) {
        return false;
    }
    let message = message.unwrap_or(EXPERIMENTAL_DEFAULT_MESSAGE);
    eprintln!("[EXPERIMENTAL] {item_name}: {message}");
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Environment variables are process-global state; serialize the
    // tests that touch them so they don't race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn wip_blocks_usage_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var(WIP_BYPASS_ENV_VAR);
        let result = check_wip_or_bypass("MyWipThing", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("MyWipThing"));
    }

    #[test]
    fn wip_uses_a_custom_message_when_given() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var(WIP_BYPASS_ENV_VAR);
        let result = check_wip_or_bypass("MyWipThing", Some("not ready yet"));
        assert_eq!(result.unwrap_err(), "[WIP] MyWipThing: not ready yet");
    }

    #[test]
    fn wip_bypasses_when_the_env_var_is_truthy() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var(WIP_BYPASS_ENV_VAR, "true");
        let result = check_wip_or_bypass("MyWipThing", None);
        std::env::remove_var(WIP_BYPASS_ENV_VAR);
        assert!(result.is_ok());
    }

    #[test]
    fn experimental_warns_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::remove_var(EXPERIMENTAL_SUPPRESS_ENV_VAR);
        assert!(warn_experimental("MyExperimentalThing", None));
    }

    #[test]
    fn experimental_is_suppressed_when_the_env_var_is_truthy() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var(EXPERIMENTAL_SUPPRESS_ENV_VAR, "1");
        let emitted = warn_experimental("MyExperimentalThing", None);
        std::env::remove_var(EXPERIMENTAL_SUPPRESS_ENV_VAR);
        assert!(!emitted);
    }
}
