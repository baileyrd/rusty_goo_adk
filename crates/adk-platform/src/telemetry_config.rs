//! Capability C0942: telemetry user-consent config, ported from
//! `google.adk.utils._telemetry_config`.
//!
//! **Forward-pull**: like [`crate::visual_builder_context`] (C0936), this
//! file lives under `utils/` in the source, not `platform/` — pulled in
//! here anyway since it's the same shape of thing this crate already
//! exists for (a small, self-contained runtime-environment primitive with
//! no natural home in a higher-level crate).
//!
//! **Adaptations, disclosed**:
//! - `pathlib.Path.home()` resolves the platform-appropriate home
//!   directory (`$HOME` on Unix, `%USERPROFILE%` on Windows via Python's
//!   own `os.path.expanduser` machinery). This port reads `$HOME` directly
//!   and nothing else — correct on Unix (this workspace's target), a
//!   disclosed gap on Windows rather than a claimed cross-platform match.
//!   No new dependency (e.g. the `dirs` crate) was added for this.
//! - The source uses Python's `logging` module for read/write failures;
//!   this port uses `eprintln!`, the same disclosed no-logging-framework
//!   gap used everywhere else in this port.
//! - `json.dump(..., indent=2)` pretty-prints the on-disk file;
//!   `rusty_serde::json` has no pretty-printer, so this port writes
//!   compact JSON instead — a purely cosmetic divergence, since the file
//!   is only ever read back by this same reader (round-trip fidelity is
//!   what matters, and compact JSON round-trips exactly).

use std::path::PathBuf;

use rusty_serde::value::Value;

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

/// `_telemetry_config.get_user_config_path` — the path to the ADK global
/// config file, `~/.adk/config.json`. Returns `None` if `$HOME` can't be
/// resolved (see the module doc's disclosed Windows gap).
pub fn get_user_config_path() -> Option<PathBuf> {
    home_dir().map(|home| home.join(".adk").join("config.json"))
}

fn read_config(path: &std::path::Path) -> Value {
    match std::fs::read_to_string(path) {
        Ok(text) => rusty_serde::json::from_str(&text).unwrap_or(Value::Map(Vec::new())),
        Err(_) => Value::Map(Vec::new()),
    }
}

/// `_telemetry_config.read_telemetry_consent` — `Some(true)`/`Some(false)`
/// if a preference was explicitly recorded, `None` if not (including when
/// `$HOME` can't be resolved, the file doesn't exist, or it fails to
/// parse — the source's own broad `except Exception` catch-all).
pub fn read_telemetry_consent() -> Option<bool> {
    let path = get_user_config_path()?;
    if !path.exists() {
        return None;
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => match rusty_serde::json::from_str::<Value>(&text) {
            Ok(Value::Map(entries)) => entries.into_iter().find_map(|(k, v)| match (k, v) {
                (key, Value::Bool(b)) if key == "telemetry" => Some(b),
                _ => None,
            }),
            Ok(_) | Err(_) => None,
        },
        Err(e) => {
            eprintln!(
                "Failed to read telemetry config from {}: {e}",
                path.display()
            );
            None
        }
    }
}

/// `_telemetry_config.write_telemetry_consent` — writes the telemetry
/// consent status to the config file, preserving any other keys already
/// present. Returns `Err` (the source re-raises) if `$HOME` can't be
/// resolved or the write fails.
pub fn write_telemetry_consent(enabled: bool) -> Result<(), String> {
    let path = get_user_config_path().ok_or("could not resolve the home directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            let message = format!(
                "Failed to write telemetry config to {}: {e}",
                path.display()
            );
            eprintln!("{message}");
            message
        })?;
    }

    let mut entries = match read_config(&path) {
        Value::Map(entries) => entries,
        _ => Vec::new(),
    };
    entries.retain(|(k, _)| k != "telemetry");
    entries.push(("telemetry".to_string(), Value::Bool(enabled)));

    let json = rusty_serde::json::to_string(&Value::Map(entries))
        .map_err(|e| format!("Failed to serialize telemetry config: {e}"))?;
    std::fs::write(&path, format!("{json}\n")).map_err(|e| {
        let message = format!(
            "Failed to write telemetry config to {}: {e}",
            path.display()
        );
        eprintln!("{message}");
        message
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serializes tests that mutate $HOME, process-wide env state.
    static HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Guarded by HOME_LOCK, so every test runs against this same
    // directory one at a time -- no cross-test collision risk despite
    // the fixed name.
    fn with_temp_home<T>(f: impl FnOnce(&std::path::Path) -> T) -> T {
        let _guard = HOME_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("adk-telemetry-config-test-home");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let previous = std::env::var("HOME").ok();
        unsafe {
            std::env::set_var("HOME", &dir);
        }
        let result = f(&dir);
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        result
    }

    #[test]
    fn read_telemetry_consent_is_none_without_a_config_file() {
        with_temp_home(|_| {
            assert_eq!(read_telemetry_consent(), None);
        });
    }

    #[test]
    fn write_then_read_round_trips_the_consent_value() {
        with_temp_home(|_| {
            write_telemetry_consent(true).unwrap();
            assert_eq!(read_telemetry_consent(), Some(true));

            write_telemetry_consent(false).unwrap();
            assert_eq!(read_telemetry_consent(), Some(false));
        });
    }

    #[test]
    fn write_telemetry_consent_preserves_other_keys() {
        with_temp_home(|_| {
            let path = get_user_config_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, r#"{"other_setting":"keep-me"}"#).unwrap();

            write_telemetry_consent(true).unwrap();

            let text = std::fs::read_to_string(&path).unwrap();
            assert!(text.contains("keep-me"));
            assert!(text.contains("\"telemetry\":true"));
        });
    }

    #[test]
    fn read_telemetry_consent_returns_none_for_malformed_json() {
        with_temp_home(|_| {
            let path = get_user_config_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "not json").unwrap();
            assert_eq!(read_telemetry_consent(), None);
        });
    }
}
