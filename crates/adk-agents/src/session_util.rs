//! Capability C0209: session-state utility functions, ported from
//! `google.adk.sessions._session_util`. Completes C0205 (`State`'s
//! `APP_PREFIX`/`USER_PREFIX`/`TEMP_PREFIX` constants, added to
//! `state.rs` in this same batch).
//!
//! **Ahead of its own caller, disclosed**: `extract_state_delta`'s
//! whole purpose in the source is splitting a flat state dict into the
//! `app_state`/`user_state`/session-state shares a persistent
//! `BaseSessionService` writes to three different storage locations,
//! but this port's `Session`/`State` have no such cross-session shared
//! storage yet (`services.rs`'s own module doc discloses this at
//! length). So this function is real and tested, but nothing in this
//! port's `SessionService`/`InMemorySessionService` actually routes
//! its `"app"`/`"user"` output anywhere yet — the same "utility ready,
//! ahead of the architecture that would consume it" situation
//! `remote_mcp_server.rs`/`artifact_util.rs` already disclosed.
//!
//! **`make_json_safe_state`, narrowed to a near-no-op, disclosed**:
//! the source's whole point is coercing a dict of arbitrary Python
//! objects (which might include non-JSON-serializable values like
//! callables) into something JSON-safe, falling back per-value to a
//! string representation on failure. This port's state is already
//! `BTreeMap<String, Value>` — `rusty_serde::value::Value` can only
//! ever hold JSON-representable variants (`Null`/`Bool`/`Int`/`UInt`/
//! `Float`/`String`/`Seq`/`Map`) by construction, so there is no value
//! this port's state can hold that could ever fail this coercion. The
//! function still exists (returning its input unchanged) so a future
//! persistent `SessionService` has the same named call site the
//! source uses before writing state to a JSON column, and so
//! `extract_json_safe_state_delta` composes the same way the source's
//! does — but its body is a real, disclosed near-no-op here, not a
//! narrowed reimplementation of genuine coercion logic.
//!
//! **`decode_model`, adapted**: the source guards only against
//! primitive non-dict values (returning `None`), letting a genuine
//! `ValidationError` on a malformed-but-dict-shaped value propagate
//! uncaught. This port collapses both cases to `None` — a real
//! narrowing, since a caller here can't distinguish "wasn't shaped
//! like this model at all" from "genuinely absent/primitive" the way
//! the source's raised exception vs. `None` return does.

use rusty_serde::value::Value;
use rusty_serde::Deserialize;
use std::collections::BTreeMap;

use crate::state::State;

/// C0209: decodes a typed model `T` from an opaque `Value`, guarding
/// against primitive non-map values (e.g. a legacy/corrupted `"null"`
/// string persisted in place of SQL NULL) the same way the source
/// guards against non-dict values before calling `model_validate`.
/// See the module doc for the disclosed "both failure modes collapse
/// to `None`" narrowing.
pub fn decode_model<T: for<'de> Deserialize<'de>>(data: Option<&Value>) -> Option<T> {
    match data {
        None | Some(Value::Null) => None,
        Some(Value::Map(_)) => rusty_serde::json::from_value(data.cloned()?).ok(),
        Some(_) => None,
    }
}

/// C0209: extracts app/user/session state deltas from a flat state
/// dictionary, keyed by scope (`"app"`/`"user"`/`"session"`) with the
/// `app:`/`user:` prefix stripped from each key. `temp:`-prefixed
/// keys are dropped entirely (never persisted), matching the source.
pub fn extract_state_delta(
    state: &BTreeMap<String, Value>,
) -> BTreeMap<String, BTreeMap<String, Value>> {
    let mut deltas: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::from([
        ("app".to_string(), BTreeMap::new()),
        ("user".to_string(), BTreeMap::new()),
        ("session".to_string(), BTreeMap::new()),
    ]);
    for (key, value) in state {
        if let Some(stripped) = key.strip_prefix(State::APP_PREFIX) {
            deltas
                .get_mut("app")
                .unwrap()
                .insert(stripped.to_string(), value.clone());
        } else if let Some(stripped) = key.strip_prefix(State::USER_PREFIX) {
            deltas
                .get_mut("user")
                .unwrap()
                .insert(stripped.to_string(), value.clone());
        } else if !key.starts_with(State::TEMP_PREFIX) {
            deltas
                .get_mut("session")
                .unwrap()
                .insert(key.clone(), value.clone());
        }
    }
    deltas
}

/// C0209: coerces a state dictionary into a JSON-serializable form —
/// see the module doc for why this is a near-no-op in this port
/// (identity, cloning its input) rather than genuine per-value
/// coercion.
pub fn make_json_safe_state(state: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    state.clone()
}

/// C0209: `extract_state_delta` coerced into a JSON-serializable form
/// — the call services persisting state to a JSON column must use
/// instead of `extract_state_delta` directly.
pub fn extract_json_safe_state_delta(
    state: &BTreeMap<String, Value>,
) -> BTreeMap<String, BTreeMap<String, Value>> {
    extract_state_delta(state)
        .into_iter()
        .map(|(scope, delta)| (scope, make_json_safe_state(&delta)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, rusty_serde::Deserialize)]
    struct Config {
        name: String,
    }

    #[test]
    fn extract_state_delta_splits_by_prefix() {
        let state = BTreeMap::from([
            ("app:theme".to_string(), Value::String("dark".to_string())),
            ("user:locale".to_string(), Value::String("en".to_string())),
            (
                "temp:scratch".to_string(),
                Value::String("drop-me".to_string()),
            ),
            ("count".to_string(), Value::Int(1)),
        ]);
        let deltas = extract_state_delta(&state);
        assert_eq!(
            deltas["app"].get("theme"),
            Some(&Value::String("dark".to_string()))
        );
        assert_eq!(
            deltas["user"].get("locale"),
            Some(&Value::String("en".to_string()))
        );
        assert_eq!(deltas["session"].get("count"), Some(&Value::Int(1)));
        assert!(!deltas["session"].contains_key("temp:scratch"));
        assert_eq!(deltas["session"].len(), 1);
    }

    #[test]
    fn extract_state_delta_is_empty_for_an_empty_state() {
        let deltas = extract_state_delta(&BTreeMap::new());
        assert!(deltas["app"].is_empty());
        assert!(deltas["user"].is_empty());
        assert!(deltas["session"].is_empty());
    }

    #[test]
    fn make_json_safe_state_returns_its_input_unchanged() {
        let state = BTreeMap::from([("k".to_string(), Value::Int(1))]);
        assert_eq!(make_json_safe_state(&state), state);
    }

    #[test]
    fn extract_json_safe_state_delta_composes_extract_and_coerce() {
        let state = BTreeMap::from([("app:theme".to_string(), Value::String("dark".to_string()))]);
        let deltas = extract_json_safe_state_delta(&state);
        assert_eq!(
            deltas["app"].get("theme"),
            Some(&Value::String("dark".to_string()))
        );
    }

    #[test]
    fn decode_model_returns_none_for_absent_data() {
        assert_eq!(decode_model::<Config>(None), None);
    }

    #[test]
    fn decode_model_returns_none_for_a_primitive_value() {
        assert_eq!(
            decode_model::<Config>(Some(&Value::String("null".to_string()))),
            None
        );
        assert_eq!(decode_model::<Config>(Some(&Value::Int(1))), None);
    }

    #[test]
    fn decode_model_decodes_a_valid_map() {
        let value = Value::Map(vec![(
            "name".to_string(),
            Value::String("agent-1".to_string()),
        )]);
        assert_eq!(
            decode_model::<Config>(Some(&value)),
            Some(Config {
                name: "agent-1".to_string()
            })
        );
    }

    #[test]
    fn decode_model_returns_none_for_a_malformed_map() {
        let value = Value::Map(vec![("wrong_field".to_string(), Value::Int(1))]);
        assert_eq!(decode_model::<Config>(Some(&value)), None);
    }
}
