//! `State`, ported from `google.adk.sessions.state` (Phase 5).
//!
//! **Disclosed forward-pull**: `State` is inventoried under Phase 5
//! (`sessions/`), but `agents::Context` cannot function without it (its
//! `.state` property is a `State`) — the same cross-phase-forward-reference
//! situation Phase 1 hit with `Event`/`LlmResponse`. `state.py` is small
//! (135 lines) and has no dependency on the rest of `sessions/`, so it is
//! pulled forward here rather than stubbed. It should be *moved* to a
//! `adk-sessions` crate (not reimplemented) once Phase 5 lands and
//! `Context`/`InvocationContext` updated to depend on that crate instead.
//!
//! **Adaptation**: the source's per-key schema validation
//! (`_validate_state_entry`, matching a key/value pair against a Pydantic
//! model's declared fields via `TypeAdapter`) has no direct Rust
//! equivalent — Rust has no runtime type-reflection registry for an
//! arbitrary user-defined struct. Schema validation is therefore not
//! implemented; `State` here is an unconditional delta-tracking map. This
//! is a real, disclosed gap (not a silent drop) to revisit if/when the
//! Rust port designs its own state-schema mechanism.

use rusty_serde::value::Value;
use std::collections::BTreeMap;

/// A state map that tracks the current value alongside the pending-commit
/// delta.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    value: BTreeMap<String, Value>,
    delta: BTreeMap<String, Value>,
}

impl State {
    pub fn new(value: BTreeMap<String, Value>, delta: BTreeMap<String, Value>) -> Self {
        Self { value, delta }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.delta.get(key).or_else(|| self.value.get(key))
    }

    pub fn get_or(&self, key: &str, default: Value) -> Value {
        self.get(key).cloned().unwrap_or(default)
    }

    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        self.value.insert(key.clone(), value.clone());
        self.delta.insert(key, value);
    }

    pub fn contains(&self, key: &str) -> bool {
        self.value.contains_key(key) || self.delta.contains_key(key)
    }

    pub fn setdefault(&mut self, key: &str, default: Value) -> Value {
        if self.contains(key) {
            self.get(key).cloned().unwrap()
        } else {
            self.set(key, default.clone());
            default
        }
    }

    pub fn has_delta(&self) -> bool {
        !self.delta.is_empty()
    }

    pub fn update(&mut self, delta: BTreeMap<String, Value>) {
        self.value.extend(delta.clone());
        self.delta.extend(delta);
    }

    pub fn to_map(&self) -> BTreeMap<String, Value> {
        let mut result = self.value.clone();
        result.extend(self.delta.clone());
        result
    }

    /// The pending-commit delta accumulated so far. Used by `Context` to
    /// sync this state's changes back into the built `EventActions`'
    /// `state_delta` (mirroring the source's reference-shared dict, where
    /// `State._delta` *is* `EventActions.state_delta` — see `context.rs`'s
    /// `Context::new`/`into_actions` for why this port copies instead).
    pub fn delta_map(&self) -> BTreeMap<String, Value> {
        self.delta.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_prefers_delta_over_value() {
        let mut value = BTreeMap::new();
        value.insert("k".to_string(), Value::String("old".to_string()));
        let mut delta = BTreeMap::new();
        delta.insert("k".to_string(), Value::String("new".to_string()));
        let state = State::new(value, delta);
        assert_eq!(state.get("k"), Some(&Value::String("new".to_string())));
    }

    #[test]
    fn set_updates_both_value_and_delta() {
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        state.set("k", Value::Int(1));
        assert!(state.has_delta());
        assert_eq!(state.get("k"), Some(&Value::Int(1)));
    }

    #[test]
    fn has_delta_is_false_for_a_freshly_loaded_state() {
        let state = State::new(BTreeMap::new(), BTreeMap::new());
        assert!(!state.has_delta());
    }

    #[test]
    fn setdefault_only_sets_when_key_absent() {
        let mut state = State::new(BTreeMap::new(), BTreeMap::new());
        let first = state.setdefault("k", Value::Int(1));
        assert_eq!(first, Value::Int(1));
        let second = state.setdefault("k", Value::Int(2));
        assert_eq!(
            second,
            Value::Int(1),
            "existing value must not be overwritten"
        );
    }

    #[test]
    fn to_map_layers_delta_over_value() {
        let mut value = BTreeMap::new();
        value.insert("a".to_string(), Value::Int(1));
        let mut state = State::new(value, BTreeMap::new());
        state.set("b", Value::Int(2));
        let map = state.to_map();
        assert_eq!(map.get("a"), Some(&Value::Int(1)));
        assert_eq!(map.get("b"), Some(&Value::Int(2)));
    }
}
