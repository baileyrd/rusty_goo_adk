//! Capability C0417: `BaseToolConfig`/`ToolArgsConfig`/`ToolConfig`,
//! ported from `google.adk.tools.tool_configs`.
//!
//! The declarative YAML/dict shape for referencing a tool by name (plus
//! optional args) instead of constructing it in code — used by an ADK
//! app config's `tools:` list. This file (130 lines in the source)
//! contains only the three data models; the actual `from_config`
//! dynamic-dispatch resolution (5 reference kinds: built-in name /
//! instance path / class+args / factory+args / function path) lives
//! elsewhere (`BaseTool.from_config`) and needs Python's `importlib` —
//! genuinely inapplicable in this port, same disclosed-inapplicable
//! precedent already established for C0939's `_lazy.accessors`. Landing
//! these three types doesn't unblock that resolution logic; it gives
//! `base_tool.rs` (C0402), `base_toolset.rs` (C0403), and
//! `example_tool.rs` (C0419) — all three of which already cite this row
//! as their reason for not porting `from_config` — a real, tested type
//! to cite instead of "not built."
//!
//! **`@experimental(FeatureName.TOOL_CONFIG)`, now a real gate**: the
//! source decorates all three classes with `@experimental`
//! (`features._feature_decorator.experimental`, the `FeatureName`-backed
//! mechanism — a different, independent system from
//! `utils.feature_decorator`'s own `experimental`, see
//! `legacy_feature_decorator.rs`'s own module doc for that distinction).
//! `_make_feature_decorator` wraps `__init__` to raise unless
//! `is_feature_enabled(FeatureName.TOOL_CONFIG)` — exactly
//! [`adk_features::feature_decorator::check_feature_enabled`]'s existing
//! contract (C0647), so each type's `new` constructor calls it directly
//! rather than re-deriving the check.
//!
//! **Disclosed narrowing**: Pydantic's `model_validate`/JSON parsing
//! always routes through `__init__`, so the source's `@experimental`
//! gate applies equally to direct construction *and* deserialization.
//! This port's derived [`Deserialize`] impls don't call the `new`
//! constructors, so only direct construction via `::new` is gated —
//! deserializing a `ToolConfig` straight from JSON bypasses the check.
//! Same shape as `ResumabilityConfig::new`'s own already-disclosed gap
//! (`app_configs.rs`).

use std::collections::BTreeMap;

use adk_features::feature_decorator::{check_feature_enabled, FeatureNotEnabledError};
use adk_features::feature_registry::FeatureName;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// `tools.tool_configs.BaseToolConfig` — the base class for all tool
/// configs. `extra="forbid"` means no fields are ever accepted; this
/// port's own subclasses (were any built) would follow the same
/// `deny_unknown_fields` convention [`ToolConfig`] already uses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct BaseToolConfig {}

impl BaseToolConfig {
    pub fn new() -> Result<Self, FeatureNotEnabledError> {
        check_feature_enabled(FeatureName::ToolConfig)?;
        Ok(Self {})
    }
}

/// `tools.tool_configs.ToolArgsConfig` — free key-value pairs for a
/// [`ToolConfig`]'s `args`. `extra="allow"` maps directly to a flattened
/// open map, the same "real fields typed, everything else collected"
/// idiom [`adk_genai::content::MediaBlobStub`]'s own `rest` field
/// already established (here there are no real fields at all — every
/// key is collected).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolArgsConfig {
    #[rusty_serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ToolArgsConfig {
    pub fn new(extra: BTreeMap<String, Value>) -> Result<Self, FeatureNotEnabledError> {
        check_feature_enabled(FeatureName::ToolConfig)?;
        Ok(Self { extra })
    }
}

/// `tools.tool_configs.ToolConfig` — a declarative reference to a tool:
/// its `name` (an ADK built-in tool name, or a fully-qualified path to a
/// user-defined tool instance/class/factory/function — see the source's
/// own extensive docstring for the five reference kinds) plus optional
/// `args`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(deny_unknown_fields)]
pub struct ToolConfig {
    pub name: String,
    #[rusty_serde(default)]
    pub args: Option<ToolArgsConfig>,
}

impl ToolConfig {
    pub fn new(
        name: impl Into<String>,
        args: Option<ToolArgsConfig>,
    ) -> Result<Self, FeatureNotEnabledError> {
        check_feature_enabled(FeatureName::ToolConfig)?;
        Ok(Self {
            name: name.into(),
            args,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_features::feature_registry::TemporaryFeatureOverride;
    use std::sync::Mutex as StdMutex;

    // `TemporaryFeatureOverride` mutates process-wide state — serialize
    // tests that touch `FeatureName::ToolConfig`, the same pattern
    // `base_retrieval_tool.rs`'s own `TEST_LOCK` already established.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn base_tool_config_round_trips_through_json() {
        let json = rusty_serde::json::to_string(&BaseToolConfig::default()).unwrap();
        let back: BaseToolConfig = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(back, BaseToolConfig::default());
    }

    #[test]
    fn base_tool_config_rejects_unknown_fields() {
        let result: Result<BaseToolConfig, _> = rusty_serde::json::from_str(r#"{"extra":true}"#);
        assert!(result.is_err());
    }

    #[test]
    fn tool_args_config_collects_every_key() {
        let json = r#"{"agent":"./another_agent.yaml","skip_summarization":true}"#;
        let args: ToolArgsConfig = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(
            args.extra.get("agent"),
            Some(&Value::String("./another_agent.yaml".to_string()))
        );
        assert_eq!(
            args.extra.get("skip_summarization"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn tool_args_config_round_trips_through_json() {
        let mut extra = BTreeMap::new();
        extra.insert("key".to_string(), Value::String("value".to_string()));
        let args = ToolArgsConfig { extra };
        let json = rusty_serde::json::to_string(&args).unwrap();
        let back: ToolArgsConfig = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(args, back);
    }

    #[test]
    fn tool_config_with_no_args_round_trips() {
        let json = r#"{"name":"google_search"}"#;
        let config: ToolConfig = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(config.name, "google_search");
        assert_eq!(config.args, None);
    }

    #[test]
    fn tool_config_with_args_round_trips() {
        let json = r#"{"name":"AgentTool","args":{"agent":"./another_agent.yaml","skip_summarization":true}}"#;
        let config: ToolConfig = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(config.name, "AgentTool");
        let args = config.args.unwrap();
        assert_eq!(
            args.extra.get("skip_summarization"),
            Some(&Value::Bool(true))
        );

        let round_tripped = rusty_serde::json::to_string(&ToolConfig {
            name: config.name.clone(),
            args: Some(args.clone()),
        })
        .unwrap();
        let back: ToolConfig = rusty_serde::json::from_str(&round_tripped).unwrap();
        assert_eq!(back.name, "AgentTool");
        assert_eq!(back.args.unwrap(), args);
    }

    #[test]
    fn tool_config_rejects_unknown_top_level_fields() {
        let result: Result<ToolConfig, _> =
            rusty_serde::json::from_str(r#"{"name":"google_search","unexpected":true}"#);
        assert!(result.is_err());
    }

    #[test]
    fn constructors_error_when_the_feature_is_disabled() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ToolConfig, false);

        assert!(BaseToolConfig::new().is_err());
        assert!(ToolArgsConfig::new(BTreeMap::new()).is_err());
        assert!(ToolConfig::new("google_search", None).is_err());
    }

    #[test]
    fn constructors_succeed_when_the_feature_is_enabled() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::ToolConfig, true);

        assert!(BaseToolConfig::new().is_ok());
        assert!(ToolArgsConfig::new(BTreeMap::new()).is_ok());
        let config = ToolConfig::new("google_search", None).unwrap();
        assert_eq!(config.name, "google_search");
    }
}
