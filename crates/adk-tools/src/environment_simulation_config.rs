//! C0486: `tools.environment_simulation.environment_simulation_config` —
//! configuration models for deterministic tool-call fault injection
//! ([`InjectedError`], [`InjectionConfig`], [`MockStrategy`],
//! [`ToolSimulationConfig`], [`EnvironmentSimulationConfig`]).
//!
//! **Adaptation**: same "plain fields + explicit `validate()`" pattern
//! `skills_models.rs` already established for pydantic `@model_validator`/
//! `@field_validator` — deserializing an invalid payload succeeds
//! structurally; call `validate()` (which cascades into every nested
//! config's own `validate()`) to enforce what the source enforces
//! automatically at construction.
//!
//! **`@experimental(FeatureName.ENVIRONMENT_SIMULATION)`, wired**: the
//! source decorates every type in this module individually (each
//! decoration re-checks at `__init__`). Since every one of those types is
//! only ever meaningfully constructed as part of building an
//! [`EnvironmentSimulationConfig`], this port collapses the check to one
//! call — [`EnvironmentSimulationConfig::validate`] — rather than
//! duplicating `check_feature_enabled` on every leaf struct; the first
//! real consumer of `crate::feature_decorator::check_feature_enabled`
//! (C0647's own doc noted the guard function existed but wasn't yet wired
//! to a call site).
//!
//! **`simulation_model_configuration`, narrowed**: the source's field is a
//! full `google.genai.types.GenerateContentConfig`, defaulted to a
//! `ThinkingConfig(include_thoughts=False, thinking_budget=10240)`. This
//! port reuses [`adk_models::llm_request::GenerateContentConfigStub`] (the
//! same narrowed placeholder `LlmRequest.config` already uses) rather than
//! inventing a second stub type — `thinking_config` on that stub is
//! already an opaque `Value`, so the default is built directly as one.

use std::collections::BTreeMap;

use adk_features::feature_decorator::{check_feature_enabled, FeatureNotEnabledError};
use adk_features::feature_registry::FeatureName;
use adk_models::llm_request::GenerateContentConfigStub;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// An error to be injected into a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InjectedError {
    /// Presents as `"error_code"` in the tool response dict.
    pub injected_http_error_code: i64,
    /// Presents as `"error_message"` in the tool response dict.
    pub error_message: String,
}

/// Injection configuration for a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InjectionConfig {
    #[rusty_serde(default = "default_injection_probability")]
    pub injection_probability: f64,
    /// Only apply injection if the request matches `match_args` — every
    /// key/value pair here must be present, with an equal value, in the
    /// tool call's actual arguments.
    #[rusty_serde(default)]
    pub match_args: Option<BTreeMap<String, Value>>,
    /// Injected latency, in seconds. May not be accurate if the
    /// interceptor is applied as an after-tool callback (source's own
    /// caveat, preserved as a doc note since this port has no
    /// before/after-tool callback dispatch to apply it against yet).
    #[rusty_serde(default)]
    pub injected_latency_seconds: f64,
    pub random_seed: Option<u64>,
    pub injected_error: Option<InjectedError>,
    pub injected_response: Option<BTreeMap<String, Value>>,
}

fn default_injection_probability() -> f64 {
    1.0
}

impl Default for InjectionConfig {
    fn default() -> Self {
        InjectionConfig {
            injection_probability: default_injection_probability(),
            match_args: None,
            injected_latency_seconds: 0.0,
            random_seed: None,
            injected_error: None,
            injected_response: None,
        }
    }
}

impl InjectionConfig {
    /// The source's `check_injected_error_or_response` (`@model_validator`)
    /// plus the `Field(le=120.0)` constraint on `injected_latency_seconds`
    /// pydantic enforces at construction.
    pub fn validate(&self) -> Result<(), String> {
        if self.injected_error.is_some() == self.injected_response.is_some() {
            return Err(
                "Either injected_error or injected_response must be set, but not both, and not \
                 neither."
                    .to_string(),
            );
        }
        if self.injected_latency_seconds > 120.0 {
            return Err(format!(
                "injected_latency_seconds must be <= 120.0, got {}",
                self.injected_latency_seconds
            ));
        }
        Ok(())
    }
}

/// Mock strategy for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MockStrategy {
    #[default]
    #[rusty_serde(rename = "MOCK_STRATEGY_UNSPECIFIED")]
    Unspecified,
    /// Use tool specifications to mock the tool response.
    #[rusty_serde(rename = "MOCK_STRATEGY_TOOL_SPEC")]
    ToolSpec,
    /// Deprecated, please use `ToolSpec` with tracing input.
    #[rusty_serde(rename = "MOCK_STRATEGY_TRACING")]
    Tracing,
}

/// Simulation configuration for a single tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSimulationConfig {
    pub tool_name: String,
    /// If provided, the tool will be injected with `injected_value` at
    /// `injection_probability` first; `mock_strategy_type` applies only
    /// if no injection config is hit.
    #[rusty_serde(default)]
    pub injection_configs: Vec<InjectionConfig>,
    #[rusty_serde(default)]
    pub mock_strategy_type: MockStrategy,
}

impl ToolSimulationConfig {
    /// The source's `check_mock_strategy_type` (`@model_validator`),
    /// cascading into each `injection_configs` entry's own `validate()`.
    pub fn validate(&self) -> Result<(), String> {
        if self.injection_configs.is_empty() && self.mock_strategy_type == MockStrategy::Unspecified
        {
            return Err(
                "If injection_configs is empty, mock_strategy_type cannot be \
                 MOCK_STRATEGY_UNSPECIFIED."
                    .to_string(),
            );
        }
        for injection_config in &self.injection_configs {
            injection_config.validate()?;
        }
        Ok(())
    }
}

/// Configuration for `EnvironmentSimulation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentSimulationConfig {
    #[rusty_serde(default)]
    pub tool_simulation_configs: Vec<ToolSimulationConfig>,
    /// The model used for internal simulator LLM calls (tool analysis,
    /// mock responses) — unused in this port's injection-only scope; see
    /// `environment_simulation_engine`'s module doc.
    #[rusty_serde(default = "default_simulation_model")]
    pub simulation_model: String,
    #[rusty_serde(default = "default_simulation_model_configuration")]
    pub simulation_model_configuration: GenerateContentConfigStub,
    /// Tracing data (e.g. a prior agent run trace in JSON string format)
    /// to provide historical context for mock generation.
    #[rusty_serde(default)]
    pub tracing: Option<String>,
    /// Environment-specific data (e.g. a minimal database dump in JSON
    /// string format), passed directly to mock strategies.
    #[rusty_serde(default)]
    pub environment_data: Option<String>,
}

fn default_simulation_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_simulation_model_configuration() -> GenerateContentConfigStub {
    GenerateContentConfigStub {
        thinking_config: Some(Value::Map(vec![
            ("include_thoughts".to_string(), Value::Bool(false)),
            ("thinking_budget".to_string(), Value::UInt(10240)),
        ])),
        ..Default::default()
    }
}

impl Default for EnvironmentSimulationConfig {
    fn default() -> Self {
        EnvironmentSimulationConfig {
            tool_simulation_configs: Vec::new(),
            simulation_model: default_simulation_model(),
            simulation_model_configuration: default_simulation_model_configuration(),
            tracing: None,
            environment_data: None,
        }
    }
}

impl EnvironmentSimulationConfig {
    /// The source's `check_tool_simulation_configs` (`@field_validator`)
    /// plus the `@experimental` gate every type in this module carries —
    /// see the module doc for why the gate is checked once here rather
    /// than on every nested type. Cascades into each tool's own
    /// `validate()`.
    pub fn validate(&self) -> Result<(), EnvironmentSimulationConfigError> {
        check_feature_enabled(FeatureName::EnvironmentSimulation)?;
        if self.tool_simulation_configs.is_empty() {
            return Err(EnvironmentSimulationConfigError::Invalid(
                "tool_simulation_configs must be provided.".to_string(),
            ));
        }
        let mut seen_tool_names = std::collections::HashSet::new();
        for tool_sim_config in &self.tool_simulation_configs {
            if !seen_tool_names.insert(tool_sim_config.tool_name.clone()) {
                return Err(EnvironmentSimulationConfigError::Invalid(format!(
                    "Duplicate tool_name found: {}",
                    tool_sim_config.tool_name
                )));
            }
            tool_sim_config
                .validate()
                .map_err(EnvironmentSimulationConfigError::Invalid)?;
        }
        Ok(())
    }
}

/// Errors from [`EnvironmentSimulationConfig::validate`].
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentSimulationConfigError {
    FeatureNotEnabled(FeatureNotEnabledError),
    Invalid(String),
}

impl std::fmt::Display for EnvironmentSimulationConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentSimulationConfigError::FeatureNotEnabled(e) => write!(f, "{e}"),
            EnvironmentSimulationConfigError::Invalid(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EnvironmentSimulationConfigError {}

impl From<FeatureNotEnabledError> for EnvironmentSimulationConfigError {
    fn from(value: FeatureNotEnabledError) -> Self {
        EnvironmentSimulationConfigError::FeatureNotEnabled(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_features::feature_registry::TemporaryFeatureOverride;

    fn error_config() -> InjectionConfig {
        InjectionConfig {
            injected_error: Some(InjectedError {
                injected_http_error_code: 500,
                error_message: "boom".to_string(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn injection_config_rejects_neither_error_nor_response() {
        let config = InjectionConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn injection_config_rejects_both_error_and_response() {
        let mut config = error_config();
        config.injected_response = Some(BTreeMap::new());
        assert!(config.validate().is_err());
    }

    #[test]
    fn injection_config_rejects_latency_over_120_seconds() {
        let mut config = error_config();
        config.injected_latency_seconds = 121.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn injection_config_accepts_error_only() {
        assert!(error_config().validate().is_ok());
    }

    #[test]
    fn injection_config_accepts_response_only() {
        let config = InjectionConfig {
            injected_response: Some(BTreeMap::new()),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn tool_simulation_config_rejects_unspecified_strategy_with_no_injections() {
        let config = ToolSimulationConfig {
            tool_name: "t".to_string(),
            injection_configs: Vec::new(),
            mock_strategy_type: MockStrategy::Unspecified,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn tool_simulation_config_accepts_a_declared_mock_strategy_with_no_injections() {
        let config = ToolSimulationConfig {
            tool_name: "t".to_string(),
            injection_configs: Vec::new(),
            mock_strategy_type: MockStrategy::ToolSpec,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn tool_simulation_config_cascades_into_injection_config_validation() {
        let config = ToolSimulationConfig {
            tool_name: "t".to_string(),
            injection_configs: vec![InjectionConfig::default()],
            mock_strategy_type: MockStrategy::Unspecified,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn environment_simulation_config_rejects_empty_tool_simulation_configs() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::EnvironmentSimulation, true);
        let config = EnvironmentSimulationConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn environment_simulation_config_rejects_duplicate_tool_names() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::EnvironmentSimulation, true);
        let tool_sim = ToolSimulationConfig {
            tool_name: "dup".to_string(),
            injection_configs: vec![error_config()],
            mock_strategy_type: MockStrategy::Unspecified,
        };
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![tool_sim.clone(), tool_sim],
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EnvironmentSimulationConfigError::Invalid(_))
        ));
    }

    #[test]
    fn environment_simulation_config_rejects_disabled_feature() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::EnvironmentSimulation, false);
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: "t".to_string(),
                injection_configs: vec![error_config()],
                mock_strategy_type: MockStrategy::Unspecified,
            }],
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EnvironmentSimulationConfigError::FeatureNotEnabled(_))
        ));
    }

    #[test]
    fn environment_simulation_config_accepts_a_valid_config() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::EnvironmentSimulation, true);
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: "t".to_string(),
                injection_configs: vec![error_config()],
                mock_strategy_type: MockStrategy::Unspecified,
            }],
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn default_simulation_model_configuration_carries_the_sources_thinking_config() {
        let default_config = default_simulation_model_configuration();
        assert_eq!(
            default_config.thinking_config,
            Some(Value::Map(vec![
                ("include_thoughts".to_string(), Value::Bool(false)),
                ("thinking_budget".to_string(), Value::UInt(10240)),
            ]))
        );
    }
}
