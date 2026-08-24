//! C0611: `eval_config.EvalConfig`/`CustomMetricConfig`/
//! `LiveModelConfig`, plus the two free functions the source declares
//! alongside them (`get_evaluation_criteria_or_default`,
//! `get_eval_metrics_from_config`), ported from
//! `google.adk.evaluation.eval_config`.
//!
//! **`criteria`/`custom_metrics`, disclosed narrowing**: the source
//! types `criteria` as `dict[str, Union[Threshold,
//! SerializeAsAny[BaseCriterion]]]` — pydantic parses every value into
//! either a bare `float` or an already-validated `BaseCriterion`
//! instance at `EvalConfig` construction time. This port keeps criterion
//! values as opaque [`Value`]s instead (the same choice already made for
//! [`crate::eval_metrics::EvalMetric::criterion`]), so an invalid
//! criterion object surfaces later — inside
//! [`get_eval_metrics_from_config`], when it's actually parsed into a
//! [`BaseCriterion`] — rather than at `EvalConfig` construction. Same
//! functional rejection, different point in the pipeline. `criteria`
//! also becomes a `HashMap` rather than preserving the source `dict`'s
//! insertion order; [`get_eval_metrics_from_config`] evaluates each
//! metric independently, so the resulting `Vec<EvalMetric>`'s order
//! doesn't affect anything downstream — disclosed rather than fixed,
//! since preserving order would mean a `Vec<(String, Value)>` at every
//! call site instead of a real map.
//!
//! **`user_simulator_config`, disclosed narrowing**: the source types
//! this as a `type`-discriminated union of `LlmBackedUserSimulatorConfig`/
//! `LlmAudioUserSimulatorConfig` (`Annotated[Union[...],
//! Field(discriminator="type")]`) — both still `REQUIRED` (C0628/C0630).
//! This port keeps it an opaque `Value` instead. The legacy-default-
//! injecting `@model_validator(mode="before")` is still ported, though —
//! see [`EvalConfig::normalize_user_simulator_config`] — since it only
//! ever touches the raw `type` key, not the (still unbuilt) typed
//! variants it would eventually discriminate into.
//!
//! **`CodeConfig`, a narrow local port, disclosed**: `CustomMetricConfig
//! .code_config`'s real type is `agents.common_configs.CodeConfig` — the
//! full YAML agent-config/reflection pipeline (`C0348`, still
//! `REQUIRED`) that type belongs to. `adk-eval` deliberately sits at the
//! bottom of this workspace's crate graph (see the crate root doc on
//! `session_details`) and can't depend on wherever C0348 eventually
//! lands, so this module ports just the one field `CustomMetricConfig`
//! actually reads (`name: String`) as its own local `CodeConfig`, rather
//! than the source's `@experimental`-gated, `AgentRefConfig`-adjacent
//! type. When C0348 lands, `agents::common_configs::CodeConfig` will be
//! a distinct (structurally identical) type from this one — that's a
//! disclosed, permanent duplication forced by the crate-graph position,
//! not an oversight to reconcile later.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_LIVE_TIMEOUT_SECONDS;
use crate::eval_metrics::{BaseCriterion, EvalMetric, MetricInfo};

const LEGACY_DEFAULT_USER_SIMULATOR_TYPE: &str = "llm_backed";

/// A narrow local port of `agents.common_configs.CodeConfig` — see this
/// module's doc for why it's not the real, still-unbuilt type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeConfig {
    pub name: String,
}

/// C0611: `eval_config.CustomMetricConfig` — configuration for a custom
/// metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct CustomMetricConfig {
    pub code_config: CodeConfig,
    #[rusty_serde(default)]
    pub metric_info: Option<MetricInfo>,
    #[rusty_serde(default)]
    pub description: String,
}

fn default_timeout_seconds() -> u64 {
    DEFAULT_LIVE_TIMEOUT_SECONDS
}

/// C0611: `eval_config.LiveModelConfig` — configuration for evaluating
/// models in Live (bidirectional streaming) mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LiveModelConfig {
    #[rusty_serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

impl Default for LiveModelConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: DEFAULT_LIVE_TIMEOUT_SECONDS,
        }
    }
}

/// C0611: `eval_config.EvalConfig` — configurations needed to run an
/// eval: metrics, their thresholds, and other properties.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalConfig {
    #[rusty_serde(default)]
    pub criteria: HashMap<String, Value>,
    #[rusty_serde(default)]
    pub custom_metrics: Option<HashMap<String, CustomMetricConfig>>,
    #[rusty_serde(default)]
    pub user_simulator_config: Option<Value>,
    #[rusty_serde(default)]
    pub live_model_config: Option<LiveModelConfig>,
}

impl EvalConfig {
    /// `eval_config.EvalConfig._inject_default_user_simulator_type` — a
    /// `@model_validator(mode="before")` in the source, so it runs
    /// automatically on every construction including deserialization.
    /// This port keeps `user_simulator_config` plainly deserializable and
    /// exposes the same check as an explicit method instead, the same
    /// "split validator that mutates" choice already made for
    /// `adk_tools::skills_models::Frontmatter::normalize_name` — call
    /// this once right after deserializing an `EvalConfig` whose
    /// `user_simulator_config` might predate the `type` discriminator.
    ///
    /// A missing `type` key, or one present with a `null` value (e.g.
    /// from dumping a base config with no discriminator set), are both
    /// treated as "no discriminator supplied" and defaulted to
    /// `"llm_backed"` — matching the source's own handling of both cases.
    pub fn normalize_user_simulator_config(&mut self) {
        let Some(Value::Map(entries)) = &mut self.user_simulator_config else {
            return;
        };
        let has_explicit_type = entries
            .iter()
            .any(|(key, value)| key == "type" && !value.is_null());
        if has_explicit_type {
            return;
        }
        entries.retain(|(key, _)| key != "type");
        entries.push((
            "type".to_string(),
            Value::String(LEGACY_DEFAULT_USER_SIMULATOR_TYPE.to_string()),
        ));
    }
}

fn default_eval_config() -> EvalConfig {
    let mut criteria = HashMap::new();
    criteria.insert("tool_trajectory_avg_score".to_string(), Value::Float(1.0));
    criteria.insert("response_match_score".to_string(), Value::Float(0.8));
    EvalConfig {
        criteria,
        ..EvalConfig::default()
    }
}

/// C0611: `eval_config.get_evaluation_criteria_or_default` — returns the
/// `EvalConfig` read from `eval_config_file_path`, if the path is given
/// and exists; otherwise a default config
/// (`tool_trajectory_avg_score: 1.0`, `response_match_score: 0.8`).
///
/// Disclosed: the source raises an uncaught `pydantic.ValidationError`
/// if the file exists but its content doesn't parse as a valid
/// `EvalConfig`; this port surfaces the same failure as `Err` instead of
/// panicking.
pub fn get_evaluation_criteria_or_default(
    eval_config_file_path: Option<&str>,
) -> Result<EvalConfig, String> {
    if let Some(path) = eval_config_file_path {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
            return rusty_serde::json::from_str(&content).map_err(|error| error.to_string());
        }
    }
    Ok(default_eval_config())
}

/// C0611: `eval_config.get_eval_metrics_from_config` — returns a list of
/// `EvalMetric`s mapped from an `EvalConfig`.
pub fn get_eval_metrics_from_config(eval_config: &EvalConfig) -> Result<Vec<EvalMetric>, String> {
    let mut eval_metric_list = Vec::new();
    for (metric_name, criterion_value) in &eval_config.criteria {
        let custom_function_path = eval_config
            .custom_metrics
            .as_ref()
            .and_then(|custom_metrics| custom_metrics.get(metric_name))
            .map(|config| config.code_config.name.clone());

        let (threshold, criterion) = match criterion_value {
            Value::Int(_) | Value::UInt(_) | Value::Float(_) => {
                let threshold = criterion_value
                    .as_f64()
                    .expect("numeric Value variant always has an f64 representation");
                let base_criterion = BaseCriterion {
                    threshold,
                    include_intermediate_responses_in_final: false,
                };
                let json =
                    rusty_serde::json::to_string(&base_criterion).map_err(|e| e.to_string())?;
                let criterion: Value =
                    rusty_serde::json::from_str(&json).map_err(|e| e.to_string())?;
                (threshold, criterion)
            }
            Value::Map(_) => {
                let json =
                    rusty_serde::json::to_string(criterion_value).map_err(|e| e.to_string())?;
                let base_criterion: BaseCriterion =
                    rusty_serde::json::from_str(&json).map_err(|_| {
                        format!("Unexpected criterion type for metric {metric_name:?}.")
                    })?;
                (base_criterion.threshold, criterion_value.clone())
            }
            other => {
                return Err(format!(
                    "Unexpected criterion type. {other:?} not supported."
                ));
            }
        };

        let mut eval_metric = EvalMetric::new(metric_name.clone())
            .with_threshold(threshold)
            .with_criterion(criterion);
        eval_metric.custom_function_path = custom_function_path.clone();
        eval_metric.set_config_custom_function_path(custom_function_path);
        eval_metric_list.push(eval_metric);
    }
    Ok(eval_metric_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_model_config_default_matches_the_source() {
        assert_eq!(LiveModelConfig::default().timeout_seconds, 300);
    }

    #[test]
    fn live_model_config_deserializes_with_a_default_from_an_empty_object() {
        let config: LiveModelConfig = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(config, LiveModelConfig::default());
    }

    #[test]
    fn custom_metric_config_defaults_description_and_metric_info() {
        let json = r#"{"codeConfig":{"name":"path.to.fn"}}"#;
        let config: CustomMetricConfig = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(config.code_config.name, "path.to.fn");
        assert_eq!(config.description, "");
        assert_eq!(config.metric_info, None);
    }

    #[test]
    fn normalize_injects_the_legacy_type_when_missing() {
        let mut config = EvalConfig {
            user_simulator_config: Some(Value::Map(vec![(
                "someField".to_string(),
                Value::String("x".to_string()),
            )])),
            ..EvalConfig::default()
        };
        config.normalize_user_simulator_config();
        assert_eq!(
            config.user_simulator_config.unwrap().get("type"),
            Some(&Value::String("llm_backed".to_string()))
        );
    }

    #[test]
    fn normalize_injects_the_legacy_type_when_explicitly_null() {
        let mut config = EvalConfig {
            user_simulator_config: Some(Value::Map(vec![("type".to_string(), Value::Null)])),
            ..EvalConfig::default()
        };
        config.normalize_user_simulator_config();
        assert_eq!(
            config.user_simulator_config.unwrap().get("type"),
            Some(&Value::String("llm_backed".to_string()))
        );
    }

    #[test]
    fn normalize_leaves_an_explicit_type_untouched() {
        let mut config = EvalConfig {
            user_simulator_config: Some(Value::Map(vec![(
                "type".to_string(),
                Value::String("llm_audio".to_string()),
            )])),
            ..EvalConfig::default()
        };
        config.normalize_user_simulator_config();
        assert_eq!(
            config.user_simulator_config.unwrap().get("type"),
            Some(&Value::String("llm_audio".to_string()))
        );
    }

    #[test]
    fn normalize_is_a_no_op_without_a_user_simulator_config() {
        let mut config = EvalConfig::default();
        config.normalize_user_simulator_config();
        assert_eq!(config.user_simulator_config, None);
    }

    #[test]
    fn get_evaluation_criteria_or_default_returns_the_default_without_a_path() {
        let config = get_evaluation_criteria_or_default(None).unwrap();
        assert_eq!(
            config.criteria.get("tool_trajectory_avg_score"),
            Some(&Value::Float(1.0))
        );
        assert_eq!(
            config.criteria.get("response_match_score"),
            Some(&Value::Float(0.8))
        );
    }

    #[test]
    fn get_evaluation_criteria_or_default_returns_the_default_for_a_missing_file() {
        let config = get_evaluation_criteria_or_default(Some("/no/such/eval_config.json")).unwrap();
        assert_eq!(config.criteria.len(), 2);
    }

    #[test]
    fn get_evaluation_criteria_or_default_reads_a_real_file() {
        let dir = std::env::temp_dir().join(format!("adk_eval_config_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("eval_config.json");
        std::fs::write(&path, r#"{"criteria":{"response_match_score":0.9}}"#).unwrap();

        let config = get_evaluation_criteria_or_default(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(
            config.criteria.get("response_match_score"),
            Some(&Value::Float(0.9))
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_eval_metrics_from_config_builds_a_base_criterion_for_a_bare_number() {
        let mut criteria = HashMap::new();
        criteria.insert("tool_trajectory_avg_score".to_string(), Value::Float(1.0));
        let config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };

        let metrics = get_eval_metrics_from_config(&config).unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].metric_name, "tool_trajectory_avg_score");
        assert_eq!(metrics[0].threshold, Some(1.0));
        assert_eq!(
            metrics[0]
                .criterion
                .as_ref()
                .and_then(|c| c.get("threshold"))
                .and_then(|v| v.as_f64()),
            Some(1.0)
        );
    }

    #[test]
    fn get_eval_metrics_from_config_reads_the_threshold_off_an_object_criterion() {
        let mut criteria = HashMap::new();
        criteria.insert(
            "final_response_match_v2".to_string(),
            Value::Map(vec![("threshold".to_string(), Value::Float(0.5))]),
        );
        let config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };

        let metrics = get_eval_metrics_from_config(&config).unwrap();
        assert_eq!(metrics[0].threshold, Some(0.5));
    }

    #[test]
    fn get_eval_metrics_from_config_wires_up_the_custom_function_path() {
        let mut criteria = HashMap::new();
        criteria.insert("my_custom_metric".to_string(), Value::Float(0.5));
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert(
            "my_custom_metric".to_string(),
            CustomMetricConfig {
                code_config: CodeConfig {
                    name: "path.to.my.metric".to_string(),
                },
                metric_info: None,
                description: String::new(),
            },
        );
        let config = EvalConfig {
            criteria,
            custom_metrics: Some(custom_metrics),
            ..EvalConfig::default()
        };

        let metrics = get_eval_metrics_from_config(&config).unwrap();
        assert_eq!(
            metrics[0].custom_function_path,
            Some("path.to.my.metric".to_string())
        );
        assert_eq!(
            metrics[0].config_custom_function_path(),
            Some("path.to.my.metric")
        );
    }

    #[test]
    fn get_eval_metrics_from_config_rejects_a_non_numeric_non_object_criterion() {
        let mut criteria = HashMap::new();
        criteria.insert("bad_metric".to_string(), Value::String("nope".to_string()));
        let config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };

        assert!(get_eval_metrics_from_config(&config).is_err());
    }

    #[test]
    fn get_eval_metrics_from_config_rejects_an_object_missing_threshold() {
        let mut criteria = HashMap::new();
        criteria.insert(
            "bad_metric".to_string(),
            Value::Map(vec![("notThreshold".to_string(), Value::Float(1.0))]),
        );
        let config = EvalConfig {
            criteria,
            ..EvalConfig::default()
        };

        assert!(get_eval_metrics_from_config(&config).is_err());
    }
}
