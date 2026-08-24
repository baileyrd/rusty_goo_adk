//! Capability C0608 (`EvalMetric`/`EvalMetricResult`/
//! `EvalMetricResultPerInvocation`) and part of C0612 (`BaseCriterion`/
//! `ToolTrajectoryCriterion`), ported from
//! `google.adk.evaluation.eval_metrics`. See the crate root doc for the
//! criterion types and `Interval`/`MetricValueInfo`/`MetricInfo` this
//! batch doesn't port, and for the disclosed `EvalStatus` wire-format
//! choice.

use rusty_serde::{Deserialize, Serialize};

use crate::eval_case::Invocation;

/// `eval_metrics.EvalStatus`. See the crate root doc for the disclosed
/// wire-format choice (variant name, not the source's underlying int
/// value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EvalStatus {
    Passed,
    Failed,
    #[default]
    NotEvaluated,
}

/// C0612 (partial): `eval_metrics.BaseCriterion`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct BaseCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
}

/// C0612 (partial): `ToolTrajectoryCriterion.MatchType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MatchType {
    #[default]
    Exact,
    InOrder,
    AnyOrder,
}

/// C0612 (partial): `eval_metrics.ToolTrajectoryCriterion`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ToolTrajectoryCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
    #[rusty_serde(default)]
    pub match_type: MatchType,
}

/// C0608: `eval_metrics.EvalMetric` — a metric used to evaluate a
/// particular aspect of an eval case.
///
/// **Adaptation**: `criterion` is the source's
/// `Optional[SerializeAsAny[BaseCriterion]]` — a polymorphic field that
/// can hold any criterion subtype. Since only `ToolTrajectoryCriterion`
/// is a typed struct in this port so far (see the crate root doc), this
/// stays an opaque `Value`; a caller that knows which criterion shape it
/// expects (e.g. [`crate::trajectory_evaluator::TrajectoryEvaluator`])
/// parses it explicitly — the same round-trip-through-JSON pattern the
/// source's own `TrajectoryEvaluator.__init__` already uses
/// (`criterion_type.model_validate(eval_metric.criterion.model_dump())`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalMetric {
    pub metric_name: String,
    #[rusty_serde(default)]
    pub threshold: Option<f64>,
    #[rusty_serde(default)]
    pub criterion: Option<rusty_serde::value::Value>,
    #[rusty_serde(default)]
    pub custom_function_path: Option<String>,
    /// Private: see the crate root doc's note on why this is a
    /// compile-time strengthening of the source's `PrivateAttr` guard,
    /// not merely a port of it.
    #[rusty_serde(skip)]
    config_custom_function_path: Option<String>,
}

impl EvalMetric {
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
            threshold: None,
            criterion: None,
            custom_function_path: None,
            config_custom_function_path: None,
        }
    }

    pub fn with_criterion(mut self, criterion: rusty_serde::value::Value) -> Self {
        self.criterion = Some(criterion);
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// The path declared for this metric in the eval config it was
    /// built from — settable only by code holding a real `EvalMetric`,
    /// never by a deserialized inbound payload.
    pub fn config_custom_function_path(&self) -> Option<&str> {
        self.config_custom_function_path.as_deref()
    }

    pub fn set_config_custom_function_path(&mut self, path: Option<String>) {
        self.config_custom_function_path = path;
    }
}

/// C0608: `eval_metrics._get_metric_threshold` — returns the configured
/// threshold or rejects an incomplete metric.
pub fn get_metric_threshold(eval_metric: &EvalMetric) -> Result<f64, String> {
    if let Some(criterion) = &eval_metric.criterion {
        if let Some(threshold) = criterion.get("threshold").and_then(|v| v.as_f64()) {
            return Ok(threshold);
        }
    }
    if let Some(threshold) = eval_metric.threshold {
        return Ok(threshold);
    }
    Err(format!(
        "Evaluation metric {:?} requires a threshold.",
        eval_metric.metric_name
    ))
}

/// C0608: `eval_metrics.EvalMetricResultDetails`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalMetricResultDetails {
    #[rusty_serde(default)]
    pub rubric_scores: Option<rusty_serde::value::Value>,
}

/// C0608: `eval_metrics.EvalMetricResult` — the actual computed
/// score/value of a particular `EvalMetric`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalMetricResult {
    pub metric_name: String,
    #[rusty_serde(default)]
    pub threshold: Option<f64>,
    #[rusty_serde(default)]
    pub criterion: Option<rusty_serde::value::Value>,
    #[rusty_serde(default)]
    pub custom_function_path: Option<String>,
    #[rusty_serde(default)]
    pub score: Option<f64>,
    pub eval_status: EvalStatus,
    #[rusty_serde(default)]
    pub details: EvalMetricResultDetails,
}

/// C0608: `eval_metrics.EvalMetricResultPerInvocation`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvalMetricResultPerInvocation {
    pub actual_invocation: Invocation,
    #[rusty_serde(default)]
    pub expected_invocation: Option<Invocation>,
    #[rusty_serde(default)]
    pub eval_metric_results: Vec<EvalMetricResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_serde::value::Value;

    #[test]
    fn eval_status_serializes_as_the_variant_name() {
        let json = rusty_serde::json::to_string(&EvalStatus::Passed).unwrap();
        assert_eq!(json, "\"Passed\"");
    }

    #[test]
    fn get_metric_threshold_prefers_the_criterion_threshold() {
        let metric = EvalMetric::new("tool_trajectory_avg_score")
            .with_threshold(0.1)
            .with_criterion(Value::Map(vec![(
                "threshold".to_string(),
                Value::Float(0.9),
            )]));
        assert_eq!(get_metric_threshold(&metric), Ok(0.9));
    }

    #[test]
    fn get_metric_threshold_falls_back_to_the_deprecated_field() {
        let metric = EvalMetric::new("tool_trajectory_avg_score").with_threshold(0.5);
        assert_eq!(get_metric_threshold(&metric), Ok(0.5));
    }

    #[test]
    fn get_metric_threshold_errors_without_either() {
        let metric = EvalMetric::new("tool_trajectory_avg_score");
        assert!(get_metric_threshold(&metric).is_err());
    }

    #[test]
    fn config_custom_function_path_is_not_settable_from_json() {
        let json = r#"{"metricName":"m","configCustomFunctionPath":"evil"}"#;
        let metric: EvalMetric = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(metric.config_custom_function_path(), None);
    }

    #[test]
    fn tool_trajectory_criterion_round_trips_with_camel_case() {
        let criterion = ToolTrajectoryCriterion {
            threshold: 1.0,
            include_intermediate_responses_in_final: false,
            match_type: MatchType::InOrder,
        };
        let json = rusty_serde::json::to_string(&criterion).unwrap();
        assert!(json.contains("\"includeIntermediateResponsesInFinal\""));
        let back: ToolTrajectoryCriterion = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(criterion, back);
    }
}
