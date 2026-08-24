//! Capability C0608 (`EvalMetric`/`EvalMetricResult`/
//! `EvalMetricResultPerInvocation`) and C0612 (`BaseCriterion`/
//! `ToolTrajectoryCriterion`/`JudgeModelOptions`/`LlmAsAJudgeCriterion`/
//! `RubricsBasedCriterion`/`HallucinationsCriterion`/
//! `LlmBackedUserSimulatorCriterion`), ported from
//! `google.adk.evaluation.eval_metrics`. See the crate root doc for the
//! disclosed `EvalStatus` wire-format choice.
//!
//! **C0612, now complete**: every criterion subtype the source declares
//! is ported. All of them (`LlmAsAJudgeCriterion` and everything that
//! extends it) exist only to *configure* an LLM-judge metric — no LLM
//! call happens inside any of these types themselves, so, like
//! `Rubric`/`RubricScore` (C0607, DONE) before them, they don't actually
//! need the still-missing `LlmAsJudge` harness (C0600's other half) to be
//! pure, useful data models. The source's class inheritance
//! (`LlmAsAJudgeCriterion(BaseCriterion)`, `RubricsBasedCriterion
//! (BaseCriterion)`, etc.) flattens into each struct declaring its own
//! full field set directly, the same choice already made for
//! `ToolTrajectoryCriterion` — no Rust struct inheritance exists, and a
//! shared-fields-via-composition alternative would change every field
//! access from `criterion.threshold` to `criterion.base.threshold`
//! everywhere a caller reads one, for no behavioral benefit.
//!
//! **`extra="allow"`, inherited**: every criterion subtype inherits
//! `BaseCriterion`'s `model_config` (pydantic subclasses inherit their
//! parent's `ConfigDict` unless they override it, and none of these do),
//! so the same `extra="allow"` narrowing already disclosed for
//! `BaseCriterion`/`ToolTrajectoryCriterion` (unknown fields don't reject
//! the payload but aren't captured/preserved either) applies to all of
//! them.
//!
//! **`JudgeModelOptions.judge_model_config`, disclosed narrowing**: the
//! source types this `Optional[google.genai.types.GenerateContentConfig]`.
//! `adk-eval` deliberately doesn't depend on `adk-models` (see the crate
//! root doc on why `session_details` stays opaque for the same reason),
//! so this stays an opaque `Value` — the same disclosed-placeholder
//! pattern `adk_models::llm_request::GenerateContentConfigStub` itself
//! already is one layer up, just without even that narrowed shape here.
//!
//! **`JudgeModelOptions.parallelism_limit`'s `Field(ge=1)`, adapted**:
//! the source enforces this constraint automatically at construction
//! (pydantic). This port keeps the field plainly `pub`/deserializable
//! and exposes the check as `JudgeModelOptions::validate()` instead — the
//! same "plain fields + explicit `validate()`" pattern used throughout
//! this crate (e.g. `eval_case::EvalCase::validate`,
//! `adk_tools::skills_models::Frontmatter::validate`).

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::eval_case::Invocation;
use crate::eval_rubrics::Rubric;

fn default_judge_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_num_samples() -> i64 {
    5
}

fn default_parallelism_limit() -> i64 {
    1
}

/// C0612: `eval_metrics.JudgeModelOptions` — options for an eval
/// metric's judge model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct JudgeModelOptions {
    #[rusty_serde(default = "default_judge_model")]
    pub judge_model: String,
    #[rusty_serde(default)]
    pub judge_model_config: Option<Value>,
    #[rusty_serde(default = "default_num_samples")]
    pub num_samples: i64,
    #[rusty_serde(default = "default_parallelism_limit")]
    pub parallelism_limit: i64,
}

impl Default for JudgeModelOptions {
    fn default() -> Self {
        Self {
            judge_model: default_judge_model(),
            judge_model_config: None,
            num_samples: default_num_samples(),
            parallelism_limit: default_parallelism_limit(),
        }
    }
}

impl JudgeModelOptions {
    /// `Field(ge=1)` on `parallelism_limit` — see this module's doc for
    /// why this is an explicit check rather than automatic.
    pub fn validate(&self) -> Result<(), String> {
        if self.parallelism_limit < 1 {
            return Err("parallelism_limit must be greater than or equal to 1".to_string());
        }
        Ok(())
    }
}

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

/// C0612: `eval_metrics.LlmAsAJudgeCriterion` — criterion when using an
/// LLM-as-a-judge metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LlmAsAJudgeCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
    #[rusty_serde(default)]
    pub judge_model_options: JudgeModelOptions,
}

/// C0612: `eval_metrics.RubricsBasedCriterion` — criterion when using a
/// rubric-based metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RubricsBasedCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
    #[rusty_serde(default)]
    pub judge_model_options: JudgeModelOptions,
    #[rusty_serde(default)]
    pub rubrics: Vec<Rubric>,
}

/// C0612: `eval_metrics.HallucinationsCriterion` — criterion to use when
/// evaluating an agent's response for hallucinations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct HallucinationsCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
    #[rusty_serde(default)]
    pub judge_model_options: JudgeModelOptions,
    #[rusty_serde(default)]
    pub evaluate_intermediate_nl_responses: bool,
}

fn default_stop_signal() -> String {
    "</finished>".to_string()
}

/// C0612: `eval_metrics.LlmBackedUserSimulatorCriterion` — criterion for
/// LLM-backed user-simulator evaluators. Extends `LlmAsAJudgeCriterion`
/// in the source; flattened here the same way every other criterion
/// subtype is (see this module's doc).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LlmBackedUserSimulatorCriterion {
    pub threshold: f64,
    #[rusty_serde(default)]
    pub include_intermediate_responses_in_final: bool,
    #[rusty_serde(default)]
    pub judge_model_options: JudgeModelOptions,
    #[rusty_serde(default = "default_stop_signal")]
    pub stop_signal: String,
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

/// `eval_metrics.PrebuiltMetrics` — the metric-name enum every metric in
/// this port (and every not-yet-built LLM-judge metric) is keyed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "snake_case")]
pub enum PrebuiltMetrics {
    ToolTrajectoryAvgScore,
    ResponseEvaluationScore,
    ResponseMatchScore,
    SafetyV1,
    FinalResponseMatchV2,
    RubricBasedFinalResponseQualityV1,
    HallucinationsV1,
    RubricBasedToolUseQualityV1,
    PerTurnUserSimulatorQualityV1,
    MultiTurnTaskSuccessV1,
    MultiTurnTrajectoryQualityV1,
    MultiTurnToolUseQualityV1,
    RubricBasedMultiTurnTrajectoryQualityV1,
}

impl PrebuiltMetrics {
    /// The wire-string value each source `PrebuiltMetrics` enum member
    /// carries explicitly (e.g. `TOOL_TRAJECTORY_AVG_SCORE =
    /// "tool_trajectory_avg_score"`), matched by hand rather than trusting
    /// this port's own `snake_case` rename to independently land on the
    /// same string for every variant (it does, verified by
    /// `prebuilt_metrics_as_str_matches_the_derived_wire_value`, but a
    /// literal match here doesn't depend on that derive continuing to
    /// agree with it).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ToolTrajectoryAvgScore => "tool_trajectory_avg_score",
            Self::ResponseEvaluationScore => "response_evaluation_score",
            Self::ResponseMatchScore => "response_match_score",
            Self::SafetyV1 => "safety_v1",
            Self::FinalResponseMatchV2 => "final_response_match_v2",
            Self::RubricBasedFinalResponseQualityV1 => "rubric_based_final_response_quality_v1",
            Self::HallucinationsV1 => "hallucinations_v1",
            Self::RubricBasedToolUseQualityV1 => "rubric_based_tool_use_quality_v1",
            Self::PerTurnUserSimulatorQualityV1 => "per_turn_user_simulator_quality_v1",
            Self::MultiTurnTaskSuccessV1 => "multi_turn_task_success_v1",
            Self::MultiTurnTrajectoryQualityV1 => "multi_turn_trajectory_quality_v1",
            Self::MultiTurnToolUseQualityV1 => "multi_turn_tool_use_quality_v1",
            Self::RubricBasedMultiTurnTrajectoryQualityV1 => {
                "rubric_based_multi_turn_trajectory_quality_v1"
            }
        }
    }
}

/// Part of C0604/C0612: `eval_metrics.Interval` — a range of numeric
/// values, e.g. `[0, 1]` or `(2, 3)` or `[-1, 6)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Interval {
    pub min_value: f64,
    #[rusty_serde(default)]
    pub open_at_min: bool,
    pub max_value: f64,
    #[rusty_serde(default)]
    pub open_at_max: bool,
}

impl Interval {
    /// A closed `[min_value, max_value]` interval — the shape every
    /// prebuilt metric in this port actually uses.
    pub fn closed(min_value: f64, max_value: f64) -> Self {
        Self {
            min_value,
            open_at_min: false,
            max_value,
            open_at_max: false,
        }
    }
}

/// Part of C0604/C0612: `eval_metrics.MetricValueInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct MetricValueInfo {
    #[rusty_serde(default)]
    pub interval: Option<Interval>,
}

/// Part of C0604/C0612: `eval_metrics.MetricInfo`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct MetricInfo {
    pub metric_name: String,
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub metric_value_info: MetricValueInfo,
}

/// C0604: `eval_metrics.MetricInfoProvider` — interface for providing
/// `MetricInfo`.
pub trait MetricInfoProvider {
    /// Returns `MetricInfo` for a given metric. The source's
    /// `ResponseEvaluatorMetricInfoProvider` is the one implementor
    /// that's actually fallible (constructed with a caller-supplied
    /// metric name it may not recognize), so this returns a `Result`
    /// rather than the source's `MetricInfo` (which just raises
    /// `ValueError` on the same case).
    fn get_metric_info(&self) -> Result<MetricInfo, String>;
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

    #[test]
    fn prebuilt_metrics_as_str_matches_the_derived_wire_value() {
        let all = [
            PrebuiltMetrics::ToolTrajectoryAvgScore,
            PrebuiltMetrics::ResponseEvaluationScore,
            PrebuiltMetrics::ResponseMatchScore,
            PrebuiltMetrics::SafetyV1,
            PrebuiltMetrics::FinalResponseMatchV2,
            PrebuiltMetrics::RubricBasedFinalResponseQualityV1,
            PrebuiltMetrics::HallucinationsV1,
            PrebuiltMetrics::RubricBasedToolUseQualityV1,
            PrebuiltMetrics::PerTurnUserSimulatorQualityV1,
            PrebuiltMetrics::MultiTurnTaskSuccessV1,
            PrebuiltMetrics::MultiTurnTrajectoryQualityV1,
            PrebuiltMetrics::MultiTurnToolUseQualityV1,
            PrebuiltMetrics::RubricBasedMultiTurnTrajectoryQualityV1,
        ];
        for metric in all {
            let derived = rusty_serde::json::to_string(&metric).unwrap();
            assert_eq!(derived, format!("\"{}\"", metric.as_str()));
        }
    }

    #[test]
    fn interval_closed_defaults_to_a_closed_range() {
        let interval = Interval::closed(0.0, 1.0);
        assert!(!interval.open_at_min);
        assert!(!interval.open_at_max);
    }

    #[test]
    fn metric_info_round_trips_through_json_with_camel_case() {
        let info = MetricInfo {
            metric_name: PrebuiltMetrics::ToolTrajectoryAvgScore.as_str().to_string(),
            description: Some("a description".to_string()),
            metric_value_info: MetricValueInfo {
                interval: Some(Interval::closed(0.0, 1.0)),
            },
        };
        let json = rusty_serde::json::to_string(&info).unwrap();
        assert!(json.contains("\"metricValueInfo\""));
        let back: MetricInfo = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn judge_model_options_defaults_match_the_source() {
        let options = JudgeModelOptions::default();
        assert_eq!(options.judge_model, "gemini-2.5-flash");
        assert_eq!(options.judge_model_config, None);
        assert_eq!(options.num_samples, 5);
        assert_eq!(options.parallelism_limit, 1);
        assert!(options.validate().is_ok());
    }

    #[test]
    fn judge_model_options_deserializes_with_defaults_from_an_empty_object() {
        let options: JudgeModelOptions = rusty_serde::json::from_str("{}").unwrap();
        assert_eq!(options, JudgeModelOptions::default());
    }

    #[test]
    fn judge_model_options_rejects_a_parallelism_limit_below_one() {
        let options = JudgeModelOptions {
            parallelism_limit: 0,
            ..JudgeModelOptions::default()
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn llm_as_a_judge_criterion_round_trips_with_camel_case() {
        let criterion = LlmAsAJudgeCriterion {
            threshold: 0.5,
            include_intermediate_responses_in_final: true,
            judge_model_options: JudgeModelOptions::default(),
        };
        let json = rusty_serde::json::to_string(&criterion).unwrap();
        assert!(json.contains("\"judgeModelOptions\""));
        let back: LlmAsAJudgeCriterion = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(criterion, back);
    }

    #[test]
    fn rubrics_based_criterion_defaults_to_no_rubrics() {
        let json = r#"{"threshold":0.5}"#;
        let criterion: RubricsBasedCriterion = rusty_serde::json::from_str(json).unwrap();
        assert!(criterion.rubrics.is_empty());
        assert_eq!(criterion.judge_model_options, JudgeModelOptions::default());
    }

    #[test]
    fn hallucinations_criterion_defaults_evaluate_intermediate_to_false() {
        let json = r#"{"threshold":0.5}"#;
        let criterion: HallucinationsCriterion = rusty_serde::json::from_str(json).unwrap();
        assert!(!criterion.evaluate_intermediate_nl_responses);
    }

    #[test]
    fn llm_backed_user_simulator_criterion_defaults_the_stop_signal() {
        let json = r#"{"threshold":0.5}"#;
        let criterion: LlmBackedUserSimulatorCriterion = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(criterion.stop_signal, "</finished>");
    }
}
