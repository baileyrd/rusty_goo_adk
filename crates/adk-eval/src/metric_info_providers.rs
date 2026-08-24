//! C0604: the 12 `MetricInfoProvider` implementors, ported from
//! `google.adk.evaluation.metric_info_providers` — 13 metrics' worth of
//! static catalog metadata (`ResponseEvaluatorMetricInfoProvider` alone
//! covers 2: `response_evaluation_score` and `response_match_score`).

use crate::eval_metrics::{
    Interval, MetricInfo, MetricInfoProvider, MetricValueInfo, PrebuiltMetrics,
};

fn zero_to_one_info(metric: PrebuiltMetrics, description: &str) -> MetricInfo {
    MetricInfo {
        metric_name: metric.as_str().to_string(),
        description: Some(description.to_string()),
        metric_value_info: MetricValueInfo {
            interval: Some(Interval::closed(0.0, 1.0)),
        },
    }
}

/// `metric_info_providers.TrajectoryEvaluatorMetricInfoProvider`.
pub struct TrajectoryEvaluatorMetricInfoProvider;

impl MetricInfoProvider for TrajectoryEvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::ToolTrajectoryAvgScore,
            "This metric compares two tool call trajectories (expected vs. actual) for the \
             same user interaction. It performs an exact match on the tool name and arguments \
             for each step in the trajectory. A score of 1.0 indicates a perfect match, while \
             0.0 indicates a mismatch. Higher values are better.",
        ))
    }
}

/// `metric_info_providers.ResponseEvaluatorMetricInfoProvider` — the one
/// implementor constructed with a caller-supplied metric name it may not
/// recognize, hence the only one whose `get_metric_info` can actually
/// fail.
pub struct ResponseEvaluatorMetricInfoProvider {
    metric_name: String,
}

impl ResponseEvaluatorMetricInfoProvider {
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
        }
    }
}

impl MetricInfoProvider for ResponseEvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        if self.metric_name == PrebuiltMetrics::ResponseEvaluationScore.as_str() {
            Ok(MetricInfo {
                metric_name: PrebuiltMetrics::ResponseEvaluationScore
                    .as_str()
                    .to_string(),
                description: Some(
                    "This metric evaluates how coherent agent's response was. Value range of \
                     this metric is [1,5], with values closer to 5 more desirable."
                        .to_string(),
                ),
                metric_value_info: MetricValueInfo {
                    interval: Some(Interval::closed(1.0, 5.0)),
                },
            })
        } else if self.metric_name == PrebuiltMetrics::ResponseMatchScore.as_str() {
            Ok(zero_to_one_info(
                PrebuiltMetrics::ResponseMatchScore,
                "This metric evaluates if the agent's final response matches a golden/expected \
                 final response using Rouge_1 metric. Value range for this metric is [0,1], \
                 with values closer to 1 more desirable.",
            ))
        } else {
            Err(format!("`{}` is not supported.", self.metric_name))
        }
    }
}

/// `metric_info_providers.SafetyEvaluatorV1MetricInfoProvider`.
pub struct SafetyEvaluatorV1MetricInfoProvider;

impl MetricInfoProvider for SafetyEvaluatorV1MetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::SafetyV1,
            "This metric evaluates the safety (harmlessness) of an Agent's Response. Value \
             range of the metric is [0, 1], with values closer to 1 to be more desirable \
             (safe).",
        ))
    }
}

/// `metric_info_providers.MultiTurnTaskSuccessV1MetricInfoProvider`.
pub struct MultiTurnTaskSuccessV1MetricInfoProvider;

impl MetricInfoProvider for MultiTurnTaskSuccessV1MetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::MultiTurnTaskSuccessV1,
            "Evaluates if the agent was able to achieve the goal or goals of the conversation. \
             Value range of the metric is [0, 1], with values closer to 1 to be more desirable \
             (safe).",
        ))
    }
}

/// `metric_info_providers.MultiTurnTrajectoryQualityV1MetricInfoProvider`.
pub struct MultiTurnTrajectoryQualityV1MetricInfoProvider;

impl MetricInfoProvider for MultiTurnTrajectoryQualityV1MetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::MultiTurnTrajectoryQualityV1,
            "Evaluates the overall trajectory of the conversation. Note that this metric is \
             different from `Multi-Turn Overall Task Success`, in the sense that task success \
             only concerns itself with the goal of whether the success was achieved or not. \
             How that was achieved is not its concern. This metric on the other hand does care \
             about the path that agent took to achieve the goal. This is a reference free \
             metric. Value range of the metric is [0, 1], with values closer to 1 to be more \
             desirable (safe).",
        ))
    }
}

/// `metric_info_providers.MultiTurnToolUseQualityV1MetricInfoProvider`.
pub struct MultiTurnToolUseQualityV1MetricInfoProvider;

impl MetricInfoProvider for MultiTurnToolUseQualityV1MetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::MultiTurnToolUseQualityV1,
            "Evaluates the function calls made during a multi-turn conversation. This is a \
             reference free metric. Value range of the metric is [0, 1], with values closer to \
             1 to be more desirable (safe).",
        ))
    }
}

/// `metric_info_providers.FinalResponseMatchV2EvaluatorMetricInfoProvider`.
pub struct FinalResponseMatchV2EvaluatorMetricInfoProvider;

impl MetricInfoProvider for FinalResponseMatchV2EvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::FinalResponseMatchV2,
            "This metric evaluates if the agent's final response matches a golden/expected \
             final response using LLM as a judge. Value range for this metric is [0,1], with \
             values closer to 1 more desirable.",
        ))
    }
}

/// `metric_info_providers.RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider`.
pub struct RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider;

impl MetricInfoProvider for RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::RubricBasedFinalResponseQualityV1,
            "This metric assess if the agent's final response against a set of rubrics using \
             LLM as a judge. Value range for this metric is [0,1], with values closer to 1 \
             more desirable.",
        ))
    }
}

/// `metric_info_providers.HallucinationsV1EvaluatorMetricInfoProvider`.
pub struct HallucinationsV1EvaluatorMetricInfoProvider;

impl MetricInfoProvider for HallucinationsV1EvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::HallucinationsV1,
            "This metric assesses whether a model response contains any false, contradictory, \
             or unsupported claims using a LLM as judge. Value range for this metric is [0,1], \
             with values closer to 1 more desirable.",
        ))
    }
}

/// `metric_info_providers.RubricBasedToolUseV1EvaluatorMetricInfoProvider`.
pub struct RubricBasedToolUseV1EvaluatorMetricInfoProvider;

impl MetricInfoProvider for RubricBasedToolUseV1EvaluatorMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::RubricBasedToolUseQualityV1,
            "This metric assess if the agent's usage of tools against a set of rubrics using \
             LLM as a judge. Value range for this metric is [0,1], with values closer to 1 \
             more desirable.",
        ))
    }
}

/// `metric_info_providers.PerTurnUserSimulatorQualityV1MetricInfoProvider`.
pub struct PerTurnUserSimulatorQualityV1MetricInfoProvider;

impl MetricInfoProvider for PerTurnUserSimulatorQualityV1MetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::PerTurnUserSimulatorQualityV1,
            "This metric evaluates if the user messages generated by a user simulator follow \
             the given conversation scenario. It validates each message separately. The \
             resulting metric computes the percentage of user messages that we mark as valid. \
             The value range for this metric is [0,1], with values closer to 1 more desirable. ",
        ))
    }
}

/// `metric_info_providers.RubricBasedMultiTurnTrajectoryMetricInfoProvider`.
pub struct RubricBasedMultiTurnTrajectoryMetricInfoProvider;

impl MetricInfoProvider for RubricBasedMultiTurnTrajectoryMetricInfoProvider {
    fn get_metric_info(&self) -> Result<MetricInfo, String> {
        Ok(zero_to_one_info(
            PrebuiltMetrics::RubricBasedMultiTurnTrajectoryQualityV1,
            "This metric evaluates the agent's multi-turn trajectory against a set of \
             user-provided rubrics using an LLM as a judge. Value range for this metric is \
             [0,1], with values closer to 1 more desirable.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_evaluator_provider_returns_the_right_metric_name() {
        let info = TrajectoryEvaluatorMetricInfoProvider
            .get_metric_info()
            .unwrap();
        assert_eq!(info.metric_name, "tool_trajectory_avg_score");
        assert_eq!(
            info.metric_value_info.interval,
            Some(Interval::closed(0.0, 1.0))
        );
    }

    #[test]
    fn response_evaluator_provider_resolves_response_evaluation_score() {
        let provider = ResponseEvaluatorMetricInfoProvider::new("response_evaluation_score");
        let info = provider.get_metric_info().unwrap();
        assert_eq!(info.metric_name, "response_evaluation_score");
        assert_eq!(
            info.metric_value_info.interval,
            Some(Interval::closed(1.0, 5.0))
        );
    }

    #[test]
    fn response_evaluator_provider_resolves_response_match_score() {
        let provider = ResponseEvaluatorMetricInfoProvider::new("response_match_score");
        let info = provider.get_metric_info().unwrap();
        assert_eq!(info.metric_name, "response_match_score");
        assert_eq!(
            info.metric_value_info.interval,
            Some(Interval::closed(0.0, 1.0))
        );
    }

    #[test]
    fn response_evaluator_provider_errors_on_an_unsupported_metric_name() {
        let provider = ResponseEvaluatorMetricInfoProvider::new("not_a_real_metric");
        assert!(provider.get_metric_info().is_err());
    }

    #[test]
    fn every_remaining_provider_returns_a_zero_to_one_interval_with_the_right_name() {
        let cases: Vec<(&str, Box<dyn MetricInfoProvider>)> = vec![
            ("safety_v1", Box::new(SafetyEvaluatorV1MetricInfoProvider)),
            (
                "multi_turn_task_success_v1",
                Box::new(MultiTurnTaskSuccessV1MetricInfoProvider),
            ),
            (
                "multi_turn_trajectory_quality_v1",
                Box::new(MultiTurnTrajectoryQualityV1MetricInfoProvider),
            ),
            (
                "multi_turn_tool_use_quality_v1",
                Box::new(MultiTurnToolUseQualityV1MetricInfoProvider),
            ),
            (
                "final_response_match_v2",
                Box::new(FinalResponseMatchV2EvaluatorMetricInfoProvider),
            ),
            (
                "rubric_based_final_response_quality_v1",
                Box::new(RubricBasedFinalResponseQualityV1EvaluatorMetricInfoProvider),
            ),
            (
                "hallucinations_v1",
                Box::new(HallucinationsV1EvaluatorMetricInfoProvider),
            ),
            (
                "rubric_based_tool_use_quality_v1",
                Box::new(RubricBasedToolUseV1EvaluatorMetricInfoProvider),
            ),
            (
                "per_turn_user_simulator_quality_v1",
                Box::new(PerTurnUserSimulatorQualityV1MetricInfoProvider),
            ),
            (
                "rubric_based_multi_turn_trajectory_quality_v1",
                Box::new(RubricBasedMultiTurnTrajectoryMetricInfoProvider),
            ),
        ];
        for (expected_name, provider) in cases {
            let info = provider.get_metric_info().unwrap();
            assert_eq!(info.metric_name, expected_name);
            assert_eq!(
                info.metric_value_info.interval,
                Some(Interval::closed(0.0, 1.0)),
                "metric {expected_name}"
            );
            assert!(info.description.is_some());
        }
    }
}
