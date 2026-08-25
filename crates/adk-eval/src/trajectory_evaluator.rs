//! Capability C0588: `TrajectoryEvaluator`, ported from
//! `google.adk.evaluation.trajectory_evaluator`. Fully portable with no
//! LLM calls and no cloud dependency — every match algorithm is plain
//! list-walking over `FunctionCall`s.

use adk_genai::content::FunctionCall;

use crate::eval_case::{get_all_tool_calls, Invocation};
use crate::eval_metrics::{get_metric_threshold, EvalMetric, MatchType, ToolTrajectoryCriterion};
use crate::evaluator::{
    validate_invocation_lengths, EvalStatus, EvaluationResult, Evaluator, PerInvocationResult,
};

/// C0588: `trajectory_evaluator.TrajectoryEvaluator` — evaluates tool
/// use trajectories for accuracy under `EXACT`/`IN_ORDER`/`ANY_ORDER`
/// match types.
pub struct TrajectoryEvaluator {
    threshold: f64,
    match_type: MatchType,
}

impl TrajectoryEvaluator {
    /// `TrajectoryEvaluator.__init__` — exactly one of `threshold` or
    /// `eval_metric` must be given.
    pub fn new(threshold: Option<f64>, eval_metric: Option<&EvalMetric>) -> Result<Self, String> {
        if threshold.is_some() && eval_metric.is_some() {
            return Err(
                "Either eval_metric should be specified or threshold should be specified."
                    .to_string(),
            );
        }

        if let Some(eval_metric) = eval_metric {
            if let Some(criterion) = &eval_metric.criterion {
                let parsed: ToolTrajectoryCriterion =
                    rusty_serde::json::from_value(criterion.clone()).map_err(|_| {
                        format!(
                            "`{}` metric expects a criterion of type `ToolTrajectoryCriterion`.",
                            eval_metric.metric_name
                        )
                    })?;
                return Ok(Self {
                    threshold: parsed.threshold,
                    match_type: parsed.match_type,
                });
            }
            let threshold = get_metric_threshold(eval_metric)?;
            return Ok(Self {
                threshold,
                match_type: MatchType::Exact,
            });
        }

        let threshold = threshold
            .ok_or_else(|| "A trajectory evaluation threshold is required.".to_string())?;
        Ok(Self {
            threshold,
            match_type: MatchType::Exact,
        })
    }

    fn calculate_tool_use_accuracy(&self, actual: &Invocation, expected: &Invocation) -> f64 {
        let actual_tool_uses = get_all_tool_calls(actual.intermediate_data_type().as_ref());
        let expected_tool_uses = get_all_tool_calls(expected.intermediate_data_type().as_ref());

        let matched = match self.match_type {
            MatchType::Exact => are_tool_calls_exact_match(&actual_tool_uses, &expected_tool_uses),
            MatchType::InOrder => {
                are_tool_calls_in_order_match(&actual_tool_uses, &expected_tool_uses)
            }
            MatchType::AnyOrder => {
                are_tool_calls_any_order_match(&actual_tool_uses, &expected_tool_uses)
            }
        };
        if matched {
            1.0
        } else {
            0.0
        }
    }

    fn eval_status(&self, score: f64) -> EvalStatus {
        if score >= self.threshold {
            EvalStatus::Passed
        } else {
            EvalStatus::Failed
        }
    }
}

fn calls_match(a: &FunctionCall, b: &FunctionCall) -> bool {
    a.name == b.name && a.args == b.args
}

/// `TrajectoryEvaluator._are_tool_calls_exact_match`.
fn are_tool_calls_exact_match(actual: &[FunctionCall], expected: &[FunctionCall]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| calls_match(a, e))
}

/// `TrajectoryEvaluator._are_tool_calls_in_order_match`.
fn are_tool_calls_in_order_match(actual: &[FunctionCall], expected: &[FunctionCall]) -> bool {
    if expected.is_empty() {
        return true;
    }
    if actual.is_empty() {
        return false;
    }

    let mut expected_it = expected.iter();
    let Some(mut current_expected) = expected_it.next() else {
        return true;
    };
    for call in actual {
        if calls_match(call, current_expected) {
            match expected_it.next() {
                Some(next) => current_expected = next,
                None => return true,
            }
        }
    }
    false
}

/// `TrajectoryEvaluator._are_tool_calls_any_order_match`.
fn are_tool_calls_any_order_match(actual: &[FunctionCall], expected: &[FunctionCall]) -> bool {
    if expected.is_empty() {
        return true;
    }
    if actual.is_empty() {
        return false;
    }

    let mut remaining: Vec<&FunctionCall> = actual.iter().collect();
    for expected_call in expected {
        let Some(pos) = remaining
            .iter()
            .position(|call| calls_match(call, expected_call))
        else {
            return false;
        };
        remaining.remove(pos);
    }
    true
}

impl Evaluator for TrajectoryEvaluator {
    fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
        conversation_scenario: Option<&rusty_serde::value::Value>,
    ) -> Result<EvaluationResult, String> {
        let _ = conversation_scenario; // not supported for per-invocation evaluation.
        let Some(expected_invocations) = expected_invocations else {
            return Err("expected_invocations is needed by this metric.".to_string());
        };
        validate_invocation_lengths(actual_invocations, Some(expected_invocations))?;

        let mut total_tool_use_accuracy = 0.0;
        let mut per_invocation_results = Vec::new();

        for (actual, expected) in actual_invocations.iter().zip(expected_invocations.iter()) {
            let score = self.calculate_tool_use_accuracy(actual, expected);
            per_invocation_results.push(PerInvocationResult {
                actual_invocation: actual.clone(),
                expected_invocation: Some(expected.clone()),
                score: Some(score),
                eval_status: self.eval_status(score),
                rubric_scores: None,
            });
            total_tool_use_accuracy += score;
        }

        if per_invocation_results.is_empty() {
            return Ok(EvaluationResult::default());
        }

        let overall_score = total_tool_use_accuracy / per_invocation_results.len() as f64;
        Ok(EvaluationResult {
            overall_score: Some(overall_score),
            overall_eval_status: self.eval_status(overall_score),
            per_invocation_results,
            overall_rubric_scores: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_case::IntermediateData;
    use adk_genai::content::Content;
    use rusty_serde::value::Value;
    use std::collections::BTreeMap;

    fn call(name: &str) -> FunctionCall {
        FunctionCall {
            partial_args: None,
            id: None,
            name: Some(name.to_string()),
            args: None,
            will_continue: None,
        }
    }

    fn call_with_args(name: &str, key: &str, value: &str) -> FunctionCall {
        let mut args = BTreeMap::new();
        args.insert(key.to_string(), Value::String(value.to_string()));
        FunctionCall {
            partial_args: None,
            id: None,
            name: Some(name.to_string()),
            args: Some(args),
            will_continue: None,
        }
    }

    fn invocation_with_calls(calls: Vec<FunctionCall>) -> Invocation {
        let value = rusty_serde::json::to_value(&IntermediateData {
            tool_uses: calls,
            ..Default::default()
        })
        .unwrap();
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: None,
            intermediate_data: Some(value),
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    #[test]
    fn new_rejects_both_threshold_and_eval_metric() {
        let metric = EvalMetric::new("tool_trajectory_avg_score").with_threshold(0.5);
        assert!(TrajectoryEvaluator::new(Some(0.5), Some(&metric)).is_err());
    }

    #[test]
    fn new_requires_a_threshold_without_an_eval_metric() {
        assert!(TrajectoryEvaluator::new(None, None).is_err());
    }

    #[test]
    fn new_reads_the_threshold_and_match_type_from_a_criterion() {
        let criterion = rusty_serde::json::to_value(&ToolTrajectoryCriterion {
            threshold: 0.8,
            include_intermediate_responses_in_final: false,
            match_type: MatchType::InOrder,
        })
        .unwrap();
        let metric = EvalMetric::new("tool_trajectory_avg_score").with_criterion(criterion);
        let evaluator = TrajectoryEvaluator::new(None, Some(&metric)).unwrap();
        assert_eq!(evaluator.threshold, 0.8);
        assert_eq!(evaluator.match_type, MatchType::InOrder);
    }

    #[test]
    fn exact_match_requires_identical_trajectories() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let actual = invocation_with_calls(vec![call("a"), call("b")]);
        let expected = invocation_with_calls(vec![call("a"), call("b")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
        assert_eq!(result.overall_eval_status, EvalStatus::Passed);
    }

    #[test]
    fn exact_match_fails_on_extra_calls() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let actual = invocation_with_calls(vec![call("a"), call("b"), call("c")]);
        let expected = invocation_with_calls(vec![call("a"), call("b")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
    }

    #[test]
    fn exact_match_compares_args_not_just_names() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let actual = invocation_with_calls(vec![call_with_args("search", "q", "cats")]);
        let expected = invocation_with_calls(vec![call_with_args("search", "q", "dogs")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
    }

    #[test]
    fn in_order_match_allows_extra_calls_between_expected_ones() {
        let mut evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        evaluator.match_type = MatchType::InOrder;
        let actual = invocation_with_calls(vec![
            call("t1"),
            call("t1.1"),
            call("t2"),
            call("t2.1"),
            call("t3"),
        ]);
        let expected = invocation_with_calls(vec![call("t1"), call("t2"), call("t3")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
    }

    #[test]
    fn in_order_match_fails_if_a_required_call_is_missing() {
        let mut evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        evaluator.match_type = MatchType::InOrder;
        let actual = invocation_with_calls(vec![call("t1"), call("t2"), call("t3")]);
        let expected = invocation_with_calls(vec![call("t1"), call("t2"), call("t3"), call("t4")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
    }

    #[test]
    fn any_order_match_ignores_call_order() {
        let mut evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        evaluator.match_type = MatchType::AnyOrder;
        let actual = invocation_with_calls(vec![call("t2"), call("t1"), call("t3")]);
        let expected = invocation_with_calls(vec![call("t1"), call("t2"), call("t3")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
    }

    #[test]
    fn any_order_match_fails_if_a_required_call_is_missing() {
        let mut evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        evaluator.match_type = MatchType::AnyOrder;
        let actual = invocation_with_calls(vec![call("t1"), call("t2")]);
        let expected = invocation_with_calls(vec![call("t1"), call("t2"), call("t3")]);
        let result = evaluator
            .evaluate_invocations(&[actual], Some(&[expected]), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
    }

    #[test]
    fn evaluate_invocations_requires_expected_invocations() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let actual = invocation_with_calls(vec![call("a")]);
        assert!(evaluator
            .evaluate_invocations(&[actual], None, None)
            .is_err());
    }

    #[test]
    fn evaluate_invocations_rejects_mismatched_lengths() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let actual = invocation_with_calls(vec![call("a")]);
        let expected = invocation_with_calls(vec![call("a")]);
        let result =
            evaluator.evaluate_invocations(&[actual.clone(), actual], Some(&[expected]), None);
        assert!(result.is_err());
    }

    #[test]
    fn overall_score_averages_across_invocations() {
        let evaluator = TrajectoryEvaluator::new(Some(1.0), None).unwrap();
        let matching = invocation_with_calls(vec![call("a")]);
        let non_matching_actual = invocation_with_calls(vec![call("a")]);
        let non_matching_expected = invocation_with_calls(vec![call("b")]);
        let result = evaluator
            .evaluate_invocations(
                &[matching.clone(), non_matching_actual],
                Some(&[matching, non_matching_expected]),
                None,
            )
            .unwrap();
        assert_eq!(result.overall_score, Some(0.5));
        assert_eq!(result.per_invocation_results.len(), 2);
    }
}
