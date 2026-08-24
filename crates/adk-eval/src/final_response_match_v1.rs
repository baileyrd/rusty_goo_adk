//! C0590: `RougeEvaluator` (metric `response_match_score`), ported from
//! `google.adk.evaluation.final_response_match_v1`. Compares an agent's
//! final response text against a golden/expected final response using
//! the ROUGE-1 F-measure computed by [`crate::rouge`].

use adk_genai::content::Content;

use crate::eval_case::Invocation;
use crate::eval_metrics::{get_metric_threshold, EvalMetric};
use crate::evaluator::{
    validate_invocation_lengths, EvalStatus, EvaluationResult, Evaluator, PerInvocationResult,
};
use crate::rouge;

/// C0590: `RougeEvaluator` — value range `[0,1]`, higher is more
/// desirable.
pub struct RougeEvaluator {
    threshold: f64,
}

impl RougeEvaluator {
    pub fn new(eval_metric: &EvalMetric) -> Result<Self, String> {
        Ok(Self {
            threshold: get_metric_threshold(eval_metric)?,
        })
    }
}

/// `final_response_match_v1._get_text_from_content` — joins each part's
/// non-empty `text` with `"\n"`. Deliberately not
/// `content_utils::extract_text_from_content` (C0927): that function
/// filters out `thought` parts and concatenates without a separator — a
/// different, file-local helper the source itself keeps separate.
fn get_text_from_content(content: Option<&Content>) -> String {
    match content {
        Some(content) if !content.parts.is_empty() => content
            .parts
            .iter()
            .filter_map(|part| part.text.as_ref().filter(|text| !text.is_empty()))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn eval_status_for(score: f64, threshold: f64) -> EvalStatus {
    if score >= threshold {
        EvalStatus::Passed
    } else {
        EvalStatus::Failed
    }
}

impl Evaluator for RougeEvaluator {
    fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
        _conversation_scenario: Option<&rusty_serde::value::Value>,
    ) -> Result<EvaluationResult, String> {
        let expected_invocations = expected_invocations
            .ok_or_else(|| "expected_invocations is required for this metric.".to_string())?;
        validate_invocation_lengths(actual_invocations, Some(expected_invocations))?;

        let mut total_score = 0.0;
        let mut per_invocation_results = Vec::new();
        for (actual, expected) in actual_invocations.iter().zip(expected_invocations.iter()) {
            let reference = get_text_from_content(expected.final_response.as_ref());
            let response = get_text_from_content(actual.final_response.as_ref());
            let score = rouge::score(&reference, &response).fmeasure;

            per_invocation_results.push(PerInvocationResult {
                actual_invocation: actual.clone(),
                expected_invocation: Some(expected.clone()),
                score: Some(score),
                eval_status: eval_status_for(score, self.threshold),
                rubric_scores: None,
            });
            total_score += score;
        }

        if per_invocation_results.is_empty() {
            return Ok(EvaluationResult::default());
        }

        let overall_score = total_score / per_invocation_results.len() as f64;
        Ok(EvaluationResult {
            overall_score: Some(overall_score),
            overall_eval_status: eval_status_for(overall_score, self.threshold),
            per_invocation_results,
            overall_rubric_scores: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_serde::value::Value;

    fn invocation(text: Option<&str>) -> Invocation {
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: text.map(Content::user_text),
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    fn metric_with_threshold(threshold: f64) -> EvalMetric {
        EvalMetric::new("response_match_score").with_threshold(threshold)
    }

    #[test]
    fn new_reads_the_threshold_from_the_metric() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.7)).unwrap();
        assert_eq!(evaluator.threshold, 0.7);
    }

    #[test]
    fn new_errors_without_a_threshold() {
        assert!(RougeEvaluator::new(&EvalMetric::new("response_match_score")).is_err());
    }

    #[test]
    fn evaluate_invocations_requires_expected_invocations() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let result = evaluator.evaluate_invocations(&[invocation(Some("hi"))], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn evaluate_invocations_rejects_mismatched_lengths() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let actual = vec![invocation(Some("hi")), invocation(Some("bye"))];
        let expected = vec![invocation(Some("hi"))];
        assert!(evaluator
            .evaluate_invocations(&actual, Some(&expected), None)
            .is_err());
    }

    #[test]
    fn scores_a_perfect_match_as_passed() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.9)).unwrap();
        let actual = vec![invocation(Some("the cat sat on the mat"))];
        let expected = vec![invocation(Some("the cat sat on the mat"))];
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
        assert_eq!(result.overall_eval_status, EvalStatus::Passed);
        assert_eq!(result.per_invocation_results.len(), 1);
    }

    #[test]
    fn scores_a_total_mismatch_as_failed() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let actual = vec![invocation(Some("apples and oranges"))];
        let expected = vec![invocation(Some("quantum entanglement physics"))];
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
        assert_eq!(result.overall_eval_status, EvalStatus::Failed);
    }

    #[test]
    fn treats_a_missing_final_response_as_empty_text() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let actual = vec![invocation(None)];
        let expected = vec![invocation(Some("hi"))];
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected), None)
            .unwrap();
        assert_eq!(result.overall_score, Some(0.0));
    }

    #[test]
    fn averages_the_score_across_invocations() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let actual = vec![invocation(Some("a b c")), invocation(Some("x y z"))];
        let expected = vec![invocation(Some("a b c")), invocation(Some("q r s"))];
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected), None)
            .unwrap();
        assert_eq!(result.per_invocation_results.len(), 2);
        assert_eq!(result.per_invocation_results[0].score, Some(1.0));
        assert_eq!(result.per_invocation_results[1].score, Some(0.0));
        assert_eq!(result.overall_score, Some(0.5));
    }

    #[test]
    fn ignores_an_unused_conversation_scenario_parameter() {
        let evaluator = RougeEvaluator::new(&metric_with_threshold(0.5)).unwrap();
        let actual = vec![invocation(Some("hi"))];
        let expected = vec![invocation(Some("hi"))];
        let scenario = Value::Map(vec![("persona".to_string(), Value::String("x".into()))]);
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected), Some(&scenario))
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
    }
}
