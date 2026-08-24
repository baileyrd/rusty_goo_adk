//! Capability C0600 (partial — see the crate root doc for the
//! `LlmAsJudge` half not built this batch), ported from
//! `google.adk.evaluation.evaluator`.

use rusty_serde::{Deserialize, Serialize};

use crate::eval_case::Invocation;
pub use crate::eval_metrics::EvalStatus;

/// `evaluator._validate_invocation_lengths` — rejects invocation lists
/// that cannot be paired without truncation.
pub fn validate_invocation_lengths(
    actual_invocations: &[Invocation],
    expected_invocations: Option<&[Invocation]>,
) -> Result<(), String> {
    if let Some(expected) = expected_invocations {
        if actual_invocations.len() != expected.len() {
            return Err(format!(
                "actual_invocations and expected_invocations must have the same length; \
                 got {} and {}.",
                actual_invocations.len(),
                expected.len()
            ));
        }
    }
    Ok(())
}

/// C0600: `evaluator.PerInvocationResult` — metric evaluation score per
/// invocation.
///
/// **Adaptation**: `rubric_scores` (`eval_rubrics.RubricScore`, C0607,
/// still `REQUIRED`) stays an opaque `Value` placeholder — see the crate
/// root doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct PerInvocationResult {
    pub actual_invocation: Invocation,
    #[rusty_serde(default)]
    pub expected_invocation: Option<Invocation>,
    #[rusty_serde(default)]
    pub score: Option<f64>,
    #[rusty_serde(default)]
    pub eval_status: EvalStatus,
    #[rusty_serde(default)]
    pub rubric_scores: Option<rusty_serde::value::Value>,
}

/// C0600: `evaluator.EvaluationResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct EvaluationResult {
    #[rusty_serde(default)]
    pub overall_score: Option<f64>,
    #[rusty_serde(default)]
    pub overall_eval_status: EvalStatus,
    #[rusty_serde(default)]
    pub per_invocation_results: Vec<PerInvocationResult>,
    #[rusty_serde(default)]
    pub overall_rubric_scores: Option<rusty_serde::value::Value>,
}

/// C0600 (partial): `evaluator.Evaluator` — a metrics evaluator
/// interface. See the crate root doc for why the source's
/// `LlmAsJudge[CriterionT]` harness isn't built on top of this yet.
///
/// **Adaptation**: sync, not `-> EvaluationResult | Awaitable[EvaluationResult]`
/// — the one implementor this batch ports
/// ([`crate::trajectory_evaluator::TrajectoryEvaluator`]) does no I/O, so
/// there's nothing to await; a future LLM-judge-backed implementor will
/// need its own async story, not modeled here yet.
pub trait Evaluator {
    /// Returns an [`EvaluationResult`] after evaluating `actual_invocations`
    /// against `expected_invocations` (a benchmark/golden set, when
    /// given).
    ///
    /// **Adaptation**: the source's `conversation_scenario: Optional[ConversationScenario]`
    /// parameter (`eval_case.ConversationScenario`, C0606, still
    /// `REQUIRED`) narrows to an opaque `Value` — see the crate root doc.
    fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
        conversation_scenario: Option<&rusty_serde::value::Value>,
    ) -> Result<EvaluationResult, String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::Content;

    fn invocation() -> Invocation {
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("hi"),
            final_response: None,
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    #[test]
    fn validate_invocation_lengths_accepts_none_expected() {
        assert!(validate_invocation_lengths(&[invocation()], None).is_ok());
    }

    #[test]
    fn validate_invocation_lengths_accepts_matching_lengths() {
        assert!(validate_invocation_lengths(&[invocation()], Some(&[invocation()])).is_ok());
    }

    #[test]
    fn validate_invocation_lengths_rejects_mismatched_lengths() {
        let result =
            validate_invocation_lengths(&[invocation(), invocation()], Some(&[invocation()]));
        assert!(result.is_err());
    }

    #[test]
    fn evaluation_result_default_is_not_evaluated() {
        assert_eq!(
            EvaluationResult::default().overall_eval_status,
            EvalStatus::NotEvaluated
        );
    }
}
