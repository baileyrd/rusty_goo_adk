//! C0592: `evaluation.final_response_match_v2`, ported from
//! `google.adk.evaluation.final_response_match_v2`.
//!
//! [`FinalResponseMatchV2Evaluator`] composes a
//! [`crate::llm_as_judge::LlmAsJudgeConfig`] directly (not
//! [`crate::rubric_based_evaluator::RubricBasedEvaluator`] — this metric
//! isn't rubric-based, it uses [`crate::eval_metrics::LlmAsAJudgeCriterion`])
//! rather than subclassing `LlmAsJudge[LlmAsAJudgeCriterion]` the way the
//! source does — see `llm_as_judge.rs`'s module doc for why this port has
//! no such trait to subclass. Its own hooks supply
//! [`evaluate_invocations_via_llm_judge`]'s four closures directly.
//!
//! **Does not implement [`crate::evaluator::Evaluator`], disclosed**: same
//! structural reason as `rubric_based_tool_use_quality_v1.rs` et al. — the
//! harness is inherently async, `Evaluator::evaluate_invocations` is
//! deliberately sync.
//!
//! **`_parse_critique`, ported as a free function**: the source's module-
//! level regexes are ported verbatim (same char-class semantics in the
//! `regex` crate as Python's `re`), including the `is_the_agent_response_valid`
//! vs. `..._invalid` field-name fallback and the `PARTIALLY_VALID` label's
//! flattened `Label::value()` (already disclosed in `llm_as_judge_utils.rs`).

use crate::eval_case::Invocation;
use crate::eval_metrics::{EvalMetric, LlmAsAJudgeCriterion};
use crate::evaluator::{EvalStatus, EvaluationResult, PerInvocationResult};
use crate::llm_as_judge::{
    evaluate_invocations_via_llm_judge, AutoRaterScore, LlmAsJudgeConfig, LlmAsJudgeError,
};
use crate::llm_as_judge_utils::{
    get_eval_status, get_text_from_content, get_text_from_invocation, Label,
};
use adk_models::llm_response::LlmResponse;
use regex::Regex;
use std::sync::OnceLock;

const AUTO_RATER_PROMPT_TEMPLATE: &str = r#"You are an expert rater for an AI agent. The AI agent is going to call an API to answer the user query and generate API tool use code based for the choice of the API and API arguments. The ideal model response should be a function call that fulfills user query, or a natural language response hedges or asks users for further clarification if a function call does not apply.
The primary focus of this rating task is to check correctness of the model responses.

The data consists of:
- A user query.
- A model generated response for the prompt. The responses can consist of:
  - Natural language, when the model is asking for clarification, or tells the user it does not possess the requested functionality / option.
  - Code, in the form of one or multiple python function calls, and additional code as needed, for when the model is fulfilling the user request.
You can use the help from a reference response annotated by a human rater. This reference response is of high quality. You can compare the agent's response with the reference response and decide if the agent's response is valid.
Note sometimes the reference response only contains the key entities of the correct answer and you need to be flexible to allow the agent response to contain more information than the reference response, or to present the key entities in a different format or structure or in shorter or longer format.
When the agent response is provided in the form of tables/dataframes or should be best provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response. Likewise, if you have the reference response, then find out the key entities and main components in them and check whether you can retrieve those from the agent response. If the prompt does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.

You should follow the constitutions below very carefully to rate the model response:
- Allow flexibility of format even when reference code only uses one of the possible format, unless API spec or user prompt has explicit format requirement
  - e.g. For state name, allow both abbreviation and full name unless API spec has explicit requirement. e.g. both 'tx' and 'Texas' should be allowed in the agent response even when reference code only uses one of them.
  - e.g. If a reference response list outputs in a list format, the agent response is allowed to use sentence format and vice versa unless user prompt explicitly asks for a specific format.
  - e.g. For numbers, allow flexibility of formatting, e.g. 1000000 vs 1,000,000.
- The model shouldn't assume that it doesn't have access to according data or incapable of answering the question if reference response is able to find a legit answer.
- If the model response contains the correct final answer, rate it as valid even when the model response contains more information than the reference response.
- If the user prompt has csv or other table format data, don't read it yourself. Trust the reference response final answer instead.
- When the validation needs maths, date calculations, do not use your own calculator. Trust the reference response final answer instead.
- Be mindful about unit of numbers. For example, if the reference response says 100 miles, but the model response says 100 km, it is invalid.
- When the agent response or the reference response is provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response and whether those match the reference response. If the user query does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.
- When the answer is in numeric format, check whether there are any format requirements in the numeric format, rounding, precision, number of decimals, etc. specified in the user query and the prompt. If there are no such instructions, then tolerate different numerical formats.
- When the answer is in numeric format and there are rounding or precision differences between the agent response and the reference response, if no further instructions are provided evaluate if the rounding strategy or precision in the agent response follows the standards for that entity. For instance, model accuracy scores must be reported with at least two decimal places (e.g., 0.798 → 0.80 is acceptable,  but 0.7 is not).

Below are the inputs:
{{
  "User prompt": {prompt},
  "Agent response": {response},
  "Reference response": {golden_response},
}}

The answer should be a json alone which follows the json structure below:
{{
  "reasoning": [reasoning],
  "is_the_agent_response_valid": [valid or invalid],
}}
Answer with assertiveness:
"#;

fn valid_field_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#""is_the_agent_response_valid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]"#,
        )
        .unwrap()
    })
}

fn invalid_field_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#""is_the_agent_response_invalid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]"#,
        )
        .unwrap()
    })
}

/// `final_response_match_v2._parse_critique` — parses the judge model
/// critique and extracts the final label.
fn parse_critique(response: &str) -> Label {
    if let Some(captures) = valid_field_pattern().captures(response) {
        let label = captures
            .get(1)
            .map_or("", |m| m.as_str())
            .trim()
            .trim_end_matches([',', '}']);
        let invalid_values: Vec<&str> = [
            Label::Invalid.value(),
            Label::Almost.value(),
            Label::False.value(),
            Label::PartiallyValid.value(),
        ]
        .concat();
        if invalid_values.contains(&label) {
            Label::Invalid
        } else if [Label::Valid.value(), Label::True.value()]
            .concat()
            .contains(&label)
        {
            Label::Valid
        } else {
            Label::NotFound
        }
    } else if let Some(captures) = invalid_field_pattern().captures(response) {
        let label = captures
            .get(1)
            .map_or("", |m| m.as_str())
            .trim()
            .trim_end_matches([',', '}']);
        if [Label::True.value(), Label::Invalid.value()]
            .concat()
            .contains(&label)
        {
            Label::Invalid
        } else {
            Label::Valid
        }
    } else {
        Label::NotFound
    }
}

/// `final_response_match_v2.FinalResponseMatchV2Evaluator`.
pub struct FinalResponseMatchV2Evaluator {
    config: LlmAsJudgeConfig<LlmAsAJudgeCriterion>,
}

impl FinalResponseMatchV2Evaluator {
    pub fn new(eval_metric: &EvalMetric) -> Result<Self, LlmAsJudgeError> {
        Ok(Self {
            config: LlmAsJudgeConfig::<LlmAsAJudgeCriterion>::new(eval_metric)?,
        })
    }

    /// `FinalResponseMatchV2Evaluator.format_auto_rater_prompt`. Panics if
    /// `expected_invocation` is `None` — mirrors the source's uncaught
    /// `ValueError`; the harness never actually calls this with `None`
    /// when `expected_invocations_required` is honored (enforced by
    /// [`evaluate_invocations_via_llm_judge`] itself before any prompt is
    /// formatted).
    fn format_auto_rater_prompt(
        &self,
        actual_invocation: &Invocation,
        expected_invocation: Option<&Invocation>,
    ) -> String {
        let expected_invocation =
            expected_invocation.expect("expected_invocation is required for this metric");
        let include_intermediate = self
            .config
            .criterion
            .include_intermediate_responses_in_final;
        let reference =
            get_text_from_invocation(expected_invocation, include_intermediate).unwrap_or_default();
        let response =
            get_text_from_invocation(actual_invocation, include_intermediate).unwrap_or_default();
        let user_prompt =
            get_text_from_content(Some(&expected_invocation.user_content)).unwrap_or_default();
        AUTO_RATER_PROMPT_TEMPLATE
            .replacen("{prompt}", &user_prompt, 1)
            .replacen("{response}", &response, 1)
            .replacen("{golden_response}", &reference, 1)
    }

    /// `FinalResponseMatchV2Evaluator.convert_auto_rater_response_to_score`.
    fn convert_auto_rater_response_to_score(&self, llm_response: &LlmResponse) -> AutoRaterScore {
        let Some(response_text) = get_text_from_content(llm_response.content.as_ref()) else {
            return AutoRaterScore::default();
        };
        match parse_critique(&response_text) {
            Label::Valid => AutoRaterScore {
                score: Some(1.0),
                rubric_scores: None,
            },
            Label::Invalid => AutoRaterScore {
                score: Some(0.0),
                rubric_scores: None,
            },
            _ => AutoRaterScore::default(),
        }
    }

    /// `FinalResponseMatchV2Evaluator.aggregate_per_invocation_samples` —
    /// majority vote; a tie (or no evaluated samples) prefers the first
    /// invalid/first sample, matching the source exactly.
    fn aggregate_per_invocation_samples(
        &self,
        per_invocation_samples: &[PerInvocationResult],
    ) -> PerInvocationResult {
        let positive: Vec<&PerInvocationResult> = per_invocation_samples
            .iter()
            .filter(|r| r.score == Some(1.0))
            .collect();
        let negative: Vec<&PerInvocationResult> = per_invocation_samples
            .iter()
            .filter(|r| r.score == Some(0.0))
            .collect();
        if positive.is_empty() && negative.is_empty() {
            per_invocation_samples[0].clone()
        } else if positive.len() > negative.len() {
            positive[0].clone()
        } else {
            negative[0].clone()
        }
    }

    /// `FinalResponseMatchV2Evaluator.aggregate_invocation_results` — the
    /// fraction of invocation results that are valid.
    fn aggregate_invocation_results(
        &self,
        per_invocation_results: &[PerInvocationResult],
    ) -> EvaluationResult {
        let mut num_valid = 0.0;
        let mut num_evaluated: i64 = 0;
        for result in per_invocation_results {
            if result.score.is_none() || result.eval_status == EvalStatus::NotEvaluated {
                continue;
            }
            num_evaluated += 1;
            num_valid += result.score.expect("checked above");
        }
        if num_evaluated == 0 {
            return EvaluationResult {
                overall_score: None,
                overall_eval_status: EvalStatus::NotEvaluated,
                per_invocation_results: per_invocation_results.to_vec(),
                overall_rubric_scores: None,
            };
        }
        let overall_score = num_valid / num_evaluated as f64;
        EvaluationResult {
            overall_score: Some(overall_score),
            overall_eval_status: get_eval_status(
                Some(overall_score),
                self.config.criterion.threshold,
            ),
            per_invocation_results: per_invocation_results.to_vec(),
            overall_rubric_scores: None,
        }
    }

    /// Drives [`evaluate_invocations_via_llm_judge`] with this evaluator's
    /// hooks. Not [`crate::evaluator::Evaluator::evaluate_invocations`] —
    /// see this module's doc.
    pub async fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
    ) -> Result<EvaluationResult, LlmAsJudgeError> {
        evaluate_invocations_via_llm_judge(
            self.config.judge_model.as_ref(),
            &self.config.criterion.judge_model_options,
            self.config.threshold,
            true,
            actual_invocations,
            expected_invocations,
            |actual, expected| self.format_auto_rater_prompt(actual, expected),
            |response| self.convert_auto_rater_response_to_score(response),
            |samples| self.aggregate_per_invocation_samples(samples),
            |results| self.aggregate_invocation_results(results),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_metrics::JudgeModelOptions;
    use adk_genai::content::{Content, Part};
    use adk_models::base_llm::{BaseLlm, BaseLlmError};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    // --- _parse_critique ---

    #[test]
    fn parse_critique_recognizes_valid() {
        let response = r#"{"reasoning": "looks right", "is_the_agent_response_valid": "valid"}"#;
        assert_eq!(parse_critique(response), Label::Valid);
    }

    #[test]
    fn parse_critique_recognizes_invalid() {
        let response = r#"{"reasoning": "wrong", "is_the_agent_response_valid": "invalid"}"#;
        assert_eq!(parse_critique(response), Label::Invalid);
    }

    #[test]
    fn parse_critique_treats_almost_and_partially_valid_as_invalid() {
        assert_eq!(
            parse_critique(r#"{"is_the_agent_response_valid": "almost"}"#),
            Label::Invalid
        );
        assert_eq!(
            parse_critique(r#"{"is_the_agent_response_valid": "partially_valid"}"#),
            Label::Invalid
        );
    }

    #[test]
    fn parse_critique_falls_back_to_the_invalid_field_name() {
        let response = r#"{"is_the_agent_response_invalid": "true"}"#;
        assert_eq!(parse_critique(response), Label::Invalid);
        let response = r#"{"is_the_agent_response_invalid": "false"}"#;
        assert_eq!(parse_critique(response), Label::Valid);
    }

    #[test]
    fn parse_critique_returns_not_found_without_a_recognizable_field() {
        assert_eq!(parse_critique("no json here"), Label::NotFound);
    }

    // --- format_auto_rater_prompt / hooks ---

    fn eval_metric() -> EvalMetric {
        let criterion = LlmAsAJudgeCriterion {
            threshold: 0.5,
            include_intermediate_responses_in_final: false,
            judge_model_options: JudgeModelOptions {
                judge_model: "gemini-2.5-flash".to_string(),
                num_samples: 1,
                ..Default::default()
            },
        };
        let value = rusty_serde::json::to_value(&criterion).unwrap();
        EvalMetric::new("final_response_match_v2").with_criterion(value)
    }

    fn invocation(user_text: &str, final_text: &str) -> Invocation {
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text(user_text),
            final_response: Some(Content::new("model", vec![Part::text(final_text)])),
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    #[test]
    fn format_auto_rater_prompt_embeds_prompt_response_and_reference() {
        let evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        let actual = invocation("what's the weather", "sunny");
        let expected = invocation("what's the weather", "it's sunny");
        let prompt = evaluator.format_auto_rater_prompt(&actual, Some(&expected));
        assert!(prompt.contains("what's the weather"));
        assert!(prompt.contains("\"Agent response\": sunny"));
        assert!(prompt.contains("\"Reference response\": it's sunny"));
    }

    #[test]
    #[should_panic(expected = "expected_invocation is required")]
    fn format_auto_rater_prompt_panics_without_an_expected_invocation() {
        let evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        let actual = invocation("hi", "hello");
        evaluator.format_auto_rater_prompt(&actual, None);
    }

    #[test]
    fn aggregate_invocation_results_computes_the_valid_fraction() {
        let evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        let results = vec![
            PerInvocationResult {
                actual_invocation: invocation("a", "b"),
                expected_invocation: None,
                score: Some(1.0),
                eval_status: EvalStatus::Passed,
                rubric_scores: None,
            },
            PerInvocationResult {
                actual_invocation: invocation("a", "b"),
                expected_invocation: None,
                score: Some(0.0),
                eval_status: EvalStatus::Failed,
                rubric_scores: None,
            },
        ];
        let summary = evaluator.aggregate_invocation_results(&results);
        assert_eq!(summary.overall_score, Some(0.5));
        assert_eq!(summary.overall_eval_status, EvalStatus::Passed);
    }

    #[test]
    fn aggregate_invocation_results_is_not_evaluated_without_any_evaluated_samples() {
        let evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        let results = vec![PerInvocationResult {
            actual_invocation: invocation("a", "b"),
            expected_invocation: None,
            score: None,
            eval_status: EvalStatus::NotEvaluated,
            rubric_scores: None,
        }];
        let summary = evaluator.aggregate_invocation_results(&results);
        assert_eq!(summary.overall_score, None);
        assert_eq!(summary.overall_eval_status, EvalStatus::NotEvaluated);
    }

    // --- evaluate_invocations, end to end with a fake judge model ---

    struct QueueLlm {
        model: String,
        queue: Mutex<Vec<LlmResponse>>,
    }

    impl BaseLlm for QueueLlm {
        fn model(&self) -> &str {
            &self.model
        }
        fn type_name(&self) -> &'static str {
            "QueueLlm"
        }
        fn generate_content_async<'a>(
            &'a self,
            _llm_request: &'a adk_models::llm_request::LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let next = self.queue.lock().unwrap().pop();
            Box::pin(async move { Ok(next.into_iter().collect()) })
        }
    }

    fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new("user", vec![Part::text(text)])),
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn evaluate_invocations_requires_expected_invocations() {
        let evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        let actual = vec![invocation("a", "b")];
        let result = evaluator.evaluate_invocations(&actual, None).await;
        assert!(matches!(
            result,
            Err(LlmAsJudgeError::ExpectedInvocationsRequired)
        ));
    }

    #[rusty_tokio::test]
    async fn evaluate_invocations_scores_a_valid_response() {
        let mut evaluator = FinalResponseMatchV2Evaluator::new(&eval_metric()).unwrap();
        evaluator.config.judge_model = Box::new(QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![text_response(
                r#"{"reasoning": "matches", "is_the_agent_response_valid": "valid"}"#,
            )]),
        });
        let actual = vec![invocation("what's the weather", "sunny")];
        let expected = vec![invocation("what's the weather", "it's sunny")];
        let result = evaluator
            .evaluate_invocations(&actual, Some(&expected))
            .await
            .unwrap();
        assert_eq!(result.overall_score, Some(1.0));
        assert_eq!(result.overall_eval_status, EvalStatus::Passed);
    }
}
