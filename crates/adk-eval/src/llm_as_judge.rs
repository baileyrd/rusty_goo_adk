//! C0600 (partial): `evaluation.llm_as_judge`, ported from
//! `google.adk.evaluation.llm_as_judge`.
//!
//! **`LlmAsJudge[CriterionT]`, translated as a harness function, not a
//! trait with default methods**: the source is an `ABC` with four
//! abstract hooks (`format_auto_rater_prompt`/
//! `convert_auto_rater_response_to_score`/`aggregate_per_invocation_samples`/
//! `aggregate_invocation_results`) — a class with any of them still
//! unimplemented can't even be instantiated in Python (`RubricBasedEvaluator`
//! itself is exactly this: it overrides three of the four and stays
//! abstract, waiting for a concrete per-metric subclass to supply
//! `format_auto_rater_prompt`). This port has no such concrete subclass
//! yet (every per-metric evaluator — `HallucinationsEvaluator`,
//! `RubricBasedFinalResponseQualityV1`, etc. — is GCP-blocked, C0591-C0598),
//! so there is nothing to make a Rust trait's `Self: Sized` requirement
//! happy either. Rather than fabricate a placeholder fourth hook (which
//! would misrepresent an unimplemented capability as a real one),
//! [`evaluate_invocations_via_llm_judge`] takes the four hooks as plain
//! closures instead — the shared harness *mechanism*, ready for a future
//! concrete evaluator to call with its own `format_auto_rater_prompt`
//! and [`RubricBasedEvaluator`]'s three already-real hook methods (see
//! that struct's own doc). Same "build the mechanism ahead of a
//! still-blocked concrete consumer" precedent already used for
//! `credential_manager.rs`/`remote_mcp_server.rs`.
//!
//! **Does not implement [`crate::evaluator::Evaluator`], disclosed**: every
//! other evaluator in this crate (`TrajectoryEvaluator`, `RougeEvaluator`,
//! `CustomMetricEvaluator`) implements that trait's sync
//! `evaluate_invocations`. [`evaluate_invocations_via_llm_judge`] can't —
//! it awaits a judge-model call, so it's inherently async, and widening
//! `Evaluator::evaluate_invocations` itself to `async fn` would be a
//! breaking change to a trait three already-shipped types implement. Per
//! this migration's standing rule, a breaking change to already-shipped
//! public surface is stop-and-ask, not auto-applied; the harness stays a
//! free async function a future concrete evaluator calls directly,
//! outside the `Evaluator` trait, exactly as `RubricBasedEvaluator` itself
//! does not implement `Evaluator` either.
//!
//! **Parallel sampling, narrowed to sequential**: the source fans out
//! `num_samples` tasks per invocation under an `asyncio.Semaphore(parallelism_limit)`
//! and gathers them concurrently. This port awaits each sample in a
//! plain sequential loop instead — the *results* are byte-for-byte
//! identical either way (grouping-by-invocation and per-invocation/overall
//! aggregation don't depend on execution order), only the wall-clock
//! parallelism is narrowed. Real concurrent fan-out over a borrowed
//! `&dyn BaseLlm` plus generic closure hooks would need either an
//! async-task-spawn story with `'static` bounds this crate has no
//! infrastructure for yet, or a hand-rolled concurrent-future-poller —
//! disproportionate to this batch; `judge_model_options.parallelism_limit`
//! is read only for documentation/future use, and has no effect on this
//! port's sequential execution.
//!
//! **A failed sample fails the whole invocation, ported exactly**: the
//! source's own logic is `if any(isinstance(r, Exception) for r in
//! invocation_result_samples): ... NOT_EVALUATED` — even one failed
//! sample discards every *successful* sample for that same invocation,
//! not just the failed one. Verified against the real source (not an
//! assumption) and ported as-is, however surprising.
//!
//! **`judge_model_config`, not applied**: the source merges
//! `self._judge_model_options.judge_model_config` into the outgoing
//! `LlmRequest.config`. `JudgeModelOptions::judge_model_config` is an
//! opaque `Value` placeholder in this port (C0612's own disclosed
//! narrowing) — merging an untyped JSON blob into the real, typed
//! `GenerateContentConfigStub` needs a schema-aware merge this batch
//! doesn't build; the outgoing request uses `GenerateContentConfigStub::default()`
//! instead. A real per-metric evaluator that needs judge-model
//! generation-config overrides will need this closed alongside it.
//!
//! **`add_default_retry_options_if_not_present`, not ported**: same
//! disclosed gap already established for `llm_backed_user_simulator.rs`
//! (C0628) — `HttpOptionsStub` has no `retry_options` field, and the
//! source itself flags this helper as eval-systems-internal-only.

use adk_genai::content::{Content, Part};
use adk_models::base_llm::BaseLlm;
use adk_models::llm_request::LlmRequest;
use adk_models::llm_response::LlmResponse;
use adk_models::registry::default_registry;
use rusty_serde::value::Value;
use rusty_serde::Deserialize;

use crate::eval_case::Invocation;
use crate::eval_metrics::{get_metric_threshold, EvalMetric, JudgeModelOptions};
use crate::eval_rubrics::RubricScore;
use crate::evaluator::{validate_invocation_lengths, EvaluationResult, PerInvocationResult};
use crate::llm_as_judge_utils::get_eval_status;

const AUTO_RATER_AUTHOR: &str = "user";

/// `llm_as_judge.AutoRaterScore`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AutoRaterScore {
    pub score: Option<f64>,
    pub rubric_scores: Option<Vec<RubricScore>>,
}

/// Error type for [`LlmAsJudgeConfig::new`]/[`evaluate_invocations_via_llm_judge`].
#[derive(Debug, rusty_err::Error)]
pub enum LlmAsJudgeError {
    #[error("`{0}` metric expects a criterion of the configured type.")]
    InvalidCriterion(String),
    #[error("expected_invocations is needed by this metric.")]
    ExpectedInvocationsRequired,
    #[error("{0}")]
    InvalidInvocations(String),
    #[error("{0}")]
    Threshold(String),
    #[error("{0}")]
    Registry(#[from] adk_models::registry::RegistryError),
    #[error("{0}")]
    Generation(#[from] adk_models::base_llm::BaseLlmError),
    #[error("LLM evaluation failed: no response received from judge model")]
    NoResponse,
}

/// Implemented by every `_CriterionT` this harness accepts
/// (`LlmAsAJudgeCriterion`/`RubricsBasedCriterion`) — both declare
/// `judge_model_options`, which is all [`LlmAsJudgeConfig::new`] reads
/// from the criterion itself.
pub trait JudgeModelOptionsProvider {
    fn judge_model_options(&self) -> &JudgeModelOptions;
}

impl JudgeModelOptionsProvider for crate::eval_metrics::LlmAsAJudgeCriterion {
    fn judge_model_options(&self) -> &JudgeModelOptions {
        &self.judge_model_options
    }
}

impl JudgeModelOptionsProvider for crate::eval_metrics::RubricsBasedCriterion {
    fn judge_model_options(&self) -> &JudgeModelOptions {
        &self.judge_model_options
    }
}

/// `LlmAsJudge.__init__` — resolved criterion, threshold, and judge
/// model, shared by every `LlmAsJudge[CriterionT]` subclass. See the
/// module doc for why this is a plain struct rather than part of a
/// trait.
pub struct LlmAsJudgeConfig<C> {
    pub criterion: C,
    pub threshold: f64,
    pub judge_model: Box<dyn BaseLlm>,
}

impl<C> LlmAsJudgeConfig<C>
where
    C: for<'de> Deserialize<'de> + JudgeModelOptionsProvider,
{
    pub fn new(eval_metric: &EvalMetric) -> Result<Self, LlmAsJudgeError> {
        let criterion_value = eval_metric
            .criterion
            .as_ref()
            .ok_or_else(|| LlmAsJudgeError::InvalidCriterion(eval_metric.metric_name.clone()))?;
        let criterion: C = parse_criterion(criterion_value)
            .map_err(|_| LlmAsJudgeError::InvalidCriterion(eval_metric.metric_name.clone()))?;
        let threshold = get_metric_threshold(eval_metric).map_err(LlmAsJudgeError::Threshold)?;
        let judge_model = setup_auto_rater(criterion.judge_model_options())?;
        Ok(Self {
            criterion,
            threshold,
            judge_model,
        })
    }
}

/// `LlmAsJudge._setup_auto_rater`.
fn setup_auto_rater(options: &JudgeModelOptions) -> Result<Box<dyn BaseLlm>, LlmAsJudgeError> {
    let registry = default_registry()
        .read()
        .expect("llm registry lock poisoned");
    Ok(registry.new_llm(&options.judge_model)?)
}

/// `LlmAsJudge.__init__`'s config round-trip — same idiom
/// `user_simulator::parse_simulator_config` already establishes.
fn parse_criterion<T>(config: &Value) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let json = rusty_serde::json::to_string(config).map_err(|error| error.to_string())?;
    rusty_serde::json::from_str(&json).map_err(|error| error.to_string())
}

/// `LlmAsJudge._evaluate_single_sample`.
async fn evaluate_single_sample(
    judge_model: &dyn BaseLlm,
    llm_request: &LlmRequest,
    actual: &Invocation,
    expected: Option<&Invocation>,
    threshold: f64,
    convert_auto_rater_response_to_score: &impl Fn(&LlmResponse) -> AutoRaterScore,
) -> Result<PerInvocationResult, LlmAsJudgeError> {
    let responses = judge_model
        .generate_content_async(llm_request, false)
        .await?;
    let Some(llm_response) = responses.into_iter().next() else {
        return Err(LlmAsJudgeError::NoResponse);
    };
    let auto_rater_score = convert_auto_rater_response_to_score(&llm_response);
    Ok(PerInvocationResult {
        actual_invocation: actual.clone(),
        expected_invocation: expected.cloned(),
        score: auto_rater_score.score,
        eval_status: get_eval_status(auto_rater_score.score, threshold),
        rubric_scores: auto_rater_score.rubric_scores,
    })
}

/// `LlmAsJudge.evaluate_invocations` — the shared harness. See the
/// module doc for why this is a free function taking the four abstract
/// hooks as closures, and for the sequential-instead-of-semaphore-gated-
/// parallel narrowing.
#[allow(clippy::too_many_arguments)]
pub async fn evaluate_invocations_via_llm_judge(
    judge_model: &dyn BaseLlm,
    judge_model_options: &JudgeModelOptions,
    threshold: f64,
    expected_invocations_required: bool,
    actual_invocations: &[Invocation],
    expected_invocations: Option<&[Invocation]>,
    format_auto_rater_prompt: impl Fn(&Invocation, Option<&Invocation>) -> String,
    convert_auto_rater_response_to_score: impl Fn(&LlmResponse) -> AutoRaterScore,
    aggregate_per_invocation_samples: impl Fn(&[PerInvocationResult]) -> PerInvocationResult,
    aggregate_invocation_results: impl Fn(&[PerInvocationResult]) -> EvaluationResult,
) -> Result<EvaluationResult, LlmAsJudgeError> {
    if expected_invocations_required && expected_invocations.is_none() {
        return Err(LlmAsJudgeError::ExpectedInvocationsRequired);
    }
    validate_invocation_lengths(actual_invocations, expected_invocations)
        .map_err(LlmAsJudgeError::InvalidInvocations)?;

    let resolved_expected: Vec<Option<&Invocation>> = match expected_invocations {
        Some(expected) => expected.iter().map(Some).collect(),
        None => vec![None; actual_invocations.len()],
    };

    let num_samples = judge_model_options.num_samples.max(0) as usize;
    let mut results_by_invocation: Vec<Vec<Result<PerInvocationResult, LlmAsJudgeError>>> =
        (0..actual_invocations.len()).map(|_| Vec::new()).collect();

    for (invocation_idx, (actual, expected)) in actual_invocations
        .iter()
        .zip(resolved_expected.iter())
        .enumerate()
    {
        let auto_rater_prompt = format_auto_rater_prompt(actual, *expected);
        let mut llm_request = LlmRequest::new(judge_model_options.judge_model.clone());
        llm_request.contents = vec![Content::new(
            AUTO_RATER_AUTHOR,
            vec![Part::text(auto_rater_prompt)],
        )];

        for _ in 0..num_samples {
            let result = evaluate_single_sample(
                judge_model,
                &llm_request,
                actual,
                *expected,
                threshold,
                &convert_auto_rater_response_to_score,
            )
            .await;
            if let Err(error) = &result {
                eprintln!("Evaluation sample failed for invocation {invocation_idx}: {error}");
            }
            results_by_invocation[invocation_idx].push(result);
        }
    }

    let mut per_invocation_results = Vec::new();
    for (invocation_idx, samples) in results_by_invocation.into_iter().enumerate() {
        if samples.is_empty() {
            continue;
        }
        if samples.iter().any(Result::is_err) {
            per_invocation_results.push(PerInvocationResult {
                actual_invocation: actual_invocations[invocation_idx].clone(),
                expected_invocation: resolved_expected[invocation_idx].cloned(),
                score: None,
                eval_status: crate::evaluator::EvalStatus::NotEvaluated,
                rubric_scores: None,
            });
        } else {
            let successful: Vec<PerInvocationResult> = samples
                .into_iter()
                .map(|r| r.expect("checked above"))
                .collect();
            per_invocation_results.push(aggregate_per_invocation_samples(&successful));
        }
    }

    if per_invocation_results.is_empty() {
        Ok(EvaluationResult::default())
    } else {
        Ok(aggregate_invocation_results(&per_invocation_results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_metrics::LlmAsAJudgeCriterion;
    use crate::evaluator::EvalStatus;
    use adk_genai::content::Content;
    use adk_models::base_llm::BaseLlmError;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    fn invocation(id: &str) -> Invocation {
        Invocation {
            invocation_id: id.to_string(),
            user_content: Content::user_text("hi"),
            final_response: Some(Content::user_text("final")),
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new(AUTO_RATER_AUTHOR, vec![Part::text(text)])),
            ..Default::default()
        }
    }

    /// Pops one queued result per call — lets a test script exactly
    /// which sample succeeds/fails, and in what order.
    struct QueueLlm {
        model: String,
        queue: Mutex<Vec<Result<LlmResponse, String>>>,
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
            _llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let next = self.queue.lock().unwrap().pop();
            Box::pin(async move {
                match next {
                    Some(Ok(response)) => Ok(vec![response]),
                    Some(Err(message)) => Err(BaseLlmError::CallFailed(message)),
                    None => Ok(vec![]),
                }
            })
        }
    }

    fn score_from_text(response: &LlmResponse) -> AutoRaterScore {
        let text = response
            .content
            .as_ref()
            .and_then(|c| c.parts.first())
            .and_then(|p| p.text.as_deref())
            .unwrap_or("");
        AutoRaterScore {
            score: text.parse::<f64>().ok(),
            rubric_scores: None,
        }
    }

    fn mean_per_invocation(samples: &[PerInvocationResult]) -> PerInvocationResult {
        let scores: Vec<f64> = samples.iter().filter_map(|s| s.score).collect();
        let mean = (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64);
        PerInvocationResult {
            actual_invocation: samples[0].actual_invocation.clone(),
            expected_invocation: samples[0].expected_invocation.clone(),
            score: mean,
            eval_status: get_eval_status(mean, 0.5),
            rubric_scores: None,
        }
    }

    fn mean_overall(results: &[PerInvocationResult]) -> EvaluationResult {
        let scores: Vec<f64> = results.iter().filter_map(|r| r.score).collect();
        let mean = (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64);
        EvaluationResult {
            overall_score: mean,
            overall_eval_status: get_eval_status(mean, 0.5),
            per_invocation_results: results.to_vec(),
            overall_rubric_scores: None,
        }
    }

    // --- LlmAsJudgeConfig::new ---

    fn eval_metric_with_criterion(value: Value) -> EvalMetric {
        EvalMetric::new("test_metric").with_criterion(value)
    }

    #[test]
    fn config_errors_without_a_criterion() {
        let eval_metric = EvalMetric::new("test_metric");
        let result = LlmAsJudgeConfig::<LlmAsAJudgeCriterion>::new(&eval_metric);
        assert!(matches!(result, Err(LlmAsJudgeError::InvalidCriterion(_))));
    }

    #[test]
    fn config_resolves_criterion_threshold_and_judge_model() {
        let criterion = LlmAsAJudgeCriterion {
            threshold: 0.7,
            include_intermediate_responses_in_final: false,
            judge_model_options: JudgeModelOptions {
                judge_model: "gemini-2.5-flash".to_string(),
                ..Default::default()
            },
        };
        let eval_metric =
            eval_metric_with_criterion(rusty_serde::json::to_value(&criterion).unwrap());
        let config = LlmAsJudgeConfig::<LlmAsAJudgeCriterion>::new(&eval_metric).unwrap();
        assert_eq!(config.threshold, 0.7);
        assert_eq!(config.judge_model.type_name(), "Gemini");
    }

    // --- evaluate_invocations_via_llm_judge ---

    #[rusty_tokio::test]
    async fn a_single_passing_sample_yields_an_overall_passed_status() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![Ok(text_response("0.9"))]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            num_samples: 1,
            ..Default::default()
        };
        let actual = vec![invocation("inv-1")];

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            false,
            &actual,
            None,
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await
        .unwrap();

        assert_eq!(result.overall_eval_status, EvalStatus::Passed);
        assert_eq!(result.per_invocation_results.len(), 1);
    }

    #[rusty_tokio::test]
    async fn multiple_samples_for_one_invocation_are_aggregated() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            // QueueLlm pops from the end, so this list is consumed
            // "1.0" then "0.0".
            queue: Mutex::new(vec![Ok(text_response("0.0")), Ok(text_response("1.0"))]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            num_samples: 2,
            ..Default::default()
        };
        let actual = vec![invocation("inv-1")];

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            false,
            &actual,
            None,
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await
        .unwrap();

        assert_eq!(result.per_invocation_results[0].score, Some(0.5));
    }

    #[rusty_tokio::test]
    async fn one_failed_sample_marks_the_whole_invocation_not_evaluated() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            // Consumed in reverse: "1.0" succeeds, then the error.
            queue: Mutex::new(vec![
                Err("safety filter".to_string()),
                Ok(text_response("1.0")),
            ]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            num_samples: 2,
            ..Default::default()
        };
        let actual = vec![invocation("inv-1")];

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            false,
            &actual,
            None,
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await
        .unwrap();

        assert_eq!(
            result.per_invocation_results[0].eval_status,
            EvalStatus::NotEvaluated
        );
        assert_eq!(result.per_invocation_results[0].score, None);
    }

    #[rusty_tokio::test]
    async fn errors_when_expected_invocations_are_required_but_missing() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            ..Default::default()
        };
        let actual = vec![invocation("inv-1")];

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            true,
            &actual,
            None,
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await;

        assert!(matches!(
            result,
            Err(LlmAsJudgeError::ExpectedInvocationsRequired)
        ));
    }

    #[rusty_tokio::test]
    async fn errors_on_mismatched_invocation_lengths() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            ..Default::default()
        };
        let actual = vec![invocation("inv-1"), invocation("inv-2")];
        let expected = vec![invocation("inv-1")];

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            false,
            &actual,
            Some(&expected),
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await;

        assert!(matches!(
            result,
            Err(LlmAsJudgeError::InvalidInvocations(_))
        ));
    }

    #[rusty_tokio::test]
    async fn empty_invocations_yield_a_default_result() {
        let llm = QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![]),
        };
        let options = JudgeModelOptions {
            judge_model: "test-model".to_string(),
            ..Default::default()
        };

        let result = evaluate_invocations_via_llm_judge(
            &llm,
            &options,
            0.5,
            false,
            &[],
            None,
            |_actual, _expected| "prompt".to_string(),
            score_from_text,
            mean_per_invocation,
            mean_overall,
        )
        .await
        .unwrap();

        assert_eq!(result, EvaluationResult::default());
    }
}
