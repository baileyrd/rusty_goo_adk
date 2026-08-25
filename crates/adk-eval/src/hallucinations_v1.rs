//! C0594: `evaluation.hallucinations_v1`, ported from
//! `google.adk.evaluation.hallucinations_v1`.
//!
//! [`HallucinationsV1Evaluator`] runs a two-stage pipeline per natural-
//! language response step: a segmenter LLM call splits the response into
//! sentences, then a validator LLM call labels each sentence
//! `supported`/`unsupported`/`contradictory`/`disputed`/`not_applicable`
//! against a constructed context (developer instructions, user prompt,
//! tool definitions, and the tool calls/outputs/NL responses of every
//! prior step). The Accuracy Score is the fraction of sentences labeled
//! `supported` or `not_applicable`.
//!
//! **Composes [`crate::llm_as_judge::LlmAsJudgeConfig`] rather than
//! reimplementing its own criterion-parsing/model-setup, disclosed
//! deviation**: the source's `HallucinationsV1Evaluator` extends
//! `Evaluator` directly (not `LlmAsJudge[CriterionT]`) and hand-rolls its
//! own `__init__`/`_setup_auto_rater` — but that hand-rolled logic is
//! functionally identical to `LlmAsJudge.__init__`'s (parse the criterion
//! out of `eval_metric.criterion`, resolve a threshold, resolve a judge
//! model off [`crate::eval_metrics::JudgeModelOptions`] via the shared
//! `LLMRegistry`). Reusing [`crate::llm_as_judge::LlmAsJudgeConfig`] here
//! avoids duplicating that setup a third time (`RubricBasedEvaluator`
//! already reuses it too) — same "compose the harness instead of
//! reinventing its already-real pieces" precedent that module's own doc
//! establishes. This needed one small, purely additive change:
//! `impl JudgeModelOptionsProvider for HallucinationsCriterion` below.
//!
//! **`judge_model_config`, not applied**: same disclosed gap
//! `llm_as_judge.rs`'s own module doc already establishes —
//! `JudgeModelOptions::judge_model_config` is an opaque `Value`
//! placeholder, so the outgoing segmenter/validator requests use
//! `GenerateContentConfigStub::default()` instead of merging it in.
//!
//! **`add_default_retry_options_if_not_present`, not ported**: same
//! disclosed gap already established for `llm_backed_user_simulator.rs`/
//! `llm_as_judge.rs` — `HttpOptionsStub` has no `retry_options` field,
//! and the source itself flags this helper as eval-systems-internal-only.
//!
//! **`_parse_validation_results`'s broad `except Exception`, not
//! ported**: the source wraps each regex-match's field extraction in a
//! `try/except Exception: logger.warning(...)` — defensive Python
//! against a possible exception in string ops on the captured groups.
//! Rust's `Regex::captures_iter` only yields matches whose capture
//! groups already exist (the pattern has no optional groups), so there
//! is no equivalent failure mode to catch here; [`parse_validation_results`]
//! ports the parsing logic without a matching try/except.
//!
//! **Tool-call/response JSON, compact not indented**: same disclosed
//! "compact JSON stand-in" precedent already established in
//! `llm_backed_user_simulator.rs`'s module doc — `rusty_serde::json`
//! has no pretty-printer, so `json.dumps(..., indent=2)` becomes a
//! single-line `rusty_serde::json::to_string`. A `None` field also
//! serializes as an explicit `null` rather than being omitted (no
//! `exclude_none=True` equivalent) — cosmetic differences in prompt
//! text, not in evaluation behavior.
//!
//! **`AppDetails::agent_details` iteration order, disclosed**: building
//! `developer_instructions` iterates every agent's instructions (unlike
//! `rubric_based_final_response_quality_v1.rs`'s single "current agent"
//! resolution) — since `AppDetails::agent_details` is a `HashMap` (an
//! already-established narrowing on that type, not one this module
//! introduces), the concatenation order of multiple agents' instruction
//! blocks is arbitrary rather than the source's dict-insertion order.
//! Each block is still fully present and self-labeled by agent name, so
//! this affects only incidental ordering, not which instructions appear.

use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};
use adk_models::llm_request::{GenerateContentConfigStub, LlmRequest};
use regex::Regex;
use rusty_serde::Serialize;

use crate::app_details::AppDetails;
use crate::eval_case::{IntermediateDataType, Invocation, InvocationEvent};
use crate::eval_metrics::{EvalMetric, HallucinationsCriterion, JudgeModelOptions};
use crate::evaluator::{
    validate_invocation_lengths, EvalStatus, EvaluationResult, PerInvocationResult,
};
use crate::llm_as_judge::{JudgeModelOptionsProvider, LlmAsJudgeConfig, LlmAsJudgeError};
use crate::llm_as_judge_utils::{
    get_eval_status, get_text_from_content, get_tool_declarations_as_json_str,
};

impl JudgeModelOptionsProvider for HallucinationsCriterion {
    fn judge_model_options(&self) -> &JudgeModelOptions {
        &self.judge_model_options
    }
}

const SEGMENTER_PROMPT: &str = r#"You are a helpful and harmless AI assistant. You will be provided with a model-generated response.
Your task is to segment the provided response sentence by sentence so that we could analyze each sentence in the future.

**Instructions:**
1. Overall, you should decompose the whole provided response into individual sentences. You should make sure the output covers ALL the sentences in the provided response block.
2. You should COPY each sentence as it is, WORD BY WORD. DO NOT modify the sentence or the surrounding punctuation.
3. If there are bullet points in the response, you should segment each bullet point into DIFFERENT sentences. If one bullet point has sub bullet points, you should further decompose sub bullet points into DIFFERENT sentences.
For example, if there are responses like "it has three criteria: * aaa. * bbb. * ccc", you should segment them into FOUR sentences: "it has three criteria", "aaa", "bbb", "ccc". Bullet points could start with numbers (1/2/3/etc) or symbols like "*", "-" etc.
4. When encountering tables, you should include the whole table in ONE sentence output.
5. Each sentence should be meaningful to further analyze on. DO NOT ONLY put symbols themselves into a sentence.
6. You should ONLY output segmented sentences in the provided response. DO NOT make up any new sentences.

**Input Format:**

The input will be the model-generated response:
* **Response:** The model-generated response to be analyzed.

**Output Format:**

For each decomposed sentence, wrap them with <sentence> and </sentence> like the following:
<sentence>...</sentence>
<sentence>...</sentence>

**Example:**

**Input:**

**Response Begin**
There are three kinds of fruits:
1. Apples are red.
2. Bananas are green.
3. Pears are purple.

For prices:
* Bananas are cheaper than apples.

Enjoy your fruit!
**Response End**

**Output:**
<sentence>There are three kinds of fruits:</sentence>
<sentence>1. Apples are red.</sentence>
<sentence>2. Bananas are green.</sentence>
<sentence>3. Pears are purple.</sentence>
<sentence>For prices:</sentence>
<sentence>* Bananas are cheaper than apples.</sentence>
<sentence>Enjoy your fruit!</sentence>

**Now, given the following response, please segment the response into sentences:**

**Input:**

**Response Begin**
{response}
**Response End**

**Your Sentence Segmentation Output:**"#;

const VALIDATOR_PROMPT: &str = r#"You are a helpful and harmless AI assistant. You will be provided with a textual context and sentences from a model-generated response.
Your task is to analyze sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Read the textual context carefully.**
2. **For each sentence, assign one of the following labels:**
    * **`supported`**: The sentence is entailed by the given context. Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence.
    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.
    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **`disputed`**: The given context contains both supporting and contradicting information. Provide both supporting and contradicting excerpt from the context.
    * **`not_applicable`**: The sentence does not require factual attribution (e.g., opinions, planning steps, greetings, questions, disclaimers, mathematical calculation).
3. **For each label, provide a short rationale explaining your decision.** The rationale should be separate from the excerpt.
4. **Be very strict with your `supported`, `contradictory` and `disputed` decisions.** Unless you can find straightforward, indisputable evidence excepts *in the context* that a sentence is `supported`, `contradictory` or `disputed`, consider it `unsupported`.  You should not employ world knowledge unless it is truly trivial.
5. "tool_outputs" blocks contain code execution results of the "tool_code" blocks immediately above them. If any sentence is based on "tool_outputs" results, first analyze if the corresponding "tool_code" is supported and if the results are error-free. Only if the "tool_code" block is supported, you can treat code execution results as correct.
6. If you need to cite multiple supporting excerpts, simply concatenate them. Excerpt could be summary from the context if it is too long.

**Input Format:**

The input will consist of two parts, clearly separated:

* **Context:**  The textual context used to generate the response.
* **Sentences:** The sentences from the model-generated response to be analyzed. Each sentence will be wrapped in <sentence>...</sentence>.

**Output Format:**

For each sentence, output a block of text with the following fields:

* sentence: The sentence being analyzed. Please directly copy the sentence which is provided.
* label: One of `supported`, `unsupported`, `contradictory`, `disputed` or `not_applicable`.
* rationale: A brief explanation for the assessment
* supporting_excerpt: A relevant excerpt from the context that supports the sentence. Only required for `supported` and `disputed` labels.
* contradicting_excerpt: A relevant excerpt from the context that contradicts with the sentence. Only required for `contradictory` and `disputed` labels.

**Example:**

**Input:**

**Context Begin**
Apples are red fruits. Bananas are yellow fruits. Pears are purple fruits. Pears are blue fruits.
**Context End**

**Sentences Begin**
<sentence>Apples are red.</sentence>
<sentence>Bananas are green.</sentence>
<sentence>Pears are purple.</sentence>
<sentence>Bananas are cheaper than apples.</sentence>
<sentence>Enjoy your fruit!</sentence>
**Sentences End**

**Output:**
sentence: Apples are red.
label: supported
rationale: The context explicitly states that apples are red.
supporting_excerpt: Apples are red fruits.
contradicting_excerpt: null

sentence: Bananas are green.
label: contradictory
rationale: The context states that bananas are yellow, not green.
supporting_excerpt: null
contradicting_excerpt: Bananas are yellow fruits.

sentence: Pears are purple.
label: disputed
rationale: The context states that pears are purple but it also states that pears are blue.
supporting_excerpt: Pears are purple fruits
contradicting_excerpt: Pears are blue fruits

sentence: Bananas are cheaper than apples.
label: unsupported
rationale: The context does not mention the price of bananas or apples.
supporting_excerpt: null
contradicting_excerpt: null

sentence: Enjoy your fruit!
label: not_applicable
rationale: This is a general expression and does not require factual attribution.
supporting_excerpt: null
contradicting_excerpt: null

**Now, please analyze the following context and sentences:**

**Input:**

**Context Begin**
{context}
**Context End**

**Sentences Begin**
{sentences}
**Sentences End**

**Output:**"#;

const POSITIVE_LABELS: &[&str] = &["supported", "not_applicable"];
const NEGATIVE_LABELS: &[&str] = &["unsupported", "contradictory", "disputed"];

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// `hallucinations_v1.EvaluationStep` — the context and natural-language
/// response to be evaluated at one step.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationStep {
    pub context: String,
    pub nl_response: String,
}

/// `hallucinations_v1._parse_sentences`.
pub fn parse_sentences(response_text: &str) -> Vec<String> {
    let pattern = Regex::new(r"(?s)<sentence>(.*?)</sentence>").expect("valid regex");
    pattern
        .captures_iter(response_text)
        .map(|cap| cap[1].to_string())
        .collect()
}

/// One parsed validation-result block from
/// [`parse_validation_results`]/`hallucinations_v1._parse_validation_results`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ValidationResult {
    pub sentence: String,
    pub label: String,
    pub rationale: String,
    pub supporting_excerpt: Option<String>,
    pub contradicting_excerpt: Option<String>,
}

fn normalize_excerpt(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("null") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// `hallucinations_v1._parse_validation_results`. See the module doc for
/// why this doesn't port the source's per-match `try/except`, and for
/// the lookahead → manual-boundary-split rewrite (Rust's `regex` crate
/// has no look-ahead, a deliberate limitation for its linear-time
/// guarantee — same class of adaptation `rubric_based_evaluator.rs`'s
/// own doc already makes for lookbehind).
pub fn parse_validation_results(response_text: &str) -> Vec<ValidationResult> {
    let text = response_text.trim();
    let boundary = Regex::new(r"(?i)\nsentence:").expect("valid regex");
    let block_pattern = Regex::new(
        r"(?is)\Asentence:(.*?)\nlabel:(.*?)\nrationale:(.*?)\nsupporting_excerpt:(.*?)\ncontradicting_excerpt:(.*)\z",
    )
    .expect("valid regex");

    // Each block starts at a `sentence:` marker; the source's trailing
    // `(?=\nsentence:|\Z)` lookahead stopped the final capture group at
    // the next such marker (or end of string) without consuming it —
    // splitting on the boundary up front, keeping `sentence:` itself as
    // part of the following block, achieves the same segmentation.
    let mut starts = vec![0usize];
    for m in boundary.find_iter(text) {
        starts.push(m.start() + 1);
    }
    starts.push(text.len());

    starts
        .windows(2)
        .filter_map(|window| block_pattern.captures(text[window[0]..window[1]].trim()))
        .map(|cap| ValidationResult {
            sentence: cap[1].trim().to_string(),
            label: cap[2].trim().to_string(),
            rationale: cap[3].trim().to_string(),
            supporting_excerpt: normalize_excerpt(&cap[4]),
            contradicting_excerpt: normalize_excerpt(&cap[5]),
        })
        .collect()
}

/// `hallucinations_v1.HallucinationsV1Evaluator`. See the module doc for
/// why this composes [`LlmAsJudgeConfig`] instead of hand-rolling its own
/// criterion/model setup, and does not implement
/// [`crate::evaluator::Evaluator`] (same async/sync structural reason as
/// every other LLM-judge-backed evaluator in this crate).
pub struct HallucinationsV1Evaluator {
    config: LlmAsJudgeConfig<HallucinationsCriterion>,
    model: String,
    model_config: GenerateContentConfigStub,
}

impl HallucinationsV1Evaluator {
    pub fn new(eval_metric: &EvalMetric) -> Result<Self, LlmAsJudgeError> {
        let config: LlmAsJudgeConfig<HallucinationsCriterion> = LlmAsJudgeConfig::new(eval_metric)?;
        let model = config.criterion.judge_model_options.judge_model.clone();
        Ok(Self {
            config,
            model,
            model_config: GenerateContentConfigStub::default(),
        })
    }

    /// `HallucinationsV1Evaluator._create_context_for_step`.
    fn create_context_for_step(
        app_details: Option<&AppDetails>,
        invocation: &Invocation,
        events: &[InvocationEvent],
    ) -> String {
        let mut developer_instructions = String::new();
        let mut tool_declarations = "Agent has no tools.".to_string();
        if let Some(app_details) = app_details {
            let instructions: Vec<String> = app_details
                .agent_details
                .iter()
                .filter(|(_, details)| !details.instructions.is_empty())
                .map(|(agent_name, details)| format!("{agent_name}:\n{}", details.instructions))
                .collect();
            developer_instructions = instructions.join("\n\n");
            tool_declarations =
                get_tool_declarations_as_json_str(app_details).unwrap_or_else(|error| error);
        }

        let mut context_parts = vec![
            format!("Developer instructions:\n{developer_instructions}\n"),
            format!(
                "User prompt:\n{}\n",
                get_text_from_content(Some(&invocation.user_content)).unwrap_or_default()
            ),
            "Tool definitions:".to_string(),
            format!("{tool_declarations}\n"),
        ];

        for event in events {
            let Some(content) = &event.content else {
                continue;
            };
            if content.parts.is_empty() {
                continue;
            }

            let tool_calls: Vec<FunctionCall> = content
                .parts
                .iter()
                .filter_map(|part| part.function_call.clone())
                .collect();
            let tool_responses: Vec<FunctionResponse> = content
                .parts
                .iter()
                .filter_map(|part| part.function_response.clone())
                .collect();
            let nl_responses: Vec<&str> = content
                .parts
                .iter()
                .filter_map(|part| part.text.as_deref())
                .filter(|text| !text.is_empty())
                .collect();

            if !nl_responses.is_empty() {
                context_parts.push(format!("{}\n", nl_responses.join("\n")));
            }

            if !tool_calls.is_empty() {
                context_parts.push("tool_calls:".to_string());
                let json = rusty_serde::json::to_string(&tool_calls).unwrap_or_default();
                context_parts.push(format!("{json}\n"));
            }
            if !tool_responses.is_empty() {
                context_parts.push("tool_outputs:".to_string());
                let json = rusty_serde::json::to_string(&tool_responses).unwrap_or_default();
                context_parts.push(format!("{json}\n"));
            }
        }

        context_parts.join("\n")
    }

    /// `HallucinationsV1Evaluator._evaluate_nl_response` — runs
    /// segmentation and validation for a single NL response. The second
    /// tuple element (an error/status message) is discarded by
    /// [`Self::evaluate_invocations`], same as the source's own
    /// `fs_score, _ = ...` call site — kept for direct testability of
    /// the two-stage pipeline.
    async fn evaluate_nl_response(
        &self,
        nl_response: &str,
        context: &str,
    ) -> (Option<f64>, String) {
        let segmenter_prompt = SEGMENTER_PROMPT.replace("{response}", nl_response);
        let mut segmenter_request = LlmRequest::new(self.model.clone());
        segmenter_request.config = self.model_config.clone();
        segmenter_request.contents = vec![Content::new("user", vec![Part::text(segmenter_prompt)])];

        let sentences = match self
            .config
            .judge_model
            .generate_content_async(&segmenter_request, false)
            .await
        {
            Ok(responses) => {
                let Some(response) = responses.into_iter().next() else {
                    return (None, "Segmenter returned no text.".to_string());
                };
                let Some(text) = get_text_from_content(response.content.as_ref()) else {
                    return (None, "Segmenter returned no text.".to_string());
                };
                parse_sentences(&text)
            }
            Err(error) => return (None, format!("Error during sentence segmentation: {error}")),
        };

        if sentences.is_empty() {
            return (None, "No sentences produced by segmenter.".to_string());
        }

        let sentences_str = sentences
            .iter()
            .map(|s| format!("<sentence>{s}</sentence>"))
            .collect::<Vec<_>>()
            .join("\n");

        let validator_prompt = VALIDATOR_PROMPT
            .replace("{context}", context)
            .replace("{sentences}", &sentences_str);
        let mut validator_request = LlmRequest::new(self.model.clone());
        validator_request.config = self.model_config.clone();
        validator_request.contents = vec![Content::new("user", vec![Part::text(validator_prompt)])];

        let validation_results = match self
            .config
            .judge_model
            .generate_content_async(&validator_request, false)
            .await
        {
            Ok(responses) => {
                let Some(response) = responses.into_iter().next() else {
                    return (None, "Sentence validator returned no text.".to_string());
                };
                let Some(text) = get_text_from_content(response.content.as_ref()) else {
                    return (None, "Sentence validator returned no text.".to_string());
                };
                parse_validation_results(&text)
            }
            Err(error) => return (None, format!("Error during sentence validation: {error}")),
        };

        let scores: Vec<f64> = validation_results
            .iter()
            .filter_map(|result| {
                let label = result.label.trim().to_lowercase();
                if POSITIVE_LABELS.contains(&label.as_str()) {
                    Some(1.0)
                } else if NEGATIVE_LABELS.contains(&label.as_str()) {
                    Some(0.0)
                } else {
                    None
                }
            })
            .collect();

        let accuracy_score = if scores.is_empty() {
            None
        } else {
            Some(mean(&scores))
        };
        let json = rusty_serde::json::to_string(&validation_results).unwrap_or_default();
        (accuracy_score, json)
    }

    /// `HallucinationsV1Evaluator._get_steps_to_evaluate`.
    fn get_steps_to_evaluate(&self, actual: &Invocation) -> Vec<EvaluationStep> {
        let mut step_evaluations = Vec::new();
        let all_events: Vec<InvocationEvent> = match actual.intermediate_data_type() {
            Some(IntermediateDataType::Events(events)) => events.invocation_events,
            _ => Vec::new(),
        };

        let events_for_context: Vec<InvocationEvent> =
            if self.config.criterion.evaluate_intermediate_nl_responses {
                let mut accumulated = Vec::new();
                for event in &all_events {
                    let nl_parts: Vec<&str> = event
                        .content
                        .as_ref()
                        .map(|content| {
                            content
                                .parts
                                .iter()
                                .filter_map(|part| part.text.as_deref())
                                .filter(|text| !text.is_empty())
                                .collect()
                        })
                        .unwrap_or_default();
                    if !nl_parts.is_empty() {
                        let context = Self::create_context_for_step(
                            actual.app_details.as_ref(),
                            actual,
                            &accumulated,
                        );
                        for nl_response in nl_parts {
                            step_evaluations.push(EvaluationStep {
                                nl_response: nl_response.to_string(),
                                context: context.clone(),
                            });
                        }
                    }
                    accumulated.push(event.clone());
                }
                accumulated
            } else {
                all_events
            };

        if let Some(final_response_text) = get_text_from_content(actual.final_response.as_ref()) {
            if !final_response_text.is_empty() {
                let context = Self::create_context_for_step(
                    actual.app_details.as_ref(),
                    actual,
                    &events_for_context,
                );
                step_evaluations.push(EvaluationStep {
                    nl_response: final_response_text,
                    context,
                });
            }
        }
        step_evaluations
    }

    /// `HallucinationsV1Evaluator._aggregate_invocation_results`.
    fn aggregate_invocation_results(
        &self,
        per_invocation_results: Vec<PerInvocationResult>,
    ) -> EvaluationResult {
        let valid_scores: Vec<f64> = per_invocation_results
            .iter()
            .filter_map(|result| result.score)
            .collect();
        if valid_scores.is_empty() {
            return EvaluationResult {
                overall_score: None,
                overall_eval_status: EvalStatus::NotEvaluated,
                per_invocation_results,
                overall_rubric_scores: None,
            };
        }

        let overall_score = mean(&valid_scores);
        EvaluationResult {
            overall_score: Some(overall_score),
            overall_eval_status: get_eval_status(
                Some(overall_score),
                self.config.criterion.threshold,
            ),
            per_invocation_results,
            overall_rubric_scores: None,
        }
    }

    /// `HallucinationsV1Evaluator.evaluate_invocations`. The source's
    /// `conversation_scenario` parameter is dropped immediately
    /// (`del conversation_scenario  # not used by this metric.`) — same
    /// omission `rubric_based_final_response_quality_v1.rs`'s own
    /// `evaluate_invocations` makes.
    pub async fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
    ) -> Result<EvaluationResult, LlmAsJudgeError> {
        validate_invocation_lengths(actual_invocations, expected_invocations)
            .map_err(LlmAsJudgeError::InvalidInvocations)?;

        let expected_by_invocation: Vec<Option<&Invocation>> = match expected_invocations {
            Some(expected) => expected.iter().map(Some).collect(),
            None => vec![None; actual_invocations.len()],
        };

        let mut per_invocation_results = Vec::new();
        for (actual, expected) in actual_invocations.iter().zip(expected_by_invocation.iter()) {
            let step_evaluations = self.get_steps_to_evaluate(actual);

            if step_evaluations.is_empty() {
                per_invocation_results.push(PerInvocationResult {
                    actual_invocation: actual.clone(),
                    expected_invocation: expected.cloned(),
                    score: None,
                    eval_status: EvalStatus::NotEvaluated,
                    rubric_scores: None,
                });
                continue;
            }

            let mut scores_per_step = Vec::new();
            for step in &step_evaluations {
                let (score, _) = self
                    .evaluate_nl_response(&step.nl_response, &step.context)
                    .await;
                if let Some(score) = score {
                    scores_per_step.push(score);
                }
            }

            let invocation_score = if scores_per_step.is_empty() {
                None
            } else {
                Some(mean(&scores_per_step))
            };

            per_invocation_results.push(PerInvocationResult {
                actual_invocation: actual.clone(),
                expected_invocation: expected.cloned(),
                score: invocation_score,
                eval_status: get_eval_status(invocation_score, self.config.criterion.threshold),
                rubric_scores: None,
            });
        }

        if per_invocation_results.is_empty() {
            Ok(EvaluationResult::default())
        } else {
            Ok(self.aggregate_invocation_results(per_invocation_results))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_details::AgentDetails;
    use crate::eval_case::InvocationEvents;
    use crate::eval_metrics::JudgeModelOptions;
    use adk_models::base_llm::{BaseLlm, BaseLlmError};
    use adk_models::llm_response::LlmResponse;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    #[test]
    fn parse_sentences_extracts_each_tagged_sentence() {
        let text = "<sentence>First one.</sentence>\n<sentence>Second one.</sentence>";
        assert_eq!(
            parse_sentences(text),
            vec!["First one.".to_string(), "Second one.".to_string()]
        );
    }

    #[test]
    fn parse_sentences_returns_empty_when_no_tags_present() {
        assert!(parse_sentences("no tags here").is_empty());
    }

    #[test]
    fn parse_validation_results_parses_multiple_blocks_with_null_excerpts() {
        let text = "sentence: Apples are red.\nlabel: supported\nrationale: The context says so.\nsupporting_excerpt: Apples are red fruits.\ncontradicting_excerpt: null\n\nsentence: Bananas are green.\nlabel: contradictory\nrationale: Contradicts the context.\nsupporting_excerpt: null\ncontradicting_excerpt: Bananas are yellow.";
        let results = parse_validation_results(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].sentence, "Apples are red.");
        assert_eq!(results[0].label, "supported");
        assert_eq!(
            results[0].supporting_excerpt.as_deref(),
            Some("Apples are red fruits.")
        );
        assert_eq!(results[0].contradicting_excerpt, None);
        assert_eq!(results[1].label, "contradictory");
        assert_eq!(results[1].supporting_excerpt, None);
        assert_eq!(
            results[1].contradicting_excerpt.as_deref(),
            Some("Bananas are yellow.")
        );
    }

    fn eval_metric() -> EvalMetric {
        let criterion = HallucinationsCriterion {
            threshold: 0.5,
            include_intermediate_responses_in_final: false,
            judge_model_options: JudgeModelOptions {
                judge_model: "gemini-2.5-flash".to_string(),
                ..Default::default()
            },
            evaluate_intermediate_nl_responses: false,
        };
        let value = rusty_serde::json::to_value(&criterion).unwrap();
        EvalMetric::new("hallucinations_v1").with_criterion(value)
    }

    fn invocation() -> Invocation {
        Invocation {
            invocation_id: "inv-1".to_string(),
            user_content: Content::user_text("Get the current time in PST."),
            final_response: Some(Content::new(
                "model",
                vec![Part::text("The current time is 10:30 AM PST.")],
            )),
            intermediate_data: Some(
                rusty_serde::json::to_value(&InvocationEvents {
                    invocation_events: vec![InvocationEvent {
                        author: "root".to_string(),
                        content: Some(Content::new(
                            "root",
                            vec![Part {
                                function_response: Some(FunctionResponse {
                                    name: Some("get_current_time".to_string()),
                                    response: Some(
                                        [(
                                            "result".to_string(),
                                            rusty_serde::value::Value::String(
                                                "10:30 AM PST".to_string(),
                                            ),
                                        )]
                                        .into_iter()
                                        .collect(),
                                    ),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }],
                        )),
                        grounding_metadata: None,
                    }],
                })
                .unwrap(),
            ),
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: Some(AppDetails {
                agent_details: [(
                    "root".to_string(),
                    AgentDetails {
                        name: "root".to_string(),
                        instructions: "You can tell the time.".to_string(),
                        tool_declarations: vec![],
                    },
                )]
                .into_iter()
                .collect(),
            }),
        }
    }

    #[test]
    fn create_context_for_step_includes_instructions_prompt_and_tool_outputs() {
        let inv = invocation();
        let events = match inv.intermediate_data_type() {
            Some(IntermediateDataType::Events(events)) => events.invocation_events,
            _ => panic!("expected events"),
        };
        let context = HallucinationsV1Evaluator::create_context_for_step(
            inv.app_details.as_ref(),
            &inv,
            &events,
        );
        assert!(context.contains("You can tell the time."));
        assert!(context.contains("Get the current time in PST."));
        assert!(context.contains("tool_outputs:"));
        assert!(context.contains("10:30 AM PST"));
    }

    #[test]
    fn get_steps_to_evaluate_only_yields_the_final_response_by_default() {
        let evaluator = HallucinationsV1Evaluator::new(&eval_metric()).unwrap();
        let steps = evaluator.get_steps_to_evaluate(&invocation());
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].nl_response, "The current time is 10:30 AM PST.");
    }

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
            _llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let next = self.queue.lock().unwrap().pop();
            Box::pin(async move { Ok(next.into_iter().collect()) })
        }
    }

    fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new("model", vec![Part::text(text)])),
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn evaluate_invocations_scores_a_fully_supported_response() {
        let mut evaluator = HallucinationsV1Evaluator::new(&eval_metric()).unwrap();
        // Popped in reverse order: validator response first pushed, segmenter second.
        evaluator.config.judge_model = Box::new(QueueLlm {
            model: "test-model".to_string(),
            queue: Mutex::new(vec![
                text_response(
                    "sentence: The current time is 10:30 AM PST.\nlabel: supported\nrationale: \
                     Matches the tool output.\nsupporting_excerpt: 10:30 AM PST\n\
                     contradicting_excerpt: null",
                ),
                text_response("<sentence>The current time is 10:30 AM PST.</sentence>"),
            ]),
        });

        let actual = vec![invocation()];
        let result = evaluator.evaluate_invocations(&actual, None).await.unwrap();
        assert_eq!(result.overall_score, Some(1.0));
        assert_eq!(result.overall_eval_status, EvalStatus::Passed);
    }

    #[rusty_tokio::test]
    async fn evaluate_invocations_is_not_evaluated_with_no_steps() {
        let evaluator = HallucinationsV1Evaluator::new(&eval_metric()).unwrap();
        let mut invocation_without_response = invocation();
        invocation_without_response.final_response = None;
        let actual = vec![invocation_without_response];
        let result = evaluator.evaluate_invocations(&actual, None).await.unwrap();
        assert_eq!(result.overall_eval_status, EvalStatus::NotEvaluated);
    }
}
