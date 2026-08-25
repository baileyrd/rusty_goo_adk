//! C0601: `evaluation.rubric_based_evaluator`, ported from
//! `google.adk.evaluation.rubric_based_evaluator`.
//!
//! [`RubricBasedEvaluator`] composes a [`crate::llm_as_judge::LlmAsJudgeConfig`]
//! (C0600's harness) rather than inheriting `LlmAsJudge[RubricsBasedCriterion]`
//! the way the source does — see `llm_as_judge.rs`'s module doc for why that
//! harness is a free function instead of a trait: `format_auto_rater_prompt`
//! has no real implementation anywhere in this port (every concrete
//! per-metric evaluator that would supply one is GCP-blocked), so
//! `RubricBasedEvaluator` provides only the three hooks the source itself
//! actually implements — [`RubricBasedEvaluator::convert_auto_rater_response_to_score`],
//! [`RubricBasedEvaluator::aggregate_per_invocation_samples`],
//! [`RubricBasedEvaluator::aggregate_invocation_results`] — for a caller to
//! pass into [`crate::llm_as_judge::evaluate_invocations_via_llm_judge`]
//! alongside its own prompt-formatting closure.
//!
//! **`_normalized_rubric_to_id_map`, ported but unread, disclosed**: the
//! source builds `self._normalized_rubric_to_id_map` in `__init__` and
//! never reads it anywhere else in the file (verified by grepping the
//! source for the attribute) — `convert_auto_rater_response_to_score`
//! builds its own fresh normalized-text map locally instead. This looks
//! like vestigial state from an earlier version, but per this migration's
//! boundary contract "looks unused" is not license to drop it: it's
//! ported faithfully as [`RubricBasedEvaluator::normalized_rubric_to_id_map`],
//! a field a caller can still read even though this port's own code never
//! consults it either, matching the source exactly.
//!
//! **Lookbehind → capture group, adaptation disclosed**: the source's
//! `_RATIONALE_PATTERN`/`_VERDICT_PATTERN` use zero-width lookbehind
//! (`(?<=Rationale: )(.*)`) to capture the rest of a line without
//! including the label in the match. Rust's `regex` crate doesn't
//! support lookbehind (a deliberate limitation, for its linear-time
//! guarantee). This port rewrites both as an ordinary capturing group
//! (`Rationale: (.*)`) and reads capture group 1 instead of group 0 —
//! behaviorally identical here, since the source's own `re.findall`
//! already returns group 1's contents whenever a pattern has exactly one
//! group (as both of these do), never the lookbehind-inclusive group 0.
//!
//! **NFKC normalization, disclosed narrowing**: [`normalize_text`] skips
//! the source's `unicodedata.normalize("NFKC", text)` step — no
//! `unicode-normalization`-equivalent crate is a dependency of
//! `adk-eval` (only `adk-tools` has one, added for `skills_models`'s
//! `Frontmatter::normalize_name`). Same disclosed gap `rouge.rs` already
//! carries for the same reason; affects only compatibility-decomposable
//! characters (full-width variants, some ligatures) in judge-model
//! output, not the common case.
//!
//! **`HashMap` iteration order, disclosed**: both aggregators group
//! rubric scores by `rubric_id` in a `HashMap` rather than preserving
//! the source dict's insertion order — each aggregated `RubricScore` is
//! self-identifying by `rubric_id`, so the order of the returned list
//! doesn't affect correctness, only incidental ordering. Same disclosed
//! choice already made for `EvalConfig.criteria`.
//!
//! **`effective_rubrics_list`, widened to interior mutability**:
//! [`RubricBasedEvaluator::create_effective_rubrics_list`]/
//! [`RubricBasedEvaluator::get_effective_rubrics_list`] were `&mut self`/
//! `&self -> &[Rubric]` when this struct had no real caller yet (every
//! concrete per-metric evaluator needing them was GCP-blocked). Their
//! first real callers — `RubricBasedToolUseV1Evaluator`/
//! `RubricBasedFinalResponseQualityV1Evaluator`/
//! `RubricBasedMultiTurnTrajectoryEvaluator` — pass `format_auto_rater_prompt`
//! to [`crate::llm_as_judge::evaluate_invocations_via_llm_judge`] as a
//! plain `Fn` closure (not `FnMut`), so recomputing the effective rubrics
//! list per invocation needs interior mutability rather than `&mut self`.
//! `effective_rubrics_list` becomes a `RefCell`, `create_effective_rubrics_list`
//! relaxes to `&self` (every existing call site already binds its
//! receiver `mut`, so this widening breaks nothing), and
//! `get_effective_rubrics_list` returns an owned `Vec<Rubric>` instead of
//! `&[Rubric]` (a `Ref` guard can't outlive the borrow, and the list is a
//! handful of short strings — cloning it is not a real cost) — the same
//! "widen a placeholder once its first real consumer needs the shape"
//! precedent already used repeatedly elsewhere in this port (e.g.
//! `PerInvocationResult::rubric_scores` itself).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use adk_models::llm_response::LlmResponse;
use regex::Regex;
use rusty_serde::{Deserialize, Serialize};

use crate::eval_metrics::{EvalMetric, RubricsBasedCriterion};
use crate::eval_rubrics::{Rubric, RubricScore};
use crate::evaluator::{EvaluationResult, PerInvocationResult};
use crate::llm_as_judge::{AutoRaterScore, LlmAsJudgeConfig, LlmAsJudgeError};
use crate::llm_as_judge_utils::{get_average_rubric_score, get_eval_status, get_text_from_content};

/// `rubric_based_evaluator.RubricResponse` — internal data model to
/// represent a rubric's response from the auto-rater.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct RubricResponse {
    #[rusty_serde(default)]
    pub rubric_id: Option<String>,
    #[rusty_serde(default)]
    pub property_text: Option<String>,
    #[rusty_serde(default)]
    pub rationale: Option<String>,
    #[rusty_serde(default)]
    pub score: Option<f64>,
}

/// `rubric_based_evaluator.AutoRaterResponseParser` — an interface for
/// parsing an auto-rater's response.
pub trait AutoRaterResponseParser {
    fn parse(&self, auto_rater_response: &str) -> Vec<RubricResponse>;
}

fn id_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*ID: (.*)$").unwrap())
}

fn property_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*Property: (.*)$").unwrap())
}

fn rationale_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Lookbehind → capture group; see this module's doc.
    RE.get_or_init(|| Regex::new(r"Rationale: (.*)").unwrap())
}

fn verdict_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"Verdict: (.*)").unwrap())
}

/// `rubric_based_evaluator.DefaultAutoRaterResponseParser` — the default
/// implementation of [`AutoRaterResponseParser`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAutoRaterResponseParser;

impl AutoRaterResponseParser for DefaultAutoRaterResponseParser {
    /// Returns a list of `RubricResponse` parsed from the auto-rater's
    /// response. Matches each ID to the property it immediately
    /// precedes (not by index) so an omitted ID line can't shift a
    /// later ID onto an earlier property. Returns an empty list if the
    /// property/rationale/verdict counts disagree — a partial parse
    /// could otherwise silently omit a failed rubric and inflate the
    /// score.
    fn parse(&self, auto_rater_response: &str) -> Vec<RubricResponse> {
        let property_captures: Vec<_> = property_pattern()
            .captures_iter(auto_rater_response)
            .collect();
        // (start offset of the whole "ID: ..." line, the captured id text).
        let id_entries: Vec<(usize, String)> = id_pattern()
            .captures_iter(auto_rater_response)
            .map(|c| {
                let start = c.get(0).expect("group 0 always matches").start();
                let id_text = c.get(1).map_or("", |m| m.as_str()).trim().to_string();
                (start, id_text)
            })
            .collect();
        let rationales: Vec<&str> = rationale_pattern()
            .captures_iter(auto_rater_response)
            .map(|c| c.get(1).map_or("", |m| m.as_str()))
            .collect();
        let scores: Vec<Option<f64>> = verdict_pattern()
            .captures_iter(auto_rater_response)
            .map(|c| {
                let verdict = c.get(1).map_or("", |m| m.as_str()).to_lowercase();
                if verdict.contains("yes") {
                    Some(1.0)
                } else if verdict.contains("no") {
                    Some(0.0)
                } else {
                    None
                }
            })
            .collect();

        if !(property_captures.len() == rationales.len() && rationales.len() == scores.len()) {
            return Vec::new();
        }

        let property_starts: Vec<usize> = property_captures
            .iter()
            .map(|c| c.get(0).expect("group 0 always matches").start())
            .collect();

        let mut rubric_responses = Vec::with_capacity(property_captures.len());
        for i in 0..property_captures.len() {
            let previous_start: i64 = if i > 0 {
                property_starts[i - 1] as i64
            } else {
                -1
            };
            let property_start = property_starts[i] as i64;

            // Match each id to the property it immediately precedes (not
            // by index) so an omitted id line can't shift a later id
            // onto an earlier property.
            let mut rubric_id: Option<String> = None;
            for (id_start, id_text) in &id_entries {
                let start = *id_start as i64;
                if previous_start < start && start < property_start {
                    rubric_id = if id_text.is_empty() {
                        None
                    } else {
                        Some(id_text.clone())
                    };
                }
            }

            let property_text = property_captures[i]
                .get(1)
                .map_or("", |m| m.as_str())
                .trim()
                .to_string();
            rubric_responses.push(RubricResponse {
                rubric_id,
                property_text: Some(property_text),
                rationale: Some(rationales[i].trim().to_string()),
                score: scores[i],
            });
        }
        rubric_responses
    }
}

/// `rubric_based_evaluator.PerInvocationResultsAggregator` — an
/// interface for aggregating per-invocation samples. AutoRaters backed
/// by an LLM have a degree of unreliability, so the same invocation may
/// be sampled more than once; this converts multiple samples into a
/// single result.
pub trait PerInvocationResultsAggregator {
    fn aggregate(
        &self,
        per_invocation_samples: &[PerInvocationResult],
        threshold: f64,
    ) -> PerInvocationResult;
}

/// `rubric_based_evaluator.MajorityVotePerInvocationResultsAggregator` —
/// aggregates per-invocation samples using majority vote: for each
/// rubric, whichever verdict (yes/no) has more samples wins; a rubric
/// with no verdict in any sample keeps its no-score entry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MajorityVotePerInvocationResultsAggregator;

/// Buckets per `rubric_id`: (no-score, positive, negative).
type ScoreCategoryBuckets = (Vec<RubricScore>, Vec<RubricScore>, Vec<RubricScore>);

impl PerInvocationResultsAggregator for MajorityVotePerInvocationResultsAggregator {
    fn aggregate(
        &self,
        per_invocation_samples: &[PerInvocationResult],
        threshold: f64,
    ) -> PerInvocationResult {
        let mut score_category_by_rubric_id: HashMap<String, ScoreCategoryBuckets> = HashMap::new();

        for sample in per_invocation_samples {
            let Some(rubric_scores) = &sample.rubric_scores else {
                continue;
            };
            for rubric_score in rubric_scores {
                let entry = score_category_by_rubric_id
                    .entry(rubric_score.rubric_id.clone())
                    .or_default();
                match rubric_score.score {
                    None => entry.0.push(rubric_score.clone()),
                    Some(1.0) => entry.1.push(rubric_score.clone()),
                    Some(_) => entry.2.push(rubric_score.clone()),
                }
            }
        }

        let mut aggregated_rubric_scores = Vec::new();
        for (no_scores, positives, negatives) in score_category_by_rubric_id.into_values() {
            if positives.is_empty() && negatives.is_empty() {
                aggregated_rubric_scores.push(no_scores[0].clone());
            } else if positives.len() > negatives.len() {
                aggregated_rubric_scores.push(positives[0].clone());
            } else {
                aggregated_rubric_scores.push(negatives[0].clone());
            }
        }

        let aggregated_overall_score = get_average_rubric_score(&aggregated_rubric_scores);
        PerInvocationResult {
            actual_invocation: per_invocation_samples[0].actual_invocation.clone(),
            expected_invocation: per_invocation_samples[0].expected_invocation.clone(),
            score: aggregated_overall_score,
            rubric_scores: Some(aggregated_rubric_scores),
            eval_status: get_eval_status(aggregated_overall_score, threshold),
        }
    }
}

/// `rubric_based_evaluator.InvocationResultsSummarizer` — an interface
/// for summarizing per-invocation results into a single result for the
/// whole eval case.
pub trait InvocationResultsSummarizer {
    fn summarize(
        &self,
        per_invocation_results: &[PerInvocationResult],
        threshold: f64,
    ) -> EvaluationResult;
}

const AGGREGATED_RATIONALE: &str = "This is an aggregated score derived from individual entries. \
     Please refer to individual entries in each invocation for actual rationale from the model.";

/// `rubric_based_evaluator.MeanInvocationResultsSummarizer` —
/// summarizes per-invocation results using the mean score of each
/// rubric across every invocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanInvocationResultsSummarizer;

impl InvocationResultsSummarizer for MeanInvocationResultsSummarizer {
    fn summarize(
        &self,
        per_invocation_results: &[PerInvocationResult],
        threshold: f64,
    ) -> EvaluationResult {
        let mut unaggregated_rubric_scores = Vec::new();
        let mut rubric_scores_by_id: HashMap<String, Vec<RubricScore>> = HashMap::new();

        for sample in per_invocation_results {
            let Some(rubric_scores) = &sample.rubric_scores else {
                continue;
            };
            for rubric_score in rubric_scores {
                rubric_scores_by_id
                    .entry(rubric_score.rubric_id.clone())
                    .or_default()
                    .push(rubric_score.clone());
                unaggregated_rubric_scores.push(rubric_score.clone());
            }
        }

        let mut aggregated_rubric_scores = Vec::new();
        for (rubric_id, rubric_scores) in &rubric_scores_by_id {
            let overall_score = get_average_rubric_score(rubric_scores);
            aggregated_rubric_scores.push(RubricScore {
                rubric_id: rubric_id.clone(),
                score: overall_score,
                rationale: Some(AGGREGATED_RATIONALE.to_string()),
            });
        }

        let aggregated_overall_score = get_average_rubric_score(&unaggregated_rubric_scores);
        EvaluationResult {
            overall_score: aggregated_overall_score,
            overall_eval_status: get_eval_status(aggregated_overall_score, threshold),
            per_invocation_results: per_invocation_results.to_vec(),
            overall_rubric_scores: Some(aggregated_rubric_scores),
        }
    }
}

fn translate_smart_chars(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2013}' | '\u{2014}' => '-',
            other => other,
        })
        .collect()
}

fn whitespace_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s+").unwrap())
}

const DECORATION_CHARS: &[char] = &[' ', '*', '_', '`', '#', '>', '-', '\u{2022}', '"', '\''];

/// `rubric_based_evaluator._normalize_text` — a normalized version of
/// the given text, so a judge model's markdown/typographic decoration
/// around an echoed rubric doesn't defeat exact-match lookup. `None`
/// (the source's non-`str` case — this port narrows `object` to
/// `Option<&str>`, its only real callers) normalizes to an empty
/// string. See this module's doc for the skipped NFKC step.
pub fn normalize_text(text: Option<&str>) -> String {
    let Some(text) = text else {
        return String::new();
    };
    let translated = translate_smart_chars(text);
    let collapsed = whitespace_pattern().replace_all(&translated, " ");
    collapsed
        .trim_matches(|c: char| DECORATION_CHARS.contains(&c))
        .to_lowercase()
}

/// `rubric_based_evaluator.RubricBasedEvaluator` — a base for rubric-based
/// evaluators. See this module's doc for why this composes
/// [`LlmAsJudgeConfig`] rather than extending a trait the way the source
/// extends `LlmAsJudge[RubricsBasedCriterion]`.
pub struct RubricBasedEvaluator {
    pub config: LlmAsJudgeConfig<RubricsBasedCriterion>,
    rubric_type: Option<String>,
    auto_rater_response_parser: Box<dyn AutoRaterResponseParser>,
    per_invocation_results_aggregator: Box<dyn PerInvocationResultsAggregator>,
    invocation_results_summarizer: Box<dyn InvocationResultsSummarizer>,
    rubrics: Vec<Rubric>,
    /// Ported but never read by this port's own code either — see this
    /// module's doc.
    normalized_rubric_to_id_map: HashMap<String, String>,
    effective_rubrics_list: RefCell<Option<Vec<Rubric>>>,
}

impl RubricBasedEvaluator {
    /// `RubricBasedEvaluator.__init__`, using the source's own default
    /// values for `auto_rater_response_parser`,
    /// `per_invocation_results_aggregator`, and
    /// `invocation_results_summarizer` — override with
    /// [`RubricBasedEvaluator::with_auto_rater_response_parser`],
    /// [`RubricBasedEvaluator::with_per_invocation_results_aggregator`], or
    /// [`RubricBasedEvaluator::with_invocation_results_summarizer`].
    pub fn new(
        eval_metric: &EvalMetric,
        rubric_type: Option<String>,
    ) -> Result<Self, LlmAsJudgeError> {
        let config = LlmAsJudgeConfig::<RubricsBasedCriterion>::new(eval_metric)?;
        let rubrics = config.criterion.rubrics.clone();
        let normalized_rubric_to_id_map = rubrics
            .iter()
            .map(|r| {
                (
                    normalize_text(r.rubric_content.text_property.as_deref()),
                    r.rubric_id.clone(),
                )
            })
            .collect();
        Ok(Self {
            config,
            rubric_type,
            auto_rater_response_parser: Box::new(DefaultAutoRaterResponseParser),
            per_invocation_results_aggregator: Box::new(MajorityVotePerInvocationResultsAggregator),
            invocation_results_summarizer: Box::new(MeanInvocationResultsSummarizer),
            rubrics,
            normalized_rubric_to_id_map,
            effective_rubrics_list: RefCell::new(None),
        })
    }

    pub fn with_auto_rater_response_parser(
        mut self,
        parser: Box<dyn AutoRaterResponseParser>,
    ) -> Self {
        self.auto_rater_response_parser = parser;
        self
    }

    pub fn with_per_invocation_results_aggregator(
        mut self,
        aggregator: Box<dyn PerInvocationResultsAggregator>,
    ) -> Self {
        self.per_invocation_results_aggregator = aggregator;
        self
    }

    pub fn with_invocation_results_summarizer(
        mut self,
        summarizer: Box<dyn InvocationResultsSummarizer>,
    ) -> Self {
        self.invocation_results_summarizer = summarizer;
        self
    }

    /// Exposes the field the source never reads either — see this
    /// module's doc.
    pub fn normalized_rubric_to_id_map(&self) -> &HashMap<String, String> {
        &self.normalized_rubric_to_id_map
    }

    /// `RubricBasedEvaluator.create_effective_rubrics_list`. See this
    /// module's doc for why this is `&self` (interior mutability) rather
    /// than `&mut self`.
    pub fn create_effective_rubrics_list(
        &self,
        invocation_rubrics: Option<&[Rubric]>,
    ) -> Result<(), String> {
        let mut rubrics_by_id: Vec<(String, Rubric)> = Vec::new();
        let mut add_rubrics = |rubrics_to_add: &[Rubric], scope_name: &str| -> Result<(), String> {
            for r in rubrics_to_add {
                if rubrics_by_id.iter().any(|(id, _)| id == &r.rubric_id) {
                    return Err(format!(
                        "Rubric with rubric_id '{}' already exists. Rubric defined in {} \
                         conflicts with an existing rubric.",
                        r.rubric_id, scope_name
                    ));
                }
                rubrics_by_id.push((r.rubric_id.clone(), r.clone()));
            }
            Ok(())
        };

        add_rubrics(&self.rubrics, "criterion")?;

        if let Some(invocation_rubrics) = invocation_rubrics {
            if !invocation_rubrics.is_empty() {
                let filtered: Vec<Rubric> = match &self.rubric_type {
                    Some(rubric_type) => invocation_rubrics
                        .iter()
                        .filter(|r| r.rubric_type.as_deref() == Some(rubric_type.as_str()))
                        .cloned()
                        .collect(),
                    None => invocation_rubrics.to_vec(),
                };
                add_rubrics(&filtered, "invocation")?;
            }
        }

        let effective: Vec<Rubric> = rubrics_by_id.into_iter().map(|(_, r)| r).collect();
        if effective.is_empty() {
            return Err("Rubrics are required.".to_string());
        }
        *self.effective_rubrics_list.borrow_mut() = Some(effective);
        Ok(())
    }

    /// `RubricBasedEvaluator.get_effective_rubrics_list`. Returns an owned
    /// clone rather than `&[Rubric]` — see this module's doc.
    pub fn get_effective_rubrics_list(&self) -> Result<Vec<Rubric>, String> {
        self.effective_rubrics_list.borrow().clone().ok_or_else(|| {
            "Effective rubrics list not initialized. Call create_effective_rubrics_list() first."
                .to_string()
        })
    }

    /// `RubricBasedEvaluator.convert_auto_rater_response_to_score`.
    pub fn convert_auto_rater_response_to_score(
        &self,
        auto_rater_response: &LlmResponse,
    ) -> AutoRaterScore {
        let response_text = get_text_from_content(auto_rater_response.content.as_ref())
            .filter(|text| !text.is_empty());
        let rubric_responses = match response_text {
            None => {
                eprintln!(
                    "Auto-rater returned an empty response; no rubric verdicts could be \
                     parsed and this sample will not be scored."
                );
                Vec::new()
            }
            Some(text) => {
                let parsed = self.auto_rater_response_parser.parse(&text);
                if parsed.is_empty() {
                    eprintln!(
                        "Auto-rater response did not match the expected \
                         Property/Rationale/Verdict format; no rubric verdicts were \
                         parsed. Raw auto-rater response: {text}"
                    );
                }
                parsed
            }
        };

        // The source's `self.get_effective_rubrics_list()` call here
        // raises an uncaught `ValueError` if `create_effective_rubrics_list`
        // was never called; this closure-typed hook can't return `Result`
        // (see this module's doc), so the equivalent is a panic rather
        // than silently scoring against no rubrics.
        let effective_rubrics = self
            .get_effective_rubrics_list()
            .expect("create_effective_rubrics_list() must be called before scoring");
        let mut normalized_rubric_to_rubric_map: HashMap<String, &Rubric> = HashMap::new();
        let mut rubric_by_id: HashMap<&str, &Rubric> = HashMap::new();
        for r in &effective_rubrics {
            normalized_rubric_to_rubric_map
                .insert(normalize_text(r.rubric_content.text_property.as_deref()), r);
            rubric_by_id.insert(r.rubric_id.as_str(), r);
        }

        let mut rubric_scores = Vec::new();
        for rubric_response in &rubric_responses {
            let mut rubric = rubric_response
                .rubric_id
                .as_deref()
                .and_then(|id| rubric_by_id.get(id).copied());
            if rubric.is_none() {
                rubric = normalized_rubric_to_rubric_map
                    .get(&normalize_text(rubric_response.property_text.as_deref()))
                    .copied();
            }
            if let Some(rubric) = rubric {
                rubric_scores.push(RubricScore {
                    rubric_id: rubric.rubric_id.clone(),
                    rationale: rubric_response.rationale.clone(),
                    score: rubric_response.score,
                });
            } else {
                eprintln!(
                    "Rubric {:?} not found in the rubrics provided to the metric.",
                    rubric_response.property_text
                );
            }
        }

        let aggregated_score = get_average_rubric_score(&rubric_scores);
        AutoRaterScore {
            score: aggregated_score,
            rubric_scores: Some(rubric_scores),
        }
    }

    /// `RubricBasedEvaluator.aggregate_per_invocation_samples`.
    pub fn aggregate_per_invocation_samples(
        &self,
        per_invocation_samples: &[PerInvocationResult],
    ) -> PerInvocationResult {
        self.per_invocation_results_aggregator
            .aggregate(per_invocation_samples, self.config.threshold)
    }

    /// `RubricBasedEvaluator.aggregate_invocation_results`.
    pub fn aggregate_invocation_results(
        &self,
        per_invocation_results: &[PerInvocationResult],
    ) -> EvaluationResult {
        self.invocation_results_summarizer
            .summarize(per_invocation_results, self.config.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_case::Invocation;
    use crate::eval_metrics::{EvalMetric, JudgeModelOptions};
    use crate::evaluator::EvalStatus;
    use adk_genai::content::{Content, Part};

    #[test]
    fn default_parser_parses_a_well_formed_response() {
        let response = "\
ID: r1
Property: The response is concise.
Rationale: It is one sentence.
Verdict: Yes

ID: r2
Property: The response is polite.
Rationale: It uses please and thank you.
Verdict: No";
        let parsed = DefaultAutoRaterResponseParser.parse(response);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].rubric_id, Some("r1".to_string()));
        assert_eq!(
            parsed[0].property_text,
            Some("The response is concise.".to_string())
        );
        assert_eq!(parsed[0].rationale, Some("It is one sentence.".to_string()));
        assert_eq!(parsed[0].score, Some(1.0));
        assert_eq!(parsed[1].rubric_id, Some("r2".to_string()));
        assert_eq!(parsed[1].score, Some(0.0));
    }

    #[test]
    fn default_parser_tolerates_a_missing_id_line() {
        let response = "\
Property: The response is concise.
Rationale: It is one sentence.
Verdict: Yes";
        let parsed = DefaultAutoRaterResponseParser.parse(response);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].rubric_id, None);
    }

    #[test]
    fn default_parser_returns_empty_on_a_mismatched_count() {
        let response = "\
Property: only a property, no rationale or verdict";
        assert!(DefaultAutoRaterResponseParser.parse(response).is_empty());
    }

    #[test]
    fn default_parser_ignores_an_unparseable_verdict() {
        let response = "\
Property: The response is concise.
Rationale: Unclear.
Verdict: Maybe";
        let parsed = DefaultAutoRaterResponseParser.parse(response);
        assert_eq!(parsed[0].score, None);
    }

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

    fn per_invocation_result(rubric_scores: Vec<RubricScore>) -> PerInvocationResult {
        PerInvocationResult {
            actual_invocation: invocation(),
            expected_invocation: None,
            score: None,
            eval_status: EvalStatus::NotEvaluated,
            rubric_scores: Some(rubric_scores),
        }
    }

    #[test]
    fn majority_vote_picks_the_majority_verdict() {
        let samples = vec![
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: Some("a".to_string()),
                score: Some(1.0),
            }]),
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: Some("b".to_string()),
                score: Some(1.0),
            }]),
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: Some("c".to_string()),
                score: Some(0.0),
            }]),
        ];
        let result = MajorityVotePerInvocationResultsAggregator.aggregate(&samples, 0.5);
        let rubric_scores = result.rubric_scores.unwrap();
        assert_eq!(rubric_scores.len(), 1);
        assert_eq!(rubric_scores[0].score, Some(1.0));
        assert_eq!(result.score, Some(1.0));
        assert_eq!(result.eval_status, EvalStatus::Passed);
    }

    #[test]
    fn majority_vote_keeps_a_no_score_entry_when_nothing_else_exists() {
        let samples = vec![per_invocation_result(vec![RubricScore {
            rubric_id: "r1".to_string(),
            rationale: None,
            score: None,
        }])];
        let result = MajorityVotePerInvocationResultsAggregator.aggregate(&samples, 0.5);
        assert_eq!(result.rubric_scores.unwrap()[0].score, None);
        assert_eq!(result.score, None);
        assert_eq!(result.eval_status, EvalStatus::NotEvaluated);
    }

    #[test]
    fn majority_vote_ignores_samples_with_no_rubric_scores() {
        let mut no_scores_sample = per_invocation_result(vec![]);
        no_scores_sample.rubric_scores = None;
        let samples = vec![
            no_scores_sample,
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(1.0),
            }]),
        ];
        let result = MajorityVotePerInvocationResultsAggregator.aggregate(&samples, 0.5);
        assert_eq!(result.rubric_scores.unwrap().len(), 1);
    }

    #[test]
    fn mean_summarizer_averages_a_rubrics_score_across_invocations() {
        let results = vec![
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(1.0),
            }]),
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(0.0),
            }]),
        ];
        let summary = MeanInvocationResultsSummarizer.summarize(&results, 0.5);
        let rubric_scores = summary.overall_rubric_scores.unwrap();
        assert_eq!(rubric_scores.len(), 1);
        assert_eq!(rubric_scores[0].score, Some(0.5));
        assert_eq!(
            rubric_scores[0].rationale,
            Some(AGGREGATED_RATIONALE.to_string())
        );
        assert_eq!(summary.overall_score, Some(0.5));
        assert_eq!(summary.overall_eval_status, EvalStatus::Passed);
        assert_eq!(summary.per_invocation_results.len(), 2);
    }

    #[test]
    fn mean_summarizer_returns_none_score_without_any_rubric_scores() {
        let mut result = per_invocation_result(vec![]);
        result.rubric_scores = None;
        let summary = MeanInvocationResultsSummarizer.summarize(&[result], 0.5);
        assert_eq!(summary.overall_score, None);
        assert_eq!(summary.overall_eval_status, EvalStatus::NotEvaluated);
    }

    #[test]
    fn normalize_text_returns_empty_string_for_none() {
        assert_eq!(normalize_text(None), "");
    }

    #[test]
    fn normalize_text_lowercases_and_collapses_whitespace() {
        assert_eq!(normalize_text(Some("  The   Response  ")), "the response");
    }

    #[test]
    fn normalize_text_strips_markdown_decoration() {
        assert_eq!(
            normalize_text(Some("**bold rubric text**")),
            "bold rubric text"
        );
        assert_eq!(normalize_text(Some("- a list item")), "a list item");
        assert_eq!(normalize_text(Some("`code span`")), "code span");
    }

    #[test]
    fn normalize_text_maps_smart_quotes_and_dashes() {
        // The leading translated `'` is then itself stripped as a
        // decoration char -- `'` is one of `_DECORATION_CHARS`, matching
        // the real source's `.strip(_DECORATION_CHARS)` behavior
        // (verified against the source logic run standalone).
        assert_eq!(
            normalize_text(Some("\u{2018}quoted\u{2019} \u{2013} text")),
            "quoted' - text"
        );
    }

    // --- RubricBasedEvaluator ---

    use crate::eval_rubrics::RubricContent;

    fn rubric(id: &str, text: &str, rubric_type: Option<&str>) -> Rubric {
        Rubric {
            rubric_id: id.to_string(),
            rubric_content: RubricContent {
                text_property: Some(text.to_string()),
            },
            description: None,
            rubric_type: rubric_type.map(|t| t.to_string()),
        }
    }

    fn eval_metric_with_criterion(criterion: &RubricsBasedCriterion) -> EvalMetric {
        let value = rusty_serde::json::to_value(criterion).unwrap();
        EvalMetric::new("rubric_metric").with_criterion(value)
    }

    fn criterion(rubrics: Vec<Rubric>) -> RubricsBasedCriterion {
        RubricsBasedCriterion {
            threshold: 0.5,
            include_intermediate_responses_in_final: false,
            judge_model_options: JudgeModelOptions {
                judge_model: "gemini-2.5-flash".to_string(),
                ..Default::default()
            },
            rubrics,
        }
    }

    fn llm_response_text(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new("model", vec![Part::text(text)])),
            ..Default::default()
        }
    }

    #[test]
    fn new_parses_criterion_rubrics_and_populates_the_normalized_map() {
        let rubrics = vec![
            rubric("r1", "The response is concise.", None),
            rubric("r2", "The response is polite.", None),
        ];
        let eval_metric = eval_metric_with_criterion(&criterion(rubrics));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        assert_eq!(evaluator.config.threshold, 0.5);
        assert_eq!(evaluator.rubrics.len(), 2);
        assert_eq!(
            evaluator
                .normalized_rubric_to_id_map()
                .get("the response is concise."),
            Some(&"r1".to_string())
        );
    }

    #[test]
    fn create_effective_rubrics_list_merges_criterion_and_invocation_rubrics() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "criterion rubric", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        let invocation_rubrics = vec![rubric("r2", "invocation rubric", None)];
        evaluator
            .create_effective_rubrics_list(Some(&invocation_rubrics))
            .unwrap();
        let effective = evaluator.get_effective_rubrics_list().unwrap();
        assert_eq!(effective.len(), 2);
    }

    #[test]
    fn create_effective_rubrics_list_rejects_a_duplicate_rubric_id() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "criterion rubric", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        let invocation_rubrics = vec![rubric("r1", "duplicate id", None)];
        let result = evaluator.create_effective_rubrics_list(Some(&invocation_rubrics));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("r1"));
    }

    #[test]
    fn create_effective_rubrics_list_filters_invocation_rubrics_by_rubric_type() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "criterion rubric", None)]));
        let evaluator =
            RubricBasedEvaluator::new(&eval_metric, Some("FINAL_RESPONSE_QUALITY".to_string()))
                .unwrap();
        let invocation_rubrics = vec![
            rubric("r2", "matching type", Some("FINAL_RESPONSE_QUALITY")),
            rubric("r3", "other type", Some("TOOL_USE_QUALITY")),
        ];
        evaluator
            .create_effective_rubrics_list(Some(&invocation_rubrics))
            .unwrap();
        let effective = evaluator.get_effective_rubrics_list().unwrap();
        assert_eq!(effective.len(), 2);
        assert!(effective.iter().any(|r| r.rubric_id == "r2"));
        assert!(!effective.iter().any(|r| r.rubric_id == "r3"));
    }

    #[test]
    fn create_effective_rubrics_list_errors_without_any_rubrics() {
        let eval_metric = eval_metric_with_criterion(&criterion(vec![]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        let result = evaluator.create_effective_rubrics_list(None);
        assert!(result.is_err());
    }

    #[test]
    fn get_effective_rubrics_list_errors_before_initialization() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "criterion rubric", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        assert!(evaluator.get_effective_rubrics_list().is_err());
    }

    #[test]
    fn convert_auto_rater_response_to_score_matches_a_rubric_by_id() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "concise", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        evaluator.create_effective_rubrics_list(None).unwrap();
        let response =
            llm_response_text("ID: r1\nProperty: concise\nRationale: It is short.\nVerdict: Yes");
        let score = evaluator.convert_auto_rater_response_to_score(&response);
        let rubric_scores = score.rubric_scores.unwrap();
        assert_eq!(rubric_scores.len(), 1);
        assert_eq!(rubric_scores[0].rubric_id, "r1");
        assert_eq!(rubric_scores[0].score, Some(1.0));
        assert_eq!(score.score, Some(1.0));
    }

    #[test]
    fn convert_auto_rater_response_to_score_falls_back_to_normalized_text_match() {
        let eval_metric = eval_metric_with_criterion(&criterion(vec![rubric(
            "r1",
            "The response is concise.",
            None,
        )]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        evaluator.create_effective_rubrics_list(None).unwrap();
        // No "ID:" line, so the parser must fall back to matching the
        // normalized property text against the rubric's own text.
        let response = llm_response_text(
            "Property: **The response is concise.**\nRationale: Short.\nVerdict: No",
        );
        let score = evaluator.convert_auto_rater_response_to_score(&response);
        let rubric_scores = score.rubric_scores.unwrap();
        assert_eq!(rubric_scores.len(), 1);
        assert_eq!(rubric_scores[0].rubric_id, "r1");
        assert_eq!(rubric_scores[0].score, Some(0.0));
    }

    #[test]
    fn convert_auto_rater_response_to_score_skips_an_unmatched_rubric() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "concise", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        evaluator.create_effective_rubrics_list(None).unwrap();
        let response = llm_response_text(
            "ID: r-unknown\nProperty: something else\nRationale: n/a\nVerdict: Yes",
        );
        let score = evaluator.convert_auto_rater_response_to_score(&response);
        assert_eq!(score.rubric_scores.unwrap().len(), 0);
        assert_eq!(score.score, None);
    }

    #[test]
    fn convert_auto_rater_response_to_score_returns_empty_for_a_blank_response() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "concise", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        evaluator.create_effective_rubrics_list(None).unwrap();
        let response = LlmResponse {
            content: None,
            ..Default::default()
        };
        let score = evaluator.convert_auto_rater_response_to_score(&response);
        assert_eq!(score.rubric_scores.unwrap().len(), 0);
        assert_eq!(score.score, None);
    }

    #[test]
    fn aggregate_per_invocation_samples_delegates_to_the_configured_aggregator() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "concise", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        let samples = vec![per_invocation_result(vec![RubricScore {
            rubric_id: "r1".to_string(),
            rationale: None,
            score: Some(1.0),
        }])];
        let result = evaluator.aggregate_per_invocation_samples(&samples);
        assert_eq!(result.score, Some(1.0));
        assert_eq!(result.eval_status, EvalStatus::Passed);
    }

    #[test]
    fn aggregate_invocation_results_delegates_to_the_configured_summarizer() {
        let eval_metric =
            eval_metric_with_criterion(&criterion(vec![rubric("r1", "concise", None)]));
        let evaluator = RubricBasedEvaluator::new(&eval_metric, None).unwrap();
        let results = vec![
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(1.0),
            }]),
            per_invocation_result(vec![RubricScore {
                rubric_id: "r1".to_string(),
                rationale: None,
                score: Some(0.0),
            }]),
        ];
        let summary = evaluator.aggregate_invocation_results(&results);
        assert_eq!(summary.overall_score, Some(0.5));
        assert_eq!(summary.per_invocation_results.len(), 2);
    }
}
