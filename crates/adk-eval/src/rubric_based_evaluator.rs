//! C0601 (partial): `evaluation.rubric_based_evaluator`, ported from
//! `google.adk.evaluation.rubric_based_evaluator`.
//!
//! **Partial**: only the harness-independent pieces are ported —
//! [`RubricResponse`], [`AutoRaterResponseParser`]/
//! [`DefaultAutoRaterResponseParser`],
//! [`PerInvocationResultsAggregator`]/
//! [`MajorityVotePerInvocationResultsAggregator`],
//! [`InvocationResultsSummarizer`]/[`MeanInvocationResultsSummarizer`],
//! and [`normalize_text`]. The source's `RubricBasedEvaluator` itself
//! extends `LlmAsJudge[RubricsBasedCriterion]` (C0600's still-deferred
//! harness) and returns `AutoRaterScore` (from `llm_as_judge.py`, also
//! unbuilt) — neither is ported this batch. Every function/type here has
//! zero dependency on that harness, the same reasoning already
//! established for the C0612 criterion types and the C0632 persona
//! system.
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

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use rusty_serde::{Deserialize, Serialize};

use crate::eval_rubrics::RubricScore;
use crate::evaluator::{EvaluationResult, PerInvocationResult};
use crate::llm_as_judge_utils::{get_average_rubric_score, get_eval_status};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_case::Invocation;
    use crate::evaluator::EvalStatus;
    use adk_genai::content::Content;

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
}
