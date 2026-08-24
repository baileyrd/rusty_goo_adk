//! Part of C0590 — the Unicode-aware tokenizer and ROUGE-1 scoring core
//! behind `RougeEvaluator` ([`crate::final_response_match_v1`]), ported
//! from `google.adk.dependencies.rouge_scorer` (a thin re-export of the
//! pip `rouge_score` package) plus `final_response_match_v1.py`'s own
//! `_UnicodeAwareTokenizer`.
//!
//! Scope: only what `RougeEvaluator` actually exercises —
//! `RougeScorer(["rouge1"], tokenizer=_UnicodeAwareTokenizer(use_stemmer=True))`.
//! `rouge_scorer.py`'s `rougeL`/`rougeLsum` (longest-common-subsequence)
//! code paths, `score_multi`, and `rouge2`+ n-gram sizes are all unreached
//! by this consumer and not ported.
//!
//! **Disclosed narrowing**: the source's `_UnicodeAwareTokenizer.tokenize`
//! opens with `unicodedata.normalize("NFKC", text)` before lowercasing.
//! NFKC (compatibility decomposition + canonical composition) mainly
//! affects things like full-width Latin/digit variants, certain ligatures,
//! and typographic-vs-plain punctuation forms — this port skips it and
//! lowercases the input as-is. Text already in normalization form C/D (the
//! overwhelming common case for real agent output) tokenizes identically
//! either way; text containing compatibility-decomposable characters may
//! tokenize slightly differently than the source. A `unicode-normalization`
//! dependency would close this gap if a real workload needs it.

use std::collections::HashMap;

use unicode_general_category::{get_general_category, GeneralCategory};

use crate::porter_stemmer;

fn is_cjk(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3040..=0x309F).contains(&code)
        || (0x30A0..=0x30FF).contains(&code)
        || (0xAC00..=0xD7AF).contains(&code)
}

fn is_non_spaced_script(c: char) -> bool {
    let code = c as u32;
    (0x0E00..=0x0E7F).contains(&code)
        || (0x0E80..=0x0EFF).contains(&code)
        || (0x1780..=0x17FF).contains(&code)
        || (0x1000..=0x109F).contains(&code)
}

fn is_mark(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || is_mark(c)
}

/// `rouge_score.tokenize.tokenize` — the plain ASCII tokenizer
/// (`DefaultTokenizer`), applied by [`tokenize`] to each ASCII "word" chunk
/// it finds. Ported exactly: lowercase, collapse any run of non-`[a-z0-9]`
/// characters to a single space, split on whitespace, stem tokens longer
/// than 3 characters (when `use_stemmer`), then keep only tokens that are
/// entirely `[a-z0-9]+`.
fn default_tokenize(text: &str, use_stemmer: bool) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut collapsed = String::with_capacity(lowered.len());
    let mut last_was_space = false;
    for c in lowered.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            collapsed.push(c);
            last_was_space = false;
        } else if !last_was_space {
            collapsed.push(' ');
            last_was_space = true;
        }
    }

    let mut tokens: Vec<String> = collapsed
        .split_whitespace()
        .map(|s| {
            if use_stemmer && s.len() > 3 {
                porter_stemmer::stem(s)
            } else {
                s.to_string()
            }
        })
        .collect();

    tokens.retain(|t| {
        !t.is_empty()
            && t.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    });
    tokens
}

/// `final_response_match_v1._UnicodeAwareTokenizer.tokenize` (`use_stemmer`
/// hardcoded `True`, the only construction `_calculate_rouge_1_scores`
/// ever uses). See the module doc for the disclosed NFKC-normalization
/// narrowing.
pub fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut processed = String::with_capacity(lowered.len());
    for c in lowered.chars() {
        if is_cjk(c) {
            processed.push(' ');
            processed.push(c);
            processed.push(' ');
        } else if is_non_spaced_script(c) {
            if is_mark(c) {
                processed.push(c);
            } else {
                processed.push(' ');
                processed.push(c);
            }
        } else if is_word_char(c) {
            processed.push(c);
        } else {
            processed.push(' ');
        }
    }

    let mut tokens = Vec::new();
    for word in processed.split_whitespace() {
        if word.is_ascii() {
            tokens.extend(default_tokenize(word, true));
        } else {
            tokens.push(word.to_string());
        }
    }
    tokens
}

/// `rouge_scorer.scoring.Score`, narrowed to just the `rouge1` type this
/// port computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rouge1Score {
    pub precision: f64,
    pub recall: f64,
    pub fmeasure: f64,
}

fn fmeasure(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn token_counts(tokens: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    counts
}

/// `RougeScorer(["rouge1"], tokenizer=_UnicodeAwareTokenizer(use_stemmer=True)).score(target, prediction)["rouge1"]`.
/// `rouge_scorer._create_ngrams`/`_score_ngrams` specialize to unigram
/// (`n=1`) multiset overlap when inlined this way — a unigram *is* a
/// token, so no ngram-tuple wrapping is needed.
pub fn score(target: &str, prediction: &str) -> Rouge1Score {
    let target_tokens = tokenize(target);
    let prediction_tokens = tokenize(prediction);
    let target_counts = token_counts(&target_tokens);
    let prediction_counts = token_counts(&prediction_tokens);

    let mut intersection = 0usize;
    for (token, &count) in &target_counts {
        let prediction_count = prediction_counts.get(token).copied().unwrap_or(0);
        intersection += count.min(prediction_count);
    }
    let target_total: usize = target_counts.values().sum();
    let prediction_total: usize = prediction_counts.values().sum();

    let precision = intersection as f64 / (prediction_total.max(1) as f64);
    let recall = intersection as f64 / (target_total.max(1) as f64);
    Rouge1Score {
        precision,
        recall,
        fmeasure: fmeasure(precision, recall),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_and_stems_plain_ascii_text() {
        assert_eq!(
            tokenize("The quick brown foxes jumps!"),
            vec!["the", "quick", "brown", "fox", "jump"]
        );
    }

    #[test]
    fn short_tokens_are_not_stemmed() {
        // "is"/"a" are len <= 3, left untouched by the stemmer gate.
        assert_eq!(tokenize("is a cat"), vec!["is", "a", "cat"]);
    }

    #[test]
    fn splits_cjk_characters_individually() {
        assert_eq!(tokenize("你好"), vec!["你", "好"]);
    }

    #[test]
    fn groups_thai_base_consonants_with_their_combining_marks() {
        // "สวัสดี" (Thai "hello") -- a mix of base consonants and vowel
        // signs; the combining marks stay attached to their preceding
        // base consonant rather than becoming standalone tokens.
        let tokens = tokenize("สวัสดี");
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|t| !t.is_ascii()));
    }

    #[test]
    fn non_ascii_word_chars_are_kept_whole_not_stemmed() {
        // "café" contains a non-ASCII letter, so the whole word is kept
        // as one token rather than run through the ASCII stemmer.
        assert_eq!(tokenize("café"), vec!["café"]);
    }

    #[test]
    fn score_is_perfect_for_identical_text() {
        let score = score("the cat sat on the mat", "the cat sat on the mat");
        assert_eq!(score.precision, 1.0);
        assert_eq!(score.recall, 1.0);
        assert_eq!(score.fmeasure, 1.0);
    }

    #[test]
    fn score_is_zero_for_disjoint_text() {
        let score = score("apples and oranges", "quantum entanglement physics");
        assert_eq!(score.fmeasure, 0.0);
    }

    #[test]
    fn score_rewards_stemmed_overlap() {
        // "running"/"runs" both stem to "run" -- the unigram overlap
        // counts them as a match even though the surface forms differ.
        let score = score("the dog is running", "the dog runs");
        assert!(score.fmeasure > 0.0);
    }

    #[test]
    fn score_handles_empty_text_without_dividing_by_zero() {
        let score = score("", "");
        assert_eq!(
            score,
            Rouge1Score {
                precision: 0.0,
                recall: 0.0,
                fmeasure: 0.0
            }
        );
    }

    /// End-to-end cross-check against the real pip `rouge_score` package
    /// (its `_UnicodeAwareTokenizer`/`RougeScorer` reassembled locally
    /// from the actual upstream source and run under real nltk 3.10.3,
    /// not reconstructed from memory) over a mix of plain ASCII,
    /// stemming-sensitive, CJK, Thai, and punctuation-heavy text pairs.
    /// `(candidate, reference, expected_precision, expected_recall,
    /// expected_fmeasure)`; `score(target=reference, prediction=candidate)`
    /// matches `_calculate_rouge_1_scores`'s own argument order.
    #[test]
    fn matches_rouge_score_end_to_end() {
        let cases: &[(&str, &str, f64, f64, f64)] = &[
            (
                "The quick brown foxes jumps!",
                "The quick brown fox jumps.",
                1.0,
                1.0,
                1.0,
            ),
            (
                "the cat sat on the mat",
                "the cat sat on the mat",
                1.0,
                1.0,
                1.0,
            ),
            (
                "apples and oranges",
                "quantum entanglement physics",
                0.0,
                0.0,
                0.0,
            ),
            (
                "the dog is running",
                "the dog runs",
                0.75,
                1.0,
                0.8571428571428571,
            ),
            ("", "", 0.0, 0.0, 0.0),
            ("你好世界", "你好", 0.5, 1.0, 0.6666666666666666),
            ("café résumé", "cafe resume", 0.0, 0.0, 0.0),
            (
                "สวัสดีครับ ผมชื่อจอห์น",
                "สวัสดีค่ะ ฉันชื่อแมรี่",
                0.5333333333333333,
                0.6153846153846154,
                0.5714285714285715,
            ),
            (
                "This is a test with 123 numbers and Punctuation!!!",
                "This test has 123 numbers, punctuation.",
                0.5555555555555556,
                0.8333333333333334,
                0.6666666666666667,
            ),
            (
                "Mixed 你好 and English text here",
                "Mixed 你好 and some English words",
                0.7142857142857143,
                0.7142857142857143,
                0.7142857142857143,
            ),
            (
                "The organization organized organizational meetings.",
                "The organizations organize organizational meeting.",
                1.0,
                1.0,
                1.0,
            ),
        ];

        for (candidate, reference, expected_p, expected_r, expected_f) in cases {
            let got = score(reference, candidate);
            let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
            assert!(
                close(got.precision, *expected_p)
                    && close(got.recall, *expected_r)
                    && close(got.fmeasure, *expected_f),
                "candidate={candidate:?} reference={reference:?} got={got:?} expected=({expected_p}, {expected_r}, {expected_f})"
            );
        }
    }
}
