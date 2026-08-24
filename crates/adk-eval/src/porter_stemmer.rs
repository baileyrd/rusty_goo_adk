//! A from-scratch port of `nltk.stem.porter.PorterStemmer` (`NLTK_EXTENSIONS`
//! mode — the only mode `rouge_score`'s `DefaultTokenizer` ever constructs,
//! see `tokenizers.DefaultTokenizer.__init__`: `porter.PorterStemmer()`,
//! no `mode` argument), which is itself part of C0590's dependency chain
//! (`google.adk.dependencies.rouge_scorer` re-exports the pip `rouge_score`
//! package, whose `DefaultTokenizer` stems ASCII tokens through this exact
//! algorithm before scoring).
//!
//! Ported from Porter, M. "An algorithm for suffix stripping." Program 14.3
//! (1980): 130-137, with NLTK's own documented extensions (the irregular-
//! forms pool, the `ies`/`ied` length-4 special cases, the relaxed step 1c
//! `y`->`i` condition, the `alli`/`fulli`/`logi` step 2 additions) — every
//! branch here mirrors nltk's `porter.py` line for line; `MARTIN_EXTENSIONS`/
//! `ORIGINAL_ALGORITHM` mode branches are omitted since no caller in this
//! dependency chain ever selects them.
//!
//! **Adaptation**: operates on `&str` assumed ASCII (`debug_assert!`ed at
//! the entry point) — the sole caller, [`crate::rouge`]'s
//! `UnicodeAwareTokenizer`, only ever stems tokens it has already confirmed
//! are ASCII (`word.isascii()` in the source), so byte indexing is safe and
//! avoids the char/byte-boundary bookkeeping a fully Unicode-generic port
//! would need for no behavioral benefit.

fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn is_consonant(word: &[u8], index: usize) -> bool {
    if is_vowel(word[index]) {
        return false;
    }
    if word[index] == b'y' {
        let mut negate = false;
        let mut i = index;
        while i > 0 && word[i] == b'y' {
            negate = !negate;
            i -= 1;
        }
        return is_vowel(word[i]) == negate;
    }
    true
}

fn consonant_flags(word: &[u8]) -> Vec<bool> {
    let mut flags: Vec<bool> = Vec::with_capacity(word.len());
    for (i, &b) in word.iter().enumerate() {
        let flag = if is_vowel(b) {
            false
        } else if b == b'y' {
            if i == 0 {
                true
            } else {
                !flags[i - 1]
            }
        } else {
            true
        };
        flags.push(flag);
    }
    flags
}

/// The "measure" `m` of a stem: the count of non-overlapping `VC`
/// occurrences in its consonant/vowel classification.
fn measure(stem: &str) -> usize {
    let flags = consonant_flags(stem.as_bytes());
    let mut count = 0;
    let mut i = 0;
    while i + 1 < flags.len() {
        if !flags[i] && flags[i + 1] {
            count += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    count
}

fn has_positive_measure(stem: &str) -> bool {
    measure(stem) > 0
}

fn contains_vowel(stem: &str) -> bool {
    consonant_flags(stem.as_bytes()).iter().any(|&c| !c)
}

fn ends_double_consonant(word: &str) -> bool {
    let bytes = word.as_bytes();
    let len = bytes.len();
    len >= 2 && bytes[len - 1] == bytes[len - 2] && is_consonant(bytes, len - 1)
}

fn ends_cvc(word: &str) -> bool {
    let bytes = word.as_bytes();
    let len = bytes.len();
    (len >= 3
        && is_consonant(bytes, len - 3)
        && !is_consonant(bytes, len - 2)
        && is_consonant(bytes, len - 1)
        && !matches!(bytes[len - 1], b'w' | b'x' | b'y'))
        || (len == 2 && !is_consonant(bytes, 0) && is_consonant(bytes, 1))
}

fn replace_suffix(word: &str, suffix: &str, replacement: &str) -> String {
    debug_assert!(word.ends_with(suffix));
    if suffix.is_empty() {
        format!("{word}{replacement}")
    } else {
        format!("{}{}", &word[..word.len() - suffix.len()], replacement)
    }
}

fn step1a(word: &str) -> String {
    if word.ends_with("ies") && word.len() == 4 {
        return replace_suffix(word, "ies", "ie");
    }
    if word.ends_with("sses") {
        return replace_suffix(word, "sses", "ss");
    }
    if word.ends_with("ies") {
        return replace_suffix(word, "ies", "i");
    }
    if word.ends_with("ss") {
        return replace_suffix(word, "ss", "ss");
    }
    if word.ends_with('s') {
        return replace_suffix(word, "s", "");
    }
    word.to_string()
}

fn step1b(word: &str) -> String {
    if word.ends_with("ied") {
        return if word.len() == 4 {
            replace_suffix(word, "ied", "ie")
        } else {
            replace_suffix(word, "ied", "i")
        };
    }

    if word.ends_with("eed") {
        let stem = replace_suffix(word, "eed", "");
        return if measure(&stem) > 0 {
            format!("{stem}ee")
        } else {
            word.to_string()
        };
    }

    let mut intermediate_stem = None;
    for suffix in ["ed", "ing"] {
        if word.ends_with(suffix) {
            let candidate = replace_suffix(word, suffix, "");
            if contains_vowel(&candidate) {
                intermediate_stem = Some(candidate);
                break;
            }
        }
    }
    let intermediate_stem = match intermediate_stem {
        Some(stem) => stem,
        None => return word.to_string(),
    };

    if intermediate_stem.ends_with("at") {
        return replace_suffix(&intermediate_stem, "at", "ate");
    }
    if intermediate_stem.ends_with("bl") {
        return replace_suffix(&intermediate_stem, "bl", "ble");
    }
    if intermediate_stem.ends_with("iz") {
        return replace_suffix(&intermediate_stem, "iz", "ize");
    }
    if ends_double_consonant(&intermediate_stem) {
        let last = *intermediate_stem.as_bytes().last().unwrap();
        return if !matches!(last, b'l' | b's' | b'z') {
            intermediate_stem[..intermediate_stem.len() - 1].to_string()
        } else {
            intermediate_stem
        };
    }
    if measure(&intermediate_stem) == 1 && ends_cvc(&intermediate_stem) {
        return format!("{intermediate_stem}e");
    }
    intermediate_stem
}

fn step1c(word: &str) -> String {
    if word.ends_with('y') {
        let stem = replace_suffix(word, "y", "");
        if stem.len() > 1 && is_consonant(stem.as_bytes(), stem.len() - 1) {
            return format!("{stem}i");
        }
    }
    word.to_string()
}

fn step2(word: &str) -> String {
    if word.ends_with("alli") && has_positive_measure(&replace_suffix(word, "alli", "")) {
        return step2(&replace_suffix(word, "alli", "al"));
    }

    macro_rules! try_rule {
        ($suffix:expr, $repl:expr) => {
            if word.ends_with($suffix) {
                let stem = replace_suffix(word, $suffix, "");
                return if has_positive_measure(&stem) {
                    format!("{stem}{}", $repl)
                } else {
                    word.to_string()
                };
            }
        };
    }

    try_rule!("ational", "ate");
    try_rule!("tional", "tion");
    try_rule!("enci", "ence");
    try_rule!("anci", "ance");
    try_rule!("izer", "ize");
    try_rule!("bli", "ble");
    try_rule!("alli", "al");
    try_rule!("entli", "ent");
    try_rule!("eli", "e");
    try_rule!("ousli", "ous");
    try_rule!("ization", "ize");
    try_rule!("ation", "ate");
    try_rule!("ator", "ate");
    try_rule!("alism", "al");
    try_rule!("iveness", "ive");
    try_rule!("fulness", "ful");
    try_rule!("ousness", "ous");
    try_rule!("aliti", "al");
    try_rule!("iviti", "ive");
    try_rule!("biliti", "ble");
    try_rule!("fulli", "ful");

    if word.ends_with("logi") {
        let stem = replace_suffix(word, "logi", "");
        let condition_stem = &word[..word.len() - 3];
        return if has_positive_measure(condition_stem) {
            format!("{stem}log")
        } else {
            word.to_string()
        };
    }

    word.to_string()
}

fn step3(word: &str) -> String {
    macro_rules! try_rule {
        ($suffix:expr, $repl:expr) => {
            if word.ends_with($suffix) {
                let stem = replace_suffix(word, $suffix, "");
                return if has_positive_measure(&stem) {
                    format!("{stem}{}", $repl)
                } else {
                    word.to_string()
                };
            }
        };
    }
    try_rule!("icate", "ic");
    try_rule!("ative", "");
    try_rule!("alize", "al");
    try_rule!("iciti", "ic");
    try_rule!("ical", "ic");
    try_rule!("ful", "");
    try_rule!("ness", "");
    word.to_string()
}

fn step4(word: &str) -> String {
    macro_rules! try_rule {
        ($suffix:expr, $repl:expr) => {
            if word.ends_with($suffix) {
                let stem = replace_suffix(word, $suffix, "");
                return if measure(&stem) > 1 {
                    format!("{stem}{}", $repl)
                } else {
                    word.to_string()
                };
            }
        };
    }
    try_rule!("al", "");
    try_rule!("ance", "");
    try_rule!("ence", "");
    try_rule!("er", "");
    try_rule!("ic", "");
    try_rule!("able", "");
    try_rule!("ible", "");
    try_rule!("ant", "");
    try_rule!("ement", "");
    try_rule!("ment", "");
    try_rule!("ent", "");
    if word.ends_with("ion") {
        let stem = replace_suffix(word, "ion", "");
        let ok = measure(&stem) > 1 && matches!(stem.as_bytes().last(), Some(b's' | b't'));
        return if ok { stem } else { word.to_string() };
    }
    try_rule!("ou", "");
    try_rule!("ism", "");
    try_rule!("ate", "");
    try_rule!("iti", "");
    try_rule!("ous", "");
    try_rule!("ive", "");
    try_rule!("ize", "");
    word.to_string()
}

fn step5a(word: &str) -> String {
    if word.ends_with('e') {
        let stem = replace_suffix(word, "e", "");
        if measure(&stem) > 1 {
            return stem;
        }
        if measure(&stem) == 1 && !ends_cvc(&stem) {
            return stem;
        }
    }
    word.to_string()
}

fn step5b(word: &str) -> String {
    if word.ends_with("ll") {
        let candidate = &word[..word.len() - 1];
        if measure(candidate) > 1 {
            return candidate.to_string();
        }
    }
    word.to_string()
}

fn irregular_form(word: &str) -> Option<&'static str> {
    match word {
        "sky" | "skies" => Some("sky"),
        "dying" => Some("die"),
        "lying" => Some("lie"),
        "tying" => Some("tie"),
        "news" => Some("news"),
        "innings" | "inning" => Some("inning"),
        "outings" | "outing" => Some("outing"),
        "cannings" | "canning" => Some("canning"),
        "howe" => Some("howe"),
        "proceed" => Some("proceed"),
        "exceed" => Some("exceed"),
        "succeed" => Some("succeed"),
        _ => None,
    }
}

/// `PorterStemmer(mode=NLTK_EXTENSIONS).stem(word, to_lowercase=True)` —
/// the only construction rouge_score's `DefaultTokenizer` ever uses.
pub fn stem(word: &str) -> String {
    let lower = word.to_lowercase();
    debug_assert!(lower.is_ascii(), "porter_stemmer::stem expects ASCII input");

    if let Some(irregular) = irregular_form(&lower) {
        return irregular.to_string();
    }
    if lower.len() <= 2 {
        return lower;
    }

    let s = step1a(&lower);
    let s = step1b(&s);
    let s = step1c(&s);
    let s = step2(&s);
    let s = step3(&s);
    let s = step4(&s);
    let s = step5a(&s);
    step5b(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 409 `(word, expected_stem)` pairs generated by actually running
    /// `nltk.stem.porter.PorterStemmer().stem()` (nltk 3.10.3, the same
    /// package `rouge_score`'s `DefaultTokenizer` depends on) over a word
    /// list covering every step's suffix rules — the paper's own worked
    /// examples, the NLTK-only irregular-forms pool and length-4 `ies`/
    /// `ied` cases, and several hundred derived/inflected common-English
    /// forms chosen to exercise multi-step chains (e.g. `"agreed"` ends
    /// step 1b as `"agree"` but step 5a then strips the trailing `e` to
    /// `"agre"` — a case a single-step-at-a-time reading of the paper's
    /// own examples would get wrong). This is the primary correctness
    /// oracle for this port, not a hand-derived table.
    #[test]
    fn matches_nltk_across_the_fixture() {
        let fixture = include_str!("porter_stemmer_fixture.txt");
        let mut checked = 0;
        for line in fixture.lines() {
            let mut parts = line.split_whitespace();
            let word = parts.next().unwrap();
            let expected = parts.next().unwrap();
            assert_eq!(stem(word), expected, "stemming {word:?}");
            checked += 1;
        }
        assert!(checked > 400, "fixture should cover several hundred words");
    }

    #[test]
    fn preserves_short_words_unchanged() {
        assert_eq!(stem("a"), "a");
        assert_eq!(stem("to"), "to");
    }

    #[test]
    fn resolves_irregular_forms() {
        assert_eq!(stem("skies"), "sky");
        assert_eq!(stem("dying"), "die");
        assert_eq!(stem("lying"), "lie");
        assert_eq!(stem("tying"), "tie");
        assert_eq!(stem("proceed"), "proceed");
    }

    #[test]
    fn applies_the_nltk_only_ies_length_four_case() {
        assert_eq!(stem("ties"), "tie");
    }

    #[test]
    fn lowercases_before_stemming() {
        assert_eq!(stem("CARESSES"), "caress");
    }
}
