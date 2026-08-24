//! C0635: misc evaluation constants, ported from
//! `google.adk.evaluation.constants` and `evaluation_constants`.

/// `constants.MISSING_EVAL_DEPENDENCIES_MESSAGE`.
pub const MISSING_EVAL_DEPENDENCIES_MESSAGE: &str =
    "Eval module is not installed, please install via `pip install \"google-adk[eval]\"`.";

/// `constants.DEFAULT_LIVE_TIMEOUT_SECONDS`.
pub const DEFAULT_LIVE_TIMEOUT_SECONDS: u64 = 300;

/// `evaluation_constants.EvalConstants` — legacy eval-file dict-key
/// names, ported as an enum-free set of `&str` constants (the source
/// itself is a plain namespace class of string literals, not a real
/// enum).
pub mod eval_constants {
    pub const QUERY: &str = "query";
    pub const EXPECTED_TOOL_USE: &str = "expected_tool_use";
    pub const RESPONSE: &str = "response";
    pub const REFERENCE: &str = "reference";
    pub const TOOL_NAME: &str = "tool_name";
    pub const TOOL_INPUT: &str = "tool_input";
    pub const MOCK_TOOL_OUTPUT: &str = "mock_tool_output";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_constants_match_the_source_dict_key_names() {
        assert_eq!(eval_constants::QUERY, "query");
        assert_eq!(eval_constants::EXPECTED_TOOL_USE, "expected_tool_use");
        assert_eq!(eval_constants::RESPONSE, "response");
        assert_eq!(eval_constants::REFERENCE, "reference");
        assert_eq!(eval_constants::TOOL_NAME, "tool_name");
        assert_eq!(eval_constants::TOOL_INPUT, "tool_input");
        assert_eq!(eval_constants::MOCK_TOOL_OUTPUT, "mock_tool_output");
    }

    #[test]
    fn default_live_timeout_matches_the_source() {
        assert_eq!(DEFAULT_LIVE_TIMEOUT_SECONDS, 300);
    }
}
