//! Capability C0944 (of C0945's larger, still-`REQUIRED` `_schema_utils.py`):
//! `_strip_json_code_fence`, ported from `google.adk.utils._schema_utils`.
//!
//! **Scope**: only this one small, self-contained, regex-free function is
//! ported here. The rest of `_schema_utils.py`
//! (`is_basemodel_schema`/`is_list_of_basemodel`/`get_list_inner_type`/
//! `schema_to_json_schema`/`validate_schema`/`validate_node_data`) is
//! fundamentally `pydantic.TypeAdapter`/`get_origin`/`get_args`-driven —
//! runtime introspection of an arbitrary Python type object (a `BaseModel`
//! class, a `list[SomeModel]` generic alias, a raw `dict`, ...) to
//! validate/convert data against it generically. Rust has no runtime type
//! object to introspect this way, and this workspace has no schema-driven
//! generic-validation layer to port it onto (no `TypeAdapter` equivalent).
//! That's tracked as its own row, C0945, left `REQUIRED` rather than
//! attempted here.
//!
//! **Adaptation**: the source uses `re.fullmatch(r"```\w*\s*(.*?)\s*```",
//! stripped, re.DOTALL)`. Rather than adding `regex` as a new dependency
//! usage site of this crate (an easy, low-risk option, but unnecessary
//! here), this port hand-rolls the equivalent match: since the pattern is
//! anchored at both ends (`re.fullmatch`) and `.` matches everything
//! (`DOTALL`), the opening/closing `` ``` `` must be the literal first and
//! last 3 characters of the trimmed string — there's no backtracking
//! ambiguity to a regex engine actually needed. Python's `\w` matches
//! Unicode word characters by default (not just ASCII); this port
//! approximates that with `char::is_alphanumeric() || c == '_'` rather
//! than restricting to ASCII — closer to the source than an ASCII-only
//! match, though not proven byte-for-byte identical across every Unicode
//! category `\w` covers.

/// `_schema_utils._strip_json_code_fence` — removes a markdown code fence
/// wrapping the entire JSON payload, if present. A model asked for
/// structured output occasionally wraps it in a `` ```json ... ``` ``
/// fence; well-formed JSON never starts with a fence, so this is a no-op
/// on valid input.
pub fn strip_json_code_fence(json_text: &str) -> &str {
    let stripped = json_text.trim();
    if stripped.len() < 6 || !stripped.starts_with("```") || !stripped.ends_with("```") {
        return json_text;
    }

    let inner = &stripped[3..stripped.len() - 3];
    let after_lang = inner.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_');
    after_lang.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_a_fence_with_a_language_tag() {
        assert_eq!(
            strip_json_code_fence("```json\n{\"a\":1}\n```"),
            "{\"a\":1}"
        );
    }

    #[test]
    fn strips_a_bare_fence_with_no_language_tag() {
        assert_eq!(strip_json_code_fence("```\n{\"a\":1}\n```"), "{\"a\":1}");
    }

    #[test]
    fn leaves_well_formed_unfenced_json_unchanged() {
        assert_eq!(strip_json_code_fence("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn leaves_a_partial_or_malformed_fence_unchanged() {
        assert_eq!(
            strip_json_code_fence("```json\n{\"a\":1}"),
            "```json\n{\"a\":1}"
        );
    }

    #[test]
    fn preserves_a_fence_nested_inside_the_payload() {
        let input = "```json\n{\"code\":\"```inner```\"}\n```";
        assert_eq!(strip_json_code_fence(input), "{\"code\":\"```inner```\"}");
    }
}
