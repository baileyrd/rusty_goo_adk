//! Capability C0938: `can_use_output_schema_with_tools`, ported from
//! `google.adk.utils.output_schema_utils`.
//!
//! **Adaptation**: the source accepts `Union[str, BaseLlm]` and special-
//! cases a `LiteLlm` instance (always `True`, since LiteLLM's own
//! per-provider tools/`response_format` compatibility handling is "strictly
//! more reliable than the `SetModelResponseTool` prompt-based workaround").
//! `models.lite_llm.LiteLlm` isn't ported in this workspace (see
//! `adk-models::registry`'s and `adk-models::ollama`'s module docs — only
//! disclosed via error-message text, no real type) — so this port narrows
//! the parameter to a bare model-name `&str`, the same representation
//! `capabilities::gemini_output_schema_and_tools` already uses, and the
//! `LiteLlm`-always-true branch is dropped rather than silently kept as
//! dead code. Disclosed narrowing, not a behavior match for a caller that
//! would pass a `LiteLlm` model.
//!
//! **`@deprecated` → `#[deprecated]`**: unlike most of this port's
//! `warnings.warn`-based Python deprecation notices (which have no direct
//! Rust equivalent and become an `eprintln!`, e.g.
//! `capabilities::legacy_output_schema_and_tools`), `typing_extensions`'s
//! `@deprecated` decorator is a *static*, type-checker-visible annotation —
//! Rust's `#[deprecated]` attribute is a genuinely close analog (a
//! compile-time warning at every call site), so it's used here instead of
//! a runtime print.

use crate::capabilities::gemini_output_schema_and_tools;

/// `output_schema_utils.can_use_output_schema_with_tools` — returns `true`
/// if output schema with tools is supported for `model_name`. See the
/// module doc for the `LiteLlm`-branch narrowing.
#[deprecated(
    note = "Use BaseLlm::capabilities().output_schema_and_tools instead. This \
            function does not honor capabilities declared by a BaseLlm subclass."
)]
pub fn can_use_output_schema_with_tools(model_name: &str) -> bool {
    gemini_output_schema_and_tools(model_name)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn delegates_to_gemini_output_schema_and_tools() {
        // Takes the shared ENV_LOCK: `gemini_output_schema_and_tools` reads
        // GOOGLE_GENAI_USE_ENTERPRISE/GOOGLE_GENAI_USE_VERTEXAI, and other
        // tests in this crate mutate those under the same lock.
        let _guard = crate::capabilities::ENV_LOCK.lock().unwrap();
        assert_eq!(
            can_use_output_schema_with_tools("gemini-2.5-flash"),
            gemini_output_schema_and_tools("gemini-2.5-flash")
        );
    }
}
