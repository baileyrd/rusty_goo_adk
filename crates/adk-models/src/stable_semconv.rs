//! C0661 (partial): `telemetry._stable_semconv`, ported from
//! `google.adk.telemetry._stable_semconv`.
//!
//! Centralizes the construction of `gen_ai.system.message`/
//! `gen_ai.user.message`/`gen_ai.choice` log bodies — pure builders, no
//! OTel SDK dependency (the source only imports OTel for the
//! type-only `AnyValue` alias, which this port represents as
//! [`rusty_serde::value::Value`]).
//!
//! **Partial**: this row's own manifest description also names the
//! experimental `gen_ai.client.inference.operation.details` log event
//! (`telemetry/_experimental_semconv.py`) — a separate, larger surface
//! (duck-typed MCP/genai object conversion, tool-definition resolution,
//! ~700 lines) that, while also free of any live OTel Span dependency
//! (its own "Section D" setters write into a caller-supplied
//! `MutableMapping`, not a real `Span`), is disproportionate to bundle
//! into this batch. Deferred to its own pass; only the three stable
//! log-body builders are ported here.
//!
//! **`system_instruction`, no `serialize_content` dispatch needed**: the
//! source's `system_message_body` calls `serialize_content` on
//! `llm_request.config.system_instruction` (a `types.ContentUnion`).
//! [`crate::llm_request::GenerateContentConfigStub::system_instruction`]
//! is already narrowed to a plain `Option<String>` in this port (that
//! narrowing predates this file — see `llm_request.rs`'s own module
//! doc), so there's no `Content`/list/raw-value union left to dispatch
//! on; the string is wrapped directly.
//!
//! **`finish_reason`, no `.value`/`str()` dispatch needed**: the source
//! reads `finish_reason.value if hasattr(finish_reason, 'value') else
//! str(finish_reason)` to normalize either a real enum member or an
//! already-stringy value. [`crate::llm_response::LlmResponse::finish_reason`]
//! is already an opaque [`Value`] holding the raw wire string (e.g.
//! `Value::String("STOP")`) in this port — there is no enum to
//! normalize away, so it's forwarded as-is.

use adk_agents::telemetry_context::TelemetryConfig;
use adk_genai::content::Content;
use rusty_serde::value::Value;
use std::collections::BTreeMap;

use crate::llm_request::LlmRequest;
use crate::llm_response::LlmResponse;

/// Stable OTel GenAI semantic-convention event names.
pub const GEN_AI_SYSTEM_MESSAGE_EVENT: &str = "gen_ai.system.message";
pub const GEN_AI_USER_MESSAGE_EVENT: &str = "gen_ai.user.message";
pub const GEN_AI_CHOICE_EVENT: &str = "gen_ai.choice";

/// Standard OTel env var controlling whether prompt/response content is
/// included in log bodies.
pub const OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: &str =
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

pub const USER_CONTENT_ELIDED: &str = "<elided>";

fn content_value(content: Option<&str>, capture_content: bool) -> Value {
    if !capture_content {
        return Value::String(USER_CONTENT_ELIDED.to_string());
    }
    match content {
        Some(text) => Value::String(text.to_string()),
        None => Value::Null,
    }
}

/// `_stable_semconv.system_message_body` — the body for a
/// `gen_ai.system.message` log event.
///
/// `do_not_elide`: when `true`, always include the content regardless of
/// [`TelemetryConfig::should_add_content_to_logs`] — the Web UI exporter
/// sets this since the UI needs the full content.
pub fn system_message_body(
    llm_request: &LlmRequest,
    telemetry_config: &TelemetryConfig,
    do_not_elide: bool,
) -> BTreeMap<String, Value> {
    let system_instruction = llm_request.config.system_instruction.as_deref();
    let mut body = BTreeMap::new();
    body.insert(
        "content".to_string(),
        content_value(
            system_instruction,
            do_not_elide || telemetry_config.should_add_content_to_logs(),
        ),
    );
    body
}

/// `_stable_semconv.user_message_body` — the body for a single
/// `gen_ai.user.message` log event. A caller emitting multiple user
/// messages (e.g. a per-content loop) calls this once per content.
pub fn user_message_body(
    content: Option<&Content>,
    telemetry_config: &TelemetryConfig,
    do_not_elide: bool,
) -> BTreeMap<String, Value> {
    let text = content
        .and_then(|c| c.parts.first())
        .and_then(|p| p.text.as_deref());
    let mut body = BTreeMap::new();
    body.insert(
        "content".to_string(),
        content_value(
            text,
            do_not_elide || telemetry_config.should_add_content_to_logs(),
        ),
    );
    body
}

/// `_stable_semconv.choice_body` — the body for a `gen_ai.choice` log
/// event. ADK always returns a single candidate, so `index` is always
/// `0`. `finish_reason` is included only when present on the response.
pub fn choice_body(
    llm_response: Option<&LlmResponse>,
    telemetry_config: &TelemetryConfig,
    do_not_elide: bool,
) -> BTreeMap<String, Value> {
    let mut body = BTreeMap::new();
    let Some(llm_response) = llm_response else {
        body.insert("content".to_string(), Value::Null);
        body.insert("index".to_string(), Value::Int(0));
        return body;
    };

    let text = llm_response
        .content
        .as_ref()
        .and_then(|c| c.parts.first())
        .and_then(|p| p.text.as_deref());
    body.insert(
        "content".to_string(),
        content_value(
            text,
            do_not_elide || telemetry_config.should_add_content_to_logs(),
        ),
    );
    body.insert("index".to_string(), Value::Int(0));
    if let Some(finish_reason) = &llm_response.finish_reason {
        body.insert("finish_reason".to_string(), finish_reason.clone());
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::Part;

    /// `TelemetryConfig::should_add_content_to_logs` is env/config
    /// resolved; every test below drives that behavior through the
    /// `do_not_elide` parameter directly instead, so a plain default
    /// config is enough and no test needs to touch process-wide env
    /// state.
    fn telemetry_config() -> TelemetryConfig {
        TelemetryConfig::default()
    }

    fn request_with_system_instruction(text: Option<&str>) -> LlmRequest {
        let mut request = LlmRequest::new("gemini-2.5-flash".to_string());
        request.config.system_instruction = text.map(str::to_string);
        request
    }

    #[test]
    fn system_message_body_elides_by_default() {
        // SAFETY (test-only, single-threaded within this process for env
        // mutation): guards against `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
        // leaking in from another test in this same binary, so this
        // assertion doesn't depend on process-wide env ordering.
        std::env::remove_var(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT);
        let request = request_with_system_instruction(Some("be helpful"));
        let config = telemetry_config();
        let body = system_message_body(&request, &config, false);
        assert_eq!(
            body.get("content"),
            Some(&Value::String(USER_CONTENT_ELIDED.to_string()))
        );
    }

    #[test]
    fn system_message_body_includes_content_when_do_not_elide_is_set() {
        let request = request_with_system_instruction(Some("be helpful"));
        let config = telemetry_config();
        let body = system_message_body(&request, &config, true);
        assert_eq!(
            body.get("content"),
            Some(&Value::String("be helpful".to_string()))
        );
    }

    #[test]
    fn system_message_body_is_null_without_a_system_instruction() {
        let request = request_with_system_instruction(None);
        let config = telemetry_config();
        let body = system_message_body(&request, &config, true);
        assert_eq!(body.get("content"), Some(&Value::Null));
    }

    #[test]
    fn user_message_body_wraps_the_first_text_part() {
        let content = Content::new("user", vec![Part::text("hello")]);
        let config = telemetry_config();
        let body = user_message_body(Some(&content), &config, true);
        assert_eq!(
            body.get("content"),
            Some(&Value::String("hello".to_string()))
        );
    }

    #[test]
    fn choice_body_is_null_content_index_zero_without_a_response() {
        let config = telemetry_config();
        let body = choice_body(None, &config, true);
        assert_eq!(body.get("content"), Some(&Value::Null));
        assert_eq!(body.get("index"), Some(&Value::Int(0)));
        assert!(!body.contains_key("finish_reason"));
    }

    #[test]
    fn choice_body_includes_finish_reason_when_present() {
        let response = LlmResponse {
            content: Some(Content::new("model", vec![Part::text("done")])),
            finish_reason: Some(Value::String("STOP".to_string())),
            ..Default::default()
        };
        let config = telemetry_config();
        let body = choice_body(Some(&response), &config, true);
        assert_eq!(
            body.get("content"),
            Some(&Value::String("done".to_string()))
        );
        assert_eq!(
            body.get("finish_reason"),
            Some(&Value::String("STOP".to_string()))
        );
    }

    #[test]
    fn choice_body_elides_content_by_default() {
        let response = LlmResponse {
            content: Some(Content::new("model", vec![Part::text("secret")])),
            ..Default::default()
        };
        let config = telemetry_config();
        let body = choice_body(Some(&response), &config, false);
        assert_eq!(
            body.get("content"),
            Some(&Value::String(USER_CONTENT_ELIDED.to_string()))
        );
    }
}
