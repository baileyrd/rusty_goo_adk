//! Capability C0134: `_build_request_log`/`_build_response_log`, ported
//! from `google.adk.models.google_llm`.
//!
//! **Adaptation, disclosed**: the source's request log includes a
//! "Functions" section (`_build_function_declaration_log` over
//! `req.config.tools[i].function_declarations`), sourced by locating the
//! first `Tool` in `config.tools` that carries `function_declarations`.
//! `LlmRequest.config.tools` stays an opaque [`rusty_serde::value::Value`]
//! placeholder in this port (C0116, Phase 8's `BaseTool` — see
//! `llm_request.rs`'s module doc), so there is no typed `Tool`/
//! `FunctionDeclaration` to walk. The "Functions" section is always empty
//! here, and the config log's own `tools` key is always fully excluded
//! (redacted) rather than partially — the same effect the source's own
//! fallback branch produces whenever it *can't* locate a function-
//! declaration-bearing tool (`tools_exclusion = True`), just taken
//! unconditionally instead of only as a fallback.
//!
//! **Adaptation, disclosed**: [`HttpOptionsStub`] only models `headers`/
//! `api_version` (see `llm_request.rs`'s module doc) — of the source's
//! full credential-bearing exclusion set (`httpx_client`/
//! `httpx_async_client`/`aiohttp_client`/`headers`/`extra_body`/
//! `client_args`/`async_client_args`), only `headers` exists to redact
//! here. It's still redacted the same way: dropped from the config log
//! entirely, never partially shown.
//!
//! **Omitted, deliberately**: no logging framework has been adopted by
//! this workspace yet (see `gemini_llm_connection.rs`'s module doc for
//! the same caveat) — these are pure string-building functions, callable
//! from anywhere once a logging decision is made. [`Gemini::generate_content`]
//! calls them under a lightweight `ADK_DEBUG_LOGGING` env-var gate (this
//! port's stand-in for the source's `logger.isEnabledFor(logging.DEBUG)`)
//! and prints via `eprintln!`, matching this codebase's existing ad hoc
//! `eprintln!`-based logging convention elsewhere.

use adk_genai::content::{Content, MediaBlobStub, Part};

use crate::generate_content_response::GenerateContentResponse;
use crate::llm_request::LlmRequest;

const NEW_LINE: &str = "\n";

/// Env var gating debug-level request/response logging — this port's
/// stand-in for `logger.isEnabledFor(logging.DEBUG)` (see the module doc).
pub fn debug_logging_enabled() -> bool {
    crate::capabilities::is_env_enabled(std::env::var("ADK_DEBUG_LOGGING").ok().as_deref(), "0")
}

/// Redacts `inline_data.data` (the actual binary payload) from a cloned
/// `Content`'s parts before logging — matching the source's
/// `_EXCLUDED_PART_FIELD = {'inline_data': {'data'}}`. Other fields
/// (`mime_type`, and anything else flattened into
/// [`MediaBlobStub::rest`]) are kept.
fn redact_content_for_log(content: &Content) -> Content {
    let mut redacted = content.clone();
    for part in &mut redacted.parts {
        redact_part_inline_data(part);
    }
    redacted
}

fn redact_part_inline_data(part: &mut Part) {
    if let Some(blob) = &mut part.inline_data {
        redact_blob_data(blob);
    }
    if let Some(blob) = &mut part.file_data {
        redact_blob_data(blob);
    }
}

fn redact_blob_data(blob: &mut MediaBlobStub) {
    if let Some(rest) = &mut blob.rest {
        rest.remove("data");
    }
}

/// C0134: `_build_request_log` — a debug-level structured request log
/// with credential-bearing fields redacted. See the module doc for what's
/// deliberately narrower than the source (no `tools`/function-declaration
/// modeling yet).
pub fn build_request_log(request: &LlmRequest) -> String {
    let contents_logs: Vec<String> = request
        .contents
        .iter()
        .map(|content| {
            let redacted = redact_content_for_log(content);
            rusty_serde::json::to_string(&redacted).unwrap_or_else(|_| "<error>".to_string())
        })
        .collect();

    let config_log = build_config_log(request);

    format!(
        "\nLLM Request:\n\
         -----------------------------------------------------------\n\
         System Instruction:\n\
         {:?}\n\
         -----------------------------------------------------------\n\
         Config:\n\
         {config_log}\n\
         -----------------------------------------------------------\n\
         Contents:\n\
         {}\n\
         -----------------------------------------------------------\n\
         Functions:\n\
         {}\n\
         -----------------------------------------------------------\n",
        request.config.system_instruction,
        contents_logs.join(NEW_LINE),
        // See the module doc: `tools` isn't modeled as typed
        // `FunctionDeclaration`s yet, so this section is always empty.
        "",
    )
}

/// The config log body: every currently-modeled `GenerateContentConfigStub`
/// field except `system_instruction` (already shown separately) and
/// `tools` (always redacted — see the module doc), with `http_options`'s
/// `headers` redacted the same way the source redacts its own
/// credential-bearing `http_options` sub-fields.
fn build_config_log(request: &LlmRequest) -> String {
    let config = &request.config;
    let mut fields: Vec<String> = Vec::new();

    if let Some(schema) = &config.response_schema {
        fields.push(format!(
            "response_schema={}",
            rusty_serde::json::to_string(schema).unwrap_or_else(|_| "<error>".to_string())
        ));
    }
    if let Some(mime_type) = &config.response_mime_type {
        fields.push(format!("response_mime_type={mime_type:?}"));
    }
    if config.tools.is_some() {
        fields.push("tools=<redacted: not modeled as typed function declarations>".to_string());
    }
    if config.thinking_config.is_some() {
        fields.push("thinking_config=<present>".to_string());
    }
    if config.safety_settings.is_some() {
        fields.push("safety_settings=<present>".to_string());
    }
    if config.tool_config.is_some() {
        fields.push("tool_config=<present>".to_string());
    }
    if let Some(cached_content) = &config.cached_content {
        fields.push(format!("cached_content={cached_content:?}"));
    }
    if let Some(http_options) = &config.http_options {
        let api_version = http_options
            .api_version
            .clone()
            .unwrap_or_else(|| "<unset>".to_string());
        let header_count = http_options.headers.as_ref().map(|h| h.len()).unwrap_or(0);
        fields.push(format!(
            "http_options={{api_version: {api_version:?}, headers: <redacted, {header_count} \
             entries>}}"
        ));
    }

    if fields.is_empty() {
        "{}".to_string()
    } else {
        format!("{{{}}}", fields.join(", "))
    }
}

/// C0134: `_build_response_log` — mirrors `resp.text`'s own
/// visible-text-only extraction (first candidate, thought parts excluded)
/// without triggering the source's warn-on-non-text-parts guard, since
/// this port has no such warning to trigger in the first place.
pub fn build_response_log(response: &GenerateContentResponse) -> String {
    let first_candidate = response.candidates.as_ref().and_then(|c| c.first());

    let text: String = first_candidate
        .and_then(|c| c.content.as_ref())
        .map(|content| {
            content
                .parts
                .iter()
                .filter(|part| part.thought != Some(true))
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let function_calls_text: Vec<String> = first_candidate
        .and_then(|c| c.content.as_ref())
        .map(|content| {
            content
                .get_function_calls()
                .iter()
                .map(|call| {
                    format!(
                        "name: {:?}, args: {:?}",
                        call.name.as_deref().unwrap_or(""),
                        call.args
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    let raw_response =
        rusty_serde::json::to_string(response).unwrap_or_else(|_| "<error>".to_string());

    format!(
        "\nLLM Response:\n\
         -----------------------------------------------------------\n\
         Text:\n\
         {text}\n\
         -----------------------------------------------------------\n\
         Function calls:\n\
         {}\n\
         -----------------------------------------------------------\n\
         Raw response:\n\
         {raw_response}\n\
         -----------------------------------------------------------\n",
        function_calls_text.join(NEW_LINE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_content_response::Candidate;
    use adk_genai::content::{Content, FunctionCall, Part};

    #[test]
    fn debug_logging_is_disabled_by_default() {
        std::env::remove_var("ADK_DEBUG_LOGGING");
        assert!(!debug_logging_enabled());
    }

    #[test]
    fn debug_logging_env_var_enables_it() {
        std::env::set_var("ADK_DEBUG_LOGGING", "true");
        assert!(debug_logging_enabled());
        std::env::remove_var("ADK_DEBUG_LOGGING");
    }

    #[test]
    fn build_request_log_shows_system_instruction_and_contents() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.config.system_instruction = Some("be helpful".to_string());
        request.contents.push(Content::user_text("hello"));

        let log = build_request_log(&request);
        assert!(log.contains("be helpful"));
        assert!(log.contains("hello"));
        assert!(log.contains("LLM Request:"));
    }

    #[test]
    fn build_request_log_redacts_inline_data_bytes_but_keeps_mime_type() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let part = Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("image/png".to_string()),
                rest: Some(rusty_serde::value::Value::Map(vec![(
                    "data".to_string(),
                    rusty_serde::value::Value::String("<huge base64 blob>".to_string()),
                )])),
            }),
            ..Default::default()
        };
        request.contents.push(Content::new("user", vec![part]));

        let log = build_request_log(&request);
        assert!(log.contains("image/png"));
        assert!(!log.contains("huge base64 blob"));
    }

    #[test]
    fn build_request_log_redacts_http_options_headers_entirely() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let mut headers = std::collections::BTreeMap::new();
        headers.insert(
            "authorization".to_string(),
            "Bearer super-secret-token".to_string(),
        );
        request.config.http_options = Some(crate::llm_request::HttpOptionsStub {
            headers: Some(headers),
            api_version: Some("v1beta".to_string()),
        });

        let log = build_request_log(&request);
        assert!(!log.contains("super-secret-token"));
        assert!(log.contains("redacted"));
        assert!(log.contains("v1beta"));
    }

    #[test]
    fn build_request_log_redacts_tools_entirely_since_they_are_not_typed_yet() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.config.tools = Some(rusty_serde::value::Value::String(
            "some-secret-tool-schema".to_string(),
        ));

        let log = build_request_log(&request);
        assert!(!log.contains("some-secret-tool-schema"));
        assert!(log.contains("redacted"));
    }

    #[test]
    fn build_response_log_extracts_visible_text_and_skips_thought_parts() {
        let mut response = GenerateContentResponse::default();
        let visible = Part {
            text: Some("hello there".to_string()),
            ..Default::default()
        };
        let thought = Part {
            text: Some("internal reasoning".to_string()),
            thought: Some(true),
            ..Default::default()
        };
        response.candidates = Some(vec![Candidate {
            content: Some(Content::new("model", vec![thought, visible])),
            ..Default::default()
        }]);

        let log = build_response_log(&response);
        // The "Text:" section excludes thought parts — matching the
        // source's `resp.text`-equivalent extraction — but the trailing
        // "Raw response:" JSON dump is intentionally unredacted (same as
        // the source's own `model_dump_json`), so it still carries the
        // thought text; only the extracted-text section needs to omit it.
        let text_section = log.split("Function calls:").next().unwrap();
        assert!(text_section.contains("hello there"));
        assert!(!text_section.contains("internal reasoning"));
    }

    #[test]
    fn build_response_log_lists_function_calls() {
        let mut response = GenerateContentResponse::default();
        let part = Part {
            function_call: Some(FunctionCall {
                name: Some("get_weather".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        response.candidates = Some(vec![Candidate {
            content: Some(Content::new("model", vec![part])),
            ..Default::default()
        }]);

        let log = build_response_log(&response);
        assert!(log.contains("get_weather"));
    }

    #[test]
    fn build_response_log_includes_the_raw_json_dump() {
        let response = GenerateContentResponse {
            model_version: Some("gemini-2.5-flash-001".to_string()),
            ..Default::default()
        };

        let log = build_response_log(&response);
        assert!(log.contains("gemini-2.5-flash-001"));
        assert!(log.contains("Raw response:"));
    }

    #[test]
    fn build_response_log_handles_an_empty_response() {
        let response = GenerateContentResponse::default();
        let log = build_response_log(&response);
        assert!(log.contains("LLM Response:"));
        assert!(log.contains("Text:"));
    }
}
