//! The wire shape of a Gemini REST API `generateContent` request body —
//! what `gemini.rs`'s real `generate_content_async` (C0125) builds and
//! sends. In the source, this JSON is produced entirely inside the
//! third-party `google-genai` SDK's own pydantic serialization of
//! `llm_request.contents`/`llm_request.config`; since that SDK isn't
//! ported, this module is the Rust-native equivalent: a real (typed, not
//! opaque) request body covering exactly the `LlmRequest` fields this
//! migration currently models (`contents`, `config.system_instruction`,
//! `config.response_mime_type`, `config.response_schema`). `config.tools`
//! isn't modeled yet (deferred with `append_tools`, C0116 — Phase 8's
//! `BaseTool`), so no `tools` key is ever sent; a real request with tools
//! configured is out of reach until that lands.

use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;
use rusty_serde::Serialize;

use crate::llm_request::LlmRequest;

/// `generationConfig` in the Gemini REST API body.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct GenerationConfigBody {
    #[rusty_serde(default)]
    pub response_mime_type: Option<String>,
    #[rusty_serde(default)]
    pub response_schema: Option<Value>,
}

/// The top-level Gemini REST API `generateContent` request body.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct GenerateContentRequestBody {
    pub contents: Vec<Content>,
    #[rusty_serde(default)]
    pub system_instruction: Option<Content>,
    #[rusty_serde(default)]
    pub generation_config: Option<GenerationConfigBody>,
}

/// Builds the request body the Gemini REST API expects from an
/// `LlmRequest`'s currently-modeled fields — see the module doc for what's
/// not yet included.
pub fn build_request_body(request: &LlmRequest) -> GenerateContentRequestBody {
    let system_instruction = request
        .config
        .system_instruction
        .as_ref()
        .map(|text| Content {
            role: None,
            parts: vec![Part::text(text.clone())],
        });

    let generation_config = if request.config.response_mime_type.is_some()
        || request.config.response_schema.is_some()
    {
        Some(GenerationConfigBody {
            response_mime_type: request.config.response_mime_type.clone(),
            response_schema: request.config.response_schema.clone(),
        })
    } else {
        None
    };

    GenerateContentRequestBody {
        contents: request.contents.clone(),
        system_instruction,
        generation_config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_request::Instructions;

    #[test]
    fn builds_contents_verbatim() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content::user_text("hello"));
        let body = build_request_body(&request);
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn carries_the_system_instruction_as_a_role_less_content() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.append_instructions(Instructions::Strings(vec!["be helpful".to_string()]));
        let body = build_request_body(&request);
        let system_instruction = body.system_instruction.unwrap();
        assert!(system_instruction.role.is_none());
        assert_eq!(
            system_instruction.parts[0].text.as_deref(),
            Some("be helpful")
        );
    }

    #[test]
    fn omits_generation_config_when_no_output_schema_is_set() {
        let request = LlmRequest::new("gemini-2.5-flash");
        let body = build_request_body(&request);
        assert!(body.generation_config.is_none());
    }

    #[test]
    fn carries_the_output_schema_into_generation_config() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.set_output_schema(Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]));
        let body = build_request_body(&request);
        let config = body.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type.as_deref(),
            Some("application/json")
        );
        assert!(config.response_schema.is_some());
    }

    #[test]
    fn serializes_with_camel_case_keys_and_no_tools_field() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content::user_text("hi"));
        let body = build_request_body(&request);
        let json = rusty_serde::json::to_string(&body).unwrap();
        assert!(json.contains("\"contents\""));
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("system_instruction"));
    }
}
