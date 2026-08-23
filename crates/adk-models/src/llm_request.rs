//! Capabilities C0114-C0115, C0117-C0118: `LlmRequest`, ported from
//! `google.adk.models.llm_request`.
//!
//! **Deferred**: `append_tools` (C0116) needs `BaseTool` (Phase 8) — nothing
//! to convert to function declarations yet, and `tools_dict` (excluded from
//! serialization in the source, populated only by `append_tools`) is
//! omitted entirely rather than kept as a dead field.
//!
//! **Adaptation**: `config`/`live_connect_config` are opaque
//! `google.genai.types.GenerateContentConfig`/`LiveConnectConfig` in the
//! source. Rather than one big opaque [`rusty_serde::value::Value`],
//! `config` is narrowed to [`GenerateContentConfigStub`] — just the
//! sub-fields `LlmRequest`'s own methods actually read or write
//! (`system_instruction`, `response_schema`, `response_mime_type`) — since
//! those methods *mutate* specific keys, not just validate presence (unlike
//! `LlmAgent`'s `generate_content_config` validator, which only checks keys
//! and can stay a generic `Value` map). `part.inline_data`/
//! `file_data`'s inner shape (`display_name`/`mime_type`/`file_uri`) is
//! opaque, so `append_instructions`' reference text omits the source's
//! descriptive suffix (`"(type: ..., 'name')"`) that would need those
//! fields — the reference id and presence-based branching are preserved.
//! `tools`/`thinking_config`/`safety_settings` on both config stubs stay
//! opaque `Value` placeholders — `tools` needs `BaseTool` (Phase 8,
//! C0116), and nothing here reads inside `thinking_config`/
//! `safety_settings`, only forwards them.
//!
//! **Adaptation, Phase 3 batch 4**: `live_connect_config` was a bare
//! `Value` placeholder ("nothing here reads it") until `Gemini.connect()`'s
//! config-preparation logic (C0131) needed real fields to mutate — now
//! [`LiveConnectConfigStub`], narrowed the same way `config` is.

use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::cache_metadata::CacheMetadata;

/// Narrowed placeholder for `google.genai.types.GenerateContentConfig` —
/// see the module doc.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GenerateContentConfigStub {
    #[rusty_serde(default)]
    pub system_instruction: Option<String>,
    #[rusty_serde(default)]
    pub response_schema: Option<Value>,
    #[rusty_serde(default)]
    pub response_mime_type: Option<String>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub tools: Option<Value>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub thinking_config: Option<Value>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub safety_settings: Option<Value>,
    /// Opaque placeholder for `types.ToolConfig` — added in Phase 3 batch 7
    /// (`GeminiContextCacheManager`, C0141/C0143) since the fingerprint and
    /// `_apply_cache_to_request` both read/clear it, without needing its
    /// internal shape.
    #[rusty_serde(default)]
    pub tool_config: Option<Value>,
    /// The active cache's resource name, set by
    /// `GeminiContextCacheManager::apply_cache_to_request` (C0143) once a
    /// request is routed through an explicit cache.
    #[rusty_serde(default)]
    pub cached_content: Option<String>,
}

/// Narrowed placeholder for `google.genai.types.HttpOptions`, as embedded
/// in [`LiveConnectConfigStub`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HttpOptionsStub {
    #[rusty_serde(default)]
    pub headers: Option<BTreeMap<String, String>>,
    #[rusty_serde(default)]
    pub api_version: Option<String>,
}

/// Narrowed placeholder for `google.genai.types.SessionResumptionConfig`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SessionResumptionStub {
    #[rusty_serde(default)]
    pub transparent: Option<bool>,
}

/// Narrowed placeholder for `google.genai.types.LiveConnectConfig` — see
/// the module doc's Phase 3 batch 4 adaptation note.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LiveConnectConfigStub {
    #[rusty_serde(default)]
    pub http_options: Option<HttpOptionsStub>,
    /// Opaque placeholder for `types.SpeechConfig`.
    #[rusty_serde(default)]
    pub speech_config: Option<Value>,
    #[rusty_serde(default)]
    pub system_instruction: Option<Content>,
    #[rusty_serde(default)]
    pub session_resumption: Option<SessionResumptionStub>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub tools: Option<Value>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub thinking_config: Option<Value>,
    /// Opaque placeholder — see the module doc.
    #[rusty_serde(default)]
    pub safety_settings: Option<Value>,
}

/// Either shape `append_instructions` accepts — the source's
/// `Union[list[str], types.Content]`.
pub enum Instructions {
    Strings(Vec<String>),
    Content(Content),
}

/// LLM request class: contents, tools, output schema, and system
/// instructions to send to the model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LlmRequest {
    pub model: Option<String>,
    pub contents: Vec<Content>,
    pub config: GenerateContentConfigStub,
    pub live_connect_config: Option<LiveConnectConfigStub>,
    pub cache_config: Option<adk_agents::context_cache_config::ContextCacheConfig>,
    pub cache_metadata: Option<CacheMetadata>,
    pub cacheable_contents_token_count: Option<i64>,
    pub previous_interaction_id: Option<String>,

    dynamic_instructions: Vec<String>,
    has_static_instruction: bool,
    static_instruction_prefix_end_index: Option<usize>,
}

impl LlmRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Default::default()
        }
    }

    /// C0115: appends instructions to the system instruction.
    ///
    /// Returns the user contents extracted from non-text parts (only when
    /// `instructions` is [`Instructions::Content`] with non-text parts;
    /// empty otherwise).
    pub fn append_instructions(&mut self, instructions: Instructions) -> Vec<Content> {
        match instructions {
            Instructions::Content(content) => {
                let mut text_parts = Vec::new();
                let mut user_contents = Vec::new();
                let mut non_text_count = 0usize;

                for part in &content.parts {
                    if let Some(text) = &part.text {
                        text_parts.push(text.clone());
                    } else if part.inline_data.is_some() {
                        let reference_id = format!("inline_data_{non_text_count}");
                        non_text_count += 1;
                        text_parts
                            .push(format!("[Reference to inline binary data: {reference_id}]"));
                        user_contents.push(Content::new(
                            "user",
                            vec![
                                Part::text(format!("Referenced inline data: {reference_id}")),
                                Part {
                                    inline_data: part.inline_data.clone(),
                                    ..Default::default()
                                },
                            ],
                        ));
                    } else if part.file_data.is_some() {
                        let reference_id = format!("file_data_{non_text_count}");
                        non_text_count += 1;
                        text_parts.push(format!("[Reference to file data: {reference_id}]"));
                        user_contents.push(Content::new(
                            "user",
                            vec![
                                Part::text(format!("Referenced file data: {reference_id}")),
                                Part {
                                    file_data: part.file_data.clone(),
                                    ..Default::default()
                                },
                            ],
                        ));
                    }
                }

                self.append_system_instruction_text(&text_parts);

                if !user_contents.is_empty() {
                    self.contents.extend(user_contents.clone());
                    self.has_static_instruction = true;
                }
                user_contents
            }
            Instructions::Strings(instructions) => {
                if instructions.is_empty() {
                    return Vec::new();
                }
                self.append_system_instruction_text(&instructions);
                Vec::new()
            }
        }
    }

    fn append_system_instruction_text(&mut self, parts: &[String]) {
        if parts.is_empty() {
            return;
        }
        let new_text = parts.join("\n\n");
        match &mut self.config.system_instruction {
            None => self.config.system_instruction = Some(new_text),
            Some(existing) => {
                existing.push_str("\n\n");
                existing.push_str(&new_text);
            }
        }
    }

    /// Dynamic instructions generated by tools, resolved into finalized
    /// contents/system instructions elsewhere (Phase 4's flow engine).
    pub fn append_dynamic_instructions(&mut self, instructions: &[String]) {
        self.dynamic_instructions.extend_from_slice(instructions);
    }

    pub fn dynamic_instructions(&self) -> &[String] {
        &self.dynamic_instructions
    }

    /// C0117: inserts request-scoped content at the current-turn boundary —
    /// before the latest ordinary user batch, but after a function response
    /// when the model is continuing a tool-call turn; a static-instruction
    /// prefix (if present) always stays first.
    pub fn insert_transient_user_content(&mut self, contents: Vec<Content>) {
        if contents.is_empty() {
            return;
        }

        if self.has_static_instruction && self.static_instruction_prefix_end_index.is_none() {
            let inserted_len = contents.len();
            self.contents.splice(0..0, contents);
            self.static_instruction_prefix_end_index = Some(inserted_len);
            return;
        }

        let mut insert_index = self.contents.len();
        for i in (0..self.contents.len()).rev() {
            let content = &self.contents[i];
            if content.role.as_deref() != Some("user") {
                insert_index = i + 1;
                break;
            }
            if content.parts.iter().any(|p| p.function_response.is_some()) {
                insert_index = i + 1;
                break;
            }
            insert_index = i;
        }

        if self.has_static_instruction {
            let prefix_end = self
                .static_instruction_prefix_end_index
                .expect("has_static_instruction implies the prefix end is tracked");
            insert_index = insert_index.max(prefix_end);
        }

        self.contents.splice(insert_index..insert_index, contents);
    }

    /// C0118: sets the output schema for the request.
    pub fn set_output_schema(&mut self, output_schema: Value) {
        self.config.response_schema = Some(output_schema);
        self.config.response_mime_type = Some("application/json".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::FunctionResponse;

    #[test]
    fn append_instructions_strings_concatenates_with_double_newline() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.append_instructions(Instructions::Strings(vec!["a".to_string()]));
        request.append_instructions(Instructions::Strings(vec!["b".to_string()]));
        assert_eq!(request.config.system_instruction.as_deref(), Some("a\n\nb"));
    }

    #[test]
    fn append_instructions_content_extracts_text_and_appends_it() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let content = Content::new("user", vec![Part::text("be helpful")]);
        let user_contents = request.append_instructions(Instructions::Content(content));
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("be helpful")
        );
        assert!(user_contents.is_empty());
    }

    #[test]
    fn append_instructions_content_turns_inline_data_into_a_reference_and_user_content() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let content = Content::new(
            "user",
            vec![Part {
                inline_data: Some(adk_genai::content::MediaBlobStub {
                    mime_type: Some("application/octet-stream".to_string()),
                    rest: None,
                }),
                ..Default::default()
            }],
        );
        let user_contents = request.append_instructions(Instructions::Content(content));
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("[Reference to inline binary data: inline_data_0]")
        );
        assert_eq!(user_contents.len(), 1);
        assert!(request.has_static_instruction);
    }

    #[test]
    fn insert_transient_user_content_is_a_noop_for_an_empty_list() {
        let mut request = LlmRequest::new("m");
        request.contents.push(Content::user_text("hi"));
        request.insert_transient_user_content(vec![]);
        assert_eq!(request.contents.len(), 1);
    }

    #[test]
    fn insert_transient_user_content_inserts_before_the_latest_user_batch() {
        let mut request = LlmRequest::new("m");
        request
            .contents
            .push(Content::new("model", vec![Part::text("reply")]));
        request.contents.push(Content::user_text("latest question"));
        request.insert_transient_user_content(vec![Content::user_text("retrieved context")]);
        assert_eq!(request.contents.len(), 3);
        assert_eq!(
            request.contents[1].parts[0].text.as_deref(),
            Some("retrieved context")
        );
    }

    #[test]
    fn insert_transient_user_content_lands_after_a_function_response_continuation() {
        let mut request = LlmRequest::new("m");
        request.contents.push(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse::default())],
        ));
        request.insert_transient_user_content(vec![Content::user_text("dynamic instruction")]);
        assert_eq!(request.contents.len(), 2);
        assert_eq!(
            request.contents[1].parts[0].text.as_deref(),
            Some("dynamic instruction")
        );
    }

    #[test]
    fn set_output_schema_sets_response_schema_and_mime_type() {
        let mut request = LlmRequest::new("m");
        request.set_output_schema(Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]));
        assert!(request.config.response_schema.is_some());
        assert_eq!(
            request.config.response_mime_type.as_deref(),
            Some("application/json")
        );
    }
}
