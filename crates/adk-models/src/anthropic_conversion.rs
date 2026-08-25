//! C0540/C0542 (partial P10 slice): pure, HTTP-free conversion helpers
//! from `models/anthropic_llm.py` — the Anthropic (Claude) model
//! backend. No new dependency: this port talks to Anthropic's Messages
//! API the same way `gemini.rs` already talks to Gemini's REST API
//! (`reqwest::blocking::Client`, since `adk-models` already depends on
//! `reqwest`/`regex` — see that file's own module doc for why the
//! blocking client, not the async one, is required under this
//! workspace's `rusty_tokio` runtime), not via the Python source's
//! `anthropic` SDK — there is no Rust equivalent of that SDK to add as a
//! dependency, and none is needed for a plain HTTPS JSON API.
//!
//! **Scope of this batch, disclosed**: [`ToolUseIdSanitizer`] (C0540),
//! the finish-reason mapping + token-usage extraction/reconciliation
//! functions (C0542), [`update_type_string`]/
//! [`function_declaration_to_tool_param`] (C0541),
//! [`build_anthropic_thinking_param`] (C0538, `thinking_budget`-only
//! subset), and now the full `Content`↔Anthropic block conversion —
//! [`part_to_message_block`]/[`content_to_message_param`] (request
//! direction) and [`content_block_to_part`] (response direction), C0539
//! — are ported here. All pure, self-contained, and testable without
//! any wire-format type beyond the minimal
//! [`AnthropicUsage`]/[`AnthropicToolParam`]/[`AnthropicThinkingParam`]/
//! [`AnthropicMessageBlock`]/[`AnthropicResponseBlock`] structs declared
//! alongside them (all ahead of their own real caller, same
//! "widen/declare ahead of a consumer" precedent used throughout this
//! port — their real consumer is the still-deferred `AnthropicLlm`
//! backend, so none of these types carry `Serialize`/`Deserialize` yet
//! either — the real HTTP-transport caller (C0536/C0537) is what will
//! need to decide the exact wire-tagging, same as [`AnthropicToolParam`]
//! already left undecided). The rest of P10 (C0536, C0537, C0543,
//! C0544 — the actual `AnthropicLlm` `BaseLlm` backend, credential
//! resolution, and SSE streaming) is real, substantial additional work
//! deliberately left for a follow-up batch: each of those needs either
//! HTTP-client wiring or new fields on
//! [`crate::llm_request::GenerateContentConfigStub`]
//! (`temperature`/`top_p`/`top_k`/`stop_sequences`/`max_output_tokens`,
//! none of which exist there yet) — real, separable units of work, not
//! something to fold into this slice. C0541, C0538, and now C0539
//! turned out **not** to need any of that — C0541 only touches
//! [`adk_genai::content::FunctionDeclaration`], C0538's
//! `thinking_budget` mapping only reads
//! [`crate::llm_request::GenerateContentConfigStub::thinking_config`],
//! and C0539's conversion functions only touch
//! [`adk_genai::content::Content`]/[`Part`] (already real, non-opaque
//! types) plus this module's own new wire-shape stand-ins — all of
//! which already have everything required. **C0538's other half stays
//! deferred**: `_build_effort_param`/`AnthropicGenerateContentConfig.effort`
//! (Anthropic's separate `reasoning_effort` request field, distinct from
//! `thinking_budget`) needs a genuinely new field on
//! `AnthropicGenerateContentConfig` (a type that doesn't exist in this
//! port yet, since the whole `AnthropicLlm`-specific config subclass is
//! part of the deferred `AnthropicLlm` backend) — left for that
//! follow-up batch rather than bolted on here.
//!
//! **C0539's `python_str_stand_in`, disclosed**: the source's
//! `function_response` branch falls back to Python's `str()`/dict-
//! `repr()` stringification for a tool-response content item that isn't
//! already `{"type": "text", "text": ...}`-shaped. This port uses
//! compact JSON instead (a bare string still round-trips unquoted), the
//! same disclosed lower-fidelity idiom
//! `llm_backed_user_simulator.rs::display_args` already establishes.
//!
//! **C0539's `thought_signature`, no byte-level codec needed**: the
//! source's `part.thought_signature` is raw `bytes`, encoded/decoded via
//! plain UTF-8 (not base64) when moving to/from Anthropic's `signature`/
//! `data` string fields. This port's [`Part::thought_signature`] is
//! already an opaque [`Value`] holding that same string directly (see
//! `content.rs`'s own doc and this crate's other `thought_signature`
//! call sites) — so conversion here is a direct `Value::String` read/
//! write, with no encode/decode step to port.
//!
//! **`to_google_genai_finish_reason`, wire string not enum**: the source
//! maps to a `types.FinishReason` enum member; this port's
//! `LlmResponse::finish_reason` is already an opaque
//! [`rusty_serde::value::Value`] holding the raw wire string (e.g.
//! `Value::String("STOP")`, per `llm_response.rs`'s own tests) — so this
//! returns that same wire string directly rather than a typed enum,
//! consistent with the "no enum to normalize away" precedent already
//! established in `stable_semconv.rs`.

use adk_genai::content::{Content, FunctionCall, FunctionResponse, MediaBlobStub, Part};
use rusty_serde::value::Value;
use std::collections::{BTreeMap, HashMap};

/// `anthropic_llm._ToolUseIdSanitizer` — maps invalid `tool_use` ids to
/// deterministic fallbacks. Reuse one instance per conversation so a
/// `tool_use` and its paired `tool_result` with the same invalid source
/// id get matching outputs.
#[derive(Debug, Default)]
pub struct ToolUseIdSanitizer {
    mapping: HashMap<String, String>,
    next_fallback: u64,
}

fn is_valid_tool_id(tool_id: &str) -> bool {
    !tool_id.is_empty()
        && tool_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl ToolUseIdSanitizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `_ToolUseIdSanitizer.sanitize`.
    pub fn sanitize(&mut self, tool_id: Option<&str>) -> String {
        if let Some(tool_id) = tool_id {
            if is_valid_tool_id(tool_id) {
                return tool_id.to_string();
            }
        }
        let key = tool_id.unwrap_or("").to_string();
        if let Some(existing) = self.mapping.get(&key) {
            return existing.clone();
        }
        let fallback = format!("toolu_fallback_{}", self.next_fallback);
        self.next_fallback += 1;
        self.mapping.insert(key, fallback.clone());
        fallback
    }
}

/// Anthropic stop-reason wire strings → the GenAI `FinishReason` wire
/// string this port's [`crate::llm_response::LlmResponse::finish_reason`]
/// already holds directly (see the module doc for why no typed enum is
/// introduced here).
///
/// `anthropic_llm._STOP_REASON_MAPPING`/`to_google_genai_finish_reason`.
pub fn to_google_genai_finish_reason(anthropic_stop_reason: Option<&str>) -> Option<Value> {
    let stop_reason = anthropic_stop_reason?;
    let mapped = match stop_reason {
        "end_turn" | "stop_sequence" | "tool_use" | "pause_turn" => "STOP",
        "max_tokens" => "MAX_TOKENS",
        "refusal" => "SAFETY",
        _ => "FINISH_REASON_UNSPECIFIED",
    };
    Some(Value::String(mapped.to_string()))
}

/// `anthropic_types.Usage`/`MessageDeltaUsage` — only the fields this
/// port's token-usage reconciliation reads. Declared ahead of its real
/// caller (`message_to_generate_content_response`, deferred — see the
/// module doc), the same precedent used throughout this port.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    /// `usage.output_tokens_details.thinking_tokens`.
    pub thinking_tokens: Option<i64>,
}

/// `_extract_prompt_token_count` — every input token billed for the
/// turn. Anthropic reports tokens served from the prompt cache and
/// tokens written to it in their own fields, disjoint from
/// `input_tokens`; the GenAI shape instead expects a single prompt count
/// with the cached portion folded in (`cached_content_token_count` is a
/// breakdown of it, not an addition to it), so this sums all three.
pub fn extract_prompt_token_count(usage: &AnthropicUsage) -> i64 {
    usage.input_tokens.unwrap_or(0)
        + usage.cache_read_input_tokens.unwrap_or(0)
        + usage.cache_creation_input_tokens.unwrap_or(0)
}

/// `_extract_cached_token_count` — Anthropic cache-read tokens, the
/// analog of `cached_content_token_count`.
pub fn extract_cached_token_count(usage: &AnthropicUsage) -> Option<i64> {
    usage.cache_read_input_tokens
}

/// `_extract_cache_creation_token_count` — Anthropic cache-write tokens,
/// the analog of `cache_creation_input_tokens`.
pub fn extract_cache_creation_token_count(usage: &AnthropicUsage) -> Option<i64> {
    usage.cache_creation_input_tokens
}

/// `_extract_thinking_token_count` — Anthropic counts extended-thinking
/// tokens inside `output_tokens`, whereas the GenAI shape keeps the
/// candidate and thought counts disjoint and sums them downstream;
/// callers subtract this from `output_tokens` to get the candidate
/// count. Clamped to `output_tokens` so that subtraction stays
/// non-negative even if the two counters ever disagree.
pub fn extract_thinking_token_count(usage: &AnthropicUsage) -> Option<i64> {
    let thinking = usage.thinking_tokens?;
    match usage.output_tokens {
        Some(output_tokens) => Some(thinking.min(output_tokens)),
        None => Some(thinking),
    }
}

const DICT_KEYS_TO_RECURSE: &[&str] = &[
    "$defs",
    "defs",
    "dependentSchemas",
    "patternProperties",
    "properties",
];

const SINGLE_KEYS_TO_RECURSE: &[&str] = &[
    "additionalProperties",
    "additional_properties",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedProperties",
];

const LIST_KEYS_TO_RECURSE: &[&str] = &[
    "allOf",
    "all_of",
    "anyOf",
    "any_of",
    "oneOf",
    "one_of",
    "prefixItems",
];

/// `anthropic_llm._update_type_string` — lowercases nested JSON-Schema
/// `"type"` strings for Anthropic compatibility, recursing into every
/// key a JSON Schema commonly nests a sub-schema under.
pub fn update_type_string(value: &mut Value) {
    match value {
        Value::Seq(items) => {
            for item in items {
                update_type_string(item);
            }
        }
        Value::Map(entries) => {
            if let Some((_, Value::String(type_name))) =
                entries.iter_mut().find(|(k, _)| k == "type")
            {
                *type_name = type_name.to_lowercase();
            }
            for key in DICT_KEYS_TO_RECURSE {
                if let Some((_, Value::Map(child_entries))) =
                    entries.iter_mut().find(|(k, _)| k == key)
                {
                    for (_, child_value) in child_entries {
                        update_type_string(child_value);
                    }
                }
            }
            for key in SINGLE_KEYS_TO_RECURSE {
                if let Some((_, child)) = entries.iter_mut().find(|(k, _)| k == key) {
                    if matches!(child, Value::Map(_) | Value::Seq(_)) {
                        update_type_string(child);
                    }
                }
            }
            for key in LIST_KEYS_TO_RECURSE {
                if let Some((_, child @ Value::Seq(_))) = entries.iter_mut().find(|(k, _)| k == key)
                {
                    update_type_string(child);
                }
            }
        }
        _ => {}
    }
}

/// `anthropic_types.ToolParam` — only the three fields the source ever
/// sets when constructing one (`anthropic_llm.py:801-805`). Declared
/// ahead of its real HTTP-transport caller, same precedent as
/// [`AnthropicUsage`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicToolParam {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// `anthropic_llm.function_declaration_to_tool_param` — converts a
/// function declaration to an Anthropic tool param.
///
/// **`parameters`-fallback, simplified — disclosed**: the source's
/// `else` branch reads `function_declaration.parameters.properties`, a
/// dict of typed `Schema` objects, and calls `.model_dump(by_alias=True,
/// exclude_none=True)` on each one to flatten it to a plain dict.
/// [`adk_genai::content::FunctionDeclaration::parameters`] is already an
/// opaque, already-flattened [`Value`] in this port (no typed `Schema`
/// object per property to begin with — same "already opaque, nothing
/// left to strip" situation this module already discloses for
/// `usage_metadata`/token extraction), so this reads `parameters`'s own
/// `"properties"`/`"required"` keys directly instead of rebuilding them
/// key-by-key.
///
/// Panics if `function_declaration.name` is `None` or empty, mirroring
/// the source's own `assert function_declaration.name` — a caller
/// invariant, not a user-reachable error path (this function has no
/// real caller yet to receive a `Result` through — see the module doc).
pub fn function_declaration_to_tool_param(
    function_declaration: &adk_genai::content::FunctionDeclaration,
) -> AnthropicToolParam {
    let name = function_declaration
        .name
        .clone()
        .filter(|n| !n.is_empty())
        .expect("function_declaration.name is required");

    let mut input_schema = if let Some(schema) = &function_declaration.parameters_json_schema {
        schema.clone()
    } else {
        let properties = function_declaration
            .parameters
            .as_ref()
            .and_then(|p| p.get("properties"))
            .cloned()
            .unwrap_or_else(|| Value::Map(Vec::new()));
        let required = function_declaration
            .parameters
            .as_ref()
            .and_then(|p| p.get("required"))
            .cloned();

        let mut schema = Value::Map(Vec::new());
        schema.insert("type", Value::String("object".to_string()));
        schema.insert("properties", properties);
        if let Some(required) = required {
            let is_empty = matches!(&required, Value::Seq(items) if items.is_empty());
            if !is_empty {
                schema.insert("required", required);
            }
        }
        schema
    };
    update_type_string(&mut input_schema);

    AnthropicToolParam {
        name,
        description: function_declaration.description.clone().unwrap_or_default(),
        input_schema,
    }
}

/// Wire-shape stand-in for Anthropic's `thinking` request param
/// (`anthropic_types.ThinkingConfigEnabledParam`/`ThinkingConfigDisabledParam`/
/// `ThinkingConfigAdaptiveParam`). Declared ahead of its real HTTP-
/// transport caller, same precedent as [`AnthropicUsage`]/
/// [`AnthropicToolParam`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinkingParam {
    Disabled,
    Adaptive,
    Enabled { budget_tokens: i64 },
}

/// `anthropic_llm._build_anthropic_thinking_param` — maps genai
/// `ThinkingConfig` to Anthropic's `thinking` request parameter.
///
/// Per `google.genai.types.ThinkingConfig`, `thinking_budget` semantics
/// are: `None` (no `thinking_config`, or a `thinking_config` present but
/// with no `thinkingBudget` key) means Anthropic requires an explicit
/// choice, surfaced as `Err` rather than silently picking a default
/// (mirroring the Anthropic API); `0` disables thinking; negative
/// (e.g. `-1` AUTOMATIC) maps to Anthropic's adaptive thinking (model
/// picks the depth); a positive budget is legacy manual mode.
///
/// **`thinking_config`, already-opaque `Value`, no widening needed**:
/// unlike the rest of P10's extended-thinking/reasoning-effort row
/// (`AnthropicGenerateContentConfig.effort`, still `REQUIRED` — a
/// genuinely new field, deferred), this only reads
/// `GenerateContentConfigStub::thinking_config`'s already-opaque
/// `Value`, which has existed since an early Phase 3 batch — no change
/// to that struct is needed. `"thinkingBudget"` is the same camelCase
/// key `llm_backed_user_simulator.rs`'s `default_model_configuration`
/// already writes into this exact field.
pub fn build_anthropic_thinking_param(
    config: Option<&crate::llm_request::GenerateContentConfigStub>,
) -> Result<Option<AnthropicThinkingParam>, String> {
    let Some(thinking_config) = config.and_then(|c| c.thinking_config.as_ref()) else {
        return Ok(None);
    };

    let Some(thinking_budget) = thinking_config
        .get("thinkingBudget")
        .and_then(Value::as_i64)
    else {
        return Err(
            "thinking_budget must be set explicitly when ThinkingConfig is provided for \
             Anthropic models. Use 0 to disable thinking, -1 for adaptive (model-chosen \
             depth), or a positive integer (>= 1024) for manual budgeting."
                .to_string(),
        );
    };

    if thinking_budget == 0 {
        return Ok(Some(AnthropicThinkingParam::Disabled));
    }
    if thinking_budget < 0 {
        return Ok(Some(AnthropicThinkingParam::Adaptive));
    }
    Ok(Some(AnthropicThinkingParam::Enabled {
        budget_tokens: thinking_budget,
    }))
}

/// `anthropic_llm.to_claude_role` — Anthropic only has `"user"`/
/// `"assistant"` roles; any genai role other than `"model"`/`"assistant"`
/// (including `None`) maps to `"user"`.
pub fn to_claude_role(role: Option<&str>) -> &'static str {
    match role {
        Some("model") | Some("assistant") => "assistant",
        _ => "user",
    }
}

const ANTHROPIC_IMAGE_MEDIA_TYPES: &[&str] =
    &["image/jpeg", "image/png", "image/gif", "image/webp"];

fn is_image_part(part: &Part) -> bool {
    part.inline_data
        .as_ref()
        .and_then(|blob| blob.mime_type.as_deref())
        .is_some_and(|mime_type| mime_type.starts_with("image/"))
}

fn is_pdf_part(part: &Part) -> bool {
    part.inline_data
        .as_ref()
        .and_then(|blob| blob.mime_type.as_deref())
        .is_some_and(|mime_type| {
            mime_type.split(';').next().unwrap_or(mime_type).trim() == "application/pdf"
        })
}

fn normalize_image_media_type(mime_type: &str) -> Result<String, String> {
    let normalized = mime_type
        .split(';')
        .next()
        .unwrap_or(mime_type)
        .trim()
        .to_lowercase();
    if ANTHROPIC_IMAGE_MEDIA_TYPES.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        Err(format!(
            "Unsupported Anthropic image MIME type: {mime_type}"
        ))
    }
}

/// A [`MediaBlobStub`]'s already-base64-encoded `"data"` wire key — this
/// port stores it opaquely in `rest` (see that struct's own doc), and
/// Anthropic's own block params want the same base64 string directly, so
/// there's no decode/re-encode round trip to do here (unlike the
/// source, whose `types.Blob.data` is raw bytes it re-encodes with
/// `base64.b64encode`).
fn blob_data_base64(blob: &MediaBlobStub) -> Option<&str> {
    blob.rest.as_ref()?.get("data")?.as_str()
}

/// Wire-shape stand-in for Anthropic's `Base64ImageSourceParam`/
/// `Base64PDFSourceParam` — declared ahead of its real HTTP-transport
/// caller, same precedent as [`AnthropicToolParam`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicBase64Source {
    pub media_type: String,
    pub data: String,
}

/// `anthropic_llm._ToolResultContentBlockParam` — the subset of block
/// types Claude accepts inside a tool result.
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicToolResultBlock {
    Text(String),
    Image(AnthropicBase64Source),
    Document(AnthropicBase64Source),
}

/// `anthropic_types.ToolResultBlockParam.content`'s `str |
/// list[_ToolResultContentBlockParam]` union.
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

/// `anthropic_llm._MessageBlockParam` — a request-side content block.
/// Declared ahead of its real HTTP-transport caller, same precedent as
/// [`AnthropicToolParam`]/[`AnthropicThinkingParam`].
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicMessageBlock {
    Text(String),
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    Image(AnthropicBase64Source),
    Document(AnthropicBase64Source),
    ToolUse {
        id: String,
        name: String,
        input: BTreeMap<String, Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        is_error: bool,
    },
}

/// `anthropic_types.MessageParam` — a full request-side message.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicMessageParam {
    pub role: &'static str,
    pub content: Vec<AnthropicMessageBlock>,
}

/// `anthropic_llm._function_response_media_blocks` — converts media a
/// tool attached to its response into tool result blocks. Media Claude
/// cannot carry in a tool result is dropped with a warning (`eprintln!`,
/// this port's established ad hoc warning convention) rather than
/// raised on, since the tool that produced it is often third-party code
/// the caller cannot change.
fn function_response_media_blocks(
    function_response: &FunctionResponse,
) -> Vec<AnthropicToolResultBlock> {
    let mut blocks = Vec::new();
    for response_part in function_response.parts.iter().flatten() {
        let Some(blob) = &response_part.inline_data else {
            continue;
        };
        let Some(data) = blob_data_base64(blob) else {
            continue;
        };
        let Some(mime_type) = &blob.mime_type else {
            continue;
        };
        let media_type = mime_type
            .split(';')
            .next()
            .unwrap_or(mime_type)
            .trim()
            .to_lowercase();
        if ANTHROPIC_IMAGE_MEDIA_TYPES.contains(&media_type.as_str()) {
            blocks.push(AnthropicToolResultBlock::Image(AnthropicBase64Source {
                media_type,
                data: data.to_string(),
            }));
        } else if media_type == "application/pdf" {
            blocks.push(AnthropicToolResultBlock::Document(AnthropicBase64Source {
                media_type,
                data: data.to_string(),
            }));
        } else {
            eprintln!(
                "Dropping tool result media of type {media_type}, which Claude cannot receive \
                 in a tool result."
            );
        }
    }
    blocks
}

/// Stand-in for Python's `str()`/dict-`repr()` stringification of a
/// tool-response value that isn't already `{"type": "text", "text":
/// ...}`-shaped — compact JSON instead, the same disclosed lower-
/// fidelity idiom `llm_backed_user_simulator.rs::display_args` already
/// establishes. A bare string round-trips unquoted, matching `str(s)`.
fn python_str_stand_in(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => rusty_serde::json::to_string(other).unwrap_or_default(),
    }
}

/// `anthropic_llm._part_to_message_block`'s `function_response` branch,
/// factored out: builds the serializable `content` text a tool's
/// response contributes, before any media blocks are appended.
fn function_response_content_text(response_data: &BTreeMap<String, Value>) -> String {
    match response_data.get("content") {
        Some(Value::Seq(items)) if !items.is_empty() => items
            .iter()
            .map(|item| match item {
                Value::Map(_)
                    if item.get("type").and_then(Value::as_str) == Some("text")
                        && item.get("text").is_some() =>
                {
                    item.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                }
                other => python_str_stand_in(other),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => match response_data.get("result") {
            Some(result) if !matches!(result, Value::Null) => match result {
                Value::Map(_) | Value::Seq(_) => {
                    rusty_serde::json::to_string(result).unwrap_or_default()
                }
                other => python_str_stand_in(other),
            },
            _ => {
                if response_data.is_empty() {
                    String::new()
                } else {
                    let value = Value::Map(
                        response_data
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    );
                    rusty_serde::json::to_string(&value).unwrap_or_default()
                }
            }
        },
    }
}

fn function_response_to_block(
    function_response: &FunctionResponse,
    sanitizer: &mut ToolUseIdSanitizer,
) -> AnthropicMessageBlock {
    let response_data = function_response.response.clone().unwrap_or_default();
    let content_text = function_response_content_text(&response_data);

    let media_blocks = function_response_media_blocks(function_response);
    let content = if media_blocks.is_empty() {
        AnthropicToolResultContent::Text(content_text)
    } else {
        let mut blocks = Vec::new();
        if !content_text.is_empty() {
            blocks.push(AnthropicToolResultBlock::Text(content_text));
        }
        blocks.extend(media_blocks);
        AnthropicToolResultContent::Blocks(blocks)
    };

    AnthropicMessageBlock::ToolResult {
        tool_use_id: sanitizer.sanitize(function_response.id.as_deref()),
        content,
        is_error: false,
    }
}

fn image_part_to_block(part: &Part) -> Result<AnthropicMessageBlock, String> {
    let blob = part
        .inline_data
        .as_ref()
        .ok_or_else(|| "Anthropic image parts require MIME type and data".to_string())?;
    let data = blob_data_base64(blob)
        .ok_or_else(|| "Anthropic image parts require MIME type and data".to_string())?;
    let mime_type = blob
        .mime_type
        .as_deref()
        .ok_or_else(|| "Anthropic image parts require MIME type and data".to_string())?;
    let media_type = normalize_image_media_type(mime_type)?;
    Ok(AnthropicMessageBlock::Image(AnthropicBase64Source {
        media_type,
        data: data.to_string(),
    }))
}

fn pdf_part_to_block(part: &Part) -> Result<AnthropicMessageBlock, String> {
    let blob = part
        .inline_data
        .as_ref()
        .ok_or_else(|| "Anthropic PDF parts require data".to_string())?;
    let data =
        blob_data_base64(blob).ok_or_else(|| "Anthropic PDF parts require data".to_string())?;
    Ok(AnthropicMessageBlock::Document(AnthropicBase64Source {
        media_type: "application/pdf".to_string(),
        data: data.to_string(),
    }))
}

/// `anthropic_llm._part_to_message_block`.
///
/// The bare `assert function_call.name` in the source's `function_call`
/// branch is ported as a Rust panic, the same "assert = caller
/// invariant" convention [`function_declaration_to_tool_param`] already
/// establishes in this file — distinct from the final `NotImplementedError`
/// case (a genuinely unsupported part shape), which this port surfaces as
/// `Err` since it's a real, reachable failure mode, not an invariant.
fn part_to_message_block_with(
    part: &Part,
    sanitizer: &mut ToolUseIdSanitizer,
) -> Result<AnthropicMessageBlock, String> {
    let has_text = part.text.as_deref().is_some_and(|text| !text.is_empty());
    let has_thought_signature = part
        .thought_signature
        .as_ref()
        .and_then(Value::as_str)
        .is_some_and(|signature| !signature.is_empty());

    if part.thought == Some(true) && has_text {
        let signature = if has_thought_signature {
            part.thought_signature
                .as_ref()
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        } else {
            String::new()
        };
        return Ok(AnthropicMessageBlock::Thinking {
            thinking: part.text.clone().unwrap_or_default(),
            signature,
        });
    }
    if part.thought == Some(true) && has_thought_signature {
        return Ok(AnthropicMessageBlock::RedactedThinking {
            data: part
                .thought_signature
                .as_ref()
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });
    }
    if let Some(text) = &part.text {
        return Ok(AnthropicMessageBlock::Text(text.clone()));
    }
    if let Some(function_call) = &part.function_call {
        let name = function_call
            .name
            .clone()
            .filter(|name| !name.is_empty())
            .expect("function_call.name is required");
        return Ok(AnthropicMessageBlock::ToolUse {
            id: sanitizer.sanitize(function_call.id.as_deref()),
            name,
            input: function_call.args.clone().unwrap_or_default(),
        });
    }
    if let Some(function_response) = &part.function_response {
        return Ok(function_response_to_block(function_response, sanitizer));
    }
    if is_image_part(part) {
        return image_part_to_block(part);
    }
    if is_pdf_part(part) {
        return pdf_part_to_block(part);
    }
    if let Some(executable_code) = &part.executable_code {
        let code = executable_code
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or_default();
        return Ok(AnthropicMessageBlock::Text(format!(
            "Code:```python\n{code}\n```"
        )));
    }
    if let Some(code_execution_result) = &part.code_execution_result {
        let output = code_execution_result
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or_default();
        return Ok(AnthropicMessageBlock::Text(format!(
            "Execution Result:```code_output\n{output}\n```"
        )));
    }

    Err(format!("Not supported yet: {part:?}"))
}

/// `anthropic_llm.part_to_message_block` — the module-level convenience
/// wrapper that builds a fresh [`ToolUseIdSanitizer`] per call.
pub fn part_to_message_block(part: &Part) -> Result<AnthropicMessageBlock, String> {
    part_to_message_block_with(part, &mut ToolUseIdSanitizer::new())
}

/// `anthropic_llm._content_to_message_param`. Image/PDF parts are
/// dropped with a warning (`eprintln!`) for any non-`"user"` role,
/// including a `None` role — matching the source's own `content.role !=
/// "user"` comparison exactly (a `None` role compares unequal to the
/// literal `"user"` too).
pub fn content_to_message_param_with(
    content: &Content,
    sanitizer: &mut ToolUseIdSanitizer,
) -> Result<AnthropicMessageParam, String> {
    let is_user = content.role.as_deref() == Some("user");
    let mut blocks = Vec::with_capacity(content.parts.len());
    for part in &content.parts {
        if !is_user && is_image_part(part) {
            eprintln!("Image data is not supported in Claude for assistant turns.");
            continue;
        }
        if !is_user && is_pdf_part(part) {
            eprintln!("PDF data is not supported in Claude for assistant turns.");
            continue;
        }
        blocks.push(part_to_message_block_with(part, sanitizer)?);
    }
    Ok(AnthropicMessageParam {
        role: to_claude_role(content.role.as_deref()),
        content: blocks,
    })
}

/// `anthropic_llm.content_to_message_param` — the module-level
/// convenience wrapper that builds a fresh [`ToolUseIdSanitizer`] per
/// call.
pub fn content_to_message_param(content: &Content) -> Result<AnthropicMessageParam, String> {
    content_to_message_param_with(content, &mut ToolUseIdSanitizer::new())
}

/// `anthropic_llm.content_block_to_part` — the response-parsing
/// direction, narrowed to the closed set of Anthropic content-block
/// variants [`AnthropicResponseBlock`] models (the same four the source
/// itself handles; every other `anthropic_types.ContentBlock` subtype
/// hits the source's own `raise NotImplementedError`, so there is no
/// fifth variant for this port's enum to be missing — unlike
/// [`part_to_message_block`], this direction cannot fail).
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicResponseBlock {
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: Option<String>,
        name: String,
        input: BTreeMap<String, Value>,
    },
}

/// `anthropic_llm.content_block_to_part`.
pub fn content_block_to_part(content_block: &AnthropicResponseBlock) -> Part {
    match content_block {
        AnthropicResponseBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut part = Part {
                text: Some(thinking.clone()),
                thought: Some(true),
                ..Default::default()
            };
            if let Some(signature) = signature.as_ref().filter(|s| !s.is_empty()) {
                part.thought_signature = Some(Value::String(signature.clone()));
            }
            part
        }
        AnthropicResponseBlock::RedactedThinking { data } => Part {
            thought: Some(true),
            thought_signature: Some(Value::String(data.clone())),
            ..Default::default()
        },
        AnthropicResponseBlock::Text { text } => Part::text(text.clone()),
        AnthropicResponseBlock::ToolUse { id, name, input } => {
            let mut part = Part::function_call(FunctionCall {
                name: Some(name.clone()),
                args: Some(input.clone()),
                ..Default::default()
            });
            if let Some(function_call) = part.function_call.as_mut() {
                function_call.id = id.clone();
            }
            part
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ToolUseIdSanitizer ---

    #[test]
    fn sanitize_passes_through_a_valid_id() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        assert_eq!(sanitizer.sanitize(Some("toolu_abc123")), "toolu_abc123");
    }

    #[test]
    fn sanitize_replaces_an_invalid_id_with_a_deterministic_fallback() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        assert_eq!(sanitizer.sanitize(Some("bad id!")), "toolu_fallback_0");
    }

    #[test]
    fn sanitize_maps_the_same_invalid_id_to_the_same_fallback() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(Some("bad id!"));
        let second = sanitizer.sanitize(Some("bad id!"));
        assert_eq!(first, second);
    }

    #[test]
    fn sanitize_gives_distinct_ids_distinct_fallbacks() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(Some("bad id 1"));
        let second = sanitizer.sanitize(Some("bad id 2"));
        assert_ne!(first, second);
    }

    #[test]
    fn sanitize_treats_a_missing_id_as_its_own_key() {
        let mut sanitizer = ToolUseIdSanitizer::new();
        let first = sanitizer.sanitize(None);
        let second = sanitizer.sanitize(None);
        assert_eq!(first, second);
        assert_eq!(first, "toolu_fallback_0");
    }

    // --- to_google_genai_finish_reason ---

    #[test]
    fn finish_reason_maps_none_to_none() {
        assert_eq!(to_google_genai_finish_reason(None), None);
    }

    #[test]
    fn finish_reason_maps_normal_stops_to_stop() {
        for reason in ["end_turn", "stop_sequence", "tool_use", "pause_turn"] {
            assert_eq!(
                to_google_genai_finish_reason(Some(reason)),
                Some(Value::String("STOP".to_string())),
                "reason {reason}"
            );
        }
    }

    #[test]
    fn finish_reason_maps_max_tokens() {
        assert_eq!(
            to_google_genai_finish_reason(Some("max_tokens")),
            Some(Value::String("MAX_TOKENS".to_string()))
        );
    }

    #[test]
    fn finish_reason_maps_refusal_to_safety() {
        assert_eq!(
            to_google_genai_finish_reason(Some("refusal")),
            Some(Value::String("SAFETY".to_string()))
        );
    }

    #[test]
    fn finish_reason_maps_unknown_to_unspecified() {
        assert_eq!(
            to_google_genai_finish_reason(Some("something_new")),
            Some(Value::String("FINISH_REASON_UNSPECIFIED".to_string()))
        );
    }

    // --- token usage extraction ---

    #[test]
    fn extract_prompt_token_count_sums_input_and_cache_fields() {
        let usage = AnthropicUsage {
            input_tokens: Some(10),
            cache_read_input_tokens: Some(5),
            cache_creation_input_tokens: Some(2),
            ..Default::default()
        };
        assert_eq!(extract_prompt_token_count(&usage), 17);
    }

    #[test]
    fn extract_prompt_token_count_treats_missing_fields_as_zero() {
        let usage = AnthropicUsage {
            input_tokens: Some(10),
            ..Default::default()
        };
        assert_eq!(extract_prompt_token_count(&usage), 10);
    }

    #[test]
    fn extract_cached_token_count_reads_cache_read_input_tokens() {
        let usage = AnthropicUsage {
            cache_read_input_tokens: Some(4),
            ..Default::default()
        };
        assert_eq!(extract_cached_token_count(&usage), Some(4));
    }

    #[test]
    fn extract_cache_creation_token_count_reads_cache_creation_input_tokens() {
        let usage = AnthropicUsage {
            cache_creation_input_tokens: Some(3),
            ..Default::default()
        };
        assert_eq!(extract_cache_creation_token_count(&usage), Some(3));
    }

    #[test]
    fn extract_thinking_token_count_is_none_without_thinking_tokens() {
        let usage = AnthropicUsage {
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), None);
    }

    #[test]
    fn extract_thinking_token_count_returns_the_raw_value_without_output_tokens() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(30),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(30));
    }

    #[test]
    fn extract_thinking_token_count_clamps_to_output_tokens() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(150),
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(100));
    }

    #[test]
    fn extract_thinking_token_count_does_not_clamp_when_within_bounds() {
        let usage = AnthropicUsage {
            thinking_tokens: Some(20),
            output_tokens: Some(100),
            ..Default::default()
        };
        assert_eq!(extract_thinking_token_count(&usage), Some(20));
    }

    // --- update_type_string ---

    fn object_map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    #[test]
    fn update_type_string_lowercases_a_top_level_type() {
        let mut schema = object_map(vec![("type", Value::String("OBJECT".to_string()))]);
        update_type_string(&mut schema);
        assert_eq!(
            schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
    }

    #[test]
    fn update_type_string_recurses_into_properties() {
        let mut schema = object_map(vec![(
            "properties",
            object_map(vec![(
                "x",
                object_map(vec![("type", Value::String("STRING".to_string()))]),
            )]),
        )]);
        update_type_string(&mut schema);
        let x_type = schema
            .get("properties")
            .and_then(|p| p.get("x"))
            .and_then(|x| x.get("type"));
        assert_eq!(x_type, Some(&Value::String("string".to_string())));
    }

    #[test]
    fn update_type_string_recurses_into_items_and_array_type_lists() {
        let mut schema = object_map(vec![
            (
                "items",
                object_map(vec![("type", Value::String("NUMBER".to_string()))]),
            ),
            (
                "anyOf",
                Value::Seq(vec![object_map(vec![(
                    "type",
                    Value::String("INTEGER".to_string()),
                )])]),
            ),
        ]);
        update_type_string(&mut schema);
        assert_eq!(
            schema.get("items").and_then(|i| i.get("type")),
            Some(&Value::String("number".to_string()))
        );
        let Some(Value::Seq(any_of)) = schema.get("anyOf") else {
            panic!("expected anyOf to remain a Seq");
        };
        assert_eq!(
            any_of[0].get("type"),
            Some(&Value::String("integer".to_string()))
        );
    }

    #[test]
    fn update_type_string_recurses_into_defs_and_pattern_properties() {
        let mut schema = object_map(vec![(
            "$defs",
            object_map(vec![(
                "Foo",
                object_map(vec![("type", Value::String("BOOLEAN".to_string()))]),
            )]),
        )]);
        update_type_string(&mut schema);
        let foo_type = schema
            .get("$defs")
            .and_then(|d| d.get("Foo"))
            .and_then(|f| f.get("type"));
        assert_eq!(foo_type, Some(&Value::String("boolean".to_string())));
    }

    #[test]
    fn update_type_string_ignores_non_dict_non_list_values() {
        let mut value = Value::String("OBJECT".to_string());
        update_type_string(&mut value);
        assert_eq!(value, Value::String("OBJECT".to_string()));
    }

    // --- function_declaration_to_tool_param ---

    use adk_genai::content::FunctionDeclaration;

    #[test]
    fn function_declaration_to_tool_param_prefers_parameters_json_schema() {
        let decl = FunctionDeclaration {
            name: Some("get_weather".to_string()),
            description: Some("Gets the weather".to_string()),
            parameters_json_schema: Some(object_map(vec![(
                "type",
                Value::String("OBJECT".to_string()),
            )])),
            parameters: Some(object_map(vec![(
                "type",
                Value::String("ignored".to_string()),
            )])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.name, "get_weather");
        assert_eq!(tool_param.description, "Gets the weather");
        assert_eq!(
            tool_param.input_schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
    }

    #[test]
    fn function_declaration_to_tool_param_builds_object_schema_from_parameters_when_json_schema_absent(
    ) {
        let decl = FunctionDeclaration {
            name: Some("get_weather".to_string()),
            parameters: Some(object_map(vec![
                (
                    "properties",
                    object_map(vec![(
                        "city",
                        object_map(vec![("type", Value::String("string".to_string()))]),
                    )]),
                ),
                (
                    "required",
                    Value::Seq(vec![Value::String("city".to_string())]),
                ),
            ])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(
            tool_param.input_schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
        assert_eq!(
            tool_param.input_schema.get("required"),
            Some(&Value::Seq(vec![Value::String("city".to_string())]))
        );
        assert!(tool_param
            .input_schema
            .get("properties")
            .and_then(|p| p.get("city"))
            .is_some());
    }

    #[test]
    fn function_declaration_to_tool_param_omits_required_when_empty() {
        let decl = FunctionDeclaration {
            name: Some("noop".to_string()),
            parameters: Some(object_map(vec![("required", Value::Seq(Vec::new()))])),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.input_schema.get("required"), None);
    }

    #[test]
    fn function_declaration_to_tool_param_defaults_description_to_empty_string() {
        let decl = FunctionDeclaration {
            name: Some("noop".to_string()),
            ..Default::default()
        };
        let tool_param = function_declaration_to_tool_param(&decl);
        assert_eq!(tool_param.description, "");
    }

    #[test]
    #[should_panic(expected = "function_declaration.name is required")]
    fn function_declaration_to_tool_param_panics_without_a_name() {
        let decl = FunctionDeclaration::default();
        let _ = function_declaration_to_tool_param(&decl);
    }

    // --- build_anthropic_thinking_param ---

    fn config_with_thinking_budget(budget: i64) -> crate::llm_request::GenerateContentConfigStub {
        crate::llm_request::GenerateContentConfigStub {
            thinking_config: Some(Value::Map(vec![(
                "thinkingBudget".to_string(),
                Value::Int(budget),
            )])),
            ..Default::default()
        }
    }

    #[test]
    fn build_anthropic_thinking_param_none_config_is_ok_none() {
        assert_eq!(build_anthropic_thinking_param(None), Ok(None));
    }

    #[test]
    fn build_anthropic_thinking_param_none_thinking_config_is_ok_none() {
        let config = crate::llm_request::GenerateContentConfigStub::default();
        assert_eq!(build_anthropic_thinking_param(Some(&config)), Ok(None));
    }

    #[test]
    fn build_anthropic_thinking_param_missing_budget_key_is_err() {
        let config = crate::llm_request::GenerateContentConfigStub {
            thinking_config: Some(Value::Map(Vec::new())),
            ..Default::default()
        };
        let result = build_anthropic_thinking_param(Some(&config));
        assert_eq!(
            result,
            Err(
                "thinking_budget must be set explicitly when ThinkingConfig is provided for \
                 Anthropic models. Use 0 to disable thinking, -1 for adaptive (model-chosen \
                 depth), or a positive integer (>= 1024) for manual budgeting."
                    .to_string()
            )
        );
    }

    #[test]
    fn build_anthropic_thinking_param_zero_budget_is_disabled() {
        let config = config_with_thinking_budget(0);
        assert_eq!(
            build_anthropic_thinking_param(Some(&config)),
            Ok(Some(AnthropicThinkingParam::Disabled))
        );
    }

    #[test]
    fn build_anthropic_thinking_param_negative_budget_is_adaptive() {
        let config = config_with_thinking_budget(-1);
        assert_eq!(
            build_anthropic_thinking_param(Some(&config)),
            Ok(Some(AnthropicThinkingParam::Adaptive))
        );
    }

    #[test]
    fn build_anthropic_thinking_param_positive_budget_is_enabled() {
        let config = config_with_thinking_budget(10240);
        assert_eq!(
            build_anthropic_thinking_param(Some(&config)),
            Ok(Some(AnthropicThinkingParam::Enabled {
                budget_tokens: 10240
            }))
        );
    }

    #[test]
    fn build_anthropic_thinking_param_reads_uint_budget() {
        let config = crate::llm_request::GenerateContentConfigStub {
            thinking_config: Some(Value::Map(vec![(
                "thinkingBudget".to_string(),
                Value::UInt(10240),
            )])),
            ..Default::default()
        };
        assert_eq!(
            build_anthropic_thinking_param(Some(&config)),
            Ok(Some(AnthropicThinkingParam::Enabled {
                budget_tokens: 10240
            }))
        );
    }

    // --- C0539: Content <-> Anthropic block conversion ---

    fn image_part(mime_type: &str, data: &str) -> Part {
        Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some(mime_type.to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(data.to_string()),
                )])),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn to_claude_role_maps_model_and_assistant_to_assistant() {
        assert_eq!(to_claude_role(Some("model")), "assistant");
        assert_eq!(to_claude_role(Some("assistant")), "assistant");
    }

    #[test]
    fn to_claude_role_defaults_everything_else_to_user() {
        assert_eq!(to_claude_role(Some("user")), "user");
        assert_eq!(to_claude_role(None), "user");
        assert_eq!(to_claude_role(Some("system")), "user");
    }

    #[test]
    fn part_to_message_block_converts_plain_text() {
        let block = part_to_message_block(&Part::text("hello")).unwrap();
        assert_eq!(block, AnthropicMessageBlock::Text("hello".to_string()));
    }

    #[test]
    fn part_to_message_block_converts_thinking_with_signature() {
        let part = Part {
            text: Some("reasoning...".to_string()),
            thought: Some(true),
            thought_signature: Some(Value::String("sig".to_string())),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Thinking {
                thinking: "reasoning...".to_string(),
                signature: "sig".to_string(),
            }
        );
    }

    #[test]
    fn part_to_message_block_thinking_without_signature_uses_empty_string() {
        let part = Part {
            text: Some("reasoning...".to_string()),
            thought: Some(true),
            thought_signature: None,
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Thinking {
                thinking: "reasoning...".to_string(),
                signature: String::new(),
            }
        );
    }

    #[test]
    fn part_to_message_block_converts_redacted_thinking() {
        let part = Part {
            text: None,
            thought: Some(true),
            thought_signature: Some(Value::String("encrypted-blob".to_string())),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::RedactedThinking {
                data: "encrypted-blob".to_string(),
            }
        );
    }

    #[test]
    fn part_to_message_block_converts_a_function_call_to_tool_use() {
        let part = Part::function_call(FunctionCall {
            id: Some("call_1".to_string()),
            name: Some("roll_die".to_string()),
            args: Some(BTreeMap::from([("sides".to_string(), Value::UInt(6))])),
            ..Default::default()
        });
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::ToolUse {
                id: "call_1".to_string(),
                name: "roll_die".to_string(),
                input: BTreeMap::from([("sides".to_string(), Value::UInt(6))]),
            }
        );
    }

    #[test]
    fn part_to_message_block_sanitizes_an_invalid_tool_use_id() {
        let part = Part::function_call(FunctionCall {
            id: Some("not valid!".to_string()),
            name: Some("roll_die".to_string()),
            ..Default::default()
        });
        let mut sanitizer = ToolUseIdSanitizer::new();
        let block = part_to_message_block_with(&part, &mut sanitizer).unwrap();
        match block {
            AnthropicMessageBlock::ToolUse { id, .. } => assert_eq!(id, "toolu_fallback_0"),
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "function_call.name is required")]
    fn part_to_message_block_panics_on_a_missing_function_call_name() {
        let part = Part::function_call(FunctionCall::default());
        let _ = part_to_message_block(&part);
    }

    #[test]
    fn part_to_message_block_converts_a_simple_text_function_response() {
        let part = Part {
            function_response: Some(FunctionResponse {
                id: Some("call_1".to_string()),
                response: Some(BTreeMap::from([(
                    "result".to_string(),
                    Value::String("42".to_string()),
                )])),
                ..Default::default()
            }),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: AnthropicToolResultContent::Text("42".to_string()),
                is_error: false,
            }
        );
    }

    #[test]
    fn part_to_message_block_function_response_extracts_content_list_text_items() {
        let part = Part {
            function_response: Some(FunctionResponse {
                id: Some("call_1".to_string()),
                response: Some(BTreeMap::from([(
                    "content".to_string(),
                    Value::Seq(vec![
                        Value::Map(vec![
                            ("type".to_string(), Value::String("text".to_string())),
                            ("text".to_string(), Value::String("first".to_string())),
                        ]),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("text".to_string())),
                            ("text".to_string(), Value::String("second".to_string())),
                        ]),
                    ]),
                )])),
                ..Default::default()
            }),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        match block {
            AnthropicMessageBlock::ToolResult { content, .. } => {
                assert_eq!(
                    content,
                    AnthropicToolResultContent::Text("first\nsecond".to_string())
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn part_to_message_block_function_response_appends_media_blocks() {
        let part = Part {
            function_response: Some(FunctionResponse {
                id: Some("call_1".to_string()),
                response: Some(BTreeMap::from([(
                    "result".to_string(),
                    Value::String("done".to_string()),
                )])),
                parts: Some(vec![adk_genai::content::FunctionResponsePart {
                    inline_data: Some(MediaBlobStub {
                        mime_type: Some("image/png".to_string()),
                        rest: Some(Value::Map(vec![(
                            "data".to_string(),
                            Value::String("aW1hZ2U=".to_string()),
                        )])),
                    }),
                    file_data: None,
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        match block {
            AnthropicMessageBlock::ToolResult { content, .. } => match content {
                AnthropicToolResultContent::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 2);
                    assert_eq!(
                        blocks[0],
                        AnthropicToolResultBlock::Text("done".to_string())
                    );
                    assert_eq!(
                        blocks[1],
                        AnthropicToolResultBlock::Image(AnthropicBase64Source {
                            media_type: "image/png".to_string(),
                            data: "aW1hZ2U=".to_string(),
                        })
                    );
                }
                other => panic!("expected Blocks, got {other:?}"),
            },
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn part_to_message_block_converts_an_image_part() {
        let part = image_part("image/png", "aW1hZ2U=");
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Image(AnthropicBase64Source {
                media_type: "image/png".to_string(),
                data: "aW1hZ2U=".to_string(),
            })
        );
    }

    #[test]
    fn part_to_message_block_rejects_an_unsupported_image_mime_type() {
        let part = image_part("image/bmp", "aW1hZ2U=");
        assert!(part_to_message_block(&part).is_err());
    }

    #[test]
    fn part_to_message_block_converts_a_pdf_part() {
        let part = image_part("application/pdf", "cGRm");
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Document(AnthropicBase64Source {
                media_type: "application/pdf".to_string(),
                data: "cGRm".to_string(),
            })
        );
    }

    #[test]
    fn part_to_message_block_renders_executable_code_as_fenced_text() {
        let part = Part {
            executable_code: Some(Value::Map(vec![(
                "code".to_string(),
                Value::String("print(1)".to_string()),
            )])),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Text("Code:```python\nprint(1)\n```".to_string())
        );
    }

    #[test]
    fn part_to_message_block_renders_code_execution_result_as_fenced_text() {
        let part = Part {
            code_execution_result: Some(Value::Map(vec![(
                "output".to_string(),
                Value::String("1\n".to_string()),
            )])),
            ..Default::default()
        };
        let block = part_to_message_block(&part).unwrap();
        assert_eq!(
            block,
            AnthropicMessageBlock::Text("Execution Result:```code_output\n1\n\n```".to_string())
        );
    }

    #[test]
    fn part_to_message_block_errors_for_an_empty_part() {
        let part = Part::default();
        assert!(part_to_message_block(&part).is_err());
    }

    #[test]
    fn content_to_message_param_maps_role_and_converts_every_part() {
        let content = Content::new("user", vec![Part::text("hi"), Part::text("there")]);
        let message = content_to_message_param(&content).unwrap();
        assert_eq!(message.role, "user");
        assert_eq!(
            message.content,
            vec![
                AnthropicMessageBlock::Text("hi".to_string()),
                AnthropicMessageBlock::Text("there".to_string())
            ]
        );
    }

    #[test]
    fn content_to_message_param_drops_image_parts_on_non_user_turns() {
        let content = Content::new(
            "model",
            vec![
                Part::text("here's an image"),
                image_part("image/png", "aW1hZ2U="),
            ],
        );
        let message = content_to_message_param(&content).unwrap();
        assert_eq!(message.role, "assistant");
        assert_eq!(
            message.content,
            vec![AnthropicMessageBlock::Text("here's an image".to_string())]
        );
    }

    #[test]
    fn content_to_message_param_drops_pdf_parts_on_non_user_turns() {
        let content = Content::new("model", vec![image_part("application/pdf", "cGRm")]);
        let message = content_to_message_param(&content).unwrap();
        assert!(message.content.is_empty());
    }

    #[test]
    fn content_to_message_param_keeps_image_parts_on_user_turns() {
        let content = Content::new("user", vec![image_part("image/png", "aW1hZ2U=")]);
        let message = content_to_message_param(&content).unwrap();
        assert_eq!(message.content.len(), 1);
    }

    #[test]
    fn content_to_message_param_treats_a_missing_role_as_non_user() {
        let content = Content {
            role: None,
            parts: vec![image_part("image/png", "aW1hZ2U=")],
        };
        let message = content_to_message_param(&content).unwrap();
        assert_eq!(message.role, "user");
        assert!(message.content.is_empty());
    }

    #[test]
    fn content_block_to_part_converts_a_text_block() {
        let part = content_block_to_part(&AnthropicResponseBlock::Text {
            text: "hi".to_string(),
        });
        assert_eq!(part.text.as_deref(), Some("hi"));
    }

    #[test]
    fn content_block_to_part_converts_a_thinking_block() {
        let part = content_block_to_part(&AnthropicResponseBlock::Thinking {
            thinking: "reasoning...".to_string(),
            signature: Some("sig".to_string()),
        });
        assert_eq!(part.text.as_deref(), Some("reasoning..."));
        assert_eq!(part.thought, Some(true));
        assert_eq!(
            part.thought_signature,
            Some(Value::String("sig".to_string()))
        );
    }

    #[test]
    fn content_block_to_part_thinking_without_signature_omits_it() {
        let part = content_block_to_part(&AnthropicResponseBlock::Thinking {
            thinking: "reasoning...".to_string(),
            signature: None,
        });
        assert_eq!(part.thought_signature, None);
    }

    #[test]
    fn content_block_to_part_converts_a_redacted_thinking_block() {
        let part = content_block_to_part(&AnthropicResponseBlock::RedactedThinking {
            data: "encrypted-blob".to_string(),
        });
        assert_eq!(part.thought, Some(true));
        assert_eq!(
            part.thought_signature,
            Some(Value::String("encrypted-blob".to_string()))
        );
        assert_eq!(part.text, None);
    }

    #[test]
    fn content_block_to_part_converts_a_tool_use_block() {
        let part = content_block_to_part(&AnthropicResponseBlock::ToolUse {
            id: Some("toolu_1".to_string()),
            name: "roll_die".to_string(),
            input: BTreeMap::from([("sides".to_string(), Value::UInt(6))]),
        });
        let function_call = part.function_call.unwrap();
        assert_eq!(function_call.id.as_deref(), Some("toolu_1"));
        assert_eq!(function_call.name.as_deref(), Some("roll_die"));
        assert_eq!(
            function_call.args,
            Some(BTreeMap::from([("sides".to_string(), Value::UInt(6))]))
        );
    }

    #[test]
    fn part_and_response_block_round_trip_thinking() {
        let original = Part {
            text: Some("reasoning...".to_string()),
            thought: Some(true),
            thought_signature: Some(Value::String("sig".to_string())),
            ..Default::default()
        };
        let block = part_to_message_block(&original).unwrap();
        let (thinking, signature) = match block {
            AnthropicMessageBlock::Thinking {
                thinking,
                signature,
            } => (thinking, signature),
            other => panic!("expected Thinking, got {other:?}"),
        };
        let round_tripped = content_block_to_part(&AnthropicResponseBlock::Thinking {
            thinking,
            signature: Some(signature),
        });
        assert_eq!(round_tripped, original);
    }
}
