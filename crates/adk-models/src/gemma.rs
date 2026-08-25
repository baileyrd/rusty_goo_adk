//! C0113/C0545/C0546/C0548: `models.gemma_llm`, ported from
//! `google.adk.models.gemma_llm`.
//!
//! [`Gemma`] adds Gemma-3-only function-calling/system-instruction
//! workarounds around [`crate::gemini::Gemini`] — Gemma 3 has no native
//! function calling or system-instruction support, so tool declarations
//! are injected as a strict-JSON text block and function calls are
//! parsed back out of the model's text response. Gemma 4+ needs none of
//! this and resolves natively to plain [`Gemini`] instead (see C0113
//! below).
//!
//! **Composition instead of the source's mixin inheritance**:
//! `GemmaFunctionCallingMixin`/`Gemma(GemmaFunctionCallingMixin, Gemini)`
//! is Python multiple inheritance — Rust has no equivalent. [`Gemma`]
//! instead holds a [`Gemini`] instance and implements [`crate::base_llm::BaseLlm`]
//! by composition: clone-then-mutate the [`LlmRequest`] before
//! delegating to the inner `Gemini::generate_content_async`, then
//! post-process each yielded [`LlmResponse`] — the same clone-then-mutate
//! idiom `functions_media.rs`'s C0195 already established for a similar
//! request/response transform.
//!
//! **C0113, resolved as a side effect**: the source relies on Python's
//! `LLMRegistry` matching `Gemini.supported_models()`'s `gemma-4.*`
//! pattern (registered first) ahead of `Gemma.supported_models()`'s
//! `gemma-.*` (registered later) so a `gemma-4-*` model name resolves
//! natively to `Gemini` while `gemma-3-*` falls through to `Gemma`. This
//! port's [`crate::registry::LlmRegistry::resolve`] already matches
//! entries in registration order (first match wins) — registering
//! `Gemma` in [`crate::registry::default_registry`] *after* the
//! pre-existing `Gemini` registration reproduces the exact same
//! precedence with no new mechanism needed.
//!
//! **`_api_backend`, disclosed narrowing**: the source's `Gemma._api_backend`
//! is a `cached_property` override that forces `GoogleLLMVariant.GEMINI_API`
//! for every consumer of `self._api_backend` on that instance. This
//! port's [`Gemini::api_backend`] is a plain inherent method deriving
//! from the process-global [`crate::capabilities::get_google_llm_variant`]
//! — it isn't virtual/overridable through composition, so [`Gemma`]'s own
//! [`Gemma::api_backend`] (added for read-parity) has no effect on the
//! inner `Gemini`'s actual auth/cache-manager backend selection inside
//! `Gemini::generate_content`. In practice this rarely matters: Vertex AI
//! already rejects Gemma-shaped model names before this port would ever
//! reach that code path (`Gemini::resolve_auth_header`'s existing
//! `VertexAiAuthNotSupported`-style gating). A real per-instance override
//! would need a new field on `Gemini` itself — left as a disclosed gap
//! rather than modifying that already-shipped struct for this batch.
//!
//! **`Gemma3Ollama`, out of scope**: the source's `Gemma3Ollama(GemmaFunctionCallingMixin,
//! LiteLlm)` (conditionally defined only when a `LiteLlm` backend is
//! importable) needs `LiteLlm` (C0557) itself, which isn't built in this
//! port at all — a much larger, separate capability. `ollama.rs`'s own
//! module doc already discloses this exact carve-out (its `ollama/(?!gemma3).*`
//! negative-lookahead pattern), so this gap was anticipated, not newly
//! discovered here.
//!
//! **`_get_last_valid_json_substring`, adapted**: the source uses
//! `json.JSONDecoder().raw_decode`, which parses a JSON value starting at
//! a given offset and reports how many characters it consumed — no
//! direct equivalent is exposed by this port's `rusty_serde::json`. This
//! port instead brace-matches a candidate JSON-object substring (quote-
//! and escape-aware) starting at each `{`, then validates it with a full
//! `rusty_serde::json::from_str::<Value>` parse — functionally
//! equivalent for well-formed JSON (both always parse a complete,
//! balanced object starting at that `{`), and both advance past a failed
//! candidate by exactly one character and keep the *last* valid match
//! found, matching the source's own search order exactly.
//!
//! **`exclude_none=True`, not replicated**: `func.model_dump_json(exclude_none=True)`/
//! `func_call.model_dump_json(exclude_none=True)` omit unset optional
//! fields from the JSON text; this port's `rusty_serde::json::to_string`
//! serializes every field, including `null` ones. Same non-stripping
//! behavior `adk-tools::append_tools::merge_declarations` already
//! establishes for the identical `FunctionDeclaration` type — a cosmetic
//! JSON-shape difference the model reads past either way, not a new
//! narrowing this batch introduces.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use regex::Regex;
use rusty_serde::value::Value;
use rusty_serde::Deserialize;

use crate::base_llm::{BaseLlm, BaseLlmError, BoxFuture};
use crate::capabilities::{GoogleLlmVariant, LlmCapabilities};
use crate::gemini::Gemini;
use crate::llm_request::{Instructions, LlmRequest};
use crate::llm_response::LlmResponse;
use adk_genai::content::{Content, FunctionCall, FunctionDeclaration, Part};

const DEFAULT_GEMMA_MODEL: &str = "gemma-3-27b-it";
const FUNCTION_DECLARATIONS_KEY: &str = "functionDeclarations";

/// `gemma_llm.GemmaFunctionCallModel` — flexible parser accepting
/// `name`/`function` and `parameters`/`args` aliases for a self-serialized
/// function-call JSON blob (C0548).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GemmaFunctionCallModel {
    #[rusty_serde(alias = "function")]
    pub name: String,
    #[rusty_serde(alias = "args")]
    pub parameters: BTreeMap<String, Value>,
}

/// C0546: `gemma_llm.Gemma` — Gemma-3-only integration composing
/// [`Gemini`]. See the module doc for the composition adaptation and the
/// `_api_backend` narrowing.
pub struct Gemma {
    pub gemini: Gemini,
}

impl Default for Gemma {
    fn default() -> Self {
        Self::new(DEFAULT_GEMMA_MODEL)
    }
}

impl Gemma {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            gemini: Gemini::new(model),
        }
    }

    /// Read-parity only — see the module doc's `_api_backend` narrowing:
    /// this has no effect on the inner `Gemini`'s own actual backend
    /// selection.
    pub fn api_backend(&self) -> GoogleLlmVariant {
        GoogleLlmVariant::GeminiApi
    }

    /// `Gemma._preprocess_request`: moves function declarations into a
    /// system-instruction text block, then prepends any remaining system
    /// instruction as a user-role message. Ported exactly, including the
    /// source's own apparent edge case: when `llm_request.contents` is
    /// empty, the instruction is dropped rather than becoming the first
    /// content (the source's `if contents:` guard skips the prepend
    /// entirely in that case, yet still unconditionally clears
    /// `config.system_instruction` afterward).
    fn preprocess_request(&self, llm_request: &mut LlmRequest) {
        move_function_calls_into_system_instruction(llm_request);

        if let Some(system_instruction) = llm_request.config.system_instruction.clone() {
            if !system_instruction.is_empty() {
                let instruction_content = Content::new(
                    "user",
                    vec![adk_genai::content::Part::text(system_instruction)],
                );
                if !llm_request.contents.is_empty()
                    && llm_request.contents[0] != instruction_content
                {
                    llm_request.contents.insert(0, instruction_content);
                }
            }
            llm_request.config.system_instruction = None;
        }
    }
}

impl BaseLlm for Gemma {
    fn model(&self) -> &str {
        self.gemini.model()
    }

    fn type_name(&self) -> &'static str {
        "Gemma"
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.gemini.capabilities()
    }

    /// C0546: `Gemma.generate_content_async` — asserts the requested
    /// model is a Gemma model, preprocesses the request (moving function
    /// declarations into a system-instruction text block), delegates to
    /// the inner `Gemini`, then extracts any function call the model
    /// self-serialized back out of each response's text.
    ///
    /// **Adaptation**: the source's bare `assert` (an uncaught
    /// `AssertionError` on failure) becomes a real `Err(BaseLlmError::
    /// CallFailed(...))` here — this port's established convention for a
    /// runtime-reachable invariant violation (e.g.
    /// `BuiltInCodeExecutor::process_llm_request`'s non-Gemini-model
    /// error), not a panic in a shared library.
    fn generate_content_async<'a>(
        &'a self,
        llm_request: &'a LlmRequest,
        stream: bool,
    ) -> BoxFuture<'a, Result<Vec<LlmResponse>, BaseLlmError>> {
        Box::pin(async move {
            let model = llm_request.model.as_deref();
            if !model.is_some_and(|m| m.starts_with("gemma-")) {
                return Err(BaseLlmError::CallFailed(format!(
                    "Requesting a non-Gemma model ({model:?}) with the Gemma LLM is not \
                     supported."
                )));
            }

            let mut request = llm_request.clone();
            self.preprocess_request(&mut request);

            let mut responses = self.gemini.generate_content_async(&request, stream).await?;
            for response in &mut responses {
                extract_function_calls_from_response(response);
            }
            Ok(responses)
        })
    }

    /// C0546: `Gemma.supported_models`.
    fn supported_models() -> Vec<&'static str>
    where
        Self: Sized,
    {
        vec!["gemma-.*"]
    }
}

/// `GemmaFunctionCallingMixin._move_function_calls_into_system_instruction`.
fn move_function_calls_into_system_instruction(llm_request: &mut LlmRequest) {
    let mut new_contents = Vec::with_capacity(llm_request.contents.len());
    for content_item in &llm_request.contents {
        let (new_parts, has_function_response, has_function_call) =
            convert_content_parts_for_gemma(content_item);
        if has_function_response {
            if !new_parts.is_empty() {
                new_contents.push(Content::new("user", new_parts));
            }
        } else if has_function_call {
            if !new_parts.is_empty() {
                new_contents.push(Content::new("model", new_parts));
            }
        } else {
            new_contents.push(content_item.clone());
        }
    }
    llm_request.contents = new_contents;

    let tools_is_empty =
        !matches!(&llm_request.config.tools, Some(Value::Seq(entries)) if !entries.is_empty());
    if tools_is_empty {
        return;
    }

    let mut all_function_declarations: Vec<FunctionDeclaration> = Vec::new();
    if let Some(Value::Seq(entries)) = &llm_request.config.tools {
        for entry in entries {
            let Value::Map(fields) = entry else {
                continue;
            };
            let Some((_, Value::Seq(declarations))) = fields
                .iter()
                .find(|(key, _)| key == FUNCTION_DECLARATIONS_KEY)
            else {
                continue;
            };
            for declaration_value in declarations {
                if let Ok(declaration) =
                    rusty_serde::json::from_value::<FunctionDeclaration>(declaration_value.clone())
                {
                    all_function_declarations.push(declaration);
                }
            }
        }
    }

    if !all_function_declarations.is_empty() {
        let system_instruction =
            build_gemma_function_system_instruction(&all_function_declarations);
        llm_request.append_instructions(Instructions::Strings(vec![system_instruction]));
    }

    llm_request.config.tools = Some(Value::Seq(Vec::new()));
}

/// `GemmaFunctionCallingMixin._extract_function_calls_from_response`.
fn extract_function_calls_from_response(llm_response: &mut LlmResponse) {
    if llm_response.partial == Some(true) || llm_response.turn_complete == Some(true) {
        return;
    }
    let Some(content) = &mut llm_response.content else {
        return;
    };
    if content.parts.is_empty() || content.parts.len() > 1 {
        return;
    }
    let Some(response_text) = content.parts[0].text.clone().filter(|t| !t.is_empty()) else {
        return;
    };

    let Some(json_candidate) = extract_json_candidate(&response_text) else {
        return;
    };

    if let Ok(parsed) = rusty_serde::json::from_str::<GemmaFunctionCallModel>(&json_candidate) {
        let function_call = FunctionCall {
            name: Some(parsed.name),
            args: Some(parsed.parameters),
            ..Default::default()
        };
        content.parts = vec![Part::function_call(function_call)];
    }
    // Malformed JSON, or JSON that doesn't validate against
    // `GemmaFunctionCallModel` (missing `name`/`parameters`): leave the
    // response as plain text, matching the source's caught
    // `JSONDecodeError`/`ValidationError`.
}

fn extract_json_candidate(response_text: &str) -> Option<String> {
    static FENCE_RE: OnceLock<Regex> = OnceLock::new();
    let fence_re =
        FENCE_RE.get_or_init(|| Regex::new(r"(?s)```(?:json|tool_code)?\s*(.*?)\s*```").unwrap());
    if let Some(captures) = fence_re.captures(response_text) {
        return Some(captures[1].to_string());
    }
    get_last_valid_json_substring(response_text)
}

/// `_get_last_valid_json_substring` — see the module doc for the
/// brace-matching adaptation.
fn get_last_valid_json_substring(text: &str) -> Option<String> {
    let mut last_valid: Option<String> = None;
    let mut search_start = 0usize;
    while let Some(relative_open) = text[search_start..].find('{') {
        let open_index = search_start + relative_open;
        if let Some(close_index) = find_matching_brace(text, open_index) {
            let candidate = &text[open_index..=close_index];
            if rusty_serde::json::from_str::<Value>(candidate).is_ok() {
                last_valid = Some(candidate.to_string());
                search_start = close_index + 1;
                continue;
            }
        }
        search_start = open_index + 1;
    }
    last_valid
}

/// Quote-and-escape-aware brace matcher: returns the byte index of the
/// `}` that closes the `{` at `open_index`, or `None` if unbalanced.
fn find_matching_brace(text: &str, open_index: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, character) in text.char_indices() {
        if index < open_index {
            continue;
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// `_convert_content_parts_for_gemma`.
fn convert_content_parts_for_gemma(content_item: &Content) -> (Vec<Part>, bool, bool) {
    let mut new_parts = Vec::with_capacity(content_item.parts.len());
    let mut has_function_response_part = false;
    let mut has_function_call_part = false;

    for part in &content_item.parts {
        if let Some(function_response) = &part.function_response {
            has_function_response_part = true;
            let response_json = function_response
                .response
                .as_ref()
                .map(|response| {
                    let value = Value::Map(
                        response
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    );
                    rusty_serde::json::to_string(&value).unwrap_or_default()
                })
                .unwrap_or_else(|| "null".to_string());
            let response_text = format!(
                "Invoking tool `{}` produced: `{response_json}`.",
                function_response.name.as_deref().unwrap_or_default()
            );
            new_parts.push(Part::text(response_text));
        } else if let Some(function_call) = &part.function_call {
            has_function_call_part = true;
            let json = rusty_serde::json::to_string(function_call).unwrap_or_default();
            new_parts.push(Part::text(json));
        } else {
            new_parts.push(part.clone());
        }
    }
    (
        new_parts,
        has_function_response_part,
        has_function_call_part,
    )
}

/// `_build_gemma_function_system_instruction`.
fn build_gemma_function_system_instruction(
    function_declarations: &[FunctionDeclaration],
) -> String {
    if function_declarations.is_empty() {
        return String::new();
    }

    let instruction_parts: Vec<String> = function_declarations
        .iter()
        .map(|declaration| rusty_serde::json::to_string(declaration).unwrap_or_default())
        .collect();

    let mut system_instruction = format!(
        "You have access to the following functions:\n[{}\n]\n",
        instruction_parts.join(",\n")
    );
    system_instruction.push_str(
        "When you call a function, you MUST respond in the format of: \
         {\"name\": function name, \"parameters\": dictionary of argument name and its value}\n\
         When you call a function, you MUST NOT include any other text in the response.\n",
    );
    system_instruction
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::FunctionResponse;

    // --- Gemma / BaseLlm surface ---

    #[test]
    fn default_model_is_gemma_3_27b() {
        assert_eq!(Gemma::default().model(), DEFAULT_GEMMA_MODEL);
    }

    #[test]
    fn type_name_is_gemma() {
        assert_eq!(Gemma::new("gemma-3-27b-it").type_name(), "Gemma");
    }

    #[test]
    fn supported_models_matches_the_source() {
        assert_eq!(Gemma::supported_models(), vec!["gemma-.*"]);
    }

    #[test]
    fn api_backend_is_always_gemini_api() {
        assert_eq!(
            Gemma::new("gemma-3-27b-it").api_backend(),
            GoogleLlmVariant::GeminiApi
        );
    }

    #[rusty_tokio::test]
    async fn generate_content_async_errors_for_a_non_gemma_model() {
        let gemma = Gemma::new("gemma-3-27b-it");
        let request = LlmRequest::new("gemini-2.5-flash");
        let error = gemma
            .generate_content_async(&request, false)
            .await
            .unwrap_err();
        assert!(matches!(error, BaseLlmError::CallFailed(_)));
    }

    // --- move_function_calls_into_system_instruction ---

    fn function_declaration_tools(declarations: Vec<FunctionDeclaration>) -> Value {
        let declaration_values = declarations
            .iter()
            .map(|d| rusty_serde::json::to_value(d).unwrap())
            .collect();
        Value::Seq(vec![Value::Map(vec![(
            FUNCTION_DECLARATIONS_KEY.to_string(),
            Value::Seq(declaration_values),
        )])])
    }

    #[test]
    fn moves_a_function_call_and_response_into_text_and_clears_tools() {
        let mut request = LlmRequest::new("gemma-3-27b-it");
        request.contents = vec![
            Content::new("user", vec![Part::text("roll a die")]),
            Content::new(
                "model",
                vec![Part::function_call(FunctionCall {
                    name: Some("roll_die".to_string()),
                    args: Some(BTreeMap::from([("sides".to_string(), Value::UInt(6))])),
                    ..Default::default()
                })],
            ),
            Content::new(
                "user",
                vec![Part::function_response(FunctionResponse {
                    name: Some("roll_die".to_string()),
                    response: Some(BTreeMap::from([("result".to_string(), Value::UInt(4))])),
                    ..Default::default()
                })],
            ),
        ];
        request.config.tools = Some(function_declaration_tools(vec![FunctionDeclaration {
            name: Some("roll_die".to_string()),
            description: Some("rolls a die".to_string()),
            ..Default::default()
        }]));

        move_function_calls_into_system_instruction(&mut request);

        assert_eq!(request.contents.len(), 3);
        assert_eq!(request.contents[1].role.as_deref(), Some("model"));
        assert!(request.contents[1].parts[0]
            .text
            .as_deref()
            .unwrap()
            .contains("roll_die"));
        assert_eq!(request.contents[2].role.as_deref(), Some("user"));
        assert!(request.contents[2].parts[0]
            .text
            .as_deref()
            .unwrap()
            .contains("Invoking tool `roll_die` produced:"));
        assert_eq!(request.config.tools, Some(Value::Seq(Vec::new())));

        let system_instruction = request.config.system_instruction.as_deref().unwrap();
        assert!(system_instruction.contains("You have access to the following functions"));
        assert!(system_instruction.contains("roll_die"));
    }

    #[test]
    fn leaves_contents_and_tools_alone_without_any_tools() {
        let mut request = LlmRequest::new("gemma-3-27b-it");
        request.contents = vec![Content::new("user", vec![Part::text("hi")])];
        move_function_calls_into_system_instruction(&mut request);
        assert_eq!(request.contents.len(), 1);
        assert_eq!(request.config.tools, None);
        assert_eq!(request.config.system_instruction, None);
    }

    // --- Gemma::preprocess_request (via the private method, same-module test) ---

    #[test]
    fn preprocess_prepends_the_system_instruction_once() {
        let gemma = Gemma::new("gemma-3-27b-it");
        let mut request = LlmRequest::new("gemma-3-27b-it");
        request.contents = vec![Content::new("user", vec![Part::text("hi")])];
        request.config.system_instruction = Some("be helpful".to_string());

        gemma.preprocess_request(&mut request);

        assert_eq!(request.contents.len(), 2);
        assert_eq!(
            request.contents[0],
            Content::new("user", vec![Part::text("be helpful")])
        );
        assert_eq!(request.config.system_instruction, None);

        // A second call with the instruction already gone is a no-op —
        // no duplicate prepend.
        gemma.preprocess_request(&mut request);
        assert_eq!(request.contents.len(), 2);
    }

    #[test]
    fn preprocess_drops_the_system_instruction_when_contents_is_empty() {
        // Matches the source's own apparent edge case — see this
        // module's doc.
        let gemma = Gemma::new("gemma-3-27b-it");
        let mut request = LlmRequest::new("gemma-3-27b-it");
        request.config.system_instruction = Some("be helpful".to_string());

        gemma.preprocess_request(&mut request);

        assert!(request.contents.is_empty());
        assert_eq!(request.config.system_instruction, None);
    }

    // --- extract_function_calls_from_response ---

    fn response_with_text(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new("model", vec![Part::text(text)])),
            ..Default::default()
        }
    }

    #[test]
    fn extracts_a_function_call_from_a_fenced_json_block() {
        let mut response = response_with_text(
            "```json\n{\"name\": \"roll_die\", \"parameters\": {\"sides\": 6}}\n```",
        );
        extract_function_calls_from_response(&mut response);
        let part = &response.content.unwrap().parts[0];
        let call = part.function_call.as_ref().unwrap();
        assert_eq!(call.name.as_deref(), Some("roll_die"));
        assert_eq!(
            call.args.as_ref().unwrap().get("sides"),
            Some(&Value::Int(6))
        );
    }

    #[test]
    fn extracts_a_function_call_using_the_function_and_args_aliases() {
        let mut response =
            response_with_text("{\"function\": \"roll_die\", \"args\": {\"sides\": 6}}");
        extract_function_calls_from_response(&mut response);
        let part = &response.content.unwrap().parts[0];
        assert_eq!(
            part.function_call.as_ref().unwrap().name.as_deref(),
            Some("roll_die")
        );
    }

    #[test]
    fn extracts_the_last_valid_json_object_when_no_fence_is_present() {
        let mut response = response_with_text(
            "Sure, here you go: {\"name\": \"roll_die\", \"parameters\": {\"sides\": 6}}",
        );
        extract_function_calls_from_response(&mut response);
        let part = &response.content.unwrap().parts[0];
        assert_eq!(
            part.function_call.as_ref().unwrap().name.as_deref(),
            Some("roll_die")
        );
    }

    #[test]
    fn leaves_plain_text_untouched_when_no_json_is_present() {
        let mut response = response_with_text("just a normal reply, no tool call here");
        extract_function_calls_from_response(&mut response);
        assert_eq!(
            response.content.unwrap().parts[0].text.as_deref(),
            Some("just a normal reply, no tool call here")
        );
    }

    #[test]
    fn leaves_text_untouched_when_the_json_does_not_match_the_model() {
        let mut response = response_with_text("{\"foo\": \"bar\"}");
        extract_function_calls_from_response(&mut response);
        assert!(response.content.unwrap().parts[0].function_call.is_none());
    }

    #[test]
    fn skips_a_partial_response() {
        let mut response = response_with_text("{\"name\": \"x\", \"parameters\": {}}");
        response.partial = Some(true);
        extract_function_calls_from_response(&mut response);
        assert!(response.content.unwrap().parts[0].function_call.is_none());
    }

    #[test]
    fn skips_a_turn_complete_response() {
        let mut response = response_with_text("{\"name\": \"x\", \"parameters\": {}}");
        response.turn_complete = Some(true);
        extract_function_calls_from_response(&mut response);
        assert!(response.content.unwrap().parts[0].function_call.is_none());
    }

    #[test]
    fn skips_a_response_with_more_than_one_part() {
        let mut response = LlmResponse {
            content: Some(Content::new(
                "model",
                vec![
                    Part::text("{\"name\": \"x\", \"parameters\": {}}"),
                    Part::text("extra part"),
                ],
            )),
            ..Default::default()
        };
        extract_function_calls_from_response(&mut response);
        assert!(response.content.unwrap().parts[0].function_call.is_none());
    }

    // --- get_last_valid_json_substring ---

    #[test]
    fn finds_the_last_of_two_json_objects() {
        let text = r#"first {"a": 1} then {"b": 2}"#;
        assert_eq!(
            get_last_valid_json_substring(text),
            Some(r#"{"b": 2}"#.to_string())
        );
    }

    #[test]
    fn returns_none_when_no_json_object_is_present() {
        assert_eq!(get_last_valid_json_substring("no braces here"), None);
    }

    #[test]
    fn handles_a_brace_inside_a_quoted_string() {
        let text = r#"{"text": "a { inside a string"}"#;
        assert_eq!(get_last_valid_json_substring(text), Some(text.to_string()));
    }

    // --- GemmaFunctionCallModel ---

    #[test]
    fn parses_the_primary_field_names() {
        let parsed: GemmaFunctionCallModel =
            rusty_serde::json::from_str(r#"{"name": "roll_die", "parameters": {"sides": 6}}"#)
                .unwrap();
        assert_eq!(parsed.name, "roll_die");
    }

    #[test]
    fn parses_the_alias_field_names() {
        let parsed: GemmaFunctionCallModel =
            rusty_serde::json::from_str(r#"{"function": "roll_die", "args": {"sides": 6}}"#)
                .unwrap();
        assert_eq!(parsed.name, "roll_die");
        assert_eq!(parsed.parameters.get("sides"), Some(&Value::Int(6)));
    }

    // --- registry precedence (C0113) ---

    #[test]
    fn a_gemma_4_model_resolves_natively_to_gemini() {
        let registry = crate::registry::default_registry().read().unwrap();
        let llm = registry.new_llm("gemma-4-27b").unwrap();
        assert_eq!(llm.type_name(), "Gemini");
    }

    #[test]
    fn a_gemma_3_model_resolves_to_gemma() {
        let registry = crate::registry::default_registry().read().unwrap();
        let llm = registry.new_llm("gemma-3-27b-it").unwrap();
        assert_eq!(llm.type_name(), "Gemma");
    }
}
