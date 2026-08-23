//! `OllamaLlm` — a real, directly-testable `BaseLlm` backend talking to a
//! local Ollama server, added at the user's explicit request alongside
//! (not instead of) the Gemini backend work.
//!
//! **Scope, disclosed**: this is deliberately *not* a port of the source's
//! `LiteLlm(BaseLlm)` class (`models/lite_llm.py`, manifest C0557 —
//! Phase 10, still `REQUIRED`). That class is a universal wrapper around
//! the third-party `litellm` Python package, which itself talks to 14
//! different providers (OpenAI, Azure, Anthropic, Bedrock, Ollama, …) each
//! with its own quirks (C0557-C0574, ~18 manifest rows). Porting all of
//! that — starting with depending on a Rust equivalent of `litellm` that
//! doesn't exist — is far larger than "connect to Ollama" and stays future
//! work. What's built here instead: a minimal backend that talks directly
//! to Ollama's own native `/api/chat` HTTP endpoint (well-documented,
//! stable, and — unlike the Gemini Live API — actually runnable and
//! testable against a real local server), covering:
//!   - Model registration for the `ollama/…`/`ollama_chat/…` prefixes
//!     (2 of the 14 provider regexes in C0560), excluding `ollama/gemma3.*`
//!     to match the source's own carve-out for `Gemma3Ollama`'s
//!     function-calling mixin (C0547, not implemented here).
//!   - The `ollama_chat` content-flattening quirk from C0567
//!     (`_flatten_ollama_content`): Ollama's chat endpoint rejects
//!     multi-part `content` when it's text-only, so text parts are joined
//!     with newlines.
//!   - Non-streaming request/response translation and a real HTTP call.
//!
//! Left out, matching scope boundaries already established elsewhere in
//! this migration: tool/function-calling (needs `BaseTool`, deferred the
//! same way as `LlmRequest.append_tools`, C0116), streaming, and every
//! non-Ollama LiteLLM provider. None of the C0540-C0587 manifest rows are
//! marked `DONE` by this work — they describe substantially more than
//! what's built here.
//!
//! No live Ollama server was reachable in the sandbox this was written in
//! (`curl http://localhost:11434` refused), so tests run against a local
//! HTTP test server speaking Ollama's documented response shape — the
//! same dependency-free pattern used for the Gemini REST/WS transports.
//! Point `OllamaLlm` at a real local Ollama instance (the default
//! `http://localhost:11434`, or `OLLAMA_HOST`) to exercise it for real.

use adk_genai::content::Content;
use rusty_serde::{Deserialize, Serialize};

use crate::base_llm::BaseLlm;
use crate::capabilities::LlmCapabilities;
use crate::llm_request::LlmRequest;
use crate::llm_response::LlmResponse;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, rusty_err::Error)]
pub enum OllamaCallError {
    #[error("Ollama requests require a model name.")]
    MissingModel,
    #[error("request to Ollama failed: {0}")]
    Transport(String),
    #[error("Ollama returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("failed to parse the Ollama response: {0}")]
    Parse(String),
}

/// Which of the two registered prefixes selected this model — controls
/// the C0567 content-flattening quirk (`ollama_chat` only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaProvider {
    /// `ollama/<model>` — LiteLLM's plain Ollama provider.
    Ollama,
    /// `ollama_chat/<model>` — LiteLLM's chat-flavored Ollama provider;
    /// the one C0567's flattening quirk actually applies to.
    OllamaChat,
}

/// `models.lite_llm.Message` (the load-bearing subset: role + flattened
/// text content — no tool calls, no multi-part media; see the module
/// doc's scope note).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
struct OllamaChatResponse {
    #[rusty_serde(default)]
    message: Option<OllamaMessage>,
    #[rusty_serde(default)]
    done: Option<bool>,
    #[rusty_serde(default)]
    prompt_eval_count: Option<i64>,
    #[rusty_serde(default)]
    eval_count: Option<i64>,
}

/// Parses a `models.lite_llm`-style `provider/model` string, matching the
/// two prefixes this module registers.
fn parse_provider(model: &str) -> Option<(OllamaProvider, &str)> {
    if let Some(rest) = model.strip_prefix("ollama_chat/") {
        return Some((OllamaProvider::OllamaChat, rest));
    }
    if let Some(rest) = model.strip_prefix("ollama/") {
        if rest.starts_with("gemma3") {
            // Matches the source's `ollama/(?!gemma3).*` carve-out for
            // `Gemma3Ollama` — not implemented here, see the module doc.
            return None;
        }
        return Some((OllamaProvider::Ollama, rest));
    }
    None
}

/// C0567's `_flatten_ollama_content`, narrowed to this module's text-only
/// scope: join every text part with a newline. Only applied for the
/// `ollama_chat` provider, matching the source's own gating.
fn flatten_text_parts(content: &Content) -> String {
    content
        .parts
        .iter()
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Maps an ADK `Content` role to Ollama/OpenAI-convention chat roles.
fn map_role(role: Option<&str>) -> String {
    match role {
        Some("model") => "assistant".to_string(),
        Some(other) => other.to_string(),
        None => "user".to_string(),
    }
}

/// Builds the Ollama chat message list from an `LlmRequest`: a leading
/// system message (if `config.system_instruction` is set), then one
/// message per `Content`, text-flattened.
fn build_messages(request: &LlmRequest) -> Vec<OllamaMessage> {
    let mut messages = Vec::new();
    if let Some(system_instruction) = &request.config.system_instruction {
        messages.push(OllamaMessage {
            role: "system".to_string(),
            content: system_instruction.clone(),
        });
    }
    for content in &request.contents {
        messages.push(OllamaMessage {
            role: map_role(content.role.as_deref()),
            content: flatten_text_parts(content),
        });
    }
    messages
}

/// A real backend for a local Ollama server. See the module doc for
/// exactly what's implemented and what's deferred.
pub struct OllamaLlm {
    pub model: String,
    pub base_url: Option<String>,
}

impl OllamaLlm {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: None,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    fn resolved_base_url(&self) -> String {
        self.base_url
            .clone()
            .or_else(|| std::env::var("OLLAMA_HOST").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
    }

    /// C0567 (partial — see the module doc): sends a real, non-streaming
    /// `/api/chat` request and maps the response into an `LlmResponse`.
    pub async fn generate_content(
        &self,
        llm_request: &LlmRequest,
    ) -> Result<LlmResponse, OllamaCallError> {
        let model = llm_request
            .model
            .clone()
            .or_else(|| Some(self.model.clone()))
            .filter(|m| !m.is_empty())
            .ok_or(OllamaCallError::MissingModel)?;
        // `build_messages` always flattens to plain-string content (see
        // the module doc's scope note — no multi-part media content is
        // modeled yet), which already satisfies C0567's flattening
        // requirement for both provider prefixes; only the model-tag
        // parsing differs between them.
        let (_provider, ollama_model_tag) =
            parse_provider(&model).ok_or(OllamaCallError::MissingModel)?;
        let messages = build_messages(llm_request);

        let body = OllamaChatRequest {
            model: ollama_model_tag.to_string(),
            messages,
            stream: false,
        };
        let body_json = rusty_serde::json::to_string(&body)
            .map_err(|e| OllamaCallError::Parse(e.to_string()))?;

        let base_url = self.resolved_base_url();
        let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
        let http = reqwest::blocking::Client::new();

        let outcome = rusty_tokio::spawn_blocking(move || -> Result<(u16, String), String> {
            let response = http
                .post(&url)
                .header("content-type", "application/json")
                .body(body_json)
                .send()
                .map_err(|e| e.to_string())?;
            let status = response.status().as_u16();
            let text = response.text().map_err(|e| e.to_string())?;
            Ok((status, text))
        })
        .await;

        let (status, text) = match outcome {
            Ok(Ok(pair)) => pair,
            Ok(Err(message)) => return Err(OllamaCallError::Transport(message)),
            Err(join_error) => return Err(OllamaCallError::Transport(join_error.to_string())),
        };

        if !(200..300).contains(&status) {
            return Err(OllamaCallError::Http { status, body: text });
        }

        let parsed: OllamaChatResponse = rusty_serde::json::from_str(&text)
            .map_err(|e| OllamaCallError::Parse(e.to_string()))?;

        let content = parsed.message.map(|m| Content {
            role: Some("model".to_string()),
            parts: vec![adk_genai::content::Part::text(m.content)],
        });

        let usage_metadata = if parsed.prompt_eval_count.is_some() || parsed.eval_count.is_some() {
            let mut entries = Vec::new();
            if let Some(count) = parsed.prompt_eval_count {
                entries.push((
                    "promptTokenCount".to_string(),
                    rusty_serde::value::Value::Int(count),
                ));
            }
            if let Some(count) = parsed.eval_count {
                entries.push((
                    "candidatesTokenCount".to_string(),
                    rusty_serde::value::Value::Int(count),
                ));
            }
            Some(rusty_serde::value::Value::Map(entries))
        } else {
            None
        };

        Ok(LlmResponse {
            content,
            usage_metadata,
            ..Default::default()
        })
    }
}

impl BaseLlm for OllamaLlm {
    fn model(&self) -> &str {
        &self.model
    }

    fn type_name(&self) -> &'static str {
        "OllamaLlm"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::default()
    }

    /// **Adaptation**: the source's own regex is `ollama/(?!gemma3).*` (a
    /// negative lookahead excluding `Gemma3Ollama`'s models). Rust's
    /// `regex` crate is a linear-time engine with no lookaround support at
    /// all, so that pattern can't be expressed as a registry regex — it
    /// would fail to compile. The exclusion is enforced in code instead
    /// (see `parse_provider`), so this registers the broad `ollama/.*`
    /// and rejects `ollama/gemma3*` at call time rather than at
    /// registry-match time. Functionally equivalent for a single-backend
    /// registration; would only diverge from the source if a competing
    /// `Gemma3Ollama`-equivalent entry were registered for the *same*
    /// model string, which doesn't exist here.
    fn supported_models() -> Vec<&'static str>
    where
        Self: Sized,
    {
        vec!["ollama/.*", "ollama_chat/.*"]
    }

    fn generate_content_async<'a>(
        &'a self,
        llm_request: &'a LlmRequest,
        stream: bool,
    ) -> crate::base_llm::BoxFuture<'a, Result<Vec<LlmResponse>, crate::base_llm::BaseLlmError>>
    {
        Box::pin(async move {
            if stream {
                return Err(crate::base_llm::BaseLlmError::CallFailed(
                    "OllamaLlm streaming generate_content_async isn't implemented yet".to_string(),
                ));
            }
            self.generate_content(llm_request)
                .await
                .map(|response| vec![response])
                .map_err(|e| crate::base_llm::BaseLlmError::CallFailed(e.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::Part;

    #[test]
    fn parse_provider_recognizes_the_ollama_prefix() {
        assert_eq!(
            parse_provider("ollama/llama3"),
            Some((OllamaProvider::Ollama, "llama3"))
        );
    }

    #[test]
    fn parse_provider_recognizes_the_ollama_chat_prefix() {
        assert_eq!(
            parse_provider("ollama_chat/llama3"),
            Some((OllamaProvider::OllamaChat, "llama3"))
        );
    }

    #[test]
    fn parse_provider_excludes_gemma3_matching_the_source_carve_out() {
        assert_eq!(parse_provider("ollama/gemma3:2b"), None);
    }

    #[test]
    fn parse_provider_rejects_unrelated_models() {
        assert_eq!(parse_provider("gemini-2.5-flash"), None);
    }

    #[test]
    fn build_messages_flattens_multiple_text_parts_with_newlines() {
        let mut request = LlmRequest::new("ollama_chat/llama3");
        request
            .contents
            .push(Content::new("user", vec![Part::text("a"), Part::text("b")]));
        let messages = build_messages(&request);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "a\nb");
    }

    #[test]
    fn build_messages_maps_the_model_role_to_assistant() {
        let mut request = LlmRequest::new("ollama/llama3");
        request
            .contents
            .push(Content::new("model", vec![Part::text("hi")]));
        let messages = build_messages(&request);
        assert_eq!(messages[0].role, "assistant");
    }

    #[test]
    fn build_messages_leads_with_a_system_message_when_present() {
        let mut request = LlmRequest::new("ollama/llama3");
        request.config.system_instruction = Some("be helpful".to_string());
        request
            .contents
            .push(Content::new("user", vec![Part::text("hi")]));
        let messages = build_messages(&request);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "be helpful");
    }

    #[test]
    fn supported_models_matches_the_source_ollama_prefixes() {
        let patterns = OllamaLlm::supported_models();
        assert_eq!(patterns.len(), 2);
        assert!(patterns.contains(&"ollama_chat/.*"));
    }

    #[test]
    fn resolved_base_url_defaults_to_localhost_11434() {
        let llm = OllamaLlm::new("ollama/llama3");
        std::env::remove_var("OLLAMA_HOST");
        assert_eq!(llm.resolved_base_url(), "http://localhost:11434");
    }

    #[test]
    fn resolved_base_url_prefers_the_explicit_field() {
        let llm = OllamaLlm::new("ollama/llama3").with_base_url("http://example.com:1234");
        assert_eq!(llm.resolved_base_url(), "http://example.com:1234");
    }

    /// One-shot local HTTP server speaking Ollama's documented `/api/chat`
    /// response shape — see the module doc for why this isn't a real
    /// Ollama instance.
    fn spawn_one_shot_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut received = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&buf[..n]);
                if let Some(header_end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&received[..header_end]);
                    let content_length: usize = headers
                        .lines()
                        .find_map(|line| {
                            let lower = line.to_ascii_lowercase();
                            lower
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if received.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
            }
            let response = format!(
                "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });
        (format!("http://{addr}"), handle)
    }

    #[rusty_tokio::test]
    async fn generate_content_parses_a_successful_ollama_response() {
        let (base_url, server) = spawn_one_shot_server(
            "HTTP/1.1 200 OK",
            r#"{"message":{"role":"assistant","content":"hi there"},"done":true,"prompt_eval_count":3,"eval_count":2}"#,
        );
        let llm = OllamaLlm::new("ollama/llama3").with_base_url(base_url);
        let mut request = LlmRequest::new("ollama/llama3");
        request.contents.push(Content::user_text("hello"));
        let response = llm.generate_content(&request).await.unwrap();
        server.join().unwrap();

        assert_eq!(
            response.content.unwrap().parts[0].text.as_deref(),
            Some("hi there")
        );
        assert!(response.usage_metadata.is_some());
    }

    #[rusty_tokio::test]
    async fn generate_content_maps_a_non_2xx_response_to_an_http_error() {
        let (base_url, server) =
            spawn_one_shot_server("HTTP/1.1 404 Not Found", r#"{"error":"model not found"}"#);
        let llm = OllamaLlm::new("ollama/does-not-exist").with_base_url(base_url);
        let request = LlmRequest::new("ollama/does-not-exist");
        let result = llm.generate_content(&request).await;
        server.join().unwrap();

        match result {
            Err(OllamaCallError::Http { status, .. }) => assert_eq!(status, 404),
            _ => panic!("expected Http error"),
        }
    }

    #[rusty_tokio::test]
    async fn generate_content_errors_for_an_unrecognized_model_string() {
        let llm = OllamaLlm::new("gemini-2.5-flash");
        let request = LlmRequest::new("gemini-2.5-flash");
        let result = llm.generate_content(&request).await;
        match result {
            Err(OllamaCallError::MissingModel) => {}
            _ => panic!("expected MissingModel"),
        }
    }
}
