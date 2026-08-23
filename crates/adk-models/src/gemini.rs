//! Capabilities C0123-C0125 (partial), C0127, C0129-C0131 (C0131 partial),
//! C0130 (partial C0133): `Gemini`, ported from `google.adk.models.google_llm`.
//!
//! **Scope of batch 2 (config layer)**: the config shape, `supported_models()`,
//! base-URL/API-version resolution, and API-client construction — pure
//! configuration logic, testable without a live network call.
//!
//! **Scope of batch 3 (this batch, real calls)**: non-streaming
//! `generate_content_async` over the real Gemini REST API
//! ([`Gemini::generate_content`]), for the Gemini-Developer-API
//! (API-key) backend only, plus [`GeminiCallError::ResourceExhausted`]
//! (C0127). **Adaptation, disclosed**: auth is resolved in one of two ways —
//! an injected [`Gemini::client`] is used exactly as configured (the caller
//! is assumed to have set up its own auth, e.g. a Vertex AI bearer token),
//! or — when no client is injected — an API key is read from
//! `GOOGLE_API_KEY`/`GEMINI_API_KEY` and sent as `x-goog-api-key`, which
//! only works for the Gemini Developer API backend. Building our own
//! Vertex AI credentials (Application Default Credentials: gcloud user
//! creds, service-account JWTs, the GCE/GKE metadata server, workload
//! identity) is a distinct, large dependency decision of its own —
//! deferred, not silently unsupported: [`GeminiCallError::VertexAiAuthNotSupported`]
//! names exactly what's missing and how to work around it today (inject a
//! pre-authed client).
//!
//! **Scope of batch 4 (this batch)**: [`Gemini::prepare_live_connect_config`]
//! and [`Gemini::live_api_version`] — everything `Gemini.connect()` does to
//! `llm_request.live_connect_config` *before* opening the actual Live
//! WebSocket connection (C0131's config-prep half): merging tracking
//! headers into an already-present `http_options` (matching the source's
//! own gating — it's never created here either), forwarding
//! `speech_config`/`tools`/`thinking_config`/`safety_settings`, the
//! unconditional `system_instruction` assignment, and validating that
//! transparent session resumption is Vertex-AI-only. All pure, in-memory
//! config mutation — testable without a live network call, same as batch
//! 2's `api_client` construction.
//!
//! Still deferred to later batches, each needing its own foundational
//! decision or wire-format work this batch doesn't have yet:
//!   - The SSE-streaming half of C0125 (`stream=true`, `StreamingResponseAggregator`),
//!     C0126 (context-cache integration), and C0128 (interactions-API
//!     delegation) — streaming needs an SSE-parsing decision on top of the
//!     transport this batch already has; caching/interactions need
//!     capabilities from later batches.
//!   - The actual Live WebSocket handshake (the rest of C0131 —
//!     `_live_api_client`, opening the connection) and all of
//!     `GeminiLlmConnection` (C0132, C0135-C0139) — `receive()` alone is a
//!     ~370-line stateful message-translation engine (grounding-metadata
//!     accumulation with index-offset merging, streamed text/thought
//!     aggregation tracked by part identity, transcription streaming,
//!     Gemini-3.x-variant-dependent tool-call buffering, session-
//!     resumption/voice-activity/GoAway passthrough) that deserves its own
//!     dedicated batch rather than being hand-waved alongside config-prep,
//!     the same way `GeminiContextCacheManager` got its own batch instead
//!     of being squeezed into this one. The WebSocket transport itself is
//!     also still undecided — `tungstenite` (the synchronous core
//!     `tokio-tungstenite` wraps) is the leading candidate, since it has
//!     the same runtime-agnostic property that made `reqwest::blocking`
//!     the right fit for the REST transport (see the load-bearing
//!     adaptation note below) — but that decision is made when the
//!     connection itself is built, not before.
//!   - C0134 (redacted debug request/response logging).
//!   - C0126 (wiring `GeminiContextCacheManager` into
//!     `generate_content_async`) — the manager itself (C0140-C0143) is
//!     built, independently, in `gemini_context_cache_manager.rs`.
//!   - `config.tools`/`FunctionDeclaration` in the request body — not
//!     modeled yet (C0116, Phase 8's `BaseTool`); see
//!     `generate_content_request.rs`'s module doc.
//!
//! **Adaptation**: the source's `api_client` is a `cached_property`
//! returning a full `google.genai.Client` (itself wrapping `httpx`/
//! `aiohttp`, ADC-based Vertex AI auth, retries, etc.). [`GeminiApiClient`]
//! here is the Rust-native equivalent scoped to what's decidable without
//! the real wire calls: a `reqwest::blocking::Client` pre-loaded with tracking
//! headers, plus the resolved base URL/API version/`enterprise` flag.
//! `client_kwargs` (the source's free-form `dict` merged into the SDK
//! constructor's kwargs, capable of overriding *any* constructor argument)
//! has no well-typed Rust equivalent without knowing which keys matter, so
//! it stays an inert opaque placeholder — like `tools_dict` in
//! `llm_request.rs`, documented rather than silently dropped.
//!
//! **Adaptation**: request/response bodies are sent/parsed via
//! `rusty_serde::json` (a `String` body plus an explicit `content-type`
//! header on the way out, `response.text()` plus `rusty_serde::json::from_str`
//! on the way back) rather than `reqwest`'s `.json()` convenience method —
//! that method requires the real `serde::Serialize`/`Deserialize` traits,
//! and this workspace deliberately has one serialization framework
//! (`rusty_serde`), not two.
//!
//! **Adaptation, load-bearing**: [`GeminiApiClient`] wraps
//! `reqwest::blocking::Client`, not the async client. `reqwest`'s async
//! transport calls straight into real `tokio::net::TcpStream`/
//! `tokio::runtime::Handle::current()`, which only exists inside an actual
//! `tokio::runtime::Runtime` — and `rusty_tokio` (this workspace's async
//! runtime, adopted in Phase 2) is a from-scratch, independent reactor, not
//! a wrapper around real tokio. The two can't share a reactor: an async
//! `reqwest::Client` call panics with "there is no reactor running" under
//! `rusty_tokio`. `reqwest::blocking::Client` sidesteps this because it
//! spins up its own private, self-contained tokio runtime internally,
//! independent of whatever ambient executor called it — so
//! [`Gemini::generate_content`] runs it inside `rusty_tokio::spawn_blocking`
//! (a genuine blocking-thread-pool offload, so a slow HTTP call doesn't
//! stall a `rusty_tokio` async worker thread) rather than calling it
//! directly.

use std::sync::{Arc, OnceLock};

use adk_genai::content::{Content, Part};
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use rusty_serde::value::Value;

use crate::base_llm::BaseLlm;
use crate::capabilities::{
    gemini_output_schema_and_tools, get_google_llm_variant, GoogleLlmVariant,
};
use crate::generate_content_request::build_request_body;
use crate::generate_content_response::GenerateContentResponse;
use crate::google_client_headers::get_tracking_headers;
use crate::llm_request::LlmRequest;
use crate::llm_response::LlmResponse;

const DEFAULT_MODEL: &str = "gemini-2.5-flash";
const API_VERSION_ENV_VAR: &str = "GOOGLE_GENAI_API_VERSION";
/// `pub(crate)`: reused by `gemini_context_cache_manager.rs` to build the
/// `cachedContents` REST endpoint URL the same way this module builds the
/// `generateContent` one.
pub(crate) const DEFAULT_GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub(crate) const DEFAULT_GEMINI_API_VERSION: &str = "v1beta";
const RESOURCE_EXHAUSTED_MITIGATION_LINK: &str = "On how to mitigate this issue, please refer to:\n\nhttps://google.github.io/adk-docs/agents/models/google-gemini/#error-code-429-resource_exhausted";

fn version_suffix_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"^/?(v[0-9][a-z0-9.-]*)/?$").expect("valid regex"))
}

/// The Rust-native stand-in for a constructed `google.genai.Client` — see
/// the module doc's adaptation note. Not constructed directly; built by
/// [`Gemini::api_client`].
pub struct GeminiApiClient {
    pub http: reqwest::blocking::Client,
    pub base_url: Option<String>,
    pub api_version: Option<String>,
    pub headers: Vec<(String, String)>,
    /// Opaque placeholder for `types.HttpRetryOptions` — nothing here reads
    /// it yet (no real call retries anything in this batch).
    pub retry_options: Option<Value>,
    /// Whether this client targets the Agent Engine "enterprise" surface
    /// (the source's `kwargs['enterprise'] = True` when `model` starts with
    /// `projects/`) — distinct from the Gemini-API/Vertex-AI backend
    /// variant, which is [`Gemini::api_backend`].
    pub enterprise: bool,
}

/// Errors from [`Gemini::generate_content`] — the concrete, structured
/// counterpart to the `BaseLlm` trait's flattened [`crate::base_llm::BaseLlmError::CallFailed`].
#[derive(Debug, rusty_err::Error)]
pub enum GeminiCallError {
    #[error("Gemini requests require a model name.")]
    MissingModel,
    /// See the module doc's disclosed adaptation: only the Gemini
    /// Developer API (API-key) backend can build its own auth today.
    #[error(
        "the Vertex AI backend isn't supported yet for real generate_content calls (needs \
         Application Default Credentials) — inject a pre-configured `client` with your own \
         auth to use Vertex AI today"
    )]
    VertexAiAuthNotSupported,
    #[error("no API key found — set GOOGLE_API_KEY or GEMINI_API_KEY")]
    MissingApiKey,
    #[error("request to the Gemini API failed: {0}")]
    Transport(String),
    #[error("Gemini API returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("failed to parse the Gemini API response: {0}")]
    Parse(String),
    /// C0127: `_ResourceExhaustedError` — a 429 response, enhanced with a
    /// link to ADK's resource-exhaustion mitigation docs.
    #[error("HTTP 429: {body}\n\n{RESOURCE_EXHAUSTED_MITIGATION_LINK}")]
    ResourceExhausted { body: String },
    /// C0131: transparent session resumption only works on the Vertex AI
    /// backend.
    #[error(
        "Transparent session resumption is only supported for Vertex AI backend. Please use \
         Vertex AI backend."
    )]
    TransparentSessionResumptionRequiresVertexAi,
}

/// C0127: maps a non-2xx HTTP response into a [`GeminiCallError`],
/// enhancing a 429 into [`GeminiCallError::ResourceExhausted`].
fn map_http_error(status: u16, body: String) -> GeminiCallError {
    if status == 429 {
        GeminiCallError::ResourceExhausted { body }
    } else {
        GeminiCallError::Http { status, body }
    }
}

fn build_http_client(headers: &[(String, String)]) -> reqwest::blocking::Client {
    let mut header_map = HeaderMap::new();
    for (key, value) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            header_map.insert(name, value);
        }
    }
    reqwest::blocking::Client::builder()
        .default_headers(header_map)
        .build()
        .expect("reqwest client with well-formed static headers must build")
}

/// C0130: `_normalize_base_url_and_api_version` — extracts a Google API
/// version path suffix (e.g. `v1beta1`) from a `*.googleapis.com` base URL.
///
/// **Adaptation**: matches against the URL's host only (ignoring an
/// explicit port), whereas the source matches against the full `netloc`
/// (host possibly including `:port`). Google API base URLs are not given
/// with an explicit port in practice, so this only diverges on a
/// synthetic/test URL that deliberately adds one.
pub fn normalize_base_url_and_api_version(
    base_url: Option<&str>,
) -> (Option<String>, Option<String>) {
    let base_url = match base_url {
        Some(b) if !b.is_empty() => b,
        _ => return (None, None),
    };

    let parsed = match reqwest::Url::parse(base_url) {
        Ok(u) => u,
        Err(_) => return (Some(base_url.to_string()), None),
    };

    let host_matches = parsed
        .host_str()
        .map(|h| h.ends_with(".googleapis.com"))
        .unwrap_or(false);
    let has_query = parsed.query().is_some();
    let has_fragment = parsed.fragment().map(|f| !f.is_empty()).unwrap_or(false);

    if !host_matches || has_query || has_fragment {
        return (Some(base_url.to_string()), None);
    }

    let path = parsed.path();
    if path.is_empty() || path == "/" {
        return (Some(base_url.to_string()), None);
    }

    match version_suffix_pattern()
        .captures(path)
        .and_then(|c| c.get(1))
    {
        Some(m) => {
            let version = m.as_str().to_string();
            let mut normalized = parsed.clone();
            normalized.set_path("/");
            (Some(normalized.to_string()), Some(version))
        }
        None => (Some(base_url.to_string()), None),
    }
}

/// `Gemini` — native Gemini/Vertex AI backend. Config shape only in this
/// batch; see the module doc for what's deferred.
pub struct Gemini {
    pub model: String,
    /// A pre-configured client to use for all API calls instead of one
    /// constructed from this instance's other fields.
    pub client: Option<Arc<GeminiApiClient>>,
    /// Opaque placeholder — see the module doc's adaptation note.
    pub client_kwargs: Option<Value>,
    pub base_url: Option<String>,
    pub api_version: Option<String>,
    /// Opaque placeholder for `types.SpeechConfig`.
    pub speech_config: Option<Value>,
    pub use_interactions_api: bool,
    /// Opaque placeholder for `types.HttpRetryOptions`.
    pub retry_options: Option<Value>,

    api_client_cache: OnceLock<Arc<GeminiApiClient>>,
}

impl Default for Gemini {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL)
    }
}

impl Gemini {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            client: None,
            client_kwargs: None,
            base_url: None,
            api_version: None,
            speech_config: None,
            use_interactions_api: false,
            retry_options: None,
            api_client_cache: OnceLock::new(),
        }
    }

    pub fn with_client(mut self, client: Arc<GeminiApiClient>) -> Self {
        self.client = Some(client);
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// C0130: resolution order — the explicit `api_version` field, then the
    /// `GOOGLE_GENAI_API_VERSION` env var, then `None` (leaving the SDK's
    /// own default to apply).
    fn configured_api_version(&self) -> Option<String> {
        if let Some(version) = &self.api_version {
            return Some(version.clone());
        }
        std::env::var(API_VERSION_ENV_VAR)
            .ok()
            .filter(|v| !v.is_empty())
    }

    fn base_url_and_api_version(&self) -> (Option<String>, Option<String>) {
        normalize_base_url_and_api_version(self.base_url.as_deref())
    }

    /// C0129: the API client — an injected [`Gemini::client`] if present,
    /// otherwise built (and cached for the lifetime of this instance) from
    /// tracking headers, resolved base URL/API version, and the
    /// `enterprise` flag.
    pub fn api_client(&self) -> Arc<GeminiApiClient> {
        if let Some(client) = &self.client {
            return client.clone();
        }
        self.api_client_cache
            .get_or_init(|| {
                let (base_url, mut api_version) = self.base_url_and_api_version();
                if api_version.is_none() {
                    api_version = self.configured_api_version();
                }
                let headers = get_tracking_headers(None);
                let enterprise = self.model.starts_with("projects/");
                Arc::new(GeminiApiClient {
                    http: build_http_client(&headers),
                    base_url,
                    api_version,
                    headers,
                    retry_options: self.retry_options.clone(),
                    enterprise,
                })
            })
            .clone()
    }

    /// The source's `_api_backend` cached_property depends on the
    /// constructed `genai.Client`'s own `.vertexai` attribute, which that
    /// SDK derives from the same environment signals ADK's own
    /// [`get_google_llm_variant`] already checks — reused directly rather
    /// than re-deriving it from a Rust-native client that has no
    /// credential-discovery logic of its own yet.
    pub fn api_backend(&self) -> GoogleLlmVariant {
        get_google_llm_variant()
    }

    /// C0131 (config-prep half): the API version for a Live connection —
    /// an embedded-in-`base_url` version takes precedence (unlike the REST
    /// `api_client`, this does *not* fall back to the explicit
    /// `api_version` field or `GOOGLE_GENAI_API_VERSION`), then the
    /// backend-specific default (`v1beta1` for Vertex AI, `v1alpha` for the
    /// Gemini Developer API).
    pub fn live_api_version(&self) -> String {
        let (_, api_version) = self.base_url_and_api_version();
        if let Some(version) = api_version {
            return version;
        }
        match self.api_backend() {
            GoogleLlmVariant::VertexAi => "v1beta1".to_string(),
            GoogleLlmVariant::GeminiApi => "v1alpha".to_string(),
        }
    }

    /// C0131 (config-prep half — see the module doc): everything
    /// `Gemini.connect()` does to `llm_request.live_connect_config` before
    /// opening the actual Live WebSocket connection (deferred — see the
    /// module doc). Mutates `llm_request` in place, matching the source
    /// (which mutates the same request object callers pass to `connect()`).
    pub fn prepare_live_connect_config(
        &self,
        llm_request: &mut LlmRequest,
    ) -> Result<(), GeminiCallError> {
        let live_api_version = self.live_api_version();
        let system_instruction_text = llm_request.config.system_instruction.clone();
        let tools = llm_request.config.tools.clone();
        let thinking_config = llm_request.config.thinking_config.clone();
        let safety_settings = llm_request.config.safety_settings.clone();
        let api_backend = self.api_backend();
        let speech_config = self.speech_config.clone();

        let config = llm_request
            .live_connect_config
            .get_or_insert_with(crate::llm_request::LiveConnectConfigStub::default);

        // Only touches headers/api_version when `http_options` is already
        // present — matches the source's own gating exactly (it never
        // creates `http_options` here either).
        if let Some(http_options) = &mut config.http_options {
            let existing: Vec<(String, String)> = http_options
                .headers
                .clone()
                .unwrap_or_default()
                .into_iter()
                .collect();
            let merged = crate::google_client_headers::merge_tracking_headers(&existing, None);
            http_options.headers = Some(merged.into_iter().collect());
            http_options.api_version = Some(live_api_version);
        }

        if let Some(speech_config) = speech_config {
            config.speech_config = Some(speech_config);
        }

        // Assigned unconditionally — see llm_request.rs's `Part::text`
        // equivalent: with no system instruction this still sends a
        // `Content(role="system", parts=[Part()])`, matching the source's
        // documented behavior rather than omitting the field.
        config.system_instruction = Some(Content {
            role: Some("system".to_string()),
            parts: vec![Part {
                text: system_instruction_text,
                ..Default::default()
            }],
        });

        let wants_transparent_resumption = config
            .session_resumption
            .as_ref()
            .and_then(|r| r.transparent)
            .unwrap_or(false);
        if wants_transparent_resumption && api_backend == GoogleLlmVariant::GeminiApi {
            return Err(GeminiCallError::TransparentSessionResumptionRequiresVertexAi);
        }

        config.tools = tools;
        if thinking_config.is_some() {
            config.thinking_config = thinking_config;
        }
        if safety_settings.is_some() && config.safety_settings.is_none() {
            config.safety_settings = safety_settings;
        }

        Ok(())
    }

    /// Resolves the auth header to attach to a real API call. `None` means
    /// "the client already carries whatever auth it needs" — either an
    /// injected [`Gemini::client`] (assumed pre-authed by the caller), or
    /// this being the one case that needs no header of its own. See the
    /// module doc's disclosed adaptation for why only the Gemini Developer
    /// API backend can build its own auth today.
    fn resolve_auth_header(&self) -> Result<Option<(&'static str, String)>, GeminiCallError> {
        if self.client.is_some() {
            return Ok(None);
        }
        match self.api_backend() {
            GoogleLlmVariant::VertexAi => Err(GeminiCallError::VertexAiAuthNotSupported),
            GoogleLlmVariant::GeminiApi => {
                let key = std::env::var("GOOGLE_API_KEY")
                    .or_else(|_| std::env::var("GEMINI_API_KEY"))
                    .map_err(|_| GeminiCallError::MissingApiKey)?;
                Ok(Some(("x-goog-api-key", key)))
            }
        }
    }

    fn generate_content_url(&self, model: &str) -> String {
        let client = self.api_client();
        let base_url = client
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_BASE_URL.to_string());
        let api_version = client
            .api_version
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_VERSION.to_string());
        let base_url = base_url.trim_end_matches('/');
        format!("{base_url}/{api_version}/models/{model}:generateContent")
    }

    /// C0125 (non-streaming only — see the module doc): sends a real,
    /// non-streaming `generateContent` request and maps the response into
    /// an [`LlmResponse`]. The concrete, structured counterpart to
    /// `BaseLlm::generate_content_async`'s trait-object-friendly
    /// `stream: false` case.
    pub async fn generate_content(
        &self,
        llm_request: &LlmRequest,
    ) -> Result<LlmResponse, GeminiCallError> {
        let mut llm_request = llm_request.clone();
        crate::base_llm::maybe_append_user_content(&mut llm_request);
        let model = llm_request
            .model
            .clone()
            .ok_or(GeminiCallError::MissingModel)?;

        let auth_header = self.resolve_auth_header()?;
        let client = self.api_client();
        let url = self.generate_content_url(&model);
        let body = build_request_body(&llm_request);
        let body_json = rusty_serde::json::to_string(&body)
            .map_err(|e| GeminiCallError::Parse(e.to_string()))?;

        // `reqwest::blocking` spins up its own private tokio runtime, so
        // this genuinely blocking call is safe to make — it just must not
        // run directly on a `rusty_tokio` async worker thread, hence the
        // `spawn_blocking` offload. See the module doc's load-bearing
        // adaptation note.
        let outcome = rusty_tokio::spawn_blocking(move || -> Result<(u16, String), String> {
            let mut request = client
                .http
                .post(&url)
                .header("content-type", "application/json")
                .body(body_json);
            if let Some((name, value)) = auth_header {
                request = request.header(name, value);
            }
            let response = request.send().map_err(|e| e.to_string())?;
            let status = response.status().as_u16();
            let text = response.text().map_err(|e| e.to_string())?;
            Ok((status, text))
        })
        .await;

        let (status, text) = match outcome {
            Ok(Ok(pair)) => pair,
            Ok(Err(message)) => return Err(GeminiCallError::Transport(message)),
            Err(join_error) => return Err(GeminiCallError::Transport(join_error.to_string())),
        };

        if !(200..300).contains(&status) {
            return Err(map_http_error(status, text));
        }

        let parsed: GenerateContentResponse = rusty_serde::json::from_str(&text)
            .map_err(|e| GeminiCallError::Parse(e.to_string()))?;
        Ok(LlmResponse::create(parsed))
    }
}

impl BaseLlm for Gemini {
    fn model(&self) -> &str {
        &self.model
    }

    fn type_name(&self) -> &'static str {
        "Gemini"
    }

    fn capabilities(&self) -> crate::capabilities::LlmCapabilities {
        crate::capabilities::LlmCapabilities {
            output_schema_and_tools: gemini_output_schema_and_tools(&self.model),
        }
    }

    /// C0125 (non-streaming only — see the module doc): delegates to
    /// [`Gemini::generate_content`], flattening its structured
    /// [`GeminiCallError`] into `BaseLlmError::CallFailed` for the
    /// trait-object-friendly contract. `stream: true` isn't implemented yet
    /// (the SSE-streaming half of C0125).
    fn generate_content_async<'a>(
        &'a self,
        llm_request: &'a LlmRequest,
        stream: bool,
    ) -> crate::base_llm::BoxFuture<'a, Result<Vec<LlmResponse>, crate::base_llm::BaseLlmError>>
    {
        Box::pin(async move {
            if stream {
                return Err(crate::base_llm::BaseLlmError::CallFailed(
                    "Gemini streaming generate_content_async isn't implemented yet (deferred \
                     to a later batch)"
                        .to_string(),
                ));
            }
            self.generate_content(llm_request)
                .await
                .map(|response| vec![response])
                .map_err(|e| crate::base_llm::BaseLlmError::CallFailed(e.to_string()))
        })
    }

    /// C0124.
    fn supported_models() -> Vec<&'static str>
    where
        Self: Sized,
    {
        vec![
            "gemini-.*",
            "gemma-4.*",
            "model-optimizer-.*",
            r"projects\/.+\/locations\/.+\/endpoints\/.+",
            r"projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `GOOGLE_GENAI_*` env vars are process-global; serializes the tests
    /// below that set/remove them so they don't race each other under the
    /// default multi-threaded test harness.
    static ENV_VAR_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn default_model_is_gemini_2_5_flash() {
        assert_eq!(Gemini::default().model, DEFAULT_MODEL);
    }

    #[test]
    fn supported_models_matches_the_source_pattern_list() {
        let patterns = Gemini::supported_models();
        assert_eq!(patterns.len(), 5);
        assert!(patterns.contains(&"gemini-.*"));
        assert!(patterns.contains(&"gemma-4.*"));
    }

    #[test]
    fn capabilities_report_output_schema_and_tools_only_on_vertex_ai() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
        std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        let gemini = Gemini::new("gemini-2.5-flash");
        assert!(!gemini.capabilities().output_schema_and_tools);
    }

    #[test]
    fn normalize_base_url_returns_none_none_for_a_missing_url() {
        assert_eq!(normalize_base_url_and_api_version(None), (None, None));
        assert_eq!(normalize_base_url_and_api_version(Some("")), (None, None));
    }

    #[test]
    fn normalize_base_url_leaves_a_non_google_host_unchanged() {
        let (url, version) = normalize_base_url_and_api_version(Some("https://example.com/v1"));
        assert_eq!(url.as_deref(), Some("https://example.com/v1"));
        assert_eq!(version, None);
    }

    #[test]
    fn normalize_base_url_strips_a_version_suffix_from_a_googleapis_host() {
        let (url, version) = normalize_base_url_and_api_version(Some(
            "https://region-aiplatform.googleapis.com/v1beta1",
        ));
        assert_eq!(
            url.as_deref(),
            Some("https://region-aiplatform.googleapis.com/")
        );
        assert_eq!(version.as_deref(), Some("v1beta1"));
    }

    #[test]
    fn normalize_base_url_leaves_a_bare_host_unchanged() {
        let (url, version) =
            normalize_base_url_and_api_version(Some("https://region-aiplatform.googleapis.com/"));
        assert_eq!(
            url.as_deref(),
            Some("https://region-aiplatform.googleapis.com/")
        );
        assert_eq!(version, None);
    }

    #[test]
    fn normalize_base_url_leaves_a_url_with_query_params_unchanged() {
        let (url, version) = normalize_base_url_and_api_version(Some(
            "https://region-aiplatform.googleapis.com/v1?x=1",
        ));
        assert_eq!(
            url.as_deref(),
            Some("https://region-aiplatform.googleapis.com/v1?x=1")
        );
        assert_eq!(version, None);
    }

    #[test]
    fn configured_api_version_prefers_the_explicit_field_over_the_env_var() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        std::env::set_var("GOOGLE_GENAI_API_VERSION", "v1-from-env");
        let gemini = Gemini::new("gemini-2.5-flash").with_api_version("v1-explicit");
        assert_eq!(
            gemini.configured_api_version().as_deref(),
            Some("v1-explicit")
        );
        std::env::remove_var("GOOGLE_GENAI_API_VERSION");
    }

    #[test]
    fn configured_api_version_falls_back_to_the_env_var() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        std::env::set_var("GOOGLE_GENAI_API_VERSION", "v1-from-env");
        let gemini = Gemini::new("gemini-2.5-flash");
        assert_eq!(
            gemini.configured_api_version().as_deref(),
            Some("v1-from-env")
        );
        std::env::remove_var("GOOGLE_GENAI_API_VERSION");
    }

    #[test]
    fn api_client_sets_enterprise_for_a_projects_style_model_name() {
        let gemini = Gemini::new("projects/p/locations/l/endpoints/e");
        assert!(gemini.api_client().enterprise);
    }

    #[test]
    fn api_client_is_not_enterprise_for_a_plain_model_name() {
        let gemini = Gemini::new("gemini-2.5-flash");
        assert!(!gemini.api_client().enterprise);
    }

    #[test]
    fn api_client_carries_tracking_headers() {
        let gemini = Gemini::new("gemini-2.5-flash");
        let client = gemini.api_client();
        assert!(client.headers.iter().any(|(k, _)| k == "x-goog-api-client"));
    }

    #[test]
    fn api_client_prefers_an_injected_client_over_building_one() {
        let injected = Arc::new(GeminiApiClient {
            http: reqwest::blocking::Client::new(),
            base_url: Some("https://injected.example.com".to_string()),
            api_version: None,
            headers: vec![],
            retry_options: None,
            enterprise: false,
        });
        let gemini = Gemini::new("gemini-2.5-flash").with_client(injected.clone());
        assert_eq!(
            gemini.api_client().base_url,
            Some("https://injected.example.com".to_string())
        );
    }

    fn clear_auth_env_vars() {
        std::env::remove_var("GOOGLE_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");
        std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
        std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
    }

    #[test]
    fn live_api_version_defaults_to_v1alpha_on_the_gemini_api_backend() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let version = gemini.live_api_version();
        clear_auth_env_vars();
        assert_eq!(version, "v1alpha");
    }

    #[test]
    fn live_api_version_defaults_to_v1beta1_on_vertex_ai() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        let gemini = Gemini::new("gemini-2.5-flash");
        let version = gemini.live_api_version();
        clear_auth_env_vars();
        assert_eq!(version, "v1beta1");
    }

    #[test]
    fn live_api_version_prefers_a_version_embedded_in_the_base_url_over_the_backend_default() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash")
            .with_base_url("https://region-aiplatform.googleapis.com/v1beta1");
        let version = gemini.live_api_version();
        clear_auth_env_vars();
        assert_eq!(version, "v1beta1");
    }

    #[test]
    fn live_api_version_ignores_the_explicit_api_version_field_and_env_var() {
        // Unlike the REST `api_client`, `_live_api_version` never falls
        // back to `self.api_version`/`GOOGLE_GENAI_API_VERSION` — matches
        // the source exactly (see `live_api_version`'s doc comment).
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_GENAI_API_VERSION", "v1-from-env");
        let gemini = Gemini::new("gemini-2.5-flash").with_api_version("v1-explicit");
        let version = gemini.live_api_version();
        clear_auth_env_vars();
        std::env::remove_var("GOOGLE_GENAI_API_VERSION");
        assert_eq!(version, "v1alpha");
    }

    #[test]
    fn prepare_live_connect_config_defaults_a_missing_config() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        assert!(request.live_connect_config.is_none());
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();
        assert!(request.live_connect_config.is_some());
    }

    #[test]
    fn prepare_live_connect_config_only_merges_tracking_headers_when_http_options_is_already_set() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        // No `http_options` was ever set, so none is created — matches the
        // source's own gating exactly.
        assert!(request.live_connect_config.unwrap().http_options.is_none());
    }

    #[test]
    fn prepare_live_connect_config_merges_tracking_headers_and_sets_the_live_api_version_when_http_options_is_present(
    ) {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.live_connect_config = Some(crate::llm_request::LiveConnectConfigStub {
            http_options: Some(crate::llm_request::HttpOptionsStub::default()),
            ..Default::default()
        });
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        let http_options = request.live_connect_config.unwrap().http_options.unwrap();
        assert_eq!(http_options.api_version.as_deref(), Some("v1alpha"));
        assert!(http_options
            .headers
            .unwrap()
            .contains_key("x-goog-api-client"));
    }

    #[test]
    fn prepare_live_connect_config_forwards_the_speech_config() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut gemini = gemini;
        gemini.speech_config = Some(Value::String("speech".to_string()));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        assert_eq!(
            request.live_connect_config.unwrap().speech_config,
            Some(Value::String("speech".to_string()))
        );
    }

    #[test]
    fn prepare_live_connect_config_always_sets_a_system_instruction_content_even_when_empty() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        let system_instruction = request
            .live_connect_config
            .unwrap()
            .system_instruction
            .unwrap();
        assert_eq!(system_instruction.role.as_deref(), Some("system"));
        assert_eq!(system_instruction.parts.len(), 1);
        assert!(system_instruction.parts[0].text.is_none());
    }

    #[test]
    fn prepare_live_connect_config_carries_the_system_instruction_text() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.config.system_instruction = Some("be helpful".to_string());
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        let system_instruction = request
            .live_connect_config
            .unwrap()
            .system_instruction
            .unwrap();
        assert_eq!(
            system_instruction.parts[0].text.as_deref(),
            Some("be helpful")
        );
    }

    #[test]
    fn prepare_live_connect_config_rejects_transparent_session_resumption_on_the_gemini_api_backend(
    ) {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.live_connect_config = Some(crate::llm_request::LiveConnectConfigStub {
            session_resumption: Some(crate::llm_request::SessionResumptionStub {
                transparent: Some(true),
            }),
            ..Default::default()
        });
        let result = gemini.prepare_live_connect_config(&mut request);
        clear_auth_env_vars();

        match result {
            Err(GeminiCallError::TransparentSessionResumptionRequiresVertexAi) => {}
            _ => panic!("expected TransparentSessionResumptionRequiresVertexAi"),
        }
    }

    #[test]
    fn prepare_live_connect_config_allows_transparent_session_resumption_on_vertex_ai() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.live_connect_config = Some(crate::llm_request::LiveConnectConfigStub {
            session_resumption: Some(crate::llm_request::SessionResumptionStub {
                transparent: Some(true),
            }),
            ..Default::default()
        });
        let result = gemini.prepare_live_connect_config(&mut request);
        clear_auth_env_vars();

        assert!(result.is_ok());
    }

    #[test]
    fn prepare_live_connect_config_unconditionally_overwrites_tools() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.config.tools = Some(Value::String("new-tools".to_string()));
        request.live_connect_config = Some(crate::llm_request::LiveConnectConfigStub {
            tools: Some(Value::String("stale-tools".to_string())),
            ..Default::default()
        });
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        assert_eq!(
            request.live_connect_config.unwrap().tools,
            Some(Value::String("new-tools".to_string()))
        );
    }

    #[test]
    fn prepare_live_connect_config_only_forwards_thinking_config_when_present() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        gemini.prepare_live_connect_config(&mut request).unwrap();
        assert!(request
            .live_connect_config
            .as_ref()
            .unwrap()
            .thinking_config
            .is_none());

        request.config.thinking_config = Some(Value::String("thinking".to_string()));
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        assert_eq!(
            request.live_connect_config.unwrap().thinking_config,
            Some(Value::String("thinking".to_string()))
        );
    }

    #[test]
    fn prepare_live_connect_config_does_not_override_an_existing_safety_settings() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.config.safety_settings = Some(Value::String("new".to_string()));
        request.live_connect_config = Some(crate::llm_request::LiveConnectConfigStub {
            safety_settings: Some(Value::String("existing".to_string())),
            ..Default::default()
        });
        gemini.prepare_live_connect_config(&mut request).unwrap();
        clear_auth_env_vars();

        assert_eq!(
            request.live_connect_config.unwrap().safety_settings,
            Some(Value::String("existing".to_string()))
        );
    }

    #[test]
    fn resolve_auth_header_uses_google_api_key_for_the_gemini_api_backend() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_API_KEY", "the-key");
        let result = Gemini::new("gemini-2.5-flash").resolve_auth_header();
        clear_auth_env_vars();
        assert_eq!(
            result.unwrap(),
            Some(("x-goog-api-key", "the-key".to_string()))
        );
    }

    #[test]
    fn resolve_auth_header_falls_back_to_gemini_api_key() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GEMINI_API_KEY", "fallback-key");
        let result = Gemini::new("gemini-2.5-flash").resolve_auth_header();
        clear_auth_env_vars();
        assert_eq!(
            result.unwrap(),
            Some(("x-goog-api-key", "fallback-key".to_string()))
        );
    }

    #[test]
    fn resolve_auth_header_errors_without_any_key() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let result = Gemini::new("gemini-2.5-flash").resolve_auth_header();
        clear_auth_env_vars();
        match result {
            Err(GeminiCallError::MissingApiKey) => {}
            _ => panic!("expected MissingApiKey"),
        }
    }

    #[test]
    fn resolve_auth_header_errors_for_vertex_ai_without_an_injected_client() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        let result = Gemini::new("gemini-2.5-flash").resolve_auth_header();
        clear_auth_env_vars();
        match result {
            Err(GeminiCallError::VertexAiAuthNotSupported) => {}
            _ => panic!("expected VertexAiAuthNotSupported"),
        }
    }

    #[test]
    fn resolve_auth_header_is_none_when_a_client_is_injected() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        let injected = Arc::new(GeminiApiClient {
            http: reqwest::blocking::Client::new(),
            base_url: None,
            api_version: None,
            headers: vec![],
            retry_options: None,
            enterprise: false,
        });
        let result = Gemini::new("gemini-2.5-flash")
            .with_client(injected)
            .resolve_auth_header();
        clear_auth_env_vars();
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn generate_content_url_uses_the_gemini_api_default_base_and_version() {
        let gemini = Gemini::new("gemini-2.5-flash");
        assert_eq!(
            gemini.generate_content_url("gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn generate_content_url_uses_a_configured_base_url_and_version() {
        let gemini = Gemini::new("gemini-2.5-flash")
            .with_base_url("http://127.0.0.1:9999/")
            .with_api_version("v1");
        assert_eq!(
            gemini.generate_content_url("gemini-2.5-flash"),
            "http://127.0.0.1:9999/v1/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn map_http_error_enhances_a_429_into_resource_exhausted() {
        match map_http_error(429, "quota exceeded".to_string()) {
            GeminiCallError::ResourceExhausted { body } => assert_eq!(body, "quota exceeded"),
            _ => panic!("expected ResourceExhausted"),
        }
    }

    #[test]
    fn map_http_error_passes_through_other_statuses() {
        match map_http_error(500, "boom".to_string()) {
            GeminiCallError::Http { status, body } => {
                assert_eq!(status, 500);
                assert_eq!(body, "boom");
            }
            _ => panic!("expected Http"),
        }
    }

    /// A one-shot local HTTP/1.1 server: accepts a single request, reads it
    /// fully (header-aware, honoring `Content-Length`), then replies with
    /// `status_line`/`body`. Dependency-free stand-in for a mock HTTP
    /// server, giving `generate_content` real, end-to-end transport
    /// coverage without a live call to Google's API.
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
            let response =
                format!("{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });
        (format!("http://{addr}"), handle)
    }

    #[rusty_tokio::test]
    // Held across `.await` deliberately: this guard only serializes test
    // env-var mutation against other tests in this file, and the awaited
    // future never touches `ENV_VAR_GUARD` itself, so there's no deadlock
    // risk — only the intended cross-test isolation.
    #[allow(clippy::await_holding_lock)]
    async fn generate_content_parses_a_successful_response() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_API_KEY", "test-key");

        let (base_url, server) = spawn_one_shot_server(
            "HTTP/1.1 200 OK",
            r#"{"modelVersion":"gemini-2.5-flash","candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}]}"#,
        );
        let gemini = Gemini::new("gemini-2.5-flash").with_base_url(base_url);
        let request = LlmRequest::new("gemini-2.5-flash");
        let response = gemini.generate_content(&request).await;
        clear_auth_env_vars();
        server.join().unwrap();

        let response = response.unwrap();
        assert_eq!(
            response.content.unwrap().parts[0].text.as_deref(),
            Some("hello")
        );
    }

    #[rusty_tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn generate_content_maps_a_429_response_to_resource_exhausted() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        std::env::set_var("GOOGLE_API_KEY", "test-key");

        let (base_url, server) =
            spawn_one_shot_server("HTTP/1.1 429 Too Many Requests", "quota exceeded");
        let gemini = Gemini::new("gemini-2.5-flash").with_base_url(base_url);
        let request = LlmRequest::new("gemini-2.5-flash");
        let response = gemini.generate_content(&request).await;
        clear_auth_env_vars();
        server.join().unwrap();

        match response {
            Err(GeminiCallError::ResourceExhausted { .. }) => {}
            _ => panic!("expected ResourceExhausted"),
        }
    }

    #[rusty_tokio::test]
    async fn generate_content_errors_without_a_model_name() {
        let gemini = Gemini::new("gemini-2.5-flash");
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.model = None;
        match gemini.generate_content(&request).await {
            Err(GeminiCallError::MissingModel) => {}
            _ => panic!("expected MissingModel"),
        }
    }

    #[rusty_tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn base_llm_generate_content_async_flattens_gemini_call_errors() {
        let _guard = ENV_VAR_GUARD.lock().unwrap();
        clear_auth_env_vars();
        let gemini = Gemini::new("gemini-2.5-flash");
        let request = LlmRequest::new("gemini-2.5-flash");
        let result = BaseLlm::generate_content_async(&gemini, &request, false).await;
        clear_auth_env_vars();
        match result {
            Err(crate::base_llm::BaseLlmError::CallFailed(message)) => {
                assert!(message.contains("GOOGLE_API_KEY"));
            }
            _ => panic!("expected CallFailed"),
        }
    }

    #[rusty_tokio::test]
    async fn base_llm_generate_content_async_rejects_streaming_for_now() {
        let gemini = Gemini::new("gemini-2.5-flash");
        let request = LlmRequest::new("gemini-2.5-flash");
        let result = BaseLlm::generate_content_async(&gemini, &request, true).await;
        match result {
            Err(crate::base_llm::BaseLlmError::CallFailed(message)) => {
                assert!(message.contains("streaming"));
            }
            _ => panic!("expected CallFailed"),
        }
    }
}
