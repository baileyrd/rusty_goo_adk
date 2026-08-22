//! Capabilities C0123, C0124, C0129, C0130 (partial C0133): `Gemini`, ported
//! from `google.adk.models.google_llm`.
//!
//! **Scope of this batch**: only the config shape, `supported_models()`,
//! base-URL/API-version resolution, and API-client construction — the parts
//! that are pure configuration logic, testable without a live network call.
//! Deferred to later batches, each needing its own foundational decision or
//! wire-format work this batch doesn't have yet:
//!   - C0125/C0126/C0128 (`generate_content_async`'s actual REST/SSE calls,
//!     context-cache integration, interactions-API delegation) — need the
//!     real `GenerateContentConfig`/`GenerateContentResponse`/`Tool`/
//!     `FunctionDeclaration` wire types (today's `LlmRequest`/`LlmResponse`
//!     only model the load-bearing subset ADK's own code reads/writes, not
//!     the full request/response bodies a real HTTP call would send/parse).
//!   - C0127 (`_ResourceExhaustedError`) — wraps an HTTP client error that
//!     only exists once C0125 makes real calls.
//!   - C0131/C0132/C0135-C0139 (Live `connect()`, computer-use/preprocess
//!     adaptation, `GeminiLlmConnection`) — need a WebSocket transport
//!     decision (this batch only decided the REST/SSE transport).
//!   - C0134 (redacted debug request/response logging) — needs the real
//!     wire types above to have fields worth redacting.
//!   - C0140-C0143 (`GeminiContextCacheManager`) — needs a SHA-256 crate
//!     decision plus the cache-creation HTTP call.
//!
//! **Adaptation**: the source's `api_client` is a `cached_property`
//! returning a full `google.genai.Client` (itself wrapping `httpx`/
//! `aiohttp`, ADC-based Vertex AI auth, retries, etc.). [`GeminiApiClient`]
//! here is the Rust-native equivalent scoped to what's decidable without
//! the real wire calls: a `reqwest::Client` pre-loaded with tracking
//! headers, plus the resolved base URL/API version/`enterprise` flag.
//! `client_kwargs` (the source's free-form `dict` merged into the SDK
//! constructor's kwargs, capable of overriding *any* constructor argument)
//! has no well-typed Rust equivalent without knowing which keys matter, so
//! it stays an inert opaque placeholder — like `tools_dict` in
//! `llm_request.rs`, documented rather than silently dropped.

use std::sync::{Arc, OnceLock};

use regex::Regex;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use rusty_serde::value::Value;

use crate::base_llm::BaseLlm;
use crate::capabilities::{
    gemini_output_schema_and_tools, get_google_llm_variant, GoogleLlmVariant,
};
use crate::google_client_headers::get_tracking_headers;

const DEFAULT_MODEL: &str = "gemini-2.5-flash";
const API_VERSION_ENV_VAR: &str = "GOOGLE_GENAI_API_VERSION";

fn version_suffix_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"^/?(v[0-9][a-z0-9.-]*)/?$").expect("valid regex"))
}

/// The Rust-native stand-in for a constructed `google.genai.Client` — see
/// the module doc's adaptation note. Not constructed directly; built by
/// [`Gemini::api_client`].
pub struct GeminiApiClient {
    pub http: reqwest::Client,
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

fn build_http_client(headers: &[(String, String)]) -> reqwest::Client {
    let mut header_map = HeaderMap::new();
    for (key, value) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            header_map.insert(name, value);
        }
    }
    reqwest::Client::builder()
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
            http: reqwest::Client::new(),
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
}
