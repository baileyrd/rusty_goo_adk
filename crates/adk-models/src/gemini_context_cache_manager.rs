//! Capabilities C0140-C0143: `GeminiContextCacheManager`, ported from
//! `google.adk.models.gemini_context_cache_manager`.
//!
//! **Adaptation, disclosed**: the source is constructed with a full
//! `google.genai.Client` (`GeminiContextCacheManager(self.api_client)`),
//! which already carries its own transport, base URL, and auth internally
//! — the manager itself never resolves auth. This port has no such
//! all-in-one client (see `gemini.rs`'s own adaptation note), so
//! [`GeminiContextCacheManager::new`] takes the same [`GeminiApiClient`]
//! `Gemini::api_client()` builds *plus* the resolved backend variant and
//! auth header a caller would compute the same way
//! `Gemini::resolve_auth_header` does — the manager is the request-scoped
//! object `google_llm.py`'s `generate_content_async` constructs fresh per
//! call (`GeminiContextCacheManager(self.api_client)`), so passing an
//! already-resolved auth header in is equivalent, not a shortcut.
//!
//! **Scope decision**: wiring this manager into `Gemini::generate_content_async`
//! (the source's `if llm_request.cache_config and not self.use_interactions_api`
//! block) is C0126 (context-cache integration), already noted as deferred in
//! `gemini.rs`'s module doc — this batch builds the manager itself,
//! independently testable against a local mock `cachedContents` endpoint,
//! the same "build it standalone, wire it in later" split used for
//! `GeminiLlmConnection` versus `Gemini::connect()`.
//!
//! **Adaptation, disclosed**: `_cache_scope`'s `project`/`location` keys
//! (Vertex-AI-only) are never populated — [`GeminiApiClient`] doesn't model
//! a Vertex AI project/location (real Vertex AI calls aren't supported yet
//! at all, see `gemini.rs`'s `GeminiCallError::VertexAiAuthNotSupported`),
//! so there's nothing to read. `backend`/`base_url` are.
//!
//! **Adaptation, disclosed**: the fingerprint's canonical JSON is a
//! Rust-native SHA-256 digest over a deterministically field-ordered
//! [`rusty_serde::value::Value`] map, not the source's
//! `json.dumps(..., sort_keys=True)` over a Python dict. The two will never
//! produce byte-identical strings, but the fingerprint is only ever compared
//! against a fingerprint this same code produced earlier
//! (`current_fingerprint == old_cache_metadata.fingerprint`) — it never
//! crosses the Rust/Python boundary — so only internal determinism matters:
//! identical logical request state must hash identically, and it does,
//! since the field order and each field's serialization are both fixed.
//! `tools`/`tool_config` stay opaque `Value`s (not modeled, C0116) so the
//! source's reordering-tolerant canonicalization (sorting function
//! declarations by name) isn't reproduced — a reordered-but-equivalent
//! tools list will (safely) miss the cache rather than (unsafely) hit a
//! stale one, which is the fail-safe direction.
//!
//! **Adaptation, disclosed**: `_estimate_request_tokens`'s per-tool
//! character count (`for tool in config.tools: if isinstance(tool,
//! types.Tool): ...`) is approximated by serializing the whole opaque
//! `tools` value once, for the same reason — it isn't modeled as a real
//! list of `Tool`s yet. Still a "rough estimate" either way, per the
//! source's own docstring.
//!
//! **Adaptation, disclosed**: `expire_time` is parsed from the Gemini API's
//! RFC 3339 `expireTime` response field via `rusty_time::DateTime::parse`,
//! which truncates to whole seconds (no sub-second fraction) — the source's
//! `datetime.timestamp()` keeps microsecond precision. `CacheMetadata`'s
//! 2-minute `expire_soon` buffer (and this manager's own expiry check)
//! swallow a sub-second difference many times over.

use std::sync::Arc;

use adk_genai::content::{Content, Part};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache_metadata::CacheMetadata;
use crate::capabilities::GoogleLlmVariant;
use crate::gemini::{GeminiApiClient, DEFAULT_GEMINI_API_BASE_URL, DEFAULT_GEMINI_API_VERSION};
use crate::llm_request::LlmRequest;
use crate::llm_response::LlmResponse;

// Named Gemini model families have documented explicit-cache floors. For
// opaque tuned-model/endpoint IDs, the server remains authoritative.
const GEMINI_2_5_MIN_CACHE_TOKENS: i64 = 2048;
const GEMINI_3_MIN_CACHE_TOKENS: i64 = 4096;

/// C0142: `_minimum_cache_tokens` — the explicit-cache token floor for a
/// named Gemini model.
fn minimum_cache_tokens(model: Option<&str>) -> Option<i64> {
    let model_name = model.unwrap_or("").rsplit('/').next().unwrap_or("");
    if model_name.starts_with("gemini-2.5-") {
        Some(GEMINI_2_5_MIN_CACHE_TOKENS)
    } else if model_name.starts_with("gemini-3") {
        Some(GEMINI_3_MIN_CACHE_TOKENS)
    } else {
        None
    }
}

fn require_model(llm_request: &LlmRequest) -> Result<&str, GeminiContextCacheError> {
    llm_request
        .model
        .as_deref()
        .ok_or(GeminiContextCacheError::MissingModel)
}

fn require_cache_config(
    llm_request: &LlmRequest,
) -> Result<&adk_agents::context_cache_config::ContextCacheConfig, GeminiContextCacheError> {
    llm_request
        .cache_config
        .as_ref()
        .ok_or(GeminiContextCacheError::MissingCacheConfig)
}

/// Errors from [`GeminiContextCacheManager`]'s cache-creation/cleanup HTTP
/// calls, plus the two "shouldn't happen" invariant violations the source
/// guards with a `RuntimeError`.
#[derive(Debug, rusty_err::Error)]
pub enum GeminiContextCacheError {
    #[error("Context caching requires a model name.")]
    MissingModel,
    #[error("Context caching requires a cache configuration.")]
    MissingCacheConfig,
    #[error("A valid cache must have active metadata.")]
    ValidCacheMissingName,
    #[error("A newly created cache must be active.")]
    NewCacheMissingName,
    #[error("request to the cache API failed: {0}")]
    Transport(String),
    #[error("cache API returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("failed to parse the cache API response: {0}")]
    Parse(String),
    #[error("the cache service returned no cache name")]
    MissingCacheName,
}

#[derive(Debug, Clone, Default, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
struct CreateCachedContentBody {
    model: String,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    contents: Option<Vec<Content>>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    tool_config: Option<Value>,
    ttl: String,
    display_name: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
struct CachedContentResource {
    #[rusty_serde(default)]
    name: Option<String>,
    #[rusty_serde(default)]
    expire_time: Option<String>,
}

/// C0140: manages the explicit context-cache lifecycle for Gemini models —
/// see the module doc for how construction differs from the source.
pub struct GeminiContextCacheManager {
    client: Arc<GeminiApiClient>,
    backend: GoogleLlmVariant,
    auth_header: Option<(&'static str, String)>,
}

impl GeminiContextCacheManager {
    pub fn new(
        client: Arc<GeminiApiClient>,
        backend: GoogleLlmVariant,
        auth_header: Option<(&'static str, String)>,
    ) -> Self {
        Self {
            client,
            backend,
            auth_header,
        }
    }

    /// C0141: the `handle_context_caching` state machine — reuse-valid →
    /// invalid-cleanup-recreate → fingerprint-mismatch → fresh-fingerprint-only
    /// → no-prior-metadata → fresh-fingerprint-only. Mutates `llm_request` in
    /// place when a cache is applied, matching the source.
    pub async fn handle_context_caching(
        &self,
        llm_request: &mut LlmRequest,
    ) -> Result<Option<CacheMetadata>, GeminiContextCacheError> {
        require_model(llm_request)?;
        require_cache_config(llm_request)?;

        let Some(existing) = llm_request.cache_metadata.clone() else {
            let cache_contents_count = self.find_count_of_contents_to_cache(&llm_request.contents);
            let fingerprint = self.generate_cache_fingerprint(llm_request, cache_contents_count);
            return Ok(Some(fingerprint_only(fingerprint, cache_contents_count)));
        };

        if self.is_cache_valid(llm_request, &existing) {
            let cache_name = existing
                .cache_name
                .clone()
                .ok_or(GeminiContextCacheError::ValidCacheMissingName)?;
            let cache_contents_count = existing.contents_count as usize;
            self.apply_cache_to_request(llm_request, cache_name, cache_contents_count);
            return Ok(Some(existing));
        }

        if let Some(cache_name) = &existing.cache_name {
            self.cleanup_cache(cache_name).await;
        }

        let previous_cache_contents_count = existing.contents_count as usize;
        let current_fingerprint =
            self.generate_cache_fingerprint(llm_request, previous_cache_contents_count);

        if current_fingerprint == existing.fingerprint {
            let current_cacheable_contents_count =
                self.find_count_of_contents_to_cache(&llm_request.contents);
            let cache_contents_count =
                previous_cache_contents_count.max(current_cacheable_contents_count);
            let current_fingerprint =
                self.generate_cache_fingerprint(llm_request, cache_contents_count);
            let cache_metadata = self
                .create_new_cache_with_contents(llm_request, cache_contents_count)
                .await;
            if let Some(cache_metadata) = cache_metadata {
                let cache_name = cache_metadata
                    .cache_name
                    .clone()
                    .ok_or(GeminiContextCacheError::NewCacheMissingName)?;
                self.apply_cache_to_request(llm_request, cache_name, cache_contents_count);
                return Ok(Some(cache_metadata));
            }
            return Ok(Some(fingerprint_only(
                current_fingerprint,
                cache_contents_count,
            )));
        }

        let cache_contents_count = self.find_count_of_contents_to_cache(&llm_request.contents);
        let fingerprint = self.generate_cache_fingerprint(llm_request, cache_contents_count);
        Ok(Some(fingerprint_only(fingerprint, cache_contents_count)))
    }

    /// C0142: `_find_count_of_contents_to_cache` — cache everything before
    /// the last continuous batch of user contents, so there's always some
    /// uncached user content left to send.
    fn find_count_of_contents_to_cache(&self, contents: &[Content]) -> usize {
        if contents.is_empty() {
            return 0;
        }
        let mut last_user_batch_start = contents.len();
        for i in (0..contents.len()).rev() {
            if contents[i].role.as_deref() == Some("user") {
                last_user_batch_start = i;
            } else {
                break;
            }
        }
        last_user_batch_start
    }

    /// C0142: `_is_cache_valid` — active (not fingerprint-only), unexpired,
    /// within its configured invocation interval, and fingerprint-compatible.
    fn is_cache_valid(&self, llm_request: &LlmRequest, cache_metadata: &CacheMetadata) -> bool {
        let (Some(_), Some(expire_time), Some(invocations_used)) = (
            cache_metadata.cache_name.as_ref(),
            cache_metadata.expire_time,
            cache_metadata.invocations_used,
        ) else {
            return false;
        };
        let Some(cache_config) = llm_request.cache_config.as_ref() else {
            return false;
        };

        if adk_platform::time::get_time() >= expire_time {
            return false;
        }
        if invocations_used > cache_config.cache_intervals {
            return false;
        }

        self.generate_cache_fingerprint(llm_request, cache_metadata.contents_count as usize)
            == cache_metadata.fingerprint
    }

    /// C0142: `_generate_cache_fingerprint` — see the module doc's
    /// disclosed adaptation for why this doesn't need to match the
    /// source's own hash byte-for-byte.
    fn generate_cache_fingerprint(
        &self,
        llm_request: &LlmRequest,
        cache_contents_count: usize,
    ) -> String {
        let mut fields: Vec<(String, Value)> = vec![
            (
                "model".to_string(),
                llm_request
                    .model
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            ),
            ("cache_scope".to_string(), self.cache_scope_value()),
        ];

        if let Some(system_instruction) = &llm_request.config.system_instruction {
            fields.push((
                "system_instruction".to_string(),
                Value::String(system_instruction.clone()),
            ));
        }
        if let Some(tools) = &llm_request.config.tools {
            fields.push(("tools".to_string(), tools.clone()));
        }
        if let Some(tool_config) = &llm_request.config.tool_config {
            fields.push(("tool_config".to_string(), tool_config.clone()));
        }
        if cache_contents_count > 0 && !llm_request.contents.is_empty() {
            let n = cache_contents_count.min(llm_request.contents.len());
            let contents_data: Vec<Value> = llm_request.contents[..n]
                .iter()
                .map(|c| rusty_serde::json::to_value(c).expect("Content always serializes"))
                .collect();
            fields.push(("cached_contents".to_string(), Value::Seq(contents_data)));
        }

        let fingerprint_str = rusty_serde::json::to_string(&Value::Map(fields))
            .expect("a Value tree of strings/opaque values always serializes");
        let digest = Sha256::digest(fingerprint_str.as_bytes());
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        hex[..16].to_string()
    }

    /// C0143: `_cache_scope` — see the module doc's disclosed adaptation
    /// for why `project`/`location` never appear.
    fn cache_scope_value(&self) -> Value {
        let mut entries = vec![(
            "backend".to_string(),
            Value::String(
                match self.backend {
                    GoogleLlmVariant::VertexAi => "vertex",
                    GoogleLlmVariant::GeminiApi => "gemini",
                }
                .to_string(),
            ),
        )];
        if let Some(base_url) = &self.client.base_url {
            entries.push(("base_url".to_string(), Value::String(base_url.clone())));
        }
        Value::Map(entries)
    }

    /// C0143: `_estimate_request_tokens` — a rough character-count-based
    /// estimate; see the module doc's disclosed adaptation for the `tools`
    /// simplification.
    fn estimate_request_tokens(
        &self,
        llm_request: &LlmRequest,
        cache_contents_count: Option<usize>,
    ) -> i64 {
        let mut total_chars: i64 = 0;

        if let Some(system_instruction) = &llm_request.config.system_instruction {
            total_chars += system_instruction.chars().count() as i64;
        }
        if let Some(tools) = &llm_request.config.tools {
            if let Ok(tool_str) = rusty_serde::json::to_string(tools) {
                total_chars += tool_str.chars().count() as i64;
            }
        }

        let contents: &[Content] = match cache_contents_count {
            Some(n) => &llm_request.contents[..n.min(llm_request.contents.len())],
            None => &llm_request.contents,
        };
        for content in contents {
            for part in &content.parts {
                if let Some(text) = &part.text {
                    total_chars += text.chars().count() as i64;
                }
            }
        }

        total_chars / 4
    }

    /// C0143: `_estimate_cacheable_prefix_tokens` — scales the one accurate
    /// token count available (the previous full-prompt count) down to the
    /// cacheable prefix's estimated share.
    fn estimate_cacheable_prefix_tokens(
        &self,
        llm_request: &LlmRequest,
        cache_contents_count: usize,
    ) -> i64 {
        let Some(full_tokens) = llm_request.cacheable_contents_token_count else {
            return 0;
        };
        if full_tokens == 0 {
            return 0;
        }

        let full_estimate = self.estimate_request_tokens(llm_request, None);
        if full_estimate <= 0 {
            return full_tokens;
        }

        let prefix_estimate = self.estimate_request_tokens(llm_request, Some(cache_contents_count));
        let ratio = (prefix_estimate as f64 / full_estimate as f64).min(1.0);
        (full_tokens as f64 * ratio) as i64
    }

    /// C0143: `_create_new_cache_with_contents` — gated on the previous
    /// response's token count clearing both `cache_config.min_tokens` and
    /// the model's own minimum cache-token floor (C0142).
    async fn create_new_cache_with_contents(
        &self,
        llm_request: &LlmRequest,
        cache_contents_count: usize,
    ) -> Option<CacheMetadata> {
        let cache_config = llm_request.cache_config.as_ref()?;

        let Some(cacheable_contents_token_count) = llm_request.cacheable_contents_token_count
        else {
            eprintln!(
                "info: no previous token count available, skipping cache creation for initial \
                 request"
            );
            return None;
        };
        if cacheable_contents_token_count < cache_config.min_tokens as i64 {
            eprintln!(
                "info: previous request too small for caching ({cacheable_contents_token_count} \
                 < {} tokens)",
                cache_config.min_tokens
            );
            return None;
        }

        let cacheable_prefix_tokens =
            self.estimate_cacheable_prefix_tokens(llm_request, cache_contents_count);
        if let Some(minimum_cache_tokens) = minimum_cache_tokens(llm_request.model.as_deref()) {
            if cacheable_prefix_tokens < minimum_cache_tokens {
                eprintln!(
                    "info: cacheable prefix below Gemini minimum cache size \
                     ({cacheable_prefix_tokens} < {minimum_cache_tokens} tokens)"
                );
                return None;
            }
        }

        match self
            .create_gemini_cache(llm_request, cache_contents_count)
            .await
        {
            Ok(metadata) => Some(metadata),
            Err(e) => {
                eprintln!("warning: failed to create cache: {e}");
                None
            }
        }
    }

    fn cached_contents_url(&self) -> String {
        let base_url = self
            .client
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_BASE_URL.to_string());
        let api_version = self
            .client
            .api_version
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_VERSION.to_string());
        let base_url = base_url.trim_end_matches('/');
        format!("{base_url}/{api_version}/cachedContents")
    }

    fn cached_content_url(&self, cache_name: &str) -> String {
        let base_url = self
            .client
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_BASE_URL.to_string());
        let api_version = self
            .client
            .api_version
            .clone()
            .unwrap_or_else(|| DEFAULT_GEMINI_API_VERSION.to_string());
        let base_url = base_url.trim_end_matches('/');
        format!("{base_url}/{api_version}/{cache_name}")
    }

    /// C0143: `_create_gemini_cache` — a real `POST .../cachedContents`
    /// call, returning cache metadata with a precise creation timestamp
    /// (matching the source's "set right after creation" comment).
    async fn create_gemini_cache(
        &self,
        llm_request: &LlmRequest,
        cache_contents_count: usize,
    ) -> Result<CacheMetadata, GeminiContextCacheError> {
        let cache_config = require_cache_config(llm_request)?;
        let model = require_model(llm_request)?.to_string();

        let cache_contents = if cache_contents_count > 0 {
            let n = cache_contents_count.min(llm_request.contents.len());
            Some(llm_request.contents[..n].to_vec())
        } else {
            None
        };
        let system_instruction =
            llm_request
                .config
                .system_instruction
                .as_ref()
                .map(|text| Content {
                    role: None,
                    parts: vec![Part::text(text.clone())],
                });
        let display_name = format!(
            "adk-cache-{}-{cache_contents_count}contents",
            adk_platform::time::get_time() as i64
        );

        let body = CreateCachedContentBody {
            model,
            contents: cache_contents,
            system_instruction,
            tools: llm_request.config.tools.clone(),
            tool_config: llm_request.config.tool_config.clone(),
            ttl: cache_config.ttl_string(),
            display_name,
        };
        let body_json = rusty_serde::json::to_string(&body)
            .map_err(|e| GeminiContextCacheError::Parse(e.to_string()))?;

        let url = self.cached_contents_url();
        let auth_header = self.auth_header.clone();
        let client = self.client.clone();

        // Same `reqwest::blocking` + `rusty_tokio::spawn_blocking` bridge as
        // `gemini.rs`'s `generate_content` — see that module's load-bearing
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
            Ok(Err(message)) => return Err(GeminiContextCacheError::Transport(message)),
            Err(join_error) => {
                return Err(GeminiContextCacheError::Transport(join_error.to_string()))
            }
        };

        if !(200..300).contains(&status) {
            return Err(GeminiContextCacheError::Http { status, body: text });
        }

        let created_at = adk_platform::time::get_time();
        let parsed: CachedContentResource = rusty_serde::json::from_str(&text)
            .map_err(|e| GeminiContextCacheError::Parse(e.to_string()))?;
        let expire_time = parsed
            .expire_time
            .as_deref()
            .and_then(|s| rusty_time::DateTime::parse(s).ok())
            .map(|dt| dt.timestamp() as f64)
            .unwrap_or(created_at + cache_config.ttl_seconds as f64);
        let cache_name = parsed
            .name
            .ok_or(GeminiContextCacheError::MissingCacheName)?;
        let fingerprint = self.generate_cache_fingerprint(llm_request, cache_contents_count);

        Ok(CacheMetadata::new(
            Some(cache_name),
            Some(expire_time),
            fingerprint,
            Some(1),
            cache_contents_count as u32,
            Some(created_at),
        )
        .expect("a freshly created cache always sets all three active-state fields"))
    }

    /// C0143: `cleanup_cache` — a best-effort `DELETE`; failures are logged,
    /// never propagated (matching the source's broad `except Exception`).
    pub async fn cleanup_cache(&self, cache_name: &str) {
        let url = self.cached_content_url(cache_name);
        let auth_header = self.auth_header.clone();
        let client = self.client.clone();
        let cache_name = cache_name.to_string();

        let outcome = rusty_tokio::spawn_blocking(move || -> Result<(u16, String), String> {
            let mut request = client.http.delete(&url);
            if let Some((name, value)) = auth_header {
                request = request.header(name, value);
            }
            let response = request.send().map_err(|e| e.to_string())?;
            let status = response.status().as_u16();
            let text = response.text().unwrap_or_default();
            Ok((status, text))
        })
        .await;

        match outcome {
            Ok(Ok((status, _))) if (200..300).contains(&status) => {}
            Ok(Ok((status, body))) => {
                eprintln!("warning: failed to cleanup cache {cache_name}: HTTP {status}: {body}")
            }
            Ok(Err(message)) => {
                eprintln!("warning: failed to cleanup cache {cache_name}: {message}")
            }
            Err(join_error) => {
                eprintln!("warning: failed to cleanup cache {cache_name}: {join_error}")
            }
        }
    }

    /// C0143: `_apply_cache_to_request` — removes the now-cached fields from
    /// the request config and truncates `contents` to the uncached suffix.
    fn apply_cache_to_request(
        &self,
        llm_request: &mut LlmRequest,
        cache_name: String,
        cache_contents_count: usize,
    ) {
        llm_request.config.system_instruction = None;
        llm_request.config.tools = None;
        llm_request.config.tool_config = None;
        llm_request.config.cached_content = Some(cache_name);

        let n = cache_contents_count.min(llm_request.contents.len());
        llm_request.contents = llm_request.contents[n..].to_vec();
    }

    /// C0143: `populate_cache_metadata_in_response`.
    pub fn populate_cache_metadata_in_response(
        &self,
        llm_response: &mut LlmResponse,
        cache_metadata: &CacheMetadata,
    ) {
        llm_response.cache_metadata = Some(cache_metadata.clone());
    }
}

fn fingerprint_only(fingerprint: String, contents_count: usize) -> CacheMetadata {
    CacheMetadata::new(None, None, fingerprint, None, contents_count as u32, None)
        .expect("fingerprint-only metadata is always internally consistent")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_request::Instructions;
    use std::io::{Read, Write};
    use std::time::Duration;

    fn gemini_api_client(base_url: String) -> Arc<GeminiApiClient> {
        Arc::new(GeminiApiClient {
            http: reqwest::blocking::Client::new(),
            base_url: Some(base_url),
            api_version: Some("v1beta".to_string()),
            headers: Vec::new(),
            retry_options: None,
            enterprise: false,
        })
    }

    fn manager(base_url: String) -> GeminiContextCacheManager {
        GeminiContextCacheManager::new(
            gemini_api_client(base_url),
            GoogleLlmVariant::GeminiApi,
            Some(("x-goog-api-key", "test-key".to_string())),
        )
    }

    fn cache_config() -> adk_agents::context_cache_config::ContextCacheConfig {
        adk_agents::context_cache_config::ContextCacheConfig {
            cache_intervals: 10,
            ttl_seconds: 1800,
            min_tokens: 100,
            create_http_options: None,
        }
    }

    fn request_with_two_turns(model: &str) -> LlmRequest {
        let mut request = LlmRequest::new(model);
        request.cache_config = Some(cache_config());
        request.contents.push(Content::user_text("hello"));
        request.contents.push(Content::new(
            "model",
            vec![Part::text("hi there, how can I help?")],
        ));
        request
            .contents
            .push(Content::user_text("what's the weather?"));
        request
    }

    // --- find_count_of_contents_to_cache ---

    #[test]
    fn find_count_is_zero_for_empty_contents() {
        let manager = manager("http://example.invalid".to_string());
        assert_eq!(manager.find_count_of_contents_to_cache(&[]), 0);
    }

    #[test]
    fn find_count_caches_everything_before_the_trailing_user_batch() {
        let manager = manager("http://example.invalid".to_string());
        let contents = vec![
            Content::user_text("a"),
            Content::new("model", vec![Part::text("b")]),
            Content::user_text("c"),
            Content::user_text("d"),
        ];
        assert_eq!(manager.find_count_of_contents_to_cache(&contents), 2);
    }

    #[test]
    fn find_count_is_zero_when_every_content_is_a_user_content() {
        let manager = manager("http://example.invalid".to_string());
        let contents = vec![Content::user_text("a"), Content::user_text("b")];
        assert_eq!(manager.find_count_of_contents_to_cache(&contents), 0);
    }

    // --- fingerprint ---

    #[test]
    fn fingerprint_is_stable_for_identical_requests() {
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let a = manager.generate_cache_fingerprint(&request, 2);
        let b = manager.generate_cache_fingerprint(&request, 2);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn fingerprint_changes_when_cached_contents_differ() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        let a = manager.generate_cache_fingerprint(&request, 2);
        request.contents[0] = Content::user_text("a completely different opener");
        let b = manager.generate_cache_fingerprint(&request, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_changes_when_the_model_differs() {
        let manager = manager("http://example.invalid".to_string());
        let request_a = request_with_two_turns("gemini-2.5-flash");
        let request_b = request_with_two_turns("gemini-3-pro");
        assert_ne!(
            manager.generate_cache_fingerprint(&request_a, 2),
            manager.generate_cache_fingerprint(&request_b, 2)
        );
    }

    #[test]
    fn fingerprint_changes_when_system_instruction_differs() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        let without = manager.generate_cache_fingerprint(&request, 2);
        request.append_instructions(Instructions::Strings(vec!["be terse".to_string()]));
        let with = manager.generate_cache_fingerprint(&request, 2);
        assert_ne!(without, with);
    }

    #[test]
    fn fingerprint_is_unaffected_by_backend_base_url_alone() {
        let request = request_with_two_turns("gemini-2.5-flash");
        let a = manager("http://one.invalid".to_string()).generate_cache_fingerprint(&request, 2);
        let b = manager("http://two.invalid".to_string()).generate_cache_fingerprint(&request, 2);
        assert_ne!(
            a, b,
            "base_url is part of cache_scope and should change the fingerprint"
        );
    }

    // --- minimum_cache_tokens ---

    #[test]
    fn minimum_cache_tokens_for_gemini_2_5_family() {
        assert_eq!(minimum_cache_tokens(Some("gemini-2.5-flash")), Some(2048));
    }

    #[test]
    fn minimum_cache_tokens_for_gemini_3_family() {
        assert_eq!(
            minimum_cache_tokens(Some("gemini-3-pro-preview")),
            Some(4096)
        );
    }

    #[test]
    fn minimum_cache_tokens_is_none_for_an_opaque_tuned_model() {
        assert_eq!(
            minimum_cache_tokens(Some("projects/p/locations/l/endpoints/123")),
            None
        );
    }

    // --- is_cache_valid ---

    #[test]
    fn fingerprint_only_metadata_is_never_a_valid_active_cache() {
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        let metadata = CacheMetadata::new(None, None, fp, None, 2, None).unwrap();
        assert!(!manager.is_cache_valid(&request, &metadata));
    }

    #[test]
    fn an_active_cache_with_a_matching_fingerprint_and_time_is_valid() {
        adk_platform::time::set_time_provider(|| 1_000.0);
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        let metadata = CacheMetadata::new(
            Some("cachedContents/abc".to_string()),
            Some(2_000.0),
            fp,
            Some(1),
            2,
            Some(1_000.0),
        )
        .unwrap();
        assert!(manager.is_cache_valid(&request, &metadata));
        adk_platform::time::reset_time_provider();
    }

    #[test]
    fn an_expired_active_cache_is_invalid() {
        adk_platform::time::set_time_provider(|| 5_000.0);
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        let metadata = CacheMetadata::new(
            Some("cachedContents/abc".to_string()),
            Some(2_000.0),
            fp,
            Some(1),
            2,
            Some(1_000.0),
        )
        .unwrap();
        assert!(!manager.is_cache_valid(&request, &metadata));
        adk_platform::time::reset_time_provider();
    }

    #[test]
    fn a_cache_past_its_invocation_interval_is_invalid() {
        adk_platform::time::set_time_provider(|| 1_000.0);
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        let metadata = CacheMetadata::new(
            Some("cachedContents/abc".to_string()),
            Some(2_000.0),
            fp,
            Some(11),
            2,
            Some(1_000.0),
        )
        .unwrap();
        assert!(!manager.is_cache_valid(&request, &metadata));
        adk_platform::time::reset_time_provider();
    }

    #[test]
    fn a_cache_with_a_stale_fingerprint_is_invalid() {
        adk_platform::time::set_time_provider(|| 1_000.0);
        let manager = manager("http://example.invalid".to_string());
        let request = request_with_two_turns("gemini-2.5-flash");
        let metadata = CacheMetadata::new(
            Some("cachedContents/abc".to_string()),
            Some(2_000.0),
            "stale-fingerprint".to_string(),
            Some(1),
            2,
            Some(1_000.0),
        )
        .unwrap();
        assert!(!manager.is_cache_valid(&request, &metadata));
        adk_platform::time::reset_time_provider();
    }

    // --- apply_cache_to_request ---

    #[test]
    fn apply_cache_truncates_contents_and_clears_config_fields() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        request.config.tools = Some(Value::String("some-tool".to_string()));
        request.config.tool_config = Some(Value::String("some-tool-config".to_string()));
        manager.apply_cache_to_request(&mut request, "cachedContents/abc".to_string(), 2);

        assert_eq!(request.contents.len(), 1);
        assert_eq!(
            request.contents[0].parts[0].text.as_deref(),
            Some("what's the weather?")
        );
        assert!(request.config.system_instruction.is_none());
        assert!(request.config.tools.is_none());
        assert!(request.config.tool_config.is_none());
        assert_eq!(
            request.config.cached_content.as_deref(),
            Some("cachedContents/abc")
        );
    }

    // --- populate_cache_metadata_in_response ---

    #[test]
    fn populate_cache_metadata_copies_it_into_the_response() {
        let manager = manager("http://example.invalid".to_string());
        let mut response = LlmResponse::default();
        let metadata = CacheMetadata::new(None, None, "fp".to_string(), None, 0, None).unwrap();
        manager.populate_cache_metadata_in_response(&mut response, &metadata);
        assert_eq!(response.cache_metadata, Some(metadata));
    }

    // --- handle_context_caching: no prior metadata ---

    #[rusty_tokio::test]
    async fn no_prior_metadata_returns_fingerprint_only_metadata() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        let result = manager
            .handle_context_caching(&mut request)
            .await
            .unwrap()
            .unwrap();
        assert!(result.cache_name.is_none());
        assert_eq!(result.contents_count, 2);
        // No cache was applied — contents are untouched.
        assert_eq!(request.contents.len(), 3);
    }

    #[rusty_tokio::test]
    async fn missing_model_is_an_error() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = LlmRequest::default();
        request.cache_config = Some(cache_config());
        let result = manager.handle_context_caching(&mut request).await;
        assert!(matches!(result, Err(GeminiContextCacheError::MissingModel)));
    }

    #[rusty_tokio::test]
    async fn missing_cache_config_is_an_error() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let result = manager.handle_context_caching(&mut request).await;
        assert!(matches!(
            result,
            Err(GeminiContextCacheError::MissingCacheConfig)
        ));
    }

    // --- handle_context_caching: reuse a valid active cache ---

    #[rusty_tokio::test]
    async fn a_valid_active_cache_is_reused_and_applied() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        adk_platform::time::set_time_provider(|| 1_000.0);
        request.cache_metadata = Some(
            CacheMetadata::new(
                Some("cachedContents/abc".to_string()),
                Some(2_000.0),
                fp,
                Some(1),
                2,
                Some(1_000.0),
            )
            .unwrap(),
        );

        let result = manager
            .handle_context_caching(&mut request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.cache_name.as_deref(), Some("cachedContents/abc"));
        assert_eq!(request.contents.len(), 1, "the cached prefix is removed");
        assert_eq!(
            request.config.cached_content.as_deref(),
            Some("cachedContents/abc")
        );
        adk_platform::time::reset_time_provider();
    }

    // --- handle_context_caching: invalid cache, fingerprint mismatch ---

    #[rusty_tokio::test]
    async fn an_invalid_cache_with_a_content_mismatch_returns_a_fresh_fingerprint() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        adk_platform::time::set_time_provider(|| 5_000.0);
        // Expired, and its recorded fingerprint no longer matches anything
        // this request could regenerate.
        request.cache_metadata = Some(
            CacheMetadata::new(
                Some("cachedContents/abc".to_string()),
                Some(2_000.0),
                "stale".to_string(),
                Some(1),
                2,
                Some(1_000.0),
            )
            .unwrap(),
        );

        let result = manager
            .handle_context_caching(&mut request)
            .await
            .unwrap()
            .unwrap();
        assert!(result.cache_name.is_none());
        assert_eq!(result.contents_count, 2);
        assert_eq!(
            result.fingerprint,
            manager.generate_cache_fingerprint(&request, 2)
        );
        adk_platform::time::reset_time_provider();
    }

    // --- handle_context_caching: invalid cache, fingerprint matches (expired-but-same-content) ---

    #[rusty_tokio::test]
    async fn expired_but_same_content_without_a_token_count_preserves_the_prefix_fingerprint() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        let fp = manager.generate_cache_fingerprint(&request, 2);
        adk_platform::time::set_time_provider(|| 5_000.0);
        request.cache_metadata = Some(
            CacheMetadata::new(
                Some("cachedContents/abc".to_string()),
                Some(2_000.0),
                fp,
                Some(1),
                2,
                Some(1_000.0),
            )
            .unwrap(),
        );
        // No `cacheable_contents_token_count` set — matches the source's
        // "skip cache creation for initial request" branch.

        let result = manager
            .handle_context_caching(&mut request)
            .await
            .unwrap()
            .unwrap();
        assert!(
            result.cache_name.is_none(),
            "cache creation is skipped without a prior token count"
        );
        assert_eq!(result.contents_count, 2);
        adk_platform::time::reset_time_provider();
    }

    // --- create_gemini_cache / cleanup_cache: real HTTP calls against a local mock server ---

    fn read_http_request(stream: &mut std::net::TcpStream) -> (String, Vec<u8>) {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
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
                let headers = String::from_utf8_lossy(&received[..header_end]).to_string();
                let content_length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                if received.len() >= header_end + 4 + content_length {
                    let request_line = headers.lines().next().unwrap_or("").to_string();
                    let body = received[header_end + 4..].to_vec();
                    return (request_line, body);
                }
            }
        }
        (String::new(), Vec::new())
    }

    fn spawn_mock_cache_server(
        response_body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let (request_line, _body) = read_http_request(&mut stream);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            request_line
        });
        (format!("http://{addr}"), handle)
    }

    #[rusty_tokio::test]
    async fn create_gemini_cache_parses_a_successful_response() {
        let (base_url, handle) = spawn_mock_cache_server(
            r#"{"name":"cachedContents/abc123","expireTime":"2026-08-23T01:30:00Z"}"#,
        );
        let manager = manager(base_url);
        let request = request_with_two_turns("gemini-2.5-flash");

        let metadata = manager.create_gemini_cache(&request, 2).await.unwrap();
        assert_eq!(
            metadata.cache_name.as_deref(),
            Some("cachedContents/abc123")
        );
        assert_eq!(metadata.contents_count, 2);
        assert_eq!(metadata.invocations_used, Some(1));
        let expected_epoch = rusty_time::DateTime::parse("2026-08-23T01:30:00Z")
            .unwrap()
            .timestamp() as f64;
        assert_eq!(metadata.expire_time, Some(expected_epoch));

        let request_line = handle.join().unwrap();
        assert!(request_line.starts_with("POST"));
        assert!(request_line.contains("/cachedContents"));
    }

    #[rusty_tokio::test]
    async fn create_gemini_cache_maps_a_non_2xx_response_to_an_http_error() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_http_request(&mut stream);
            let body = r#"{"error":{"message":"below minimum cache size"}}"#;
            let response = format!(
                "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });
        let manager = manager(format!("http://{addr}"));
        let request = request_with_two_turns("gemini-2.5-flash");

        let result = manager.create_gemini_cache(&request, 2).await;
        assert!(matches!(
            result,
            Err(GeminiContextCacheError::Http { status: 400, .. })
        ));
        handle.join().unwrap();
    }

    #[rusty_tokio::test]
    async fn cleanup_cache_sends_a_delete_to_the_cache_specific_url() {
        let (base_url, handle) = spawn_mock_cache_server("{}");
        let manager = manager(base_url);
        manager.cleanup_cache("cachedContents/abc123").await;
        let request_line = handle.join().unwrap();
        assert!(request_line.starts_with("DELETE"));
        assert!(request_line.contains("/cachedContents/abc123"));
    }

    #[rusty_tokio::test]
    async fn create_new_cache_with_contents_skips_creation_below_min_tokens() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        request.cacheable_contents_token_count = Some(10);
        let result = manager.create_new_cache_with_contents(&request, 2).await;
        assert!(result.is_none());
    }

    #[rusty_tokio::test]
    async fn create_new_cache_with_contents_skips_creation_below_the_model_floor() {
        let manager = manager("http://example.invalid".to_string());
        let mut request = request_with_two_turns("gemini-2.5-flash");
        // Clears `min_tokens` (100) but the per-request estimate of the
        // 2-content prefix is far below Gemini 2.5's 2048-token floor.
        request.cacheable_contents_token_count = Some(500);
        let result = manager.create_new_cache_with_contents(&request, 2).await;
        assert!(result.is_none());
    }
}
