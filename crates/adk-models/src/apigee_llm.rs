//! C0549/C0550/C0551 (P10, first Apigee slice): the model-string DSL,
//! `ApiType`, and constructor/identification logic from
//! `models/apigee_llm.py`'s `ApigeeLlm(Gemini)`.
//!
//! **Scope of this batch, disclosed**: only the pure, HTTP-free config
//! and identity layer is ported here — [`ApiType`] (C0549's enum half),
//! [`validate_model_string`] (C0549's DSL-validation half),
//! [`ApigeeLlm::new`]/[`ApigeeLlmConfig`] (C0550's constructor, including
//! both conflicting-options warnings), and
//! [`identify_vertexai`]/[`identify_api_version`]/[`get_model_id`]
//! (C0551, including the `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION`
//! required-env-var checks when Vertex-routed). The rest of P10's Apigee
//! rows (C0552-C0556 — the GENAI-path `HttpOptions` override, the
//! `CompletionsHTTPClient` for the OpenAI-compatible chat-completions
//! path, request-payload construction, response parsing, and non-Gemini
//! preprocessing) are real, separable HTTP-calling work needing an async
//! HTTP client wired into `BaseLlm::generate_content_async` — deliberately
//! left for a follow-up batch, the same "config layer first, wire calls
//! later" split this port already used for the native Gemini backend
//! (`gemini.rs`'s own module doc) and for the Anthropic backend
//! (`anthropic_conversion.rs`).
//!
//! **Not yet registered into [`crate::registry::default_registry`]**:
//! unlike `Gemini`/`Gemma`/`OllamaLlm`, [`ApigeeLlm`] doesn't implement
//! [`crate::base_llm::BaseLlm`] yet — there is no real
//! `generate_content_async` to back it (that's the deferred C0552-C0556
//! work above), and registering a backend that can't actually generate
//! content would be worse than not registering it. Registration lands
//! alongside that follow-up batch.
//!
//! **Composition instead of the source's inheritance**: `ApigeeLlm(Gemini)`
//! is a Python subclass; this port instead holds a [`crate::gemini::Gemini`]
//! by composition — the same adaptation `gemma.rs`'s module doc already
//! established for `Gemma(GemmaFunctionCallingMixin, Gemini)`.
//!
//! **`credentials`, opaque placeholder**: the source's `credentials` is a
//! `google.auth.credentials.Credentials` (import guarded by
//! `TYPE_CHECKING`, meaning even the source never actually touches its
//! internals at this layer) — represented here as an opaque
//! [`rusty_serde::value::Value`] placeholder, only stored and passed
//! through, matching the "opaque placeholder, forwarded not read"
//! convention `llm_request.rs`'s own module doc already establishes for
//! `tools`/`thinking_config`/`safety_settings`.
//!
//! **`ApiType::parse`, string-parsing parity**: the source's
//! `ApiType(str, enum.Enum)` accepts either an `ApiType` member or a raw
//! string (`ApigeeLlm.__init__`'s `if isinstance(api_type, str): api_type
//! = ApiType(api_type)`), with `_missing_` mapping only the empty
//! string/`None` to `UNKNOWN` — any other unrecognized string still
//! raises (via `Enum.__new__`'s normal `ValueError`, since `_missing_`
//! falls through to `super()._missing_(value)`, which returns `None`).
//! [`ApiType::parse`] preserves that exact fallback shape as a `Result`.

use std::collections::HashMap;
use std::sync::Arc;

use rusty_serde::value::Value;

use crate::capabilities::is_enterprise_mode_enabled;
use crate::gemini::{Gemini, GeminiApiClient};

const APIGEE_PROXY_URL_ENV_VAR: &str = "APIGEE_PROXY_URL";
const PROJECT_ENV_VAR: &str = "GOOGLE_CLOUD_PROJECT";
const LOCATION_ENV_VAR: &str = "GOOGLE_CLOUD_LOCATION";

/// `ApigeeLlm.ApiType` — the supported API types for the Apigee LLM
/// backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiType {
    #[default]
    Unknown,
    ChatCompletions,
    Genai,
}

impl ApiType {
    /// `ApiType(value)`/`ApiType._missing_` — see the module doc for the
    /// exact fallback shape being preserved.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "" | "unknown" => Ok(Self::Unknown),
            "chat_completions" => Ok(Self::ChatCompletions),
            "genai" => Ok(Self::Genai),
            other => Err(format!("'{other}' is not a valid ApiType")),
        }
    }
}

/// `apigee_llm._validate_model_string` — validates
/// `apigee/[<provider>/][<version>/]<model_id>`, where `provider` is one
/// of `vertex_ai`/`gemini`/`openai` and `version` starts with `v`.
pub fn validate_model_string(model: &str) -> bool {
    let Some(rest) = model.strip_prefix("apigee/") else {
        return false;
    };
    if rest.is_empty() {
        return false;
    }
    let components: Vec<&str> = rest.split('/').collect();
    match components.len() {
        1 => true,
        3 => {
            matches!(components[0], "vertex_ai" | "gemini" | "openai")
                && components[1].starts_with('v')
        }
        2 => {
            matches!(components[0], "vertex_ai" | "gemini" | "openai")
                || components[0].starts_with('v')
        }
        _ => false,
    }
}

/// `apigee_llm._identify_vertexai` — an Apigee-routed model is Vertex AI
/// when the resolved [`ApiType`] permits it and either the model string
/// explicitly names `vertex_ai` or enterprise mode is on (and the
/// `gemini`/`openai` providers are never Vertex-routed regardless).
pub fn identify_vertexai(model: &str, api_type: ApiType) -> bool {
    if !matches!(api_type, ApiType::Genai | ApiType::Unknown) {
        return false;
    }
    if model.starts_with("apigee/gemini/") || model.starts_with("apigee/openai/") {
        return false;
    }
    model.starts_with("apigee/vertex_ai/") || is_enterprise_mode_enabled()
}

/// `apigee_llm._identify_api_version` — the middle DSL component, when
/// present and not itself a provider name.
pub fn identify_api_version(model: &str) -> String {
    let rest = model.strip_prefix("apigee/").unwrap_or(model);
    let components: Vec<&str> = rest.split('/').collect();
    if components.len() == 3 {
        return components[1].to_string();
    }
    if components.len() == 2
        && !matches!(components[0], "vertex_ai" | "gemini")
        && components[0].starts_with('v')
    {
        return components[0].to_string();
    }
    String::new()
}

/// `apigee_llm._get_model_id` — the last DSL component.
pub fn get_model_id(model: Option<&str>) -> Result<String, String> {
    let model = model
        .filter(|m| !m.is_empty())
        .ok_or_else(|| "Model is not set.".to_string())?;
    let rest = model.strip_prefix("apigee/").unwrap_or(model);
    Ok(rest.rsplit('/').next().unwrap_or(rest).to_string())
}

/// Constructor parameters for [`ApigeeLlm::new`] — mirrors
/// `ApigeeLlm.__init__`'s keyword arguments (all optional except
/// `model`, passed separately).
#[derive(Clone, Default)]
pub struct ApigeeLlmConfig {
    pub proxy_url: Option<String>,
    pub custom_headers: HashMap<String, String>,
    /// Opaque placeholder for `types.HttpRetryOptions` — see
    /// [`crate::gemini::Gemini::retry_options`].
    pub retry_options: Option<Value>,
    pub api_type: ApiType,
    /// Opaque placeholder — see the module doc.
    pub credentials: Option<Value>,
    pub client: Option<Arc<GeminiApiClient>>,
}

#[derive(Debug, rusty_err::Error)]
pub enum ApigeeLlmError {
    #[error("Invalid model string: {0}")]
    InvalidModelString(String),
    #[error("The {0} environment variable must be set.")]
    MissingEnvVar(&'static str),
}

/// `ApigeeLlm(Gemini)` — routes Gemini-shaped requests through an Apigee
/// proxy. See the module doc for what's ported in this batch vs.
/// deferred.
pub struct ApigeeLlm {
    pub gemini: Gemini,
    pub api_type: ApiType,
    pub proxy_url: Option<String>,
    pub custom_headers: HashMap<String, String>,
    /// Opaque placeholder — see the module doc.
    pub credentials: Option<Value>,
    pub isvertexai: bool,
    pub project: Option<String>,
    pub location: Option<String>,
    pub api_version: String,
    pub user_agent: String,
}

impl ApigeeLlm {
    /// `ApigeeLlm.__init__`.
    pub fn new(model: impl Into<String>, config: ApigeeLlmConfig) -> Result<Self, ApigeeLlmError> {
        let model = model.into();
        if !validate_model_string(&model) {
            return Err(ApigeeLlmError::InvalidModelString(model));
        }

        let api_type = if config.api_type != ApiType::Unknown {
            config.api_type
        } else if model.starts_with("apigee/gemini/") || model.starts_with("apigee/vertex_ai/") {
            ApiType::Genai
        } else if model.starts_with("apigee/openai/") {
            ApiType::ChatCompletions
        } else {
            ApiType::Genai
        };

        let isvertexai = identify_vertexai(&model, api_type);
        let (project, location) = if isvertexai {
            let project = std::env::var(PROJECT_ENV_VAR)
                .ok()
                .filter(|v| !v.is_empty())
                .ok_or(ApigeeLlmError::MissingEnvVar(PROJECT_ENV_VAR))?;
            let location = std::env::var(LOCATION_ENV_VAR)
                .ok()
                .filter(|v| !v.is_empty())
                .ok_or(ApigeeLlmError::MissingEnvVar(LOCATION_ENV_VAR))?;
            (Some(project), Some(location))
        } else {
            (None, None)
        };

        let api_version = identify_api_version(&model);
        let proxy_url = config
            .proxy_url
            .or_else(|| std::env::var(APIGEE_PROXY_URL_ENV_VAR).ok());

        let mut gemini = Gemini::new(model);
        gemini.client = config.client.clone();
        gemini.retry_options = config.retry_options;

        if config.client.is_some() {
            if proxy_url.is_some() || !config.custom_headers.is_empty() {
                eprintln!(
                    "UserWarning: Both client and proxy_url/custom_headers were provided. The \
                     injected client will be used as-is for GENAI calls, and \
                     proxy_url/custom_headers will be ignored. Ensure the injected client is \
                     pre-configured with the correct proxy and headers."
                );
            }
            if api_type == ApiType::ChatCompletions {
                eprintln!(
                    "UserWarning: An injected client was provided but ApiType is \
                     CHAT_COMPLETIONS. The injected client will be ignored for \
                     CHAT_COMPLETIONS calls."
                );
            }
        }

        Ok(Self {
            api_version,
            gemini,
            api_type,
            proxy_url,
            custom_headers: config.custom_headers,
            credentials: config.credentials,
            isvertexai,
            project,
            location,
            user_agent: format!("google-adk/{}", env!("CARGO_PKG_VERSION")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::ENV_LOCK;

    // --- ApiType::parse ---

    #[test]
    fn api_type_parse_empty_string_is_unknown() {
        assert_eq!(ApiType::parse(""), Ok(ApiType::Unknown));
    }

    #[test]
    fn api_type_parse_known_values() {
        assert_eq!(ApiType::parse("unknown"), Ok(ApiType::Unknown));
        assert_eq!(ApiType::parse("genai"), Ok(ApiType::Genai));
        assert_eq!(
            ApiType::parse("chat_completions"),
            Ok(ApiType::ChatCompletions)
        );
    }

    #[test]
    fn api_type_parse_rejects_unrecognized_values() {
        assert!(ApiType::parse("bogus").is_err());
    }

    // --- validate_model_string ---

    #[test]
    fn validate_model_string_rejects_missing_apigee_prefix() {
        assert!(!validate_model_string("gemini-2.5-flash"));
    }

    #[test]
    fn validate_model_string_rejects_empty_model_id() {
        assert!(!validate_model_string("apigee/"));
    }

    #[test]
    fn validate_model_string_accepts_bare_model_id() {
        assert!(validate_model_string("apigee/gemini-2.5-flash"));
    }

    #[test]
    fn validate_model_string_accepts_provider_and_model_id() {
        assert!(validate_model_string("apigee/vertex_ai/gemini-2.5-flash"));
        assert!(validate_model_string("apigee/openai/gpt-4"));
    }

    #[test]
    fn validate_model_string_accepts_version_and_model_id() {
        assert!(validate_model_string("apigee/v1/gemini-2.5-flash"));
    }

    #[test]
    fn validate_model_string_rejects_unknown_two_component_prefix() {
        assert!(!validate_model_string("apigee/bogus/gemini-2.5-flash"));
    }

    #[test]
    fn validate_model_string_accepts_provider_version_and_model_id() {
        assert!(validate_model_string(
            "apigee/vertex_ai/v1beta/gemini-2.5-flash"
        ));
    }

    #[test]
    fn validate_model_string_rejects_bad_provider_in_three_components() {
        assert!(!validate_model_string("apigee/bogus/v1/gemini-2.5-flash"));
    }

    #[test]
    fn validate_model_string_rejects_bad_version_in_three_components() {
        assert!(!validate_model_string(
            "apigee/vertex_ai/beta/gemini-2.5-flash"
        ));
    }

    #[test]
    fn validate_model_string_rejects_more_than_three_components() {
        assert!(!validate_model_string("apigee/a/b/c/d"));
    }

    // --- identify_vertexai ---

    #[test]
    fn identify_vertexai_false_for_explicit_gemini_provider() {
        let _guard = ENV_LOCK.lock().unwrap();
        assert!(!identify_vertexai(
            "apigee/gemini/gemini-2.5-flash",
            ApiType::Genai
        ));
    }

    #[test]
    fn identify_vertexai_false_for_explicit_openai_provider() {
        let _guard = ENV_LOCK.lock().unwrap();
        assert!(!identify_vertexai("apigee/openai/gpt-4", ApiType::Genai));
    }

    #[test]
    fn identify_vertexai_true_for_explicit_vertex_ai_provider() {
        let _guard = ENV_LOCK.lock().unwrap();
        assert!(identify_vertexai(
            "apigee/vertex_ai/gemini-2.5-flash",
            ApiType::Genai
        ));
    }

    #[test]
    fn identify_vertexai_false_for_chat_completions_api_type() {
        let _guard = ENV_LOCK.lock().unwrap();
        assert!(!identify_vertexai(
            "apigee/vertex_ai/gemini-2.5-flash",
            ApiType::ChatCompletions
        ));
    }

    #[test]
    fn identify_vertexai_follows_enterprise_mode_when_no_explicit_provider() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        assert!(!identify_vertexai(
            "apigee/gemini-2.5-flash",
            ApiType::Unknown
        ));
        unsafe {
            std::env::set_var("GOOGLE_GENAI_USE_ENTERPRISE", "true");
        }
        let result = identify_vertexai("apigee/gemini-2.5-flash", ApiType::Unknown);
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
        }
        assert!(result);
    }

    // --- identify_api_version ---

    #[test]
    fn identify_api_version_empty_for_bare_model_id() {
        assert_eq!(identify_api_version("apigee/gemini-2.5-flash"), "");
    }

    #[test]
    fn identify_api_version_reads_the_version_when_no_provider() {
        assert_eq!(identify_api_version("apigee/v1/gemini-2.5-flash"), "v1");
    }

    #[test]
    fn identify_api_version_empty_when_the_two_component_prefix_is_a_provider() {
        assert_eq!(
            identify_api_version("apigee/vertex_ai/gemini-2.5-flash"),
            ""
        );
    }

    #[test]
    fn identify_api_version_reads_the_middle_component_in_three_part_form() {
        assert_eq!(
            identify_api_version("apigee/vertex_ai/v1beta/gemini-2.5-flash"),
            "v1beta"
        );
    }

    // --- get_model_id ---

    #[test]
    fn get_model_id_errors_on_none() {
        assert!(get_model_id(None).is_err());
    }

    #[test]
    fn get_model_id_errors_on_empty_string() {
        assert!(get_model_id(Some("")).is_err());
    }

    #[test]
    fn get_model_id_strips_apigee_prefix_only() {
        assert_eq!(
            get_model_id(Some("apigee/gemini-2.5-flash")),
            Ok("gemini-2.5-flash".to_string())
        );
    }

    #[test]
    fn get_model_id_strips_provider_and_version() {
        assert_eq!(
            get_model_id(Some("apigee/vertex_ai/v1beta/gemini-2.5-flash")),
            Ok("gemini-2.5-flash".to_string())
        );
    }

    // --- ApigeeLlm::new ---

    #[test]
    fn new_rejects_an_invalid_model_string() {
        let _guard = ENV_LOCK.lock().unwrap();
        let result = ApigeeLlm::new("not-apigee-prefixed", ApigeeLlmConfig::default());
        assert!(matches!(result, Err(ApigeeLlmError::InvalidModelString(_))));
    }

    #[test]
    fn new_infers_genai_for_gemini_and_vertex_ai_prefixes() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
        }
        let llm =
            ApigeeLlm::new("apigee/gemini/gemini-2.5-flash", ApigeeLlmConfig::default()).unwrap();
        assert_eq!(llm.api_type, ApiType::Genai);
        assert!(!llm.isvertexai);
    }

    #[test]
    fn new_infers_chat_completions_for_openai_prefix() {
        let _guard = ENV_LOCK.lock().unwrap();
        let llm = ApigeeLlm::new("apigee/openai/gpt-4", ApigeeLlmConfig::default()).unwrap();
        assert_eq!(llm.api_type, ApiType::ChatCompletions);
        assert!(!llm.isvertexai);
    }

    #[test]
    fn new_honors_an_explicit_api_type_over_inference() {
        let _guard = ENV_LOCK.lock().unwrap();
        let llm = ApigeeLlm::new(
            "apigee/openai/gpt-4",
            ApigeeLlmConfig {
                api_type: ApiType::Genai,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(llm.api_type, ApiType::Genai);
    }

    #[test]
    fn new_requires_project_and_location_env_vars_when_vertex_routed() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var(PROJECT_ENV_VAR);
            std::env::remove_var(LOCATION_ENV_VAR);
        }
        let result = ApigeeLlm::new(
            "apigee/vertex_ai/gemini-2.5-flash",
            ApigeeLlmConfig::default(),
        );
        assert!(matches!(result, Err(ApigeeLlmError::MissingEnvVar(v)) if v == PROJECT_ENV_VAR));
    }

    #[test]
    fn new_succeeds_when_vertex_routed_with_project_and_location_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(PROJECT_ENV_VAR, "my-project");
            std::env::set_var(LOCATION_ENV_VAR, "us-central1");
        }
        let result = ApigeeLlm::new(
            "apigee/vertex_ai/gemini-2.5-flash",
            ApigeeLlmConfig::default(),
        );
        unsafe {
            std::env::remove_var(PROJECT_ENV_VAR);
            std::env::remove_var(LOCATION_ENV_VAR);
        }
        let llm = result.unwrap();
        assert!(llm.isvertexai);
        assert_eq!(llm.project.as_deref(), Some("my-project"));
        assert_eq!(llm.location.as_deref(), Some("us-central1"));
    }

    #[test]
    fn new_resolves_proxy_url_from_env_when_not_given() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
            std::env::set_var(APIGEE_PROXY_URL_ENV_VAR, "https://proxy.example.com");
        }
        let llm = ApigeeLlm::new("apigee/gemini-2.5-flash", ApigeeLlmConfig::default());
        unsafe {
            std::env::remove_var(APIGEE_PROXY_URL_ENV_VAR);
        }
        assert_eq!(
            llm.unwrap().proxy_url.as_deref(),
            Some("https://proxy.example.com")
        );
    }

    #[test]
    fn new_prefers_explicit_proxy_url_over_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("GOOGLE_GENAI_USE_ENTERPRISE");
            std::env::remove_var("GOOGLE_GENAI_USE_VERTEXAI");
            std::env::set_var(APIGEE_PROXY_URL_ENV_VAR, "https://env.example.com");
        }
        let llm = ApigeeLlm::new(
            "apigee/gemini-2.5-flash",
            ApigeeLlmConfig {
                proxy_url: Some("https://explicit.example.com".to_string()),
                ..Default::default()
            },
        );
        unsafe {
            std::env::remove_var(APIGEE_PROXY_URL_ENV_VAR);
        }
        assert_eq!(
            llm.unwrap().proxy_url.as_deref(),
            Some("https://explicit.example.com")
        );
    }

    #[test]
    fn new_identifies_api_version_from_the_model_string() {
        let _guard = ENV_LOCK.lock().unwrap();
        let llm = ApigeeLlm::new(
            "apigee/gemini/v1/gemini-2.5-flash",
            ApigeeLlmConfig::default(),
        )
        .unwrap();
        assert_eq!(llm.api_version, "v1");
    }
}
