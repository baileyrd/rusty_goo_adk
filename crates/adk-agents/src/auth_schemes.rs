//! Capability C0503 (plus C0498's wrap-up): `auth.auth_schemes`, ported
//! from `google.adk.auth.auth_schemes`.
//!
//! **`AuthScheme`, adapted**: the source's `AuthScheme = Union[SecurityScheme,
//! OpenIdConnectWithConfig, CustomAuthScheme]`, where `SecurityScheme` is
//! itself `fastapi.openapi.models.SecurityScheme` — a further union over
//! the OpenAPI 3.0 Security Scheme Object's four shapes (API key, HTTP,
//! OAuth2, OpenID Connect). This port models that whole tree as nested
//! Rust enums ([`SecurityScheme`] inside [`AuthScheme`]) rather than a
//! single flat enum, matching the source's own two-level union structure.
//! Both enums are `#[rusty_serde(untagged)]` (the same pattern
//! `skills_models::ResourceContent` already established) — each
//! variant's inner struct has a field set no other variant shares
//! (`in_`/`name` for API key, `scheme` for HTTP, `flows` for OAuth2,
//! `open_id_connect_url` for plain OpenID Connect), so structural
//! matching during deserialization recovers the right variant without a
//! discriminant tag, the same way the source's own `type` field is
//! informative but not required for `Union` resolution via pydantic's
//! smart-union matching.
//!
//! **`OAuthFlow`, narrowed**: the OpenAPI spec defines four distinct
//! "OAuth Flow Object" shapes (`OAuthFlowImplicit`/`OAuthFlowPassword`/
//! `OAuthFlowClientCredentials`/`OAuthFlowAuthorizationCode`), each
//! requiring a different subset of `authorizationUrl`/`tokenUrl` — a
//! type-level distinction fastapi's pydantic models enforce per-flow.
//! This port collapses all four into one [`OAuthFlow`] struct with both
//! URLs optional; which of [`OAuthFlows`]'s four `Option` fields is
//! populated still identifies the grant type (exactly what
//! [`OAuthGrantType::from_flow`] reads), so no information is lost —
//! only the per-flow required-field type-checking is, a narrowing (a
//! malformed flow that fastapi's pydantic model would reject at
//! construction deserializes here without error).
//!
//! **`CustomAuthScheme`'s extensibility, preserved via flatten**: the
//! source subclasses `BaseModelWithConfig` (`extra="allow"`) specifically
//! so developer-defined subclasses can add their own typed fields — unlike
//! `auth_credential.rs`'s structs (where `extra="allow"` covers
//! incidental/unmodeled metadata and is disclosed there as "dropped, not
//! preserved"), extensibility is this type's entire purpose. This port
//! preserves it with `#[rusty_serde(flatten)] extra: Option<Value>` (the
//! same flattened-catch-all pattern `adk_genai::content::MediaBlobStub
//! ::rest` already established) rather than dropping unknown fields —
//! a caller can still read/round-trip a custom scheme's own fields
//! through `extra`, just untyped rather than reified as named Rust
//! fields the way a Python subclass would declare them. The other
//! `AuthScheme` branches (`SecurityScheme`'s OpenAPI-defined shapes,
//! `OpenIdConnectWithConfig`) do *not* get this treatment — whether
//! fastapi's `SecurityBase` genuinely allows extra fields by default
//! isn't something this port can verify without the `fastapi` package
//! installed, so only `CustomAuthScheme` (this port's own established
//! `BaseModelWithConfig`-equivalent convention) gets it.
//!
//! **`ExtendedOAuth2`, flattened**: the source's `class
//! ExtendedOAuth2(OAuth2):` inherits every `OAuth2` field and adds
//! `issuer_url`. Rust has no struct inheritance, so
//! [`ExtendedOAuth2`] repeats `OAuth2Scheme`'s fields directly plus
//! `issuer_url` — the same "flatten inherited fields into the subclass
//! struct" pattern `adk_events::event::Event`'s own module doc already
//! established for `LlmResponse`'s inherited fields.

use std::collections::BTreeMap;

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// `fastapi.openapi.models.SecuritySchemeType` (OpenAPI 3.0's Security
/// Scheme Object `type` enum). Re-exported as [`AuthSchemeType`] to match
/// the source's `AuthSchemeType = SecuritySchemeType` alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecuritySchemeType {
    #[rusty_serde(rename = "apiKey")]
    ApiKey,
    #[rusty_serde(rename = "http")]
    Http,
    #[rusty_serde(rename = "oauth2")]
    OAuth2,
    #[rusty_serde(rename = "openIdConnect")]
    OpenIdConnect,
    #[rusty_serde(rename = "mutualTLS")]
    MutualTls,
}

impl SecuritySchemeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SecuritySchemeType::ApiKey => "apiKey",
            SecuritySchemeType::Http => "http",
            SecuritySchemeType::OAuth2 => "oauth2",
            SecuritySchemeType::OpenIdConnect => "openIdConnect",
            SecuritySchemeType::MutualTls => "mutualTLS",
        }
    }
}

/// `AuthSchemeType` — re-exports `SecuritySchemeType` (source: `AuthSchemeType
/// = SecuritySchemeType`).
pub type AuthSchemeType = SecuritySchemeType;

/// `fastapi.openapi.models.APIKeyIn` — where an API key credential is
/// carried on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiKeyIn {
    #[rusty_serde(rename = "query")]
    Query,
    #[rusty_serde(rename = "header")]
    Header,
    #[rusty_serde(rename = "cookie")]
    Cookie,
}

/// The OpenAPI 3.0 API Key Security Scheme Object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct ApiKeyScheme {
    #[rusty_serde(default)]
    pub description: Option<String>,
    #[rusty_serde(rename = "in")]
    pub in_: ApiKeyIn,
    pub name: String,
}

/// The OpenAPI 3.0 HTTP Security Scheme Object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct HttpScheme {
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub scheme: String,
    #[rusty_serde(default)]
    pub bearer_format: Option<String>,
}

/// The OpenAPI 3.0 OAuth Flow Object — see the module doc for why this
/// port collapses the spec's four distinct flow shapes into one.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct OAuthFlow {
    #[rusty_serde(default)]
    pub authorization_url: Option<String>,
    #[rusty_serde(default)]
    pub token_url: Option<String>,
    #[rusty_serde(default)]
    pub refresh_url: Option<String>,
    #[rusty_serde(default)]
    pub scopes: BTreeMap<String, String>,
}

/// The OpenAPI 3.0 OAuth Flows Object.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OAuthFlows {
    #[rusty_serde(default)]
    pub implicit: Option<OAuthFlow>,
    #[rusty_serde(default)]
    pub password: Option<OAuthFlow>,
    #[rusty_serde(default)]
    pub client_credentials: Option<OAuthFlow>,
    #[rusty_serde(default)]
    pub authorization_code: Option<OAuthFlow>,
}

/// The OpenAPI 3.0 OAuth2 Security Scheme Object.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OAuth2Scheme {
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub flows: OAuthFlows,
}

/// The OpenAPI 3.0 OpenID Connect Security Scheme Object (the *plain*
/// one, distinct from [`OpenIdConnectWithConfig`] below, which inlines
/// the discovery document instead of just pointing at its URL).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct OpenIdConnectScheme {
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub open_id_connect_url: String,
}

/// `fastapi.openapi.models.SecurityScheme` — see the module doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(untagged)]
pub enum SecurityScheme {
    ApiKey(ApiKeyScheme),
    Http(HttpScheme),
    /// Boxed: `OAuth2Scheme` (via `OAuthFlows`) is much larger than the
    /// other variants — boxing keeps `SecurityScheme` itself small
    /// rather than every variant paying for the largest one's size.
    OAuth2(Box<OAuth2Scheme>),
    OpenIdConnect(OpenIdConnectScheme),
}

impl SecurityScheme {
    pub fn scheme_type(&self) -> SecuritySchemeType {
        match self {
            SecurityScheme::ApiKey(_) => SecuritySchemeType::ApiKey,
            SecurityScheme::Http(_) => SecuritySchemeType::Http,
            SecurityScheme::OAuth2(_) => SecuritySchemeType::OAuth2,
            SecurityScheme::OpenIdConnect(_) => SecuritySchemeType::OpenIdConnect,
        }
    }
}

/// `auth.auth_schemes.OpenIdConnectWithConfig` — inlines the OIDC
/// discovery-document fields directly rather than just an
/// `openIdConnectUrl` pointer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct OpenIdConnectWithConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[rusty_serde(default)]
    pub userinfo_endpoint: Option<String>,
    #[rusty_serde(default)]
    pub revocation_endpoint: Option<String>,
    #[rusty_serde(default)]
    pub token_endpoint_auth_methods_supported: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub grant_types_supported: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub scopes: Option<Vec<String>>,
}

/// `auth.auth_schemes.CustomAuthScheme` — a flexible model for custom
/// authentication schemes. See the module doc for why `extra` is
/// preserved (flattened) rather than dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomAuthScheme {
    #[rusty_serde(rename = "type")]
    pub type_: String,
    #[rusty_serde(flatten, default)]
    pub extra: Option<Value>,
}

/// `auth.auth_schemes.AuthScheme` — see the module doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(untagged)]
pub enum AuthScheme {
    /// Boxed: `SecurityScheme` (via `OAuth2Scheme`/`OAuthFlows`) is much
    /// larger than the other variants — see `SecurityScheme::OAuth2`'s
    /// own doc.
    Security(Box<SecurityScheme>),
    OpenIdConnectWithConfig(OpenIdConnectWithConfig),
    Custom(CustomAuthScheme),
}

impl AuthScheme {
    /// The source's `type_.name if type_ and hasattr(type_, "name") else
    /// str(type_)` (`AuthConfig.get_credential_key`) — see the module doc
    /// for why this port doesn't reproduce Python's enum-`.name`-vs-`str()`
    /// branching verbatim; any stable, type-derived string satisfies the
    /// same purpose (key uniqueness), which this preserves.
    pub fn type_name(&self) -> String {
        match self {
            AuthScheme::Security(scheme) => scheme.scheme_type().as_str().to_string(),
            AuthScheme::OpenIdConnectWithConfig(_) => {
                SecuritySchemeType::OpenIdConnect.as_str().to_string()
            }
            AuthScheme::Custom(custom) => custom.type_.clone(),
        }
    }

    /// The source's `model_extra` read — only [`AuthScheme::Custom`]
    /// carries one in this port (see the module doc).
    pub fn extra(&self) -> Option<&Value> {
        match self {
            AuthScheme::Custom(custom) => custom.extra.as_ref(),
            _ => None,
        }
    }

    /// A clone with `extra` cleared — the source's `model_copy(deep=True)`
    /// and `model_extra.clear()` step, which `AuthConfig.get_credential_key`
    /// runs before digesting the scheme (`auth_tool.rs`, C0504).
    pub fn without_extra(&self) -> AuthScheme {
        match self {
            AuthScheme::Custom(custom) => AuthScheme::Custom(CustomAuthScheme {
                type_: custom.type_.clone(),
                extra: None,
            }),
            other => other.clone(),
        }
    }
}

/// Represents the OAuth2 flow (or grant type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "snake_case")]
pub enum OAuthGrantType {
    ClientCredentials,
    AuthorizationCode,
    Implicit,
    Password,
}

impl OAuthGrantType {
    /// Converts an `OAuthFlows` object to an `OAuthGrantType`.
    pub fn from_flow(flow: &OAuthFlows) -> Option<Self> {
        if flow.client_credentials.is_some() {
            return Some(OAuthGrantType::ClientCredentials);
        }
        if flow.authorization_code.is_some() {
            return Some(OAuthGrantType::AuthorizationCode);
        }
        if flow.implicit.is_some() {
            return Some(OAuthGrantType::Implicit);
        }
        if flow.password.is_some() {
            return Some(OAuthGrantType::Password);
        }
        None
    }
}

/// `auth.auth_schemes.ExtendedOAuth2` — OAuth2 scheme with auto-discovery
/// for endpoints. See the module doc for why `OAuth2Scheme`'s fields are
/// repeated here rather than nested.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ExtendedOAuth2 {
    #[rusty_serde(default)]
    pub description: Option<String>,
    pub flows: OAuthFlows,
    /// Used for endpoint discovery.
    #[rusty_serde(default)]
    pub issuer_url: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_grant_type_from_flow_prefers_client_credentials() {
        let flows = OAuthFlows {
            client_credentials: Some(OAuthFlow::default()),
            authorization_code: Some(OAuthFlow::default()),
            ..Default::default()
        };
        assert_eq!(
            OAuthGrantType::from_flow(&flows),
            Some(OAuthGrantType::ClientCredentials)
        );
    }

    #[test]
    fn oauth_grant_type_from_flow_falls_back_through_each_variant() {
        assert_eq!(
            OAuthGrantType::from_flow(&OAuthFlows {
                authorization_code: Some(OAuthFlow::default()),
                ..Default::default()
            }),
            Some(OAuthGrantType::AuthorizationCode)
        );
        assert_eq!(
            OAuthGrantType::from_flow(&OAuthFlows {
                implicit: Some(OAuthFlow::default()),
                ..Default::default()
            }),
            Some(OAuthGrantType::Implicit)
        );
        assert_eq!(
            OAuthGrantType::from_flow(&OAuthFlows {
                password: Some(OAuthFlow::default()),
                ..Default::default()
            }),
            Some(OAuthGrantType::Password)
        );
    }

    #[test]
    fn oauth_grant_type_from_flow_returns_none_for_an_empty_flows() {
        assert_eq!(OAuthGrantType::from_flow(&OAuthFlows::default()), None);
    }

    #[test]
    fn security_scheme_round_trips_an_api_key_scheme() {
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })));
        let json = rusty_serde::json::to_string(&scheme).expect("serialize");
        let round_tripped: AuthScheme = rusty_serde::json::from_str(&json).expect("deserialize");
        assert_eq!(scheme, round_tripped);
        assert_eq!(scheme.type_name(), "apiKey");
    }

    #[test]
    fn security_scheme_round_trips_an_http_scheme() {
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "bearer".to_string(),
            bearer_format: Some("JWT".to_string()),
        })));
        let json = rusty_serde::json::to_string(&scheme).expect("serialize");
        let round_tripped: AuthScheme = rusty_serde::json::from_str(&json).expect("deserialize");
        assert_eq!(scheme, round_tripped);
    }

    #[test]
    fn security_scheme_round_trips_an_oauth2_scheme() {
        let scheme =
            AuthScheme::Security(Box::new(SecurityScheme::OAuth2(Box::new(OAuth2Scheme {
                description: None,
                flows: OAuthFlows {
                    client_credentials: Some(OAuthFlow {
                        token_url: Some("https://example.com/token".to_string()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            }))));
        let json = rusty_serde::json::to_string(&scheme).expect("serialize");
        let round_tripped: AuthScheme = rusty_serde::json::from_str(&json).expect("deserialize");
        assert_eq!(scheme, round_tripped);
        assert_eq!(scheme.type_name(), "oauth2");
    }

    #[test]
    fn open_id_connect_with_config_round_trips_and_reports_its_type_name() {
        let scheme = AuthScheme::OpenIdConnectWithConfig(OpenIdConnectWithConfig {
            authorization_endpoint: "https://example.com/authorize".to_string(),
            token_endpoint: "https://example.com/token".to_string(),
            userinfo_endpoint: None,
            revocation_endpoint: None,
            token_endpoint_auth_methods_supported: None,
            grant_types_supported: None,
            scopes: Some(vec!["openid".to_string()]),
        });
        let json = rusty_serde::json::to_string(&scheme).expect("serialize");
        let round_tripped: AuthScheme = rusty_serde::json::from_str(&json).expect("deserialize");
        assert_eq!(scheme, round_tripped);
        assert_eq!(scheme.type_name(), "openIdConnect");
    }

    #[test]
    fn custom_auth_scheme_preserves_unknown_fields_via_extra() {
        let json = r#"{"type":"my_custom_scheme","apiVersion":"v2","nested":{"a":1}}"#;
        let scheme: AuthScheme = rusty_serde::json::from_str(json).expect("deserialize");
        assert_eq!(scheme.type_name(), "my_custom_scheme");
        let extra = scheme.extra().expect("expected extra fields");
        assert_eq!(
            extra.get("apiVersion"),
            Some(&Value::String("v2".to_string()))
        );
    }

    #[test]
    fn non_custom_schemes_have_no_extra() {
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "basic".to_string(),
            bearer_format: None,
        })));
        assert_eq!(scheme.extra(), None);
    }
}
