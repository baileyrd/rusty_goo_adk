//! Capability C0533: `auth.oauth2_discovery`'s two metadata models,
//! ported from `google.adk.auth.oauth2_discovery`.
//!
//! **Scope**: only the two data models this batch builds —
//! [`AuthorizationServerMetadata`] (RFC8414) and
//! [`ProtectedResourceMetadata`] (RFC9728). `OAuth2DiscoveryManager`'s
//! actual `.well-known` candidate-path traversal and issuer/resource
//! mix-up-attack validation (`discover_auth_server_metadata`/
//! `discover_resource_metadata`, C0534/C0535) need an async HTTP client
//! this port hasn't adopted anywhere yet — a separate, still-open batch.
//!
//! **Wire shape, not camelCase**: unlike this crate's Google-genai-facing
//! types (`Part`/`FunctionResponse`/etc., which use `alias_generator=to_camel`
//! to match Google's own API convention), these two models mirror RFC8414/
//! RFC9728's actual JSON field names verbatim — both RFCs specify
//! snake_case field names (`authorization_endpoint`, `scopes_supported`,
//! ...) — so no `rename_all` is applied here.
//!
//! **`@experimental`, same precedent as `ResumabilityConfig`**: the
//! source decorates both classes with `@experimental`, which wraps
//! `__init__` to warn on every *construction* (confirmed by reading
//! `_create_decorator`'s class-decorating branch) — not on class
//! definition, and not on deserialization specifically (pydantic's
//! `model_validate` also calls `__init__`, so the source warns on parse
//! too; this port's derive-based `Deserialize` bypasses any constructor
//! function, same already-accepted narrowing `ResumabilityConfig::new`
//! established: the explicit constructor warns, deserialization doesn't).

use rusty_serde::{Deserialize, Serialize};

use adk_features::legacy_feature_decorator::warn_experimental;

/// `auth.oauth2_discovery.AuthorizationServerMetadata` — the OAuth2
/// authorization server metadata document per RFC8414.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[rusty_serde(default)]
    pub scopes_supported: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub registration_endpoint: Option<String>,
}

impl AuthorizationServerMetadata {
    /// Constructs the metadata, emitting the source's `@experimental`
    /// warning — see the module doc.
    pub fn new(
        issuer: impl Into<String>,
        authorization_endpoint: impl Into<String>,
        token_endpoint: impl Into<String>,
        scopes_supported: Option<Vec<String>>,
        registration_endpoint: Option<String>,
    ) -> Self {
        warn_experimental("AuthorizationServerMetadata", None);
        Self {
            issuer: issuer.into(),
            authorization_endpoint: authorization_endpoint.into(),
            token_endpoint: token_endpoint.into(),
            scopes_supported,
            registration_endpoint,
        }
    }
}

/// `auth.oauth2_discovery.ProtectedResourceMetadata` — the OAuth2
/// protected resource metadata document per RFC9728.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtectedResourceMetadata {
    pub resource: String,
    #[rusty_serde(default)]
    pub authorization_servers: Vec<String>,
}

impl ProtectedResourceMetadata {
    /// Constructs the metadata, emitting the source's `@experimental`
    /// warning — see the module doc.
    pub fn new(resource: impl Into<String>, authorization_servers: Vec<String>) -> Self {
        warn_experimental("ProtectedResourceMetadata", None);
        Self {
            resource: resource.into(),
            authorization_servers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_server_metadata_round_trips_through_json() {
        let metadata = AuthorizationServerMetadata::new(
            "https://issuer.example.com",
            "https://issuer.example.com/authorize",
            "https://issuer.example.com/token",
            Some(vec!["openid".to_string(), "email".to_string()]),
            Some("https://issuer.example.com/register".to_string()),
        );
        let json = rusty_serde::json::to_string(&metadata).unwrap();
        let round_tripped: AuthorizationServerMetadata =
            rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(round_tripped, metadata);
        assert!(json.contains("authorization_endpoint"));
        assert!(!json.contains("authorizationEndpoint"));
    }

    #[test]
    fn authorization_server_metadata_defaults_optional_fields_when_absent() {
        let json = r#"{"issuer":"i","authorization_endpoint":"a","token_endpoint":"t"}"#;
        let metadata: AuthorizationServerMetadata = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(metadata.scopes_supported, None);
        assert_eq!(metadata.registration_endpoint, None);
    }

    #[test]
    fn protected_resource_metadata_round_trips_through_json() {
        let metadata = ProtectedResourceMetadata::new(
            "https://resource.example.com",
            vec!["https://issuer.example.com".to_string()],
        );
        let json = rusty_serde::json::to_string(&metadata).unwrap();
        let round_tripped: ProtectedResourceMetadata = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(round_tripped, metadata);
    }

    #[test]
    fn protected_resource_metadata_defaults_authorization_servers_to_an_empty_list() {
        let json = r#"{"resource":"r"}"#;
        let metadata: ProtectedResourceMetadata = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(metadata.authorization_servers, Vec::<String>::new());
    }
}
