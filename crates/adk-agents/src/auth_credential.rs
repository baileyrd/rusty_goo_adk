//! Capabilities C0494-C0497/C0499: the credential-scheme data models,
//! ported from `google.adk.auth.auth_credential`.
//!
//! **Scope of this batch**: only `auth_credential.py`'s own content.
//! `AuthScheme`/`OpenIdConnectWithConfig` (C0498) live in the separate
//! `auth_schemes.py` and aren't ported here — `OAuth2Auth` (this file)
//! already carries every field the OpenID Connect scheme reuses, so
//! nothing here is blocked on that follow-up, it's just a distinct
//! source file left for its own batch. `auth/__init__.py`'s re-export
//! asymmetry (C0493) is likewise not this batch's concern — see the
//! module doc note below for why it doesn't have a direct analogue in
//! this port's flat `pub mod` convention.
//!
//! **Adaptation, shared by every struct in this file**: the source's
//! `BaseModelWithConfig` sets Pydantic's `extra="allow"` — callers may
//! attach arbitrary unmodeled keys, which are preserved (redacted in
//! `repr`, never dropped) alongside the declared fields. A Rust struct
//! has a fixed field set; there is no catch-all map to stash unknown
//! keys into here, so **an extra key round-tripped through one of these
//! structs is silently dropped, not preserved-but-redacted** — a real,
//! disclosed behavior gap from the source's "keep it, just don't leak
//! it" semantics. `hide_input_in_errors=True` and the custom
//! `__repr_args__` secret redaction have no analogue either: this port
//! has no schema-validation-error-message layer to harden, and Rust's
//! derived `Debug` isn't used to serialize these structs anywhere in
//! this port yet, so there's nothing to redact.
//!
//! **Widened from placeholder**: `adk_agents::services::AuthCredential`
//! was previously `pub type AuthCredential = Value` (Phase 6). This
//! batch promotes it to the real struct defined here — the same
//! "widen a placeholder to a real type once a real consumer needs its
//! structure" precedent as `EventCompaction.compacted_content`
//! (C0185) and `services::{MemoryEntry, SearchMemoryResponse}`
//! (C0423). `BaseCredentialService`/`CredentialManager` are still
//! Phase 9 placeholders themselves — nothing here *produces* a real
//! `AuthCredential` from an actual auth flow yet, only the shape is
//! real and tested.

use std::collections::BTreeMap;

use rusty_serde::{Deserialize, Serialize};

/// `auth.auth_credential.AuthCredentialTypes` (C0494).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthCredentialTypes {
    #[rusty_serde(rename = "apiKey")]
    ApiKey,
    #[rusty_serde(rename = "http")]
    Http,
    #[rusty_serde(rename = "oauth2")]
    OAuth2,
    #[rusty_serde(rename = "openIdConnect")]
    OpenIdConnect,
    #[rusty_serde(rename = "serviceAccount")]
    ServiceAccount,
}

/// `auth.auth_credential.HttpCredentials` — part of C0496.
///
/// **Adaptation**: the source overrides `model_validate` to read only
/// `username`/`password`/`token` out of an arbitrary input dict,
/// discarding anything else even before `extra="allow"` would apply.
/// This port's derived `Deserialize` already ignores any JSON key that
/// isn't one of this struct's three fields by default (no
/// `deny_unknown_fields`), which has the identical net effect without
/// needing a hand-written override.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HttpCredentials {
    #[rusty_serde(default)]
    pub username: Option<String>,
    #[rusty_serde(default)]
    pub password: Option<String>,
    #[rusty_serde(default)]
    pub token: Option<String>,
}

impl HttpCredentials {
    pub fn new() -> Self {
        Self::default()
    }
}

/// `auth.auth_credential.HttpAuth` — C0496. `scheme` is any RFC7235
/// HTTP Authorization scheme name (e.g. `"basic"`, `"bearer"`, or any
/// other IANA-registered value) — represented as a plain `String`,
/// matching the source's own unconstrained `str` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HttpAuth {
    pub scheme: String,
    pub credentials: HttpCredentials,
    #[rusty_serde(default)]
    pub additional_headers: Option<BTreeMap<String, String>>,
}

impl HttpAuth {
    pub fn new(scheme: impl Into<String>, credentials: HttpCredentials) -> Self {
        Self {
            scheme: scheme.into(),
            credentials,
            additional_headers: None,
        }
    }
}

/// `OAuth2Auth.token_endpoint_auth_method`'s `Literal[...]` constraint,
/// ported as a real enum rather than a loosely-typed `String` — the
/// four values are fixed and exhaustive in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[rusty_serde(rename_all = "snake_case")]
pub enum TokenEndpointAuthMethod {
    #[default]
    ClientSecretBasic,
    ClientSecretPost,
    ClientSecretJwt,
    PrivateKeyJwt,
}

/// `auth.auth_credential.OAuth2Auth` — C0497.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OAuth2Auth {
    #[rusty_serde(default)]
    pub client_id: Option<String>,
    #[rusty_serde(default)]
    pub client_secret: Option<String>,
    #[rusty_serde(default)]
    pub auth_uri: Option<String>,
    #[rusty_serde(default)]
    pub nonce: Option<String>,
    #[rusty_serde(default)]
    pub state: Option<String>,
    #[rusty_serde(default)]
    pub redirect_uri: Option<String>,
    #[rusty_serde(default)]
    pub auth_response_uri: Option<String>,
    #[rusty_serde(default)]
    pub auth_code: Option<String>,
    #[rusty_serde(default)]
    pub access_token: Option<String>,
    #[rusty_serde(default)]
    pub refresh_token: Option<String>,
    #[rusty_serde(default)]
    pub id_token: Option<String>,
    #[rusty_serde(default)]
    pub expires_at: Option<i64>,
    #[rusty_serde(default)]
    pub expires_in: Option<i64>,
    #[rusty_serde(default)]
    pub audience: Option<String>,
    #[rusty_serde(default)]
    pub prompt: Option<String>,
    #[rusty_serde(default)]
    pub code_verifier: Option<String>,
    #[rusty_serde(default)]
    pub code_challenge_method: Option<String>,
    #[rusty_serde(default)]
    pub token_endpoint_auth_method: Option<TokenEndpointAuthMethod>,
}

impl Default for OAuth2Auth {
    fn default() -> Self {
        Self {
            client_id: None,
            client_secret: None,
            auth_uri: None,
            nonce: None,
            state: None,
            redirect_uri: None,
            auth_response_uri: None,
            auth_code: None,
            access_token: None,
            refresh_token: None,
            id_token: None,
            expires_at: None,
            expires_in: None,
            audience: None,
            prompt: None,
            code_verifier: None,
            code_challenge_method: None,
            token_endpoint_auth_method: Some(TokenEndpointAuthMethod::ClientSecretBasic),
        }
    }
}

impl OAuth2Auth {
    pub fn new() -> Self {
        Self::default()
    }
}

/// `auth.auth_credential.ServiceAccountCredential` — part of C0499.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceAccountCredential {
    #[rusty_serde(rename = "type", default)]
    pub type_: String,
    pub project_id: String,
    pub private_key_id: String,
    pub private_key: String,
    pub client_email: String,
    pub client_id: String,
    pub auth_uri: String,
    pub token_uri: String,
    pub auth_provider_x509_cert_url: String,
    pub client_x509_cert_url: String,
    pub universe_domain: String,
}

/// `auth.auth_credential.ServiceAccount` — C0499. Construction is
/// fallible: [`ServiceAccount::new`] runs the same two checks as the
/// source's `_validate_config` `model_validator`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceAccount {
    #[rusty_serde(default)]
    pub service_account_credential: Option<ServiceAccountCredential>,
    #[rusty_serde(default)]
    pub scopes: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub use_default_credential: bool,
    #[rusty_serde(default)]
    pub use_id_token: bool,
    #[rusty_serde(default)]
    pub audience: Option<String>,
}

/// Raised by [`ServiceAccount::new`] when the field combination the
/// source's `_validate_config` rejects is given.
#[derive(Debug, rusty_err::Error)]
pub enum ServiceAccountError {
    #[error("service_account_credential is required when use_default_credential is False.")]
    MissingCredential,
    #[error(
        "audience is required when use_id_token is True. Set it to the URL of the target \
         service (e.g. 'https://my-service.run.app')."
    )]
    MissingAudience,
}

impl ServiceAccount {
    pub fn new(
        service_account_credential: Option<ServiceAccountCredential>,
        scopes: Option<Vec<String>>,
        use_default_credential: bool,
        use_id_token: bool,
        audience: Option<String>,
    ) -> Result<Self, ServiceAccountError> {
        if !use_default_credential && service_account_credential.is_none() {
            return Err(ServiceAccountError::MissingCredential);
        }
        if use_id_token && audience.is_none() {
            return Err(ServiceAccountError::MissingAudience);
        }
        Ok(Self {
            service_account_credential,
            scopes,
            use_default_credential,
            use_id_token,
            audience,
        })
    }
}

/// `auth.auth_credential.AuthCredential` — the umbrella credential type
/// (C0493-C0497/C0499 combined). Widens
/// `adk_agents::services::AuthCredential` from its former `Value`
/// placeholder — see the module doc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthCredential {
    pub auth_type: AuthCredentialTypes,
    #[rusty_serde(default)]
    pub resource_ref: Option<String>,
    #[rusty_serde(default)]
    pub api_key: Option<String>,
    #[rusty_serde(default)]
    pub http: Option<HttpAuth>,
    #[rusty_serde(default)]
    pub service_account: Option<ServiceAccount>,
    #[rusty_serde(default)]
    pub oauth2: Option<OAuth2Auth>,
}

impl AuthCredential {
    pub fn new(auth_type: AuthCredentialTypes) -> Self {
        Self {
            auth_type,
            resource_ref: None,
            api_key: None,
            http: None,
            service_account: None,
            oauth2: None,
        }
    }

    /// C0495: an API Key credential — `AuthCredential(auth_type=API_KEY,
    /// api_key=...)`.
    pub fn api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Self::new(AuthCredentialTypes::ApiKey)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_credential_types_serialize_to_their_wire_strings() {
        assert_eq!(
            rusty_serde::json::to_value(&AuthCredentialTypes::ApiKey).unwrap(),
            rusty_serde::value::Value::String("apiKey".to_string())
        );
        assert_eq!(
            rusty_serde::json::to_value(&AuthCredentialTypes::OpenIdConnect).unwrap(),
            rusty_serde::value::Value::String("openIdConnect".to_string())
        );
        assert_eq!(
            rusty_serde::json::to_value(&AuthCredentialTypes::ServiceAccount).unwrap(),
            rusty_serde::value::Value::String("serviceAccount".to_string())
        );
    }

    #[test]
    fn auth_credential_types_round_trip_through_json() {
        let value = rusty_serde::json::to_value(&AuthCredentialTypes::OAuth2).unwrap();
        let parsed: AuthCredentialTypes = rusty_serde::json::from_value(value).unwrap();
        assert_eq!(parsed, AuthCredentialTypes::OAuth2);
    }

    #[test]
    fn http_credentials_deserialize_ignores_unknown_keys() {
        let value = rusty_serde::value::Value::Map(vec![
            (
                "username".to_string(),
                rusty_serde::value::Value::String("alice".to_string()),
            ),
            (
                "unexpected_field".to_string(),
                rusty_serde::value::Value::String("dropped".to_string()),
            ),
        ]);
        let parsed: HttpCredentials = rusty_serde::json::from_value(value).unwrap();
        assert_eq!(parsed.username.as_deref(), Some("alice"));
        assert_eq!(parsed.password, None);
    }

    #[test]
    fn api_key_credential_matches_the_source_example() {
        let credential = AuthCredential::api_key("1234");
        assert_eq!(credential.auth_type, AuthCredentialTypes::ApiKey);
        assert_eq!(credential.api_key.as_deref(), Some("1234"));
    }

    #[test]
    fn http_auth_credential_matches_the_source_example() {
        let credential = AuthCredential {
            http: Some(HttpAuth::new(
                "basic",
                HttpCredentials {
                    username: Some("user".to_string()),
                    password: Some("password".to_string()),
                    token: None,
                },
            )),
            ..AuthCredential::new(AuthCredentialTypes::Http)
        };
        let http = credential.http.as_ref().unwrap();
        assert_eq!(http.scheme, "basic");
        assert_eq!(http.credentials.username.as_deref(), Some("user"));
    }

    #[test]
    fn oauth2_auth_defaults_token_endpoint_auth_method_to_client_secret_basic() {
        let oauth2 = OAuth2Auth::new();
        assert_eq!(
            oauth2.token_endpoint_auth_method,
            Some(TokenEndpointAuthMethod::ClientSecretBasic)
        );
    }

    #[test]
    fn token_endpoint_auth_method_serializes_as_snake_case() {
        assert_eq!(
            rusty_serde::json::to_value(&TokenEndpointAuthMethod::PrivateKeyJwt).unwrap(),
            rusty_serde::value::Value::String("private_key_jwt".to_string())
        );
    }

    #[test]
    fn service_account_requires_a_credential_unless_using_default_credential() {
        let error = ServiceAccount::new(None, None, false, false, None).unwrap_err();
        assert!(matches!(error, ServiceAccountError::MissingCredential));
    }

    #[test]
    fn service_account_allows_no_credential_when_using_default_credential() {
        let account = ServiceAccount::new(None, None, true, false, None).unwrap();
        assert!(account.use_default_credential);
        assert!(account.service_account_credential.is_none());
    }

    #[test]
    fn service_account_requires_audience_when_using_id_token() {
        let error = ServiceAccount::new(None, None, true, true, None).unwrap_err();
        assert!(matches!(error, ServiceAccountError::MissingAudience));
    }

    #[test]
    fn service_account_succeeds_with_id_token_and_audience() {
        let account = ServiceAccount::new(
            None,
            None,
            true,
            true,
            Some("https://my-service.run.app".to_string()),
        )
        .unwrap();
        assert!(account.use_id_token);
        assert_eq!(
            account.audience.as_deref(),
            Some("https://my-service.run.app")
        );
    }

    #[test]
    fn service_account_credential_type_field_maps_from_the_type_key() {
        let value = rusty_serde::value::Value::Map(vec![
            (
                "type".to_string(),
                rusty_serde::value::Value::String("service_account".to_string()),
            ),
            (
                "project_id".to_string(),
                rusty_serde::value::Value::String("proj".to_string()),
            ),
            (
                "private_key_id".to_string(),
                rusty_serde::value::Value::String("kid".to_string()),
            ),
            (
                "private_key".to_string(),
                rusty_serde::value::Value::String("key".to_string()),
            ),
            (
                "client_email".to_string(),
                rusty_serde::value::Value::String("a@b.iam.gserviceaccount.com".to_string()),
            ),
            (
                "client_id".to_string(),
                rusty_serde::value::Value::String("cid".to_string()),
            ),
            (
                "auth_uri".to_string(),
                rusty_serde::value::Value::String("https://accounts.google.com".to_string()),
            ),
            (
                "token_uri".to_string(),
                rusty_serde::value::Value::String(
                    "https://oauth2.googleapis.com/token".to_string(),
                ),
            ),
            (
                "auth_provider_x509_cert_url".to_string(),
                rusty_serde::value::Value::String("https://certs".to_string()),
            ),
            (
                "client_x509_cert_url".to_string(),
                rusty_serde::value::Value::String("https://client-certs".to_string()),
            ),
            (
                "universe_domain".to_string(),
                rusty_serde::value::Value::String("googleapis.com".to_string()),
            ),
        ]);
        let parsed: ServiceAccountCredential = rusty_serde::json::from_value(value).unwrap();
        assert_eq!(parsed.type_, "service_account");
        assert_eq!(parsed.project_id, "proj");
    }
}
