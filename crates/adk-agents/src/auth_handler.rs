//! Capabilities C0506, C0507, C0508: `auth.auth_handler.AuthHandler`,
//! ported from `google.adk.auth.auth_handler`.
//!
//! **`exchange_auth_token`, not ported**: the source always constructs a
//! concrete `OAuth2CredentialExchanger()` internally and calls
//! `.exchange(...)` on it. This port only has the abstract contract
//! (`base_credential_exchanger::BaseCredentialExchanger`, C0523) — the
//! concrete `OAuth2CredentialExchanger` (C0524) stays its own blocked row,
//! needing the same missing authlib-equivalent HTTP-exchange machinery
//! `create_oauth2_session` (C0530) is blocked on. Since there is nothing
//! for `exchange_auth_token` to actually call, it is left unported rather
//! than stubbed against a hypothetical injected exchanger nobody can
//! provide yet.
//!
//! **[`AuthHandler::parse_and_store_auth_response`], `Partial:`**: writes
//! `auth_config.exchanged_auth_credential` into `state["temp:" +
//! credential_key]` exactly as the source does, and returns early for a
//! non-OAuth2/OIDC scheme (both fully ported, no blocker). For an
//! OAuth2/OIDC scheme the source *overwrites* that same key with the
//! result of `exchange_auth_token` — the piece this port can't do (see
//! above), so an OAuth2/OIDC scheme's credential is left un-exchanged in
//! state here, a real and disclosed narrowing.
//!
//! **[`AuthHandler::generate_auth_uri`], `Partial:`**: the source tries
//! `authlib`'s `OAuth2Session` first and only falls back to a bare
//! `raw_auth_credential.model_copy(deep=True)` (or `None`) when
//! `AUTHLIB_AVAILABLE` is `False`. This port has no authlib-equivalent
//! OAuth2 client (same missing-crate gap C0530/C0524/C0526 are blocked
//! on), so it always takes that fallback branch — which is also,
//! genuinely, this port's entire *reachable* behavior today, not a
//! simplification of something else that would otherwise run.
//! [`resolve_authorization_endpoint_and_scopes`] still ports the source's
//! flow-priority endpoint/scope resolution as a pure, tested, standalone
//! function — the "widen/build a placeholder ahead of its still-blocked
//! caller" precedent (`reflect_retry_utils.rs`,
//! `runner::get_function_responses_from_content`): once an OAuth2 client
//! lands, `generate_auth_uri` has this ready to call instead of
//! reimplementing it from scratch. The source's PKCE
//! `code_verifier`-generation and `code_challenge_method == 'S256'`
//! validation live entirely inside the authlib-only branch this port
//! never reaches, so neither is ported.
//!
//! **`_validate`/the source's `not auth_scheme` checks, N/A**: this
//! port's [`crate::auth_tool::AuthConfig::auth_scheme`] is a required
//! `AuthScheme` field, not `Optional` — so the source's `if not
//! self.auth_config.auth_scheme` guards (`_validate`,
//! `_build_credential_from_string`'s no-scheme branch) describe a state
//! this port's type system already makes unreachable.
//! [`AuthHandler::validate`] is still ported as a no-op for parity (it is
//! never called by any other source method either — grep-verified — so
//! this matches the source's own dead-code shape, not a narrowing).
//!
//! **`isinstance(val, AuthCredential)`, structural**: same "Value
//! round-trip, not Python identity" adaptation `functions.rs`'s
//! `as_function_response_part` (C0195) already established — nothing in
//! this port ever stores a native `AuthCredential` directly inside
//! `State` (only its `Value` serialization), so the source's
//! `isinstance(val, AuthCredential)` and `isinstance(val, dict)` branches
//! collapse into one `Value::Map` case here.

use crate::auth_credential::{
    AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials, OAuth2Auth,
};
use crate::auth_schemes::{AuthScheme, SecurityScheme, SecuritySchemeType};
use crate::auth_tool::AuthConfig;
use crate::oauth2_util::{normalize_oauth_scopes, OAuthScopes};
use crate::state::State;

use rusty_serde::value::Value;

/// Raised by [`AuthHandler::generate_auth_request`] — matches the
/// source's `raise ValueError(...)` messages verbatim (parameterized by
/// the scheme's [`AuthScheme::type_name`]).
#[derive(Debug, rusty_err::Error)]
pub enum AuthHandlerError {
    #[error("credential_key is empty.")]
    EmptyCredentialKey,
    #[error("Auth Scheme {0} requires auth_credential.")]
    MissingAuthCredential(String),
    #[error("Auth Scheme {0} requires oauth2 in auth_credential.")]
    MissingOAuth2Credential(String),
    #[error(
        "Auth Scheme {0} requires both client_id and client_secret in auth_credential.oauth2."
    )]
    MissingClientIdOrSecret(String),
}

/// `auth.auth_handler.AuthHandler` — see the module doc for what's
/// ported in full versus disclosed as `Partial:`.
pub struct AuthHandler {
    pub auth_config: AuthConfig,
}

impl AuthHandler {
    pub fn new(auth_config: AuthConfig) -> Self {
        Self { auth_config }
    }

    /// `AuthHandler._validate` — see the module doc for why this is
    /// unconditionally `Ok`.
    pub fn validate(&self) -> Result<(), AuthHandlerError> {
        Ok(())
    }

    /// `AuthHandler.parse_and_store_auth_response` — see the module doc
    /// for the `Partial:` disclosure on its OAuth2/OIDC branch.
    pub fn parse_and_store_auth_response(&self, state: &mut State) -> Result<(), AuthHandlerError> {
        let credential_key = self
            .auth_config
            .credential_key
            .as_deref()
            .filter(|key| !key.is_empty())
            .ok_or(AuthHandlerError::EmptyCredentialKey)?;
        let temp_credential_key = format!("temp:{credential_key}");

        let value = self
            .auth_config
            .exchanged_auth_credential
            .as_ref()
            .map(|credential| {
                rusty_serde::json::to_value(credential).expect("AuthCredential always serializes")
            })
            .unwrap_or(Value::Null);
        state.set(temp_credential_key, value);

        // The OAuth2/OIDC branch would overwrite the key above with
        // `exchange_auth_token`'s result here — unported, see the module
        // doc.
        Ok(())
    }

    /// `AuthHandler.get_auth_response` — reads `state["temp:" +
    /// credential_key]`, falling back to the bare `credential_key`.
    pub fn get_auth_response(&self, state: &State) -> Option<AuthCredential> {
        let credential_key = self.auth_config.credential_key.as_deref()?;
        let temp_credential_key = format!("temp:{credential_key}");

        if let Some(value) = state.get(&temp_credential_key) {
            if let Some(credential) = self.credential_from_value(value) {
                return Some(credential);
            }
        }
        if let Some(value) = state.get(credential_key) {
            if let Some(credential) = self.credential_from_value(value) {
                return Some(credential);
            }
        }
        None
    }

    fn credential_from_value(&self, value: &Value) -> Option<AuthCredential> {
        match value {
            Value::Map(_) => rusty_serde::json::from_value(value.clone()).ok(),
            Value::String(token) if !token.is_empty() => {
                Some(self.build_credential_from_string(token))
            }
            _ => None,
        }
    }

    /// `AuthHandler._build_credential_from_string` (C0507) — maps a bare
    /// token string to a credential shaped by the configured scheme,
    /// falling back to an OAuth2 access token for every other scheme
    /// (OAuth2/OIDC explicitly, `Custom` implicitly — matching the
    /// source's own `else` catch-all).
    pub fn build_credential_from_string(&self, token: &str) -> AuthCredential {
        match &self.auth_config.auth_scheme {
            AuthScheme::Security(security) => match security.as_ref() {
                SecurityScheme::ApiKey(_) => AuthCredential {
                    api_key: Some(token.to_string()),
                    ..AuthCredential::new(AuthCredentialTypes::ApiKey)
                },
                SecurityScheme::Http(http_scheme) => AuthCredential {
                    // The source's `getattr(auth_scheme, "scheme",
                    // "bearer")`: this port's `HttpScheme.scheme` is a
                    // required field, never absent, so the "bearer"
                    // default never applies here.
                    http: Some(HttpAuth::new(
                        http_scheme.scheme.clone(),
                        HttpCredentials {
                            token: Some(token.to_string()),
                            ..Default::default()
                        },
                    )),
                    ..AuthCredential::new(AuthCredentialTypes::Http)
                },
                SecurityScheme::OAuth2(_) | SecurityScheme::OpenIdConnect(_) => {
                    oauth2_access_token_credential(token)
                }
            },
            AuthScheme::OpenIdConnectWithConfig(_) | AuthScheme::Custom(_) => {
                oauth2_access_token_credential(token)
            }
        }
    }

    /// `AuthHandler.generate_auth_request` (C0508).
    pub fn generate_auth_request(&self) -> Result<AuthConfig, AuthHandlerError> {
        if !is_oauth2_or_openid_connect_scheme(&self.auth_config.auth_scheme) {
            return Ok(self.auth_config.clone());
        }

        let already_has_auth_uri = self
            .auth_config
            .exchanged_auth_credential
            .as_ref()
            .and_then(|credential| credential.oauth2.as_ref())
            .and_then(|oauth2| oauth2.auth_uri.as_ref())
            .is_some();
        if already_has_auth_uri {
            return Ok(self.auth_config.clone());
        }

        let type_name = self.auth_config.auth_scheme.type_name();

        let Some(raw_credential) = &self.auth_config.raw_auth_credential else {
            return Err(AuthHandlerError::MissingAuthCredential(type_name));
        };
        let Some(oauth2) = &raw_credential.oauth2 else {
            return Err(AuthHandlerError::MissingOAuth2Credential(type_name));
        };

        if oauth2.auth_uri.is_some() {
            return Ok(AuthConfig {
                auth_scheme: self.auth_config.auth_scheme.clone(),
                raw_auth_credential: self.auth_config.raw_auth_credential.clone(),
                exchanged_auth_credential: self.auth_config.raw_auth_credential.clone(),
                credential_key: self.auth_config.credential_key.clone(),
            });
        }

        if oauth2.client_id.is_none() || oauth2.client_secret.is_none() {
            return Err(AuthHandlerError::MissingClientIdOrSecret(type_name));
        }

        Ok(AuthConfig {
            auth_scheme: self.auth_config.auth_scheme.clone(),
            raw_auth_credential: self.auth_config.raw_auth_credential.clone(),
            exchanged_auth_credential: self.generate_auth_uri(),
            credential_key: self.auth_config.credential_key.clone(),
        })
    }

    /// `AuthHandler.generate_auth_uri` — see the module doc for why this
    /// port always takes the source's `not AUTHLIB_AVAILABLE` fallback.
    pub fn generate_auth_uri(&self) -> Option<AuthCredential> {
        self.auth_config.raw_auth_credential.clone()
    }
}

fn oauth2_access_token_credential(token: &str) -> AuthCredential {
    AuthCredential {
        oauth2: Some(OAuth2Auth {
            access_token: Some(token.to_string()),
            ..OAuth2Auth::default()
        }),
        ..AuthCredential::new(AuthCredentialTypes::OAuth2)
    }
}

/// The source's `isinstance(auth_scheme, SecurityBase) and auth_scheme.type_
/// in (oauth2, openIdConnect)` guard, ported structurally: both
/// [`SecurityScheme::OAuth2`]/[`SecurityScheme::OpenIdConnect`] (fastapi's
/// plain OIDC pointer) and [`AuthScheme::OpenIdConnectWithConfig`] subclass
/// `SecurityBase` in the source, so all three match here; [`AuthScheme::Custom`]
/// does not (it never subclasses `SecurityBase`), regardless of what string
/// its own `type` field happens to hold.
fn is_oauth2_or_openid_connect_scheme(auth_scheme: &AuthScheme) -> bool {
    match auth_scheme {
        AuthScheme::Security(security) => matches!(
            security.scheme_type(),
            SecuritySchemeType::OAuth2 | SecuritySchemeType::OpenIdConnect
        ),
        AuthScheme::OpenIdConnectWithConfig(_) => true,
        AuthScheme::Custom(_) => false,
    }
}

/// The pure half of `generate_auth_uri`'s authlib-only branch — see the
/// module doc for why it's ported standalone, ahead of its still-blocked
/// caller. Mirrors the source's exact flow-priority resolution: for an
/// OAuth2 scheme, the first populated flow wins in `implicit`,
/// `authorizationCode`, `clientCredentials`, `password` order (reading
/// `authorizationUrl` for the first two, `tokenUrl` for the last two,
/// matching the source's own per-flow field choice); for
/// [`AuthScheme::OpenIdConnectWithConfig`], its own
/// `authorization_endpoint`/`scopes`. Returns `(None, [])` for any scheme
/// that isn't OAuth2/OIDC shaped.
pub fn resolve_authorization_endpoint_and_scopes(
    auth_scheme: &AuthScheme,
) -> (Option<String>, Vec<String>) {
    match auth_scheme {
        AuthScheme::OpenIdConnectWithConfig(oidc) => {
            let scopes = oidc.scopes.clone().map(OAuthScopes::List);
            (
                Some(oidc.authorization_endpoint.clone()),
                normalize_oauth_scopes(scopes.as_ref()),
            )
        }
        AuthScheme::Security(security) => match security.as_ref() {
            SecurityScheme::OAuth2(oauth2_scheme) => {
                let flows = &oauth2_scheme.flows;
                let endpoint = flows
                    .implicit
                    .as_ref()
                    .and_then(|flow| flow.authorization_url.clone())
                    .or_else(|| {
                        flows
                            .authorization_code
                            .as_ref()
                            .and_then(|flow| flow.authorization_url.clone())
                    })
                    .or_else(|| {
                        flows
                            .client_credentials
                            .as_ref()
                            .and_then(|flow| flow.token_url.clone())
                    })
                    .or_else(|| {
                        flows
                            .password
                            .as_ref()
                            .and_then(|flow| flow.token_url.clone())
                    });

                let scopes = flows
                    .implicit
                    .as_ref()
                    .or(flows.authorization_code.as_ref())
                    .or(flows.client_credentials.as_ref())
                    .or(flows.password.as_ref())
                    .map(|flow| OAuthScopes::Described(flow.scopes.clone()));

                (endpoint, normalize_oauth_scopes(scopes.as_ref()))
            }
            _ => (None, Vec::new()),
        },
        AuthScheme::Custom(_) => (None, Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_schemes::{
        ApiKeyIn, ApiKeyScheme, HttpScheme, OAuth2Scheme, OAuthFlow, OAuthFlows,
        OpenIdConnectWithConfig,
    };
    use std::collections::BTreeMap;

    fn empty_state() -> State {
        State::new(BTreeMap::new(), BTreeMap::new())
    }

    fn oauth2_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::OAuth2(Box::new(OAuth2Scheme {
            description: None,
            flows: OAuthFlows {
                authorization_code: Some(OAuthFlow {
                    authorization_url: Some("https://example.com/authorize".to_string()),
                    token_url: Some("https://example.com/token".to_string()),
                    refresh_url: None,
                    scopes: BTreeMap::from([("read".to_string(), "Read access".to_string())]),
                }),
                ..Default::default()
            },
        }))))
    }

    fn api_key_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })))
    }

    fn config_with_scheme(auth_scheme: AuthScheme) -> AuthConfig {
        AuthConfig::new(auth_scheme, None, None, Some("my_key".to_string()))
    }

    #[test]
    fn get_auth_response_returns_none_without_a_credential_key() {
        let config = AuthConfig::new(api_key_scheme(), None, None, None);
        let handler = AuthHandler::new(AuthConfig {
            credential_key: None,
            ..config
        });
        assert_eq!(handler.get_auth_response(&empty_state()), None);
    }

    #[test]
    fn get_auth_response_reads_the_temp_prefixed_key_first() {
        let handler = AuthHandler::new(config_with_scheme(api_key_scheme()));
        let credential = AuthCredential::api_key("secret");
        let mut state = empty_state();
        state.set(
            "temp:my_key",
            rusty_serde::json::to_value(&credential).unwrap(),
        );

        assert_eq!(handler.get_auth_response(&state), Some(credential));
    }

    #[test]
    fn get_auth_response_falls_back_to_the_bare_key() {
        let handler = AuthHandler::new(config_with_scheme(api_key_scheme()));
        let credential = AuthCredential::api_key("secret");
        let mut state = empty_state();
        state.set("my_key", rusty_serde::json::to_value(&credential).unwrap());

        assert_eq!(handler.get_auth_response(&state), Some(credential));
    }

    #[test]
    fn get_auth_response_builds_a_credential_from_a_bare_string() {
        let handler = AuthHandler::new(config_with_scheme(api_key_scheme()));
        let mut state = empty_state();
        state.set("temp:my_key", Value::String("raw-token".to_string()));

        let credential = handler.get_auth_response(&state).unwrap();
        assert_eq!(credential.api_key.as_deref(), Some("raw-token"));
    }

    #[test]
    fn build_credential_from_string_shapes_an_api_key_credential() {
        let handler = AuthHandler::new(config_with_scheme(api_key_scheme()));
        let credential = handler.build_credential_from_string("secret");
        assert_eq!(credential.auth_type, AuthCredentialTypes::ApiKey);
        assert_eq!(credential.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn build_credential_from_string_shapes_an_http_credential() {
        let http_scheme = AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "basic".to_string(),
            bearer_format: None,
        })));
        let handler = AuthHandler::new(config_with_scheme(http_scheme));
        let credential = handler.build_credential_from_string("secret");
        assert_eq!(credential.auth_type, AuthCredentialTypes::Http);
        let http = credential.http.unwrap();
        assert_eq!(http.scheme, "basic");
        assert_eq!(http.credentials.token.as_deref(), Some("secret"));
    }

    #[test]
    fn build_credential_from_string_falls_back_to_oauth2_for_oauth2_and_custom_schemes() {
        let oauth2_handler = AuthHandler::new(config_with_scheme(oauth2_scheme()));
        let oauth2_credential = oauth2_handler.build_credential_from_string("token");
        assert_eq!(oauth2_credential.auth_type, AuthCredentialTypes::OAuth2);
        assert_eq!(
            oauth2_credential.oauth2.unwrap().access_token.as_deref(),
            Some("token")
        );

        let custom_scheme = AuthScheme::Custom(crate::auth_schemes::CustomAuthScheme {
            type_: "my_custom".to_string(),
            extra: None,
        });
        let custom_handler = AuthHandler::new(config_with_scheme(custom_scheme));
        let custom_credential = custom_handler.build_credential_from_string("token");
        assert_eq!(custom_credential.auth_type, AuthCredentialTypes::OAuth2);
    }

    #[test]
    fn generate_auth_request_returns_a_clone_for_a_non_oauth2_scheme() {
        let handler = AuthHandler::new(config_with_scheme(api_key_scheme()));
        let result = handler.generate_auth_request().unwrap();
        assert_eq!(result, handler.auth_config);
    }

    #[test]
    fn generate_auth_request_returns_a_clone_when_an_auth_uri_is_already_exchanged() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.exchanged_auth_credential = Some(AuthCredential {
            oauth2: Some(OAuth2Auth {
                auth_uri: Some("https://example.com/authorize?state=1".to_string()),
                ..OAuth2Auth::default()
            }),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        });
        let handler = AuthHandler::new(config);
        let result = handler.generate_auth_request().unwrap();
        assert_eq!(result, handler.auth_config);
    }

    #[test]
    fn generate_auth_request_errors_without_a_raw_credential() {
        let handler = AuthHandler::new(config_with_scheme(oauth2_scheme()));
        assert!(handler.generate_auth_request().is_err());
    }

    #[test]
    fn generate_auth_request_errors_without_oauth2_in_the_raw_credential() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.raw_auth_credential = Some(AuthCredential::api_key("not-oauth2"));
        let handler = AuthHandler::new(config);
        assert!(handler.generate_auth_request().is_err());
    }

    #[test]
    fn generate_auth_request_errors_without_client_id_or_secret() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.raw_auth_credential = Some(AuthCredential {
            oauth2: Some(OAuth2Auth::default()),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        });
        let handler = AuthHandler::new(config);
        assert!(handler.generate_auth_request().is_err());
    }

    #[test]
    fn generate_auth_request_reuses_an_auth_uri_already_present_on_the_raw_credential() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.raw_auth_credential = Some(AuthCredential {
            oauth2: Some(OAuth2Auth {
                auth_uri: Some("https://example.com/authorize?state=2".to_string()),
                ..OAuth2Auth::default()
            }),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        });
        let handler = AuthHandler::new(config);
        let result = handler.generate_auth_request().unwrap();
        assert_eq!(
            result.exchanged_auth_credential,
            handler.auth_config.raw_auth_credential
        );
    }

    #[test]
    fn generate_auth_request_falls_through_to_generate_auth_uri() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.raw_auth_credential = Some(AuthCredential {
            oauth2: Some(OAuth2Auth {
                client_id: Some("id".to_string()),
                client_secret: Some("secret".to_string()),
                ..OAuth2Auth::default()
            }),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        });
        let handler = AuthHandler::new(config);
        let result = handler.generate_auth_request().unwrap();
        assert_eq!(
            result.exchanged_auth_credential,
            handler.auth_config.raw_auth_credential
        );
    }

    #[test]
    fn generate_auth_uri_deep_copies_the_raw_credential() {
        let mut config = config_with_scheme(oauth2_scheme());
        config.raw_auth_credential = Some(AuthCredential::api_key("raw"));
        let handler = AuthHandler::new(config);
        assert_eq!(
            handler.generate_auth_uri(),
            handler.auth_config.raw_auth_credential
        );
    }

    #[test]
    fn generate_auth_uri_returns_none_without_a_raw_credential() {
        let handler = AuthHandler::new(config_with_scheme(oauth2_scheme()));
        assert_eq!(handler.generate_auth_uri(), None);
    }

    #[test]
    fn parse_and_store_auth_response_errors_on_an_empty_credential_key() {
        let mut config = config_with_scheme(api_key_scheme());
        config.credential_key = Some(String::new());
        let handler = AuthHandler::new(config);
        assert!(handler
            .parse_and_store_auth_response(&mut empty_state())
            .is_err());
    }

    #[test]
    fn parse_and_store_auth_response_writes_the_exchanged_credential_under_the_temp_key() {
        let mut config = config_with_scheme(api_key_scheme());
        config.exchanged_auth_credential = Some(AuthCredential::api_key("secret"));
        let handler = AuthHandler::new(config);
        let mut state = empty_state();

        handler.parse_and_store_auth_response(&mut state).unwrap();

        let stored = state.get("temp:my_key").unwrap();
        let credential: AuthCredential = rusty_serde::json::from_value(stored.clone()).unwrap();
        assert_eq!(credential.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn resolve_authorization_endpoint_and_scopes_prefers_implicit_over_authorization_code() {
        let mut config = config_with_scheme(oauth2_scheme());
        if let AuthScheme::Security(security) = &mut config.auth_scheme {
            if let SecurityScheme::OAuth2(oauth2_scheme) = security.as_mut() {
                oauth2_scheme.flows.implicit = Some(OAuthFlow {
                    authorization_url: Some("https://example.com/implicit".to_string()),
                    token_url: None,
                    refresh_url: None,
                    scopes: BTreeMap::from([("write".to_string(), "Write access".to_string())]),
                });
            }
        }
        let (endpoint, scopes) = resolve_authorization_endpoint_and_scopes(&config.auth_scheme);
        assert_eq!(endpoint.as_deref(), Some("https://example.com/implicit"));
        assert_eq!(scopes, vec!["write".to_string()]);
    }

    #[test]
    fn resolve_authorization_endpoint_and_scopes_uses_the_authorization_code_flow() {
        let (endpoint, scopes) = resolve_authorization_endpoint_and_scopes(&oauth2_scheme());
        assert_eq!(endpoint.as_deref(), Some("https://example.com/authorize"));
        assert_eq!(scopes, vec!["read".to_string()]);
    }

    #[test]
    fn resolve_authorization_endpoint_and_scopes_reads_open_id_connect_with_config() {
        let scheme = AuthScheme::OpenIdConnectWithConfig(OpenIdConnectWithConfig {
            authorization_endpoint: "https://example.com/oidc/authorize".to_string(),
            token_endpoint: "https://example.com/oidc/token".to_string(),
            userinfo_endpoint: None,
            revocation_endpoint: None,
            token_endpoint_auth_methods_supported: None,
            grant_types_supported: None,
            scopes: Some(vec!["openid".to_string()]),
        });
        let (endpoint, scopes) = resolve_authorization_endpoint_and_scopes(&scheme);
        assert_eq!(
            endpoint.as_deref(),
            Some("https://example.com/oidc/authorize")
        );
        assert_eq!(scopes, vec!["openid".to_string()]);
    }

    #[test]
    fn resolve_authorization_endpoint_and_scopes_is_empty_for_a_non_oauth2_scheme() {
        let (endpoint, scopes) = resolve_authorization_endpoint_and_scopes(&api_key_scheme());
        assert_eq!(endpoint, None);
        assert_eq!(scopes, Vec::<String>::new());
    }
}
