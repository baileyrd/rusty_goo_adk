//! Capabilities C0517/C0519/C0520/C0521 (C0518 partial — see below):
//! `CredentialManager`, ported from `google.adk.auth.credential_manager`.
//!
//! Orchestrates the full lifecycle of an authentication credential —
//! loading, exchanging, refreshing, and caching — for a single
//! [`AuthConfig`]. Every dependency this class needs already exists in
//! this port: [`AuthProviderRegistry`]/[`BaseAuthProvider`],
//! [`CredentialExchangerRegistry`]/[`BaseCredentialExchanger`],
//! [`CredentialRefresherRegistry`]/[`BaseCredentialRefresher`], and
//! [`Context`]'s already-wired `request_credential`/`get_auth_response`/
//! `load_credential`/`save_credential` methods (C0527).
//!
//! **C0518, only partially portable, disclosed**: the source's
//! `__init__` registers three concrete default implementors —
//! `OAuth2CredentialExchanger` (C0524), `ServiceAccountCredentialExchanger`,
//! and `OAuth2CredentialRefresher` (C0526) — none of which are built in
//! this port yet (all three need an authlib-equivalent HTTP credential
//! exchange, the same gap already disclosed on `auth_handler.rs`).
//! [`CredentialManager::new`] therefore registers nothing by default;
//! [`CredentialManager::register_credential_exchanger`] stays the only
//! way in until those land — the same "build the orchestrator ahead of
//! a still-blocked concrete implementor" precedent already accepted for
//! `remote_mcp_server.rs`/`environment_simulation_engine.rs`. C0518
//! itself stays `REQUIRED` in the manifest, not `DONE`, since the
//! capability it actually names (the three default registrations)
//! isn't ported.
//!
//! **`_rehydrate_custom_scheme`, dropped**: the source rehydrates a
//! generic `CustomAuthScheme` into a more specific registered subclass
//! via `__subclasses__()` runtime reflection. This port's
//! `AuthScheme::Custom` variant only ever holds a plain
//! [`crate::auth_schemes::CustomAuthScheme`] — there is no subclass
//! hierarchy to rehydrate into, so the source's `type(x) is
//! CustomAuthScheme` check (always true here) and the rehydration call
//! it guards both collapse away.
//!
//! **`_populate_auth_scheme` (OAuth2 auto-discovery), unreachable —
//! disclosed pre-existing gap**: the source only auto-discovers when
//! `auth_scheme` is an `ExtendedOAuth2` (a real Python subclass of
//! `OAuth2`, structurally compatible with the wider `AuthScheme` union
//! via duck typing). This port's `ExtendedOAuth2`
//! (`auth_schemes.rs`) was built as a separate, flattened struct
//! outside the `AuthScheme` enum entirely (`lib.rs`'s own doc: "the
//! same 'flatten inherited fields' pattern" — a tree-fusion gap in the
//! same family as C0092) — so no `AuthScheme` value in this port can
//! ever *be* one. [`CredentialManager::populate_auth_scheme`] is kept
//! as a real, called step (matching this port's own precedent for a
//! correctly-wired-but-structurally-unreachable branch, e.g.
//! `workflow_graph_validation::validate_chat_agent_wiring`) rather than
//! silently dropped, but always returns `false` given the current
//! `AuthScheme` shape. `_missing_oauth_info` itself is unaffected and
//! ports in full — it only checks the plain `OAuth2Scheme` case.
//!
//! **`hasattr(context, "request_credential")`, moot**: `CallbackContext`
//! is already a unified alias for `Context` in this port (C0048) — the
//! source's runtime check that a plain `CallbackContext` (as opposed to
//! a `ToolContext`) lacks `request_credential` has no analog here,
//! since every `Context` value already has the method.

use std::sync::{Arc, Mutex, OnceLock};

use crate::auth_credential::{AuthCredential, AuthCredentialTypes};
use crate::auth_provider_registry::AuthProviderRegistry;
use crate::auth_schemes::{AuthScheme, SecurityScheme, SecuritySchemeType};
use crate::auth_tool::AuthConfig;
use crate::base_auth_provider::BaseAuthProvider;
use crate::base_credential_exchanger::{BaseCredentialExchanger, CredentialExchangeError};
use crate::base_credential_refresher::CredentialRefresherError;
use crate::context::{Context, ContextError};
use crate::credential_exchanger_registry::CredentialExchangerRegistry;
use crate::credential_refresher_registry::CredentialRefresherRegistry;

#[derive(Debug, rusty_err::Error)]
pub enum CredentialManagerError {
    #[error(
        "No auth provider registered for custom auth scheme {0:?}. Register it using \
         `register_auth_provider`."
    )]
    NoProviderForCustomScheme(String),
    #[error("AuthProvider did not return a credential.")]
    ProviderReturnedNoCredential,
    #[error("raw_auth_credential is required for auth_scheme type {0:?}")]
    RawAuthCredentialRequired(SecuritySchemeType),
    #[error("auth_config.raw_credential.oauth2 required for credential type {0:?}")]
    MissingOAuth2Credential(AuthCredentialTypes),
    #[error("OAuth scheme info is missing, and auto-discovery has failed to fill them in.")]
    OAuthInfoMissing,
    #[error("client credentials flow requires raw_auth_credential to be set")]
    MissingRawCredentialForClientCredentialsFlow,
    #[error("{0}")]
    Context(#[from] ContextError),
    #[error("{0}")]
    Exchange(#[from] CredentialExchangeError),
    #[error("{0}")]
    Refresh(#[from] CredentialRefresherError),
}

/// `CredentialManager._auth_provider_registry`/`_registry_lock`: a
/// process-wide (not per-instance) registry — see this crate's own
/// `OnceLock<Mutex<_>>`-inside-an-accessor-fn convention (matching
/// `adk-eval::metric_evaluator_registry::default_registry` and
/// `adk-features::feature_registry`).
pub fn default_auth_provider_registry() -> &'static Mutex<AuthProviderRegistry> {
    static REGISTRY: OnceLock<Mutex<AuthProviderRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(AuthProviderRegistry::new()))
}

/// `CredentialManager.register_auth_provider` (C0517): public API for
/// registering a custom auth provider, process-wide. A collision on an
/// already-registered scheme kind is silently ignored (matching the
/// source's own `logger.warning`-then-`continue`; no logging framework
/// is adopted in this port, the same already-disclosed narrowing used
/// throughout).
pub fn register_auth_provider(provider: Arc<dyn BaseAuthProvider>) {
    let mut registry = default_auth_provider_registry()
        .lock()
        .expect("auth provider registry mutex poisoned");
    for &scheme_kind in provider.supported_auth_schemes() {
        if let Some(existing) = registry.get_provider(scheme_kind) {
            if !Arc::ptr_eq(&existing, &provider) {
                continue;
            }
        }
        registry.register(scheme_kind, provider.clone());
    }
}

/// C0517/C0519-C0521: `CredentialManager` — see this module's own doc
/// for what's narrowed.
pub struct CredentialManager {
    auth_config: AuthConfig,
    exchanger_registry: CredentialExchangerRegistry,
    refresher_registry: CredentialRefresherRegistry,
}

fn auth_scheme_type(scheme: &AuthScheme) -> Option<SecuritySchemeType> {
    match scheme {
        AuthScheme::Security(security) => Some(match security.as_ref() {
            SecurityScheme::ApiKey(_) => SecuritySchemeType::ApiKey,
            SecurityScheme::Http(_) => SecuritySchemeType::Http,
            SecurityScheme::OAuth2(_) => SecuritySchemeType::OAuth2,
            SecurityScheme::OpenIdConnect(_) => SecuritySchemeType::OpenIdConnect,
        }),
        AuthScheme::OpenIdConnectWithConfig(_) => Some(SecuritySchemeType::OpenIdConnect),
        AuthScheme::Custom(_) => None,
    }
}

impl CredentialManager {
    /// `CredentialManager.__init__` — narrowed, see this module's own
    /// doc on C0518: registers no default exchangers/refreshers.
    pub fn new(auth_config: AuthConfig) -> Self {
        Self {
            auth_config,
            exchanger_registry: CredentialExchangerRegistry::new(),
            refresher_registry: CredentialRefresherRegistry::new(),
        }
    }

    /// `CredentialManager.register_credential_exchanger` (C0521).
    pub fn register_credential_exchanger(
        &mut self,
        credential_type: AuthCredentialTypes,
        exchanger: Arc<dyn BaseCredentialExchanger>,
    ) {
        self.exchanger_registry.register(credential_type, exchanger);
    }

    /// `CredentialManager.request_credential` (C0521) — see this
    /// module's own doc for why the source's `hasattr` guard is moot
    /// here. Not `async`: the underlying `Context::request_credential`
    /// this delegates to is itself synchronous.
    pub fn request_credential(&self, context: &mut Context) -> Result<(), ContextError> {
        context.request_credential(self.auth_config.clone())
    }

    /// `CredentialManager.get_auth_credential` (C0519): the full
    /// credential-resolution state machine.
    pub async fn get_auth_credential(
        &mut self,
        context: &mut Context,
    ) -> Result<Option<AuthCredential>, CredentialManagerError> {
        // Step 0: custom-scheme dispatch.
        if let AuthScheme::Custom(custom_scheme) = &self.auth_config.auth_scheme {
            let scheme_type = custom_scheme.type_.clone();
            let provider = {
                let registry = default_auth_provider_registry()
                    .lock()
                    .expect("auth provider registry mutex poisoned");
                registry.get_provider_for_scheme(&self.auth_config.auth_scheme)
            }
            .ok_or(CredentialManagerError::NoProviderForCustomScheme(
                scheme_type,
            ))?;

            let provided_credential = provider
                .get_auth_credential(&self.auth_config, context)
                .await
                .ok_or(CredentialManagerError::ProviderReturnedNoCredential)?;

            if let Some(oauth2) = &provided_credential.oauth2 {
                if oauth2.access_token.is_none() && oauth2.auth_uri.is_some() {
                    // User consent is required: save the auth URI and
                    // return `None` to signal it's needed.
                    self.auth_config.exchanged_auth_credential = Some(provided_credential);
                    return Ok(None);
                }
            }
            return Ok(Some(provided_credential));
        }

        // Step 1: validate credential configuration.
        self.validate_credential().await?;

        // Step 2: already-ready fast path.
        let raw_auth_credential = self.auth_config.raw_auth_credential.clone();
        if self.is_credential_ready() {
            if let Some(raw) = &raw_auth_credential {
                return Ok(Some(raw.clone()));
            }
        }

        let is_service_account = raw_auth_credential
            .as_ref()
            .is_some_and(|c| c.auth_type == AuthCredentialTypes::ServiceAccount);

        // Step 3: try to load an existing processed credential.
        let mut credential = if is_service_account {
            None
        } else {
            self.load_existing_credential(context).await
        };

        // Step 4: fall back to a stored auth response.
        let mut was_from_auth_response = false;
        if credential.is_none() {
            credential = self.load_from_auth_response(context);
            was_from_auth_response = true;
        }

        // Step 5: client-credentials flow, or signal user authorization is needed.
        let mut credential = match credential {
            Some(credential) => credential,
            None => {
                if self.is_client_credentials_flow() {
                    raw_auth_credential.clone().ok_or(
                        CredentialManagerError::MissingRawCredentialForClientCredentialsFlow,
                    )?
                } else {
                    return Ok(None);
                }
            }
        };

        // Step 6: exchange (e.g. service account -> access token).
        let (exchanged, was_exchanged) = self.exchange_credential(credential).await?;
        credential = exchanged;

        // Step 7: refresh if expired (only when not already exchanged).
        let mut was_refreshed = false;
        if !was_exchanged {
            let (refreshed, refreshed_flag) = self.refresh_credential(credential).await?;
            credential = refreshed;
            was_refreshed = refreshed_flag;
        }

        // Step 8: persist if this run actually changed anything.
        if (was_from_auth_response || was_exchanged || was_refreshed) && !is_service_account {
            self.save_credential(context, credential.clone()).await?;
        }

        Ok(Some(credential))
    }

    async fn load_existing_credential(&self, context: &Context) -> Option<AuthCredential> {
        self.load_from_credential_service(context).await
    }

    async fn load_from_credential_service(&self, context: &Context) -> Option<AuthCredential> {
        context
            .load_credential(&self.auth_config)
            .await
            .ok()
            .flatten()
    }

    fn load_from_auth_response(&self, context: &Context) -> Option<AuthCredential> {
        context.get_auth_response(&self.auth_config)
    }

    async fn exchange_credential(
        &self,
        credential: AuthCredential,
    ) -> Result<(AuthCredential, bool), CredentialManagerError> {
        let Some(exchanger) = self.exchanger_registry.get_exchanger(credential.auth_type) else {
            return Ok((credential, false));
        };
        let result = exchanger
            .exchange(&credential, Some(&self.auth_config.auth_scheme))
            .await?;
        Ok((result.credential, result.was_exchanged))
    }

    async fn refresh_credential(
        &self,
        credential: AuthCredential,
    ) -> Result<(AuthCredential, bool), CredentialManagerError> {
        let Some(refresher) = self.refresher_registry.get_refresher(credential.auth_type) else {
            return Ok((credential, false));
        };
        if refresher
            .is_refresh_needed(&credential, Some(&self.auth_config.auth_scheme))
            .await
        {
            let refreshed = refresher
                .refresh(&credential, Some(&self.auth_config.auth_scheme))
                .await?;
            Ok((refreshed, true))
        } else {
            Ok((credential, false))
        }
    }

    fn is_credential_ready(&self) -> bool {
        match &self.auth_config.raw_auth_credential {
            Some(credential) => matches!(
                credential.auth_type,
                AuthCredentialTypes::ApiKey | AuthCredentialTypes::Http
            ),
            None => false,
        }
    }

    async fn validate_credential(&mut self) -> Result<(), CredentialManagerError> {
        if self.auth_config.raw_auth_credential.is_none()
            && matches!(
                auth_scheme_type(&self.auth_config.auth_scheme),
                Some(SecuritySchemeType::OAuth2) | Some(SecuritySchemeType::OpenIdConnect)
            )
        {
            return Err(CredentialManagerError::RawAuthCredentialRequired(
                auth_scheme_type(&self.auth_config.auth_scheme).expect("checked above"),
            ));
        }

        if let Some(raw_credential) = &self.auth_config.raw_auth_credential {
            if matches!(
                raw_credential.auth_type,
                AuthCredentialTypes::OAuth2 | AuthCredentialTypes::OpenIdConnect
            ) && raw_credential.oauth2.is_none()
            {
                return Err(CredentialManagerError::MissingOAuth2Credential(
                    raw_credential.auth_type,
                ));
            }
        }

        if self.missing_oauth_info() && !self.populate_auth_scheme().await {
            return Err(CredentialManagerError::OAuthInfoMissing);
        }

        Ok(())
    }

    async fn save_credential(
        &self,
        context: &mut Context,
        credential: AuthCredential,
    ) -> Result<(), CredentialManagerError> {
        if context.invocation_context().credential_service.is_none() {
            return Ok(());
        }
        let mut auth_config_to_save = self.auth_config.clone();
        auth_config_to_save.exchanged_auth_credential = Some(credential);
        context.save_credential(&auth_config_to_save).await?;
        Ok(())
    }

    /// `_populate_auth_scheme` — see this module's own doc: always
    /// `false` given the current `AuthScheme` shape's `ExtendedOAuth2`
    /// gap.
    async fn populate_auth_scheme(&mut self) -> bool {
        false
    }

    /// `_missing_oauth_info` — ports in full; unaffected by the
    /// `ExtendedOAuth2` gap since it only inspects the plain
    /// `OAuth2Scheme` case.
    fn missing_oauth_info(&self) -> bool {
        let AuthScheme::Security(security) = &self.auth_config.auth_scheme else {
            return false;
        };
        let SecurityScheme::OAuth2(oauth2_scheme) = security.as_ref() else {
            return false;
        };
        let flows = &oauth2_scheme.flows;
        flows
            .implicit
            .as_ref()
            .is_some_and(|f| f.authorization_url.is_none())
            || flows
                .password
                .as_ref()
                .is_some_and(|f| f.token_url.is_none())
            || flows
                .client_credentials
                .as_ref()
                .is_some_and(|f| f.token_url.is_none())
            || flows
                .authorization_code
                .as_ref()
                .is_some_and(|f| f.authorization_url.is_none())
            || flows
                .authorization_code
                .as_ref()
                .is_some_and(|f| f.token_url.is_none())
    }

    fn is_client_credentials_flow(&self) -> bool {
        match &self.auth_config.auth_scheme {
            AuthScheme::Security(security) => match security.as_ref() {
                SecurityScheme::OAuth2(oauth2_scheme) => {
                    oauth2_scheme.flows.client_credentials.is_some()
                }
                _ => false,
            },
            AuthScheme::OpenIdConnectWithConfig(oidc) => oidc
                .grant_types_supported
                .as_ref()
                .is_some_and(|types| types.iter().any(|t| t == "client_credentials")),
            AuthScheme::Custom(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::OAuth2Auth;
    use crate::auth_schemes::{
        ApiKeyIn, ApiKeyScheme, OAuth2Scheme, OAuthFlow, OAuthFlows, OpenIdConnectWithConfig,
    };
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    fn api_key_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })))
    }

    fn complete_oauth2_flows() -> OAuthFlows {
        OAuthFlows {
            authorization_code: Some(OAuthFlow {
                authorization_url: Some("https://example.com/authorize".to_string()),
                token_url: Some("https://example.com/token".to_string()),
                refresh_url: None,
                scopes: Default::default(),
            }),
            ..Default::default()
        }
    }

    fn oauth2_scheme(flows: OAuthFlows) -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::OAuth2(Box::new(OAuth2Scheme {
            description: None,
            flows,
        }))))
    }

    #[rusty_tokio::test]
    async fn api_key_credential_is_returned_via_the_fast_path() {
        let credential = AuthCredential::api_key("secret");
        let auth_config = AuthConfig::new(api_key_scheme(), Some(credential.clone()), None, None);
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let result = manager.get_auth_credential(&mut context).await.unwrap();
        assert_eq!(result, Some(credential));
    }

    #[rusty_tokio::test]
    async fn oauth2_without_a_raw_credential_is_rejected() {
        let auth_config = AuthConfig::new(oauth2_scheme(complete_oauth2_flows()), None, None, None);
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let err = manager.get_auth_credential(&mut context).await.unwrap_err();
        assert!(matches!(
            err,
            CredentialManagerError::RawAuthCredentialRequired(SecuritySchemeType::OAuth2)
        ));
    }

    #[rusty_tokio::test]
    async fn incomplete_oauth_flows_are_rejected_since_auto_discovery_cannot_run() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth::default());
        let incomplete_flows = OAuthFlows {
            authorization_code: Some(OAuthFlow {
                authorization_url: None,
                token_url: Some("https://example.com/token".to_string()),
                refresh_url: None,
                scopes: Default::default(),
            }),
            ..Default::default()
        };
        let auth_config = AuthConfig::new(
            oauth2_scheme(incomplete_flows),
            Some(credential),
            None,
            None,
        );
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let err = manager.get_auth_credential(&mut context).await.unwrap_err();
        assert!(matches!(err, CredentialManagerError::OAuthInfoMissing));
    }

    #[rusty_tokio::test]
    async fn client_credentials_flow_uses_the_raw_credential_directly() {
        let client_credentials_flow = OAuthFlows {
            client_credentials: Some(OAuthFlow {
                authorization_url: None,
                token_url: Some("https://example.com/token".to_string()),
                refresh_url: None,
                scopes: Default::default(),
            }),
            ..Default::default()
        };
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth::default());
        let auth_config = AuthConfig::new(
            oauth2_scheme(client_credentials_flow),
            Some(credential.clone()),
            None,
            None,
        );
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let result = manager.get_auth_credential(&mut context).await.unwrap();
        assert_eq!(result, Some(credential));
    }

    #[rusty_tokio::test]
    async fn no_existing_credential_and_no_client_credentials_flow_returns_none() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth::default());
        let auth_config = AuthConfig::new(
            oauth2_scheme(complete_oauth2_flows()),
            Some(credential),
            None,
            None,
        );
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let result = manager.get_auth_credential(&mut context).await.unwrap();
        assert_eq!(result, None);
    }

    fn oidc_scheme(grant_types_supported: Option<Vec<String>>) -> AuthScheme {
        AuthScheme::OpenIdConnectWithConfig(OpenIdConnectWithConfig {
            authorization_endpoint: "https://example.com/authorize".to_string(),
            token_endpoint: "https://example.com/token".to_string(),
            userinfo_endpoint: None,
            revocation_endpoint: None,
            token_endpoint_auth_methods_supported: None,
            grant_types_supported,
            scopes: None,
        })
    }

    #[test]
    fn is_client_credentials_flow_checks_oidc_grant_types() {
        let auth_config = AuthConfig::new(
            oidc_scheme(Some(vec!["client_credentials".to_string()])),
            None,
            None,
            None,
        );
        let manager = CredentialManager::new(auth_config);
        assert!(manager.is_client_credentials_flow());

        let auth_config = AuthConfig::new(oidc_scheme(None), None, None, None);
        let manager = CredentialManager::new(auth_config);
        assert!(!manager.is_client_credentials_flow());
    }

    struct StubProvider {
        credential: AuthCredential,
        called: Arc<AtomicBool>,
    }

    impl BaseAuthProvider for StubProvider {
        fn supported_auth_schemes(&self) -> &'static [crate::base_auth_provider::AuthSchemeKind] {
            &[crate::base_auth_provider::AuthSchemeKind::Custom]
        }

        fn get_auth_credential<'a>(
            &'a self,
            _auth_config: &'a AuthConfig,
            _context: &'a mut Context,
        ) -> crate::services::BoxFuture<'a, Option<AuthCredential>> {
            self.called.store(true, Ordering::SeqCst);
            let credential = self.credential.clone();
            Box::pin(async move { Some(credential) })
        }
    }

    // `register_auth_provider` is process-wide (the source's own
    // `_auth_provider_registry` is a class attribute, not per-instance —
    // C0517), and `AuthProviderRegistry` keys by the coarse
    // `AuthSchemeKind::Custom` bucket rather than per scheme `type_`
    // string (an already-shipped narrowing from an earlier batch). Both
    // scenarios therefore have to run in one test, in this exact order,
    // rather than as two independent `#[test]` functions: once anything
    // registers a `Custom`-kind provider, every custom scheme in this
    // process answers through it, so a separate "no provider" test
    // could pass or fail depending on unrelated test execution order.
    #[rusty_tokio::test]
    async fn custom_scheme_dispatch_errors_without_a_provider_then_succeeds_once_registered() {
        let custom_scheme = AuthScheme::Custom(crate::auth_schemes::CustomAuthScheme {
            type_: "credential_manager_test_custom_scheme".to_string(),
            extra: None,
        });
        let auth_config = AuthConfig::new(custom_scheme.clone(), None, None, None);
        let mut manager = CredentialManager::new(auth_config);
        let mut context = ctx();
        let err = manager.get_auth_credential(&mut context).await.unwrap_err();
        assert!(matches!(
            err,
            CredentialManagerError::NoProviderForCustomScheme(_)
        ));

        let called = Arc::new(AtomicBool::new(false));
        let credential = AuthCredential::api_key("from-provider");
        register_auth_provider(Arc::new(StubProvider {
            credential: credential.clone(),
            called: called.clone(),
        }));

        let auth_config = AuthConfig::new(custom_scheme, None, None, None);
        let mut manager = CredentialManager::new(auth_config);
        let result = manager.get_auth_credential(&mut context).await.unwrap();
        assert_eq!(result, Some(credential));
        assert!(called.load(Ordering::SeqCst));
    }
}
