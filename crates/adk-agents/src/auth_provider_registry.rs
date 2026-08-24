//! Capability C0516 (part 2): `auth.auth_provider_registry`, ported from
//! `google.adk.auth.auth_provider_registry`.
//!
//! **Registry key, narrowed**: the source keys its provider map by
//! `type[AuthScheme]` — an actual Python class object, so two distinct
//! `CustomAuthScheme` subclasses (e.g. `MySso1Scheme`/`MySso2Scheme`)
//! register and resolve independently. This port keys by
//! [`crate::base_auth_provider::AuthSchemeKind`] instead (see that
//! module's own doc) — a real, disclosed narrowing: every custom scheme
//! collapses to one [`AuthSchemeKind::Custom`] key, so at most one
//! provider can be registered for *all* custom schemes at once, not one
//! per exact subclass. `Security`/`OpenIdConnectWithConfig` aren't
//! narrowed the same way — the source's `type[AuthScheme]` for those two
//! branches is already just the one `SecurityScheme`/
//! `OpenIdConnectWithConfig` class each, matching this port's one
//! discriminant each.

use std::collections::HashMap;
use std::sync::Arc;

use crate::auth_schemes::AuthScheme;
use crate::base_auth_provider::{AuthSchemeKind, BaseAuthProvider};

/// `auth.auth_provider_registry.AuthProviderRegistry` — registry for
/// auth provider instances.
#[derive(Default)]
pub struct AuthProviderRegistry {
    providers: HashMap<AuthSchemeKind, Arc<dyn BaseAuthProvider>>,
}

impl AuthProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a provider instance for an auth scheme kind.
    pub fn register(
        &mut self,
        auth_scheme_kind: AuthSchemeKind,
        provider: Arc<dyn BaseAuthProvider>,
    ) {
        self.providers.insert(auth_scheme_kind, provider);
    }

    /// Gets the provider instance registered for the given kind, if any.
    pub fn get_provider(
        &self,
        auth_scheme_kind: AuthSchemeKind,
    ) -> Option<Arc<dyn BaseAuthProvider>> {
        self.providers.get(&auth_scheme_kind).cloned()
    }

    /// Gets the provider instance registered for an `AuthScheme`
    /// instance's kind, if any — the source's `get_provider` overload
    /// that accepts an instance rather than a type.
    pub fn get_provider_for_scheme(
        &self,
        auth_scheme: &AuthScheme,
    ) -> Option<Arc<dyn BaseAuthProvider>> {
        self.get_provider(auth_scheme_kind_of(auth_scheme))
    }
}

fn auth_scheme_kind_of(auth_scheme: &AuthScheme) -> AuthSchemeKind {
    match auth_scheme {
        AuthScheme::Security(_) => AuthSchemeKind::Security,
        AuthScheme::OpenIdConnectWithConfig(_) => AuthSchemeKind::OpenIdConnectWithConfig,
        AuthScheme::Custom(_) => AuthSchemeKind::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredential;
    use crate::auth_schemes::CustomAuthScheme;
    use crate::auth_tool::AuthConfig;
    use crate::context::CallbackContext;
    use crate::services::BoxFuture;

    struct StubProvider;

    impl BaseAuthProvider for StubProvider {
        fn get_auth_credential<'a>(
            &'a self,
            _auth_config: &'a AuthConfig,
            _context: &'a mut CallbackContext,
        ) -> BoxFuture<'a, Option<AuthCredential>> {
            Box::pin(async { None })
        }
    }

    fn custom_scheme() -> AuthScheme {
        AuthScheme::Custom(CustomAuthScheme {
            type_: "custom".to_string(),
            extra: None,
        })
    }

    #[test]
    fn get_provider_returns_none_when_nothing_is_registered() {
        let registry = AuthProviderRegistry::new();
        assert!(registry.get_provider(AuthSchemeKind::Custom).is_none());
    }

    #[test]
    fn register_and_get_provider_round_trips() {
        let mut registry = AuthProviderRegistry::new();
        registry.register(AuthSchemeKind::Custom, Arc::new(StubProvider));
        assert!(registry.get_provider(AuthSchemeKind::Custom).is_some());
        assert!(registry.get_provider(AuthSchemeKind::Security).is_none());
    }

    #[test]
    fn get_provider_for_scheme_resolves_by_the_schemes_kind() {
        let mut registry = AuthProviderRegistry::new();
        registry.register(AuthSchemeKind::Custom, Arc::new(StubProvider));
        assert!(registry.get_provider_for_scheme(&custom_scheme()).is_some());
    }

    #[test]
    fn registering_again_for_the_same_kind_replaces_the_provider() {
        let mut registry = AuthProviderRegistry::new();
        registry.register(AuthSchemeKind::Custom, Arc::new(StubProvider));
        registry.register(AuthSchemeKind::Custom, Arc::new(StubProvider));
        assert!(registry.get_provider(AuthSchemeKind::Custom).is_some());
    }
}
