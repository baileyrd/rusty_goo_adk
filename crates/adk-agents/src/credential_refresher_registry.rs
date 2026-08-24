//! Capability C0525 (registry half): `auth.refresher.credential_refresher_registry`,
//! ported from `google.adk.auth.refresher.credential_refresher_registry`.
//!
//! Same shape as `credential_exchanger_registry.rs` (C0523) — keyed
//! directly on `AuthCredentialTypes`, no discriminant adaptation needed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::auth_credential::AuthCredentialTypes;
use crate::base_credential_refresher::BaseCredentialRefresher;

/// `auth.refresher.credential_refresher_registry.CredentialRefresherRegistry`
/// — registry for credential refresher instances.
#[derive(Default)]
pub struct CredentialRefresherRegistry {
    refreshers: HashMap<AuthCredentialTypes, Arc<dyn BaseCredentialRefresher>>,
}

impl CredentialRefresherRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a refresher instance for a credential type.
    pub fn register(
        &mut self,
        credential_type: AuthCredentialTypes,
        refresher: Arc<dyn BaseCredentialRefresher>,
    ) {
        self.refreshers.insert(credential_type, refresher);
    }

    /// Gets the refresher instance for a credential type, if registered.
    pub fn get_refresher(
        &self,
        credential_type: AuthCredentialTypes,
    ) -> Option<Arc<dyn BaseCredentialRefresher>> {
        self.refreshers.get(&credential_type).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredential;
    use crate::auth_schemes::AuthScheme;
    use crate::base_credential_refresher::CredentialRefresherError;
    use crate::services::BoxFuture;

    struct StubRefresher;

    impl BaseCredentialRefresher for StubRefresher {
        fn is_refresh_needed<'a>(
            &'a self,
            _auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, bool> {
            Box::pin(async move { false })
        }

        fn refresh<'a>(
            &'a self,
            auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, Result<AuthCredential, CredentialRefresherError>> {
            let credential = auth_credential.clone();
            Box::pin(async move { Ok(credential) })
        }
    }

    #[test]
    fn get_refresher_returns_none_when_nothing_is_registered() {
        let registry = CredentialRefresherRegistry::new();
        assert!(registry
            .get_refresher(AuthCredentialTypes::OAuth2)
            .is_none());
    }

    #[test]
    fn register_and_get_refresher_round_trips() {
        let mut registry = CredentialRefresherRegistry::new();
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubRefresher));
        assert!(registry
            .get_refresher(AuthCredentialTypes::OAuth2)
            .is_some());
        assert!(registry
            .get_refresher(AuthCredentialTypes::ApiKey)
            .is_none());
    }

    #[test]
    fn registering_again_for_the_same_type_replaces_the_refresher() {
        let mut registry = CredentialRefresherRegistry::new();
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubRefresher));
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubRefresher));
        assert!(registry
            .get_refresher(AuthCredentialTypes::OAuth2)
            .is_some());
    }
}
