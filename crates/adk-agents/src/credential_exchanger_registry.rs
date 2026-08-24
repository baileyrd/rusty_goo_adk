//! Capability C0523 (registry half): `auth.exchanger.credential_exchanger_registry`,
//! ported from `google.adk.auth.exchanger.credential_exchanger_registry`.
//!
//! **Registry key**: the source keys by `AuthCredentialTypes` — an
//! already-closed, non-open enum (unlike `AuthProviderRegistry`'s
//! `type[AuthScheme]`, where a real Python class needed a discriminant
//! adaptation), so this registers directly on that enum with no
//! narrowing needed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::auth_credential::AuthCredentialTypes;
use crate::base_credential_exchanger::BaseCredentialExchanger;

/// `auth.exchanger.credential_exchanger_registry.CredentialExchangerRegistry`
/// — registry for credential exchanger instances.
#[derive(Default)]
pub struct CredentialExchangerRegistry {
    exchangers: HashMap<AuthCredentialTypes, Arc<dyn BaseCredentialExchanger>>,
}

impl CredentialExchangerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an exchanger instance for a credential type.
    pub fn register(
        &mut self,
        credential_type: AuthCredentialTypes,
        exchanger: Arc<dyn BaseCredentialExchanger>,
    ) {
        self.exchangers.insert(credential_type, exchanger);
    }

    /// Gets the exchanger instance for a credential type, if registered.
    pub fn get_exchanger(
        &self,
        credential_type: AuthCredentialTypes,
    ) -> Option<Arc<dyn BaseCredentialExchanger>> {
        self.exchangers.get(&credential_type).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredential;
    use crate::auth_schemes::AuthScheme;
    use crate::base_credential_exchanger::{CredentialExchangeError, ExchangeResult};
    use crate::services::BoxFuture;

    struct StubExchanger;

    impl BaseCredentialExchanger for StubExchanger {
        fn exchange<'a>(
            &'a self,
            auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, Result<ExchangeResult, CredentialExchangeError>> {
            let credential = auth_credential.clone();
            Box::pin(async move {
                Ok(ExchangeResult {
                    credential,
                    was_exchanged: true,
                })
            })
        }
    }

    #[test]
    fn get_exchanger_returns_none_when_nothing_is_registered() {
        let registry = CredentialExchangerRegistry::new();
        assert!(registry
            .get_exchanger(AuthCredentialTypes::OAuth2)
            .is_none());
    }

    #[test]
    fn register_and_get_exchanger_round_trips() {
        let mut registry = CredentialExchangerRegistry::new();
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubExchanger));
        assert!(registry
            .get_exchanger(AuthCredentialTypes::OAuth2)
            .is_some());
        assert!(registry
            .get_exchanger(AuthCredentialTypes::ApiKey)
            .is_none());
    }

    #[test]
    fn registering_again_for_the_same_type_replaces_the_exchanger() {
        let mut registry = CredentialExchangerRegistry::new();
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubExchanger));
        registry.register(AuthCredentialTypes::OAuth2, Arc::new(StubExchanger));
        assert!(registry
            .get_exchanger(AuthCredentialTypes::OAuth2)
            .is_some());
    }
}
