//! Capability C0525 (interface half): `auth.refresher.base_credential_refresher`,
//! ported from `google.adk.auth.refresher.base_credential_refresher`.
//!
//! Credential refreshers check whether a credential is expired/needs
//! refreshing, and refresh it if necessary. The concrete
//! `OAuth2CredentialRefresher` (C0526) needing `authlib`-equivalent
//! expiry-check/HTTP-refresh machinery stays its own, still-blocked row
//! — this batch is the abstract contract and registry only.

use crate::auth_credential::AuthCredential;
use crate::auth_schemes::AuthScheme;
use crate::services::BoxFuture;

/// Base exception for credential refresh errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialRefresherError(pub String);

impl std::fmt::Display for CredentialRefresherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CredentialRefresherError {}

/// `auth.refresher.base_credential_refresher.BaseCredentialRefresher` —
/// base interface for credential refreshers.
pub trait BaseCredentialRefresher: Send + Sync {
    /// Checks if `auth_credential` needs to be refreshed.
    fn is_refresh_needed<'a>(
        &'a self,
        auth_credential: &'a AuthCredential,
        auth_scheme: Option<&'a AuthScheme>,
    ) -> BoxFuture<'a, bool>;

    /// Refreshes `auth_credential` if needed.
    fn refresh<'a>(
        &'a self,
        auth_credential: &'a AuthCredential,
        auth_scheme: Option<&'a AuthScheme>,
    ) -> BoxFuture<'a, Result<AuthCredential, CredentialRefresherError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredentialTypes;

    struct NeverExpiringRefresher;

    impl BaseCredentialRefresher for NeverExpiringRefresher {
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

    struct FailingRefresher;

    impl BaseCredentialRefresher for FailingRefresher {
        fn is_refresh_needed<'a>(
            &'a self,
            _auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, bool> {
            Box::pin(async move { true })
        }

        fn refresh<'a>(
            &'a self,
            _auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, Result<AuthCredential, CredentialRefresherError>> {
            Box::pin(async move { Err(CredentialRefresherError("refresh failed".to_string())) })
        }
    }

    #[rusty_tokio::test]
    async fn a_never_expiring_refresher_reports_no_refresh_needed() {
        let refresher = NeverExpiringRefresher;
        let credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        assert!(!refresher.is_refresh_needed(&credential, None).await);
        assert_eq!(
            refresher.refresh(&credential, None).await.unwrap(),
            credential
        );
    }

    #[rusty_tokio::test]
    async fn a_failing_refresher_returns_an_error() {
        let refresher = FailingRefresher;
        let credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        assert!(refresher.is_refresh_needed(&credential, None).await);
        let err = refresher.refresh(&credential, None).await.unwrap_err();
        assert_eq!(err.to_string(), "refresh failed");
    }
}
