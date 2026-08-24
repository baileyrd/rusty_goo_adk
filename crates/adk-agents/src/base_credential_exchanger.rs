//! Capability C0523 (interface half): `auth.exchanger.base_credential_exchanger`,
//! ported from `google.adk.auth.exchanger.base_credential_exchanger`.
//!
//! Credential exchangers are responsible for exchanging credentials from
//! one format or scheme to another (e.g. an OAuth2 authorization code
//! for an access token). The concrete `OAuth2CredentialExchanger`
//! (C0524) needing `authlib`-equivalent HTTP exchange machinery stays
//! its own, still-blocked row — this batch is the abstract contract and
//! registry only.

use crate::auth_credential::AuthCredential;
use crate::auth_schemes::AuthScheme;
use crate::services::BoxFuture;

/// Base exception for credential exchange errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialExchangeError(pub String);

impl std::fmt::Display for CredentialExchangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CredentialExchangeError {}

/// `auth.exchanger.base_credential_exchanger.ExchangeResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExchangeResult {
    pub credential: AuthCredential,
    pub was_exchanged: bool,
}

/// `auth.exchanger.base_credential_exchanger.BaseCredentialExchanger` —
/// base interface for credential exchangers.
pub trait BaseCredentialExchanger: Send + Sync {
    /// Exchanges `auth_credential` if needed.
    fn exchange<'a>(
        &'a self,
        auth_credential: &'a AuthCredential,
        auth_scheme: Option<&'a AuthScheme>,
    ) -> BoxFuture<'a, Result<ExchangeResult, CredentialExchangeError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredentialTypes;

    struct NoopExchanger;

    impl BaseCredentialExchanger for NoopExchanger {
        fn exchange<'a>(
            &'a self,
            auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, Result<ExchangeResult, CredentialExchangeError>> {
            let credential = auth_credential.clone();
            Box::pin(async move {
                Ok(ExchangeResult {
                    credential,
                    was_exchanged: false,
                })
            })
        }
    }

    struct FailingExchanger;

    impl BaseCredentialExchanger for FailingExchanger {
        fn exchange<'a>(
            &'a self,
            _auth_credential: &'a AuthCredential,
            _auth_scheme: Option<&'a AuthScheme>,
        ) -> BoxFuture<'a, Result<ExchangeResult, CredentialExchangeError>> {
            Box::pin(async move { Err(CredentialExchangeError("exchange failed".to_string())) })
        }
    }

    #[rusty_tokio::test]
    async fn a_noop_exchanger_returns_the_credential_unexchanged() {
        let exchanger = NoopExchanger;
        let credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        let result = exchanger.exchange(&credential, None).await.unwrap();
        assert_eq!(result.credential, credential);
        assert!(!result.was_exchanged);
    }

    #[rusty_tokio::test]
    async fn a_failing_exchanger_returns_an_error() {
        let exchanger = FailingExchanger;
        let credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        let err = exchanger.exchange(&credential, None).await.unwrap_err();
        assert_eq!(err.to_string(), "exchange failed");
    }
}
