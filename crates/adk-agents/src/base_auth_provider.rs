//! Capability C0516 (part 1): `auth.base_auth_provider`, ported from
//! `google.adk.auth.base_auth_provider`.
//!
//! **`supported_auth_schemes`, adapted**: the source returns `tuple[type[AuthScheme],
//! ...]` — actual Python *classes*, letting a subclass declare exactly
//! which `AuthScheme` subtype(s) (including a specific custom subclass)
//! it handles. Rust's [`crate::auth_schemes::AuthScheme`] is a closed
//! enum, not an open class hierarchy, so this port returns
//! [`AuthSchemeKind`] discriminants instead — see
//! `auth_provider_registry`'s own module doc for the resulting
//! registry-side narrowing (one provider per discriminant, not one per
//! exact custom subclass).
//!
//! **`@experimental(FeatureName.PLUGGABLE_AUTH)`, not yet wired**: unlike
//! `app_configs`'s C0283/C0284 (this session's first real call site for
//! the *bare* `@experimental`/`check_feature_enabled` guard functions),
//! this trait has no single natural call site to fire a check from — a
//! Rust trait itself can't run code at "declaration" time the way a
//! Python class decorator does, and every concrete implementor is still
//! unbuilt (`CredentialManager`, C0517, the only real caller, is a
//! separate unported row). Left as a documented gap rather than forced
//! onto an arbitrary method.

use crate::auth_credential::AuthCredential;
use crate::auth_tool::AuthConfig;
use crate::context::CallbackContext;
use crate::services::BoxFuture;

/// A discriminant for [`crate::auth_schemes::AuthScheme`]'s variants —
/// see the module doc for why this replaces the source's `type[AuthScheme]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AuthSchemeKind {
    Security,
    OpenIdConnectWithConfig,
    Custom,
}

/// `auth.base_auth_provider.BaseAuthProvider` — abstract base for custom
/// authentication providers.
pub trait BaseAuthProvider: Send + Sync {
    /// The `AuthScheme` kinds this provider supports. Empty by default
    /// (matches the source's `return ()`); a provider overrides this to
    /// enable single-argument registration.
    fn supported_auth_schemes(&self) -> &'static [AuthSchemeKind] {
        &[]
    }

    /// Provides an `AuthCredential` asynchronously, or `None` if
    /// unavailable.
    fn get_auth_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        context: &'a mut CallbackContext,
    ) -> BoxFuture<'a, Option<AuthCredential>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticProvider {
        credential: Option<AuthCredential>,
    }

    impl BaseAuthProvider for StaticProvider {
        fn supported_auth_schemes(&self) -> &'static [AuthSchemeKind] {
            &[AuthSchemeKind::Custom]
        }

        fn get_auth_credential<'a>(
            &'a self,
            _auth_config: &'a AuthConfig,
            _context: &'a mut CallbackContext,
        ) -> BoxFuture<'a, Option<AuthCredential>> {
            let credential = self.credential.clone();
            Box::pin(async move { credential })
        }
    }

    #[test]
    fn default_supported_auth_schemes_is_empty() {
        struct Minimal;
        impl BaseAuthProvider for Minimal {
            fn get_auth_credential<'a>(
                &'a self,
                _auth_config: &'a AuthConfig,
                _context: &'a mut CallbackContext,
            ) -> BoxFuture<'a, Option<AuthCredential>> {
                Box::pin(async { None })
            }
        }
        assert_eq!(Minimal.supported_auth_schemes(), &[] as &[AuthSchemeKind]);
    }

    #[rusty_tokio::test]
    async fn a_provider_returns_its_configured_credential() {
        use crate::auth_credential::AuthCredentialTypes;
        use crate::auth_schemes::{AuthScheme, CustomAuthScheme};
        use crate::invocation_context::InvocationContextBuilder;
        use crate::session::Session;

        let credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        let provider = StaticProvider {
            credential: Some(credential.clone()),
        };
        let auth_config = AuthConfig::new(
            AuthScheme::Custom(CustomAuthScheme {
                type_: "custom".to_string(),
                extra: None,
            }),
            None,
            None,
            None,
        );
        let invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let mut context = CallbackContext::new(invocation_context);
        let result = provider
            .get_auth_credential(&auth_config, &mut context)
            .await;
        assert_eq!(result, Some(credential));
    }
}
