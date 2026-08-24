//! Capability C0529: `auth.credential_service.session_state_credential_service`,
//! ported from `google.adk.auth.credential_service.session_state_credential_service`.
//!
//! Stores credentials directly under `callback_context.state[credential_key]`
//! — the source's own doc explicitly warns this "may not be secure, use
//! at your own risk," preserved verbatim below.
//!
//! **`AuthCredential`, round-tripped through `Value`**: `Context::state`
//! is a `Value`-typed map (this crate's own established shape, not
//! specific to this batch), so a credential is serialized via
//! `rusty_serde::json::to_value`/`from_value` on the way in/out rather
//! than stored as a native Rust value.
//!
//! **Explicit `Value::Null` on a missing credential, preserved**: the
//! source's `save_credential` always writes
//! `callback_context.state[key] = auth_config.exchanged_auth_credential`,
//! even when that's `None` — overwriting a prior stored value with
//! `null` rather than leaving it untouched. Unlike
//! `InMemoryCredentialService` (whose private map has no other reader),
//! session state *is* externally observable, so this port reproduces
//! the overwrite faithfully rather than skipping the write.

use crate::auth_credential::AuthCredential;
use crate::auth_tool::AuthConfig;
use crate::context::Context;
use crate::services::{BoxFuture, CredentialService};
use rusty_serde::value::Value;

/// `auth.credential_service.session_state_credential_service.SessionStateCredentialService`.
/// Storing a credential in session may not be secure — use at your own
/// risk.
#[derive(Debug, Default, Clone, Copy)]
pub struct SessionStateCredentialService;

impl SessionStateCredentialService {
    pub fn new() -> Self {
        Self
    }
}

impl CredentialService for SessionStateCredentialService {
    fn load_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a Context,
    ) -> BoxFuture<'a, Option<AuthCredential>> {
        Box::pin(async move {
            let key = auth_config.credential_key.as_deref()?;
            let value = callback_context.state().get(key)?;
            rusty_serde::json::from_value(value.clone()).ok()
        })
    }

    fn save_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let Some(key) = auth_config.credential_key.clone() else {
                return;
            };
            let value = match &auth_config.exchanged_auth_credential {
                Some(credential) => rusty_serde::json::to_value(credential).unwrap_or(Value::Null),
                None => Value::Null,
            };
            callback_context.state_mut().set(key, value);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_schemes::{AuthScheme, CustomAuthScheme};
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn scheme() -> AuthScheme {
        AuthScheme::Custom(CustomAuthScheme {
            type_: "custom".to_string(),
            extra: None,
        })
    }

    fn context() -> Context {
        let invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(invocation_context)
    }

    #[rusty_tokio::test]
    async fn load_credential_returns_none_when_unset() {
        let service = SessionStateCredentialService::new();
        let ctx = context();
        let auth_config = AuthConfig::new(scheme(), None, None, Some("key".to_string()));
        assert_eq!(service.load_credential(&auth_config, &ctx).await, None);
    }

    #[rusty_tokio::test]
    async fn save_then_load_round_trips_through_session_state() {
        let service = SessionStateCredentialService::new();
        let mut ctx = context();
        let credential = AuthCredential::api_key("secret");
        let auth_config = AuthConfig::new(
            scheme(),
            None,
            Some(credential.clone()),
            Some("key".to_string()),
        );
        service.save_credential(&auth_config, &mut ctx).await;
        assert_eq!(
            service.load_credential(&auth_config, &ctx).await,
            Some(credential)
        );
    }

    #[rusty_tokio::test]
    async fn saving_with_no_exchanged_credential_overwrites_with_null() {
        let service = SessionStateCredentialService::new();
        let mut ctx = context();
        let with_value = AuthConfig::new(
            scheme(),
            None,
            Some(AuthCredential::api_key("secret")),
            Some("key".to_string()),
        );
        service.save_credential(&with_value, &mut ctx).await;

        let without_value = AuthConfig::new(scheme(), None, None, Some("key".to_string()));
        service.save_credential(&without_value, &mut ctx).await;

        assert_eq!(ctx.state().get("key"), Some(&Value::Null));
        assert_eq!(service.load_credential(&without_value, &ctx).await, None);
    }

    #[rusty_tokio::test]
    async fn stores_directly_under_the_credential_key_in_state() {
        let service = SessionStateCredentialService::new();
        let mut ctx = context();
        let credential = AuthCredential::api_key("secret");
        let auth_config = AuthConfig::new(
            scheme(),
            None,
            Some(credential),
            Some("my_credential_key".to_string()),
        );
        service.save_credential(&auth_config, &mut ctx).await;
        assert!(ctx.state().get("my_credential_key").is_some());
    }
}
