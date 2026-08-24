//! Capability C0528: `auth.credential_service.in_memory_credential_service`,
//! ported from `google.adk.auth.credential_service.in_memory_credential_service`.
//!
//! **`app_name`/`user_id`, adapted**: the source reads
//! `callback_context._invocation_context.app_name`/`.user_id` directly.
//! This port's `InvocationContext` has no such direct fields — only
//! nested under `.session.app_name`/`.session.user_id` (this crate's own
//! established shape) — so this reads through `.session` instead; same
//! values, one more level of nesting.
//!
//! **Value-vs-absent-key, disclosed as a non-issue**: the source's
//! `save_credential` always writes into its dict, even when
//! `exchanged_auth_credential` is `None` — but `dict.get(key)` returns
//! `None` whether the key is absent or explicitly mapped to `None`,
//! and nothing else ever inspects this service's private map. So this
//! port simply skips the insert when there's nothing to store, rather
//! than modeling an `Option<AuthCredential>` value type — behaviorally
//! identical through the only observable interface (`load_credential`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::auth_credential::AuthCredential;
use crate::auth_tool::AuthConfig;
use crate::context::Context;
use crate::services::{BoxFuture, CredentialService};

/// app_name -> user_id -> credential_key -> credential.
type CredentialBuckets = BTreeMap<String, BTreeMap<String, BTreeMap<String, AuthCredential>>>;

/// `auth.credential_service.in_memory_credential_service.InMemoryCredentialService`
/// — a process-local credential store, not persisted across restarts.
#[derive(Default)]
pub struct InMemoryCredentialService {
    credentials: Mutex<CredentialBuckets>,
}

impl InMemoryCredentialService {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CredentialService for InMemoryCredentialService {
    fn load_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a Context,
    ) -> BoxFuture<'a, Option<AuthCredential>> {
        Box::pin(async move {
            let session = &callback_context.invocation_context().session;
            let key = auth_config.credential_key.as_deref()?;
            let credentials = self
                .credentials
                .lock()
                .expect("in-memory credential service mutex poisoned");
            credentials
                .get(&session.app_name)
                .and_then(|by_user| by_user.get(&session.user_id))
                .and_then(|by_key| by_key.get(key))
                .cloned()
        })
    }

    fn save_credential<'a>(
        &'a self,
        auth_config: &'a AuthConfig,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let session = &callback_context.invocation_context().session;
            let Some(key) = auth_config.credential_key.clone() else {
                return;
            };
            let mut credentials = self
                .credentials
                .lock()
                .expect("in-memory credential service mutex poisoned");
            let bucket = credentials
                .entry(session.app_name.clone())
                .or_default()
                .entry(session.user_id.clone())
                .or_default();
            match &auth_config.exchanged_auth_credential {
                Some(credential) => {
                    bucket.insert(key, credential.clone());
                }
                None => {
                    bucket.remove(&key);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredentialTypes;
    use crate::auth_schemes::{AuthScheme, CustomAuthScheme};
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn scheme() -> AuthScheme {
        AuthScheme::Custom(CustomAuthScheme {
            type_: "custom".to_string(),
            extra: None,
        })
    }

    fn context_for(app_name: &str, user_id: &str) -> Context {
        let invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new(app_name, user_id, "s1")).build();
        Context::new(invocation_context)
    }

    #[rusty_tokio::test]
    async fn load_credential_returns_none_when_nothing_saved() {
        let service = InMemoryCredentialService::new();
        let context = context_for("app", "user");
        let auth_config = AuthConfig::new(scheme(), None, None, Some("key".to_string()));
        assert_eq!(service.load_credential(&auth_config, &context).await, None);
    }

    #[rusty_tokio::test]
    async fn save_then_load_round_trips_the_credential() {
        let service = InMemoryCredentialService::new();
        let mut context = context_for("app", "user");
        let credential = AuthCredential::api_key("secret");
        let auth_config = AuthConfig::new(
            scheme(),
            None,
            Some(credential.clone()),
            Some("key".to_string()),
        );
        service.save_credential(&auth_config, &mut context).await;
        assert_eq!(
            service.load_credential(&auth_config, &context).await,
            Some(credential)
        );
    }

    #[rusty_tokio::test]
    async fn credentials_are_scoped_by_app_and_user() {
        let service = InMemoryCredentialService::new();
        let mut context_a = context_for("app-a", "user-1");
        let context_b = context_for("app-b", "user-1");
        let credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        let auth_config =
            AuthConfig::new(scheme(), None, Some(credential), Some("key".to_string()));
        service.save_credential(&auth_config, &mut context_a).await;
        assert_eq!(
            service.load_credential(&auth_config, &context_b).await,
            None
        );
    }

    #[rusty_tokio::test]
    async fn saving_with_no_exchanged_credential_clears_a_prior_value() {
        let service = InMemoryCredentialService::new();
        let mut context = context_for("app", "user");
        let auth_config_with_value = AuthConfig::new(
            scheme(),
            None,
            Some(AuthCredential::api_key("secret")),
            Some("key".to_string()),
        );
        service
            .save_credential(&auth_config_with_value, &mut context)
            .await;

        let auth_config_without_value =
            AuthConfig::new(scheme(), None, None, Some("key".to_string()));
        service
            .save_credential(&auth_config_without_value, &mut context)
            .await;

        assert_eq!(
            service
                .load_credential(&auth_config_without_value, &context)
                .await,
            None
        );
    }
}
