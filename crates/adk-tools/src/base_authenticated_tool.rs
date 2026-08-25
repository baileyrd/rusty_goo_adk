//! C0412: `tools.base_authenticated_tool`, ported from
//! `google.adk.tools.base_authenticated_tool`.
//!
//! A [`BaseTool`] that resolves an [`AuthConfig`] credential via
//! [`CredentialManager`] before invocation, returning a configurable
//! "Pending User Authorization" placeholder if no credential is yet
//! available instead of running the wrapped logic.
//!
//! **Composition instead of an abstract method, same as `FunctionTool`**:
//! the source is an `ABC` with an abstract `_run_async_impl(args,
//! tool_context, credential)` every subclass overrides. Rust has no
//! abstract methods, so [`BaseAuthenticatedTool`] instead takes a boxed
//! [`AuthenticatedRunFn`] closure at construction — the same
//! composition-over-inheritance shape [`crate::function_tool::FunctionTool`]
//! already established for its own wrapped closure.
//!
//! **`if auth_config and auth_config.auth_scheme:`, collapses**: the
//! source's guard checks two things — `auth_config` being falsy (`None`),
//! and `auth_config.auth_scheme` being falsy (`None`, since Python's
//! `auth_scheme` is itself optional at the type-hint level even though
//! rarely actually unset). This port's [`AuthConfig::auth_scheme`] is a
//! plain, non-`Option` [`adk_agents::auth_schemes::AuthScheme`] — a
//! constructed `AuthConfig` can never have a falsy `auth_scheme` — so the
//! guard collapses to just `auth_config.is_some()`.
//!
//! **`CredentialManager`, behind an async mutex**: [`BaseTool::run_async`]
//! takes `&self`, but [`CredentialManager::get_auth_credential`] needs
//! `&mut self` and is itself `async`. A [`std::sync::Mutex`] guard can't
//! safely be held across an `.await`, so the credential manager is
//! wrapped in [`rusty_tokio::sync::Mutex`] instead — an async-aware
//! mutex, locked for the duration of the credential-resolution call and
//! released before the wrapped run implementation is invoked.
//!
//! **Error propagation, adapted**: the source lets any exception from
//! `get_auth_credential`/`request_credential` propagate uncaught (there's
//! no `try`/`except` around either call). This port surfaces both as
//! [`ToolError::CredentialResolutionFailed`] instead, since [`BaseTool::
//! run_async`] already returns a `Result` rather than raising.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_agents::auth_credential::AuthCredential;
use adk_agents::auth_tool::AuthConfig;
use adk_agents::credential_manager::CredentialManager;
use adk_features::feature_decorator::{check_feature_enabled, FeatureNotEnabledError};
use adk_features::feature_registry::FeatureName;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

/// The wrapped `_run_async_impl` closure's shape: takes the raw tool-call
/// args, a mutable [`ToolContext`], and the resolved credential (`None`
/// when the tool has no [`AuthConfig`] to resolve), returns the tool's
/// JSON result.
pub type AuthenticatedRunFn = Arc<
    dyn for<'a> Fn(
            &'a BTreeMap<String, Value>,
            &'a mut ToolContext,
            Option<AuthCredential>,
        ) -> BoxFuture<'a, Value>
        + Send
        + Sync,
>;

/// C0412: `base_authenticated_tool.BaseAuthenticatedTool`. See the module
/// doc for the composition-over-inheritance and async-mutex adaptations.
pub struct BaseAuthenticatedTool {
    name: String,
    description: String,
    credentials_manager: Option<rusty_tokio::sync::Mutex<CredentialManager>>,
    response_for_auth_required: Option<Value>,
    run_async_impl: AuthenticatedRunFn,
}

impl BaseAuthenticatedTool {
    /// `BaseAuthenticatedTool.__init__`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        auth_config: Option<AuthConfig>,
        response_for_auth_required: Option<Value>,
        run_async_impl: AuthenticatedRunFn,
    ) -> Result<Self, FeatureNotEnabledError> {
        check_feature_enabled(FeatureName::BaseAuthenticatedTool)?;
        let credentials_manager = auth_config
            .map(CredentialManager::new)
            .map(rusty_tokio::sync::Mutex::new);
        Ok(Self {
            name: name.into(),
            description: description.into(),
            credentials_manager,
            response_for_auth_required,
            run_async_impl,
        })
    }
}

impl BaseTool for BaseAuthenticatedTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let mut credential: Option<AuthCredential> = None;
            if let Some(manager) = &self.credentials_manager {
                let mut manager = manager.lock().await;
                credential = manager
                    .get_auth_credential(tool_context)
                    .await
                    .map_err(|error| ToolError::CredentialResolutionFailed(error.to_string()))?;
                if credential.is_none() {
                    manager.request_credential(tool_context).map_err(|error| {
                        ToolError::CredentialResolutionFailed(error.to_string())
                    })?;
                    return Ok(self.response_for_auth_required.clone().unwrap_or_else(|| {
                        Value::String("Pending User Authorization.".to_string())
                    }));
                }
            }

            Ok((self.run_async_impl)(args, tool_context, credential).await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::auth_schemes::{ApiKeyIn, ApiKeyScheme, AuthScheme, SecurityScheme};
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_features::feature_registry::TemporaryFeatureOverride;

    fn ctx() -> Context {
        let mut context = Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        context.set_function_call_id(Some("fc-1".to_string()));
        context
    }

    fn api_key_auth_config(raw: Option<AuthCredential>) -> AuthConfig {
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })));
        AuthConfig::new(scheme, raw, None, None)
    }

    fn echo_impl() -> AuthenticatedRunFn {
        Arc::new(|args, _ctx, credential| {
            let value = args.get("value").cloned().unwrap_or(Value::Null);
            let credential_present = credential.is_some();
            Box::pin(async move {
                Value::Map(vec![
                    ("value".to_string(), value),
                    (
                        "had_credential".to_string(),
                        Value::Bool(credential_present),
                    ),
                ])
            })
        })
    }

    #[test]
    fn errors_when_the_feature_is_disabled() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::BaseAuthenticatedTool, false);
        let result = BaseAuthenticatedTool::new("t", "d", None, None, echo_impl());
        assert!(result.is_err());
    }

    #[rusty_tokio::test]
    async fn without_an_auth_config_it_skips_straight_to_the_inner_impl() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::BaseAuthenticatedTool, true);
        let tool = BaseAuthenticatedTool::new("t", "d", None, None, echo_impl()).unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::Map(vec![
                ("value".to_string(), Value::String("hi".to_string())),
                ("had_credential".to_string(), Value::Bool(false)),
            ])
        );
    }

    #[rusty_tokio::test]
    async fn without_a_ready_credential_it_requests_one_and_returns_the_placeholder() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::BaseAuthenticatedTool, true);
        let auth_config = api_key_auth_config(None);
        let tool =
            BaseAuthenticatedTool::new("t", "d", Some(auth_config), None, echo_impl()).unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::String("Pending User Authorization.".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn a_custom_response_for_auth_required_is_used_instead_of_the_default() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::BaseAuthenticatedTool, true);
        let auth_config = api_key_auth_config(None);
        let custom = Value::String("please authenticate".to_string());
        let tool = BaseAuthenticatedTool::new(
            "t",
            "d",
            Some(auth_config),
            Some(custom.clone()),
            echo_impl(),
        )
        .unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, custom);
    }

    #[rusty_tokio::test]
    async fn with_a_ready_credential_it_invokes_the_inner_impl_with_it() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::BaseAuthenticatedTool, true);
        let raw = AuthCredential::api_key("k".to_string());
        let auth_config = api_key_auth_config(Some(raw));
        let tool =
            BaseAuthenticatedTool::new("t", "d", Some(auth_config), None, echo_impl()).unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::Map(vec![
                ("value".to_string(), Value::String("hi".to_string())),
                ("had_credential".to_string(), Value::Bool(true)),
            ])
        );
    }
}
