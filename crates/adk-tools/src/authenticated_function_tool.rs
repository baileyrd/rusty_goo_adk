//! C0412: `tools.authenticated_function_tool`, ported from
//! `google.adk.tools.authenticated_function_tool`.
//!
//! A [`FunctionTool`] that resolves an [`AuthConfig`] credential via
//! [`CredentialManager`] before invocation, injecting the resolved
//! credential into the call args before delegating to the wrapped
//! function — same credential-gating shape as
//! [`crate::base_authenticated_tool::BaseAuthenticatedTool`], composed
//! around a [`FunctionTool`] instead of an arbitrary closure.
//!
//! **Composition instead of inheritance**: the source subclasses
//! `FunctionTool` directly. This port instead wraps an already-built
//! [`FunctionTool`] as a field — the same composition-over-inheritance
//! shape [`BaseAuthenticatedTool`] and `FunctionTool` itself already
//! established — and delegates every [`BaseTool`] method except
//! `run_async` straight through to it.
//!
//! **`_ignore_params.append("credential")`/`inspect.signature`, moot**:
//! the source's `_ignore_params` mechanism excludes `credential` from
//! the auto-derived `FunctionDeclaration` its reflection-based
//! `FunctionTool` base builds from the wrapped callable's real Python
//! signature; `inspect.signature(self.func)` then re-checks that same
//! real signature at call time to decide whether to actually inject
//! `credential` into `args_to_call`. Neither mechanism has anything to
//! act on here: this port's [`FunctionTool`] takes an already-built,
//! hand-written [`FunctionDeclaration`] rather than deriving one via
//! reflection (see that module's own doc), so a caller building a
//! credential-consuming function's declaration simply omits `credential`
//! from it directly — there is no auto-derived schema to filter — and
//! every wrapped closure has the same fixed `(&BTreeMap<String, Value>,
//! &mut ToolContext)` shape regardless of whether it happens to use a
//! `credential` key, so [`AuthenticatedFunctionTool::run_async`]
//! unconditionally inserts `"credential"` into `args_to_call` rather than
//! probing for a "does the function want it" signature that doesn't
//! exist in Rust.

use std::collections::BTreeMap;

use adk_agents::auth_credential::AuthCredential;
use adk_agents::auth_tool::AuthConfig;
use adk_agents::credential_manager::CredentialManager;
use adk_features::feature_decorator::{check_feature_enabled, FeatureNotEnabledError};
use adk_features::feature_registry::FeatureName;
use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::function_tool::FunctionTool;
use crate::tool_context::ToolContext;

const CREDENTIAL_ARG_KEY: &str = "credential";

/// C0412: `authenticated_function_tool.AuthenticatedFunctionTool`. See
/// the module doc for the composition and `_ignore_params` adaptations.
pub struct AuthenticatedFunctionTool {
    inner: FunctionTool,
    credentials_manager: Option<rusty_tokio::sync::Mutex<CredentialManager>>,
    response_for_auth_required: Option<Value>,
}

impl AuthenticatedFunctionTool {
    /// `AuthenticatedFunctionTool.__init__`.
    pub fn new(
        inner: FunctionTool,
        auth_config: Option<AuthConfig>,
        response_for_auth_required: Option<Value>,
    ) -> Result<Self, FeatureNotEnabledError> {
        check_feature_enabled(FeatureName::AuthenticatedFunctionTool)?;
        let credentials_manager = auth_config
            .map(CredentialManager::new)
            .map(rusty_tokio::sync::Mutex::new);
        Ok(Self {
            inner,
            credentials_manager,
            response_for_auth_required,
        })
    }
}

impl BaseTool for AuthenticatedFunctionTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        self.inner.get_declaration()
    }

    fn check_require_confirmation<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        self.inner.check_require_confirmation(args, tool_context)
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

            let mut args_to_call = args.clone();
            args_to_call.insert(
                CREDENTIAL_ARG_KEY.to_string(),
                credential
                    .map(|credential| {
                        rusty_serde::json::to_value(&credential).unwrap_or(Value::Null)
                    })
                    .unwrap_or(Value::Null),
            );
            self.inner.run_async(&args_to_call, tool_context).await
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
    use std::sync::Arc;

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

    fn echo_function_tool() -> FunctionTool {
        FunctionTool::new(
            "echo",
            "echoes back its args, including any injected credential",
            FunctionDeclaration {
                name: Some("echo".to_string()),
                ..Default::default()
            },
            vec![],
            Arc::new(|args, _ctx| {
                let args = Value::Map(args.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
                Box::pin(async move { args })
            }),
        )
    }

    #[test]
    fn errors_when_the_feature_is_disabled() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::AuthenticatedFunctionTool, false);
        let result = AuthenticatedFunctionTool::new(echo_function_tool(), None, None);
        assert!(result.is_err());
    }

    #[rusty_tokio::test]
    async fn without_an_auth_config_credential_is_null_in_the_call_args() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::AuthenticatedFunctionTool, true);
        let tool = AuthenticatedFunctionTool::new(echo_function_tool(), None, None).unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result.get(CREDENTIAL_ARG_KEY), Some(&Value::Null));
    }

    #[rusty_tokio::test]
    async fn without_a_ready_credential_it_requests_one_and_returns_the_placeholder() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::AuthenticatedFunctionTool, true);
        let auth_config = api_key_auth_config(None);
        let tool =
            AuthenticatedFunctionTool::new(echo_function_tool(), Some(auth_config), None).unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::String("Pending User Authorization.".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn with_a_ready_credential_it_injects_the_credential_into_the_call_args() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::AuthenticatedFunctionTool, true);
        let raw = AuthCredential::api_key("k".to_string());
        let auth_config = api_key_auth_config(Some(raw.clone()));
        let tool =
            AuthenticatedFunctionTool::new(echo_function_tool(), Some(auth_config), None).unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let expected_credential = rusty_serde::json::to_value(&raw).unwrap();
        assert_eq!(result.get("value"), Some(&Value::String("hi".to_string())));
        assert_eq!(result.get(CREDENTIAL_ARG_KEY), Some(&expected_credential));
    }

    #[rusty_tokio::test]
    async fn name_and_description_delegate_to_the_inner_function_tool() {
        let _guard = TemporaryFeatureOverride::new(FeatureName::AuthenticatedFunctionTool, true);
        let tool = AuthenticatedFunctionTool::new(echo_function_tool(), None, None).unwrap();
        assert_eq!(tool.name(), "echo");
        assert_eq!(
            tool.description(),
            "echoes back its args, including any injected credential"
        );
    }
}
