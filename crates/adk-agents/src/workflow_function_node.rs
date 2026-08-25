//! Capabilities C0313-C0315 (narrowed — see below): `FunctionNode`,
//! ported from `google.adk.workflow._function_node`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc
//! for the standing crate-placement decision.
//!
//! **Reflective parameter binding/type coercion, narrowed away
//! entirely**: the bulk of the source (`_bind_parameters`/
//! `_coerce_param`/`_infer_schemas_from_func_signature`/
//! `_infer_schemas_for_state_mode`/`_type_adapters`/`get_type_hints`)
//! exists to bind a *reflectively-introspected* Python function's named
//! parameters from `ctx.state`/`node_input`, coercing each via a
//! `TypeAdapter` built from the parameter's runtime type annotation.
//! Rust has no equivalent to introspecting a closure's parameter names
//! or building a validator from a type at runtime — a wrapped function
//! here ([`FunctionNodeBody::call`]) already receives `(ctx,
//! node_input)` directly and is responsible for its own parameter
//! extraction, the same "caller supplies the resolved bits" adaptation
//! already established elsewhere in this port (e.g.
//! `adk-flows::functions::execute_function_calls`'s `tools_dict`).
//! `parameter_binding`'s `'state'`/`'node_input'` distinction and
//! `model_copy`'s bound-method rebinding accordingly have nothing left
//! to port either — both exist only to serve the reflective binding
//! this port doesn't have.
//!
//! **`_to_event`'s normalization, already subsumed**: pass-through of
//! `Event`/`RequestInput`, and wrapping any other returned value as
//! `Event(output=...)`, is already exactly what [`NodeYield`]/
//! `BaseNode::run` (C0295) do generically for every node — nothing
//! `FunctionNode`-specific to add. Per-yield `state_delta` attachment
//! is already narrower in this port by design: `workflow_node_runner
//! ::NodeRunner`'s trailing flush (C0312) moves any state changes made
//! during a node's run onto one synthesized event at the end, not onto
//! each yield as it's produced — already disclosed there.
//!
//! **What's actually left to port, and genuinely new**: the
//! `auth_config` gate (C0314) — the one piece of `_function_node.py`
//! not already covered by the above. [`FunctionNodeBody`] (a trait, not
//! a raw `Fn` closure bound — the same struct+trait "override point"
//! shape used everywhere else in this port, sidestepping the
//! higher-ranked-lifetime inference friction a generic `for<'a> Fn(...)
//! -> BoxFuture<'a, _>` bound runs into) is the thin remainder of
//! "wraps a function as a node."

use std::future::Future;
use std::pin::Pin;

use rusty_serde::value::Value;

use crate::auth_tool::AuthConfig;
use crate::context::Context;
use crate::workflow_base_node::{BaseNode, BaseNodeError, NodeBehavior, NodeRunError, NodeYield};
use crate::workflow_hitl_utils::{
    create_auth_request_event, has_auth_credential, process_auth_resume,
};
use crate::workflow_retry_config::RetryConfig;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The wrapped node logic — `FunctionNode`'s override point, analogous
/// to the source's wrapped Python function. See this module's own doc
/// for why this is a trait rather than a raw closure bound.
pub trait FunctionNodeBody: Send + Sync + 'static {
    fn call<'a>(
        &'a self,
        ctx: &'a mut Context,
        node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>>;
}

#[derive(Debug, rusty_err::Error)]
pub enum FunctionNodeError {
    #[error(
        "FunctionNode with auth_config requires rerun_on_resume=true. The node must rerun after credentials are provided."
    )]
    AuthConfigRequiresRerunOnResume,
    #[error("{0}")]
    Build(#[from] BaseNodeError),
}

struct FunctionNode {
    auth_config: Option<AuthConfig>,
    body: Box<dyn FunctionNodeBody>,
}

impl NodeBehavior for FunctionNode {
    fn run_impl<'a>(
        &'a self,
        ctx: &'a mut Context,
        node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
        Box::pin(async move {
            // --- Auth gate (C0314) ---
            if let Some(auth_config) = &self.auth_config {
                let interrupt_id = format!("wf_auth:{}", ctx.node_path());
                let auth_response = ctx.resume_inputs().get(&interrupt_id).cloned();
                if let Some(auth_response) = auth_response {
                    process_auth_resume(
                        &auth_response,
                        auth_config,
                        ctx.state_mut(),
                        &interrupt_id,
                    )
                    .map_err(|e| -> NodeRunError { e.to_string().into() })?;
                } else if !has_auth_credential(auth_config, ctx.state()) {
                    let event =
                        create_auth_request_event(auth_config, &interrupt_id, ctx.state_mut())
                            .map_err(|e| -> NodeRunError { e.to_string().into() })?;
                    return Ok(vec![NodeYield::Event(Box::new(event))]);
                }
            }

            self.body.call(ctx, node_input).await
        })
    }
}

/// `FunctionNode.__init__`: builds a [`BaseNode`] wrapping `body`,
/// gated behind `auth_config` if set. Errors if `auth_config` is set
/// without `rerun_on_resume` — the node must rerun after credentials
/// are provided, matching the source's own constructor-time check.
#[allow(clippy::too_many_arguments)]
pub fn function_node(
    name: impl Into<String>,
    rerun_on_resume: bool,
    retry_config: Option<RetryConfig>,
    timeout: Option<f64>,
    auth_config: Option<AuthConfig>,
    body: impl FunctionNodeBody,
) -> Result<BaseNode, FunctionNodeError> {
    if auth_config.is_some() && !rerun_on_resume {
        return Err(FunctionNodeError::AuthConfigRequiresRerunOnResume);
    }
    let behavior = FunctionNode {
        auth_config,
        body: Box::new(body),
    };
    BaseNode::build(
        name,
        String::new(),
        rerun_on_resume,
        false,
        retry_config,
        timeout,
        None,
        None,
        None,
        behavior,
    )
    .map_err(FunctionNodeError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::AuthCredential;
    use crate::auth_schemes::{ApiKeyIn, ApiKeyScheme, AuthScheme, SecurityScheme};
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    struct Echo;
    impl FunctionNodeBody for Echo {
        fn call<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move { Ok(vec![NodeYield::Data(node_input)]) })
        }
    }

    #[rusty_tokio::test]
    async fn runs_the_body_when_no_auth_config_is_set() {
        let node = function_node("echo", false, None, None, None, Echo).unwrap();
        let mut ctx = ctx();
        let events = node
            .run(&mut ctx, Value::String("hi".to_string()))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].output, Some(Value::String("hi".to_string())));
    }

    #[test]
    fn auth_config_without_rerun_on_resume_is_rejected() {
        let err =
            function_node("echo", false, None, None, Some(api_key_config()), Echo).unwrap_err();
        assert!(matches!(
            err,
            FunctionNodeError::AuthConfigRequiresRerunOnResume
        ));
    }

    fn api_key_config() -> AuthConfig {
        AuthConfig::new(
            AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
                description: None,
                in_: ApiKeyIn::Header,
                name: "X-Api-Key".to_string(),
            }))),
            Some(AuthCredential::api_key("placeholder")),
            None,
            Some("echo_key".to_string()),
        )
    }

    #[rusty_tokio::test]
    async fn yields_an_auth_request_when_no_credential_is_stored() {
        let node = function_node("echo", true, None, None, Some(api_key_config()), Echo).unwrap();
        let mut ctx = ctx();
        let events = node.run(&mut ctx, Value::Null).await.unwrap();
        assert_eq!(events.len(), 1);
        assert!(crate::workflow_hitl_utils::has_auth_request_function_call(
            &events[0]
        ));
    }

    #[rusty_tokio::test]
    async fn runs_the_body_once_a_credential_is_already_stored() {
        let node = function_node("echo", true, None, None, Some(api_key_config()), Echo).unwrap();
        let mut ctx = ctx();
        let credential = AuthCredential::api_key("secret");
        ctx.state_mut().set(
            "temp:echo_key",
            rusty_serde::json::to_value(&credential).unwrap(),
        );
        let events = node
            .run(&mut ctx, Value::String("hi".to_string()))
            .await
            .unwrap();
        assert_eq!(events[0].output, Some(Value::String("hi".to_string())));
    }

    #[rusty_tokio::test]
    async fn resuming_with_a_credential_stores_it_and_runs_the_body() {
        let node = function_node("echo", true, None, None, Some(api_key_config()), Echo).unwrap();
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        // A node-scoped resume context the way `NodeRunner` would build one:
        // `node_path()` is "echo@1", so the auth gate's interrupt id is
        // "wf_auth:echo@1" -- keyed here to match.
        let mut resume_inputs = std::collections::BTreeMap::new();
        resume_inputs.insert(
            "wf_auth:echo@1".to_string(),
            Value::String("resumed-key".to_string()),
        );
        let mut node_ctx = Context::for_node(
            ic,
            "",
            &[],
            None,
            "echo",
            "1",
            resume_inputs,
            1,
            false,
            true,
        );

        let events = node
            .run(&mut node_ctx, Value::String("hi".to_string()))
            .await
            .unwrap();
        assert_eq!(events[0].output, Some(Value::String("hi".to_string())));
        assert!(has_auth_credential(&api_key_config(), node_ctx.state()));
    }
}
