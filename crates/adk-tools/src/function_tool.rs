//! Capability C0404 (partial): `FunctionTool`, ported from
//! `google.adk.tools.function_tool`.
//!
//! **Adaptation, fundamental**: the source wraps an arbitrary Python
//! callable and uses runtime reflection (`inspect.signature`,
//! `get_type_hints`) to: detect whether/where it wants a `ToolContext`
//! parameter, build a `FunctionDeclaration` from its signature
//! (`_automatic_function_calling_util.build_function_declaration`, cached
//! via `functools.lru_cache`), coerce raw JSON args into the exact
//! Pydantic-model parameter types the function declares
//! (`_preprocess_args`), and compute which parameters are mandatory
//! (`_get_mandatory_args`, from `inspect.signature`'s default-value info).
//! None of this is available in Rust: functions have no runtime-inspectable
//! signatures, and static typing means a caller must already know its
//! parameter types at compile time. So this port's `FunctionTool`:
//! - always calls its wrapped closure with `(&BTreeMap<String, Value>, &mut
//!   ToolContext)` — no context-parameter detection needed, since the
//!   signature is fixed rather than discovered.
//! - takes an already-built [`FunctionDeclaration`] and an explicit
//!   `required_args` list from its constructor, rather than deriving
//!   either from the wrapped closure. Building a `FunctionDeclaration`
//!   from a Rust function signature (the equivalent of
//!   `build_function_declaration`) needs compile-time reflection (e.g. a
//!   proc macro) this port doesn't have yet.
//! - does no argument type-coercion (`_preprocess_args`'s Pydantic
//!   conversion): a Rust closure body converts its own args out of the
//!   `Value` map (typically via `rusty_serde::json::from_value`) however
//!   it needs to, since there's no schema-driven layer above it to do this
//!   generically.
//!
//! **Not** ported: `input_stream` injection (live/bidirectional-streaming
//! tools) — needs `InvocationContext::active_streaming_tools` wiring this
//! port's `adk-agents` doesn't consume yet (its own module doc discloses
//! the same gap); the sync/async callable-runner distinction
//! (`_SYNC_CALLABLE_RUNNER`/`_invoke_callable`) — every closure this
//! wrapper calls is already async by construction, so there's no sync
//! case to special-case; `_detect_error_in_response` (a telemetry hook,
//! Phase 12, not built).

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_confirmation::ToolConfirmation;
use crate::tool_context::ToolContext;

/// The wrapped closure's shape: takes the raw tool-call args and a mutable
/// [`ToolContext`], returns the tool's JSON result. Always async, and
/// always takes both parameters — see the module doc for why no context-
/// parameter auto-detection is needed here.
pub type ToolFn = Arc<
    dyn for<'a> Fn(&'a BTreeMap<String, Value>, &'a mut ToolContext) -> BoxFuture<'a, Value>
        + Send
        + Sync,
>;

/// A predicate deciding, per call, whether a tool invocation requires
/// confirmation — the closure form of `require_confirmation`.
pub type RequireConfirmationFn = Arc<
    dyn for<'a> Fn(&'a BTreeMap<String, Value>, &'a mut ToolContext) -> BoxFuture<'a, bool>
        + Send
        + Sync,
>;

/// `Union[bool, Callable[..., bool]]` for `require_confirmation`.
#[derive(Clone)]
pub enum RequireConfirmation {
    Bool(bool),
    Predicate(RequireConfirmationFn),
}

impl From<bool> for RequireConfirmation {
    fn from(value: bool) -> Self {
        RequireConfirmation::Bool(value)
    }
}

/// C0404: wraps a Rust closure as a [`BaseTool`]. See the module doc for
/// the scope this port narrows the source's runtime-reflection-driven
/// behavior down to.
pub struct FunctionTool {
    name: String,
    description: String,
    declaration: FunctionDeclaration,
    required_args: Vec<String>,
    require_confirmation: RequireConfirmation,
    func: ToolFn,
}

impl FunctionTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        declaration: FunctionDeclaration,
        required_args: Vec<String>,
        func: ToolFn,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            declaration,
            required_args,
            require_confirmation: RequireConfirmation::Bool(false),
            func,
        }
    }

    pub fn with_require_confirmation(
        mut self,
        require_confirmation: impl Into<RequireConfirmation>,
    ) -> Self {
        self.require_confirmation = require_confirmation.into();
        self
    }

    fn mandatory_args(&self) -> &[String] {
        &self.required_args
    }
}

impl BaseTool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(self.declaration.clone())
    }

    fn check_require_confirmation<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        Box::pin(async move {
            match &self.require_confirmation {
                RequireConfirmation::Bool(value) => *value,
                RequireConfirmation::Predicate(predicate) => predicate(args, tool_context).await,
            }
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let missing: Vec<&String> = self
                .mandatory_args()
                .iter()
                .filter(|name| !args.contains_key(*name))
                .collect();
            if !missing.is_empty() {
                let missing_str = missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let error = format!(
                    "Invoking `{}()` failed as the following mandatory input parameters are not present:\n{missing_str}\nYou could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters.",
                    self.name
                );
                return Ok(error_response(error));
            }

            if self.check_require_confirmation(args, tool_context).await {
                match tool_context.tool_confirmation() {
                    None => {
                        let _ = tool_context.request_confirmation(
                            Some(format!(
                                "Please approve or reject the tool call {}() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
                                self.name
                            )),
                            None,
                        );
                        tool_context.actions_mut().skip_summarization = true;
                        return Ok(error_response(
                            "This tool call requires confirmation, please approve or reject."
                                .to_string(),
                        ));
                    }
                    Some(confirmation_value) => {
                        let confirmed = rusty_serde::json::from_value::<ToolConfirmation>(
                            confirmation_value.clone(),
                        )
                        .map(|confirmation| confirmation.confirmed)
                        .unwrap_or(false);
                        if !confirmed {
                            return Ok(error_response("This tool call is rejected.".to_string()));
                        }
                    }
                }
            }

            Ok((self.func)(args, tool_context).await)
        })
    }
}

fn error_response(error: String) -> Value {
    Value::Map(vec![("error".to_string(), Value::String(error))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    fn echo_tool() -> FunctionTool {
        FunctionTool::new(
            "echo",
            "echoes back its `value` argument",
            FunctionDeclaration {
                name: Some("echo".to_string()),
                ..Default::default()
            },
            vec!["value".to_string()],
            Arc::new(|args, _ctx| {
                let value = args.get("value").cloned().unwrap_or(Value::Null);
                Box::pin(async move { value })
            }),
        )
    }

    #[rusty_tokio::test]
    async fn runs_the_wrapped_closure_and_returns_its_result() {
        let tool = echo_tool();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String("hi".to_string()));
    }

    #[rusty_tokio::test]
    async fn reports_missing_mandatory_arguments_without_invoking_the_closure() {
        let tool = echo_tool();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert!(fields.iter().any(|(k, _)| k == "error"));
            }
            other => panic!("expected an error map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn default_require_confirmation_is_false() {
        let tool = echo_tool();
        let mut context = ctx();
        let args = BTreeMap::new();
        assert!(!tool.check_require_confirmation(&args, &mut context).await);
    }

    #[rusty_tokio::test]
    async fn requests_confirmation_and_skips_summarization_on_first_call() {
        let tool = echo_tool().with_require_confirmation(true);
        let mut context = ctx();
        context.set_function_call_id(Some("fc-1".to_string()));
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => assert!(fields.iter().any(|(k, _)| k == "error")),
            other => panic!("expected an error map, got {other:?}"),
        }
        assert!(context.actions().skip_summarization);
        assert!(context
            .actions()
            .requested_tool_confirmations
            .contains_key("fc-1"));
    }

    #[rusty_tokio::test]
    async fn runs_the_tool_once_confirmation_is_confirmed() {
        let tool = echo_tool().with_require_confirmation(true);
        let mut context = ctx();
        context.set_tool_confirmation(Some(
            rusty_serde::json::to_value(&ToolConfirmation {
                hint: String::new(),
                confirmed: true,
                payload: None,
            })
            .unwrap(),
        ));
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String("hi".to_string()));
    }

    #[rusty_tokio::test]
    async fn rejects_the_tool_call_when_confirmation_is_denied() {
        let tool = echo_tool().with_require_confirmation(true);
        let mut context = ctx();
        context.set_tool_confirmation(Some(
            rusty_serde::json::to_value(&ToolConfirmation {
                hint: String::new(),
                confirmed: false,
                payload: None,
            })
            .unwrap(),
        ));
        let mut args = BTreeMap::new();
        args.insert("value".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert_eq!(
                    error.1,
                    Value::String("This tool call is rejected.".to_string())
                );
            }
            other => panic!("expected an error map, got {other:?}"),
        }
    }
}
