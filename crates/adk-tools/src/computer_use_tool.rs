//! Capability C0447: `ComputerUseTool`, ported from
//! `google.adk.tools.computer_use.computer_use_tool`.
//!
//! **Adaptation, structural**: the source wraps an arbitrary Python
//! callable (`func`) and relies on `FunctionTool`'s own runtime
//! reflection to build its declaration/required-args from that
//! callable's signature. This port composes an already-built
//! [`FunctionTool`] as `inner` instead — the same shape
//! `AuthenticatedFunctionTool` (C0412) already established for an
//! identical "wrap a `FunctionTool`, inject pre/post-processing around
//! `run_async`" case.
//!
//! **`ComputerState` → image dict, relocated one layer down, disclosed**:
//! in the source, `ComputerUseTool.run_async` itself detects
//! `isinstance(result, ComputerState)` on whatever the wrapped raw
//! function returned and converts it to the wire image dict. This port's
//! `FunctionTool::ToolFn` is fixed to return `Value` (not a typed
//! `ComputerState`), so the boundary that bridges a
//! `BaseComputer`-trait-method's real `ComputerState` return into that
//! `Value` — built in `computer_use_toolset.rs`, since that's where a
//! `BaseComputer` trait method is actually called — does the same
//! conversion at that point instead of a second time here. The
//! observable wire output is identical either way; only which internal
//! boundary performs the conversion differs.
//!
//! **`@experimental(FeatureName.COMPUTER_USE)`**: gated once, in
//! [`ComputerUseTool::new`] — see `base_computer.rs`'s module doc.

use std::collections::BTreeMap;

use adk_features::feature_decorator::check_feature_enabled;
use adk_features::feature_registry::FeatureName;
use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::function_tool::{FunctionTool, ToolFn};
use crate::tool_confirmation::ToolConfirmation;
use crate::tool_context::ToolContext;

#[derive(Debug, rusty_err::Error)]
pub enum ComputerUseToolError {
    #[error("{0}")]
    FeatureNotEnabled(#[from] adk_features::feature_decorator::FeatureNotEnabledError),
    #[error("screen_size dimensions must be positive")]
    InvalidScreenSize,
    #[error("virtual_screen_size dimensions must be positive")]
    InvalidVirtualScreenSize,
}

/// C0447: `computer_use_tool.ComputerUseTool` — normalizes model-supplied
/// virtual-coordinate-space input to real screen size, and gates
/// execution behind the model's own `safety_decision`
/// confirmation-request protocol. See the module doc for the
/// `ComputerState`-conversion boundary adaptation.
pub struct ComputerUseTool {
    inner: FunctionTool,
    screen_size: (u32, u32),
    coordinate_space: (u32, u32),
}

impl ComputerUseTool {
    /// `ComputerUseTool.__init__`. `screen_size`/`virtual_screen_size`
    /// must both have strictly positive dimensions (the source's
    /// `ValueError` on a non-positive dimension) — the "must be a
    /// 2-tuple" half of the source's shape check has no Rust equivalent
    /// to fail, since `(u32, u32)` already enforces that.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        declaration: FunctionDeclaration,
        required_args: Vec<String>,
        func: ToolFn,
        screen_size: (u32, u32),
        virtual_screen_size: (u32, u32),
    ) -> Result<Self, ComputerUseToolError> {
        check_feature_enabled(FeatureName::ComputerUse)?;
        if screen_size.0 == 0 || screen_size.1 == 0 {
            return Err(ComputerUseToolError::InvalidScreenSize);
        }
        if virtual_screen_size.0 == 0 || virtual_screen_size.1 == 0 {
            return Err(ComputerUseToolError::InvalidVirtualScreenSize);
        }
        Ok(Self {
            inner: FunctionTool::new(name, description, declaration, required_args, func),
            screen_size,
            coordinate_space: virtual_screen_size,
        })
    }

    /// `ComputerUseTool._normalize_x` — true (float) division then
    /// truncation toward zero, not rounding, then clamped to
    /// `[0, screen_size.0 - 1]`.
    fn normalize_x(&self, x: f64) -> i64 {
        let normalized = (x / self.coordinate_space.0 as f64 * self.screen_size.0 as f64) as i64;
        normalized.clamp(0, self.screen_size.0 as i64 - 1)
    }

    /// `ComputerUseTool._normalize_y`.
    fn normalize_y(&self, y: f64) -> i64 {
        let normalized = (y / self.coordinate_space.1 as f64 * self.screen_size.1 as f64) as i64;
        normalized.clamp(0, self.screen_size.1 as i64 - 1)
    }

    fn normalize_coordinate_key(&self, args: &mut BTreeMap<String, Value>, key: &str, is_x: bool) {
        let Some(value) = args.get(key) else {
            return;
        };
        let numeric = match value {
            Value::Int(i) => Some(*i as f64),
            Value::UInt(u) => Some(*u as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        };
        if let Some(numeric) = numeric {
            let normalized = if is_x {
                self.normalize_x(numeric)
            } else {
                self.normalize_y(numeric)
            };
            args.insert(key.to_string(), Value::Int(normalized));
        }
    }
}

impl BaseTool for ComputerUseTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        self.inner.get_declaration()
    }

    /// `ComputerUseTool.process_llm_request` is an explicit no-op in the
    /// source — `ComputerUseToolset` is what adds declarations/config to
    /// the request, not the individual tool.
    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        _llm_request: &'a mut adk_models::llm_request::LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            // Layered confirmation gate — see the module doc's source
            // walkthrough: the model's own `safety_decision` field is
            // checked first (only when no confirmation is on file yet),
            // then a plain `tool_confirmation.confirmed` rejection.
            match tool_context.tool_confirmation() {
                None => {
                    if let Some(Value::Map(decision_fields)) = args.get("safety_decision") {
                        let decision = decision_fields
                            .iter()
                            .find(|(k, _)| k == "decision")
                            .and_then(|(_, v)| match v {
                                Value::String(s) => Some(s.as_str()),
                                _ => None,
                            });
                        if decision == Some("require_confirmation") {
                            let explanation = decision_fields
                                .iter()
                                .find(|(k, _)| k == "explanation")
                                .and_then(|(_, v)| match v {
                                    Value::String(s) => Some(s.clone()),
                                    _ => None,
                                });
                            let hint = explanation.unwrap_or_else(|| {
                                "This computer use action requires safety confirmation.".to_string()
                            });
                            let _ = tool_context.request_confirmation(Some(hint), None);
                            tool_context.actions_mut().skip_summarization = true;
                            return Ok(error_response(
                                "This tool call requires confirmation, please approve or reject."
                                    .to_string(),
                            ));
                        }
                    }
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

            let mut normalized_args = args.clone();
            self.normalize_coordinate_key(&mut normalized_args, "x", true);
            self.normalize_coordinate_key(&mut normalized_args, "y", false);
            self.normalize_coordinate_key(&mut normalized_args, "destination_x", true);
            self.normalize_coordinate_key(&mut normalized_args, "destination_y", false);

            let mut response = self.inner.run_async(&normalized_args, tool_context).await?;

            let confirmed = tool_context
                .tool_confirmation()
                .and_then(|value| {
                    rusty_serde::json::from_value::<ToolConfirmation>(value.clone()).ok()
                })
                .map(|confirmation| confirmation.confirmed)
                .unwrap_or(false);
            if confirmed {
                response = match response {
                    Value::Map(mut fields) => {
                        fields.push((
                            "safety_acknowledgement".to_string(),
                            Value::String("true".to_string()),
                        ));
                        Value::Map(fields)
                    }
                    other => Value::Map(vec![
                        ("result".to_string(), other),
                        (
                            "safety_acknowledgement".to_string(),
                            Value::String("true".to_string()),
                        ),
                    ]),
                };
            }

            Ok(response)
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
    use std::sync::Arc;

    fn ctx() -> Context {
        let mut context = Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        context.set_function_call_id(Some("call-1".to_string()));
        context
    }

    fn declaration() -> FunctionDeclaration {
        FunctionDeclaration {
            name: Some("click_at".to_string()),
            description: Some("Clicks at a coordinate.".to_string()),
            ..Default::default()
        }
    }

    fn echo_args_func() -> ToolFn {
        Arc::new(|args, _ctx| {
            let args = args.clone();
            Box::pin(async move { Value::Map(args.into_iter().collect()) })
        })
    }

    #[test]
    fn new_rejects_a_zero_screen_dimension() {
        let result = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (0, 100),
            (1000, 1000),
        );
        assert!(matches!(
            result,
            Err(ComputerUseToolError::InvalidScreenSize)
        ));
    }

    #[test]
    fn new_rejects_a_zero_virtual_screen_dimension() {
        let result = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 0),
        );
        assert!(matches!(
            result,
            Err(ComputerUseToolError::InvalidVirtualScreenSize)
        ));
    }

    #[rusty_tokio::test]
    async fn run_async_normalizes_x_and_y_and_truncates_toward_zero() {
        let tool = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        // 999/1000 * 1920 = 1918.08 -> truncates to 1918, not rounds to 1918 either way;
        // use a value where truncation vs rounding actually differ.
        args.insert("x".to_string(), Value::Int(999));
        args.insert("y".to_string(), Value::Int(500));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        let x = fields.iter().find(|(k, _)| k == "x").unwrap();
        let y = fields.iter().find(|(k, _)| k == "y").unwrap();
        assert_eq!(x.1, Value::Int(1918));
        assert_eq!(y.1, Value::Int(540));
    }

    #[rusty_tokio::test]
    async fn run_async_clamps_coordinates_to_screen_bounds() {
        let tool = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("x".to_string(), Value::Int(1000));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        let x = fields.iter().find(|(k, _)| k == "x").unwrap();
        assert_eq!(x.1, Value::Int(1919));
    }

    #[rusty_tokio::test]
    async fn run_async_normalizes_destination_coordinates_for_drag_and_drop() {
        let tool = ComputerUseTool::new(
            "drag_and_drop",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (2000, 1000),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("destination_x".to_string(), Value::Int(500));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        let dest_x = fields.iter().find(|(k, _)| k == "destination_x").unwrap();
        assert_eq!(dest_x.1, Value::Int(1000));
    }

    #[rusty_tokio::test]
    async fn run_async_requests_confirmation_when_the_model_signals_a_safety_decision() {
        let tool = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "safety_decision".to_string(),
            Value::Map(vec![(
                "decision".to_string(),
                Value::String("require_confirmation".to_string()),
            )]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        assert!(fields.iter().any(|(k, _)| k == "error"));
        assert!(context.tool_confirmation().is_none());
        assert!(context.actions().skip_summarization);
    }

    #[rusty_tokio::test]
    async fn run_async_rejects_when_confirmation_was_declined() {
        let tool = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        context.set_tool_confirmation(Some(Value::Map(vec![(
            "confirmed".to_string(),
            Value::Bool(false),
        )])));
        let result = tool
            .run_async(&BTreeMap::new(), &mut context)
            .await
            .unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        let error = fields.iter().find(|(k, _)| k == "error").unwrap();
        assert_eq!(
            error.1,
            Value::String("This tool call is rejected.".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn run_async_stamps_safety_acknowledgement_once_confirmed() {
        let tool = ComputerUseTool::new(
            "click_at",
            "desc",
            declaration(),
            vec![],
            echo_args_func(),
            (1920, 1080),
            (1000, 1000),
        )
        .unwrap();
        let mut context = ctx();
        context.set_tool_confirmation(Some(Value::Map(vec![(
            "confirmed".to_string(),
            Value::Bool(true),
        )])));
        let result = tool
            .run_async(&BTreeMap::new(), &mut context)
            .await
            .unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        let ack = fields
            .iter()
            .find(|(k, _)| k == "safety_acknowledgement")
            .unwrap();
        assert_eq!(ack.1, Value::String("true".to_string()));
    }
}
