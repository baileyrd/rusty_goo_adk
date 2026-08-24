//! Capability C0437: `SetModelResponseTool`, ported from
//! `google.adk.tools.set_model_response_tool`.
//!
//! **Fundamental adaptation, disclosed**: the source dynamically builds a
//! Python function signature from `output_schema` at runtime
//! (`inspect.Parameter` per Pydantic model field, or a single `items`/
//! `response` parameter for `list[BaseModel]`/raw-schema shapes), then
//! validates the model's call args against that same schema via Pydantic
//! (`model_validate`/`TypeAdapter`, catching `ValidationError` for a
//! retry-oriented error message). This port has neither Python's runtime
//! type introspection nor a Pydantic-equivalent JSON-schema validator
//! (the same "no compile-time reflection" limitation `function_tool.rs`'s
//! own module doc already discloses), so:
//! - `output_schema` is taken as an already-opaque JSON-schema `Value`
//!   (matching `LlmRequest.config.response_schema`'s own opaque-`Value`
//!   treatment, C0118) and used directly as the declaration's
//!   `parameters` — there is no dynamic per-field signature synthesis or
//!   `Field(description=...)` re-application (`_merge_json_schema_descriptions`/
//!   `_apply_descriptions_to_schema_properties`).
//! - `run_async` has no schema to validate the call's `args` against, so
//!   it can't distinguish "regular object schema" from "`list[BaseModel]`"
//!   from "raw non-object schema" the way the source's
//!   `_is_basemodel`/`_is_list_of_basemodel` flags do. Instead it uses the
//!   same two single-key conventions the source's own dynamic signature
//!   would produce for the non-object cases: an `items` key unwraps to
//!   its value (the `list[BaseModel]` shape), a `response` key unwraps to
//!   its value (the raw-schema shape), and otherwise every arg becomes a
//!   field of the result object (the regular-object shape) — a reasonable
//!   but *not* type-verified stand-in.
//! - The `ValidationError`-triggered retry-with-feedback path
//!   (`{"error": "Validation Error found:\n..."}`) is not ported — there
//!   is no validation to fail. A caller relying on that retry loop won't
//!   get it; this is a real, disclosed capability gap, not a silent one.

use std::collections::BTreeMap;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

const DESCRIPTION: &str = "Set your final response using the required output schema. Use this tool to provide your final structured answer instead of outputting text directly.";

/// C0437: internal tool used for the output-schema workaround — lets the
/// model set its final structured response via a tool call when
/// `output_schema` is configured alongside other tools.
pub struct SetModelResponseTool {
    output_schema: Value,
}

impl SetModelResponseTool {
    pub fn new(output_schema: Value) -> Self {
        Self { output_schema }
    }
}

fn args_to_response(args: &BTreeMap<String, Value>) -> Value {
    if args.len() == 1 {
        if let Some(items) = args.get("items") {
            return items.clone();
        }
        if let Some(response) = args.get("response") {
            return response.clone();
        }
    }
    Value::Map(
        args.iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

impl BaseTool for SetModelResponseTool {
    fn name(&self) -> &str {
        "set_model_response"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters: Some(self.output_schema.clone()),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let result = args_to_response(args);
            tool_context.actions_mut().set_model_response = Some(result.clone());
            Ok(result)
        })
    }
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

    fn object_schema() -> Value {
        Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![(
                    "answer".to_string(),
                    Value::Map(vec![(
                        "type".to_string(),
                        Value::String("string".to_string()),
                    )]),
                )]),
            ),
        ])
    }

    #[test]
    fn get_declaration_uses_the_output_schema_as_parameters() {
        let tool = SetModelResponseTool::new(object_schema());
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(declaration.name.as_deref(), Some("set_model_response"));
        assert_eq!(declaration.parameters, Some(object_schema()));
    }

    #[rusty_tokio::test]
    async fn run_async_sets_the_action_and_returns_the_object_fields() {
        let tool = SetModelResponseTool::new(object_schema());
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("answer".to_string(), Value::String("42".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::Map(vec![(
                "answer".to_string(),
                Value::String("42".to_string())
            )])
        );
        assert_eq!(context.actions().set_model_response, Some(result));
    }

    #[rusty_tokio::test]
    async fn run_async_unwraps_a_sole_items_argument() {
        let tool = SetModelResponseTool::new(Value::Null);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "items".to_string(),
            Value::Seq(vec![Value::String("a".to_string())]),
        );

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Seq(vec![Value::String("a".to_string())]));
    }

    #[rusty_tokio::test]
    async fn run_async_unwraps_a_sole_response_argument() {
        let tool = SetModelResponseTool::new(Value::Null);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("response".to_string(), Value::Int(7));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Int(7));
    }

    #[rusty_tokio::test]
    async fn run_async_treats_multiple_args_as_object_fields_even_if_one_is_named_items() {
        let tool = SetModelResponseTool::new(Value::Null);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("items".to_string(), Value::Seq(vec![]));
        args.insert("other".to_string(), Value::Bool(true));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => assert_eq!(fields.len(), 2),
            other => panic!("expected a map, got {other:?}"),
        }
    }
}
