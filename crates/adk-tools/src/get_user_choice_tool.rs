//! Capability C0421: `get_user_choice`/`get_user_choice_tool`, ported from
//! `google.adk.tools.get_user_choice_tool`.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::function_tool::FunctionTool;
use crate::long_running_tool::LongRunningFunctionTool;
use crate::tool_context::ToolContext;

/// C0421: provides the options to the user and asks them to choose one.
/// Always defers to client-side resolution — sets `skip_summarization`
/// and returns `None`, matching the source exactly.
pub fn get_user_choice(_args: &BTreeMap<String, Value>, tool_context: &mut ToolContext) -> Value {
    tool_context.actions_mut().skip_summarization = true;
    Value::Null
}

fn options_schema() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("object".to_string())),
        (
            "properties".to_string(),
            Value::Map(vec![(
                "options".to_string(),
                Value::Map(vec![
                    ("type".to_string(), Value::String("array".to_string())),
                    (
                        "items".to_string(),
                        Value::Map(vec![(
                            "type".to_string(),
                            Value::String("string".to_string()),
                        )]),
                    ),
                ]),
            )]),
        ),
        (
            "required".to_string(),
            Value::Seq(vec![Value::String("options".to_string())]),
        ),
    ])
}

/// C0421: `get_user_choice_tool` — a [`LongRunningFunctionTool`] wrapping
/// [`get_user_choice`].
pub fn get_user_choice_tool() -> LongRunningFunctionTool {
    LongRunningFunctionTool::new(FunctionTool::new(
        "get_user_choice",
        "Provides the options to the user and asks them to choose one.",
        FunctionDeclaration {
            name: Some("get_user_choice".to_string()),
            description: Some(
                "Provides the options to the user and asks them to choose one.".to_string(),
            ),
            parameters: Some(options_schema()),
            ..Default::default()
        },
        vec!["options".to_string()],
        Arc::new(|args, ctx| {
            let value = get_user_choice(args, ctx);
            Box::pin(async move { value })
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_tool::BaseTool;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[test]
    fn sets_skip_summarization_and_returns_null() {
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "options".to_string(),
            Value::Seq(vec![Value::String("a".to_string())]),
        );
        let result = get_user_choice(&args, &mut context);
        assert_eq!(result, Value::Null);
        assert!(context.actions().skip_summarization);
    }

    #[test]
    fn get_user_choice_tool_is_long_running() {
        assert!(get_user_choice_tool().is_long_running());
    }

    #[rusty_tokio::test]
    async fn get_user_choice_tool_run_async_defers_and_skips_summarization() {
        let tool = get_user_choice_tool();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "options".to_string(),
            Value::Seq(vec![
                Value::String("a".to_string()),
                Value::String("b".to_string()),
            ]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Null);
        assert!(context.actions().skip_summarization);
    }
}
