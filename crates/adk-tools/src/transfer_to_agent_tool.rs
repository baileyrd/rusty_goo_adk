//! Capability C0436: `transfer_to_agent`/`TransferToAgentTool`, ported from
//! `google.adk.tools.transfer_to_agent_tool`.
//!
//! This unblocks the `TransferToAgentTool`-building half of C0171
//! (`agent_transfer.rs`'s request processor) that its own module doc
//! already disclosed as deferred until `BaseTool` existed — building and
//! attaching a real tool instance into `LlmRequest.config.tools` is
//! wiring left for a follow-up batch, since it needs the not-yet-built
//! "resolve `InvocationContext.agent` to a concrete `LlmAgent`" piece
//! every other Phase 4 processor is blocked on too.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::function_tool::FunctionTool;
use crate::tool_context::ToolContext;

/// Transfers the query to another agent — sets
/// `tool_context.actions.transfer_to_agent`. For most use cases, prefer
/// [`TransferToAgentTool`] (enum-constrained agent names, preventing a
/// hallucinated target) over calling this directly.
pub fn transfer_to_agent(args: &BTreeMap<String, Value>, tool_context: &mut ToolContext) -> Value {
    if let Some(Value::String(agent_name)) = args.get("agent_name") {
        tool_context.actions_mut().transfer_to_agent = Some(agent_name.clone());
    }
    Value::Null
}

fn declaration(agent_names: &[String]) -> FunctionDeclaration {
    FunctionDeclaration {
        name: Some("transfer_to_agent".to_string()),
        description: Some(
            "Transfer the query to another agent that is more suitable to answer the user's \
             query according to the agent's description."
                .to_string(),
        ),
        parameters: Some(Value::Map(vec![
            ("type".to_string(), Value::String("object".to_string())),
            (
                "properties".to_string(),
                Value::Map(vec![(
                    "agent_name".to_string(),
                    Value::Map(vec![
                        ("type".to_string(), Value::String("string".to_string())),
                        (
                            "enum".to_string(),
                            Value::Seq(agent_names.iter().cloned().map(Value::String).collect()),
                        ),
                    ]),
                )]),
            ),
            (
                "required".to_string(),
                Value::Seq(vec![Value::String("agent_name".to_string())]),
            ),
        ])),
        ..Default::default()
    }
}

/// C0436: a specialized [`FunctionTool`] for agent transfer, adding a JSON
/// Schema `enum` constraint to the `agent_name` parameter — restricting
/// choices to only valid agents, preventing the model from hallucinating
/// an invalid target.
pub struct TransferToAgentTool {
    inner: FunctionTool,
}

impl TransferToAgentTool {
    pub fn new(agent_names: Vec<String>) -> Self {
        Self {
            inner: FunctionTool::new(
                "transfer_to_agent",
                "Transfer the query to another agent.",
                declaration(&agent_names),
                vec!["agent_name".to_string()],
                Arc::new(|args, tool_context| {
                    let value = transfer_to_agent(args, tool_context);
                    Box::pin(async move { value })
                }),
            ),
        }
    }
}

impl BaseTool for TransferToAgentTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        self.inner.get_declaration()
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        self.inner.run_async(args, tool_context)
    }

    fn check_require_confirmation<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        self.inner.check_require_confirmation(args, tool_context)
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

    fn agent_name_enum(declaration: &FunctionDeclaration) -> Vec<String> {
        let Some(Value::Map(schema)) = &declaration.parameters else {
            panic!("expected a schema map");
        };
        let (_, Value::Map(properties)) = schema.iter().find(|(k, _)| k == "properties").unwrap()
        else {
            panic!("expected a properties map");
        };
        let (_, Value::Map(agent_name_schema)) =
            properties.iter().find(|(k, _)| k == "agent_name").unwrap()
        else {
            panic!("expected an agent_name schema map");
        };
        let (_, Value::Seq(values)) = agent_name_schema.iter().find(|(k, _)| k == "enum").unwrap()
        else {
            panic!("expected an enum seq");
        };
        values
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => panic!("expected a string, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn transfer_to_agent_sets_the_action() {
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "agent_name".to_string(),
            Value::String("billing_agent".to_string()),
        );
        let result = transfer_to_agent(&args, &mut context);
        assert_eq!(result, Value::Null);
        assert_eq!(
            context.actions().transfer_to_agent.as_deref(),
            Some("billing_agent")
        );
    }

    #[test]
    fn transfer_to_agent_is_a_no_op_without_an_agent_name() {
        let mut context = ctx();
        transfer_to_agent(&BTreeMap::new(), &mut context);
        assert_eq!(context.actions().transfer_to_agent, None);
    }

    #[test]
    fn declaration_constrains_agent_name_to_the_given_enum() {
        let tool = TransferToAgentTool::new(vec!["agent_a".to_string(), "agent_b".to_string()]);
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(
            agent_name_enum(&declaration),
            vec!["agent_a".to_string(), "agent_b".to_string()]
        );
    }

    #[rusty_tokio::test]
    async fn run_async_sets_the_transfer_action() {
        let tool = TransferToAgentTool::new(vec!["agent_a".to_string()]);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "agent_name".to_string(),
            Value::String("agent_a".to_string()),
        );

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Null);
        assert_eq!(
            context.actions().transfer_to_agent.as_deref(),
            Some("agent_a")
        );
    }
}
