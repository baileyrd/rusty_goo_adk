//! C0490: `NodeTool`, ported from `google.adk.tools._node_tool`.
//!
//! Wraps a workflow [`BaseNode`] (a `Workflow`/loop/join/parallel-worker
//! node — anything but an `Agent`) as a callable [`BaseTool`], so an
//! `LlmAgent` can invoke it like any other function.
//!
//! **`isinstance(node, BaseAgent)` guard, no equivalent needed**:
//! `BaseNode` and `BaseAgent` are disjoint Rust types — a `BaseAgent`
//! value cannot be passed where [`NodeTool::new`] expects a `BaseNode` in
//! the first place, so the source's runtime rejection has nothing to
//! check here.
//!
//! **`FunctionNode`/`parameter_binding` rebinding, no equivalent needed**:
//! `workflow_function_node.rs`'s own module doc already discloses this
//! port's `FunctionNode` has no `parameter_binding` concept at all (its
//! body always receives `(ctx, node_input)` directly) — and the concrete
//! `FunctionNode` type isn't even `pub`, so a caller in this crate
//! couldn't downcast to detect one either way. Nothing to rebind.
//!
//! **`BaseModel`-schema branches, unreachable by construction**: the
//! source's `_get_declaration`/`run_async` each branch on whether
//! `input_schema` is a Pydantic class or a plain JSON-Schema dict.
//! [`BaseNode::input_schema`] is always an opaque [`Value`] in this port
//! (already-disclosed narrowing, `workflow_base_node.rs`'s own doc) —
//! never a class — so only the "dict schema" branch of each method is
//! ever reachable, and is the only one ported.
//!
//! **Object-schema wrapping and the `run_node` call, ported faithfully**:
//! [`NodeTool::get_declaration`] wraps a non-object input schema under
//! `{"type":"object","properties":{"request":<schema>},"required":["request"]}`
//! (the GenAI API requires an object-typed `parameters_json_schema`), and
//! [`NodeTool::run_async`] correspondingly extracts `args["request"]`
//! rather than passing `args` through directly whenever the schema needed
//! that wrapping — both real behaviors in the source, verified directly
//! against `_node_tool.py` rather than assumed.
//!
//! **`NodeInterruptedError`, not re-raised — this port's `Context::run_node`
//! already made that choice**: the source re-raises `NodeInterruptedError`
//! uncaught so it unwinds past `run_async` and pauses the invocation.
//! This port's [`adk_agents::context::Context::run_node`] instead returns
//! `Ok(RunNodeOutcome::Interrupted(_))` (a value, not an error) and
//! already propagates the interrupt ids onto the calling `Context`
//! in-place — `adk_agents::workflow_parallel_worker::ParallelWorker` is
//! the established precedent for this: on `Interrupted`, a caller just
//! returns without producing a normal result and lets the caller read the
//! interrupt off `ctx` afterward. `NodeTool::run_async` does the same,
//! returning `Ok(Value::Null)`.
//!
//! **Every other node-run failure stringified into a successful result,
//! ported faithfully (however surprising)**: the source's catch-all
//! `except Exception as e: return f'Error running node {name}: {e}'`
//! turns a failed node run into ordinary tool *output*, not a raised
//! tool error — verified against the real source rather than assumed to
//! be a mistake, and ported as `Ok(Value::String(...))`, not
//! `Err(ToolError::...)`.

use std::collections::BTreeMap;

use adk_agents::context::RunNodeOptions;
use adk_agents::workflow_base_node::{BaseNode, BaseNodeError};
use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

#[derive(Debug, rusty_err::Error)]
pub enum NodeToolError {
    #[error(
        "Node '{0}' does not have an input_schema defined. NodeTool requires an explicit \
         input_schema on the wrapped node."
    )]
    MissingInputSchema(String),
    #[error("{0}")]
    BaseNode(#[from] BaseNodeError),
}

/// Whether `schema` is a JSON-Schema object with `"type": "object"` —
/// the GenAI API's requirement `_get_declaration`/`run_async` both check
/// before deciding whether to wrap/unwrap under a `"request"` property.
fn is_object_schema(schema: &Value) -> bool {
    matches!(
        schema.get("type"),
        Some(Value::String(type_name)) if type_name == "object"
    )
}

fn wrap_as_request_schema(schema: Value) -> Value {
    let mut properties = Value::Map(Vec::new());
    properties.insert("request", schema);
    let mut wrapped = Value::Map(Vec::new());
    wrapped.insert("type", Value::String("object".to_string()));
    wrapped.insert("properties", properties);
    wrapped.insert(
        "required",
        Value::Seq(vec![Value::String("request".to_string())]),
    );
    wrapped
}

/// `_node_tool.NodeTool` — a tool wrapper that executes a [`BaseNode`].
pub struct NodeTool {
    node: BaseNode,
    name: String,
    description: String,
}

impl NodeTool {
    /// `NodeTool.__init__`. See the module doc for the two source checks
    /// (`isinstance(node, BaseAgent)`, `FunctionNode` rebinding) that have
    /// no equivalent here.
    pub fn new(
        node: BaseNode,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<Self, NodeToolError> {
        if node.input_schema().is_none() {
            return Err(NodeToolError::MissingInputSchema(node.name().to_string()));
        }
        let name = name.unwrap_or_else(|| node.name().to_string());
        let description = description
            .filter(|d| !d.is_empty())
            .or_else(|| (!node.description().is_empty()).then(|| node.description().to_string()))
            .unwrap_or_else(|| format!("Executes the node: {}", node.name()));
        Ok(Self {
            node,
            name,
            description,
        })
    }
}

impl BaseTool for NodeTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn is_long_running(&self) -> bool {
        true
    }

    /// `NodeTool._get_declaration`.
    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        let schema = self
            .node
            .input_schema()
            .cloned()
            .expect("NodeTool::new already checked input_schema is present");
        let schema = if is_object_schema(&schema) {
            schema
        } else {
            wrap_as_request_schema(schema)
        };
        Some(FunctionDeclaration {
            name: Some(self.name.clone()),
            description: Some(self.description.clone()),
            parameters_json_schema: Some(schema),
            response_json_schema: self.node.output_schema().cloned(),
            ..Default::default()
        })
    }

    /// `NodeTool.run_async`.
    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let input_schema = self
                .node
                .input_schema()
                .expect("NodeTool::new already checked input_schema is present");
            let node_input = if is_object_schema(input_schema) {
                let mut map = Value::Map(Vec::new());
                for (key, value) in args {
                    map.insert(key.clone(), value.clone());
                }
                map
            } else {
                args.get("request").cloned().unwrap_or(Value::Null)
            };

            let fc_id = tool_context.function_call_id().unwrap_or("None");
            let segment = format!("{}@{fc_id}", self.name);
            let tool_branch = match tool_context.branch() {
                Some(base) => format!("{base}.{segment}"),
                None => segment,
            };

            let outcome = tool_context
                .run_node(
                    self.node.clone(),
                    node_input,
                    RunNodeOptions {
                        override_branch: Some(tool_branch),
                        use_sub_branch: false,
                        raise_on_wait: true,
                        ..Default::default()
                    },
                )
                .await;

            match outcome {
                Ok(adk_agents::context::RunNodeOutcome::Completed(output)) => {
                    Ok(output.output.unwrap_or_else(|| {
                        let mut result = Value::Map(Vec::new());
                        result.insert("result", Value::Null);
                        result
                    }))
                }
                // The interrupt is already recorded on `tool_context` by
                // `run_node` itself — see the module doc.
                Ok(adk_agents::context::RunNodeOutcome::Interrupted(_)) => Ok(Value::Null),
                Err(error) => Ok(Value::String(format!(
                    "Error running node {}: {error}",
                    self.name
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::base_agent::BaseAgent;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_agents::workflow_base_node::{NodeBehavior, NodeRunError, NodeYield};
    use adk_events::RequestInput;
    use std::future::Future;
    use std::pin::Pin;

    fn ctx() -> Context {
        let agent = BaseAgent::new("agent", adk_agents::base_agent::NoopBehavior).unwrap();
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(agent)
            .build();
        Context::new(ic)
    }

    fn object_schema() -> Value {
        let mut properties = Value::Map(Vec::new());
        let mut x_schema = Value::Map(Vec::new());
        x_schema.insert("type", Value::String("integer".to_string()));
        properties.insert("x", x_schema);
        let mut schema = Value::Map(Vec::new());
        schema.insert("type", Value::String("object".to_string()));
        schema.insert("properties", properties);
        schema
    }

    fn string_schema() -> Value {
        let mut schema = Value::Map(Vec::new());
        schema.insert("type", Value::String("string".to_string()));
        schema
    }

    type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    struct EchoNode;
    impl NodeBehavior for EchoNode {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            node_input: Value,
        ) -> BoxFut<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move { Ok(vec![NodeYield::Data(node_input)]) })
        }
    }

    struct InterruptingNode;
    impl NodeBehavior for InterruptingNode {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFut<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move {
                Ok(vec![NodeYield::RequestInput(RequestInput::new(
                    Some("confirm?".to_string()),
                    None,
                    None,
                ))])
            })
        }
    }

    struct FailingNode;
    impl NodeBehavior for FailingNode {
        fn run_impl<'a>(
            &'a self,
            _ctx: &'a mut Context,
            _node_input: Value,
        ) -> BoxFut<'a, Result<Vec<NodeYield>, NodeRunError>> {
            Box::pin(async move { Err("boom".into()) })
        }
    }

    #[test]
    fn new_rejects_a_node_without_an_input_schema() {
        let node = BaseNode::new("n", EchoNode).unwrap();
        let Err(err) = NodeTool::new(node, None, None) else {
            panic!("expected a MissingInputSchema error");
        };
        assert!(matches!(err, NodeToolError::MissingInputSchema(name) if name == "n"));
    }

    #[test]
    fn new_defaults_name_and_description_from_the_node() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(object_schema()),
            None,
            None,
            EchoNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        assert_eq!(tool.name(), "n");
        assert_eq!(tool.description(), "Executes the node: n");
    }

    #[test]
    fn get_declaration_passes_an_object_schema_through_unwrapped() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(object_schema()),
            None,
            None,
            EchoNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let decl = tool.get_declaration().unwrap();
        assert_eq!(decl.parameters_json_schema, Some(object_schema()));
    }

    #[test]
    fn get_declaration_wraps_a_non_object_schema_under_request() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(string_schema()),
            None,
            None,
            EchoNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let decl = tool.get_declaration().unwrap();
        let schema = decl.parameters_json_schema.unwrap();
        assert_eq!(
            schema.get("type"),
            Some(&Value::String("object".to_string()))
        );
        assert_eq!(
            schema.get("properties").and_then(|p| p.get("request")),
            Some(&string_schema())
        );
        assert_eq!(
            schema.get("required"),
            Some(&Value::Seq(vec![Value::String("request".to_string())]))
        );
    }

    #[rusty_tokio::test]
    async fn run_async_passes_object_args_through_directly() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(object_schema()),
            None,
            None,
            EchoNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("x".to_string(), Value::Int(3));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result.get("x"), Some(&Value::Int(3)));
    }

    #[rusty_tokio::test]
    async fn run_async_unwraps_the_request_property_for_a_non_object_schema() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(string_schema()),
            None,
            None,
            EchoNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String("hi".to_string()));
    }

    #[rusty_tokio::test]
    async fn run_async_returns_null_and_records_the_interrupt_when_the_node_pauses() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(object_schema()),
            None,
            None,
            InterruptingNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::Null);
        assert!(!context.interrupt_ids().is_empty());
    }

    #[rusty_tokio::test]
    async fn run_async_stringifies_a_node_run_failure_into_a_successful_result() {
        let node = BaseNode::build(
            "n",
            "",
            true,
            false,
            None,
            None,
            Some(object_schema()),
            None,
            None,
            FailingNode,
        )
        .unwrap();
        let tool = NodeTool::new(node, None, None).unwrap();
        let mut context = ctx();
        let args = BTreeMap::new();
        let result = tool.run_async(&args, &mut context).await.unwrap();
        let Value::String(message) = result else {
            panic!("expected a string result, got {result:?}");
        };
        // `Context::run_node` wraps a dynamic node's own failure in
        // `WorkflowNodeError::DynamicNodeFail`, whose `Display` shows only
        // its own `message` field ("Dynamic node {name} failed"), not the
        // wrapped error's text verbatim — already-established,
        // already-disclosed behavior (`context.rs`), not a bug.
        assert!(message.starts_with("Error running node n:"));
        assert!(message.contains("Dynamic node n failed"));
    }
}
