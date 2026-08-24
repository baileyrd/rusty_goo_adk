//! Capability C0419: `ExampleTool`, ported from
//! `google.adk.tools.example_tool`.
//!
//! **Not** ported: `from_config` — needs `ToolArgsConfig`/YAML
//! tool-reference config (C0417), not built in this port yet, the same
//! blocker `base_tool.rs`'s own module doc already discloses for every
//! `from_config` classmethod.
//!
//! `name`/`description` are set but, per the source's own comment, never
//! actually used — this tool only mutates `llm_request`, never appears in
//! `llm_request.config.tools` (its `get_declaration` uses the trait
//! default of `None`, so the default `process_llm_request` add-declaration
//! behavior is fully overridden rather than invoked), and is never called
//! by the model (its `run_async` uses the trait default `NotImplemented`,
//! matching the source not overriding it either).

use adk_examples::base_example_provider::BaseExampleProvider;
use adk_examples::example::Example;
use adk_examples::example_util::{build_example_si, ExamplesSource};
use adk_genai::content::Content;
use adk_models::llm_request::{Instructions, LlmRequest};

use crate::base_tool::{BaseTool, BoxFuture};
use crate::tool_context::ToolContext;

/// `Union[list[Example], BaseExampleProvider]` for [`ExampleTool::new`].
pub enum ExampleSource {
    List(Vec<Example>),
    Provider(Box<dyn BaseExampleProvider + Send + Sync>),
}

/// C0419: a tool that adds (few-shot) examples to the LLM request.
pub struct ExampleTool {
    examples: ExampleSource,
}

impl ExampleTool {
    pub fn new(examples: ExampleSource) -> Self {
        Self { examples }
    }
}

impl BaseTool for ExampleTool {
    fn name(&self) -> &str {
        "example_tool"
    }

    fn description(&self) -> &str {
        "example tool"
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let Some(user_content_value) = tool_context.invocation_context().user_content.clone()
            else {
                return;
            };
            let Ok(user_content) = rusty_serde::json::from_value::<Content>(user_content_value)
            else {
                return;
            };
            let Some(text) = user_content
                .parts
                .first()
                .and_then(|part| part.text.as_deref())
            else {
                return;
            };

            let instruction = match &self.examples {
                ExampleSource::List(list) => build_example_si(
                    ExamplesSource::List(list),
                    text,
                    llm_request.model.as_deref(),
                ),
                ExampleSource::Provider(provider) => build_example_si(
                    ExamplesSource::Provider(provider.as_ref()),
                    text,
                    llm_request.model.as_deref(),
                ),
            };
            llm_request.append_instructions(Instructions::Strings(vec![instruction]));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_genai::content::Part;

    fn example() -> Example {
        Example::new(
            Content::new("user", vec![Part::text("hi")]),
            vec![Content::new("model", vec![Part::text("hello")])],
        )
    }

    #[rusty_tokio::test]
    async fn injects_an_instruction_built_from_the_user_content_text() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.user_content = Some(
            rusty_serde::json::to_value(&Content::new("user", vec![Part::text("test query")]))
                .unwrap(),
        );
        let mut ctx = Context::new(ic);
        let mut request = LlmRequest::new("gemini-2.5-flash");

        let tool = ExampleTool::new(ExampleSource::List(vec![example()]));
        tool.process_llm_request(&mut ctx, &mut request).await;

        let system_instruction = request.config.system_instruction.unwrap();
        assert!(system_instruction.contains("<EXAMPLES>"));
        assert!(system_instruction.contains("hi"));
        assert!(system_instruction.contains("hello"));
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_without_user_content() {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let mut ctx = Context::new(ic);
        let mut request = LlmRequest::new("gemini-2.5-flash");

        let tool = ExampleTool::new(ExampleSource::List(vec![example()]));
        tool.process_llm_request(&mut ctx, &mut request).await;

        assert!(request.config.system_instruction.is_none());
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_when_the_first_part_has_no_text() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.user_content = Some(
            rusty_serde::json::to_value(&Content::new("user", vec![Part::default()])).unwrap(),
        );
        let mut ctx = Context::new(ic);
        let mut request = LlmRequest::new("gemini-2.5-flash");

        let tool = ExampleTool::new(ExampleSource::List(vec![example()]));
        tool.process_llm_request(&mut ctx, &mut request).await;

        assert!(request.config.system_instruction.is_none());
    }

    struct StaticProvider(Vec<Example>);

    impl BaseExampleProvider for StaticProvider {
        fn get_examples(&self, _query: &str) -> Vec<Example> {
            self.0.clone()
        }
    }

    #[rusty_tokio::test]
    async fn injects_an_instruction_built_from_a_provider() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.user_content = Some(
            rusty_serde::json::to_value(&Content::new("user", vec![Part::text("test query")]))
                .unwrap(),
        );
        let mut ctx = Context::new(ic);
        let mut request = LlmRequest::new("gemini-2.5-flash");

        let tool = ExampleTool::new(ExampleSource::Provider(Box::new(StaticProvider(vec![
            example(),
        ]))));
        tool.process_llm_request(&mut ctx, &mut request).await;

        let system_instruction = request.config.system_instruction.unwrap();
        assert!(system_instruction.contains("<EXAMPLES>"));
    }

    #[test]
    fn name_and_description_are_not_used_but_are_set() {
        let tool = ExampleTool::new(ExampleSource::List(vec![]));
        assert_eq!(tool.name(), "example_tool");
        assert_eq!(tool.description(), "example tool");
        assert!(tool.get_declaration().is_none());
    }
}
