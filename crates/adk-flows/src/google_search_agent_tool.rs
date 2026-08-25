//! C0429: `tools.google_search_agent_tool`, ported from
//! `google.adk.tools.google_search_agent_tool`.
//!
//! A workaround wrapping a dedicated single-tool sub-agent so
//! `google_search` can coexist with other tools on the same agent —
//! Gemini restricts `google_search` to sole-tool use, so the source
//! (and this port) instead delegates search requests to a nested
//! sub-agent whose only tool is `google_search`, and wraps that
//! sub-agent as an [`AgentTool`] the parent agent can call alongside its
//! other tools. Explicitly marked temporary in the source ("TODO: Remove
//! once the workaround is no longer needed").
//!
//! **`GoogleSearchAgentTool`, collapses to `AgentTool` itself**: the
//! source's subclass adds nothing beyond calling
//! `AgentTool.__init__(agent=agent, propagate_grounding_metadata=True)`.
//! This port's [`AgentTool::new`] has no `propagate_grounding_metadata`
//! parameter at all — that flag's actual consumer, the
//! grounding-metadata-from-session-state workaround in
//! `_handle_after_model_callback`, is already disclosed as blocked on
//! C0092 in `agent_tool.rs`'s own module doc (needs `agent.canonical_tools()`
//! to know it's dealing with a `GoogleSearchAgentTool` specifically).
//! With nothing left for a subclass to add, this batch exposes
//! [`create_google_search_agent_tool`] directly, returning a plain
//! [`AgentTool`] — a distinct `GoogleSearchAgentTool` newtype would only
//! wrap it with no new behavior.
//!
//! **`LlmAgent` has no `name`/`description` fields**: those live on
//! [`BaseAgent`] in this port (`llm_agent.rs`'s own established split),
//! so [`create_google_search_agent`] passes them to [`BaseAgent::build`]
//! directly rather than as `LlmAgent` constructor arguments the way the
//! source's fused `LlmAgent(name=..., description=..., ...)` does.

use std::sync::Arc;

use adk_agents::base_agent::{BaseAgent, BaseAgentError};
use adk_agents::llm_agent::{Instruction, LlmAgent, ModelRef};
use adk_tools::agent_tool::AgentTool;
use adk_tools::google_search_tool::GoogleSearchTool;

use crate::llm_flow::{LlmFlow, LlmFlowError};

const GOOGLE_SEARCH_AGENT_NAME: &str = "google_search_agent";
const GOOGLE_SEARCH_AGENT_DESCRIPTION: &str =
    "An agent for performing Google search using the `google_search` tool";
const GOOGLE_SEARCH_AGENT_INSTRUCTION: &str = "You are a specialized Google search agent.\n\n\
     When given a search query, use the `google_search` tool to find the related information.";

/// Error type for [`create_google_search_agent`]/[`create_google_search_agent_tool`].
#[derive(Debug, rusty_err::Error)]
pub enum CreateGoogleSearchAgentError {
    #[error("{0}")]
    Flow(#[from] LlmFlowError),
    #[error("{0}")]
    Agent(#[from] BaseAgentError),
}

/// `google_search_agent_tool.create_google_search_agent` — builds the
/// dedicated sub-agent whose only tool is `google_search`. See the
/// module doc for why `name`/`description` are set via [`BaseAgent::build`]
/// rather than as `LlmAgent` fields.
pub fn create_google_search_agent(
    model: ModelRef,
) -> Result<BaseAgent, CreateGoogleSearchAgentError> {
    let mut llm_agent = LlmAgent::new(model);
    llm_agent.instruction = Instruction::Static(GOOGLE_SEARCH_AGENT_INSTRUCTION.to_string());

    let flow = LlmFlow::new(llm_agent)?.with_tools(vec![Arc::new(GoogleSearchTool::new())]);

    Ok(BaseAgent::build(
        GOOGLE_SEARCH_AGENT_NAME,
        GOOGLE_SEARCH_AGENT_DESCRIPTION,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        flow,
    )?)
}

/// `GoogleSearchAgentTool` — see the module doc for why this port
/// exposes it as a plain [`AgentTool`] rather than a distinct newtype.
pub fn create_google_search_agent_tool(
    model: ModelRef,
) -> Result<AgentTool, CreateGoogleSearchAgentError> {
    Ok(AgentTool::new(create_google_search_agent(model)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_sub_agent_named_google_search_agent() {
        let agent =
            create_google_search_agent(ModelRef::Name("gemini-2.0-flash".to_string())).unwrap();
        assert_eq!(agent.name(), GOOGLE_SEARCH_AGENT_NAME);
        assert_eq!(agent.description(), GOOGLE_SEARCH_AGENT_DESCRIPTION);
    }

    #[test]
    fn the_wrapped_agent_tool_carries_the_same_sub_agent() {
        let tool = create_google_search_agent_tool(ModelRef::Name("gemini-2.0-flash".to_string()))
            .unwrap();
        assert_eq!(
            adk_tools::base_tool::BaseTool::name(&tool),
            GOOGLE_SEARCH_AGENT_NAME
        );
    }
}
