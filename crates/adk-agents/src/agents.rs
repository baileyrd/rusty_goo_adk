//! Capability C0034: the `agents` package's public export surface,
//! ported from `google.adk.agents.__init__`.
//!
//! **Eager vs. lazy has no Rust equivalent, so it's dropped**: same
//! precedent `plugins.rs`'s own doc already established for
//! `google.adk.plugins.__init__` — the source's `_LAZY_MEMBERS`/
//! `__getattr__` split exists only to avoid importing heavy agent
//! modules at package-load time; a Rust `pub use` has no such cost to
//! defer, so every name below is a plain re-export regardless of which
//! list (`_LAZY_MEMBERS`/`__all__`) it was on in the source.
//!
//! **Not re-exported here, disclosed — genuinely unbuilt in this port**:
//! `ManagedAgent` (Agent Engine's Managed Agents API), `BaseAgentConfig`/
//! `LlmAgentConfig`/`LoopAgentConfig`/`ParallelAgentConfig`/
//! `SequentialAgentConfig` (the deprecated YAML-config-loading pipeline,
//! blocked on the same still-deferred YAML-parsing dependency as C0047),
//! and `McpInstructionProvider` (needs a real MCP client transport,
//! itself out of scope — see `adk-tools`'s own MCP module docs). This
//! module re-exports only what actually exists; revisit each once its
//! own blocker clears.

pub use crate::base_agent::BaseAgent;
pub use crate::context::Context;
pub use crate::invocation_context::InvocationContext;
pub use crate::live_request::{LiveRequest, LiveRequestQueue};
pub use crate::llm_agent::LlmAgent;
/// `agents.Agent` — the source's own alias for [`LlmAgent`].
pub use crate::llm_agent::LlmAgent as Agent;
pub use crate::loop_agent::LoopAgent;
pub use crate::parallel_agent::ParallelAgent;
pub use crate::run_config::RunConfig;
pub use crate::sequential_agent::SequentialAgent;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_is_an_alias_for_llm_agent() {
        fn assert_same_type<T>(_: &Option<T>, _: &Option<T>) {}
        let agent: Option<Agent> = None;
        let llm_agent: Option<LlmAgent> = None;
        assert_same_type(&agent, &llm_agent);
    }

    #[test]
    fn every_re_exported_agent_type_is_reachable_through_the_facade() {
        let _base_agent_type_check: Option<BaseAgent> = None;
        let _context_type_check: Option<Context> = None;
        let _invocation_context_type_check: Option<InvocationContext> = None;
        let _live_request_type_check: Option<LiveRequest> = None;
        let _live_request_queue_type_check: Option<LiveRequestQueue> = None;
        let _loop_agent_type_check: Option<LoopAgent> = None;
        let _parallel_agent_type_check: Option<ParallelAgent> = None;
        let _run_config_type_check: Option<RunConfig> = None;
        let _sequential_agent_type_check: Option<SequentialAgent> = None;
    }
}
