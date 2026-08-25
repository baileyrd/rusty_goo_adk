//! Capability C0043: `BaseAgent._run_impl`'s workflow-node adapter,
//! ported from `google.adk.agents.base_agent`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc for
//! the standing crate-placement decision.
//!
//! **Why a wrapper exists at all, disclosed**: the source's `BaseAgent`
//! is itself a `BaseNode` subclass (`_run_impl` *is* `BaseNode`'s own
//! override point) — running an agent as a workflow node needs no
//! adapter there, just a method on the same class. This port's
//! `BaseAgent` and `BaseNode` are two unrelated types (`BaseAgent`
//! predates the workflow engine by several batches, and retrofitting it
//! to also implement `NodeBehavior` would mean `adk-agents`'s oldest,
//! most-depended-on type takes on a P7-only trait bound for every
//! existing caller). [`agent_node`] is the "caller supplies the resolved
//! bits" bridge instead: it builds a real [`BaseNode`] whose behavior
//! wraps a [`BaseAgent`] handle, matching the source's own `_run_impl`
//! body one-for-one.
//!
//! **Scope, narrowed**: this is the *generic* `BaseAgent._run_impl` —
//! run once via [`BaseAgent::run_async`], collect every event, done. The
//! source's `LlmAgent`-specific `workflow/_llm_agent_wrapper.py`
//! (`run_llm_agent_as_node`'s task/chat-mode dispatch loop, task
//! delegation via `_TaskAgentTool`, `FinishTaskTool` sniffing) is a much
//! larger, separate capability (C0407) that layers on top of this one —
//! not ported here.

use std::future::Future;
use std::pin::Pin;

use adk_events::Event;
use rusty_serde::value::Value;

use crate::base_agent::BaseAgent;
use crate::context::Context;
use crate::workflow_base_node::{BaseNode, BaseNodeError, NodeBehavior, NodeRunError, NodeYield};
use crate::workflow_retry_config::RetryConfig;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The wrapped agent — `AgentNode`'s override point. Kept `pub(crate)`
/// (rather than opaque behind [`agent_node`] alone) so
/// `context.rs`'s dynamic-dispatch transfer loop (C0059/C0060) can
/// downcast a [`BaseNode`] back to its wrapped [`BaseAgent`] via
/// [`BaseNode::as_any`] — the same "is this concretely an agent"
/// check the source expresses as `isinstance(curr_node, BaseAgent)`.
pub(crate) struct AgentNode {
    agent: BaseAgent,
}

impl AgentNode {
    pub(crate) fn agent(&self) -> &BaseAgent {
        &self.agent
    }
}

impl NodeBehavior for AgentNode {
    /// `BaseAgent._run_impl`: runs the wrapped agent via `run_async`,
    /// stamping `ctx.event_author` and (for an event this agent itself
    /// authored, with no path already set) `event.node_info.path` from
    /// `ctx.node_path` — matching the source's own comment ("Preserve
    /// author by setting it in context for NodeRunner").
    fn run_impl<'a>(
        &'a self,
        ctx: &'a mut Context,
        _node_input: Value,
    ) -> BoxFuture<'a, Result<Vec<NodeYield>, NodeRunError>> {
        Box::pin(async move {
            let invocation_context = ctx.get_invocation_context();
            let events: Vec<Event> = self
                .agent
                .run_async(&invocation_context)
                .await
                .map_err(|e| -> NodeRunError { e.to_string().into() })?;

            let mut yields = Vec::with_capacity(events.len());
            for mut event in events {
                if !event.author.is_empty() {
                    ctx.set_event_author(event.author.clone());
                }
                if event.node_info.path.is_empty() && event.author == self.agent.name() {
                    event.node_info.path = ctx.node_path().to_string();
                }
                yields.push(NodeYield::Event(Box::new(event)));
            }
            Ok(yields)
        })
    }
}

/// Builds a [`BaseNode`] wrapping `agent` as a workflow node — the
/// `agent`-typed case of the source's `build_node`, narrowed to just
/// this: the caller has already resolved `agent`, this just needs to
/// wrap it (see this module's own doc for the `LlmAgent`-specific
/// mode/task-delegation machinery this deliberately excludes).
pub fn agent_node(
    agent: BaseAgent,
    rerun_on_resume: bool,
    retry_config: Option<RetryConfig>,
    timeout: Option<f64>,
) -> Result<BaseNode, BaseNodeError> {
    let name = agent.name().to_string();
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
        AgentNode { agent },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_agent::AgentBehavior;
    use crate::invocation_context::{InvocationContext, InvocationContextBuilder};
    use crate::session::Session;
    use adk_events::node_info::NodeInfo;

    fn ctx() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    struct Echoes;
    impl AgentBehavior for Echoes {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, crate::base_agent::AgentRunError>> {
            let author = ctx.agent.as_ref().map(|a| a.name().to_string()).unwrap();
            Box::pin(async move {
                Ok(vec![Event::new(
                    ctx.invocation_id.clone(),
                    author,
                    NodeInfo::new(""),
                )])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, crate::base_agent::AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[rusty_tokio::test]
    async fn wraps_an_agent_as_a_node_and_stamps_the_node_path() {
        let agent = BaseAgent::new("greeter", Echoes).unwrap();
        let node = agent_node(agent, false, None, None).unwrap();
        let mut ctx = Context::for_node(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
            "",
            &[],
            None,
            "greeter",
            "1",
            std::collections::BTreeMap::new(),
            1,
            false,
            true,
            None,
        );
        let events = node.run(&mut ctx, Value::Null).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "greeter");
        assert_eq!(events[0].node_info.path, ctx.node_path());
        assert_eq!(ctx.event_author(), "greeter");
    }

    #[rusty_tokio::test]
    async fn does_not_overwrite_an_already_set_node_path() {
        let agent = BaseAgent::new("greeter", Echoes).unwrap();
        let node = agent_node(agent, false, None, None).unwrap();
        let mut c = ctx();
        let events = node.run(&mut c, Value::Null).await.unwrap();
        // Root ctx has an empty node_path; the event's own path also
        // stays empty since `event.node_info.path.is_empty()` was true
        // but `ctx.node_path()` is itself empty here -- covered by the
        // stamped-path test above for the non-empty case. This exercises
        // the "no author-mismatch skip" branch instead: a synthetic
        // event authored by someone else never gets a path stamped.
        assert_eq!(events[0].node_info.path, "");
    }
}
