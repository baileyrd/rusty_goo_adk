//! Capability C0335 (partial): `SequentialAgent`, ported from
//! `google.adk.agents.sequential_agent`.
//!
//! **Adaptation, cross-sub-agent state visibility**: the source's
//! `Context.state`'s backing dict IS `EventActions.state_delta` by
//! reference (documented in `context.rs`'s own module doc), so a state
//! change one sub-agent makes is visible to the next sub-agent
//! immediately, before the `Runner` ever formally persists anything. This
//! port's `State`/`Context` copy instead of sharing by reference (the
//! same already-disclosed departure), so without a fix, a later sub-agent
//! in the same `SequentialAgent` run would never see an earlier one's
//! state changes — breaking exactly the "each step feeds the next" value
//! a sequential chain exists for. Fixed here by applying each sub-agent
//! event's `state_delta` onto `ctx.session.state` directly as the loop
//! processes it (mirroring, at a smaller scope, what
//! `SessionService::append_event`'s own state-merge step already does at
//! the persistence layer) — a sub-agent's own `BaseAgent::run_async`
//! clones `ctx.session` when it builds its working copy, so this update
//! is what makes it visible to the *next* sub-agent's clone.
//!
//! **Not ported this batch**: `_run_live_impl` (the `task_completed`
//! tool/instruction auto-injection for live mode) — needs `LlmAgent.tools`
//! wired to real `BaseTool`/`FunctionTool` instances (`canonical_tools`,
//! C0092, still a `ToolUnion` placeholder) to append a real completion
//! tool; `SequentialAgentConfig`/YAML config loading (C0338, deprecated
//! surface, needs the config-resolution pipeline C0348 discloses as
//! unbuilt); the `@deprecated`/`@experimental` decorators themselves (no
//! Rust equivalent — documented here instead, matching every other
//! decorator-only capability in this migration).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use adk_events::node_info::NodeInfo;
use adk_events::Event;
use rusty_serde::value::Value;

use crate::base_agent::{AgentBehavior, AgentRunError};
use crate::invocation_context::InvocationContext;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const CURRENT_SUB_AGENT_KEY: &str = "current_sub_agent";

/// C0335: resumable state for [`SequentialAgent`] — which sub-agent to
/// resume from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SequentialAgentState {
    pub current_sub_agent: String,
}

impl SequentialAgentState {
    fn to_raw(&self) -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert(
            CURRENT_SUB_AGENT_KEY.to_string(),
            Value::String(self.current_sub_agent.clone()),
        );
        map
    }

    fn from_raw(raw: &HashMap<String, Value>) -> Self {
        let current_sub_agent = match raw.get(CURRENT_SUB_AGENT_KEY) {
            Some(Value::String(name)) => name.clone(),
            _ => String::new(),
        };
        Self { current_sub_agent }
    }
}

/// C0335: a shell agent that runs its sub-agents in sequence, in tree
/// order — plug this in as a [`crate::base_agent::BaseAgent`]'s behavior.
#[derive(Debug, Default)]
pub struct SequentialAgent;

impl SequentialAgent {
    /// Resolves which index to resume from: 0 for a fresh run, the
    /// matched sub-agent's index for a resume, `sub_agents.len()` if the
    /// tracked state says the run already finished, or 0 (restart) if the
    /// tracked sub-agent name no longer exists in the tree.
    fn start_index(
        agent_state: Option<&SequentialAgentState>,
        sub_agents: &[crate::base_agent::BaseAgent],
    ) -> usize {
        let Some(agent_state) = agent_state else {
            return 0;
        };
        if agent_state.current_sub_agent.is_empty() {
            return sub_agents.len();
        }
        sub_agents
            .iter()
            .position(|sub_agent| sub_agent.name() == agent_state.current_sub_agent)
            .unwrap_or(0)
    }
}

impl AgentBehavior for SequentialAgent {
    fn run_async_impl<'a>(
        &'a self,
        ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async move {
            let Some(agent) = ctx.agent.clone() else {
                return Ok(Vec::new());
            };
            let sub_agents = agent.sub_agents().to_vec();
            if sub_agents.is_empty() {
                return Ok(Vec::new());
            }

            let mut events = Vec::new();
            let agent_state = ctx
                .agent_states
                .get(agent.name())
                .map(SequentialAgentState::from_raw);
            let start_index = Self::start_index(agent_state.as_ref(), &sub_agents);
            let mut resuming_sub_agent = agent_state.is_some();

            for sub_agent in &sub_agents[start_index..] {
                if !resuming_sub_agent && ctx.is_resumable() {
                    let state = SequentialAgentState {
                        current_sub_agent: sub_agent.name().to_string(),
                    };
                    ctx.set_agent_state(agent.name(), Some(state.to_raw()), false);
                    let mut marker =
                        Event::new(ctx.invocation_id.clone(), agent.name(), NodeInfo::new(""));
                    marker.branch = ctx.branch.clone();
                    marker.actions.agent_state = ctx.agent_states.get(agent.name()).cloned();
                    events.push(marker);
                }

                let produced = sub_agent.run_async(ctx).await?;
                let mut pause_invocation = false;
                for event in produced {
                    for (key, value) in event.actions.state_delta.iter() {
                        ctx.session.state.insert(key.clone(), value.clone());
                    }
                    if ctx.should_pause_invocation(&event) {
                        pause_invocation = true;
                    }
                    events.push(event);
                }

                if pause_invocation {
                    return Ok(events);
                }

                resuming_sub_agent = false;
            }

            if ctx.is_resumable() {
                ctx.set_agent_state(agent.name(), None, true);
                let mut marker =
                    Event::new(ctx.invocation_id.clone(), agent.name(), NodeInfo::new(""));
                marker.branch = ctx.branch.clone();
                marker.actions.end_of_agent = true;
                events.push(marker);
            }

            Ok(events)
        })
    }

    fn run_live_impl<'a>(
        &'a self,
        ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async move {
            let Some(agent) = ctx.agent.clone() else {
                return Ok(Vec::new());
            };
            let mut events = Vec::new();
            for sub_agent in agent.sub_agents().to_vec() {
                events.extend(sub_agent.run_live(ctx).await?);
            }
            Ok(events)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_agent::{AgentRunError as Error, BaseAgent, NoopBehavior};
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    struct RecordingBehavior {
        name: &'static str,
        state_key: Option<&'static str>,
    }

    impl AgentBehavior for RecordingBehavior {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, Error>> {
            let name = self.name;
            let state_key = self.state_key;
            Box::pin(async move {
                let mut event = Event::new(ctx.invocation_id.clone(), name, NodeInfo::new(""));
                if let Some(key) = state_key {
                    event
                        .actions
                        .state_delta
                        .insert(key.to_string(), Value::String(name.to_string()));
                }
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, Error>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    fn sequential_with(sub_agents: Vec<BaseAgent>) -> BaseAgent {
        BaseAgent::build(
            "sequential",
            "",
            sub_agents,
            Vec::new(),
            Vec::new(),
            SequentialAgent,
        )
        .unwrap()
    }

    fn parent_ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    #[rusty_tokio::test]
    async fn runs_every_sub_agent_in_order() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
            },
        )
        .unwrap();
        let b = BaseAgent::new(
            "b",
            RecordingBehavior {
                name: "b",
                state_key: None,
            },
        )
        .unwrap();
        let seq = sequential_with(vec![a, b]);

        let events = seq.run_async(&parent_ctx()).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].author, "a");
        assert_eq!(events[1].author, "b");
    }

    #[rusty_tokio::test]
    async fn returns_no_events_without_sub_agents() {
        let seq = sequential_with(Vec::new());
        let events = seq.run_async(&parent_ctx()).await.unwrap();
        assert!(events.is_empty());
    }

    #[rusty_tokio::test]
    async fn a_later_sub_agent_sees_an_earlier_ones_state_delta() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: Some("shared"),
            },
        )
        .unwrap();
        let b = BaseAgent::new("b", NoopBehavior).unwrap();
        let seq = sequential_with(vec![a, b]);

        let mut ctx = parent_ctx();
        ctx.agent = Some(seq.clone());
        SequentialAgent.run_async_impl(&mut ctx).await.unwrap();

        assert_eq!(
            ctx.session.state.get("shared"),
            Some(&Value::String("a".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn resumable_run_emits_agent_state_markers_and_a_final_end_of_agent_marker() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
            },
        )
        .unwrap();
        let seq = sequential_with(vec![a]);

        let mut ctx = parent_ctx();
        ctx.resumability_config =
            Some(crate::app_configs::ResumabilityConfig { is_resumable: true });
        ctx.agent = Some(seq.clone());

        let events = SequentialAgent.run_async_impl(&mut ctx).await.unwrap();
        // marker(current_sub_agent=a), a's own event, marker(end_of_agent).
        assert_eq!(events.len(), 3);
        assert!(events[0].actions.agent_state.is_some());
        assert_eq!(events[1].author, "a");
        assert!(events[2].actions.end_of_agent);
    }

    #[rusty_tokio::test]
    async fn resumes_from_the_tracked_sub_agent() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
            },
        )
        .unwrap();
        let b = BaseAgent::new(
            "b",
            RecordingBehavior {
                name: "b",
                state_key: None,
            },
        )
        .unwrap();
        let seq = sequential_with(vec![a, b]);

        let mut ctx = parent_ctx();
        ctx.resumability_config =
            Some(crate::app_configs::ResumabilityConfig { is_resumable: true });
        ctx.agent = Some(seq.clone());
        ctx.set_agent_state(
            "sequential",
            Some(
                SequentialAgentState {
                    current_sub_agent: "b".to_string(),
                }
                .to_raw(),
            ),
            false,
        );

        let events = SequentialAgent.run_async_impl(&mut ctx).await.unwrap();
        // No marker re-emitted for "b" (resuming_sub_agent=true skips it),
        // just b's own event, then the end-of-agent marker.
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].author, "b");
        assert!(events[1].actions.end_of_agent);
    }
}
