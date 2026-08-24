//! Capability C0337 (partial): `LoopAgent`, ported from
//! `google.adk.agents.loop_agent`.
//!
//! Structurally, this is `SequentialAgent` (same module, same
//! `crate::sequential_agent`, own module doc) wrapped in an outer loop
//! that restarts from the first sub-agent, up to `max_iterations` times
//! (or forever if unset), stopping early the moment a sub-agent escalates.
//! It reuses `SequentialAgent`'s own already-disclosed adaptation
//! verbatim: this port's `Context`/`State` copy rather than share state
//! by reference, so `run_async_impl` applies each produced event's
//! `state_delta` onto `ctx.session.state` directly as it processes it —
//! without this, a sub-agent in iteration 2 would never see state a
//! sub-agent in iteration 1 set.
//!
//! **Not ported this batch**: `_run_live_impl` — the source itself raises
//! `NotImplementedError` for live mode (never implemented upstream
//! either), so this port's `run_live_impl` does the same;
//! `LoopAgentConfig`/`_parse_config`/YAML config loading (C0338, needs
//! the config-resolution pipeline C0348 discloses as unbuilt — this is
//! also how the source reads `max_iterations` from YAML; construct a
//! `LoopAgent` directly with [`LoopAgent::with_max_iterations`] instead).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use adk_events::node_info::NodeInfo;
use adk_events::Event;
use rusty_serde::value::Value;

use crate::base_agent::{AgentBehavior, AgentRunError, BaseAgent};
use crate::invocation_context::InvocationContext;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const CURRENT_SUB_AGENT_KEY: &str = "current_sub_agent";
const TIMES_LOOPED_KEY: &str = "times_looped";

/// C0337: resumable state for [`LoopAgent`] — which sub-agent to resume
/// from, and how many full loop iterations have completed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopAgentState {
    pub current_sub_agent: String,
    pub times_looped: u32,
}

impl LoopAgentState {
    fn to_raw(&self) -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert(
            CURRENT_SUB_AGENT_KEY.to_string(),
            Value::String(self.current_sub_agent.clone()),
        );
        map.insert(
            TIMES_LOOPED_KEY.to_string(),
            Value::Int(self.times_looped as i64),
        );
        map
    }

    fn from_raw(raw: &HashMap<String, Value>) -> Self {
        let current_sub_agent = match raw.get(CURRENT_SUB_AGENT_KEY) {
            Some(Value::String(name)) => name.clone(),
            _ => String::new(),
        };
        let times_looped = match raw.get(TIMES_LOOPED_KEY) {
            Some(Value::Int(n)) if *n >= 0 => *n as u32,
            _ => 0,
        };
        Self {
            current_sub_agent,
            times_looped,
        }
    }
}

fn agent_state_marker(ctx: &InvocationContext, agent_name: &str) -> Event {
    let mut marker = Event::new(ctx.invocation_id.clone(), agent_name, NodeInfo::new(""));
    marker.branch = ctx.branch.clone();
    marker
}

/// C0337: a shell agent that runs its sub-agents in a loop, stopping when
/// a sub-agent escalates or [`LoopAgent::max_iterations`] is reached (if
/// set) — plug this in as a [`crate::base_agent::BaseAgent`]'s behavior.
#[derive(Debug, Clone, Default)]
pub struct LoopAgent {
    /// The maximum number of loop iterations. `None` runs indefinitely
    /// until a sub-agent escalates.
    pub max_iterations: Option<u32>,
}

impl LoopAgent {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Resolves `(times_looped, start_index)` from the tracked state —
    /// `times_looped` is preserved even when the tracked sub-agent name
    /// no longer exists in the tree (only `start_index` restarts at 0 in
    /// that case, matching the source).
    fn start_state(agent_state: Option<&LoopAgentState>, sub_agents: &[BaseAgent]) -> (u32, usize) {
        let Some(agent_state) = agent_state else {
            return (0, 0);
        };
        if agent_state.current_sub_agent.is_empty() {
            return (agent_state.times_looped, 0);
        }
        let start_index = sub_agents
            .iter()
            .position(|sub_agent| sub_agent.name() == agent_state.current_sub_agent)
            .unwrap_or(0);
        (agent_state.times_looped, start_index)
    }
}

impl AgentBehavior for LoopAgent {
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
                .map(LoopAgentState::from_raw);
            let mut is_resuming_at_current_agent = agent_state.is_some();
            let (mut times_looped, mut start_index) =
                Self::start_state(agent_state.as_ref(), &sub_agents);

            let mut should_exit = false;
            let mut pause_invocation = false;

            while (self.max_iterations.is_none_or(|max| times_looped < max))
                && !(should_exit || pause_invocation)
            {
                for sub_agent in &sub_agents[start_index..] {
                    if ctx.is_resumable() && !is_resuming_at_current_agent {
                        let state = LoopAgentState {
                            current_sub_agent: sub_agent.name().to_string(),
                            times_looped,
                        };
                        ctx.set_agent_state(agent.name(), Some(state.to_raw()), false);
                        let mut marker = agent_state_marker(ctx, agent.name());
                        marker.actions.agent_state = ctx.agent_states.get(agent.name()).cloned();
                        events.push(marker);
                    }
                    is_resuming_at_current_agent = false;

                    let produced = sub_agent.run_async(ctx).await?;
                    for event in produced {
                        for (key, value) in event.actions.state_delta.iter() {
                            ctx.session.state.insert(key.clone(), value.clone());
                        }
                        if event.actions.escalate {
                            should_exit = true;
                        }
                        if ctx.should_pause_invocation(&event) {
                            pause_invocation = true;
                        }
                        events.push(event);
                    }

                    if should_exit || pause_invocation {
                        break;
                    }
                }

                if !pause_invocation {
                    start_index = 0;
                    times_looped += 1;
                    ctx.reset_sub_agent_states(agent.name());
                }
            }

            if pause_invocation {
                return Ok(events);
            }

            if ctx.is_resumable() {
                ctx.set_agent_state(agent.name(), None, true);
                let mut marker = agent_state_marker(ctx, agent.name());
                marker.actions.end_of_agent = true;
                events.push(marker);
            }

            Ok(events)
        })
    }

    fn run_live_impl<'a>(
        &'a self,
        _ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async { Err(Box::new(LoopAgentError::LiveNotSupported) as AgentRunError) })
    }
}

/// [`AgentRunError`] is a boxed `std::error::Error`, not this crate's own
/// `rusty_err::Error` — implemented by hand for that reason (matching
/// `parallel_agent.rs`'s own choice).
#[derive(Debug)]
pub enum LoopAgentError {
    LiveNotSupported,
}

impl std::fmt::Display for LoopAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LiveNotSupported => f.write_str("This is not supported yet for LoopAgent."),
        }
    }
}

impl std::error::Error for LoopAgentError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    struct RecordingBehavior {
        name: &'static str,
        state_key: Option<&'static str>,
        escalate: bool,
    }

    impl AgentBehavior for RecordingBehavior {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
            let name = self.name;
            let state_key = self.state_key;
            let escalate = self.escalate;
            let invocation_id = ctx.invocation_id.clone();
            Box::pin(async move {
                let mut event = Event::new(invocation_id, name, NodeInfo::new(""));
                if let Some(key) = state_key {
                    event
                        .actions
                        .state_delta
                        .insert(key.to_string(), Value::String(name.to_string()));
                }
                event.actions.escalate = escalate;
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    fn loop_with(sub_agents: Vec<BaseAgent>, loop_agent: LoopAgent) -> BaseAgent {
        BaseAgent::build("loop", "", sub_agents, Vec::new(), Vec::new(), loop_agent).unwrap()
    }

    fn parent_ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    #[rusty_tokio::test]
    async fn runs_until_max_iterations_is_reached() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
                escalate: false,
            },
        )
        .unwrap();
        let looped = loop_with(vec![a], LoopAgent::new().with_max_iterations(3));

        let events = looped.run_async(&parent_ctx()).await.unwrap();
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e.author == "a"));
    }

    #[rusty_tokio::test]
    async fn stops_early_on_escalate() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
                escalate: true,
            },
        )
        .unwrap();
        let looped = loop_with(vec![a], LoopAgent::new().with_max_iterations(10));

        let events = looped.run_async(&parent_ctx()).await.unwrap();
        assert_eq!(
            events.len(),
            1,
            "should stop after the first escalating iteration"
        );
    }

    #[rusty_tokio::test]
    async fn runs_forever_without_max_iterations_until_escalate() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
                escalate: false,
            },
        )
        .unwrap();
        let b = BaseAgent::new(
            "b",
            RecordingBehavior {
                name: "b",
                state_key: None,
                escalate: true,
            },
        )
        .unwrap();
        // No max_iterations: relies solely on b's escalate to terminate.
        let looped = loop_with(vec![a, b], LoopAgent::new());

        let events = looped.run_async(&parent_ctx()).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].author, "a");
        assert_eq!(events[1].author, "b");
    }

    #[rusty_tokio::test]
    async fn returns_no_events_without_sub_agents() {
        let looped = loop_with(Vec::new(), LoopAgent::new().with_max_iterations(1));
        let events = looped.run_async(&parent_ctx()).await.unwrap();
        assert!(events.is_empty());
    }

    #[rusty_tokio::test]
    async fn a_later_iteration_sees_an_earlier_ones_state_delta() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: Some("shared"),
                escalate: false,
            },
        )
        .unwrap();
        let looped = loop_with(vec![a], LoopAgent::new().with_max_iterations(2));

        let mut ctx = parent_ctx();
        ctx.agent = Some(looped.clone());
        LoopAgent::new()
            .with_max_iterations(2)
            .run_async_impl(&mut ctx)
            .await
            .unwrap();

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
                escalate: false,
            },
        )
        .unwrap();
        let looped = loop_with(vec![a], LoopAgent::new().with_max_iterations(1));

        let mut ctx = parent_ctx();
        ctx.resumability_config =
            Some(crate::invocation_context::ResumabilityConfigStub { is_resumable: true });
        ctx.agent = Some(looped.clone());

        let events = LoopAgent::new()
            .with_max_iterations(1)
            .run_async_impl(&mut ctx)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);
        assert!(events[0].actions.agent_state.is_some());
        assert_eq!(events[1].author, "a");
        assert!(events[2].actions.end_of_agent);
    }

    #[rusty_tokio::test]
    async fn run_live_impl_is_not_supported() {
        let mut ctx = parent_ctx();
        let err = LoopAgent::new().run_live_impl(&mut ctx).await.unwrap_err();
        assert_eq!(err.to_string(), "This is not supported yet for LoopAgent.");
    }
}
