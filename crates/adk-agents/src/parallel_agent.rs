//! Capability C0336 (partial): `ParallelAgent`, ported from
//! `google.adk.agents.parallel_agent`.
//!
//! **Adaptation, concurrency model**: the source's `_merge_agent_run`
//! merges sub-agent event *streams* through a queue with per-event
//! backpressure (each sub-agent waits for the runner to consume an event
//! before producing its next one), so it can cancel the remaining
//! sub-agents the instant one escalates. This port's
//! [`crate::base_agent::AgentBehavior`] returns a fully-collected
//! `Vec<Event>` per run rather than a live stream (an adaptation already
//! disclosed back in `base_agent.rs`'s own module doc) — so there is no
//! partial result to race against or cancel mid-flight; a sub-agent's
//! `run_async` call is already atomic by the time anything can observe
//! its output. Sub-agents still run with genuine concurrency (via
//! `rusty_tokio::spawn`, one task per sub-agent), but: (1) escalate
//! detection happens after every included sub-agent has already run to
//! completion, not the instant it occurs, and (2) a sibling already
//! mid-flight when one escalates is **not** cancelled — it finishes
//! normally. Both are direct, disclosed consequences of the earlier
//! streaming-vs-`Vec` decision, not a new gap this batch introduces.
//!
//! **Adaptation, cross-branch state visibility**: each sub-agent runs
//! against its own branched clone of the `InvocationContext` (matching
//! the source's own `_create_branch_ctx_for_sub_agent`, which likewise
//! `model_copy()`s the context per sub-agent) — but the source's copy is
//! shallow, so `agent_states`/`end_of_agents`/session state remain the
//! SAME shared dicts across every branch and the parent; a sub-agent
//! marking itself done is instantly visible to the parent's own view.
//! This port's `InvocationContext::clone()` is a real deep clone (already
//! disclosed for `SequentialAgent`, for the same reason), so nothing a
//! sub-agent's own branch mutates is visible on the parent's `ctx`
//! automatically. This batch propagates what it reasonably can back onto
//! the parent post-hoc: every produced event's `state_delta` is applied
//! to `ctx.session.state` (same fix as `SequentialAgent`). Full nested
//! resumability propagation (a sub-agent tree's own internal
//! `agent_states`/`end_of_agents` writes reaching the parent mid-turn) is
//! NOT implemented — the "already finished in a previous turn" skip
//! check reads `ctx.end_of_agents` as already populated by the caller
//! (e.g. `populate_invocation_agent_states` at a resumed turn's start,
//! which reads real session history, not per-turn sub-agent-branch
//! state), and the "did every current sub-agent finish this turn"
//! determination is derived from whether this turn's run completed
//! without pausing, not from replaying each sub-agent's own nested
//! agent-state events — correct for the common (non-nested-resumable)
//! case, narrower for a sub-agent that is itself independently paused
//! mid-tree. Flagged, not silently dropped.
//!
//! **Not ported this batch**: `_run_live_impl` — the source itself
//! raises `NotImplementedError` for live mode (never implemented
//! upstream either), so this port's `run_live_impl` does the same;
//! `ParallelAgentConfig`/YAML config loading (C0338, needs the
//! config-resolution pipeline C0348 discloses as unbuilt).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use adk_events::branch_path::BranchPath;
use adk_events::node_info::NodeInfo;
use adk_events::Event;

use crate::base_agent::{AgentBehavior, AgentRunError, BaseAgent};
use crate::invocation_context::InvocationContext;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// [`AgentRunError`] is a boxed `std::error::Error`, not this crate's own
/// `rusty_err::Error` — implemented by hand for that reason (matching the
/// same choice `base_agent.rs`'s own test doubles make).
#[derive(Debug)]
pub enum ParallelAgentError {
    SubAgentTaskFailed,
    LiveNotSupported,
}

impl std::fmt::Display for ParallelAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubAgentTaskFailed => f.write_str("a sub-agent's task panicked or was cancelled"),
            Self::LiveNotSupported => f.write_str("This is not supported yet for ParallelAgent."),
        }
    }
}

impl std::error::Error for ParallelAgentError {}

/// C0336: a shell agent that runs its sub-agents in parallel, each in its
/// own isolated branch — plug this in as a
/// [`crate::base_agent::BaseAgent`]'s behavior. See the module doc for
/// the concurrency-model and cross-branch-state adaptations.
#[derive(Debug, Default)]
pub struct ParallelAgent;

fn branch_ctx_for_sub_agent(
    agent: &BaseAgent,
    sub_agent: &BaseAgent,
    ctx: &InvocationContext,
) -> InvocationContext {
    let mut sub_ctx = ctx.clone();
    let base = BranchPath::from_string(ctx.branch.as_deref().unwrap_or(""));
    let branch_name = format!("{}.{}", agent.name(), sub_agent.name());
    sub_ctx.branch = Some(base.create_sub_branch(branch_name, None).to_dotted_string());
    sub_ctx
}

fn agent_state_marker(ctx: &InvocationContext, agent_name: &str) -> Event {
    let mut marker = Event::new(ctx.invocation_id.clone(), agent_name, NodeInfo::new(""));
    marker.branch = ctx.branch.clone();
    marker
}

impl AgentBehavior for ParallelAgent {
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
            let already_started = ctx.agent_states.contains_key(agent.name());
            if ctx.is_resumable() && !already_started {
                ctx.set_agent_state(agent.name(), Some(HashMap::new()), false);
                let mut marker = agent_state_marker(ctx, agent.name());
                marker.actions.agent_state = ctx.agent_states.get(agent.name()).cloned();
                events.push(marker);
            }

            let mut handles = Vec::new();
            for sub_agent in &sub_agents {
                if ctx
                    .end_of_agents
                    .get(sub_agent.name())
                    .copied()
                    .unwrap_or(false)
                {
                    // Already finished in a previous (paused) run.
                    continue;
                }
                let sub_ctx = branch_ctx_for_sub_agent(&agent, sub_agent, ctx);
                let sub_agent = sub_agent.clone();
                handles.push(rusty_tokio::spawn(async move {
                    sub_agent.run_async(&sub_ctx).await
                }));
            }

            for handle in handles {
                let produced = handle.await.map_err(|_| {
                    Box::new(ParallelAgentError::SubAgentTaskFailed) as AgentRunError
                })??;
                for event in produced {
                    // Escalate detection isn't tracked here: the source's
                    // `escalated or all(...)` end-of-agent gate collapses
                    // to always-true at this point in this port's model
                    // (see the module doc) — there's no early-cancellation
                    // path for it to also gate.
                    for (key, value) in event.actions.state_delta.iter() {
                        ctx.session.state.insert(key.clone(), value.clone());
                    }
                    events.push(event);
                }
            }

            if events
                .iter()
                .any(|event| ctx.should_pause_invocation(event))
            {
                return Ok(events);
            }

            // Reaching this point (no pause above) means every sub-agent
            // that ran this turn finished, and every sub-agent that
            // didn't run was already finished before this turn started
            // (that's the skip condition above) — so all sub-agents are
            // now done. See the module doc: "all finished" is derived
            // from this turn completing without a pause, not from
            // replaying each sub-agent's own nested agent-state events.
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
        Box::pin(async { Err(Box::new(ParallelAgentError::LiveNotSupported) as AgentRunError) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use rusty_serde::value::Value;

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

    fn parallel_with(sub_agents: Vec<BaseAgent>) -> BaseAgent {
        BaseAgent::build(
            "parallel",
            "",
            sub_agents,
            Vec::new(),
            Vec::new(),
            ParallelAgent,
        )
        .unwrap()
    }

    fn parent_ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    #[rusty_tokio::test]
    async fn runs_every_sub_agent_concurrently_and_collects_all_events() {
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
                escalate: false,
            },
        )
        .unwrap();
        let par = parallel_with(vec![a, b]);

        let events = par.run_async(&parent_ctx()).await.unwrap();
        let mut authors: Vec<&str> = events.iter().map(|e| e.author.as_str()).collect();
        authors.sort();
        assert_eq!(authors, vec!["a", "b"]);
    }

    #[rusty_tokio::test]
    async fn returns_no_events_without_sub_agents() {
        let par = parallel_with(Vec::new());
        let events = par.run_async(&parent_ctx()).await.unwrap();
        assert!(events.is_empty());
    }

    #[rusty_tokio::test]
    async fn merges_every_sub_agents_state_delta_onto_the_parent_context() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: Some("from_a"),
                escalate: false,
            },
        )
        .unwrap();
        let b = BaseAgent::new(
            "b",
            RecordingBehavior {
                name: "b",
                state_key: Some("from_b"),
                escalate: false,
            },
        )
        .unwrap();
        let par = parallel_with(vec![a, b]);

        let mut ctx = parent_ctx();
        ctx.agent = Some(par.clone());
        ParallelAgent.run_async_impl(&mut ctx).await.unwrap();

        assert_eq!(
            ctx.session.state.get("from_a"),
            Some(&Value::String("a".to_string()))
        );
        assert_eq!(
            ctx.session.state.get("from_b"),
            Some(&Value::String("b".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn each_sub_agent_runs_in_its_own_branch() {
        struct BranchCapturingBehavior;
        impl AgentBehavior for BranchCapturingBehavior {
            fn run_async_impl<'a>(
                &'a self,
                ctx: &'a mut InvocationContext,
            ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
                let branch = ctx.branch.clone();
                let invocation_id = ctx.invocation_id.clone();
                Box::pin(async move {
                    let mut event = Event::new(invocation_id, "child", NodeInfo::new(""));
                    event.branch = branch;
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
        let child = BaseAgent::new("child", BranchCapturingBehavior).unwrap();
        let par = parallel_with(vec![child]);

        let events = par.run_async(&parent_ctx()).await.unwrap();
        assert_eq!(events[0].branch.as_deref(), Some("parallel.child"));
    }

    #[rusty_tokio::test]
    async fn resumable_run_emits_start_and_end_of_agent_markers() {
        let a = BaseAgent::new(
            "a",
            RecordingBehavior {
                name: "a",
                state_key: None,
                escalate: false,
            },
        )
        .unwrap();
        let par = parallel_with(vec![a]);

        let mut ctx = parent_ctx();
        ctx.resumability_config =
            Some(crate::app_configs::ResumabilityConfig { is_resumable: true });
        ctx.agent = Some(par.clone());

        let events = ParallelAgent.run_async_impl(&mut ctx).await.unwrap();
        assert!(events[0].actions.agent_state.is_some());
        assert_eq!(events[1].author, "a");
        assert!(events[2].actions.end_of_agent);
    }

    #[rusty_tokio::test]
    async fn skips_a_sub_agent_already_marked_finished_in_a_previous_run() {
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
                escalate: false,
            },
        )
        .unwrap();
        let par = parallel_with(vec![a, b]);

        let mut ctx = parent_ctx();
        ctx.agent = Some(par.clone());
        ctx.end_of_agents.insert("a".to_string(), true);

        let events = ParallelAgent.run_async_impl(&mut ctx).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "b");
    }

    #[rusty_tokio::test]
    async fn run_live_impl_is_not_supported() {
        let mut ctx = parent_ctx();
        let err = ParallelAgent.run_live_impl(&mut ctx).await.unwrap_err();
        assert_eq!(
            err.to_string(),
            "This is not supported yet for ParallelAgent."
        );
    }
}
