//! Capability C0173: the `compaction` request processor, ported from
//! `google.adk.flows.llm_flows.compaction`.
//!
//! Runs token-threshold event compaction *before* contents are assembled
//! for a model call — distinct from `crate::apps_compaction`'s
//! post-invocation trigger (C0871/C0872, `Runner`-level, runs only after
//! an invocation finishes) and from `crate::compaction` (C0185,
//! compaction-aware history reconstruction at contents-build time). All
//! three read the same `EventCompaction`/`EventsCompactionConfig` types
//! but serve different points in the request lifecycle, matching the
//! source's own three distinct modules
//! (`flows/llm_flows/compaction.py`/`apps/compaction.py`/
//! `flows/llm_flows/_content_compaction.py`).
//!
//! **Adaptation, disclosed**: the source's `CompactionRequestProcessor
//! .run_async` is an `AsyncGenerator[Event, None]` that never actually
//! yields (both `return` statements precede a dead `yield`) — it only
//! mutates `invocation_context` as a side effect. [`apply_compaction_processor`]
//! returns `Result<(), CompactionTriggerError>` instead, matching every
//! other `LlmFlow::preprocess` step that only mutates `ctx`/`llm_request`
//! (`apply_context_cache`, `apply_output_schema_processor`).
//!
//! **Adaptation, disclosed**: the source's `require_agent` raises a
//! `TypeError` when `invocation_context.agent` is unset — a real
//! invariant violation for a live invocation, but every direct unit-test
//! call site in this crate (and any future caller that hasn't wired
//! `InvocationContext.agent`) would otherwise need a throwaway agent just
//! to exercise the no-op path. [`apply_compaction_processor`] treats a
//! missing `ctx.agent` as "nothing to compact against" and returns
//! `Ok(())` instead of erroring — the same "defensive, no full tree = skip
//! the enhancement" posture `instructions.rs::resolve_root_global_instruction`
//! already established for its own no-tree fallback.

use adk_agents::invocation_context::InvocationContext;

use crate::apps_compaction::{run_compaction_for_token_threshold_config, CompactionTriggerError};

/// `CompactionRequestProcessor.run_async`: if `ctx.events_compaction_config`
/// is fully configured for token-threshold compaction and the current
/// prompt token count crosses the threshold, summarizes the
/// retention-window candidates into a compaction event, appends it via
/// `ctx.session_service`, and marks `ctx.token_compaction_checked` —
/// which `Runner`'s post-invocation sliding-window trigger (C0872) reads
/// back as `skip_token_compaction`, so a flow-level compaction this round
/// isn't immediately redone at the end of the same invocation. A no-op
/// (not an error) whenever compaction isn't configured, isn't yet
/// triggered, or `ctx.agent` is unset — see the module doc.
pub async fn apply_compaction_processor(
    ctx: &mut InvocationContext,
) -> Result<(), CompactionTriggerError> {
    let Some(config) = ctx.events_compaction_config.clone() else {
        return Ok(());
    };
    let Some(agent) = ctx.agent.clone() else {
        return Ok(());
    };

    let agent_name = agent.name().to_string();
    let current_branch = ctx.branch.clone();
    let compaction_event = run_compaction_for_token_threshold_config(
        &config,
        &agent,
        &ctx.session.events,
        &agent_name,
        current_branch.as_deref(),
    )
    .await?;

    let Some(compaction_event) = compaction_event else {
        return Ok(());
    };

    let session_service = ctx.session_service.clone();
    session_service
        .append_event(&mut ctx.session, compaction_event)
        .await;
    ctx.token_compaction_checked = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::app_configs::{BaseEventsSummarizer, EventsCompactionConfig};
    use adk_agents::base_agent::{BaseAgent, NoopBehavior};
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::services::BoxFuture;
    use adk_agents::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_events::Event;
    use std::sync::Arc;

    struct StubSummarizer {
        event: Option<Event>,
    }

    impl BaseEventsSummarizer for StubSummarizer {
        fn maybe_summarize_events<'a>(
            &'a self,
            _events: &'a [Event],
        ) -> BoxFuture<'a, Option<Event>> {
            Box::pin(async move { self.event.clone() })
        }
    }

    fn agent_named(name: &str) -> BaseAgent {
        BaseAgent::new(name, NoopBehavior).unwrap()
    }

    fn ctx_with(
        agent: Option<BaseAgent>,
        config: Option<EventsCompactionConfig>,
    ) -> InvocationContext {
        let mut builder = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"));
        if let Some(agent) = agent {
            builder = builder.agent(agent);
        }
        let mut ctx = builder.build();
        ctx.events_compaction_config = config;
        ctx
    }

    fn token_threshold_config(
        summarizer: Option<Arc<dyn BaseEventsSummarizer>>,
    ) -> EventsCompactionConfig {
        EventsCompactionConfig {
            summarizer,
            token_threshold: Some(10),
            event_retention_size: Some(1),
            ..Default::default()
        }
    }

    fn event_with_usage(invocation_id: &str, prompt_tokens: i64) -> Event {
        let mut e = Event::new(invocation_id, "user", NodeInfo::new(""));
        e.usage_metadata = Some(rusty_serde::value::Value::Map(vec![(
            "promptTokenCount".to_string(),
            rusty_serde::value::Value::Int(prompt_tokens),
        )]));
        e
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_without_a_compaction_config() {
        let mut ctx = ctx_with(Some(agent_named("root")), None);
        apply_compaction_processor(&mut ctx).await.unwrap();
        assert!(!ctx.token_compaction_checked);
        assert!(ctx.session.events.is_empty());
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_without_an_agent_in_context() {
        let config = token_threshold_config(Some(Arc::new(StubSummarizer {
            event: Some(Event::new("inv-2", "model", NodeInfo::new(""))),
        })));
        let mut ctx = ctx_with(None, Some(config));
        ctx.session.events.push(event_with_usage("inv-1", 100));
        apply_compaction_processor(&mut ctx).await.unwrap();
        assert!(!ctx.token_compaction_checked);
    }

    #[rusty_tokio::test]
    async fn is_a_no_op_below_the_token_threshold() {
        let config = token_threshold_config(Some(Arc::new(StubSummarizer {
            event: Some(Event::new("inv-2", "model", NodeInfo::new(""))),
        })));
        let mut ctx = ctx_with(Some(agent_named("root")), Some(config));
        ctx.session.events.push(event_with_usage("inv-1", 1));
        apply_compaction_processor(&mut ctx).await.unwrap();
        assert!(!ctx.token_compaction_checked);
        assert!(ctx.session.events.len() == 1);
    }

    #[rusty_tokio::test]
    async fn appends_the_compaction_event_and_marks_token_compaction_checked() {
        let compaction_event = Event::new("inv-summary", "model", NodeInfo::new(""));
        let config = token_threshold_config(Some(Arc::new(StubSummarizer {
            event: Some(compaction_event),
        })));
        let mut ctx = ctx_with(Some(agent_named("root")), Some(config));
        // `event_retention_size: Some(1)` (see `token_threshold_config`)
        // means at least 2 candidate events are needed before there's
        // anything left to compact once the retained tail is set aside.
        ctx.session
            .events
            .push(Event::new("inv-0", "user", NodeInfo::new("")));
        ctx.session.events.push(event_with_usage("inv-1", 100));

        apply_compaction_processor(&mut ctx).await.unwrap();

        assert!(ctx.token_compaction_checked);
        assert!(ctx
            .session
            .events
            .iter()
            .any(|e| e.invocation_id == "inv-summary"));
    }

    #[rusty_tokio::test]
    async fn does_not_mark_token_compaction_checked_when_the_summarizer_yields_nothing() {
        let config = token_threshold_config(Some(Arc::new(StubSummarizer { event: None })));
        let mut ctx = ctx_with(Some(agent_named("root")), Some(config));
        ctx.session.events.push(event_with_usage("inv-1", 100));

        apply_compaction_processor(&mut ctx).await.unwrap();

        assert!(!ctx.token_compaction_checked);
    }
}
