//! Capability C0833-C0926 (partial): `Runner`, ported from
//! `google.adk.runners`.
//!
//! **Scoping**: `runners.py` is 2609 lines / 94 capability rows, and most
//! of it depends on infrastructure this port doesn't have yet:
//! - An `App` type (`apps.app.App`, Phase 7) — `Runner.__init__`'s
//!   app/agent/node mutual-exclusivity contract, `App.model_construct`
//!   wrapping, and `app.root_agent`/`app.plugins` resolution (C0840-C0850)
//!   are all N/A here: this port's `Runner` only ever wraps a concrete
//!   [`BaseAgent`] directly, so there's no app/agent/node union to
//!   validate against.
//! - The workflow/node/task-delegation engine (`BaseNode`,
//!   `Context.run_node`, `DynamicNodeScheduler`, task-scope tracking,
//!   Phase 7) — confirmed absent from this port. Every node/task-mode
//!   code path (`_run_node_async`/`_run_node_live`, task-scope resolution,
//!   resumability) is out of scope until that engine exists.
//! - A real plugin system (`plugins/`, Phase 7) — `PluginManager` is
//!   currently a hardcoded no-op stub exposing only the per-agent
//!   `before`/`after_agent_callback` hooks `BaseAgent::run_async` already
//!   calls through it (so those ARE exercised, transitively, by this
//!   batch's `Runner::run_async` calling `agent.run_async`). The
//!   Runner-*level* plugin hooks `_exec_with_plugin` wraps a whole turn
//!   with (`run_before_run_callback`/`run_on_event_callback`/
//!   `run_after_run_callback`, plus `_notify_run_error`'s
//!   `on_run_error_callback`) don't exist on `PluginManager` at all yet —
//!   rather than invent placeholder methods for a shape that isn't real
//!   until Phase 7 defines it, `Runner` doesn't store a `PluginManager`
//!   field or call through one at its own level; this is a genuine,
//!   disclosed gap, not a silent no-op.
//! - Live mode (`run_live`), `rewind_async`, `run_debug` — each needs its
//!   own supporting infrastructure (live request queue wiring into
//!   `InvocationContext`, artifact-versioned rewind deltas, etc.) beyond
//!   this batch's scope.
//!
//! What's left, and what this batch builds, is the "legacy" (plain
//! `BaseAgent`, single always-non-resumable turn) execution path
//! (C0884-C0886 in spirit): [`Runner::run_async`] fetches-or-creates a
//! session, appends the user message, drives `agent.run_async` (the real
//! orchestration primitive `adk_agents::base_agent::BaseAgent` already
//! provides), and persists the resulting events. Also built: the
//! constructor/field contract (C0840-C0845, narrowed — see above) and
//! [`Runner::close`] (C0924, partial — closes what actually exists today:
//! `session_service.flush()`; toolset collection (C0922/C0923) is
//! deferred until `LlmAgent.tools` holds real `BaseToolset` instances
//! instead of the `ToolUnion` placeholder, and there is no plugin
//! registration to close).
//!
//! **Not ported this batch**: `_find_agent_to_run` (C0907-C0910, picks up
//! a resumed multi-turn conversation's last-active agent — needs
//! resumability, always false here); `_resolve_invocation_id` (C0855,
//! same reason); `Runner.run()`'s sync thread-bridging wrapper
//! (C0877-C0880, a local-testing convenience — this port's whole call
//! surface is already async-native, so there's less need for it, and it
//! can be added later without disturbing anything here);
//! `InMemoryRunner` (C0926, needs `InMemoryArtifactService`/
//! `InMemoryMemoryService`, neither built yet); compaction
//! (`_run_post_invocation_compaction`, C0871-C0872, needs
//! `events_compaction_config` wiring, Phase 7); agent-origin inference
//! and its warnings (C0851-C0854) — Rust has no runtime module-path
//! reflection to inspect, and no logging framework is adopted to warn
//! through even if it did.

use std::sync::Arc;

use adk_agents::base_agent::BaseAgent;
use adk_agents::invocation_context::InvocationContextBuilder;
use adk_agents::services::{
    new_invocation_context_id, ArtifactService, CredentialService, MemoryService, SessionService,
};
use adk_agents::session::Session;
use adk_errors::already_exists::AlreadyExistsError;
use adk_errors::session_not_found::SessionNotFoundError;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_genai::content::Content;

#[derive(Debug, rusty_err::Error)]
pub enum RunnerError {
    #[error("{0}")]
    SessionNotFound(SessionNotFoundError),
    #[error("{0}")]
    AlreadyExists(AlreadyExistsError),
    #[error("run_async rejects a user-authored new_message containing a function call")]
    NewMessageContainsFunctionCall,
    #[error("agent run failed: {0}")]
    AgentRun(String),
}

/// C0840-C0845 (narrowed, see the module doc): the core execution engine.
/// Wraps exactly one [`BaseAgent`] — no `App`/bare-node union, since
/// neither exists in this port.
pub struct Runner {
    app_name: String,
    agent: BaseAgent,
    session_service: Arc<dyn SessionService + Send + Sync>,
    artifact_service: Option<Arc<dyn ArtifactService + Send + Sync>>,
    memory_service: Option<Arc<dyn MemoryService + Send + Sync>>,
    credential_service: Option<Arc<dyn CredentialService + Send + Sync>>,
    /// C0844: stored for parity with the source's constructor contract,
    /// though nothing yet uses it — `close()` has no registered plugins
    /// to apply a timeout to until `plugins/` (Phase 7) lands.
    plugin_close_timeout: f64,
    /// C0845: the single switch controlling whether `run_async` creates
    /// a missing session or reports it as not found.
    auto_create_session: bool,
}

impl Runner {
    /// C0841 (narrowed): `app_name` and `agent` are always both required
    /// here — there's no `App`/bare-node alternative to be mutually
    /// exclusive against.
    pub fn new(
        app_name: impl Into<String>,
        agent: BaseAgent,
        session_service: Arc<dyn SessionService + Send + Sync>,
    ) -> Self {
        Self {
            app_name: app_name.into(),
            agent,
            session_service,
            artifact_service: None,
            memory_service: None,
            credential_service: None,
            plugin_close_timeout: 5.0,
            auto_create_session: false,
        }
    }

    pub fn with_artifact_service(
        mut self,
        service: Arc<dyn ArtifactService + Send + Sync>,
    ) -> Self {
        self.artifact_service = Some(service);
        self
    }

    pub fn with_memory_service(mut self, service: Arc<dyn MemoryService + Send + Sync>) -> Self {
        self.memory_service = Some(service);
        self
    }

    pub fn with_credential_service(
        mut self,
        service: Arc<dyn CredentialService + Send + Sync>,
    ) -> Self {
        self.credential_service = Some(service);
        self
    }

    pub fn with_plugin_close_timeout(mut self, seconds: f64) -> Self {
        self.plugin_close_timeout = seconds;
        self
    }

    pub fn with_auto_create_session(mut self, auto_create_session: bool) -> Self {
        self.auto_create_session = auto_create_session;
        self
    }

    pub fn app_name(&self) -> &str {
        &self.app_name
    }

    pub fn agent(&self) -> &BaseAgent {
        &self.agent
    }

    /// C0844: seconds `close()` will eventually bound each toolset's
    /// `close()` call by, once toolset collection (C0922/C0923) is wired.
    pub fn plugin_close_timeout(&self) -> f64 {
        self.plugin_close_timeout
    }

    /// C0873 (narrowed — no `GetSessionConfig`, see `adk-agents::services`'
    /// own disclosed scope cut): gets the named session, or creates it
    /// (empty of history) if missing and [`Runner::with_auto_create_session`]
    /// is set; otherwise reports it as not found.
    async fn get_or_create_session(
        &self,
        user_id: &str,
        session_id: &str,
    ) -> Result<Session, RunnerError> {
        if let Some(session) = self
            .session_service
            .get_session(&self.app_name, user_id, session_id)
            .await
        {
            return Ok(session);
        }
        if !self.auto_create_session {
            return Err(RunnerError::SessionNotFound(SessionNotFoundError::new(
                format!("Session not found: {session_id}"),
            )));
        }
        self.session_service
            .create_session(&self.app_name, user_id, None, Some(session_id.to_string()))
            .await
            .map_err(RunnerError::AlreadyExists)
    }

    /// C0884-C0886 (narrowed "legacy" path, see the module doc): runs one
    /// turn of the wrapped agent against `new_message`, returning the
    /// events it produced (the appended user-message event itself isn't
    /// included — the caller already knows what it sent, matching the
    /// source's own yielded-events contract).
    pub async fn run_async(
        &self,
        user_id: &str,
        session_id: &str,
        new_message: Content,
    ) -> Result<Vec<Event>, RunnerError> {
        if new_message
            .parts
            .iter()
            .any(|part| part.function_call.is_some())
        {
            return Err(RunnerError::NewMessageContainsFunctionCall);
        }

        let mut session = self.get_or_create_session(user_id, session_id).await?;

        let invocation_id = new_invocation_context_id();
        let mut invocation_context =
            InvocationContextBuilder::new(invocation_id.clone(), session.clone())
                .agent(self.agent.clone())
                .build();
        invocation_context.session_service = self.session_service.clone();
        invocation_context.artifact_service = self.artifact_service.clone();
        invocation_context.memory_service = self.memory_service.clone();
        invocation_context.credential_service = self.credential_service.clone();
        invocation_context.user_content = rusty_serde::json::to_value(&new_message).ok();

        let mut user_event = Event::new(invocation_id, "user", NodeInfo::new("root"));
        user_event.content = Some(new_message);
        self.session_service
            .append_event(&mut session, user_event)
            .await;
        invocation_context.session = session.clone();

        let events = self
            .agent
            .run_async(&invocation_context)
            .await
            .map_err(|error| RunnerError::AgentRun(error.to_string()))?;

        for event in &events {
            self.session_service
                .append_event(&mut session, event.clone())
                .await;
        }

        Ok(events)
    }

    /// C0924 (partial, see the module doc): closes what actually exists
    /// today — `session_service.flush()`. Toolset collection and plugin
    /// closing are both no-ops to port against right now (see the module
    /// doc), so they're omitted rather than faked.
    pub async fn close(&self) {
        self.session_service.flush().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::services::InMemorySessionService;
    use adk_genai::content::Part;

    struct EchoBehavior;

    impl adk_agents::base_agent::AgentBehavior for EchoBehavior {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let mut event = Event::new(
                    ctx.invocation_id.clone(),
                    "echo_agent",
                    NodeInfo::new("root"),
                );
                event.content = Some(Content::user_text("echo"));
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut adk_agents::invocation_context::InvocationContext,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Vec<Event>, adk_agents::base_agent::AgentRunError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    fn runner(auto_create_session: bool) -> Runner {
        let agent = BaseAgent::new("echo_agent", EchoBehavior).unwrap();
        Runner::new("app", agent, Arc::new(InMemorySessionService::new()))
            .with_auto_create_session(auto_create_session)
    }

    #[rusty_tokio::test]
    async fn run_async_reports_a_missing_session_when_auto_create_is_off() {
        let runner = runner(false);
        let err = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, RunnerError::SessionNotFound(_)));
    }

    #[rusty_tokio::test]
    async fn run_async_auto_creates_a_missing_session_and_drives_the_agent() {
        let runner = runner(true);
        let events = runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "echo_agent");
    }

    #[rusty_tokio::test]
    async fn run_async_persists_both_the_user_message_and_the_agent_events() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("hi"))
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        assert_eq!(session.events.len(), 2);
        assert_eq!(session.events[0].author, "user");
        assert_eq!(session.events[1].author, "echo_agent");
    }

    #[rusty_tokio::test]
    async fn run_async_rejects_a_new_message_containing_a_function_call() {
        let runner = runner(true);
        let message = Content::new(
            "user",
            vec![Part {
                function_call: Some(adk_genai::content::FunctionCall {
                    name: Some("do_thing".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        );
        let err = runner.run_async("user", "s1", message).await.unwrap_err();
        assert!(matches!(err, RunnerError::NewMessageContainsFunctionCall));
    }

    #[rusty_tokio::test]
    async fn close_flushes_the_session_service() {
        let runner = runner(true);
        runner.close().await;
    }

    #[rusty_tokio::test]
    async fn run_async_reuses_an_existing_session() {
        let runner = runner(true);
        runner
            .run_async("user", "s1", Content::user_text("first"))
            .await
            .unwrap();
        runner
            .run_async("user", "s1", Content::user_text("second"))
            .await
            .unwrap();

        let session = runner
            .session_service
            .get_session("app", "user", "s1")
            .await
            .unwrap();
        // 2 user events + 2 echoed agent events across both turns.
        assert_eq!(session.events.len(), 4);
    }
}
