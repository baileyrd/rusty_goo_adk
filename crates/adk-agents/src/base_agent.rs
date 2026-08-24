//! Capabilities C0035-C0047: `BaseAgent`/`BaseAgentState`, ported from
//! `google.adk.agents.base_agent`.
//!
//! **Adaptation, tree ownership**: Python's `parent_agent`/`sub_agents` are
//! plain object references, mutated once at tree-construction time. Rust has
//! no shared-mutable-object-graph equivalent without indirection; this port
//! uses `Arc<BaseAgentData>` (shared, cheap-clone handles — same
//! `SharedRng`-style pattern used for `adk_platform::random`) with a
//! `std::sync::OnceLock<Weak<BaseAgentData>>` back-pointer for
//! `parent_agent`. `OnceLock` mirrors the source's "can only be adopted as a
//! sub-agent once" invariant directly: a second `set()` fails, which is
//! exactly the `ValueError` the source raises on a double-add. `Weak` (not a
//! second strong `Arc`) avoids a parent/child reference cycle that would
//! leak memory (Python's GC handles cycles; Rust's `Arc` does not).
//!
//! **Adaptation, callbacks**: modeled directly as the *resolved* list form
//! (what the source calls `canonical_before_agent_callbacks`/
//! `canonical_after_agent_callbacks`) rather than a single-or-list union
//! type, since that resolved form is all any caller ever needs. Callback
//! bodies are synchronous closures for this batch (the source allows
//! `async def` callbacks and awaits them if awaitable) — no callback in this
//! codebase yet needs to `.await` anything; revisit if a later phase's
//! callback genuinely needs asynchronous work.
//!
//! **Adaptation, `_run_async_impl`/`_run_live_impl`**: the source's
//! `AsyncGenerator[Event, None]` is represented here as an eagerly-collected
//! `Vec<Event>` rather than a live channel/stream, since no concrete agent
//! subclass in this batch (`LlmAgent` lands in Phase 2 batch 2) yet needs
//! incremental/backpressured yielding — `LiveRequestQueue` already
//! demonstrates the channel-based pattern this would use if/when a
//! concrete subclass needs it. This only changes *when* events become
//! available to a caller, not their content or order.
//!
//! **Deferred**: `BaseAgent._run_impl` (C0043, the workflow-node adapter)
//! needs `workflow::BaseNode` (Phase 7) and is not implemented — `BaseAgent`
//! is a standalone struct here, not a graph node, until that phase lands.
//! OTel span/context propagation (`_instrumentation.record_agent_invocation`,
//! `opentelemetry::context::attach`/`detach`) is Phase 12 and is a no-op
//! here. `PluginManager`'s agent-level hooks (C0354) are real (see
//! `services.rs`) and wired here: `run_async`/`run_live` now read the
//! actual `PluginManager` off the built `InvocationContext` (`ctx.plugin_manager`)
//! instead of constructing a fresh, always-empty one — a latent bug this
//! batch fixes, since a `Runner`-configured `PluginManager` would
//! otherwise silently never run.
//! `BaseAgent::from_config`/`_parse_config`/`BaseAgentConfig` (C0047, the
//! deprecated YAML-loading pipeline) is a data-shape-only capability for now
//! (see `base_agent_config.rs`); its dynamic agent/callback *resolution*
//! (`config_agent_utils.py`, dotted-path dynamic loading with no Rust
//! equivalent) is flagged, not silently dropped, as needing its own design
//! decision before implementation.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, Weak};

use adk_events::Event;
use adk_genai::content::Content;

use crate::context::Context;
use crate::invocation_context::InvocationContext;
use crate::services::PluginManager;

/// A callback invoked before/after an agent's run. Returning `Some(content)`
/// short-circuits the chain (see [`run_before_agent_callbacks`]/
/// [`run_after_agent_callbacks`]).
pub type AgentCallback = Arc<dyn Fn(&mut Context) -> Option<Content> + Send + Sync>;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The error type an [`AgentBehavior`] run can fail with — the source
/// allows raising any `Exception`; this is narrowed to a boxed
/// `std::error::Error` since nothing in this batch needs to match on a
/// specific behavior-error variant (only propagate it after notifying the
/// error callback, per C0041/C0046).
pub type AgentRunError = Box<dyn std::error::Error + Send + Sync>;

/// The abstract behavior every concrete agent type supplies — the source's
/// `_run_async_impl`/`_run_live_impl`. See the module doc for the
/// `Vec<Event>` (vs. streaming) adaptation.
pub trait AgentBehavior: Send + Sync {
    fn run_async_impl<'a>(
        &'a self,
        ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>>;

    fn run_live_impl<'a>(
        &'a self,
        ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>>;
}

/// A behavior that produces no events — the "abstract, unimplemented" case
/// (the source's `_run_async_impl` raises `NotImplementedError` by default).
/// Used directly by [`BaseAgent::new`] and as a test double.
#[derive(Debug, Default)]
pub struct NoopBehavior;

impl AgentBehavior for NoopBehavior {
    fn run_async_impl<'a>(
        &'a self,
        _ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn run_live_impl<'a>(
        &'a self,
        _ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

/// Capability C0039: base class for every agent's resumable state.
/// Experimental in the source (`FeatureName.AGENT_STATE`) — see
/// `context_cache_config.rs`'s module doc for why that's documented, not
/// runtime-enforced, here.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BaseAgentState;

#[derive(Debug, rusty_err::Error)]
pub enum BaseAgentError {
    #[error("Found invalid agent name: `{0}`. Agent name must be a valid identifier. It should start with a letter (a-z, A-Z) or an underscore (_), and can only contain letters, digits (0-9), and underscores.")]
    InvalidName(String),
    #[error("Agent name cannot be `user`. `user` is reserved for end-user's input.")]
    ReservedName,
    #[error("Agent `{0}` already has a parent agent, current parent: `{1}`, trying to add: `{2}`")]
    AlreadyHasParent(String, String, String),
    #[error("Cannot update `parent_agent` field in clone. Parent agent is set only when the parent agent is instantiated with the sub-agents.")]
    ParentAgentInClone,
    #[error("Cannot update nonexistent fields in agent: {0}")]
    UnknownCloneField(String),
}

fn validate_name(name: &str) -> Result<(), BaseAgentError> {
    let is_identifier = {
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        }
    };
    if !is_identifier {
        return Err(BaseAgentError::InvalidName(name.to_string()));
    }
    if name == "user" {
        return Err(BaseAgentError::ReservedName);
    }
    Ok(())
}

struct BaseAgentData {
    name: String,
    description: String,
    sub_agents: Vec<BaseAgent>,
    parent_agent: OnceLock<Weak<BaseAgentData>>,
    before_agent_callback: Vec<AgentCallback>,
    after_agent_callback: Vec<AgentCallback>,
    behavior: Box<dyn AgentBehavior>,
}

/// Capabilities C0035-C0047: `BaseAgent`. A cheap-clone handle sharing one
/// underlying [`BaseAgentData`] — see the module doc's tree-ownership note.
#[derive(Clone)]
pub struct BaseAgent(Arc<BaseAgentData>);

impl BaseAgent {
    pub fn new(
        name: impl Into<String>,
        behavior: impl AgentBehavior + 'static,
    ) -> Result<Self, BaseAgentError> {
        Self::build(
            name,
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            behavior,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        name: impl Into<String>,
        description: impl Into<String>,
        sub_agents: Vec<BaseAgent>,
        before_agent_callback: Vec<AgentCallback>,
        after_agent_callback: Vec<AgentCallback>,
        behavior: impl AgentBehavior + 'static,
    ) -> Result<Self, BaseAgentError> {
        let name = name.into();
        validate_name(&name)?;

        // C0037: duplicate sub-agent names are warned about, not an error.
        let mut seen = std::collections::BTreeSet::new();
        for sub_agent in &sub_agents {
            if !seen.insert(sub_agent.name().to_string()) {
                eprintln!(
                    "Found duplicate sub-agent names: `{}`. All sub-agents must have unique names.",
                    sub_agent.name()
                );
            }
        }

        let agent = BaseAgent(Arc::new(BaseAgentData {
            name,
            description: description.into(),
            sub_agents,
            parent_agent: OnceLock::new(),
            before_agent_callback,
            after_agent_callback,
            behavior: Box::new(behavior),
        }));

        // C0036: set each sub-agent's parent exactly once; error on double-add.
        for sub_agent in &agent.0.sub_agents {
            let weak_self = Arc::downgrade(&agent.0);
            if sub_agent.0.parent_agent.set(weak_self).is_err() {
                let current_parent = sub_agent
                    .parent_agent()
                    .map(|p| p.name().to_string())
                    .unwrap_or_default();
                return Err(BaseAgentError::AlreadyHasParent(
                    sub_agent.name().to_string(),
                    current_parent,
                    agent.name().to_string(),
                ));
            }
        }

        Ok(agent)
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn description(&self) -> &str {
        &self.0.description
    }

    pub fn sub_agents(&self) -> &[BaseAgent] {
        &self.0.sub_agents
    }

    /// C0036: the parent agent, if this agent has been adopted as a
    /// sub-agent.
    pub fn parent_agent(&self) -> Option<BaseAgent> {
        self.0
            .parent_agent
            .get()
            .and_then(|weak| weak.upgrade())
            .map(BaseAgent)
    }

    /// C0044: the root of this agent's tree.
    pub fn root_agent(&self) -> BaseAgent {
        let mut root = self.clone();
        while let Some(parent) = root.parent_agent() {
            root = parent;
        }
        root
    }

    /// C0044: finds the agent with the given name in this agent and its
    /// descendants (self-first, then descendants, first match wins).
    pub fn find_agent(&self, name: &str) -> Option<BaseAgent> {
        if self.name() == name {
            return Some(self.clone());
        }
        self.find_sub_agent(name)
    }

    /// C0044: finds the agent with the given name among this agent's
    /// descendants only.
    pub fn find_sub_agent(&self, name: &str) -> Option<BaseAgent> {
        for sub_agent in self.sub_agents() {
            if let Some(found) = sub_agent.find_agent(name) {
                return Some(found);
            }
        }
        None
    }

    /// C0040: deep-copies this agent, including its sub-agents.
    ///
    /// `update` may rename this agent or replace non-tree fields; it may
    /// never set `parent_agent` (rejected, matching the source) nor name a
    /// field this agent doesn't have.
    pub fn clone_with(&self, update: BaseAgentUpdate) -> Result<BaseAgent, BaseAgentError> {
        if update.parent_agent_requested {
            return Err(BaseAgentError::ParentAgentInClone);
        }
        let name = update.name.unwrap_or_else(|| self.name().to_string());
        let description = update
            .description
            .unwrap_or_else(|| self.description().to_string());
        let sub_agents = match update.sub_agents {
            Some(replacement) => replacement,
            None => self
                .sub_agents()
                .iter()
                .map(|sub_agent| sub_agent.clone_with(BaseAgentUpdate::default()))
                .collect::<Result<Vec<_>, _>>()?,
        };
        BaseAgent::build(
            name,
            description,
            sub_agents,
            self.0.before_agent_callback.clone(),
            self.0.after_agent_callback.clone(),
            NoopBehavior,
        )
    }

    fn canonical_before_agent_callbacks(&self) -> &[AgentCallback] {
        &self.0.before_agent_callback
    }

    fn canonical_after_agent_callbacks(&self) -> &[AgentCallback] {
        &self.0.after_agent_callback
    }

    /// C0038/C0045: runs the before-agent callback chain (plugin hooks
    /// first, then this agent's own canonical callbacks only if no plugin
    /// short-circuited), stopping at the first non-`None` result. Emits a
    /// state-delta-only event if a callback mutated state without returning
    /// content.
    async fn handle_before_agent_callback(
        &self,
        ctx: &InvocationContext,
        plugin_manager: &PluginManager,
    ) -> Option<Event> {
        let mut callback_ctx = Context::new(ctx.clone());
        let mut content = plugin_manager
            .run_before_agent_callback(self, &mut callback_ctx)
            .await;

        if content.is_none() {
            for callback in self.canonical_before_agent_callbacks() {
                content = callback(&mut callback_ctx);
                if content.is_some() {
                    break;
                }
            }
        }

        if let Some(content) = content {
            let mut event = Event::new(
                &ctx.invocation_id,
                self.name(),
                adk_events::node_info::NodeInfo::new(""),
            );
            event.branch = ctx.branch.clone();
            event.actions = callback_ctx.into_actions();
            event.content = Some(content);
            return Some(event);
        }

        if callback_ctx.state().has_delta() {
            let mut event = Event::new(
                &ctx.invocation_id,
                self.name(),
                adk_events::node_info::NodeInfo::new(""),
            );
            event.branch = ctx.branch.clone();
            event.actions = callback_ctx.into_actions();
            return Some(event);
        }

        None
    }

    /// C0038/C0045: after-agent callback chain — same shape as
    /// [`Self::handle_before_agent_callback`], but never sets
    /// `end_invocation` (that's a before-callback-only effect).
    async fn handle_after_agent_callback(
        &self,
        ctx: &InvocationContext,
        plugin_manager: &PluginManager,
    ) -> Option<Event> {
        let mut callback_ctx = Context::new(ctx.clone());
        let mut content = plugin_manager
            .run_after_agent_callback(self, &mut callback_ctx)
            .await;

        if content.is_none() {
            for callback in self.canonical_after_agent_callbacks() {
                content = callback(&mut callback_ctx);
                if content.is_some() {
                    break;
                }
            }
        }

        if content.is_some() || callback_ctx.state().has_delta() {
            let mut event = Event::new(
                &ctx.invocation_id,
                self.name(),
                adk_events::node_info::NodeInfo::new(""),
            );
            event.branch = ctx.branch.clone();
            event.actions = callback_ctx.into_actions();
            event.content = content;
            return Some(event);
        }

        None
    }

    /// C0046: notification-only error callback. This always runs every
    /// registered plugin regardless of what any of them do (C0357/C0360);
    /// the triggering error is always the caller's to re-raise/propagate,
    /// never this method's.
    async fn handle_agent_error_callback(
        &self,
        ctx: &InvocationContext,
        plugin_manager: &PluginManager,
        error: &AgentRunError,
    ) {
        let mut callback_ctx = Context::new(ctx.clone());
        plugin_manager
            .run_on_agent_error_callback(self, &mut callback_ctx, error)
            .await;
    }

    /// C0041: primary text-conversation entrypoint. Wraps
    /// `_run_async_impl` with before/after-agent callbacks,
    /// end-invocation short-circuiting, and error-callback notification —
    /// a failed `_run_async_impl` is always re-raised to the caller after
    /// the notification runs (C0046), never swallowed.
    pub async fn run_async(
        &self,
        parent_context: &InvocationContext,
    ) -> Result<Vec<Event>, AgentRunError> {
        let mut ctx = parent_context.with_agent(self.clone());
        let plugin_manager = ctx.plugin_manager.clone();
        let mut events = Vec::new();

        if let Some(event) = self
            .handle_before_agent_callback(&ctx, &plugin_manager)
            .await
        {
            events.push(event);
        }
        if ctx.end_invocation {
            return Ok(events);
        }

        match self.0.behavior.run_async_impl(&mut ctx).await {
            Ok(produced) => events.extend(produced),
            Err(error) => {
                self.handle_agent_error_callback(&ctx, &plugin_manager, &error)
                    .await;
                return Err(error);
            }
        }

        if ctx.end_invocation {
            return Ok(events);
        }

        if let Some(event) = self
            .handle_after_agent_callback(&ctx, &plugin_manager)
            .await
        {
            events.push(event);
        }

        Ok(events)
    }

    /// C0042: audio/video entrypoint. Same callback wrapping (and error
    /// re-raise) as [`Self::run_async`]; the source marks this `@final`
    /// (not overridable by subclasses) — there is no Rust equivalent to
    /// enforce that at compile time for a trait-object-backed design, so
    /// it is documented here instead.
    pub async fn run_live(
        &self,
        parent_context: &InvocationContext,
    ) -> Result<Vec<Event>, AgentRunError> {
        let mut ctx = parent_context.with_agent(self.clone());
        let plugin_manager = ctx.plugin_manager.clone();
        let mut events = Vec::new();

        if let Some(event) = self
            .handle_before_agent_callback(&ctx, &plugin_manager)
            .await
        {
            events.push(event);
        }
        if ctx.end_invocation {
            return Ok(events);
        }

        match self.0.behavior.run_live_impl(&mut ctx).await {
            Ok(produced) => events.extend(produced),
            Err(error) => {
                self.handle_agent_error_callback(&ctx, &plugin_manager, &error)
                    .await;
                return Err(error);
            }
        }

        if let Some(event) = self
            .handle_after_agent_callback(&ctx, &plugin_manager)
            .await
        {
            events.push(event);
        }

        Ok(events)
    }
}

/// Field updates for [`BaseAgent::clone_with`] (C0040).
#[derive(Default)]
pub struct BaseAgentUpdate {
    pub name: Option<String>,
    pub description: Option<String>,
    pub sub_agents: Option<Vec<BaseAgent>>,
    parent_agent_requested: bool,
}

impl BaseAgentUpdate {
    /// Marks that the caller tried to set `parent_agent` — always rejected
    /// by [`BaseAgent::clone_with`], matching the source.
    pub fn with_parent_agent_rejected() -> Self {
        Self {
            parent_agent_requested: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    #[test]
    fn valid_identifier_names_are_accepted() {
        assert!(BaseAgent::new("agent_1", NoopBehavior).is_ok());
    }

    #[test]
    fn invalid_identifier_names_are_rejected() {
        assert!(matches!(
            BaseAgent::new("123bad", NoopBehavior),
            Err(BaseAgentError::InvalidName(_))
        ));
    }

    #[test]
    fn the_name_user_is_reserved() {
        assert!(matches!(
            BaseAgent::new("user", NoopBehavior),
            Err(BaseAgentError::ReservedName)
        ));
    }

    #[test]
    fn base_agent_state_defaults_are_equal() {
        assert_eq!(BaseAgentState, BaseAgentState);
    }

    #[test]
    fn duplicate_sub_agent_names_are_allowed_not_errored() {
        let a = BaseAgent::new("dup", NoopBehavior).unwrap();
        let b = BaseAgent::new("dup", NoopBehavior).unwrap();
        // C0037: the source only warns (logs) on duplicate sub-agent names,
        // it does not raise — construction must still succeed.
        assert!(BaseAgent::build("parent", "", vec![a, b], vec![], vec![], NoopBehavior).is_ok());
    }

    #[test]
    fn adopting_a_sub_agent_sets_its_parent() {
        let child = BaseAgent::new("child", NoopBehavior).unwrap();
        let parent = BaseAgent::build(
            "parent",
            "",
            vec![child.clone()],
            vec![],
            vec![],
            NoopBehavior,
        )
        .unwrap();
        assert_eq!(child.parent_agent().unwrap().name(), "parent");
        assert_eq!(parent.sub_agents()[0].name(), "child");
    }

    #[test]
    fn adopting_the_same_agent_instance_twice_errors() {
        let child = BaseAgent::new("child", NoopBehavior).unwrap();
        let _first_parent =
            BaseAgent::build("p1", "", vec![child.clone()], vec![], vec![], NoopBehavior).unwrap();
        let second = BaseAgent::build("p2", "", vec![child], vec![], vec![], NoopBehavior);
        assert!(matches!(second, Err(BaseAgentError::AlreadyHasParent(..))));
    }

    #[test]
    fn root_agent_walks_to_the_top_of_the_tree() {
        let grandchild = BaseAgent::new("grandchild", NoopBehavior).unwrap();
        let child = BaseAgent::build(
            "child",
            "",
            vec![grandchild.clone()],
            vec![],
            vec![],
            NoopBehavior,
        )
        .unwrap();
        let _root =
            BaseAgent::build("root", "", vec![child], vec![], vec![], NoopBehavior).unwrap();
        assert_eq!(grandchild.root_agent().name(), "root");
    }

    #[test]
    fn find_agent_checks_self_before_descendants() {
        let child = BaseAgent::new("child", NoopBehavior).unwrap();
        let root = BaseAgent::build("root", "", vec![child], vec![], vec![], NoopBehavior).unwrap();
        assert_eq!(root.find_agent("root").unwrap().name(), "root");
        assert_eq!(root.find_agent("child").unwrap().name(), "child");
        assert!(root.find_agent("missing").is_none());
    }

    #[test]
    fn clone_with_deep_copies_sub_agents_with_fresh_parent_links() {
        let child = BaseAgent::new("child", NoopBehavior).unwrap();
        let root = BaseAgent::build("root", "", vec![child], vec![], vec![], NoopBehavior).unwrap();
        let cloned = root.clone_with(BaseAgentUpdate::default()).unwrap();
        assert_eq!(cloned.name(), "root");
        assert_eq!(cloned.sub_agents()[0].name(), "child");
        assert_eq!(
            cloned.sub_agents()[0].parent_agent().unwrap().name(),
            "root"
        );
        assert!(
            !Arc::ptr_eq(&cloned.0, &root.0),
            "clone must be a distinct agent instance"
        );
    }

    #[test]
    fn clone_with_rejects_setting_parent_agent() {
        let agent = BaseAgent::new("solo", NoopBehavior).unwrap();
        let result = agent.clone_with(BaseAgentUpdate::with_parent_agent_rejected());
        assert!(matches!(result, Err(BaseAgentError::ParentAgentInClone)));
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_content_skips_the_agent_run_and_ends_invocation() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let before: AgentCallback = Arc::new(move |_ctx| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Some(Content::user_text("skip"))
        });
        let agent =
            BaseAgent::build("agent", "", vec![], vec![before], vec![], NoopBehavior).unwrap();
        let events = agent.run_async(&ctx()).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].content, Some(Content::user_text("skip")));
    }

    #[rusty_tokio::test]
    async fn callback_chain_stops_at_first_non_none_result() {
        let first: AgentCallback = Arc::new(|_ctx| None);
        let second: AgentCallback = Arc::new(|_ctx| Some(Content::user_text("second")));
        let third_ran = Arc::new(AtomicUsize::new(0));
        let third_ran_clone = third_ran.clone();
        let third: AgentCallback = Arc::new(move |_ctx| {
            third_ran_clone.fetch_add(1, Ordering::SeqCst);
            Some(Content::user_text("third"))
        });
        let agent = BaseAgent::build(
            "agent",
            "",
            vec![],
            vec![first, second, third],
            vec![],
            NoopBehavior,
        )
        .unwrap();
        let events = agent.run_async(&ctx()).await.unwrap();
        assert_eq!(
            third_ran.load(Ordering::SeqCst),
            0,
            "chain must stop at `second`"
        );
        assert_eq!(events[0].content, Some(Content::user_text("second")));
    }

    #[rusty_tokio::test]
    async fn after_agent_callback_content_is_appended_as_an_extra_event() {
        let after: AgentCallback = Arc::new(|_ctx| Some(Content::user_text("done")));
        let agent =
            BaseAgent::build("agent", "", vec![], vec![], vec![after], NoopBehavior).unwrap();
        let events = agent.run_async(&ctx()).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].content, Some(Content::user_text("done")));
    }

    #[rusty_tokio::test]
    async fn no_callbacks_and_a_noop_impl_yields_no_events() {
        let agent = BaseAgent::new("agent", NoopBehavior).unwrap();
        let events = agent.run_async(&ctx()).await.unwrap();
        assert!(events.is_empty());
    }

    #[derive(Debug)]
    struct BoomError;
    impl std::fmt::Display for BoomError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "boom")
        }
    }
    impl std::error::Error for BoomError {}

    struct FailingBehavior;
    impl AgentBehavior for FailingBehavior {
        fn run_async_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
            Box::pin(async { Err(Box::new(BoomError) as AgentRunError) })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
            Box::pin(async { Err(Box::new(BoomError) as AgentRunError) })
        }
    }

    /// C0041/C0046: a failing `_run_async_impl` is always re-raised to the
    /// caller (never swallowed) after the error callback is notified.
    #[rusty_tokio::test]
    async fn run_async_re_raises_the_behavior_error() {
        let agent = BaseAgent::new("agent", FailingBehavior).unwrap();
        let err = agent.run_async(&ctx()).await.unwrap_err();
        assert_eq!(err.to_string(), "boom");
    }

    #[rusty_tokio::test]
    async fn run_live_re_raises_the_behavior_error() {
        let agent = BaseAgent::new("agent", FailingBehavior).unwrap();
        let err = agent.run_live(&ctx()).await.unwrap_err();
        assert_eq!(err.to_string(), "boom");
    }

    struct ShortCircuitingPlugin;

    impl crate::services::BasePlugin for ShortCircuitingPlugin {
        fn name(&self) -> &str {
            "short_circuiting_plugin"
        }

        fn before_agent_callback<'a>(
            &'a self,
            _agent: &'a BaseAgent,
            _callback_context: &'a mut Context,
        ) -> crate::services::BoxFuture<'a, Option<Content>> {
            Box::pin(async { Some(Content::user_text("short-circuited by a plugin")) })
        }
    }

    /// Proves the fix disclosed in this module's doc: `run_async` reads
    /// the real, configured `PluginManager` off the built
    /// `InvocationContext` rather than constructing a fresh, always-empty
    /// one — a registered plugin's `before_agent_callback` must actually
    /// run and be able to short-circuit the agent.
    #[rusty_tokio::test]
    async fn run_async_honors_a_plugin_registered_on_the_invocation_context() {
        let mut plugin_manager = crate::services::PluginManager::new();
        plugin_manager
            .register_plugin(std::sync::Arc::new(ShortCircuitingPlugin))
            .unwrap();
        let parent_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
                .plugin_manager(plugin_manager)
                .build();

        let agent = BaseAgent::new("agent", NoopBehavior).unwrap();
        let events = agent.run_async(&parent_context).await.unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].content,
            Some(Content::user_text("short-circuited by a plugin"))
        );
    }
}
