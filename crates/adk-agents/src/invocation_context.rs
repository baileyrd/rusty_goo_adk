//! Capabilities C0066-C0071: `InvocationContext`, ported from
//! `google.adk.agents.invocation_context`.
//!
//! **Scope note**: this batch omits `live_request_queue`,
//! `active_streaming_tools`, `active_non_blocking_tool_tasks`,
//! `transcription_cache`, and the input/output realtime audio caches —
//! live-mode fields whose sharing semantics (Python's `model_copy` shares
//! the same live object across context copies) need a concrete live-mode
//! consumer to shape correctly. `LiveRequestQueue` itself is fully
//! implemented and tested (`live_request.rs`); only its wiring into this
//! struct is deferred. C0066 is left `REQUIRED`, not `DONE`, for this
//! reason — flagged, not silently dropped.
//!
//! **Adaptation**: `resumability_config`/`events_compaction_config` are
//! `apps._configs` types (Phase 7). `events_compaction_config` is an opaque
//! placeholder (nothing in this batch reads its fields); `resumability_config`
//! is narrowed to just the one boolean field (`is_resumable`) this batch's
//! logic (`populate_invocation_agent_states`) actually needs — see
//! [`ResumabilityConfigStub`]'s doc.
//!
//! **Deferred** (blocked on Phase 3's real `Content`/`Part` — `Event`
//! currently types those as opaque `Value` placeholders, so
//! `get_function_calls`/`get_function_responses` don't exist yet):
//! `should_pause_invocation`, `_find_matching_function_call`,
//! `stamp_event_branch_context` (part of C0071). `_get_events` (also part of
//! C0071) needs none of that and is implemented.
//!
//! **Deferred**: `_enqueue_event` (C0069) needs the `_event_queue` this
//! batch omits along with the other live-mode fields (see the scope note
//! above) — its consumer (the `Runner` main loop, C0833-C0926) isn't built
//! yet either, so there's nothing to exercise it against.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use adk_events::Event;
use rusty_serde::value::Value;

use crate::base_agent::BaseAgent;
use crate::context_cache_config::ContextCacheConfig;
use crate::run_config::RunConfig;
use crate::services::{
    ArtifactService, AuthCredential, CredentialService, MemoryService, PluginManager,
    SessionService,
};
use crate::session::Session;

/// Narrowed placeholder for `apps._configs.ResumabilityConfig` (Phase 7) —
/// see the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResumabilityConfigStub {
    pub is_resumable: bool,
}

#[derive(Debug, rusty_err::Error)]
pub enum InvocationContextError {
    #[error("Max number of llm calls limit of `{0}` exceeded")]
    LlmCallsLimitExceeded(i64),
}

#[derive(Debug, Default)]
struct InvocationCostManager {
    number_of_llm_calls: i64,
}

impl InvocationCostManager {
    fn increment_and_enforce_llm_calls_limit(
        &mut self,
        run_config: Option<&RunConfig>,
    ) -> Result<(), InvocationContextError> {
        self.number_of_llm_calls += 1;
        if let Some(run_config) = run_config {
            if run_config.max_llm_calls > 0 && self.number_of_llm_calls > run_config.max_llm_calls {
                return Err(InvocationContextError::LlmCallsLimitExceeded(
                    run_config.max_llm_calls,
                ));
            }
        }
        Ok(())
    }
}

/// Capability C0068: caches an audio data chunk before it's flushed to the
/// session/artifact service. `data` is an opaque `google.genai.types.Blob`
/// placeholder (see `run_config`'s module doc for the rationale).
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeCacheEntry {
    pub role: String,
    pub data: Value,
    pub timestamp: f64,
}

#[derive(Clone)]
pub struct InvocationContext {
    pub artifact_service: Option<Arc<dyn ArtifactService + Send + Sync>>,
    pub session_service: Arc<dyn SessionService + Send + Sync>,
    pub memory_service: Option<Arc<dyn MemoryService + Send + Sync>>,
    pub credential_service: Option<Arc<dyn CredentialService + Send + Sync>>,
    pub context_cache_config: Option<ContextCacheConfig>,

    pub invocation_id: String,
    pub branch: Option<String>,
    pub isolation_scope: Option<String>,
    pub agent: Option<BaseAgent>,
    pub user_content: Option<Value>,
    pub session: Session,
    pub node_path: Option<String>,

    pub agent_states: BTreeMap<String, HashMap<String, Value>>,
    pub end_of_agents: BTreeMap<String, bool>,
    pub end_invocation: bool,

    pub run_config: Option<RunConfig>,
    pub resumability_config: Option<ResumabilityConfigStub>,
    pub events_compaction_config: Option<Value>,
    pub token_compaction_checked: bool,

    pub plugin_manager: PluginManager,
    pub canonical_tools_cache: Option<Vec<Value>>,
    pub credential_by_key: BTreeMap<String, AuthCredential>,
    pub custom_metadata: BTreeMap<String, Value>,

    invocation_cost_manager: Arc<Mutex<InvocationCostManager>>,
}

impl InvocationContext {
    /// C0067: tracks and enforces `run_config.max_llm_calls` (only if
    /// positive) for this invocation. Shared via `Arc<Mutex<_>>` across every
    /// per-agent-run copy of this context so the count is per-*invocation*,
    /// not per-agent-run — mirroring the source's `PrivateAttr` shared
    /// across `model_copy`.
    pub fn increment_llm_call_count(&self) -> Result<(), InvocationContextError> {
        self.invocation_cost_manager
            .lock()
            .expect("invocation cost manager mutex poisoned")
            .increment_and_enforce_llm_calls_limit(self.run_config.as_ref())
    }

    pub fn is_resumable(&self) -> bool {
        self.resumability_config
            .map(|config| config.is_resumable)
            .unwrap_or(false)
    }

    /// Builds a copy of this context with `agent` set — the source's
    /// `_create_invocation_context` (`model_copy(update={'agent': self})`).
    pub fn with_agent(&self, agent: BaseAgent) -> Self {
        let mut clone = self.clone();
        clone.agent = Some(agent);
        clone
    }

    /// C0070: sets/clears/resets an agent's resumable state.
    pub fn set_agent_state(
        &mut self,
        agent_name: &str,
        agent_state: Option<HashMap<String, Value>>,
        end_of_agent: bool,
    ) {
        if end_of_agent {
            self.end_of_agents.insert(agent_name.to_string(), true);
            self.agent_states.remove(agent_name);
        } else if let Some(agent_state) = agent_state {
            self.agent_states
                .insert(agent_name.to_string(), agent_state);
            self.end_of_agents.insert(agent_name.to_string(), false);
        } else {
            self.end_of_agents.remove(agent_name);
            self.agent_states.remove(agent_name);
        }
    }

    /// C0070: resets the state of every sub-agent of the given agent.
    pub fn reset_sub_agent_states(&mut self, agent_name: &str) {
        let Some(root) = self.agent.clone() else {
            return;
        };
        let Some(agent) = root.find_agent(agent_name) else {
            return;
        };
        for sub_agent in agent.sub_agents().to_vec() {
            self.set_agent_state(sub_agent.name(), None, false);
            self.reset_sub_agent_states(sub_agent.name());
        }
    }

    /// C0070: rebuilds agent states from history for a resumable session.
    pub fn populate_invocation_agent_states(&mut self) {
        if !self.is_resumable() {
            return;
        }
        let events = self.get_events(true, false);
        for event in events {
            let key = if event.node_info.path.is_empty() {
                event.author.clone()
            } else {
                event.node_info.path.clone()
            };
            if event.actions.end_of_agent {
                self.end_of_agents.insert(key.clone(), true);
                self.agent_states.remove(&key);
            } else if let Some(agent_state) = event.actions.agent_state.clone() {
                self.agent_states.insert(key.clone(), agent_state);
                self.end_of_agents.insert(key, false);
            } else if event.author != "user"
                && event.content.is_some()
                && !self.agent_states.contains_key(&key)
            {
                self.agent_states.insert(key.clone(), HashMap::new());
                self.end_of_agents.insert(key, false);
            }
        }
    }

    /// C0071 (partial): events from the current session, optionally
    /// filtered by invocation and/or branch. Branch matching here covers
    /// only the direct/descendant-branch prefix check — the source's
    /// function-response/function-call cross-branch leak guard needs
    /// `Event::get_function_calls`/`get_function_responses`, which don't
    /// exist until Phase 3's `Content`/`Part` land.
    pub fn get_events(&self, current_invocation: bool, current_branch: bool) -> Vec<Event> {
        let mut results: Vec<Event> = self.session.events.clone();
        if current_invocation {
            results.retain(|event| event.invocation_id == self.invocation_id);
        }
        if current_branch {
            results.retain(|event| match (&event.branch, &self.branch) {
                (None, _) | (_, None) => event.branch == self.branch,
                (Some(event_branch), Some(branch)) => {
                    event_branch == branch || event_branch.starts_with(&format!("{branch}."))
                }
            });
        }
        results
    }
}

/// Builds an [`InvocationContext`] for tests and call sites that don't need
/// every field customized — the source constructs `InvocationContext`
/// directly via its pydantic constructor; this mirrors that with sensible
/// defaults (a no-op `SessionService`, no optional services).
pub struct InvocationContextBuilder {
    invocation_id: String,
    session: Session,
    agent: Option<BaseAgent>,
    run_config: Option<RunConfig>,
    branch: Option<String>,
}

struct NoopSessionService;
impl SessionService for NoopSessionService {}

impl InvocationContextBuilder {
    pub fn new(invocation_id: impl Into<String>, session: Session) -> Self {
        Self {
            invocation_id: invocation_id.into(),
            session,
            agent: None,
            run_config: None,
            branch: None,
        }
    }

    pub fn agent(mut self, agent: BaseAgent) -> Self {
        self.agent = Some(agent);
        self
    }

    pub fn run_config(mut self, run_config: RunConfig) -> Self {
        self.run_config = Some(run_config);
        self
    }

    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    pub fn build(self) -> InvocationContext {
        InvocationContext {
            artifact_service: None,
            session_service: Arc::new(NoopSessionService),
            memory_service: None,
            credential_service: None,
            context_cache_config: None,
            invocation_id: self.invocation_id,
            branch: self.branch,
            isolation_scope: None,
            agent: self.agent,
            user_content: None,
            session: self.session,
            node_path: None,
            agent_states: BTreeMap::new(),
            end_of_agents: BTreeMap::new(),
            end_invocation: false,
            run_config: self.run_config,
            resumability_config: None,
            events_compaction_config: None,
            token_compaction_checked: false,
            plugin_manager: PluginManager,
            canonical_tools_cache: None,
            credential_by_key: BTreeMap::new(),
            custom_metadata: BTreeMap::new(),
            invocation_cost_manager: Arc::new(Mutex::new(InvocationCostManager::default())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    #[test]
    fn increment_llm_call_count_is_unbounded_without_a_run_config() {
        let ic = ctx();
        for _ in 0..10 {
            ic.increment_llm_call_count().unwrap();
        }
    }

    #[test]
    fn increment_llm_call_count_enforces_a_positive_max() {
        let run_config = RunConfig {
            max_llm_calls: 2,
            ..Default::default()
        };
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .run_config(run_config)
            .build();
        ic.increment_llm_call_count().unwrap();
        ic.increment_llm_call_count().unwrap();
        let err = ic.increment_llm_call_count().unwrap_err();
        assert!(matches!(
            err,
            InvocationContextError::LlmCallsLimitExceeded(2)
        ));
    }

    #[test]
    fn increment_llm_call_count_is_shared_across_with_agent_copies() {
        let ic = ctx();
        let agent = BaseAgent::new("a", crate::base_agent::NoopBehavior).unwrap();
        let child = ic.with_agent(agent);
        ic.increment_llm_call_count().unwrap();
        child.increment_llm_call_count().unwrap();
        assert_eq!(
            ic.invocation_cost_manager
                .lock()
                .unwrap()
                .number_of_llm_calls,
            2,
            "the cost manager must be shared, not duplicated, across context copies"
        );
    }

    fn one_field_state() -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert("k".to_string(), Value::Int(1));
        map
    }

    #[test]
    fn set_agent_state_end_of_agent_clears_state_and_marks_done() {
        let mut ic = ctx();
        ic.set_agent_state("a", Some(one_field_state()), false);
        ic.set_agent_state("a", None, true);
        assert_eq!(ic.end_of_agents.get("a"), Some(&true));
        assert!(!ic.agent_states.contains_key("a"));
    }

    #[test]
    fn set_agent_state_with_no_state_clears_both_maps() {
        let mut ic = ctx();
        ic.set_agent_state("a", Some(one_field_state()), false);
        ic.set_agent_state("a", None, false);
        assert!(!ic.agent_states.contains_key("a"));
        assert!(!ic.end_of_agents.contains_key("a"));
    }

    #[test]
    fn get_events_filters_by_invocation_id() {
        let mut session = Session::new("app", "user", "s1");
        let node_info = adk_events::node_info::NodeInfo::new("root");
        session
            .events
            .push(Event::new("inv-1", "user", node_info.clone()));
        session.events.push(Event::new("inv-2", "user", node_info));
        let ic = InvocationContextBuilder::new("inv-1", session).build();
        let events = ic.get_events(true, false);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].invocation_id, "inv-1");
    }

    #[test]
    fn get_events_filters_descendant_branches() {
        let mut session = Session::new("app", "user", "s1");
        let node_info = adk_events::node_info::NodeInfo::new("root");
        let mut on_branch = Event::new("inv-1", "user", node_info.clone());
        on_branch.branch = Some("a.b".to_string());
        let mut off_branch = Event::new("inv-1", "user", node_info);
        off_branch.branch = Some("z".to_string());
        session.events.push(on_branch);
        session.events.push(off_branch);
        let ic = InvocationContextBuilder::new("inv-1", session)
            .branch("a")
            .build();
        let events = ic.get_events(false, true);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].branch.as_deref(), Some("a.b"));
    }

    #[test]
    fn populate_invocation_agent_states_is_a_noop_when_not_resumable() {
        let mut ic = ctx();
        ic.populate_invocation_agent_states();
        assert!(ic.agent_states.is_empty());
    }

    #[test]
    fn populate_invocation_agent_states_rebuilds_from_history() {
        let mut session = Session::new("app", "user", "s1");
        let node_info = adk_events::node_info::NodeInfo::new("planner");
        let mut event = Event::new("inv-1", "planner", node_info);
        event.actions.agent_state = Some(one_field_state());
        session.events.push(event);
        let mut ic = InvocationContextBuilder::new("inv-1", session).build();
        ic.resumability_config = Some(ResumabilityConfigStub { is_resumable: true });
        ic.populate_invocation_agent_states();
        assert_eq!(ic.agent_states.get("planner"), Some(&one_field_state()));
        assert_eq!(ic.end_of_agents.get("planner"), Some(&false));
    }

    #[test]
    fn reset_sub_agent_states_clears_every_descendant() {
        let child =
            crate::base_agent::BaseAgent::new("child", crate::base_agent::NoopBehavior).unwrap();
        let root = crate::base_agent::BaseAgent::build(
            "root",
            "",
            vec![child],
            vec![],
            vec![],
            crate::base_agent::NoopBehavior,
        )
        .unwrap();
        let mut ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(root)
            .build();
        ic.set_agent_state("child", Some(one_field_state()), false);
        assert!(ic.agent_states.contains_key("child"));
        ic.reset_sub_agent_states("root");
        assert!(!ic.agent_states.contains_key("child"));
    }

    #[test]
    fn realtime_cache_entry_carries_role_data_and_timestamp() {
        let entry = RealtimeCacheEntry {
            role: "user".to_string(),
            data: Value::String("chunk".to_string()),
            timestamp: 1.5,
        };
        assert_eq!(entry.role, "user");
        assert_eq!(entry.timestamp, 1.5);
    }
}
