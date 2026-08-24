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
//! **C0071 now complete**: `get_events`/`should_pause_invocation`/
//! `find_matching_function_call`/`stamp_event_branch_context` all needed
//! `Event::get_function_calls`/`get_function_responses`, blocked until
//! Phase 3 gave `Event.content` a real `Content`/`Part` structure to
//! inspect — landed alongside this batch, so all four are implemented and
//! tested here. `find_event_by_function_call_id` is pulled forward from
//! the source's `flows.llm_flows.functions` (Phase 4 isn't built) since
//! `find_matching_function_call` needs it and it doesn't depend on
//! anything else in that module.
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
    ArtifactService, AuthCredential, BoxFuture, CredentialService, MemoryService, PluginManager,
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

    /// C0071: events from the current session, optionally filtered by
    /// invocation and/or branch.
    pub fn get_events(&self, current_invocation: bool, current_branch: bool) -> Vec<Event> {
        let mut results: Vec<Event> = self.session.events.clone();
        if current_invocation {
            results.retain(|event| event.invocation_id == self.invocation_id);
        }
        if current_branch {
            results.retain(|event| self.is_branch_match(event));
        }
        results
    }

    /// The predicate behind `get_events(current_branch=True)`: for a
    /// user-authored event carrying function responses, first checks that
    /// at least one response id matches a function-call id issued on this
    /// branch or a descendant sub-branch (guarding against event leakage
    /// across parallel/unrelated branches); then applies the ordinary
    /// direct/descendant-branch prefix match.
    fn is_branch_match(&self, event: &Event) -> bool {
        if event.author == "user" {
            let frs = event.get_function_responses();
            if !frs.is_empty() {
                if let Some(branch) = &self.branch {
                    let fr_ids: std::collections::HashSet<&str> =
                        frs.iter().filter_map(|fr| fr.id.as_deref()).collect();
                    if !fr_ids.is_empty() {
                        let branch_fc_ids: std::collections::HashSet<&str> = self
                            .session
                            .events
                            .iter()
                            .filter(|e| {
                                e.branch.as_deref().is_some_and(|b| {
                                    b == branch || b.starts_with(&format!("{branch}."))
                                })
                            })
                            .flat_map(|e| e.get_function_calls())
                            .filter_map(|fc| fc.id.as_deref())
                            .collect();
                        if fr_ids.is_disjoint(&branch_fc_ids) {
                            return false;
                        }
                    }
                }
            }
            match (&event.branch, &self.branch) {
                (None, _) | (_, None) => true,
                (Some(event_branch), Some(branch)) => {
                    event_branch == branch || event_branch.starts_with(&format!("{branch}."))
                }
            }
        } else {
            event.branch == self.branch
        }
    }

    /// C0071: whether to pause the invocation right after `event` —
    /// true iff `event` carries an unresolved long-running function call
    /// that isn't already being answered by a nested sub-branch.
    pub fn should_pause_invocation(&self, event: &Event) -> bool {
        let long_running_ids = match &event.long_running_tool_ids {
            Some(ids) if !ids.is_empty() => ids,
            _ => return false,
        };
        let function_calls = event.get_function_calls();
        if function_calls.is_empty() {
            return false;
        }

        let events = &self.session.events;
        for fc in &function_calls {
            let Some(fc_id) = fc.id.as_deref() else {
                continue;
            };
            if !long_running_ids.iter().any(|id| id == fc_id) {
                continue;
            }
            let event_index = events.iter().position(|e| e.id == event.id);
            let is_resolving_sub_branch = event_index.is_some_and(|index| {
                events[index + 1..].iter().any(|e| {
                    e.author == "user"
                        && e.branch
                            .as_ref()
                            .map(|b| branch_path_run_ids(b).contains(&fc_id.to_string()))
                            .unwrap_or(false)
                })
            });
            if !is_resolving_sub_branch {
                return true;
            }
        }
        false
    }

    /// C0071: finds the function-call event in the current invocation that
    /// matches `function_response_event`'s function response id.
    pub fn find_matching_function_call(&self, function_response_event: &Event) -> Option<Event> {
        let function_responses = function_response_event.get_function_responses();
        if function_responses.is_empty() {
            return None;
        }
        let events = self.get_events(true, false);
        let search_space: &[Event] = if events
            .last()
            .is_some_and(|last| last.id == function_response_event.id)
        {
            &events[..events.len() - 1]
        } else {
            &events[..]
        };
        let function_response_id = function_responses[0].id.as_deref()?;
        find_event_by_function_call_id(search_space, function_response_id)
    }

    /// C0071: stamps `event` with the branch (and, if unset, isolation
    /// scope) of its matching function-call event.
    pub fn stamp_event_branch_context(&self, event: &mut Event) {
        if let Some(function_call_event) = self.find_matching_function_call(event) {
            event.branch = function_call_event.branch.clone();
            if event.isolation_scope.is_none() && function_call_event.isolation_scope.is_some() {
                event.isolation_scope = function_call_event.isolation_scope.clone();
            }
        }
    }
}

/// Shared by [`InvocationContext::should_pause_invocation`] — the
/// `_BranchPath`-tagged run ids embedded in a dot-separated branch string
/// (mirrors `_BranchPath.from_string(branch).run_ids`).
fn branch_path_run_ids(branch: &str) -> Vec<String> {
    adk_events::branch_path::BranchPath::from_string(branch)
        .run_ids()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// C0071 helper (the source's `flows.llm_flows.functions.find_event_by_function_call_id`,
/// pulled forward since `InvocationContext::find_matching_function_call`
/// needs it and the full `flows/` module is Phase 4): finds the function
/// call event matching `function_call_id`, searching backward.
fn find_event_by_function_call_id(events: &[Event], function_call_id: &str) -> Option<Event> {
    events.iter().rev().find_map(|event| {
        event
            .get_function_calls()
            .iter()
            .any(|fc| fc.id.as_deref() == Some(function_call_id))
            .then(|| event.clone())
    })
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
impl SessionService for NoopSessionService {
    fn create_session<'a>(
        &'a self,
        app_name: &'a str,
        user_id: &'a str,
        state: Option<BTreeMap<String, Value>>,
        session_id: Option<String>,
    ) -> BoxFuture<'a, Result<Session, adk_errors::already_exists::AlreadyExistsError>> {
        Box::pin(async move {
            Ok(Session {
                id: session_id.unwrap_or_default(),
                app_name: app_name.to_string(),
                user_id: user_id.to_string(),
                state: state.unwrap_or_default(),
                events: Vec::new(),
            })
        })
    }

    fn get_session<'a>(
        &'a self,
        _app_name: &'a str,
        _user_id: &'a str,
        _session_id: &'a str,
    ) -> BoxFuture<'a, Option<Session>> {
        Box::pin(async { None })
    }

    fn list_sessions<'a>(
        &'a self,
        _app_name: &'a str,
        _user_id: Option<&'a str>,
    ) -> BoxFuture<'a, Vec<Session>> {
        Box::pin(async { Vec::new() })
    }

    fn delete_session<'a>(
        &'a self,
        _app_name: &'a str,
        _user_id: &'a str,
        _session_id: &'a str,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }
}

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

    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{Content, FunctionCall, FunctionResponse, Part};

    fn fc_event(invocation_id: &str, id: &str) -> Event {
        let mut event = Event::new(invocation_id, "agent", NodeInfo::new("root"));
        event.content = Some(Content::new(
            "model",
            vec![Part::function_call(FunctionCall {
                id: Some(id.to_string()),
                name: Some("tool".to_string()),
                ..Default::default()
            })],
        ));
        event
    }

    fn fr_event(invocation_id: &str, id: &str) -> Event {
        let mut event = Event::new(invocation_id, "user", NodeInfo::new("root"));
        event.content = Some(Content::new(
            "user",
            vec![Part::function_response(FunctionResponse {
                id: Some(id.to_string()),
                name: Some("tool".to_string()),
                ..Default::default()
            })],
        ));
        event
    }

    #[test]
    fn should_pause_invocation_is_true_for_an_unresolved_long_running_call() {
        let mut event = fc_event("inv-1", "fc-1");
        event.set_long_running_tool_ids(["fc-1"]);
        let ic = ctx();
        assert!(ic.should_pause_invocation(&event));
    }

    #[test]
    fn should_pause_invocation_is_false_without_a_matching_long_running_id() {
        let mut event = fc_event("inv-1", "fc-1");
        event.set_long_running_tool_ids(["some-other-id"]);
        let ic = ctx();
        assert!(!ic.should_pause_invocation(&event));
    }

    #[test]
    fn should_pause_invocation_is_false_once_a_sub_branch_resolves_it() {
        let mut event = fc_event("inv-1", "fc-1");
        event.set_long_running_tool_ids(["fc-1"]);

        let mut resolving_user_event = Event::new("inv-1", "user", NodeInfo::new("root"));
        resolving_user_event.branch = Some("root@fc-1".to_string());

        let mut session = Session::new("app", "user", "s1");
        session.events.push(event.clone());
        session.events.push(resolving_user_event);
        let ic = InvocationContextBuilder::new("inv-1", session).build();

        assert!(!ic.should_pause_invocation(&event));
    }

    #[test]
    fn find_matching_function_call_locates_the_originating_call() {
        let call = fc_event("inv-1", "fc-1");
        let response = fr_event("inv-1", "fc-1");
        let mut session = Session::new("app", "user", "s1");
        session.events.push(call.clone());
        session.events.push(response.clone());
        let ic = InvocationContextBuilder::new("inv-1", session).build();

        let found = ic.find_matching_function_call(&response).unwrap();
        assert_eq!(found.id, call.id);
    }

    #[test]
    fn find_matching_function_call_returns_none_without_a_response() {
        let ic = ctx();
        let plain_event = Event::new("inv-1", "user", NodeInfo::new("root"));
        assert!(ic.find_matching_function_call(&plain_event).is_none());
    }

    #[test]
    fn stamp_event_branch_context_copies_branch_from_the_matching_call() {
        let mut call = fc_event("inv-1", "fc-1");
        call.branch = Some("root.worker".to_string());
        let mut response = fr_event("inv-1", "fc-1");

        let mut session = Session::new("app", "user", "s1");
        session.events.push(call);
        session.events.push(response.clone());
        let ic = InvocationContextBuilder::new("inv-1", session).build();

        ic.stamp_event_branch_context(&mut response);
        assert_eq!(response.branch.as_deref(), Some("root.worker"));
    }

    #[test]
    fn get_events_branch_filter_excludes_a_function_response_for_an_unrelated_branch() {
        // A function call issued on branch "a", and a function response
        // event whose only matching call was never issued anywhere in "a"'s
        // branch tree — the response must not leak into "a"'s event view.
        let mut call = fc_event("inv-1", "fc-1");
        call.branch = Some("a".to_string());
        let mut response = fr_event("inv-1", "fc-1");
        response.branch = Some("a".to_string());

        let mut unrelated_response = fr_event("inv-1", "fc-2");
        unrelated_response.branch = Some("a".to_string());
        let unrelated_response_id = unrelated_response.id.clone();

        let mut session = Session::new("app", "user", "s1");
        session.events.push(call);
        session.events.push(response.clone());
        session.events.push(unrelated_response);
        let ic = InvocationContextBuilder::new("inv-1", session)
            .branch("a")
            .build();

        let events = ic.get_events(false, true);
        let ids: Vec<&str> = events.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&response.id.as_str()));
        assert!(
            !ids.contains(&unrelated_response_id.as_str()),
            "the response to a call never issued on this branch tree must be filtered out"
        );
    }
}
