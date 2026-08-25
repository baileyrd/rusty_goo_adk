//! Capability C0322: `ReplayManager`, ported from
//! `google.adk.workflow.utils._replay_manager`. Part of the P7
//! workflow/graph engine — see `workflow_rehydration_utils.rs`'s module
//! doc for why this batch (P7 Chunk 4) has no caller yet and is still a
//! legitimate, independently-testable batch.
//!
//! Unifies event rehydration ([`crate::workflow_rehydration_utils`]),
//! replay interception, and sequence-barrier synchronization
//! ([`crate::workflow_replay_sequence_barrier::ReplaySequenceBarrier`])
//! across static and (once built) dynamic nodes. Every method here is
//! directly testable against a constructed [`Context`]/session-events
//! fixture, with no `Workflow` orchestrator required.
//!
//! **Identity-based indexing, faithful**: the source deliberately
//! indexes/merges events by Python object identity (`id(e)`), not
//! `Event.__eq__`, to keep `get_events_for_rehydration` O(1)-ish rather
//! than quadratic in session size (its own comment: `e not in
//! node_events` would "invoke `Event.__eq__` per probe... making this
//! loop quadratic"). This port reproduces that with `Event::id` (this
//! port's UUID-stamped identity field, `Event::new_id`) as the
//! membership key instead of Rust reference identity — a `HashSet<
//! &str>`/`HashSet<String>` keyed on each event's own `id` field is the
//! direct, safe equivalent of Python's `id(e)` (object identity) here,
//! not a narrowing: this port's own `Event::id` is generated fresh per
//! event specifically to serve as a stable identity, per `event.rs`'s
//! own C0018 doc.

use std::collections::{BTreeMap, HashMap};

use adk_events::node_path_builder::NodePathBuilder;
use adk_events::Event;

use crate::context::Context;
use crate::workflow_hitl_utils::get_request_input_interrupt_ids;
use crate::workflow_rehydration_utils::{
    direct_child_toward, is_terminal_event, reconstruct_node_states, ChildScanState,
};
use crate::workflow_replay_sequence_barrier::ReplaySequenceBarrier;

/// `ReplayManager`: unifies rehydration, replay interception, and
/// sequence-barrier synchronization across static and dynamic nodes.
#[derive(Default)]
pub struct ReplayManager {
    recovered_executions: BTreeMap<String, ChildScanState>,
    sequence_barrier: Option<ReplaySequenceBarrier>,
    parent_sequence_barriers: HashMap<String, ReplaySequenceBarrier>,
    events_by_parent: HashMap<String, Vec<Event>>,
    transitive_events_by_parent: HashMap<String, Vec<Event>>,
    indexed_event_count: Option<usize>,
}

impl ReplayManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn recovered_executions(&self) -> &BTreeMap<String, ChildScanState> {
        &self.recovered_executions
    }

    pub fn sequence_barrier(&self) -> Option<&ReplaySequenceBarrier> {
        self.sequence_barrier.as_ref()
    }

    /// `ReplayManager._ensure_index`: ensures event indexes are
    /// initialized and up-to-date with the current session. In
    /// multi-turn sessions new events accrue on each turn, so the index
    /// is rebuilt whenever the event count changes.
    fn ensure_index(&mut self, ctx: &Context) {
        let events = &ctx.invocation_context().session.events;
        if self.indexed_event_count != Some(events.len()) {
            self.build_event_index(events, &ctx.invocation_context().invocation_id);
            self.indexed_event_count = Some(events.len());
        }
    }

    /// `ReplayManager._build_event_index`: builds the index of events
    /// grouped by parent path (both direct and transitive). The index
    /// intentionally spans every invocation in the session so
    /// multi-turn context stays visible during rehydration — a caller
    /// needing a single invocation filters by `invocation_id` itself.
    fn build_event_index(&mut self, events: &[Event], _invocation_id: &str) {
        self.events_by_parent.clear();
        self.transitive_events_by_parent.clear();
        let mut fc_to_parent: HashMap<String, String> = HashMap::new();

        for event in events {
            if event.author == "user" {
                self.index_user_event(event, &mut fc_to_parent);
                continue;
            }

            let path = event.node_info.path.as_str();
            if path.is_empty() {
                continue;
            }

            let path_builder = NodePathBuilder::from_string(path);
            let parent_path = path_builder
                .parent()
                .map(|p| p.to_slash_string())
                .unwrap_or_default();

            self.add_event_to_index(&parent_path, event.clone());

            let mut interrupt_ids: Vec<String> =
                event.long_running_tool_ids.clone().unwrap_or_default();
            interrupt_ids.extend(get_request_input_interrupt_ids(event));
            for fid in interrupt_ids {
                fc_to_parent.insert(fid, parent_path.clone());
            }
        }
    }

    /// `ReplayManager._index_user_event`: routes a user response event
    /// to its parent path based on the function-call ids it responds
    /// to; a general user prompt with no matching call indexes under
    /// the root (`""`).
    fn index_user_event(&mut self, event: &Event, fc_to_parent: &mut HashMap<String, String>) {
        let Some(content) = &event.content else {
            return;
        };
        let mut matched = false;
        let mut added_parents: std::collections::HashSet<String> = std::collections::HashSet::new();
        for part in &content.parts {
            let Some(fr) = &part.function_response else {
                continue;
            };
            let Some(fr_id) = &fr.id else { continue };
            let Some(parent) = fc_to_parent.get(fr_id).cloned() else {
                continue;
            };
            if added_parents.insert(parent.clone()) {
                self.add_event_to_index(&parent, event.clone());
                matched = true;
            }
        }

        if !matched {
            self.events_by_parent
                .entry(String::new())
                .or_default()
                .push(event.clone());
            self.transitive_events_by_parent
                .entry(String::new())
                .or_default()
                .push(event.clone());
        }
    }

    /// `ReplayManager._add_event_to_index`: indexes an event under its
    /// direct parent and every ancestor path up to root.
    fn add_event_to_index(&mut self, parent_path: &str, event: Event) {
        self.events_by_parent
            .entry(parent_path.to_string())
            .or_default()
            .push(event.clone());

        let mut curr = if parent_path.is_empty() {
            None
        } else {
            Some(NodePathBuilder::from_string(parent_path))
        };
        while let Some(builder) = curr {
            let path_str = builder.to_slash_string();
            if path_str.is_empty() {
                break;
            }
            self.transitive_events_by_parent
                .entry(path_str)
                .or_default()
                .push(event.clone());
            curr = builder.parent();
        }

        self.transitive_events_by_parent
            .entry(String::new())
            .or_default()
            .push(event);
    }

    /// `ReplayManager.get_events_for_rehydration`: retrieves
    /// pre-filtered session events relevant to rehydrating `node_path`,
    /// querying the pre-indexed transitive events under the node's
    /// parent path (rather than an O(N) linear scan), with top-level
    /// user prompts merged back in so multi-turn context stays visible,
    /// preserving strict session-chronological ordering.
    pub fn get_events_for_rehydration(&mut self, ctx: &Context, node_path: &str) -> Vec<Event> {
        if node_path.is_empty() {
            return Vec::new();
        }

        self.ensure_index(ctx);
        let path_builder = NodePathBuilder::from_string(node_path);
        let Some(parent_builder) = path_builder.parent() else {
            return ctx.invocation_context().session.events.clone();
        };
        let parent_path = parent_builder.to_slash_string();
        if parent_path.is_empty() {
            return ctx.invocation_context().session.events.clone();
        }

        let node_events = self
            .transitive_events_by_parent
            .get(&parent_path)
            .cloned()
            .unwrap_or_default();
        if node_events.is_empty() {
            return ctx.invocation_context().session.events.clone();
        }

        let root_events = self.events_by_parent.get("").cloned().unwrap_or_default();
        let node_event_ids: std::collections::HashSet<&str> =
            node_events.iter().map(|e| e.id.as_str()).collect();
        let user_prompts: Vec<&Event> = root_events
            .iter()
            .filter(|e| e.author == "user" && !node_event_ids.contains(e.id.as_str()))
            .collect();

        if user_prompts.is_empty() {
            return node_events;
        }

        let mut event_ids: std::collections::HashSet<String> =
            node_event_ids.iter().map(|id| id.to_string()).collect();
        event_ids.extend(user_prompts.iter().map(|e| e.id.clone()));

        ctx.invocation_context()
            .session
            .events
            .iter()
            .filter(|e| event_ids.contains(&e.id))
            .cloned()
            .collect()
    }

    /// `ReplayManager._scan_sequence`: extracts the chronological child
    /// completion sequence under `base_path`.
    fn scan_sequence(
        &self,
        events: &[Event],
        ctx: &Context,
        base_path: &str,
        strict_direct_child: bool,
    ) -> Vec<String> {
        let base_path_builder = NodePathBuilder::from_string(base_path);
        let mut sequence: Vec<String> = Vec::new();
        let invocation_id = &ctx.invocation_context().invocation_id;

        for event in events {
            if !invocation_id.is_empty() && &event.invocation_id != invocation_id {
                continue;
            }

            let event_path_builder = NodePathBuilder::from_string(event.node_info.path.as_str());
            if event_path_builder.segments().len() <= base_path_builder.segments().len()
                || !event_path_builder.is_descendant_of(&base_path_builder)
            {
                continue;
            }

            let Some(child_path) = direct_child_toward(&base_path_builder, &event_path_builder)
            else {
                continue;
            };
            if strict_direct_child && event_path_builder != child_path {
                continue;
            }

            let segment = child_path.leaf_segment();

            if is_terminal_event(event) {
                sequence.retain(|s| s != &segment);
                sequence.push(segment);
            }
        }

        sequence
    }

    /// `ReplayManager.scan_workflow_events`: scans session events for
    /// direct child workflow nodes and initializes the sequence
    /// barrier.
    pub fn scan_workflow_events(
        &mut self,
        ctx: &Context,
    ) -> Result<(BTreeMap<String, ChildScanState>, Vec<String>), String> {
        let events = ctx.invocation_context().session.events.clone();
        self.build_event_index(&events, &ctx.invocation_context().invocation_id);

        let filtered_events = self
            .transitive_events_by_parent
            .get(ctx.node_path())
            .cloned()
            .unwrap_or_default();
        let raw_results = reconstruct_node_states(
            &filtered_events,
            ctx.node_path(),
            &ctx.invocation_context().invocation_id,
            true,
        )?;

        let transitive_events = self
            .transitive_events_by_parent
            .get(ctx.node_path())
            .cloned()
            .unwrap_or_default();
        let sequence = self.scan_sequence(&transitive_events, ctx, ctx.node_path(), false);

        self.recovered_executions = raw_results.clone();
        self.sequence_barrier = Some(ReplaySequenceBarrier::new(sequence.clone()));
        Ok((raw_results, sequence))
    }

    /// `ReplayManager.prepare_parent_sequence_barrier`: ensures a
    /// sequence barrier is set up for dynamic nodes under `parent_path`.
    pub fn prepare_parent_sequence_barrier(&mut self, ctx: &Context, parent_path: &str) {
        if self.parent_sequence_barriers.contains_key(parent_path) {
            return;
        }
        self.ensure_index(ctx);

        let events = self
            .events_by_parent
            .get(parent_path)
            .cloned()
            .unwrap_or_default();
        let seq = self.scan_sequence(&events, ctx, parent_path, true);
        self.parent_sequence_barriers
            .insert(parent_path.to_string(), ReplaySequenceBarrier::new(seq));
    }

    /// `ReplayManager.advance_sequence`: advances the sequence barrier
    /// if one is initialized for `parent_path`.
    pub fn advance_sequence(&mut self, parent_path: &str, key: &str) {
        if let Some(barrier) = self.parent_sequence_barriers.get_mut(parent_path) {
            barrier.check_and_advance(key);
        }
    }

    /// `ReplayManager.wait_sequence`: waits for the sequence barrier if
    /// one is initialized for `parent_path`.
    pub async fn wait_sequence(&self, parent_path: &str, key: &str) -> Result<(), String> {
        if let Some(barrier) = self.parent_sequence_barriers.get(parent_path) {
            barrier.wait(key).await
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use adk_events::node_info::NodeInfo;
    use rusty_serde::value::Value;

    /// `node_path` is the desired final path (e.g. `"wf@1"`) — split
    /// into `(name, run_id)` so `Context::for_node`'s own `node@run_id`
    /// derivation produces it exactly, rather than passing the whole
    /// tagged string as the node name (which would double-tag it into
    /// `"wf@1@1"`).
    fn ctx_at(node_path: &str, events: Vec<Event>) -> Context {
        let mut session = Session::new("app", "user", "s1");
        session.events = events;
        let ic = InvocationContextBuilder::new("inv-1", session).build();
        let (name, run_id) = match node_path.split_once('@') {
            Some((name, run_id)) => (name, run_id.to_string()),
            None => (node_path, "1".to_string()),
        };
        Context::for_node(
            ic,
            "",
            &[],
            None,
            name,
            run_id,
            BTreeMap::new(),
            1,
            false,
            false,
            None,
        )
    }

    fn event_at(path: &str) -> Event {
        Event::new(
            "inv-1".to_string(),
            "child".to_string(),
            NodeInfo::new(path),
        )
    }

    #[test]
    fn scan_workflow_events_recovers_direct_child_output() {
        let mut child_event = event_at("wf@1/child@1");
        child_event.output = Some(Value::String("done".to_string()));
        let ctx = ctx_at("wf@1", vec![child_event]);

        let mut manager = ReplayManager::new();
        let (recovered, sequence) = manager.scan_workflow_events(&ctx).unwrap();

        assert!(recovered.contains_key("child@1"));
        assert_eq!(sequence, vec!["child@1".to_string()]);
    }

    #[test]
    fn get_events_for_rehydration_returns_empty_for_a_root_path() {
        let ctx = ctx_at("wf@1", Vec::new());
        let mut manager = ReplayManager::new();
        assert!(manager.get_events_for_rehydration(&ctx, "").is_empty());
    }

    #[test]
    fn get_events_for_rehydration_merges_top_level_user_prompts() {
        let mut user_event = Event::new("inv-1".to_string(), "user".to_string(), NodeInfo::new(""));
        user_event.content = Some(adk_genai::content::Content {
            role: Some("user".to_string()),
            parts: vec![adk_genai::content::Part {
                text: Some("hi".to_string()),
                ..Default::default()
            }],
        });
        let mut child_event = event_at("wf@1/child@1");
        child_event.output = Some(Value::String("done".to_string()));

        let ctx = ctx_at("wf@1", vec![user_event.clone(), child_event.clone()]);
        let mut manager = ReplayManager::new();
        let events = manager.get_events_for_rehydration(&ctx, "wf@1/child@1");

        assert!(events.iter().any(|e| e.id == user_event.id));
        assert!(events.iter().any(|e| e.id == child_event.id));
    }

    #[rusty_tokio::test]
    async fn prepare_and_advance_parent_sequence_barrier() {
        let mut dynamic_event = event_at("wf@1/dyn1@1");
        dynamic_event.output = Some(Value::String("done".to_string()));
        let ctx = ctx_at("wf@1", vec![dynamic_event]);

        let mut manager = ReplayManager::new();
        manager.prepare_parent_sequence_barrier(&ctx, "wf@1");
        manager.advance_sequence("wf@1", "dyn1@1");
        manager.wait_sequence("wf@1", "dyn1@1").await.unwrap();
    }

    #[test]
    fn ensure_index_rebuilds_when_the_event_count_changes() {
        let mut manager = ReplayManager::new();
        let ctx1 = ctx_at("wf@1", vec![event_at("wf@1/a@1")]);
        manager.ensure_index(&ctx1);
        assert_eq!(manager.indexed_event_count, Some(1));

        let ctx2 = ctx_at("wf@1", vec![event_at("wf@1/a@1"), event_at("wf@1/b@1")]);
        manager.ensure_index(&ctx2);
        assert_eq!(manager.indexed_event_count, Some(2));
    }
}
