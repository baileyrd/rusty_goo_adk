//! Capabilities C0016-C0024: `Event`, ported from `google.adk.events.event`.

use crate::event_actions::EventActions;
use crate::node_info::NodeInfo;
use adk_platform::time::get_time;
use adk_platform::uuid::new_uuid;
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One event in a session's history.
///
/// **Adaptation (capability C0017)**: the source's `Event` extends
/// `LlmResponse` (from `models/`, phase P3) via Python inheritance, gaining
/// ~20 fields (content, grounding metadata, usage metadata, transcriptions,
/// ...) on top of the ones `events/event.py` itself declares. Rust has no
/// struct inheritance, and P3 hasn't landed yet, so those inherited fields
/// are flattened directly onto this struct (matching the actual flat JSON
/// wire shape either way) and typed as JSON [`Value`] placeholders for now.
/// They narrow to concrete types (`Content`, `GroundingMetadata`,
/// `UsageMetadata`, ...) once P3's `models/` crate lands, at which point
/// this struct is expected to become `#[rusty_serde(flatten)] base:
/// LlmResponse` plus its own fields, rather than one flat struct — noted
/// here so that refactor isn't a surprise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct Event {
    // --- Event's own fields (events/event.py) ---
    pub invocation_id: String,
    pub author: String,
    #[rusty_serde(default)]
    pub actions: EventActions,
    #[rusty_serde(default)]
    pub output: Option<Value>,
    pub node_info: NodeInfo,
    /// Kept pre-sorted on every mutation (see [`Event::set_long_running_tool_ids`])
    /// so serialization always emits a stable, sorted list rather than
    /// relying on hash-iteration order — this is capability C0018.
    #[rusty_serde(default)]
    pub long_running_tool_ids: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub branch: Option<String>,
    /// Internal — mirrors the source's own warning: "DO NOT USE THIS FIELD
    /// DIRECTLY... may change without notice."
    #[rusty_serde(default)]
    pub isolation_scope: Option<String>,
    pub id: String,
    pub timestamp: f64,

    // --- Fields inherited from LlmResponse in the source (P3 placeholder) ---
    #[rusty_serde(default)]
    pub content: Option<Value>,
    #[rusty_serde(default)]
    pub grounding_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub partial: Option<bool>,
    #[rusty_serde(default)]
    pub turn_complete: Option<bool>,
    #[rusty_serde(default)]
    pub turn_complete_reason: Option<Value>,
    #[rusty_serde(default)]
    pub finish_reason: Option<String>,
    #[rusty_serde(default)]
    pub error_code: Option<String>,
    #[rusty_serde(default)]
    pub error_message: Option<String>,
    #[rusty_serde(default)]
    pub interrupted: Option<bool>,
    #[rusty_serde(default)]
    pub custom_metadata: Option<HashMap<String, Value>>,
    #[rusty_serde(default)]
    pub usage_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub live_session_resumption_update: Option<Value>,
    #[rusty_serde(default)]
    pub live_session_id: Option<String>,
    #[rusty_serde(default)]
    pub go_away: Option<Value>,
    #[rusty_serde(default)]
    pub voice_activity: Option<Value>,
    #[rusty_serde(default)]
    pub input_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub output_transcription: Option<Value>,
    #[rusty_serde(default)]
    pub avg_logprobs: Option<f64>,
    #[rusty_serde(default)]
    pub logprobs_result: Option<Value>,
    #[rusty_serde(default)]
    pub cache_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub citation_metadata: Option<Value>,
    #[rusty_serde(default)]
    pub interaction_id: Option<String>,
    #[rusty_serde(default)]
    pub environment_id: Option<String>,
}

impl Event {
    /// Constructs a new event with an auto-assigned [`Event::new_id`] and
    /// current timestamp — mirrors the source's `model_post_init` /
    /// `Event.new_id()` behavior (capability C0024).
    pub fn new(
        invocation_id: impl Into<String>,
        author: impl Into<String>,
        node_info: NodeInfo,
    ) -> Self {
        Self {
            invocation_id: invocation_id.into(),
            author: author.into(),
            actions: EventActions::default(),
            output: None,
            node_info,
            long_running_tool_ids: None,
            branch: None,
            isolation_scope: None,
            id: Self::new_id(),
            timestamp: get_time(),
            content: None,
            grounding_metadata: None,
            partial: None,
            turn_complete: None,
            turn_complete_reason: None,
            finish_reason: None,
            error_code: None,
            error_message: None,
            interrupted: None,
            custom_metadata: None,
            usage_metadata: None,
            live_session_resumption_update: None,
            live_session_id: None,
            go_away: None,
            voice_activity: None,
            input_transcription: None,
            output_transcription: None,
            avg_logprobs: None,
            logprobs_result: None,
            cache_metadata: None,
            citation_metadata: None,
            interaction_id: None,
            environment_id: None,
        }
    }

    /// Static id generator, delegating to `adk_platform::uuid::new_uuid`
    /// (capability C0024).
    pub fn new_id() -> String {
        new_uuid()
    }

    /// Sets `long_running_tool_ids`, keeping it sorted (capability C0018).
    pub fn set_long_running_tool_ids<I, S>(&mut self, ids: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut sorted: Vec<String> = ids.into_iter().map(Into::into).collect();
        sorted.sort();
        sorted.dedup();
        self.long_running_tool_ids = Some(sorted);
    }

    /// Convenience-kwarg-routing equivalent (capability C0019): the
    /// source's `message=` constructor kwarg, routed to `content`.
    ///
    /// **Adaptation**: a placeholder until P3's `Content` type lands — for
    /// now this just stores the given [`Value`] as-is on `content`,
    /// without the source's `t_content` transformer or its "raises if both
    /// `message` and `content` are given" validation (there is no separate
    /// `content` constructor arg to conflict with here yet).
    pub fn with_message(mut self, message: Value) -> Self {
        self.content = Some(message);
        self
    }

    /// Convenience-kwarg-routing equivalent (capability C0019): the
    /// source's `state=` constructor kwarg, routed to `actions.state_delta`.
    pub fn with_state(mut self, state: HashMap<String, Value>) -> Self {
        self.actions.state_delta.extend(state);
        self
    }

    /// Convenience-kwarg-routing equivalent (capability C0019): the
    /// source's `route=` constructor kwarg, routed to `actions.route`.
    pub fn with_route(mut self, route: Value) -> Self {
        self.actions.route = Some(route);
        self
    }

    /// Convenience-kwarg-routing equivalent (capability C0019): the
    /// source's `node_path=` constructor kwarg, routed to `node_info.path`.
    pub fn with_node_path(mut self, node_path: impl Into<String>) -> Self {
        self.node_info.path = node_path.into();
        self
    }

    /// `message` getter (capability C0020) — reads `content`.
    pub fn message(&self) -> Option<&Value> {
        self.content.as_ref()
    }

    /// `message` setter (capability C0020) — writes (or clears, on `None`)
    /// `content`.
    pub fn set_message(&mut self, message: Option<Value>) {
        self.content = message;
    }

    /// `node_name` property (capability C0021): empty when
    /// `actions.agent_state` is set or `actions.end_of_agent` is true,
    /// else the node's plain name.
    pub fn node_name(&self) -> String {
        if self.actions.agent_state.is_some() || self.actions.end_of_agent {
            return String::new();
        }
        self.node_info.name()
    }

    /// `is_final_response()` (capability C0022).
    ///
    /// **Partial implementation, not yet full parity**: the source's full
    /// rule also inspects `content` for function-call/function-response
    /// parts and checks [`Event::has_trailing_code_execution_result`] — both
    /// require the real `Content`/`Part` structure from P3's `models/`
    /// crate, which doesn't exist yet (`content` here is an untyped JSON
    /// placeholder). This implementation covers the two branches that
    /// don't need `Content`'s structure (the `skip_summarization`/
    /// `long_running_tool_ids` short-circuit, and the `partial` check);
    /// the function-call-aware branch is a follow-up once P3 lands, and
    /// this capability's manifest row stays `REQUIRED` until then rather
    /// than being marked done on partial coverage.
    pub fn is_final_response(&self) -> bool {
        if self.actions.skip_summarization || self.long_running_tool_ids.is_some() {
            return true;
        }
        if self.partial == Some(true) {
            return false;
        }
        // TODO(P3): also require no function calls / no function
        // responses / no trailing code-execution result once `content`
        // has real structure to inspect.
        true
    }

    /// `has_trailing_code_execution_result()` (capability C0023).
    ///
    /// **Not yet implemented pending P3**: this fundamentally requires
    /// inspecting the last part of `content` for a `code_execution_result`
    /// — impossible to do meaningfully against the current untyped JSON
    /// placeholder. Returns `false` unconditionally for now; this
    /// capability's manifest row stays `REQUIRED` until P3's `Content`/
    /// `Part` types land and this can be implemented for real.
    pub fn has_trailing_code_execution_result(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event() -> Event {
        Event::new("inv-1", "test_agent", NodeInfo::new("root"))
    }

    /// Parity test for capability C0024: `id` is auto-assigned (non-empty,
    /// unique per construction) and `timestamp` is a plausible current
    /// wall-clock time.
    #[test]
    fn new_assigns_unique_id_and_current_timestamp() {
        let a = sample_event();
        let b = sample_event();
        assert_ne!(a.id, b.id);
        assert!(!a.id.is_empty());
        assert!(
            a.timestamp > 1_767_225_600.0,
            "expected a post-2026 timestamp"
        );
    }

    /// Parity test for capability C0018: `long_running_tool_ids` is always
    /// stored (and therefore serialized) sorted, regardless of insertion
    /// order.
    #[test]
    fn long_running_tool_ids_are_kept_sorted() {
        let mut event = sample_event();
        event.set_long_running_tool_ids(["zeta", "alpha", "mu"]);
        assert_eq!(
            event.long_running_tool_ids.as_deref(),
            Some(&["alpha".to_string(), "mu".to_string(), "zeta".to_string()][..])
        );
    }

    /// Parity test for capability C0019: builder-style convenience
    /// routing for `state`/`route`/`node_path`, mirroring the source's
    /// constructor-kwarg routing.
    #[test]
    fn builder_methods_route_to_the_expected_fields() {
        let mut state = HashMap::new();
        state.insert("count".to_string(), Value::Int(1));
        let event = sample_event()
            .with_state(state)
            .with_route(Value::String("agent_b".to_string()))
            .with_node_path("root/child");

        assert_eq!(event.actions.state_delta.get("count"), Some(&Value::Int(1)));
        assert_eq!(
            event.actions.route,
            Some(Value::String("agent_b".to_string()))
        );
        assert_eq!(event.node_info.path, "root/child");
    }

    /// Parity test for capability C0020: `message`/`set_message` read and
    /// write `content`.
    #[test]
    fn message_getter_and_setter_operate_on_content() {
        let mut event = sample_event();
        assert_eq!(event.message(), None);
        event.set_message(Some(Value::String("hi".to_string())));
        assert_eq!(event.message(), Some(&Value::String("hi".to_string())));
        event.set_message(None);
        assert_eq!(event.message(), None);
    }

    /// Parity test for capability C0021: `node_name` is empty when
    /// `actions.end_of_agent` or `actions.agent_state` is set, else the
    /// node's plain name.
    #[test]
    fn node_name_is_empty_for_agent_state_events() {
        let mut event = sample_event();
        assert_eq!(event.node_name(), "root");

        event.actions.end_of_agent = true;
        assert_eq!(event.node_name(), "");

        let mut other = sample_event();
        other.actions.agent_state = Some(HashMap::new());
        assert_eq!(other.node_name(), "");
    }

    /// Parity test for capability C0022 (partial): the
    /// `skip_summarization`/`long_running_tool_ids`/`partial` branches
    /// that don't need `Content` structure.
    #[test]
    fn is_final_response_covers_the_non_content_branches() {
        let mut event = sample_event();
        assert!(
            event.is_final_response(),
            "no partial/tool-id flags => final by default"
        );

        event.partial = Some(true);
        assert!(!event.is_final_response());

        event.partial = None;
        event.actions.skip_summarization = true;
        assert!(event.is_final_response());

        let mut with_tool_ids = sample_event();
        with_tool_ids.partial = Some(true);
        with_tool_ids.set_long_running_tool_ids(["tool1"]);
        assert!(
            with_tool_ids.is_final_response(),
            "long_running_tool_ids short-circuits even when partial"
        );
    }

    #[test]
    fn round_trips_through_json_with_camel_case() {
        let event = sample_event();
        let json = rusty_serde::json::to_string(&event).unwrap();
        assert!(json.contains("\"invocationId\""));
        assert!(json.contains("\"nodeInfo\""));
        let back: Event = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(event, back);
    }
}
