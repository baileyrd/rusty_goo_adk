//! Capability C0320: workflow rehydration utilities, ported from
//! `google.adk.workflow.utils._rehydration_utils`. Part of the P7
//! workflow/graph engine — see `workflow_node_state.rs`'s module doc
//! for the standing crate-placement decision.
//!
//! **No caller yet, disclosed**: this batch (P7 Chunk 4 — this file,
//! `workflow_replay_sequence_barrier.rs`, `workflow_replay_interceptor.rs`,
//! and `workflow_replay_manager.rs`) has no caller in this port yet, the
//! same way Chunk 1's pure-data primitives had none until Chunk 2/3 used
//! them: the only caller in the source, `Workflow` (C0298-C0306), is
//! still blocked on `Graph::from_edge_items` (C0327, itself blocked on
//! `FunctionNode`/`BaseTool`) and `DynamicNodeScheduler` (C0318/C0319,
//! confirmed still blocked — its `__call__` needs `Context
//! ::_run_node_standalone` to dispatch over an arbitrary [`BaseNode`]
//! reference, and this port's `BaseNode` is a concrete struct with no
//! dynamic-dispatch seam yet). Every function here is directly testable
//! today against constructed `Context`/`Event` fixtures without needing
//! `Workflow` or dynamic dispatch to exist — that testability is what
//! makes this a legitimate batch despite having no wired-up caller.
//!
//! **`_process_rehydrated_output`/`_validate_resume_response`, narrowed**:
//! the source validates/coerces rehydrated data against a node's
//! `output_schema` (a real Python type/pydantic model/`TypeAdapter`) or
//! an interrupt's JSON-Schema `response_schema`. `workflow_base_node.rs`
//! already discloses `BaseNode::output_schema` as an opaque `Value`
//! placeholder this port never interprets, so there is no real schema to
//! validate *against* here either — [`process_rehydrated_output`] only
//! distinguishes "some schema is configured" (attempt a JSON parse,
//! falling back to the raw text — the same "don't block resumption on
//! schema drift" intent the source's own `ValidationError` fallback
//! expresses) from "no schema" (always return the raw text).
//! [`validate_resume_response`] narrows further: it only implements the
//! JSON-Schema primitive `"type"` coercions (`integer`/`number`/`string`/
//! `boolean`) the source's own `type_mapping` dict also special-cases;
//! `"array"`/`"object"` (the source's dynamic `pydantic.create_model`
//! path) pass through unvalidated rather than building a schema-driven
//! validator, matching this port's standing "no per-key schema
//! mechanism" narrowing (`state.rs`'s own module doc).
//!
//! **`ChildOutput`, adaptation disclosed**: `_ChildScanState.output` is
//! `Any` in the source — either a plain rehydrated value or a raw
//! `types.Content` (when recovered from a `message_as_output` event,
//! before `_process_rehydrated_output` normalizes it). This port
//! materializes that as an explicit two-variant enum rather than an
//! opaque `Value`, the same "pick the actual shapes exercised" approach
//! already used elsewhere for a narrow `Any`.

use std::collections::{BTreeMap, HashSet};

use adk_events::node_path_builder::NodePathBuilder;
use adk_events::Event;
use adk_genai::content::Content;
use adk_genai::content_utils::extract_text_from_content;
use rusty_serde::value::Value;

use crate::workflow_base_node::BaseNode;
use crate::workflow_hitl_utils::{
    get_request_input_interrupt_ids, REQUEST_INPUT_FUNCTION_CALL_NAME,
};

const RESULT_KEY: &str = "result";

/// `_ChildScanState`: state accumulated for a child node during event
/// scanning.
#[derive(Debug, Clone, Default)]
pub struct ChildScanState {
    pub run_id: Option<String>,
    pub output: Option<ChildOutput>,
    /// The child's emitted route, if any — kept as the raw `Value` an
    /// event's `actions.route` already carries (this port never refines
    /// it into `workflow_graph::RouteValue` at this layer, matching the
    /// source's own dynamically-typed `child.route = event.actions.route`
    /// assignment).
    pub route: Option<Value>,
    pub branch: Option<String>,
    pub isolation_scope: Option<String>,
    pub transfer_to_agent: Option<String>,
    pub interrupt_ids: HashSet<String>,
    pub resolved_ids: HashSet<String>,
    pub resolved_responses: BTreeMap<String, Value>,
}

/// See this module's own doc for why `_ChildScanState.output`'s `Any`
/// materializes as this two-variant enum.
#[derive(Debug, Clone)]
pub enum ChildOutput {
    Value(Value),
    Content(Content),
}

/// `_wrap_response`: wraps a value into a response map suitable for a
/// `FunctionResponse`. If `value` is already a map, returns it as-is;
/// otherwise wraps as `{"result": value}`.
pub fn wrap_response(value: Value) -> BTreeMap<String, Value> {
    match value {
        Value::Map(entries) => entries.into_iter().collect(),
        other => BTreeMap::from([(RESULT_KEY.to_string(), other)]),
    }
}

/// `_unwrap_response`: unwraps a `FunctionResponse`'s response map to
/// the original value. If `data` has exactly one key, `"result"`,
/// extracts the value — string values are JSON-parsed when possible
/// (the web frontend wraps user text as `{"result": text}` without
/// parsing). Otherwise returns `data` unchanged, as a `Value::Map`.
pub fn unwrap_response(data: &BTreeMap<String, Value>) -> Value {
    if data.len() == 1 {
        if let Some(value) = data.get(RESULT_KEY) {
            if let Value::String(s) = value {
                if let Ok(parsed) = rusty_serde::json::from_str::<Value>(s) {
                    return parsed;
                }
            }
            return value.clone();
        }
    }
    Value::Map(data.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// `_extract_schema_from_event`: extracts the response schema from an
/// event if it's a `RequestInput` function call matching `interrupt_id`.
pub fn extract_schema_from_event(event: &Event, interrupt_id: &str) -> Option<Value> {
    let content = event.content.as_ref()?;
    for part in &content.parts {
        let Some(fc) = &part.function_call else {
            continue;
        };
        if fc.name.as_deref() == Some(REQUEST_INPUT_FUNCTION_CALL_NAME)
            && fc.id.as_deref() == Some(interrupt_id)
        {
            if let Some(args) = &fc.args {
                if let Some(schema) = args.get("response_schema") {
                    return Some(schema.clone());
                }
            }
        }
    }
    None
}

/// `_process_rehydrated_output` — see this module's own doc for the
/// schema-coercion narrowing.
pub fn process_rehydrated_output(node: &BaseNode, output: Option<&ChildOutput>) -> Option<Value> {
    let content = match output {
        Some(ChildOutput::Content(content)) => content,
        Some(ChildOutput::Value(value)) => return Some(value.clone()),
        None => return None,
    };

    let text = extract_text_from_content(Some(content)).trim().to_string();
    if text.is_empty() {
        return None;
    }

    if node.output_schema().is_some() {
        if let Ok(parsed) = rusty_serde::json::from_str::<Value>(&text) {
            return Some(parsed);
        }
    }
    Some(Value::String(text))
}

fn coerce_to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Int(i) => Some(*i),
        Value::UInt(u) => i64::try_from(*u).ok(),
        Value::Float(f) if f.fract() == 0.0 => Some(*f as i64),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn coerce_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::UInt(u) => Some(*u as f64),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn coerce_to_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn coerce_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::UInt(u) => u.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

/// `_validate_resume_response` — see this module's own doc for the
/// primitive-types-only narrowing.
pub fn validate_resume_response(
    response_data: Value,
    schema: Option<&Value>,
) -> Result<Value, String> {
    let Some(schema) = schema else {
        return Ok(response_data);
    };
    let Some(type_str) = schema.get("type").and_then(Value::as_str) else {
        return Ok(response_data);
    };
    match type_str {
        "integer" => coerce_to_i64(&response_data)
            .map(Value::Int)
            .ok_or_else(|| format!("Failed to coerce data to integer: {response_data:?}")),
        "number" => coerce_to_f64(&response_data)
            .map(Value::Float)
            .ok_or_else(|| format!("Failed to coerce data to number: {response_data:?}")),
        "string" => Ok(Value::String(coerce_to_string(&response_data))),
        "boolean" => coerce_to_bool(&response_data)
            .map(Value::Bool)
            .ok_or_else(|| format!("Failed to coerce data to boolean: {response_data:?}")),
        _ => Ok(response_data),
    }
}

/// Truncates `descendant` down to the single segment directly below
/// `base` — `_NodePathBuilder.get_direct_child(descendant)` in the
/// source, a different overload than this port's own
/// [`NodePathBuilder::append`]-based `get_direct_child` (which builds a
/// child path from a bare name, not from another builder). Kept local
/// to this module rather than added to the already-shipped
/// `node_path_builder.rs` under the same name, which would collide.
/// Returns `None` when `descendant` isn't a proper descendant of `base`.
pub(crate) fn direct_child_toward(
    base: &NodePathBuilder,
    descendant: &NodePathBuilder,
) -> Option<NodePathBuilder> {
    let base_len = base.segments().len();
    let descendant_segments = descendant.segments();
    if descendant_segments.len() <= base_len {
        return None;
    }
    if descendant_segments[..base_len] != base.segments()[..] {
        return None;
    }
    let truncated = &descendant_segments[..base_len + 1];
    let slash = truncated
        .iter()
        .map(|segment| match &segment.run_id {
            Some(run_id) => format!("{}@{}", segment.node, run_id),
            None => segment.node.clone(),
        })
        .collect::<Vec<_>>()
        .join("/");
    Some(NodePathBuilder::from_string(&slash))
}

/// Strict "proper descendant" check — `_NodePathBuilder.is_descendant_of`
/// in the source excludes exact self-match; this port's own
/// [`NodePathBuilder::is_descendant_of`] (C0032, already shipped) does
/// not, so this adds the length guard locally rather than changing that
/// already-tested method's behavior.
fn is_proper_descendant(path: &NodePathBuilder, ancestor: &NodePathBuilder) -> bool {
    path.segments().len() > ancestor.segments().len() && path.is_descendant_of(ancestor)
}

/// `_reconstruct_node_states`: scans session events to reconstruct node
/// states for resume.
pub fn reconstruct_node_states(
    events: &[Event],
    base_path: &str,
    invocation_id: &str,
    group_by_direct_child: bool,
) -> Result<BTreeMap<String, ChildScanState>, String> {
    let mut scan_states: BTreeMap<String, ChildScanState> = BTreeMap::new();
    let mut interrupt_owner: BTreeMap<String, String> = BTreeMap::new();
    let mut schemas_by_id: BTreeMap<String, Value> = BTreeMap::new();

    let base_path_builder = NodePathBuilder::from_string(base_path);

    let get_owner_key = |event_path_builder: &NodePathBuilder| -> Option<String> {
        if group_by_direct_child {
            if !is_proper_descendant(event_path_builder, &base_path_builder) {
                return None;
            }
            direct_child_toward(&base_path_builder, event_path_builder)
                .map(|child| child.leaf_segment())
        } else if *event_path_builder == base_path_builder
            || is_proper_descendant(event_path_builder, &base_path_builder)
        {
            Some(base_path.to_string())
        } else {
            None
        }
    };

    for event in events {
        if !invocation_id.is_empty() && event.invocation_id != invocation_id {
            continue;
        }

        // 1. Handle FunctionResponse (user responses to interrupts).
        if event.author == "user" {
            if let Some(content) = &event.content {
                for part in &content.parts {
                    let Some(fr) = &part.function_response else {
                        continue;
                    };
                    let Some(fr_id) = &fr.id else { continue };
                    let Some(owner) = interrupt_owner.get(fr_id).cloned() else {
                        continue;
                    };
                    let state = scan_states.entry(owner).or_default();
                    state.resolved_ids.insert(fr_id.clone());
                    let mut response_data =
                        unwrap_response(fr.response.as_ref().unwrap_or(&BTreeMap::new()));

                    if let Some(schema) = schemas_by_id.get(fr_id) {
                        response_data = validate_resume_response(response_data, Some(schema))
                            .map_err(|e| format!("Validation failed for interrupt {fr_id}: {e}"))?;
                    }

                    state
                        .resolved_responses
                        .insert(fr_id.clone(), response_data);
                }
            }
            continue;
        }

        // 2. Match events under base_path.
        let event_node_path = event.node_info.path.as_str();
        let event_path_builder = NodePathBuilder::from_string(event_node_path);
        let Some(owner_key) = get_owner_key(&event_path_builder) else {
            continue;
        };

        // 3. Initialize state for the owner if needed.
        let state = scan_states.entry(owner_key.clone()).or_insert_with(|| {
            let owner_path_builder = NodePathBuilder::from_string(&owner_key);
            ChildScanState {
                run_id: owner_path_builder.run_id().map(str::to_string),
                ..Default::default()
            }
        });
        if let Some(scope) = &event.isolation_scope {
            state.isolation_scope = Some(scope.clone());
        }

        // 4. Determine if event is a direct child or a delegated output.
        let is_direct = if group_by_direct_child {
            event_path_builder.is_direct_child_of(&base_path_builder)
        } else {
            event_path_builder == base_path_builder
        };

        let mut has_output = event.output.is_some();
        let mut use_message_as_output = false;
        if !has_output
            && event.node_info.message_as_output.unwrap_or(false)
            && event.content.is_some()
        {
            has_output = true;
            use_message_as_output = true;
        }

        let mut is_delegated = false;
        if has_output {
            if let Some(output_for) = &event.node_info.output_for {
                is_delegated = if !group_by_direct_child {
                    output_for.iter().any(|p| p == base_path)
                } else {
                    let owner_full_path = base_path_builder
                        .append(owner_key.clone(), None)
                        .to_slash_string();
                    output_for.contains(&owner_full_path)
                };
            }
        }

        // 5. Extract output and route.
        if is_direct || is_delegated {
            if let Some(output) = &event.output {
                state.output = Some(ChildOutput::Value(output.clone()));
                state.branch = event.branch.clone();
            } else if use_message_as_output {
                if let Some(content) = &event.content {
                    state.output = Some(ChildOutput::Content(content.clone()));
                }
            }
            if let Some(route) = &event.actions.route {
                state.route = Some(route.clone());
            }
            if let Some(transfer) = &event.actions.transfer_to_agent {
                state.transfer_to_agent = Some(transfer.clone());
            }
        }

        // 6. Extract interrupts and their schemas.
        let mut interrupt_ids_to_process: HashSet<String> = event
            .long_running_tool_ids
            .as_ref()
            .map(|ids| ids.iter().cloned().collect())
            .unwrap_or_default();
        interrupt_ids_to_process.extend(get_request_input_interrupt_ids(event));

        for interrupt_id in &interrupt_ids_to_process {
            state.interrupt_ids.insert(interrupt_id.clone());
            interrupt_owner.insert(interrupt_id.clone(), owner_key.clone());

            if let Some(schema) = extract_schema_from_event(event, interrupt_id) {
                schemas_by_id.insert(interrupt_id.clone(), schema);
            }
        }
    }

    Ok(scan_states)
}

/// `is_terminal_event`: whether an event represents a terminal
/// execution outcome (output, route, error, or interrupt).
pub fn is_terminal_event(event: &Event) -> bool {
    if event.output.is_some() {
        return true;
    }
    if event.node_info.message_as_output.unwrap_or(false) && event.content.is_some() {
        return true;
    }
    if event.actions.route.is_some() {
        return true;
    }
    if event
        .long_running_tool_ids
        .as_ref()
        .is_some_and(|ids| !ids.is_empty())
    {
        return true;
    }
    if event.error_code.is_some() {
        return true;
    }
    crate::workflow_hitl_utils::has_request_input_function_call(event)
        || crate::workflow_hitl_utils::has_auth_request_function_call(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{FunctionResponse, Part};

    fn event_at(path: &str, invocation_id: &str) -> Event {
        Event::new(
            invocation_id.to_string(),
            "child".to_string(),
            NodeInfo::new(path),
        )
    }

    #[test]
    fn wrap_response_wraps_a_non_map_value() {
        let wrapped = wrap_response(Value::Int(7));
        assert_eq!(wrapped.get(RESULT_KEY), Some(&Value::Int(7)));
    }

    #[test]
    fn wrap_response_passes_through_an_already_mapped_value() {
        let mut entries = BTreeMap::new();
        entries.insert("a".to_string(), Value::Bool(true));
        let wrapped = wrap_response(Value::Map(entries.clone().into_iter().collect()));
        assert_eq!(wrapped, entries);
    }

    #[test]
    fn unwrap_response_extracts_a_single_result_key() {
        let mut data = BTreeMap::new();
        data.insert(RESULT_KEY.to_string(), Value::Int(42));
        assert_eq!(unwrap_response(&data), Value::Int(42));
    }

    #[test]
    fn unwrap_response_json_parses_a_string_result() {
        let mut data = BTreeMap::new();
        data.insert(RESULT_KEY.to_string(), Value::String("[1,2,3]".to_string()));
        assert_eq!(
            unwrap_response(&data),
            Value::Seq(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );
    }

    #[test]
    fn unwrap_response_returns_multi_key_maps_unchanged() {
        let mut data = BTreeMap::new();
        data.insert("a".to_string(), Value::Int(1));
        data.insert("b".to_string(), Value::Int(2));
        let unwrapped = unwrap_response(&data);
        assert!(matches!(unwrapped, Value::Map(_)));
    }

    #[test]
    fn validate_resume_response_coerces_a_string_to_an_integer() {
        let schema = Value::Map(vec![(
            "type".to_string(),
            Value::String("integer".to_string()),
        )]);
        let coerced =
            validate_resume_response(Value::String(" 42 ".to_string()), Some(&schema)).unwrap();
        assert_eq!(coerced, Value::Int(42));
    }

    #[test]
    fn validate_resume_response_coerces_a_string_to_a_boolean() {
        let schema = Value::Map(vec![(
            "type".to_string(),
            Value::String("boolean".to_string()),
        )]);
        let coerced =
            validate_resume_response(Value::String("true".to_string()), Some(&schema)).unwrap();
        assert_eq!(coerced, Value::Bool(true));
    }

    #[test]
    fn validate_resume_response_passes_through_without_a_schema() {
        let coerced = validate_resume_response(Value::String("hi".to_string()), None).unwrap();
        assert_eq!(coerced, Value::String("hi".to_string()));
    }

    #[test]
    fn validate_resume_response_passes_through_object_schemas_unvalidated() {
        let schema = Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]);
        let data = Value::Map(vec![("k".to_string(), Value::Int(1))]);
        let coerced = validate_resume_response(data.clone(), Some(&schema)).unwrap();
        assert_eq!(coerced, data);
    }

    #[test]
    fn is_terminal_event_is_true_for_output() {
        let mut event = event_at("a@1", "inv-1");
        event.output = Some(Value::Int(1));
        assert!(is_terminal_event(&event));
    }

    #[test]
    fn is_terminal_event_is_false_for_a_silent_state_only_event() {
        let event = event_at("a@1", "inv-1");
        assert!(!is_terminal_event(&event));
    }

    #[test]
    fn reconstruct_node_states_recovers_output_under_the_base_path() {
        let mut event = event_at("a@1", "inv-1");
        event.output = Some(Value::String("done".to_string()));
        let states = reconstruct_node_states(&[event], "a@1", "inv-1", false).unwrap();
        let state = states.get("a@1").unwrap();
        assert!(
            matches!(state.output, Some(ChildOutput::Value(Value::String(ref s))) if s == "done")
        );
    }

    #[test]
    fn reconstruct_node_states_groups_by_direct_child() {
        let mut event = event_at("a@1/b@1", "inv-1");
        event.output = Some(Value::Int(1));
        let states = reconstruct_node_states(&[event], "a@1", "inv-1", true).unwrap();
        assert!(states.contains_key("b@1"));
    }

    #[test]
    fn reconstruct_node_states_tracks_interrupts_and_resolves_them() {
        let mut request = event_at("a@1", "inv-1");
        request.set_long_running_tool_ids(["fc-1".to_string()]);

        let mut response = Event::new("inv-1".to_string(), "user".to_string(), NodeInfo::new(""));
        response.content = Some(adk_genai::content::Content {
            role: Some("user".to_string()),
            parts: vec![Part::function_response(FunctionResponse {
                id: Some("fc-1".to_string()),
                name: None,
                response: Some(BTreeMap::from([(
                    RESULT_KEY.to_string(),
                    Value::String("42".to_string()),
                )])),
                parts: None,
            })],
        });

        let states = reconstruct_node_states(&[request, response], "a@1", "inv-1", false).unwrap();
        let state = states.get("a@1").unwrap();
        assert!(state.resolved_ids.contains("fc-1"));
        assert_eq!(state.resolved_responses.get("fc-1"), Some(&Value::Int(42)));
    }

    #[test]
    fn reconstruct_node_states_ignores_events_from_a_different_invocation() {
        let mut event = event_at("a@1", "inv-2");
        event.output = Some(Value::Int(1));
        let states = reconstruct_node_states(&[event], "a@1", "inv-1", false).unwrap();
        assert!(states.is_empty());
    }
}
