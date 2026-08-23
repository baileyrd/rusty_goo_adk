//! Part of capability C0181-C0189: the standalone event/content-list
//! transforms `_get_contents`'s pipeline is built from, ported from
//! `google.adk.flows.llm_flows.contents`.
//!
//! **Scope, disclosed**: this batch ports every helper in `contents.py`
//! that operates purely on `Event`/`Content` values already in hand —
//! function-call/response pairing and rearrangement (C0186, C0187),
//! empty/invisible-part filtering (C0189), branch/event-kind filtering
//! (part of C0183), and the function-call-id preservation mechanism (part
//! of C0181, `copy_content_for_request`). **Not** included in this batch,
//! each deserving its own dedicated treatment:
//!   - `_get_contents`/`_get_current_turn_contents` themselves — the
//!     ~185-line orchestrating functions that call everything in this
//!     file in sequence, plus the `_ContentLlmRequestProcessor` that
//!     wraps them (C0181-C0183, C0189's own top-level wiring).
//!   - Cross-agent transcript fencing (C0184, `_fencing.py`) — prompt-
//!     injection-relevant, deserves focused attention on its own.
//!   - Compaction-aware history reconstruction (C0185,
//!     `_content_compaction.py`) — needs `EventCompaction` semantics this
//!     batch doesn't build.
//!   - C0181's actual *policy* of which model backends need FC-id
//!     preservation (Anthropic/LiteLLM/OpenAIResponsesLlm/Interactions-API
//!     Gemini) — those backends don't exist in this port yet (Phase 10,
//!     C0557+). `copy_content_for_request` below is the *mechanism* both
//!     this batch and that future policy share.
//!
//! **Adaptation, disclosed**: `Event.input_transcription`/
//! `output_transcription` stay opaque `Value` placeholders (see
//! `adk-events`'s own module doc) — this batch only checks their
//! presence (`contains_empty_content`); reading their `text` field is
//! deferred to whichever future batch needs it.

use adk_events::Event;
use adk_genai::content::{Content, FunctionCall, Part};

pub const AF_FUNCTION_CALL_ID_PREFIX: &str = "adk-";
pub const REQUEST_EUC_FUNCTION_CALL_NAME: &str = "adk_request_credential";
pub const REQUEST_CONFIRMATION_FUNCTION_CALL_NAME: &str = "adk_request_confirmation";

#[derive(Debug, rusty_err::Error)]
pub enum ContentsError {
    #[error("At least one function_response event is required.")]
    NoFunctionResponseEvents,
    #[error("There should be at least one function_response part.")]
    NoFunctionResponseParts,
    #[error(
        "Last response event should only contain the responses for the function calls in the \
         same function call event. Function call ids found: {call_ids:?}, function response ids \
         provided: {response_ids:?}"
    )]
    ResponseIdsNotASubsetOfCallIds {
        call_ids: Vec<Option<String>>,
        response_ids: Vec<Option<String>>,
    },
    #[error("No function call event found for function response ids: {0:?}")]
    NoMatchingFunctionCallEvent(Vec<Option<String>>),
}

/// C0189: `_is_part_invisible`.
pub fn is_part_invisible(part: &Part, include_thoughts: bool) -> bool {
    if part.function_call.is_some() || part.function_response.is_some() {
        return false;
    }
    if part.thought_signature.is_some() {
        return false;
    }
    if part.tool_call.is_some() || part.tool_response.is_some() {
        return false;
    }
    (part.thought == Some(true) && !include_thoughts)
        || !(part.text.is_some()
            || part.inline_data.is_some()
            || part.file_data.is_some()
            || part.executable_code.is_some()
            || part.code_execution_result.is_some())
}

/// C0189: `_contains_empty_content`.
pub fn contains_empty_content(event: &Event, include_thoughts: bool) -> bool {
    if event.actions.compaction.is_some() {
        return false;
    }
    let content_is_empty = match &event.content {
        None => true,
        Some(content) => {
            content.role.is_none()
                || content.parts.is_empty()
                || content
                    .parts
                    .iter()
                    .all(|p| is_part_invisible(p, include_thoughts))
        }
    };
    content_is_empty && event.output_transcription.is_none() && event.input_transcription.is_none()
}

/// C0183 (part): `_is_event_belongs_to_branch`.
pub fn is_event_belongs_to_branch(invocation_branch: Option<&str>, event: &Event) -> bool {
    let (Some(invocation_branch), Some(event_branch)) =
        (invocation_branch, event.branch.as_deref())
    else {
        return true;
    };
    let inv_path = adk_events::branch_path::BranchPath::from_string(invocation_branch);
    let evt_path = adk_events::branch_path::BranchPath::from_string(event_branch);
    inv_path == evt_path || inv_path.is_descendant_of(&evt_path)
}

fn is_function_call_event(event: &Event, function_name: &str) -> bool {
    let Some(content) = &event.content else {
        return false;
    };
    content.parts.iter().any(|part| {
        part.function_call
            .as_ref()
            .is_some_and(|fc| fc.name.as_deref() == Some(function_name))
            || part
                .function_response
                .as_ref()
                .is_some_and(|fr| fr.name.as_deref() == Some(function_name))
    })
}

pub fn is_auth_event(event: &Event) -> bool {
    is_function_call_event(event, REQUEST_EUC_FUNCTION_CALL_NAME)
}

pub fn is_request_confirmation_event(event: &Event) -> bool {
    is_function_call_event(event, REQUEST_CONFIRMATION_FUNCTION_CALL_NAME)
}

pub fn is_adk_framework_event(event: &Event) -> bool {
    is_function_call_event(event, "adk_framework")
}

/// C0183 (part): `_is_direct_transfer`.
pub fn is_direct_transfer(event: &Event) -> bool {
    if event.actions.transfer_to_agent.is_some() {
        return true;
    }
    let Some(content) = &event.content else {
        return false;
    };
    content.parts.iter().any(|p| {
        p.function_call
            .as_ref()
            .is_some_and(|fc| fc.name.as_deref() == Some("transfer_to_agent"))
    })
}

/// `_is_live_model_media_event_with_inline_data`.
pub fn is_live_model_media_event_with_inline_data(event: &Event) -> bool {
    let Some(content) = &event.content else {
        return false;
    };
    content.parts.iter().any(|part| {
        part.inline_data
            .as_ref()
            .and_then(|blob| blob.mime_type.as_deref())
            .map(|mime| {
                let mime = mime.to_ascii_lowercase();
                mime.starts_with("audio/")
                    || mime.starts_with("video/")
                    || mime.starts_with("image/")
            })
            .unwrap_or(false)
    })
}

/// C0183 (part): `_should_include_event_in_context`.
pub fn should_include_event_in_context(
    current_branch: Option<&str>,
    event: &Event,
    isolation_scope: Option<&str>,
    include_thoughts: bool,
) -> bool {
    if event.isolation_scope.as_deref() != isolation_scope {
        return false;
    }
    !(contains_empty_content(event, include_thoughts)
        || !is_event_belongs_to_branch(current_branch, event)
        || is_adk_framework_event(event)
        || is_auth_event(event)
        || is_request_confirmation_event(event))
}

/// C0181 (mechanism only — see the module doc): `_copy_content_for_request`.
pub fn copy_content_for_request(
    content: &Content,
    strip_client_function_call_ids: bool,
) -> Content {
    let mut new_content = content.clone();
    if !strip_client_function_call_ids {
        return new_content;
    }
    for part in &mut new_content.parts {
        if let Some(fc) = &mut part.function_call {
            if fc
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with(AF_FUNCTION_CALL_ID_PREFIX))
            {
                fc.id = None;
            }
        }
        if let Some(fr) = &mut part.function_response {
            if fr
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with(AF_FUNCTION_CALL_ID_PREFIX))
            {
                fr.id = None;
            }
        }
    }
    new_content
}

/// C0187: `_drop_orphaned_function_responses`. Unlike the source (which
/// only logs), returns the dropped ids alongside the filtered events so a
/// caller decides how to surface them — no logging framework has been
/// adopted by this workspace yet (see `debug_log.rs`'s module doc for the
/// same class of omission).
pub fn drop_orphaned_function_responses(events: &[Event]) -> (Vec<Event>, Vec<String>) {
    let mut call_ids = std::collections::HashSet::new();
    for event in events {
        for fc in event.get_function_calls() {
            if let Some(id) = &fc.id {
                call_ids.insert(id.clone());
            }
        }
    }

    let mut orphaned_ids = Vec::new();
    let mut result_events = Vec::with_capacity(events.len());
    for event in events {
        let Some(content) = &event.content else {
            result_events.push(event.clone());
            continue;
        };
        if content.parts.is_empty() || event.get_function_responses().is_empty() {
            result_events.push(event.clone());
            continue;
        }

        let mut kept_parts = Vec::with_capacity(content.parts.len());
        for part in &content.parts {
            if let Some(response) = &part.function_response {
                if let Some(id) = &response.id {
                    if !call_ids.contains(id) {
                        orphaned_ids.push(id.clone());
                        continue;
                    }
                }
            }
            kept_parts.push(part.clone());
        }

        if kept_parts.is_empty() {
            continue;
        }
        if kept_parts.len() != content.parts.len() {
            let mut new_event = event.clone();
            if let Some(new_content) = &mut new_event.content {
                new_content.parts = kept_parts;
            }
            result_events.push(new_event);
        } else {
            result_events.push(event.clone());
        }
    }

    (result_events, orphaned_ids)
}

/// C0187: `_merge_function_response_events`.
pub fn merge_function_response_events(
    function_response_events: &[Event],
) -> Result<Event, ContentsError> {
    let (first, rest) = function_response_events
        .split_first()
        .ok_or(ContentsError::NoFunctionResponseEvents)?;

    let mut merged_event = first.clone();
    let Some(merged_content) = &mut merged_event.content else {
        return Err(ContentsError::NoFunctionResponseParts);
    };
    if merged_content.parts.is_empty() {
        return Err(ContentsError::NoFunctionResponseParts);
    }

    let mut index_by_call_id: std::collections::HashMap<Option<String>, usize> =
        std::collections::HashMap::new();
    for (idx, part) in merged_content.parts.iter().enumerate() {
        if let Some(response) = &part.function_response {
            index_by_call_id.insert(response.id.clone(), idx);
        }
    }

    for event in rest {
        let Some(event_content) = &event.content else {
            return Err(ContentsError::NoFunctionResponseParts);
        };
        if event_content.parts.is_empty() {
            return Err(ContentsError::NoFunctionResponseParts);
        }
        for part in &event_content.parts {
            if let Some(response) = &part.function_response {
                match index_by_call_id.get(&response.id) {
                    Some(&idx) => merged_content.parts[idx] = part.clone(),
                    None => {
                        merged_content.parts.push(part.clone());
                        index_by_call_id
                            .insert(response.id.clone(), merged_content.parts.len() - 1);
                    }
                }
            } else {
                merged_content.parts.push(part.clone());
            }
        }
    }

    Ok(merged_event)
}

/// C0187: `_rearrange_events_for_latest_function_response`.
pub fn rearrange_events_for_latest_function_response(
    events: Vec<Event>,
) -> Result<Vec<Event>, ContentsError> {
    if events.len() < 2 {
        return Ok(events);
    }

    let function_responses = events.last().unwrap().get_function_responses();
    if function_responses.is_empty() {
        return Ok(events);
    }
    let mut function_response_ids: std::collections::HashSet<Option<String>> =
        function_responses.iter().map(|fr| fr.id.clone()).collect();

    let second_last_calls = events[events.len() - 2].get_function_calls();
    if second_last_calls
        .iter()
        .any(|fc| function_response_ids.contains(&fc.id))
    {
        // The latest function_response is already matched.
        return Ok(events);
    }

    let mut function_call_event_idx = None;
    for idx in (0..events.len() - 1).rev() {
        let function_calls = events[idx].get_function_calls();
        if function_calls.is_empty() {
            continue;
        }
        if let Some(matching) = function_calls
            .iter()
            .find(|fc| function_response_ids.contains(&fc.id))
        {
            let _ = matching;
            let call_ids: std::collections::HashSet<Option<String>> =
                function_calls.iter().map(|fc| fc.id.clone()).collect();
            if !function_response_ids.is_subset(&call_ids) {
                return Err(ContentsError::ResponseIdsNotASubsetOfCallIds {
                    call_ids: function_calls.iter().map(|fc| fc.id.clone()).collect(),
                    response_ids: function_response_ids.into_iter().collect(),
                });
            }
            function_response_ids = call_ids;
            function_call_event_idx = Some(idx);
            break;
        }
    }

    let Some(function_call_event_idx) = function_call_event_idx else {
        return Err(ContentsError::NoMatchingFunctionCallEvent(
            function_response_ids.into_iter().collect(),
        ));
    };

    let mut function_response_events = Vec::new();
    for event in &events[function_call_event_idx + 1..events.len() - 1] {
        let responses = event.get_function_responses();
        if !responses.is_empty()
            && responses
                .iter()
                .any(|fr| function_response_ids.contains(&fr.id))
        {
            function_response_events.push(event.clone());
        }
    }
    function_response_events.push(events.last().unwrap().clone());

    let mut result_events = events[..function_call_event_idx + 1].to_vec();
    result_events.push(merge_function_response_events(&function_response_events)?);
    Ok(result_events)
}

/// C0186: `_rearrange_events_for_async_function_responses_in_history`.
pub fn rearrange_events_for_async_function_responses_in_history(
    events: Vec<Event>,
) -> Result<Vec<Event>, ContentsError> {
    let mut call_event_indices_by_id: std::collections::HashMap<Option<String>, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, event) in events.iter().enumerate() {
        if !event.get_function_responses().is_empty() {
            continue;
        }
        for fc in event.get_function_calls() {
            call_event_indices_by_id
                .entry(fc.id.clone())
                .or_default()
                .push(i);
        }
    }

    let mut response_event_index_by_call: std::collections::HashMap<
        (Option<String>, usize),
        usize,
    > = std::collections::HashMap::new();
    let mut history_has_function_responses = false;
    for (i, event) in events.iter().enumerate() {
        for fr in event.get_function_responses() {
            history_has_function_responses = true;
            let Some(call_event_indices) = call_event_indices_by_id.get(&fr.id) else {
                continue;
            };
            if call_event_indices.is_empty() {
                continue;
            }
            let preceding_calls = call_event_indices.partition_point(|&idx| idx < i);
            let owning_call_event_index = call_event_indices[preceding_calls.saturating_sub(1)];
            response_event_index_by_call.insert((fr.id.clone(), owning_call_event_index), i);
        }
    }

    if !history_has_function_responses {
        return Ok(events);
    }

    let mut result_events = Vec::with_capacity(events.len());
    for (i, event) in events.iter().enumerate() {
        if !event.get_function_responses().is_empty() {
            continue;
        }
        let function_calls: Vec<&FunctionCall> = event.get_function_calls();
        if !function_calls.is_empty() {
            let mut response_indices = std::collections::BTreeSet::new();
            for fc in &function_calls {
                if let Some(&idx) = response_event_index_by_call.get(&(fc.id.clone(), i)) {
                    response_indices.insert(idx);
                }
            }
            result_events.push(event.clone());
            if response_indices.is_empty() {
                continue;
            }
            if response_indices.len() == 1 {
                let idx = *response_indices.iter().next().unwrap();
                result_events.push(events[idx].clone());
            } else {
                let merge_events: Vec<Event> = response_indices
                    .iter()
                    .map(|&idx| events[idx].clone())
                    .collect();
                result_events.push(merge_function_response_events(&merge_events)?);
            }
        } else {
            result_events.push(event.clone());
        }
    }

    Ok(result_events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use rusty_serde::value::Value;

    fn event(author: &str) -> Event {
        Event::new("inv-1", author, NodeInfo::new("root"))
    }

    fn event_with_content(author: &str, content: Content) -> Event {
        let mut e = event(author);
        e.content = Some(content);
        e
    }

    fn fc_part(id: &str, name: &str) -> Part {
        Part::function_call(FunctionCall {
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            args: None,
            will_continue: None,
        })
    }

    fn fr_part(id: &str, name: &str) -> Part {
        Part::function_response(adk_genai::content::FunctionResponse {
            id: Some(id.to_string()),
            name: Some(name.to_string()),
            response: None,
        })
    }

    // --- is_part_invisible / contains_empty_content ---

    #[test]
    fn a_plain_text_part_is_visible() {
        assert!(!is_part_invisible(&Part::text("hi"), false));
    }

    #[test]
    fn a_thought_only_part_is_invisible_unless_thoughts_are_included() {
        let part = Part {
            text: Some("thinking".to_string()),
            thought: Some(true),
            ..Default::default()
        };
        assert!(is_part_invisible(&part, false));
        assert!(!is_part_invisible(&part, true));
    }

    #[test]
    fn a_function_call_part_is_never_invisible_even_if_marked_thought() {
        let part = Part {
            thought: Some(true),
            ..fc_part("id1", "tool")
        };
        assert!(!is_part_invisible(&part, false));
    }

    #[test]
    fn a_thought_signature_alone_is_never_invisible() {
        let part = Part {
            thought_signature: Some(Value::String("sig".to_string())),
            ..Default::default()
        };
        assert!(!is_part_invisible(&part, false));
    }

    #[test]
    fn a_tool_call_part_is_never_invisible() {
        let part = Part {
            tool_call: Some(Value::String("call".to_string())),
            ..Default::default()
        };
        assert!(!is_part_invisible(&part, false));
    }

    #[test]
    fn a_compaction_event_is_never_empty() {
        let mut e = event("user");
        e.actions.compaction = Some(adk_events::event_compaction::EventCompaction {
            start_timestamp: 1.0,
            end_timestamp: 2.0,
            compacted_content: Content::user_text("summary"),
        });
        assert!(!contains_empty_content(&e, false));
    }

    #[test]
    fn an_event_with_only_a_thought_part_is_empty() {
        let e = event_with_content(
            "model",
            Content::new(
                "model",
                vec![Part {
                    text: Some("t".to_string()),
                    thought: Some(true),
                    ..Default::default()
                }],
            ),
        );
        assert!(contains_empty_content(&e, false));
    }

    // --- branch / event-kind filters ---

    #[test]
    fn events_with_no_branch_info_always_belong() {
        assert!(is_event_belongs_to_branch(None, &event("user")));
    }

    #[test]
    fn a_descendant_branch_belongs() {
        let mut e = event("user");
        e.branch = Some("root".to_string());
        assert!(is_event_belongs_to_branch(Some("root.child"), &e));
    }

    #[test]
    fn a_sibling_branch_does_not_belong() {
        let mut e = event("user");
        e.branch = Some("root.sibling".to_string());
        assert!(!is_event_belongs_to_branch(Some("root.child"), &e));
    }

    #[test]
    fn is_auth_event_detects_the_request_credential_function_call() {
        let e = event_with_content(
            "user",
            Content::new(
                "model",
                vec![fc_part("id1", REQUEST_EUC_FUNCTION_CALL_NAME)],
            ),
        );
        assert!(is_auth_event(&e));
    }

    #[test]
    fn is_direct_transfer_detects_the_transfer_to_agent_action() {
        let mut e = event("model");
        e.actions.transfer_to_agent = Some("sub_agent".to_string());
        assert!(is_direct_transfer(&e));
    }

    #[test]
    fn is_live_model_media_event_detects_audio_inline_data() {
        let e = event_with_content(
            "model",
            Content::new(
                "model",
                vec![Part {
                    inline_data: Some(adk_genai::content::MediaBlobStub {
                        mime_type: Some("audio/pcm".to_string()),
                        rest: None,
                    }),
                    ..Default::default()
                }],
            ),
        );
        assert!(is_live_model_media_event_with_inline_data(&e));
    }

    // --- copy_content_for_request ---

    #[test]
    fn strips_client_function_call_ids_when_requested() {
        let content = Content::new("model", vec![fc_part("adk-123", "tool")]);
        let copied = copy_content_for_request(&content, true);
        assert!(copied.parts[0].function_call.as_ref().unwrap().id.is_none());
    }

    #[test]
    fn preserves_function_call_ids_when_not_stripping() {
        let content = Content::new("model", vec![fc_part("adk-123", "tool")]);
        let copied = copy_content_for_request(&content, false);
        assert_eq!(
            copied.parts[0]
                .function_call
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some("adk-123")
        );
    }

    #[test]
    fn non_adk_prefixed_ids_survive_stripping() {
        let content = Content::new("model", vec![fc_part("server-issued-1", "tool")]);
        let copied = copy_content_for_request(&content, true);
        assert_eq!(
            copied.parts[0]
                .function_call
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some("server-issued-1")
        );
    }

    // --- drop_orphaned_function_responses ---

    #[test]
    fn drops_a_function_response_with_no_matching_call() {
        let events = vec![event_with_content(
            "user",
            Content::new("user", vec![fr_part("missing-id", "tool")]),
        )];
        let (result, orphaned) = drop_orphaned_function_responses(&events);
        assert!(result.is_empty());
        assert_eq!(orphaned, vec!["missing-id".to_string()]);
    }

    #[test]
    fn keeps_a_function_response_with_a_matching_call() {
        let events = vec![
            event_with_content("model", Content::new("model", vec![fc_part("id1", "tool")])),
            event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")])),
        ];
        let (result, orphaned) = drop_orphaned_function_responses(&events);
        assert_eq!(result.len(), 2);
        assert!(orphaned.is_empty());
    }

    // --- merge_function_response_events ---

    #[test]
    fn merge_function_response_events_requires_at_least_one_event() {
        let err = merge_function_response_events(&[]).unwrap_err();
        assert!(matches!(err, ContentsError::NoFunctionResponseEvents));
    }

    #[test]
    fn merges_later_responses_into_the_first_event_by_id() {
        let first = event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let mut second_response = fr_part("id1", "tool");
        if let Some(fr) = &mut second_response.function_response {
            fr.response = Some(std::collections::BTreeMap::from([(
                "result".to_string(),
                Value::String("done".to_string()),
            )]));
        }
        let second = event_with_content("user", Content::new("user", vec![second_response]));
        let merged = merge_function_response_events(&[first, second]).unwrap();
        assert_eq!(merged.content.unwrap().parts.len(), 1);
    }

    // --- rearrange_events_for_latest_function_response ---

    #[test]
    fn rearrange_latest_is_a_no_op_with_fewer_than_two_events() {
        let events = vec![event("user")];
        let result = rearrange_events_for_latest_function_response(events.clone()).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn rearrange_latest_merges_intervening_async_progress_events() {
        let call = event_with_content("model", Content::new("model", vec![fc_part("id1", "tool")]));
        let progress =
            event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let final_response =
            event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let events = vec![call, progress, final_response];
        let result = rearrange_events_for_latest_function_response(events).unwrap();
        // [call, merged-response]
        assert_eq!(result.len(), 2);
    }

    // --- rearrange_events_for_async_function_responses_in_history ---

    #[test]
    fn async_rearrange_is_a_no_op_without_any_function_responses() {
        let events = vec![event("user"), event("model")];
        let result =
            rearrange_events_for_async_function_responses_in_history(events.clone()).unwrap();
        assert_eq!(result.len(), events.len());
    }

    #[test]
    fn async_rearrange_attaches_the_response_right_after_its_call() {
        let call = event_with_content("model", Content::new("model", vec![fc_part("id1", "tool")]));
        let unrelated = event("user");
        let response =
            event_with_content("user", Content::new("user", vec![fr_part("id1", "tool")]));
        let events = vec![call, unrelated, response];
        let result = rearrange_events_for_async_function_responses_in_history(events).unwrap();
        // The response is moved to sit right after its call; the unrelated
        // event that originally separated them is pushed back afterward.
        assert_eq!(result.len(), 3);
        assert!(!result[0].get_function_calls().is_empty());
        assert!(!result[1].get_function_responses().is_empty());
        assert!(
            result[2].get_function_calls().is_empty()
                && result[2].get_function_responses().is_empty()
        );
    }
}
