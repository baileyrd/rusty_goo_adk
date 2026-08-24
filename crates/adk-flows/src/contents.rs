//! Capabilities C0181-C0189: `_get_contents`'s full pipeline, ported from
//! `google.adk.flows.llm_flows.contents`.
//!
//! This module now covers everything in `contents.py` except the
//! `_ContentLlmRequestProcessor` wiring itself: the standalone event/
//! content-list transforms (function-call/response pairing and
//! rearrangement — C0186/C0187; empty/invisible-part filtering — C0189;
//! branch/event-kind filtering — part of C0183; the function-call-id
//! preservation mechanism — part of C0181, `copy_content_for_request`),
//! and now the top-level orchestration itself: [`get_contents`]/
//! [`get_current_turn_contents`] (C0181-C0183, C0188, C0189's own
//! top-level wiring) and [`build_task_input_user_content`], each calling
//! into `crate::fencing` (C0184) and `crate::compaction` (C0185) in
//! sequence exactly as the source does.
//!
//! **Not** included, each its own dedicated future batch:
//!   - The `_ContentLlmRequestProcessor` itself, which decides *when* to
//!     call [`get_contents`] vs [`get_current_turn_contents`]
//!     (`agent.include_contents`), computes `preserve_function_call_ids`
//!     from the agent's canonical model type, and wires in
//!     `_add_model_input_context_to_user_content`/
//!     `_add_instructions_to_user_content`. This needs `LlmAgent` wired
//!     into `BaseAgent`'s tree and a real `InvocationContext.agent` —
//!     the same blocker every other Phase 4 processor has disclosed.
//!   - C0181's actual *policy* of which model backends need FC-id
//!     preservation (Anthropic/LiteLLM/OpenAIResponsesLlm/Interactions-API
//!     Gemini) — those backends don't exist in this port yet (Phase 10,
//!     C0557+). `copy_content_for_request` below is the *mechanism* both
//!     this batch and that future policy share.
//!
//! **Adaptation, disclosed**: `Event.input_transcription`/
//! `output_transcription` stay opaque `Value` placeholders (see
//! `adk-events`'s own module doc); [`get_contents`]'s transcription-
//! coalescing step (C0188) reads their `text` key out of the opaque
//! value via [`transcription_text`] rather than a typed field access.

use adk_events::Event;
use adk_genai::content::{Content, FunctionCall, Part};
use rusty_serde::value::Value;
use std::collections::HashMap;

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
        if function_calls
            .iter()
            .any(|fc| function_response_ids.contains(&fc.id))
        {
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

fn transcription_text(value: &Value) -> Option<&str> {
    value.get("text").and_then(Value::as_str)
}

const SINGLE_TURN_NUDGE: &str = "Important: You will not receive any user replies or \
     clarifications. Complete the task using only the information provided above.";

/// C0181-C0183 (part): `_build_task_input_user_content`.
///
/// A task agent runs under `isolation_scope=<fc_id>`, where `fc_id` matches
/// the `function_call.id` that delegated to it. The FC itself lives on a
/// parent event (typically the chat coordinator's), so it is filtered out
/// of the task agent's own content by the isolation_scope filter. This
/// rebuilds it as a user-role text content so the task agent's LLM sees
/// its task as the first turn.
///
/// When no matching FC is found (a task dispatched directly by a workflow
/// node, not via FC delegation), falls back to `user_content`. Returns
/// `None` if neither source yields content.
pub fn build_task_input_user_content(
    all_events: &[Event],
    isolation_scope: &str,
    is_single_turn: bool,
    user_content: Option<&Content>,
) -> Option<Content> {
    for event in all_events {
        let Some(content) = &event.content else {
            continue;
        };
        for part in &content.parts {
            let Some(fc) = &part.function_call else {
                continue;
            };
            if fc.id.as_deref() != Some(isolation_scope) {
                continue;
            }
            if fc.args.as_ref().is_none_or(|args| args.is_empty()) {
                continue;
            }
            let text = fc
                .args
                .as_ref()
                .map(|args| rusty_serde::json::to_string(args).unwrap_or_default())
                .unwrap_or_default();
            let mut parts = vec![Part::text(text)];
            if is_single_turn {
                parts.push(Part::text(SINGLE_TURN_NUDGE));
            }
            return Some(Content::new("user", parts));
        }
    }

    let user_content = user_content?;
    if user_content.parts.is_empty() {
        return None;
    }
    let mut parts = user_content.parts.clone();
    if is_single_turn {
        parts.push(Part::text(SINGLE_TURN_NUDGE));
    }
    Some(Content::new("user", parts))
}

/// Coalesces adjacent content-less transcription-fragment events into one
/// ordinary text event (C0188), for use inside [`get_contents`]'s main
/// loop. Returns `None` when this iteration's fragment isn't the last of
/// its run yet (the caller should skip it and keep accumulating).
fn coalesce_transcription_event(
    events_to_process: &[Event],
    i: usize,
    accumulated_input: &mut String,
    accumulated_output: &mut String,
) -> Option<Event> {
    let event = &events_to_process[i];
    if event.content.is_some() {
        return Some(event.clone());
    }
    let n = events_to_process.len();

    if let Some(text) = event
        .input_transcription
        .as_ref()
        .and_then(transcription_text)
    {
        if !text.is_empty() {
            accumulated_input.push_str(text);
            let next_has_text = i + 1 < n
                && events_to_process[i + 1]
                    .input_transcription
                    .as_ref()
                    .and_then(transcription_text)
                    .is_some_and(|t| !t.is_empty());
            if next_has_text {
                return None;
            }
            let mut new_event = event.clone();
            new_event.input_transcription = None;
            new_event.content = Some(Content::new(
                "user",
                vec![Part::text(std::mem::take(accumulated_input))],
            ));
            return Some(new_event);
        }
    }

    if let Some(text) = event
        .output_transcription
        .as_ref()
        .and_then(transcription_text)
    {
        if !text.is_empty() {
            accumulated_output.push_str(text);
            let next_has_text = i + 1 < n
                && events_to_process[i + 1]
                    .output_transcription
                    .as_ref()
                    .and_then(transcription_text)
                    .is_some_and(|t| !t.is_empty());
            if next_has_text {
                return None;
            }
            let mut new_event = event.clone();
            new_event.output_transcription = None;
            new_event.content = Some(Content::new(
                "model",
                vec![Part::text(std::mem::take(accumulated_output))],
            ));
            return Some(new_event);
        }
    }

    Some(event.clone())
}

/// C0181-C0183, C0188, C0189: `_get_contents` — the full pipeline that
/// turns raw session events into the `LlmRequest.contents` list.
///
/// Applies (in order): rewind filtering (via `adk_events::rewind::apply_rewinds`),
/// branch/isolation-scope/event-kind filtering, compaction resolution (via
/// `crate::compaction`), transcription-fragment coalescing, cross-agent
/// message fencing (via `crate::fencing`), orphaned-response dropping,
/// both function-call/response rearrangement passes, function-call-id
/// stripping, and (for scoped agents) prepending the task's originating
/// input as a synthetic leading user turn.
#[allow(clippy::too_many_arguments)]
pub fn get_contents(
    current_branch: Option<&str>,
    events: &[Event],
    agent_name: &str,
    preserve_function_call_ids: bool,
    isolation_scope: Option<&str>,
    is_single_turn: bool,
    user_content: Option<&Content>,
    include_thoughts_from_other_agents: bool,
) -> Result<Vec<Content>, ContentsError> {
    let rewind_filtered_events = adk_events::rewind::apply_rewinds(events);

    let raw_filtered_events: Vec<Event> = rewind_filtered_events
        .into_iter()
        .filter(|e| {
            let include_thoughts = include_thoughts_from_other_agents
                && crate::fencing::is_other_agent_reply(agent_name, e);
            should_include_event_in_context(current_branch, e, isolation_scope, include_thoughts)
        })
        .collect();

    let has_compaction_events = raw_filtered_events
        .iter()
        .any(|e| e.actions.compaction.is_some());

    let events_to_process = if has_compaction_events {
        let processed =
            crate::compaction::process_compaction_events(&raw_filtered_events, agent_name);
        crate::compaction::recover_compacted_function_calls(processed, &raw_filtered_events)
    } else {
        raw_filtered_events
    };

    let mut fc_author_by_id: HashMap<String, String> = HashMap::new();
    for e in &events_to_process {
        if let Some(content) = &e.content {
            for part in &content.parts {
                if let Some(fc) = &part.function_call {
                    if let Some(id) = &fc.id {
                        fc_author_by_id.insert(id.clone(), e.author.clone());
                    }
                }
            }
        }
    }

    let mut filtered_events: Vec<Event> = Vec::new();
    let mut accumulated_input = String::new();
    let mut accumulated_output = String::new();
    for i in 0..events_to_process.len() {
        let Some(event) = coalesce_transcription_event(
            &events_to_process,
            i,
            &mut accumulated_input,
            &mut accumulated_output,
        ) else {
            continue;
        };

        let mut is_other_reply = crate::fencing::is_other_agent_reply(agent_name, &event);

        if !is_other_reply {
            if let Some(content) = &event.content {
                for part in &content.parts {
                    let Some(fr) = &part.function_response else {
                        continue;
                    };
                    let Some(resp_id) = &fr.id else { continue };
                    let Some(call_author) = fc_author_by_id.get(resp_id) else {
                        continue;
                    };
                    if call_author != agent_name && call_author != "user" {
                        is_other_reply = true;
                        break;
                    }
                }
            }
        }

        if is_other_reply {
            if let Some(converted) = crate::fencing::present_other_agent_message(
                &event,
                include_thoughts_from_other_agents,
            ) {
                filtered_events.push(converted);
            }
        } else {
            filtered_events.push(event);
        }
    }

    let (filtered_events, _dropped_ids) = drop_orphaned_function_responses(&filtered_events);
    let result_events = rearrange_events_for_latest_function_response(filtered_events)?;
    let result_events = rearrange_events_for_async_function_responses_in_history(result_events)?;

    let mut contents: Vec<Content> = Vec::new();
    for event in &result_events {
        if let Some(content) = &event.content {
            contents.push(copy_content_for_request(
                content,
                !preserve_function_call_ids,
            ));
        }
    }

    if let Some(isolation_scope) = isolation_scope {
        if let Some(leading) =
            build_task_input_user_content(events, isolation_scope, is_single_turn, user_content)
        {
            contents.insert(0, leading);
        }
    }

    Ok(contents)
}

/// C0181-C0183: `_get_current_turn_contents` — the `include_contents='none'`
/// mode: finds the latest event that starts the current turn (a real user
/// turn, or another agent's reply — but never a direct `transfer_to_agent`
/// hop) and delegates to [`get_contents`] from there, dropping everything
/// before it.
#[allow(clippy::too_many_arguments)]
pub fn get_current_turn_contents(
    current_branch: Option<&str>,
    events: &[Event],
    agent_name: &str,
    preserve_function_call_ids: bool,
    isolation_scope: Option<&str>,
    is_single_turn: bool,
    user_content: Option<&Content>,
    include_thoughts_from_other_agents: bool,
) -> Result<Vec<Content>, ContentsError> {
    for i in (0..events.len()).rev() {
        let event = &events[i];
        let include_thoughts = include_thoughts_from_other_agents
            && crate::fencing::is_other_agent_reply(agent_name, event);
        let included = should_include_event_in_context(
            current_branch,
            event,
            isolation_scope,
            include_thoughts,
        );
        let is_turn_start =
            event.author == "user" || crate::fencing::is_other_agent_reply(agent_name, event);
        if included && is_turn_start && !is_direct_transfer(event) {
            return get_contents(
                current_branch,
                &events[i..],
                agent_name,
                preserve_function_call_ids,
                isolation_scope,
                is_single_turn,
                user_content,
                include_thoughts_from_other_agents,
            );
        }
    }
    Ok(Vec::new())
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
            ..Default::default()
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

    // --- build_task_input_user_content ---

    fn fc_with_args(id: &str, args: Vec<(&str, Value)>) -> Part {
        Part::function_call(FunctionCall {
            id: Some(id.to_string()),
            name: Some("delegate".to_string()),
            args: Some(args.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
            will_continue: None,
        })
    }

    #[test]
    fn build_task_input_finds_the_delegating_call_by_isolation_scope() {
        let events = vec![event_with_content(
            "coordinator",
            Content::new(
                "model",
                vec![fc_with_args(
                    "fc-1",
                    vec![("city", Value::String("Boston".to_string()))],
                )],
            ),
        )];
        let content = build_task_input_user_content(&events, "fc-1", false, None).unwrap();
        assert_eq!(content.role.as_deref(), Some("user"));
        assert!(content.parts[0].text.as_ref().unwrap().contains("Boston"));
    }

    #[test]
    fn build_task_input_falls_back_to_user_content_when_no_matching_call_exists() {
        let events = vec![event("coordinator")];
        let fallback = Content::user_text("do the task");
        let content =
            build_task_input_user_content(&events, "fc-missing", false, Some(&fallback)).unwrap();
        assert_eq!(content.parts[0].text.as_deref(), Some("do the task"));
    }

    #[test]
    fn build_task_input_returns_none_with_no_call_and_no_fallback() {
        let events = vec![event("coordinator")];
        assert!(build_task_input_user_content(&events, "fc-missing", false, None).is_none());
    }

    #[test]
    fn build_task_input_appends_the_single_turn_nudge() {
        let fallback = Content::user_text("do the task");
        let content =
            build_task_input_user_content(&[], "fc-missing", true, Some(&fallback)).unwrap();
        assert_eq!(content.parts.len(), 2);
        assert_eq!(content.parts[1].text.as_deref(), Some(SINGLE_TURN_NUDGE));
    }

    // --- get_contents ---

    #[test]
    fn get_contents_converts_content_events_and_strips_synthetic_ids() {
        let events = vec![
            event_with_content("user", Content::user_text("hi")),
            event_with_content(
                "model",
                Content::new("model", vec![fc_part("adk-1", "tool")]),
            ),
        ];
        let contents = get_contents(None, &events, "", false, None, false, None, false).unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hi"));
        assert!(contents[1].parts[0]
            .function_call
            .as_ref()
            .unwrap()
            .id
            .is_none());
    }

    #[test]
    fn get_contents_preserves_synthetic_ids_when_requested() {
        let events = vec![event_with_content(
            "model",
            Content::new("model", vec![fc_part("adk-1", "tool")]),
        )];
        let contents = get_contents(None, &events, "", true, None, false, None, false).unwrap();
        assert_eq!(
            contents[0].parts[0]
                .function_call
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some("adk-1")
        );
    }

    #[test]
    fn get_contents_drops_events_annulled_by_a_rewind() {
        let mut rewind_event = event("user");
        rewind_event.actions.rewind_before_invocation_id = Some("inv-1".to_string());
        rewind_event.content = Some(Content::user_text("this should be dropped"));
        let events = vec![
            event_with_content("user", Content::user_text("kept before rewind target")),
            rewind_event,
        ];
        // Both events share invocation_id "inv-1" (the `event()` helper's
        // default), so the rewind should annul everything including itself.
        let contents = get_contents(None, &events, "", false, None, false, None, false).unwrap();
        assert!(contents.is_empty());
    }

    #[test]
    fn get_contents_fences_another_agents_reply_as_user_context() {
        let events = vec![event_with_content(
            "agent_b",
            Content::new("model", vec![Part::text("some other agent's turn")]),
        )];
        let contents =
            get_contents(None, &events, "agent_a", false, None, false, None, false).unwrap();
        assert_eq!(contents.len(), 1);
        let text = contents[0].parts[1].text.as_deref().unwrap();
        assert!(text.starts_with("[agent_b] said:"));
    }

    #[test]
    fn get_contents_coalesces_split_input_transcription_fragments() {
        let transcription =
            |text: &str| Value::Map(vec![("text".to_string(), Value::String(text.to_string()))]);
        let mut first = event("user");
        first.input_transcription = Some(transcription("hello "));
        let mut second = event("user");
        second.input_transcription = Some(transcription("world"));

        let contents =
            get_contents(None, &[first, second], "", false, None, false, None, false).unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("hello world"));
    }

    #[test]
    fn get_contents_prepends_the_task_input_for_a_scoped_agent() {
        let events = vec![event_with_content(
            "coordinator",
            Content::new(
                "model",
                vec![fc_with_args(
                    "fc-1",
                    vec![("task", Value::String("summarize".to_string()))],
                )],
            ),
        )];
        let contents =
            get_contents(None, &events, "", false, Some("fc-1"), false, None, false).unwrap();
        // The delegating FC's own event is unscoped, so it's filtered out by
        // isolation_scope, leaving only the synthetic leading task input.
        assert_eq!(contents.len(), 1);
        assert!(contents[0].parts[0]
            .text
            .as_ref()
            .unwrap()
            .contains("summarize"));
    }

    // --- get_current_turn_contents ---

    #[test]
    fn get_current_turn_contents_starts_from_the_latest_user_turn() {
        let events = vec![
            event_with_content("user", Content::user_text("first turn")),
            event_with_content("model", Content::new("model", vec![Part::text("reply 1")])),
            event_with_content("user", Content::user_text("second turn")),
            event_with_content("model", Content::new("model", vec![Part::text("reply 2")])),
        ];
        let contents =
            get_current_turn_contents(None, &events, "", false, None, false, None, false).unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("second turn"));
        assert_eq!(contents[1].parts[0].text.as_deref(), Some("reply 2"));
    }

    #[test]
    fn get_current_turn_contents_is_empty_when_no_turn_start_qualifies() {
        let events = vec![event_with_content(
            "model",
            Content::new("model", vec![Part::text("no user turn yet")]),
        )];
        let contents =
            get_current_turn_contents(None, &events, "", false, None, false, None, false).unwrap();
        assert!(contents.is_empty());
    }

    #[test]
    fn get_current_turn_contents_skips_a_direct_transfer_event_as_a_turn_start() {
        let earlier_user_turn = event_with_content("user", Content::user_text("real user turn"));
        let mut transfer_call = event_with_content(
            "user",
            Content::new("user", vec![fc_part("fc-1", "transfer_to_agent")]),
        );
        transfer_call.actions.transfer_to_agent = Some("sub_agent".to_string());
        let events = vec![earlier_user_turn, transfer_call];

        let contents =
            get_current_turn_contents(None, &events, "", false, None, false, None, false).unwrap();
        // The trailing transfer event (authored "user" but a direct
        // transfer) is skipped as a turn start; the real user turn before it
        // anchors the current turn instead, so both events are included.
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0].parts[0].text.as_deref(), Some("real user turn"));
    }
}
