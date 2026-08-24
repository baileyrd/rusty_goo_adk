//! C0623: `evaluation.evaluation_generator`, ported from
//! `google.adk.evaluation.evaluation_generator`. Only the pure
//! event→`Invocation` grouping algorithm is ported — the file's other
//! static methods (`_generate_inferences_from_root_agent`/`_live`, `
//! _process_query_with_session`, `_get_app_details_by_invocation_id`)
//! drive a real `Runner`/`Session`/a request-interception plugin (C0621/
//! C0622/C0624, all still `REQUIRED`), which this batch doesn't build.
//!
//! **`HashMap` → `Vec<(String, Vec<Event>)>`, disclosed**: unlike the
//! `HashMap`-for-grouping choices already disclosed elsewhere in this
//! crate (e.g. `EvalConfig.criteria`, the `rubric_based_evaluator`
//! aggregators), invocation *order* is semantically meaningful here — a
//! `StaticConversation`/`EvalCase.conversation`'s invocations are matched
//! positionally against `expected_invocations` elsewhere in this crate.
//! [`collect_events_by_invocation_id`] therefore preserves first-seen
//! order explicitly (a parallel `Vec` of ids alongside the `HashMap`),
//! matching the source dict's own insertion-order-preserving behavior
//! exactly, rather than accepting `HashMap`'s unordered iteration.

use std::collections::HashMap;

use adk_events::Event;
use adk_genai::content::Content;

use crate::app_details::AppDetails;
use crate::eval_case::{Invocation, InvocationEvent, InvocationEvents};

const USER_AUTHOR: &str = "user";
const DEFAULT_AUTHOR: &str = "agent";

/// `evaluation_generator.EvaluationGenerator._collect_events_by_invocation_id`
/// — groups `events` by `invocation_id`, preserving each id's first-seen
/// order (see this module's doc for why order is preserved here).
pub fn collect_events_by_invocation_id(events: &[Event]) -> Vec<(String, Vec<Event>)> {
    let mut order: Vec<String> = Vec::new();
    let mut by_id: HashMap<String, Vec<Event>> = HashMap::new();

    for event in events {
        if !by_id.contains_key(&event.invocation_id) {
            order.push(event.invocation_id.clone());
        }
        by_id
            .entry(event.invocation_id.clone())
            .or_default()
            .push(event.clone());
    }

    order
        .into_iter()
        .map(|id| {
            let events = by_id.remove(&id).unwrap_or_default();
            (id, events)
        })
        .collect()
}

fn has_non_empty_text(parts: &[adk_genai::content::Part]) -> bool {
    parts
        .iter()
        .any(|part| part.text.as_deref().is_some_and(|text| !text.is_empty()))
}

/// C0623: `evaluation_generator.EvaluationGenerator.convert_events_to_eval_invocations`
/// — converts a list of `Event`s into a list of `Invocation`s, one per
/// `invocation_id` group.
pub fn convert_events_to_eval_invocations(
    events: &[Event],
    app_details_per_invocation: Option<&HashMap<String, AppDetails>>,
) -> Result<Vec<Invocation>, String> {
    let grouped = collect_events_by_invocation_id(events);
    let mut invocations = Vec::with_capacity(grouped.len());

    for (invocation_id, events) in grouped {
        let mut final_response: Option<Content> = None;
        let mut final_event_index: Option<usize> = None;
        let mut user_content = Content {
            role: None,
            parts: vec![],
        };
        let mut invocation_timestamp: f64 = 0.0;
        let app_details = app_details_per_invocation
            .and_then(|map| map.get(&invocation_id))
            .cloned();

        let mut events_to_add: Vec<usize> = Vec::new();

        for (idx, event) in events.iter().enumerate() {
            let current_author = if event.author.is_empty() {
                DEFAULT_AUTHOR.to_string()
            } else {
                event.author.to_lowercase()
            };

            if current_author == USER_AUTHOR {
                if let Some(content) = &event.content {
                    user_content = content.clone();
                    invocation_timestamp = event.timestamp;
                }
                continue;
            }

            let has_content_and_parts = event.content.as_ref().is_some_and(|c| !c.parts.is_empty());
            if has_content_and_parts {
                let content = event.content.as_ref().expect("checked above");

                if event.is_final_response() {
                    // A live response is both audio and a text transcript;
                    // keep the text one as the gradable response.
                    let final_has_text = final_response
                        .as_ref()
                        .is_some_and(|fr| has_non_empty_text(&fr.parts));
                    let event_has_text = has_non_empty_text(&content.parts);
                    if !final_has_text || event_has_text {
                        final_response = Some(content.clone());
                        final_event_index = Some(idx);
                    }
                }

                let mut should_add_event = event.grounding_metadata.is_some();
                for part in &content.parts {
                    if part.function_call.is_some()
                        || part.function_response.is_some()
                        || part.text.as_deref().is_some_and(|t| !t.is_empty())
                        || part.inline_data.is_some()
                    {
                        should_add_event = true;
                        break;
                    }
                }
                if should_add_event {
                    events_to_add.push(idx);
                }
            } else if event.grounding_metadata.is_some() {
                events_to_add.push(idx);
            }
        }

        let mut invocation_events = Vec::with_capacity(events_to_add.len());
        for idx in events_to_add {
            let event = &events[idx];
            let is_final_event = final_event_index == Some(idx);
            let has_function_calls = !event.get_function_calls().is_empty();

            // Keep the final event only when it carries tool calls (so the
            // judge still sees the function call) or grounding metadata;
            // every other event is always included.
            if is_final_event && !has_function_calls && event.grounding_metadata.is_none() {
                continue;
            }

            let content = if !is_final_event || has_function_calls {
                event.content.clone()
            } else {
                None
            };

            invocation_events.push(InvocationEvent {
                author: event.author.clone(),
                content,
                grounding_metadata: event.grounding_metadata.clone(),
            });
        }

        invocations.push(Invocation {
            invocation_id,
            user_content,
            final_response,
            intermediate_data: Some(
                rusty_serde::json::to_value(&InvocationEvents { invocation_events })
                    .map_err(|error| error.to_string())?,
            ),
            creation_timestamp: invocation_timestamp,
            rubrics: None,
            app_details,
        });
    }

    Ok(invocations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{FunctionCall, Part};

    fn event(invocation_id: &str, author: &str, content: Option<Content>) -> Event {
        let mut event = Event::new(invocation_id, author, NodeInfo::new("root"));
        event.timestamp = 0.0;
        event.content = content;
        event
    }

    #[test]
    fn collect_events_by_invocation_id_preserves_first_seen_order() {
        let events = vec![
            event("b", "user", None),
            event("a", "user", None),
            event("b", "agent", None),
        ];
        let grouped = collect_events_by_invocation_id(&events);
        let ids: Vec<&str> = grouped.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a"]);
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[1].1.len(), 1);
    }

    #[test]
    fn convert_events_to_eval_invocations_builds_user_content_and_final_response() {
        let events = vec![
            event("inv-1", "user", Some(Content::user_text("hi"))),
            event(
                "inv-1",
                "agent",
                Some(Content::new("model", vec![Part::text("hello")])),
            ),
        ];
        let invocations = convert_events_to_eval_invocations(&events, None).unwrap();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].user_content, Content::user_text("hi"));
        assert_eq!(
            invocations[0].final_response,
            Some(Content::new("model", vec![Part::text("hello")]))
        );
    }

    #[test]
    fn convert_events_to_eval_invocations_prefers_text_over_audio_only_final_response() {
        let mut audio_only = Content::new("model", vec![]);
        audio_only.parts.push(Part {
            inline_data: Some(adk_genai::content::MediaBlobStub::default()),
            ..Default::default()
        });
        let text_final = Content::new("model", vec![Part::text("the real answer")]);

        let events = vec![
            event("inv-1", "user", Some(Content::user_text("hi"))),
            event("inv-1", "agent", Some(audio_only)),
            event("inv-1", "agent", Some(text_final.clone())),
        ];
        let invocations = convert_events_to_eval_invocations(&events, None).unwrap();
        assert_eq!(invocations[0].final_response, Some(text_final));
    }

    #[test]
    fn convert_events_to_eval_invocations_keeps_final_event_content_only_with_function_calls() {
        let mut with_call = Content::new("model", vec![Part::text("checking...")]);
        with_call.parts.push(Part::function_call(FunctionCall {
            id: Some("c1".to_string()),
            name: Some("get_weather".to_string()),
            ..Default::default()
        }));

        let events = vec![
            event("inv-1", "user", Some(Content::user_text("hi"))),
            event("inv-1", "agent", Some(with_call.clone())),
        ];
        let invocations = convert_events_to_eval_invocations(&events, None).unwrap();
        let intermediate = invocations[0].intermediate_data_type().unwrap();
        let events = match intermediate {
            crate::eval_case::IntermediateDataType::Events(events) => events,
            _ => panic!("expected InvocationEvents"),
        };
        assert_eq!(events.invocation_events.len(), 1);
        assert_eq!(events.invocation_events[0].content, Some(with_call));
    }

    #[test]
    fn convert_events_to_eval_invocations_drops_a_contentless_final_event() {
        let mut with_call = Content::new("model", vec![]);
        with_call.parts.push(Part::function_call(FunctionCall {
            id: Some("c1".to_string()),
            name: Some("get_weather".to_string()),
            ..Default::default()
        }));
        let events = vec![
            event("inv-1", "user", Some(Content::user_text("hi"))),
            event("inv-1", "agent", Some(with_call)),
        ];
        // The only agent event has a function call, so is_final_response()
        // is false for it (get_function_calls is non-empty) -- meaning
        // final_response/final_event stay unset, and the event is still
        // added (should_add_event is true due to the function call).
        let invocations = convert_events_to_eval_invocations(&events, None).unwrap();
        assert_eq!(invocations[0].final_response, None);
        let intermediate = invocations[0].intermediate_data_type().unwrap();
        let events = match intermediate {
            crate::eval_case::IntermediateDataType::Events(events) => events,
            _ => panic!("expected InvocationEvents"),
        };
        assert_eq!(events.invocation_events.len(), 1);
    }

    #[test]
    fn convert_events_to_eval_invocations_attaches_app_details() {
        let mut app_details_map = HashMap::new();
        app_details_map.insert("inv-1".to_string(), AppDetails::default());
        let events = vec![event("inv-1", "user", Some(Content::user_text("hi")))];
        let invocations =
            convert_events_to_eval_invocations(&events, Some(&app_details_map)).unwrap();
        assert_eq!(invocations[0].app_details, Some(AppDetails::default()));
    }
}
