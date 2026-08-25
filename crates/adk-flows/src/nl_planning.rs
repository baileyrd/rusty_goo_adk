//! Capabilities C0176/C0179: `_nl_planning`'s request/response
//! processors, ported from
//! `google.adk.flows.llm_flows._nl_planning`.
//!
//! **Free functions taking `Option<&dyn BasePlanner>`, not
//! `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor` trait objects**:
//! the same "caller supplies the resolved bits" adaptation every sibling
//! processor in `llm_flow.rs` already makes (`basic`, `identity`,
//! `instructions`, `output_schema`, ...) — `LlmFlow::preprocess`/
//! `postprocess` call these directly with `self.planner.as_deref()`
//! rather than through an assembled processor list. The source's own
//! `_get_planner` (resolve from `InvocationContext.agent`, defaulting an
//! unrecognized `agent.planner` type to `PlanReActPlanner`) has no
//! equivalent here: `LlmAgent.planner` is still an opaque, unread
//! `Value` placeholder (C0088) with no real trait-object storage to
//! resolve *from* — the same C0092 tree-fusion gap `tools`/`tools_dict`
//! already discloses. [`crate::llm_flow::LlmFlow::with_planner`] is the
//! caller-supplied substitute.
//!
//! **The response processor's method-identity skip check, adapted**:
//! the source skips calling `process_planning_response` only when
//! `type(planner).process_planning_response is
//! BuiltInPlanner.process_planning_response` — an *unbound-method
//! identity* check, distinguishing "this literally is (or inherits,
//! unoverridden) `BuiltInPlanner`'s own no-op" from "this is a subclass
//! that overrides it." Rust has no subclassing, and this port has
//! exactly two concrete `BasePlanner` types (no `BuiltInPlanner`
//! subclass exists, or could meaningfully exist as a distinct type,
//! without actual subclassing) — so this adapts to a concrete-type
//! downcast (`planner.as_any().downcast_ref::<BuiltInPlanner>()`),
//! skipping exactly the same set of planners the source's identity
//! check would (the literal `BuiltInPlanner`, never `PlanReActPlanner`
//! or any other `BasePlanner` implementor), which is the *effective*
//! behavior two of the source's own regression tests
//! (`test_overridden_subclass_process_planning_response_called`/
//! `test_process_planning_response_not_called_without_override`) guard
//! — this port just has no subclass case for that identity check to
//! ever distinguish from the literal type in the first place.
//!
//! **`isolation_scope`, deliberately not copied onto the state-update
//! event**: the source's `Event(invocation_id=..., author=..., branch=...,
//! actions=...)` never sets an `isolation_scope` (it has no such
//! parameter on this constructor call) — unlike the model-response event
//! `llm_flow.rs::postprocess` builds in the same function, which does set
//! `event.isolation_scope = ctx.isolation_scope.clone()`. Matched here
//! exactly: [`apply_nl_planning_response`] returns bare
//! [`adk_events::EventActions`] for the caller to build an event
//! around, and `llm_flow.rs`'s own wiring leaves that event's
//! `isolation_scope` at its default `None`, not copying the pattern used
//! for the model-response event right next to it.

use adk_agents::context::CallbackContext;
use adk_agents::invocation_context::InvocationContext;
use adk_agents::readonly_context::ReadonlyContext;
use adk_events::EventActions;
use adk_models::llm_request::{Instructions, LlmRequest};
use adk_models::llm_response::LlmResponse;

use crate::planners::{BasePlanner, BuiltInPlanner};

/// `_NlPlanningRequestProcessor.run_async` — `planner` is the caller's
/// already-resolved planner (see the module doc); `None` is a no-op,
/// matching the source's `if not planner: return`.
pub fn apply_nl_planning_request(
    planner: Option<&dyn BasePlanner>,
    readonly_context: &ReadonlyContext,
    llm_request: &mut LlmRequest,
) {
    let Some(planner) = planner else {
        return;
    };

    if let Some(built_in) = planner.as_any().downcast_ref::<BuiltInPlanner>() {
        built_in.apply_thinking_config(llm_request);
        return;
    }

    if let Some(instruction) = planner.build_planning_instruction(readonly_context, llm_request) {
        llm_request.append_instructions(Instructions::Strings(vec![instruction]));
    }
    remove_thought_from_request(llm_request);
}

/// `_remove_thought_from_request`.
fn remove_thought_from_request(llm_request: &mut LlmRequest) {
    for content in &mut llm_request.contents {
        for part in &mut content.parts {
            part.thought = None;
        }
    }
}

/// `_NlPlanningResponse.run_async` — returns `Some(actions)` when the
/// planner produced a session-state delta the caller should wrap into a
/// state-update event (see the module doc for why `isolation_scope`
/// isn't set on it here). `None` means no event should be emitted for
/// this response, matching every one of the source's early `return`s
/// (no content/parts, no planner, or the `BuiltInPlanner` no-op skip).
pub fn apply_nl_planning_response(
    planner: Option<&dyn BasePlanner>,
    invocation_context: &InvocationContext,
    llm_response: &mut LlmResponse,
) -> Option<EventActions> {
    let has_parts = llm_response
        .content
        .as_ref()
        .is_some_and(|content| !content.parts.is_empty());
    if !has_parts {
        return None;
    }

    let planner = planner?;
    if planner.as_any().downcast_ref::<BuiltInPlanner>().is_some() {
        return None;
    }

    let mut callback_context = CallbackContext::new(invocation_context.clone());
    let response_parts = llm_response
        .content
        .as_ref()
        .map(|content| content.parts.clone())
        .unwrap_or_default();
    let processed_parts = planner.process_planning_response(&mut callback_context, response_parts);
    if let Some(processed_parts) = processed_parts {
        if let Some(content) = &mut llm_response.content {
            content.parts = processed_parts;
        }
    }

    if callback_context.state().has_delta() {
        Some(callback_context.into_actions())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planners::PlanReActPlanner;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_genai::content::{Content, Part};
    use rusty_serde::value::Value;

    fn invocation_context() -> InvocationContext {
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build()
    }

    fn readonly_context() -> ReadonlyContext {
        ReadonlyContext::new(invocation_context())
    }

    struct CustomPlanner;

    impl BasePlanner for CustomPlanner {
        fn build_planning_instruction(
            &self,
            _readonly_context: &ReadonlyContext,
            _llm_request: &LlmRequest,
        ) -> Option<String> {
            Some("Custom instruction".to_string())
        }

        fn process_planning_response(
            &self,
            callback_context: &mut CallbackContext,
            response_parts: Vec<Part>,
        ) -> Option<Vec<Part>> {
            callback_context
                .state_mut()
                .update([("planned".to_string(), Value::Bool(true))].into());
            Some(response_parts)
        }
    }

    struct SilentPlanner;

    impl BasePlanner for SilentPlanner {
        fn build_planning_instruction(
            &self,
            _readonly_context: &ReadonlyContext,
            _llm_request: &LlmRequest,
        ) -> Option<String> {
            None
        }

        fn process_planning_response(
            &self,
            _callback_context: &mut CallbackContext,
            response_parts: Vec<Part>,
        ) -> Option<Vec<Part>> {
            Some(response_parts)
        }
    }

    #[test]
    fn apply_nl_planning_request_is_a_no_op_without_a_planner() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content {
            role: Some("model".to_string()),
            parts: vec![Part {
                thought: Some(true),
                ..Default::default()
            }],
        });
        apply_nl_planning_request(None, &readonly_context(), &mut request);
        assert_eq!(request.config.system_instruction, None);
        assert_eq!(request.contents[0].parts[0].thought, Some(true));
    }

    #[test]
    fn apply_nl_planning_request_only_applies_thinking_config_for_a_built_in_planner() {
        let planner = BuiltInPlanner::new(Value::Bool(true));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content {
            role: Some("model".to_string()),
            parts: vec![Part {
                thought: Some(true),
                ..Default::default()
            }],
        });
        apply_nl_planning_request(Some(&planner), &readonly_context(), &mut request);
        assert_eq!(request.config.thinking_config, Some(Value::Bool(true)));
        assert_eq!(request.config.system_instruction, None);
        // `contents` is left untouched for a `BuiltInPlanner` — matches
        // the source's own `test_built_in_planner_content_list_unchanged`.
        assert_eq!(request.contents[0].parts[0].thought, Some(true));
    }

    #[test]
    fn apply_nl_planning_request_appends_the_instruction_for_a_non_built_in_planner() {
        let planner = PlanReActPlanner;
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.append_instructions(Instructions::Strings(vec![
            "Original instruction".to_string()
        ]));
        apply_nl_planning_request(Some(&planner), &readonly_context(), &mut request);
        let instruction = request.config.system_instruction.unwrap();
        assert!(instruction.starts_with("Original instruction\n\n"));
    }

    #[test]
    fn apply_nl_planning_request_strips_thought_flags_for_a_non_built_in_planner() {
        let planner = CustomPlanner;
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content {
            role: Some("model".to_string()),
            parts: vec![Part {
                thought: Some(true),
                ..Default::default()
            }],
        });
        apply_nl_planning_request(Some(&planner), &readonly_context(), &mut request);
        assert_eq!(request.contents[0].parts[0].thought, None);
    }

    #[test]
    fn apply_nl_planning_request_uses_a_custom_base_planners_instruction() {
        let planner = CustomPlanner;
        let mut request = LlmRequest::new("gemini-2.5-flash");
        apply_nl_planning_request(Some(&planner), &readonly_context(), &mut request);
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("Custom instruction")
        );
    }

    fn response_with_text(text: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content {
                role: Some("model".to_string()),
                parts: vec![Part::text(text)],
            }),
            ..Default::default()
        }
    }

    #[test]
    fn apply_nl_planning_response_is_a_no_op_without_content() {
        let mut response = LlmResponse::default();
        let result = apply_nl_planning_response(None, &invocation_context(), &mut response);
        assert!(result.is_none());
    }

    #[test]
    fn apply_nl_planning_response_is_a_no_op_without_a_planner() {
        let mut response = response_with_text("hi");
        let result = apply_nl_planning_response(None, &invocation_context(), &mut response);
        assert!(result.is_none());
    }

    #[test]
    fn apply_nl_planning_response_skips_the_built_in_planners_no_op() {
        let planner = BuiltInPlanner::new(Value::Bool(true));
        let mut response = response_with_text("hi");
        let result =
            apply_nl_planning_response(Some(&planner), &invocation_context(), &mut response);
        assert!(result.is_none());
        assert_eq!(
            response.content.unwrap().parts[0].text.as_deref(),
            Some("hi")
        );
    }

    #[test]
    fn apply_nl_planning_response_returns_none_without_a_state_delta() {
        let planner = SilentPlanner;
        let mut response = response_with_text("hi");
        let result =
            apply_nl_planning_response(Some(&planner), &invocation_context(), &mut response);
        assert!(result.is_none());
    }

    #[test]
    fn apply_nl_planning_response_replaces_parts_and_returns_the_state_delta() {
        let planner = CustomPlanner;
        let mut response = response_with_text("hi");
        let result =
            apply_nl_planning_response(Some(&planner), &invocation_context(), &mut response)
                .unwrap();
        assert_eq!(result.state_delta.get("planned"), Some(&Value::Bool(true)));
        assert_eq!(
            response.content.unwrap().parts[0].text.as_deref(),
            Some("hi")
        );
    }
}
