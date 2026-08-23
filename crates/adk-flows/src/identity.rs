//! Capability C0169: the `identity` request processor's core logic, ported
//! from `google.adk.flows.llm_flows.identity`.
//!
//! **Scope, disclosed**: same shape as `basic.rs` (see that module's own
//! doc for the full rationale) — [`apply_identity`] is a free function,
//! not yet a real [`crate::processor::BaseLlmRequestProcessor`] reading
//! through `InvocationContext`, since `LlmAgent` doesn't implement
//! `AgentBehavior` yet.
//!
//! **Adaptation, disclosed**: the source reads `agent.name`/
//! `agent.description` — fields `LlmAgent` inherits from `BaseAgent` in
//! the source, but this port's standalone `LlmAgent` struct (see
//! `llm_agent.rs`'s own module doc) has neither field yet, since it isn't
//! wired into `BaseAgent`'s tree. [`apply_identity`] takes `agent_name`/
//! `agent_description` as explicit parameters instead of reading them off
//! an `LlmAgent` value — once real tree placement lands, a caller passes
//! `agent.name()`/`agent.description()` (from `BaseAgent`) through
//! unchanged; this function's own logic doesn't need to change.

use adk_agents::llm_agent::AgentMode;
use adk_models::llm_request::{Instructions, LlmRequest};

/// C0169: the identity instruction text, or `None` for a single-turn
/// agent (which gets no identity instruction at all, matching the
/// source's `mode != 'single_turn'` gate).
pub fn identity_instruction(
    agent_name: &str,
    agent_description: Option<&str>,
    mode: Option<AgentMode>,
) -> Option<String> {
    if mode == Some(AgentMode::SingleTurn) {
        return None;
    }
    let mut instruction = format!("You are an agent. Your internal name is \"{agent_name}\".");
    if let Some(description) = agent_description.filter(|d| !d.is_empty()) {
        instruction.push_str(&format!(" The description about you is \"{description}\"."));
    }
    Some(instruction)
}

/// C0169: `_build_identity_request` — appends the identity instruction to
/// `llm_request`, if any. See the module doc for what's deferred.
pub fn apply_identity(
    agent_name: &str,
    agent_description: Option<&str>,
    mode: Option<AgentMode>,
    llm_request: &mut LlmRequest,
) {
    if let Some(instruction) = identity_instruction(agent_name, agent_description, mode) {
        llm_request.append_instructions(Instructions::Strings(vec![instruction]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_instruction_includes_name_and_description() {
        let instruction =
            identity_instruction("weather_agent", Some("reports the weather"), None).unwrap();
        assert_eq!(
            instruction,
            "You are an agent. Your internal name is \"weather_agent\". The description about \
             you is \"reports the weather\"."
        );
    }

    #[test]
    fn identity_instruction_omits_the_description_sentence_when_absent() {
        let instruction = identity_instruction("weather_agent", None, None).unwrap();
        assert_eq!(
            instruction,
            "You are an agent. Your internal name is \"weather_agent\"."
        );
    }

    #[test]
    fn identity_instruction_omits_the_description_sentence_when_empty() {
        let instruction = identity_instruction("weather_agent", Some(""), None).unwrap();
        assert_eq!(
            instruction,
            "You are an agent. Your internal name is \"weather_agent\"."
        );
    }

    #[test]
    fn identity_instruction_is_none_for_a_single_turn_agent() {
        assert!(identity_instruction("weather_agent", None, Some(AgentMode::SingleTurn)).is_none());
    }

    #[test]
    fn identity_instruction_is_present_for_chat_and_task_modes() {
        assert!(identity_instruction("weather_agent", None, Some(AgentMode::Chat)).is_some());
        assert!(identity_instruction("weather_agent", None, Some(AgentMode::Task)).is_some());
    }

    #[test]
    fn apply_identity_appends_the_instruction_to_the_request() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        apply_identity(
            "weather_agent",
            Some("reports the weather"),
            None,
            &mut request,
        );
        assert!(request
            .config
            .system_instruction
            .as_deref()
            .unwrap()
            .contains("weather_agent"));
    }

    #[test]
    fn apply_identity_is_a_no_op_for_a_single_turn_agent() {
        let mut request = LlmRequest::new("gemini-2.5-flash");
        apply_identity(
            "weather_agent",
            None,
            Some(AgentMode::SingleTurn),
            &mut request,
        );
        assert!(request.config.system_instruction.is_none());
    }
}
