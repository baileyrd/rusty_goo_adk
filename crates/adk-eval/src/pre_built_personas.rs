//! C0632 (part 2): `evaluation.simulation.pre_built_personas`, ported
//! from `google.adk.evaluation.simulation.pre_built_personas`.
//!
//! **`PreBuiltBehaviors`, adaptation disclosed**: the source is an
//! `enum.Enum` whose 11 members each carry a `UserBehavior` *instance* as
//! their value (Python enum values need not be hashable). This port
//! makes [`PreBuiltBehaviors`] a plain unit-variant enum instead, with
//! [`PreBuiltBehaviors::user_behavior`] building the corresponding
//! (owned, heap-allocating) [`UserBehavior`] on demand — the same
//! "enum + method returning the associated value" shape already used for
//! `eval_metrics::PrebuiltMetrics::as_str`, just returning an owned
//! struct instead of a `&'static str` since `UserBehavior` holds
//! `Vec<String>` fields with no `const`-friendly representation.
//!
//! **Behavior/persona text, verbatim**: every `name`/`description`/
//! `behavior_instructions`/`violation_rubrics` string below is copied
//! character-for-character from the source, including its own internal
//! inconsistencies (e.g. "Plan.When" with no space, "Response response"
//! doubled, "a a direct" doubled, "inconsist" for "inconsistent", an
//! unterminated `"` inside one `TONE_PROFESSIONAL` rubric, and
//! `END_NO_TROUBLESHOOTING`'s description starting with a leading
//! space) — these are grading-rubric/prompt text a judge model reads,
//! not code, so "fixing" them here would silently diverge this port's
//! prompts from the source's.
//!
//! **`_PreBuiltPersonas`, not ported as a public type**: the source's
//! enum is itself private (leading underscore) and used only inside
//! `get_default_persona_registry`. This port keeps the same shape —
//! three private functions ([`expert_persona`]/[`novice_persona`]/
//! [`evaluator_persona`]), not a public Rust enum, since nothing outside
//! this module reads them individually either.

use crate::user_simulator_personas::{UserBehavior, UserPersona, UserPersonaRegistry};

/// C0632: `pre_built_personas.PreBuiltBehaviors` — atomic behaviors that
/// can be mixed and matched to form personas. See this module's doc for
/// the enum-vs-instance-value adaptation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreBuiltBehaviors {
    AdvanceDetailOriented,
    AdvanceGoalOriented,
    AnswerRelevantOnly,
    AnswerAll,
    CorrectAgent,
    DoNotCorrectAgent,
    TroubleshootOnce,
    EndLimitedTroubleshooting,
    EndNoTroubleshooting,
    ToneProfessional,
    ToneConversational,
}

impl PreBuiltBehaviors {
    /// Builds the `UserBehavior` this variant represents.
    pub fn user_behavior(&self) -> UserBehavior {
        match self {
            Self::AdvanceDetailOriented => UserBehavior {
                name: "Advance in the Agent succeeds".to_string(),
                description: "The Generated User Response should stick to the Conversation \
                    Plan.When starting a new request, the Generated User Response should \
                    provide all the information required to accomplish a high-level goal."
                    .to_string(),
                behavior_instructions: vec![
                    "If the Agent succeeds, make the next request from the Conversation Plan."
                        .to_string(),
                    "Skip redundant requests already fulfilled by the Agent.".to_string(),
                    "When making a new request, state both the high-level goal you want to \
                        achieve next AND any additional details you need to achieve that goal."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Generated User Response repeats a high-level goal that was already \
                        completed in previous turns."
                        .to_string(),
                    "The Generated User Response provides details for a high-level goal that \
                        was already completed."
                        .to_string(),
                    "The Generated User Response response agrees to change the topic or \
                        perform a task not listed in the Conversation Plan."
                        .to_string(),
                    "The Generated User Response invents a new goal not present in the \
                        Conversation Plan."
                        .to_string(),
                    "The Generated User Response invents details (e.g., a made-up phone \
                        number or address) not provided in the Conversation Plan."
                        .to_string(),
                    "The Generated User Response only provides the high-level goal and the \
                        Agent has to ask for additional details."
                        .to_string(),
                    "The Generated User Response tries to accomplish more than one \
                        high-level task in a single turn."
                        .to_string(),
                ],
            },
            Self::AdvanceGoalOriented => UserBehavior {
                name: "Advance if the Agent succeeds".to_string(),
                description: "The Generated User Response should stick to the Conversation \
                    Plan as much as possible. It may deviate in response to Agent requests. \
                    The User Simulator starts with high-level goals, expecting the Agent to \
                    ask for specific details."
                    .to_string(),
                behavior_instructions: vec![
                    "If the Agent succeeds, make the next request from the Conversation Plan."
                        .to_string(),
                    "Skip redundant requests already fulfilled by the Agent.".to_string(),
                    "When making a request, state only the high-level goal you want to \
                        achieve next."
                        .to_string(),
                    "Do NOT provide any additional information related to the high-level \
                        goal. The Agent must ask for it."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Generated User Response repeats a high-level goal that was already \
                        completed in previous turns."
                        .to_string(),
                    "The Generated User Response provides details for a high-level goal that \
                        was already completed."
                        .to_string(),
                    "The Generated User Response invents a new goal not present in the \
                        Conversation Plan or in the Agent's messages."
                        .to_string(),
                    "The Generated User Response invents details (e.g., a made-up phone \
                        number or address) not provided in the Conversation Plan or in the \
                        Agent's messages."
                        .to_string(),
                    "The Generated User Response provides specific details for a high-level \
                        goal (email content, recipient address, phone numbers) BEFORE the \
                        Agent has explicitly asked for them."
                        .to_string(),
                    "The Generated User Response tries to accomplish more than one \
                        high-level task in a single turn."
                        .to_string(),
                ],
            },
            Self::AnswerRelevantOnly => UserBehavior {
                name: "Answer only relevant questions".to_string(),
                description: "The User Simulator should not answer questions that are not \
                    relevant to the high-level goals in the Conversation Plan (e.g., \"How \
                    is your day going?\"). If all questions the Agent asked are not \
                    relevant, the User Simulator should enforce the Conversation Plan (e.g., \
                    \"Please stick to writing the email.\")."
                    .to_string(),
                behavior_instructions: vec![
                    "Only answer the Agent's questions using information from the \
                        Conversation Plan."
                        .to_string(),
                    "Do NOT provide any additional information the Agent did not explicitly \
                        ask for."
                        .to_string(),
                    "If you do not have the information requested by the Agent, inform the \
                        Agent. Do NOT make up information that is not in the Conversation \
                        Plan."
                        .to_string(),
                    "Do NOT answer questions that are not relevant to the high level goals \
                        in the Conversation Plan."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Agent asked a question that is not relevant to the high-level goal \
                        and the Generated User Response responds to it."
                        .to_string(),
                ],
            },
            Self::AnswerAll => UserBehavior {
                name: "Answer all questions".to_string(),
                description: "The User Simulator should address EVERY question that the \
                    Agent asked, e.g., if the Agent asks \"How is your day going?\", the \
                    User Simulator should respond."
                    .to_string(),
                behavior_instructions: vec![
                    "Only answer the Agent's questions using information from the \
                        Conversation Plan."
                        .to_string(),
                    "Do NOT provide any additional information the Agent did not explicitly \
                        ask for."
                        .to_string(),
                    "If you do not have the information requested by the Agent, inform the \
                        Agent. Do NOT make up information that is not in the Conversation \
                        Plan. Acknowledge you don't know the information."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Agent asked a question (or multiple questions), and the Generated \
                        User Response failed to address one or all of them."
                        .to_string(),
                    "The Agent asked for information NOT in the Conversation Plan, and the \
                        Generated User Response made up an answer instead of stating, e.g., \
                        \"I don't know\" or \"I don't have that info.\""
                        .to_string(),
                ],
            },
            Self::CorrectAgent => UserBehavior {
                name: "Correct the Agent if it makes a mistake".to_string(),
                description: "The User Simulator should catch and correct the Agent's \
                    mistakes."
                    .to_string(),
                behavior_instructions: vec![
                    "Challenge illogical or incorrect statements made by the Agent.".to_string(),
                    "If the Agent did an incorrect operation, ask the Agent to fix it.".to_string(),
                ],
                violation_rubrics: vec![
                    "The Agent provided incorrect information, and the Generated User \
                        Response continues as if it was correct."
                        .to_string(),
                    "The Agent made a dangerous assumption (e.g., sending an email without \
                        asking for the content first), and the Generated User Response \
                        continues without correcting the Agent."
                        .to_string(),
                ],
            },
            Self::DoNotCorrectAgent => UserBehavior {
                name: "Do not correct the Agent".to_string(),
                description: "The User Simulator should end the conversation when the Agent \
                    provides an illogical or incorrect statement."
                    .to_string(),
                behavior_instructions: vec![
                    "If the Agent made an illogical or incorrect statement, end the \
                        conversation with `{{ stop_signal }}`."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Agent makes a mistake or an assumption and the Generated User \
                        Response corrects the Agent."
                        .to_string(),
                ],
            },
            Self::TroubleshootOnce => UserBehavior {
                name: "Troubleshoot once (if necessary)".to_string(),
                description: "The User Simulator should only troubleshoot the Agent ONCE. \
                    Troubleshooting is defined as the User Simulator helping the Agent after \
                    the Agent fails to execute an action (e.g., calls a function incorrectly) \
                    or fails to provide a response expected by the Conversation Plan. \
                    Answering a clarification question from the Agent is NOT \
                    troubleshooting. NOTE: Please check the conversation history count for \
                    Agent errors."
                    .to_string(),
                behavior_instructions: vec![
                    "If the Agent failed to complete a request for the first time, \
                        troubleshoot the failure."
                        .to_string(),
                    "You should only troubleshoot ONCE per conversation. DO NOT \
                        troubleshoot again if the Conversation History shows that the you \
                        have already tried to troubleshoot any request."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Generated User Response ends the conversation immediately after \
                        the first Agent failure."
                        .to_string(),
                    "On the second Agent failure, the Generated User Response response \
                        continues the conversation without using `{{ stop_signal }}`."
                        .to_string(),
                    "After the second Agent failure, the Generated User Response tries to \
                        continue the conversation or continues addressing failures without \
                        using `{{ stop_signal }}`."
                        .to_string(),
                ],
            },
            Self::EndLimitedTroubleshooting => UserBehavior {
                name: "End the conversation appropriately".to_string(),
                description: "A conversation is complete if ANY of the following stop \
                    conditions are true:\n- The Agent has confirmed the completion of all \
                    the high-level goals in the Conversation Plan.\n- The Agent \
                    successfully transferred the User Simulator to a human/live agent.\n- \
                    The Agent failed more than once.\nThe Agent fails if it is unable to \
                    execute an action (e.g., calls a function incorrectly) or fails to \
                    provide a response expected by the Conversation Plan. Asking a \
                    clarification question is not a failure."
                    .to_string(),
                behavior_instructions: vec![
                    "End the conversation only when any of the stopping conditions are met; \
                        do NOT end prematurely."
                        .to_string(),
                    "When ending the conversation because the Agent has completed all the \
                        high-level goals, you must wait until the Agent has confirmed the \
                        completion of all the goals before ending."
                        .to_string(),
                    "Output `{{ stop_signal }}` as part of your response to indicate that \
                        the conversation with the Agent is over."
                        .to_string(),
                    "Pay attention to the Conversation History and count the number of \
                        Agent failures. A second failure should trigger the end of the \
                        conversation."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The conversation meets one of the stop conditions above, but the \
                        Generated User Response did not use `{{ stop_signal }}`."
                        .to_string(),
                    "The Generated User Response used `{{ stop_signal }}` but the \
                        conversation does not meet any of the stop conditions above."
                        .to_string(),
                ],
            },
            Self::EndNoTroubleshooting => UserBehavior {
                name: "End the conversation appropriately".to_string(),
                description: " A conversation is considered completed if ANY of the \
                    following stop conditions are true:\n- The Agent has confirmed the \
                    completion of all the high-level goals in the Conversation Plan.\n- The \
                    Agent successfully transferred the User Simulator to a human/live \
                    agent.\n- The Agent failed.\nThe Agent fails if it is unable to execute \
                    an action (e.g., calls a function incorrectly) or fails to provide a \
                    response expected by the Conversation Plan. Asking a clarification \
                    question is not a failure."
                    .to_string(),
                behavior_instructions: vec![
                    "End the conversation when any of the stopping conditions are met; do \
                        NOT end prematurely."
                        .to_string(),
                    "When ending the conversation because the Agent has completed all the \
                        high-level goals, you must wait until the Agent has confirmed the \
                        completion of all the goals before ending."
                        .to_string(),
                    "Output `{{ stop_signal }}` as part of your response to indicate that \
                        the conversation with the Agent is over."
                        .to_string(),
                    "Pay attention to the last Agent message in the Conversation History. \
                        If the Agent message contains a failure, end the conversation."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The conversation meets one of the stop conditions above, but the \
                        Generated User Response did not use `{{ stop_signal }}`."
                        .to_string(),
                    "The Generated User Response used `{{ stop_signal }}` but the \
                        conversation does not meet any of the stop conditions above."
                        .to_string(),
                    "On the first Agent failure, the Generated User Response continues the \
                        conversation without using `{{ stop_signal }}`."
                        .to_string(),
                    "After the first Agent failure, the Generated User Response tries to \
                        continue the conversation without using `{{ stop_signal }}`."
                        .to_string(),
                ],
            },
            Self::ToneProfessional => UserBehavior {
                name: "Professional tone".to_string(),
                description: "The User Simulator use clear, technical language. NOTE: \
                    `{{ stop_signal }}` is appropriate language."
                    .to_string(),
                behavior_instructions: vec![
                    "The User Simulator should use clear, technical language.".to_string(),
                    "Avoid slang, frequent abbreviations, emojis, or excessive social \
                        filler and personal asides."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Generated User Response includes slang (e.g., \"gimme,\" \
                        \"kinda,\" \"lol\"), frequent abbreviations (e.g., \"info,\" \
                        \"btw\"), or emojis."
                        .to_string(),
                    "The Generated User Response includes significant social filler or \
                        personal asides, e.g., \"Hi there! I hope you're having a good day."
                        .to_string(),
                    "The Generated User Response is a \"wall of text\" where a a direct \
                        sentence would suffice."
                        .to_string(),
                    "The tone of the Generated User Response is inconsist with previous \
                        user turns (if present)."
                        .to_string(),
                ],
            },
            Self::ToneConversational => UserBehavior {
                name: "Conversational tone".to_string(),
                description: "The User Simulator sounds informal. NOTE: \
                    `{{ stop_signal }}` is appropriate language."
                    .to_string(),
                behavior_instructions: vec![
                    "The User Simulator should sound like a normal human having a casual \
                        conversation."
                        .to_string(),
                    "Avoid answers that are too formal in nature or employ overly polite \
                        phrases and expressions."
                        .to_string(),
                    "Avoid answers that lack natural conversational framing, for example, \
                        sterile or purely functional responses."
                        .to_string(),
                ],
                violation_rubrics: vec![
                    "The Generated User Response is sterile and purely functional (direct \
                        commands) with no natural conversational framing."
                        .to_string(),
                    "The Generated User Response is too formal in nature, employing overly \
                        polite phrases and expressions."
                        .to_string(),
                    "The Generated User Response is a \"wall of text\" where a simple \
                        sentence would suffice."
                        .to_string(),
                    "The tone of the Generated User Response is inconsist with previous \
                        user turns (if present)."
                        .to_string(),
                ],
            },
        }
    }
}

/// `_PreBuiltPersonas.EXPERT`.
fn expert_persona() -> UserPersona {
    UserPersona {
        id: "EXPERT".to_string(),
        description: "An Expert knows exactly what they want and views the Agent as a tool \
            to execute their commands as efficiently as possible. Experts have little \
            patience for chit-chat or unnecessary questions."
            .to_string(),
        behaviors: vec![
            PreBuiltBehaviors::AdvanceDetailOriented.user_behavior(),
            PreBuiltBehaviors::AnswerRelevantOnly.user_behavior(),
            PreBuiltBehaviors::CorrectAgent.user_behavior(),
            PreBuiltBehaviors::TroubleshootOnce.user_behavior(),
            PreBuiltBehaviors::EndLimitedTroubleshooting.user_behavior(),
            PreBuiltBehaviors::ToneProfessional.user_behavior(),
        ],
    }
}

/// `_PreBuiltPersonas.NOVICE`.
fn novice_persona() -> UserPersona {
    UserPersona {
        id: "NOVICE".to_string(),
        description: "A Novice is trying to solve a problem they don't fully understand, \
            and they rely heavily on the Agent for guidance. Novices are patient with the \
            Agent's questions, but are unable to troubleshoot the Agent's mistakes. Novices \
            are also unable to correct the Agent."
            .to_string(),
        behaviors: vec![
            PreBuiltBehaviors::AdvanceGoalOriented.user_behavior(),
            PreBuiltBehaviors::DoNotCorrectAgent.user_behavior(),
            PreBuiltBehaviors::AnswerAll.user_behavior(),
            PreBuiltBehaviors::EndNoTroubleshooting.user_behavior(),
            PreBuiltBehaviors::ToneConversational.user_behavior(),
        ],
    }
}

/// `_PreBuiltPersonas.EVALUATOR`.
fn evaluator_persona() -> UserPersona {
    UserPersona {
        id: "EVALUATOR".to_string(),
        description: "An Evaluator is trying to assess whether the Agent can help \
            accomplish the goals in the Conversation Plan."
            .to_string(),
        behaviors: vec![
            PreBuiltBehaviors::AdvanceDetailOriented.user_behavior(),
            PreBuiltBehaviors::AnswerRelevantOnly.user_behavior(),
            PreBuiltBehaviors::EndNoTroubleshooting.user_behavior(),
            PreBuiltBehaviors::DoNotCorrectAgent.user_behavior(),
            PreBuiltBehaviors::ToneConversational.user_behavior(),
        ],
    }
}

/// C0632: `pre_built_personas.get_default_persona_registry`.
pub fn get_default_persona_registry() -> UserPersonaRegistry {
    let mut registry = UserPersonaRegistry::new();
    registry.register_persona("EXPERT", expert_persona());
    registry.register_persona("NOVICE", novice_persona());
    registry.register_persona("EVALUATOR", evaluator_persona());
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_three_pre_built_personas() {
        let registry = get_default_persona_registry();
        assert!(registry.get_persona("EXPERT").is_ok());
        assert!(registry.get_persona("NOVICE").is_ok());
        assert!(registry.get_persona("EVALUATOR").is_ok());
        assert_eq!(registry.get_registered_personas().len(), 3);
    }

    #[test]
    fn expert_has_six_behaviors() {
        assert_eq!(expert_persona().behaviors.len(), 6);
    }

    #[test]
    fn novice_has_five_behaviors() {
        assert_eq!(novice_persona().behaviors.len(), 5);
    }

    #[test]
    fn evaluator_has_five_behaviors() {
        assert_eq!(evaluator_persona().behaviors.len(), 5);
    }

    #[test]
    fn every_pre_built_behavior_has_non_empty_content() {
        let all = [
            PreBuiltBehaviors::AdvanceDetailOriented,
            PreBuiltBehaviors::AdvanceGoalOriented,
            PreBuiltBehaviors::AnswerRelevantOnly,
            PreBuiltBehaviors::AnswerAll,
            PreBuiltBehaviors::CorrectAgent,
            PreBuiltBehaviors::DoNotCorrectAgent,
            PreBuiltBehaviors::TroubleshootOnce,
            PreBuiltBehaviors::EndLimitedTroubleshooting,
            PreBuiltBehaviors::EndNoTroubleshooting,
            PreBuiltBehaviors::ToneProfessional,
            PreBuiltBehaviors::ToneConversational,
        ];
        assert_eq!(all.len(), 11);
        for behavior in all {
            let user_behavior = behavior.user_behavior();
            assert!(!user_behavior.name.is_empty());
            assert!(!user_behavior.description.is_empty());
            assert!(!user_behavior.behavior_instructions.is_empty());
            assert!(!user_behavior.violation_rubrics.is_empty());
        }
    }

    #[test]
    fn tone_professional_preserves_the_sources_unterminated_quote() {
        let behavior = PreBuiltBehaviors::ToneProfessional.user_behavior();
        assert!(behavior.violation_rubrics[1].contains("\"Hi there! I hope you're having a"));
    }

    #[test]
    fn end_no_troubleshooting_preserves_the_sources_leading_space() {
        let behavior = PreBuiltBehaviors::EndNoTroubleshooting.user_behavior();
        assert!(behavior.description.starts_with(" A conversation"));
    }
}
