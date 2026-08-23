//! Capability C0171 (partial): the `agent_transfer` request processor,
//! ported from `google.adk.flows.llm_flows.agent_transfer`.
//!
//! Computes which sibling/parent agents the current agent may hand a
//! conversation off to, and builds the instruction text telling the model
//! about them.
//!
//! **Adaptation, disclosed**: the source's `_get_transfer_targets` reads
//! `mode`/`disallow_transfer_to_parent`/`disallow_transfer_to_peers`
//! straight off any `BaseAgent` via `getattr`/`hasattr` (`LlmAgent`
//! extends `BaseAgent` in the source's inheritance model). This port's
//! `BaseAgent` (Phase 2) and `LlmAgent` (Phase 2/4) are two separate,
//! unfused types — `LlmAgent` isn't wired into `BaseAgent`'s tree yet, the
//! standing blocker every Phase 4 processor in this crate discloses. So
//! [`get_transfer_targets`] takes an `llm_mode` callback the caller
//! supplies: `None` for a `BaseAgent` with no corresponding `LlmAgent`
//! config (`not hasattr(agent, 'mode')`/`not hasattr(agent,
//! 'disallow_transfer_to_parent')` in the source — a workflow node, for
//! instance), `Some(mode)` for one that has an `LlmAgent` (defaulting an
//! unset `mode` to [`AgentMode::Chat`], since the source's `getattr(...,
//! None)` and an actually-`None` `LlmAgent.mode` are indistinguishable to
//! this check either way).
//!
//! **Not** ported: `_get_incompatible_builtin_tool_error` (needs
//! `GoogleSearchTool`/`VertexAiSearchTool`/`EnterpriseWebSearchTool`,
//! Phase 8) and actually building/attaching a `TransferToAgentTool`
//! (needs `BaseTool`, also Phase 8, the same blocker `output_schema.rs`
//! already discloses for `SetModelResponseTool`).

use adk_agents::base_agent::BaseAgent;
use adk_agents::llm_agent::AgentMode;

/// `_get_transfer_targets`: the agents `agent` may transfer to — its
/// sub-agents (excluding single-turn/task-mode ones), then (if the parent
/// is itself LLM-orchestrated) the parent, then peer agents (the parent's
/// other sub-agents, excluding `agent` itself and excluding single-turn/
/// task-mode ones) — each gated by the corresponding `disallow_transfer_*`
/// flag.
pub fn get_transfer_targets(
    agent: &BaseAgent,
    llm_mode: &dyn Fn(&BaseAgent) -> Option<AgentMode>,
    disallow_transfer_to_parent: bool,
    disallow_transfer_to_peers: bool,
) -> Vec<BaseAgent> {
    let is_valid_target = |a: &BaseAgent| {
        !matches!(
            llm_mode(a),
            Some(AgentMode::Task) | Some(AgentMode::SingleTurn)
        )
    };

    let mut result: Vec<BaseAgent> = agent
        .sub_agents()
        .iter()
        .filter(|sub_agent| is_valid_target(sub_agent))
        .cloned()
        .collect();

    let Some(parent) = agent.parent_agent() else {
        return result;
    };
    if llm_mode(&parent).is_none() {
        return result;
    }

    if !disallow_transfer_to_parent {
        result.push(parent.clone());
    }

    if !disallow_transfer_to_peers {
        for peer in parent.sub_agents() {
            if peer.name() != agent.name() && is_valid_target(peer) {
                result.push(peer.clone());
            }
        }
    }

    result
}

/// `_build_transfer_instruction_body`: the agent-tree-agnostic transfer
/// instruction text — works with any `target_agents` exposing a name and
/// description.
pub fn build_transfer_instruction_body(tool_name: &str, target_agents: &[BaseAgent]) -> String {
    let mut available_agent_names: Vec<&str> = target_agents.iter().map(BaseAgent::name).collect();
    available_agent_names.sort_unstable();
    let formatted_agent_names = available_agent_names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");

    let target_agents_info = target_agents
        .iter()
        .map(|a| {
            format!(
                "\nAgent name: {}\nAgent description: {}\n",
                a.name(),
                a.description()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "\nYou have a list of other agents to transfer to:\n\n{target_agents_info}\n\nIf you \
         are the best to answer the question according to your description,\nyou can answer \
         it.\n\nIf another agent is better for answering the question according to its\n\
         description, call `{tool_name}` function to transfer the question to that\nagent. \
         When transferring, do not generate any text other than the function\ncall.\n\n\
         **NOTE**: the only available agents for `{tool_name}` function are\n\
         {formatted_agent_names}.\n"
    )
}

/// `_build_transfer_instructions`: the agent-tree variant — delegates to
/// [`build_transfer_instruction_body`], then appends a parent-transfer
/// suggestion if applicable. Empty for a task/single-turn agent (which
/// never initiates a transfer).
pub fn build_transfer_instructions(
    tool_name: &str,
    agent_mode: Option<AgentMode>,
    target_agents: &[BaseAgent],
    parent_agent: Option<&BaseAgent>,
    disallow_transfer_to_parent: bool,
) -> String {
    if matches!(
        agent_mode,
        Some(AgentMode::Task) | Some(AgentMode::SingleTurn)
    ) {
        return String::new();
    }

    let mut instructions = build_transfer_instruction_body(tool_name, target_agents);

    if let Some(parent) = parent_agent {
        if !disallow_transfer_to_parent {
            instructions.push_str(&format!(
                "\nIf neither you nor the other agents are best for the question, transfer to \
                 your parent agent {}.\n",
                parent.name()
            ));
        }
    }

    instructions
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::base_agent::NoopBehavior;

    fn agent(name: &str, description: &str) -> BaseAgent {
        BaseAgent::build(
            name,
            description,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            NoopBehavior,
        )
        .unwrap()
    }

    fn agent_with_sub_agents(
        name: &str,
        description: &str,
        sub_agents: Vec<BaseAgent>,
    ) -> BaseAgent {
        BaseAgent::build(
            name,
            description,
            sub_agents,
            Vec::new(),
            Vec::new(),
            NoopBehavior,
        )
        .unwrap()
    }

    fn always_chat(_: &BaseAgent) -> Option<AgentMode> {
        Some(AgentMode::Chat)
    }

    // --- get_transfer_targets ---

    #[test]
    fn includes_sub_agents_by_default() {
        let sub1 = agent("sub1", "d1");
        let sub2 = agent("sub2", "d2");
        let main = agent_with_sub_agents("main", "d", vec![sub1, sub2]);
        let targets = get_transfer_targets(&main, &always_chat, false, false);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn excludes_single_turn_and_task_mode_sub_agents() {
        let sub1 = agent("sub1", "d1");
        let sub2 = agent("sub2", "d2");
        let main = agent_with_sub_agents("main", "d", vec![sub1, sub2]);
        let mode_of = |a: &BaseAgent| {
            if a.name() == "sub2" {
                Some(AgentMode::SingleTurn)
            } else {
                Some(AgentMode::Chat)
            }
        };
        let targets = get_transfer_targets(&main, &mode_of, false, false);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].name(), "sub1");
    }

    #[test]
    fn includes_the_parent_and_peers_by_default() {
        let sub_agent = agent("sub_agent", "d");
        let main = agent_with_sub_agents("main", "d", vec![sub_agent]);
        let peer = agent("peer", "d");
        let _parent = agent_with_sub_agents("parent", "d", vec![main.clone(), peer]);

        let targets = get_transfer_targets(&main, &always_chat, false, false);
        let names: Vec<&str> = targets.iter().map(BaseAgent::name).collect();
        assert!(names.contains(&"sub_agent"));
        assert!(names.contains(&"parent"));
        assert!(names.contains(&"peer"));
    }

    #[test]
    fn excludes_the_parent_when_disallowed() {
        let main = agent("main", "d");
        let _parent = agent_with_sub_agents("parent", "d", vec![main.clone()]);
        let targets = get_transfer_targets(&main, &always_chat, true, false);
        assert!(!targets.iter().any(|a| a.name() == "parent"));
    }

    #[test]
    fn excludes_peers_when_disallowed() {
        let main = agent("main", "d");
        let peer = agent("peer", "d");
        let _parent = agent_with_sub_agents("parent", "d", vec![main.clone(), peer]);
        let targets = get_transfer_targets(&main, &always_chat, false, true);
        assert!(!targets.iter().any(|a| a.name() == "peer"));
        // The parent itself is still a valid target.
        assert!(targets.iter().any(|a| a.name() == "parent"));
    }

    #[test]
    fn a_non_llm_parent_is_not_a_transfer_target_and_has_no_peers() {
        let main = agent("main", "d");
        let peer = agent("peer", "d");
        let _parent = agent_with_sub_agents("parent", "d", vec![main.clone(), peer]);
        let no_llm_info = |_: &BaseAgent| None;
        let targets = get_transfer_targets(&main, &no_llm_info, false, false);
        assert!(targets.is_empty());
    }

    // --- build_transfer_instruction_body / build_transfer_instructions ---

    #[test]
    fn matches_the_sources_exact_instruction_text_without_a_parent() {
        let agent1 = agent("agent1", "First sub-agent");
        let agent2 = agent("agent2", "Second sub-agent");
        let body = build_transfer_instruction_body("transfer_to_agent", &[agent1, agent2]);

        let expected = "\nYou have a list of other agents to transfer to:\n\n\nAgent name: \
             agent1\nAgent description: First sub-agent\n\n\nAgent name: agent2\nAgent \
             description: Second sub-agent\n\n\nIf you are the best to answer the question \
             according to your description,\nyou can answer it.\n\nIf another agent is better \
             for answering the question according to its\ndescription, call `transfer_to_agent` \
             function to transfer the question to that\nagent. When transferring, do not \
             generate any text other than the function\ncall.\n\n**NOTE**: the only available \
             agents for `transfer_to_agent` function are\n`agent1`, `agent2`.";
        assert!(body.contains(expected));
    }

    #[test]
    fn matches_the_sources_exact_instruction_text_with_a_parent() {
        let sub_agent = agent("sub_agent", "Sub agent");
        let parent_agent = agent("parent_agent", "Parent agent");
        let instructions = build_transfer_instructions(
            "transfer_to_agent",
            None,
            &[sub_agent, parent_agent.clone()],
            Some(&parent_agent),
            false,
        );

        let expected = "\nYou have a list of other agents to transfer to:\n\n\nAgent name: \
             sub_agent\nAgent description: Sub agent\n\n\nAgent name: parent_agent\nAgent \
             description: Parent agent\n\n\nIf you are the best to answer the question \
             according to your description,\nyou can answer it.\n\nIf another agent is better \
             for answering the question according to its\ndescription, call `transfer_to_agent` \
             function to transfer the question to that\nagent. When transferring, do not \
             generate any text other than the function\ncall.\n\n**NOTE**: the only available \
             agents for `transfer_to_agent` function are\n`parent_agent`, `sub_agent`.\n\nIf \
             neither you nor the other agents are best for the question, transfer to your \
             parent agent parent_agent.";
        assert!(instructions.contains(expected));
    }

    #[test]
    fn is_empty_for_a_task_mode_agent() {
        let instructions = build_transfer_instructions(
            "transfer_to_agent",
            Some(AgentMode::Task),
            &[agent("a", "d")],
            None,
            false,
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn is_empty_for_a_single_turn_agent() {
        let instructions = build_transfer_instructions(
            "transfer_to_agent",
            Some(AgentMode::SingleTurn),
            &[agent("a", "d")],
            None,
            false,
        );
        assert!(instructions.is_empty());
    }

    #[test]
    fn omits_the_parent_suggestion_when_transfer_to_parent_is_disallowed() {
        let parent_agent = agent("parent_agent", "d");
        let instructions = build_transfer_instructions(
            "transfer_to_agent",
            None,
            std::slice::from_ref(&parent_agent),
            Some(&parent_agent),
            true,
        );
        assert!(!instructions.contains("transfer to your parent agent"));
    }
}
