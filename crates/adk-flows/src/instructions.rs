//! Capability C0170: the `instructions` request processor's core logic,
//! ported from `google.adk.flows.llm_flows.instructions`.
//!
//! **Scope, disclosed**: same shape as `basic.rs`/`identity.rs` (see
//! those modules' own docs for the full rationale) —
//! [`build_instructions`] is a free function taking `&LlmAgent` directly,
//! not yet a real [`crate::processor::BaseLlmRequestProcessor`].
//!
//! **`global_instruction`, now a real root-agent walk**: the source
//! resolves the deprecated `global_instruction` from the invocation's
//! *root* agent (`root_agent.canonical_global_instruction`) — a
//! cross-tree broadcast. [`build_instructions`] now does the same via
//! `ctx.agent()`'s `.root_agent()` plus `AgentBehavior::as_any`'s
//! downcast onto `crate::llm_flow::LlmFlow` (see `base_agent.rs`'s own
//! doc for that mechanism) — mirroring the source's
//! `hasattr(root_agent, 'global_instruction')` gate: if the root isn't
//! LLM-backed, no `global_instruction` applies, full stop, no fallback.
//! **Disclosed narrowing**: when `ctx.agent()` is unset entirely (no
//! tree context at all — every direct unit-test call site in this
//! module, and any future caller that hasn't wired `InvocationContext
//! .agent`), this falls back to `agent`'s own `global_instruction`
//! field, since there's no tree to consult at all. Once a real caller
//! always sets `ctx.agent`, this fallback is unreachable — it exists
//! only to keep `build_instructions` usable without a full tree in
//! place, exactly the "no tree, so 'this agent' and 'the root agent'
//! are the same agent" case this module previously handled by reading
//! `agent`'s own field unconditionally.
//!
//! **Adaptation, disclosed**: `agent.static_instruction` is an opaque
//! `google.genai.types.ContentUnion` placeholder (`Union[str, list[str],
//! types.Content, types.PartUnion, ...]` in the source, normalized by the
//! SDK's own `_transformers.t_content` — not ported, Phase 3 scope only
//! covers ADK's own code, not the whole Gemini SDK type system).
//! [`build_instructions`] handles the two shapes it can interpret without
//! that transformer: a `Value::Map` already shaped like a real `Content`
//! (deserialized directly), and a `Value::String` (wrapped as a single
//! text part) — covering the common cases a caller would actually set.
//! Anything else names [`InstructionsError::UnsupportedStaticInstruction`]
//! rather than silently dropping or misinterpreting it.

use adk_agents::llm_agent::LlmAgent;
use adk_agents::readonly_context::ReadonlyContext;
use adk_genai::content::{Content, Part};
use adk_models::llm_request::{Instructions, LlmRequest};
use rusty_serde::value::Value;

use crate::instructions_utils::{inject_session_state, InjectSessionStateError};

#[derive(Debug, rusty_err::Error)]
pub enum InstructionsError {
    #[error("{0}")]
    InjectSessionState(#[from] InjectSessionStateError),
    #[error("static_instruction is neither a Content-shaped map nor a plain string: {0}")]
    UnsupportedStaticInstruction(String),
}

fn static_instruction_to_content(value: &Value) -> Result<Content, InstructionsError> {
    match value {
        Value::String(text) => Ok(Content {
            role: None,
            parts: vec![Part::text(text.clone())],
        }),
        Value::Map(_) => rusty_serde::json::from_value(value.clone())
            .map_err(|e| InstructionsError::UnsupportedStaticInstruction(e.to_string())),
        other => Err(InstructionsError::UnsupportedStaticInstruction(format!(
            "{other:?}"
        ))),
    }
}

/// Resolves the deprecated `global_instruction`'s `(text,
/// bypass_state_injection)` pair from the invocation's root agent — see
/// the module doc for the exact semantics and its one disclosed
/// fallback.
fn resolve_root_global_instruction(
    agent: &LlmAgent,
    ctx: &ReadonlyContext,
) -> Option<(String, bool)> {
    match ctx.agent() {
        None => agent.global_instruction.is_set().then(|| {
            adk_agents::llm_agent::canonical_global_instruction(&agent.global_instruction, ctx)
        }),
        Some(current_agent) => {
            let root = current_agent.root_agent();
            let llm_flow = root.as_any().downcast_ref::<crate::llm_flow::LlmFlow>()?;
            llm_flow.llm_agent.global_instruction.is_set().then(|| {
                adk_agents::llm_agent::canonical_global_instruction(
                    &llm_flow.llm_agent.global_instruction,
                    ctx,
                )
            })
        }
    }
}

fn process_agent_instruction(
    agent: &LlmAgent,
    ctx: &ReadonlyContext,
) -> Result<String, InjectSessionStateError> {
    let (raw_si, bypass_state_injection) =
        adk_agents::llm_agent::canonical_instruction(&agent.instruction, ctx);
    if bypass_state_injection {
        Ok(raw_si)
    } else {
        inject_session_state(&raw_si, ctx)
    }
}

/// C0170: `_build_instructions` — appends global/static/dynamic
/// instructions to `llm_request`. See the module doc for what's deferred.
pub fn build_instructions(
    agent: &LlmAgent,
    ctx: &ReadonlyContext,
    llm_request: &mut LlmRequest,
) -> Result<(), InstructionsError> {
    // Deprecated global instruction — see the module doc's root-agent walk.
    if let Some((raw_si, bypass_state_injection)) = resolve_root_global_instruction(agent, ctx) {
        let si = if bypass_state_injection {
            raw_si
        } else {
            inject_session_state(&raw_si, ctx)?
        };
        llm_request.append_instructions(Instructions::Strings(vec![si]));
    }

    if let Some(static_instruction) = &agent.static_instruction {
        let content = static_instruction_to_content(static_instruction)?;
        llm_request.append_instructions(Instructions::Content(content));
    }

    if agent.instruction.is_set() && agent.static_instruction.is_none() {
        let si = process_agent_instruction(agent, ctx)?;
        llm_request.append_instructions(Instructions::Strings(vec![si]));
    } else if agent.instruction.is_set() && agent.static_instruction.is_some() {
        let si = process_agent_instruction(agent, ctx)?;
        llm_request
            .contents
            .push(Content::new("user", vec![Part::text(si)]));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::llm_agent::{Instruction, ModelRef};
    use adk_agents::session::Session;

    fn ctx() -> ReadonlyContext {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ReadonlyContext::new(ic)
    }

    fn agent() -> LlmAgent {
        LlmAgent::new(ModelRef::Name("gemini-2.5-flash".to_string()))
    }

    #[test]
    fn a_plain_instruction_becomes_the_system_instruction() {
        let mut a = agent();
        a.instruction = Instruction::Static("be helpful".to_string());
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("be helpful")
        );
    }

    #[test]
    fn no_instructions_set_appends_nothing() {
        let a = agent();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        assert!(request.config.system_instruction.is_none());
        assert!(request.contents.is_empty());
    }

    #[test]
    fn global_instruction_is_appended_first() {
        let mut a = agent();
        a.global_instruction = Instruction::Static("global rule".to_string());
        a.instruction = Instruction::Static("local rule".to_string());
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("global rule\n\nlocal rule")
        );
    }

    #[test]
    fn a_string_static_instruction_becomes_a_text_content() {
        let mut a = agent();
        a.static_instruction = Some(Value::String("static text".to_string()));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        // append_instructions(Content) extracts text and folds it into
        // system_instruction, matching llm_request.rs's own contract.
        assert!(request
            .config
            .system_instruction
            .as_deref()
            .unwrap()
            .contains("static text"));
    }

    #[test]
    fn dynamic_instruction_becomes_user_content_when_static_instruction_is_present() {
        let mut a = agent();
        a.static_instruction = Some(Value::String("static text".to_string()));
        a.instruction = Instruction::Static("dynamic text".to_string());
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        let last = request.contents.last().unwrap();
        assert_eq!(last.role.as_deref(), Some("user"));
        assert_eq!(last.parts[0].text.as_deref(), Some("dynamic text"));
    }

    #[test]
    fn instruction_state_injection_can_be_bypassed_by_a_provider() {
        let mut a = agent();
        a.instruction = Instruction::Provider(std::sync::Arc::new(|_ctx: &ReadonlyContext| {
            "raw {not_injected}".to_string()
        }));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&a, &ctx(), &mut request).unwrap();
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("raw {not_injected}")
        );
    }

    #[test]
    fn an_unsupported_static_instruction_shape_errors() {
        let mut a = agent();
        a.static_instruction = Some(Value::Int(123));
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let err = build_instructions(&a, &ctx(), &mut request).unwrap_err();
        assert!(matches!(
            err,
            InstructionsError::UnsupportedStaticInstruction(_)
        ));
    }

    /// Builds a 2-level tree (root, with one sub-agent named "sub") and
    /// returns a `ReadonlyContext` whose `.agent()` is the sub-agent —
    /// mirroring what `Runner`/`BaseAgent::run_async` set up for real.
    /// Also returns the root `BaseAgent` handle: the sub-agent only
    /// holds a `Weak` back-reference to it (C0036), so the caller must
    /// keep the root alive for as long as the returned context is used,
    /// or the parent-walk will find nothing to upgrade.
    fn tree_running_the_sub_agent(
        root_behavior: impl adk_agents::base_agent::AgentBehavior,
        sub_behavior: impl adk_agents::base_agent::AgentBehavior,
    ) -> (ReadonlyContext, adk_agents::base_agent::BaseAgent) {
        let sub_agent = adk_agents::base_agent::BaseAgent::new("sub", sub_behavior).unwrap();
        let root_agent = adk_agents::base_agent::BaseAgent::build(
            "root",
            "",
            vec![sub_agent.clone()],
            vec![],
            vec![],
            root_behavior,
        )
        .unwrap();

        let sub_agent_handle = sub_agent.parent_agent().unwrap().find_agent("sub").unwrap();
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(sub_agent_handle)
            .build();
        (ReadonlyContext::new(ic), root_agent)
    }

    #[test]
    fn global_instruction_is_resolved_from_the_real_tree_root_not_the_running_agent() {
        // Root carries the global instruction; the sub-agent (the one
        // actually running) does not. build_instructions is called with
        // the sub-agent's own LlmAgent, but a ctx whose `.agent` is the
        // sub BaseAgent — proving the root-walk finds the root's field
        // rather than reading the passed-in agent's own (unset) field.
        let mut root_llm_agent = agent();
        root_llm_agent.global_instruction = Instruction::Static("root rule".to_string());
        let root_behavior = crate::llm_flow::LlmFlow::new(root_llm_agent).unwrap();
        let sub_behavior = crate::llm_flow::LlmFlow::new(agent()).unwrap();

        let (ctx_with_tree, _root) = tree_running_the_sub_agent(root_behavior, sub_behavior);

        let sub_llm_agent = agent();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&sub_llm_agent, &ctx_with_tree, &mut request).unwrap();
        assert_eq!(
            request.config.system_instruction.as_deref(),
            Some("root rule")
        );
    }

    #[test]
    fn global_instruction_is_absent_when_the_tree_root_is_not_llm_backed() {
        let sub_behavior = crate::llm_flow::LlmFlow::new(agent()).unwrap();
        let (ctx_with_tree, _root) =
            tree_running_the_sub_agent(adk_agents::base_agent::NoopBehavior, sub_behavior);

        let sub_llm_agent = agent();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        build_instructions(&sub_llm_agent, &ctx_with_tree, &mut request).unwrap();
        assert!(request.config.system_instruction.is_none());
    }
}
