//! Capability C0049: `ReadonlyContext`, ported from
//! `google.adk.agents.readonly_context`.

use rusty_serde::value::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::invocation_context::InvocationContext;
use crate::run_config::RunConfig;
use crate::services::AuthCredential;
use crate::session::Session;

/// A read-only view over an [`InvocationContext`].
pub struct ReadonlyContext {
    pub(crate) invocation_context: InvocationContext,
}

impl ReadonlyContext {
    pub fn new(invocation_context: InvocationContext) -> Self {
        Self { invocation_context }
    }

    /// The user content that started this invocation. READONLY.
    pub fn user_content(&self) -> Option<&Value> {
        self.invocation_context.user_content.as_ref()
    }

    pub fn invocation_id(&self) -> &str {
        &self.invocation_context.invocation_id
    }

    /// The name of the agent currently running. `"unknown"` if none is set.
    pub fn agent_name(&self) -> &str {
        match &self.invocation_context.agent {
            Some(agent) => agent.name(),
            None => "unknown",
        }
    }

    /// The agent currently running, if one is set. Lets a caller walk the
    /// tree (`.parent_agent()`/`.root_agent()`) or downcast onto a
    /// concrete `AgentBehavior` (`.as_any()`) — needed for a cross-tree
    /// lookup like the deprecated `global_instruction`'s root-agent
    /// resolution (`instructions.rs`, C0170).
    pub fn agent(&self) -> Option<&crate::base_agent::BaseAgent> {
        self.invocation_context.agent.as_ref()
    }

    /// The state of the current session, as a read-only view. READONLY.
    pub fn state(&self) -> &BTreeMap<String, Value> {
        &self.invocation_context.session.state
    }

    pub fn session(&self) -> &Session {
        &self.invocation_context.session
    }

    /// The invocation's artifact service, if one is configured. Added for
    /// `adk-flows`'s `inject_session_state` (C0170), which needs it to
    /// resolve `{artifact.name}` template references.
    pub fn artifact_service(
        &self,
    ) -> Option<&Arc<dyn crate::services::ArtifactService + Send + Sync>> {
        self.invocation_context.artifact_service.as_ref()
    }

    /// The id of the user. READONLY.
    pub fn user_id(&self) -> &str {
        &self.invocation_context.session.user_id
    }

    pub fn run_config(&self) -> Option<&RunConfig> {
        self.invocation_context.run_config.as_ref()
    }

    /// The custom metadata dictionary, as a read-only view.
    pub fn custom_metadata(&self) -> &BTreeMap<String, Value> {
        &self.invocation_context.custom_metadata
    }

    /// Gets a resolved credential by key for this invocation.
    pub fn get_credential(&self, key: &str) -> Option<&AuthCredential> {
        self.invocation_context.credential_by_key.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;

    #[test]
    fn agent_name_falls_back_to_unknown() {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let ctx = ReadonlyContext::new(ic);
        assert_eq!(ctx.agent_name(), "unknown");
    }

    #[test]
    fn agent_name_reflects_the_set_agent() {
        let agent =
            crate::base_agent::BaseAgent::new("planner", crate::base_agent::NoopBehavior).unwrap();
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(agent)
            .build();
        let ctx = ReadonlyContext::new(ic);
        assert_eq!(ctx.agent_name(), "planner");
    }

    #[test]
    fn user_id_and_session_come_from_the_session() {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "alice", "s1")).build();
        let ctx = ReadonlyContext::new(ic);
        assert_eq!(ctx.user_id(), "alice");
        assert_eq!(ctx.session().id, "s1");
    }

    #[test]
    fn get_credential_looks_up_by_key() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let credential = AuthCredential::api_key("secret");
        ic.credential_by_key
            .insert("k".to_string(), credential.clone());
        let ctx = ReadonlyContext::new(ic);
        assert_eq!(ctx.get_credential("k"), Some(&credential));
        assert_eq!(ctx.get_credential("missing"), None);
    }
}
