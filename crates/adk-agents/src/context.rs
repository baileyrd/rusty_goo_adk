//! Capabilities C0048, C0050-C0058, C0061-C0065: `Context` (`CallbackContext`
//! is now a unified alias for it), ported from `google.adk.agents.context`.
//!
//! **Deferred** (need `workflow::BaseNode`/the graph engine, Phase 7):
//! `Context.run_node`/`_run_node_internal`/`_run_node_standalone` (C0059,
//! C0060) — dynamic node execution is inseparable from the workflow
//! scheduler that doesn't exist yet. `node`/`parent_ctx`/`node_path`/`run_id`
//! (workflow-specific fields) are likewise omitted; `Context` here covers
//! only the agent-callback/tool-context surface this batch actually
//! exercises and tests.
//!
//! **Adaptation**: `telemetry_context` (Phase 12) is omitted — nothing in
//! this batch reads it.

use std::collections::HashSet;

use adk_events::ui_widget::UiWidget;
use adk_events::EventActions;
use rusty_serde::value::Value;

use crate::invocation_context::InvocationContext;
use crate::services::{self, AuthConfig, AuthCredential};
use crate::state::State;

#[derive(Debug, rusty_err::Error)]
pub enum ContextError {
    #[error("Output already set. A node can produce at most one output.")]
    OutputAlreadySet,
    #[error("Artifact service is not initialized.")]
    ArtifactServiceUnset,
    #[error("Credential service is not initialized.")]
    CredentialServiceUnset,
    #[error("request_credential requires function_call_id. This method can only be used in a tool context, not a callback context. Consider using save_credential/load_credential instead.")]
    RequestCredentialNeedsFunctionCallId,
    #[error("request_confirmation requires function_call_id. This method can only be used in a tool context.")]
    RequestConfirmationNeedsFunctionCallId,
    #[error("Cannot add session to memory: memory service is not available.")]
    MemoryServiceUnsetForSession,
    #[error("Cannot add events to memory: memory service is not available.")]
    MemoryServiceUnsetForEvents,
    #[error("Cannot add memory: memory service is not available.")]
    MemoryServiceUnsetForMemory,
    #[error("Memory service is not available.")]
    MemoryServiceUnsetForSearch,
    #[error("UI widget with ID '{0}' already exists in the current event actions.")]
    DuplicateUiWidget(String),
}

/// C0048: `CallbackContext` in the source is now a unified alias for
/// `Context` (no longer a distinct class) — mirrored directly here.
pub type CallbackContext = Context;

/// The context within an agent run.
pub struct Context {
    invocation_context: InvocationContext,
    event_actions: EventActions,
    state: State,
    function_call_id: Option<String>,
    isolation_scope: Option<String>,
    output: Option<Value>,
    route: Option<Value>,
    interrupt_ids: HashSet<String>,
    event_author: String,
    tool_confirmation: Option<Value>,
}

impl Context {
    pub fn new(invocation_context: InvocationContext) -> Self {
        let state = State::new(invocation_context.session.state.clone(), Default::default());
        let isolation_scope = invocation_context.isolation_scope.clone();
        Self {
            state,
            function_call_id: None,
            isolation_scope,
            output: None,
            route: None,
            interrupt_ids: HashSet::new(),
            event_author: String::new(),
            event_actions: EventActions::default(),
            invocation_context,
            tool_confirmation: None,
        }
    }

    pub fn invocation_context(&self) -> &InvocationContext {
        &self.invocation_context
    }

    pub fn branch(&self) -> Option<&str> {
        self.invocation_context.branch.as_deref()
    }

    pub fn custom_metadata(&self) -> &std::collections::BTreeMap<String, Value> {
        &self.invocation_context.custom_metadata
    }

    /// C0051.
    pub fn function_call_id(&self) -> Option<&str> {
        self.function_call_id.as_deref()
    }

    pub fn set_function_call_id(&mut self, value: Option<String>) {
        self.function_call_id = value;
    }

    /// The tool confirmation of the current tool call, if the inbound
    /// function response carried one. Stays a plain (opaque) `Value` here
    /// rather than a typed `ToolConfirmation` — that type lives in
    /// `adk-tools` (Phase 8), which depends on `adk-agents`, not the
    /// other way around, so this crate can't hold it as a typed field
    /// without a cycle. `adk_tools::function_tool::FunctionTool` narrows
    /// it via `ToolConfirmation`'s own (de)serialization.
    pub fn tool_confirmation(&self) -> Option<&Value> {
        self.tool_confirmation.as_ref()
    }

    pub fn set_tool_confirmation(&mut self, value: Option<Value>) {
        self.tool_confirmation = value;
    }

    /// C0052: internal mechanism — do not use directly outside the
    /// framework (see the source docstring).
    pub fn isolation_scope(&self) -> Option<&str> {
        self.isolation_scope.as_deref()
    }

    pub fn set_isolation_scope(&mut self, value: Option<String>) {
        self.isolation_scope = value;
    }

    /// C0053: the delta-aware state of the current session.
    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    pub fn actions(&self) -> &EventActions {
        &self.event_actions
    }

    pub fn actions_mut(&mut self) -> &mut EventActions {
        &mut self.event_actions
    }

    /// Consumes this `Context`, returning its accumulated event actions —
    /// used by `BaseAgent`'s callback wrapping to build the resulting
    /// `Event`. Syncs `state`'s pending delta into `state_delta` first: the
    /// source's `State._delta` *is* `EventActions.state_delta` (the same
    /// dict object, by reference), so a direct `ctx.state[key] = value`
    /// mutation is automatically visible on `event_actions.state_delta`
    /// there. This port's `State` owns its delta rather than sharing it by
    /// reference, so this sync step reproduces the same end result at the
    /// one point it's actually observed.
    pub fn into_actions(mut self) -> EventActions {
        self.event_actions.state_delta = self.state.delta_map().into_iter().collect();
        self.event_actions
    }

    /// C0054: at most one output per execution.
    pub fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    pub fn set_output(&mut self, value: Value) -> Result<(), ContextError> {
        if self.output.is_some() {
            return Err(ContextError::OutputAlreadySet);
        }
        self.output = Some(value);
        Ok(())
    }

    /// C0055: routing value for conditional edges, independent of output.
    pub fn route(&self) -> Option<&Value> {
        self.route.as_ref()
    }

    pub fn set_route(&mut self, value: Value) {
        self.route = Some(value);
    }

    /// C0056: interrupt IDs accumulated during this execution. Read-only —
    /// returns a copy, matching the source's `set(self._interrupt_ids)`.
    pub fn interrupt_ids(&self) -> HashSet<String> {
        self.interrupt_ids.clone()
    }

    /// C0057.
    pub fn event_author(&self) -> &str {
        &self.event_author
    }

    pub fn set_event_author(&mut self, value: impl Into<String>) {
        self.event_author = value.into();
    }

    /// C0058: a copy of the invocation context with the proxy session and
    /// isolation scope applied.
    pub fn get_invocation_context(&self) -> InvocationContext {
        let mut ctx = self.invocation_context.clone();
        ctx.isolation_scope = self.isolation_scope.clone();
        ctx
    }

    // ------------------------------------------------------------------
    // Artifact methods (C0061)
    // ------------------------------------------------------------------

    pub async fn load_artifact(
        &self,
        filename: &str,
        version: Option<i64>,
    ) -> Result<Option<Value>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.load_artifact(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            version,
        ))
    }

    pub async fn save_artifact(
        &mut self,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
    ) -> Result<i64, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        let version = service.save_artifact(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            artifact,
            custom_metadata,
        );
        self.event_actions
            .artifact_delta
            .insert(filename.to_string(), version);
        Ok(version)
    }

    pub async fn get_artifact_version(
        &self,
        filename: &str,
        version: Option<i64>,
    ) -> Result<Option<Value>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.get_artifact_version(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            filename,
            version,
        ))
    }

    pub async fn list_artifacts(&self) -> Result<Vec<String>, ContextError> {
        let service = self
            .invocation_context
            .artifact_service
            .as_ref()
            .ok_or(ContextError::ArtifactServiceUnset)?;
        Ok(service.list_artifact_keys(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
        ))
    }

    // ------------------------------------------------------------------
    // Credential methods (C0062)
    // ------------------------------------------------------------------

    pub async fn save_credential(&self, auth_config: &AuthConfig) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .credential_service
            .as_ref()
            .ok_or(ContextError::CredentialServiceUnset)?;
        service.save_credential(auth_config);
        Ok(())
    }

    pub async fn load_credential(
        &self,
        auth_config: &AuthConfig,
    ) -> Result<Option<AuthCredential>, ContextError> {
        let service = self
            .invocation_context
            .credential_service
            .as_ref()
            .ok_or(ContextError::CredentialServiceUnset)?;
        Ok(service.load_credential(auth_config))
    }

    /// C0062: requests a credential for the current tool call. Requires
    /// `function_call_id` — for callback contexts, use
    /// `save_credential`/`load_credential` instead.
    pub fn request_credential(&mut self, auth_config: AuthConfig) -> Result<(), ContextError> {
        let function_call_id = self
            .function_call_id
            .clone()
            .ok_or(ContextError::RequestCredentialNeedsFunctionCallId)?;
        self.event_actions
            .requested_auth_configs
            .insert(function_call_id, auth_config);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Tool methods (C0063)
    // ------------------------------------------------------------------

    /// C0063: requests confirmation for the current tool call. Requires
    /// `function_call_id`.
    pub fn request_confirmation(
        &mut self,
        hint: Option<String>,
        payload: Option<Value>,
    ) -> Result<(), ContextError> {
        let function_call_id = self
            .function_call_id
            .clone()
            .ok_or(ContextError::RequestConfirmationNeedsFunctionCallId)?;
        let mut confirmation = std::collections::BTreeMap::new();
        if let Some(hint) = hint {
            confirmation.insert("hint".to_string(), Value::String(hint));
        }
        if let Some(payload) = payload {
            confirmation.insert("payload".to_string(), payload);
        }
        self.event_actions.requested_tool_confirmations.insert(
            function_call_id,
            Value::Map(confirmation.into_iter().collect()),
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Memory methods (C0064)
    // ------------------------------------------------------------------

    pub async fn add_session_to_memory(&self) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForSession)?;
        service.add_session_to_memory(&self.invocation_context.session);
        Ok(())
    }

    pub async fn add_events_to_memory(
        &self,
        events: &[adk_events::Event],
        custom_metadata: Option<&std::collections::BTreeMap<String, Value>>,
    ) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForEvents)?;
        service.add_events_to_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            &self.invocation_context.session.id,
            events,
            custom_metadata,
        );
        Ok(())
    }

    pub async fn add_memory(
        &self,
        memories: &[services::MemoryEntry],
        custom_metadata: Option<&std::collections::BTreeMap<String, Value>>,
    ) -> Result<(), ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForMemory)?;
        service.add_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            memories,
            custom_metadata,
        );
        Ok(())
    }

    pub async fn search_memory(
        &self,
        query: &str,
    ) -> Result<services::SearchMemoryResponse, ContextError> {
        let service = self
            .invocation_context
            .memory_service
            .as_ref()
            .ok_or(ContextError::MemoryServiceUnsetForSearch)?;
        Ok(service.search_memory(
            &self.invocation_context.session.app_name,
            &self.invocation_context.session.user_id,
            query,
        ))
    }

    // ------------------------------------------------------------------
    // UI widget methods (C0065)
    // ------------------------------------------------------------------

    pub fn render_ui_widget(&mut self, ui_widget: UiWidget) -> Result<(), ContextError> {
        services::render_ui_widget(&mut self.event_actions.render_ui_widgets, ui_widget)
            .map_err(ContextError::DuplicateUiWidget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;

    fn context() -> Context {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(ic)
    }

    #[test]
    fn output_can_only_be_set_once() {
        let mut ctx = context();
        ctx.set_output(Value::Int(1)).unwrap();
        let err = ctx.set_output(Value::Int(2)).unwrap_err();
        assert!(matches!(err, ContextError::OutputAlreadySet));
        assert_eq!(ctx.output(), Some(&Value::Int(1)));
    }

    #[test]
    fn event_author_defaults_to_empty_string() {
        let ctx = context();
        assert_eq!(ctx.event_author(), "");
    }

    #[test]
    fn interrupt_ids_returns_an_independent_copy() {
        let mut ctx = context();
        ctx.interrupt_ids.insert("i1".to_string());
        let mut copy = ctx.interrupt_ids();
        copy.insert("i2".to_string());
        assert_eq!(
            ctx.interrupt_ids().len(),
            1,
            "mutating the copy must not affect the original"
        );
    }

    #[test]
    fn request_credential_requires_function_call_id() {
        let mut ctx = context();
        let err = ctx.request_credential(Value::Null).unwrap_err();
        assert!(matches!(
            err,
            ContextError::RequestCredentialNeedsFunctionCallId
        ));
    }

    #[test]
    fn request_credential_stores_it_keyed_by_function_call_id() {
        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        ctx.request_credential(Value::String("auth".to_string()))
            .unwrap();
        assert!(ctx.actions().requested_auth_configs.contains_key("fc-1"));
    }

    #[test]
    fn request_confirmation_requires_function_call_id() {
        let mut ctx = context();
        let err = ctx.request_confirmation(None, None).unwrap_err();
        assert!(matches!(
            err,
            ContextError::RequestConfirmationNeedsFunctionCallId
        ));
    }

    #[rusty_tokio::test]
    async fn artifact_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.load_artifact("f", None).await.unwrap_err();
        assert!(matches!(err, ContextError::ArtifactServiceUnset));
    }

    #[rusty_tokio::test]
    async fn memory_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.add_session_to_memory().await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForSession));
        let err = ctx.search_memory("q").await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForSearch));
    }

    #[test]
    fn render_ui_widget_rejects_duplicate_ids() {
        let mut ctx = context();
        ctx.render_ui_widget(UiWidget::new("w1", "mcp", Value::Null))
            .unwrap();
        let err = ctx
            .render_ui_widget(UiWidget::new("w1", "mcp", Value::Null))
            .unwrap_err();
        assert!(matches!(err, ContextError::DuplicateUiWidget(id) if id == "w1"));
    }

    #[test]
    fn state_has_delta_is_false_for_a_freshly_built_context() {
        let ctx = context();
        assert!(!ctx.state().has_delta());
    }

    #[test]
    fn callback_context_is_the_same_type_as_context() {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let callback_ctx: CallbackContext = Context::new(ic);
        assert!(!callback_ctx.state().has_delta());
    }

    #[test]
    fn isolation_scope_can_be_read_and_overridden() {
        let mut ctx = context();
        assert_eq!(ctx.isolation_scope(), None);
        ctx.set_isolation_scope(Some("scope-1".to_string()));
        assert_eq!(ctx.isolation_scope(), Some("scope-1"));
    }

    #[test]
    fn route_is_independent_of_output() {
        let mut ctx = context();
        ctx.set_route(Value::String("branch-a".to_string()));
        ctx.set_output(Value::Int(1)).unwrap();
        assert_eq!(ctx.route(), Some(&Value::String("branch-a".to_string())));
        assert_eq!(ctx.output(), Some(&Value::Int(1)));
    }

    #[test]
    fn event_author_can_be_overridden() {
        let mut ctx = context();
        ctx.set_event_author("workflow");
        assert_eq!(ctx.event_author(), "workflow");
    }

    #[test]
    fn get_invocation_context_applies_the_current_isolation_scope() {
        let mut ctx = context();
        ctx.set_isolation_scope(Some("scope-1".to_string()));
        let copy = ctx.get_invocation_context();
        assert_eq!(copy.isolation_scope, Some("scope-1".to_string()));
    }

    #[rusty_tokio::test]
    async fn credential_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.save_credential(&Value::Null).await.unwrap_err();
        assert!(matches!(err, ContextError::CredentialServiceUnset));
        let err = ctx.load_credential(&Value::Null).await.unwrap_err();
        assert!(matches!(err, ContextError::CredentialServiceUnset));
    }

    #[test]
    fn request_confirmation_stores_it_keyed_by_function_call_id() {
        let mut ctx = context();
        ctx.set_function_call_id(Some("fc-1".to_string()));
        ctx.request_confirmation(Some("pick one".to_string()), None)
            .unwrap();
        assert!(ctx
            .actions()
            .requested_tool_confirmations
            .contains_key("fc-1"));
    }

    #[rusty_tokio::test]
    async fn remaining_memory_methods_raise_when_service_unset() {
        let ctx = context();
        let err = ctx.add_events_to_memory(&[], None).await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForEvents));
        let err = ctx.add_memory(&[], None).await.unwrap_err();
        assert!(matches!(err, ContextError::MemoryServiceUnsetForMemory));
    }

    struct FakeArtifactService;
    impl services::ArtifactService for FakeArtifactService {
        fn load_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            _version: Option<i64>,
        ) -> Option<Value> {
            Some(Value::String(format!("contents of {filename}")))
        }

        fn save_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _artifact: Value,
            _custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
        ) -> i64 {
            1
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<services::ArtifactVersion> {
            None
        }

        fn list_artifact_keys(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
        ) -> Vec<String> {
            vec!["f.txt".to_string()]
        }
    }

    #[rusty_tokio::test]
    async fn artifact_methods_delegate_to_a_configured_service() {
        let mut ic =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ic.artifact_service = Some(std::sync::Arc::new(FakeArtifactService));
        let mut ctx = Context::new(ic);

        let loaded = ctx.load_artifact("f.txt", None).await.unwrap();
        assert_eq!(loaded, Some(Value::String("contents of f.txt".to_string())));

        let version = ctx
            .save_artifact("f.txt", Value::String("data".to_string()), None)
            .await
            .unwrap();
        assert_eq!(version, 1);
        assert_eq!(ctx.actions().artifact_delta.get("f.txt"), Some(&1));

        let keys = ctx.list_artifacts().await.unwrap();
        assert_eq!(keys, vec!["f.txt".to_string()]);
    }
}
