//! Placeholder service traits for capabilities C0061-C0064, forward-referencing
//! phases not yet built: `BaseArtifactService`/`ArtifactVersion` (Phase 6),
//! `BaseSessionService` (Phase 5), `BaseMemoryService`/`SearchMemoryResponse`/
//! `MemoryEntry` (Phase 6), `BaseCredentialService`/`AuthCredential`/
//! `AuthConfig` (Phase 9).
//!
//! **Disclosed adaptation**: the source's service methods are `async def`.
//! Since no concrete backend exists yet (nothing here performs real I/O),
//! these placeholder traits are synchronous; `Context`'s own methods stay
//! `async fn` (preserving the `.await`-able call shape callers already use)
//! and simply call through. Revisit — trait methods become `async fn` too —
//! once a real backend (network/disk I/O) lands in its own phase.

use adk_events::ui_widget::UiWidget;
use adk_platform::uuid::new_uuid;
use rusty_serde::value::Value;
use std::collections::BTreeMap;

use crate::session::Session;

/// Placeholder for `auth.auth_credential.AuthCredential` (Phase 9).
pub type AuthCredential = Value;
/// Placeholder for `auth.auth_tool.AuthConfig` (Phase 9).
pub type AuthConfig = Value;
/// Placeholder for `artifacts.base_artifact_service.ArtifactVersion` (Phase 6).
pub type ArtifactVersion = Value;
/// Placeholder for `memory.base_memory_service.SearchMemoryResponse` (Phase 6).
pub type SearchMemoryResponse = Value;
/// Placeholder for `memory.memory_entry.MemoryEntry` (Phase 6).
pub type MemoryEntry = Value;

/// Placeholder for `sessions.base_session_service.BaseSessionService`
/// (Phase 5). Marker only — nothing in this batch calls through it yet;
/// `InvocationContext` merely needs the field type to exist.
pub trait SessionService {}

/// Placeholder for `artifacts.base_artifact_service.BaseArtifactService`
/// (Phase 6).
pub trait ArtifactService {
    fn load_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<Value>;

    fn save_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<BTreeMap<String, Value>>,
    ) -> i64;

    fn get_artifact_version(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<ArtifactVersion>;

    fn list_artifact_keys(&self, app_name: &str, user_id: &str, session_id: &str) -> Vec<String>;
}

/// Placeholder for `memory.base_memory_service.BaseMemoryService` (Phase 6).
pub trait MemoryService {
    fn add_session_to_memory(&self, session: &Session);

    fn add_events_to_memory(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        events: &[adk_events::Event],
        custom_metadata: Option<&BTreeMap<String, Value>>,
    );

    fn add_memory(
        &self,
        app_name: &str,
        user_id: &str,
        memories: &[MemoryEntry],
        custom_metadata: Option<&BTreeMap<String, Value>>,
    );

    fn search_memory(&self, app_name: &str, user_id: &str, query: &str) -> SearchMemoryResponse;
}

/// Placeholder for `auth.credential_service.base_credential_service.BaseCredentialService`
/// (Phase 9).
pub trait CredentialService {
    fn save_credential(&self, auth_config: &AuthConfig);
    fn load_credential(&self, auth_config: &AuthConfig) -> Option<AuthCredential>;
}

/// Placeholder for `plugins.plugin_manager.PluginManager` (Phase 7).
///
/// Structurally faithful for the "zero plugins registered" case: with no
/// `Plugin` trait/registration surface yet, every hook correctly returns
/// `None` — exactly what a real `PluginManager` with an empty plugin list
/// would also return. Becomes a real iterate-and-short-circuit loop once
/// `plugins/` lands.
#[derive(Debug, Default, Clone)]
pub struct PluginManager;

impl PluginManager {
    pub fn run_before_agent_callback(&self) -> Option<adk_genai::content::Content> {
        None
    }

    pub fn run_after_agent_callback(&self) -> Option<adk_genai::content::Content> {
        None
    }

    pub fn run_on_agent_error_callback(&self) {
        // No plugins registered yet; nothing to notify.
    }
}

/// Renders a UI widget by appending it to the given actions list, raising
/// (as `Err`) on a duplicate widget id. Shared by `Context::render_ui_widget`
/// (C0065) — factored out here since it only needs `UiWidget`, not any
/// context state.
/// Returns `Err(widget_id)` on a duplicate id — the caller formats the
/// user-facing message (see `Context::render_ui_widget`'s `ContextError`).
pub fn render_ui_widget(
    widgets: &mut Option<Vec<UiWidget>>,
    widget: UiWidget,
) -> Result<(), String> {
    let list = widgets.get_or_insert_with(Vec::new);
    if list.iter().any(|existing| existing.id == widget.id) {
        return Err(widget.id);
    }
    list.push(widget);
    Ok(())
}

/// Generates a fresh invocation id, mirroring
/// `invocation_context.new_invocation_context_id`.
pub fn new_invocation_context_id() -> String {
    format!("e-{}", new_uuid())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_manager_with_no_plugins_always_returns_none() {
        let manager = PluginManager;
        assert_eq!(manager.run_before_agent_callback(), None);
        assert_eq!(manager.run_after_agent_callback(), None);
    }

    #[test]
    fn render_ui_widget_rejects_duplicate_ids() {
        let mut widgets = None;
        render_ui_widget(&mut widgets, UiWidget::new("w1", "mcp", Value::Null)).unwrap();
        let err =
            render_ui_widget(&mut widgets, UiWidget::new("w1", "mcp", Value::Null)).unwrap_err();
        assert!(err.contains("w1"));
    }

    #[test]
    fn new_invocation_context_id_is_prefixed() {
        assert!(new_invocation_context_id().starts_with("e-"));
    }
}
