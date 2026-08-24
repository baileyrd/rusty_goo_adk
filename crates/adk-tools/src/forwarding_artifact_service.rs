//! Capability C0489 (partial): `ForwardingArtifactService`, ported from
//! `tools/_forwarding_artifact_service.py`.
//!
//! Routes a nested `Runner`'s artifact reads/writes back through the
//! parent tool context's own real artifact backend, so an agent invoked
//! via [`crate::agent_tool::AgentTool`] can see and persist real
//! artifacts instead of running with no artifact service at all — a gap
//! `agent_tool.rs`'s own module doc has disclosed since C0406 first
//! landed.
//!
//! **Disclosed adaptation**: the source updates
//! `tool_context.actions.artifact_delta` synchronously, inline, as each
//! individual `save_artifact` call happens mid-nested-run — it awaits
//! the parent `ToolContext`'s own `save_artifact` method, which needs
//! `&mut self`. This port's `ArtifactService` trait is fully
//! synchronous and `&self`-only, so a `ForwardingArtifactService`
//! instance can't hold a live mutable borrow of the parent `Context`
//! across the whole nested run. Instead, version numbers are
//! accumulated into a shared `artifact_delta` map as they're produced,
//! and [`crate::agent_tool::AgentTool::run_async`] merges that map into
//! the parent tool context's own actions once the nested run completes
//! — the same post-hoc-merge idiom that file already uses for state
//! deltas (see its per-event `state_delta` forwarding). Reads
//! (`load_artifact`/`list_artifact_keys`/`list_versions`/
//! `list_artifact_versions`/`get_artifact_version`) and the write
//! itself (`save_artifact`/`delete_artifact`) still route straight
//! through to the parent's real backend as they happen — only the
//! delta *bookkeeping* is deferred.
//!
//! `app_name`/`user_id`/`session_id` are always the parent's own
//! (copied once at construction) — every trait method here ignores the
//! caller-supplied identifiers, matching the source's own
//! `del app_name, user_id, session_id` in every override.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use adk_agents::services::{ArtifactService, ArtifactVersion};
use rusty_serde::value::Value;

use crate::tool_context::ToolContext;

pub struct ForwardingArtifactService {
    parent_service: Arc<dyn ArtifactService + Send + Sync>,
    parent_app_name: String,
    parent_user_id: String,
    parent_session_id: String,
    artifact_delta: Mutex<HashMap<String, i64>>,
}

impl ForwardingArtifactService {
    /// `None` when the parent tool context has no artifact service of
    /// its own to forward to — there is nothing to wrap.
    pub fn new(tool_context: &ToolContext) -> Option<Self> {
        let invocation_context = tool_context.invocation_context();
        let parent_service = invocation_context.artifact_service.clone()?;
        Some(Self {
            parent_service,
            parent_app_name: invocation_context.session.app_name.clone(),
            parent_user_id: invocation_context.session.user_id.clone(),
            parent_session_id: invocation_context.session.id.clone(),
            artifact_delta: Mutex::new(HashMap::new()),
        })
    }

    /// Drains the versions accumulated by `save_artifact` calls so far
    /// — see the module doc's disclosed post-hoc-merge adaptation.
    pub fn take_artifact_delta(&self) -> HashMap<String, i64> {
        std::mem::take(&mut self.artifact_delta.lock().unwrap())
    }
}

impl ArtifactService for ForwardingArtifactService {
    fn load_artifact(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<Value> {
        self.parent_service.load_artifact(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
            version,
        )
    }

    fn save_artifact(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
        filename: &str,
        artifact: Value,
        custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
    ) -> i64 {
        let version = self.parent_service.save_artifact(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
            artifact,
            custom_metadata,
        );
        self.artifact_delta
            .lock()
            .unwrap()
            .insert(filename.to_string(), version);
        version
    }

    fn get_artifact_version(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
        filename: &str,
        version: Option<i64>,
    ) -> Option<ArtifactVersion> {
        self.parent_service.get_artifact_version(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
            version,
        )
    }

    fn list_artifact_keys(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
    ) -> Vec<String> {
        self.parent_service.list_artifact_keys(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
        )
    }

    fn delete_artifact(&self, _app_name: &str, _user_id: &str, _session_id: &str, filename: &str) {
        self.parent_service.delete_artifact(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
        )
    }

    fn list_versions(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
        filename: &str,
    ) -> Vec<i64> {
        self.parent_service.list_versions(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
        )
    }

    fn list_artifact_versions(
        &self,
        _app_name: &str,
        _user_id: &str,
        _session_id: &str,
        filename: &str,
    ) -> Vec<ArtifactVersion> {
        self.parent_service.list_artifact_versions(
            &self.parent_app_name,
            &self.parent_user_id,
            &self.parent_session_id,
            filename,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    struct StubArtifactService {
        stored: Mutex<HashMap<String, Value>>,
    }

    impl StubArtifactService {
        fn new() -> Self {
            Self {
                stored: Mutex::new(HashMap::new()),
            }
        }
    }

    impl ArtifactService for StubArtifactService {
        fn load_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            _version: Option<i64>,
        ) -> Option<Value> {
            self.stored.lock().unwrap().get(filename).cloned()
        }

        fn save_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
            artifact: Value,
            _custom_metadata: Option<std::collections::BTreeMap<String, Value>>,
        ) -> i64 {
            self.stored
                .lock()
                .unwrap()
                .insert(filename.to_string(), artifact);
            7
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<ArtifactVersion> {
            None
        }

        fn list_artifact_keys(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
        ) -> Vec<String> {
            self.stored.lock().unwrap().keys().cloned().collect()
        }

        fn delete_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            filename: &str,
        ) {
            self.stored.lock().unwrap().remove(filename);
        }

        fn list_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<i64> {
            vec![1]
        }

        fn list_artifact_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<ArtifactVersion> {
            Vec::new()
        }
    }

    fn ctx_with_artifact_service(service: Arc<dyn ArtifactService + Send + Sync>) -> Context {
        let mut invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("parent-app", "parent-user", "s1"))
                .build();
        invocation_context.artifact_service = Some(service);
        Context::new(invocation_context)
    }

    fn ctx_without_artifact_service() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("parent-app", "parent-user", "s1"))
                .build(),
        )
    }

    #[test]
    fn new_is_none_without_a_parent_artifact_service() {
        let ctx = ctx_without_artifact_service();
        assert!(ForwardingArtifactService::new(&ctx).is_none());
    }

    #[test]
    fn save_and_load_round_trip_through_the_parent_service() {
        let parent = Arc::new(StubArtifactService::new());
        let ctx = ctx_with_artifact_service(parent);
        let forwarding = ForwardingArtifactService::new(&ctx).unwrap();

        let version = forwarding.save_artifact(
            "ignored-app",
            "ignored-user",
            "ignored-session",
            "f.txt",
            Value::String("hello".to_string()),
            None,
        );
        assert_eq!(version, 7);

        let loaded = forwarding.load_artifact(
            "ignored-app",
            "ignored-user",
            "ignored-session",
            "f.txt",
            None,
        );
        assert_eq!(loaded, Some(Value::String("hello".to_string())));
    }

    #[test]
    fn save_artifact_always_targets_the_parents_own_identity() {
        let parent = Arc::new(StubArtifactService::new());
        let ctx = ctx_with_artifact_service(parent.clone());
        let forwarding = ForwardingArtifactService::new(&ctx).unwrap();

        forwarding.save_artifact(
            "some-other-app",
            "some-other-user",
            "some-other-session",
            "f.txt",
            Value::String("hello".to_string()),
            None,
        );

        // Loading directly from the parent service under the *parent's*
        // identity (not the caller-supplied one above) must see it.
        let loaded = parent.load_artifact("parent-app", "parent-user", "s1", "f.txt", None);
        assert_eq!(loaded, Some(Value::String("hello".to_string())));
    }

    #[test]
    fn take_artifact_delta_drains_accumulated_versions() {
        let parent = Arc::new(StubArtifactService::new());
        let ctx = ctx_with_artifact_service(parent);
        let forwarding = ForwardingArtifactService::new(&ctx).unwrap();

        forwarding.save_artifact(
            "a",
            "u",
            "s",
            "f.txt",
            Value::String("hello".to_string()),
            None,
        );

        let delta = forwarding.take_artifact_delta();
        assert_eq!(delta.get("f.txt"), Some(&7));
        assert!(forwarding.take_artifact_delta().is_empty());
    }

    #[test]
    fn delete_and_list_route_through_the_parent_service() {
        let parent = Arc::new(StubArtifactService::new());
        let ctx = ctx_with_artifact_service(parent);
        let forwarding = ForwardingArtifactService::new(&ctx).unwrap();

        forwarding.save_artifact("a", "u", "s", "f.txt", Value::String("x".to_string()), None);
        assert_eq!(forwarding.list_artifact_keys("a", "u", "s"), vec!["f.txt"]);

        forwarding.delete_artifact("a", "u", "s", "f.txt");
        assert!(forwarding.list_artifact_keys("a", "u", "s").is_empty());
    }
}
