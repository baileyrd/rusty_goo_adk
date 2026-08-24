//! Capability C0406 (partial): `AgentTool`, ported from
//! `google.adk.tools.agent_tool`.
//!
//! **Scope, disclosed**: several pieces of infrastructure the source
//! leans on don't exist in this port yet, so this batch narrows to the
//! core nested-`Runner` turn:
//! - `_get_input_schema`/`_get_output_schema` walk to a concrete
//!   `LlmAgent` to read its `input_schema`/`output_schema`. This port's
//!   `BaseAgent` is a type-erased wrapper (`Arc<dyn AgentBehavior>`) with
//!   no way to recover a concrete `LlmAgent` from it — the same "resolve
//!   `InvocationContext.agent` to a concrete `LlmAgent`" blocker every
//!   Phase 4 processor already discloses. So `get_declaration` always
//!   uses the generic `{"request": string}` parameter shape (the
//!   source's own no-input-schema fallback), and `run_async` never
//!   schema-validates the merged response text.
//! - `InMemoryMemoryService` is Phase 6, not built — the nested `Runner`
//!   runs with no memory service.
//! - Plugin propagation (`include_plugins`) needs `Runner` to accept a
//!   `PluginManager`, which `adk-runners::runner`'s own module doc
//!   already discloses `Runner` doesn't do yet ("a genuine, disclosed
//!   gap, not a silent no-op"). So `include_plugins` has no observable
//!   effect here — there's nothing to propagate into.
//! - `propagate_grounding_metadata` and `code_execution_result`/
//!   `executable_code` part-to-text extraction aren't ported — those
//!   `Part` fields are opaque placeholders in this port (`adk-genai`'s
//!   own disclosed scope), so only `part.text` is extractable.
//!
//! **What is ported**: spins up an isolated in-memory session (forwarding
//! the parent's non-`_adk`-prefixed state as the child's initial state,
//! matching the source's own filter), runs the wrapped agent for one
//! turn via the real `adk_runners::Runner`, forwards state deltas back to
//! the parent tool context as each event arrives, and merges the last
//! response event's non-thought text parts into the tool's return value
//! — falling back to the last error message when there's no usable text.
//! Also installs a [`crate::forwarding_artifact_service
//! ::ForwardingArtifactService`] (C0489 partial) on the nested `Runner`
//! whenever the parent tool context has a real artifact service of its
//! own, so the nested agent can read/write real artifacts — see that
//! module's own doc for its disclosed post-hoc artifact-delta-merge
//! adaptation, applied here the same way state deltas already are.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_agents::base_agent::BaseAgent;
use adk_agents::services::{InMemorySessionService, SessionService};
use adk_genai::content::{Content, FunctionDeclaration, Part};
use adk_runners::runner::Runner;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::forwarding_artifact_service::ForwardingArtifactService;
use crate::tool_context::ToolContext;

fn part_to_text(part: &Part) -> String {
    part.text.clone().unwrap_or_default()
}

fn declaration_schema() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("object".to_string())),
        (
            "properties".to_string(),
            Value::Map(vec![(
                "request".to_string(),
                Value::Map(vec![(
                    "type".to_string(),
                    Value::String("string".to_string()),
                )]),
            )]),
        ),
        (
            "required".to_string(),
            Value::Seq(vec![Value::String("request".to_string())]),
        ),
    ])
}

/// C0406: wraps a [`BaseAgent`] as a callable tool. Direct use is
/// discouraged by the source in favor of `sub_agents`/single-turn mode
/// where possible — ported here as the fallback path regardless.
pub struct AgentTool {
    agent: BaseAgent,
    skip_summarization: bool,
}

impl AgentTool {
    pub fn new(agent: BaseAgent) -> Self {
        Self {
            agent,
            skip_summarization: false,
        }
    }

    pub fn with_skip_summarization(mut self, skip_summarization: bool) -> Self {
        self.skip_summarization = skip_summarization;
        self
    }
}

impl BaseTool for AgentTool {
    fn name(&self) -> &str {
        self.agent.name()
    }

    fn description(&self) -> &str {
        self.agent.description()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters: Some(declaration_schema()),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            if self.skip_summarization {
                tool_context.actions_mut().skip_summarization = true;
            }

            let request_text = match args.get("request") {
                Some(Value::String(text)) => text.clone(),
                _ => rusty_serde::json::to_string(&Value::Map(
                    args.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                ))
                .unwrap_or_default(),
            };
            let content = Content::new("user", vec![Part::text(request_text)]);

            let parent_user_id = tool_context.invocation_context().session.user_id.clone();
            let parent_app_name = tool_context.invocation_context().session.app_name.clone();
            let child_app_name = if parent_app_name.is_empty() {
                self.agent.name().to_string()
            } else {
                parent_app_name
            };

            let filtered_state: BTreeMap<String, Value> = tool_context
                .state()
                .to_map()
                .into_iter()
                .filter(|(key, _)| !key.starts_with("_adk"))
                .collect();

            let session_service: Arc<dyn SessionService + Send + Sync> =
                Arc::new(InMemorySessionService::new());
            let session = session_service
                .create_session(&child_app_name, &parent_user_id, Some(filtered_state), None)
                .await
                .map_err(|err| ToolError::NestedRunFailed(err.to_string()))?;

            let forwarding_artifact_service =
                ForwardingArtifactService::new(tool_context).map(Arc::new);
            let mut runner = Runner::new(child_app_name, self.agent.clone(), session_service);
            if let Some(forwarding_artifact_service) = &forwarding_artifact_service {
                runner = runner.with_artifact_service(forwarding_artifact_service.clone());
            }

            let events = runner
                .run_async(&session.user_id, &session.id, content)
                .await
                .map_err(|err| ToolError::NestedRunFailed(err.to_string()))?;

            let mut last_content: Option<Content> = None;
            let mut last_error_message: Option<String> = None;
            for event in &events {
                if !event.actions.state_delta.is_empty() {
                    let delta: BTreeMap<String, Value> = event
                        .actions
                        .state_delta
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    tool_context.state_mut().update(delta);
                }
                if let Some(error_message) = &event.error_message {
                    last_error_message = Some(error_message.clone());
                }
                if let Some(event_content) = &event.content {
                    last_content = Some(event_content.clone());
                }
            }

            if let Some(forwarding_artifact_service) = &forwarding_artifact_service {
                tool_context
                    .actions_mut()
                    .artifact_delta
                    .extend(forwarding_artifact_service.take_artifact_delta());
            }

            runner.close().await;

            let Some(last_content) = last_content else {
                return Ok(Value::String(last_error_message.unwrap_or_default()));
            };
            let merged_text = last_content
                .parts
                .iter()
                .filter(|part| part.thought != Some(true))
                .map(part_to_text)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if merged_text.is_empty() {
                if let Some(error_message) = last_error_message {
                    return Ok(Value::String(error_message));
                }
            }
            Ok(Value::String(merged_text))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::base_agent::{AgentBehavior, NoopBehavior};
    use adk_agents::context::Context;
    use adk_agents::invocation_context::{InvocationContext, InvocationContextBuilder};
    use adk_agents::services::ArtifactService;
    use adk_agents::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_events::Event;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[test]
    fn declaration_uses_the_generic_request_parameter() {
        let agent = BaseAgent::new("helper", NoopBehavior).unwrap();
        let tool = AgentTool::new(agent);
        let declaration = tool.get_declaration().unwrap();
        assert_eq!(declaration.name.as_deref(), Some("helper"));
        assert_eq!(declaration.parameters, Some(declaration_schema()));
    }

    #[rusty_tokio::test]
    async fn skip_summarization_sets_the_action_before_running() {
        let agent = BaseAgent::new("helper", NoopBehavior).unwrap();
        let tool = AgentTool::new(agent).with_skip_summarization(true);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));

        let _ = tool.run_async(&args, &mut context).await;
        assert!(context.actions().skip_summarization);
    }

    #[rusty_tokio::test]
    async fn a_no_op_nested_agent_falls_back_to_an_empty_string() {
        let agent = BaseAgent::new("helper", NoopBehavior).unwrap();
        let tool = AgentTool::new(agent);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String(String::new()));
    }

    type TestFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

    struct RepliesWithText;

    impl AgentBehavior for RepliesWithText {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut InvocationContext,
        ) -> TestFuture<'a, Result<Vec<Event>, adk_agents::base_agent::AgentRunError>> {
            Box::pin(async move {
                let mut event =
                    Event::new(ctx.invocation_id.clone(), "helper", NodeInfo::new("helper"));
                event.content = Some(Content::new(
                    "model",
                    vec![Part::text("hello from nested agent")],
                ));
                event
                    .actions
                    .state_delta
                    .insert("nested_key".to_string(), Value::Bool(true));
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> TestFuture<'a, Result<Vec<Event>, adk_agents::base_agent::AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[rusty_tokio::test]
    async fn merges_the_final_response_text_and_forwards_state_delta() {
        let agent = BaseAgent::new("helper", RepliesWithText).unwrap();
        let tool = AgentTool::new(agent);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String("hello from nested agent".to_string()));
        assert_eq!(context.state().get("nested_key"), Some(&Value::Bool(true)));
    }

    struct StubParentArtifactService {
        stored: std::sync::Mutex<BTreeMap<String, Value>>,
    }

    impl StubParentArtifactService {
        fn new() -> Self {
            Self {
                stored: std::sync::Mutex::new(BTreeMap::new()),
            }
        }
    }

    impl adk_agents::services::ArtifactService for StubParentArtifactService {
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
            _custom_metadata: Option<BTreeMap<String, Value>>,
        ) -> i64 {
            self.stored
                .lock()
                .unwrap()
                .insert(filename.to_string(), artifact);
            3
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<adk_agents::services::ArtifactVersion> {
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
            Vec::new()
        }

        fn list_artifact_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<adk_agents::services::ArtifactVersion> {
            Vec::new()
        }
    }

    fn ctx_with_artifact_service(
        service: Arc<dyn adk_agents::services::ArtifactService + Send + Sync>,
    ) -> Context {
        let mut invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        invocation_context.artifact_service = Some(service);
        Context::new(invocation_context)
    }

    struct SavesAnArtifact;

    impl AgentBehavior for SavesAnArtifact {
        fn run_async_impl<'a>(
            &'a self,
            ctx: &'a mut InvocationContext,
        ) -> TestFuture<'a, Result<Vec<Event>, adk_agents::base_agent::AgentRunError>> {
            Box::pin(async move {
                let version = ctx.artifact_service.as_ref().unwrap().save_artifact(
                    &ctx.session.app_name,
                    &ctx.session.user_id,
                    &ctx.session.id,
                    "nested.txt",
                    Value::String("nested contents".to_string()),
                    None,
                );
                let mut event =
                    Event::new(ctx.invocation_id.clone(), "helper", NodeInfo::new("helper"));
                event.content = Some(Content::new(
                    "model",
                    vec![Part::text(format!("saved v{version}"))],
                ));
                Ok(vec![event])
            })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> TestFuture<'a, Result<Vec<Event>, adk_agents::base_agent::AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[rusty_tokio::test]
    async fn forwards_artifact_saves_to_the_parents_real_service_and_merges_the_delta() {
        let agent = BaseAgent::new("helper", SavesAnArtifact).unwrap();
        let tool = AgentTool::new(agent);
        let parent_service = Arc::new(StubParentArtifactService::new());
        let mut context = ctx_with_artifact_service(parent_service.clone());
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));

        let _ = tool.run_async(&args, &mut context).await.unwrap();

        assert_eq!(
            parent_service.load_artifact("app", "user", "s1", "nested.txt", None),
            Some(Value::String("nested contents".to_string()))
        );
        assert_eq!(context.actions().artifact_delta.get("nested.txt"), Some(&3));
    }

    #[rusty_tokio::test]
    async fn runs_normally_when_the_parent_has_no_artifact_service() {
        let agent = BaseAgent::new("helper", RepliesWithText).unwrap();
        let tool = AgentTool::new(agent);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("request".to_string(), Value::String("hi".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(result, Value::String("hello from nested agent".to_string()));
        assert!(context.actions().artifact_delta.is_empty());
    }
}
