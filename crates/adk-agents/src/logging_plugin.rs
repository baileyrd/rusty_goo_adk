//! Capability C0362 (partial): `plugins.logging_plugin.LoggingPlugin`,
//! ported from `google.adk.plugins.logging_plugin`.
//!
//! A console-debugging plugin that logs at each callback point — "not a
//! replacement of existing logging in ADK," per the source's own doc,
//! just a terminal-visible trace and a demo of the plugin surface.
//!
//! **6 of 13 hooks ported, 7 deferred**: `on_user_message_callback`,
//! `before_run_callback`, `on_event_callback`, `after_run_callback`,
//! `before_agent_callback`, `after_agent_callback` touch only fields
//! already present in this port. The remaining 7
//! (`before_model_callback`/`after_model_callback`/`on_model_error_callback`
//! and `before_tool_callback`/`after_tool_callback`/`on_tool_error_callback`)
//! need `BasePlugin`'s model-level and tool-level hooks (C0355/C0356),
//! which don't exist yet — `adk-tools`/`adk-models` already depend on
//! `adk-agents`, so `adk-agents::services::BasePlugin` can't grow a
//! `LlmRequest`/`LlmResponse`/`BaseTool`/`ToolContext`-shaped method
//! without a dependency cycle; that needs a crate above both, the same
//! disclosed blocker `services.rs`'s own module doc already names. This
//! ships as an explicitly disclosed Partial, matching this crate's
//! established convention for a capability blocked mid-way by another
//! not-yet-landed one (e.g. C0895/C0170).
//!
//! **`print()`, not the `logging` module**: the source's own `_log`
//! calls bare `print()` with ANSI grey codes (`\033[90m`...`\033[0m`) —
//! it's a stdout console logger by design, not routed through Python's
//! `logging` module. This port's `println!` is therefore the faithful
//! translation, not a substitution for a missing logging framework (the
//! "no logging framework adopted" disclosure used elsewhere in this
//! migration doesn't apply here — the source itself doesn't use one for
//! this specific plugin).
//!
//! **`callback_context.agent_name`, not needed**: the source's
//! `before_agent_callback`/`after_agent_callback` read the running
//! agent's name off `callback_context.agent_name` (a derived property).
//! This port's `BasePlugin::before_agent_callback`/`after_agent_callback`
//! already receive `agent: &BaseAgent` directly as their own parameter
//! (this crate's own established hook signature, not specific to this
//! plugin) — so this reads `agent.name()` straight from that instead.
//!
//! **Printed output, not parity-tested**: matching this crate's general
//! posture toward other `println!`/`eprintln!` side effects (e.g.
//! `run_config.rs`'s env-var-parse warning), the exact printed text
//! isn't asserted against — [`format_content`]/[`format_args`] (the
//! source's `_format_content`/`_format_args`, where the actual
//! formatting logic lives) are unit-tested directly as pure functions,
//! and the hooks themselves are tested for their return-value contract
//! (always `None`, never short-circuits, never panics).

use std::collections::BTreeMap;

use adk_events::Event;
use adk_genai::content::Content;
use rusty_serde::value::Value;

use crate::base_agent::BaseAgent;
use crate::context::Context;
use crate::services::{BasePlugin, BoxFuture};

/// `plugins.logging_plugin.LoggingPlugin`.
pub struct LoggingPlugin {
    name: String,
}

impl LoggingPlugin {
    pub fn new() -> Self {
        Self {
            name: "logging_plugin".to_string(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn log(&self, message: impl AsRef<str>) {
        println!("\x1b[90m[{}] {}\x1b[0m", self.name, message.as_ref());
    }
}

impl Default for LoggingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// `_format_content`: renders each part as `kind: value`, truncating any
/// `text` part past `max_length`.
pub fn format_content(content: Option<&Content>, max_length: usize) -> String {
    let Some(content) = content else {
        return "None".to_string();
    };
    if content.parts.is_empty() {
        return "None".to_string();
    }

    content
        .parts
        .iter()
        .map(|part| {
            if let Some(text) = &part.text {
                let trimmed = text.trim();
                if trimmed.chars().count() > max_length {
                    let truncated: String = trimmed.chars().take(max_length).collect();
                    format!("text: '{truncated}...'")
                } else {
                    format!("text: '{trimmed}'")
                }
            } else if let Some(function_call) = &part.function_call {
                format!(
                    "function_call: {}",
                    function_call.name.as_deref().unwrap_or_default()
                )
            } else if let Some(function_response) = &part.function_response {
                format!(
                    "function_response: {}",
                    function_response.name.as_deref().unwrap_or_default()
                )
            } else if part.code_execution_result.is_some() {
                "code_execution_result".to_string()
            } else {
                "other_part".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

/// `_format_args`: `{key: value, ...}`-style rendering, truncated past
/// `max_length`.
pub fn format_args(args: &BTreeMap<String, Value>, max_length: usize) -> String {
    if args.is_empty() {
        return "{}".to_string();
    }
    let formatted = format!(
        "{}",
        Value::Map(args.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    );
    if formatted.chars().count() > max_length {
        let truncated: String = formatted.chars().take(max_length).collect();
        format!("{truncated}...}}")
    } else {
        formatted
    }
}

fn agent_label(agent: Option<&BaseAgent>) -> &str {
    agent.map(BaseAgent::name).unwrap_or("Unknown")
}

impl BasePlugin for LoggingPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_user_message_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
        user_message: &'a Content,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            let ic = invocation_context.invocation_context();
            self.log("🚀 USER MESSAGE RECEIVED");
            self.log(format!("   Invocation ID: {}", ic.invocation_id));
            self.log(format!("   Session ID: {}", ic.session.id));
            self.log(format!("   User ID: {}", ic.session.user_id));
            self.log(format!("   App Name: {}", ic.session.app_name));
            self.log(format!("   Root Agent: {}", agent_label(ic.agent.as_ref())));
            self.log(format!(
                "   User Content: {}",
                format_content(Some(user_message), 200)
            ));
            if let Some(branch) = &ic.branch {
                self.log(format!("   Branch: {branch}"));
            }
            None
        })
    }

    fn before_run_callback<'a>(
        &'a self,
        invocation_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            let ic = invocation_context.invocation_context();
            self.log("🏃 INVOCATION STARTING");
            self.log(format!("   Invocation ID: {}", ic.invocation_id));
            self.log(format!(
                "   Starting Agent: {}",
                agent_label(ic.agent.as_ref())
            ));
            None
        })
    }

    fn on_event_callback<'a>(
        &'a self,
        _invocation_context: &'a mut Context,
        event: &'a Event,
    ) -> BoxFuture<'a, Option<Event>> {
        Box::pin(async move {
            self.log("📢 EVENT YIELDED");
            self.log(format!("   Event ID: {}", event.id));
            self.log(format!("   Author: {}", event.author));
            self.log(format!(
                "   Content: {}",
                format_content(event.content.as_ref(), 200)
            ));
            self.log(format!("   Final Response: {}", event.is_final_response()));

            let function_calls = event.get_function_calls();
            if !function_calls.is_empty() {
                let names: Vec<&str> = function_calls
                    .iter()
                    .map(|fc| fc.name.as_deref().unwrap_or_default())
                    .collect();
                self.log(format!("   Function Calls: {names:?}"));
            }

            let function_responses = event.get_function_responses();
            if !function_responses.is_empty() {
                let names: Vec<&str> = function_responses
                    .iter()
                    .map(|fr| fr.name.as_deref().unwrap_or_default())
                    .collect();
                self.log(format!("   Function Responses: {names:?}"));
            }

            if let Some(ids) = &event.long_running_tool_ids {
                if !ids.is_empty() {
                    self.log(format!("   Long Running Tools: {ids:?}"));
                }
            }

            None
        })
    }

    fn after_run_callback<'a>(&'a self, invocation_context: &'a mut Context) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let ic = invocation_context.invocation_context();
            self.log("✅ INVOCATION COMPLETED");
            self.log(format!("   Invocation ID: {}", ic.invocation_id));
            self.log(format!(
                "   Final Agent: {}",
                agent_label(ic.agent.as_ref())
            ));
        })
    }

    fn before_agent_callback<'a>(
        &'a self,
        agent: &'a BaseAgent,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            self.log("🤖 AGENT STARTING");
            self.log(format!("   Agent Name: {}", agent.name()));
            self.log(format!(
                "   Invocation ID: {}",
                callback_context.invocation_context().invocation_id
            ));
            if let Some(branch) = &callback_context.invocation_context().branch {
                self.log(format!("   Branch: {branch}"));
            }
            None
        })
    }

    fn after_agent_callback<'a>(
        &'a self,
        agent: &'a BaseAgent,
        callback_context: &'a mut Context,
    ) -> BoxFuture<'a, Option<Content>> {
        Box::pin(async move {
            self.log("🤖 AGENT COMPLETED");
            self.log(format!("   Agent Name: {}", agent.name()));
            self.log(format!(
                "   Invocation ID: {}",
                callback_context.invocation_context().invocation_id
            ));
            None
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invocation_context::InvocationContextBuilder;
    use crate::session::Session;
    use adk_events::node_info::NodeInfo;
    use adk_genai::content::{FunctionCall, Part};

    fn context() -> Context {
        let invocation_context =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        Context::new(invocation_context)
    }

    #[test]
    fn format_content_renders_none_for_an_empty_or_missing_content() {
        assert_eq!(format_content(None, 200), "None");
        assert_eq!(
            format_content(Some(&Content::new("user", vec![])), 200),
            "None"
        );
    }

    #[test]
    fn format_content_truncates_long_text() {
        let content = Content::user_text("a".repeat(250));
        let rendered = format_content(Some(&content), 200);
        assert!(rendered.starts_with("text: 'aaa"));
        assert!(rendered.ends_with("...'"));
    }

    #[test]
    fn format_content_labels_function_calls_and_responses() {
        let content = Content::new(
            "model",
            vec![Part {
                function_call: Some(FunctionCall {
                    name: Some("do_thing".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        );
        assert_eq!(
            format_content(Some(&content), 200),
            "function_call: do_thing"
        );
    }

    #[test]
    fn format_args_renders_empty_braces_for_no_args() {
        assert_eq!(format_args(&BTreeMap::new(), 300), "{}");
    }

    #[test]
    fn format_args_truncates_past_max_length() {
        let mut args = BTreeMap::new();
        args.insert("key".to_string(), Value::String("v".repeat(400)));
        let rendered = format_args(&args, 10);
        assert!(rendered.chars().count() <= 14);
        assert!(rendered.ends_with("...}"));
    }

    #[rusty_tokio::test]
    async fn on_user_message_callback_never_short_circuits() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        let message = Content::user_text("hi");
        assert_eq!(
            plugin.on_user_message_callback(&mut ctx, &message).await,
            None
        );
    }

    #[rusty_tokio::test]
    async fn before_run_callback_never_short_circuits() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        assert_eq!(plugin.before_run_callback(&mut ctx).await, None);
    }

    #[rusty_tokio::test]
    async fn on_event_callback_never_replaces_the_event() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        let event = Event::new("inv-1", "model", NodeInfo::new("root"));
        assert_eq!(plugin.on_event_callback(&mut ctx, &event).await, None);
    }

    #[rusty_tokio::test]
    async fn after_run_callback_runs_without_error() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        plugin.after_run_callback(&mut ctx).await;
    }

    #[rusty_tokio::test]
    async fn before_agent_callback_never_short_circuits() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        let agent = BaseAgent::new("agent", crate::base_agent::NoopBehavior).unwrap();
        assert_eq!(plugin.before_agent_callback(&agent, &mut ctx).await, None);
    }

    #[rusty_tokio::test]
    async fn after_agent_callback_never_short_circuits() {
        let plugin = LoggingPlugin::new();
        let mut ctx = context();
        let agent = BaseAgent::new("agent", crate::base_agent::NoopBehavior).unwrap();
        assert_eq!(plugin.after_agent_callback(&agent, &mut ctx).await, None);
    }
}
