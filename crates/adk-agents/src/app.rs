//! Capabilities C0279-C0280: `App`, ported from `google.adk.apps.app`.
//!
//! **`root_agent`, narrowed**: the source's field is
//! `Union[BaseAgent, BaseNode, None]` (the source's own comment: "Change to
//! `Union[BaseAgent, BaseNode, None]` after dependency is fixed" — this is
//! already the aspirational, not-yet-real shape even in the source). The
//! `BaseNode`/workflow-graph engine (C0298-C0306) isn't built in this port,
//! so `App::root_agent` is `BaseAgent`-only for now and is a required
//! constructor argument rather than an `Option` — the source's
//! `_validate` model-validator rejects a `None` root_agent anyway, so this
//! just enforces the same invariant at the type level instead of at
//! runtime.
//!
//! **Not wired into `Runner`, deliberately, this batch**: `Runner::new`'s
//! constructor already shipped (PR #101 onward) taking an agent + services
//! directly. Accepting an `App` there instead (C0840-C0850) would change
//! that already-shipped signature — left for a follow-up batch once `App`
//! exists and can be reviewed on its own.
//!
//! **App-name validation, a distinct rule from agent-name validation**: the
//! source's `_VALID_APP_NAME_RE` (`^[a-zA-Z][a-zA-Z0-9_-]*$`) additionally
//! permits hyphens, which `base_agent::validate_name`'s identifier check
//! does not — so this is a new, separate validator, not a reuse of the
//! agent one.

use std::sync::Arc;

use crate::app_configs::{EventsCompactionConfig, ResumabilityConfig};
use crate::base_agent::BaseAgent;
use crate::context_cache_config::ContextCacheConfig;
use crate::services::BasePlugin;

/// The source's `ValueError`/`TypeError` from `validate_app_name` and
/// `App._validate`.
#[derive(Debug, rusty_err::Error)]
pub enum AppError {
    #[error(
        "Invalid app name '{0}': must start with a letter and can only consist of letters, \
         digits, underscores, and hyphens."
    )]
    InvalidName(String),
    #[error("App name cannot be 'user'; reserved for end-user input.")]
    ReservedName,
}

/// C0279: `apps.app.validate_app_name` — ensures the provided application
/// name is safe and intuitive.
pub fn validate_app_name(name: &str) -> Result<(), AppError> {
    let is_valid = {
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() => {
                chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            }
            _ => false,
        }
    };
    if !is_valid {
        return Err(AppError::InvalidName(name.to_string()));
    }
    if name == "user" {
        return Err(AppError::ReservedName);
    }
    Ok(())
}

/// C0280: `apps.app.App` — the top-level container for an agentic system
/// powered by LLMs.
pub struct App {
    pub name: String,
    pub root_agent: BaseAgent,
    pub plugins: Vec<Arc<dyn BasePlugin>>,
    pub events_compaction_config: Option<EventsCompactionConfig>,
    pub context_cache_config: Option<ContextCacheConfig>,
    pub resumability_config: Option<ResumabilityConfig>,
}

impl App {
    /// C0280: `App._validate` requires `root_agent` be provided at all —
    /// modeled here as a required constructor argument (see the module
    /// doc) rather than an `Option` callers can omit.
    pub fn new(name: impl Into<String>, root_agent: BaseAgent) -> Result<Self, AppError> {
        let name = name.into();
        validate_app_name(&name)?;
        Ok(App {
            name,
            root_agent,
            plugins: Vec::new(),
            events_compaction_config: None,
            context_cache_config: None,
            resumability_config: None,
        })
    }

    pub fn with_plugin(mut self, plugin: Arc<dyn BasePlugin>) -> Self {
        self.plugins.push(plugin);
        self
    }

    pub fn with_events_compaction_config(mut self, config: EventsCompactionConfig) -> Self {
        self.events_compaction_config = Some(config);
        self
    }

    pub fn with_context_cache_config(mut self, config: ContextCacheConfig) -> Self {
        self.context_cache_config = Some(config);
        self
    }

    pub fn with_resumability_config(mut self, config: ResumabilityConfig) -> Self {
        self.resumability_config = Some(config);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_agent::AgentBehavior;
    use crate::invocation_context::InvocationContext;
    use adk_events::Event;
    use std::future::Future;
    use std::pin::Pin;

    type TestBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    struct NoopBehavior;

    impl AgentBehavior for NoopBehavior {
        fn run_async_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> TestBoxFuture<'a, Result<Vec<Event>, crate::base_agent::AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn run_live_impl<'a>(
            &'a self,
            _ctx: &'a mut InvocationContext,
        ) -> TestBoxFuture<'a, Result<Vec<Event>, crate::base_agent::AgentRunError>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    fn agent(name: &str) -> BaseAgent {
        BaseAgent::new(name, NoopBehavior).expect("valid agent")
    }

    #[test]
    fn validate_app_name_accepts_letters_digits_underscore_and_hyphen() {
        assert!(validate_app_name("my-app_2").is_ok());
    }

    #[test]
    fn validate_app_name_rejects_a_leading_digit() {
        match validate_app_name("2app") {
            Err(AppError::InvalidName(name)) => assert_eq!(name, "2app"),
            other => panic!("expected InvalidName, got {other:?}"),
        }
    }

    #[test]
    fn validate_app_name_rejects_an_interior_invalid_character() {
        match validate_app_name("my app") {
            Err(AppError::InvalidName(name)) => assert_eq!(name, "my app"),
            other => panic!("expected InvalidName, got {other:?}"),
        }
    }

    #[test]
    fn validate_app_name_rejects_the_reserved_name_user() {
        match validate_app_name("user") {
            Err(AppError::ReservedName) => {}
            other => panic!("expected ReservedName, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_an_invalid_name() {
        assert!(App::new("2app", agent("root")).is_err());
    }

    #[test]
    fn new_defaults_plugins_and_configs_empty() {
        let app = App::new("my-app", agent("root")).expect("valid app");
        assert_eq!(app.name, "my-app");
        assert_eq!(app.root_agent.name(), "root");
        assert!(app.plugins.is_empty());
        assert!(app.events_compaction_config.is_none());
        assert!(app.context_cache_config.is_none());
        assert!(app.resumability_config.is_none());
    }

    #[test]
    fn builders_round_trip_the_configured_values() {
        let app = App::new("my-app", agent("root"))
            .expect("valid app")
            .with_events_compaction_config(EventsCompactionConfig {
                compaction_interval: Some(5),
                overlap_size: Some(1),
                ..Default::default()
            })
            .with_context_cache_config(ContextCacheConfig::default())
            .with_resumability_config(ResumabilityConfig::new(true));

        assert!(app.events_compaction_config.is_some());
        assert!(app.context_cache_config.is_some());
        assert!(app.resumability_config.unwrap().is_resumable);
    }
}
