//! Capabilities C0643-C0646/C0648-C0649: the feature-flag registry,
//! ported from `google.adk.features._feature_registry`.
//!
//! **Scope of this batch**: `_feature_registry.py`'s content only.
//! `experimental`/`working_in_progress`/`stable` (C0647,
//! `_feature_decorator.py`) has no clean Rust analog — Rust has no
//! runtime decorators to gate an arbitrary object behind a feature
//! flag the way a Python `@decorator` can wrap a function/class. That
//! row stays `REQUIRED`, undecided, rather than force-fit a partial
//! substitute here. Since decorator-driven auto-registration
//! (`_register_feature`) is the *only* way the source ever adds a
//! `FeatureName` not already in the static table, and this batch
//! doesn't port the decorators, this port's registry is a fixed,
//! exhaustive `match` rather than a mutable `dict` — every
//! `FeatureName` variant is guaranteed a [`FeatureConfig`] at compile
//! time, so [`feature_config`] can never "miss" the way
//! `_get_feature_config` can return `None` in the source. That makes
//! the source's "raises `ValueError`` for an unregistered name" branch
//! (`is_feature_enabled`, `override_feature_enabled`,
//! `temporary_feature_override` all have one) structurally
//! unreachable here — not a narrowing, a compile-time strengthening
//! of the same guarantee.
//!
//! **Member count, disclosed**: the manifest's own C0643 row estimates
//! "37 members"; this file's `FeatureName` has 38, counted directly
//! off the source file at port time (`_feature_registry.py`, read
//! 2026-08-24) — the manifest description is an approximation, the
//! source is ground truth (same reconciliation this session already
//! did for `OAuth2Auth`'s field count).
//!
//! **The "private" member**: `_MCP_GRACEFUL_ERROR_HANDLING`'s leading
//! underscore is a Python-convention-only privacy marker ("nothing
//! should import this enum member by name"), not enforced by the
//! language — the source's own registry still keys a real dict entry
//! off it. Rust has no equivalent partial-privacy signal for one enum
//! variant among public ones, so [`FeatureName::McpGracefulErrorHandling`]
//! is a public variant like any other, carrying the same "internal
//! kill-switch, don't reference directly" doc note instead of any
//! enforced restriction.
//!
//! **`_emit_non_stable_warning_once`, adapted**: this workspace has no
//! logging/warning framework (a repeatedly disclosed gap this
//! session — e.g. `preload_memory_tool.rs`'s dropped
//! `logging.warning`). The once-per-feature *tracking* state (a
//! process-wide set, matching the source's `_WARNED_FEATURES`) is
//! ported faithfully; the notice itself is emitted via `eprintln!`
//! rather than `warnings.warn` — the same precedent
//! `adk_models::capabilities::is_enterprise_mode_enabled`'s own
//! deprecation notice already established, not a new adaptation
//! invented here. Callers can't filter/capture it the way Python's
//! `warnings` module allows, a real but narrow gap.
//!
//! **`temporary_feature_override`, adapted**: the source is a
//! `@contextmanager` (`with temporary_feature_override(...): ...`,
//! restoring on exit including via exception unwind). This port is an
//! RAII guard ([`TemporaryFeatureOverride`]) whose [`Drop`] impl
//! restores the prior state — the standard Rust idiom for "run this on
//! scope exit, including on panic-unwind," matching the source's
//! `try`/`finally` semantics exactly (a `Drop` runs during unwinding
//! just as `finally` does).

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// `features._feature_registry.FeatureName` — 38 members (see the
/// module doc for the manifest's "37" estimate vs. this count).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureName {
    AgentConfig,
    AgentState,
    AuthenticatedFunctionTool,
    BaseAuthenticatedTool,
    BigQueryToolset,
    BigQueryToolConfig,
    BigtableToolSettings,
    BigtableToolset,
    ComputerUse,
    DataAgentToolConfig,
    DataAgentToolset,
    DynamicInstructionRouting,
    DaytonaEnvironment,
    E2bEnvironment,
    EnvironmentSimulation,
    EventarcToolConfig,
    EventarcToolset,
    GcsAdminToolset,
    GcsToolSettings,
    GcsToolset,
    GoogleCredentialsConfig,
    GoogleTool,
    JsonSchemaForFuncDecl,
    McpAgentServer,
    /// Private in the source (leading underscore) — see the module
    /// doc. Flipped via `ADK_ENABLE_MCP_GRACEFUL_ERROR_HANDLING=1`.
    McpGracefulErrorHandling,
    ProgressiveSseStreaming,
    PubsubToolConfig,
    PubsubToolset,
    SkillToolset,
    SpannerToolset,
    SpannerAdminToolset,
    SpannerToolSettings,
    SpannerVectorStore,
    ToolConfig,
    ToolConfirmation,
    PluggableAuth,
    SnakeCaseSkillName,
    InMemorySessionServiceLightCopy,
}

impl FeatureName {
    /// The registry key / env-var-name-fragment string, matching the
    /// source `Enum`'s `.value` exactly.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AgentConfig => "AGENT_CONFIG",
            Self::AgentState => "AGENT_STATE",
            Self::AuthenticatedFunctionTool => "AUTHENTICATED_FUNCTION_TOOL",
            Self::BaseAuthenticatedTool => "BASE_AUTHENTICATED_TOOL",
            Self::BigQueryToolset => "BIG_QUERY_TOOLSET",
            Self::BigQueryToolConfig => "BIG_QUERY_TOOL_CONFIG",
            Self::BigtableToolSettings => "BIGTABLE_TOOL_SETTINGS",
            Self::BigtableToolset => "BIGTABLE_TOOLSET",
            Self::ComputerUse => "COMPUTER_USE",
            Self::DataAgentToolConfig => "DATA_AGENT_TOOL_CONFIG",
            Self::DataAgentToolset => "DATA_AGENT_TOOLSET",
            Self::DynamicInstructionRouting => "DYNAMIC_INSTRUCTION_ROUTING",
            Self::DaytonaEnvironment => "DAYTONA_ENVIRONMENT",
            Self::E2bEnvironment => "E2B_ENVIRONMENT",
            Self::EnvironmentSimulation => "ENVIRONMENT_SIMULATION",
            Self::EventarcToolConfig => "EVENTARC_TOOL_CONFIG",
            Self::EventarcToolset => "EVENTARC_TOOLSET",
            Self::GcsAdminToolset => "GCS_ADMIN_TOOLSET",
            Self::GcsToolSettings => "GCS_TOOL_SETTINGS",
            Self::GcsToolset => "GCS_TOOLSET",
            Self::GoogleCredentialsConfig => "GOOGLE_CREDENTIALS_CONFIG",
            Self::GoogleTool => "GOOGLE_TOOL",
            Self::JsonSchemaForFuncDecl => "JSON_SCHEMA_FOR_FUNC_DECL",
            Self::McpAgentServer => "MCP_AGENT_SERVER",
            Self::McpGracefulErrorHandling => "MCP_GRACEFUL_ERROR_HANDLING",
            Self::ProgressiveSseStreaming => "PROGRESSIVE_SSE_STREAMING",
            Self::PubsubToolConfig => "PUBSUB_TOOL_CONFIG",
            Self::PubsubToolset => "PUBSUB_TOOLSET",
            Self::SkillToolset => "SKILL_TOOLSET",
            Self::SpannerToolset => "SPANNER_TOOLSET",
            Self::SpannerAdminToolset => "SPANNER_ADMIN_TOOLSET",
            Self::SpannerToolSettings => "SPANNER_TOOL_SETTINGS",
            Self::SpannerVectorStore => "SPANNER_VECTOR_STORE",
            Self::ToolConfig => "TOOL_CONFIG",
            Self::ToolConfirmation => "TOOL_CONFIRMATION",
            Self::PluggableAuth => "PLUGGABLE_AUTH",
            Self::SnakeCaseSkillName => "SNAKE_CASE_SKILL_NAME",
            Self::InMemorySessionServiceLightCopy => "IN_MEMORY_SESSION_SERVICE_LIGHT_COPY",
        }
    }
}

/// `features._feature_registry.FeatureStage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureStage {
    /// Work in progress, not functioning completely. ADK internal
    /// development only.
    Wip,
    /// Feature works but the API may change.
    Experimental,
    /// Production-ready, no breaking changes without a MAJOR version
    /// bump.
    Stable,
}

impl FeatureStage {
    fn label(self) -> &'static str {
        match self {
            Self::Wip => "WIP",
            Self::Experimental => "EXPERIMENTAL",
            Self::Stable => "STABLE",
        }
    }
}

/// `features._feature_registry.FeatureConfig`.
#[derive(Debug, Clone, Copy)]
pub struct FeatureConfig {
    pub stage: FeatureStage,
    pub default_on: bool,
}

const fn config(stage: FeatureStage, default_on: bool) -> FeatureConfig {
    FeatureConfig { stage, default_on }
}

/// `features._feature_registry._get_feature_config` — a fixed,
/// exhaustive match rather than a runtime dict lookup; see the module
/// doc for why that makes the source's `None` case unreachable here.
pub fn feature_config(feature_name: FeatureName) -> FeatureConfig {
    use FeatureName::*;
    use FeatureStage::{Experimental, Stable, Wip};
    match feature_name {
        AgentConfig => config(Experimental, true),
        AgentState => config(Experimental, true),
        AuthenticatedFunctionTool => config(Experimental, true),
        BaseAuthenticatedTool => config(Experimental, true),
        BigQueryToolset => config(Stable, true),
        BigQueryToolConfig => config(Stable, true),
        BigtableToolSettings => config(Experimental, true),
        BigtableToolset => config(Experimental, true),
        ComputerUse => config(Experimental, true),
        DataAgentToolConfig => config(Stable, true),
        DataAgentToolset => config(Stable, true),
        DynamicInstructionRouting => config(Experimental, false),
        DaytonaEnvironment => config(Experimental, true),
        E2bEnvironment => config(Experimental, true),
        EnvironmentSimulation => config(Experimental, true),
        EventarcToolConfig => config(Experimental, true),
        EventarcToolset => config(Experimental, true),
        GcsAdminToolset => config(Experimental, true),
        GcsToolSettings => config(Experimental, true),
        GcsToolset => config(Experimental, true),
        GoogleCredentialsConfig => config(Experimental, true),
        GoogleTool => config(Experimental, true),
        JsonSchemaForFuncDecl => config(Experimental, true),
        McpAgentServer => config(Experimental, true),
        McpGracefulErrorHandling => config(Experimental, true),
        ProgressiveSseStreaming => config(Experimental, true),
        PubsubToolConfig => config(Experimental, true),
        PubsubToolset => config(Experimental, true),
        SkillToolset => config(Stable, true),
        SpannerToolset => config(Experimental, true),
        SpannerAdminToolset => config(Experimental, true),
        SpannerToolSettings => config(Experimental, true),
        SpannerVectorStore => config(Experimental, true),
        ToolConfig => config(Experimental, true),
        ToolConfirmation => config(Experimental, true),
        PluggableAuth => config(Experimental, true),
        SnakeCaseSkillName => config(Experimental, false),
        InMemorySessionServiceLightCopy => config(Wip, false),
    }
}

fn overrides() -> &'static Mutex<HashMap<FeatureName, bool>> {
    static OVERRIDES: OnceLock<Mutex<HashMap<FeatureName, bool>>> = OnceLock::new();
    OVERRIDES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn warned() -> &'static Mutex<HashSet<FeatureName>> {
    static WARNED: OnceLock<Mutex<HashSet<FeatureName>>> = OnceLock::new();
    WARNED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// C0648: `ADK_ENABLE_<NAME>`/`ADK_DISABLE_<NAME>` truthiness check —
/// `'1'`/`'true'`, case-insensitive.
fn is_env_enabled(var_name: &str) -> bool {
    std::env::var(var_name)
        .map(|value| {
            let lower = value.to_lowercase();
            lower == "true" || lower == "1"
        })
        .unwrap_or(false)
}

/// C0649: emits the one-time non-stable-feature notice — see the
/// module doc for the `eprintln!`-not-`warnings.warn` adaptation.
fn emit_non_stable_warning_once(feature_name: FeatureName, stage: FeatureStage) {
    let mut warned = warned().lock().unwrap();
    if warned.insert(feature_name) {
        eprintln!(
            "[{}] feature {} is enabled.",
            stage.label(),
            feature_name.as_str()
        );
    }
}

/// C0646: programmatically overrides a feature's enabled state,
/// process-wide — highest priority, superseding environment
/// variables and registry defaults.
pub fn override_feature_enabled(feature_name: FeatureName, enabled: bool) {
    overrides().lock().unwrap().insert(feature_name, enabled);
}

/// C0645: checks whether a feature is enabled at runtime. Priority
/// order (highest to lowest): programmatic overrides
/// ([`override_feature_enabled`]/[`TemporaryFeatureOverride`]),
/// `ADK_ENABLE_<NAME>`/`ADK_DISABLE_<NAME>` environment variables,
/// then the registry default.
pub fn is_feature_enabled(feature_name: FeatureName) -> bool {
    let stage = feature_config(feature_name).stage;

    if let Some(&enabled) = overrides().lock().unwrap().get(&feature_name) {
        if enabled && stage != FeatureStage::Stable {
            emit_non_stable_warning_once(feature_name, stage);
        }
        return enabled;
    }

    let enable_var = format!("ADK_ENABLE_{}", feature_name.as_str());
    let disable_var = format!("ADK_DISABLE_{}", feature_name.as_str());
    if is_env_enabled(&enable_var) {
        if stage != FeatureStage::Stable {
            emit_non_stable_warning_once(feature_name, stage);
        }
        return true;
    }
    if is_env_enabled(&disable_var) {
        return false;
    }

    let default_on = feature_config(feature_name).default_on;
    if stage != FeatureStage::Stable && default_on {
        emit_non_stable_warning_once(feature_name, stage);
    }
    default_on
}

/// C0646: an RAII guard restoring a feature's prior override state on
/// drop — the source's `temporary_feature_override` `@contextmanager`,
/// ported as the standard Rust "run on scope exit" idiom (see the
/// module doc for why this matches the source's `try`/`finally`
/// semantics, including under panic-unwind).
#[must_use = "the override is restored when this guard is dropped; binding it to `_` restores immediately"]
pub struct TemporaryFeatureOverride {
    feature_name: FeatureName,
    had_override: bool,
    original_value: bool,
}

impl TemporaryFeatureOverride {
    pub fn new(feature_name: FeatureName, enabled: bool) -> Self {
        let mut overrides = overrides().lock().unwrap();
        let had_override = overrides.contains_key(&feature_name);
        let original_value = overrides.get(&feature_name).copied().unwrap_or(false);
        overrides.insert(feature_name, enabled);
        Self {
            feature_name,
            had_override,
            original_value,
        }
    }
}

impl Drop for TemporaryFeatureOverride {
    fn drop(&mut self) {
        let mut overrides = overrides().lock().unwrap();
        if self.had_override {
            overrides.insert(self.feature_name, self.original_value);
        } else {
            overrides.remove(&self.feature_name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // Overrides/env vars are process-wide state; serialize tests that
    // touch them so they don't race each other.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn clear_overrides() {
        overrides().lock().unwrap().clear();
        warned().lock().unwrap().clear();
    }

    #[test]
    fn registry_default_is_used_absent_override_or_env_var() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        unsafe {
            std::env::remove_var("ADK_ENABLE_DYNAMIC_INSTRUCTION_ROUTING");
            std::env::remove_var("ADK_DISABLE_DYNAMIC_INSTRUCTION_ROUTING");
        }
        assert!(!is_feature_enabled(FeatureName::DynamicInstructionRouting));
        assert!(is_feature_enabled(FeatureName::AgentConfig));
    }

    #[test]
    fn enable_env_var_turns_on_a_default_off_feature() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        unsafe {
            std::env::set_var("ADK_ENABLE_DYNAMIC_INSTRUCTION_ROUTING", "true");
        }
        let enabled = is_feature_enabled(FeatureName::DynamicInstructionRouting);
        unsafe {
            std::env::remove_var("ADK_ENABLE_DYNAMIC_INSTRUCTION_ROUTING");
        }
        assert!(enabled);
    }

    #[test]
    fn disable_env_var_turns_off_a_default_on_feature() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        unsafe {
            std::env::set_var("ADK_DISABLE_AGENT_CONFIG", "1");
        }
        let enabled = is_feature_enabled(FeatureName::AgentConfig);
        unsafe {
            std::env::remove_var("ADK_DISABLE_AGENT_CONFIG");
        }
        assert!(!enabled);
    }

    #[test]
    fn programmatic_override_wins_over_env_vars() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        unsafe {
            std::env::set_var("ADK_DISABLE_AGENT_CONFIG", "1");
        }
        override_feature_enabled(FeatureName::AgentConfig, true);
        let enabled = is_feature_enabled(FeatureName::AgentConfig);
        unsafe {
            std::env::remove_var("ADK_DISABLE_AGENT_CONFIG");
        }
        clear_overrides();
        assert!(enabled);
    }

    #[test]
    fn temporary_override_restores_the_prior_registry_default_on_drop() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        assert!(!is_feature_enabled(FeatureName::DynamicInstructionRouting));
        {
            let _temp = TemporaryFeatureOverride::new(FeatureName::DynamicInstructionRouting, true);
            assert!(is_feature_enabled(FeatureName::DynamicInstructionRouting));
        }
        assert!(!is_feature_enabled(FeatureName::DynamicInstructionRouting));
    }

    #[test]
    fn temporary_override_restores_a_prior_programmatic_override_on_drop() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        override_feature_enabled(FeatureName::AgentConfig, false);
        {
            let _temp = TemporaryFeatureOverride::new(FeatureName::AgentConfig, true);
            assert!(is_feature_enabled(FeatureName::AgentConfig));
        }
        assert!(!is_feature_enabled(FeatureName::AgentConfig));
        clear_overrides();
    }

    #[test]
    fn as_str_matches_the_source_enum_value() {
        assert_eq!(FeatureName::AgentConfig.as_str(), "AGENT_CONFIG");
        assert_eq!(
            FeatureName::McpGracefulErrorHandling.as_str(),
            "MCP_GRACEFUL_ERROR_HANDLING"
        );
        assert_eq!(
            FeatureName::InMemorySessionServiceLightCopy.as_str(),
            "IN_MEMORY_SESSION_SERVICE_LIGHT_COPY"
        );
    }

    #[test]
    fn stable_features_never_warn() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        assert_eq!(
            feature_config(FeatureName::BigQueryToolset).stage,
            FeatureStage::Stable
        );
        assert!(is_feature_enabled(FeatureName::BigQueryToolset));
        assert!(warned().lock().unwrap().is_empty());
    }

    #[test]
    fn non_stable_default_on_feature_warns_exactly_once() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_overrides();
        assert!(is_feature_enabled(FeatureName::AgentConfig));
        assert!(is_feature_enabled(FeatureName::AgentConfig));
        assert!(warned().lock().unwrap().contains(&FeatureName::AgentConfig));
    }
}
