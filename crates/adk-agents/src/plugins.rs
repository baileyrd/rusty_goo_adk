//! Capability C0352 (partial): the plugin package's public export
//! surface, ported from `google.adk.plugins.__init__`.
//!
//! **Eager vs. lazy has no Rust equivalent, so it's dropped**: the
//! source's `_LAZY_MEMBERS`/`__getattr__` split exists only to avoid
//! importing heavy plugin modules (`debug_logging_plugin`, etc.) at
//! package-load time; Rust has no import-time cost to defer in the first
//! place (a `pub mod`/`pub use` declaration doesn't eagerly execute
//! anything), so every name below is a plain `pub use` regardless of
//! which list it was on in the source.
//!
//! **Not re-exported here, disclosed — deliberately excluded, matching
//! the source's own `__all__`/`_LAZY_MEMBERS` omissions exactly**:
//! [`crate::save_files_as_artifacts_plugin::SaveFilesAsArtifactsPlugin`]
//! (real in this port, but never surfaced from the source's own package
//! root either) plus `BigQueryAgentAnalyticsPlugin`/
//! `GlobalInstructionPlugin`/`ContextFilterPlugin`/
//! `MultimodalToolResultsPlugin`/`AutoTracingPlugin` (none of which exist
//! in this port yet either way).
//!
//! **Not yet portable, disclosed**: `DebugLoggingPlugin`/
//! `ReflectAndRetryModelPlugin`/`ReflectAndRetryToolPlugin` are in the
//! source's `__all__`/`_LAZY_MEMBERS`, but none of the three exist as
//! Rust types in this port yet — all three need [`crate::services::
//! BasePlugin`]'s model-level and/or tool-level hooks (C0355/C0356),
//! which are themselves blocked on the `adk-agents`↔`adk-models`/
//! `adk-tools` crate-cycle already documented on `BasePlugin` itself.
//! This module re-exports only what actually exists; revisit once
//! C0355/C0356 unblock.

pub use crate::logging_plugin::LoggingPlugin;
pub use crate::services::{BasePlugin, PluginManager};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct StubPlugin;
    impl BasePlugin for StubPlugin {
        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn base_plugin_and_plugin_manager_are_reachable_through_the_facade() {
        let mut manager = PluginManager::new();
        manager.register_plugin(Arc::new(StubPlugin)).unwrap();
        assert!(manager.get_plugin("stub").is_some());
    }

    #[test]
    fn logging_plugin_is_reachable_through_the_facade() {
        let plugin = LoggingPlugin::default();
        assert_eq!(BasePlugin::name(&plugin), "logging_plugin");
    }
}
