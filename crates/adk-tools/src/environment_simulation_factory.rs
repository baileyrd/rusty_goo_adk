//! C0488 (partial): `tools.environment_simulation.environment_simulation_factory`
//! — [`EnvironmentSimulationFactory::create_callback`], a factory
//! producing a `before_tool_callback`-shaped closure around one
//! [`EnvironmentSimulationEngine`].
//!
//! **Deferred, disclosed**: `create_plugin`/`EnvironmentSimulationPlugin`
//! has no port — it needs `BasePlugin`'s `before_tool_callback` hook,
//! which this port's `BasePlugin` trait doesn't expose yet (same
//! deferral this manifest's C0356 row already established for
//! plugin-level tool hooks generally).
//!
//! **Not wired to a real dispatch mechanism, disclosed**: `create_callback`
//! still produces a real, independently callable closure with the
//! source's exact shape (`Fn(tool, args) -> Future<Output = Option<dict>>`,
//! `tool_context` dropped — see `environment_simulation_engine`'s own
//! module doc for why the injection-only `simulate()` never reads it), but
//! nothing in this port's `before_tool_callback` dispatch
//! (`adk_agents::llm_agent::LlmAgent::before_tool_callback`, typed
//! `Vec<LlmCallback>` where `LlmCallback = Fn(&mut Context) -> Option<Value>`)
//! can accept a callback of this shape — that type has no `tool`/`args`
//! parameters to give it. Wiring this in is its own follow-up batch, once
//! `before_tool_callback`'s signature (or a new tool-scoped callback type)
//! widens to carry them; that's a breaking change to already-shipped
//! public surface, out of scope here.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rusty_serde::value::Value;

use crate::base_tool::BaseTool;
use crate::environment_simulation_config::EnvironmentSimulationConfig;
use crate::environment_simulation_engine::EnvironmentSimulationEngine;

/// The source's `Callable[[BaseTool, Dict[str, Any], Any],
/// Awaitable[Optional[Dict[str, Any]]]]` — `tool_context` dropped, see the
/// module doc.
pub type SimulationCallback = Arc<
    dyn Fn(
            Arc<dyn BaseTool>,
            BTreeMap<String, Value>,
        ) -> Pin<Box<dyn Future<Output = Option<BTreeMap<String, Value>>> + Send>>
        + Send
        + Sync,
>;

/// Factory for creating `EnvironmentSimulation` instances.
pub struct EnvironmentSimulationFactory;

impl EnvironmentSimulationFactory {
    /// Creates a callback closure for `EnvironmentSimulation` — usable as
    /// a `before_tool_callback`/`after_tool_callback` once this port's
    /// tool-callback dispatch can carry `(tool, args)` (see the module
    /// doc).
    pub fn create_callback(config: EnvironmentSimulationConfig) -> SimulationCallback {
        let engine = Arc::new(EnvironmentSimulationEngine::new(config));
        Arc::new(
            move |tool: Arc<dyn BaseTool>, args: BTreeMap<String, Value>| {
                let engine = engine.clone();
                Box::pin(async move { engine.simulate(tool.as_ref(), &args).await })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment_simulation_config::{
        InjectedError, InjectionConfig, MockStrategy, ToolSimulationConfig,
    };

    struct StubTool;

    impl BaseTool for StubTool {
        fn name(&self) -> &str {
            "t"
        }
        fn description(&self) -> &str {
            ""
        }
    }

    #[rusty_tokio::test]
    async fn the_created_callback_delegates_to_the_engines_simulate() {
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: "t".to_string(),
                injection_configs: vec![InjectionConfig {
                    injection_probability: 1.0,
                    random_seed: Some(1),
                    injected_error: Some(InjectedError {
                        injected_http_error_code: 429,
                        error_message: "rate limited".to_string(),
                    }),
                    ..Default::default()
                }],
                mock_strategy_type: MockStrategy::Unspecified,
            }],
            ..Default::default()
        };
        let callback = EnvironmentSimulationFactory::create_callback(config);
        let tool: Arc<dyn BaseTool> = Arc::new(StubTool);
        let result = callback(tool, BTreeMap::new())
            .await
            .expect("expected an injected error");
        assert_eq!(result.get("error_code"), Some(&Value::Int(429)));
    }

    #[rusty_tokio::test]
    async fn the_created_callback_returns_none_for_an_unconfigured_tool() {
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: "other".to_string(),
                injection_configs: vec![InjectionConfig {
                    injected_response: Some(BTreeMap::new()),
                    ..Default::default()
                }],
                mock_strategy_type: MockStrategy::Unspecified,
            }],
            ..Default::default()
        };
        let callback = EnvironmentSimulationFactory::create_callback(config);
        let tool: Arc<dyn BaseTool> = Arc::new(StubTool);
        let result = callback(tool, BTreeMap::new()).await;
        assert_eq!(result, None);
    }
}
