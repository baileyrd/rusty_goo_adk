//! C0487 (partial): `tools.environment_simulation.environment_simulation_engine`
//! — [`EnvironmentSimulationEngine::simulate`]'s injection-only path: a
//! `before_tool_callback` running per-tool injection checks (probability
//! roll against a seedable PRNG, `match_args` filtering, optional
//! latency, then an injected error or response).
//!
//! **Deferred, disclosed**: the source falls back to an LLM-synthesized
//! mock response (`_create_mock_strategy`/`ToolConnectionAnalyzer.analyze`/
//! `agent.canonical_tools`) when no injection config is hit and a
//! `mock_strategy_type` is configured. This port has no LLM-invocation
//! path wired to a mock-response synthesizer yet, so that branch is a
//! disclosed no-op — [`EnvironmentSimulationEngine::simulate`] returns
//! `None` (same as the source's own "no mock strategy configured"
//! no-op branch) and does not attempt tool-connection analysis. The
//! `tool_context`/`canonical_tools`/`ToolConnectionAnalyzer` machinery
//! this deferred branch would need stays unported (see
//! `tool_connection_map`'s own module doc for the C0488 split).
//!
//! **Not wired to a real dispatch mechanism, disclosed**: nothing in this
//! port's `before_tool_callback` type (`adk_agents::llm_agent::LlmCallback`)
//! accepts a `(tool, args)` pair the way the source's `before_tool_callback`
//! does — see `environment_simulation_factory`'s own module doc for the
//! same gap on the factory side. This engine is a complete, independently
//! testable unit; it just has no real caller yet.
//!
//! **PRNG algorithm mismatch, disclosed**: `adk_platform::random::Rng`
//! (xorshift128+/SplitMix64-seeded) matches Python's `random.random()`
//! *range* (`[0, 1)`) but not its Mersenne-Twister *algorithm* — a
//! `random_seed` reproduces the same roll deterministically within this
//! port, not the same roll Python's `random.Random(seed).random()` would
//! produce for the same seed. Same narrowing already disclosed for
//! `adk_platform::random`'s own module doc.

use std::collections::BTreeMap;
use std::sync::Mutex;

use adk_platform::random::Rng;
use rusty_serde::value::Value;

use crate::base_tool::BaseTool;
use crate::environment_simulation_config::{
    EnvironmentSimulationConfig, MockStrategy, ToolSimulationConfig,
};

/// Core engine to handle the simulation logic (injection-only path — see
/// the module doc for the deferred LLM mock-strategy fallback).
pub struct EnvironmentSimulationEngine {
    tool_sim_configs: BTreeMap<String, ToolSimulationConfig>,
    rng: Mutex<Rng>,
}

impl EnvironmentSimulationEngine {
    pub fn new(config: EnvironmentSimulationConfig) -> Self {
        let tool_sim_configs = config
            .tool_simulation_configs
            .into_iter()
            .map(|c| (c.tool_name.clone(), c))
            .collect();
        EnvironmentSimulationEngine {
            tool_sim_configs,
            rng: Mutex::new(Rng::from_entropy()),
        }
    }

    /// Simulates a tool call. Returns `None` if `tool` has no simulation
    /// config, no injection config was hit and no mock strategy is
    /// configured, or (disclosed narrowing) a mock strategy *is*
    /// configured — see the module doc.
    pub async fn simulate(
        &self,
        tool: &dyn BaseTool,
        args: &BTreeMap<String, Value>,
    ) -> Option<BTreeMap<String, Value>> {
        let tool_sim_config = self.tool_sim_configs.get(tool.name())?;

        for injection_config in &tool_sim_config.injection_configs {
            if let Some(match_args) = &injection_config.match_args {
                let matches = match_args.iter().all(|(k, v)| args.get(k) == Some(v));
                if !matches {
                    continue;
                }
            }

            let roll = {
                let mut rng = self
                    .rng
                    .lock()
                    .expect("environment simulation rng poisoned");
                if let Some(seed) = injection_config.random_seed {
                    *rng = Rng::from_seed(seed);
                }
                rng.next_f64()
            };

            if roll < injection_config.injection_probability {
                if injection_config.injected_latency_seconds > 0.0 {
                    rusty_tokio::time::sleep(std::time::Duration::from_secs_f64(
                        injection_config.injected_latency_seconds,
                    ))
                    .await;
                }
                if let Some(injected_error) = &injection_config.injected_error {
                    let mut response = BTreeMap::new();
                    response.insert(
                        "error_code".to_string(),
                        Value::Int(injected_error.injected_http_error_code),
                    );
                    response.insert(
                        "error_message".to_string(),
                        Value::String(injected_error.error_message.clone()),
                    );
                    return Some(response);
                }
                if let Some(injected_response) = &injection_config.injected_response {
                    return Some(injected_response.clone());
                }
            }
        }

        if tool_sim_config.mock_strategy_type == MockStrategy::Unspecified {
            eprintln!(
                "environment_simulation: tool '{}' did not hit any injection config and has no \
                 mock strategy configured. Returning no-op.",
                tool.name()
            );
        } else {
            eprintln!(
                "environment_simulation: tool '{}' has a mock strategy configured, but this \
                 port has no LLM-synthesized mock-response path yet (deferred, see \
                 environment_simulation_engine's module doc). Returning no-op.",
                tool.name()
            );
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment_simulation_config::{InjectedError, InjectionConfig};

    struct StubTool {
        name: &'static str,
    }

    impl BaseTool for StubTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            ""
        }
    }

    fn config_with(
        tool_name: &str,
        injection_configs: Vec<InjectionConfig>,
    ) -> EnvironmentSimulationConfig {
        EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: tool_name.to_string(),
                injection_configs,
                mock_strategy_type: MockStrategy::Unspecified,
            }],
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn returns_none_for_a_tool_with_no_simulation_config() {
        let engine = EnvironmentSimulationEngine::new(config_with("configured", vec![]));
        let result = engine
            .simulate(
                &StubTool {
                    name: "unconfigured",
                },
                &BTreeMap::new(),
            )
            .await;
        assert_eq!(result, None);
    }

    #[rusty_tokio::test]
    async fn injects_an_error_when_probability_is_1_and_seeded() {
        let injection = InjectionConfig {
            injection_probability: 1.0,
            random_seed: Some(42),
            injected_error: Some(InjectedError {
                injected_http_error_code: 503,
                error_message: "unavailable".to_string(),
            }),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let result = engine
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        let result = result.expect("expected an injected error");
        assert_eq!(result.get("error_code"), Some(&Value::Int(503)));
        assert_eq!(
            result.get("error_message"),
            Some(&Value::String("unavailable".to_string()))
        );
    }

    #[rusty_tokio::test]
    async fn injects_a_response_when_probability_is_1() {
        let mut response = BTreeMap::new();
        response.insert("ok".to_string(), Value::Bool(true));
        let injection = InjectionConfig {
            injection_probability: 1.0,
            random_seed: Some(7),
            injected_response: Some(response.clone()),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let result = engine
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        assert_eq!(result, Some(response));
    }

    #[rusty_tokio::test]
    async fn never_injects_when_probability_is_0() {
        let injection = InjectionConfig {
            injection_probability: 0.0,
            random_seed: Some(1),
            injected_error: Some(InjectedError {
                injected_http_error_code: 500,
                error_message: "boom".to_string(),
            }),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let result = engine
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        assert_eq!(result, None);
    }

    #[rusty_tokio::test]
    async fn skips_an_injection_config_whose_match_args_do_not_match() {
        let injection = InjectionConfig {
            injection_probability: 1.0,
            random_seed: Some(1),
            match_args: Some(BTreeMap::from([(
                "user_id".to_string(),
                Value::String("alice".to_string()),
            )])),
            injected_error: Some(InjectedError {
                injected_http_error_code: 500,
                error_message: "boom".to_string(),
            }),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let mut args = BTreeMap::new();
        args.insert("user_id".to_string(), Value::String("bob".to_string()));
        let result = engine.simulate(&StubTool { name: "t" }, &args).await;
        assert_eq!(result, None);
    }

    #[rusty_tokio::test]
    async fn applies_an_injection_config_whose_match_args_match() {
        let injection = InjectionConfig {
            injection_probability: 1.0,
            random_seed: Some(1),
            match_args: Some(BTreeMap::from([(
                "user_id".to_string(),
                Value::String("alice".to_string()),
            )])),
            injected_response: Some(BTreeMap::new()),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let mut args = BTreeMap::new();
        args.insert("user_id".to_string(), Value::String("alice".to_string()));
        args.insert("extra".to_string(), Value::Bool(true));
        let result = engine.simulate(&StubTool { name: "t" }, &args).await;
        assert_eq!(result, Some(BTreeMap::new()));
    }

    #[rusty_tokio::test]
    async fn falls_through_to_no_op_when_no_injection_hits_and_no_mock_strategy() {
        let injection = InjectionConfig {
            injection_probability: 0.0,
            random_seed: Some(1),
            injected_response: Some(BTreeMap::new()),
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let result = engine
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        assert_eq!(result, None);
    }

    #[rusty_tokio::test]
    async fn deferred_mock_strategy_returns_none_rather_than_synthesizing() {
        let config = EnvironmentSimulationConfig {
            tool_simulation_configs: vec![ToolSimulationConfig {
                tool_name: "t".to_string(),
                injection_configs: vec![],
                mock_strategy_type: MockStrategy::ToolSpec,
            }],
            ..Default::default()
        };
        let engine = EnvironmentSimulationEngine::new(config);
        let result = engine
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        assert_eq!(result, None);
    }

    #[rusty_tokio::test]
    async fn a_reseeded_run_reproduces_the_same_roll_deterministically() {
        let injection = InjectionConfig {
            injection_probability: 0.5,
            random_seed: Some(123),
            injected_response: Some(BTreeMap::new()),
            ..Default::default()
        };
        let engine_a = EnvironmentSimulationEngine::new(config_with("t", vec![injection.clone()]));
        let engine_b = EnvironmentSimulationEngine::new(config_with("t", vec![injection]));
        let result_a = engine_a
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        let result_b = engine_b
            .simulate(&StubTool { name: "t" }, &BTreeMap::new())
            .await;
        assert_eq!(result_a, result_b);
    }
}
