//! Capability C0637 (data types): `optimization.data_types`, ported from
//! `google.adk.optimization.data_types`.
//!
//! **Placement, disclosed**: `optimization/` is a distinct top-level
//! source package from `agents/`, but its every type reaches directly
//! into `LlmAgent` (`Agent = LlmAgent`, this crate's own type) with no
//! other new dependency needed — the same "no dedicated crate for a
//! handful of types that only need what their one real consumer crate
//! already has" placement `app_configs.rs` already established for
//! `apps/_configs.py`.
//!
//! **Generic bound → trait, adaptation**: the source uses pydantic
//! generics (`SamplingResultT = TypeVar("SamplingResultT",
//! bound=SamplingResult)`, `AgentWithScoresT = TypeVar(...,
//! bound=AgentWithScores)`) so a concrete optimizer/sampler pair can
//! share a richer result shape via subclassing (e.g.
//! `UnstructuredSamplingResult`). This port models each bound as a trait
//! ([`SamplingResult`]/[`AgentWithScores`]) that
//! [`crate::sampler::Sampler`]/[`crate::agent_optimizer::AgentOptimizer`]'s
//! generic parameters are bounded by, rather than a concrete base struct
//! callers subclass — Rust has no struct inheritance, so a subclass's
//! extra fields (`UnstructuredSamplingResult.data`) are declared
//! directly on its own struct instead (the same "flatten inherited
//! fields into the subclass struct" pattern already established for
//! `ExtendedOAuth2`, `auth_schemes.rs`).
//!
//! **`Agent`/`LlmAgent`, held via `Arc`**: `LlmAgent` (this port's
//! `Agent` alias target) derives neither `Clone` nor `Debug` — sharing
//! one across `AgentWithScores`/optimizer call sites needs a handle, so
//! [`BaseAgentWithScores::optimized_agent`] is `Arc<LlmAgent>` rather
//! than an owned value, the same "cheap shareable handle" role
//! `BaseAgent`'s own internal `Arc<BaseAgentData>` wrapper already plays.

use std::collections::BTreeMap;
use std::sync::Arc;

use rusty_serde::value::Value;

use crate::llm_agent::LlmAgent;

/// `optimization.data_types.SamplingResult` — see the module doc for why
/// this is a trait, not a base struct.
pub trait SamplingResult: Send + Sync {
    /// A map from example UID to the agent's overall score on that
    /// example (higher is better).
    fn scores(&self) -> &BTreeMap<String, f64>;
}

/// `optimization.data_types.SamplingResult` (the concrete/default case —
/// just the required `scores` field).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaseSamplingResult {
    pub scores: BTreeMap<String, f64>,
}

impl SamplingResult for BaseSamplingResult {
    fn scores(&self) -> &BTreeMap<String, f64> {
        &self.scores
    }
}

/// `optimization.data_types.UnstructuredSamplingResult` — evaluation
/// result providing per-example unstructured evaluation data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnstructuredSamplingResult {
    pub scores: BTreeMap<String, f64>,
    /// A map from example UID to JSON-serializable evaluation data
    /// useful for agent optimization (inputs/trajectories/metrics
    /// recommended). Must be provided if requested by the optimizer.
    pub data: Option<BTreeMap<String, BTreeMap<String, Value>>>,
}

impl SamplingResult for UnstructuredSamplingResult {
    fn scores(&self) -> &BTreeMap<String, f64> {
        &self.scores
    }
}

/// `optimization.data_types.AgentWithScores` — see the module doc.
pub trait AgentWithScores: Send + Sync {
    fn optimized_agent(&self) -> &Arc<LlmAgent>;
    fn overall_score(&self) -> Option<f64>;
}

/// `optimization.data_types.AgentWithScores` (the concrete/default
/// case). Optimizers may use `overall_score` and can return custom
/// metrics by implementing [`AgentWithScores`] on their own struct
/// instead.
#[derive(Clone)]
pub struct BaseAgentWithScores {
    pub optimized_agent: Arc<LlmAgent>,
    pub overall_score: Option<f64>,
}

impl std::fmt::Debug for BaseAgentWithScores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `LlmAgent` (this port's own type) doesn't implement `Debug`
        // (see `llm_agent.rs`), so it's summarized as an opaque handle
        // here instead.
        f.debug_struct("BaseAgentWithScores")
            .field("optimized_agent", &"<LlmAgent>")
            .field("overall_score", &self.overall_score)
            .finish()
    }
}

impl AgentWithScores for BaseAgentWithScores {
    fn optimized_agent(&self) -> &Arc<LlmAgent> {
        &self.optimized_agent
    }
    fn overall_score(&self) -> Option<f64> {
        self.overall_score
    }
}

/// `optimization.data_types.OptimizerResult` — a list of optimized
/// agents which cannot be considered strictly better than one another
/// (a Pareto front: <https://en.wikipedia.org/wiki/Pareto_front>), along
/// with scores.
#[derive(Debug, Clone)]
pub struct OptimizerResult<A: AgentWithScores> {
    pub optimized_agents: Vec<A>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_sampling_result_reports_its_scores() {
        let result = BaseSamplingResult {
            scores: BTreeMap::from([("ex-1".to_string(), 0.5)]),
        };
        assert_eq!(result.scores().get("ex-1"), Some(&0.5));
    }

    #[test]
    fn unstructured_sampling_result_reports_its_scores_and_data() {
        let result = UnstructuredSamplingResult {
            scores: BTreeMap::from([("ex-1".to_string(), 1.0)]),
            data: Some(BTreeMap::from([(
                "ex-1".to_string(),
                BTreeMap::from([("trajectory".to_string(), Value::String("ok".to_string()))]),
            )])),
        };
        assert_eq!(result.scores().get("ex-1"), Some(&1.0));
        assert_eq!(
            result
                .data
                .as_ref()
                .and_then(|d| d.get("ex-1"))
                .and_then(|d| d.get("trajectory")),
            Some(&Value::String("ok".to_string()))
        );
    }

    #[test]
    fn base_agent_with_scores_reports_its_agent_and_score() {
        use crate::llm_agent::ModelRef;

        let agent = Arc::new(LlmAgent::new(ModelRef::Name("gemini-test".to_string())));
        let scored = BaseAgentWithScores {
            optimized_agent: agent.clone(),
            overall_score: Some(0.75),
        };
        assert!(Arc::ptr_eq(scored.optimized_agent(), &agent));
        assert_eq!(scored.overall_score(), Some(0.75));
    }

    #[test]
    fn optimizer_result_holds_a_pareto_front_of_agents() {
        use crate::llm_agent::ModelRef;

        let a = BaseAgentWithScores {
            optimized_agent: Arc::new(LlmAgent::new(ModelRef::Name("a".to_string()))),
            overall_score: Some(0.9),
        };
        let b = BaseAgentWithScores {
            optimized_agent: Arc::new(LlmAgent::new(ModelRef::Name("b".to_string()))),
            overall_score: Some(0.8),
        };
        let result = OptimizerResult {
            optimized_agents: vec![a, b],
        };
        assert_eq!(result.optimized_agents.len(), 2);
    }
}
