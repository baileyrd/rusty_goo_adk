//! Capability C0636: `optimization.agent_optimizer`, ported from
//! `google.adk.optimization.agent_optimizer`.

use crate::llm_agent::LlmAgent;
use crate::optimization_data_types::{AgentWithScores, OptimizerResult, SamplingResult};
use crate::sampler::Sampler;
use crate::services::BoxFuture;

/// `optimization.agent_optimizer.AgentOptimizer` — base interface for
/// agent optimizers.
pub trait AgentOptimizer<R: SamplingResult, A: AgentWithScores>: Send + Sync {
    /// Runs the optimizer.
    ///
    /// `initial_agent` is the agent to be optimized; `sampler` is the
    /// interface used to get training/validation example UIDs, request
    /// agent evaluations, and get data useful for optimizing the agent.
    ///
    /// Returns the final result of the optimization process: the
    /// optimized agent instances with their scores on the validation
    /// examples, and any optimization metadata.
    fn optimize<'a>(
        &'a self,
        initial_agent: &'a LlmAgent,
        sampler: &'a dyn Sampler<R>,
    ) -> BoxFuture<'a, OptimizerResult<A>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_agent::ModelRef;
    use crate::optimization_data_types::BaseAgentWithScores;
    use crate::sampler::ExampleSet;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Default)]
    struct StubSamplingResult {
        scores: BTreeMap<String, f64>,
    }

    impl SamplingResult for StubSamplingResult {
        fn scores(&self) -> &BTreeMap<String, f64> {
            &self.scores
        }
    }

    struct StubSampler;

    impl Sampler<StubSamplingResult> for StubSampler {
        fn get_train_example_ids(&self) -> Vec<String> {
            vec!["train-1".to_string()]
        }
        fn get_validation_example_ids(&self) -> Vec<String> {
            vec!["val-1".to_string()]
        }
        fn sample_and_score<'a>(
            &'a self,
            _candidate: &'a LlmAgent,
            _example_set: ExampleSet,
            _batch: Option<&'a [String]>,
            _capture_full_eval_data: bool,
        ) -> crate::services::BoxFuture<'a, StubSamplingResult> {
            Box::pin(async move {
                StubSamplingResult {
                    scores: BTreeMap::from([("val-1".to_string(), 1.0)]),
                }
            })
        }
    }

    /// An optimizer that just re-wraps `initial_agent`, scored via one
    /// validation-set pass — enough to exercise the trait's wiring
    /// end-to-end without a real optimization strategy.
    struct PassthroughOptimizer;

    impl AgentOptimizer<StubSamplingResult, BaseAgentWithScores> for PassthroughOptimizer {
        fn optimize<'a>(
            &'a self,
            initial_agent: &'a LlmAgent,
            sampler: &'a dyn Sampler<StubSamplingResult>,
        ) -> crate::services::BoxFuture<'a, OptimizerResult<BaseAgentWithScores>> {
            Box::pin(async move {
                let validation_ids = sampler.get_validation_example_ids();
                let result = sampler
                    .sample_and_score(initial_agent, ExampleSet::Validation, None, false)
                    .await;
                let overall_score = if validation_ids.is_empty() {
                    None
                } else {
                    Some(result.scores().values().sum::<f64>() / validation_ids.len() as f64)
                };
                OptimizerResult {
                    optimized_agents: vec![BaseAgentWithScores {
                        optimized_agent: Arc::new(LlmAgent::new(initial_agent.model.clone())),
                        overall_score,
                    }],
                }
            })
        }
    }

    #[rusty_tokio::test]
    async fn optimize_returns_a_scored_agent_from_the_sampler() {
        let optimizer = PassthroughOptimizer;
        let sampler = StubSampler;
        let initial_agent = LlmAgent::new(ModelRef::Name("gemini-test".to_string()));

        let result = optimizer.optimize(&initial_agent, &sampler).await;

        assert_eq!(result.optimized_agents.len(), 1);
        assert_eq!(result.optimized_agents[0].overall_score, Some(1.0));
    }
}
