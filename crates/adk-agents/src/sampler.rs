//! Capability C0637 (partial): `optimization.sampler`, ported from
//! `google.adk.optimization.sampler`.
//!
//! **Default parameter values, adapted**: the source's `sample_and_score(
//! ..., example_set: _ExampleSet = VALIDATION_SET, batch: Optional[...]
//! = None, capture_full_eval_data: bool = False)` has Python-style
//! default argument values. Rust trait methods can't declare defaults —
//! every caller passes every argument explicitly.
//! `TRAIN_SET`/`VALIDATION_SET` (the source's `ClassVar` string
//! constants used as those defaults) aren't ported as trait constants
//! either — nothing needs them as anything other than [`ExampleSet`]
//! enum variants once the defaulting itself is gone.

use crate::llm_agent::LlmAgent;
use crate::optimization_data_types::SamplingResult;
use crate::services::BoxFuture;

/// `optimization.sampler._ExampleSet` — which batch of examples to draw
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleSet {
    Train,
    Validation,
}

/// `optimization.sampler.Sampler` — the interface an optimizer uses to
/// get training/validation example UIDs and request agent evaluations.
/// A developer implements this against their own evaluation service for
/// it to work with an [`crate::agent_optimizer::AgentOptimizer`].
pub trait Sampler<R: SamplingResult>: Send + Sync {
    /// The UIDs of examples to use for training the agent.
    fn get_train_example_ids(&self) -> Vec<String>;

    /// The UIDs of examples to use for validating the optimized agent.
    fn get_validation_example_ids(&self) -> Vec<String>;

    /// Evaluates `candidate` on the given batch of examples (or every
    /// example in `example_set` if `batch` is `None`). If
    /// `capture_full_eval_data` is `false`, it's enough to only
    /// calculate scores for each example; if `true`, this should also
    /// capture whatever else the optimizer needs (e.g. outputs,
    /// trajectories, tool calls).
    fn sample_and_score<'a>(
        &'a self,
        candidate: &'a LlmAgent,
        example_set: ExampleSet,
        batch: Option<&'a [String]>,
        capture_full_eval_data: bool,
    ) -> BoxFuture<'a, R>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_agent::ModelRef;
    use crate::optimization_data_types::BaseSamplingResult;
    use std::collections::BTreeMap;

    struct StubSampler;

    impl Sampler<BaseSamplingResult> for StubSampler {
        fn get_train_example_ids(&self) -> Vec<String> {
            vec!["train-1".to_string(), "train-2".to_string()]
        }

        fn get_validation_example_ids(&self) -> Vec<String> {
            vec!["val-1".to_string()]
        }

        fn sample_and_score<'a>(
            &'a self,
            _candidate: &'a LlmAgent,
            example_set: ExampleSet,
            batch: Option<&'a [String]>,
            _capture_full_eval_data: bool,
        ) -> BoxFuture<'a, BaseSamplingResult> {
            let ids = match batch {
                Some(ids) => ids.to_vec(),
                None => match example_set {
                    ExampleSet::Train => self.get_train_example_ids(),
                    ExampleSet::Validation => self.get_validation_example_ids(),
                },
            };
            Box::pin(async move {
                let scores = ids
                    .into_iter()
                    .map(|id| (id, 1.0))
                    .collect::<BTreeMap<_, _>>();
                BaseSamplingResult { scores }
            })
        }
    }

    #[rusty_tokio::test]
    async fn sample_and_score_uses_the_given_batch_when_provided() {
        use crate::optimization_data_types::SamplingResult;

        let sampler = StubSampler;
        let agent = LlmAgent::new(ModelRef::Name("gemini-test".to_string()));
        let batch = vec!["custom-1".to_string()];
        let result = sampler
            .sample_and_score(&agent, ExampleSet::Validation, Some(&batch), false)
            .await;
        assert_eq!(result.scores().len(), 1);
        assert!(result.scores().contains_key("custom-1"));
    }

    #[rusty_tokio::test]
    async fn sample_and_score_falls_back_to_the_example_sets_full_id_list() {
        use crate::optimization_data_types::SamplingResult;

        let sampler = StubSampler;
        let agent = LlmAgent::new(ModelRef::Name("gemini-test".to_string()));
        let result = sampler
            .sample_and_score(&agent, ExampleSet::Train, None, false)
            .await;
        assert_eq!(result.scores().len(), 2);
    }

    #[test]
    fn get_train_and_validation_example_ids_are_distinct() {
        let sampler = StubSampler;
        assert_eq!(sampler.get_train_example_ids().len(), 2);
        assert_eq!(sampler.get_validation_example_ids().len(), 1);
    }
}
