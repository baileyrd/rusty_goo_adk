//! C0599: `evaluation.custom_metric_evaluator`, ported from
//! `google.adk.evaluation.custom_metric_evaluator`.
//!
//! **`_get_metric_function`, adaptation disclosed**: the source resolves
//! a custom metric function at runtime via `importlib.import_module` +
//! `getattr` on a dotted path (`"my_package.my_module.my_function"`).
//! Rust has no equivalent — there's no dynamic module loader, and even
//! if there were, a function pulled out of an arbitrary compiled binary
//! this way couldn't be called with a known signature. This port
//! replaces dynamic import with an explicit registration API
//! ([`register_custom_metric_function`]): the embedding application
//! registers each custom metric function it wants reachable by the same
//! dotted-path string an eval config would name, once, before running
//! evals — the same "class → registered constructor closure, keyed by a
//! string the source already treats as an identifier" adaptation already
//! used for `user_simulator::register_user_simulator`.
//!
//! **Sync only, disclosed narrowing**: the source's custom metric
//! functions may be sync or `async` (`inspect.isawaitable` branches at
//! call time). This port's [`CustomMetricFn`] is sync-only, matching the
//! already-established `Evaluator` trait itself being sync (see
//! `evaluator`'s module doc) — a custom function that needs to await
//! something has nowhere to do so through this trait yet, same gap as
//! any other `Evaluator` implementor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rusty_serde::value::Value;

use crate::eval_case::Invocation;
use crate::eval_metrics::EvalMetric;
use crate::evaluator::{EvaluationResult, Evaluator};

/// A registered custom metric function. See this module's doc for why
/// this replaces the source's `importlib`-based dynamic import.
pub type CustomMetricFn = Arc<
    dyn Fn(
            &EvalMetric,
            &[Invocation],
            Option<&[Invocation]>,
            Option<&Value>,
        ) -> Result<EvaluationResult, String>
        + Send
        + Sync,
>;

fn registry() -> &'static Mutex<HashMap<String, CustomMetricFn>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, CustomMetricFn>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The extension point custom metric functions register under — the
/// same dotted-path string an eval config's `code_config.name` (or a
/// metric's `custom_function_path`) already names.
pub fn register_custom_metric_function(
    custom_function_path: impl Into<String>,
    function: CustomMetricFn,
) {
    registry()
        .lock()
        .expect("custom metric function registry lock poisoned")
        .insert(custom_function_path.into(), function);
}

/// `custom_metric_evaluator._get_metric_function` — returns the
/// registered custom metric function for the given path.
fn get_metric_function(custom_function_path: &str) -> Result<CustomMetricFn, String> {
    registry()
        .lock()
        .expect("custom metric function registry lock poisoned")
        .get(custom_function_path)
        .cloned()
        .ok_or_else(|| {
            format!("Could not import custom metric function from {custom_function_path:?}")
        })
}

/// C0599: `custom_metric_evaluator._CustomMetricEvaluator` — evaluator
/// for custom metrics. Ported without the leading underscore: the
/// source's own `metric_evaluator_registry.py` imports and constructs it
/// directly across the module boundary despite the naming convention.
pub struct CustomMetricEvaluator {
    eval_metric: EvalMetric,
    metric_function: CustomMetricFn,
}

impl CustomMetricEvaluator {
    pub fn new(eval_metric: EvalMetric, custom_function_path: &str) -> Result<Self, String> {
        let metric_function = get_metric_function(custom_function_path)?;
        Ok(Self {
            eval_metric,
            metric_function,
        })
    }
}

impl Evaluator for CustomMetricEvaluator {
    /// Clears `threshold` on a copy of the configured metric before
    /// calling the custom function — matching the source's
    /// `eval_metric.model_copy(deep=True)` then `eval_metric.threshold =
    /// None`. The deprecated top-level `threshold` field is superseded
    /// by `criterion.threshold`; clearing it here stops a custom
    /// function from reading the stale value.
    fn evaluate_invocations(
        &self,
        actual_invocations: &[Invocation],
        expected_invocations: Option<&[Invocation]>,
        conversation_scenario: Option<&Value>,
    ) -> Result<EvaluationResult, String> {
        let mut eval_metric = self.eval_metric.clone();
        eval_metric.threshold = None;
        (self.metric_function)(
            &eval_metric,
            actual_invocations,
            expected_invocations,
            conversation_scenario,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::EvalStatus;

    fn eval_metric() -> EvalMetric {
        EvalMetric::new("my_custom_metric").with_threshold(0.75)
    }

    #[test]
    fn unregistered_path_is_an_error() {
        let result = CustomMetricEvaluator::new(eval_metric(), "no.such.module.function");
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .contains("Could not import custom metric function"));
    }

    #[test]
    fn evaluate_invocations_calls_the_registered_function_with_threshold_cleared() {
        register_custom_metric_function(
            "adk_eval_tests.custom_metric_evaluator.clears_threshold",
            Arc::new(|metric, _actual, _expected, _scenario| {
                assert_eq!(metric.threshold, None);
                Ok(EvaluationResult {
                    overall_score: Some(1.0),
                    overall_eval_status: EvalStatus::Passed,
                    per_invocation_results: vec![],
                    overall_rubric_scores: None,
                })
            }),
        );

        let evaluator = CustomMetricEvaluator::new(
            eval_metric(),
            "adk_eval_tests.custom_metric_evaluator.clears_threshold",
        )
        .unwrap();
        let result = evaluator.evaluate_invocations(&[], None, None).unwrap();
        assert_eq!(result.overall_score, Some(1.0));
    }

    #[test]
    fn evaluate_invocations_leaves_the_original_metrics_threshold_untouched() {
        register_custom_metric_function(
            "adk_eval_tests.custom_metric_evaluator.noop",
            Arc::new(|_metric, _actual, _expected, _scenario| Ok(EvaluationResult::default())),
        );

        let evaluator = CustomMetricEvaluator::new(
            eval_metric(),
            "adk_eval_tests.custom_metric_evaluator.noop",
        )
        .unwrap();
        evaluator.evaluate_invocations(&[], None, None).unwrap();
        assert_eq!(evaluator.eval_metric.threshold, Some(0.75));
    }

    #[test]
    fn propagates_the_custom_functions_error() {
        register_custom_metric_function(
            "adk_eval_tests.custom_metric_evaluator.always_fails",
            Arc::new(|_metric, _actual, _expected, _scenario| Err("boom".to_string())),
        );

        let evaluator = CustomMetricEvaluator::new(
            eval_metric(),
            "adk_eval_tests.custom_metric_evaluator.always_fails",
        )
        .unwrap();
        assert_eq!(
            evaluator.evaluate_invocations(&[], None, None),
            Err("boom".to_string())
        );
    }
}
