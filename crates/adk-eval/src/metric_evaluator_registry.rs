//! C0603 (partial): `evaluation.metric_evaluator_registry`, ported from
//! `google.adk.evaluation.metric_evaluator_registry`.
//!
//! **Partial, same shape as C0600's own split**: `_register_standard_metrics`
//! seeds 13 evaluators in the source; only [`crate::trajectory_evaluator::TrajectoryEvaluator`]
//! is actually built AND registrable in this port so far, so
//! [`MetricEvaluatorRegistry::new`] registers just that one. Four more
//! (`FinalResponseMatchV2Evaluator`/`RubricBasedFinalResponseQualityV1Evaluator`/
//! `RubricBasedToolUseV1Evaluator`/`RubricBasedMultiTurnTrajectoryEvaluator`,
//! C0592/C0593/C0595/C0598) are now built too, once C0600's `LlmAsJudge`
//! harness landed — but **cannot** register here regardless: they're
//! inherently async (the harness awaits a judge-model call), while this
//! registry's [`EvaluatorFactory`] constructs a sync `Box<dyn Evaluator
//! + Send + Sync>` and `Evaluator::evaluate_invocations` is deliberately
//! sync (see `evaluator.rs`'s own doc) — a structural mismatch, not a
//! temporary gap this registry will close once more evaluators exist. The
//! remaining 8 (`ResponseEvaluator`/`SafetyEvaluatorV1`/the Vertex-delegated
//! multi-turn metrics/`HallucinationsV1`/`PerTurnUserSimulatorQualityV1`)
//! stay `REQUIRED` under their own rows (C0591/C0594/C0596/C0597) — GCP-blocked,
//! not on anything this crate builds.
//!
//! **`type[Evaluator]` → tagged factory closure, adaptation disclosed**:
//! the source stores the concrete `Evaluator` *class* per metric name and
//! later does `issubclass(evaluator_type, _CustomMetricEvaluator)` to
//! decide how to construct one (positional `eval_metric` vs. keyword
//! `eval_metric`+`custom_function_path`). Rust has no runtime class
//! objects to store or `issubclass` against. This port tags each
//! registry entry with which construction it needs at the point of
//! registration instead ([`RegisteredEvaluator::Factory`] for anything
//! [`register_evaluator`](MetricEvaluatorRegistry::register_evaluator)
//! adds, [`RegisteredEvaluator::Custom`] for anything
//! [`register_custom_metrics_from_config`] adds) — the same information
//! the source's `issubclass` check recovers at lookup time, just decided
//! once at registration instead of every lookup.
//!
//! **`DEFAULT_METRIC_EVALUATOR_REGISTRY`, adaptation disclosed**: the
//! source is a mutable module-level singleton that
//! `register_custom_metrics_from_config` defaults to and mutates in
//! place when no registry is passed explicitly. This port exposes the
//! equivalent as [`default_registry`] (a lazily-initialized,
//! mutex-guarded static — the same pattern already used for
//! `user_simulator`'s config→simulator registry) but drops the
//! "optional parameter defaulting to it" convenience:
//! [`register_custom_metrics_from_config`] always takes an explicit
//! `&mut MetricEvaluatorRegistry`. A caller of this port that wants the
//! shared default locks [`default_registry`] itself and passes the
//! guard's `&mut` — returning a borrow that outlives a `MutexGuard`
//! isn't expressible the way Python's shared-object-identity default
//! is.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use adk_errors::not_found::NotFoundError;

use crate::custom_metric_evaluator::CustomMetricEvaluator;
use crate::eval_config::EvalConfig;
use crate::eval_metrics::{EvalMetric, Interval, MetricInfo, MetricInfoProvider, MetricValueInfo};
use crate::evaluator::Evaluator;
use crate::metric_info_providers::TrajectoryEvaluatorMetricInfoProvider;
use crate::trajectory_evaluator::TrajectoryEvaluator;

/// Constructs an `Evaluator` for one metric configuration. See this
/// module's doc for why this replaces the source's stored `type[Evaluator]`.
pub type EvaluatorFactory =
    Arc<dyn Fn(&EvalMetric) -> Result<Box<dyn Evaluator + Send + Sync>, String> + Send + Sync>;

#[derive(Clone)]
enum RegisteredEvaluator {
    /// Constructed via `factory(eval_metric)`.
    Factory(EvaluatorFactory),
    /// Constructed via `CustomMetricEvaluator::new`, using a
    /// `custom_function_path` looked up separately.
    Custom,
}

/// C0603: `metric_evaluator_registry.MetricEvaluatorRegistry` — a
/// registry for metric evaluators.
#[derive(Clone)]
pub struct MetricEvaluatorRegistry {
    registry: HashMap<String, (RegisteredEvaluator, MetricInfo)>,
    custom_function_paths: HashMap<String, String>,
}

impl MetricEvaluatorRegistry {
    /// Seeds the standard metrics this port has built so far. See this
    /// module's doc for which ones that is.
    pub fn new() -> Self {
        let mut registry = Self {
            registry: HashMap::new(),
            custom_function_paths: HashMap::new(),
        };
        register_standard_metrics(&mut registry);
        registry
    }

    /// `MetricEvaluatorRegistry.get_evaluator` — returns a new `Evaluator`
    /// instance for the given metric.
    pub fn get_evaluator(
        &self,
        eval_metric: &EvalMetric,
    ) -> Result<Box<dyn Evaluator + Send + Sync>, NotFoundError> {
        let (evaluator, _metric_info) =
            self.registry.get(&eval_metric.metric_name).ok_or_else(|| {
                NotFoundError::new(format!(
                    "{} not found in registry.",
                    eval_metric.metric_name
                ))
            })?;
        match evaluator {
            RegisteredEvaluator::Factory(factory) => {
                factory(eval_metric).map_err(NotFoundError::new)
            }
            RegisteredEvaluator::Custom => {
                let custom_function_path =
                    self.custom_function_path(eval_metric).ok_or_else(|| {
                        NotFoundError::new(format!(
                            "No custom function registered for {}.",
                            eval_metric.metric_name
                        ))
                    })?;
                let evaluator =
                    CustomMetricEvaluator::new(eval_metric.clone(), &custom_function_path)
                        .map_err(NotFoundError::new)?;
                Ok(Box::new(evaluator))
            }
        }
    }

    /// `MetricEvaluatorRegistry._custom_function_path` — returns the
    /// module path to import for a custom metric, if known. The
    /// incoming metric's own `custom_function_path` field is not
    /// consulted, as it can be set by whoever built the request; only
    /// paths this registry itself recorded (via [`Self::register`] or
    /// the metric's private `_config_custom_function_path`, set only by
    /// code holding a real `EvalMetric`) are trusted.
    fn custom_function_path(&self, eval_metric: &EvalMetric) -> Option<String> {
        if let Some(path) = self.custom_function_paths.get(&eval_metric.metric_name) {
            return Some(path.clone());
        }
        eval_metric
            .config_custom_function_path()
            .map(str::to_string)
    }

    /// `MetricEvaluatorRegistry.register_evaluator` — registers an
    /// evaluator given the metric info. Updates an existing mapping, if
    /// one is already registered for the metric name.
    pub fn register_evaluator(&mut self, metric_info: MetricInfo, factory: EvaluatorFactory) {
        self.register(metric_info, RegisteredEvaluator::Factory(factory), None);
    }

    /// `MetricEvaluatorRegistry._register` — registers an evaluator
    /// along with the function path it may need. A path already
    /// recorded for the metric is kept when this registration doesn't
    /// carry one, so re-registering an evaluator doesn't drop it.
    fn register(
        &mut self,
        metric_info: MetricInfo,
        evaluator: RegisteredEvaluator,
        custom_function_path: Option<String>,
    ) {
        let metric_name = metric_info.metric_name.clone();
        self.registry
            .insert(metric_name.clone(), (evaluator, metric_info));
        if let Some(path) = custom_function_path {
            self.custom_function_paths.insert(metric_name, path);
        }
    }

    /// `MetricEvaluatorRegistry.get_registered_metrics` — returns the
    /// `MetricInfo` for every metric registered so far.
    pub fn get_registered_metrics(&self) -> Vec<MetricInfo> {
        self.registry
            .values()
            .map(|(_, metric_info)| metric_info.clone())
            .collect()
    }

    /// `MetricEvaluatorRegistry.fork` — returns an isolated copy of this
    /// registry. Registrations made afterwards on either copy are
    /// invisible to the other.
    pub fn fork(&self) -> Self {
        self.clone()
    }
}

impl Default for MetricEvaluatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn register_standard_metrics(registry: &mut MetricEvaluatorRegistry) {
    registry.register_evaluator(
        TrajectoryEvaluatorMetricInfoProvider
            .get_metric_info()
            .expect("TrajectoryEvaluatorMetricInfoProvider::get_metric_info never fails"),
        Arc::new(|eval_metric| {
            TrajectoryEvaluator::new(None, Some(eval_metric))
                .map(|evaluator| Box::new(evaluator) as Box<dyn Evaluator + Send + Sync>)
        }),
    );
}

/// `metric_evaluator_registry.DEFAULT_METRIC_EVALUATOR_REGISTRY`. See
/// this module's doc for the adaptation from a mutable module-level
/// singleton to a lazily-initialized, mutex-guarded static.
pub fn default_registry() -> &'static Mutex<MetricEvaluatorRegistry> {
    static REGISTRY: OnceLock<Mutex<MetricEvaluatorRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(MetricEvaluatorRegistry::new()))
}

fn default_metric_info(
    metric_name: impl Into<String>,
    description: impl Into<String>,
) -> MetricInfo {
    MetricInfo {
        metric_name: metric_name.into(),
        description: Some(description.into()),
        metric_value_info: MetricValueInfo {
            interval: Some(Interval::closed(0.0, 1.0)),
        },
    }
}

/// C0603: `metric_evaluator_registry.register_custom_metrics_from_config`
/// — registers the custom metrics declared in `eval_config` into
/// `metric_evaluator_registry`. An entry without a `metric_info` gets a
/// default one with a `[0.0, 1.0]` value interval.
pub fn register_custom_metrics_from_config(
    eval_config: &EvalConfig,
    metric_evaluator_registry: &mut MetricEvaluatorRegistry,
) {
    let Some(custom_metrics) = &eval_config.custom_metrics else {
        return;
    };
    for (metric_name, config) in custom_metrics {
        let metric_info = match &config.metric_info {
            Some(metric_info) => {
                let mut metric_info = metric_info.clone();
                metric_info.metric_name = metric_name.clone();
                metric_info
            }
            None => default_metric_info(metric_name.clone(), config.description.clone()),
        };
        metric_evaluator_registry.register(
            metric_info,
            RegisteredEvaluator::Custom,
            Some(config.code_config.name.clone()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_metric_evaluator::register_custom_metric_function;
    use crate::eval_config::CodeConfig;
    use crate::evaluator::EvaluationResult;

    #[test]
    fn new_registers_the_trajectory_evaluator() {
        let registry = MetricEvaluatorRegistry::new();
        let metric = EvalMetric::new("tool_trajectory_avg_score").with_threshold(1.0);
        assert!(registry.get_evaluator(&metric).is_ok());
        assert_eq!(registry.get_registered_metrics().len(), 1);
    }

    #[test]
    fn get_evaluator_errors_for_an_unregistered_metric() {
        let registry = MetricEvaluatorRegistry::new();
        let metric = EvalMetric::new("no_such_metric").with_threshold(1.0);
        assert!(registry.get_evaluator(&metric).is_err());
    }

    #[test]
    fn register_evaluator_updates_an_existing_mapping() {
        let mut registry = MetricEvaluatorRegistry::new();
        let metric_info = TrajectoryEvaluatorMetricInfoProvider
            .get_metric_info()
            .unwrap();
        registry.register_evaluator(
            metric_info,
            Arc::new(|_eval_metric| Err("replaced".to_string())),
        );
        let metric = EvalMetric::new("tool_trajectory_avg_score").with_threshold(1.0);
        let err = registry.get_evaluator(&metric).err().unwrap();
        assert_eq!(err.message, "replaced");
    }

    #[test]
    fn fork_is_isolated_from_the_original() {
        let mut original = MetricEvaluatorRegistry::new();
        let forked = original.fork();

        let metric_info = MetricInfo {
            metric_name: "only_on_original".to_string(),
            description: None,
            metric_value_info: MetricValueInfo { interval: None },
        };
        original.register_evaluator(metric_info, Arc::new(|_| Err("n/a".to_string())));

        assert_eq!(original.get_registered_metrics().len(), 2);
        assert_eq!(forked.get_registered_metrics().len(), 1);
    }

    #[test]
    fn register_custom_metrics_from_config_uses_a_default_metric_info_when_absent() {
        register_custom_metric_function(
            "adk_eval_tests.metric_evaluator_registry.my_custom_metric",
            Arc::new(|_metric, _actual, _expected, _scenario| Ok(EvaluationResult::default())),
        );

        let mut eval_config = EvalConfig::default();
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert(
            "my_custom_metric".to_string(),
            crate::eval_config::CustomMetricConfig {
                code_config: CodeConfig {
                    name: "adk_eval_tests.metric_evaluator_registry.my_custom_metric".to_string(),
                },
                metric_info: None,
                description: "A custom metric.".to_string(),
            },
        );
        eval_config.custom_metrics = Some(custom_metrics);

        let mut registry = MetricEvaluatorRegistry::new();
        register_custom_metrics_from_config(&eval_config, &mut registry);

        let metric = EvalMetric::new("my_custom_metric").with_threshold(0.5);
        assert!(registry.get_evaluator(&metric).is_ok());

        let registered = registry
            .get_registered_metrics()
            .into_iter()
            .find(|info| info.metric_name == "my_custom_metric")
            .unwrap();
        assert_eq!(registered.description, Some("A custom metric.".to_string()));
        assert_eq!(
            registered.metric_value_info.interval,
            Some(Interval::closed(0.0, 1.0))
        );
    }

    #[test]
    fn get_evaluator_errors_when_no_custom_function_path_is_known() {
        let mut registry = MetricEvaluatorRegistry::new();
        registry.register(
            MetricInfo {
                metric_name: "orphaned_custom_metric".to_string(),
                description: None,
                metric_value_info: MetricValueInfo { interval: None },
            },
            RegisteredEvaluator::Custom,
            None,
        );
        let metric = EvalMetric::new("orphaned_custom_metric").with_threshold(0.5);
        let err = registry.get_evaluator(&metric).err().unwrap();
        assert!(err.message.contains("No custom function registered"));
    }

    #[test]
    fn default_registry_is_shared_across_calls() {
        {
            let mut guard = default_registry().lock().unwrap();
            let metric_info = MetricInfo {
                metric_name: "shared_default_registry_marker".to_string(),
                description: None,
                metric_value_info: MetricValueInfo { interval: None },
            };
            guard.register_evaluator(metric_info, Arc::new(|_| Err("registered".to_string())));
        }
        // A fresh lock acquisition sees the registration made under the
        // previous one -- proving `default_registry()` returns the same
        // static, not a fresh instance each call.
        let guard = default_registry().lock().unwrap();
        let metric = EvalMetric::new("shared_default_registry_marker").with_threshold(0.5);
        assert_eq!(
            guard.get_evaluator(&metric).err().unwrap().message,
            "registered"
        );
    }
}
