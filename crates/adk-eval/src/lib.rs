//! Evaluation framework core, ported from `google.adk.evaluation`
//! (Phase 11) — the first landing in this capability area.
//!
//! **Scope of this first batch**: the pure-computation core needed to
//! run [`trajectory_evaluator::TrajectoryEvaluator`] end-to-end with no
//! LLM calls and no cloud dependency: [`eval_case`] (`Invocation` and its
//! nested types, C0605), [`evaluator`] (the `Evaluator` trait, C0600 —
//! partial, see below), [`eval_metrics`] (`EvalMetric`/`EvalMetricResult`,
//! C0608; `BaseCriterion`/`ToolTrajectoryCriterion`, part of C0612), and
//! [`trajectory_evaluator`] (C0588, DONE in full).
//!
//! **C0600, partial**: the source's `evaluator.py` also defines
//! `LlmAsJudge[CriterionT]` — a generic harness that samples a judge LLM
//! several times per invocation under a parallelism-limiting semaphore,
//! tolerating (logging, not failing on) individual sample failures. That
//! needs a real LLM-invocation path this batch doesn't build on; only
//! the plain `Evaluator` trait/`PerInvocationResult`/`EvaluationResult`
//! half of C0600 is ported here.
//!
//! **C0612, partial**: only `BaseCriterion`/`ToolTrajectoryCriterion`
//! (plus its nested `MatchType`) are ported. `LlmAsAJudgeCriterion`/
//! `RubricsBasedCriterion`/`HallucinationsCriterion`/
//! `LlmBackedUserSimulatorCriterion`/`JudgeModelOptions` all exist only
//! to configure an LLM-judge metric, which isn't built this batch;
//! `Interval`/`MetricValueInfo`/`MetricInfo`/`MetricInfoProvider` are
//! metric-catalog metadata types with no consumer yet either.
//!
//! **C0606/C0607/C0610, not ported — opaque placeholders**:
//! `Invocation.rubrics`/`.app_details` (their real types,
//! `eval_rubrics.Rubric` and `app_details.AppDetails`, are each their own
//! still-`REQUIRED` manifest row) and `Evaluator::evaluate_invocations`'s
//! `conversation_scenario` parameter (`eval_case.ConversationScenario`,
//! C0606) stay opaque `Value` placeholders — the same "widen once a real
//! consumer needs the structure, not before" convention used throughout
//! this port. `TrajectoryEvaluator` itself never reads any of the three
//! (the source explicitly `del`s `conversation_scenario`, "not supported
//! for per-invocation evaluation"), so none of this batch's own logic is
//! narrowed by leaving them opaque.
//!
//! **`EvalStatus`, wire format disclosed**: the source's `EvalStatus` is
//! a plain (non-`str`) `Enum` with int values (`PASSED = 1`, ...); under
//! Pydantic v2's default enum serialization that means the wire form is
//! the bare integer, not a readable string. No cross-language consumer
//! of this wire format exists anywhere in this workspace yet (this is
//! the first `evaluation/` capability landed), so this port serializes
//! `EvalStatus` as its variant name (`"PASSED"`/`"FAILED"`/
//! `"NOT_EVALUATED"`, matching the source's own enum member *names*)
//! instead — a disclosed, purely cosmetic choice favoring readability
//! over exactly replicating a wire quirk nothing yet depends on.
//!
//! **`EvalMetric`'s private-attribute spoofing guard — a compile-time
//! strengthening, not a port**: the source's `_config_custom_function_path`
//! is a `PrivateAttr`, specifically so that a metric deserialized from an
//! inbound (potentially attacker-controlled) payload can never set it —
//! only code holding a real `EvalMetric` instance can. This port's
//! equivalent field is a private (non-`pub`) struct field with no
//! `Deserialize` support at all; a value parsed from JSON via this
//! port's derived `Deserialize` structurally cannot populate a private
//! field, so the same guarantee holds automatically rather than needing
//! `PrivateAttr`'s runtime enforcement.

pub mod eval_case;
pub mod eval_metrics;
pub mod evaluator;
pub mod trajectory_evaluator;
