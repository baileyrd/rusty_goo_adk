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
//! **Second batch**: [`eval_rubrics`] (`Rubric`/`RubricContent`/
//! `RubricScore`, C0607), [`app_details`] (`AgentDetails`/`AppDetails`,
//! C0610), [`conversation_scenarios`] (`ConversationScenario`/
//! `ConversationGenerationConfig`, C0606), [`eval_set`] (`EvalSet`, the
//! rest of C0607), [`eval_result`] (`EvalCaseResult`/`EvalSetResult`,
//! C0609), and [`constants`] (misc eval constants, C0635) — plus
//! [`eval_case`] grows `SessionInput`/`SessionState`/`EvalCase` (closing
//! out C0606) and `Invocation.rubrics`/`.app_details` widen from opaque
//! `Value` to the now-real [`eval_rubrics::Rubric`]/
//! [`app_details::AppDetails`] types, closing that first batch's
//! disclosed gap.
//!
//! **Still opaque, disclosed**: [`conversation_scenarios::ConversationScenario::user_persona`]
//! (real type + registry resolution belongs to the persona system, its
//! own still-`REQUIRED` row C0632) and [`eval_result::EvalCaseResult::session_details`]
//! (real type `adk_agents::session::Session` already exists, but pulling
//! `adk-agents` into this crate's dependency graph for one unread
//! passthrough field would invert `adk-eval`'s deliberate bottom-of-the-
//! graph position — see that module's own doc). `Evaluator::evaluate_invocations`'s
//! `conversation_scenario` parameter also stays opaque `Value` —
//! `TrajectoryEvaluator`/`RougeEvaluator` (the only `Evaluator`s built so
//! far) never read any of these, matching the source's own `del
//! conversation_scenario, "not supported for per-invocation evaluation"`.
//!
//! **Third batch — local persistence**: [`path_validation`] (C0614),
//! [`eval_sets_manager`] (the `EvalSetsManager` trait + shared
//! [`eval_sets_manager::EvalManagerError`], part of C0613),
//! [`eval_sets_manager_utils`]/[`eval_set_results_manager_utils`]
//! (support functions shared across implementors), and two concrete
//! implementors each: [`in_memory_eval_sets_manager`]/
//! [`local_eval_sets_manager`] (C0613), [`eval_set_results_manager`]
//! (the trait, C0615) with [`local_eval_set_results_manager`] (C0615).
//! `GcsEvalSetsManager`/`GcsEvalSetResultsManager` stay `REQUIRED` — no
//! GCS SDK dependency is decided anywhere in this workspace yet, and
//! nothing else in this crate needs one added just to close those two
//! rows out.
//!
//! Adds `adk-errors` (for `NotFoundError`/`AlreadyExistsError`-shaped
//! error variants — already a lightweight leaf crate, only depending on
//! `rusty_err`) and `adk-platform` (for `uuid::new_uuid`/`time::get_time`,
//! the workspace's existing provider-swappable ID/clock abstractions —
//! also lightweight, only `rusty_uuid` + `rusty_serde`) as new
//! dependencies. Neither widens `adk-eval`'s graph anywhere near as much
//! as `adk-agents` would (see [`eval_result`]'s doc on why
//! `session_details` stays opaque instead).
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

pub mod app_details;
pub mod constants;
pub mod conversation_scenarios;
pub mod eval_case;
pub mod eval_metrics;
pub mod eval_result;
pub mod eval_rubrics;
pub mod eval_set;
pub mod eval_set_results_manager;
pub mod eval_set_results_manager_utils;
pub mod eval_sets_manager;
pub mod eval_sets_manager_utils;
pub mod evaluator;
pub mod final_response_match_v1;
pub mod in_memory_eval_sets_manager;
pub mod local_eval_set_results_manager;
pub mod local_eval_sets_manager;
pub mod path_validation;
mod porter_stemmer;
pub mod rouge;
pub mod trajectory_evaluator;
