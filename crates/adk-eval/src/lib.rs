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
//! **C0612**: `BaseCriterion`/`ToolTrajectoryCriterion` (plus its nested
//! `MatchType`) landed in this first batch; `Interval`/`MetricValueInfo`/
//! `MetricInfo`/`MetricInfoProvider` landed alongside the metric-catalog
//! providers (C0604); `LlmAsAJudgeCriterion`/`RubricsBasedCriterion`/
//! `HallucinationsCriterion`/`LlmBackedUserSimulatorCriterion`/
//! `JudgeModelOptions` — the remaining criterion subtypes, which exist
//! only to *configure* an LLM-judge metric but don't call one
//! themselves — landed in the fourth batch below, completing C0612.
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
//! **Eighth batch**: [`evaluation_generator`] (the pure event→`Invocation`
//! grouping algorithm, C0623 — DONE) and [`agent_evaluator`] (`AgentEvaluator`'s
//! file/dataset-loading and legacy-format helpers, C0619 — partial;
//! `migrate_eval_data_to_new_schema`, C0620 — DONE). Neither needs a real
//! `Runner`/LLM-invocation path. `evaluation_generator`'s
//! `_collect_events_by_invocation_id` swaps its source `dict`'s grouping
//! for a `HashMap` + parallel order-preserving `Vec<String>`, disclosed in
//! that module's own doc as a case where (unlike this crate's other
//! `HashMap`-for-grouping choices) iteration order is semantically
//! load-bearing, since invocations are matched positionally against
//! `expected_invocations` elsewhere. `agent_evaluator`'s `DatasetInput`
//! enum models `_load_dataset`'s actual reachable `isinstance` dispatch
//! (`str` path / `list[str]` of paths) rather than its broader, partly
//! unreachable type hint; `AgentEvaluator.evaluate`/`evaluate_eval_set`
//! themselves stay `REQUIRED`, needing C0621/C0622/C0624's still-unbuilt
//! inference generation. No new dependency.
//!
//! **Seventh batch**: [`llm_as_judge_utils`] (`Label`, text-extraction/
//! score/JSON-serialization helpers, C0947 — a genuine inventory gap
//! discovered and added to the manifest mid-window, not folded into any
//! other row) and [`rubric_based_evaluator`] (`RubricResponse`,
//! `AutoRaterResponseParser`/`DefaultAutoRaterResponseParser`,
//! `PerInvocationResultsAggregator`/
//! `MajorityVotePerInvocationResultsAggregator`,
//! `InvocationResultsSummarizer`/`MeanInvocationResultsSummarizer`,
//! `normalize_text`, C0601 — partial, the source's
//! `RubricBasedEvaluator` class itself stays unbuilt, needing C0600's
//! still-deferred `LlmAsJudge` harness). Neither needs that harness
//! itself — same reasoning as the C0612 criterion types and C0632
//! persona system. `evaluator::PerInvocationResult::rubric_scores`/
//! `EvaluationResult::overall_rubric_scores` widen from opaque `Value`
//! to real `Vec<eval_rubrics::RubricScore>` this batch, now that a real
//! consumer (`rubric_based_evaluator`'s aggregators) needs the
//! structure — same "widen once a real consumer needs it" pattern as
//! `Invocation.rubrics`/`.app_details`. Adds `regex` (already a
//! workspace dependency, new usage site) for
//! `DefaultAutoRaterResponseParser`'s line-anchored patterns, rewriting
//! the source's two lookbehind patterns as ordinary capture groups
//! (Rust's `regex` crate has no lookbehind support) — disclosed in that
//! module's own doc.
//!
//! **Sixth batch**: [`base_eval_service`] (`BaseEvalService`/
//! `EvaluateConfig`/`InferenceConfig`/`InferenceRequest`/
//! `InferenceResult`/`EvaluateRequest`, C0616), [`custom_metric_evaluator`]
//! (`CustomMetricEvaluator`, C0599), and [`metric_evaluator_registry`]
//! (`MetricEvaluatorRegistry`, C0603 — partial, registers only
//! `TrajectoryEvaluator` among the 13 standard evaluators; the rest land
//! alongside their own still-`REQUIRED` rows). None need GCP or an
//! LLM-invocation path. Two registry-shaped adaptations, disclosed in
//! their own module docs: `custom_metric_evaluator`'s dynamic
//! `importlib` import becomes an explicit registration API (same
//! "class → registered closure keyed by a string" pattern as
//! `user_simulator`'s config→simulator registry); `metric_evaluator_registry`'s
//! `DEFAULT_METRIC_EVALUATOR_REGISTRY` mutable singleton becomes a
//! lazily-initialized, mutex-guarded static.
//!
//! **Fifth batch — user-simulator core + persona system**:
//! [`user_simulator`] (`UserSimulator`/`BaseUserSimulatorConfig`/
//! `NextUserMessage`/`Status` + the config→simulator registry, C0626),
//! [`static_user_simulator`] (`StaticUserSimulator`, C0629),
//! [`user_simulator_personas`] (`UserBehavior`/`UserPersona`/
//! `UserPersonaRegistry`) and [`pre_built_personas`] (`PreBuiltBehaviors`,
//! the built-in EXPERT/NOVICE/EVALUATOR personas, and
//! `get_default_persona_registry`, together C0632). None of these need
//! an LLM-invocation path — `StaticUserSimulator` replays a
//! pre-authored list, and the persona system is pure data — so, like
//! the criterion types in the fourth batch, they're useful without the
//! still-unbuilt `LlmAsJudge`/LLM-backed-simulator infrastructure.
//! `UserSimulatorProvider` (C0627) and the LLM-backed/audio simulators
//! (C0628/C0630) stay `REQUIRED` — deliberately not attempted this
//! batch, since they need that infrastructure for real. Adds
//! `adk-events` (for the real `Event` type `UserSimulator
//! ::get_next_user_message` takes — already a lightweight leaf crate,
//! only `adk-genai` + `adk-platform`) as a new dependency.
//!
//! **Fourth batch**: [`eval_config`] (`EvalConfig`/`CustomMetricConfig`/
//! `LiveModelConfig`, C0611) and the rest of [`eval_metrics`]'s criterion
//! types (`JudgeModelOptions`/`LlmAsAJudgeCriterion`/
//! `RubricsBasedCriterion`/`HallucinationsCriterion`/
//! `LlmBackedUserSimulatorCriterion`, closing out C0612 — see that
//! module's own doc for why these are pure data models despite existing
//! only to configure an LLM-judge metric, same reasoning as `Rubric`/
//! `RubricScore` before them).
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

pub mod agent_evaluator;
pub mod app_details;
pub mod audio_utils;
pub mod base_eval_service;
pub mod constants;
pub mod conversation_scenarios;
pub mod custom_metric_evaluator;
pub mod eval_case;
pub mod eval_config;
pub mod eval_metrics;
pub mod eval_result;
pub mod eval_rubrics;
pub mod eval_set;
pub mod eval_set_results_manager;
pub mod eval_set_results_manager_utils;
pub mod eval_sets_manager;
pub mod eval_sets_manager_utils;
pub mod evaluation_generator;
pub mod evaluator;
pub mod final_response_match_v1;
pub mod in_memory_eval_sets_manager;
pub mod llm_as_judge_utils;
pub mod local_eval_set_results_manager;
pub mod local_eval_sets_manager;
pub mod metric_evaluator_registry;
pub mod metric_info_providers;
pub mod path_validation;
mod porter_stemmer;
pub mod pre_built_personas;
pub mod rouge;
pub mod rubric_based_evaluator;
pub mod static_user_simulator;
pub mod trajectory_evaluator;
pub mod user_simulator;
pub mod user_simulator_personas;
