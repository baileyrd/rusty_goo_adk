# Changelog

All notable changes to this repo are documented here.
Format: Added / Changed / Deprecated / Removed / Fixed / Security, newest first.

## [Unreleased]
### Added
- P10 Apigee backend, first slice (C0549/C0550/C0551, all DONE): new
  `crates/adk-models/src/apigee_llm.rs` — `ApiType`/`validate_model_string`
  (the `apigee/[provider/][version/]model_id` DSL), `ApigeeLlm::new`'s
  full constructor (proxy_url with `APIGEE_PROXY_URL` env fallback,
  custom_headers, retry_options/client passthrough to a held `Gemini`,
  the two conflicting-options `eprintln!` warnings), and
  `identify_vertexai`/`identify_api_version`/`get_model_id` (including
  the `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` required-env-var
  checks when Vertex-routed). `ApigeeLlm(Gemini)` is ported as
  composition (holds a `Gemini` by value) rather than the source's
  subclassing, same adaptation `gemma.rs` already established. Not yet
  implementing `BaseLlm` or registered into `default_registry` — this
  slice is config/identity only; the HTTP-calling half (C0552-C0556)
  is deliberately deferred to a follow-up batch. 35 new unit tests.
- P10 Anthropic (Claude) backend, third slice (C0538, partial):
  `crates/adk-models/src/anthropic_conversion.rs` gains
  `AnthropicThinkingParam`/`build_anthropic_thinking_param` — maps genai
  `ThinkingConfig` to Anthropic's `thinking` request parameter
  (`thinking_budget`-only subset: absent budget → `Err` with the
  source's guidance message, `0` → disabled, negative → adaptive,
  positive → enabled with the given token budget). Reads
  `GenerateContentConfigStub::thinking_config`'s already-opaque `Value`
  via the same `"thinkingBudget"` key already used by
  `llm_backed_user_simulator.rs` — no struct widening needed, correcting
  C0542's own evidence which had flagged this row as blocked on one.
  `_build_effort_param`/`AnthropicGenerateContentConfig.effort` and the
  sampling-params warning stay deferred, needing the still-unbuilt
  `AnthropicLlm` backend's own config type.
- P10 Anthropic (Claude) backend, second slice (C0541, DONE):
  `crates/adk-models/src/anthropic_conversion.rs` gains `update_type_string`
  (recursive JSON-Schema `"type"`-string lowercasing, over the exact
  same dict/single/list key lists as the source) and
  `function_declaration_to_tool_param`/`AnthropicToolParam`. Turned out
  not to need the `GenerateContentConfigStub` widening the first slice's
  own evidence flagged as a blocker for the rest of P10 — this only
  touches `adk_genai::content::FunctionDeclaration`, which already has
  everything required. The `parameters`-fallback branch is simplified,
  disclosed: this port's `parameters` is already an opaque, already-
  flattened `Value` (no typed `Schema` per property to model_dump), so
  it reads `parameters`'s own `"properties"`/`"required"` keys directly.
- P10 Anthropic (Claude) backend, first slice (C0540/C0542, both DONE):
  `crates/adk-models/src/anthropic_conversion.rs` — `ToolUseIdSanitizer`
  (C0540, deterministic `toolu_fallback_N` placeholders for invalid
  tool-use ids) and `to_google_genai_finish_reason`/`AnthropicUsage`/
  `extract_prompt_token_count`/`extract_cached_token_count`/
  `extract_cache_creation_token_count`/`extract_thinking_token_count`
  (C0542, finish-reason mapping + token-usage extraction/reconciliation).
  Both pure, self-contained, no new dependency — this port talks to
  Anthropic's Messages API via plain `reqwest::blocking` (same recipe
  `gemini.rs` already established), not the Python source's `anthropic`
  SDK. Deliberately a small first slice of the 9-row P10 phase: the
  actual `AnthropicLlm` `BaseLlm` backend, extended-thinking mapping,
  the full content↔block conversion (image/PDF/tool-result media
  handling), and SSE streaming (C0536/C0537/C0538/C0539/C0543/C0544)
  are disclosed at length on C0542's own manifest row as separable,
  larger units of future work — not blocked, just genuinely bigger than
  this batch.
- P12 telemetry pure-logic batch (C0668/C0669/C0672/C0673, all DONE;
  C0661, partial): `crates/adk-models/src/token_usage.rs` (`TokenUsage`,
  C0668 — token-usage attribute names/aggregation, reading the same
  opaque camelCase `usage_metadata` keys `cache_performance_analyzer`
  already reads directly), `crates/adk-models/src/stable_semconv.rs`
  (`system_message_body`/`user_message_body`/`choice_body`, C0661
  partial — the three stable-semconv log-body builders; the experimental
  `gen_ai.client.inference.operation.details` event is a separate, larger
  surface deferred to its own batch), `crates/adk-agents/src/adk_attributes.rs`
  (the 6 `adk.experimental.skill.*` attribute-name constants, C0669),
  and `crates/adk-agents/src/schema_version.rs` (GCP + generic OTel
  exporter env-var name/default constants, C0672/C0673 — bundled
  alongside the already-DONE C0671, same "declare ahead of a blocked
  consumer" precedent). None of these five needs an OTel SDK dependency
  or touches any already-shipped public surface — every function is a
  pure data/string transform, and every constants-only row has no
  consumer yet (the real span/metric-emission machinery is a much
  larger, still-unbuilt surface).
- `NodeTool` (C0490, DONE): `crates/adk-tools/src/node_tool.rs` wraps a
  workflow `BaseNode` as a callable `BaseTool`, unblocked once
  `workflow::BaseNode`/`Context::run_node` landed — C0491's evidence
  previously called this row blocked on that; corrected. Ports the
  object-schema-wrapping declaration (`{"type":"object","properties":
  {"request":<schema>},...}` for a non-object `input_schema`) and its
  matching `run_async` unwrap, faithfully, including the surprising
  catch-all that stringifies a failed node run into a normal tool
  *result* rather than a raised tool error. An interrupted run returns
  `Ok(Value::Null)` with no separate propagation step, since
  `Context::run_node` already records the interrupt on the calling
  `Context` in place (same contract `ParallelWorker` already relies on).
  Disclosed, no-equivalent-needed: the `isinstance(node, BaseAgent)`
  guard (foreclosed by the type system) and the `FunctionNode`/
  `parameter_binding` rebinding block (this port's `FunctionNode` has no
  such concept, and isn't even a public type). 8 new tests.
- P11 local LLM-judge metrics batch (C0592/C0593/C0595/C0598, all DONE):
  `crates/adk-eval/src/final_response_match_v2.rs` (`FinalResponseMatchV2Evaluator`),
  `rubric_based_tool_use_quality_v1.rs` (`RubricBasedToolUseV1Evaluator`),
  `rubric_based_final_response_quality_v1.rs`
  (`RubricBasedFinalResponseQualityV1Evaluator`), and
  `rubric_based_multi_turn_trajectory_evaluator.rs`
  (`RubricBasedMultiTurnTrajectoryEvaluator`) — four evaluators unblocked
  once C0600's `LlmAsJudge` harness landed, none of them actually GCP-gated
  (`metric_evaluator_registry.rs`'s own module doc previously lumped them in
  with the GCP-blocked metrics; corrected). None can register into
  `MetricEvaluatorRegistry`: the harness is inherently async, while
  `Evaluator::evaluate_invocations` and `MetricEvaluatorRegistry`'s
  `EvaluatorFactory` are both deliberately sync — a structural mismatch,
  disclosed on each evaluator and on the registry's module doc.
- **Widened** `RubricBasedEvaluator::create_effective_rubrics_list`/
  `get_effective_rubrics_list` (`&mut self`/`&self -> &[Rubric]` to
  `&self` via interior-mutable `RefCell`/`&self -> Vec<Rubric>`): three of
  the four new evaluators pass `format_auto_rater_prompt` to the harness
  as a plain `Fn` closure, needing interior mutability to recompute the
  effective-rubrics cache per invocation. Backwards-compatible — every
  existing call site already bound its receiver `mut`, and this is
  `RubricBasedEvaluator`'s first real caller.
- **Manifest housekeeping:** corrected `metric_evaluator_registry.rs`'s
  stale module doc, which blamed all 12 unregistered evaluators on a
  harness that shipped the day before this batch.
- P5 Session/State closure batch (C0204 DONE; C0206/C0209/C0211 manifest
  corrections): `crates/adk-agents/src/session.rs`'s `Session` now derives
  `Serialize`/`Deserialize` with `#[rusty_serde(rename_all = "camelCase",
  deny_unknown_fields)]` (matching the source's `alias_generator=to_camel`
  + `extra='forbid'`) and gains a private, never-serialized
  `storage_update_marker: Option<String>` field (matching the source's
  `_storage_update_marker` `PrivateAttr`) — unread by anything in this
  crate today, its real caller being a future `DatabaseSessionService`
  (C0221). Purely additive: nothing in the crate serialized/deserialized
  `Session` before this change.
- **Manifest housekeeping:** C0209 (`_session_util` state-scoping helpers)
  and C0211 (`InMemorySessionService`'s storage shape + disclosed-N/A
  `*_sync` mirrors) were both already fully implemented and tested, but
  left at `REQUIRED` — corrected to `DONE`. C0206's evidence wrongly
  claimed `get_user_state` wasn't ported; it was (as part of the earlier
  C0214 batch) — corrected in place. C0210 (package export surface) and
  C0205/C0206's `StateSchemaError` piece were re-verified directly against
  the Python source and confirmed genuinely blocked (three of four
  session backends, and the whole `adk-sessions` crate, don't exist yet;
  schema-reflection has no Rust equivalent without a new dependency) —
  left `REQUIRED` with an expanded disclosure rather than stubbed.
- Plugin package close/export closure (C0352/C0361, both DONE — plus
  C0360's stale manifest status corrected): new `crates/adk-agents/src/
  plugins.rs` re-exports `BasePlugin`/`PluginManager`/`LoggingPlugin`
  (the plugin package's public export surface, C0352) — only what
  actually exists in this port; `DebugLoggingPlugin`/
  `ReflectAndRetryModelPlugin`/`ReflectAndRetryToolPlugin` stay
  unexported since they don't exist as Rust types yet (blocked on
  C0355/C0356, see below). `PluginManager` gains `set_close_timeout`
  and `PluginManager::close` (C0361) now enforces a per-plugin timeout,
  aggregating every timed-out plugin's name into one
  `PluginCloseError::Failed`, rather than returning `()` unconditionally.
  `Runner::close` (`adk-runners`) now actually applies
  `plugin_close_timeout` — previously a dead field — by cloning
  `PluginManager` and calling `set_close_timeout` before `close()`,
  logging (not propagating) a close failure so `Runner::close`'s own
  `()` return type doesn't change for its existing cross-crate caller.
- **Disclosed narrowing (C0361):** `BasePlugin::close` returns `()`,
  not a `Result` — this port's hooks have no fallible-close channel at
  all, so the only failure `PluginCloseError` can represent is a
  plugin that doesn't finish within the configured timeout; a
  panicking `close()` still propagates unchanged.
- **Manifest housekeeping:** C0360 (`PluginManager`'s early-exit/
  notify-all dispatch contract) was fully implemented and tested for
  every hook this port has today, but its manifest status was never
  flipped from `REQUIRED` to `DONE` — corrected, noting that C0355/
  C0356's eventual hooks will apply this same already-established
  pattern rather than needing new dispatch work of their own.
  Investigated C0355/C0356 (`BasePlugin`'s model-level/tool-level
  hooks) directly and confirmed they remain correctly blocked on the
  `adk-agents`↔`adk-models`/`adk-tools` crate-cycle already documented
  — no change needed there, just re-verified rather than assumed.
- **Tests:** `crates/adk-agents/src/plugins.rs::tests::*` (2 tests);
  `crates/adk-agents/src/services.rs::tests::
  {close_reports_a_timeout_for_a_slow_plugin,
  close_aggregates_multiple_slow_plugins_into_one_error}`;
  `crates/adk-runners/src/runner.rs::tests::
  close_applies_the_configured_plugin_close_timeout`.
- P7 node-wrapping trio (C0296/C0317/C0326, all DONE — see below for
  disclosed narrowings): new `crates/adk-agents/src/
  workflow_parallel_worker.rs` (`ParallelWorker`/`parallel_worker_node`,
  C0317) — runs a wrapped node once per list-input item, empty-list
  short-circuit, non-list input wrapped as a single item; and new
  `crates/adk-agents/src/workflow_node_factory.rs` (`build_node`/
  `is_node_like`, C0326; `node`/`NodeFactoryError`, C0296) — converts a
  `NodeLike` to a concrete `BaseNode`, and composes that with
  `parallel_worker_node` when `parallel_worker=true` is requested.
- **Disclosed narrowing (C0317):** `ParallelWorker` dispatches
  sequentially, not concurrently — `Context::run_node` needs `&mut
  Context` for a node's entire execution, the same constraint that
  already forced `Workflow`'s own LOOP phase (C0301) to bypass it and
  build a bespoke concurrent combinator instead; `ParallelWorker` has
  no such machinery to reuse, so items run one at a time through the
  already-shipped `Context::run_node` directly. Deterministic
  earliest-index failure is preserved by construction rather than by
  replicating the source's stable-sort tiebreak; "cancel remaining
  in-flight items"/5s drain timeout are not applicable (nothing is ever
  concurrently in-flight); an interrupted item stops the batch and
  yields nothing, relying on `Context::run_node`'s own already-tested
  interrupt-id propagation. `max_parallel_workers` is preserved as a
  validated (`>= 1`) field for API-shape fidelity but has no effect
  under sequential dispatch.
- **Disclosed narrowing (C0326):** `build_node` narrows to the
  `BaseNode`/`START` cases of `NodeLike` — `BaseTool`→`_ToolNode` stays
  out of scope (the `adk-tools`/`adk-agents` crate-cycle, same as
  C0355/C0356); `LlmAgent`/task-mode-`RemoteA2aAgent` auto-defaulting
  stays out of scope (needs the C0092 tree-fusion gap); `callable`→
  `FunctionNode` has no runtime dispatch to build since Rust has no
  `isinstance`/`callable()` check (a caller constructs `FunctionNode`
  directly instead). Within the `BaseNode` branch, only the
  no-overrides case is supported (`node_like.model_copy(update=kwargs)`
  has no equivalent since `NodeBehavior` has no `Clone` bound — adding
  one would be a breaking supertrait change to every existing
  implementor, its own stop-and-ask) — an override attempt returns
  `BuildNodeError::OverridesNotSupportedForBaseNode`. `is_node_like`
  narrows to a disclosed constant `true`, since `NodeLike` is already a
  closed, exhaustively-matched enum.
- **Disclosed narrowing (C0296):** `node(...)` narrows to its "wrap an
  already-resolved `NodeLike`" overload — the bare-decorator overload
  (`@node`/`@node()` on a function) has no Rust equivalent, so
  `auth_config`/`parameter_binding` (which belong to `FunctionNode`'s
  own constructor) aren't parameters here. `Node` (the subclassable
  base class) is a **permanent** narrowing: this port's `NodeBehavior`
  trait-object design already is the Rust equivalent of "subclass and
  override" — any implementor wanting parallel-worker fan-out just
  wraps its own built `BaseNode` with `parallel_worker_node` directly.
- Also corrected two stale manifest rows while in this area: C0307
  (`NodeState`/`NodeStatus`/`Trigger`) was fully complete from an
  earlier batch but still showed `REQUIRED`; C0298 (`Workflow`'s own
  struct-skeleton row) is updated with the full, compiler-verified
  detail on the `NodeBehavior`/`BaseNode` wiring blocker (see the
  dedicated PR that disclosed it).
- **Tests:** `crates/adk-agents/src/workflow_parallel_worker.rs::tests::*`
  (7 tests); `crates/adk-agents/src/workflow_node_factory.rs::tests::*`
  (9 tests).
- `Workflow` FINALIZE phase (C0306, DONE — see below for disclosed
  adaptations): `Workflow::finalize` propagates interrupt ids onto `ctx`
  or sets `ctx.output` from the single terminal node's output (erroring
  via new `WorkflowError::MultipleTerminalOutputs` if more than one
  terminal node produced output). `Workflow::cleanup_all_tasks` drains
  any still-pending node futures and marks their nodes `CANCELLED`.
  Small helpers `Workflow::collect_remaining_interrupts` (gathers
  interrupt ids from nodes still `WAITING` after the LOOP phase) and
  `Workflow::has_terminal_output` (whether any terminal node produced
  output) round out the batch, plus a no-op `Workflow::
  validate_output_data` matching `validate_state_schema`'s
  already-established shape.
- **Disclosed adaptation:** `finalize` is `async fn`, unlike the
  source's sync `_finalize` — it unions two separate interrupt-id
  accumulators, `WorkflowLoopState::interrupt_ids` (static graph nodes)
  and `DynamicNodeScheduler::interrupt_ids` (dynamic `ctx.run_node()`
  nodes), since the source's single `_LoopState(DynamicNodeState)`
  inheritance makes these the same Python attribute but this port's two
  unrelated structs keep them separate; reading the scheduler's side
  needs its async `Mutex` guard.
- **Disclosed no-op:** `cleanup_all_tasks`'s dynamic-task cancellation
  half (`loop_state.get_dynamic_tasks()`/`loop_state.runs`) has no
  equivalent here — this port's dynamic dispatch always executes inline
  via `DynamicNodeScheduler::call` (already-disclosed C0318/C0319
  narrowing), never as a separately-scheduled task. Only the
  static-node half (draining `pending_tasks`, marking `CANCELLED`) is
  ported, and needs no `async`/explicit cancellation either — dropping
  a boxed `PendingNodeFuture` without polling it to completion already
  is cancellation (Rust's standard async-cancellation-by-drop model).
- New `WorkflowError::MultipleTerminalOutputs`/`WorkflowError::
  Context(#[from] ContextError)` variants.
- **Known limitation:** `Workflow` is still not wired into a
  `BaseNode`/`NodeBehavior` — that glue (`setup` → `run_loop` →
  `cleanup_all_tasks` → `collect_remaining_interrupts` → `finalize`,
  matching `_run_impl`'s own sequencing) is deliberately a separate
  follow-up, since it also needs to settle how `Workflow`'s own
  `max_concurrency`/`rerun_on_resume` fields map onto `BaseNode::
  build`'s constructor parameters.
- **Tests:** `crates/adk-agents/src/workflow_workflow.rs::tests::
  {finalize_propagates_interrupt_ids_and_skips_terminal_output,
  finalize_sets_ctx_output_from_the_single_terminal_output,
  finalize_is_a_no_op_with_no_terminal_output_and_no_interrupts,
  finalize_errors_when_more_than_one_terminal_node_has_output,
  cleanup_all_tasks_drains_pending_tasks_and_marks_nodes_cancelled,
  collect_remaining_interrupts_gathers_from_waiting_nodes_only,
  has_terminal_output_checks_only_terminal_node_outputs}`.
- `Workflow` LOOP driver + completion handling (C0301/C0304/C0305, all
  DONE — see below for disclosed adaptations): `Workflow::run_loop`
  (C0301) schedules ready nodes and drains pending tasks until none
  remain, processing every batch of simultaneous completions in
  deterministic order (recovered-sequence first via the new
  `ReplaySequenceBarrier::sequence` accessor, then stable insertion
  order for anything not in the recovered sequence). On the first node
  error in a batch it marks that node `FAILED`, sets `WorkflowLoopState
  ::error_shut_down`, and stops — without processing any other
  completions in the same batch, matching the source's own behavior.
  New free fn `wait_for_completions`: a hand-rolled `std::future::
  poll_fn` combinator (no `rusty_tokio::task::JoinSet` — `Send +
  'static` incompatible with `PendingNodeFuture`'s non-`'static`
  `&Context` borrow — and no `rusty_tokio::select!`, a fixed-arity
  macro) that polls every entry in `pending_tasks` and collects **all**
  that are `Ready` in one pass, not just the first, since the
  deterministic-order sort only matters when more than one completion
  can be observed per call. `Workflow::handle_completion` (C0304) ports
  the source's exact 3-way outcome routing (interrupt → `WAITING` +
  propagate interrupt ids; `wait_for_output` with nothing yet →
  `WAITING`; else → `COMPLETED`, caching output/branch then buffering
  downstream triggers) — no longer `async`, since the checkpoint
  builders it calls are already pure `Option<Event>` functions.
  `Workflow::buffer_downstream_triggers` (C0305) finds downstream edges
  via `Graph::get_next_pending_nodes` and buffers a `Trigger` per
  target: normal successors get the completing node's own output/branch
  (sub-branching on fan-out), a `JoinNode`-like target only fires once
  every predecessor has `COMPLETED`, with input built as a
  `{predecessor_name: output}` map and branch set to the common prefix
  of every predecessor's cached branch — reusing the already-shipped
  `workflow_join_node::common_branch_prefix` rather than porting a
  second copy of the source's module-level `get_common_branch_prefix`
  (the same n-ary common-prefix computation, one pairwise-reduced, one
  all-at-once, over the same associative operation). New
  `crate::workflow_graph::value_to_route_spec` converts a completed
  node's raw, dynamically-typed emitted `ctx.route` into the typed
  `RouteSpec` `get_next_pending_nodes` needs.
- **Disclosed divergence:** `run_loop` takes `ctx: &Context`, not
  `&mut Context` — every pending node future borrows `ctx` immutably
  for as long as it's outstanding (new tasks can be scheduled on any
  loop iteration), so an `&mut Context` parameter would conflict with
  those outstanding borrows for the loop's entire duration. The
  captured error is returned as `RunLoopOutcome{events, error:
  Option<(message, node_path)>}` instead of being written onto `ctx`
  inline the way the source's `_run_loop` does — the caller (the
  not-yet-built `NodeBehavior` wiring) applies it onto its own
  `&mut Context` once every pending future has gone out of scope.
- **Disclosed no-op:** the source's final "await fire-and-forget
  dynamic tasks" step in `_run_loop` has no equivalent here — this
  port's `Context::run_node` (Mode 1) always executes and awaits a
  `DynamicNodeScheduler::call` inline (already-disclosed C0318/C0319
  narrowing), so there is no separately-scheduled dynamic task list to
  await.
- **Changed:** `ReplaySequenceBarrier::check_and_advance` widened from
  `&mut self` to `&self` (its `current_index`/`unblocked` state moved
  into an interior `std::sync::Mutex`, never held across an `.await`),
  and gained a `sequence()` accessor — both needed once `run_loop`
  reads/advances the `Arc`-shared barrier across concurrently pending
  node futures.
- **Known limitation:** still not wired into a `BaseNode`/
  `NodeBehavior`. `Workflow._finalize`/`_cleanup_all_tasks` (C0306) —
  terminal-output collection, remaining-interrupt propagation, and
  leftover-task cleanup — and the `NodeBehavior` wiring itself are
  separate follow-up work.
- **Tests:** `crates/adk-agents/src/workflow_workflow.rs::tests::
  {run_loop_drives_a_linear_workflow_to_completion,
  run_loop_stops_and_returns_the_first_error,
  handle_completion_moves_to_waiting_and_collects_interrupt_ids_on_interrupt,
  handle_completion_waits_when_a_wait_for_output_node_has_nothing_yet,
  handle_completion_completes_and_buffers_a_downstream_trigger,
  buffer_downstream_triggers_fans_out_with_sub_branching,
  buffer_downstream_triggers_fires_a_join_only_once_every_predecessor_completes}`;
  `crates/adk-agents/src/workflow_graph.rs::tests::
  {value_to_route_spec_converts_each_scalar_variant,
  value_to_route_spec_converts_a_seq_to_many_filtering_unconvertible_entries,
  value_to_route_spec_is_none_for_unconvertible_values}`.
- `Workflow` node-scheduling primitives (C0302/C0303, both DONE — see
  below for narrowings): `Workflow::schedule_ready_nodes`/
  `start_node_task` and their helpers (`has_waiting_task_agent`,
  `at_concurrency_limit`, `prepare_node_state_for_starting`, free fns
  `next_run_id`/`compute_isolation_scope_for_node`/
  `create_node_state_for_new_run`), plus the resumability-checkpoint
  builders `Workflow::node_checkpoint_event`/
  `maybe_reemit_replayed_output_event`/`end_of_agent_event`. `Workflow
  ::start_node_task` (C0303) is a disclosed, deliberate divergence from
  the source's literal call chain: rather than routing through
  `ctx.run_node()`/`DynamicNodeScheduler` (which both need exclusive
  `&mut Context` access for a node's whole execution — structurally
  incompatible with the LOOP phase's purpose of running graph nodes
  concurrently, not just a performance tradeoff), it dispatches via
  `NodeRunner::run` directly, the only primitive whose shared `&Context`
  borrow lets multiple pending node futures be built and polled at once.
  Fidelity is preserved via `start_node_task`'s own inline
  replay-interception check against `loop_state.recovered_executions`.
  New `PendingNodeFuture<'a>` type alias (a local boxed, non-`'static`
  future per pending node — `rusty_tokio::task::JoinSet` needs `Send +
  'static`, which `NodeRunner::run`'s `&Context`-borrowing future can't
  satisfy without a breaking rework). `WorkflowLoopState` grows `nodes`/
  `replayed_nodes`/(private) `pending_tasks` fields and a lifetime
  parameter; `sequence_barrier` is now `Arc`-wrapped so multiple pending
  futures can share one barrier. The three checkpoint builders are
  redesigned as pure functions returning `Option<Event>` (this port's
  "eagerly collected `Vec`" adaptation) rather than enqueuing onto a live
  per-invocation event queue, for the not-yet-built LOOP driver (C0301)
  to push into its own accumulator — gated on `InvocationContext::
  is_resumable`, matching the source. `has_waiting_task_agent` narrows
  to always `false` (needs a `node.mode == "task"` check this port's
  `BaseNode` has no equivalent for, the same C0092 LlmAgent tree-fusion
  gap already disclosed elsewhere). Still not wired into a `BaseNode`/
  `NodeBehavior` — nothing yet drives `pending_tasks` to completion; that
  lands with the LOOP driver (C0301) and the rest of the batch
  (C0304-C0306).
- `Workflow` struct skeleton + SETUP phase (C0298/C0299/C0300, all DONE —
  see below for narrowings): new `crates/adk-agents/src/
  workflow_workflow.rs`. `Workflow::new` builds and validates a `Graph`
  from `edges` (`rerun_on_resume` defaults `true`, unlike `BaseNode`'s
  own `false`); `Workflow::validate_state_schema` (C0299) is a disclosed
  no-op — needs `FunctionNode`'s wrapped-function signature
  introspection, which this port doesn't have, over an already-opaque
  `state_schema`. `WorkflowLoopState::setup` (C0300) ports the SETUP
  phase: rehydrates recovered executions via a fresh `ReplayManager`,
  warns (to stderr) when `resume_inputs` were given but nothing was
  recovered, seeds triggers for `START`'s direct successors via the new
  order-preserving `WorkflowLoopState::push_trigger` (a `Vec<(String,
  Vec<Trigger>)>`, not a `HashMap`/`BTreeMap` — the source relies on
  `dict` insertion order for deterministic scheduling), and installs a
  freshly built `DynamicNodeScheduler` on `ctx` via the new
  `Context::set_workflow_scheduler`. New `ReplayManager::
  take_sequence_barrier` (additive — `ReplaySequenceBarrier` isn't
  `Clone`, so moving it off the `ReplayManager` that built it needs an
  explicit take rather than a borrow). **Not yet wired into a
  `BaseNode`/`NodeBehavior`**: the LOOP phase (C0301-C0305:
  `_run_loop`/`_schedule_ready_nodes`/`_handle_completion`/
  `_buffer_downstream_triggers`) and FINALIZE (C0306) are a separate,
  larger follow-up batch — `_run_impl`'s orchestration loop can't
  meaningfully run without them, so `Workflow` stays a standalone,
  directly-unit-tested struct for now (the same "build the layer, defer
  the caller" shape already used for `NodeRunner`/`ReplayManager`/
  `check_interception`). `DynamicNodeScheduler::new` always builds its
  own fresh `ReplayManager` rather than sharing the Workflow's own (the
  source's `_LoopState(DynamicNodeState)` inheritance shares one
  instance) — disclosed as a narrowing, not a correctness gap (neither
  scan mutates session state, so a second instance just redundantly
  rebuilds the same index).
- `DynamicNodeScheduler`/`ScheduleDynamicNode` (C0318, DONE — concurrent-
  task dedup disclosed as not ported) and its rehydration/execution/
  outcome-classification internals (C0319, DONE — cancellation clean-
  removal disclosed as not applicable): new `crates/adk-agents/src/
  workflow_dynamic_node_scheduler.rs`. Backs `Context::run_node`'s Mode 1
  (Workflow-scheduled dynamic dispatch), previously disclosed as
  unconditional dead code — `Context` now gains `workflow_scheduler`
  (inherited-or-freshly-created per node context, via `Context::for_node`,
  mirroring the source's `_derive_scheduler`) and `child_run_counters`
  fields, and `Context::run_node`'s loop branches on whether a scheduler
  is present: Mode 1 through `DynamicNodeScheduler::call` for any node
  context, Mode 2 (standalone, via `NodeRunner` directly) still for a
  direct call on a root context. Fresh/completed-dedup/waiting-resume
  3-case dispatch and the 5-way run-outcome classification are fully
  ported and tested end-to-end (a node body dynamically dispatching the
  same target twice with the same run id fast-forwards the second call).
  `check_interception` (C0321) regains its "Case 1" (same-turn completed/
  waiting interception), restored now that `DynamicNodeRun` exists to
  back it — previously narrowed out for lack of a `current_run` type.
  Concurrent-task dedup (`asyncio.Task`-based "await the already-in-flight
  call") stays unported: nothing in this port spawns concurrent tasks
  that could race on the same dynamic-node path, and `Workflow`'s own
  concurrent LOOP phase (C0300-C0306, the only future source of that) isn't
  built. Corrected a stale disclosure in C0059/C0060's own evidence along
  the way (Mode 1 is no longer dead code) and in C0146's (the multi-step
  tool-call loop and before/after-model callback dispatch were already
  shipped, contrary to what that row's text still said).
- Dynamic workflow-node dispatch: `Context::run_node` (C0059, DONE) and
  its in-place agent-transfer loop (C0060, DONE). New `RunNodeOptions`/`RunNodeOutcome`/
  `RunNodeOutput` on `context.rs`: events are returned (not enqueued, this
  port's usual "eagerly collected `Vec`" adaptation) and an interrupted
  child surfaces as `RunNodeOutcome::Interrupted` rather than the source's
  raised (deliberately non-catchable) `NodeInterruptedError` — a
  `NodeBehavior::run_impl` calling `run_node` observes it directly instead
  of needing a dedicated exception catch. New `Context::for_node` field
  `node_rerun_on_resume` (mirrors the source's `self._node_rerun_on_resume`,
  the *caller's* own rerun-on-resume flag `run_node` validates against —
  not the dispatched node's). New `crates/adk-agents/src/
  workflow_transfer_utils.rs`: `resolve_and_derive_transfer_context`
  (C0325, DONE) resolves self/child/sibling/parent-climb/root-bypass-
  fallback/unrelated transfer targets against a local, per-call
  `ChainFrame` ancestry list built by `run_node`'s own loop — adapted from
  the source's permanent `Context.node`/`Context.parent_ctx` object
  references, deliberately never added to this port's `Context` (would
  mean either an `Arc`-shared `Context`, a breaking change to every
  existing `&mut Context` call site, or an owned `Context` tree with no
  behavioral payoff). New `crates/adk-agents/src/workflow_agent_node.rs`:
  `agent_node`/`AgentNode` (C0043, DONE) wraps a `BaseAgent` as a
  `workflow_base_node::BaseNode`, the generic `BaseAgent._run_impl`
  adapter — ported as a standalone wrapper rather than a `BaseAgent`
  method, since this port's `BaseAgent` and `BaseNode` are unrelated types
  (the source's `BaseAgent` is itself a `BaseNode` subclass). Scoped to
  the generic run-once-collect-every-event adapter; `LlmAgent`'s
  task/chat-mode dispatch loop and task delegation (`_llm_agent_wrapper.py`)
  is the separate, larger C0407.
- `_nl_planning`'s request/response processors (C0176/C0179, both DONE,
  closing C0200 as a side effect): `LlmFlow` gains a `planner: Option<
  Arc<dyn BasePlanner>>` field and `with_planner` builder — the same
  "caller supplies the resolved bits" shape already established for
  `tools`/`tools_dict`. New `crates/adk-flows/src/nl_planning.rs`:
  `apply_nl_planning_request` applies `BuiltInPlanner`'s thinking config
  or (for any other planner) appends its planning instruction and strips
  pre-existing `thought` flags, wired into `LlmFlow::preprocess` right
  after `context_cache`; `apply_nl_planning_response` runs
  `process_planning_response` for any non-`BuiltInPlanner`, replaces the
  response's parts, and surfaces a state-update event when the planner
  touched session state — wired into `LlmFlow::postprocess`, emitted
  before the model-response event. `BasePlanner` widens to `AsAny + Send
  + Sync` (additive — its only two impls and its only consumer are in
  this same batch) so the response processor can downcast to the
  concrete `BuiltInPlanner` type, replacing the source's Python-only
  unbound-method-identity check with the same effective skip set.
  Corrected a stale claim in `C0144`'s own evidence along the way:
  `compaction`/`interactions_processor`/`output_schema_processor` were
  already wired into `LlmFlow::preprocess` in earlier batches, contrary
  to what that row's text still said.
- Computer-use tool trio (`tools/computer_use/`): `BaseComputer`
  (C0445, DONE) — the full browser-automation trait contract
  (click/hover/type/scroll/wait/navigate/search/key-combo/drag-drop/
  screenshot/state); `ComputerUseTool` (C0447, DONE) — normalizes
  model-supplied virtual-coordinate-space input to real screen size and
  gates execution behind the model's own safety-confirmation protocol;
  `ComputerUseToolset` (C0446, partial) — a fixed 15-entry tool table
  (14 real actions plus a confirmed, faithfully-replicated source quirk:
  `initialize` leaks into the tool set too, since the source's own
  `EXCLUDED_METHODS` doesn't list it), `navigate` SSRF-hardened by
  reusing `load_web_page`'s validation (`load_web_page.rs`'s SSRF core
  is now `pub(crate)`, plus a new `resolve_direct_addresses` combining
  resolve+block-check). One source check — a raw-backslash-in-netloc
  special case — isn't ported: verified empirically that this port's
  WHATWG-spec-compliant URL parser already resolves that exact input the
  same way a real browser would, closing the parser-disagreement the
  check exists to prevent. `adapt_computer_use_tool` isn't ported —
  narrows `gemini.rs::Gemini::preprocess_request`'s (C0132) own
  disclosed gap down to exactly this one function, blocked on a
  crate-graph-direction issue (`LlmRequest.tools_dict` would need
  `BaseTool`, which lives in a crate `adk-models` can't depend on).
- `Gemini::connect_live` (C0131, DONE) opens a real Live API WebSocket
  connection, sends the `BidiGenerateContentSetup` envelope, and returns
  a ready `GeminiLlmConnection` — the handshake half `Gemini::connect()`
  was waiting on since an earlier batch shipped only its config-prep
  half (`prepare_live_connect_config`/`live_api_version`). New
  `crates/adk-models/src/live_setup_request.rs`: the setup-envelope wire
  shape, read directly out of the installed `google-genai` package's
  `_live_converters.py`. `LiveWsConnection::connect_with_headers` (new,
  additive) attaches `x-goog-api-key` to the WebSocket handshake, the
  same way the REST transport attaches auth. Wired as `Gemini`'s first
  real `BaseLlm::connect` override; Gemini-API-key backend only, mirroring
  `generate_content`'s already-disclosed Vertex AI narrowing. Also
  `Gemini::preprocess_request` (C0132, partial): strips unsupported
  `labels`/inline `display_name` on the Gemini API backend and sanitizes
  every content part's `inline_data` via `as_safe_part_for_llm` — the
  computer-use `wait`-function adaptation stays out of scope, blocked on
  `ComputerUseToolset` (C0446), same disclosed gap C0195 already names.
  `as_safe_part_for_llm` relocates from `adk-tools::load_artifacts_tool`
  to new `crates/adk-genai/src/safe_part.rs` (re-exported from its old
  path, no caller-visible change) — a crate-graph-direction fix: the
  source has `models` import from `tools`, but this workspace's crate
  graph runs the other way, so the function moves to the common ancestor
  both crates already depend on.
- New `crates/adk-eval/src/llm_as_judge.rs` (C0600, DONE) and
  `RubricBasedEvaluator` in `crates/adk-eval/src/rubric_based_evaluator.rs`
  (C0601, DONE), ported from `evaluation/llm_as_judge.py` and
  `evaluation/rubric_based_evaluator.py`. `evaluate_invocations_via_llm_judge`
  is the `LlmAsJudge[CriterionT]` harness: builds one judge-model request
  per invocation, samples it `num_samples` times, and — verified against
  the real source, not assumed — marks the *whole* invocation
  `NOT_EVALUATED` if even one sample fails, discarding its successful
  samples too. Ported as a free async function taking the source's four
  abstract hooks as closures rather than a trait: no concrete per-metric
  evaluator exists yet to supply a real `format_auto_rater_prompt`
  (every one is GCP-blocked, C0591-C0598), and it also can't implement
  `Evaluator` itself without a breaking async-widening of that
  already-shipped trait. `RubricBasedEvaluator` composes an
  `LlmAsJudgeConfig<RubricsBasedCriterion>` and provides the three hooks
  the source itself actually implements: rubric-set merging
  (criterion+invocation scope, duplicate-`rubric_id` rejection, optional
  `rubric_type` filtering), ID-then-normalized-text rubric-response
  matching, and the majority-vote/mean-score aggregator delegations.
  Adaptations disclosed in the module docs: sequential instead of
  semaphore-bounded concurrent sampling (identical results, only
  wall-clock parallelism narrows); `judge_model_config` merge and
  `add_default_retry_options_if_not_present` not applied (existing
  disclosed gaps); lookbehind regexes become capture groups; NFKC
  normalization skipped (same gap as `rouge.rs`). 33 new tests
  (8 + 25).
- New `crates/adk-models/src/gemma.rs`: `Gemma`/`GemmaFunctionCallingMixin`/
  `GemmaFunctionCallModel` (C0113/C0545/C0546/C0548, all DONE), ported
  from `models/gemma_llm.py`. `Gemma` composes a `Gemini` instance
  (Rust has no multiple inheritance for the source's mixin) to work
  around Gemma 3's lack of native function calling/system-instruction
  support: tool declarations get injected as a strict-JSON text
  system-instruction block, function calls are parsed back out of the
  model's text response (fenced code block or the last valid JSON
  substring), and any system instruction is converted to a prepended
  user-role message. Registered in `default_registry` after `Gemini`,
  whose own `gemma-4.*` pattern already precedes `Gemma`'s broader
  `gemma-.*` — resolving C0113's registration-precedence requirement
  (Gemma 4+ still resolves natively to `Gemini`) as a side effect, no
  new mechanism needed. Adaptations, disclosed at length in the module
  doc: `_get_last_valid_json_substring`'s `json.JSONDecoder.raw_decode`
  becomes a quote/escape-aware brace-matcher; `Gemma::api_backend()` is
  read-parity only (has no effect on the inner `Gemini`'s actual
  backend selection, which is process-global, not per-instance); the
  source's bare `assert` on a non-Gemma model becomes a real `Err`
  instead of a panic. `Gemma3Ollama` (C0547) stays out of scope — it
  needs `LiteLlm` (C0557), not built in this port at all. 24 new tests.
- `crates/adk-eval/src/user_simulator.rs`: `UserSimulatorProvider` now
  genuinely dispatches `"llm_backed"` scenario cases (C0627, partial →
  closes its own disclosed gap), ported from
  `user_simulator_provider.py`'s module-level
  `register_user_simulator(LlmBackedUserSimulatorConfig,
  LlmBackedUserSimulator)`. Since Rust has no module-import-time side
  effects, this port instead seeds the built-in registration in
  `registry()`'s own lazy static — the same shape
  `metric_evaluator_registry::default_registry` already established for
  its own built-in evaluators. `SimulatorFactory`/`register_user_simulator`/
  `create_user_simulator` widen to also take the `ConversationScenario`
  a scenario-driven simulator's constructor needs (zero external
  callers verified). The `None`-config legacy-default path and an
  explicit `"llm_backed"` config both now resolve to a real
  `LlmBackedUserSimulator` instead of erroring. The audio-decorator
  composition (`_LlmAudioUserSimulator`, C0630) stays out of scope —
  that type still doesn't exist in this port. 1 new test, 2 existing
  tests updated to reflect the now-successful dispatch.
- New `crates/adk-eval/src/llm_backed_user_simulator.rs`:
  `LlmBackedUserSimulatorConfig`/`LlmBackedUserSimulator` (C0628, DONE),
  ported from `evaluation/simulation/llm_backed_user_simulator(_prompts).py`.
  Generates the next user message via an LLM: first invocation returns
  the scenario's static starting prompt, later ones prompt the model
  with a summarized conversation history and classify the result
  (success / stop-signal-detected / turn-limit-reached / a genuine
  generation failure). `adk-eval` gains new `adk-models`/`rusty_tokio`
  dependencies (verified no cycle). `UserSimulator::get_next_user_message`
  widens from sync/infallible to `async`/`Result<NextUserMessage, String>`
  in the same batch (zero external callers verified) — the source's own
  documented `raise RuntimeError` failure path has no `Status` variant
  to map onto. Narrowed, disclosed at length in the module doc: only
  the flat, no-persona prompt template renders (the persona-decorated
  template needs real Jinja2 loops/filters this port doesn't have);
  `is_valid_user_simulator_template` narrows from Jinja AST inspection
  to a regex placeholder-presence check; `add_default_retry_options_if_not_present`
  isn't ported (the source flags it internal-only). Not auto-registered
  under `"llm_backed"` — this port has no module-load-time registration
  side effects. 19 new tests.
- `crates/adk-flows/src/llm_flow.rs`: `LlmFlow`'s tool-declaration
  commit phase (C0151, partial), ported from `base_llm_flow.py`'s
  `_process_agent_tools`. `LlmFlow::preprocess` now calls each
  caller-supplied tool's `process_llm_request` serially, in
  `LlmFlow::tools`' list order, right after `output_schema` — matching
  the source's own ordering guarantee that later tools can observe
  earlier tools' mutations to the outgoing `LlmRequest` (e.g.
  `GoogleSearchTool` writing `llm_request.model`). New `LlmFlow::tools`
  field + `with_tools` builder, kept separate from the pre-existing
  `tools_dict`/`with_tools_dict` since a `HashMap` has no ordering.
  Still narrowed: automatic resolution from `agent.tools` stays blocked
  on C0092. 2 new tests.
- New `crates/adk-flows/src/google_search_agent_tool.rs`:
  `create_google_search_agent`/`create_google_search_agent_tool`
  (C0429, DONE), ported from `tools/google_search_agent_tool.py`. Builds
  the dedicated single-tool sub-agent (`LlmAgent` + `LlmFlow` wired with
  just `GoogleSearchTool`, via the C0151 batch above) that lets
  `google_search` coexist with other tools on an agent — Gemini
  restricts `google_search` to sole-tool use. Adaptation, disclosed:
  `GoogleSearchAgentTool`'s source subclass adds nothing over a plain
  `AgentTool` in this port, since its only added behavior
  (`propagate_grounding_metadata`) has no consumer here yet (that
  workaround is itself blocked on C0092) — so this batch exposes the
  constructor functions directly rather than a no-op newtype. 2 new
  tests.
- New `crates/adk-tools/src/base_authenticated_tool.rs` and
  `crates/adk-tools/src/authenticated_function_tool.rs`: `BaseAuthenticatedTool`/
  `AuthenticatedFunctionTool` (C0412, DONE), ported from
  `tools/base_authenticated_tool.py`/`tools/authenticated_function_tool.py`.
  Both resolve an `AuthConfig` credential via `CredentialManager` before
  invocation, returning a configurable "Pending User Authorization"
  placeholder when no credential is available yet instead of running
  the wrapped logic. `BaseAuthenticatedTool` takes a boxed closure
  standing in for the source's abstract `_run_async_impl`
  (composition, same shape `FunctionTool` already uses for its wrapped
  closure); `AuthenticatedFunctionTool` composes an inner `FunctionTool`
  rather than subclassing it. Both hold their `CredentialManager` behind
  a `rusty_tokio::sync::Mutex` since `get_auth_credential` is itself
  `async` and needs `&mut self`. Gated by their existing
  `FeatureName::BaseAuthenticatedTool`/`FeatureName::AuthenticatedFunctionTool`
  registry entries. `GoogleTool`/`_google_credentials.py` (C0413/C0414)
  stays explicitly out of this batch — it needs a real OAuth2/ADC
  client stack (`google.auth`/`google.oauth2`), a new third-party
  dependency this batch doesn't add. 10 new tests.
- New `crates/adk-eval/src/local_eval_service.rs`: `generate_final_eval_status`
  (C0617, partial), ported from `evaluation/local_eval_service.py`'s
  `LocalEvalService._generate_final_eval_status` — the one synchronous,
  dependency-free method on that `@experimental` orchestrator class.
  Rolls up a list of per-metric `EvalMetricResult`s into one overall
  `EvalStatus`: a `Passed` result keeps scanning (a later result can
  still override it), `NotEvaluated` is skipped, and a `Failed` result
  short-circuits immediately, even before results that would otherwise
  have passed. `LocalEvalService`'s remaining nine methods stay
  unported — all `async`, needing semaphore-bounded concurrency and a
  `Runner`-driven inference generator (C0621/C0622, still `REQUIRED`).
  6 new tests.
- `crates/adk-eval/src/evaluation_generator.rs`: `process_query_with_session`
  (C0624, partial), ported from `evaluation_generator.py`'s
  `EvaluationGenerator._process_query_with_session` — replays recorded
  `Session` events to fill in `actual_tool_use`/`response` for a list of
  eval-dataset entries without invoking a `Runner`, matching the
  source's own accumulate-across-every-matching-user-event behavior
  exactly. `adk-eval` gains a new dependency on `adk-agents` for the
  real `Session` type (verified: no cycle — `adk-agents`'s own
  dependency chain never reaches back to `adk-eval`). Not ported:
  `generate_responses_from_session`'s JSON-file-reading wrapper —
  disclosed in the module doc as genuinely out of scope, since it has
  zero callers besides its own test and `Session` doesn't derive
  `Deserialize` yet. 4 new tests.
- `crates/adk-flows/src/code_execution.rs`: the general (non-built-in)
  code-executor response path (C0180, `DONE`), ported from
  `flows/llm_flows/_code_execution.py`'s general-executor branch.
  `apply_code_execution_response` (now `&mut InvocationContext`) gains a
  second branch alongside the pre-existing `BuiltInCodeExecutor` one:
  checks the per-invocation error-retry limit via `CodeExecutorContext`,
  extracts and truncates the first fenced code block
  (`extract_code_and_truncate_content`), emits a code event, resolves a
  stateful-or-per-invocation execution id (new
  `get_or_set_execution_id`), calls `BaseCodeExecutor::execute_code`, and
  post-processes the result (new `post_process_code_execution_result`) —
  updating the error-retry counter, saving every output file as an
  artifact (base64-encoded `MediaBlobStub` `Part`, the same convention
  `file_artifact_service.rs`/`load_artifacts_tool.rs` already use), and
  clearing the original response content so the agent loops for another
  turn. Disclosed in the module doc: `CodeExecutorContext`'s buffered
  nested `_code_execution_context` sub-dict isn't auto-flushed to session
  state on drop, so each scoped instance that mutates it explicitly
  applies `get_state_delta()` back before the borrow ends; Python's
  `get_content_as_bytes` union-resolution helper isn't needed since this
  port's `File.content` is already normalized to `Vec<u8>`. Still not
  wired into `LlmFlow::postprocess` (same standing C0092 blocker every
  sibling processor in this file discloses). 8 new tests.
- `crates/adk-flows/src/llm_flow.rs`: before/after/on-error model
  callback dispatch (C0153/C0154/C0155, all `DONE`), ported from
  `flows/llm_flows/base_llm_flow.py`'s `_handle_before_model_callback`/
  `_handle_after_model_callback`/`_run_and_handle_error`.
  `LlmFlow::handle_before_model_callback` can short-circuit a model call
  entirely; `LlmFlow::handle_after_model_callback` can replace the
  model's response; `LlmFlow::handle_on_model_error_callback` can
  substitute a response instead of propagating a model-call error — all
  three mirror `BaseAgent::handle_before_agent_callback`'s short-circuit
  shape (C0038/C0045), wired around the existing `LlmFlow::call_model`
  in `LlmFlow::run_one_step` without touching that already-shipped
  method's own signature. Narrowed, disclosed at length in the module
  doc: the plugin-manager half of dispatch doesn't run (a pre-existing
  `adk-agents`↔`adk-models` crate-cycle block on `PluginManager`'s own
  side); the `google_search_agent` grounding-metadata workaround isn't
  ported (needs `agent.canonical_tools()`, blocked on C0092); and a
  callback's state-delta mutations aren't threaded back onto the
  resulting event, since each dispatch phase builds an independent,
  discarded `Context` rather than sharing the source's single
  `CallbackContext`. 8 new tests.
- New `crates/adk-flows/src/auth_preprocessor.rs`: the auth
  request/response processor (C0511-C0515, all `DONE`), ported from
  `auth/auth_preprocessor.py`. Handles the round-trip for a tool that
  needs end-user credentials: scans the last user-authored event for
  `adk_request_credential` responses (C0515), reconciles+pins each
  response back onto server-issued `auth_scheme`/`credential_key`
  values via `_merge_credential_oauth2_fields` (C0512, ignoring an
  unrequested response), stores each credential
  (`_store_auth_and_collect_resume_targets`, C0514), and re-executes
  the original tool call(s) that needed it. **Fully ported including
  tool re-execution, not stubbed**: `tools_dict: &ToolsDict` is a
  caller-supplied parameter (the same adaptation
  `request_confirmation.rs`'s C0172 batch already established for the
  structurally identical confirmation round-trip), so the terminal
  `handle_function_calls_async` step reuses the already-shipped
  `functions::execute_function_calls` directly — still not wired into
  `LlmFlow::preprocess`, pending the same `agent.canonical_tools()`
  blocker (C0092) `request_confirmation.rs` discloses. `TOOLSET_AUTH_CREDENTIAL_ID_PREFIX`
  (C0513) is kept as one shared constant, per that row's own note (the
  source duplicates it independently in `base_llm_flow.py`). Narrowed,
  disclosed: `_merge_credential_oauth2_fields`'s `token_endpoint_auth_method`
  merge always adopts the source's value (this port has no
  `model_fields_set`-equivalent "was this explicitly set" tracking).
  13 new tests, including an end-to-end test exercising the full
  store-credential → resume → re-execute path against a real tool.
- New `crates/adk-genai/src/serialization.rs`: telemetry JSON
  serialization helpers (C0680), ported from `telemetry/_serialization.py`
  (`safe_json_serialize`/`serialize_content`) plus
  `telemetry/_experimental_semconv.py`'s `_safe_json_serialize_no_whitespaces`.
  Both `safe_json_serialize` functions fall back to the literal
  `"<not serializable>"` string rather than propagating an error;
  `serialize_content` dispatches across a new `ContentUnion` enum
  (`Content`/`Text`/`List`/`Value`, the same `isinstance`-replacement
  adaptation `content_utils::ToUserContentInput` already established),
  matching the source's own type-inconsistent return (a structured value
  for `Content`, a JSON string for the catch-all branch). Narrowed,
  disclosed: the source's per-leaf `default`-hook recovery (a single
  non-serializable object nested anywhere inside an otherwise-normal
  structure falls back individually) has no port — a Rust `Serialize`
  bound is a compile-time whole-type guarantee, so this port can only
  fall back for the whole call; and `safe_json_serialize`/
  `_safe_json_serialize_no_whitespaces` collapse to byte-identical
  output, since `rusty_serde::json::to_string` has no
  whitespace-including/custom-separator mode. No new dependency —
  `adk-genai` depends only on `rusty_serde`. 8 new tests.
- `crates/adk-runners/src/runner.rs`: four node-path resolution helpers
  from `runners.py` (C0834/C0856/C0857/C0858), all built ahead of their
  own caller (`_run_node_async`, the workflow/node/task-delegation turn
  loop — still out of scope pending that engine's own wiring into
  `Runner`). `find_active_task_scope` (`_find_active_task_scope`) is a
  two-pass backward scan over session events locating a still-open task
  delegation's scope, closed only by a terminal `finish_task` function
  response (a validation-error response leaves it open) — its
  `FINISH_TASK_TOOL_NAME`/`FINISH_TASK_SUCCESS_RESULT`/
  `FINISH_TASK_ERROR_RESULT` constants and per-event predicate are
  duplicated locally from `adk-tools::finish_task_tool`, since
  `adk-tools` depends on `adk-runners` and a reverse dependency would be
  a crate cycle. `extract_resume_inputs`/`validate_new_message`/
  `resolve_invocation_id_from_fr` round out the resume-detection/
  invocation-id-resolution trio, ported exactly including the source's
  truthy-id filtering and dual error conditions (unmatched function
  response ids, function responses spanning multiple invocations) — the
  latter's error message sorts its offending ids for deterministic
  output, a disclosed cosmetic divergence from the source's
  hash-order-dependent Python `set` interpolation. 20 new tests.
- `crates/adk-agents/src/services.rs`/`session.rs`: completed the
  in-memory session-service's state-scoping architecture (C0208/C0212/
  C0214, C0204 partial), ported from `sessions/in_memory_session_service.py`
  and `sessions/session.py`. `InMemorySessionService` now carries shared
  `app_state`/`user_state` maps and routes `app:`/`user:`-prefixed state
  deltas into them (via the already-shipped C0209 `extract_state_delta`)
  from both `create_session`'s initial state and `append_event`'s
  `state_delta`; every read (`get_session`/`list_sessions`/
  `create_session`'s own return value) re-merges that shared state back
  onto the session, matching the source's `_merge_state` — verified with
  a test where a change made through one session becomes visible on a
  completely different, never-touched sibling session's next read.
  `SessionService::get_user_state` (C0214) is a new trait method
  defaulting to `GetUserStateError::NotSupported` (mirroring the source's
  `NotImplementedError` default), with `InMemorySessionService` overriding
  it with a real read of the shared user-state map. `Session::last_update_time`
  (C0204, partial) is set at creation and bumped on every `append_event`,
  so `list_sessions` (C0208) now sorts by `(last_update_time, user_id, id)`
  ascending — the source's real `ListSessionsResponse` ordering, not
  insertion order. `copy_session`'s `IN_MEMORY_SESSION_SERVICE_LIGHT_COPY`
  feature check (C0212) is genuinely read at every one of the source's
  three copy call sites, but disclosed as behaviorally inert in this
  port: this port's owned, non-aliased `Session` has no cheaper "shallow
  copy" tier the way Python's reference-typed `list`/`dict` do, so both
  strategies converge to the same `.clone()`. 8 new tests.
- New `crates/adk-agents/src/credential_manager.rs`: `CredentialManager`
  (C0517/C0519/C0520/C0521, C0518 partial) — the master credential-
  resolution orchestrator, ported from `auth/credential_manager.py`.
  `get_auth_credential`'s full 9-step state machine (custom-scheme
  dispatch, validate, fast-path for API_KEY/HTTP, load-existing,
  load-from-auth-response, client-credentials fallback or
  return-None-for-consent, exchange, refresh, save) ports in full,
  including a source quirk preserved faithfully rather than "fixed":
  step 4 sets `was_from_auth_response = true` unconditionally on entry
  to that branch, even when the lookup itself also finds nothing.
  `register_auth_provider`/`default_auth_provider_registry` (C0517) use
  this crate's own established `OnceLock<Mutex<_>>`-inside-an-accessor-fn
  singleton-registry convention. Narrowed, disclosed at length in the
  module doc: `CredentialManager::new` (C0518) registers no default
  exchangers/refreshers, since none of `OAuth2CredentialExchanger`/
  `ServiceAccountCredentialExchanger`/`OAuth2CredentialRefresher` exist
  in this port yet (still blocked on an authlib-equivalent HTTP
  exchange dependency, C0524/C0526); `_populate_auth_scheme` (OAuth2
  auto-discovery, C0520) is kept as a real, called step but always
  returns `false`, since this port's `ExtendedOAuth2` is a flattened
  struct outside the `AuthScheme` enum (a pre-existing tree-fusion gap
  from an earlier batch) and no `AuthScheme` value can ever be the
  `ExtendedOAuth2` the source's discovery check requires;
  `_rehydrate_custom_scheme`'s `__subclasses__()` reflection has no
  port, since this port's `AuthScheme::Custom` holds only a plain
  `CustomAuthScheme`, no subclass hierarchy to rehydrate into; and
  `request_credential`'s `hasattr` guard is moot given `CallbackContext`
  is already a unified alias for `Context` (C0048). 7 new tests.
- New `crates/adk-agents/src/workflow_graph_parser.rs`: P7 workflow/graph
  engine Chunk 6 — `parse_edge_items`/`Graph::from_edge_items` (C0327,
  narrowed), the chain-building convenience syntax on top of `Edge`/
  `Graph` (Chunk 2). Plain chains, fan-out tuples, routing-map dicts,
  and the consecutive-routing-map/empty-routing-map rejections all port
  in full. Narrowed, disclosed at length in the module doc: `NodeLike`
  keeps only `BaseNode`/`"START"` — no `BaseAgent`/`BaseTool`/raw-callable
  chain elements, since those need `build_node`/`is_node_like` (C0326),
  itself still blocked on the `adk-tools`/`adk-agents` crate cycle
  (C0355/C0356) `workflow_graph.rs`'s own module doc already disclosed.
  `_get_or_build_node` degenerates to identity dedup via the
  already-shipped `BaseNode::ptr_eq`, since `build_node` is a no-op
  passthrough for the only case this narrowed `NodeLike` can produce;
  the source's runtime `isinstance`/`is_node_like` validation checks are
  dropped since this port's typed enums make invalid values
  unrepresentable at parse time. 9 new tests.
- New `crates/adk-agents/src/workflow_{function_node,join_node}.rs`: P7
  workflow/graph engine Chunk 5 — `FunctionNode` (C0313/C0314/C0315, all
  DONE, heavily narrowed) and `JoinNode` (C0316, narrowed — `_ToolNode`
  half stays out of scope). `FunctionNode`'s `auth_config` gate (C0314)
  ports in full, reusing Chunk 2's HITL utilities verbatim; the source's
  reflective per-parameter binding/type-coercion machinery
  (`_bind_parameters`/`_coerce_param`/`get_type_hints`/`TypeAdapter`,
  C0313/C0315's bulk) has no Rust equivalent — Rust can't introspect a
  closure's parameter names or build a validator from a runtime type —
  so a wrapped `FunctionNodeBody::call` receives `(ctx, node_input)`
  directly, the same "caller supplies the resolved bits" adaptation
  already established elsewhere. `_to_event`'s normalization and
  per-yield `state_delta` attachment are already fully subsumed by
  existing infrastructure (`NodeYield`/`BaseNode::run`, C0295; `NodeRunner`'s
  trailing flush, C0312) — nothing left to add. `JoinNode` ports its
  `requires_all_predecessors=true` override and pass-through `_run_impl`
  in full, plus `_get_common_branch_prefix` (dead code even in the
  source, ported anyway per this migration's standing rule) as a thin
  wrapper around the already-shipped `BranchPath::common_prefix`.
  `NodeBehavior` grows a new `requires_all_predecessors` default trait
  method (additive, `false` by default) and `BaseNode` a delegating
  accessor. 15 new tests.
- New `crates/adk-agents/src/workflow_{rehydration_utils,replay_interceptor,
  replay_manager,replay_sequence_barrier}.rs`: P7 workflow/graph engine
  Chunk 4 — the replay/rehydration stack (C0320/C0321/C0322/C0323, all
  DONE), unblocked now that `NodeRunner` (Chunk 3) exists. Rebuilds
  per-node state from session events (`reconstruct_node_states`,
  `is_terminal_event`), decides whether a node should re-run, fast-forward,
  or stay waiting on resume (`check_interception`, `create_mock_context`),
  and unifies event indexing/sequence-barrier synchronization for
  deterministic replay ordering (`ReplayManager`, `ReplaySequenceBarrier`).
  Every piece is directly testable today against constructed `Context`/
  `Event` fixtures — the same "no caller yet" situation Chunk 1's pure-data
  primitives were in before Chunk 2/3 used them; the only real caller,
  `Workflow` (C0298-C0306), stays blocked on `Graph::from_edge_items`
  (C0327) and `DynamicNodeScheduler` (C0318/C0319, confirmed still blocked:
  its `__call__` needs a dynamic-dispatch seam over `BaseNode` that this
  port's concrete-struct `BaseNode` doesn't have). Narrowed, disclosed at
  length in the module docs: `process_rehydrated_output`/
  `validate_resume_response` can't run real schema validation (no
  TypeAdapter/pydantic — `BaseNode::output_schema` stays an opaque `Value`
  placeholder) and narrow to JSON-parse-or-fallback plus primitive-type
  coercion only; `check_interception` drops the same-turn `DynamicNodeRun`
  case (C0318/C0319 still blocked) and the `isinstance(node, Workflow)`
  check (`Workflow` isn't built, so nothing can ever be one). Also fixed,
  discovered while porting this batch: `workflow_hitl_utils
  ::create_request_input_event` was building its `response_schema`
  function-call arg under the wrong (camelCase) wire key — the source
  explicitly re-adds it as snake_case after its `by_alias=True` dump; and
  `adk-events::node_path_builder::NodePathBuilder::leaf_segment` was
  wrongly aliased to `node_name` (dropping `run_id`) instead of the
  source's own distinct, run_id-preserving `leaf_segment` property — fixed
  since this batch is `leaf_segment`'s first real caller. 35 new tests.
- New `crates/adk-agents/src/workflow_node_runner.rs`: P7 workflow/graph
  engine Chunk 3 — `NodeRunner` (C0310/C0311/C0312, all DONE), the
  per-node executor directly below the still-unbuilt `Workflow`
  orchestrator (C0298-C0306). Drives `BaseNode::run` with retry (per
  the node's `retry_config`, reusing Chunk 1's `should_retry_node`/
  `get_retry_delay`), a `rusty_tokio::time::timeout`-enforced timeout
  (→ `WorkflowNodeError::Timeout`), and result-tracking (output/route/
  interrupt-ids, "native event" author filtering, branch/author/
  invocation-id/output-for stamping). `crates/adk-agents/src/context.rs`
  grows a new `Context::for_node` constructor (additive — `node_path`/
  `run_id`/`attempt_count`/`resume_inputs`/`output_for_ancestors`/
  emitted-flags/error fields; `Context::new`'s existing call sites
  untouched) computing the node path via `NodePathBuilder`, the same
  singleton-safe pattern already established for `BaseNode::start`'s
  identity. `State` grows `take_delta` (returns and clears the pending
  state delta, `pub(crate)`) for the trailing flush step. Adaptations,
  disclosed at length in the module doc: `BaseNode::run` already
  returns one eagerly-collected `Vec<Event>` rather than streaming
  through a live queue (the same adaptation `AgentBehavior
  ::run_async_impl` established), so `NodeRunner::run` returns that
  same shape instead of enqueueing, and the source's per-event
  delta-flush collapses into one trailing flush since nothing here
  emits partial events yet; `WorkflowNodeError`/`ContextError` are this
  port's own sovereign `rusty_err::Error` (not `std::error::Error`), so
  they're bridged into `NodeRunError` via a local newtype/string
  conversion rather than a direct `impl std::error::Error` (which would
  conflict with `rusty_err`'s own blanket bridge the other way); the
  source's `NodeInterruptedError` catch and resume-inputs session-history
  rehydration aren't ported since nothing in this port can raise the
  former or build the latter yet (both need still-deferred dynamic
  dispatch / the replay stack, C0059/C0060, C0320-C0323). 7 new tests.
- New `crates/adk-agents/src/workflow_{base_node,graph,graph_validation,
  hitl_utils}.rs`: P7 workflow/graph engine Chunk 2 — `BaseNode`
  (C0294/C0295), `Edge`/`Graph`/routing (C0297, narrowed), `validate_graph`
  (C0328), and the HITL utilities `BaseNode::run` itself calls (C0329),
  all now DONE. `NodeBehavior` (trait, override point `run_impl`) +
  `BaseNode` (`Arc`-backed handle) mirror the `AgentBehavior`/`BaseAgent`
  split already established, including a `NoopNodeBehavior` mirroring
  `NoopBehavior` (used by the `START` sentinel and as a test double) and
  reusing `base_agent::AsAny` for the downcast escape hatch. `start()`
  caches a single `BaseNode` behind a `OnceLock` so it always returns the
  same singleton instance — load-bearing for `Graph::new`'s identity-based
  node dedup and `validate_graph`'s duplicate-name detection, both of
  which compare nodes by `Arc` pointer identity. `Graph::get_next_pending_nodes`
  ports the source's full route-matching logic (untagged edges always
  fire, `DEFAULT_ROUTE` only as a fallback, a dead-end warning). All 9 of
  `validate_graph`'s checks run in the source's exact order; the
  chat-agent-wiring check is ported as a real, called step but is a
  disclosed structural no-op — it's only reachable in the source because
  `LlmAgent` IS a `BaseNode` there (the already-known C0092 tree-fusion
  gap), and this port's `BaseNode`/`LlmAgent` are separate types. HITL
  utilities (`create_request_input_event`/`create_auth_request_event`/
  `process_auth_resume`/`has_auth_credential`) port in full, including the
  OAuth-state CSRF-style mismatch check. Narrowed, disclosed at length in
  `workflow_graph.rs`'s module doc: `Graph::from_edge_items`/
  `parse_edge_items` (C0327) and `build_node`/`is_node_like` (C0326) are
  deferred — they need `FunctionNode`/`_ToolNode`/`_ParallelWorker`
  (C0313/C0316/C0317, not yet built) and `BaseTool`, and `adk-tools`
  (home of `BaseTool`) already depends on `adk-agents`, so a dependency
  back would be the same crate-cycle shape already disclosed for
  C0355/C0356. 34 new tests across the four modules.
- New `crates/adk-agents/src/workflow_{node_status,node_state,trigger,
  errors,retry_config,retry_utils}.rs`: the first batch of the
  previously-entirely-unbuilt P7 workflow/graph engine (C0294-C0339,
  ~6,300 Python source lines) — the pure-data slice with zero
  dependency on `BaseAgent`/`BaseTool`/`Context`/`Event` (C0307/C0308/
  C0309/C0324, all now DONE). `NodeStatus` (7 variants), `NodeState`
  (8 fields, including the `attempt_count`/`run_counter`
  exclude-at-default serialization behavior), `Trigger` (4 fields),
  `NodeInterruptedError`/`WorkflowNodeError` (`NodeTimeoutError`/
  `DynamicNodeFailError`), `RetryConfig`, and `should_retry_node`/
  `get_retry_delay` (16 tests, 11 ported directly from the source's own
  test suite, including a 2000-sample statistical cap-before-jitter
  test). Landed as flat `workflow_*.rs` modules inside `adk-agents`
  rather than a new crate — `adk-agents` already discloses needing
  `workflow::BaseNode` once it exists (`base_agent.rs`/`context.rs`/
  `app.rs`), and the eventual `Workflow` orchestrator needs
  `agents.context.Context` directly, so two separate crates would hit
  a mutual-dependency cycle Cargo can't express, the same shape already
  disclosed for C0355/C0356. Adaptations, disclosed: `NodeStatus`
  serializes as its variant name rather than the source's underlying
  int value (same cosmetic choice `EvalStatus` already established);
  `RetryConfig.exceptions` accepts `Option<Vec<String>>` directly
  rather than porting the source's class-object-or-string duality
  (nothing in Rust to normalize *from*); `should_retry_node` takes the
  failed exception's type name as a caller-supplied `&str` (Rust has
  no generic way to recover a short type name from an arbitrary `&dyn
  std::error::Error`); jitter uses `adk_platform::random`'s existing
  seeded RNG seam, a different (and already-disclosed) PRNG than
  Python's. Not ported: `BaseNode`/`Graph`/`Edge` (C0294/C0295/C0297)
  and everything downstream — separate, larger follow-up batches,
  several already sized in the scoping notes for this area.
- `crates/adk-eval/src/user_simulator.rs`: `UserSimulatorProvider`
  (C0627, partial) — routes a per-`EvalCase` `UserSimulator`: a case
  carrying a static `conversation` always gets a `StaticUserSimulator`
  (config-agnostic); a case carrying a `conversation_scenario` dispatches
  through `create_user_simulator` (C0626) keyed by the configured `type`
  discriminator string, defaulting to `"llm_backed"` (matching the
  source's `_LEGACY_DEFAULT_CONFIG_TYPE`) when no config is supplied.
  Adaptation, disclosed: stores the config as an opaque `Value` and
  reads its `"type"` field rather than a typed instance, matching
  `create_user_simulator`'s own registry-by-discriminator-string shape.
  Not ported, disclosed: the audio-decorator composition
  (`_LlmAudioUserSimulator` wrapping) — neither `LlmBackedUserSimulator`
  (C0628) nor `_LlmAudioUserSimulator`/`LlmAudioUserSimulatorConfig`
  (C0630) exist in this port yet, so `create_user_simulator`'s existing
  "no simulator registered" error is the correct behavior until those
  land. 6 new tests.
### Fixed
- `capability-manifest.md`, C0200's Evidence column: corrected a
  Status/Evidence mismatch (Status stayed `REQUIRED` but the Evidence
  text started with a stale `"DONE:"` prefix instead of the `"Partial:"`
  convention every sibling row uses for a row with real remaining scope
  — `BasePlanner`'s hooks are ported, but wiring into `_nl_planning`
  (C0176/C0179) is still open).
### Added
- New `crates/adk-flows/src/code_execution.rs`: the `_code_execution`
  request/response processors' `BuiltInCodeExecutor` branches
  (C0177/C0180, partial). `apply_code_execution_request` delegates to
  `BuiltInCodeExecutor::process_llm_request` (Gemini tool marker or a
  non-Gemini error) then always converts trailing executable-code/
  execution-result parts to text using the executor's own delimiters,
  regardless of executor kind — matching the source's unconditional pass.
  `apply_code_execution_response` skips streaming partials, then for a
  `BuiltInCodeExecutor` saves every generated image part to the artifact
  service (`display_name`-or-UTC-timestamp filename fallback) and always
  yields exactly one event carrying the resulting `artifact_delta`.
  Adaptation, disclosed: both take `code_executor: &dyn BaseCodeExecutor`
  as a caller-supplied parameter (same C0092 "caller supplies the
  resolved bits" precedent as `agent_transfer.rs`/`request_confirmation.rs`
  — `LlmAgent.code_executor` stays an opaque `Value`), and aren't wired
  into `LlmFlow::preprocess`/`postprocess` yet — left ready for a future
  C0092-unblocking batch. Adds `BaseCodeExecutor: AsAny` (blanket-
  implemented, purely additive, same pattern as `adk-agents`/
  `adk-models`'s own `AsAny`) so a resolved executor can be downcast onto
  `BuiltInCodeExecutor`. Not ported this batch, disclosed: the general
  (non-built-in) executor's `optimize_data_file` data-file extraction/
  `explore_df`-injection request path, and its extract-execute-emit-
  events response path with per-invocation error-retry tracking — real
  additional surface area, not a blocker, left for a follow-up batch.
  9 new tests.
- `crates/adk-flows/src/llm_flow.rs`: wired transfer-to-agent recursion
  into `LlmFlow::run_one_step` (rest of C0158, now DONE) — when a
  function-response event's `transfer_to_agent` action is set, resolves
  the target via `agent_transfer::get_agent_to_run` (C0159; uses the
  current agent's own `disallow_transfer_to_peers`, so this doesn't touch
  the C0092 tree-fusion gap) and recursively calls
  `agent_to_run.run_async(ctx)`, extending the step with the nested run's
  events — matching the source's `_postprocess_handle_function_calls_
  async` exactly, including running unconditionally after auth/
  confirmation/`set_model_response` synthesis. Confirmed against the
  Python source that no branch extension happens for this path (only
  `ctx.agent` swaps, inside the target's own `run_async`) — a genuinely
  new finding this batch, since `ParallelAgent`/`SequentialAgent`'s own
  branch-extension convention is a caller-side choice, not part of
  `run_async`'s contract. Adds two additive `LlmFlowError` variants
  (`GetAgentToRun`, `NestedAgentRun`); `run_one_step`'s own signature is
  unchanged. No new dependency, no breaking change. 2 new tests: a
  two-agent tree where the root transfers to a child agent that produces
  the final response; an error case for an unknown transfer target.
  Also fixed a cluster of stale "needs `BaseTool`/`App` (Phase N, not
  built)" claims this and the prior two batches' own existence falsify:
  `app.rs`/`lib.rs` (adk-agents) and manifest row C0280 no longer claim
  `App` isn't wired into `Runner` (it is, via `Runner::from_app`);
  `agent_transfer.rs` no longer claims `TransferToAgentTool` needs
  `BaseTool` (it's built, C0436); `functions_utils.rs`'s
  `get_long_running_function_calls` doc now correctly cites C0092 (no
  automatic `tools_dict` resolution) rather than a nonexistent `BaseTool`
  gap; `functions.rs` no longer claims auth-event synthesis isn't wired
  into the turn loop (it is, C0158); `llm_request.rs`/
  `generate_content_request.rs`/`gemini.rs`/`debug_log.rs` no longer
  claim `append_tools`/`BaseTool` (C0116) don't exist — they do; the real
  remaining gap is that nothing downstream reads `config.tools` back out
  of its opaque placeholder into a typed REST body/log; `services.rs`'s
  `MemoryEntry` doc no longer claims the backing memory service is
  unbuilt (`InMemoryMemoryService` already produces real values;
  `VertexAiMemoryBankService` is the genuinely out-of-scope remainder);
  `llm_agent.rs`'s `ToolUnion` doc no longer claims `BaseTool`/
  `BaseToolset` don't exist (they do; the real gap is still C0092).
- `crates/adk-flows/src/llm_flow.rs`: wired auth-request/tool-confirmation-
  request/`set_model_response`-final-event synthesis into `LlmFlow::run_one_step`
  (C0158), plus the `end_invocation` short-circuit (rest of C0149) — a
  direct follow-up to the turn-loop batch below. Mirrors
  `_postprocess_handle_function_calls_async`'s exact yield order after a
  function call executes: auth-request event (`generate_auth_event`,
  `functions_utils.rs`, C0504 — setting `ctx.end_invocation = true` when
  one is yielded), tool-confirmation-request event
  (`generate_request_confirmation_event`), the function-response event
  itself, then (conditionally) a synthesized final event when the response
  carries a validated `set_model_response` result
  (`create_final_model_response_event`/`get_structured_model_response`,
  `output_schema.rs`, C0178). `run_one_step` now returns an empty step
  immediately once `ctx.end_invocation` is set, since a function-response
  event isn't itself a final response and `run_async`'s outer loop would
  otherwise issue one more invalid model call after an auth request. Not
  ported: recursive re-run of a transferred sub-agent (`transfer_to_agent`
  action handling) — `get_agent_to_run` (C0159, DONE) already provides the
  resolution half. No new dependency, no breaking change to `LlmFlow`'s
  public surface. 3 new tests.
- `crates/adk-flows/src/llm_flow.rs`: the multi-step tool-calling turn
  loop (C0148/C0149/C0151/C0152) — `LlmFlow::run_async`, a new outer loop
  mirroring the source's `run_async`/`_run_one_step_async` structure that
  repeats `run_one_step` (preprocess → call model → postprocess →
  execute any function calls the response carries) until a step's last
  event is a final response. `LlmFlow` gains a `pub tools_dict: ToolsDict`
  field (default empty) plus a `with_tools_dict` builder — the resolved
  `name -> BaseTool` map the loop dispatches function calls against,
  supplied by the caller rather than auto-resolved from `agent.tools`
  (still blocked on C0092's tree-fusion gap), the same "caller supplies
  the resolved bits" adaptation already established by
  `request_confirmation.rs`/`agent_transfer.rs`. `run_one_step` now
  executes a step's function calls via `execute_function_calls` and
  appends the resulting function-response event to the step.
  `AgentBehavior::run_async_impl` now calls `run_async` instead of
  `run_one_step` once. Disclosed adaptation: since this port materializes
  each step's events as a `Vec<Event>` rather than yielding them
  cooperatively, `run_async` explicitly appends each step's events onto
  `ctx.session` between iterations (via `ctx.session_service`) so the
  next step's `preprocess` sees them — this means every event now flows
  through `session_service.append_event` twice (once from the loop,
  once from `Runner`'s own top-level append), safe only because
  `InMemorySessionService` already deduplicates a redelivered event by
  id+equality. Not ported this batch, disclosed in the file's own module
  doc: auth/tool-confirmation-request event synthesis, transfer-to-agent
  recursion, and `set_model_response` final-event synthesis. 4 new tests.
  Bundled stale-blocker corrections (the loop's own existence falsifies
  these): `llm_flow.rs`'s module doc no longer claims the turn loop
  "has nothing to loop on yet" without `BaseTool`; `functions.rs`'s
  module doc no longer claims auth-event synthesis "needs `AuthConfig`,
  Phase 9, not built" (stale since C0504 landed; the real gap is that
  `run_one_step` doesn't call it yet); `basic.rs`'s task-mode
  `output_schema` comment no longer claims it needs `BaseTool` (stale
  since `FinishTaskTool` shipped; the real blocker is C0092); manifest
  rows C0148/C0149/C0150/C0151/C0152/C0156 updated with real, tested
  evidence in place of the stale "needs `BaseTool` (Phase 8)" reasoning;
  manifest rows C0842/C0843 updated from "N/A: `App` doesn't exist" to
  cite `App`/`Runner::from_app`, both DONE, matching sibling rows
  C0841/C0846/C0848/C0849/C0850's already-corrected treatment.
- New `crates/adk-tools/src/tool_configs.rs`: `BaseToolConfig`/
  `ToolArgsConfig`/`ToolConfig` (C0417) — the declarative YAML/dict
  tool-reference data shape (a tool `name` plus optional free-form
  `args`), gated behind `FeatureName::ToolConfig` via
  `adk_features::feature_decorator::check_feature_enabled` (C0647),
  matching the source's `@experimental` decorator. Doesn't unblock
  `BaseTool.from_config`'s actual dynamic-dispatch resolution (5
  reference kinds, needs Python's `importlib`) — `base_tool.rs`,
  `base_toolset.rs`, and `example_tool.rs` updated to cite that as
  disclosed-inapplicable (same precedent as C0939's `_lazy.accessors`)
  rather than "C0417 not built." 9 new tests.
- New `crates/adk-tools/src/finish_task_tool.rs`: `FinishTaskTool`
  (C0099) — signals `LlmAgent` task completion, wraps a non-object
  `output_schema` under a `result` key (hoisting `$defs` to the wrapped
  schema's root so `$ref` pointers stay valid), and appends its "don't
  call this prematurely" instruction via `process_llm_request`. Plus its
  two pure helpers, `get_output_wrapper_key` and
  `is_finish_task_terminal_fr`. Disclosed narrowings: takes
  `output_schema: Option<Value>` directly rather than a whole `LlmAgent`
  (the source's own `self._task_agent_name` is set but never read
  anywhere in the source tree); `run_async` never validates `args`
  against the schema and always succeeds, since this port has no
  Pydantic-equivalent validator (same limitation `set_model_response_tool.rs`,
  C0437, already discloses). Scoped to the tool itself — not the
  task-mode turn loop that would construct and wire it into a running
  agent (C0333/C0834/C0887), which needs infrastructure this port
  doesn't have yet. 13 new tests.
- New `crates/adk-flows/src/compaction_request_processor.rs`: the
  `compaction` request processor (C0173), wired into
  `LlmFlow::preprocess` right after `instructions`/`identity` and before
  `interactions_processor`/`contents`, matching the source's own
  `REQUEST_PROCESSORS` ordering. Runs token-threshold event compaction
  before contents are assembled, appends the resulting event via
  `ctx.session_service`, and marks `ctx.token_compaction_checked` — which
  `Runner`'s post-invocation sliding-window trigger (C0872) already reads
  back as `skip_token_compaction`, so flow-level compaction isn't
  immediately redone at the end of the same invocation.
  `apps_compaction.rs::run_compaction_for_token_threshold` split into
  itself (unchanged, the `App`-level `""`/`None` wrapper) plus a new
  `run_compaction_for_token_threshold_config` taking `agent_name`/
  `current_branch` explicitly, since the new processor is a second real
  caller with real values for both. `LlmFlow::preprocess` widened from
  `&InvocationContext` to `&mut InvocationContext` to support it. 5 new
  tests.
### Fixed
- `crates/adk-tools/src/agent_tool.rs`: `AgentTool`'s nested `Runner` now
  gets an `InMemoryMemoryService` (matching the source's own
  `memory_service=InMemoryMemoryService()`, unconditional and never
  parent-forwarded, unlike the artifact service). Corrects a stale
  blocker claim (same pattern as C0172/C0178/C0196/C0125): the module
  doc said this was "Phase 6, not built" — false, it shipped long ago
  and `adk-tools` already depends on `adk-agents`. The manifest row's
  own evidence also repeated a second stale claim (that
  `ForwardingArtifactService` wasn't ported either) well after it had
  actually landed — corrected in place. 1 new test.
- `crates/adk-genai/src/content.rs`: new `PartialArg` type and a
  `partial_args: Option<Vec<PartialArg>>` field on `FunctionCall`, needed
  for progressive SSE streaming's incrementally-streamed function-call
  arguments.
- `crates/adk-models/src/streaming_utils.rs`: `StreamingResponseAggregator`
  now ports the source's progressive SSE streaming mode
  (`FeatureName::ProgressiveSseStreaming`, closing out the rest of C0125)
  alongside the already-shipped legacy mode — ordered part accumulation
  (only merging consecutive text parts of the same thought/non-thought
  type), and JSONPath-addressed incremental function-call argument
  assembly (`$.location`, `$.units.precision`, ...) via a new
  `value_from_partial_arg`/`set_value_by_json_path` pair. Corrects a stale
  blocker claim (same pattern as C0172/C0178/C0196): the module doc said
  this needed "typed function-call/tool machinery (C0116, Phase 8's
  `BaseTool`)" — false, progressive mode never touches `BaseTool` at all.
  `adk-models` gained direct `adk-features`/`adk-events` dependencies (both
  already transitive via `adk-agents`, so no new crate cycle) for the
  feature check and for reproducing `generate_client_function_call_id`
  locally (its real home, `adk-flows`, depends on `adk-models`, not the
  reverse — disclosed duplication, not a dependency). 10 new
  progressive-mode tests; all 11 pre-existing legacy-mode tests now
  explicitly force the flag off, since `ProgressiveSseStreaming` defaults
  on and would otherwise change their behavior.
### Fixed
- `crates/adk-models/src/gemini.rs`: one pre-existing streaming test
  (`generate_content_stream_yields_partials_then_a_final_aggregate`)
  hardcoded the legacy aggregation mode's response shape; now explicitly
  forces `ProgressiveSseStreaming` off, since the flag defaults on.
- `crates/adk-flows/src/functions_utils.rs`: closed out the remaining gap
  in C0196 — `build_auth_request_event`, `generate_auth_event`, and
  `generate_request_confirmation_event` are now real, tested code.
  Corrects a stale blocker claim (same pattern as C0172/C0178): the
  module doc said these needed `AuthConfig` (Phase 9, "doesn't exist in
  this port yet"), but `AuthConfig` (C0504), `AuthToolArguments`, and
  `ToolConfirmation` all landed long ago and are already `adk-flows`
  dependencies. `EventActions.requested_auth_configs`/
  `requested_tool_confirmations` stay `Value`-typed for good (`adk-events`
  sits beneath `adk-agents`/`adk-tools`, so a real crate cycle — not
  staleness — blocks typing them directly); the two `generate_*`
  functions round-trip each entry through `rusty_serde::json::from_value`
  and silently drop malformed ones, and both new builders sort by key
  instead of relying on `HashMap`'s absent insertion order — both
  disclosed narrowings. 11 new tests.
- `crates/adk-events/src/event_actions.rs`: corrected the module doc's
  claim that `AuthConfig`/`ToolConfirmation` "aren't built yet" — both
  exist now; the `Value`-typed fields are a permanent adaptation forced
  by the crate dependency direction, not a placeholder waiting on a
  future phase.
- `crates/adk-tools/src/tool_context.rs`: closed out C0415 — added the
  `AuthCredential`/`AuthHandler`/`AuthConfig` back-compat re-exports the
  module doc had marked as blocked on "Phase 9, which doesn't exist."
  All three types exist in `adk-agents`, already a dependency of
  `adk-tools`; ported as plain `pub use` re-exports (this port's
  "static instead of lazy" substitute for the source's
  `__getattr__`-based lazy import). 3 new tests.
- `crates/adk-agents/src/auth_tool.rs`: `AuthToolArguments` now derives
  `Serialize`/`Deserialize`, needed for `build_auth_request_event` to
  round-trip it through `Value`. Matches `AuthConfig`'s own already-
  shipped snake_case wire shape for internal consistency, rather than
  introducing a fresh camelCase/snake_case mismatch inside one nested
  value.
- `crates/adk-flows/src/request_confirmation.rs`: removed a duplicate
  `REQUEST_CONFIRMATION_FUNCTION_CALL_NAME` constant introduced in the
  C0172 batch — `contents.rs` already defined the same public constant
  with the same value; now imported from there instead.
- `crates/adk-flows/src/request_confirmation.rs`: closed most of the
  remaining gap in C0172 (`request_confirmation`) — `resolve_confirmation_targets`
  (the full 3-check validation: registered in session history, author
  match, tool-registered + requires-confirmation-or-dynamically-requested,
  name/args match) and `process_request_confirmations` (the processor's
  Steps 1-4: parse the last user event's approvals, dedup already-consumed
  confirmations, resolve targets, re-execute via `functions::execute_function_calls`)
  are now real, tested code. Corrects a stale blocker claim (same pattern
  as C0178): `BaseTool`/`ToolConfirmation`/`ToolContext` all exist in
  `adk-tools`, which `adk-flows` already depends on. Still `Partial:` —
  not wired into `LlmFlow::preprocess`, since `tools_dict` auto-resolution
  needs `LlmAgent.canonical_tools()` (C0092, still blocked). 16 new
  tests.
- `crates/adk-flows/src/output_schema.rs`/`llm_flow.rs`: closed out C0178
  (`_output_schema_processor`) — `apply_output_schema_processor` now
  actually injects the synthetic `set_model_response` tool + its
  instruction into the request, and is wired directly into
  `LlmFlow::preprocess` (near the end, matching the source's own
  `REQUEST_PROCESSORS` ordering after `contents`/`context_cache`). Two
  stale blocker claims are corrected: `LlmRequest::append_tools` (a
  *method* on `LlmRequest` inside `adk-models`) really is blocked by the
  `adk-models`/`adk-tools` crate cycle, but the free-function
  `append_tools`/`merge_declarations` (C0116) lives in `adk-tools`,
  which `adk-flows` already depends on; and "needs `InvocationContext.agent`
  to resolve a concrete `LlmAgent`" was never actually true — every
  sibling processor (`basic`/`identity`/`instructions`) already solves
  this by taking `&LlmAgent` as a plain free-function parameter. 5 new
  tests.
### Added
- New `crates/adk-agents/tests/artifact_service_cross_backend.rs` (C0278
  REQUIRED+Partial) — a cross-backend `ArtifactService` parity suite:
  10 shared behaviors (load-empty, save/load/delete, key/version
  listing, nested-artifact-reference isolation, user-prefix scoping,
  namespaced `user_id`), each run once against `InMemoryArtifactService`
  and once against `FileArtifactService` through the same `&dyn
  ArtifactService` trait object — proving the two backends genuinely
  behave identically, not just that each independently passes its own
  unit tests. `GcsArtifactService` (the source's third parametrized
  backend) stays excluded, still blocked on the P5/P6 GCP-SDK pocket.
  20 new tests.
### Fixed
- `crates/adk-agents/src/context.rs`: `Context::request_credential` now
  routes through `AuthHandler::generate_auth_request` instead of
  serializing the caller's raw `AuthConfig` verbatim — for an
  OAuth2/OIDC scheme this validates the raw credential and substitutes
  a freshly generated `exchanged_auth_credential`, matching the
  source's own `AuthHandler(auth_config).generate_auth_request()` call.
  Also added the missing `Context::get_auth_response` (the source's
  `AuthHandler(auth_config).get_auth_response(self.state())` two-line
  wrapper, absent from this port entirely until now). C0062 was marked
  `DONE` before `auth_handler.rs` (C0506-C0508) existed and never
  revisited once it landed — corrected its evidence to cite the real
  coverage. 4 new tests.
### Added
- New `crates/adk-agents/src/auth_handler.rs`: `AuthHandler` (C0507 DONE,
  C0506/C0508 REQUIRED+Partial) — `get_auth_response`/
  `parse_and_store_auth_response`/`_build_credential_from_string`
  (C0507)/`generate_auth_request`/`generate_auth_uri`, plus a new pure
  standalone `resolve_authorization_endpoint_and_scopes` helper (built
  ahead of its still-blocked authlib-only caller). `get_auth_response`
  and `_build_credential_from_string` ported in full; `generate_auth_uri`
  always takes the source's own `not AUTHLIB_AVAILABLE` fallback (this
  port has no authlib-equivalent OAuth2 client, the same missing-crate
  gap C0530/C0524/C0526 are blocked on — and genuinely this port's
  entire reachable behavior today); `parse_and_store_auth_response`'s
  OAuth2/OIDC token-exchange branch stays unported (`exchange_auth_token`
  needs a concrete `OAuth2CredentialExchanger`, C0524, blocked on the
  same missing crate). 22 new tests.
- New `crates/adk-tools/src/base_retrieval_tool.rs`:
  `retrieval_tool_declaration`/`BaseRetrievalTool` (C0482 DONE) — the
  shared `query`-string `FunctionDeclaration` every retrieval tool
  exposes, gated on `FeatureName::JsonSchemaForFuncDecl` exactly like
  the source. The source's abstract class becomes a free function (same
  shape as `request_input_tool.rs`'s `parameters_schema()`) plus a thin
  `BaseRetrievalTool: BaseTool` marker supertrait, since Rust can't have
  a subtrait override an inherited default trait method
  (`BaseTool::get_declaration`) for just its own implementers. Also
  gives a future `cli.agent_graph`
  `isinstance(tool_or_agent, BaseRetrievalTool)` port (C0281, deferred)
  a real marker to check against. 5 new tests.
- New `crates/adk-agents/src/reflect_retry_utils.rs`:
  `TrackingScope`/`resolve_scope_key`/`ScopedFailureTracker` (C0370
  DONE) — the invocation-vs-global failure-tracking layer behind the
  still-unbuilt reflect-and-retry plugins (C0368/C0369). `asyncio.Lock`
  ported as `std::sync::Mutex` (a synchronous-critical-section lock, no
  `await` inside — the same shape `InMemoryMemoryService`/
  `InMemoryCredentialService` already use); `increment`/`reset` stay
  `async fn` to mirror the source's own `async def` signatures. Built
  ahead of its callers, same as recent precedent
  (`get_function_responses_from_content`,
  `session_util.rs`/`artifact_util.rs`) — C0368/C0369 both route through
  the still-blocked `before_model_callback`/`before_tool_callback`
  hooks (C0355/C0356), but this module depends on neither. 8 new tests.
- New `crates/adk-agents/src/oauth2_discovery.rs`:
  `AuthorizationServerMetadata` (RFC8414) and `ProtectedResourceMetadata`
  (RFC9728), C0533's two data models. Field names stay literal
  snake_case (both RFCs specify snake_case wire fields, unlike this
  crate's Google-genai-facing camelCase types); `@experimental`'s
  warn-on-construction ported via an explicit `::new()` calling
  `warn_experimental`, the same `ResumabilityConfig::new` precedent.
  4 new tests.
- `adk-agents::oauth2_discovery::OAuth2DiscoveryManager` (C0534/C0535
  DONE) — `discover_auth_server_metadata`/`discover_resource_metadata`,
  trying the RFC-specified `.well-known` candidate endpoints in order
  and validating the returned `issuer`/`resource` field against the
  requested URL, the explicit defense against OAuth "IdP mix-up"
  attacks; swallows per-candidate HTTP/parse errors and falls through to
  the next candidate. `reqwest::blocking` does the fetching inside
  `rusty_tokio::spawn_blocking`, the same bridging pattern
  `load_web_page.rs` established — a new usage site of the
  already-adopted `reqwest` dependency added to `adk-agents` for this
  (see `crates/adk-agents/Cargo.toml`). Verified against a new local
  multi-connection mock HTTP server (`spawn_mock_server`), since the
  candidate-fallback logic needs several distinct responses from the
  same mock host in one test. 7 new tests.
- `adk-runners::runner::get_function_responses_from_content` (C0835
  DONE) — extracts every `FunctionResponse` from a `Content`'s parts,
  `[]` for `None`/no-parts, reusing the already-real
  `Content::get_function_responses`. Built ahead of its own caller
  (`_resolve_invocation_id`, C0855 — needs resumability wiring `Runner`
  doesn't have yet). 3 new tests.
- `adk-agents::services::{GetSessionConfig, SessionService::get_session_with_config}`
  (C0207 DONE) — `num_recent_events`/`after_timestamp` session-history
  bounding, ported as a new default trait method on `SessionService`
  (additive; doesn't touch `get_session`'s already-shipped signature or
  any of its ~26 existing call sites) that defers to `get_session` and
  applies the trimming generically in one shared place, so every
  current implementer (`InMemorySessionService`/`NoopSessionService`/a
  test-only `FakeSessionService`) gets correct, identical behavior for
  free. Replicates the source's own Python-truthiness quirk where
  `after_timestamp: Some(0.0)` is treated as unset, not "keep nothing."
  Not yet wired: `RunConfig.get_session_config` is still an opaque
  `Value` placeholder (C0875), so nothing on the `Runner` side threads
  a real `GetSessionConfig` through this new method yet — updated the
  citation text on C0873/C0891/C0914 and two code comments in
  `runner.rs` accordingly, rather than leaving them saying
  `GetSessionConfig` doesn't exist at all. 7 new tests.
- `adk-flows::llm_flow::{apply_no_content_error, should_skip_empty_response}`
  (C0156 partial DONE) — the two remaining `_postprocess_async` guard
  clauses: converting a non-partial, non-streaming STOP-with-no-content
  response into a `MODEL_RETURNED_NO_CONTENT` error (excluded for SSE
  streaming, where a terminal finish-only chunk legitimately follows
  content already streamed earlier), and skipping the model-response
  event entirely when a response carries no content, no error, no
  interruption, and no grounding metadata. Applied in the source's own
  sequential order — a response the first check rewrites into an error
  is never skipped by the second. 14 new tests.
- Manifest evidence closure for `Gemini::generate_content_async`'s
  dual-mode streaming contract (C0102 DONE) — already correctly
  implemented and, it turns out, already partly tested directly via
  the `BaseLlm` trait method; added the one missing case (`stream=false`
  success, "exactly one response, not partial"). 1 new test.
### Added
- `adk-flows::functions_media::{as_function_response_part, extract_media_from_entry,
  extract_multimodal_parts}` (C0195 DONE) — `functions.py`'s
  multimodal tool-result extraction, pulling media (an image, audio
  clip, or document) out of a tool's returned dict/list/tuple (bounded
  to one level of container nesting) before the rest of the result is
  coerced to a plain JSON dict, wired into
  `adk-flows::functions::build_function_response_content`. New schema:
  `adk_genai::content::FunctionResponse` gained a `parts:
  Option<Vec<FunctionResponsePart>>` field (additive; every existing
  construction site updated to use `..Default::default()`);
  `FunctionResponsePart` is a new type reusing `MediaBlobStub` for the
  same `mime_type`-real/rest-opaque shape `Part::inline_data`/
  `Part::file_data` already use. Disclosed adaptation: the source's
  `isinstance(value, types.Part)` check has no direct equivalent since
  this port's tools only ever return an already-JSON-shaped `Value` —
  the check here is structural instead (round-trips through
  `rusty_serde::json::from_value::<Part>` and carries a populated
  `inline_data`/`file_data`), looser than the source's identity check
  but the only representation available. 13 new tests.
### Added
- Manifest evidence closures for two `runners.py` module-level helpers
  already fully implemented and exercised, but previously undocumented
  as such: **C0833** (`_notify_run_error`, best-effort
  `on_run_error_callback` notification) — already wired at
  `Runner::run_async_with_config`'s one unhandled-agent-error path via
  `PluginManager::run_on_run_error_callback`; the source's
  "logs+suppresses callback exceptions" behavior is N/A by construction
  here, since this port's plugin callback trait methods return `()`, not
  `Result` — nothing to catch or suppress. **C0836**
  (`_apply_run_config_custom_metadata`) — already implemented as
  `apply_run_config_custom_metadata`; added 3 new direct unit tests
  (previously only exercised indirectly through `run_async` tests).
### Added
- `adk-runners::Runner::run` (C0877/C0878/C0879/C0880 DONE) — a
  synchronous wrapper around `run_async_with_config`, ported from
  `runners.py`'s local-testing/convenience-only `Runner.run(...)`.
  `Runner` now derives `Clone` (every field is a cheap `Arc`/value clone)
  so `run` can move an owned copy onto a background OS thread via
  `adk_platform::thread::create_thread` (C0005, new `adk-runners` →
  `adk-platform` dependency) running its own `rusty_tokio::Runtime` —
  verified callable from inside an already-running `rusty_tokio` runtime
  without deadlocking. Disclosed narrowing: the source bridges events one
  at a time through a blocking `queue.Queue`; this port's
  `run_async_with_config` already collapses to a single batched
  `Result<Vec<Event>, RunnerError>` (an already-established narrowing),
  so `run` collapses to one background computation whose whole result
  becomes available at once. A panic on the background thread (this
  port's only abnormal-termination case — no `Exception`/`BaseException`/
  `SystemExit`/`CancelledError` hierarchy to distinguish) is caught by
  `join()` and re-wrapped into `RunnerError::AgentRun` naming the panic
  payload, rather than re-panicking the calling thread. C0880 also closes
  a small pre-existing gap: `run_async_with_config` didn't accept
  `state_delta` at all before this batch — added as a new
  `Option<HashMap<String, Value>>` parameter, applied onto the appended
  user event's `actions.state_delta` when non-empty (mirroring
  `_append_new_message_to_session`'s own truthiness check), forwarded
  straight through from `run`. Also fixed a stale evidence note on C0886
  (compaction was closed in an earlier PR this session but its "not
  ported" wording never got updated) and a stale module-doc line
  (`run` was still listed under "not ported this batch"). 5 new tests.
### Added
- `adk-runners::Runner::run_async_with_config` (C0918 N/A-by-existing-design,
  C0919 partial DONE) — now also patches `context_cache_config`/
  `resumability_config`/`events_compaction_config` onto the
  `InvocationContext` it builds; previously these three configs lived on
  `Runner` but were never wired onto the context the agent/callbacks
  actually ran with. `adk-agents::invocation_context::InvocationContextBuilder`
  is confirmed to already play the role of the source's
  `_create_invocation_context` factory (C0918) — no Rust subclassing to
  override, so nothing further to port. Not ported: the `support_cfc`
  (Compositional Function Calling) branch — `LlmAgent.code_executor` is
  still an opaque `Value` placeholder (C0088), the same
  architecture-investment blocker as C0092/C0429. 2 new tests, using a new
  `ConfigCapturingBehavior` test double.
### Added
- `adk-runners::Runner::{run_debug, run_debug_with_config}` (C0911/C0912/C0913
  DONE, C0914 N/A) — a debugging/experimentation convenience ported from
  `runners.py`'s `run_debug`, plus a new `DebugMessages` type normalizing a
  bare string or a list of strings. `run_debug` matches the source's
  documented defaults (`user_id="debug_user_id"`, `session_id=
  "debug_session_id"`, reusing them across calls continues the same
  session); `run_debug_with_config` is the full-control form, the same
  split already established by `run_async`/`run_async_with_config` since
  Rust has no keyword-argument defaults. Session lookup is unconditional
  get-or-create, bypassing `Runner::with_auto_create_session` entirely
  (C0912); drives `run_async_with_config` once per message and calls
  `adk_events::debug_output::print_event` per event unless `quiet`,
  returning the full flat event list across every message, not just the
  last (C0913). Disclosed narrowing: the source's `logger.info(...)` calls
  around session creation and each message have no destination in this
  port (no logging framework adopted anywhere in this crate); C0914's
  `run_config.get_session_config` forwarding is N/A — this port's
  `SessionService::get_session` has no config parameter to forward one to.
  6 new tests. Also corrected two stale manifest/doc notes touched by this
  batch: C0924's evidence understated `Runner::close` (it already closes
  both the plugin manager and the session service, not just the latter),
  and C0925 (`__aenter__`/`__aexit__`) is now disclosed N/A — Rust has no
  async-context-manager protocol and `Drop` can't run an async close.
### Added
- `adk-flows::apps_compaction::{run_compaction_for_token_threshold,
  run_compaction_for_sliding_window}` (C0293 DONE) — the two
  post-invocation compaction trigger entrypoints, ported from
  `apps/compaction.py`. `Runner` (C0871/C0872 DONE) gets a new
  `events_compaction_config` field (populated only via `Runner::from_app`,
  matching `context_cache_config`'s existing sourcing rule) and a
  `CompactionTrigger` trait-object extension point
  (`Runner::with_compaction_trigger`), called after every produced
  event has already been appended and just before returning from
  `run_async_with_config`. **Load-bearing architecture note**: this
  port's crate layering (`adk-tools`/`adk-flows` both depend on
  `adk-runners`, not the reverse) makes it impossible for
  `adk-runners` to call the real trigger logic directly without a
  crate-graph cycle — `adk_flows::apps_compaction::with_real_compaction_trigger`
  (in the same file, which already depends on `adk-runners`) wires the
  real implementation in; a `Runner` built via `from_app` alone has
  compaction configured but inert until that wiring is applied. 13 new
  tests (11 for the decision logic, 2 end-to-end for the wiring). New
  dependency: `adk-flows` now depends on `adk-runners` (no cycle —
  `adk-runners` depends on neither `adk-flows` nor `adk-tools`).
### Added
- `adk-runners::Runner::rewind_async` (C0891/C0894 DONE) + new
  `crates/adk-runners/src/rewind.rs`
  (`compute_state_delta_for_rewind`/`compute_artifact_delta_for_rewind`,
  C0892/C0893 DONE) — rewinds a session to before a given invocation by
  appending a single reversing-delta event (never a destructive
  truncation of `session.events`); `adk_events::rewind::apply_rewinds`
  (already DONE) is what interprets it downstream. The state-delta
  helper replays history to reconstruct state at the rewind point,
  treating an explicit `Value::Null` in a historical delta as a
  tombstone; the artifact-delta helper restores changed artifacts as
  brand-new versions (never rewriting history), marking a
  never-existed-at-rewind-point artifact inaccessible via the same
  `rusty_serde::json::to_value`-of-a-`Part` representation
  `save_files_as_artifacts_plugin.rs` already established. Narrowed:
  no `run_config`/`GetSessionConfig` parameter, the same already-
  disclosed C0873 narrowing. Corrects `runner.rs`'s module doc, which
  previously listed `rewind_async` among the not-yet-built pieces.
  14 new tests (10 pure-function tests in `rewind.rs`, 4 in
  `runner.rs`'s own test module verifying the wiring). No new
  dependency.
### Added
- `adk-flows::apps_compaction::{ensure_compaction_summarizer,
  events_to_compact_for_token_threshold, longest_self_contained_prefix,
  safe_token_compaction_split_index}` (C0290/C0291/C0292 DONE) — the
  rest of `apps/compaction.py`'s pure logic, continuing the same file
  PR #112 started. `ensure_compaction_summarizer` resolves an
  `EventsCompactionConfig`'s summarizer (existing one, or a new
  `LlmEventSummarizer` from the agent's already-resolved canonical
  model via `LlmFlow::model`), reusing the `agent.as_any()
  .downcast_ref::<LlmFlow>()` pattern `instructions.rs` already
  established for recovering a concrete `LlmAgent`-backed behavior.
  Disclosed: resolves-and-returns rather than mutating `config
  .summarizer` in place (no interior mutability on this port's
  `EventsCompactionConfig`) — in-place caching is left to whatever
  wires C0293. `events_to_compact_for_token_threshold`/
  `longest_self_contained_prefix`/`safe_token_compaction_split_index`
  port in full, including the prior-compaction-summary seeding and the
  responses-close-before-calls-open ordering. C0293 (the two
  `Runner`-facing trigger entrypoints) deliberately left for a later,
  larger batch needing real `App`/`Runner` wiring. 14 new tests. No
  new dependency.
### Added
- `adk-flows::apps_compaction::{latest_compaction_event,
  estimate_prompt_token_count, latest_prompt_token_count}` (C0288
  partial, C0289 DONE) — dedup and token-estimation logic ported from
  `apps/compaction.py`, distinct from this crate's sibling
  `compaction.rs` (`flows/llm_flows/_content_compaction.py`, C0185,
  a different source module needing the same subsumption-detection
  shape for a different purpose). `latest_compaction_event` finds the
  latest non-subsumed compaction range (a range fully contained by/
  identical to a later one is subsumed and ignored). Token estimation
  reuses the real `adk-flows::contents::get_contents` prompt-assembly
  path (4 chars/token), preferring a real `usage_metadata
  .promptTokenCount` (the already-established key-read convention)
  over the estimate. Not ported: the OTel-traced summarization wrapper
  (C0288's other half — needs span/tracer machinery not adopted in
  this port) and C0290-C0293 (summarizer lazy-init, the safe-window
  split logic, and the two `Runner`-facing trigger entrypoints —
  deliberately left for a follow-up batch). 11 new tests. No new
  dependency.
### Added
- `adk-tools::llm_event_summarizer::LlmEventSummarizer` (C0286/C0287
  DONE) — the LLM-based sliding-window-compaction summarizer,
  implementing `adk-agents::app_configs::BaseEventsSummarizer` (C0285).
  Formats a conversation history (text, thoughts — skipping ones from a
  prior compaction event, tool calls/responses truncated at 2000
  chars), drives one non-streaming LLM call, and wraps the result into
  an `Event` carrying an `EventCompaction` action with `role` forced to
  `"model"` and `author` forced to `"user"`. Lands in `adk-tools`, not
  `adk-agents`, since it needs a real `adk-models::BaseLlm` and
  `adk-models` already depends on `adk-agents` — disclosed in the
  module doc, the same supporting-crate placement
  `forwarding_artifact_service.rs` (C0489) already used. `args`/
  `response` formatting reuses the compact-JSON-instead-of-`str()`
  divergence `adk-events::debug_output` (C0933) already established.
  Also corrects `app_configs.rs`'s stale "still LLM-blocked" claim
  about C0286/C0287. 10 new tests. No new dependency.
### Added
- `adk-tools::forwarding_artifact_service::ForwardingArtifactService`
  (C0489 partial) — routes a nested `AgentTool` `Runner`'s artifact
  reads/writes back through the parent tool context's own real
  artifact backend, closing a gap `agent_tool.rs`'s module doc has
  disclosed since C0406: the nested run previously had no artifact
  service at all. Disclosed adaptation: this port's synchronous,
  `&self`-only `ArtifactService` trait can't hold a live mutable
  borrow of the parent `Context` across the nested run (unlike the
  source, which awaits the parent's own async `save_artifact`
  inline), so saved versions accumulate in a shared map and merge into
  the parent's `artifact_delta` action once the nested run completes —
  the same post-hoc-merge idiom `agent_tool.rs` already uses for state
  deltas. 8 new tests (6 in the new module, 2 end-to-end in
  `agent_tool`). No new dependency.
### Added
- `adk-runners::Runner::from_app` (C0846/C0849 DONE) — an additive
  second constructor building a `Runner` from a resolved `App`,
  deriving `context_cache_config`/`resumability_config`/`plugins` from
  it rather than accepting them as direct arguments (matching the
  source's "never accepted as direct constructor args" rule for the
  first two). `Runner::new`'s already-shipped signature is untouched.
  `app_name` defaults to `app.name`, with an optional override
  parameter matching the source's `app_name or app.name`. Also closes
  out C0847/C0848/C0850 as N/A (agent-origin inference and the
  deprecated back-compat wrapper have nothing to port in this shape)
  and corrects the module doc's stale "`App` type... N/A here" claim.
  6 new tests. No new dependency.
### Added
- `adk-agents::app::{App, validate_app_name, AppError}` (C0279/C0280
  DONE) — the top-level container binding a root agent + app-wide
  settings (`plugins`, `events_compaction_config`,
  `context_cache_config`, `resumability_config`). `root_agent` narrows
  from the source's `Union[BaseAgent, BaseNode, None]` to `BaseAgent`-only
  and becomes a required constructor argument (the `BaseNode`/workflow
  graph engine, C0298-C0306, isn't built in this port, and the source's
  own `_validate` model-validator already rejects `None`).
  `validate_app_name` is a new, distinct validator from
  `base_agent::validate_name` — the app-name regex additionally permits
  hyphens. Deliberately not wired into `Runner`'s constructor this
  batch — a follow-up once `App` exists and can be reviewed on its own.
  7 new tests. No new dependency.
### Added
- `adk-runners::Runner::in_memory` (C0926 DONE) — `InMemoryRunner`,
  narrowed from a `Runner` subclass to a constructor: pre-wires
  `InMemorySessionService`/`InMemoryArtifactService`/`InMemoryMemoryService`
  for testing and development, defaulting `app_name` to the literal
  `"InMemoryRunner"`. `plugins`/`plugin_close_timeout` aren't forwarded
  as separate parameters — already reachable through `Runner`'s existing
  `.with_plugin`/`.with_plugin_close_timeout` builders. Also corrects a
  stale module doc claiming `InMemoryArtifactService`/`InMemoryMemoryService`
  were "neither built yet" — both have been for several batches. 2 new
  tests. No new dependency.
### Added
- `adk-models::base_llm::{AsAny}` (new) — the same downcast mechanism
  `AgentBehavior::as_any` established in `adk-agents` (PR #104), now on
  `BaseLlm` too: lets `adk-flows::llm_flow::LlmFlow` detect whether its
  resolved `Arc<dyn BaseLlm>` is a `Gemini` without `adk-models` needing
  to know about `adk-flows`. Purely additive. A same-crate regression
  test guards the same `.as_ref()`-before-downcast trap PR #104 already
  caught once for `Box<dyn AgentBehavior>`.
- `adk-flows::llm_flow::LlmFlow::preprocess` now wires
  `interactions_processor` (C0174 DONE): gates on the downcast above
  plus `Gemini::use_interactions_api`, calling
  `interactions::find_previous_interaction_state` to set
  `LlmRequest::previous_interaction_id` when set. Corrects 2 stale
  module docs (`llm_flow.rs`, `interactions.rs`) that described this as
  blocked on "no downcasting mechanism." 4 new tests (3 in `adk-flows`,
  1 `AsAny` regression test in `adk-models`). No new dependency.
### Added
- `adk-tools::vertex_ai_search_tool::{VertexAiSearchTool, VertexAiSearchConfig}`
  (C0433 DONE) — a built-in Gemini retrieval tool bound to a Vertex AI
  Search data-store/search-engine (mutually exclusive, constructor
  validated). Ports the full populated
  `{"retrieval":{"vertexAiSearch":{...}}}` config shape. The source's
  subclass-based dynamic-filter customization point becomes an optional
  closure field (`with_config_builder`) — the same "overridable Python
  method → closure field" adaptation `base_agent.rs`'s `AgentCallback`
  already established. 8 new tests. No new dependency.
### Added
- `adk-agents::base_agent::{AgentBehavior::as_any, AsAny, BaseAgent::as_any}`
  (new) — a downcast escape hatch letting code holding a `BaseAgent`
  (type-erased: `Box<dyn AgentBehavior>`) recover a concrete behavior type
  for a cross-tree lookup (an ancestor's `LlmAgent`-specific fields).
  Purely additive via a blanket-implemented `AsAny` supertrait — no
  existing `AgentBehavior` implementor needs changes. Also adds
  `adk-agents::readonly_context::ReadonlyContext::agent`, exposing the
  running agent so callers can walk `.parent_agent()`/`.root_agent()`.
- `adk-flows::instructions::build_instructions` now genuinely walks to the
  tree root for the deprecated `global_instruction` field (C0170), using
  the new downcast mechanism, matching the source's
  `hasattr(root_agent, 'global_instruction')` gate — closing a
  previously-disclosed narrowing (reading the running agent's own field
  as a same-agent stand-in). 2 new tests build a real 2-level tree and
  prove a sub-agent's `LlmAgent` picks up the *root's* field. Falls back
  to the passed-in agent's own field only when no tree context is set at
  all (a remaining disclosed narrowing, see the module doc).
### Fixed
- `BaseAgent::as_any` (added in this same change): `Box<dyn AgentBehavior>`
  is itself `Sized + 'static`, so a naive `AsAny` blanket impl also
  (over-broadly) covers the `Box` itself — method resolution picks that
  outer impl before reaching the supertrait vtable, silently returning
  the `Box`'s own `TypeId` instead of the concrete behavior's, so every
  downcast would always fail. Fixed by forcing an explicit `.as_ref()`
  deref to `&dyn AgentBehavior` before calling `.as_any()`. Caught by a
  same-crate regression test before this line ever shipped in a release.
### Added
- `adk-agents::logging_plugin::LoggingPlugin` (C0362 partial) — an
  ANSI-grey console-debugging plugin. 6 of 13 hooks ported
  (`on_user_message_callback`/`before_run_callback`/`on_event_callback`/
  `after_run_callback`/`before_agent_callback`/`after_agent_callback`),
  including `format_content`/`format_args`, the source's own
  `_format_content`/`_format_args` truncation helpers. 11 new tests.
  The remaining 7 hooks (model-level/tool-level) stay N/A, blocked on
  C0355/C0356's already-disclosed crate-cycle blocker. No new
  dependency.
### Added
- `adk-agents::save_files_as_artifacts_plugin::SaveFilesAsArtifactsPlugin`
  (C0367 DONE) — saves `inline_data` parts from a user message as artifacts
  (20MB size cap), replacing each with a placeholder text part and,
  optionally, a `file_data` reference part when the saved artifact's
  `canonical_uri` uses a model-accessible scheme (`gs`/`https`/`http`).
  10 new tests total. Reads `MediaBlobStub`'s flattened `rest` map for
  `displayName`/`data`, reusing `file_artifact_service`'s `base64_decode`
  (promoted `pub(crate)`).
### Fixed
- `adk-runners::runner::merge_context_state_into_session` (new): a real
  bug fix surfaced while porting `SaveFilesAsArtifactsPlugin` — a
  run-level plugin hook's state mutations (`on_user_message_callback`,
  `before_run_callback`, `on_event_callback`, `after_run_callback`) were
  previously invisible to any other hook, since this port's `Context`
  clones `InvocationContext` rather than sharing it by reference the way
  the source's raw shared `session.state` dict does. Now applied after
  every run-level hook call in `Runner::run_async_with_config`; verified
  end-to-end by a new test proving `SaveFilesAsArtifactsPlugin`'s
  stash-in-`on_user_message_callback`-then-flush-in-`before_agent_callback`
  pattern actually works through `Runner`. No new dependency.
### Added
- `adk-runners::Runner` now stores a real `PluginManager` (`Runner::with_plugin`)
  and wires its run-level hooks into a new `Runner::run_async_with_config`
  (`run_async` delegates to it with a default `RunConfig`) — the Rust analog
  of `_exec_with_plugin`/`_handle_new_message`/`_append_new_message_to_session`:
  `on_user_message_callback`, the deprecated `save_input_blobs_as_artifacts`
  blob-saving path (C0898/C0899 DONE), `before_run_callback` (early-exit
  support), `on_event_callback` merging (`merge_output_event`, C0896 DONE),
  event-persistence gating (`should_append_event`, C0895 partial — only the
  non-live half is reachable), and `after_run_callback`/`on_run_error_callback`
  (C0357 DONE). Closes out C0353's remaining run-level call-site gap and
  C0886's plugin-wrapping gap (compaction is now the only thing left on that
  row). `Runner::close` now also closes every registered plugin. 10 new
  tests. No new dependency.
### Added
- `adk-agents::{base_credential_exchanger::BaseCredentialExchanger,
  credential_exchanger_registry::CredentialExchangerRegistry}` (C0523
  DONE), `adk-agents::{base_credential_refresher::BaseCredentialRefresher,
  credential_refresher_registry::CredentialRefresherRegistry}` (C0525
  DONE), `adk-agents::in_memory_credential_service::
  InMemoryCredentialService` (C0528 DONE), and
  `adk-agents::session_state_credential_service::
  SessionStateCredentialService` (C0529 DONE). 18 new tests. Both
  registries key directly on `AuthCredentialTypes` (given new `Hash`/
  `PartialOrd`/`Ord` derives) — already a closed enum, so no discriminant
  adaptation was needed the way `AuthProviderRegistry`'s `type[AuthScheme]`
  required. Implementing the two credential services required widening
  two long-stale placeholders: `adk-agents::services::AuthConfig` (was
  `Value`) now re-exports `auth_tool::AuthConfig`, and
  `adk-agents::services::CredentialService` (C0527 DONE as a discovered
  side effect) grows from a synchronous, context-free trait into a real
  async, `Context`-taking one — safe since grep confirmed zero prior
  implementors or call sites. `Context::save_credential`'s receiver
  changes from `&self` to `&mut self` to match (also zero external call
  sites). No new dependency.
### Added
- `adk-agents::telemetry_context::{ContentCapturingMode, TelemetryConfig,
  SemconvStabilityOptIn}` (C0651/C0652 DONE, plus 5 of C0670's 6 env-var
  constants) and `adk-agents::schema_version::resolve_schema_version`
  (C0679 DONE, plus its own env-var constant closing out C0670, and the
  `GOOGLE_CLOUD_AGENT_ENGINE_*` constants for C0671). 23 new tests.
  `RunConfig::telemetry` widens from a bare `Value` placeholder to the
  real `TelemetryConfig`. Ports every resolution property's full
  precedence ladder (admin lock, per-request field, `OTEL_*` env var,
  default) and `resolve_schema_version`'s env-override → Agent-Engine
  auto-detect → default-1 precedence. Pure env-var-precedence logic
  only — no OTel SDK/span/tracer machinery, a much larger still-unported
  surface. No new dependency.
### Fixed
- `capability-manifest.md`: C0505 and C0798 were both fully covered by
  already-merged work (C0504's `auth_tool.rs` batch and C0942's
  `telemetry_config.rs` respectively) but were never cross-linked and
  still read `REQUIRED`. Both now point at their real evidence — no new
  code, a manifest-hygiene fix only.
### Added
- `adk-agents::agent_optimizer::AgentOptimizer` (C0636 DONE),
  `adk-agents::sampler::{Sampler, ExampleSet}`, and
  `adk-agents::optimization_data_types::{SamplingResult,
  BaseSamplingResult, UnstructuredSamplingResult, AgentWithScores,
  BaseAgentWithScores, OptimizerResult}` (C0637 DONE). 8 new tests.
  Ports `optimization/`'s pure interfaces in full — no LLM call lives in
  either row (concrete LLM-touching optimizers/samplers stay their own,
  still-blocked rows). The source's pydantic-generic bounds become
  traits (`SamplingResult`/`AgentWithScores`) rather than base structs
  callers subclass; `UnstructuredSamplingResult`'s extra `data` field is
  declared directly on its own struct (same "flatten inherited fields"
  pattern as `ExtendedOAuth2`). `BaseAgentWithScores::optimized_agent`
  holds an `Arc<LlmAgent>` handle since `LlmAgent` has neither `Clone`
  nor `Debug`. Disclosed narrowing: `sample_and_score`'s Python-style
  default parameter values have no Rust equivalent (every caller passes
  every argument explicitly). No new dependency.
- `adk-agents::auth_schemes::{SecuritySchemeType, AuthSchemeType, ApiKeyIn,
  ApiKeyScheme, HttpScheme, OAuthFlow, OAuthFlows, OAuth2Scheme,
  OpenIdConnectScheme, SecurityScheme, OpenIdConnectWithConfig,
  CustomAuthScheme, AuthScheme, OAuthGrantType, ExtendedOAuth2}` (C0503
  DONE, plus C0498's wrap-up), `adk-agents::auth_tool::{AuthConfig,
  AuthToolArguments, stable_digest}` (C0504 DONE),
  `adk-agents::auth_headers::build_auth_headers` (C0522 DONE), and
  `adk-agents::{base_auth_provider::{AuthSchemeKind, BaseAuthProvider},
  auth_provider_registry::AuthProviderRegistry}` (C0516 DONE). 37 new
  tests. Closes out most of C0493's remaining unported top-level names
  (only `AuthHandler`, C0506, stays open). `AuthScheme`'s two-level
  union ports as nested `#[rusty_serde(untagged)]` enums;
  `CustomAuthScheme`'s developer-extensibility is preserved via a
  flattened `extra: Option<Value>` catch-all (not dropped, since
  extensibility is this type's whole purpose); `AuthProviderRegistry`
  keys by an `AuthSchemeKind` discriminant instead of the source's
  `type[AuthScheme]` (a real narrowing — every custom scheme collapses
  to one key, not one per exact subclass). Disclosed narrowing:
  the OpenAPI spec's four distinct `OAuthFlow*` shapes collapse into one
  lenient `OAuthFlow` struct; `build_auth_headers`'s API-key fallback
  for a non-`APIKey` scheme (an unsound `hasattr`-guarded read in the
  source, flagged there with `# type: ignore[union-attr]`) falls
  through to `None` rather than reproducing a latent crash;
  `stable_digest` isn't byte-identical to Python's digest (different
  JSON serializers), only equally deterministic. New dependency: `sha2`
  for `adk-agents` (already a workspace dependency, new usage site).
- `adk-agents::app_configs::{ResumabilityConfig, EventsCompactionConfig,
  EventsCompactionConfigError, BaseEventsSummarizer}` (C0283/C0284/C0285
  DONE). 18 new tests. Ports `ResumabilityConfig`'s `is_resumable` field,
  `EventsCompactionConfig`'s full validator (both-or-neither per trigger
  pair, at-least-one-trigger-mode, plus the `Field(gt=0)`/`Field(ge=0)`
  constraints pydantic enforces ahead of it), and the `BaseEventsSummarizer`
  trait. `InvocationContext::resumability_config`/`::events_compaction_config`
  upgraded from a narrowed stub/opaque placeholder to the real types. Also
  the first real call site for `adk_features::legacy_feature_decorator
  ::warn_experimental` (C0797's guard function, landed but unwired).
  `summarizer: Optional[BaseEventsSummarizer]` (an arbitrary, non-pydantic
  field in the source) becomes `Option<Arc<dyn BaseEventsSummarizer>>`,
  with no `Serialize`/`Deserialize` derive on `EventsCompactionConfig` at
  all (`Debug`/`Clone` implemented by hand instead). New dependency:
  `adk-features` for `adk-agents` (existing zero-dependency internal
  crate, new usage site, no cycle).
### Fixed
- `adk-tools::environment_simulation_config`'s tests: four tests
  (`environment_simulation_config_rejects_empty_tool_simulation_configs`,
  `::rejects_duplicate_tool_names`, `::rejects_disabled_feature`,
  `::accepts_a_valid_config`) raced each other under the default parallel
  test runner via `TemporaryFeatureOverride`'s process-wide override map,
  causing an intermittent failure. Serialized with a local `TEST_LOCK`
  mutex, same pattern `adk_features::feature_registry`'s own tests already
  use.
### Added
- `adk-tools::environment_simulation_config::{InjectedError, InjectionConfig,
  MockStrategy, ToolSimulationConfig, EnvironmentSimulationConfig}` (C0486
  DONE). 13 new tests. Ports every validator: injected-error-xor-injected-
  response plus the `injected_latency_seconds <= 120.0` constraint, the
  empty-injection-configs-requires-a-mock-strategy check, and the
  non-empty/no-duplicate-tool_name check — all cascading correctly. Also
  the first real call site for `adk_features::feature_decorator
  ::check_feature_enabled` (C0647's guard function, landed but unwired
  until now): the `@experimental(FeatureName.ENVIRONMENT_SIMULATION)` gate
  every source type carries is checked once at `EnvironmentSimulationConfig
  ::validate` rather than duplicated per leaf struct.
  `simulation_model_configuration` reuses `adk_models::llm_request
  ::GenerateContentConfigStub` instead of a new stub type. No new
  dependency.
- `adk-tools::environment_simulation_engine::EnvironmentSimulationEngine`
  (C0487 partial — the injection-only `before_tool_callback` path). 9 new
  tests. Ports per-tool-config lookup, `match_args` filtering, an
  optional-reseed-then-roll probability check against
  `adk_platform::random::Rng`, injected latency via `rusty_tokio::time
  ::sleep`, and the injected-error/injected-response dict shape. Deferred,
  disclosed in the module doc: the LLM-synthesized mock-response fallback
  — this port has no LLM-invocation path to drive it. No new dependency.
- `adk-tools::tool_connection_map::{StatefulParameter, ToolConnectionMap}`
  + `adk-tools::environment_simulation_factory
  ::{EnvironmentSimulationFactory::create_callback, SimulationCallback}`
  (C0488 partial). 4 new tests. Ports the pure tool-connection data types
  and a real `create_callback` closure with the source's exact
  `Fn(tool, args) -> Future<Output = Option<dict>>` shape. Deferred,
  disclosed in each module's own doc: `ToolConnectionAnalyzer`/
  `ToolSpecMockStrategy` (LLM-blocked) and `EnvironmentSimulationPlugin`/
  `create_plugin` (needs a `BasePlugin` tool-hook this port doesn't expose
  yet, same gap as the existing C0356 deferral); `create_callback`'s
  output also has no real dispatch target yet, since this port's
  `before_tool_callback` type takes no `tool`/`args` parameters. No new
  dependency.
- `adk-tools::skill_toolset::SkillScriptCodeExecutor` +
  `{python_str_literal, python_bytes_literal, python_list_literal,
  python_dict_literal}` (C0410 DONE — the `code_executor`-configured
  path of `RunSkillScriptTool`, closing this row out in full). 11 new
  tests, including two end-to-end runs against real `python3`/`bash`
  interpreters and a byte-for-byte round-trip test for the Python-
  literal escaping. Generates the same self-extracting Python wrapper
  source the original does (embedding every skill resource as a Python
  literal, extracting to a temp dir, then `runpy.run_path`-ing a `.py`
  target or `subprocess.run`-ing a `.sh`/`.bash` target through `bash`
  with a JSON-envelope result), executed via `rusty_tokio::spawn_blocking`
  (the `asyncio.to_thread` equivalent). Also adds the
  `code_executor`/`environment` mutual-exclusivity check the constructor
  was missing. Disclosed narrowing: the Python-literal helpers are
  round-trip-correct but not byte-identical to CPython's `repr()`
  (adaptive quote selection); the source's `except SystemExit as e:`
  branch is dead code for this port's only concrete `BaseCodeExecutor`
  (`UnsafeLocalCodeExecutor`, always subprocess-based, with no exit-code
  field on `CodeExecutionResult` to inspect). No new dependency.
- `adk-tools::skill_toolset::{AdditionalTool, SkillToolsetConfig
  ::additional_tools, SkillToolset::{resolve_additional_tools_from_state,
  clone_with_updated_skills}}` (C0950 DONE, closing the gap discovered
  last batch). 6 new tests. Ports the activated-skill → `adk_additional_tools`
  name-set → candidate-tool resolution pipeline (from provided tools and
  provided toolsets via `get_tools_with_prefix`), the core-tool-name-
  collision skip, and `clone_with_updated_skills`'s exact field carry-
  forward (faithfully including the source's own omission of
  `tool_name_prefix`/`tool_filter` from the clone). Disclosed narrowing:
  the source's `ToolUnion`'s bare-`Callable` branch (wrapped via
  `FunctionTool(callable)`'s `inspect.signature` reflection) has no port
  — `FunctionTool`'s own module doc already discloses this port has no
  such runtime reflection, so `AdditionalTool` only models the two
  Rust-expressible branches (`Tool`/`Toolset`). No new dependency.
- `adk-tools::skill_registry::SkillRegistry` (C0395 DONE) +
  `adk-tools::skill_instructions_utils::inject_session_state` (C0401 DONE,
  a local duplicate of `adk_flows::instructions_utils::inject_session_state`
  C0170, avoiding a crate-graph cycle) +
  `adk-tools::skill_toolset::{SkillToolset, SkillToolsetConfig,
  ListSkillsTool, SearchSkillsTool, LoadSkillTool}` (C0408 DONE) +
  `LoadSkillResourceTool` (C0409 DONE) + `RunSkillScriptTool` (C0410
  partial — `environment`-configured path only, `code_executor` path
  deferred to its own batch, needing `_SkillScriptCodeExecutor`'s
  from-scratch Python-wrapper-generation design) +
  `build_skill_system_instruction`/`default_skill_system_instruction`
  (C0411 DONE). 30 new tests. Widened
  `adk-tools::skills_models::Resources` from `String`-only to a real
  `ResourceContent` (`Text`/`Bytes`) enum, now that
  `LoadSkillResourceTool` is a real consumer needing the binary branch.
  Central architectural adaptation, disclosed at length in
  `skill_toolset`'s module doc: the source's tools each hold a live
  back-reference to their owning toolset (a reference cycle Rust can't
  replicate); this port pulls the shared mutable state into one
  `SkillCoreState` behind an `Arc` every tool clones a handle to, the
  same pattern `environment_toolset.rs` (C0440) already established.
  Disclosed narrowing: the source's per-invocation skill-fetch cache
  coalesces concurrent in-flight fetches via a shared `asyncio.Future`;
  this port keeps the 16-turn FIFO caching behavior exactly but two
  concurrent calls for an uncached skill each independently fetch,
  rather than one awaiting the other. New manifest row **C0950**
  (REQUIRED, not implemented) covers a discovered gap:
  `SkillToolset.additional_tools`/`_resolve_additional_tools_from_state`/
  `clone_with_updated_skills`. No new dependency.
- `adk-tools::base_environment::{BaseEnvironment, ExecutionResult,
  EnvironmentError}` (C0948 DONE — a genuine inventory gap: `environment/`
  had no manifest row at all, despite 4 existing rows already referencing
  it) + `adk-tools::local_environment::LocalEnvironment` (C0949 DONE) +
  `adk-tools::environment_toolset::EnvironmentToolset` (C0440 DONE) +
  `adk-tools::{execute_tool::ExecuteTool, read_file_tool::ReadFileTool,
  edit_file_tool::EditFileTool, write_file_tool::WriteFileTool}` (C0441-
  C0444 DONE). 34 new tests. No new dependency (`regex`/`rusty_tokio`
  already workspace dependencies of `adk-tools`). Notable adaptations,
  disclosed in each module's own doc: `BaseEnvironment::is_initialized`
  is a required trait method (Rust traits carry no data, unlike the
  source's class-level attribute every subclass inherits for free);
  `initialize()` returns `Result` even in the trivial default case, since
  a real implementor's IO can fail where the source lets an exception
  propagate uncaught; `write_file(content: str | bytes)` collapses to
  `&[u8]` without losing behavior (the source's str branch already
  disables newline translation, so both branches produce identical
  bytes); `LocalEnvironment`'s path resolution is lexical
  (`os.path.normpath`-style), not `Path.resolve()`-based, reusing the
  "path safety by construction, not by canonicalize" pattern already
  established in `file_artifact_service.rs` (C0268-C0269); a timed-out
  command carries no partial stdout/stderr, the same disclosed gap
  already established in `bash_tool.rs` (C0418); `EnvironmentToolset`'s
  uncaught initialize-failure becomes a panic rather than widening the
  already-shipped, infallible `BaseToolset` trait (C0403).
- `adk-eval::evaluation_generator::{collect_events_by_invocation_id,
  convert_events_to_eval_invocations}` (C0623 DONE) +
  `adk-eval::agent_evaluator::{load_json, find_config_for_test_file,
  get_initial_session, DatasetInput, load_dataset, validate_input,
  get_eval_set_from_old_format, load_eval_set_from_file,
  migrate_eval_data_to_new_schema}` (C0619 partial, C0620 DONE). 25 new
  tests total. Neither needs a real `Runner`/LLM-invocation path.
  `evaluation_generator` preserves invocation insertion order via a
  parallel `Vec<String>` alongside its grouping `HashMap` — unlike this
  crate's other `HashMap`-for-grouping choices, order here is
  semantically load-bearing (invocations are matched positionally
  against `expected_invocations` elsewhere). `agent_evaluator`'s
  `DatasetInput` enum (`Path`/`Paths`) models `_load_dataset`'s actual
  reachable `isinstance` dispatch rather than its broader, partly
  unreachable type hint; cross-verified the assert-vs-`ValidationError`
  control flow in `_load_eval_set_from_file` against the real source
  logic run standalone. `AgentEvaluator.evaluate`/`evaluate_eval_set`
  themselves stay `REQUIRED`, needing C0621/C0622/C0624's still-unbuilt
  inference generation. No new dependency.
- `adk-eval::llm_as_judge_utils::{Label, get_text_from_content,
  get_text_from_invocation, get_eval_status, get_average_rubric_score,
  get_tool_declarations_as_json_str,
  get_tool_calls_and_responses_as_json_str,
  get_grounding_metadata_as_json_str}` (C0947 DONE — closes the
  inventory gap added last batch) +
  `adk-eval::rubric_based_evaluator::{RubricResponse,
  AutoRaterResponseParser, DefaultAutoRaterResponseParser,
  PerInvocationResultsAggregator,
  MajorityVotePerInvocationResultsAggregator,
  InvocationResultsSummarizer, MeanInvocationResultsSummarizer,
  normalize_text}` (C0601 partial — `RubricBasedEvaluator` itself stays
  unbuilt, needing C0600's still-deferred `LlmAsJudge` harness and
  `AutoRaterScore`). Neither needs that harness to be useful — same
  reasoning as the C0612 criterion types and C0632 persona system.
  Cross-verified `get_text_from_content`'s `Some("")`-vs-`None`
  truthiness edge case and `DefaultAutoRaterResponseParser`'s parsing
  (well-formed response, missing-ID tolerance, mismatched-count
  rejection, unparseable verdict) directly against the real source
  logic run standalone. Widened `evaluator::PerInvocationResult
  ::rubric_scores`/`EvaluationResult::overall_rubric_scores` from
  opaque `Value` to real `Vec<RubricScore>`, now that the new
  aggregators are real consumers. Notable adaptations, disclosed in
  each module's doc: `get_text_from_content` splits by type (no
  function overloading in Rust); `Label`'s inconsistent per-variant
  `.value` shape becomes uniform; the source's two lookbehind regex
  patterns become ordinary capture groups (Rust's `regex` crate has no
  lookbehind); `normalize_text` skips NFKC normalization (same gap
  `rouge.rs` already carries). New dependency: `regex` (already a
  workspace dependency, new usage site in `adk-eval`). 31 new tests.
- `adk-eval::base_eval_service::{BaseEvalService, EvaluateConfig,
  InferenceConfig, InferenceRequest, InferenceStatus, InferenceResult,
  EvaluateRequest}` (C0616 DONE) + `adk-eval::custom_metric_evaluator::{
  CustomMetricEvaluator, register_custom_metric_function}` (C0599 DONE) +
  `adk-eval::metric_evaluator_registry::{MetricEvaluatorRegistry,
  default_registry, register_custom_metrics_from_config}` (C0603
  partial — registers only `TrajectoryEvaluator` among the 13 standard
  evaluators; the other 12 stay `REQUIRED` under their own rows, blocked
  on GCP or the still-deferred `LlmAsJudge` harness). Cross-verified
  `InferenceConfig`'s defaults/camelCase wire form directly against a
  real pydantic model built from the source. Notable adaptations,
  disclosed at length in each module's doc: `BaseEvalService`'s
  async-generator methods become `Vec`-returning (no async story yet in
  this crate); `custom_metric_evaluator`'s `importlib`-based dynamic
  dispatch becomes an explicit registration API keyed by the same
  dotted-path string, same pattern as `user_simulator`'s registry;
  `MetricEvaluatorRegistry`'s stored `type[Evaluator]` +
  `issubclass`-based construction dispatch becomes a tagged factory
  closure decided once at registration; its
  `DEFAULT_METRIC_EVALUATOR_REGISTRY` mutable singleton becomes a
  lazily-initialized mutex-guarded static, with
  `register_custom_metrics_from_config` always taking an explicit
  `&mut MetricEvaluatorRegistry` rather than defaulting to it. Also adds
  a manifest row, C0947, for `evaluation/llm_as_judge_utils.py` — a
  genuine inventory gap discovered while scoping this batch (the source
  file existed with no manifest row at all); not implemented this
  batch. No new dependencies. 17 new tests.
- `adk-eval::user_simulator::{UserSimulator, BaseUserSimulatorConfig,
  NextUserMessage, Status, register_user_simulator,
  create_user_simulator}` (C0626 DONE) +
  `adk-eval::static_user_simulator::StaticUserSimulator` (C0629 DONE) +
  `adk-eval::user_simulator_personas::{UserBehavior, UserPersona,
  UserPersonaRegistry}` + `adk-eval::pre_built_personas::{
  PreBuiltBehaviors, get_default_persona_registry}` (C0632 DONE) — the
  user-simulator core interface and the built-in EXPERT/NOVICE/EVALUATOR
  persona system. Cross-verified all 11 pre-built behaviors' and all 3
  personas' text/composition byte-for-byte against the real source
  module (zero mismatches, including the source's own typos/quirks,
  deliberately preserved as judge-model prompt content). Two disclosed
  adaptations: the config→simulator registry keys by the config's
  `type` discriminator string rather than by config *class* (Rust types
  aren't runtime dict keys); `UserSimulator`'s two methods are required
  trait methods rather than the source's non-abstract,
  `NotImplementedError`-by-default shape. `UserSimulatorProvider`
  (C0627) and the LLM-backed/audio simulators (C0628/C0630) stay
  `REQUIRED` — they need a real LLM-invocation path this batch doesn't
  build. New dependency: `adk-events` (already a lightweight leaf
  crate, new usage site for the real `Event` type). 27 new tests.
- `adk-eval::eval_config::{EvalConfig, CustomMetricConfig, LiveModelConfig,
  CodeConfig}` (C0611 DONE) + `adk-eval::eval_metrics::{JudgeModelOptions,
  LlmAsAJudgeCriterion, RubricsBasedCriterion, HallucinationsCriterion,
  LlmBackedUserSimulatorCriterion}` (C0612 DONE, completing it) —
  `eval_config.py`/the rest of `eval_metrics.py`'s criterion types.
  `EvalConfig::normalize_user_simulator_config` ports the legacy-
  default-injecting validator explicitly (missing/null `type` defaults
  to `"llm_backed"`); cross-checked its four cases, plus
  `JudgeModelOptions`'s defaults and `ge=1` rejection, directly against
  the real pydantic validator/model logic run standalone. Every
  criterion subtype flattens the source's class inheritance into its
  own full field set — confirmed none of them need the still-missing
  `LlmAsJudge` harness to be useful pure data models, same reasoning
  already established for `Rubric`/`RubricScore`. Disclosed narrowing:
  `EvalConfig.criteria`'s values and `user_simulator_config` stay
  opaque `Value`; `CustomMetricConfig.code_config` is a narrow local
  `CodeConfig` (`{name: String}`), not the real
  `agents.common_configs.CodeConfig` (C0348, still unbuilt and
  out of `adk-eval`'s crate-graph reach). No new dependencies. 24 new
  tests (17 in `eval_config`, 7 in `eval_metrics`).
- `adk-tools::skills_models::{Frontmatter, Script, Resources, Skill}`
  (C0393 DONE, C0394 DONE) — the `skills/models.py` data models:
  `Frontmatter`'s `name`/`description`/`license`/`compatibility`/
  `allowed_tools`/`metadata` fields with NFKC-normalized, kebab/
  snake-case `name` validation (branching on
  `FeatureName::SnakeCaseSkillName`, same 64/1024/500-char limits and
  `adk_additional_tools`/`adk_inject_state` metadata checks as the
  source), `Script`'s `Display` impl, `Resources`'s six
  get/list accessors, and `Skill`'s `name()`/`description()`
  delegation plus a `pub(crate)` `_uri` provenance field. Cross-checked
  the 64-char, snake_case-rejected-by-default, and `allowed-tools`
  alias/dump behavior directly against a real `pydantic`-backed
  `Frontmatter` from the source. New dependencies: `adk-features`
  (existing internal crate, new usage site) and `unicode-normalization`
  (small, well-audited, near-zero-transitive-deps, same bar as
  `adk-eval`'s `unicode-general-category`). 19 new tests.
- `adk-features::feature_decorator::check_feature_enabled` (C0647
  partial) and `adk-features::legacy_feature_decorator::{check_wip_or_bypass,
  warn_experimental}` (C0797 partial) — the `experimental`/
  `working_in_progress`/`stable` decorator mechanisms, ported as plain
  guard functions (Rust has no runtime decorators). Confirmed, not
  assumed, that `utils/feature_decorator.py` is a genuine second,
  independent feature-gating system (no registry involvement at all)
  rather than the same mechanism reached two ways. Wiring either guard
  into the source's actual decorated call sites across the codebase is
  its own larger undertaking, not done this batch. 7 new tests.
- `adk-eval::audio_utils` (C0625) — `resample_pcm16`/`to_live_input`/
  `parse_sample_rate`, a linear-interpolation PCM16 resampler (24kHz
  TTS → 16kHz Live API) and hand-rolled `rate=` mime-type parameter
  parser (no `regex` dependency needed). Cross-checked against the real
  `google.adk.evaluation._audio_utils` functions run directly. 15 new
  tests.
- `adk-eval::eval_metrics::{PrebuiltMetrics, Interval, MetricValueInfo,
  MetricInfo, MetricInfoProvider}` + `adk-eval::metric_info_providers`
  (C0604 DONE, closes the `Interval`/`MetricValueInfo`/`MetricInfo`/
  `MetricInfoProvider` slice of C0612 too) — all 12 concrete provider
  implementors (covering all 13 `PrebuiltMetrics`). Verified the
  source's two providers that pass a bare `PrebuiltMetrics` enum member
  (not `.value`) to a `str`-typed field are not a bug — confirmed live
  that pydantic v2 unwraps a plain-`Enum` member the same way `.value`
  would; this port uses `.as_str()` uniformly for all 13. 8 new tests.
- `adk-tools::skills_prompt::format_skills_as_xml` (C0400) — renders
  skills into an `<available_skills>` XML block for LLM instructions,
  HTML-escaped (`html.escape`'s default `quote=True` behavior ported
  exactly, cross-checked against real Python output). First landing in
  the `skills/` capability area — the full `Frontmatter`/`Skill` models
  (C0393/C0394) and disk-loading (C0396, needs a YAML crate decision)
  stay `REQUIRED`; this function only reads `name`/`description`, so it
  narrows to a minimal local `SkillSummary` struct instead of depending
  on either unbuilt model.
- `adk-tools::gemini_schema_util` (C0489 partial) — OpenAPI/JSON-Schema →
  Gemini-Schema conversion (`to_snake_case`, `dereference_schema` with
  circular-`$ref` guarding, `sanitize_schema_type`,
  `sanitize_schema_formats_for_gemini`, `to_gemini_schema`), operating on
  `Value` throughout. End-to-end cross-checked against the real
  `google.adk.tools._gemini_schema_util` source (imported and run
  directly from the checked-out `google/adk-python` repo) over 11
  fixtures spanning `$ref`/`$defs` resolution, circular refs, nullable
  type lists, `oneOf`→`anyOf` widening, camelCase key conversion, and
  per-type format allow-listing. Disclosed scope boundary: the source's
  final step delegates to the third-party `google-genai` SDK's own
  ~380-line `Schema.from_json_schema` (outside `google/adk-python`'s own
  source tree); this port stops at `_gemini_schema_util.py`'s own
  boundary and returns `Value` (this workspace's existing Gemini-schema
  representation) rather than replicating the SDK-internal step too.
- `adk-tools::mcp_conversion_utils` (C0455 partial) —
  `adk_to_mcp_tool_type`/`gemini_to_json_schema`, ported from
  `mcp_tool/conversion_utils.py`, backed by a narrowed local `McpTool`
  struct rather than a real `mcp` crate dependency (this port has none).
  `session_context.py`'s `SessionContext` (real async `mcp.ClientSession`
  pooling) stays `REQUIRED`. 30 new tests across both modules.
- `adk-eval` local persistence batch (C0613/C0615 partial, C0614 DONE) —
  `eval_sets_manager::EvalSetsManager` trait + shared `EvalManagerError`,
  `eval_sets_manager_utils`/`eval_set_results_manager_utils` support
  functions, `in_memory_eval_sets_manager::InMemoryEvalSetsManager`,
  `local_eval_sets_manager::LocalEvalSetsManager` (real `.evalset.json`
  file I/O, old→new legacy-format schema migration on read, verified
  against a real legacy-format fixture), `path_validation::validate_path_segment`
  (path-traversal/null-byte hardening, applied at every filesystem path
  construction site), `eval_set_results_manager::EvalSetResultsManager`
  trait, and `local_eval_set_results_manager::LocalEvalSetResultsManager`
  (with the back-compat double-JSON-decode fallback for legacy result
  files, verified by actually double-encoding a result and reading it
  back). `GcsEvalSetsManager`/`GcsEvalSetResultsManager` stay `REQUIRED`
  — no GCS SDK dependency is decided in this workspace. Adds `adk-errors`
  and `adk-platform` as new `adk-eval` dependencies — both lightweight
  leaf-ish crates, deliberately chosen over `adk-agents` (see the crate
  root doc). Disclosed narrowing: writes always include every field
  (`rusty_serde::json` has no pretty-printer or `skip_serializing_if`
  support), unlike the source's sparse, pretty-printed
  `exclude_unset`/`exclude_defaults`/`exclude_none` output — files
  round-trip correctly either way. 40 new tests.
- `adk-eval` data-model batch (C0606, C0607, C0609, C0610, C0635) —
  `eval_case::EvalCase`/`SessionInput`/`SessionState` (`conversation` XOR
  `conversation_scenario`, enforced via `EvalCase::validate()` rather
  than automatically on deserialization, disclosed in the module doc),
  `conversation_scenarios::ConversationScenario`/
  `ConversationGenerationConfig`, `eval_set::EvalSet`,
  `eval_rubrics::Rubric`/`RubricContent`/`RubricScore`,
  `app_details::AgentDetails`/`AppDetails`,
  `eval_result::EvalCaseResult`/`EvalSetResult`, and
  `constants::{MISSING_EVAL_DEPENDENCIES_MESSAGE,
  DEFAULT_LIVE_TIMEOUT_SECONDS, eval_constants}`. Closes the disclosed
  gap from the first `adk-eval` batch: `Invocation.rubrics`/
  `.app_details` now use the real `Rubric`/`AppDetails` types instead of
  opaque `Value`. Disclosed narrowings: `ConversationScenario.user_persona`
  (real type + registry resolution is the persona system, C0632, still
  `REQUIRED`) and `EvalCaseResult.session_details` (real type
  `adk_agents::session::Session` exists but pulling `adk-agents` into
  `adk-eval`'s dependency graph for one unread passthrough field would
  invert its deliberate bottom-of-the-graph position) both stay opaque
  `Value`. 21 new tests.
- `adk-eval::final_response_match_v1::RougeEvaluator` (C0590) — the
  `response_match_score` metric, ROUGE-1 F-measure between an agent's
  final response and a golden/expected response, using a hand-ported
  `nltk.stem.porter.PorterStemmer` (`adk-eval::porter_stemmer`, verified
  against 409 word/stem pairs from real nltk 3.10.3 — an earlier
  hand-derived expectation table, read from Porter's paper's own
  per-step examples in isolation, was caught wrong by this cross-check
  and replaced) and a Unicode-aware tokenizer (`adk-eval::rouge`)
  correctly handling CJK (character-level splitting), Thai/Lao/Khmer/
  Myanmar (grapheme-cluster grouping via combining-mark detection), and
  ASCII (routed through the stemmer) text — the stock ASCII-only ROUGE
  tokenizer would score any of the former scripts as zero overlap.
  End-to-end cross-checked against the real upstream `rouge_score`
  package source (fetched and run locally under real `nltk`) over 11
  candidate/reference pairs; all match to floating-point precision.
  Adds `unicode-general-category` (zero transitive dependencies,
  `no_std`, static Unicode 16.0.0 tables) as a new `adk-eval` dependency
  for cross-script Mark-category classification — judged impractical to
  hand-roll accurately, adopted directly under the same well-audited/
  non-sovereignty-sensitive bar the `regex` crate was. Disclosed
  narrowing: `unicodedata.normalize("NFKC", ...)` is skipped (affects
  only compatibility-decomposable characters, not the common case).
- New `adk-eval` crate: the pure-computation core of `google.adk.evaluation`
  (Phase 11's first landing) — `eval_case::Invocation`/`IntermediateData`/
  `InvocationEvents` + accessor helpers (C0605), `evaluator::Evaluator`
  trait/`PerInvocationResult`/`EvaluationResult` (C0600, partial),
  `eval_metrics::EvalMetric`/`EvalMetricResult`/
  `EvalMetricResultPerInvocation`/`BaseCriterion`/`ToolTrajectoryCriterion`
  (C0608, C0612 partial), and `trajectory_evaluator::TrajectoryEvaluator`
  (C0588, DONE in full) — the `tool_trajectory_avg_score` metric,
  comparing actual vs expected tool-call trajectories under EXACT/
  IN_ORDER/ANY_ORDER match algorithms. 31 new tests. Depends only on
  `adk-genai` + `rusty_serde` (bottom of the crate graph). Disclosed
  narrowings: `EvalMetric`'s private `config_custom_function_path` is a
  compile-time strengthening (no `Deserialize` support at all, not
  `PrivateAttr`'s runtime guard) of the source's inbound-payload-spoofing
  protection; `EvalStatus` serializes as its variant name rather than the
  source's underlying bare-integer Pydantic-v2 enum value (no
  cross-language consumer yet); `Invocation.rubrics`/`.app_details`/
  `EvalMetric.criterion`/`Evaluator::evaluate_invocations`'s
  `conversation_scenario` stay opaque `Value` placeholders (their real
  types are C0606/C0607/C0610, each independently still `REQUIRED`, and
  `TrajectoryEvaluator` never reads their structure). `LlmAsJudge` (the
  generic LLM-judge-sampling harness) and all LLM-judge criterion types
  are out of scope for this batch — no LLM-invocation path built on yet.
- `adk-tools::unsafe_local_code_executor::UnsafeLocalCodeExecutor`
  (C0385), ported from `google.adk.code_executors.unsafe_local_code_executor`
  — bare, zero-sandboxed subprocess code execution (same host/creds/
  network/filesystem access as this process). The embedded `_RUNNER`
  Python script, `__main__`-guard detection, and `stateful`/
  `optimize_data_file` rejection are all ported exactly; 6 new tests
  include real end-to-end subprocess execution (stdout capture, a
  traceback-on-error case, and a real 1-second timeout kill). This
  completes the `code_executors/` area's non-cloud-backend scope
  (C0383-C0385, C0390-C0391); `ContainerCodeExecutor`/`GkeCodeExecutor`/
  cloud executors still need a new SDK dependency decision.
  Disclosed adaptations: `sys.executable` becomes a configurable
  `python_executable` command (default `"python3"`) since this port was
  never a Python interpreter itself — running this executor still
  genuinely requires a Python interpreter on the host; the process-
  group SIGTERM→grace→SIGKILL sequence narrows to an immediate
  `Child::kill()`, the same disclosed `killpg`-equivalent gap already
  established by `bash_tool.rs` (C0418); stdin/stdout/stderr are
  handled by three dedicated OS threads since `execute_code` is
  synchronous (matching the source's own non-`async` signature).
- `adk-tools::built_in_code_executor::BuiltInCodeExecutor` (C0384) and
  `adk-tools::code_executor_context::CodeExecutorContext` (C0390),
  ported from `google.adk.code_executors.{built_in_code_executor,
  code_executor_context}`. `BuiltInCodeExecutor::process_llm_request`
  reuses `append_built_in_tool_marker` (`"codeExecution"` wire key,
  same helper `GoogleSearchTool`/etc. use) and preserves the raise-for-
  unsupported-model behavior, since it isn't a `BaseTool` and so isn't
  bound by that trait's usual error-dropping signature narrowing.
  `CodeExecutorContext` preserves the source's real distinction between
  its nested, flush-on-`get_state_delta` context sub-dict and the
  already-live root session-state keys; `File.content` round-trips as
  base64 text (this port's state has no raw-bytes `Value` variant).
  Adds `adk-platform` as a new direct dependency of `adk-tools`
  (already vetted workspace-wide, new usage site only).
- `adk-tools::code_execution_utils::{File, CodeExecutionInput,
  CodeExecutionResult, get_encoded_file_content,
  extract_code_and_truncate_content, build_executable_code_part,
  build_code_execution_result_part, convert_code_execution_parts}`
  (C0391) and `adk-tools::base_code_executor::{BaseCodeExecutor,
  CodeExecutorConfig}` (C0383), ported from
  `google.adk.code_executors.{code_execution_utils, base_code_executor}` —
  the first landing in a new `code_executors/` capability area. Reads/
  writes `Part.executable_code`/`code_execution_result` (opaque `Value`
  placeholders) by known Gemini wire keys, the same established pattern
  used elsewhere for opaque-boundary fields. A small base64 codec is
  duplicated locally (no workspace `base64` dependency; neither existing
  hand-rolled copy has the right shape for reuse here).
- `adk-agents::file_artifact_service::FileArtifactService` (C0268-C0274),
  ported from `google.adk.artifacts.file_artifact_service` — a
  filesystem-backed `ArtifactService` implementation: full storage
  layout (nested filenames, `user:`-namespace sharing across sessions),
  `mkdir`-staging-directory atomic crash-safe writes with rename-to-
  publish, path-traversal/rooted/drive-qualified filename rejection,
  `metadata.json`-name collision protection, and always-recomputed
  `canonical_uri`. Disclosed adaptations: no `_umask_derived_file_mode`
  equivalent needed (this port's atomic-write helper already gets
  umask-derived permissions the way the source's `mkstemp` doesn't);
  path-traversal prevention is by lexical construction rather than
  filesystem-resolving canonicalize (Rust's `canonicalize` needs the
  target to already exist); `canonical_uri` is a hand-rolled, purely
  lexical `file://` URI builder for the same reason; a small base64
  codec is duplicated locally (can't reuse `adk-tools`'s own hand-rolled
  one without a crate-graph cycle).
- Also closed out C0266/C0267 (`InMemoryArtifactService`'s empty-
  artifact sentinel and artifact-reference resolution) — both were
  already implemented but had no dedicated parity test; added one each.
- `adk-flows::cache_performance_analyzer::{CachePerformanceAnalyzer,
  CachePerformanceReport, CachePerformanceStats}` (C0946), ported from
  `google.adk.utils.cache_performance_analyzer` — analyzes
  `GeminiContextCacheManager` cache-hit/refresh performance from a
  session's event history. `Event.cache_metadata`/`usage_metadata` stay
  opaque `Value` placeholders (parsed into `CacheMetadata` on demand,
  the same idiom `context_cache.rs`'s C0175 already established); the
  source's untyped `Dict[str, Any]` return becomes a closed
  `CachePerformanceReport` enum; a missing session becomes an explicit
  `Err` rather than the source's implicit `AttributeError` risk.
  `@experimental` (C0797, still unresolved) isn't represented. Adds
  `adk-errors` as a new direct (test-only) dependency of `adk-flows` —
  an already-vetted workspace dependency, new usage site only.
- `adk-genai::schema_utils::strip_json_code_fence` (C0944), ported from
  `google.adk.utils._schema_utils`'s `_strip_json_code_fence` —
  hand-rolled rather than adding `regex` as a new usage site of the
  crate, since the source's `re.fullmatch(..., re.DOTALL)` reduces to
  plain string slicing once anchored at both ends.
- Manifest rows C0943 (`yaml_utils.py`, flagged — needs a new YAML
  dependency decision), C0945 (the rest of `_schema_utils.py`, flagged
  — blocked on missing generic runtime type-reflection/validation
  machinery this workspace doesn't have a `TypeAdapter` equivalent
  for), and C0946 (`cache_performance_analyzer.py`, deferred — a
  promising future-batch candidate, its own dependencies already exist).
  C0941's evidence corrected with a deeper investigation of exactly
  what blocks it (`LlmAgent` isn't wired into `BaseAgent`'s tree, and
  `ToolUnion` is still an opaque placeholder).
- `adk-platform::telemetry_config::{get_user_config_path,
  read_telemetry_consent, write_telemetry_consent}` (C0942), ported from
  `google.adk.utils._telemetry_config` — reads/writes the ADK global
  telemetry-consent preference at `~/.adk/config.json`, preserving any
  other keys already in the file. `pathlib.Path.home()`'s cross-platform
  resolution narrows to reading `$HOME` directly (disclosed Windows gap,
  no new dependency added); the on-disk file is compact JSON rather than
  the source's pretty-printed `indent=2` (`rusty_serde::json` has no
  pretty-printer) — a cosmetic-only divergence.
- Manifest rows C0940 (`context_utils.py`) and C0941 (`agent_info.py`)
  added: C0940 flagged — its `find_context_parameter` reflection-based
  Context-parameter detection is already handled differently (a fixed
  closure signature, disclosed in `function_tool.rs`'s own C0404 doc)
  for the one caller this workspace has built; C0941 is genuinely
  portable (every dependency already exists) but left `REQUIRED` for
  its own dedicated future batch rather than rushed into this one.
- `adk-errors::missing_extra::missing_extra` (C0935), ported from
  `google.adk.utils._dependency` — builds the standard "install this
  extra" message for a missing optional dependency.
- `adk-platform::visual_builder_context::{is_visual_builder,
  set_visual_builder}` (C0936), ported from
  `google.adk.utils._telemetry_context`'s `_is_visual_builder` context
  flag. `contextvars.ContextVar`'s async-task scoping becomes a
  `thread_local!` (disclosed narrowing to thread-scoping, same as
  `ClientLabelScope`, C0932).
- `adk-models::output_schema_utils::can_use_output_schema_with_tools`
  (C0938), ported from `google.adk.utils.output_schema_utils` — a
  deprecated wrapper delegating to `gemini_output_schema_and_tools`.
  `@deprecated` becomes Rust's own `#[deprecated]` attribute. Disclosed
  narrowing: the source's `LiteLlm`-instance always-`True` special case
  is dropped, since `LiteLlm` isn't ported in this workspace.
- Manifest row C0937 (`_serialized_base_model.SerializedBaseModel`)
  added and marked `DONE`, linked to the pre-existing repo-wide
  `#[rusty_serde(rename_all = "camelCase")]` convention across 16
  files — a structural capability that needed no new code, just its
  own tracked row and evidence.
- `adk-genai::json_utils::safe_json_loads` (C0931), ported from
  `google.adk.utils._json_utils` — generic over the target `Deserialize`
  type (the source returns a dynamically-typed `Any`), returning a
  uniform `Result<T, String>` with an optional context label folded
  into the error message.
- `adk-models::google_client_headers::{ClientLabelScope, EVAL_CLIENT_LABEL}`
  (C0932), ported from `google.adk.utils._client_labels_utils` —
  previously deferred (folded into C0133's evidence) since only the two
  unconditional tracking labels were ported. The source's
  `@contextmanager` becomes an RAII guard (`Drop`-based restore), the
  same pattern as `adk-features::TemporaryFeatureOverride`. `contextvars`'
  async-task-scoping becomes a `thread_local!` (only thread-scoped, a
  disclosed narrowing).
- `adk-events::debug_output::print_event` (C0933), ported from
  `google.adk.utils._debug_output` — prints an `Event` to stdout,
  showing tool calls/results/code execution/inline-or-file-data parts
  only when `verbose`.
- `adk-models::vertex_ai_utils::get_express_mode_api_key` (C0934),
  ported from `google.adk.utils.vertex_ai_utils`.
- `adk-models::capabilities::{GoogleLlmVariant, get_google_llm_variant}`
  gained its own manifest row (C0930, `variant_utils.py`) and a
  dedicated parity test — already ported in an earlier forward-pull
  batch but never linked or directly tested.
- 4 new tests for `adk-models::capabilities::is_enterprise_mode_enabled`
  (Phase 16, C0796) covering its `GOOGLE_GENAI_USE_ENTERPRISE` vs.
  deprecated `GOOGLE_GENAI_USE_VERTEXAI` precedence — the function
  itself was already ported in an earlier Phase 3 forward-pull batch
  but had no dedicated test and was never linked back to its manifest
  row until now.
- `adk-genai::content_utils::{extract_text_from_content, to_user_content,
  ToUserContentInput, SKIP_THOUGHT_SIGNATURE_VALIDATOR}` (Phase 2,
  C0927/C0928/C0929) — ported from `google.adk.utils.content_utils`.
  `extract_text_from_content` ported exactly; `to_user_content`'s
  runtime `isinstance` dispatch becomes an explicit
  `ToUserContentInput` enum (`Content`/`Text`/`Value`), with the
  source's `BaseModel`/dict/list/anything-else branches all folding
  into the `Value` variant, non-string values formatted as compact
  JSON rather than Python's `str()`/`repr()` (disclosed, low-severity).
  `SKIP_THOUGHT_SIGNATURE_VALIDATOR` is ported as a constant, ahead of
  its own caller (`ReflectAndRetryToolCallsPlugin`, not built).
  Consolidates `is_audio_part`/`filter_audio_parts` — previously a
  private duplicate local to `adk-models::gemini_llm_connection`
  (C0136) — into this shared module as the single source of truth.
  These three functions had no manifest rows at all before this
  batch; found and added per the boundary contract. 13 new tests.
- `adk-agents::session_util::{decode_model, extract_state_delta,
  make_json_safe_state, extract_json_safe_state_delta}` (Phase 5,
  C0209 partial) plus `State::{APP_PREFIX, USER_PREFIX, TEMP_PREFIX}`
  (completing C0205's evidence — the dict-with-pending-delta wrapper
  itself was already ported in an earlier batch, just never linked
  back to its manifest row) — ported from
  `google.adk.sessions._session_util`/`state.py`'s prefix constants.
  Reconciles `services.rs`'s own pre-existing private
  `TEMP_STATE_PREFIX` duplicate to reference `State::TEMP_PREFIX` as
  the single source of truth. `extract_state_delta`/
  `extract_json_safe_state_delta` ported exactly; `make_json_safe_state`
  is a disclosed near-no-op (this port's `Value` can only ever hold
  JSON-representable variants by construction, so there's no value
  that could fail the source's coercion); `decode_model` collapses the
  source's two distinct failure modes (primitive-non-dict vs.
  malformed-dict) to a uniform `None`, a disclosed narrowing. Nothing
  in this port's `SessionService` routes `extract_state_delta`'s
  output to real cross-session shared storage yet — ahead of its own
  caller, same as `remote_mcp_server.rs`. 8 new tests.
- New `adk-features` crate:
  `adk_features::feature_registry::{FeatureName, FeatureStage,
  FeatureConfig, feature_config, is_feature_enabled,
  override_feature_enabled, TemporaryFeatureOverride}` (Phase 12,
  C0643-C0646/C0648-C0649) — the feature-flag registry ported from
  `google.adk.features._feature_registry`. The three-tier
  `is_feature_enabled` precedence (programmatic override → env var →
  registry default), the `ADK_ENABLE_<NAME>`/`ADK_DISABLE_<NAME>`
  convention, and the once-per-process non-stable-feature notice are
  all ported and tested. `temporary_feature_override`'s
  `@contextmanager` becomes an RAII guard (`TemporaryFeatureOverride`,
  restore-on-`Drop`) — the standard Rust idiom for "run on scope exit
  including unwind," matching the source's `try`/`finally` semantics.
  The registry itself is a fixed, exhaustive `match` rather than a
  mutable dict (nothing in this batch's scope calls the source's
  decorator-driven dynamic registration), which makes the source's
  "raises on an unregistered name" branches structurally unreachable
  here — a compile-time strengthening, not a narrowing. Not this
  batch: the `experimental`/`working_in_progress`/`stable` decorators
  (C0647) — Rust has no runtime-decorator analog to gate an arbitrary
  object behind a flag; left `REQUIRED`, undecided. 9 new tests.
- `adk-agents::in_memory_artifact_service::InMemoryArtifactService`
  (Phase 6, C0265) — the first real `ArtifactService` implementation:
  path-keyed `Vec<ArtifactEntry>` storage (version = list length at
  save time, monotonic per path), the `"user:"`-namespace vs.
  session-scoped path split, MIME-type detection including the
  artifact-reference-URI resolve-and-validate-scope recursion via
  `artifact_util` (C0262-C0264, its first real caller), empty-artifact
  sentinel checks on load, and the `memory://` canonical URI scheme.
  Widens `adk_agents::services::ArtifactVersion` from a `Value`
  placeholder to a real struct and extends `ArtifactService` with
  `delete_artifact`/`list_versions`/`list_artifact_versions` to match
  the source's full abstract interface. Disclosed, predating this
  batch: the trait's `session_id` is a required `&str` everywhere (not
  `Optional[str]`), so `list_artifact_keys` always returns the
  combined session+user listing; `artifact`/return values stay opaque
  `Value` at the trait boundary, deserialized/reserialized internally.
  Where the source raises `InputValidationError`/`ValueError`, this
  port panics — the closest analog to an uncaught exception through
  this trait's non-`Result` methods. 12 new tests.
- `adk-agents::artifact_util::{ParsedArtifactUri, parse_artifact_uri,
  get_artifact_uri, is_artifact_ref, validate_artifact_reference_scope,
  validate_path_segment}` (Phase 6, C0262/C0263/C0264) — the canonical
  `artifact://apps/{app}/users/{user}/[sessions/{sid}/]artifacts/{filename}/versions/{v}`
  URI scheme (parse/construct, round-trip tested), the cross-tenant
  artifact-reference-escape security boundary, and the path-segment
  validator every artifact backend needs for its app/user/session
  identifiers (rejects empty/null-byte/absolute/drive-qualified/
  traversal segments). Pure string/regex logic, no I/O, building on
  the already-real `InputValidationError`. Adapted: `is_artifact_ref`
  reads `"fileUri"` out of `Part.file_data`'s opaque flattened `rest`
  map (this port's `MediaBlobStub` has no typed `file_uri` field),
  the same pattern `load_artifacts_tool.rs` already uses for
  `inline_data.rest.get("data")`. Not built: `InMemoryArtifactService`
  (C0265) doesn't exist yet, so nothing produces a real artifact-backed
  `Part` in a live turn yet — this utility is real and tested, ahead
  of its own only caller. 17 new tests.
- `adk-agents::in_memory_memory_service::{InMemoryMemoryService,
  format_timestamp, UNKNOWN_SESSION_ID}` (Phase 6, C0244/C0245 partial/
  C0247/C0248/C0249) — the first real `MemoryService` implementation:
  a `Mutex`-guarded in-process keyword-search memory backend
  ("prototyping purpose only", matching the source's own docstring).
  Ports `add_session_to_memory` (wholesale per-session overwrite),
  `add_events_to_memory` (additive, dedup by event id), and
  `search_memory` (Unicode-aware `\w+` tokenization + non-ASCII
  substring fallback, snapshot-under-lock-then-score-outside-lock,
  10-result cap, stable ties) exactly. Also ports `format_timestamp`
  using a hand-rolled, deterministic public-domain epoch-to-calendar
  algorithm rather than a date/time crate. Disclosed: this port
  formats timestamps in UTC, not the host's local timezone the way the
  source does (true local-time conversion needs a full IANA
  timezone-database crate, not added); `add_memory` (never overridden
  by the source, defaulting to `NotImplementedError`) has no `Result`
  path through the pre-existing `MemoryService` trait, so it panics via
  `unimplemented!()` instead — the closest analog to an uncaught
  exception. 11 new tests.
- `adk-agents::oauth2_util::{normalize_oauth_scopes, OAuthScopes,
  is_non_mtls_googleapis_endpoint, effective_googleapis_endpoint,
  use_client_cert_effective, update_credential_with_tokens}` (Phase 9,
  C0509/C0531 partial/C0532) — `_normalize_oauth_scopes` (dict-or-list
  scopes → a flat list), the pure env-var/URL-string half of the
  mTLS-endpoint-routing logic (`.googleapis.com` → `.mtls.googleapis.com`
  rewriting, hand-rolled hostname extraction, no new dependency added),
  and `update_credential_with_tokens` (copies token fields from an
  opaque token map into `OAuth2Auth`). Disclosed: the real
  client-certificate loading/mounting half of mTLS routing
  (`configure_session_for_mtls`/`MtlsClientCerts`) needs
  `google.auth.transport.mtls`, not a workspace dependency — unported;
  `use_client_cert_effective` always takes the source's env-var-fallback
  branch since `google.auth`'s cert-availability probe isn't available.
  17 new tests.
- `adk-agents::auth_credential` camelCase wire format (Phase 9,
  C0501, partial) — `#[rusty_serde(rename_all = "camelCase")]` on
  `HttpCredentials`/`HttpAuth`/`OAuth2Auth`/`ServiceAccount`/
  `AuthCredential`, matching `BaseModelWithConfig`'s camelCase alias
  generator. Deliberately excludes `ServiceAccountCredential`: its
  fields mirror a real downloaded GCP service-account JSON key file,
  which is itself snake_case — applying camelCase there would have
  broken parsing an actual key file, the one real input format that
  struct exists for. Disclosed: `populate_by_name=True`'s dual-name
  accept (snake_case *or* camelCase on input) has no port — this
  port's `Deserialize` only accepts the one configured wire name. Also
  marks C0502 (`AuthCredential.resource_ref`) DONE — already ported in
  the prior batch, just not yet reflected in the manifest. 3 new
  tests.
- `adk-agents::auth_credential::{AuthCredentialTypes, HttpCredentials,
  HttpAuth, TokenEndpointAuthMethod, OAuth2Auth,
  ServiceAccountCredential, ServiceAccount, AuthCredential}` (Phase 9,
  C0494/C0495/C0496/C0497/C0499, mostly DONE) — the credential-scheme
  data models from `auth.auth_credential`, starting Phase 9. Widens
  `adk_agents::services::AuthCredential` from a `Value` placeholder to
  this real struct (same "widen from placeholder" precedent as
  `MemoryEntry`/`SearchMemoryResponse`, C0423). `ServiceAccount::new`
  is fallible, running the source's `_validate_config` validator's
  exact two checks. Disclosed: `extra="allow"` (arbitrary preserved-
  but-redacted extra keys) has no Rust analogue — an unmodeled key is
  silently dropped, not preserved; non-repr secret fields have no
  redaction surface to port since this port's `Debug` isn't used to
  serialize these structs anywhere. Not this batch: `AuthScheme`/
  `OpenIdConnectWithConfig` (C0498, a separate source file) and the
  `auth/__init__.py` re-export-asymmetry behavior itself (C0493 —
  this port has no crate-root re-export layer for any module, so the
  asymmetry has no distinguishing analogue). 12 new tests.
- `adk-tools::remote_mcp_server::{RemoteMcpServer, HeaderProvider}`
  (Phase 8, C0491, partial) — the declarative model for a server-side
  MCP server used by the Managed Agents API (`url`/`name`/`headers`/
  `allowed_tools`/`header_provider`), plus a new `resolved_headers`
  method implementing the documented `header_provider`-wins-on-
  conflict merge behavior. Nothing in this port constructs one in a
  live turn yet — the Managed Agents API `interactions.create` request
  path (`ManagedAgent`) is a separate, unbuilt capability; this row is
  ahead of its own only caller, not blocked on a missing dependency.
  5 new tests.
- `adk-tools::request_input_tool::{request_input, request_input_tool,
  REQUEST_INPUT_FUNCTION_CALL_NAME}` (Phase 8, C0492, partial) — a
  `LongRunningFunctionTool` asking the user a free-text/structured
  question (`message` + optional `response_schema`) and returning
  `Null` to trigger the long-running-interrupt mechanism, matching
  `_request_input_func` exactly. Forward-references
  `REQUEST_INPUT_FUNCTION_CALL_NAME` ("adk_request_input") here since
  `adk-flows::functions`'s own HITL wiring for that constant isn't
  ported yet and `adk-tools` doesn't depend on `adk-flows`. Not
  ported: the source's `logging.info` call (no logging framework
  adopted yet). 6 new tests.
- `adk-tools::{google_search_tool::GoogleSearchTool,
  google_maps_grounding_tool::GoogleMapsGroundingTool,
  enterprise_search_tool::EnterpriseWebSearchTool,
  url_context_tool::UrlContextTool}` (Phase 8, C0428/C0430/C0431/
  C0432, partial) — the four built-in Gemini grounding tools: each
  checks whether the request targets a Gemini model (or has the
  model-ID check disabled) and, if so, appends a built-in-tool marker
  (e.g. `{"googleSearch": {}}`) to `llm_request.config.tools` via a
  new shared `append_tools::append_built_in_tool_marker` helper. Also
  adds `adk-tools::model_name_utils::{is_gemini_model_id_check_disabled,
  is_managed_agent}`. Disclosed narrowings, shared across all four:
  `process_llm_request` has no `Result` path, so an unsupported model
  silently skips the marker instead of the source's `ValueError`;
  `is_managed_agent()` always returns `false` (no such field on this
  port's `LlmRequest`); `GoogleSearchTool.bypass_multi_tools_limit` is
  stored but unenforced (no multi-tool-limit check exists yet).
  `EnterpriseWebSearchTool`/`GoogleMapsGroundingTool` match the
  source's own omission of an `_is_managed_agent` check for those two
  tools exactly. 12 new tests.
- `adk-tools::load_mcp_resource_tool::{LoadMcpResourceTool,
  McpResourceProvider}` (Phase 8, C0426, partial) — the full
  list→instruction-inject→function-response-detect→per-resource-read→
  append flow, reusing `load_artifacts_tool::maybe_base64_to_bytes`
  (C0425) for the same decode-with-placeholder-fallback shape. No real
  `McpToolset` exists yet (C0540-C0542, its own larger capability), so
  this defines a minimal `McpResourceProvider` trait carrying just the
  two operations this tool calls — the same placeholder-trait pattern
  `adk-agents::services` already uses for `MemoryService`/
  `ArtifactService`. Refined `maybe_base64_to_bytes` (shared with
  C0425): a non-empty input with no recognizable base64 characters
  now correctly reports a decode failure instead of silently
  producing an empty byte vector. 7 new tests.
- `adk-tools::load_web_page::{load_web_page, LoadWebPageTool}` (Phase
  8, C0427, partial) — fetches a URL and extracts its text, with the
  source's SSRF-hardening core ported in full: URL/host/port
  validation, `localhost` rejection, DNS resolution with every
  resolved address vetted, IPv4/IPv6 global-reachability
  classification transcribed field-for-field from CPython 3.11's
  `ipaddress.py` (ground-truthed against a 30+-address battery run
  through the real module), embedded-IPv4-in-IPv6 checks (mapped/
  6to4/NAT64/deprecated-compatible forms — verified this catches
  `64:ff9b::169.254.169.254`-style addresses the plain IPv6
  `is_global` check alone misses), IP-pinned fetch via
  `reqwest::blocking::ClientBuilder::resolve`, and disabled redirects.
  New `adk-tools` dependencies: `reqwest` (new usage site of an
  already-vetted workspace dependency), `url` (the exact crate
  `reqwest::Url` re-exports), `regex` (new usage site). Disclosed
  narrowings: no proxy-env-var-aware branching (always does the
  direct IP-pinned fetch — strictly more restrictive, not a safety
  regression); a regex-based HTML text extractor stands in for
  `BeautifulSoup`+`lxml`; the live-fetch path itself is untested in
  this sandboxed environment (network-dependent, and its own targets
  like `127.0.0.1` are correctly rejected by design). 15 new tests.
- `adk-tools::load_artifacts_tool::{LoadArtifactsTool,
  as_safe_part_for_llm}` (Phase 8, C0425, partial) — lists artifacts
  and injects them into the LLM request on demand: MIME
  normalization/classification, a hand-rolled base64 decoder
  (standard-strict then URL-safe-lenient), text-like decoding, and a
  binary-placeholder fallback; the full list→instruction-inject→
  `load_artifacts`-function-response-detect→per-artifact-load→append
  flow, including the session-scoped-then-`user:`-prefixed
  cross-session fallback. Disclosed narrowings (module doc, at
  length): no DOCX regex text extraction (needs a zip reader, not a
  workspace dependency), no spreadsheet parsing (needs a `pandas`
  equivalent — disabled by default upstream too), no `process_artifact`
  custom-callback override. 11 new tests.
- `adk-tools::bash_tool::ExecuteBashTool` (Phase 8, C0418, partial) —
  runs a validated bash command in a workspace directory via
  `rusty_tokio::process::Command` (no shell, matching the source's own
  `create_subprocess_exec`); mandatory confirmation gate on every
  invocation regardless of policy; command-prefix allowlist and
  blocked-operator validation; stdout/stderr capture (replicating the
  source's own "empty bytes → placeholder text" quirk) and Python's
  negative-on-signal `returncode` convention. 15 new tests, covering
  13 of the source's own `test_bash_tool.py` cases in spirit.
  Disclosed narrowings (module doc, at length): no `setrlimit`
  resource-limit enforcement (no `libc` binding — the policy fields
  don't exist rather than existing unenforced), a timeout kills only
  the immediate child not its process group (no `killpg` equivalent),
  no partial-output capture on timeout, and a hand-rolled
  POSIX-ish `shlex.split` stand-in.
- `adk-tools::agent_tool::AgentTool` (Phase 8, C0406, partial) — wraps a
  `BaseAgent` as a callable tool: spins up an isolated
  `InMemorySessionService` session (forwarding the parent's
  non-`_adk`-prefixed state), runs the wrapped agent for one turn via
  the real `adk_runners::Runner`, forwards state deltas back to the
  parent tool context per-event, and merges the last response event's
  non-thought text parts into the return value. New `adk-tools` →
  `adk-runners` dependency edge (verified non-circular). Adds a new
  `ToolError::NestedRunFailed` variant for session-creation/nested-run
  failures. Not ported (disclosed at length in the module doc):
  input/output-schema-driven declaration and validation (needs a
  concrete `LlmAgent` resolved from `BaseAgent`, same blocker every
  Phase 4 processor discloses), `ForwardingArtifactService`,
  `InMemoryMemoryService` (Phase 6), `include_plugins` propagation
  (`Runner` doesn't accept a `PluginManager` yet), grounding-metadata
  propagation, and code-execution part-to-text extraction. 4 new
  tests.
- `adk-tools::set_model_response_tool::SetModelResponseTool` (Phase 8,
  C0437, partial) — the output-schema workaround letting a model set
  its final structured response via a tool call when `output_schema`
  and other tools coexist. Sets the already-real
  `EventActions.set_model_response` field `output_schema.rs`'s
  `get_structured_model_response` (C0178) reads back. Fundamental
  adaptation, disclosed at length in the module doc: no Python-style
  runtime type introspection or Pydantic-equivalent JSON-schema
  validator exists in this port, so `output_schema` is taken as an
  already-opaque `Value` used directly as the declaration's
  `parameters`, and `run_async` uses an `items`/`response`-single-key
  convention to approximate the source's
  `BaseModel`/`list[BaseModel]`/raw-schema branches. The
  `ValidationError`-triggered retry-with-feedback path isn't ported —
  a real, disclosed capability gap. Unblocks the tool-existence half
  of C0171/C0178's own disclosed gaps (request-processor wiring stays
  deferred on the separate "resolve a concrete `LlmAgent`" blocker
  every Phase 4 processor shares). 5 new tests.
- `adk-tools::transfer_to_agent_tool::{transfer_to_agent,
  TransferToAgentTool}` (Phase 8, C0436, DONE) — the bare function
  sets `EventActions.transfer_to_agent`; the class variant wraps a
  `FunctionTool` and adds a JSON-schema `enum` constraint to the
  `agent_name` parameter, restricting choices to valid agents and
  preventing the model from hallucinating a target. Unblocks the
  `TransferToAgentTool`-building half of C0171 (`agent_transfer.rs`)
  that its own module doc disclosed as deferred until `BaseTool`
  existed. 4 new tests.
- `adk-tools::{load_memory_tool, preload_memory_tool, memory_entry_utils}`
  (Phase 8, C0423/C0424, DONE) — `LoadMemoryTool`/`load_memory` (wraps
  the already-real `Context::search_memory` in a `FunctionTool`,
  appends the "you have memory" instruction) and `PreloadMemoryTool`
  (automatically searches memory each turn and injects matches as
  transient user content, with the `Time: .../author: text`
  rendering). Promotes `adk_agents::services::{MemoryEntry,
  SearchMemoryResponse}` from opaque `Value` placeholders to real
  structs matching the source's pydantic models — `BaseMemoryService`
  itself stays an unbuilt Phase 6 trait; nothing produces real values
  yet, only consumes the shape. 11 new tests.
- New `adk-examples` crate (Phase 17, C0829/C0831/C0832, DONE) —
  `Example`/`BaseExampleProvider` (the few-shot-example extension
  point) and `example_util::{convert_examples_to_text,
  build_example_si, get_latest_message_from_user}`. All 10 of the
  source's own `test_example_util.py` cases ported, including the
  gemini-2-vs-other-model fence-style switch. Adaptations disclosed at
  length in the module doc: a compact-JSON stand-in for Python's
  dict/object `str()`/`repr()` (function-call-arg and
  function-response rendering), and `FunctionCall.args`'s `BTreeMap`
  rendering multi-argument calls in sorted-key rather than call-site
  order. `VertexAiExampleStore` (C0830) stays `REQUIRED`, deferred for
  the same Vertex-AI-auth reason already disclosed by
  `gemini_context_cache_manager.rs`. 14 new tests.
- `adk-tools::example_tool::ExampleTool` (Phase 8, C0419, DONE) —
  wires the new `adk-examples` crate into a `BaseTool`: reads the tool
  context's opaque `user_content` back into a typed `Content`, builds
  the examples instruction, and appends it to the LLM request via the
  already-real `LlmRequest::append_instructions`. 5 new tests.
- `adk-tools::{exit_loop_tool, long_running_tool, get_user_choice_tool}`
  (Phase 8, C0420-C0422, DONE) — `exit_loop` (sets
  `escalate`+`skip_summarization` to break a loop-type agent),
  `LongRunningFunctionTool` (wraps a `FunctionTool` by composition —
  no struct inheritance in Rust — delegating every method except
  `is_long_running` and `get_declaration`, which appends the "don't
  call again while pending" instruction), and `get_user_choice`/
  `get_user_choice_tool` (always defers to client-side resolution). 9
  new tests.
- `adk-flows::planners` (Phase 4, C0200-C0203, DONE) — `BasePlanner`
  trait, `BuiltInPlanner` (thinking-config passthrough, both hooks
  no-ops), and `PlanReActPlanner` (prompted Plan-Re-Act: 5-tag NL
  instruction injection, and tagged-response splitting into
  thought/answer/tool-call parts). All 8 of the source's own
  `test_plan_re_act_planner.py` tests ported 1:1, including the
  leading-parallel-function-call regression test, plus 2 new edge
  cases. Not yet wired into a real `BaseLlmRequestProcessor`/
  `BaseLlmResponseProcessor` for `_nl_planning` (C0176/C0179) — same
  "needs `InvocationContext.agent` to resolve a concrete `LlmAgent`"
  blocker already disclosed by every other Phase 4 processor. 13 new
  tests.
- `adk-flows::functions` (Phase 4, C0191/C0192, partial) — the
  `functions.py` execution core: `get_tool` resolves a `FunctionCall`
  against a `ToolsDict`, `execute_single_function_call` runs a tool via
  `BaseTool::run_async` and builds its response `Event`, and
  `execute_function_calls` dispatches many calls concurrently (one
  `rusty_tokio::spawn` task per call, matching the precedent already
  established by `ParallelAgent`), filters by ID, and merges the
  results via the already-built
  `functions_utils::merge_parallel_function_response_events`. Adds a
  new `adk-flows` → `adk-tools` dependency edge. Disclosed omissions
  (module doc has the full list): tool-level before/after/on-error
  callback dispatch (same crate-graph constraint that already excludes
  tool hooks from `BasePlugin`), auth-request/tool-confirmation-request
  event synthesis (needs Phase 9's `AuthConfig`), the long-running
  `_defers_response` empty-response skip (a real design gap —
  `BaseTool::run_async`'s `Result<Value, ToolError>` can't express "no
  response yet"), and OTel tracing (Phase 12). 7 new tests.
- `adk-agents::loop_agent::LoopAgent` (Phase 7 batch 4, C0337, partial)
  — structurally `SequentialAgent` wrapped in an outer loop that
  restarts from the first sub-agent up to `max_iterations` times (or
  forever if unset), stopping on escalate; resets sub-agent states
  between iterations via `ctx.reset_sub_agent_states` (already built).
  Reuses `SequentialAgent`'s own state-delta-propagation fix verbatim
  so a sub-agent in iteration 2 sees state a sub-agent in iteration 1
  set. 7 new tests.
- `adk-agents::parallel_agent::ParallelAgent` (Phase 7 batch 3, C0336,
  partial) — runs its sub-agents with genuine concurrency
  (`rusty_tokio::spawn`, one task per sub-agent), each in its own
  isolated branch (`BranchPath::create_sub_branch`). Adaptation,
  disclosed at length in the module doc: since `AgentBehavior` returns
  a fully-collected `Vec<Event>` rather than a live stream, there's no
  partial result to cancel mid-flight, so `escalate`-triggered early
  cancellation of still-running siblings isn't implemented (a sibling
  already in flight finishes normally). Each sub-agent's `state_delta`
  is applied onto the parent's `ctx.session.state` post-hoc (same fix
  as `SequentialAgent`) to restore cross-branch state visibility, since
  this port's `InvocationContext::clone()` is a real deep clone unlike
  the source's shallow `model_copy()`. 7 new tests.
- `adk-agents::sequential_agent::SequentialAgent` (Phase 7 batch 2,
  C0335, partial) — a `BaseAgent`-pluggable `AgentBehavior` running its
  sub-agents in order, with resumability (start-index resumption,
  agent-state markers, restart-from-beginning if a tracked sub-agent
  was removed, pause-on-long-running-call). Adaptation, disclosed at
  length in the module doc: this port's `Context`/`State` copy rather
  than share-by-reference, so `run_async_impl` explicitly applies each
  produced event's `state_delta` onto `ctx.session.state` as it
  processes it — otherwise a later sub-agent would never see an
  earlier one's state changes, breaking the entire point of a
  sequential chain. Live mode's `task_completed` tool auto-injection is
  deferred (needs `canonical_tools`, still a placeholder). 5 new tests.
- Starts Phase 7 (`plugins/`): `adk-agents::services::BasePlugin` — a
  real plugin trait replacing the old hardcoded-`None` `PluginManager`
  stub. Agent-level hooks (`before_agent_callback`/`after_agent_callback`,
  C0354, done) are wired into `BaseAgent::run_async`/`run_live` for
  real, fixing a latent bug along the way: those methods previously
  constructed a fresh, always-empty `PluginManager` locally instead of
  reading the actual one off the passed-in `InvocationContext`, so a
  configured `PluginManager` would have silently never run. Run-level
  hooks (`on_user_message_callback`/`before_run_callback`/
  `on_event_callback`/`after_run_callback`, C0353) and the
  notification-only hooks (`on_agent_error_callback`/`on_run_error_callback`,
  C0357) are defined and dispatchable but only `on_agent_error_callback`
  has a call site yet — the rest need `adk-runners::Runner`'s own
  wiring, a follow-up batch. `PluginManager` gets real `register_plugin`/
  `get_plugin`/`set_skip_closing_plugins` (C0359, done), first-non-None
  short-circuit dispatch with registration-order/plugin-before-agent
  precedence (C0358, done), and a sequential (matching the source's
  actual, not documented, behavior) `close()` (C0361, partial — no
  per-plugin timeout/failure-aggregation yet, since no plugin
  implementation exists that can fail to close). Model-level
  (`before_model_callback`/etc, C0355) and tool-level
  (`before_tool_callback`/etc, C0356) hooks are deferred — they'd need
  `adk-agents` to depend on `adk-models`/`adk-tools`, which already
  depend on `adk-agents`, the same crate-graph constraint
  `LlmRequest::append_tools` (C0116) already disclosed. 13 new tests
  across `adk-agents`.
- New `adk-runners` crate: `Runner` (C0840-C0845/C0873/C0884/C0886/
  C0888/C0924, all partial except C0888 done) — the core execution
  engine's "legacy" (plain `BaseAgent`, single always-non-resumable
  turn) path. `Runner::new(app_name, agent, session_service)` +
  `.with_artifact_service`/`.with_memory_service`/
  `.with_credential_service`/`.with_plugin_close_timeout`/
  `.with_auto_create_session`; `run_async` fetches-or-creates a
  session, rejects a `new_message` containing a function call, appends
  the user message, drives `agent.run_async`, and persists the
  resulting events; `close()` flushes the session service. No `App`
  type or workflow/node/task engine exists in this port yet (both
  Phase 7), so there's no app/agent/node union to validate against and
  no node/task/live/rewind/debug execution path — this crate wraps
  exactly one concrete `BaseAgent` directly. 6 new tests.
- Starts `Runner` (`runners.py`, C0833-C0926):
  `adk-agents::services::SessionService` upgraded from an empty
  marker trait to a real (if narrowed) port of `BaseSessionService`
  (C0206 partial) — `create_session`/`get_session`/`list_sessions`/
  `delete_session`, plus a concrete `append_event` default (temp-state
  apply/trim, session-state update) and a no-op `flush`. Object-safe
  via a boxed-future method pattern (mirrors
  `adk_tools::base_tool::BaseTool`), since `InvocationContext` stores
  it as `Arc<dyn SessionService>`. Adds `InMemorySessionService`
  (C0211 partial, C0213 done): nested-map storage behind a `Mutex`,
  with a real `append_event` override that dedups a re-delivered event
  against the canonical stored session and mirrors the appended
  event/state back onto it (since `get_session`/`create_session` hand
  callers their own clone). Narrowed, disclosed: no app:/user:
  state-prefix scoping or `get_user_state` (C0209/C0214, need their
  own architecture); no `last_update_time`-based `list_sessions`
  ordering or `StaleSessionError` (no such field on the placeholder
  `Session` yet); no `GetSessionConfig` event-trimming
  (`RunConfig.get_session_config` is still an opaque placeholder). 17
  new tests. Full workspace gate green.
- `adk-tools`: `FunctionTool` (Phase 8 batch 3, C0404 partial, C0405
  done) — wraps a Rust closure as a `BaseTool`. Since Rust has no
  runtime function-signature reflection, `FunctionTool::new` takes an
  already-built `FunctionDeclaration` and an explicit `required_args`
  list instead of deriving either from the wrapped closure, which
  always takes `(&BTreeMap<String, Value>, &mut ToolContext)` — no
  context-parameter auto-detection, no Pydantic-style argument
  coercion, no sync/async runner distinction (disclosed at length in
  the module doc). The `require_confirmation` gate (bool or async
  predicate) is fully ported, needing a small addition to
  `adk-agents::context::Context`: a `tool_confirmation`/
  `set_tool_confirmation` field (kept as an opaque `Value`, not a
  typed `ToolConfirmation`, to avoid a crate cycle).
- `adk-tools`: `BaseToolset` (Phase 8 batch 2, C0403, partial) — the
  base trait for a tool collection: `tool_filter`/`tool_name_prefix` as
  trait methods (not fields), a `ToolFilter` enum standing in for the
  source's `Union[ToolPredicate, List[str]]` (no runtime `isinstance`
  dispatch in Rust), and `get_tools_with_prefix` built as a default
  method over an explicit `PrefixCache`-behind-a-`Mutex` each
  implementor owns (the source's per-invocation cache is normal mutable
  instance state, which a `&self` trait method can't hold directly). A
  `PrefixedTool` wrapper (delegates to an inner `Arc<dyn BaseTool>`,
  overrides `name`/`get_declaration`) replaces the source's
  `copy.copy(tool)` + closure rewrite. `from_config` deferred (needs
  `ToolArgsConfig`, C0417, same gap as `BaseTool`).
- `adk-tools` crate (Phase 8 batch 1): `BaseTool` (C0402, partial —
  `from_config`/`SelfTool` deferred pending C0417), `ToolContext` (C0415,
  partial — `Context` type alias; Auth back-compat re-exports deferred to
  Phase 9), `ToolConfirmation`/`from_response_dict` (C0416, done), and
  `LlmRequest.append_tools`/`merge_declarations` (C0116, done). New crate
  sits alongside `adk-flows` (depends on `adk-agents`+`adk-genai`+
  `adk-models`) rather than nesting inside it, since `append_tools` needs
  `LlmRequest` while `BaseTool` needs `Context` — avoiding a crate-graph
  cycle by keeping `append_tools` a free function rather than a real
  `LlmRequest` method, the same "processor as a free function" pattern
  `adk-flows` already uses throughout. Adds `FunctionDeclaration` to
  `adk-genai` (real `name`/`description`, opaque `parameters`/`response`
  schemas) needed for `append_tools`'s dedup-by-name logic.
- Standard governance file set (repo-config): README, ARCHITECTURE, CONTRIBUTING,
  CODE_OF_CONDUCT, SECURITY, RELEASE_NOTES, ADR seed, PR/issue templates,
  `.gitattributes`.
- `capability-manifest.md`: full 831-row capability roadmap for the
  google/adk-python migration, grouped into 17 dependency-ordered phases.
- Cargo workspace + Phase 1 crates (`adk-platform`, `adk-errors`,
  `adk-events`), implementing capabilities C0001-C0033 (minus C0022/C0023,
  partially blocked on Phase 3) against `rusty_uuid`/`rusty_time`/
  `rusty_err`/`rusty_serde` sibling crates.
- `adk-agents` crate (Phase 2 batch 1): `BaseAgent`, `Context`/
  `ReadonlyContext`/`InvocationContext`, `RunConfig`, `LiveRequestQueue`,
  and supporting types (C0035-C0078 minus C0043/C0047/C0059/C0060/C0066/
  C0069/C0071, deferred pending later phases). Adopts `rusty_tokio` as the
  async runtime.
- 94 capability rows (C0833-C0926) filling the `runners.py` inventory gap
  flagged at row C0788.
- `LlmAgent` config shape + self-contained resolution helpers (Phase 2
  batch 2): `canonical_instruction`/`canonical_global_instruction`,
  `generate_content_config` validation, the `_llm_flow` decision,
  `set_default_model`/`set_default_live_model`, and the model/tool callback
  chain contract, plus `TaskRequest`/`TaskResult`/`DefaultTaskInput`/
  `DefaultTaskOutput` (C0079/C0081-C0089/C0091/C0093/C0100). `LlmAgent`
  isn't yet wired into `BaseAgent`'s tree — deferred pending Phase 3/4/8.
- `adk-genai` crate: a minimal real subset of `google.genai.types`
  (`Content`/`Part`/`FunctionCall`/`FunctionResponse`) needed to give
  `Event`/`LlmRequest`/`LlmResponse` real (not opaque-placeholder)
  behavior.
- `adk-models` crate (Phase 3 batch 1): `BaseLlm`/`LlmCapabilities`/
  `BaseLlmConnection`/`LLMRegistry`/`LlmRequest`/`LlmResponse`/
  `CacheMetadata` (C0101, C0103-C0108, C0110, C0114-C0115, C0117-C0120,
  C0122). Adopts the `regex` crate for model-name pattern matching. The
  native Gemini backend (C0123-C0143) is deferred to a follow-up batch
  pending an HTTP client decision.
- Retroactively completed, now that `adk-genai::Content` exists:
  `Event::is_final_response`/`has_trailing_code_execution_result` (Phase
  1, C0022/C0023), `InvocationContext`'s FC-matching methods (Phase 2,
  completing C0071), and `LlmAgent`'s `_get_subagent_to_resume`/
  `__maybe_save_output_to_state`/`__maybe_accumulate_streaming_output`
  (Phase 2, C0094/C0095).
- `Gemini` client-construction/config layer (Phase 3 batch 2): the config
  shape (`model`/`client`/`base_url`/`api_version`/`speech_config`/
  `use_interactions_api`/`retry_options`), `supported_models()`,
  `api_client`/`GeminiApiClient` construction, and base-URL/API-version
  resolution (C0123, C0124, C0129, C0130). Adopts `reqwest` (`rustls-tls`)
  as the HTTP client after checking every sibling Rusty-Mill repo for an
  HTTP/TLS candidate (none exists) — the REST/SSE transport decision;
  the Live API's WebSocket transport is a separate, later decision.
- Real, non-streaming `Gemini.generate_content_async` (Phase 3 batch 3):
  `LlmResponse.create()` mapping a raw `GenerateContentResponse` into an
  `LlmResponse` (C0121), the Gemini REST API request-body wire shape
  (`generate_content_request.rs`), and `Gemini::generate_content` — a
  real HTTP POST to the Gemini Developer API, including
  `_ResourceExhaustedError`'s 429-with-mitigation-link enhancement
  (C0127). Auth: an injected `client` is used as-is; otherwise an API key
  is read from `GOOGLE_API_KEY`/`GEMINI_API_KEY` (Gemini Developer API
  only — Vertex AI's own auth (ADC) is a distinct, deferred dependency
  decision, and returns a named "not supported yet, inject your own
  client" error rather than failing silently).
- **Fixed**: `adk_genai::content` (`Content`/`Part`/`FunctionCall`/
  `FunctionResponse`) now serializes multi-word fields as camelCase
  (`functionCall`, `inlineData`, `willContinue`, etc.), matching the
  source's `alias_generator=to_camel` pydantic config. This was a latent
  gap from Phase 3 batch 1 — the initial cut used bare field names —
  that batch 3 needed fixed to build a wire-accurate Gemini REST request
  body, but which also affects ADK's own event/session JSON.
- **New dependency, disclosed:** Phase 3 batch 3 discovered that
  `reqwest`'s *async* client requires a genuine `tokio` reactor
  underneath, and `rusty_tokio` (Phase 2's runtime) is a from-scratch,
  independent runtime — the two can't share a reactor (an async
  `reqwest::Client` call panics with "there is no reactor running" under
  `rusty_tokio`). Switched to `reqwest::blocking::Client` (which spins up
  its own private, self-contained tokio runtime internally, so it's safe
  under any ambient executor), driven via `rusty_tokio::spawn_blocking` so
  a slow HTTP call doesn't stall an async worker thread. Documented in
  the workspace `Cargo.toml` and `gemini.rs`'s module doc.
- `Gemini::prepare_live_connect_config`/`Gemini::live_api_version` (Phase
  3 batch 4, C0131 config-prep half): everything `Gemini.connect()` does
  to `llm_request.live_connect_config` before opening the actual Live
  WebSocket connection — tracking-header merging (gated on `http_options`
  already being present, matching the source's own behavior), forwarding
  `speech_config`/`tools`/`thinking_config`/`safety_settings`, the
  unconditional `system_instruction` assignment, and validating
  transparent session resumption is Vertex-AI-only. Extends
  `LlmRequest.live_connect_config` from an opaque placeholder to a real
  narrowed `LiveConnectConfigStub`.
- `GeminiLlmConnection` send-side methods (Phase 3 batch 5): `send_history`
  (audio-part filtering, Gemini-3.x response-trigger nudge),
  `send_content`/`send_content_partial` (function-response routing,
  Gemini-3.x realtime-text routing), `send_realtime` (Blob/ActivityStart/
  ActivityEnd/audio-stream-end dispatch) — C0135-C0138. `receive()`
  (C0139, a ~370-line stateful message-translation engine) is deferred to
  its own batch.
- **New dependency, disclosed:** `tungstenite` (not `tokio-tungstenite`)
  as the WebSocket transport for the Gemini Live API — checked every
  sibling Rusty-Mill repo (none exists); adopted for the same
  runtime-agnostic reason `reqwest::blocking` fit the REST transport
  (`rusty_tokio` can't share a reactor with anything assuming real
  tokio), bridged via `rusty_tokio::spawn_blocking`. `rustls-tls-webpki-roots`
  keeps TLS pure-Rust.
- **Fixed, disclosed:** the Gemini REST request body (`generate_content_request.rs`)
  and the new Live message envelopes now use `skip_serializing_if` to
  omit absent optional fields instead of sending them as explicit
  `null` — discovered while testing the Live envelopes; applied to both
  wire-format modules for consistency.
- **Adaptation, disclosed:** `Part.inline_data`/`file_data` narrowed from
  an opaque `Value` placeholder to `MediaBlobStub` (`mime_type` modeled,
  the rest of the payload flattened so round-tripping doesn't lose it) —
  needed to port `utils/content_utils.py`'s `is_audio_part`/
  `filter_audio_parts` for real. `BaseLlmConnection::send_realtime`'s
  `blob` parameter is retroactively upgraded from an opaque `Value` to
  the real `RealtimeInput` enum for the same reason.
- `GeminiLlmConnection::receive()` (Phase 3 batch 6, C0139): the Live
  event→`LlmResponse` translation engine — usage-metadata remap,
  cross-message grounding-metadata accumulation with index-offset
  merging, streamed text/thought aggregation, transcription streaming
  (persisted across `receive()` calls, model-variant-dependent),
  Gemini-3.x-variant tool-call buffering, session-resumption/
  voice-activity/GoAway passthrough. Split into a pure, directly-tested
  `process_message` core plus a thin real-socket read loop.
- **Scope decision:** the SSE-streaming half of `generate_content_async`
  (`stream=true`, `StreamingResponseAggregator`), context-cache
  integration, interactions-API delegation, the actual Live WebSocket
  handshake (the rest of C0131 — opening the connection to Google's Live
  endpoint), `_adapt_computer_use_tool`/`_preprocess_request` (C0132),
  redacted request/response logging, and `GeminiContextCacheManager` are
  deferred to further batches — see `crates/adk-models/src/gemini.rs`'s
  and `gemini_llm_connection.rs`'s module docs for exactly what's left
  and why.
- `OllamaLlm` — a real, directly-testable `BaseLlm` backend talking to a
  local Ollama server's native `/api/chat` endpoint, added at the user's
  explicit request alongside (not instead of) the Gemini backend work.
  **Not** a port of the source's `LiteLlm(BaseLlm)` class (Phase 10,
  C0557 — a universal wrapper around the third-party `litellm` package
  covering 14 providers with their own quirks, C0557-C0574); no
  manifest rows are marked `DONE` by this addition, since it covers
  substantially less. Model registration for `ollama/…`/`ollama_chat/…`
  (2 of the 14 provider regexes in C0560, excluding `ollama/gemma3.*` to
  match the source's own `Gemma3Ollama` carve-out — enforced in code
  rather than in the registry regex, since Rust's `regex` crate has no
  lookahead support and the source's own pattern uses one), and the
  `ollama_chat` content-flattening quirk from C0567. Tool-calling,
  streaming, and every non-Ollama provider stay out of scope. No Ollama
  server was reachable in the sandbox this was built in, so tests run
  against a local HTTP test server speaking Ollama's documented response
  shape — the same dependency-free pattern used for the Gemini
  transports.
- `GeminiContextCacheManager` (Phase 3 batch 7, C0140-C0143): the full
  explicit context-cache lifecycle — the `handle_context_caching` state
  machine (reuse-valid → invalid-cleanup-recreate → fingerprint-mismatch →
  fresh-fingerprint-only → no-prior-metadata → fresh-fingerprint-only),
  model-specific minimum cache-token floors (`gemini-2.5-*`→2048,
  `gemini-3*`→4096), SHA-256 cache-validity fingerprinting, gated cache
  creation (token-minimum + prefix-token estimation), best-effort cleanup,
  and request truncation to the uncached suffix. Built and tested
  standalone against a local mock `cachedContents` HTTP endpoint — wiring
  it into `Gemini::generate_content_async` (C0126) stays deferred, the
  same "build it standalone, wire it in later" split used for
  `GeminiLlmConnection` versus `Gemini::connect()`.
- **New dependency, disclosed:** `sha2` (RustCrypto) for the cache
  manager's fingerprint hash — checked every sibling Rusty-Mill repo (none
  has a hashing/crypto candidate); pure Rust, no system OpenSSL, same
  rationale as `regex`. Reused `rusty_time` (already a workspace
  dependency) to parse the Gemini API's RFC 3339 `expireTime` response
  field into a Unix timestamp.
- **Adaptation, disclosed:** the cache fingerprint's canonical JSON is a
  Rust-native SHA-256 digest over a deterministically field-ordered
  `Value` map, not the source's `json.dumps(sort_keys=True)` over a Python
  dict — it's only ever compared against a fingerprint this same code
  produced earlier, never against Python's, so only internal determinism
  matters. `LlmRequest.config` gained `tool_config`/`cached_content`
  fields (opaque placeholders) the cache manager needed to read/clear.
- Runnable demo examples for both backends —
  `cargo run -p adk-models --example gemini_demo` /
  `--example ollama_demo`. Each tries a real server first (a real Gemini
  API key from the environment; a real local Ollama at `OLLAMA_HOST`/
  `localhost:11434`) and falls back to a local one-shot mock server
  speaking the same wire shape when none is available, so both run with
  zero setup. Verified against the real Gemini API in this session
  (`gemini-3.5-flash-lite` returned a real model response) and against
  the mock fallback for both backends.
- `Gemini::apply_tracking_headers` (Phase 3 batch 8, closing C0133): the
  REST-path counterpart to `prepare_live_connect_config`'s tracking-header
  merge — unconditionally creates `llm_request.config.http_options` (if
  absent) and merges ADK's tracking headers into it, then attaches those
  headers explicitly on every real `generate_content` request so tracking
  survives even an injected `client` this port can't introspect for
  pre-existing default headers. `LlmRequest.config` gained a new
  `http_options: Option<HttpOptionsStub>` field (reusing the existing
  stub type from the Live-connect config).
- `debug_log::build_request_log`/`build_response_log` (Phase 3 batch 9,
  C0134): redacted debug-level request/response logging, wired into
  `Gemini::generate_content` under a lightweight `ADK_DEBUG_LOGGING`
  env-var gate (no logging framework has been adopted by this workspace
  yet). Redacts `inline_data`/`file_data`'s binary payload (keeping
  `mime_type`), `http_options.headers` (the credential-bearing field this
  port models), and — since `config.tools` isn't typed yet (C0116) —
  fully excludes `tools` from the config log and leaves the "Functions"
  section always empty, the same effect the source's own fallback
  exclusion branch produces, just taken unconditionally.
  `GenerateContentResponse` and its nested types gained a `Serialize`
  impl (previously deserialize-only) so a parsed response can round-trip
  into the log's raw-JSON dump.
- **Partial, disclosed:** wired `GeminiContextCacheManager` into
  `Gemini::generate_content` (Phase 3 batch 10, the non-streaming half of
  C0126) — invoked in the same place the source does, gated on
  `llm_request.cache_config.is_some() && !self.use_interactions_api`,
  populating the resulting cache metadata into the returned
  `LlmResponse`. The streaming half (populating cache metadata into a
  `StreamingResponseAggregator`'s responses) stays deferred alongside
  C0125's own SSE-streaming gap — there's no streaming path to populate
  yet, so C0126 stays `REQUIRED` in the manifest rather than `DONE` until
  that lands too.
- `Gemini::generate_content_stream` and `streaming_utils::StreamingResponseAggregator`
  (Phase 3 batch 11, closing C0125's SSE-streaming half and C0126's
  streaming half): a real `streamGenerateContent?alt=sse` call, parsed
  into Server-Sent-Events and aggregated into partial `LlmResponse`s plus
  one final aggregated response, wired into
  `BaseLlm::generate_content_async`'s `stream: true` branch (previously
  always an error). Cache metadata is populated only into the final
  aggregated response, matching the source. **Scope, disclosed:** only
  the source's "non-progressive" (legacy, text-only) aggregation mode is
  ported — its newer JSONPath-addressed partial-function-call-argument
  streaming mode is feature-flagged in the source and needs typed
  function-call/tool machinery (C0116, Phase 8) this port doesn't have
  yet. **Adaptation:** the aggregator returns a `Vec<LlmResponse>` per
  chunk rather than a true generator, and the full SSE body is read up
  front rather than incrementally — `generate_content_async`'s own
  contract already collects everything into one `Vec` before returning,
  so neither loses anything a caller could observe.
- **Phase 4 started**: `adk-flows` crate (`google.adk.flows`), depending on
  `adk-agents`, `adk-events`, and `adk-models` together — the crate-graph
  reason is disclosed in its own module doc (`adk-models` already depends
  on `adk-agents` for `ContextCacheConfig`, so `adk-agents` depending back
  on `adk-models` for `LlmAgent`'s model-resolution methods would make the
  two crates depend on each other).
  - `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor` (C0147): the
    processor interfaces `BaseLlmFlow`'s whole request/response pipeline
    is built from. `run_async` returns a boxed future resolving to
    `Result<Vec<Event>, ProcessorError>` rather than an `AsyncGenerator`,
    the same adaptation `BaseLlm::generate_content_async` already made.
  - `canonical_model`/`canonical_live_model` (C0080/C0090, partial): free
    functions (not `LlmAgent` methods, for the crate-graph reason above)
    resolving `LlmAgent.model` to a real `BaseLlm` via a new process-wide
    default `LlmRegistry` (`adk_models::registry::default_registry`,
    C0111 partial — pre-populated with `Gemini` and `OllamaLlm`, the two
    concrete backends this migration has actually built). Still
    `REQUIRED`, disclosed: ancestor-agent-chain fallback (`LlmAgent` isn't
    wired into `BaseAgent`'s tree yet), the source's memoization cache,
    and resolving `ModelRef::Instance` (a live `BaseLlm` instance passed
    directly rather than a name) — the last one is blocked on a deliberate
    crate-dependency restructuring (extracting `ContextCacheConfig` into a
    shared lower crate), not just more code.
  - `basic::build_basic_request` (C0168, partial): the `basic` request
    processor's full behavior — assembles `LlmRequest.model`/`config`/
    `output_schema`/`live_connect_config` from `LlmAgent`'s canonical
    settings and `RunConfig` (labels merging, http_options headers
    merging with mutation-leakage protection, the full live-connect
    surface including the Gemini-3.x-live affective-dialog/proactivity
    suppression). `LlmRequest.config`/`live_connect_config` gained several
    new opaque-placeholder fields sourced straight from `RunConfig`'s own
    same-named fields (`labels`, `response_modalities`,
    `output_audio_transcription`, `input_audio_transcription`,
    `realtime_input_config`, `explicit_vad_signal`, `translation_config`,
    `enable_affective_dialog`, `proactivity`, `history_config`,
    `context_window_compression`, `avatar_config`). **Scope, disclosed:**
    a free function, not yet a real `BaseLlmRequestProcessor` reading
    through `InvocationContext` — that needs `LlmAgent` wired into
    `BaseAgent`'s tree first (a separate, larger Phase 4 piece).
  - `identity::apply_identity` (C0169, partial): the `identity` request
    processor — appends "You are an agent. Your internal name is..."
    (plus a description sentence when set) unless the agent is in
    single-turn mode. Takes `agent_name`/`agent_description` as explicit
    parameters rather than reading them off `LlmAgent` (which has neither
    field yet, since it isn't wired into `BaseAgent`'s tree), same scope
    note as `basic`.
  - `instructions::build_instructions` (C0170, partial): the
    `instructions` request processor — global/static/dynamic instruction
    assembly, deferring to a new `instructions_utils::inject_session_state`
    (the regex-based `{state_var}`/`{artifact.name}` template engine) for
    state/artifact interpolation. `ReadonlyContext` gained an
    `artifact_service()` accessor and `Instruction::is_set` became `pub`
    to support it. **Scope, disclosed:** global_instruction reads the
    given agent's own field rather than walking to the tree root (no tree
    yet); `static_instruction` only interprets a plain string or an
    already-`Content`-shaped value, not the source's full `ContentUnion`
    transformer; Jinja2-mode template rendering is out of scope (an
    explicit opt-in nothing in this port ever requests).
  - `contents` module (C0181, C0183, C0186, C0187, C0189): the standalone
    event/content-list transforms `flows/llm_flows/contents.py`'s
    `_get_contents` pipeline is built from — invisible-part/empty-content
    filtering (C0189, DONE), branch-membership and event-kind filtering
    (`is_event_belongs_to_branch`, `is_direct_transfer`, `is_auth_event`,
    `is_request_confirmation_event`, `is_adk_framework_event`,
    `should_include_event_in_context` — C0183, partial: the visibility
    predicates are done, wiring the already-built
    `adk_events::rewind::apply_rewinds` into the top-level orchestration
    is deferred), function-call-id preservation mechanism
    (`copy_content_for_request` — C0181, partial: mechanism only, the
    backend-specific policy is deferred to Phase 10), and orphan-response
    dropping plus both async/latest function-response rearrangement
    passes (`drop_orphaned_function_responses`,
    `rearrange_events_for_latest_function_response`,
    `rearrange_events_for_async_function_responses_in_history`,
    `merge_function_response_events` — C0186/C0187, DONE; `bisect_left`
    translated as `Vec::partition_point`). Added `Part.tool_call`/
    `tool_response: Option<Value>` (opaque placeholders for a server-side
    tool call/result, distinct from `function_call`/`function_response`)
    for `is_part_invisible`'s "never invisible" exception. **Scope,
    disclosed:** the `_get_contents`/`_get_current_turn_contents`
    orchestration and `_ContentLlmRequestProcessor` wiring, cross-agent
    transcript fencing (C0184), and compaction-aware history
    reconstruction (C0185) are each deferred to their own dedicated
    batches.
  - `fencing` module (C0184): prompt-injection fencing for cross-agent
    transcript relaying, ported from `_fencing.py` in full —
    `quote_untrusted`/`elide_quote_markers` (marker wrapping with
    literal-marker elision so a payload can't forge its own end marker),
    `is_other_agent_reply` (live/bidi-aware other-agent detection), and
    `present_other_agent_message` (reformats another agent's event as
    fenced `[agent_name] said:`/`thought:`/tool-call/tool-result user
    context, relaying blob parts unfenced). Adaptation: `function_call.args`/
    `function_response.response` are rendered as compact JSON rather than
    Python's `str(dict)` repr, the same disclosed stand-in
    `instructions_utils::value_to_display_string` already uses.
    **Scope, disclosed:** the caller that decides *when* to apply this to
    an event — `contents.py`'s `_get_contents` orchestration — remains
    deferred with the rest of that orchestration.
  - `compaction` module (C0185), ported from `_content_compaction.py` in
    full: `process_compaction_events` (resolves overlapping compaction
    summaries, keeping only non-subsumed ones; materializes each
    surviving summary as a synthetic event attributed to the given agent
    name, filtering out raw events inside any kept compaction range) and
    `recover_compacted_function_calls` (re-injects a compacted
    function-call event — verbatim, to preserve parallel-call thought
    signatures — ahead of a surviving function-response that would
    otherwise be orphaned, along with any compacted sibling responses).
    Required narrowing `EventCompaction.compacted_content` (C0027) from a
    placeholder JSON `Value` to a real `adk_genai::content::Content`,
    now that Phase 3 landed the type it was always meant to become.
    Adaptation: the source's defensive `is None` checks on
    `EventCompaction`'s fields are omitted since Rust's type system
    already guarantees they're present.
  - `contents::get_contents`/`get_current_turn_contents` (C0181-C0183,
    C0188, C0189's top-level wiring; C0190 fully DONE): the top-level
    `_get_contents`/`_get_current_turn_contents` orchestration, calling
    into `apply_rewinds`, the visibility predicates, `crate::compaction`,
    `crate::fencing`, both function-response rearrangement passes, and
    `copy_content_for_request` in the same sequence as the source.
    `coalesce_transcription_event` (C0188) merges adjacent content-less
    transcription fragments into one text event; `build_task_input_user_content`
    rebuilds a task agent's originating delegation FC (or the invocation's
    `user_content` fallback) as its synthetic first turn. Adaptation:
    `copy_content_for_request` does a full Rust clone rather than the
    source's shallow-copy-for-mutation-safety optimization — a strictly
    safer superset this port doesn't need the performance trade for yet.
    **Scope, disclosed:** the `_ContentLlmRequestProcessor` itself — which
    chooses between the two entry points via `agent.include_contents`,
    computes `preserve_function_call_ids` from the agent's canonical
    model type, and wires in `_add_model_input_context_to_user_content`/
    `_add_instructions_to_user_content` — remains deferred pending
    `LlmAgent` wired into `BaseAgent`'s tree, the same blocker every other
    Phase 4 processor has disclosed.
  - `interactions` module (C0174, partial): the `interactions_processor`
    request processor's core logic — `is_event_in_branch`/
    `find_previous_interaction_state` finds the most recent branch-aware
    `interaction_id`/`environment_id` for the current agent, to enable
    stateful conversation chaining via the Gemini Interactions API.
  - `context_cache` module (C0175, partial): the `context_cache_processor`
    request processor's core logic — `find_cache_info_from_events` scans
    session history backward for the agent's most recent cache metadata
    (incrementing `invocations_used` when it's an active cache carried
    over from a prior invocation) and prompt token count;
    `apply_context_cache` assembles both into `LlmRequest`'s already-real
    `cache_config`/`cache_metadata`/`cacheable_contents_token_count`
    fields. Adaptation: `Event.cache_metadata` (opaque `Value`) is parsed
    back into a real `CacheMetadata` via its own `Deserialize` impl rather
    than `Event` holding a typed field directly — `adk-events` sits below
    `adk-models` in the crate graph and depending on it would cycle;
    `usage_metadata`'s `promptTokenCount` key is read directly since no
    typed `UsageMetadata` exists in this port yet.
  - Both new modules are free-function core logic only, not yet wired as
    real `BaseLlmRequestProcessor`s — same "needs `LlmAgent` in
    `BaseAgent`'s tree" scope note as `basic`/`identity`/`instructions`.
  - `output_schema` module (C0178, partial): the `_output_schema_processor`
    request processor's gating decision
    (`should_apply_output_schema_processor` — output_schema set, tools
    non-empty, the model can't honor both together, and not task mode),
    its instruction text, and the two standalone helpers that read back a
    completed structured response (`create_final_model_response_event`,
    `get_structured_model_response`). **Not** ported: actually injecting
    a `SetModelResponseTool` into the request — both the tool itself and
    `LlmRequest::append_tools` (C0116) need `BaseTool` (Phase 8), which
    doesn't exist in this port yet.
  - `agent_transfer` module (C0171, partial): the `agent_transfer`
    request processor's transfer-target computation and instruction-text
    generation — `get_transfer_targets` (sub-agents, then the parent and
    peers if the parent is itself LLM-orchestrated, each gated by the
    corresponding `disallow_transfer_*` flag and excluding single-turn/
    task-mode agents) and `build_transfer_instruction_body`/
    `build_transfer_instructions` (byte-for-byte parity with the source's
    own literal expected instruction text, verified against 2 tests
    copied from the source's test file). Adaptation: takes an `llm_mode`
    callback rather than reading `mode`/`disallow_transfer_to_*` straight
    off any `BaseAgent`, since this port's `BaseAgent` and `LlmAgent` are
    separate unfused types (disclosed in the module doc). **Not**
    ported: `_get_incompatible_builtin_tool_error` and building/attaching
    a real `TransferToAgentTool` — both need `BaseTool` (Phase 8), the
    same blocker `output_schema.rs` already discloses.
  - `request_confirmation` module (C0172, partial): the
    `request_confirmation` request processor's pure, tool-infrastructure-
    free dedup pre-pass — `get_original_function_call_args` (extracts the
    `originalFunctionCall` payload out of an `adk_request_confirmation`
    call's args) and `map_confirmation_to_original_fc_ids` (maps a
    confirmation call's id back to the original function-call id it
    confirms, so already-consumed confirmations can be dropped cheaply
    before expensive re-validation). **Not** ported: parsing a
    `ToolConfirmation`, resolving/validating the confirmed tool against
    session history, and re-executing it — all need `BaseTool`/
    `ToolConfirmation`/`ToolContext` (Phase 8/9), which don't exist in
    this port yet.
  - `functions_utils` module (C0196, partial): a slice of `functions.py`'s
    helpers — `merge_parallel_function_response_events` (concatenates
    parallel tool-call response events' content parts and deep-merges
    their `EventActions`, round-tripping through `Value` via
    `to_value`/`from_value` and generically deep-merging the resulting
    maps rather than hand-writing bespoke per-field rules — mirrors the
    source's own `model_dump`+`deep_merge_dicts`+`model_validate` exactly,
    including `render_ui_widgets` aggregating additively across events
    instead of last-wins) and the client function-call-id lifecycle
    helpers (`generate_client_function_call_id`,
    `populate_client_function_call_id`, `remove_client_function_call_id`,
    `get_long_running_function_calls`, `find_event_by_function_call_id`,
    `find_matching_function_call`). **Not** ported:
    `build_auth_request_event`/`generate_auth_event`/
    `generate_request_confirmation_event` — need `AuthConfig` (Phase 9).
  - `llm_flow` module (C0144, C0146, C0148-C0150, C0153, C0156, partial;
    C0157 DONE): `LlmFlow`, the first concrete `AgentBehavior` this port
    builds for an `LlmAgent` — a real, working (if narrowed) turn:
    `preprocess` (`basic`→`identity`→`instructions`→contents→
    `context_cache`, in order) → `call_model` (`BaseLlm::generate_content_async`,
    resolved once at construction rather than per-call) → `postprocess`
    (`finalize_model_response_event`, C0157 now DONE — a full,
    field-by-field faithful `LlmResponse`→`Event` shallow-copy). Verified
    end-to-end: `BaseAgent::run_async` driving a real `BaseAgent` wired
    with `LlmFlow` against a fake `BaseLlm`, no stubs on the
    `AgentBehavior` seam itself. **Scope, disclosed:** no multi-step
    tool-call loop, no `interactions_processor`/`preserve_function_call_ids`
    wiring (undetectable through the type-erased `Arc<dyn BaseLlm>`), no
    telemetry spans or before/after-model callback dispatch, no live mode
    (`run_live_impl` returns a clear "not implemented" error) — each maps
    to its own still-`REQUIRED` manifest row. `LlmFlow` resolving its
    model once at construction is a real (if narrow) instance of the
    memoization cache `canonical_model.rs` disclosed as missing.
  - `agent_transfer::get_agent_to_run` (C0159, DONE): resolves a
    `transfer_to_agent` target by name from the tree root
    (`root_agent()`/`find_agent()`), raising on an unknown target or a
    disallowed sibling transfer. Adaptation: parent-agent identity is
    compared by name rather than Pydantic-style object equality, since
    `BaseAgent` (wrapping a type-erased `Box<dyn AgentBehavior>`) has no
    `PartialEq` — sibling names are already expected unique within one
    tree (`BaseAgent::build` warns on duplicates).
### Changed
### Fixed
- `InMemoryArtifactService::save_artifact` (Phase 6, C0259) now panics
  on a malformed `artifact` value instead of silently substituting an
  empty `Part` — matching the source's `ensure_part`/`model_validate`
  raising a `ValidationError` rather than losing data quietly. Closes
  out C0259 (`ArtifactVersion`/`ensure_part`) and C0260
  (`BaseArtifactService`'s full 7-method abstract interface) as `DONE`
  — both already fully satisfied by the C0265 batch, just not yet
  reflected in the manifest. 1 new test.
### Security

<!-- ## [0.1.0] - YYYY-MM-DD
### Added
- Initial release -->
