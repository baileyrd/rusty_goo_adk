# Release Notes

<!--
Two variants, pick the one that fits this repo's actual unit of change:

1. No version tags yet (pre-1.0, nothing published) — track by PR instead, same way
   AISF does it: one entry per merged PR against main, reverse chronological, each
   linking to its PR and (where one exists) to the doc that covers the change in full
   detail. Use "## PR #N — <summary>" headers.

2. Actual version tags exist — use "## vX.Y.Z - YYYY-MM-DD" headers instead, each
   linking to the PRs it shipped and a compare link to the previous tag. Add an
   "### Upgrade notes" subsection under any entry with a breaking change.

Either way, keep the tone AISF's file uses: bolded category tags inline in the
bullet (**Added:** / **Changed:** / **Fixed:**), not separate subheaders per
category — and state known limitations or deliberate scope cuts plainly instead of
leaving them implied.
-->

Notable changes to this repo, one entry per merged PR against `main`, newest first.

---

## PR #TBD — `auth/oauth2_discovery.py`/`runners.py`: two small data-only ports (C0533, C0835)
**2026-08-24** · (link added once this PR is opened)

Two small, unrelated capabilities landed together.

- **Added:** `crates/adk-agents/src/oauth2_discovery.rs` —
  `AuthorizationServerMetadata` (RFC8414) and `ProtectedResourceMetadata`
  (RFC9728), C0533's two data models. Field names stay literal
  snake_case (both RFCs specify snake_case wire fields themselves,
  unlike this crate's Google-genai-facing camelCase types); the
  source's `@experimental` decorator warns on every *construction* (not
  class definition) — ported via an explicit `::new()` calling
  `warn_experimental`, the same `ResumabilityConfig::new` precedent.
  `OAuth2DiscoveryManager`'s actual `.well-known` fetching/mix-up-attack
  validation logic stays its own rows (C0534/C0535) — needs an async
  HTTP client this port hasn't adopted anywhere yet.
- **Added:** `adk_runners::runner::get_function_responses_from_content`
  (C0835) — extracts every `FunctionResponse` from a `Content`'s parts,
  `[]` for `None`/no-parts, reusing the already-real
  `Content::get_function_responses`. Built ahead of its own caller
  (`_resolve_invocation_id`, C0855 — needs resumability wiring `Runner`
  doesn't have yet), the same precedent `session_util.rs`/`artifact_util.rs`
  already established.
- **Scope:** 7 new tests (4 for the OAuth2 discovery models, 3 for
  `get_function_responses_from_content`).

---

## PR #TBD — `sessions/base_session_service.py`: `GetSessionConfig` (C0207)
**2026-08-24** · (link added once this PR is opened)

Ports the session-history-bounding config every `SessionService`
backend is expected to honor identically.

- **Added:** `adk_agents::services::GetSessionConfig` —
  `num_recent_events`/`after_timestamp`, both optional and composable.
- **Added:** `adk_agents::services::SessionService::get_session_with_config`
  — a new default trait method (additive; `get_session`'s
  already-shipped signature and its ~26 existing call sites are
  untouched) that defers to `get_session` and applies the trimming
  generically in one shared place. `num_recent_events` tail-slices
  first (`Some(0)` or negative returns no events); `after_timestamp`
  then drops everything older — replicating the source's own Python
  truthiness quirk where `after_timestamp: Some(0.0)` is treated the
  same as unset, not "return nothing." Every current implementer
  (`InMemorySessionService`, `NoopSessionService`, a test-only
  `FakeSessionService`) gets correct, identical behavior for free —
  matching the manifest row's own "every backend must honor
  identically" framing more faithfully than a per-backend override
  would.
- **Not yet wired:** `RunConfig.get_session_config` is still an opaque
  `Value` placeholder (C0875), so no `adk-runners` call site threads a
  real `GetSessionConfig` through this new method yet — built ahead of
  its own caller, the same precedent `session_util.rs`/`artifact_util.rs`
  already established. Updated the citation text on C0873/C0891/C0914's
  manifest evidence and two `runner.rs` doc comments to say so
  precisely, rather than leaving them saying `GetSessionConfig` doesn't
  exist at all.
- **Scope:** 7 new tests.

---

## PR #TBD — `base_llm_flow.py`/`base_llm.py`: postprocess no-content guards + streaming-contract evidence (C0156, C0102)
**2026-08-24** · (link added once this PR is opened)

Two small, unrelated gaps found while scoping the next batch, landed
together since both are quick.

- **Added:** `adk_flows::llm_flow::apply_no_content_error` — converts a
  non-partial, non-streaming response that finished with STOP but
  carries no content into a `MODEL_RETURNED_NO_CONTENT` error (an
  otherwise-silent empty final response made actionable instead).
  Excluded for SSE streaming, where a terminal finish-only chunk
  legitimately follows content already streamed in earlier chunks. An
  existing `error_message` is preserved over the default.
- **Added:** `adk_flows::llm_flow::should_skip_empty_response` — skips
  the model-response event entirely when a response carries no
  content, no error code, no interruption, and no grounding metadata.
  Both guards run in `postprocess`, in the source's own sequential
  order: a response the no-content-error check rewrites into an error
  is never skipped by the empty-response check afterward.
- **Confirmed, not new work:** `Gemini::generate_content_async` (the
  `BaseLlm` trait method, C0102's dual-mode streaming contract) was
  already correctly implemented, and — checking the existing test
  suite more carefully than the last scoping pass did — already
  partly tested directly via the trait method (error-flattening for
  `stream=false`, and the `stream=true` partials-then-final-aggregate
  shape). The one real gap: a `stream=false` *success* case asserting
  "exactly one response, not partial." Added that one test and closed
  the manifest row's evidence.
- **Scope:** 15 new tests total (14 for the `postprocess` guards, 1 for
  `generate_content_async`).

---

## PR #TBD — `functions.py`: multimodal tool-result extraction (C0195)
**2026-08-24** · (link added once this PR is opened)

Ports the piece of `functions.py` that lets a tool hand back an image,
audio clip, or document instead of encoding it into text — a tool
result is otherwise required to be JSON-serializable.

- **Added:** `adk_flows::functions_media::{as_function_response_part,
  extract_media_from_entry, extract_multimodal_parts}` — pulls media
  out of a tool's returned dict/list/tuple, bounded to one level of
  container nesting (`_MAX_MEDIA_CONTAINER_DEPTH`), matching the
  source's exact recursion shape (a container found at that depth is
  kept as-is rather than recursed into a second time) and its
  "`remaining or {}`" empty-container coercion.
- **Added:** `adk_genai::content::FunctionResponse::parts:
  Option<Vec<FunctionResponsePart>>` — a new, additive field (every
  existing construction site updated), and `FunctionResponsePart`
  itself, a new type reusing `MediaBlobStub` for the same
  `mime_type`-real/rest-opaque shape `Part::inline_data`/
  `Part::file_data` already use.
- **Wired:** `adk_flows::functions::build_function_response_content`
  now extracts media before coercing the remainder to a dict, and
  attaches it to the built `FunctionResponse::parts`.
- **Disclosed adaptation:** the source's `_as_function_response_part`
  checks `isinstance(value, types.Part)` — a real Python
  object-identity check. This port's tools only ever return an
  already-JSON-shaped `Value` (there's no way to embed a typed `Part`
  object inside an arbitrary result tree), so the check here is
  necessarily structural instead: a `Value::Map` that round-trips
  through `rusty_serde::json::from_value::<Part>` (the same
  opaque-payload round-trip convention `load_artifacts_tool.rs`
  already uses) and carries a populated `inline_data`/`file_data`
  counts as media. This is looser than the source's identity check — a
  plain dict that happens to carry `inlineData`/`fileData`-shaped keys
  would also match here — but it's the only representation a tool
  constructing media in this port actually has to work with.
- **Not ported:** computer-use image decoding
  (`_try_decode_computer_use_image`) — needs `ComputerUseTool`, not
  built in this port.
- **Scope:** 13 new tests.

---

## PR #TBD — `runners.py`: two module-level helper closures (C0833, C0836)
**2026-08-24** · (link added once this PR is opened)

Two rows found already fully implemented while scoping the next batch —
closed with proper evidence rather than left understating real coverage.

- **C0833** (`_notify_run_error`): already wired — `Runner::run_async_with_config`'s
  one unhandled-`agent.run_async`-error path already calls
  `PluginManager::run_on_run_error_callback` as a best-effort notifier,
  and the original run error always still propagates afterward. The
  source's "logs+suppresses any exception the callback itself raises"
  behavior is N/A by construction, not by omission: this port's plugin
  callback trait methods return `()`, not `Result` — there's no
  exception channel for a callback to raise through in the first place.
- **C0836** (`_apply_run_config_custom_metadata`): already implemented
  and correct (`run_config.custom_metadata` merges in first,
  `event.custom_metadata`'s own keys layered on top so they win on
  conflict; a no-op when the config side is `None` or `Some` but empty)
  — but only ever exercised indirectly through `run_async` tests. Added
  3 direct unit tests covering each case by name.
- **Scope:** 3 new tests, no behavior changes.

---

## PR #TBD — `runners.py`: `Runner::run` sync wrapper (C0877, C0878, C0879, C0880)
**2026-08-24** · (link added once this PR is opened)

Ports the local-testing/convenience-only synchronous entrypoint, so a
caller that isn't already inside `async` code (or is inside one and
can't `await`) can still drive the runner.

- **Added:** `adk_runners::Runner::run(user_id, session_id, new_message,
  state_delta, run_config)` — spins a dedicated OS thread via
  `adk_platform::thread::create_thread` (C0005) running its own
  `rusty_tokio::Runtime`, mirroring the source's `asyncio.run(...)` on a
  background thread. `Runner` now derives `Clone` so an owned copy can
  move onto that thread — every field is already a cheap `Arc`/value
  clone. New dependency: `adk-runners` now depends on `adk-platform`.
- **Verified:** callable from inside an already-running `rusty_tokio`
  runtime without deadlocking — the actual reason the source needs a
  separate thread rather than just calling `asyncio.run()` on the
  caller's own thread.
- **Disclosed narrowing:** the source bridges events one at a time
  through a blocking `queue.Queue`, preserving "events yielded before a
  failure ARE yielded before the exception" (C0879) across the thread
  boundary. This port's `run_async_with_config` already collapses to a
  single batched `Result<Vec<Event>, RunnerError>` rather than a stream
  — an already-established narrowing from the source's own async
  generator — so `run` collapses to one background computation whose
  whole result becomes available at once; there's no partial-events
  case left to preserve.
- **C0878:** Rust has no `Exception`/`BaseException`/`SystemExit`/
  `CancelledError` hierarchy to distinguish. The structural analog is
  `JoinHandle::join`'s own `Result`: a normal `Err(RunnerError::AgentRun(..))`
  from the background computation surfaces unchanged; a *panic* on the
  background thread — this port's only abnormal-termination case — is
  caught by `join()` and re-wrapped into `RunnerError::AgentRun` naming
  the panic payload, rather than re-panicking the calling thread.
- **C0880:** closes a small pre-existing gap found while wiring this up
  — `run_async_with_config` didn't accept `state_delta` at all before
  this PR (the source's own `run_async` does). Added as a new
  `Option<HashMap<String, Value>>` parameter, applied onto the appended
  user event's `actions.state_delta` when non-empty (mirroring
  `_append_new_message_to_session`'s own `if state_delta:` truthiness
  check), then forwarded straight through from `run`.
- **Housekeeping:** corrected a stale C0886 evidence note (compaction
  was closed by an earlier PR this session, but its "not ported"
  wording was never updated) and a stale module-doc line still listing
  `run` under "not ported this batch".
- **Scope:** 5 new tests.

---

## PR #TBD — `runners.py`: invocation-context config wiring (C0918 N/A, C0919 partial)
**2026-08-24** · (link added once this PR is opened)

Closes a real gap found while scoping the next batch: `Runner` already
held `context_cache_config`/`resumability_config`/`events_compaction_config`,
but `run_async_with_config` never patched them onto the
`InvocationContext` it built — so the agent, plugin callbacks, and every
other consumer of the context never actually saw them.

- **Fixed:** `adk_runners::Runner::run_async_with_config` now also
  copies `context_cache_config`/`resumability_config`/
  `events_compaction_config` onto the built `InvocationContext`,
  alongside the four services it already patched.
- **Confirmed, not new code:** `adk_agents::invocation_context::InvocationContextBuilder`
  already plays the role of the source's `_create_invocation_context`
  factory (C0918) — this port has no subclassing to override, which is
  that factory's only documented purpose in the source, so there's
  nothing left to port.
- **Disclosed narrowing:** the source's `support_cfc` (Compositional
  Function Calling, experimental) branch — validating the resolved
  model name and force-installing a `BuiltInCodeExecutor` on the agent
  — has nothing to port onto: `LlmAgent.code_executor` is still an
  opaque `Value` placeholder (C0088), the same architecture-investment
  blocker already disclosed for C0092/C0429 (`GoogleSearchAgentTool`/
  `canonical_tools`).
- **Scope:** 2 new tests, via a new `ConfigCapturingBehavior` test
  double that captures the configs an `InvocationContext` actually
  carried into an agent run.

---

## PR #TBD — `runners.py`: `Runner::run_debug` (C0911, C0912, C0913, C0914 N/A)
**2026-08-24** · (link added once this PR is opened)

Ports the debugging/experimentation convenience method developers use to
quickly exercise an agent without dealing with session management,
content formatting, or event streaming directly.

- **Added:** `adk_runners::Runner::run_debug(user_messages)` — defaults
  `user_id`/`session_id` to the source's own `"debug_user_id"`/
  `"debug_session_id"` literals (reusing them across calls continues the
  same debug conversation, intentional and documented) and every other
  parameter to the source's own defaults.
- **Added:** `adk_runners::Runner::run_debug_with_config(user_messages,
  user_id, session_id, run_config, quiet, verbose)` — the full-control
  form. Rust has no keyword-argument defaults, so the source's single
  method with defaulted kwargs splits into two, the same split
  `run_async`/`run_async_with_config` already established.
- **Added:** `adk_runners::runner::DebugMessages` — normalizes a bare
  `&str`/`String` or a `Vec<&str>`/`Vec<String>` to the message list,
  mirroring the source's `str | list[str]` parameter.
- **Behavior:** session lookup is unconditional get-or-create — it
  bypasses `Runner::with_auto_create_session` entirely, unlike
  `run_async`'s own session handling (C0912). Drives
  `run_async_with_config` once per message and prints each produced
  event via `adk_events::debug_output::print_event` (already DONE)
  unless `quiet`, returning the full flat event list across *every*
  message, not just the last (C0913).
- **Disclosed narrowing:** the source also logs `"Created new
  session"`/`"Continue session"`/`"User > %s"` via Python's `logging`
  module at various points; this port has no logging framework adopted
  anywhere in this crate, so those log lines have no destination —
  only the actual stdout output (`print_event`) is preserved. C0914
  (forwarding `run_config.get_session_config`) is N/A: this port's
  `SessionService::get_session` has no config parameter to forward one
  to (an already-disclosed narrowing from `Runner::get_or_create_session`).
- **Housekeeping:** corrected two stale manifest/doc notes touched while
  in this file — C0924's evidence understated `Runner::close` (it
  already closes both the plugin manager and the session service, not
  just the session service); C0925 (`__aenter__`/`__aexit__`) is now
  disclosed N/A, since Rust has no async-context-manager protocol and
  `Drop` cannot run an async close.
- **Scope:** 6 new tests.

---

## PR #TBD — `apps/compaction.py`/`runners.py`: post-invocation compaction trigger (C0293, C0871, C0872)
**2026-08-24** · (link added once this PR is opened)

Ports the last piece of `apps/compaction.py` and wires it into
`Runner` — closing out the compaction feature this session's earlier
PRs built the pure logic for.

- **Added:** `adk_flows::apps_compaction::{run_compaction_for_token_threshold,
  run_compaction_for_sliding_window}` — the two trigger entrypoints,
  narrowed to `Option<Event>` (the source's `AsyncGenerator[Event, None]`
  never yields more than one) and to take `agent`/`config`/raw session
  events directly rather than an `App`. Neither performs the append
  itself — the caller does, the same split `Runner::rewind_async`
  (prior PR) already established.
- **Added:** `adk_runners::runner::{Runner::events_compaction_config,
  Runner::with_compaction_trigger, CompactionTrigger}` — a new config
  field (sourced only via `Runner::from_app`, matching
  `context_cache_config`'s rule) and a trait-object extension point
  for the actual trigger logic, called right after every produced
  event has already been appended and just before `run_async_with_config`
  returns.
- **Load-bearing architecture note, disclosed:** this port's crate
  layering — `adk-tools` and `adk-flows` both depend on `adk-runners`,
  never the reverse — means `adk-runners` itself cannot call
  `run_compaction_for_sliding_window` without creating a crate-graph
  cycle. `adk_flows::apps_compaction::with_real_compaction_trigger`
  (added alongside the trigger functions, in a crate that already
  depends on `adk-runners`) wires the real implementation into any
  `Runner`. A `Runner` built via `from_app` alone has compaction
  *configured* but stays inert until that wiring is applied — the same
  "overridable behavior → injected trait object" pattern this crate
  already uses for `ArtifactService`/`SessionService`.
- **Scope:** new dependency — `adk-flows` now depends on `adk-runners`
  (verified no cycle: `adk-runners` depends on neither `adk-flows` nor
  `adk-tools`). 13 new tests (11 for the decision logic, 2 end-to-end
  for the `Runner` wiring).

---

## PR #TBD — `runners.py`: `Runner::rewind_async` (C0891, C0892, C0893, C0894)
**2026-08-24** · (link added once this PR is opened)

Ports session rewind — a forward-only append of a reversing delta
event, never a destructive truncation of session history.

- **Added:** `adk_runners::Runner::rewind_async(user_id, session_id,
  rewind_before_invocation_id)` — gets-or-creates the session (honoring
  `auto_create_session`, C0894), linear-scans for the target
  invocation's first event, computes reversing state/artifact deltas,
  and appends a single new user-authored event carrying them.
  `adk_events::rewind::apply_rewinds` (already DONE) is what
  interprets `rewind_before_invocation_id` downstream.
- **Added:** new `crates/adk-runners/src/rewind.rs` —
  `compute_state_delta_for_rewind` (C0892) replays state deltas
  strictly before the rewind point to reconstruct state at that point
  (an explicit `Value::Null` in a historical delta is a tombstone),
  then diffs against current state; `compute_artifact_delta_for_rewind`
  (C0893) restores each changed artifact as a brand-new version
  (rewind never rewrites history), marking artifacts that didn't exist
  yet at the rewind point as inaccessible via the same
  `rusty_serde::json::to_value`-of-a-`Part` representation
  `save_files_as_artifacts_plugin.rs` already established.
- **Disclosed:** no `run_config`/`GetSessionConfig` parameter —
  `Runner::get_or_create_session` doesn't thread one, the same
  already-disclosed C0873 narrowing.
- **Corrected:** `runner.rs`'s module doc previously listed
  `rewind_async` among the pieces still needing infrastructure — it's
  built now.
- **Scope:** no new dependency. 14 new tests.

---

## PR #TBD — `apps/compaction.py`: summarizer init + safe-window logic (C0290, C0291, C0292)
**2026-08-24** · (link added once this PR is opened)

Continues the `apps_compaction.rs` file from the prior PR, porting the
rest of the module's pure logic.

- **Added:** `adk_flows::apps_compaction::ensure_compaction_summarizer`
  — resolves an `EventsCompactionConfig`'s summarizer: the existing
  one if set, otherwise a new `LlmEventSummarizer` built from the
  agent's already-resolved canonical model (`LlmFlow::model`). The
  `isinstance(agent, LlmAgent)` check reuses the `agent.as_any()
  .downcast_ref::<LlmFlow>()` pattern `instructions.rs` already
  established.
- **Added:** `adk_flows::apps_compaction::{events_to_compact_for_token_threshold,
  longest_self_contained_prefix, safe_token_compaction_split_index}` —
  the token-threshold compaction candidate selection, the
  open-obligation-tracking safe-prefix logic, and the
  orphan-avoiding retention split, all ported in full.
- **Disclosed adaptation:** the source mutates `config.summarizer` in
  place; this port's `EventsCompactionConfig` has no interior
  mutability, so `ensure_compaction_summarizer` resolves and returns
  instead — in-place caching (if wanted) is left to whatever wires
  C0293.
- **Scope:** C0293 (the two `Runner`-facing trigger entrypoints) needs
  real `App`/`Runner` wiring and a `BaseSessionService::append_event`
  call — a genuinely larger, separate batch, deliberately left for
  later.
- **Scope:** no new dependency. 14 new tests.

---

## PR #TBD — `apps/compaction.py`: dedup + token estimation (C0288 partial, C0289)
**2026-08-24** · (link added once this PR is opened)

Ports the pure, self-contained pieces of the sliding-window/token-
threshold compaction trigger logic.

- **Added:** `adk_flows::apps_compaction::latest_compaction_event` —
  the latest non-subsumed compaction event by stream order (a range
  fully contained by, or identical to, a later one is subsumed; an
  identical-range tie breaks toward the later event).
- **Added:** `adk_flows::apps_compaction::{estimate_prompt_token_count,
  latest_prompt_token_count}` — an approximate prompt token count (4
  chars/token) over the real `adk-flows::contents::get_contents`
  prompt-assembly path, and a lookup that prefers a real
  `usage_metadata.promptTokenCount` over the estimate.
- **Disclosed placement:** lands in `adk-flows`, distinct from this
  crate's own `compaction.rs` (`flows/llm_flows/_content_compaction.py`,
  C0185) — a different source module that happens to need the same
  subsumption-detection shape for a different purpose (filtering
  contents, not deciding what to summarize next).
- **Disclosed adaptation:** `_count_chars_in_content`'s `json.dumps`-
  with-`str()`-fallback narrows to just the `json.dumps` path, since
  this port's `args`/`response` are already-typed `BTreeMap` values
  and always serializable — there's no failure case to fall back from.
  `usage_metadata.prompt_token_count` reads through the
  `"promptTokenCount"`-key convention `cache_performance_analyzer.rs`/
  `context_cache.rs` already established.
- **Scope:** the OTel-traced summarization wrapper (C0288's other
  half) and C0290-C0293 (summarizer lazy-init, the safe-window split
  logic, and the two `Runner`-facing trigger entrypoints) are
  deliberately left for a follow-up batch — this one stays to the pure
  functions with no `Runner`/`App` wiring dependency.
- **Scope:** no new dependency. 11 new tests.

---

## PR #TBD — `apps/llm_event_summarizer.py`: `LlmEventSummarizer` (C0286, C0287)
**2026-08-24** · (link added once this PR is opened)

Ports the LLM-based summarizer for sliding-window event compaction.

- **Added:** `adk_tools::llm_event_summarizer::LlmEventSummarizer` —
  implements `adk-agents`'s `BaseEventsSummarizer` trait (C0285).
  Formats a conversation history (text, thoughts — skipping ones
  emitted by a prior compaction event so its reasoning doesn't leak
  into the next summary, tool calls/responses truncated at 2000
  chars), drives one non-streaming LLM call, and returns a new `Event`
  carrying an `EventCompaction` action, forcing `role='model'` on the
  summary content and `author='user'` on the synthesized event.
- **Disclosed placement:** lands in `adk-tools`, not `adk-agents`
  (where `BaseEventsSummarizer` itself lives) — this type needs a real
  `adk-models::BaseLlm`, and `adk-models` already depends on
  `adk-agents`, so `adk-agents` adding `adk-models` as a dependency
  would be a crate-graph cycle. `adk-tools` already depends on all
  three crates needed. Same supporting-crate placement
  `forwarding_artifact_service.rs` (C0489, prior PR) already used.
- **Disclosed adaptation:** `args`/`response` formatting uses
  `rusty_serde::json::to_string` instead of Python's `str()` — the
  same divergence `adk-events::debug_output` (C0933) already
  established for the same two fields.
- **Corrected:** `app_configs.rs`'s module doc previously claimed the
  compaction machinery was "still LLM-blocked" — only the *trigger*
  logic deciding when to compact (`apps/compaction.py`, C0288/C0289)
  is still unbuilt; the summarizer itself is done.
- **Scope:** no new dependency. 10 new tests.

---

## PR #TBD — `tools/_forwarding_artifact_service.py`: `ForwardingArtifactService` (C0489 partial)
**2026-08-24** · (link added once this PR is opened)

Gives a nested `AgentTool` run access to the parent's real artifact
backend, closing a gap disclosed since `AgentTool` first landed.

- **Added:** `adk_tools::forwarding_artifact_service
  ::ForwardingArtifactService` — implements the port's `ArtifactService`
  trait, routing every read/write straight through to the parent tool
  context's own real backend under the parent's own
  `app_name`/`user_id`/`session_id` (never the caller-supplied ones,
  matching the source's `del app_name, user_id, session_id`).
  `AgentTool::run_async` installs one on the nested `Runner` whenever
  the parent has a real artifact service of its own.
- **Disclosed:** the source updates the parent's `artifact_delta`
  action synchronously, inline, by awaiting the parent `ToolContext`'s
  own async `save_artifact` method (which needs `&mut self`). This
  port's `ArtifactService` trait is fully synchronous and `&self`-only,
  so a `ForwardingArtifactService` can't hold a live mutable borrow of
  the parent `Context` across the whole nested run. Saved versions
  instead accumulate in a shared map, merged into the parent's
  `artifact_delta` once the nested run completes — the same
  post-hoc-merge idiom `agent_tool.rs` already uses for state deltas.
  Reads and the write itself still happen live against the parent's
  real backend; only the delta bookkeeping is deferred.
- **Scope:** no new dependency. 8 new tests (6 in the new module, 2
  end-to-end in `agent_tool`).

---

## PR #TBD — `runners.py`: `Runner::from_app` (C0846, C0847, C0848, C0849, C0850)
**2026-08-24** · (link added once this PR is opened)

Wires the `App` model shipped last PR into `Runner`, additively.

- **Added:** `adk_runners::Runner::from_app(app, app_name_override,
  session_service) -> Result<Runner, PluginManagerError>` — the single
  normalization path from a resolved `App` to a `Runner`. Derives
  `context_cache_config`/`resumability_config` from the app (never
  accepted as direct constructor arguments, matching the source) and
  folds `app.plugins` into the registered plugin set via the existing
  `Runner::with_plugin`, so a duplicate plugin name surfaces the same
  error it would through that method directly. `app_name` defaults to
  `app.name`; the override parameter matches the source's `app_name or
  app.name`.
- **Scope:** `Runner::new`'s already-shipped signature is untouched —
  this is a second, additive constructor, not a breaking change to the
  first.
- **Closed as N/A:** C0847 (`_enforce_app_name_alignment`/
  `_warn_uncached_agent_transfer`) and C0850 (the deprecated
  `_validate_runner_params` back-compat wrapper) both depend on
  `_infer_agent_origin` (C0851, already N/A — no Rust module-path
  reflection) or on logging machinery not adopted in this port; C0848
  (`_require_root_agent`) is N/A because `Runner::agent` is always a
  concrete `BaseAgent` here, never a bare node.
- **Corrected:** `runner.rs`'s own module doc previously claimed the
  `App`-dependent rows were N/A "no `App` type exists here" — `App`
  landed last PR (C0279/C0280); the doc now reflects that only the
  agent-origin-reflection/back-compat pieces stay N/A.
- **Scope:** no new dependency. 6 new tests.

---

## PR #TBD — `apps/app.py`: `App` model (C0279, C0280)
**2026-08-24** · (link added once this PR is opened)

Ports the top-level `App` container — the entry point binding a root
agent plus app-wide settings (plugins, event compaction, context
caching, resumability).

- **Added:** `adk_agents::app::{App, validate_app_name, AppError}`.
  `App::new(name, root_agent)` validates the name and defaults every
  other field empty/unset; `.with_plugin`/`.with_events_compaction_config`/
  `.with_context_cache_config`/`.with_resumability_config` builders set
  the rest, reusing the four already-DONE config/plugin types directly.
- **Narrowed:** `root_agent` goes from the source's
  `Union[BaseAgent, BaseNode, None]` to `BaseAgent`-only and becomes a
  required constructor argument rather than an `Option` — the
  `BaseNode`/workflow-graph engine (C0298-C0306) isn't built in this
  port, and the source's own `_validate` model-validator already
  rejects a `None` root_agent, so this enforces the same invariant at
  the type level instead of at runtime.
- **Disclosed:** `validate_app_name` is a new, distinct validator from
  `base_agent::validate_name` — the source's app-name regex
  (`^[a-zA-Z][a-zA-Z0-9_-]*$`) additionally permits hyphens, which
  agent names don't.
- **Scope:** deliberately not wired into `Runner`'s constructor this
  batch (C0840-C0850) — that would change `Runner::new`'s already-shipped
  signature; left for a follow-up batch once `App` exists and can be
  reviewed on its own. No new dependency. 7 new tests.

---

## PR #TBD — `runners.py`: `InMemoryRunner` (C0926)
**2026-08-24** · (link added once this PR is opened)

Ports the testing/development convenience `Runner` subclass, and
corrects a stale module doc along the way.

- **Added:** `adk_runners::Runner::in_memory(agent) -> Runner` — narrowed
  from a `Runner` subclass to a constructor, matching `Runner`'s own
  C0841 narrowing (no `App`/bare-node union exists for a subclass to be
  an alternative *of*). Pre-wires `InMemorySessionService`/
  `InMemoryArtifactService`/`InMemoryMemoryService`; `credential_service`
  stays unset, matching the source. `app_name` defaults to the literal
  `"InMemoryRunner"` unconditionally (the source's app-vs-app_name
  conditional doesn't apply — there's no `App` here).
- **Disclosed:** `plugins`/`plugin_close_timeout` aren't forwarded as
  separate constructor parameters — already reachable through `Runner`'s
  existing `.with_plugin`/`.with_plugin_close_timeout` builders.
- **Corrected:** `runner.rs`'s own module doc previously claimed
  `InMemoryArtifactService`/`InMemoryMemoryService` were "neither built
  yet" — both have been DONE for several batches.
- **Scope:** no new dependency. 2 new tests.

---

## PR #TBD — `BaseLlm::as_any` downcast + real `interactions_processor` wiring (C0174)
**2026-08-24** · (link added once this PR is opened)

Extends the `AsAny` downcast pattern PR #104 established for
`AgentBehavior` to `BaseLlm`, and uses it to close a disclosed gap in
`LlmFlow::preprocess`.

- **Added:** `adk_models::base_llm::AsAny` — lets `LlmFlow` detect
  whether its resolved `Arc<dyn BaseLlm>` is a `Gemini`, without
  `adk-models` needing to know about `adk-flows`. Purely additive: every
  existing `BaseLlm` implementor needs zero changes.
- **Closed:** `interactions_processor` (C0174 DONE) — `LlmFlow::preprocess`
  now gates on the downcast plus `Gemini::use_interactions_api`, calling
  `interactions::find_previous_interaction_state` to set
  `LlmRequest::previous_interaction_id` when the branch-aware lookup
  finds one. This wiring needed no `InvocationContext.agent`-resolution
  fix (unlike other Phase 4 processors) — `LlmFlow` already owns
  concrete `self.llm_agent`/`self.model` fields directly.
- **Corrected** 2 stale module docs (`llm_flow.rs`, `interactions.rs`)
  that described this as blocked on "no downcasting mechanism," now
  that one exists.
- **Disclosed, still correctly deferred:** `preserve_function_call_ids`
  (C0181) is *not* unlocked by this same mechanism — it needs detecting
  Anthropic/LiteLLM/OpenAIResponsesLlm backends, three of which don't
  exist in this port at all yet (only `Gemini`/`Ollama` do), so the
  downcast alone has nothing to downcast to for that row.
- **Scope:** no new dependency. 4 new tests (3 in `adk-flows` covering
  the gate-on/gate-off/non-Gemini cases, 1 `AsAny` regression test in
  `adk-models`).

---

## PR #TBD — `tools/vertex_ai_search_tool.py`: `VertexAiSearchTool` (C0433)
**2026-08-24** · (link added once this PR is opened)

Ports the last unstarted member of the built-in-grounding-tool family
(alongside `GoogleSearchTool`/`EnterpriseWebSearchTool`/
`GoogleMapsGroundingTool`/`UrlContextTool`, all already landed).

- **Added:** `adk_tools::vertex_ai_search_tool::{VertexAiSearchTool,
  VertexAiSearchConfig}` (C0433 DONE) — mutual-exclusivity constructor
  validation (`data_store_id`/`search_engine_id`, and `data_store_specs`
  requiring `search_engine_id`) and the full populated
  `{"retrieval":{"vertexAiSearch":{datastore, dataStoreSpecs, engine,
  filter, maxResults}}}` config shape.
- **Adapted:** the source's subclass-based dynamic-filter customization
  point (override `_build_vertex_ai_search_config` to set a per-request
  filter from session state) becomes an optional closure field
  (`with_config_builder`) — the same "overridable Python method →
  closure field" adaptation `base_agent.rs`'s `AgentCallback` already
  established for `before_agent_callback`/`after_agent_callback`.
- **Disclosed** (shared with every sibling built-in grounding tool): an
  unsupported model simply doesn't get the tool appended rather than the
  source's `ValueError` (no `Result` channel through
  `process_llm_request`); `bypass_multi_tools_limit` is stored but not
  enforced (deferred with C0171, same as `GoogleSearchTool`);
  `data_store_specs` entries round-trip as opaque values (an unmodeled
  third-party SDK type); `logger.debug` isn't reproduced.
- **Scope:** no new dependency. 8 new tests.

---

## PR #TBD — `AgentBehavior::as_any` downcast + real `global_instruction` root-walk (C0170)
**2026-08-24** · (link added once this PR is opened)

A small, foundational design-decision batch: adds a downcast escape hatch
for `BaseAgent`'s type-erased behavior, then uses it to close a
long-disclosed narrowing in the deprecated `global_instruction` field's
resolution.

- **Added:** `adk_agents::base_agent::{AgentBehavior::as_any, AsAny,
  BaseAgent::as_any}` — lets code holding a `BaseAgent` recover a concrete
  behavior type for a cross-tree lookup (e.g. an ancestor's
  `LlmAgent`-specific fields), without `adk-agents` needing to know about
  `adk-flows::llm_flow::LlmFlow` or any other concrete behavior type.
  Purely additive: every existing `AgentBehavior` implementor needs zero
  changes, via a blanket-implemented `AsAny` supertrait.
- **Added:** `adk_agents::readonly_context::ReadonlyContext::agent`,
  exposing the running agent for tree-walking.
- **Fixed a real bug** caught before it ever shipped: `Box<dyn
  AgentBehavior>` is itself `Sized + 'static`, so `AsAny`'s blanket impl
  also (over-broadly) matches the `Box` itself — Rust's method resolution
  picks that outer match before reaching the supertrait vtable, so every
  downcast would have silently always failed. Fixed by forcing an
  explicit `.as_ref()` deref before calling `.as_any()`; a same-crate
  regression test now guards it.
- **Closed:** `adk_flows::instructions::build_instructions`'s
  `global_instruction` resolution (C0170) now genuinely walks to the tree
  root (`ctx.agent().root_agent()` + the new downcast), matching the
  source's `hasattr(root_agent, 'global_instruction')` gate — verified by
  a real 2-level-tree test proving a sub-agent's `LlmAgent` picks up the
  *root's* field, not its own. Falls back to the passed-in agent's own
  field only when no tree context is set at all.
- **Corrected 3 stale module docs** (`llm_agent.rs`, `canonical_model.rs`,
  `instructions.rs`) that still claimed `LlmAgent` wasn't wired into
  `BaseAgent`'s tree — it has been since `LlmFlow` landed (Phase 4 batch
  14); `canonical_model.rs`'s ancestor-chain fallback (C0080/C0090) stays
  deferred for a different, still-real reason: it's called once at
  `LlmFlow::new` construction time, before any tree placement exists to
  walk — a separate, larger change (deferring resolution to first-use)
  than this batch's scope.
- **Scope:** no new dependency, no breaking change. 3 new tests (1 in
  `adk-agents`, 2 in `adk-flows`).

---

## PR #TBD — `plugins/logging_plugin.py`: `LoggingPlugin` (C0362, partial)
**2026-08-24** · (link added once this PR is opened)

Ports the console-debugging plugin's 6 run-level and agent-level hooks —
the remaining 7 (model/tool-level) stay blocked on C0355/C0356 as
already disclosed.

- **Added:** `adk_agents::logging_plugin::LoggingPlugin` (C0362 partial):
  `on_user_message_callback`, `before_run_callback`, `on_event_callback`,
  `after_run_callback`, `before_agent_callback`, `after_agent_callback`,
  plus the `format_content`/`format_args` truncation helpers
  (`_format_content`/`_format_args` in the source).
- **Disclosed:** the source's `_log` calls bare `print()` with ANSI grey
  codes, not Python's `logging` module — so this port's `println!` is a
  faithful translation, not the "no logging framework adopted"
  substitution used elsewhere in this migration.
- **Scope:** no new dependency. 11 new tests.

---

## PR #TBD — `plugins/save_files_as_artifacts_plugin.py`: `SaveFilesAsArtifactsPlugin` (C0367)
**2026-08-24** · (link added once this PR is opened)

Ports the plugin that saves files embedded in a user message as artifacts —
and, along the way, fixes a real gap in `Runner`'s plugin wiring that this
plugin's own two-hook (`on_user_message_callback` → `before_agent_callback`)
design surfaced.

- **Added:** `adk_agents::save_files_as_artifacts_plugin::SaveFilesAsArtifactsPlugin`
  (C0367 DONE) — the 20MB size cap, the `gs`/`https`/`http`-only
  model-accessible-URI check for attaching a `file_data` reference part,
  and the pending-delta-then-flush pattern all port in full.
- **Fixed:** `adk_runners::runner::merge_context_state_into_session` (new) —
  a run-level plugin hook's state mutations were previously invisible to
  any other hook in the same turn, since this port's `Context` clones
  `InvocationContext` rather than sharing it by reference the way the
  source's raw `session.state` dict does. Applied after every run-level
  hook call in `Runner::run_async_with_config`; verified end-to-end by a
  new test proving this exact plugin's stash-then-flush pattern works
  through `Runner`.
- **Disclosed:** `MediaBlobStub.display_name`/`.data` are read out of its
  flattened `rest` map (an already-established narrowing, not new here);
  no per-part `try/except`-style fallback since `ArtifactService::save_artifact`
  is infallible in this port (a pre-existing trait shape); the source's
  `DeprecationWarning`/logging isn't reproduced (no logging framework
  adopted, an already-established scope cut).
- **Scope:** no new dependency. 10 new tests (9 in `adk-agents`, 1
  end-to-end in `adk-runners`).

---

## PR #TBD — `runners.py`: run-level plugin wiring (C0353, C0357, C0886, C0895-C0899)
**2026-08-24** · (link added once this PR is opened)

Wires `Runner` up to a real `PluginManager` and ports `_exec_with_plugin`/
`_handle_new_message`/`_append_new_message_to_session`'s run-level plugin
harness — the last major gap `Runner`'s own module doc had flagged as a
follow-up since the plugin system itself first landed.

- **Added:** `Runner::with_plugin` (register a plugin, erroring on a
  duplicate name) and a new `Runner::run_async_with_config` (accepts a
  per-call `RunConfig`; `Runner::run_async` now delegates to it with a
  default).
- **Closed:** C0353 (all 5 run-level `BasePlugin` hooks now have real call
  sites) and C0357 (`on_run_error_callback` now has one too) as DONE.
- **Closed:** C0886's remaining plugin-wrapping gap — compaction is now the
  only thing left open on that row.
- **Added:** `should_append_event` (C0895, partial — only the non-live half
  is reachable, `Runner` has no live-mode path yet), `merge_output_event`
  (C0896 DONE), and the deprecated blob-saving path
  (`Runner::maybe_save_input_blobs_as_artifacts`, C0898/C0899 DONE).
- **Changed:** `Runner::close` now also closes every registered plugin.
- **Disclosed:** `on_event_callback` in this port returns a full
  replacement `Event`, not a partial-update object gated by the source's
  `model_fields_set`, so `merge_output_event` merges by restoring
  `id`/`invocation_id`/`timestamp` and a blank `author` rather than
  field-by-field; `RunConfig` has no constructor-vs-mutation distinction
  to preserve for C0899's read guard (a plain public-field struct, same
  as every other config type in this crate); the source's
  `DeprecationWarning` for the blob-saving path has no Rust equivalent
  (no warnings/logging framework adopted).
- **Scope:** no new dependency. 10 new tests.

---

## PR #TBD — `auth/{exchanger,refresher,credential_service}/`: auth-service cluster (C0523, C0525, C0527, C0528, C0529)
**2026-08-24** · (link added once this PR is opened)

Ports the exchanger/refresher registries and the two concrete
`CredentialService` implementations. Closes the long-standing
`services::AuthConfig`/`services::CredentialService` placeholder gap as
a discovered side effect — the abstract `BaseCredentialService`
interface (C0527) had no real implementor until this batch gave it two.

- **Added:** `adk_agents::{base_credential_exchanger::
  BaseCredentialExchanger, credential_exchanger_registry::
  CredentialExchangerRegistry}` (C0523 DONE).
- **Added:** `adk_agents::{base_credential_refresher::
  BaseCredentialRefresher, credential_refresher_registry::
  CredentialRefresherRegistry}` (C0525 DONE).
- **Added:** `adk_agents::in_memory_credential_service::
  InMemoryCredentialService` (C0528 DONE) and
  `adk_agents::session_state_credential_service::
  SessionStateCredentialService` (C0529 DONE).
- **Closed:** C0527 (`BaseCredentialService`) as a discovered side
  effect — the trait pre-existed as a stale placeholder with zero real
  implementors; this batch's two credential services are its first.
- **Changed:** `adk_agents::services::AuthConfig` widens from a bare
  `Value` placeholder to re-export `auth_tool::AuthConfig`.
  `adk_agents::services::CredentialService` widens from a synchronous,
  context-free trait to a real async, `Context`-taking one (via this
  crate's `BoxFuture` convention) — safe since grep confirmed zero
  prior implementors or call sites. `Context::save_credential`'s
  receiver changes from `&self` to `&mut self` to match (also zero
  external call sites). `AuthCredentialTypes` gains `Hash`/
  `PartialOrd`/`Ord` derives so both registries can key on it directly
  — it's already a closed enum, so (unlike `AuthProviderRegistry`'s
  `type[AuthScheme]`) no discriminant-collapse adaptation was needed.
- **Disclosed:** `InMemoryCredentialService` skips storing an explicit
  absent-value marker on a `None` exchanged credential (its private
  map has no other reader, so this is behaviorally identical through
  the only observable interface); `SessionStateCredentialService`
  preserves the source's explicit-`null`-overwrite instead, since
  session state is externally observable.
- **Scope:** no new dependency. 18 new tests.

---

## PR #TBD — `telemetry/`: per-request config — `TelemetryConfig`/`resolve_schema_version` (C0651, C0652, C0670, C0671, C0679)
**2026-08-24** · (link added once this PR is opened)

Ports `telemetry/context.py`/`telemetry/_schema_version.py`'s pure
env-var-precedence logic. No OTel SDK/span/tracer machinery — that's a
much larger, still-unported surface; this batch is the resolution
logic a caller's `RunConfig` carries and how it resolves against env
vars.

- **Added:** `adk_agents::telemetry_context::{ContentCapturingMode,
  TelemetryConfig, SemconvStabilityOptIn}` (C0651/C0652 DONE).
- **Added:** `adk_agents::schema_version::resolve_schema_version`
  (C0679 DONE).
- **Closed:** C0670 (telemetry config env-var names) and C0671
  (Agent-Engine env-var names) as a side effect — all ported as real,
  actually-declared `pub const`s rather than just documented strings.
- **Changed:** `RunConfig::telemetry` widens from a bare `Value`
  placeholder to the real `TelemetryConfig`.
- **Fixed (manifest hygiene, no new code):** C0505 and C0798 were both
  already fully covered by earlier merged work (C0504's `auth_tool.rs`,
  C0942's `telemetry_config.rs`) but were never cross-linked and still
  read `REQUIRED`. Both now point at their real evidence.
- **Scope:** no new dependency. 23 new tests.

---

## PR #TBD — `optimization/`: `AgentOptimizer`/`Sampler`/data types (C0636, C0637)
**2026-08-24** · (link added once this PR is opened)

Ports `optimization/`'s pure interfaces — no LLM call lives in either
row; concrete LLM-touching optimizers/samplers (`SimplePromptOptimizer`,
GEPA, `LocalEvalSampler`) stay their own, still-blocked rows.

- **Added:** `adk_agents::agent_optimizer::AgentOptimizer` (C0636 DONE).
- **Added:** `adk_agents::sampler::{Sampler, ExampleSet}` and
  `adk_agents::optimization_data_types::{SamplingResult,
  BaseSamplingResult, UnstructuredSamplingResult, AgentWithScores,
  BaseAgentWithScores, OptimizerResult}` (C0637 DONE).
- **Disclosed narrowing:** the source's pydantic-generic bounds
  (`SamplingResultT`/`AgentWithScoresT`) become traits rather than base
  structs callers subclass; `sample_and_score`'s Python-style default
  parameter values have no Rust equivalent, so every caller passes
  every argument explicitly; `BaseAgentWithScores::optimized_agent`
  holds an `Arc<LlmAgent>` handle since `LlmAgent` has neither `Clone`
  nor `Debug`.
- **Scope:** no new dependency (folded into `adk-agents`, which already
  owns `LlmAgent` — the same placement reasoning `app_configs.rs`
  established for `apps/_configs.py`). 8 new tests.

---

## PR #TBD — `auth/`: auth-scheme cluster — `AuthScheme`/`AuthConfig`/`build_auth_headers`/`AuthProviderRegistry` (C0503, C0504, C0522, C0516, C0498)
**2026-08-24** · (link added once this PR is opened)

Ports the OpenAPI-security-scheme side of `auth/`, building on the
credential-scheme batch from an earlier PR. Closes out most of C0493's
remaining unported top-level names — only `AuthHandler` (C0506) stays
open.

- **Added:** `adk_agents::auth_schemes::{SecuritySchemeType,
  AuthSchemeType, ApiKeyIn, ApiKeyScheme, HttpScheme, OAuthFlow,
  OAuthFlows, OAuth2Scheme, OpenIdConnectScheme, SecurityScheme,
  OpenIdConnectWithConfig, CustomAuthScheme, AuthScheme, OAuthGrantType,
  ExtendedOAuth2}` (C0503 DONE, plus C0498's wrap-up).
- **Added:** `adk_agents::auth_tool::{AuthConfig, AuthToolArguments,
  stable_digest}` (C0504 DONE) — including `AuthConfig`'s
  `credential_key` auto-derivation and the dynamic-OAuth2-field-clearing
  step before hashing a raw credential.
- **Added:** `adk_agents::auth_headers::build_auth_headers` (C0522
  DONE) — OAuth2/HTTP-bearer/HTTP-basic/HTTP-other/API-key header
  construction.
- **Added:** `adk_agents::{base_auth_provider::{AuthSchemeKind,
  BaseAuthProvider}, auth_provider_registry::AuthProviderRegistry}`
  (C0516 DONE) — pluggable custom-scheme auth-provider registration and
  lookup.
- **Disclosed narrowing:** the OpenAPI spec's four distinct
  `OAuthFlow*` shapes collapse into one lenient `OAuthFlow` struct
  (which of `OAuthFlows`'s four fields is populated still identifies
  the grant type); `AuthProviderRegistry` keys by an `AuthSchemeKind`
  discriminant rather than the source's `type[AuthScheme]` — every
  custom scheme collapses to one key, not one per exact subclass;
  `build_auth_headers`'s API-key fallback for a non-`APIKey` scheme (an
  unsound `hasattr`-guarded read in the source) falls through to `None`
  rather than reproducing a latent crash; `stable_digest` isn't
  byte-identical to Python's digest, only equally deterministic.
- **Scope:** one new dependency (`sha2` for `adk-agents` — already a
  workspace dependency, new usage site). 37 new tests.

---

## PR #TBD — `apps/_configs.py`: `ResumabilityConfig`/`EventsCompactionConfig`/`BaseEventsSummarizer` (C0283, C0284, C0285)
**2026-08-24** · (link added once this PR is opened)

Closes a real `InvocationContext` placeholder gap: `resumability_config`
was a narrowed stub and `events_compaction_config` an opaque `Value` —
both now hold the source's real config types.

- **Added:** `adk_agents::app_configs::{ResumabilityConfig,
  EventsCompactionConfig, EventsCompactionConfigError,
  BaseEventsSummarizer}` (C0283/C0284/C0285 DONE).
- **Ported:** `EventsCompactionConfig`'s full validator — both-or-neither
  per trigger pair, at-least-one-trigger-mode, plus the `Field(gt=0)`/
  `Field(ge=0)` constraints pydantic enforces ahead of it.
- **Wired:** `adk_features::legacy_feature_decorator::warn_experimental`
  (C0797's guard function, landed but unwired previously) — its first
  real call site, firing from `ResumabilityConfig::new` and
  `EventsCompactionConfig::validate`.
- **Updated:** `InvocationContext::resumability_config`/
  `::events_compaction_config` field types, plus every test call site in
  `loop_agent.rs`/`parallel_agent.rs`/`sequential_agent.rs` that
  constructed the old stub.
- **Disclosed narrowing:** `summarizer: Optional[BaseEventsSummarizer]`
  (an arbitrary, non-pydantic field in the source) becomes
  `Option<Arc<dyn BaseEventsSummarizer>>`; `EventsCompactionConfig` has no
  `Serialize`/`Deserialize` derive at all (a trait object has no wire
  representation), with `Debug`/`Clone` implemented by hand. Nothing
  reads into the compaction machinery's fields yet — the actual
  compaction trigger logic (`LlmEventSummarizer`, C0286/C0287) is
  LLM-blocked and still unported.
- **Also fixed:** an intermittent test race in
  `adk_tools::environment_simulation_config`'s tests (`TemporaryFeatureOverride`
  mutating process-wide state across parallel test threads), serialized
  with a local `TEST_LOCK` mutex.
- **Scope:** one new dependency (`adk-features` for `adk-agents` — an
  existing zero-dependency internal crate, new usage site, no cycle). 18
  new tests.

---

## PR #TBD — `tools/environment_simulation/`: config models + injection-only engine (C0486, C0487 partial, C0488 partial)
**2026-08-24** · (link added once this PR is opened)

Ports deterministic tool-call fault injection for agent testing —
`EnvironmentSimulationConfig` and the injection path of
`EnvironmentSimulationEngine.simulate()` — deferring the LLM-synthesized
mock-strategy fallback the source falls back to when no injection hits.

- **Added:** `adk_tools::environment_simulation_config::{InjectedError,
  InjectionConfig, MockStrategy, ToolSimulationConfig,
  EnvironmentSimulationConfig}` (C0486 DONE). Every validator ported:
  injected-error-xor-injected-response, the `injected_latency_seconds <=
  120.0` constraint, the empty-injection-configs-requires-a-mock-strategy
  check, and the non-empty/no-duplicate-tool_name check.
- **Added:** `adk_tools::environment_simulation_engine
  ::EnvironmentSimulationEngine` (C0487 partial). The injection-only path:
  per-tool-config lookup, `match_args` filtering, a reseed-then-roll
  probability check, injected latency, and the injected-error/injected-
  response dict shape.
- **Added:** `adk_tools::tool_connection_map::{StatefulParameter,
  ToolConnectionMap}` + `adk_tools::environment_simulation_factory
  ::EnvironmentSimulationFactory::create_callback` (C0488 partial).
- **Wired:** `adk_features::feature_decorator::check_feature_enabled`
  (C0647's guard function, landed but unwired previously) — its first
  real call site, gating `EnvironmentSimulationConfig::validate` behind
  `FeatureName::EnvironmentSimulation`.
- **Disclosed narrowing:** the LLM-synthesized mock-response fallback
  (`ToolConnectionAnalyzer`/`ToolSpecMockStrategy`) is blocked — this
  port has no LLM-invocation path to drive it; `EnvironmentSimulationPlugin`/
  `create_plugin` is blocked on the same `BasePlugin` tool-hook gap as
  the existing C0356 deferral; `create_callback`'s output has no real
  dispatch target yet, since this port's `before_tool_callback` type
  takes no `tool`/`args` parameters; `adk_platform::random::Rng` matches
  Python's `random.random()` range but not its Mersenne-Twister
  algorithm, so a `random_seed` reproduces the same roll deterministically
  within this port only.
- **Scope:** no new dependency. 26 new tests.

---

## PR #TBD — `tools/skill_toolset.py`: `RunSkillScriptTool`'s `code_executor` path, closing C0410
**2026-08-24** · (link added once this PR is opened)

Ports `_SkillScriptCodeExecutor` in full, closing out C0410 (previously
partial — the `environment`-configured path only).

- **Added:** `adk_tools::skill_toolset::SkillScriptCodeExecutor` — the
  self-extracting Python wrapper generator (`.py` via `runpy.run_path`,
  `.sh`/`.bash` via `subprocess.run` + JSON envelope), executed against
  `BaseCodeExecutor::execute_code` via `rusty_tokio::spawn_blocking`.
- **Added:** `python_str_literal`/`python_bytes_literal`/
  `python_list_literal`/`python_dict_literal` — a Python `repr()`-
  equivalent, cross-verified round-trip-correct against a real `python3`
  interpreter.
- **Added:** the `code_executor`/`environment` mutual-exclusivity check
  the constructor was missing.
- **Verified end-to-end** against real `python3`/`bash` interpreters: a
  `.py` script reading `sys.argv` built from tool-call `args`, and a
  `.sh` script whose JSON-enveloped stdout/stderr/returncode round-trip
  correctly.
- **Disclosed narrowing:** the Python-literal helpers are round-trip-
  correct but not byte-identical to CPython's `repr()` (adaptive quote
  selection); the source's `except SystemExit as e:` branch is dead code
  for this port's only concrete `BaseCodeExecutor`
  (`UnsafeLocalCodeExecutor`, always subprocess-based, with no exit-code
  field on `CodeExecutionResult` to inspect).
- **Scope:** no new dependency. 11 new tests.

---

## PR #TBD — `tools/skill_toolset.py`: `SkillToolset.additional_tools` (C0950)
**2026-08-24** · (link added once this PR is opened)

Closes the `additional_tools`/`_resolve_additional_tools_from_state`/
`clone_with_updated_skills` gap discovered while building the prior
`SkillToolset` batch.

- **Added:** `adk_tools::skill_toolset::{AdditionalTool,
  SkillToolsetConfig::additional_tools,
  SkillToolset::resolve_additional_tools_from_state,
  SkillToolset::clone_with_updated_skills}` (C0950 DONE).
- **Ported:** the activated-skill → `adk_additional_tools` name-set →
  candidate-tool resolution pipeline (provided tools and provided
  toolsets via `get_tools_with_prefix`), the core-tool-name-collision
  skip, and `clone_with_updated_skills`'s exact field carry-forward —
  including the source's own omission of `tool_name_prefix`/`tool_filter`
  from the clone, reproduced faithfully rather than "fixed."
- **Disclosed narrowing:** the source's `ToolUnion`'s bare-`Callable`
  branch (`FunctionTool(callable)` via `inspect.signature` reflection)
  has no port — `FunctionTool`'s own module doc already discloses this
  port has no such runtime reflection.
- **Scope:** no new dependency. 6 new tests.

---

## PR #TBD — `tools/skill_toolset.py`: the SkillToolset stack (C0395, C0401, C0408-C0411), plus a discovered `additional_tools` gap
**2026-08-24** · (link added once this PR is opened)

Builds `SkillToolset` and four of its five tools on top of the
`BaseEnvironment`/`BaseCodeExecutor`/skill-model infrastructure landed in
earlier batches.

- **Added:** `adk_tools::skill_registry::SkillRegistry` (C0395 DONE).
- **Added:** `adk_tools::skill_instructions_utils::inject_session_state`
  (C0401 DONE) — a local duplicate of the C0170 port in `adk-flows`
  (avoiding a crate-graph cycle), exercised via `LoadSkillTool`'s
  `adk_inject_state` interpolation.
- **Added:** `adk_tools::skill_toolset::{SkillToolset, SkillToolsetConfig,
  ListSkillsTool, SearchSkillsTool, LoadSkillTool}` (C0408 DONE),
  `LoadSkillResourceTool` (C0409 DONE), `RunSkillScriptTool` (C0410
  partial — `environment` path only), `build_skill_system_instruction`/
  `default_skill_system_instruction` (C0411 DONE).
- **Widened:** `adk_tools::skills_models::Resources` from `String`-only
  to a real `ResourceContent` (`Text`/`Bytes`) enum, now that
  `LoadSkillResourceTool` needs the binary branch.
- **Architectural adaptation:** the source's tools hold a live
  back-reference to their owning toolset (a Python reference cycle);
  this port shares one `SkillCoreState` behind an `Arc` instead, the
  same pattern `EnvironmentToolset` (previous PR) already established.
- **Disclosed narrowing:** the source's per-invocation skill-fetch cache
  coalesces concurrent in-flight fetches via a shared `asyncio.Future`;
  this port keeps the 16-turn FIFO caching exactly but doesn't coalesce
  concurrent fetches for the same uncached skill.
- **Deferred:** `RunSkillScriptTool`'s `code_executor` path (needs
  `_SkillScriptCodeExecutor`'s from-scratch Python-wrapper-generation
  design — its own batch). New manifest row **C0950** (REQUIRED, not
  implemented) covers a discovered gap: `SkillToolset.additional_tools`/
  `_resolve_additional_tools_from_state`/`clone_with_updated_skills`.
- **Scope:** no new dependency. 30 new tests.

---

## PR #TBD — `tools/environment/`: the environment-toolset stack, plus a discovered `environment/` inventory gap (C0948, C0949, C0440-C0444)
**2026-08-24** · (link added once this PR is opened)

Closes an inventory gap a background scoping agent found (`environment/`
had no manifest row despite 4 existing rows already depending on it),
then builds the `EnvironmentToolset` stack on top of it in the same PR.

- **Added:** `adk_tools::base_environment::{BaseEnvironment,
  ExecutionResult, EnvironmentError}` (C0948 DONE, new gap-fill row) —
  the abstract code-execution-environment contract.
- **Added:** `adk_tools::local_environment::LocalEnvironment` (C0949
  DONE) — subprocess-shell execution with timeout, blocking file I/O
  offloaded to `rusty_tokio::spawn_blocking`, lexical path-escape
  rejection, auto-created-vs-explicit working-directory lifecycle.
- **Added:** `adk_tools::environment_toolset::EnvironmentToolset` (C0440
  DONE) — bundles the four tools below, injects the environment-level
  system instruction.
- **Added:** `adk_tools::{execute_tool::ExecuteTool,
  read_file_tool::ReadFileTool, edit_file_tool::EditFileTool,
  write_file_tool::WriteFileTool}` (C0441-C0444 DONE).
- **Scope:** no new dependency (`regex`/`rusty_tokio` already workspace
  dependencies of `adk-tools`). 34 new tests. Notable adaptations:
  `BaseEnvironment::is_initialized` becomes a required trait method
  (Rust traits carry no data); `write_file`'s `str | bytes` union
  collapses to `&[u8]` without losing behavior (the source's str branch
  already disables newline translation); `LocalEnvironment`'s path
  resolution is lexical, reusing the "path safety by construction"
  pattern already established in `file_artifact_service.rs`; a timed-out
  command carries no partial output, the same disclosed gap already
  established in `bash_tool.rs`; `EnvironmentToolset`'s uncaught
  initialize-failure becomes a panic rather than widening the already-
  shipped, infallible `BaseToolset` trait.

---

## PR #TBD — `evaluation/`: event→`Invocation` grouping + `AgentEvaluator`'s dataset/legacy-format helpers (C0623, C0619 partial, C0620)
**2026-08-24** · (link added once this PR is opened)

Ports the pure-computation parts of `evaluation_generator.py` and
`agent_evaluator.py` that don't need a real `Runner`/LLM-invocation path.

- **Added:** `adk_eval::evaluation_generator::{collect_events_by_invocation_id,
  convert_events_to_eval_invocations}` (C0623 DONE) — the event→`Invocation`
  grouping algorithm: text-over-audio-only final-response preference,
  should-add-event inclusion rules, and final-event content-dedup. Order
  preservation (unlike this crate's other `HashMap`-for-grouping choices)
  is semantically load-bearing here, so it's kept via a parallel
  `Vec<String>` alongside the grouping `HashMap`.
- **Added:** `adk_eval::agent_evaluator::{load_json,
  find_config_for_test_file, get_initial_session, DatasetInput,
  load_dataset, validate_input, get_eval_set_from_old_format,
  load_eval_set_from_file}` (C0619 partial) — everything except
  `AgentEvaluator.evaluate`/`evaluate_eval_set` themselves, which need
  C0621/C0622/C0624's still-unbuilt inference generation. Cross-verified
  the assert-vs-`ValidationError` control flow in
  `_load_eval_set_from_file` against the real source logic run
  standalone. `DatasetInput` models `_load_dataset`'s actual reachable
  `isinstance` dispatch, not its broader, partly-unreachable type hint.
- **Added:** `adk_eval::agent_evaluator::migrate_eval_data_to_new_schema`
  (C0620 DONE) — the schema-migration utility, exactly ported.
- **Scope:** no new dependency. 25 new tests.

---

## PR #TBD — `evaluation/`: `llm_as_judge_utils` + the rubric-evaluator's harness-independent parts (C0947, C0601)
**2026-08-24** · (link added once this PR is opened)

Closes out the inventory gap flagged last batch (`llm_as_judge_utils.py`
had no manifest row at all) and, since it's the direct dependency of
`rubric_based_evaluator.py`, ports that file's own harness-independent
parts in the same PR.

- **Added:** `adk_eval::llm_as_judge_utils::{Label, get_text_from_content,
  get_text_from_invocation, get_eval_status, get_average_rubric_score,
  get_tool_declarations_as_json_str,
  get_tool_calls_and_responses_as_json_str,
  get_grounding_metadata_as_json_str}` (C0947) — every function ported.
- **Added:** `adk_eval::rubric_based_evaluator::{RubricResponse,
  AutoRaterResponseParser, DefaultAutoRaterResponseParser,
  PerInvocationResultsAggregator,
  MajorityVotePerInvocationResultsAggregator,
  InvocationResultsSummarizer, MeanInvocationResultsSummarizer,
  normalize_text}` (C0601, partial) — the pluggable response parser
  (nearest-preceding-property ID matching, robust to a dropped ID
  line), the majority-vote aggregator, the mean-score summarizer, and
  rubric-text normalization.
- **Not this batch, still `REQUIRED`:** `RubricBasedEvaluator` itself —
  extends `LlmAsJudge[RubricsBasedCriterion]` (C0600's still-deferred
  harness) and returns `AutoRaterScore` (`llm_as_judge.py`, also
  unbuilt). Neither of this batch's additions needs that harness to be
  useful pure data/computation, the same reasoning already established
  for the C0612 criterion types and the C0632 persona system.
- **Widened:** `evaluator::PerInvocationResult::rubric_scores`/
  `EvaluationResult::overall_rubric_scores`, from opaque `Value` to
  real `Vec<eval_rubrics::RubricScore>` — the new aggregators are real
  consumers that need the structure, same "widen once a real consumer
  needs it" pattern already used for `Invocation.rubrics`/
  `.app_details`.
- **Verification:** cross-checked `get_text_from_content`'s
  `Some("")`-vs-`None` truthiness edge case (parts present but none
  carry non-empty text → `Some("")`, not `None`) and
  `DefaultAutoRaterResponseParser`'s parsing across four cases
  (well-formed response, missing-ID tolerance, mismatched-count
  rejection, unparseable verdict) directly against the real source
  logic, run standalone.
- **Adaptation, disclosed:** `get_text_from_content` splits into two
  functions by type (`get_text_from_content`/`get_text_from_invocation`)
  since the source overloads one function over `Union[Content,
  Invocation]` and Rust has no function overloading. `Label`'s
  inconsistent per-variant `.value` shape (a 3-tuple for one member,
  plain strings elsewhere) becomes a uniform `&'static [&'static str]`
  for every variant — a strict improvement, not a narrowing.
- **Adaptation, disclosed:** the source's `_RATIONALE_PATTERN`/
  `_VERDICT_PATTERN` use zero-width lookbehind
  (`(?<=Rationale: )(.*)`); Rust's `regex` crate has no lookbehind
  support (a deliberate limitation for its linear-time guarantee), so
  both become ordinary capture groups instead — behaviorally identical
  here, since the source's own `re.findall` already returns group 1's
  contents for both, never the lookbehind-inclusive group 0.
- **Disclosed narrowing:** `normalize_text` skips the source's NFKC
  normalization step — same gap `rouge.rs` already carries, no
  `unicode-normalization`-equivalent crate is a dependency of
  `adk-eval`. Both aggregators group rubric scores by `rubric_id` in a
  `HashMap` rather than preserving dict insertion order — each
  aggregated score is self-identifying by ID, so list order doesn't
  affect correctness. `_ToolCallAndResponse.tool_response`'s
  `Union[FunctionResponse, str]` narrows to an opaque `Value`.
- **New dependency:** `regex` (already a workspace dependency, new
  usage site in `adk-eval`).
- 31 new tests.

---

## PR #TBD — `evaluation/`: eval-service interface + metric-evaluator registry (C0599, C0603, C0616)
**2026-08-24** · (link added once this PR is opened)

Rounds out the pure-data-model slice of `evaluation/`: the request/
result shapes an eval service works with, the custom-metric extension
point, and the registry that maps a metric name to the evaluator that
scores it.

- **Added:** `adk_eval::base_eval_service::{EvaluateConfig,
  InferenceConfig, InferenceRequest, InferenceStatus, InferenceResult,
  EvaluateRequest, BaseEvalService}` (C0616) — every field ported;
  cross-verified `InferenceConfig`'s defaults and camelCase wire form
  directly against a real pydantic model built from the source.
- **Added:** `adk_eval::custom_metric_evaluator::{CustomMetricEvaluator,
  register_custom_metric_function}` (C0599) — the deep-copy-then-clear-
  `threshold` step before calling a custom function (so it only ever
  reads `criterion.threshold`, never the deprecated top-level field),
  and error propagation.
- **Added:** `adk_eval::metric_evaluator_registry::{
  MetricEvaluatorRegistry, default_registry,
  register_custom_metrics_from_config}` (C0603, partial) —
  `get_evaluator`/`register_evaluator`/`get_registered_metrics`/`fork`,
  seeding only `TrajectoryEvaluator` among the 13 standard evaluators
  (the only one this port has actually built so far).
- **Adaptation, disclosed:** `BaseEvalService`'s `perform_inference`/
  `evaluate` are `AsyncGenerator`s in the source; this port's trait
  returns a fully-materialized `Vec` instead — the same "collected, not
  a live stream" choice already disclosed for `BaseAgent::run_async_impl`.
- **Adaptation, disclosed:** `custom_metric_evaluator`'s source resolves
  a scoring function at runtime via `importlib.import_module`+`getattr`
  on a dotted path. Rust has no dynamic module loader, so this port
  replaces it with an explicit registration API
  (`register_custom_metric_function`) keyed by the same dotted-path
  string — the same "class → registered closure, keyed by a string"
  pattern already used for `user_simulator::register_user_simulator`.
  Custom functions are sync-only (the source supports sync+async),
  matching the already-sync `Evaluator` trait (C0600).
- **Adaptation, disclosed:** `MetricEvaluatorRegistry`'s stored
  `type[Evaluator]` class + runtime `issubclass` check (to decide how
  to construct an evaluator) becomes a tagged factory closure
  (`Factory`/`Custom`), decided once at registration instead of every
  lookup. Its `DEFAULT_METRIC_EVALUATOR_REGISTRY` mutable module-level
  singleton becomes `default_registry()`, a lazily-initialized
  mutex-guarded static (same pattern as `user_simulator`'s own
  registry); `register_custom_metrics_from_config` always takes an
  explicit `&mut MetricEvaluatorRegistry` rather than defaulting to it,
  since a borrow can't outlive a `MutexGuard` the way Python's
  shared-object-identity default can.
- **Not this batch, still `REQUIRED`:** the other 12 standard
  evaluators (`ResponseEvaluator`, `SafetyEvaluatorV1`, the multi-turn
  and rubric-based evaluators, `PerTurnUserSimulatorQualityV1`) under
  C0591-C0598 — each blocked on GCP or the still-deferred `LlmAsJudge`
  harness, not on anything this batch builds. Registering each is one
  call, added alongside its own evaluator once it lands.
- **Manifest housekeeping:** adds C0947 for
  `evaluation/llm_as_judge_utils.py` — a genuine inventory gap found
  while scoping this batch (the source file existed with no manifest
  row at all, not folded into any other row). Landed in a separate,
  earlier PR as a pure bookkeeping fix; flagged as a future-batch
  candidate, not implemented yet.
- **New dependencies:** none.
- 17 new tests.

---

## PR #TBD — `evaluation/simulation/`: user-simulator core + persona system (C0626, C0629, C0632)
**2026-08-24** · (link added once this PR is opened)

The user-simulator interface and the built-in personas that steer it:
what a simulator's config/status shapes look like, one concrete
implementor (`StaticUserSimulator`), and the EXPERT/NOVICE/EVALUATOR
personas with their 11 atomic, mix-and-match behaviors.

- **Added:** `adk_eval::user_simulator::{BaseUserSimulatorConfig, Status,
  NextUserMessage, UserSimulator, parse_simulator_config,
  register_user_simulator, create_user_simulator}` (C0626) — the
  `UserSimulator` trait, its `NextUserMessage`/`Status` result shape
  (with the XOR "message iff `SUCCESS`" validator ported as an explicit
  `NextUserMessage::validate()`), and the config→simulator dispatch
  registry.
- **Added:** `adk_eval::static_user_simulator::StaticUserSimulator`
  (C0629) — replays a pre-authored invocation list in order, returning
  `Status::StopSignalDetected` once exhausted.
- **Added:** `adk_eval::user_simulator_personas::{UserBehavior,
  UserPersona, UserPersonaRegistry}` + `adk_eval::pre_built_personas::{
  PreBuiltBehaviors, get_default_persona_registry}` (C0632) — the
  persona system: 11 atomic behaviors (steering instructions +
  violation rubrics each), composed into the 3 built-in personas
  (EXPERT: 6 behaviors, NOVICE: 5, EVALUATOR: 5).
- **Verification:** cross-checked every behavior's and persona's text
  byte-for-byte against the real source module (executed directly with
  stub model classes, since the full package import chain needs
  `opentelemetry` — same gap noted in the C0611/C0612 entry below).
  Zero mismatches, including the source's own internal
  inconsistencies — "Plan.When" with no space, "Response response"
  doubled, "a a direct" doubled, "inconsist" for "inconsistent", an
  unterminated `"` inside one `TONE_PROFESSIONAL` violation rubric, and
  `END_NO_TROUBLESHOOTING`'s description starting with a leading space.
  These are preserved deliberately: it's judge-model prompt/rubric
  text, not code, so "fixing" it here would silently diverge this
  port's prompts from the source's actual behavior.
- **Adaptation, disclosed:** the source's config→simulator registry
  (`register_user_simulator`) keys by the concrete
  `BaseUserSimulatorConfig` *subclass itself* — Python classes are
  hashable and usable as dict keys, but Rust types aren't runtime
  values. This port keys the registry by the `type` discriminator
  *string* each config carries instead (exactly what `EvalConfig`'s own
  discriminated-union deserialization already dispatches on), with a
  constructor closure as the registered value rather than a class
  object.
- **Adaptation, disclosed:** the source's `UserSimulator` `ABC` marks
  neither `get_next_user_message` nor `get_simulation_evaluator`
  `@abstractmethod` — both just raise `NotImplementedError` if a
  subclass forgets to override them, a runtime-only failure. This
  port's `UserSimulator` trait makes both required methods instead — a
  compile-time strengthening, not a behavior change for any
  implementor that does override both (as every real simulator does).
  `PreBuiltBehaviors` (an enum-with-instance-values in the source)
  becomes a plain unit-variant enum with a `user_behavior()` accessor
  building the owned value on demand, the same shape already used for
  `eval_metrics::PrebuiltMetrics::as_str`.
- **Not this batch, still `REQUIRED`:** `UserSimulatorProvider` (C0627,
  the actual per-`EvalCase` factory that reads this registry) and the
  LLM-backed/audio simulators (C0628/C0630) — all three need a real
  LLM-invocation path this batch doesn't build.
- **New dependency:** `adk-events` — an already lightweight leaf crate
  (`adk-genai` + `adk-platform` only), new usage site for the real
  `Event` type `UserSimulator::get_next_user_message` takes.
- 27 new tests.

---

## PR #TBD — `evaluation/`: `EvalConfig` + the rest of the criterion types (C0611, C0612)
**2026-08-24** · (link added once this PR is opened)

Closes out `evaluation/eval_metrics.py`'s criterion-type hierarchy and
ports `evaluation/eval_config.py` in full — the config surface a
developer actually authors to run an eval (which metrics, what
thresholds, custom metrics, live-mode timeout).

- **Added:** `adk_eval::eval_metrics::{JudgeModelOptions,
  LlmAsAJudgeCriterion, RubricsBasedCriterion, HallucinationsCriterion,
  LlmBackedUserSimulatorCriterion}` (C0612, now complete) — the
  remaining criterion subtypes, flattening the source's class
  inheritance into each struct's own full field set (no Rust struct
  inheritance), the same choice already made for `ToolTrajectoryCriterion`.
  Realized these don't actually need the still-missing `LlmAsJudge`
  harness to be useful, pure data models — they only *configure* an
  LLM-judge metric, no LLM call happens inside any of them, the same
  reasoning already established for `Rubric`/`RubricScore` (C0607).
- **Added:** `adk_eval::eval_config::{EvalConfig, CustomMetricConfig,
  LiveModelConfig, CodeConfig, get_evaluation_criteria_or_default,
  get_eval_metrics_from_config}` (C0611) — `EvalConfig`'s full field
  set plus both free functions the source declares alongside it.
- **Adaptation, disclosed:** the source's legacy-default-injecting
  `@model_validator(mode="before")` on `EvalConfig.user_simulator_config`
  runs automatically on every construction including deserialization.
  This port exposes it as an explicit `EvalConfig::normalize_user_simulator_config`
  instead (the same "split validator that mutates" pattern already
  used for `adk_tools::skills_models::Frontmatter::normalize_name`) —
  call it once after deserializing a config that might predate the
  `type` discriminator. Likewise `JudgeModelOptions.parallelism_limit`'s
  `Field(ge=1)` becomes an explicit `JudgeModelOptions::validate()`
  rather than an automatic construction-time check.
- **Verification:** cross-checked `normalize_user_simulator_config`'s
  four cases (missing `type`, explicit-`null` `type`, explicit
  non-null `type`, no config at all) and `JudgeModelOptions`'s
  defaults/`ge=1` rejection directly against the real pydantic
  validator and model logic, run standalone (the full package import
  chain pulls in `opentelemetry` and other transitive dependencies not
  installed in this environment, so the validator/model logic was
  reproduced verbatim from the source file rather than imported
  through the whole `google.adk` package).
- **Disclosed narrowing:** `EvalConfig.criteria`'s values stay opaque
  `Value` (same choice already made for `EvalMetric::criterion`) and
  become a `HashMap` rather than preserving the source dict's
  insertion order — each metric evaluates independently, so this
  doesn't change behavior, only the order of the returned
  `Vec<EvalMetric>`. `EvalConfig.user_simulator_config` also stays
  opaque `Value` — its real discriminated-union type
  (`LlmBackedUserSimulatorConfig`/`LlmAudioUserSimulatorConfig`) depends
  on C0628/C0630, both still `REQUIRED`. `CustomMetricConfig.code_config`'s
  type is a narrow local `CodeConfig` (`{name: String}` only), not the
  real `agents.common_configs.CodeConfig` — that type belongs to the
  still-`REQUIRED`, unbuilt C0348 YAML config pipeline, and `adk-eval`
  deliberately sits at the bottom of this workspace's crate graph and
  can't depend on wherever C0348 eventually lands. A disclosed,
  permanent, structurally-identical duplication, not an oversight to
  reconcile later. `JudgeModelOptions.judge_model_config` (source:
  `Optional[GenerateContentConfig]`) stays opaque `Value` for the same
  crate-graph-position reason.
- **New dependencies:** none.
- 24 new tests (17 in `eval_config`, 7 in `eval_metrics`).

---

## PR #TBD — `skills/`: `Frontmatter`/`Resources`/`Script`/`Skill` data models (C0393, C0394)
**2026-08-24** · (link added once this PR is opened)

The `skills/models.py` data models: L1 discovery metadata
(`Frontmatter`), L3 on-demand content (`Resources`/`Script`), and the
combined `Skill` type. This is the second of three `skills/` pieces —
`format_skills_as_xml` (C0400) landed in PR #81; `SkillRegistry`/
`load_skill_from_dir` (C0395/C0396) are still open, the latter blocked
on a YAML crate decision.

- **Added:** `adk_tools::skills_models::Frontmatter` (C0393) — `name`
  (kebab-case by default, or kebab-or-snake-case once
  `FeatureName::SnakeCaseSkillName` is enabled, ≤64 chars, NFKC-
  normalized first), `description` (required, ≤1024 chars),
  `license`, `compatibility` (≤500 chars), `allowed_tools` (accepts
  both `allowed_tools` and the YAML-friendly `allowed-tools` wire
  names on the way in, always serializes as `allowed-tools`), and
  `metadata` (validates `adk_additional_tools` is a list and
  `adk_inject_state` is a bool when present). Cross-checked the
  64-char-name, default-snake_case-rejected, and `allowed-tools`
  alias/dump behavior directly against a real `pydantic`-backed
  `Frontmatter` instantiated from the source — messages and behavior
  match exactly.
- **Added:** `adk_tools::skills_models::{Script, Resources}` (C0394)
  — `Script`'s `Display` impl (mirrors `__str__`, returning the raw
  script source) and `Resources`'s six accessor methods
  (`get_reference`/`get_asset`/`get_script`/`list_references`/
  `list_assets`/`list_scripts`) over its `references`/`assets`/
  `scripts` maps.
- **Added:** `adk_tools::skills_models::Skill` (C0394) — combines
  `frontmatter`/`instructions`/`resources`, with `name()`/
  `description()` delegating to `self.frontmatter`, and a
  `pub(crate)` `_uri` provenance field (`uri()`/`set_uri()`
  accessors) matching the source's `PrivateAttr`-style privacy.
- **Adaptation, disclosed:** the source's `@field_validator`s
  (pydantic) run automatically on construction, including
  deserialization. This port keeps `Frontmatter`'s fields plainly
  `pub`/deserializable and exposes the checks as an explicit
  `Frontmatter::validate()` — the same "plain fields + explicit
  `validate()`" pattern already established for
  `adk_eval::eval_case::EvalCase`. NFKC normalization is a separate
  `normalize_name()` step (call before `validate()`) rather than
  folded into validation itself, since a Rust validator can't rewrite
  the struct it borrows the way a pydantic validator mutates and
  returns the field value.
- **Disclosed narrowing:** `Frontmatter`'s `extra="allow"` narrows the
  same way `adk_eval::eval_case::SessionInput` already discloses — an
  unrecognized inbound field no longer rejects the payload, but isn't
  captured anywhere either (silently dropped, not preserved).
  `Resources`'s `dict[str, str | bytes]` values (`references`/
  `assets`) narrow to `String` only — no `str`-or-`bytes` union
  representation exists yet in this codebase, and every other
  content field ported so far (`Script.src`, `Skill.instructions`) is
  text; widen once a caller actually loads a binary resource.
- **New dependencies:** `adk-features` (an existing internal
  workspace crate, new usage site — `Frontmatter::validate` needs
  `FeatureName::SnakeCaseSkillName`/`is_feature_enabled` to branch its
  name pattern the way the source does) and `unicode-normalization`
  (NFKC has no workspace-internal equivalent; same "small,
  well-audited, near-zero-transitive-dependency" bar already used to
  adopt `adk-eval`'s `unicode-general-category` — one dependency of
  its own, `tinyvec`, and the same `unicode-rs` maintainers).
- 19 new tests in `adk-tools::skills_models`, including one
  demonstrating NFKC normalization on a fullwidth-hyphen name and one
  demonstrating snake_case acceptance once the feature flag is
  enabled via `TemporaryFeatureOverride`.

---

## PR #TBD — `features/`: feature-gating decorators as guard functions (C0647 partial, C0797 partial)
**2026-08-24** · (link added once this PR is opened)

Two feature-gating mechanisms in the source, both keyed off decorators
Rust has no runtime equivalent for. Ported as plain guard functions —
called manually at the top of a guarded function/constructor body
instead of wrapping it automatically.

- **Added:** `feature_decorator::check_feature_enabled` (C0647
  partial) — the shared runtime check every one of the source's
  `working_in_progress`/`experimental`/`stable` decorators performs
  (raise unless the feature is enabled at call time). This port's
  `feature_config` (from a prior batch, C0643-C0646) is already a
  fixed, exhaustive `match` — every `FeatureName` carries exactly one
  hardcoded stage, so there's no way for a caller to independently
  assert a *different* stage the way choosing the wrong Python
  decorator could. That makes the source's three-way split, and its
  decoration-time stage-mismatch check, structurally moot here rather
  than narrowed — collapsed into one guard function.
- **Added:** `legacy_feature_decorator::{check_wip_or_bypass,
  warn_experimental}` (C0797 partial) — `utils/feature_decorator.py`'s
  own `working_in_progress`/`experimental`. **Verified, not assumed**:
  reading both source files side by side confirms this is a genuine,
  independent duplicate of `features/_feature_decorator.py` — no
  `FeatureName`/registry involvement at all, just a message, a label,
  and an environment-variable escape hatch (`ADK_ALLOW_WIP_FEATURES`/
  `ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS`), checked at call time.
  Ported exactly, including the env-var truthy set and default
  messages verbatim; the source's message includes the decorated
  object's `__name__` via runtime reflection this port has no
  equivalent for, so `item_name` is an explicit caller-supplied
  argument instead.
- **Not done this batch:** wiring either guard into the source's actual
  decorated call sites across the codebase — each is its own
  not-yet-ported piece of production code, a separate, larger
  undertaking. Both rows stay `REQUIRED` (marked `Partial`).

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-features +7 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `evaluation/`: audio resampling + metric-catalog providers (C0604, C0625) — pure-computation leftovers
**2026-08-24** · (link added once this PR is opened)

Two genuine outliers among `evaluation/`'s remaining rows (mostly LLM-
judge/GCS/simulator work): pure data and computation, no network, no
cloud SDK.

- **Added:** `audio_utils::{resample_pcm16, to_live_input,
  parse_sample_rate}` (C0625) — a linear-interpolation 16-bit PCM
  resampler (24kHz TTS output → 16kHz Live API input) plus a
  `rate=<digits>` mime-type parameter parser, hand-rolled rather than
  adding a `regex` dependency (`adk-eval` doesn't have one) since
  `;`-delimited-parameter matching is straightforward to reimplement
  directly. Cross-checked against the real
  `google.adk.evaluation._audio_utils` functions, run directly from the
  checked-out `google/adk-python` repo — both the resampler's exact
  interpolated output and the mime-type parser's edge cases (case
  insensitivity, surrounding whitespace, a later `;`-parameter, a
  non-digit suffix) match.
- **Added:** `eval_metrics::{PrebuiltMetrics, Interval, MetricValueInfo,
  MetricInfo, MetricInfoProvider}` and `metric_info_providers` (C0604) —
  all 12 concrete `MetricInfoProvider` implementors, covering all 13
  `PrebuiltMetrics` (`ResponseEvaluatorMetricInfoProvider` alone covers
  2: `response_evaluation_score` and `response_match_score`). Also
  closes the `Interval`/`MetricValueInfo`/`MetricInfo`/
  `MetricInfoProvider` slice of C0612, previously unported.
- **Verified, not assumed:** the source's
  `PerTurnUserSimulatorQualityV1MetricInfoProvider`/
  `RubricBasedMultiTurnTrajectoryMetricInfoProvider` pass the bare
  `PrebuiltMetrics` enum member (not `.value`) into `MetricInfo`'s
  `str`-typed `metric_name` field — everywhere else in the file uses
  `.value` explicitly. This looked like a real bug worth flagging or
  replicating. Checked live instead: pydantic v2 unwraps a plain-`Enum`
  member assigned to a `str` field via its `.value` automatically,
  producing the identical string either way. Not a bug — this port uses
  `.as_str()` uniformly for all 13 metrics, matching actual runtime
  behavior rather than the surface-level code-review impression.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-eval +23 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `skills/`: `format_skills_as_xml` (C0400) — first landing in the skills/ area
**2026-08-24** · (link added once this PR is opened)

Small, self-contained follow-on flagged during the previous PR's own
review. First landing in the `skills/` capability area — everything
else there (`Frontmatter`/`Skill` models, disk/GCS loading, zip-bomb
defense, `SkillToolset`) stays `REQUIRED`; this one function doesn't
depend on any of it.

- **Added:** `skills_prompt::format_skills_as_xml` — renders an
  `<available_skills>` XML block for LLM instructions, HTML-escaped.
  Ported exactly: the empty-list sentinel, the per-skill `<name>`/
  `<description>` block shape, and `html.escape`'s default
  `quote=True` character-escaping table — cross-checked against real
  Python `html.escape` output for `<script>`/`&`/`"`/`'`.
- **Disclosed narrowing:** the source's `list[Union[Frontmatter,
  Skill]]` parameter narrows to a minimal local `SkillSummary{name,
  description}` struct — this function only ever reads those two
  fields off either shape, and both real models are their own
  still-`REQUIRED` rows (C0393/C0394). Widen once a real caller needs
  to pass one directly.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-tools +4 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `tools/`: OpenAPI↔Gemini-Schema and MCP schema conversion (C0489 partial, C0455 partial)
**2026-08-24** · (link added once this PR is opened)

New capability area for this session: `tools/`'s schema-conversion
infrastructure backing `McpToolset` and OpenAPI-defined function tools.
Both directions are pure tree transforms over `Value` — no `mcp` crate,
no `google-genai` SDK dependency.

- **Added:** `gemini_schema_util::to_gemini_schema` (and its pieces —
  `to_snake_case`, `dereference_schema`, `sanitize_schema_type`,
  `sanitize_schema_formats_for_gemini`) — OpenAPI v3.1/JSON-Schema →
  Gemini-schema conversion: `$ref`/`$defs`/`definitions` dereferencing
  with circular-ref guarding, camelCase→snake_case wire-key conversion,
  Gemini-supported-field sanitization, `oneOf`→`anyOf` widening with
  accumulation across both keywords, nullable-type-list collapsing, and
  per-type `format` allow-listing (`int32`/`int64` for numeric types,
  `date-time`/`enum` for string).
- **Verification:** end-to-end cross-checked against the real
  `google.adk.tools._gemini_schema_util` source — imported directly
  from the checked-out `google/adk-python` repo and run locally, not
  reconstructed from memory — over 11 fixtures covering every branch
  above, including the trickier case where the top-level schema is
  itself a `$ref` sibling to `$defs`. All match structurally.
- **Disclosed scope boundary:** the source's `_to_gemini_schema` ends by
  calling `google.genai.types.Schema.from_json_schema(...)` — a ~380-line
  method belonging to the third-party `google-genai` SDK, not
  `google.adk` itself, which re-derefs `$ref`s a second time and applies
  its own stricter per-JSON-Schema-type field allow-list. This lives
  outside `google/adk-python`'s own source tree (the boundary this
  migration ports), and this workspace has no typed Gemini `Schema`
  struct to begin with — `FunctionDeclaration.parameters` is already
  just `Value`. This port covers everything `_gemini_schema_util.py`
  itself does and stops there; the returned `Value` may retain a few
  fields the real SDK step would additionally prune.
- **Added:** `mcp_conversion_utils::{adk_to_mcp_tool_type,
  gemini_to_json_schema}`, ported from `mcp_tool/conversion_utils.py`.
  `gemini_to_json_schema` is the reverse mapping (Gemini schema → JSON
  Schema), a self-contained per-type field mapping with no SDK-internal
  dependency. `adk_to_mcp_tool_type` is backed by a narrowed local
  `McpTool{name, description, input_schema}` struct rather than the real
  `mcp.types.Tool` — this port has no `mcp` crate dependency anywhere.
  `session_context.py`'s `SessionContext` (real async `mcp.ClientSession`
  pooling) stays `REQUIRED`.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-tools +30 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `evaluation/`: local `EvalSetsManager`/`EvalSetResultsManager` persistence (C0613/C0615 partial, C0614)
**2026-08-24** · (link added once this PR is opened)

Follow-on to the `EvalSet`/`EvalCase` data-model PR — the local-disk
persistence layer that actually stores and retrieves them. `Gcs*`
implementors of both manager traits stay `REQUIRED`; no GCS SDK
dependency is decided anywhere in this workspace yet.

- **Added:** `eval_sets_manager::EvalSetsManager` trait + a shared
  `EvalManagerError` enum (`NotFound`/`InvalidPath`/`InvalidArgument`/
  `Io` variants) used by every manager in this PR — the source mixes
  plain `ValueError` and the typed `NotFoundError`; this port keeps that
  same split as distinct variants instead of collapsing everything into
  one string, so a caller can match on "not found" specifically.
- **Added:** `eval_sets_manager_utils` (`get_eval_set_from_app_and_id`/
  `get_eval_case_from_eval_set`/`add_eval_case_to_eval_set`/
  `update_eval_case_in_eval_set`/`delete_eval_case_from_eval_set`),
  shared by every `EvalSetsManager` implementor.
- **Added:** `in_memory_eval_sets_manager::InMemoryEvalSetsManager` —
  `Mutex`-guarded maps (the trait takes `&self`, to stay object-safe
  alongside the local/disk implementor too), the same interior-mutability
  pattern `adk_agents::in_memory_memory_service::InMemoryMemoryService`
  already uses for the identical shape of problem.
- **Added:** `local_eval_sets_manager::LocalEvalSetsManager` — real
  `std::fs` I/O, one `.evalset.json` file per `(app_name, eval_set_id)`.
  `load_eval_set_from_file` tries the current typed schema first, falling
  back to converting the legacy JSON-array format
  (`convert_eval_set_to_pydantic_schema`) on a parse failure — verified
  against an actual legacy-format fixture file, not just unit-level
  pieces of the conversion.
- **Added:** `path_validation::validate_path_segment` (C0614) —
  empty/null-byte/path-separator/traversal-segment rejection, applied at
  every filesystem-path construction site in both local managers.
- **Added:** `eval_set_results_manager::EvalSetResultsManager` trait and
  `eval_set_results_manager_utils` (`create_eval_set_result`/
  `parse_eval_set_result_json`, with the back-compat double-JSON-decode
  fallback for legacy result files — verified by actually double-encoding
  a real result and reading it back, not just exercising the happy path).
- **Added:** `local_eval_set_results_manager::LocalEvalSetResultsManager`
  — stores results under `<agents_dir>/<app_name>/.adk/eval_history/`.
- **New dependencies:** `adk-errors` (for `NotFoundError`/
  `InputValidationError`-shaped error variants) and `adk-platform` (for
  `uuid::new_uuid`/`time::get_time`, the workspace's existing
  provider-swappable id/clock abstractions). Both are lightweight —
  `adk-errors` only depends on `rusty_err`, `adk-platform` only on
  `rusty_uuid` + `rusty_serde` — chosen deliberately over `adk-agents`
  (which would be needed for `EvalCaseResult.session_details`'s real
  type, still left opaque — see the previous PR).
- **Disclosed adaptation:** `validate_path_segment`'s source raises a
  plain `ValueError`; this port uses `adk_errors::input_validation::InputValidationError`
  (`ValueErrorLike`), the same typed stand-in this codebase already uses
  for the artifacts subsystem's identical path-traversal/null-byte
  rejection category.
- **Disclosed adaptation:** `LocalEvalSetsManager::_validate_id`'s
  `^[a-zA-Z0-9_]+$` regex check is hand-rolled (non-empty, every
  character ASCII-alphanumeric-or-underscore) rather than adding a
  `regex` dependency for one trivial pattern.
- **Disclosed narrowing:** the source's `model_dump_json(indent=2,
  exclude_unset=True, exclude_defaults=True, exclude_none=True)` writes
  pretty-printed, sparse JSON. `rusty_serde::json` has no pretty-printer
  and this port's structs don't use `skip_serializing_if` anywhere, so
  every write is a compact JSON object with every field present
  (including `null`s and defaults) — round-trips correctly through this
  same port's own `Deserialize` either way, just denser on disk.
- **Disclosed narrowing:** the legacy-format converter's `value_str`
  helper defaults every missing JSON key to `""` uniformly, where the
  source mixes required-key direct indexing (`KeyError` on a malformed
  file) with `.get(key, default)` for a few optional ones.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-eval +40 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `evaluation/`: `EvalSet`/`EvalCase` data model + rubrics/app-details/results (C0606, C0607, C0609, C0610, C0635)
**2026-08-24** · (link added once this PR is opened)

Closes the "second-tier" data-model layer under `evaluation/`: everything
`EvalSet`/`EvalCase` need to actually represent an eval fixture and its
results on disk, once C0605/C0608's core landed. Also closes a disclosed
gap from the first `evaluation/` PR: `Invocation.rubrics`/`.app_details`
now carry the real `Rubric`/`AppDetails` types instead of opaque `Value`.

- **Added:** `eval_case::EvalCase`/`SessionInput`/`SessionState`/
  `StaticConversation`. `EvalCase`'s `conversation`-XOR-
  `conversation_scenario` constraint (the source's `@model_validator`)
  is exposed as `EvalCase::validate()` rather than enforced automatically
  on deserialization — disclosed in the module doc, matching this
  codebase's existing `ServiceAccount`-style pattern (plainly
  deserializable fields, explicit validation on construction) rather
  than baking a Python-style post-init hook into `Deserialize`.
- **Added:** `conversation_scenarios::ConversationScenario`/
  `ConversationScenarios`/`ConversationGenerationConfig`.
- **Added:** `eval_set::EvalSet`.
- **Added:** `eval_rubrics::Rubric`/`RubricContent`/`RubricScore`.
  `RubricContent.text_property` has no default in the source
  (`Field(description=...)`, no `default=`), so — like pydantic — it's
  required at construction even though its value may be `null`.
- **Added:** `app_details::AgentDetails`/`AppDetails`, including both
  methods (`get_developer_instructions`/`get_tools_by_agent_name`).
- **Added:** `eval_result::EvalCaseResult`/`EvalSetResult`, including
  both explicitly-deprecated back-compat fields.
- **Added:** `constants::{MISSING_EVAL_DEPENDENCIES_MESSAGE,
  DEFAULT_LIVE_TIMEOUT_SECONDS, eval_constants}`.
- **Disclosed adaptation:** `Rubric`'s source `type` field is
  `rubric_type` at the Rust level (`r#type` as a field name produces
  unparsable proc-macro token streams under this codebase's derive
  macro) — `#[rusty_serde(rename = "type")]` keeps the wire shape
  unchanged.
- **Disclosed narrowing:** `ConversationScenario.user_persona` stays
  opaque `Value` — its real type and the source's string-id-to-persona
  registry resolution belong to the persona system, C0632, still
  `REQUIRED`; neither `Evaluator` built so far reads it.
- **Disclosed narrowing:** `EvalCaseResult.session_details` stays opaque
  `Value` — its real type (`adk_agents::session::Session`) already
  exists in this workspace, but `adk-eval` deliberately stays at the
  bottom of the crate graph (only `adk-genai` + `rusty_serde`); pulling
  in `adk-agents`'s own dependency tree for one unread passthrough field
  would invert that design intentionally.
- **Disclosed narrowing:** `EvalCase`/`SessionInput`'s source
  `extra="allow"` config (vs. the base `EvalBaseModel`'s `extra="forbid"`)
  narrows to "an unrecognized field no longer rejects the payload" — this
  port has no catch-all side channel preserving the extra field the way
  pydantic does.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-eval +21 new tests; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `evaluation/`: `RougeEvaluator` (C0590) — hand-ported Porter stemmer + Unicode-aware ROUGE-1 tokenizer
**2026-08-24** · (link added once this PR is opened)

Follow-on to the previous `evaluation/` PR. Adds the `response_match_score`
metric — the second-tier candidate flagged in that PR's own review as
"more effort than the trajectory metric": it requires hand-writing a
ROUGE-1 scorer plus the exact Porter-stemming and CJK/Thai/Lao/Khmer/
Myanmar-aware tokenization the source's own `RougeEvaluator` depends on
(the source itself just glues to the pip `rouge_score`/`nltk` packages).

- **Added:** `adk-eval::porter_stemmer` — a from-scratch port of
  `nltk.stem.porter.PorterStemmer` (`NLTK_EXTENSIONS` mode, the only mode
  `rouge_score`'s `DefaultTokenizer` ever constructs). Every step (1a
  through 5b), the irregular-forms pool, and the NLTK-only extensions
  (length-4 `ies`/`ied` cases, the relaxed step 1c condition, the
  `alli`/`fulli`/`logi` step 2 additions) are ported line-for-line from
  nltk's own source. Verified against 409 `(word, stem)` pairs generated
  by actually running nltk 3.10.3 in this environment — not reconstructed
  from memory. This cross-check caught a real mistake in an earlier
  hand-derived expectation table: reading Porter's paper's own per-step
  worked examples in isolation gives `"agreed"` → `"agree"` (step 1b's own
  example), but the *full* pipeline continues through step 5a, which
  strips the trailing `e` to give `"agre"` — the actual nltk output. The
  fixture-based test replaced the wrong hand-derived table before this
  PR, not after.
- **Added:** `adk-eval::rouge` — `_UnicodeAwareTokenizer` (CJK characters
  split individually; Thai/Lao/Khmer/Myanmar grouped into grapheme
  clusters by attaching combining marks to their preceding base
  consonant; ASCII words routed through the stemmer) and `RougeScorer`'s
  `rouge1` unigram-overlap precision/recall/F-measure. Only the `rouge1`
  path `RougeEvaluator` uses is ported — `rougeL`/`rougeLsum`/`rouge2+`/
  `score_multi` are unreached by this consumer.
- **Added:** `adk-eval::final_response_match_v1::RougeEvaluator` —
  `evaluate_invocations` ported exactly (per-invocation ROUGE-1
  F-measure vs. threshold, mean overall score), including the
  `expected_invocations`-required check and the file-local
  `_get_text_from_content` helper (join non-empty part text with `"\n"`
  — deliberately not reusing `content_utils::extract_text_from_content`,
  C0927, which filters `thought` parts and concatenates without a
  separator: a different, purpose-built helper the source itself keeps
  file-local).
- **Verification:** end-to-end cross-checked against the real upstream
  `rouge_score` package source (fetched from `google-research/
  google-research` on GitHub, reassembled locally, and run under real
  `nltk`) over 11 candidate/reference pairs spanning plain ASCII,
  stemming-sensitive text, CJK, Thai, and punctuation-heavy input — every
  precision/recall/F-measure triple matches to floating-point precision.
- **New dependency:** `unicode-general-category` (zero transitive
  dependencies, `no_std`, static Unicode 16.0.0 tables) — needed for the
  source's `unicodedata.category(char).startswith("M")` combining-mark
  checks, which span far more scripts (Devanagari matras, Arabic
  diacritics, etc., not just the four non-spaced scripts this metric
  specifically targets) than a hand-embedded table could accurately
  cover. Adopted directly under the same well-audited, non-sovereignty-
  sensitive bar the workspace's `regex` dependency was adopted under,
  rather than stopping to ask, since it is pure text-processing
  infrastructure with no I/O/crypto/network surface.
- **Disclosed narrowing:** `unicodedata.normalize("NFKC", text)` is
  skipped — no normalization-table dependency was added. NFKC mainly
  affects compatibility-decomposable characters (full-width Latin/digit
  variants, some ligatures); text already in common normalization forms
  tokenizes identically either way.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-eval +24 new tests — porter_stemmer, rouge, final_response_match_v1 — all passing; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — Start `evaluation/` pure-scoring core: `TrajectoryEvaluator` + `Invocation`/`EvalMetric` (C0588, C0600 partial, C0605, C0608, C0612 partial)
**2026-08-24** · (link added once this PR is opened)

New `adk-eval` crate — the first landing in `google.adk.evaluation`
(Phase 11), scoped to the pure-computation core needed to run
`TrajectoryEvaluator` end-to-end with no LLM calls and no cloud
dependency: `eval_case::Invocation` and its nested types (C0605),
`evaluator::Evaluator` trait (C0600, partial), `eval_metrics::EvalMetric`/
`EvalMetricResult`/`BaseCriterion`/`ToolTrajectoryCriterion` (C0608,
C0612 partial), and `trajectory_evaluator::TrajectoryEvaluator` (C0588,
DONE in full — the `tool_trajectory_avg_score` metric). Depends only on
`adk-genai` + `rusty_serde`, at the bottom of the crate graph.

- **Added:** `adk-eval::eval_case::{Invocation, IntermediateData,
  InvocationEvent, InvocationEvents, IntermediateDataType}` plus
  `get_all_tool_calls`/`get_all_tool_responses`/
  `get_all_tool_calls_with_responses`. The source's
  `Union[IntermediateData, InvocationEvents]` (no shared wire tag)
  resolves via `Invocation::intermediate_data_type()`, which attempts a
  JSON-round-trip parse into each shape in turn — centralized into one
  method rather than repeated at every call site the way the source's
  `isinstance` checks are.
- **Added:** `adk-eval::evaluator::{Evaluator, PerInvocationResult,
  EvaluationResult}` and `validate_invocation_lengths`. The `Evaluator`
  trait is sync, not `async` — its one implementor this batch does no I/O.
- **Added:** `adk-eval::eval_metrics::{EvalMetric, EvalMetricResult,
  EvalMetricResultDetails, EvalMetricResultPerInvocation, EvalStatus,
  BaseCriterion, MatchType, ToolTrajectoryCriterion}` and
  `get_metric_threshold`.
- **Added:** `adk-eval::trajectory_evaluator::TrajectoryEvaluator` — the
  exact 3-branch constructor validation logic (threshold XOR
  eval_metric-with-criterion), and all three match algorithms
  (`are_tool_calls_exact_match`/`are_tool_calls_in_order_match`/
  `are_tool_calls_any_order_match`) against the source's own documented
  edge cases, built on a `calls_match` helper that deliberately compares
  only `name`+`args` (unlike derived `PartialEq`, which also compares
  `id`/`will_continue`).
- **Disclosed, compile-time-strengthening adaptation:** `EvalMetric`'s
  source `_config_custom_function_path` `PrivateAttr` (guards against an
  inbound-payload spoofing that field) ports as a private struct field
  with `#[rusty_serde(skip)]` — it structurally cannot be populated by
  this port's derived `Deserialize` at all, a stronger guarantee than the
  source's runtime-enforced `PrivateAttr`.
- **Disclosed, cosmetic adaptation:** `EvalStatus` serializes as its
  variant name (`"Passed"`/`"Failed"`/`"NotEvaluated"`) rather than the
  source's underlying bare-integer Pydantic-v2 enum value, since no
  cross-language consumer of this brand-new capability area exists yet.
- **Disclosed narrowing:** `Invocation.rubrics`/`.app_details`,
  `EvalMetric.criterion`, and `Evaluator::evaluate_invocations`'s
  `conversation_scenario` parameter stay opaque `Value` placeholders —
  their real types (`eval_rubrics.Rubric`/`RubricScore`,
  `app_details.AppDetails`, `eval_case.ConversationScenario`) are each
  their own still-`REQUIRED` manifest row (C0606/C0607/C0610), and
  `TrajectoryEvaluator` never reads any of the three — the source itself
  explicitly `del`s `conversation_scenario`, "not supported for
  per-invocation evaluation."
- **Not in scope this batch:** `LlmAsJudge[CriterionT]` (the generic
  LLM-judge-sampling harness, needs a real LLM-invocation path) and its
  criterion types (`LlmAsAJudgeCriterion`/`RubricsBasedCriterion`/
  `HallucinationsCriterion`/`LlmBackedUserSimulatorCriterion`/
  `JudgeModelOptions`), plus the metric-catalog metadata types
  (`Interval`/`MetricValueInfo`/`MetricInfo`/`MetricInfoProvider`) — all
  left `REQUIRED` (C0600/C0612 marked `Partial` in the manifest).

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-eval +31 new tests, all passing; zero regressions elsewhere)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `UnsafeLocalCodeExecutor` (C0385) — completes `code_executors/`'s non-cloud scope
**2026-08-24** · (link added once this PR is opened)

Closes out the `code_executors/` area's local (non-cloud-backend)
scope. `ContainerCodeExecutor`/`GkeCodeExecutor`/cloud executors still
need a new SDK dependency decision and stay flagged `REQUIRED`.

- **Added:** `adk-tools::unsafe_local_code_executor::UnsafeLocalCodeExecutor`
  — bare, zero-sandboxed subprocess code execution (same host/creds/
  network/filesystem access as this process, matching the source
  exactly). The embedded `_RUNNER` Python script (byte-for-byte
  identical, including the frame-stripped traceback printing),
  `__main__`-guard detection, and `stateful`/`optimize_data_file`
  rejection are all ported exactly. 6 new tests, including real
  end-to-end subprocess execution: stdout capture, a traceback-on-error
  case, and a real 1-second timeout kill.
- **Disclosed adaptation:** `sys.executable` (the source re-invokes
  itself) becomes a configurable `python_executable` command (default
  `"python3"` on `PATH`) — this port was never a Python interpreter, so
  running this executor still genuinely requires a real Python
  interpreter installed on the host, inherent to the capability itself.
- **Disclosed adaptation:** `PYTHONPATH` isn't fabricated from a
  nonexistent `sys.path` equivalent — left to inherit whatever the
  parent process already has.
- **Disclosed adaptation:** the process-group SIGTERM→5s-grace→SIGKILL
  sequence narrows to an immediate `Child::kill()` (SIGKILL only,
  immediate child only) — the same disclosed `killpg`-equivalent gap
  already established by `bash_tool.rs` (C0418), not a new one.
- **Disclosed adaptation:** stdin/stdout/stderr are handled by three
  dedicated OS threads (mirroring what Python's `subprocess.communicate()`
  does internally), since `execute_code` is a synchronous method,
  matching the source's own non-`async` signature.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-tools +6 new tests, including real subprocess execution, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `code_executors/`: `BuiltInCodeExecutor` + `CodeExecutorContext` (C0384, C0390)
**2026-08-24** · (link added once this PR is opened)

Continues the `code_executors/` area started in the previous PR.
`UnsafeLocalCodeExecutor` (C0385) is left for its own batch;
`ContainerCodeExecutor`/`GkeCodeExecutor`/cloud executors still need a
new SDK dependency decision.

- **Added:** `adk-tools::built_in_code_executor::BuiltInCodeExecutor`
  (C0384) — `process_llm_request` reuses `append_built_in_tool_marker`
  (`"codeExecution"` wire key) and preserves the raise-for-unsupported-
  model behavior, since `BuiltInCodeExecutor` isn't a `BaseTool` and so
  isn't bound by that trait's usual error-dropping signature narrowing.
- **Disclosed adaptation:** the source's `execute_code` override
  actually returns `None` at runtime despite its own declared non-
  `Optional` return type (the source's own `# type: ignore[empty-body]`
  acknowledges the mismatch). This port keeps `BaseCodeExecutor::execute_code`'s
  trait contract honestly non-`Option` (matching what every other
  executor truly returns) and uses `CodeExecutionResult::default()` as
  the closest-fitting sentinel here instead.
- **Added:** `adk-tools::code_executor_context::CodeExecutorContext`
  (C0390) — every method ported exactly, including the real
  distinction between the nested, flush-on-`get_state_delta` context
  sub-dict (execution id/processed file names) and the already-live
  root session-state keys (input files/error counts/results).
- **Disclosed adaptation:** `File.content` round-trips as base64 text
  (reusing `code_execution_utils`'s codec) rather than the source's raw
  `bytes` in a plain dict, since this port's state has no raw-bytes
  `Value` variant — the encoding round-trips exactly, no data loss.
- Adds `adk-platform` as a new direct dependency of `adk-tools`
  (already vetted workspace-wide, new usage site only).

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-tools +10 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — Start `code_executors/`: `CodeExecutionUtils` + `BaseCodeExecutor` (C0383, C0391)
**2026-08-24** · (link added once this PR is opened)

The first landing in a new capability area, `code_executors/` — none of
it existed in this workspace before. This batch covers the data/utility
layer; `BuiltInCodeExecutor`/`CodeExecutorContext`/`UnsafeLocalCodeExecutor`
are tracked separately for future batches (`ContainerCodeExecutor`/
`GkeCodeExecutor`/cloud-backend executors need a new SDK dependency
decision and stay flagged `REQUIRED`).

- **Added:** `adk-tools::code_execution_utils::{File, CodeExecutionInput,
  CodeExecutionResult, get_encoded_file_content,
  extract_code_and_truncate_content, build_executable_code_part,
  build_code_execution_result_part, convert_code_execution_parts}`
  (C0391) — every function ported exactly, including the truthy
  (not merely non-empty-Optional) text-part filter and the
  "convert a trailing result part to text only when `Content` has
  exactly one part" guard.
- **Added:** `adk-tools::base_code_executor::{BaseCodeExecutor,
  CodeExecutorConfig}` (C0383) — the source's Pydantic `BaseModel`
  (fields + one abstract method) splits into a plain config struct a
  concrete executor embeds, plus a trait with `config()`/`execute_code()`.
- **Disclosed adaptation:** `Part.executable_code`/`code_execution_result`
  (opaque `Value` placeholders per `adk-genai::content`'s own doc) are
  read/written by known Gemini wire keys (`code`/`language`/`outcome`/
  `output`) — the same established opaque-boundary-field pattern used
  elsewhere, not a widening of `Part`.
- **Disclosed adaptation:** `File.content`'s `str | bytes` union
  narrows to always-raw `Vec<u8>`. A small base64 codec is duplicated
  locally (no `base64` crate is a workspace dependency, and neither
  existing hand-rolled copy in this port has the right visibility/
  byte-vs-str shape for reuse here).

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-tools +14 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `FileArtifactService` (C0268-C0274, plus C0266/C0267 parity tests)
**2026-08-24** · (link added once this PR is opened)

A filesystem-backed `ArtifactService` implementation — this repo's
second full `ArtifactService` backend, alongside `InMemoryArtifactService`.

- **Added:** `adk-agents::file_artifact_service::FileArtifactService` —
  full storage layout (`root/apps/{app}/users/{user}/[sessions/{sid}/]artifacts/{path}/versions/{version}/{filename}`
  + sibling `metadata.json`), nested filenames, `user:`-namespace
  sharing across sessions, `mkdir`-staging-directory atomic crash-safe
  writes with rename-to-publish, path-traversal/rooted/drive-qualified
  filename rejection, `metadata.json`-name collision protection, and
  always-recomputed `canonical_uri`.
- **Disclosed adaptation:** no `_umask_derived_file_mode` equivalent
  needed — this port's atomic-write helper creates its temp file via
  `std::fs::write`, which already gets the OS's normal umask-derived
  permissions the way the source's `tempfile.mkstemp` (hardcoded 0600)
  doesn't, so the mismatch that function exists to fix never arises.
- **Disclosed adaptation:** path-traversal prevention is by lexical
  construction (reject any `..` segment, then join with no filesystem
  interaction) rather than the source's filesystem-resolving,
  symlink-following `Path.resolve(strict=False)` re-check — Rust's
  `canonicalize` needs the target to already exist, which a
  not-yet-created artifact version usually doesn't. A stronger
  guarantee for the traversal-prevention property specifically, though
  it doesn't replicate the source's symlinked-scope-root behavior.
- **Disclosed adaptation:** `canonical_uri` is a hand-rolled, purely
  lexical `file://` URI builder for the same not-yet-existent-path
  reason.
- **Disclosed adaptation:** a small base64 codec is duplicated locally
  — `adk-tools::load_artifacts_tool` already hand-rolled one, but
  `adk-tools` depends on `adk-agents`, not the reverse, so reusing it
  would need a crate-graph cycle.
- **Also closed out:** C0266/C0267 (`InMemoryArtifactService`'s
  empty-artifact sentinel and artifact-reference resolution) — both
  were already implemented as part of C0265's original batch but had
  no dedicated parity test; added one each.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-agents +18 new tests: 16 for `FileArtifactService`, 2 for C0266/C0267, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `CachePerformanceAnalyzer` (C0946)
**2026-08-24** · (link added once this PR is opened)

Closes out C0946, the one capability flagged in the previous PR as a
genuinely tractable next batch (unlike C0941/C0945's deeper blockers).

- **Added:** `adk-flows::cache_performance_analyzer::{CachePerformanceAnalyzer,
  CachePerformanceReport, CachePerformanceStats}` — analyzes
  `GeminiContextCacheManager` cache-hit/refresh performance from a
  session's event history (token totals, hit/utilization ratios,
  invocation counts, cache-refresh count, latest cache name), including
  the source's truthy-int treatment of `prompt_token_count`/
  `cached_content_token_count`.
- **Disclosed adaptation:** `Event.cache_metadata`/`usage_metadata` stay
  opaque `Value` placeholders — parsed into `CacheMetadata` on demand,
  and `usage_metadata`'s `promptTokenCount`/`cachedContentTokenCount`
  keys read directly, the same idiom `context_cache.rs`'s C0175 already
  established (`adk-events` can't depend on `adk-models` without a
  cycle).
- **Disclosed adaptation:** the source's untyped `Dict[str, Any]` return
  becomes a closed `CachePerformanceReport` enum — a strict improvement,
  not a narrowing, since no consumer needs a serialized wire form yet.
- **Disclosed adaptation, compile-time strengthening:** a missing
  session becomes an explicit `Err(SessionNotFound)` rather than the
  source's implicit `AttributeError`-on-`None` risk.
- **Not represented:** `@experimental` (`utils/feature_decorator.py`,
  C0797) — that decorator's own manifest row is still unresolved
  (possibly a second, parallel feature-gating mechanism), so this port
  doesn't guess at a representation for it here.
- Adds `adk-errors` as a new direct dependency of `adk-flows`
  (test-only use, naming a trait method's error type) — an
  already-vetted workspace dependency, new usage site only.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-flows +4 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `utils/` sweep 4: `_strip_json_code_fence`; flag `yaml_utils`/rest of `_schema_utils`/`cache_performance_analyzer`; correct C0941
**2026-08-24** · (link added once this PR is opened)

Closes out the `utils/` sweep's last few files.

- **Added:** `adk-genai::schema_utils::strip_json_code_fence` (C0944) —
  hand-rolled rather than adding `regex` as a new usage site, since the
  source's anchored-fullmatch-with-DOTALL regex reduces to plain string
  slicing.
- **Added, flagged:** manifest row C0943 (`yaml_utils.py`) — needs a
  new YAML crate dependency decision (no YAML crate is a dependency
  anywhere in this workspace); hand-rolling a YAML parser/dumper isn't
  the "simple and deterministic" tier this port's hand-roll precedent
  covers.
- **Added, flagged:** manifest row C0945 (the rest of
  `_schema_utils.py`) — every function dispatches on an arbitrary
  Python type object at runtime (`TypeAdapter`, `get_origin`/`get_args`);
  this workspace has no generic-validation layer to port it onto.
- **Added, deferred:** manifest row C0946
  (`cache_performance_analyzer.py`) — its dependencies (`CacheMetadata`,
  `Event.cache_metadata`/`.author`) already exist, unlike C0941/C0945's
  deeper blockers, making it a good future-batch candidate; not
  attempted in this batch.
- **Corrected:** C0941's evidence (`agent_info.py`) after investigating
  it for a possible follow-up batch — it's blocked on prerequisite
  architecture, not merely "larger than one batch": `LlmAgent` has no
  `name`/`sub_agents` fields (those live only on `BaseAgent`, and
  `LlmAgent` isn't wired into its tree yet), and `LlmAgent::tools` is
  `Vec<ToolUnion>` where every variant is still an opaque placeholder.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-genai +5 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `utils/` sweep 3: `_telemetry_config`, plus flagging `context_utils`/`agent_info`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk-platform::telemetry_config::{get_user_config_path,
  read_telemetry_consent, write_telemetry_consent}` (C0942) —
  reads/writes the ADK global telemetry-consent preference at
  `~/.adk/config.json`, preserving other keys already in the file.
- **Disclosed adaptation:** `pathlib.Path.home()` narrows to reading
  `$HOME` directly (no new `dirs`-crate dependency; a disclosed gap on
  Windows). The on-disk file is compact JSON, not the source's
  pretty-printed `indent=2` — cosmetic only, since this same code is
  the file's only reader.
- **Added, flagged:** manifest row C0940 (`context_utils.py`) — `Aclosing`
  has no Rust equivalent (no async-generator protocol to close);
  `find_context_parameter`'s reflection-based Context-parameter
  detection is already handled differently for this workspace's one
  built caller (`FunctionTool`'s fixed closure signature, per its own
  C0404 module doc) — left `REQUIRED` since other source callers
  (`mcp_tool.py`, the automatic-function-calling util) aren't ported
  here yet.
- **Added, deferred:** manifest row C0941 (`agent_info.py`) — genuinely
  portable (every dependency it needs already exists in this
  workspace) but a larger, multi-type-spanning capability than fit
  alongside this batch's smaller rows; left `REQUIRED` for its own
  future batch.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-platform +4 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `utils/` sweep 2: `_dependency`/`_telemetry_context`/`_serialized_base_model`/`output_schema_utils`
**2026-08-24** · (link added once this PR is opened)

The remaining small `utils/` files, found continuing the same sweep as the
previous PR. Four new manifest rows (C0935-C0938).

- **Added:** `adk-errors::missing_extra::missing_extra` (C0935) —
  the standard "install this extra" message for a missing optional
  dependency. Returns a plain `String` (no caller in this workspace
  needs it yet; none of the source's optional-dependency-gated
  subsystems that call it are built here).
- **Added:** `adk-platform::visual_builder_context::{is_visual_builder,
  set_visual_builder}` (C0936) — forward-pulled into `adk-platform`
  even though its source file is `utils/`, not `platform/`, since it's
  the same shape of thing that crate already exists for.
- **Disclosed adaptation:** the source's `contextvars.ContextVar` is
  async-task-scoped; this port uses a `thread_local!` instead (only
  thread-scoped), the same narrowing already disclosed for
  `ClientLabelScope` (C0932).
- **Added:** manifest row C0937 (`SerializedBaseModel`) — no new code;
  this is a structural convention every applicable struct in this port
  already satisfies via `#[rusty_serde(rename_all = "camelCase")]`
  (16 files, cross-referenced in the row's evidence). Disclosed gap:
  the source's `populate_by_name=True` (accepting the original
  snake_case field name on input too) hasn't been verified to have an
  equivalent here.
- **Added:** `adk-models::output_schema_utils::can_use_output_schema_with_tools`
  (C0938) — a deprecated wrapper delegating to
  `gemini_output_schema_and_tools`. Python's `@deprecated` becomes
  Rust's own `#[deprecated]` attribute (a genuinely close analog, both
  static/type-checker-visible). Disclosed narrowing: the source's
  `LiteLlm`-instance always-`True` special case is dropped, since
  `LiteLlm` isn't ported in this workspace.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-errors +1, adk-platform +2, adk-models +1 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `utils/` sweep: `variant_utils`/`_json_utils`/`_client_labels_utils`/`_debug_output`/`vertex_ai_utils`
**2026-08-24** · (link added once this PR is opened)

A sweep of `google/adk/utils/` files never fully inventoried, found while
looking for the next small self-contained batch after C0796. Five new/
completed manifest rows (C0930-C0934).

- **Added:** `adk-genai::json_utils::safe_json_loads` (C0931) — generic
  over the target `Deserialize` type, since Rust has no equivalent to the
  source's dynamically-typed `Any` return; returns `Result<T, String>`
  with an optional context label folded into the message.
- **Added:** `adk-models::google_client_headers::{ClientLabelScope,
  EVAL_CLIENT_LABEL}` (C0932) — the `_client_labels_utils.py` piece this
  file's module doc previously deferred (only the two unconditional
  tracking labels were ported). The source's `@contextmanager` becomes an
  RAII guard (`Drop`-based restore, same pattern as
  `adk-features::TemporaryFeatureOverride`).
- **Disclosed adaptation:** the source's `contextvars.ContextVar` is
  async-task-scoped (follows a value across an `.await` resuming on a
  different worker thread); this port uses a `thread_local!` instead,
  which is only thread-scoped. Nothing in this workspace exercises that
  boundary yet, so this is a disclosed gap, not a proven bug.
- **Added:** `adk-events::debug_output::print_event` (C0933) — prints an
  `Event` to stdout; text parts always shown, tool calls/results/code
  execution/inline-or-file-data parts only when `verbose`. Tool-call
  args/response rendering uses compact JSON instead of Python's
  `str()`/`repr()` (same disclosed divergence as `to_user_content`,
  C0928); `_truncate` truncates by byte length, walking back to the
  nearest `char` boundary rather than the source's by-code-point slicing.
- **Added:** `adk-models::vertex_ai_utils::get_express_mode_api_key`
  (C0934), ported exactly.
- **Added:** a dedicated parity test for
  `adk-models::capabilities::get_google_llm_variant`, and its own
  manifest row (C0930, `variant_utils.py`) — already ported in an
  earlier forward-pull batch but never linked or directly tested.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (adk-genai +3, adk-events +6, adk-models +6 new tests, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — Close out C0796 (`env_utils.py`)
**2026-08-24** · (link added once this PR is opened)

- **Added:** 4 new tests for `is_enterprise_mode_enabled`'s
  `GOOGLE_GENAI_USE_ENTERPRISE`-vs.-deprecated-`GOOGLE_GENAI_USE_VERTEXAI`
  precedence. Both `is_env_enabled`/`is_enterprise_mode_enabled` were
  already ported in an earlier Phase 3 forward-pull batch (needed by
  `BaseLlm.capabilities`'s deprecated name-based fallback) but never
  linked back to their manifest row, and `is_enterprise_mode_enabled`
  itself had no dedicated test until now.
- **Disclosed adaptation:** `is_env_enabled` here takes an
  already-looked-up value rather than the source's env-var *name* —
  every call site in this port already has its own value in hand.
  `adk-features::feature_registry`'s own `is_env_enabled` is a
  separate, intentionally undeduplicated local copy (per its own
  module doc) that does take a name directly, for its dynamically
  constructed `ADK_ENABLE_<NAME>` variable.

Manifest row C0796 updated to `DONE`. `CHANGELOG.md` and
`RELEASE_NOTES.md` updated.

## Test plan

- [x] `cargo build --workspace`
- [x] `cargo test --workspace` (4 new tests in `adk-models`, 216 total, all passing)
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo fmt --check`

---

## PR #TBD — `content_utils` shared module (`adk-genai`)
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_genai::content_utils::{extract_text_from_content,
  to_user_content, ToUserContentInput,
  SKIP_THOUGHT_SIGNATURE_VALIDATOR}` (C0927/C0928/C0929), ported from
  `google.adk.utils.content_utils`.
- **Consolidation:** `is_audio_part`/`filter_audio_parts` (C0136) were
  already ported, but as a private duplicate local to
  `adk-models::gemini_llm_connection` — `content_utils.py` itself
  wasn't in scope yet when that earlier batch needed just those two
  functions. Moved here as the single source of truth;
  `gemini_llm_connection.rs` now calls through to this module instead
  of keeping its own copy.
- **Newly found, previously uninventoried:** `extract_text_from_content`/
  `to_user_content`/`SKIP_THOUGHT_SIGNATURE_VALIDATOR` had no manifest
  rows at all before this batch — only 2 of `content_utils.py`'s 5
  exports were ever inventoried. Per the boundary contract, a
  capability missing from the original inventory still gets tracked
  once found — added as C0927/C0928/C0929, appended rather than
  renumbering the existing sequential range. Also corrects the
  manifest's stale "832 rows, C0001-C0832" footer comment, which
  predated several later append batches.
- **Adaptation (`to_user_content`):** the source's runtime `isinstance`
  dispatch across `Content`/`str`/`BaseModel`/`dict`/`list`/anything
  else becomes an explicit `ToUserContentInput` enum — Rust has no
  runtime `isinstance`, and callers already know which shape they hold.
  A `BaseModel` input becomes whatever the caller's own typed struct
  serializes to via `rusty_serde::json::to_value` first, matching the
  "the boundary already deals in `Value`" convention used throughout
  this port.
- **Disclosed, low-severity:** the source's "anything else →
  `str(value)`" catch-all has no Rust equivalent; non-string,
  non-`Content` values are formatted as compact JSON instead (e.g. a
  bool renders `true`/`false` rather than Python's `True`/`False`).
- **Disclosed, ahead of its own caller:** `SKIP_THOUGHT_SIGNATURE_VALIDATOR`'s
  sole source consumer (`ReflectAndRetryToolCallsPlugin`) isn't built
  in this workspace yet.
- 13 new tests.

---

## PR #TBD — Fix `InMemoryArtifactService` malformed-input handling; close out C0259/C0260/C0261
**2026-08-24** · (link added once this PR is opened)

- **Fixed:** `InMemoryArtifactService::save_artifact` now panics on a
  malformed `artifact` value instead of silently substituting an
  empty `Part` — matching the source's `ensure_part`/`model_validate`
  raising a `ValidationError` rather than losing data quietly. This
  is `ensure_part`'s (C0259) entire behavior in this port, disclosed:
  the source's `Union[Part, dict]` input collapses to a single `Value`
  at this trait boundary, since nothing here can pass an
  already-constructed `Part` object through `ArtifactService::save_artifact`
  — only a `Value`.
- **Also:** marks C0259 (`ArtifactVersion`/`ensure_part`) and C0260
  (`BaseArtifactService`'s full 7-method abstract interface) `DONE` —
  both were already fully satisfied by the `InMemoryArtifactService`
  batch (PR #61), just not yet reflected in the manifest. Fills in
  C0261's evidence (package lazy-export): the eager-vs-lazy
  `__init__.py` import distinction has no analogue in this port's flat
  `pub mod` crate structure, the same disclosure C0493 already gave
  for `auth/__init__.py`'s own asymmetry.
- 1 new test.

---

## PR #TBD — Session-state utilities (`_session_util`)
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::session_util::{decode_model,
  extract_state_delta, make_json_safe_state,
  extract_json_safe_state_delta}` (C0209, partial) plus
  `State::{APP_PREFIX, USER_PREFIX, TEMP_PREFIX}` (completing C0205's
  manifest evidence — the dict-with-pending-delta wrapper itself was
  already ported in an earlier Phase 2 forward-pull batch, just never
  linked back to its row).
- Reconciles `services.rs`'s own pre-existing private
  `TEMP_STATE_PREFIX` duplicate constant to reference `State::TEMP_PREFIX`
  as the single source of truth.
- **Ported exactly:** `extract_state_delta`/`extract_json_safe_state_delta`
  (app/user/session prefix split, `temp:` keys dropped).
- **Disclosed near-no-op:** `make_json_safe_state` is effectively an
  identity function here — this port's state is already
  `BTreeMap<String, Value>`, and `Value` can only ever hold
  JSON-representable variants by construction, so there is no value
  this port's state can hold that could fail the source's coercion.
  The function still exists so `extract_json_safe_state_delta` composes
  the same way and a future persistent backend has the named call site
  ready.
- **Disclosed narrowing:** `decode_model` collapses the source's two
  distinct failure modes (a primitive non-dict value → `None`; a
  malformed-but-dict-shaped value → a raised `ValidationError`) to a
  uniform `None` — a caller here can't distinguish the two the way the
  source's exception-vs-`None` split does.
- **Disclosed, ahead of its own caller:** nothing in this port's
  `SessionService`/`InMemorySessionService` routes
  `extract_state_delta`'s "app"/"user" output to any real cross-session
  shared storage yet (`Session`/`State` have no such architecture) —
  this utility is real and tested, ahead of the architecture that
  would consume it, the same situation `remote_mcp_server.rs` disclosed.
- 8 new tests.

---

## PR #TBD — Feature-flag registry (`adk-features`)
**2026-08-24** · (link added once this PR is opened)

- **Added:** a new `adk-features` crate —
  `feature_registry::{FeatureName, FeatureStage, FeatureConfig,
  feature_config, is_feature_enabled, override_feature_enabled,
  TemporaryFeatureOverride}` (C0643-C0646/C0648-C0649), ported from
  `google.adk.features._feature_registry`. The three-tier
  `is_feature_enabled` precedence (programmatic override → `ADK_ENABLE_
  <NAME>`/`ADK_DISABLE_<NAME>` env vars → registry default), and the
  once-per-process non-stable-feature notice, all ported and tested.
- **Adaptation:** `temporary_feature_override`'s `@contextmanager`
  becomes an RAII guard (`TemporaryFeatureOverride`, restore-on-`Drop`)
  — the standard Rust idiom for "run on scope exit including unwind,"
  matching the source's `try`/`finally` semantics exactly.
- **Compile-time strengthening, not a narrowing:** the registry is a
  fixed, exhaustive `match` over every `FeatureName` variant rather
  than a mutable dict (nothing in this batch's scope calls the
  source's decorator-driven dynamic registration) — so there is no
  `FeatureName` value this port can construct that lacks a registry
  entry, making the source's "raises on an unregistered name" branches
  structurally unreachable here.
- **Disclosed:** the notice emitted when a non-stable feature turns on
  uses `eprintln!` rather than `warnings.warn` — this workspace has no
  logging/warning framework, following the same precedent
  `adk_models::capabilities::is_enterprise_mode_enabled`'s own
  deprecation notice already established. Member count: 38 counted
  directly off the source file, not the 37 the manifest description
  estimates — source is ground truth.
- **Not this batch:** the `experimental`/`working_in_progress`/`stable`
  decorators (C0647) — Rust has no runtime-decorator analog to gate an
  arbitrary object behind a flag; left `REQUIRED`, undecided.
- 9 new tests.

---

## PR #TBD — `InMemoryArtifactService`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::in_memory_artifact_service::InMemoryArtifactService`
  (C0265) — the first real `ArtifactService` implementation: path-keyed
  `Vec<ArtifactEntry>` storage (version = list length at save time,
  monotonic per path), the `"user:"`-namespace vs. session-scoped path
  split, MIME-type detection (inline data/text/file-reference branches,
  including the artifact-reference-URI resolve-and-validate-scope
  recursion via `artifact_util`, C0262-C0264 — its first real caller),
  the empty-artifact sentinel checks on load, and the `memory://`
  canonical URI scheme.
- **Also:** widens `adk_agents::services::ArtifactVersion` from a
  `Value` placeholder to a real struct (same "widen from placeholder"
  precedent as `AuthCredential`/`MemoryEntry`) and extends
  `ArtifactService` with `delete_artifact`/`list_versions`/
  `list_artifact_versions` to match the source's full abstract
  interface — the pre-existing `FakeArtifactService` test double in
  `context.rs` updated to match.
- **Disclosed, predating this batch:** the trait's `session_id` is a
  required `&str` on every method, not the source's `Optional[str]` —
  so the source's "no session in play" branches have no representable
  path through this trait signature; `list_artifact_keys` always
  returns the combined session+user listing. `artifact`/return values
  stay opaque `Value` at the trait boundary rather than a typed
  `Part`; this service deserializes/reserializes internally, the same
  pattern `ExampleTool`/`PreloadMemoryTool`/`LoadArtifactsTool` use.
- **Disclosed:** version lookup only supports `None` (latest) or a
  non-negative index, not the source's Python-list negative indexing —
  unreachable in practice since `save_artifact` only ever returns
  non-negative version numbers.
- **Disclosed:** where the source raises `InputValidationError`/
  `ValueError` (invalid path segment, unsupported artifact type,
  invalid/out-of-scope artifact reference), this port panics — the
  closest Rust analog to an uncaught exception through this trait's
  non-`Result` methods, the same pattern
  `InMemoryMemoryService::add_memory` already disclosed.
- 12 new tests.

---

## PR #TBD — `artifact_util` (artifact URI scheme + path safety)
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::artifact_util::{ParsedArtifactUri,
  parse_artifact_uri, get_artifact_uri, is_artifact_ref,
  validate_artifact_reference_scope, validate_path_segment}`
  (C0262/C0263/C0264) — the canonical
  `artifact://apps/{app}/users/{user}/[sessions/{sid}/]artifacts/{filename}/versions/{v}`
  URI scheme (parse/construct, round-trip tested), the security
  boundary preventing cross-tenant artifact-reference escapes, and the
  path-segment validator every artifact backend needs for app/user/
  session identifiers (rejects empty/null-byte/absolute/drive-qualified/
  traversal segments, both `/`- and `\`-separated).
- Pure string/regex logic, no I/O — builds on the already-real
  `adk_errors::input_validation::InputValidationError`.
- **Adaptation:** `is_artifact_ref` reads `"fileUri"` out of
  `Part.file_data`'s opaque flattened `rest` map rather than a typed
  field — this port's `MediaBlobStub` has no typed `file_uri` field
  (`adk-genai::content`'s own documented narrowing) — the same
  read-a-flattened-field pattern `load_artifacts_tool.rs` already uses
  for `inline_data.rest.get("data")`.
- **Disclosed, not built yet:** `InMemoryArtifactService` (C0265)
  doesn't exist in this port yet, so nothing produces a real
  artifact-backed `Part` in a live turn to exercise these against —
  this utility is real and tested, ahead of its own only caller, the
  same situation `remote_mcp_server.rs` disclosed.
- 17 new tests.

---

## PR #TBD — `InMemoryMemoryService`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::in_memory_memory_service::{InMemoryMemoryService,
  format_timestamp, UNKNOWN_SESSION_ID}` (C0243 partial/C0244/C0245
  partial/C0247/C0248/C0249) — the first real implementation of the
  `MemoryService` trait (a Phase 6 placeholder), the same
  "first real backend narrows a placeholder trait to its actual
  contract" moment `InMemorySessionService` was for `SessionService`.
  A `Mutex`-guarded, in-process keyword-search memory backend —
  "prototyping purpose only", matching the source's own docstring.
- **Ported exactly:** `add_session_to_memory` (wholesale per-session
  overwrite) vs. `add_events_to_memory` (additive, dedup by event id)
  — distinct semantics, each covered by its own test; `search_memory`'s
  keyword-match scoring (Unicode-aware `\w+` tokenization, a substring
  fallback for non-ASCII query words, snapshot-under-the-lock then
  score-outside-it to avoid the source's own documented
  "dictionary changed size during iteration" race, a 10-result cap,
  and a stable sort so equally-scored memories stay in insertion
  order); only events with non-empty content parts are retained on any
  write path.
- **Also ported:** `format_timestamp`, using a hand-rolled, well-known
  public-domain epoch-to-calendar algorithm (Howard Hinnant's
  `civil_from_days`) rather than pulling in a date/time crate.
- **Disclosed narrowing (`format_timestamp`):** the source's bare
  `datetime.fromtimestamp(timestamp)` uses the host's *local*
  timezone; this port formats in UTC, since true local-time conversion
  needs a full IANA timezone-database crate (`chrono-tz` or similar),
  not added in this batch. Since `MemoryEntry.timestamp` is forwarded
  to the LLM verbatim rather than parsed back, this is real but
  low-severity (a different wall-clock hour shown to the LLM, not
  corrupted data).
- **Disclosed narrowing (`add_memory`):** the source's `BaseMemoryService`
  gives `add_memory` a default that raises `NotImplementedError`
  unless overridden, and `InMemoryMemoryService` never overrides it.
  This port's pre-existing `MemoryService` trait method has no
  `Result` to signal "unsupported" through, so
  `InMemoryMemoryService::add_memory` panics via `unimplemented!()` —
  the closest Rust analog to an uncaught exception.
- **Disclosed, already narrowed by an earlier batch:** the source's
  `add_events_to_memory(session_id: str | None = None)` falls back to
  an `__unknown_session_id__` sentinel when no session id is given;
  this port's pre-existing `MemoryService::add_events_to_memory` trait
  signature takes a required `&str`, and its sole caller
  (`Context::add_events_to_memory`, built before any real
  `MemoryService` existed) always supplies the real session id — so
  that fallback path isn't reachable through `Context` today. The
  sentinel constant is kept as `UNKNOWN_SESSION_ID` for a follow-up
  that corrects the trait signature.
- 11 new tests.

---

## PR #TBD — OAuth2/mTLS utility functions
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::oauth2_util::{normalize_oauth_scopes,
  OAuthScopes, is_non_mtls_googleapis_endpoint,
  effective_googleapis_endpoint, use_client_cert_effective,
  update_credential_with_tokens}` (C0509/C0531 partial/C0532) — three
  small, self-contained OAuth2/mTLS helpers that build on the already-
  real `AuthCredential`/`OAuth2Auth` without needing `authlib` or
  `google-auth`: `_normalize_oauth_scopes` (dict-or-list scopes → a
  flat list, C0509, DONE), the pure env-var/URL-string half of Google
  token-endpoint mTLS routing (C0531, partial), and
  `update_credential_with_tokens` (C0532, DONE).
- **No new dependency:** URL-host inspection/rewriting for the mTLS
  routing piece is done with plain string operations, not a URL-
  parsing crate — the same "hand-roll a small self-contained algorithm"
  precedent as `bash_tool.rs`'s `shlex_split`.
- **Disclosed, C0531 scoped to its portable half:** the real
  client-certificate loading/mounting (`configure_session_for_mtls`,
  `MtlsClientCerts`, `get_api_endpoint`'s cert-availability check)
  needs `google.auth.transport.mtls` — no such crate is a workspace
  dependency, so it's unported. `use_client_cert_effective` always
  takes the source's `ImportError` env-var-fallback branch, since
  `google.auth`'s own `mtls.should_use_client_cert()` probe isn't
  available here — this port can report "the env var says use a cert"
  but never "a cert is actually available". Because the real cert step
  is unported, the full call-site gating (rewrite the token endpoint
  only once a certificate is actually mounted) isn't wired up
  anywhere yet — left with `create_oauth2_session` (C0530), still
  fully `REQUIRED`.
- 17 new tests.

---

## PR #TBD — Auth credential camelCase wire format
**2026-08-24** · (link added once this PR is opened)

- **Added:** `#[rusty_serde(rename_all = "camelCase")]` on
  `HttpCredentials`/`HttpAuth`/`OAuth2Auth`/`ServiceAccount`/
  `AuthCredential` (C0501, partial), matching
  `BaseModelWithConfig`'s `alias_generator=alias_generators.to_camel`.
- **Deliberate exception:** `ServiceAccountCredential` keeps its
  snake_case field names. Its fields mirror a real downloaded GCP
  service-account JSON key file verbatim, and that file format is
  itself snake_case (`project_id`, `private_key_id`, ...) — the
  source's `populate_by_name=True` is what lets Pydantic accept either
  form there, but this port's `rename_all` sets one fixed wire name
  with no dual-name accept, so applying camelCase would have broken
  parsing an actual key file, the opposite of matching the source.
- **Disclosed narrowing:** `populate_by_name=True`'s dual-name accept
  (snake_case *or* camelCase on input) has no port for the structs
  that *do* get `rename_all` — this port's `Deserialize` only accepts
  the one configured wire name.
- **Also:** marks manifest row C0502 (`AuthCredential.resource_ref`)
  `DONE` — the field was already ported in the prior `auth_credential`
  batch (PR #56), just not yet reflected in the manifest. Marks C0500
  (secret redaction) with detailed `Partial:` evidence: this port has
  no `Debug`/log output path for these structs to harden yet, so
  there's nothing to redact.
- 3 new tests.

---

## PR #TBD — Auth credential schemes (Phase 9 start)
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::auth_credential::{AuthCredentialTypes,
  HttpCredentials, HttpAuth, TokenEndpointAuthMethod, OAuth2Auth,
  ServiceAccountCredential, ServiceAccount, AuthCredential}`
  (C0494/C0495/C0496/C0497/C0499) — the credential-scheme data models
  ported from `google.adk.auth.auth_credential`, starting this repo's
  Phase 9 (auth). Widens `adk_agents::services::AuthCredential` from
  its former `Value` placeholder to the real struct here, the same
  "widen from placeholder once a real consumer needs the shape"
  precedent as `MemoryEntry`/`SearchMemoryResponse` (C0423).
- **Adaptation:** `ServiceAccount::new` is a fallible constructor
  running the source's `_validate_config` `model_validator`'s exact
  two checks (a credential is required unless
  `use_default_credential`; `audience` is required when
  `use_id_token`), returning `Result<Self, ServiceAccountError>` in
  place of Pydantic's constructor-time `ValueError` — both reject the
  same invalid states with the same messages.
- **Disclosed narrowing, shared by every struct in this batch:** the
  source's `extra="allow"` lets callers attach arbitrary unmodeled
  keys, preserved (redacted in `repr`) rather than dropped. A Rust
  struct has a fixed field set — an unmodeled key round-tripped
  through one of these structs is silently dropped, not
  preserved-but-redacted. Non-repr secret fields (`Field(repr=False)`)
  have no redaction surface to port either: this port's derived
  `Debug` isn't used to serialize/log these structs anywhere yet.
- **Not this batch:** `AuthScheme`/`OpenIdConnectWithConfig` (C0498)
  live in the separate `auth_schemes.py`, left for their own batch —
  `OAuth2Auth` already carries every field the OpenID Connect scheme
  reuses, so nothing here is blocked on that follow-up. The
  `auth/__init__.py` re-export-asymmetry behavior itself (C0493) isn't
  addressed either: this port's crates never re-export any module's
  contents at the crate root for *any* module, so the specific
  "some names get a shortcut, others don't" asymmetry the source
  exhibits has no distinguishing analogue to replicate here.
- 12 new tests.

---

## PR #TBD — `RemoteMcpServer`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::remote_mcp_server::{RemoteMcpServer,
  HeaderProvider}` (C0491, partial) — the declarative model describing
  a server-side MCP server for the Managed Agents API: `ManagedAgent`
  forwards the server's URL and headers to `interactions.create`, and
  the Interactions backend opens the MCP session and runs the tools —
  ADK itself never connects to this server, contrast with the
  client-side `McpToolset`. Adds a new `resolved_headers` method
  implementing the documented `header_provider`-wins-on-conflict merge
  order.
- **Disclosed:** `extra='forbid'`/`arbitrary_types_allowed` (Pydantic
  schema/validation concerns) have no Rust equivalent to port — a
  struct literal can't carry unknown fields in the first place.
- **Disclosed, not built yet:** nothing in this port constructs a
  `RemoteMcpServer` in a live turn — the Managed Agents API
  `interactions.create` request path (`ManagedAgent`) is a separate,
  larger, unbuilt capability. Unlike sibling row C0490 (`NodeTool`,
  blocked outright on the unbuilt `workflow::BaseNode` graph engine),
  this one has no missing dependency — it's simply ahead of its own
  only caller.
- 5 new tests.

---

## PR #TBD — `request_input_tool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::request_input_tool::{request_input,
  request_input_tool, REQUEST_INPUT_FUNCTION_CALL_NAME}` (C0492,
  partial) — a `LongRunningFunctionTool` asking the user a question
  (`message`, mandatory) with an optional `response_schema`, always
  returning `Null` to trigger the long-running-interrupt mechanism —
  ported from `_request_input_tool.py`'s `_request_input_func` and
  `request_input` instance.
- **Disclosed forward-reference:** `REQUEST_INPUT_FUNCTION_CALL_NAME`
  ("adk_request_input") is defined in this new module rather than in
  `adk-flows::functions` — its actual source location
  (`flows/llm_flows/functions.py`) — because `adk-flows::functions`
  (C0191/C0192, already ported) hasn't built the HITL request-input
  wiring that constant belongs to yet, and `adk-tools` doesn't depend
  on `adk-flows`. A follow-up batch wiring that piece should reuse
  this constant's value rather than defining a second copy.
- **Disclosed narrowing:** the source's `logging.info` call on each
  invocation isn't ported — no logging framework is adopted by this
  workspace yet (same disclosed omission as `preload_memory_tool.rs`).
- 6 new tests.

---

## PR #TBD — Built-in Gemini grounding tools
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::{google_search_tool::GoogleSearchTool,
  google_maps_grounding_tool::GoogleMapsGroundingTool,
  enterprise_search_tool::EnterpriseWebSearchTool,
  url_context_tool::UrlContextTool}` (C0428/C0430/C0431/C0432,
  partial) — the four built-in Gemini grounding tools, each a thin
  `BaseTool` whose `process_llm_request` checks whether the request
  targets a Gemini model (or has the model-ID check disabled) and, if
  so, appends a built-in-tool marker object (e.g. `{"googleSearch":
  {}}`) to `llm_request.config.tools`. Adds two new shared helpers:
  `adk_tools::append_tools::append_built_in_tool_marker` and
  `adk_tools::model_name_utils::{is_gemini_model_id_check_disabled,
  is_managed_agent}`.
- **Disclosed narrowing, shared by all four:** `BaseTool::process_llm_request`
  returns `BoxFuture<'a, ()>` with no `Result` to propagate through —
  the same structural gap disclosed for every other built-in-tool
  port in this workspace. An unsupported model here simply doesn't
  get its marker appended, rather than the source's hard `ValueError`.
- **Disclosed narrowing:** `is_managed_agent()` always returns `false`
  — this port's `LlmRequest` has no `_is_managed_agent` field to check.
- **Disclosed narrowing:** `GoogleSearchTool.bypass_multi_tools_limit`
  is stored for API-shape parity but nothing in this port enforces
  the "Gemini restricts `google_search` to sole-tool use" limitation
  it would bypass — that enforcement is deferred with the rest of the
  still-unresolved C0171 request-processor-wiring gap.
- **Matched, not narrowed:** `EnterpriseWebSearchTool` and
  `GoogleMapsGroundingTool` never check `_is_managed_agent` in the
  source either — this port matches that omission exactly rather than
  adding a check the source itself doesn't have.
- 12 new tests.

---

## PR #TBD — `LoadMcpResourceTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::load_mcp_resource_tool::{LoadMcpResourceTool,
  McpResourceProvider}` (C0426, partial) — the full
  list→instruction-inject→function-response-detect→per-resource-read→
  append flow, reusing `load_artifacts_tool::maybe_base64_to_bytes`
  (C0425) for the same decode-with-placeholder-fallback shape the
  source's `_mcp_content_to_part` uses.
- **Disclosed narrowing:** no real `McpToolset` exists yet — the
  actual MCP client (stdio/SSE/streamable-HTTP transport) is its own,
  much larger capability (C0540-C0542), not built in this port. This
  batch defines a minimal `McpResourceProvider` trait carrying just
  the two operations this tool actually calls
  (`list_resources`/`read_resource`) — the same "placeholder trait,
  forward-referencing a not-yet-built phase" pattern
  `adk-agents::services` already uses for `MemoryService`/
  `ArtifactService`, so `LoadMcpResourceTool` itself is fully real and
  tested against a stub provider even though no real MCP client
  exists yet to plug into it.
- **Bugfix, shared with C0425:** refined `maybe_base64_to_bytes` — a
  non-empty input with no recognizable base64 characters at all
  previously decoded to a silent empty byte vector rather than
  signaling failure (the lenient URL-safe fallback pass never
  early-returns `None` by construction). It now correctly reports a
  decode failure in that case, matching the source's actual intent
  even though Python's own lenient `urlsafe_b64decode` fallback has
  the identical latent quirk. This makes the "could not be decoded"
  placeholder path genuinely reachable rather than dead code.
- 7 new tests.

## PR #TBD — `load_web_page`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::load_web_page::{load_web_page,
  LoadWebPageTool}` (C0427, partial) — fetches a URL and extracts its
  text. The source's substantial SSRF-hardening core is ported in
  full: URL scheme/host/port validation, `localhost`/`*.localhost`
  rejection, DNS resolution with every resolved address vetted before
  any connection is attempted, IP-literal vetting, IPv4/IPv6
  global-reachability classification, embedded-IPv4-in-IPv6 checks
  (mapped/6to4/NAT64/deprecated-compatible forms), IP-pinned
  connection with the original Host header preserved, and disabled
  redirects.
- **Verification methodology:** the IP-classification logic was
  transcribed field-for-field from CPython 3.11's own `ipaddress.py`
  (consulted locally at `/usr/lib/python3.11/ipaddress.py` — the exact
  reference the source's `is_global` calls resolve to), then
  ground-truthed by running a 30+-address battery through the real
  Python `ipaddress` module before writing the Rust tests. The
  battery specifically confirms the embedded-IPv4 check catches cases
  the plain IPv6 `is_global` check alone misses: `64:ff9b::169.254.169.254`
  and `::169.254.169.254` both read as "global" by the raw check
  (NAT64/IPv4-compatible ranges aren't in IPv6's private-network
  list) but are correctly blocked once the embedded IPv4 target
  (169.254.169.254, the cloud metadata endpoint) is extracted and
  checked — exactly the vulnerability class this hardening exists to
  close.
- **Implementation notes:** IP-pinning uses
  `reqwest::blocking::ClientBuilder::resolve` rather than a hand-rolled
  connection adapter (the source's own `_PinnedAddressAdapter`) —
  `resolve()` keeps the request URL pointed at the original hostname,
  so the correct Host header is sent automatically without manual
  header surgery. New `adk-tools` dependencies: `reqwest` (already a
  workspace dependency via `adk-models`/`gemini.rs` — this is a new
  usage site, not a new supply-chain surface), `url` (the exact crate
  `reqwest::Url` re-exports, added explicitly for `url::Host`'s
  already-parsed Domain/Ipv4/Ipv6 variant), `regex` (already a
  workspace dependency, new usage site).
- **Disclosed narrowings:** no proxy-aware branching — this port
  never reads `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and always does
  the direct IP-pinned fetch (strictly more restrictive than the
  source, not a safety regression, but a real behavior gap for
  proxy-dependent environments). HTML text extraction has no HTML5
  parser behind it — a regex-based tag/script/style stripper plus a
  handful of common named-entity decodes stands in for
  `BeautifulSoup`+`lxml`, with no DOM-aware whitespace collapsing.
  The live-fetch path itself (the real `reqwest` GET) has no
  automated test in this sandboxed environment — it depends on
  outbound network/DNS this test run may not have, and the tool's own
  safety design correctly rejects `127.0.0.1`/`localhost` before any
  request is attempted, so a local mock server can't stand in without
  weakening the real check being tested. Every other piece is
  covered: 15 new tests.

## PR #TBD — `LoadArtifactsTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::load_artifacts_tool::{LoadArtifactsTool,
  as_safe_part_for_llm}` (C0425, partial) — lists artifacts and
  injects them into the LLM request on demand. Ports MIME
  normalization/classification (Gemini-supported-inline-prefix/type
  checks, the SVG/XML unsupported-subtype exclusion), a hand-rolled
  base64 decoder (standard-strict then URL-safe-lenient, mirroring the
  source's own two-attempt fallback), text-like decoding, the
  binary-placeholder fallback, and the full
  list→instruction-inject→`load_artifacts`-function-response-detect→
  per-artifact-load→append-to-`llm_request.contents` flow (including
  the session-scoped-then-`user:`-prefixed cross-session fallback).
  `tool_context.load_artifact`/`list_artifacts`'s opaque `Value`
  results are parsed back into a typed `Part` via its own
  `Deserialize` impl, the same pattern `ExampleTool`/
  `PreloadMemoryTool` use for `user_content`.
- **Disclosed narrowings (module doc, at length):** no DOCX regex
  text extraction — needs a zip reader, and no such crate is a
  workspace dependency; a `.docx` artifact falls through to the
  generic binary-placeholder response instead of extracted text. No
  spreadsheet parsing — needs a `pandas` equivalent this port has
  none of, though this is disabled by default upstream too
  (`enable_spreadsheet_parsing=False`), so it's the same
  optional-dependency treatment the source itself gives it, not a
  narrowing unique to this port. No `process_artifact` custom-callback
  override — every artifact goes through the built-in safety
  conversion.
- 11 new tests.

## PR #TBD — `ExecuteBashTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::bash_tool::ExecuteBashTool` (C0418, partial) —
  runs a validated bash command in a workspace directory via
  `rusty_tokio::process::Command` (no shell — matches the source's own
  `create_subprocess_exec`, so shell operators like `|`/`;`/`&&` are
  never interpreted specially by either port). Ports the command
  validation (prefix allowlist, blocked operators), the mandatory
  confirmation gate (every invocation requires confirmation regardless
  of policy, matching the source exactly), stdout/stderr capture
  (replicating the source's own "empty bytes → `<no _ captured>`
  placeholder" quirk), and Python's negative-on-signal `returncode`
  convention. 15 new tests, covering 13 of the source's own
  `test_bash_tool.py` cases in spirit (confirmation request/reject/
  confirm, prefix allowlist, blocked operators, timeout, nonzero exit,
  stderr capture, cwd).
- **Disclosed narrowings (module doc, at length):** the source's
  `BashToolPolicy.max_memory_bytes`/`max_file_size_bytes`/
  `max_child_processes`/implicit `RLIMIT_CORE` suppression are
  enforced via a `preexec_fn` calling POSIX `setrlimit()`; this port
  has no `libc`/`setrlimit` binding, and adding one wasn't judged
  worth it for three calls, so those fields don't exist on this port's
  `BashToolPolicy` at all (a config field with no enforcement behind
  it would be worse than no field). A timeout kills only the immediate
  child (`Command::kill_on_drop`) rather than the source's whole
  process group (`os.killpg`) — this port's `Child` has no
  `killpg`-equivalent, so a grandchild the command spawned can survive
  a timeout. A timeout's response carries no partial stdout/stderr —
  the source re-invokes `communicate()` after killing to capture
  whatever was buffered; this port's `Child::wait_with_output`
  consumes the child as one unit with no drain-then-kill-then-redrain
  split. `shlex.split` is replicated with a hand-rolled POSIX-ish word
  splitter (quotes + backslash escaping), not the full shlex grammar.

## PR #TBD — `AgentTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::agent_tool::AgentTool` (C0406, partial) —
  wraps a `BaseAgent` as a callable tool. Spins up an isolated
  `InMemorySessionService` session (forwarding the parent's
  non-`_adk`-prefixed state as the child's initial state, matching
  the source's own filter), runs the wrapped agent for one turn via
  the real `adk_runners::Runner`, forwards state deltas back to the
  parent tool context as each event arrives, and merges the last
  response event's non-thought text parts into the tool's return
  value — falling back to the last error message when there's no
  usable text. Adds a new `adk-tools` → `adk-runners` dependency edge
  (verified non-circular) and a new `ToolError::NestedRunFailed`
  variant for session-creation/nested-run failures.
- **Known limitations (disclosed in the module doc):**
  input/output-schema-driven declaration and response validation
  aren't ported — this port's `BaseAgent` is type-erased with no way
  to recover a concrete `LlmAgent` to read `input_schema`/
  `output_schema` from, the same blocker every Phase 4 processor
  already discloses, so `get_declaration` always uses the generic
  `{"request": string}` fallback shape. `ForwardingArtifactService`
  isn't built (nested run has no artifact service).
  `InMemoryMemoryService` is Phase 6 (nested run has no memory
  service). `include_plugins` has no observable effect —
  `adk_runners::Runner` doesn't accept a `PluginManager` yet, its own
  module doc already discloses this as a genuine gap.
  `propagate_grounding_metadata` and
  `code_execution_result`/`executable_code` part-to-text extraction
  aren't ported (opaque `Part` placeholders in `adk-genai`).
  `support_cfc` disabling isn't ported (no CFC concept in this port).
- 4 new tests.

## PR #TBD — `SetModelResponseTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::set_model_response_tool::SetModelResponseTool`
  (C0437, partial) — the internal output-schema workaround: lets the
  model set its final structured response via a tool call when
  `output_schema` is configured alongside other tools. Sets the
  already-real `EventActions.set_model_response` field
  `output_schema.rs`'s `get_structured_model_response` (C0178)
  reads back.
- **Fundamental adaptation (disclosed in the module doc):** the
  source dynamically builds a Python function signature from
  `output_schema` at runtime and validates the model's call args
  against that schema via Pydantic. This port has neither Python's
  runtime type introspection nor a Pydantic-equivalent JSON-schema
  validator, so `output_schema` is taken as an already-opaque `Value`
  used directly as the declaration's `parameters` — no dynamic
  per-field signature synthesis, no `Field(description=...)`
  re-application. `run_async` can't distinguish "regular object
  schema" from "`list[BaseModel]`" from "raw non-object schema" the
  way the source's `_is_basemodel`/`_is_list_of_basemodel` flags do;
  it uses the same `items`/`response`-single-key convention the
  source's own dynamic signature would produce for the non-object
  cases instead — a reasonable but *not* type-verified stand-in. The
  `ValidationError`-triggered retry-with-feedback path isn't ported —
  there's no validation to fail, a real disclosed gap, not a silent
  one.
- **Unblocks:** the tool-existence half of C0171 (`TransferToAgentTool`
  wiring) and C0178 (`SetModelResponseTool` wiring)'s own disclosed
  gaps — both tools now exist; the remaining gap in each is purely
  the request-processor wiring, which needs `InvocationContext.agent`
  to resolve a concrete `LlmAgent`, the same blocker every other
  Phase 4 processor already discloses.
- 5 new tests.

## PR #TBD — `TransferToAgentTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::transfer_to_agent_tool::{transfer_to_agent,
  TransferToAgentTool}` (C0436, DONE). The bare `transfer_to_agent`
  function sets `EventActions.transfer_to_agent` (already real,
  C0018). `TransferToAgentTool` wraps a `FunctionTool` and builds a
  JSON-schema `agent_name` parameter with an `enum` constraint from
  the given agent names — restricting choices to valid agents,
  preventing the model from hallucinating an invalid target.
- **Unblocks:** the `TransferToAgentTool`-building half of C0171
  (`adk-flows::agent_transfer`) that its own module doc already
  disclosed as deferred "until `BaseTool` existed." Actually attaching
  a built instance into `LlmRequest.config.tools` is left for a
  follow-up batch — it needs the still-unbuilt "resolve
  `InvocationContext.agent` to a concrete `LlmAgent`" wiring every
  other Phase 4 processor is blocked on too.
- 4 new tests.

## PR #TBD — `LoadMemoryTool`, `PreloadMemoryTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::load_memory_tool::{load_memory, LoadMemoryTool}`
  (C0423, DONE) — wraps the already-real `Context::search_memory`
  (Phase 2) in a `FunctionTool`; `process_llm_request` merges the
  declaration then appends the "you have memory" instruction, matching
  the source's `super().process_llm_request()` + append.
- **Added:** `adk_tools::preload_memory_tool::PreloadMemoryTool`
  (C0424, DONE) — automatically (never model-invoked) searches memory
  each turn: parses the opaque `user_content` back into a typed
  `Content` (same pattern `ExampleTool`/C0419 uses), builds the
  `Time: .../author: text` per-memory rendering exactly, and injects
  it via the already-real `LlmRequest::insert_transient_user_content`
  (C0117).
- **Added:** `adk_tools::memory_entry_utils::extract_text` — joins a
  memory entry's text parts, matching `_memory_entry_utils.py`.
- **Widened:** `adk_agents::services::{MemoryEntry,
  SearchMemoryResponse}` promoted from opaque `Value` placeholders to
  real structs matching the source's pydantic models field-for-field
  — the same "widen from placeholder once a real consumer needs the
  shape" precedent already used for `EventCompaction.compacted_content`
  (C0185). `BaseMemoryService` itself is still an unbuilt Phase 6
  trait — nothing in this workspace produces real values of these
  types yet, only consumes the shape.
- **Adaptation:** since `FunctionTool`'s wrapped-closure signature has
  no `Result` to propagate an error through, `load_memory` reports a
  missing memory service as an `{"error": ...}` response value rather
  than a raised exception. `PreloadMemoryTool` swallows a failed
  `search_memory` call silently (matching the source's own
  exception-swallowing control flow) but doesn't log a warning — no
  logging framework has been adopted by this workspace yet.
- 11 new tests across the three modules.

## PR #TBD — New `adk-examples` crate + `ExampleTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** a new `adk-examples` crate (C0829/C0831/C0832, DONE),
  porting `google.adk.examples` in full apart from
  `VertexAiExampleStore`: `Example`/`BaseExampleProvider` (the
  few-shot-example extension point) and
  `example_util::{convert_examples_to_text, build_example_si,
  get_latest_message_from_user}`. Sits alongside `adk-tools` in the
  crate graph (depends on `adk-agents`/`adk-events`/`adk-genai`; not
  depended on by anything above `adk-tools`), so `adk-tools` can
  depend on it without a cycle.
- **Test parity:** all 10 of the source's own `test_example_util.py`
  cases ported, parametrized across `gemini-2.5-flash`/
  `llama3_vertex_agent`/`None` exactly like the source's own pytest
  parametrization — including the model-family-dependent
  function-call-fence-style switch. Plus 4 new tests for
  `get_latest_message_from_user`. 14 new tests total.
- **Adaptation (disclosed in the module doc):** the source's
  function-call-arg rendering for a non-string value, and its
  function-response rendering (`part.function_response.__dict__`),
  both rely on Python's dict/object `str()`/`repr()`. This port has no
  equivalent, so a compact-JSON stand-in is used instead — the same
  disclosed pattern as `adk-flows::instructions_utils::
  value_to_display_string`, duplicated locally here (not reused
  directly, to avoid an `adk-examples`→`adk-flows` crate-graph cycle).
  `FunctionCall.args` being a `BTreeMap` also means a multi-argument
  function call renders its arguments in sorted-key order, not the
  source's call-site order.
- **Added:** `adk_tools::example_tool::ExampleTool` (C0419, DONE) —
  wires the new crate into a `BaseTool`: reads the tool context's
  opaque `user_content` back into a typed `Content` (the same
  "opaque `Value` parsed back via its own `Deserialize` impl" pattern
  `context_cache.rs` already established), builds the examples
  instruction, and appends it via the already-real
  `LlmRequest::append_instructions`. `name`/`description` are set but
  unused (matches the source's own comment); `get_declaration`/
  `run_async` use the `BaseTool` trait defaults, matching the source
  not overriding either. 5 new tests.
- **Known limitation:** `VertexAiExampleStore` (C0830) stays
  `REQUIRED` — it needs a real Vertex AI Example Store client/
  credentials this workspace doesn't have, the same deferral class as
  `gemini_context_cache_manager.rs`'s own disclosed Vertex-AI-auth
  gap. `ExampleTool::from_config` (YAML tool-reference config, C0417)
  also stays undone, same as every other tool's `from_config`.

## PR #TBD — Phase 8: `exit_loop`, `LongRunningFunctionTool`, `get_user_choice`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::exit_loop_tool::exit_loop` (C0420, DONE) — sets
  `escalate`+`skip_summarization` on the tool context's actions to
  break a loop-type agent, matching the source verbatim.
- **Added:** `adk_tools::long_running_tool::LongRunningFunctionTool`
  (C0422, DONE) — wraps a `FunctionTool` by composition (Rust has no
  struct inheritance to subclass `FunctionTool` the way the source
  does), delegating every `BaseTool` method except `is_long_running`
  (always `true`) and `get_declaration` (appends a "don't call again
  while pending" instruction to the description).
- **Added:** `adk_tools::get_user_choice_tool::{get_user_choice,
  get_user_choice_tool}` (C0421, DONE) — `get_user_choice` ignores its
  `options` arg and always sets `skip_summarization`+returns `Null`
  (always defers to client-side resolution, matching the source);
  `get_user_choice_tool()` wraps it in the new
  `LongRunningFunctionTool`.
- **Adaptation:** since this port's `FunctionTool` needs an explicit
  `FunctionDeclaration` (no runtime reflection over Rust function
  signatures — `function_tool.rs`'s own module doc already discloses
  this), `get_user_choice_tool`'s declaration uses a hand-written
  JSON-schema `parameters` value in place of the source's
  inferred-from-signature `options: list[str]`.
- 9 new tests across the three modules.

## PR #TBD — Phase 4: Planners (`BasePlanner`, `BuiltInPlanner`, `PlanReActPlanner`)
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_flows::planners` (C0200-C0203, DONE). `BasePlanner` is
  the trait every planner implements (`build_planning_instruction`,
  `process_planning_response`). `BuiltInPlanner` wraps a model's native
  thinking features — both hooks are no-ops, and
  `apply_thinking_config` sets `LlmRequest.config.thinking_config`
  (an opaque `Value` placeholder, matching the field it writes into).
  `PlanReActPlanner` is the model-agnostic prompted Plan-Re-Act
  planner: `build_planning_instruction` returns a byte-for-byte port
  of the source's 5-tag NL instruction
  (`PLANNING`/`REPLANNING`/`REASONING`/`ACTION`/`FINAL_ANSWER`), and
  `process_planning_response` splits a tagged model response back into
  thought/answer/tool-call parts — stopping at the first (group of)
  function calls, splitting text on the final-answer tag into a
  thought-prefixed reasoning block plus a clean answer suffix, and
  stripping/marking leading-tagged text as a thought part otherwise.
- **Test parity:** all 8 tests from the source's own
  `tests/unittests/planners/test_plan_re_act_planner.py` ported 1:1,
  including the leading-parallel-function-call regression test (the
  source's own comment notes an earlier `> 0` index-guard bug that
  silently dropped every function call after the first when index 0
  itself was the first call) — this port uses `Option<usize>` instead
  of a `-1` sentinel, which structurally can't reproduce that bug.
  Plus 2 new edge-case tests (empty response list, an empty-named
  function call). 13 new tests total.
- **Known limitation:** `BasePlanner` isn't yet wired into a real
  `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor` for
  `_nl_planning` (C0176/C0179) — that needs
  `InvocationContext.agent` to resolve a concrete `LlmAgent`'s
  configured planner, the same blocker every other Phase 4 processor
  (`basic`, `identity`, `instructions`, `context_cache_processor`, ...)
  already discloses.

## PR #TBD — Phase 4: `functions.py` execution core
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_flows::functions` (C0191/C0192, partial) — the
  `functions.py` execution core. `get_tool` resolves a `FunctionCall`
  against a `ToolsDict`; `create_tool_context` builds a `ToolContext`
  carrying the function-call ID and an optional serialized
  `ToolConfirmation`; `execute_single_function_call` runs a tool via
  `BaseTool::run_async` and builds its response `Event`;
  `execute_function_calls` dispatches many calls concurrently (one
  `rusty_tokio::spawn` task per call, the same pattern already
  established by `ParallelAgent`), filters by ID, and merges results
  via the already-built `functions_utils::merge_parallel_function_response_events`.
  Adds a new `adk-flows` → `adk-tools` dependency edge (verified
  non-circular). 7 new tests.
- **Known limitations (disclosed in the module doc):** no tool-level
  before/after/on-error callback dispatch — the `PluginManager` half is
  excluded by the same crate-graph constraint already disclosed for
  `BasePlugin` (`adk-tools` already depends on `adk-agents`, so
  `adk-agents` can't depend back for `BaseTool`), and the
  `LlmAgent.canonical_*_tool_callbacks` half isn't built; no
  auth-request/tool-confirmation-request event synthesis (needs Phase
  9's `AuthConfig`); no long-running/`_defers_response` empty-response
  skip — a genuine design gap, since `BaseTool::run_async`'s
  `Result<Value, ToolError>` contract has no way to signal "no response
  yet" distinct from a real value; no multimodal-part
  extraction/computer-use image decoding/`AgentTool`
  skip-summarization special case (those types don't exist in this
  port); no `response_scheduling` forwarding (`FunctionResponse`
  doesn't model that field); no OTel tracing (Phase 12); no defensive
  args deep-copy (unneeded — args already arrive as an owned clone
  under Rust's ownership model); no explicit cancel-and-await-siblings
  on one call's failure (`execute_function_calls`'s `handle.await` loop
  propagates the first error but doesn't cancel remaining in-flight
  tasks — the same limitation `ParallelAgent`'s own module doc already
  discloses).

## PR #TBD — Phase 7 batch 4: `LoopAgent`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::loop_agent::LoopAgent` (C0337, partial) — a
  `BaseAgent`-pluggable `AgentBehavior`, structurally `SequentialAgent`
  wrapped in an outer loop: restarts from the first sub-agent up to
  `max_iterations` times (`None` runs indefinitely), stopping the
  moment a sub-agent escalates. Resets every sub-agent's tracked state
  between iterations via `ctx.reset_sub_agent_states` (already built
  for a prior batch). Resumability mirrors `SequentialAgent`'s
  (`LoopAgentState` additionally tracks `times_looped`, preserved even
  when the tracked sub-agent name no longer exists in the tree — only
  the resume index restarts at 0 in that case, matching the source).
- **Adaptation:** reuses `SequentialAgent`'s own already-disclosed
  state-delta-propagation fix verbatim — this port's `Context`/`State`
  copy rather than share state by reference, so each produced event's
  `state_delta` is applied onto `ctx.session.state` directly as the
  loop processes it; without this, a sub-agent in iteration 2 would
  never see state a sub-agent in iteration 1 set.
- **Not ported:** `_run_live_impl` matches the source exactly (raises
  `NotImplementedError` — never implemented upstream either);
  `LoopAgentConfig`/`_parse_config`/YAML config loading (C0338, needs
  the config-resolution pipeline C0348 discloses as unbuilt) —
  construct a `LoopAgent` directly with `with_max_iterations` instead.
- 7 new tests (`adk-agents`, 150 total). Full workspace gate green.
- This completes the "deprecated-but-active" multi-agent trio
  (`SequentialAgent`/`ParallelAgent`/`LoopAgent`) started in Phase 7
  batches 2-3.

## PR #TBD — Phase 7 batch 3: `ParallelAgent`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::parallel_agent::ParallelAgent` (C0336,
  partial) — a `BaseAgent`-pluggable `AgentBehavior` running its
  sub-agents with genuine concurrency, each in its own isolated branch
  (`BranchPath::create_sub_branch`, already built for a prior batch).
  Resumability mirrors `SequentialAgent`'s: an is-resumable-only start
  marker, a skip for any sub-agent already marked finished in a
  previous (paused) run, and a final end-of-agent marker.
- **Adaptation, disclosed at length in the module doc:** the source's
  `_merge_agent_run` merges sub-agent event *streams* through a queue
  with per-event backpressure, letting it cancel remaining sub-agents
  the instant one escalates. This port's `AgentBehavior` returns a
  fully-collected `Vec<Event>` per run (already disclosed in
  `base_agent.rs`), so there is no partial result to race against or
  cancel mid-flight — sub-agents still run with real concurrency (via
  `rusty_tokio::spawn`), but escalate detection happens only after
  every included sub-agent has already run to completion, and a
  sibling already mid-flight when one escalates is not cancelled. Both
  are direct, disclosed consequences of the earlier streaming-vs-`Vec`
  decision, not a new gap.
- **Adaptation, disclosed at length in the module doc:** the source's
  per-sub-agent context copy is shallow, so `agent_states`/
  `end_of_agents`/session state stay the same shared dicts across every
  branch and the parent — a sub-agent marking itself done is instantly
  visible to the parent. This port's `InvocationContext::clone()` is a
  real deep clone (the same already-disclosed departure
  `SequentialAgent` has), so nothing a sub-agent's own branch mutates
  reaches the parent automatically. Every produced event's
  `state_delta` is applied onto `ctx.session.state` post-hoc to restore
  cross-branch visibility. Full nested-resumability propagation isn't
  implemented — "all sub-agents finished" is derived from this turn
  completing without a pause rather than replaying each sub-agent's own
  nested agent-state events; correct for the common case, narrower for
  an independently-paused nested sub-agent tree.
- **Not ported:** `ParallelAgentConfig`/YAML config loading (C0338,
  needs the config-resolution pipeline C0348 discloses as unbuilt).
  `_run_live_impl` matches the source exactly (raises
  `NotImplementedError` — never implemented upstream either).
- 7 new tests (`adk-agents`, 143 total). Full workspace gate green.

## PR #TBD — Phase 7 batch 2: `SequentialAgent`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_agents::sequential_agent::SequentialAgent` (C0335,
  partial) — a `BaseAgent`-pluggable `AgentBehavior` running its
  sub-agents in tree order. Ported faithfully: start-index resumption
  (resumes from the tracked sub-agent, restarts from the beginning if
  that sub-agent was removed from the tree since — silently, no
  logging framework adopted), resumable-only agent-state marker events
  (one before each fresh sub-agent, one final end-of-agent marker),
  and pause-on-long-running-call (`should_pause_invocation`, already
  built).
- **Adaptation, disclosed at length in the module doc:** the source's
  `Context.state` backing dict *is* `EventActions.state_delta` by
  reference, so one sub-agent's state change is visible to the next
  immediately. This port's `Context`/`State` copy instead of sharing
  by reference (an already-disclosed departure) — without a fix, a
  later sub-agent in the same `SequentialAgent` run would never see an
  earlier one's state changes, breaking the entire point of a
  sequential chain. Fixed by applying each produced event's
  `state_delta` onto `ctx.session.state` directly as the loop
  processes it, mirroring (at a smaller scope) what
  `SessionService::append_event`'s state-merge already does at the
  persistence layer — a sub-agent's own `BaseAgent::run_async` clones
  `ctx.session` when it builds its working copy, so this update is
  what the *next* sub-agent's clone actually sees.
- **Not ported:** `_run_live_impl`'s `task_completed` tool/instruction
  auto-injection (needs `canonical_tools`, C0092, still a `ToolUnion`
  placeholder — `run_live_impl` here is a plain pass-through with no
  injection); `SequentialAgentConfig`/YAML config loading (C0338,
  needs the config-resolution pipeline C0348 discloses as unbuilt).
- 5 new tests (`adk-agents`, 136 total). Full workspace gate green.

## PR #TBD — Start Phase 7 batch 1: a real `BasePlugin` + `PluginManager`
**2026-08-24** · (link added once this PR is opened)

- **Scoping note:** Phase 7 (`plugins/`, the workflow/node graph engine,
  `App`, deprecated-but-active `SequentialAgent`/`ParallelAgent`/
  `LoopAgent`, `RemoteA2aAgent`, YAML agent config — 103 capability
  rows) is the largest remaining phase. The graph/node/task-delegation
  engine alone is a multi-batch undertaking; a real plugin system, by
  contrast, is self-contained, already has a real (if stubbed)
  consumer in `BaseAgent`, and was explicitly disclosed as a gap by
  both `adk-tools::FunctionTool`'s and `adk-runners::Runner`'s own
  module docs. So this batch starts there.
- **Added:** `adk_agents::services::BasePlugin` — replaces the old
  hardcoded `PluginManager` stub (whose hooks took no parameters and
  always returned `None`) with a real trait: agent-level hooks
  (`before_agent_callback`/`after_agent_callback`, C0354, done),
  run-level hooks (`on_user_message_callback`/`before_run_callback`/
  `on_event_callback`/`after_run_callback`, C0353, partial — defined
  and dispatchable, no call site wired yet), and notification-only
  hooks (`on_agent_error_callback`/`on_run_error_callback`, C0357,
  partial — only the agent-level one has a call site so far). Every
  hook defaults to a no-op, matching the source's own `pass`-bodied
  `BasePlugin` base class.
- **Added:** a real `PluginManager` (C0358-C0361) — `register_plugin`
  (rejects a duplicate name), `get_plugin`, `set_skip_closing_plugins`;
  dispatch runs registered plugins in registration order, and for the
  short-circuiting hooks the first plugin to return `Some(..)` stops
  the rest and its value propagates; the notification-only hooks
  always run every plugin regardless. `close()` closes plugins
  sequentially, matching the source's *actual* implementation (a
  deliberate choice to avoid anyio/MCP task-local-context issues), not
  its docstring's claim of running them concurrently — this port's own
  capability manifest already flagged that inconsistency as one to
  resolve, not replicate blindly.
- **Fixed a latent bug:** `BaseAgent::run_async`/`run_live` previously
  constructed a fresh, always-empty `PluginManager` locally on every
  call instead of reading the real one off the passed-in
  `InvocationContext` (`ctx.plugin_manager`) — meaning a `PluginManager`
  configured anywhere upstream (e.g. by a `Runner`) would have silently
  never run. Both methods now use `ctx.plugin_manager`, and a new
  end-to-end test (`run_async_honors_a_plugin_registered_on_the_invocation_context`)
  proves a registered plugin's `before_agent_callback` actually fires
  and can short-circuit the agent.
- **Not ported:** model-level (`before_model_callback`/
  `after_model_callback`/`on_model_error_callback`, C0355) and
  tool-level (`before_tool_callback`/`after_tool_callback`/
  `on_tool_error_callback`, C0356) hooks — typing these needs
  `LlmRequest`/`LlmResponse` (`adk-models`) and `BaseTool`/`ToolContext`
  (`adk-tools`), and `adk-tools` already depends on `adk-agents`, so
  `adk-agents` depending back on either would be the same crate-graph
  cycle `LlmRequest::append_tools` (C0116) already had to avoid. A
  unified `BasePlugin` spanning all four hook levels needs its own home
  above `adk-tools`/`adk-models`, deferred to a follow-up. Also not
  ported: per-plugin `close()` timeout and failure aggregation (no
  plugin implementation exists yet that can fail to close); wrapping a
  panicking plugin hook the way the source wraps a raised exception
  (this port's hooks return a value rather than raising, so there's no
  exception channel to intercept — the same posture `AgentCallback`
  closures already have).
- 13 new tests (`adk-agents`, 131 total). Full workspace gate green.

## PR #TBD — Start `Runner` batch 2: the `Runner` struct + legacy `run_async`
**2026-08-24** · (link added once this PR is opened)

- **Added:** a new `adk-runners` crate with `Runner` — the "legacy"
  (plain `BaseAgent`, no node/task/live/rewind/debug) execution path,
  built on top of batch 1's `SessionService`.
  - `Runner::new(app_name, agent, session_service)` — since no `App`
    type exists in this port (Phase 7), `Runner` only ever wraps one
    concrete `BaseAgent` directly; there's no app/agent/node
    mutual-exclusivity contract to enforce (C0840-C0842, N/A) since
    only one combination is representable at all. `.with_artifact_service`/
    `.with_memory_service`/`.with_credential_service`/
    `.with_plugin_close_timeout`/`.with_auto_create_session` (C0844,
    C0845) round out construction.
  - `Runner::run_async(user_id, session_id, new_message)` (C0884,
    C0886, C0888, partial/done — see below): rejects a `new_message`
    carrying a function call (C0888, done); fetches-or-creates the
    session per `auto_create_session` (C0873, partial); builds an
    `InvocationContext` wired with the Runner's own services; appends
    the user message event; drives `agent.run_async(&invocation_context)`
    — the real orchestration primitive `BaseAgent` already provides;
    persists every resulting event; returns them (the appended user
    event itself isn't returned, matching the source's own
    yielded-events contract).
  - `Runner::close()` (C0924, partial) — flushes the session service.
- **Not ported:** the workflow/node/task-delegation engine (Phase 7,
  confirmed absent) — so no node/task-mode paths, no
  `_find_agent_to_run`/resumed-conversation continuation (C0907-C0910),
  no `_resolve_invocation_id` (C0855, needs resumability, always false
  here); a real plugin system (Phase 7) — `Runner` doesn't store or
  call through a `PluginManager` at its own level at all, rather than
  invent placeholder methods for a shape (`run_before_run_callback`/
  `run_on_event_callback`/`run_after_run_callback`) that isn't real
  until Phase 7 defines it (the per-agent `before`/`after_agent_callback`
  hooks *are* exercised transitively, via `agent.run_async`); toolset
  collection/closing in `close()` (C0922/C0923 — needs `LlmAgent.tools`
  wired to real `BaseToolset` instances instead of the `ToolUnion`
  placeholder); `Runner.run()`'s sync thread-bridging wrapper
  (C0877-C0880 — a local-testing convenience with less need in an
  already async-native codebase); `InMemoryRunner` (C0926 — needs
  `InMemoryArtifactService`/`InMemoryMemoryService`, neither built);
  compaction (C0871-C0872, needs `events_compaction_config` wiring);
  agent-origin inference/warnings (C0851-C0854 — no Rust module-path
  reflection, no logging framework adopted).
- 6 new tests (`adk-runners`). Full workspace gate green.

## PR #TBD — Start `Runner` batch 1: a real `SessionService` + `InMemorySessionService`
**2026-08-24** · (link added once this PR is opened)

- **Scoping note:** `Runner` (`runners.py`, 2609 lines, C0833-C0926 — 94
  capability rows) is the core execution engine. Most of it depends on
  infrastructure this port doesn't have yet: an `App` type (Phase 7),
  the workflow/node/task-delegation engine (`BaseNode`, `Context.run_node`,
  Phase 7, confirmed absent), and a real plugin system (`PluginManager`
  is currently a hardcoded no-op stub). What Runner's own legacy
  (non-node) turn orchestration genuinely needs *first* is a working
  session backend — and `SessionService` was, until now, an empty
  marker trait (a Phase 5 forward-reference stub with zero methods).
  So this first batch builds that real prerequisite rather than a
  Runner shell that can't yet do anything.
- **Added:** `adk_agents::services::SessionService` upgraded to a real
  (if narrowed) port of `sessions.base_session_service.BaseSessionService`
  (C0206, partial) — `create_session`/`get_session`/`list_sessions`/
  `delete_session` (required), plus a concrete `append_event` default
  (drops partial events; applies-then-trims `temp:`-scoped state delta;
  updates session state) and a no-op `flush`, both ported directly from
  the source's own non-abstract defaults. Methods return a boxed future
  (`BoxFuture`) rather than using native `async fn`, since
  `InvocationContext` stores this trait as `Arc<dyn SessionService>` —
  the same object-safety pattern `adk_tools::base_tool::BaseTool`
  already established for the same reason.
- **Added:** `InMemorySessionService` (C0211, partial; C0213, done) —
  nested-map storage (`app_name -> user_id -> session_id -> Session`)
  behind a `Mutex`. Its `append_event` is a real override (not the
  shared default): since `get_session`/`create_session` hand callers
  their own clone of a session (mirroring the source's `_copy_session`),
  appending to that clone alone would never become visible to a later
  `get_session` call. The override dedups a re-delivered event against
  the *canonical stored* session's events (matched by id, then full
  equality) before applying anything, then mirrors the resulting
  state/event onto that canonical copy — and returns the event
  unstored (not raised) if the session's app/user/id isn't (or is no
  longer) in the store.
- **Adaptation, disclosed:** the source logs a warning on that
  unknown-session case; this port has no logging framework adopted yet
  (the same "no logging framework" substitution used throughout this
  migration), so it's silent instead. The deprecated `*_sync` mirror
  methods aren't ported — Rust has no sync/async method-duplication
  problem to begin with (nothing here has a legacy sync API predating
  an async one).
- **Not ported:** app:/user: state-prefix scoping and cross-session
  shared state maps (`_session_util.extract_state_delta`, `get_user_state`,
  C0209/C0214) — the placeholder `Session`/`State` have no prefix-scoping
  concept yet, a Phase 5 concern of their own; `last_update_time`-based
  `list_sessions` ordering and `StaleSessionError` (no such field exists
  on the placeholder `Session`, and the source's own in-memory backend
  never raises `StaleSessionError` either — that's a persistent-backend
  concern); `GetSessionConfig` event-trimming (`num_recent_events`/
  `after_timestamp`) — `RunConfig.get_session_config` is already an
  opaque `Value` placeholder with nothing typed to apply it against.
- 17 new tests (`adk-agents`, 124 total). Full workspace gate green.
- **Next:** the `Runner` struct itself (constructor, `close()`, the
  legacy single-agent `run_async` turn) — deferred to keep this batch
  to one clean, testable piece.

## PR #TBD — Phase 8 batch 3: `FunctionTool`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::function_tool::FunctionTool` (C0404, partial;
  C0405, done) — wraps a Rust closure as a `BaseTool`.
  - **Fundamental adaptation, disclosed at length in the module doc:**
    the source wraps an arbitrary Python callable and uses runtime
    reflection (`inspect.signature`, `get_type_hints`) to detect a
    context parameter, build a cached `FunctionDeclaration` from the
    signature, coerce raw JSON args into Pydantic model types, and
    compute mandatory parameters. None of this exists in Rust —
    functions have no runtime-inspectable signatures. So
    `FunctionTool::new` takes an already-built `FunctionDeclaration`
    and an explicit `required_args` list; the wrapped closure always
    takes `(&BTreeMap<String, Value>, &mut ToolContext)` (no context-
    parameter detection needed, since the signature is fixed rather
    than discovered); there's no generic argument-coercion layer (the
    closure body converts its own args via `rusty_serde::json::from_value`
    as needed); there's no sync/async runner distinction (every
    closure here is already async by construction).
  - The `require_confirmation` gate (`RequireConfirmation::Bool` or
    `::Predicate`) is fully ported: missing-mandatory-arg pre-check
    returns a `{"error": ...}` value (not a Rust error); a required-
    but-unanswered confirmation calls `request_confirmation`, sets
    `actions.skip_summarization`, and returns the "please
    approve/reject" error; an answered-but-rejected confirmation
    returns "this tool call is rejected"; a confirmed one runs the
    closure.
  - **Adds** `tool_confirmation`/`set_tool_confirmation` to
    `adk-agents::context::Context` — needed for the gate above. Kept
    as an opaque `Value`, not the real `ToolConfirmation` type, since
    `adk-tools` (which owns `ToolConfirmation`) depends on
    `adk-agents`, not the reverse; `FunctionTool` narrows it via
    `ToolConfirmation`'s own (de)serialization.
- **Not ported:** `input_stream` injection (live/bidirectional-
  streaming tools — needs `active_streaming_tools` wiring `adk-agents`
  doesn't consume yet); `_detect_error_in_response` (telemetry, Phase
  12).
- 6 new tests (`adk-tools`, 30 total), 0 new tests needed in
  `adk-agents` beyond the existing `Context` coverage (the new field
  is exercised transitively by `function_tool`'s own tests). Full
  workspace gate green.

## PR #TBD — Phase 8 batch 2: `BaseToolset`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_tools::base_toolset::BaseToolset` (C0403, partial) —
  the base trait for a tool collection.
  - `tool_filter`/`tool_name_prefix` are trait methods, not fields
    (same `BaseTool` precedent). `ToolFilter` (an enum of
    `Predicate(Arc<dyn Fn(&dyn BaseTool, Option<&ReadonlyContext>) ->
    bool>)` / `Names(Vec<String>)`) stands in for the source's
    `Union[ToolPredicate, List[str]]` — Rust has no runtime
    `isinstance` dispatch to distinguish the two at call time the way
    the source does.
  - `get_tools_with_prefix` (per-invocation-cached, prefixes tool +
    declaration names) is a default trait method built over an
    explicit `PrefixCache` behind a `Mutex` that each implementor owns
    and exposes via a required `prefix_cache()` method — the source's
    `_cached_invocation_id`/`_cached_prefixed_tools` are plain mutable
    instance state on a normal Python object, which a Rust `&self`
    trait method has no equivalent for without an owned, explicit cache
    field.
  - A `PrefixedTool` wrapper (delegates to an inner `Arc<dyn BaseTool>`,
    overrides `name()`/`get_declaration()`) replaces the source's
    `copy.copy(tool)` plus closure-rewrite of `_get_declaration` — there's
    no way to "shallow-copy and monkey-patch a method" on a Rust trait
    object.
  - `get_auth_config` returns the pre-existing
    `adk_agents::services::AuthConfig` opaque `Value` placeholder.
- **Not ported:** `from_config` (needs `ToolArgsConfig`, C0417, the
  same gap `BaseTool` (C0402) already discloses); the source's
  `@final` on `get_tools_with_prefix` (no Rust equivalent — an
  implementor could in principle override the default method, which
  the source disallows).
- 9 new tests (24 total in `adk-tools`). Full workspace gate green.

## PR #TBD — Phase 8 batch 1: `adk-tools` — `BaseTool`, `ToolContext`, `ToolConfirmation`, `append_tools`
**2026-08-24** · (link added once this PR is opened)

- **Added:** a new `adk-tools` crate, the start of Phase 8
  (`google.adk.tools`). Placed parallel to `adk-flows` (same dependency
  level: `adk-agents` + `adk-genai` + `adk-models`), not nested inside
  it or merged into `adk-models` — a literal `LlmRequest.append_tools`
  *method* would need `adk-models` to depend on `adk-tools` for
  `BaseTool` while `adk-tools` needs `LlmRequest` from `adk-models`, a
  crate-graph cycle. `append_tools` is a free function instead, the
  same "processor as a free function, not a method" pattern `adk-flows`
  already uses throughout.
  - `BaseTool` (C0402, partial): the trait every tool implements.
    Every source instance attribute (`name`/`description`/
    `is_long_running`/`custom_metadata`/`response_scheduling`) becomes
    a trait method, not a field — matching
    `adk_models::base_llm::BaseLlm`'s established precedent. Not
    ported: `from_config` (needs `ToolArgsConfig`, C0417, not built)
    and the `SelfTool` generic-return pattern (no Rust equivalent for a
    trait method returning `Self` through a trait object).
  - `ToolContext` (C0415, partial): `pub type ToolContext =
    adk_agents::context::Context` — the source's alias is the whole
    capability, since this port's `Context` (Phase 2) already covers
    it. Not ported: the lazy `AuthCredential`/`AuthHandler`/
    `AuthConfig` back-compat re-exports (Phase 9, not built).
  - `ToolConfirmation`/`from_response_dict` (C0416, done): parses
    either a direct dict or the ADK client's wrapped
    `{'response': '<json string>'}` format.
  - `LlmRequest.append_tools`/`merge_declarations` (C0116, done):
    merges each tool's `FunctionDeclaration` into the one
    `functionDeclarations`-carrying entry in `config.tools`, without
    disturbing unrelated built-in-tool marker entries (Google Search,
    etc). `config.tools` stays the pre-existing opaque `Value`
    placeholder rather than a typed `Vec<Tool>` — narrowing it is a
    natural follow-up once more of Phase 8 exists to justify it.
- **Added:** `FunctionDeclaration` to `adk-genai` — real `name`/
  `description` (needed for `append_tools`'s dedup-by-name logic),
  opaque `parameters`/`parameters_json_schema`/`response`/
  `response_json_schema` (ADK's own code only forwards these, never
  inspects their shape).
- **Adaptation, disclosed:** a default trait method can't coerce
  `&Self` into `&dyn BaseTool` (`E0277`, `Self` not `Sized` — the only
  fix the compiler suggests, `where Self: Sized`, would make the
  method uncallable through a trait object at all, which dynamic
  dispatch here requires). Resolved by giving `append_tools` an
  object-free core, `merge_declarations`, that takes plain
  `(String, FunctionDeclaration)` pairs; `BaseTool`'s default
  `process_llm_request` calls it directly on `self.name()`/
  `self.get_declaration()` (always legal on `&Self`) instead of ever
  constructing a trait object.
- **Adaptation, disclosed:** the source only *logs* a warning on a
  duplicate tool name within one `append_tools` call (last wins); no
  logging framework is adopted in this port yet
  (`functions_utils.rs`/`contents.rs` disclose the same substitution),
  so `append_tools`/`merge_declarations` return the shadowed names
  instead — both declarations are still advertised to the model
  either way. The source's `tools_dict` (name → `BaseTool` map,
  excluded from serialization) isn't tracked — nothing in this port
  yet consumes it (function-call dispatch, C0191, needs `BaseTool`
  resolution this batch doesn't wire).
- 15 new tests (`adk-tools`), 2 new tests (`adk-genai`'s
  `FunctionDeclaration`). Full workspace gate green.

## PR #TBD — Phase 4 batch 15: `_get_agent_to_run` — transfer target resolution
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_flows::agent_transfer::get_agent_to_run` (C0159, DONE) —
  resolves a `transfer_to_agent` target by name, searching the whole
  agent tree from the root down (`root_agent()`/`find_agent()`, both
  already real). Raises when the named agent isn't found anywhere in
  the tree, and when the transferring agent has
  `disallow_transfer_to_peers` set and the target is a sibling (shares
  the same parent, isn't itself).
- **Adaptation, disclosed:** the source compares `parent_agent` by
  Pydantic model equality (effectively object identity); this port's
  `BaseAgent` wraps a type-erased `Box<dyn AgentBehavior>` and so can't
  derive `PartialEq`, so parent identity is compared by name instead —
  `BaseAgent::build` already treats sibling name uniqueness as expected
  within one tree, so this is a reasonable proxy.
- 7 new tests. Full workspace gate green (172 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 14: `LlmFlow` — a real, working `AgentBehavior` for `LlmAgent`
**2026-08-24** · (link added once this PR is opened)

- **Added:** `adk_flows::llm_flow::LlmFlow` — the first concrete
  `AgentBehavior` this port builds for an `LlmAgent`. Wired as a
  `BaseAgent`'s behavior, it drives a real (if narrowed) non-live turn
  end-to-end through `BaseAgent::run_async`:
  - `preprocess` (C0150, partial): assembles the `LlmRequest` by running
    `basic` → `identity` → `instructions` → contents (full history via
    `get_contents`, or current-turn-only via `get_current_turn_contents`
    per `LlmAgent.include_contents`) → `context_cache`, in the source's
    own load-bearing order.
  - `call_model` (C0153, partial): calls the resolved `BaseLlm`. `LlmFlow`
    resolves its model once, at construction (`LlmFlow::new`), rather
    than re-resolving through `canonical_model` on every call — a real
    (if narrow) instance of the memoization cache `canonical_model.rs`'s
    own module doc disclosed as missing. `LlmFlow::with_model` injects
    any `Arc<dyn BaseLlm>` (a test double included) without touching the
    process-wide model registry.
  - `postprocess` (C0156, partial) + `finalize_model_response_event`
    (**C0157, now fully DONE**): converts each `LlmResponse` into a
    finalized `Event` — every field with a real `Event` counterpart is
    shallow-copied only when non-`None`, matching the source exactly.
    Adaptation: `finish_reason` narrows `LlmResponse`'s opaque `Value` to
    `Event`'s `String` via `Value::as_str`; `cache_metadata` widens
    `LlmResponse`'s real `CacheMetadata` back to `Event`'s opaque `Value`
    via `to_value`, since `Event` holding the concrete type directly
    would need an `adk-events`→`adk-models` crate-graph cycle (the same
    constraint `event_compaction.rs`/`context_cache.rs` already disclose).
- **Verified end-to-end**, not just unit-tested in isolation: a test
  builds a real `BaseAgent` wired with `LlmFlow`, calls
  `BaseAgent::run_async` against a fake `BaseLlm`, and asserts on the
  resulting event — no stubs on the `AgentBehavior` seam itself.
- **Scope, disclosed:** no multi-step tool-call loop (C0148, C0149,
  C0151, C0152, C0158, C0159 — without `BaseTool`, Phase 8, a model is
  never handed any tools, so it has nothing to call and the source's
  "call model, run tools, call model again" loop has nothing to loop
  on yet); no `interactions_processor` wiring or
  `preserve_function_call_ids` policy (both need to detect whether the
  resolved model is a Gemini-with-Interactions-API/Anthropic/LiteLLM
  backend — undetectable through the type-erased `Arc<dyn BaseLlm>` this
  port resolves to, no downcasting mechanism); no telemetry spans or
  before/after/on-error model callback dispatch (C0154, C0155 —
  `LlmAgent.before_model_callback`/`after_model_callback` exist as real
  fields, but dispatching them needs a `Context`-based short-circuit
  path like `BaseAgent`'s existing agent-level callbacks); no live mode
  (C0161-C0167 — `run_live_impl` returns a clear
  `LlmFlowError::LiveNotImplemented` rather than pretending).
- Also needed a small error-system bridge: `rusty_err::Error` (this
  port's own sovereign error trait) and `std::error::Error` are
  deliberately separate traits, and `BaseAgent`'s `AgentRunError` is a
  `Box<dyn std::error::Error + Send + Sync>` — a tiny `BoxedLlmFlowError`
  wrapper (carrying just the rendered message) bridges an `LlmFlowError`
  across that boundary, mirroring the shape `BaseAgent`'s own tests
  already use for a boxed error.
- 6 new tests. Full workspace gate green (165 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 13: `functions.py` merge + client-id helpers
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::functions_utils` (C0196, partial) — a slice of
  `functions.py`'s helpers:
  - `merge_parallel_function_response_events`: concatenates parallel
    tool-call response events' content parts and deep-merges their
    `EventActions`. The merge round-trips each event's `actions` through
    `Value` (`rusty_serde::json::to_value`/`from_value`) and generically
    deep-merges the resulting maps — mirroring the source's own
    `model_dump(exclude_none=True)` + `deep_merge_dicts` +
    `model_validate` approach exactly, rather than hand-writing bespoke
    per-field merge rules that could silently diverge from it. A
    `None`/absent field from a later event never overwrites an earlier
    value (matching `exclude_none=True`); `render_ui_widgets` is popped
    out before the generic merge and aggregated additively across every
    event instead of last-wins, then reattached — verified by dedicated
    tests for both behaviors.
  - Client function-call-id lifecycle helpers:
    `generate_client_function_call_id`, `populate_client_function_call_id`,
    `remove_client_function_call_id`, `get_long_running_function_calls`
    (takes an `is_long_running` callback rather than a
    `tools_dict: dict[str, BaseTool]`), `find_event_by_function_call_id`,
    `find_matching_function_call`.
- **Scope, disclosed:** `build_auth_request_event`/`generate_auth_event`/
  `generate_request_confirmation_event` (the "auth/confirmation request
  events" half of C0196) are not ported — they need `AuthConfig`
  (Phase 9), which doesn't exist in this port yet.
- 13 new tests. Full workspace gate green (159 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 12: `request_confirmation` dedup pre-pass
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::request_confirmation` (C0172, partial) — the
  `request_confirmation` request processor's pure, tool-infrastructure-
  free dedup pre-pass:
  - `get_original_function_call_args`: extracts the `originalFunctionCall`
    payload out of an `adk_request_confirmation` call's args, `None` if
    absent or not itself a map (malformed).
  - `map_confirmation_to_original_fc_ids`: maps each confirmation
    function-call id back to the original function-call id it confirms,
    scanning every function call in session events — the cheap,
    validation-free pre-pass so already-consumed confirmations can be
    dropped *before* the expensive strict re-validation the source
    performs downstream.
- **Scope, disclosed:** parsing a `ToolConfirmation` out of a
  confirmation response, resolving/validating the confirmed tool against
  session history (`_resolve_confirmation_targets`), and re-executing it
  (`functions.handle_function_call_list_async`) are **not** ported — all
  need `BaseTool`/`ToolConfirmation`/`ToolContext` (Phase 8/9), which
  don't exist in this port yet, the same blocker `agent_transfer.rs`/
  `output_schema.rs` already disclose for their own Phase 8 gaps.
- 6 new tests. Full workspace gate green (146 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 11: `agent_transfer` — transfer targets + instruction text
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::agent_transfer` (C0171, partial) — the
  `agent_transfer` request processor's transfer-target computation and
  instruction-text generation:
  - `get_transfer_targets`: sub-agents (excluding single-turn/task-mode
    ones), then — if the parent is itself LLM-orchestrated — the parent
    and its other sub-agents (peers, excluding the current agent),
    gated by the corresponding `disallow_transfer_to_parent`/
    `disallow_transfer_to_peers` flags.
  - `build_transfer_instruction_body`/`build_transfer_instructions`: the
    exact instruction text telling the model about its transfer targets,
    plus an appended parent-transfer suggestion when applicable. 2 tests
    assert byte-for-byte containment of the source's own literal
    `expected_content` strings, copied straight from
    `test_agent_transfer_system_instructions.py`.
- **Adaptation, disclosed:** takes an `llm_mode: &dyn Fn(&BaseAgent) ->
  Option<AgentMode>` callback rather than reading `mode`/
  `disallow_transfer_to_*` straight off any `BaseAgent` (as the source
  does via `getattr`/`hasattr`, since `LlmAgent` extends `BaseAgent`
  there) — this port's `BaseAgent` (Phase 2) and `LlmAgent` (Phase 2/4)
  are two separate, unfused types; `None` signals a `BaseAgent` with no
  corresponding `LlmAgent` config (e.g. a workflow node).
- **Scope, disclosed:** `_get_incompatible_builtin_tool_error` (needs
  `GoogleSearchTool`/`VertexAiSearchTool`/`EnterpriseWebSearchTool`) and
  actually building/attaching a `TransferToAgentTool` are not ported —
  both need `BaseTool` (Phase 8), the same blocker `output_schema.rs`
  already discloses for `SetModelResponseTool`.
- 11 new tests. Full workspace gate green (140 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 10: `_output_schema_processor` gating + read-back helpers
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::output_schema` (C0178, partial) — the
  `_output_schema_processor` request processor's gating decision and
  read-back helpers:
  - `should_apply_output_schema_processor`: whether the processor would
    inject a `set_model_response` tool for this request — `output_schema`
    set, `tools` non-empty, the resolved model's capabilities can't honor
    output schema and tools together, and the agent isn't in task mode.
  - `OUTPUT_SCHEMA_TOOL_INSTRUCTION`: the instruction text appended
    alongside the injected tool, verbatim from the source.
  - `create_final_model_response_event`/`get_structured_model_response`:
    builds a plain-text model-response event from a validated
    `set_model_response` result, and reads one back out of a function-
    response event's `actions.set_model_response`.
- **Scope, disclosed:** actually injecting a `SetModelResponseTool` into
  the request (`llm_request.append_tools`) is **not** ported — both the
  tool itself and `LlmRequest::append_tools` (C0116) need `BaseTool`
  (Phase 8), which doesn't exist in this port yet; `append_tools`'s own
  module doc in `adk-models` already discloses this same blocker.
- 8 new tests. Full workspace gate green (129 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 9: `interactions_processor` + `context_cache_processor`
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::interactions` (C0174, partial) — the
  `interactions_processor` request processor's core logic:
  `is_event_in_branch` (whether an event belongs to a branch, or the
  root when unbranched) and `find_previous_interaction_state` (scans
  session events in reverse for the current agent's most recent
  `interaction_id`/`environment_id`, skipping events outside the current
  branch) — used to chain stateful Gemini Interactions API conversations
  across turns.
- **Added:** `adk_flows::context_cache` (C0175, partial) — the
  `context_cache_processor` request processor's core logic:
  `find_cache_info_from_events` scans session history backward for the
  agent's most recent cache metadata (incrementing `invocations_used`
  when the cache is active and carried over from a prior invocation) and
  prompt token count; `apply_context_cache` assembles both into
  `LlmRequest`'s existing real `cache_config`/`cache_metadata`/
  `cacheable_contents_token_count` fields.
- **Adaptation, disclosed:** `Event.cache_metadata` (an opaque `Value`
  placeholder) is parsed back into a real
  `adk_models::cache_metadata::CacheMetadata` via its own `Deserialize`
  impl (`rusty_serde::json::from_value`) rather than `Event` holding a
  typed field directly — `adk-events` sits below `adk-models` in the
  crate graph, so depending on it directly would cycle (the same
  constraint `adk-flows`'s own top-level module doc already discloses
  for `canonical_model`). `usage_metadata`'s `promptTokenCount` key is
  read directly since no typed `UsageMetadata` exists in this port yet.
- **Scope, disclosed:** both modules are free-function core logic only —
  not yet wired as real `BaseLlmRequestProcessor`s reading through
  `InvocationContext`, the same "needs `LlmAgent` wired into
  `BaseAgent`'s tree" scope note every other Phase 4 processor
  (`basic`/`identity`/`instructions`) has already disclosed.
- 18 new tests (10 in `interactions.rs`, 8 in `context_cache.rs`). Full
  workspace gate green (121 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 8: `_get_contents`/`_get_current_turn_contents` orchestration
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::contents::get_contents`/`get_current_turn_contents`
  — the top-level orchestration `_get_contents`/`_get_current_turn_contents`
  from `contents.py`, completing C0181-C0183/C0188/C0189's own top-level
  wiring and fully finishing C0190:
  - `get_contents`: applies (in order) rewind filtering via
    `adk_events::rewind::apply_rewinds`, branch/isolation-scope/event-kind
    visibility filtering, compaction resolution via `crate::compaction`,
    transcription-fragment coalescing (C0188, new
    `coalesce_transcription_event`), cross-agent message fencing via
    `crate::fencing`, orphaned-response dropping, both function-call/
    response rearrangement passes, function-call-id stripping
    (`copy_content_for_request`), and — for scoped (task/single-turn)
    agents — prepending the originating delegation input as a synthetic
    leading user turn (new `build_task_input_user_content`).
  - `get_current_turn_contents`: the `include_contents='none'` mode —
    finds the latest event that starts the current turn (a real user
    turn or another agent's reply, never a direct `transfer_to_agent`
    hop) and delegates to `get_contents` from there.
- **Adaptation, disclosed:** `copy_content_for_request` does a full Rust
  clone rather than the source's shallow-copy-for-mutation-safety
  optimization — a strictly safer superset (no nested-field-sharing
  hazard for downstream mutators to worry about) that this port doesn't
  need the performance trade for yet.
- **Scope, disclosed:** the `_ContentLlmRequestProcessor` itself remains
  deferred — it decides *when* to call `get_contents` vs
  `get_current_turn_contents` (`agent.include_contents`), computes
  `preserve_function_call_ids` from the agent's canonical model type
  (Anthropic/LiteLLM/OpenAIResponsesLlm/Interactions-API Gemini — none of
  which exist in this port yet), and wires in
  `_add_model_input_context_to_user_content`/
  `_add_instructions_to_user_content`. All of this needs `LlmAgent` wired
  into `BaseAgent`'s tree and a real `InvocationContext.agent` — the same
  blocker every other Phase 4 processor (`basic`, `identity`,
  `instructions`) has already disclosed.
- Minor cleanup: `rearrange_events_for_latest_function_response`'s
  slightly awkward `let _ = matching;` binding (flagged during the prior
  batch's review) is now a plain `.any(...)` check.
- 14 new tests. Full workspace gate green (103 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 7: `_content_compaction.py`, compaction-aware history reconstruction
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::compaction`, porting `flows/llm_flows/_content_compaction.py`
  (C0185) in full:
  - `process_compaction_events`: resolves overlapping compaction
    summaries (a wider or later-indexed summary subsumes a narrower or
    earlier one it fully covers), materializes each surviving summary as
    a synthetic event at its compaction end timestamp — attributed to
    the given agent name (falling back to `"model"`) so the agent reads
    its own compacted history as its own prior turns — and filters raw
    events whose timestamp falls inside any kept compaction range.
  - `recover_compacted_function_calls`: re-injects a compacted
    function-call event verbatim (preserving parallel-call thought
    signatures, which only the first part carries) ahead of a surviving
    function-response that would otherwise be orphaned — the clearest
    case being a long-running tool call compacted along with its
    placeholder response, with the real result arriving after resume.
    Recovers compacted sibling responses too, so a parallel sibling
    doesn't surface as a phantom pending call.
- **Changed:** `EventCompaction.compacted_content` (C0027, `adk-events`)
  narrowed from a placeholder JSON `Value` to a real
  `adk_genai::content::Content` — `process_compaction_events` needs a
  genuine `Content` to build the synthetic summary event's own `content`
  field, and `adk-events` already depends on `adk-genai` for `Event.content`
  itself, so this closes a gap that was only left open pending Phase 3
  (which has since landed).
- **Adaptation, disclosed:** the source's defensive `is None` checks on
  `EventCompaction.start_timestamp`/`end_timestamp`/`compacted_content`
  are omitted — the Rust struct's fields are already non-optional
  (matching the source's own pydantic model, which declares them
  required), so the type system already guarantees they're present
  whenever `actions.compaction` is `Some`.
- 9 new tests. Full workspace gate green (90 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 6: `_fencing.py`, prompt-injection fencing for cross-agent transcripts
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::fencing`, porting `flows/llm_flows/_fencing.py`
  (C0184) in full:
  - `quote_untrusted`/`elide_quote_markers`: wraps relayed text between
    literal begin/end markers and states, in the message itself, that the
    fenced content is data to read and never instructions to follow.
    Markers already present in the payload are elided first, so quoted
    content can't forge the end of its own block and keep speaking as the
    framework.
  - `is_other_agent_reply`: whether an event is a reply from an agent
    other than the current one, including the live/bidi-mode carve-out
    where any non-user author (even the current agent, post-transfer) is
    treated as "other".
  - `present_other_agent_message`: reformats another agent's event as
    `role='user'` context — `[agent_name] said:`/`thought:` for text,
    `[agent_name] called tool ... with parameters:` /
    `` [agent_name] `tool` tool returned result: `` for function
    calls/responses (each fenced via `quote_untrusted`), and blob parts
    (`inline_data`/`file_data`/`executable_code`/`code_execution_result`)
    relayed unfenced on their own part type — fencing means flattening
    into the text channel, which blobs can't do at all and which would
    drop code/output pairing.
- **Adaptation, disclosed:** `function_call.args`/
  `function_response.response` (both `BTreeMap<String, Value>`) are
  rendered as compact JSON rather than reproducing Python's `str(dict)`
  repr — the same lower-fidelity stand-in
  `instructions_utils::value_to_display_string` already disclosed for
  dict/list values. `None` (an absent map) still renders as the literal
  string `"None"`, matching Python's `str(None)` exactly.
- **Scope, disclosed:** this module is fully self-contained and needed no
  new subsystems (`Event`/`Content`/`Part` were already real types). What
  still needs building is the *caller* — `contents.py`'s `_get_contents`
  orchestration decides *when* an event needs this applied, and that
  orchestration remains its own deferred future batch (see the
  batch-5 entry below).
- 14 new tests. Full workspace gate green (81 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 5: `contents.py`'s standalone event/content transforms
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::contents` — the standalone, non-orchestrating
  helpers `flows/llm_flows/contents.py`'s `_get_contents` pipeline is
  built from:
  - `is_part_invisible`/`contains_empty_content` (C0189, DONE): a part is
    never invisible if it carries a function call/response, a
    thought_signature, or a server-side tool call/result; an event is
    empty only if every part is invisible and it carries no
    transcription/compaction action.
  - `is_event_belongs_to_branch`, `is_direct_transfer`, `is_auth_event`,
    `is_request_confirmation_event`, `is_adk_framework_event`,
    `should_include_event_in_context` (C0183, partial): branch-membership
    and event-kind visibility predicates.
  - `copy_content_for_request` (C0181, partial): strips synthetic
    `adk-*`-prefixed function-call/response ids when requested — the
    mechanism a future backend-specific policy will call into.
  - `drop_orphaned_function_responses`, `merge_function_response_events`,
    `rearrange_events_for_latest_function_response`,
    `rearrange_events_for_async_function_responses_in_history`
    (C0186/C0187, DONE): orphan-response dropping and both
    function-call/response rearrangement passes. `bisect_left` translated
    as `Vec::partition_point`.
- **Added:** `Part.tool_call`/`tool_response: Option<Value>` in
  `adk-genai` — opaque placeholders for a server-side (model-run) tool
  call/result, distinct from `function_call`/`function_response`, needed
  for `is_part_invisible`'s "never invisible" exception.
- **Adaptation, disclosed:** `drop_orphaned_function_responses` returns
  the dropped ids alongside the filtered events instead of logging a
  warning internally — no logging framework has been adopted by this
  workspace yet, the same class of omission `debug_log.rs` already
  disclosed.
- **Scope, disclosed:** this batch deliberately excludes
  `_get_contents`/`_get_current_turn_contents` themselves (the ~185-line
  orchestrating functions, plus the `_ContentLlmRequestProcessor` that
  wraps them), cross-agent transcript fencing (C0184, `_fencing.py` —
  prompt-injection-relevant, deserves its own focused batch), and
  compaction-aware history reconstruction (C0185, needs `EventCompaction`
  semantics this batch doesn't build). C0181's actual FC-id-preservation
  *policy* stays deferred to Phase 10 (the backends it applies to don't
  exist in this port yet).
- 24 new tests. Full workspace gate green.

## PR #TBD — Phase 4 batch 4: the `instructions` request processor
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::instructions::build_instructions` — the
  `instructions` request processor: appends the deprecated
  `global_instruction` (with state injection), the `static_instruction`
  (stable prefix, via `Instructions::Content`), and the dynamic
  `instruction` — as the system instruction when no static instruction
  exists, or as a dynamic user-content turn when one does — matching the
  source's branching exactly.
- **Added:** `adk_flows::instructions_utils::inject_session_state` — the
  regex-based `{state_var}`/`{state_var?}`/`{artifact.name}` template
  engine every agent instruction string is rendered through. Handles
  `app:`/`user:`/`temp:`-prefixed and bare state variable names, optional
  (`?`-suffixed) references, artifact lookups through the invocation's
  `ArtifactService`, and Python-matching `None`/`True`/`False` value
  rendering.
- `ReadonlyContext` gained an `artifact_service()` accessor (needed by
  `inject_session_state`'s artifact-lookup branch); `Instruction::is_set`
  became `pub` (needed by `build_instructions` to decide whether
  `global_instruction`/`instruction` has anything to append).
- **New dependency, disclosed:** `regex` in `adk-flows` — already a
  vetted workspace dependency (adopted in `adk-models` for model-name
  pattern matching), reused here for the template-variable pattern
  (`\{+[^{}]*\}+`) rather than hand-rolling an equivalent scanner.
- **Scope decision, disclosed:** only the source's default regex-based
  template engine is ported — Jinja2 mode is an explicit opt-in
  (`use_jinja2=True`) nothing in this port's own processors ever
  requests, and needs an optional Python package with no Rust
  equivalent decision made yet. `global_instruction` reads the given
  agent's own field rather than walking to the invocation's root agent
  (no `BaseAgent` tree yet — the same deferral `canonical_model` already
  disclosed for ancestor-chain fallback). `static_instruction` only
  interprets a plain string or an already-`Content`-shaped value, not the
  source's full `ContentUnion` transformer (which needs the whole
  `google-genai` SDK type system, out of scope per Phase 3's own module
  docs).
- **Adaptation, disclosed:** `str(value)`'s exact Python formatting isn't
  reproduced for floats/dicts/lists (JSON formatting stands in instead);
  it is exact for strings, `None`/`True`/`False`, and integers — the
  common case for instruction template variables.
- 17 new tests (7 in `instructions.rs`, 10 in `instructions_utils.rs`).
  Full workspace gate green (43 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 3: the `identity` request processor
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::identity::apply_identity`/`identity_instruction`
  — the `identity` request processor: appends "You are an agent. Your
  internal name is \"...\"." (plus a description sentence when the agent
  has one) unless the agent is in single-turn mode, matching the source's
  `mode != 'single_turn'` gate exactly.
- **Scope/adaptation, disclosed** (same shape as the `basic` processor,
  PR before this one): a free function, not yet a real
  `BaseLlmRequestProcessor`. Takes `agent_name`/`agent_description` as
  explicit parameters rather than reading them off `LlmAgent` — the
  source inherits these from `BaseAgent`, but this port's standalone
  `LlmAgent` struct has neither field yet (it isn't wired into
  `BaseAgent`'s tree). Once real tree placement lands, a caller passes
  `agent.name()`/`agent.description()` straight through — this function's
  own logic doesn't change.
- 7 new tests. Full workspace gate green (26 passing in `adk-flows`).

## PR #TBD — Phase 4 batch 2: the `basic` request processor
**2026-08-23** · (link added once this PR is opened)

- **Added:** `adk_flows::basic::build_basic_request` — the full behavior
  of the source's `basic` request processor (`_build_basic_request`):
  resolves the agent's canonical model onto `LlmRequest.model`, rebuilds
  `LlmRequest.config` from the agent's validated `generate_content_config`
  (deserialized fresh rather than copied field-by-field — see the module
  doc for why that's equivalent here), merges any pre-existing
  `http_options` headers back in (RunConfig's headers win on conflict),
  merges `RunConfig.labels` into `config.labels`, gates `output_schema`
  on the model's `output_schema_and_tools` capability and task mode, and
  populates the full `live_connect_config` surface from `RunConfig`
  (including the Gemini-3.x-live-specific suppression of
  `enable_affective_dialog`/`proactivity`).
- `LlmRequest.config` gained a `labels` field; `LlmRequest.live_connect_config`
  gained `response_modalities`/`output_audio_transcription`/
  `input_audio_transcription`/`realtime_input_config`/`explicit_vad_signal`/
  `translation_config`/`enable_affective_dialog`/`proactivity`/
  `history_config`/`context_window_compression`/`avatar_config` — each
  sourced straight from `RunConfig`'s own already-opaque same-named field.
- **Scope decision, disclosed:** `build_basic_request` is a free function
  taking `&LlmAgent`/`&RunConfig` directly, not yet a real
  `BaseLlmRequestProcessor` reading through `InvocationContext`. The
  source's `as_llm_agent` narrows `invocation_context.agent: BaseAgent`
  down to a concrete `LlmAgent` via a Python `cast()` (a runtime no-op);
  this port's `InvocationContext.agent: Option<BaseAgent>` has no
  equivalent — `LlmAgent` doesn't implement `AgentBehavior` yet (flagged
  in `llm_agent.rs`'s own module doc as blocked on exactly the Phase 4
  work this crate is starting). Wiring `build_basic_request` into a real
  trait impl is deferred to whichever future batch gives `LlmAgent` real
  tree placement — the behavior itself is fully ported and tested now, so
  that future wiring is only plumbing, not new logic.
- **Adaptation, disclosed:** `_merge_run_config_http_options`'s
  `timeout`/`retry_options`/`extra_body` fields aren't modeled in
  `HttpOptionsStub` yet — only the `headers` merge is ported.
  `RunConfig.session_resumption` is deserialized best-effort into
  `SessionResumptionStub`; a shape mismatch just doesn't get copied
  forward (RunConfig's opaque per-field values were never validated
  against this port's narrower stub types the way `generate_content_config`
  was, so failing loudly here would be the wrong default).
- 10 new tests. Full workspace gate green (19 passing in `adk-flows`).

## PR #TBD — Start Phase 4: adk-flows, processor interfaces, canonical_model
**2026-08-23** · (link added once this PR is opened)

- **Added:** new `adk-flows` crate (Phase 4, `google.adk.flows`) —
  everything downstream of Phase 3's Gemini backend that needs `LlmAgent`
  (`adk-agents`) and `BaseLlm`/`LLMRegistry` (`adk-models`) together.
- **Added, C0147:** `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor` —
  the processor interfaces `BaseLlmFlow`'s whole request/response pipeline
  is built from (`basic`, `identity`, `instructions`, `nl_planning`, etc.
  will each implement one of these in follow-up batches). `run_async`
  returns a boxed future resolving to `Result<Vec<Event>, ProcessorError>`
  rather than an `AsyncGenerator` — the same adaptation
  `BaseLlm::generate_content_async` already made in Phase 3.
- **Added, C0080/C0090 (partial):** `canonical_model`/`canonical_live_model`
  — resolves `LlmAgent.model` to a real `BaseLlm` via a new process-wide
  default registry, retroactively completing what `llm_agent.rs`'s own
  module doc flagged as blocked on `LLMRegistry` back when Phase 3 hadn't
  built any real backend yet.
- **Added, C0111 (partial):** `adk_models::registry::default_registry` — a
  real process-wide `LlmRegistry`, pre-populated with `Gemini` and
  `OllamaLlm` (the two concrete backends this migration has actually
  built). Resolving a Claude/OpenAI/Apigee/OCIGenAI/LiteLLM-provider model
  name still falls through to the existing named "install this pip extra"
  error — no backend exists for any of those yet.
- **Architectural decision, disclosed:** `canonical_model`/`canonical_live_model`
  are free functions in `adk-flows`, not methods on `LlmAgent` itself
  (unlike the source, where they're properties). `adk-models` already
  depends on `adk-agents` (for `ContextCacheConfig`, used by
  `LlmRequest.cache_config`); having `adk-agents` depend back on
  `adk-models` for these two methods would make the two crates depend on
  each other, which Cargo doesn't allow. `adk-flows` sits above both
  instead. Fixing this for real (so these can become genuine `LlmAgent`
  methods) means extracting `ContextCacheConfig` into a shared lower
  crate both `adk-agents` and `adk-models` can depend on — a deliberate
  restructuring, left for when/if it's actually needed, not done as a
  side effect of this batch.
- **Scope decision, disclosed:** three real gaps stay `REQUIRED` in the
  manifest rather than flipping C0080/C0090 to `DONE`: ancestor-agent-chain
  fallback (needs `LlmAgent` wired into `BaseAgent`'s tree — not done yet,
  itself blocked on Phase 3/4/8 per `llm_agent.rs`'s own module doc), the
  source's `_resolved_model` memoization/invalidation-on-reassignment
  cache (this port re-resolves via the registry on every call), and
  `ModelRef::Instance` (a live `BaseLlm` instance passed directly rather
  than a model name) — blocked on the same crate-dependency restructuring
  named above.
- 12 new tests (9 in `adk-flows`, 3 in `adk-models::registry`). Full
  workspace gate green (212 passing in `adk-models`, 9 in the new
  `adk-flows`).

## PR #TBD — SSE streaming for Gemini (Phase 3 batch 11, closing C0125 and C0126)
**2026-08-23** · (link added once this PR is opened)

- **Added:** `Gemini::generate_content_stream` — a real
  `POST .../models/{model}:streamGenerateContent?alt=sse` call, its
  response body parsed into Server-Sent-Events (`parse_sse_events`) and
  fed through `crate::streaming_utils::StreamingResponseAggregator` to
  produce a `Vec<LlmResponse>`: one per-chunk partial (marked
  `partial: true`), a flushed merged-text event whenever buffered text
  needs to give way to a non-text/terminal chunk, and one final
  non-partial aggregated response. Wired into
  `BaseLlm::generate_content_async`'s `stream: true` branch, which
  previously always returned an error — closes C0125.
- **Added:** `crates/adk-models/src/streaming_utils.rs` —
  `StreamingResponseAggregator`, ported from
  `google.adk.utils.streaming_utils`. Tracks buffered text/thought text,
  the last-reported usage/grounding/citation metadata, and the finish
  reason across chunks; `close()` produces the final aggregated response,
  surfacing a non-STOP finish reason (or prompt feedback, when there's no
  candidate at all) as an error, matching the source's precedence exactly.
- **Refactored:** extracted `Gemini::prepare_call` — the setup shared by
  both the non-streaming and streaming real calls (`maybe_append_user_content`,
  the model-name check, auth resolution, C0126's cache-manager invocation,
  `apply_tracking_headers`) — out of `generate_content`, so
  `generate_content_stream` doesn't duplicate it.
- Closes the streaming half of **C0126**: cache metadata is populated only
  into the final aggregated streaming response, never the partials,
  matching the source's own `if cache_metadata and cache_manager is not
  None: cache_manager.populate_cache_metadata_in_response(close_result,
  cache_metadata)` — which only runs after the aggregator's `close()`.
- **Scope decision, disclosed:** the source's `StreamingResponseAggregator`
  has two aggregation modes switched on a feature flag
  (`FeatureName.PROGRESSIVE_SSE_STREAMING`): the legacy text-only mode
  this PR ports, and a newer mode that preserves part ordering and streams
  function-call arguments incrementally via JSONPath-addressed partial
  args. This workspace has adopted no feature-flag registry (Phase 12's
  `features/` isn't built) and no typed function-call/tool machinery to
  stream partial arguments into (`config.tools`/`FunctionDeclaration` stay
  opaque, C0116, Phase 8's `BaseTool`) — the progressive mode has nothing
  to be built on top of yet, so only the legacy mode is ported.
- **Adaptation, disclosed:** the source's `process_response` is an
  `AsyncGenerator`; the ported `process_response` returns a plain
  `Vec<LlmResponse>` instead, and the whole SSE response body is read up
  front rather than incrementally (via `reqwest::blocking`'s `.text()`)
  — `BaseLlm::generate_content_async`'s own contract already collects a
  whole call's responses into one `Vec` before returning anything to the
  caller, so there's no incremental consumer either change could lose
  fidelity against.
- **Fixed, disclosed (caught while porting, not by a failing test):** the
  source's per-chunk text check (`if ... parts[0].text:`) relies on
  Python's string truthiness — an empty string is falsy. The Rust port's
  first draft checked `Option::is_some()` instead, which would have
  treated a chunk with `text: ""` as "has real text to accumulate" rather
  than falling through to the flush/passthrough branch. Fixed before
  landing, with a dedicated regression test
  (`an_empty_string_text_part_is_treated_as_no_text_matching_pythons_truthiness`).
- 8 new tests in `gemini.rs` (SSE parsing, end-to-end streaming through
  both `generate_content_stream` and the `BaseLlm` trait method, cache
  metadata placement) plus 11 tests in `streaming_utils.rs` (text/thought
  accumulation, flush timing including the inline-data and empty-string
  edge cases, usage-metadata persistence, error/finish-reason precedence
  in `close()`). Full workspace gate green (209 passing in `adk-models`).

## PR #TBD — Wire GeminiContextCacheManager into generate_content (Phase 3 batch 10, non-streaming half of C0126)
**2026-08-23** · (link added once this PR is opened)

- **Added:** `Gemini::generate_content` now invokes
  `GeminiContextCacheManager` in the same place the source's
  `generate_content_async` does — right after the model-name check, before
  `apply_tracking_headers` merges in tracking headers — gated on
  `llm_request.cache_config.is_some() && !self.use_interactions_api`,
  matching the source's own gate exactly. The resulting cache metadata
  (fingerprint-only or an active cache) is populated into the returned
  `LlmResponse` via `GeminiContextCacheManager::populate_cache_metadata_in_response`.
- Added `GeminiCallError::ContextCache(String)`, flattening a
  `GeminiContextCacheError` into `generate_content`'s own error type — the
  same treatment every other real call in this module gets.
- **Scope decision, disclosed:** this closes only the non-streaming half
  of C0126. The source also populates cache metadata into every response
  a streaming call yields (`StreamingResponseAggregator`'s partials plus
  its final aggregate) — there's no streaming path in this port yet to
  populate (the SSE-streaming half of C0125 is still deferred), so C0126
  stays `REQUIRED` in the capability manifest rather than flipping to
  `DONE`, the same "stays REQUIRED until every half lands" treatment
  already applied to C0131.
- No OTel span wraps the call, matching this workspace's not having
  adopted a tracing framework yet (same class of omission as C0134's
  logging-framework caveat).
- 3 new tests: fingerprint-only cache metadata populated when
  `cache_config` is set (against a local mock `generateContent` server),
  cache handling skipped when `use_interactions_api` is `true`, and an
  explicit assertion that `cache_metadata` stays `None` without a
  `cache_config`. The cache-manager's own state-machine behavior (reuse/
  invalidate/fingerprint-match/mismatch) is already exhaustively covered
  by `gemini_context_cache_manager.rs`'s own 29 tests — this batch only
  needed to verify the wiring itself. Full workspace gate green (194
  passing in `adk-models`).

## PR #TBD — Redacted debug logging (Phase 3 batch 9, closing C0134)
**2026-08-23** · (link added once this PR is opened)

- **Added:** `crates/adk-models/src/debug_log.rs` — `build_request_log`/
  `build_response_log`, ported from `google_llm.py`'s
  `_build_request_log`/`_build_response_log`. Builds a structured
  debug-level string for a request (system instruction, config,
  contents, function declarations) or response (extracted text, function
  calls, raw JSON dump), with credential-bearing fields redacted before
  they can reach a log: `inline_data`/`file_data`'s binary payload
  (`mime_type` and everything else is kept), and `http_options.headers`.
  Wired into `Gemini::generate_content`, gated by a new
  `ADK_DEBUG_LOGGING` env var (this port's lightweight stand-in for the
  source's `logger.isEnabledFor(logging.DEBUG)` — no logging framework
  decision has been made for this workspace yet, same caveat already
  disclosed in `gemini_llm_connection.rs`'s module doc).
- **Adaptation, disclosed:** `config.tools` isn't modeled as typed
  `FunctionDeclaration`s yet (C0116, Phase 8's `BaseTool`), so this port
  can't walk into a tool's function declarations the way the source
  does. The "Functions" log section is always empty, and the config
  log's `tools` key is always fully redacted (never partially shown) —
  the same effect the source's own fallback-exclusion branch produces
  whenever it can't locate a function-declaration-bearing tool, just
  taken unconditionally here instead of only as a fallback. Of the
  source's full `http_options` credential-bearing exclusion set
  (`httpx_client`/`httpx_async_client`/`aiohttp_client`/`headers`/
  `extra_body`/`client_args`/`async_client_args`), only `headers` exists
  to redact in this port's narrower `HttpOptionsStub` — it's still
  dropped from the log entirely, never partially shown.
- `GenerateContentResponse` and its nested `Candidate`/`PromptFeedback`
  types gained a `Serialize` impl (`skip_serializing_if` on every
  `Option`, matching `exclude_none=True`) — previously deserialize-only,
  since nothing needed to re-serialize a parsed response before the
  response log needed to dump one back into raw JSON.
- 10 new tests covering both redaction paths, the empty-response case,
  function-call extraction, and the env-var gate. Full workspace gate
  green (192 passing in `adk-models`).

## PR #TBD — Gemini REST-path tracking headers (Phase 3 batch 8, closing C0133)
**2026-08-23** · (link added once this PR is opened)

- **Added:** `Gemini::apply_tracking_headers` — the REST-call counterpart
  to `prepare_live_connect_config`'s tracking-header merge, closing out
  C0133 (the Live-path half landed in Phase 3 batch 4; `get_tracking_headers`/
  `merge_tracking_headers` themselves landed in batch 2). Unconditionally
  creates `llm_request.config.http_options` if it's absent — a genuine,
  previously-undocumented divergence from the Live path's own gating
  (`prepare_live_connect_config` only touches an *already-present*
  `http_options`, matching the source's own two different gates at its two
  call sites) — merges ADK's tracking headers into it (overriding any
  pre-set value for the same header, same de-dup-by-token behavior as the
  Live path), and forwards the resolved `api_version` when one is
  configured.
- Wired into `Gemini::generate_content`: called right after
  `maybe_append_user_content`, and the resolved tracking headers are
  attached explicitly to the outgoing HTTP request (in addition to
  whatever the client's own default headers are).
- **Adaptation, disclosed:** the source's SDK fully replaces the outgoing
  request's headers server-side before sending; `reqwest` has no
  "override this default header" primitive exposed to callers, and this
  port can't introspect an already-built injected `reqwest::blocking::Client`'s
  default headers to merge against them the way Python reads back
  `http_options.headers or {}`. So tracking headers are attached via a
  second explicit `.header()` call — the ADK tracking value is always
  present on the wire either way, but when the client also sets a default
  header of the same name, the outgoing request may carry two header
  lines rather than one merged value. The important behavioral case (an
  injected client is still guaranteed to carry ADK's tracking header) is
  preserved and tested.
- `LlmRequest.config` (`GenerateContentConfigStub`) gains a new
  `http_options: Option<HttpOptionsStub>` field, reusing the existing stub
  type from `live_connect_config`.
- 4 new tests, including one against a local mock HTTP server asserting
  the tracking header actually reaches the wire for an injected client.
  Full workspace gate green.

## PR #TBD — GeminiContextCacheManager (Phase 3 batch 7, C0140-C0143)
**2026-08-23** · (link added once this PR is opened)

- **Added:** `crates/adk-models/src/gemini_context_cache_manager.rs` — a
  full port of `GeminiContextCacheManager`, the explicit context-cache
  lifecycle manager for Gemini models. Covers the `handle_context_caching`
  state machine (reuse a valid active cache → clean up and recreate an
  invalid one → fall back to a fingerprint-only record when recreation
  isn't warranted or fails), model-specific minimum cache-token floors
  (`gemini-2.5-*` → 2048, `gemini-3*` → 4096), SHA-256 cache-validity
  fingerprinting over model/scope/system-instruction/tools/first-N-contents,
  gated cache creation (previous-response token count vs. both
  `cache_config.min_tokens` and the model's own floor, scaled by an
  estimated cacheable-prefix share), best-effort cache cleanup, and
  request truncation to the uncached suffix.
- **Scope decision:** wiring this manager into
  `Gemini::generate_content_async` (C0126 — the source's
  `if llm_request.cache_config and not self.use_interactions_api` block)
  stays deferred, already noted in `gemini.rs`'s module doc. This PR
  builds and tests the manager standalone against a local mock
  `cachedContents` HTTP endpoint — the same "build it standalone, wire it
  in later" split used for `GeminiLlmConnection` versus `Gemini::connect()`
  two batches ago.
- **Adaptation, disclosed:** the source is constructed with one
  all-in-one `google.genai.Client` that already carries its own auth;
  this port's `GeminiContextCacheManager::new` instead takes the same
  `GeminiApiClient` `Gemini::api_client()` builds, plus the resolved
  backend variant and a pre-resolved auth header — equivalent, since the
  source itself constructs a fresh manager per call
  (`GeminiContextCacheManager(self.api_client)`).
- **Adaptation, disclosed:** the cache fingerprint never crosses the
  Rust/Python boundary (it's only ever compared against a fingerprint
  this same code produced earlier), so it doesn't need to match the
  source's `json.dumps(sort_keys=True)` output byte-for-byte — only be
  internally deterministic, which a fixed-field-order `Value` JSON dump
  is. `tools`/`tool_config` stay opaque `Value` placeholders (not modeled
  yet, C0116), so unlike the source's own reordering-tolerant
  canonicalization, a reordered-but-equivalent tools list will (safely)
  miss the cache rather than (unsafely) hit a stale one.
- **New dependency, disclosed:** `sha2` (RustCrypto) — checked every
  sibling Rusty-Mill repo for a hashing/crypto candidate first, none
  exists; pure Rust, no system OpenSSL, same rationale as `regex`. Also
  newly wired in: `rusty_time` (already a workspace dependency, unused
  until now) to parse the Gemini API's RFC 3339 `expireTime` response
  field into a Unix timestamp.
- **Adaptation, disclosed:** `_cache_scope`'s `project`/`location` keys
  (Vertex-AI-only) never populate — `GeminiApiClient` doesn't model a
  Vertex AI project/location, matching `gemini.rs`'s existing
  `GeminiCallError::VertexAiAuthNotSupported` deferral. `LlmRequest.config`
  (`GenerateContentConfigStub`) gained two new opaque fields this batch
  needed: `tool_config: Option<Value>` and `cached_content: Option<String>`.
- **Testing caveat:** no real Gemini `cachedContents` endpoint was
  exercised — cache-creation/cleanup tests run against a local hand-rolled
  HTTP mock server (the same dependency-free pattern used throughout this
  migration's Gemini REST/Live tests), so the request/response wire shape
  is a best-effort reconstruction of the documented public API, not
  verified against the real service the way `gemini_demo`'s
  `generateContent` call was in the previous PR.
- 29 new tests; full workspace gate (`cargo build/test/clippy/fmt`) green.

## PR #TBD — Runnable demo examples for Gemini and Ollama
**2026-08-23** · (link added once this PR is opened)

- **Added, at the user's request:** `examples/gemini_demo.rs` and
  `examples/ollama_demo.rs` in `adk-models` — runnable, end-to-end
  demonstrations of each backend. `cargo run -p adk-models --example
  gemini_demo` (or `ollama_demo`) builds a real `LlmRequest`, sends it,
  and prints the model's reply.
- Each demo tries a real server first — `gemini_demo` uses a real
  `GOOGLE_API_KEY`/`GEMINI_API_KEY` from the environment if set;
  `ollama_demo` probes `OLLAMA_HOST`/`localhost:11434` for a real
  Ollama instance — and falls back to a local one-shot mock server
  speaking the same wire shape when neither is available, so both run
  with zero setup.
- **Verified for real in this session:** `gemini_demo` against the
  actual Gemini API (an available key made the real call; older model
  names returned real 404s pointing at their replacements, and
  `gemini-3.5-flash-lite` returned a genuine model response) — the
  first live, non-mocked confirmation that the Phase 3 REST
  implementation (batches 2-3) actually works against the real
  service. Both demos also verified against their mock fallback (no
  Ollama server was reachable in this sandbox).

## PR #TBD — OllamaLlm: a real, testable local-Ollama backend
**2026-08-23** · (link added once this PR is opened)

- **Added, at the user's request:** `OllamaLlm` — a `BaseLlm` backend that
  talks directly to a local Ollama server's native `/api/chat` HTTP
  endpoint. Requested explicitly as a parallel track alongside the
  ongoing Gemini backend work (not a replacement for it): Ollama's API
  is well-documented, stable, and — unlike Gemini's Live API — actually
  runnable and verifiable against a real local server.
- **Scope, disclosed:** this is *not* a port of the source's
  `LiteLlm(BaseLlm)` class (`models/lite_llm.py`, Phase 10, C0557 — a
  universal wrapper around the third-party `litellm` Python package,
  itself covering 14 providers with substantial provider-specific
  behavior, C0557-C0574, ~18 manifest rows). Porting that starts with a
  Rust equivalent of `litellm` that doesn't exist and is a far larger
  undertaking than "connect to Ollama." What's built here instead:
  model registration for the `ollama/…`/`ollama_chat/…` prefixes (2 of
  the 14 provider regexes from C0560), the `ollama_chat`
  content-flattening quirk from C0567 (`_flatten_ollama_content` —
  Ollama's chat endpoint rejects multi-part `content` when it's
  text-only), and non-streaming request/response translation over a
  real HTTP call. Tool-calling (needs `BaseTool`, deferred the same way
  as `LlmRequest.append_tools`/C0116), streaming, and every non-Ollama
  provider stay out of scope. No manifest rows are marked `DONE` — the
  C0540-C0587 LiteLLM cluster describes substantially more than this
  covers, and this addition isn't itself a capability discovered in the
  source inventory, so it doesn't claim a manifest row of its own.
- **Adaptation, disclosed:** the source's own registration regex is
  `ollama/(?!gemma3).*` (a negative lookahead excluding `Gemma3Ollama`'s
  models). Rust's `regex` crate has no lookaround support at all, so
  that pattern can't be expressed as a registry regex. The exclusion is
  enforced in code instead (`parse_provider` rejects `ollama/gemma3*` at
  call time) — functionally equivalent here, since no competing
  `Gemma3Ollama`-equivalent entry exists to conflict with.
- **Testing note:** no Ollama server was reachable in the sandbox this
  was built in (`curl http://localhost:11434` refused), so the 13 new
  tests run against a local, dependency-free HTTP test server speaking
  Ollama's documented `/api/chat` response shape — the same pattern used
  for the Gemini REST/WS transports. Point `OllamaLlm` at a real local
  Ollama instance (default `http://localhost:11434`, or `OLLAMA_HOST`)
  to exercise it against the real thing.

## PR #TBD — Phase 3 batch 6: GeminiLlmConnection.receive()
**2026-08-23** · (link added once this PR is opened)

- **Added:** `GeminiLlmConnection::receive()` (C0139) — the Live
  event→`LlmResponse` translation engine, ported branch-for-branch from
  the source's ~370-line `receive()`: usage-metadata remap (Live API's
  `responseTokenCount` → `GenerateContentResponseUsageMetadata`'s
  `candidatesTokenCount`, etc.), cross-message grounding-metadata
  accumulation (append-unique string-list fields, chunk extension,
  index-offset-shifted support remapping), streamed text/thought
  aggregation (tracked by part *index* rather than the source's `id()`
  object identity — equivalent here, since parts in one list never alias
  or reorder), transcription streaming persisted across `receive()`
  calls (matching the source's own instance-field buffers), Gemini-3.x-
  variant-dependent tool-call buffering (immediate yield vs. buffered
  until `turn_complete`), and session-resumption/voice-activity/GoAway
  passthrough. Split into [`process_message`] — a pure function taking
  one parsed message plus mutable aggregation state, fully unit-tested
  without any socket — and a thin `receive()` loop that reads real
  frames through it. 13 new tests, including two end-to-end reads over
  a real local WebSocket connection.
- **New wire-shape module:** `live_server_message.rs` — `LiveServerMessage`
  and its nested shapes (`LiveUsageMetadata`/`ServerContent`/
  `Transcription`/`ToolCall`), continuing the same "minimal real subset,
  best-effort protocol reconstruction" discipline as the send-side
  envelopes from batch 5.
- **Adaptation, deliberate:** `grounding_metadata` stays an opaque
  `Value` end to end, and its merge logic operates generically over
  `Value::Map` entries (append-unique any list-of-strings field,
  special-cased `grounding_chunks`/`grounding_supports`, overwrite
  otherwise) rather than naming every `GroundingMetadata` sub-field —
  actually *more* faithful here than a fully-typed struct would be,
  since the source's own `_merge_grounding_metadata` is itself generic
  over whatever keys a response happens to contain.
- **Omitted, disclosed:** the source's one warning-only branch (logging
  when `retrieval_queries` is present but `grounding_chunks` is empty,
  with no effect on any yielded response) is dropped rather than
  plumbed through a logging framework this workspace hasn't adopted.
- **Confidence caveat carried forward:** same as batch 5's send-side
  envelopes — `LiveServerMessage`'s shape is a best-effort
  reconstruction of Google's public Multimodal Live API, not something
  `google/adk-python` itself specifies, and remains unverified against a
  live endpoint.
- 1 row marked `DONE` with test-name evidence (C0139). Phase 3 is now
  26 of 43 rows complete; the remaining rows split into the actual Live
  WebSocket handshake (the rest of C0131), computer-use/preprocess
  adaptation (C0132), redacted debug logging (C0133/C0134),
  `GeminiContextCacheManager` (C0140-C0143), SSE streaming/context-cache/
  interactions-API integration (C0125/C0126/C0128), and the LiteLLM/
  lazy-provider-registration cluster (C0109/C0111-C0113).

## PR #TBD — Phase 3 batch 5: GeminiLlmConnection send-side methods
**2026-08-22** · (link added once this PR is opened)

- **Added:** `GeminiLlmConnection` — `BaseLlmConnection`'s Gemini Live
  implementation. `send_history` (filters audio parts from replayed
  history, sends a Gemini-3.x placeholder nudge so a trailing user turn
  actually triggers a response), `send_content`/`send_content_partial`
  (function responses route via a tool-response envelope; a single
  non-partial text part on Gemini 3.x Live routes via realtime input
  instead of client content), `send_realtime` (type-dispatches
  Blob/ActivityStart/ActivityEnd/audio-stream-end, with Gemini-3.x/
  3.5-Live-Translate routing audio/image blobs through dedicated
  realtime-input fields) — C0135-C0138. 18 new tests, all against a
  dependency-free local WebSocket test server (`tungstenite`'s own
  server-side `accept()`, symmetric with the client-side test in
  `live_connection.rs`).
- **New dependency, disclosed:** `tungstenite` (not `tokio-tungstenite`)
  as the Gemini Live API's WebSocket transport. Checked every sibling
  Rusty-Mill repo (none exists). Adopted the plain synchronous crate,
  not the tokio-wrapped one, for the identical reason batch 3 adopted
  `reqwest::blocking` over `reqwest`'s async client: it needs no real
  `tokio` reactor, so it works safely under `rusty_tokio` (this
  workspace's from-scratch, independent runtime) via
  `rusty_tokio::spawn_blocking`. `rustls-tls-webpki-roots` keeps TLS
  pure-Rust, matching the REST transport's TLS choice.
- **Fixed, disclosed:** discovered while testing the new Live message
  envelopes that neither they nor batch 3's REST request body
  (`generate_content_request.rs`) used `skip_serializing_if` — every
  absent optional field was serialized as an explicit `null` rather than
  omitted. Fixed in both places; a test now locks in the REST body's
  fix, and the Live envelope test explicitly asserts no `null` appears.
- **Adaptation, disclosed:** `Part.inline_data`/`file_data` — opaque
  `Value` placeholders since Phase 3 batch 1 — are narrowed to
  `MediaBlobStub` (`mime_type` modeled; the rest of the payload
  flattened via `#[rusty_serde(flatten)]` so nothing is lost on
  round-trip) because `send_history` genuinely needs
  `utils/content_utils.py`'s `is_audio_part`/`filter_audio_parts`, which
  branch on `mime_type`. `BaseLlmConnection::send_realtime`'s `blob`
  parameter is retroactively upgraded the same way, from an opaque
  `Value` to a real `RealtimeInput` enum (the source's
  `Union[Blob, ActivityStart, ActivityEnd, LiveClientRealtimeInput]`) —
  only `LiveClientRealtimeInput.audio_stream_end` is modeled, matching
  the source's own admittedly-incomplete handling of that variant
  ("Unary LiveClientRealtimeInput not fully supported yet").
- **Adaptation, disclosed (confidence caveat):** the Gemini Live API's
  exact WebSocket message envelopes (`BidiGenerateContentClientContent`/
  `RealtimeInput`/`ToolResponse`) are Google's public Multimodal Live API
  wire protocol — not part of `google/adk-python`'s own source, which
  only ever talks to the opaque third-party `google.genai.live.AsyncSession`.
  This batch's envelope shapes are a best-effort reconstruction of that
  public protocol, built with the same "minimal real subset" discipline
  as the rest of Phase 3, but — unlike the REST `generateContent` body,
  a simpler and extremely well-known shape — unverified against a live
  Gemini Live endpoint. Flagged in `gemini_llm_connection.rs`'s module
  doc rather than presented as certain.
- **Scope decision, sized deliberately:** `receive()` (C0139) — a
  ~370-line stateful message-translation engine (grounding-metadata
  accumulation with index-offset merging, streamed text/thought
  aggregation tracked by part identity, transcription streaming,
  Gemini-3.x-variant-dependent tool-call buffering, session-resumption/
  voice-activity/GoAway passthrough) — and the actual WebSocket handshake
  to Google's Live endpoint (the rest of C0131) are deferred to their own
  batch(es), for the same reason `GeminiContextCacheManager` got its own
  batch: it's too large and too distinct to fold in here.
- 4 rows marked `DONE` with per-row test-name evidence
  (C0135-C0138); C0131 (the handshake), C0132, and C0139 stay `REQUIRED`.

## PR #TBD — Phase 3 batch 4: Gemini Live connect() config-prep
**2026-08-22** · (link added once this PR is opened)

- **Added:** `Gemini::live_api_version` and `Gemini::prepare_live_connect_config`
  — the config-prep half of C0131 (`Gemini.connect()`): merges tracking
  headers into `live_connect_config.http_options` only when it's already
  present (matches the source's own gating exactly — it's never created
  there either), forwards `speech_config`/`tools`/`thinking_config`/
  `safety_settings`, unconditionally sets `system_instruction` (even when
  empty, matching a documented source behavior), and rejects transparent
  session resumption on the Gemini Developer API backend (Vertex AI
  only). 18 new tests.
- **Adaptation:** `LlmRequest.live_connect_config` — previously an opaque
  `Value` placeholder ("nothing here reads it") — is now a real, narrowed
  `LiveConnectConfigStub` (`http_options`/`speech_config`/
  `system_instruction`/`session_resumption`/`tools`/`thinking_config`/
  `safety_settings`), since this batch's config-prep logic needed real
  fields to mutate. `GenerateContentConfigStub` similarly grows
  `tools`/`thinking_config`/`safety_settings` (opaque placeholders,
  forwarded but not inspected).
- **Scope decision, sized deliberately:** this batch stops at
  config-prep — pure in-memory mutation, testable without a live network
  call, mirroring batch 2's `api_client` construction. The actual Live
  WebSocket handshake and all of `GeminiLlmConnection`
  (`send_history`/`send_content`/`send_realtime`/`receive()`) are
  deferred to their own batch: `receive()` alone is a ~370-line stateful
  message-translation engine (grounding-metadata accumulation with
  index-offset merging, streamed text/thought aggregation tracked by
  part identity, transcription streaming, Gemini-3.x-variant-dependent
  tool-call buffering, session-resumption/voice-activity/GoAway
  passthrough) — it deserves the same dedicated-batch treatment
  `GeminiContextCacheManager` already got, not being folded into this
  one. The WebSocket transport itself is also still undecided;
  `tungstenite` (the synchronous core `tokio-tungstenite` wraps) is the
  leading candidate for the same reason `reqwest::blocking` fit the REST
  transport — runtime-agnostic, so it doesn't hit the same reactor
  conflict with `rusty_tokio` batch 3 discovered — but that decision is
  made when the connection itself is built.
- No manifest rows moved to `DONE` this batch (C0131 as a whole still
  needs the real connection to close out) — documented here instead,
  per the same convention used for C0125's still-open streaming half.

## PR #TBD — Phase 3 batch 3: Gemini generate_content_async (non-streaming)
**2026-08-22** · (link added once this PR is opened)

- **Added:** `LlmResponse.create()` (C0121) — maps a raw
  `GenerateContentResponse` into an `LlmResponse` (normal/error/
  block-reason/empty-no-candidates branches), backed by a new
  `generate_content_response` module modeling the Gemini REST API's
  response wire shape (minimal real subset, same treatment as
  `adk_genai::content`). Added `generate_content_request` (the REST
  request-body wire shape) and `Gemini::generate_content` — a real,
  non-streaming HTTP POST to the Gemini Developer API, plus
  `_ResourceExhaustedError`'s 429-with-mitigation-link enhancement
  (C0127). 25 new tests, including an end-to-end round-trip against a
  dependency-free local test HTTP server.
- **Adaptation, disclosed:** auth resolves one of two ways — an injected
  `client` is trusted as-is (assumed pre-authed, e.g. for Vertex AI), or
  an API key is read from `GOOGLE_API_KEY`/`GEMINI_API_KEY` (Gemini
  Developer API backend only). Building real Vertex AI credentials
  (Application Default Credentials) is a distinct, large dependency
  decision, deferred — attempting it without an injected client returns
  a named `VertexAiAuthNotSupported` error rather than failing silently
  or attempting a doomed call.
- **Fixed, disclosed:** `adk_genai::content` now serializes multi-word
  fields as camelCase (`functionCall`, `inlineData`, `willContinue`,
  etc.) instead of bare snake_case field names — a latent gap from Phase
  3 batch 1 this batch needed fixed to build a wire-accurate request
  body, but which also corrects ADK's own event/session JSON to match
  the source's `alias_generator=to_camel` convention. No prior test
  pinned the old (incorrect) casing.
- **New dependency decision, disclosed:** discovered that `reqwest`'s
  async client requires a genuine `tokio` reactor, which `rusty_tokio`
  (a from-scratch, independent runtime adopted in Phase 2) doesn't
  provide — the two can't share a reactor. Switched `GeminiApiClient` to
  `reqwest::blocking::Client` (self-contained, safe under any ambient
  executor) driven via `rusty_tokio::spawn_blocking`, so a slow HTTP call
  offloads to a real blocking thread rather than stalling an async
  worker. Documented in the workspace `Cargo.toml` and `gemini.rs`.
- **Scope decision:** SSE-streaming `generate_content_async`
  (`stream=true`), context-cache integration, interactions-API
  delegation, the Live `connect()`/`GeminiLlmConnection`, redacted debug
  logging, and `GeminiContextCacheManager` remain deferred — see
  `crates/adk-models/src/gemini.rs`'s module doc.
- 2 rows marked `DONE` with per-row test-name evidence (C0121, C0127);
  C0125 stays `REQUIRED` since it bundles the still-undone SSE-streaming
  half. Phase 3 sits at 21 of 43 rows complete.

## PR #TBD — Phase 3 batch 2: Gemini client construction & config layer
**2026-08-22** · (link added once this PR is opened)

- **Added:** the `Gemini` struct (`model`/`client`/`client_kwargs`/
  `base_url`/`api_version`/`speech_config`/`use_interactions_api`/
  `retry_options`), `supported_models()`, `api_client`/`GeminiApiClient`
  construction (tracking headers, `enterprise` flag, base-URL/API-version
  resolution), and `_configured_api_version`/
  `_normalize_base_url_and_api_version` (C0123, C0124, C0129, C0130). Also
  added `google_client_headers` (`get_client_labels`/`get_tracking_headers`/
  `merge_tracking_headers`, C0133 partial). 25 new tests.
- **New dependency, disclosed:** the `reqwest` crate (`rustls-tls`
  feature, no native/OpenSSL TLS) as the HTTP/SSE transport for the Gemini
  REST API. This is the sovereignty-sensitive dependency decision the
  Phase 3 batch 1 `regex` note explicitly contrasted itself with — every
  sibling Rusty-Mill repo under the platform directory
  (`rusty_time`/`rusty_err`/`rustils_async`/`rusty_tokio`/`rusty_json`/
  `rusty_std`/`rusty_uuid`/`rusty_serde`) was checked for an HTTP/TLS
  candidate the manifest's own sibling-check methodology names
  (`rusty_http`/`rusty_request`, `rusty_tls`) — none exists. Only the
  REST/SSE transport is decided here; the Live API's WebSocket transport
  is a separate decision for the batch that implements `connect()`.
- **Adaptation, disclosed:** `client_kwargs` (the source's free-form dict
  merged into the `genai.Client` constructor, capable of overriding any
  constructor argument) has no well-typed Rust equivalent without knowing
  which keys matter, so it stays an inert opaque placeholder — documented
  in `gemini.rs`'s module doc, same treatment as `tools_dict` in
  `llm_request.rs`.
- **Scope decision:** this batch covers only pure configuration logic
  testable without a live network call. Deferred to further batches:
  `generate_content_async`'s actual REST/SSE wire calls, context-cache
  integration, and interactions-API delegation (C0125/C0126/C0128 — need
  real `GenerateContentConfig`/`GenerateContentResponse`/`Tool`/
  `FunctionDeclaration` wire types beyond today's load-bearing-subset
  `LlmRequest`/`LlmResponse`); `_ResourceExhaustedError` (C0127 — wraps an
  HTTP error that only exists once real calls exist); the Live
  `connect()`/`GeminiLlmConnection`/computer-use preprocessing
  (C0131/C0132/C0135-C0139 — need a WebSocket transport decision);
  redacted request/response debug logging (C0134 — needs the real wire
  types above); and `GeminiContextCacheManager` (C0140-C0143 — needs a
  SHA-256 crate decision plus the cache-creation HTTP call). See
  `crates/adk-models/src/gemini.rs`'s module doc for the full breakdown.
- 4 rows marked `DONE` with per-row test-name evidence in
  `capability-manifest.md`; Phase 3 sits at 19 of 43 rows complete.

## PR #TBD — Phase 3 batch 1: model layer core (BaseLlm/LlmRequest/LlmResponse/LLMRegistry)
**2026-08-21** · (link added once this PR is opened)

- **Added:** `adk-genai` crate — a minimal, *real* (not opaque) Rust subset
  of `google.genai.types`: `Content`/`Part`/`FunctionCall`/`FunctionResponse`,
  covering exactly the fields `google/adk-python`'s own code touches
  (confirmed by grepping every `part.*`/`content.*` access across the
  source tree). This isn't an ADK capability of its own — these types
  belong to the third-party `google-genai` package — but `LlmRequest`/
  `LlmResponse` (this batch's own capabilities) can't have real behavior
  without it.
- **Added:** `adk-models` crate — `BaseLlm` (trait, capabilities self-report
  with deprecated name-based fallback, `supported_models`, default
  `NotImplementedError`-equivalent `generate_content_async`/`connect`),
  `BaseLlmConnection` (live-connection trait), `LLMRegistry` (regex-based
  model-name dispatch, `prefix:model` override syntax, helpful
  unresolvable-model errors), `LlmRequest` (`append_instructions`,
  `insert_transient_user_content`, `set_output_schema`), `LlmResponse`
  (`get_function_calls`/`get_function_responses`), `CacheMetadata`
  (two-state active/fingerprint-only model). 37 new tests.
- **Retroactive completions, now that real `Content`/`Part` exist:**
  `Event::is_final_response`/`has_trailing_code_execution_result` (Phase 1,
  C0022/C0023 — left partial there, now finished), `InvocationContext`'s
  `should_pause_invocation`/`find_matching_function_call`/
  `stamp_event_branch_context` (Phase 2, completing C0071), and
  `LlmAgent`'s `_get_subagent_to_resume`/`__maybe_save_output_to_state`/
  `__maybe_accumulate_streaming_output` (Phase 2, C0094/C0095). 20 total
  rows across three phases moved to `DONE` by this batch alone.
- **New dependency:** the `regex` crate, for `LLMRegistry`'s model-name
  pattern matching — no sibling `rusty_regex` exists in the platform
  directory, and this isn't a sovereignty-sensitive surface the way an
  HTTP client or crypto library would be.
- **Scope decision:** the native Gemini backend (`Gemini`,
  `GeminiLlmConnection`, `GeminiContextCacheManager` — C0123-C0143, 21
  rows) is deferred to a follow-up batch. It needs a real HTTP/WebSocket
  client to Google's API — a dependency decision on the same scale as
  Phase 2's async-runtime choice — and deserves its own sibling-repo check
  before hand-rolling, per the migration's own step 3.
- **Known gaps, flagged not hidden:** `LlmResponse.create()` (needs the raw
  Gemini SDK response type), `LlmRequest.append_tools` (needs `BaseTool`,
  Phase 8), the LiteLLM provider-list fallback and real lazy provider
  registrations (`Gemini`/`Claude`/`LiteLlm`/etc. — no backends exist yet
  to register) all stay `REQUIRED`.
- 20 rows marked `DONE` with per-row test-name evidence in
  `capability-manifest.md` (15 in Phase 3, plus the 5 retroactive
  completions above); Phase 3 sits at 15 of 43 rows complete, with the
  remaining 28 concentrated in the deferred Gemini-backend cluster.

## PR #TBD — Phase 2 batch 2: LlmAgent config shape + task-delegation models
**2026-08-21** · (link added once this PR is opened)

- **Added:** `LlmAgent`'s config fields and the handful of methods that
  don't need Phase 3/4/8 to exist: `canonical_instruction`/
  `canonical_global_instruction` (string-or-provider resolution), the
  `generate_content_config` validator (rejects `tools`/`system_instruction`/
  `response_schema`/`http_options.base_url` set there), the `_llm_flow`
  single-vs-auto flow *decision*, `set_default_model`/`set_default_live_model`,
  and the 6 model/tool callback fields' stop-at-first-non-`None` chain
  contract. Plus `TaskRequest`/`TaskResult`/`DefaultTaskInput`/
  `DefaultTaskOutput` (task-delegation payload shapes — fully self-contained,
  no forward references). 21 new tests (155 total workspace-wide), clippy/fmt
  clean.
- **Scope decision:** `LlmAgent` is implemented as a standalone struct this
  batch, not yet wired into `BaseAgent`'s tree/`AgentBehavior` — its real
  behavior (`canonical_model`, `canonical_tools`, `_run_async_impl`) is
  driven entirely by types this migration hasn't built yet (`BaseLlm`/
  `LlmRequest`/`LlmResponse`/`LLMRegistry` — Phase 3; `BaseTool`/
  `BaseToolset`/`ToolContext` — Phase 8; `BaseLlmFlow`/`SingleFlow`/
  `AutoFlow`/planners — Phase 4). Wiring it into the tree now would mean
  giving it a `_run_async_impl` that can't actually run anything; that
  integration happens once those phases land.
- **Known gaps, flagged not hidden** — left `REQUIRED`: `canonical_model`/
  `canonical_live_model` (need `LLMRegistry`), `canonical_tools` (needs
  `BaseTool` resolution), `_get_subagent_to_resume`/
  `__maybe_save_output_to_state`/`__maybe_accumulate_streaming_output`
  (blocked on Phase 3's real `Content`/`Part` — `Event.get_function_calls`/
  `get_function_responses` don't exist yet), `_pre_validate_tools`'s
  tool/sub-agent auto-wrapping (needs `BaseNode`/`BaseTool`/
  `FinishTaskTool`), the deprecated YAML config pipeline (needs the same
  design decision as `BaseAgent::from_config`), and `FinishTaskTool` itself
  (needs `BaseTool`/`LlmRequest`/`ToolContext`).
- 13 more Phase 2 rows marked `DONE` (51 of 66 in the `agents/` phase now
  complete) with per-row test-name evidence in `capability-manifest.md`.
  Phase 2's remaining 15 rows are all blocked on later phases landing
  first — Phase 2 itself is otherwise complete for what's buildable today.

## PR #TBD — Phase 2 batch 1: BaseAgent + Context/InvocationContext/RunConfig
**2026-08-21** · (link added once this PR is opened)

- **Added:** `adk-agents` crate covering `BaseAgent` (name/parent-tree
  validation, before/after-agent callback chains with plugin-precedence and
  error-notification-then-reraise, clone, tree-walk helpers) and the
  `Context`/`ReadonlyContext`/`InvocationContext` family (state, output,
  route, interrupt-ids, event-author, artifact/credential/memory
  tool-context methods, agent-state machine, per-invocation LLM-call-limit
  enforcement), plus `RunConfig`/`ToolThreadPoolConfig`/`LiveRequest`/
  `LiveRequestQueue`/`ActiveStreamingTool`/`ContextCacheConfig`/
  `TranscriptionEntry`/`StreamingMode`. 73 new tests (134 total across the
  workspace), clippy/fmt clean.
- **Runners.py inventory gap closed:** a dedicated read of `runners.py`
  (2609 lines, flagged in row C0788 as outside the original 8-agent
  inventory's scope) added 94 granular capability rows (C0833-C0926,
  Phase 2) — the `Runner`/`InMemoryRunner` execution engine's actual
  sub-capabilities, previously referenced only piecemeal by other reports.
- **Async runtime decision:** `rusty_tokio` adopted (pinned git-rev) — the
  first capability needing one, `LiveRequestQueue`'s `asyncio.Queue`
  equivalent. Chosen over `rustils_async`, which is a narrow OS-level
  async-I/O primitives layer (process/fs/net) with no task-spawning or
  channel API; `rusty_tokio` is a general-purpose runtime matching what
  ADK's in-process task/queue orchestration needs.
- **Disclosed adaptations** (each documented in its module's doc comment):
  tree ownership via `Arc`+`OnceLock` back-pointer (mirrors the "adopted as
  a sub-agent only once" invariant as a single `OnceLock::set` failure);
  `_run_async_impl`/`_run_live_impl` represented as eagerly-collected
  `Vec<Event>` rather than a live stream, since no concrete agent subclass
  yet needs incremental yielding; `sessions.state.State` (Phase 5) and a
  minimal `Session` placeholder pulled forward since `Context` cannot
  compile without them; `PluginManager` (Phase 7) is real but structurally
  always returns `None` for the zero-plugins-registered case.
- **Known gaps, flagged not hidden:** `BaseAgent._run_impl` (needs
  `workflow::BaseNode`, Phase 7), `BaseAgent::from_config` dynamic
  resolution (needs a design decision — no Rust equivalent to Python's
  dotted-path dynamic loading), `Context.run_node`/`_run_node_internal`
  (needs the Phase 7 workflow engine), `InvocationContext`'s live-mode
  fields and `_enqueue_event` (need a concrete live-mode consumer),
  and `should_pause_invocation`/`_find_matching_function_call`/
  `stamp_event_branch_context` (blocked on Phase 3's real `Content`/`Part`)
  all stay `REQUIRED` in the manifest, not marked `DONE`.
- `LlmAgent` (C0079-C0100) is deferred to a follow-up batch — it has the
  heaviest forward-reference surface (models/`BaseLlm`, tools, planners,
  flows) of anything in Phase 2.
- 38 of the addressed Phase 2 rows marked `DONE` with per-row test-name
  evidence in `capability-manifest.md`.

## PR #TBD — Phase 1: platform/errors/events primitives (C0001-C0033)
**2026-08-21** · (link added once this PR is opened)

- **Added:** Cargo workspace with three crates —
  `adk-platform` (random/time/uuid providers, thread factory),
  `adk-errors` (6 error types + the `ValueErrorLike` marker trait), and
  `adk-events` (`Event`/`EventActions`/`RequestInput`/`NodeInfo`/
  `EventCompaction`/`UiWidget`/branch-path and node-path helpers/
  `apply_rewinds`) — implementing all 33 Phase 1 capabilities. 61 tests,
  `cargo clippy -- -D warnings` clean, `cargo fmt --check` clean.
- **Sibling-crate decisions:** `rusty_uuid`, `rusty_time`, and `rusty_err`
  adopted as pinned git-rev workspace dependencies after inspection (per the
  user's direction to check `rustils_async`/`rusty_tokio` and
  `rusty_serde`/`rusty_json`/`rusty_uuid`/`rusty_rand` first); `rusty_serde`
  chosen over `rusty_json` for serialization (fully sovereign, zero
  external deps) at the user's explicit direction. No async runtime is
  adopted yet — Phase 1 needs none; the tokio/rusty_tokio/rustils_async
  choice is deferred to whichever later phase first needs an executor.
- **Adaptation, disclosed:** the source's `ContextVar`-scoped provider
  pattern (random/time/uuid) is ported to `thread_local!` instead, since no
  async runtime is chosen yet — documented inline in each provider module
  and revisitable once a runtime lands.
- **Adaptation, disclosed:** `Event(LlmResponse)` needs ~20 fields from
  `LlmResponse` (Phase 3, not yet built); those fields are flattened onto
  `Event` as typed placeholders (mostly `rusty_serde::value::Value`) with a
  documented plan to replace them with `#[rusty_serde(flatten)] base:
  LlmResponse` once Phase 3 lands.
- **Known gap, flagged not hidden:** C0022 (`is_final_response`) and C0023
  (`has_trailing_code_execution_result`) are left `REQUIRED`, not `DONE` —
  both need real `Content`/`Part` inspection from Phase 3 for true parity;
  this port's versions cover only the branches that don't depend on those
  types.
- 31 of the 33 Phase 1 manifest rows marked `DONE` with per-row test-name
  evidence in `capability-manifest.md`; C0022/C0023 remain `REQUIRED`.

## PR #TBD — Capability roadmap: full 831-row manifest for the google/adk-python migration
**2026-08-21** · (link added once this PR is opened)

- **Added:** `capability-manifest.md` — the complete capability inventory for
  the migration, one row per capability with a stable `C####` id, grouped into
  17 dependency-ordered phases (P1 core primitives through P17 deferred
  decisions). Built from 8 parallel read-only inventory passes over
  `google/adk-python`'s ~206k lines / 28 top-level modules.
- **Known limitation:** the `Existing RustyMill impl` column is populated only
  where a repo/purpose match was obvious from `platform-directory.md`'s
  heuristic — a full `scan_platform_repos.sh` pass per capability wasn't run
  at this scale; it's re-checked per-issue when each capability is actually
  worked (per the rust-migration skill's step 3).
- **Known gap, flagged not hidden:** `runners.py` (the core `Runner`/
  `InMemoryRunner` execution engine, 2609 lines) sits outside the 28 scoped
  module directories and wasn't deep-dived by any of the 8 inventory agents —
  row `C0788` flags this explicitly as a required follow-up read before P2
  can be considered fully scoped, rather than silently omitting it.
- Every row defaults `REQUIRED` per the migration's boundary contract;
  `scripts/check_manifest_coverage.sh` confirms all 831 rows parse and are
  correctly in a non-terminal state (nothing migrated yet, as expected).
- No GitHub issues have been filed yet — the user asked for the manifest
  organized into phases first, to review and pace the work themselves.

## PR #TBD — Bootstrap governance files; begin rust-migration of google/adk-python
**2026-08-21** · (link added once this PR is opened)

- **Added:** standard governance file set (repo-config) — README, ARCHITECTURE,
  CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG, RELEASE_NOTES, ADR seed,
  PR/issue templates, `.gitattributes` (`eol=lf`).
- **Known limitation:** no Cargo workspace exists yet, so no CI workflow was
  added (repo-config skips CI generation when there's no stack manifest to run
  against) and README's Getting Started section is a placeholder.
- This is the first PR of an ongoing rust-migration-skill loop porting
  [google/adk-python](https://github.com/google/adk-python) to Rust; see
  `capability-manifest.md` (added in a follow-up PR) for the tracked capability
  list.
