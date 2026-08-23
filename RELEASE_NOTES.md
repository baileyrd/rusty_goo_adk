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
