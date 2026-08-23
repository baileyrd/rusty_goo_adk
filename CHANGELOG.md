# Changelog

All notable changes to this repo are documented here.
Format: Added / Changed / Deprecated / Removed / Fixed / Security, newest first.

## [Unreleased]
### Added
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
### Changed
### Fixed
### Security

<!-- ## [0.1.0] - YYYY-MM-DD
### Added
- Initial release -->
