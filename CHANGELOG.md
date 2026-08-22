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
- **Scope decision:** the SSE-streaming half of `generate_content_async`
  (`stream=true`, `StreamingResponseAggregator`), context-cache
  integration, interactions-API delegation, the Live `connect()`/
  `GeminiLlmConnection`, redacted request/response logging, and
  `GeminiContextCacheManager` (C0125's streaming half, C0126, C0128,
  C0131-C0143) are deferred to further batches — see
  `crates/adk-models/src/gemini.rs`'s module doc for exactly what's left
  and why.
### Changed
### Fixed
### Security

<!-- ## [0.1.0] - YYYY-MM-DD
### Added
- Initial release -->
