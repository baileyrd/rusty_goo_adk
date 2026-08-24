//! Capabilities C0144-C0157 (partial): the `BaseLlmFlow`/`SingleFlow`
//! turn-orchestration engine, ported from `google.adk.flows.llm_flows`.
//!
//! [`LlmFlow`] is the first concrete [`AgentBehavior`] this port builds for
//! an [`LlmAgent`] — it's what an `LlmAgent` is actually *for*: given a
//! [`BaseAgent`] wired with `LlmFlow` as its behavior, `BaseAgent::run_async`
//! now drives a real (if narrowed) request → model call → response turn,
//! using every processor this crate has built so far
//! ([`crate::basic`], [`crate::identity`], [`crate::instructions`],
//! [`crate::contents`], [`crate::context_cache`]).
//!
//! **Scope, disclosed** — this is the single non-live step
//! (`_run_one_step_async`'s core, not the full engine):
//!   - **No turn loop / tool execution** (C0148, C0149, C0151, C0152,
//!     C0158, C0159): without `BaseTool` (Phase 8) a model backend is
//!     never handed any tools, so it has nothing to call — the source's
//!     multi-step "call model, run tools, call model again" loop has
//!     nothing to loop on yet. `run_one_step` is exactly one model call;
//!     [`AgentBehavior::run_async_impl`] calls it once and returns.
//!   - **`interactions_processor` (C0174), now wired**: `preprocess` gates
//!     on `self.model.as_ref().as_any().downcast_ref::<Gemini>()` (the
//!     `AsAny` downcast mechanism `adk-models::base_llm` now provides,
//!     mirroring `AgentBehavior::as_any`'s already-reviewed pattern in
//!     `adk-agents`) plus [`Gemini::use_interactions_api`]
//!     (`adk_models::gemini`) — both real fields, no longer blocked.
//!   - **No telemetry spans, before/after/on-error model callback
//!     dispatch** (C0154, C0155): `LlmAgent.before_model_callback`/
//!     `after_model_callback` exist as real fields but dispatching them
//!     needs a `Context`-based short-circuit path like the agent-level
//!     callbacks `BaseAgent` already has (C0038/C0045) — a follow-up, not
//!     built here.
//!   - **No live mode** (C0161-C0167): [`AgentBehavior::run_live_impl`]
//!     returns [`LlmFlowError::LiveNotImplemented`].
//!   - **`preserve_function_call_ids`** is always `false` — the source's
//!     policy needs detecting Anthropic/LiteLLM/OpenAIResponsesLlm/
//!     Interactions-API-Gemini backends. Unlike `interactions_processor`
//!     above, the downcast mechanism alone doesn't close this: three of
//!     those four backends don't exist in this port at all yet (only
//!     `Gemini`/`Ollama` do), so there's nothing to downcast *to* for
//!     them — still correctly deferred (see `contents.rs`'s own
//!     disclosure for C0181).
//!   - **`ctx.user_content`** stays an opaque `Value` placeholder (see
//!     `invocation_context.rs`'s own module doc), so it's never forwarded
//!     into [`crate::contents::get_contents`]'s `user_content` parameter
//!     (`None` always) — narrowing it to a real `Content` is its own
//!     future batch.
//!
//! **Adaptation**: `LlmFlow` resolves its model once, at construction
//! (`LlmFlow::new`), rather than through `canonical_model` on every call —
//! a real (if narrow) instance of the source's own disclosed-missing
//! memoization cache (see `canonical_model.rs`'s module doc). This also
//! makes `LlmFlow` trivially testable: `LlmFlow::with_model` injects any
//! `Arc<dyn BaseLlm>`, including a test double, without touching the
//! process-wide registry `canonical_model` otherwise hits.

use std::sync::Arc;

use adk_agents::base_agent::{AgentBehavior, AgentRunError};
use adk_agents::invocation_context::{InvocationContext, InvocationContextError};
use adk_agents::llm_agent::{AgentMode, IncludeContents, LlmAgent};
use adk_agents::readonly_context::ReadonlyContext;
use adk_agents::run_config::RunConfig;
use adk_agents::streaming_mode::StreamingMode;
use adk_events::node_info::NodeInfo;
use adk_events::Event;
use adk_models::base_llm::{BaseLlm, BaseLlmError};
use adk_models::llm_request::LlmRequest;
use adk_models::llm_response::LlmResponse;

use crate::canonical_model::{canonical_model, CanonicalModelError};
use crate::contents::{get_contents, get_current_turn_contents, ContentsError};
use crate::context_cache::{apply_context_cache, ContextCacheError};
use crate::identity::apply_identity;
use crate::instructions::{build_instructions, InstructionsError};
use crate::interactions::find_previous_interaction_state;
use crate::processor::BoxFuture;
use crate::{basic, basic::BasicRequestError};

#[derive(Debug, rusty_err::Error)]
pub enum LlmFlowError {
    #[error("{0}")]
    CanonicalModel(#[from] CanonicalModelError),
    #[error("{0}")]
    BasicRequest(#[from] BasicRequestError),
    #[error("{0}")]
    Instructions(#[from] InstructionsError),
    #[error("{0}")]
    Contents(#[from] ContentsError),
    #[error("{0}")]
    ContextCache(#[from] ContextCacheError),
    #[error("{0}")]
    InvocationContext(#[from] InvocationContextError),
    #[error("model call failed: {0}")]
    ModelCall(#[from] BaseLlmError),
    #[error("live mode isn't implemented yet in this port (C0161-C0167)")]
    LiveNotImplemented,
}

/// Bridges an `LlmFlowError` into [`AgentRunError`]
/// (`Box<dyn std::error::Error + Send + Sync>`). `rusty_err::Error` (this
/// crate's own sovereign error trait, used everywhere else in this port)
/// and `std::error::Error` are deliberately separate traits — `rusty_err`
/// bridges *from* `std::error::Error` for free, not the other way, and
/// implementing `std::error::Error` directly on `LlmFlowError` would
/// conflict with that blanket impl — so this tiny wrapper carries just the
/// rendered message across the boundary, the same shape `BaseAgent`'s own
/// tests use for a boxed error.
#[derive(Debug)]
struct BoxedLlmFlowError(String);

impl std::fmt::Display for BoxedLlmFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BoxedLlmFlowError {}

impl From<LlmFlowError> for AgentRunError {
    fn from(error: LlmFlowError) -> Self {
        Box::new(BoxedLlmFlowError(error.to_string()))
    }
}

const NO_CONTENT_ERROR_CODE: &str = "MODEL_RETURNED_NO_CONTENT";
const NO_CONTENT_ERROR_MESSAGE: &str =
    "The model returned no content (finish_reason=STOP with empty parts).";

/// C0156: a non-streaming turn that finishes with STOP but has no
/// content parts would otherwise be skipped by
/// [`should_skip_empty_response`] and become a silent empty final
/// response; surfaces it as an actionable error instead. Streaming
/// (`StreamingMode::Sse`) is excluded because a terminal finish-only
/// chunk legitimately follows content already streamed in earlier
/// chunks. `finish_reason` compares against the literal `"STOP"` — both
/// sides of this check are opaque-`Value`/placeholder types pending a
/// real typed `FinishReason` (see [`finalize_model_response_event`]'s
/// own doc for the same narrowing).
fn apply_no_content_error(response: &mut LlmResponse, run_config: &RunConfig) {
    let is_stop = response.finish_reason.as_ref().and_then(|v| v.as_str()) == Some("STOP");
    let has_no_content = match &response.content {
        None => true,
        Some(content) => content.parts.is_empty(),
    };
    if !response.partial.unwrap_or(false)
        && response.error_code.is_none()
        && is_stop
        && has_no_content
        && run_config.streaming_mode != StreamingMode::Sse
    {
        response.error_code = Some(NO_CONTENT_ERROR_CODE.to_string());
        response.error_message = Some(
            response
                .error_message
                .clone()
                .unwrap_or_else(|| NO_CONTENT_ERROR_MESSAGE.to_string()),
        );
    }
}

/// C0156: mirrors "skip the model response event if there is no content
/// and no error code" — needed upstream for the code executor to
/// trigger another loop (that consumer, C0158's function-call/tool-loop
/// delegation, isn't built in this port; the skip check itself has no
/// such dependency and applies regardless).
fn should_skip_empty_response(response: &LlmResponse) -> bool {
    response.content.is_none()
        && response.error_code.is_none()
        && !response.interrupted.unwrap_or(false)
        && response.grounding_metadata.is_none()
}

/// C0157: `_finalize_model_response_event` — shallow-copies every
/// non-`None` `LlmResponse` field onto `event`. `finish_reason` is
/// narrowed from `LlmResponse`'s opaque `Value` to `Event`'s `String` via
/// `Value::as_str` (both sides are placeholders pending a real typed
/// `FinishReason`; only a string-shaped value round-trips). `cache_metadata`
/// goes the other way — `LlmResponse`'s real `CacheMetadata` re-widens to
/// `Event`'s opaque `Value` via `to_value`, since `Event` can't hold the
/// concrete type without depending on `adk-models` (a crate-graph cycle,
/// same constraint `event_compaction.rs` and `context_cache.rs` disclose).
pub fn finalize_model_response_event(event: &mut Event, response: &LlmResponse) {
    if let Some(content) = &response.content {
        event.content = Some(content.clone());
    }
    if let Some(v) = &response.grounding_metadata {
        event.grounding_metadata = Some(v.clone());
    }
    if let Some(v) = response.partial {
        event.partial = Some(v);
    }
    if let Some(v) = response.turn_complete {
        event.turn_complete = Some(v);
    }
    if let Some(v) = &response.turn_complete_reason {
        event.turn_complete_reason = Some(v.clone());
    }
    if let Some(v) = response.finish_reason.as_ref().and_then(|v| v.as_str()) {
        event.finish_reason = Some(v.to_string());
    }
    if let Some(v) = &response.error_code {
        event.error_code = Some(v.clone());
    }
    if let Some(v) = &response.error_message {
        event.error_message = Some(v.clone());
    }
    if let Some(v) = response.interrupted {
        event.interrupted = Some(v);
    }
    if let Some(v) = &response.custom_metadata {
        event.custom_metadata = Some(v.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    }
    if let Some(v) = &response.usage_metadata {
        event.usage_metadata = Some(v.clone());
    }
    if let Some(v) = &response.live_session_resumption_update {
        event.live_session_resumption_update = Some(v.clone());
    }
    if let Some(v) = &response.live_session_id {
        event.live_session_id = Some(v.clone());
    }
    if let Some(v) = &response.go_away {
        event.go_away = Some(v.clone());
    }
    if let Some(v) = &response.voice_activity {
        event.voice_activity = Some(v.clone());
    }
    if let Some(v) = &response.input_transcription {
        event.input_transcription = Some(v.clone());
    }
    if let Some(v) = &response.output_transcription {
        event.output_transcription = Some(v.clone());
    }
    if let Some(v) = response.avg_logprobs {
        event.avg_logprobs = Some(v);
    }
    if let Some(v) = &response.logprobs_result {
        event.logprobs_result = Some(v.clone());
    }
    if let Some(cache_metadata) = &response.cache_metadata {
        if let Ok(v) = rusty_serde::json::to_value(cache_metadata) {
            event.cache_metadata = Some(v);
        }
    }
    if let Some(v) = &response.citation_metadata {
        event.citation_metadata = Some(v.clone());
    }
    if let Some(v) = &response.interaction_id {
        event.interaction_id = Some(v.clone());
    }
    if let Some(v) = &response.environment_id {
        event.environment_id = Some(v.clone());
    }
}

/// The turn-orchestration engine for one [`LlmAgent`] — see the module doc
/// for exactly what's wired and what's deferred.
pub struct LlmFlow {
    pub llm_agent: LlmAgent,
    pub model: Arc<dyn BaseLlm>,
}

impl LlmFlow {
    /// Resolves `llm_agent.model` once via [`canonical_model`] and holds
    /// onto it — see the module doc's memoization note.
    pub fn new(llm_agent: LlmAgent) -> Result<Self, LlmFlowError> {
        let model = canonical_model(&llm_agent)?;
        Ok(Self { llm_agent, model })
    }

    /// Builds an `LlmFlow` from an already-resolved model — the seam a
    /// test double (or a future live-instance-model agent) plugs into
    /// without touching the process-wide registry.
    pub fn with_model(llm_agent: LlmAgent, model: Arc<dyn BaseLlm>) -> Self {
        Self { llm_agent, model }
    }

    /// C0150 (partial): assembles the `LlmRequest` — `basic` → `identity`
    /// → `instructions` → contents (full history or current-turn-only,
    /// per `include_contents`) → `context_cache`. See the module doc for
    /// what's not wired (toolset auth/tool resolution, dynamic
    /// instructions, `interactions_processor`).
    pub async fn preprocess(&self, ctx: &InvocationContext) -> Result<LlmRequest, LlmFlowError> {
        let agent_name = ctx
            .agent
            .as_ref()
            .map(|a| a.name().to_string())
            .unwrap_or_default();
        let agent_description = ctx.agent.as_ref().map(|a| a.description().to_string());

        let default_run_config = RunConfig::default();
        let run_config = ctx.run_config.as_ref().unwrap_or(&default_run_config);

        let mut request = LlmRequest::default();
        basic::build_basic_request(&self.llm_agent, run_config, &mut request)?;
        apply_identity(
            &agent_name,
            agent_description.as_deref(),
            self.llm_agent.mode,
            &mut request,
        );

        let readonly_ctx = ReadonlyContext::new(ctx.clone());
        build_instructions(&self.llm_agent, &readonly_ctx, &mut request)?;

        let events = ctx.get_events(false, false);

        // C0174: interactions_processor — gate on the resolved model
        // being a Gemini with `use_interactions_api` set. `self.model`
        // is already resolved once at construction (this struct's own
        // disclosed memoization adaptation), so this downcasts it
        // directly rather than re-resolving via `canonical_model`.
        if let Some(gemini) = self
            .model
            .as_ref()
            .as_any()
            .downcast_ref::<adk_models::gemini::Gemini>()
        {
            if gemini.use_interactions_api {
                let (previous_interaction_id, _environment_id) =
                    find_previous_interaction_state(&events, &agent_name, ctx.branch.as_deref());
                request.previous_interaction_id = previous_interaction_id;
            }
        }

        let is_single_turn = self.llm_agent.mode == Some(AgentMode::SingleTurn);
        let preserve_function_call_ids = false;
        let include_thoughts_from_other_agents = run_config.include_thoughts_from_other_agents;

        request.contents = if self.llm_agent.include_contents == IncludeContents::Default
            && request.previous_interaction_id.is_none()
        {
            get_contents(
                ctx.branch.as_deref(),
                &events,
                &agent_name,
                preserve_function_call_ids,
                ctx.isolation_scope.as_deref(),
                is_single_turn,
                None,
                include_thoughts_from_other_agents,
            )?
        } else {
            get_current_turn_contents(
                ctx.branch.as_deref(),
                &events,
                &agent_name,
                preserve_function_call_ids,
                ctx.isolation_scope.as_deref(),
                is_single_turn,
                None,
                false,
            )?
        };

        apply_context_cache(
            &mut request,
            ctx.context_cache_config.as_ref(),
            &events,
            &agent_name,
            &ctx.invocation_id,
        )?;

        Ok(request)
    }

    /// C0153 (partial): calls the resolved model. No telemetry spans, no
    /// before/after-model callback dispatch, no live branch — see the
    /// module doc.
    pub async fn call_model(&self, request: &LlmRequest) -> Result<Vec<LlmResponse>, LlmFlowError> {
        self.model
            .generate_content_async(request, false)
            .await
            .map_err(LlmFlowError::from)
    }

    /// C0156 (partial): converts each `LlmResponse` into an `Event` via
    /// [`finalize_model_response_event`] — first applying the
    /// no-content-error conversion ([`apply_no_content_error`]) and the
    /// empty-response skip ([`should_skip_empty_response`]), in that
    /// order (a response the first rewrites into an error is never
    /// skipped by the second, matching the source's own sequential
    /// mutate-then-check). No function-call delegation — see the module
    /// doc.
    pub fn postprocess(&self, ctx: &InvocationContext, responses: Vec<LlmResponse>) -> Vec<Event> {
        let agent_name = ctx
            .agent
            .as_ref()
            .map(|a| a.name().to_string())
            .unwrap_or_default();
        let node_path = ctx.node_path.clone().unwrap_or_default();
        let default_run_config = RunConfig::default();
        let run_config = ctx.run_config.as_ref().unwrap_or(&default_run_config);

        responses
            .into_iter()
            .filter_map(|mut response| {
                apply_no_content_error(&mut response, run_config);
                if should_skip_empty_response(&response) {
                    return None;
                }
                let mut event = Event::new(
                    ctx.invocation_id.clone(),
                    agent_name.clone(),
                    NodeInfo::new(node_path.clone()),
                );
                event.branch = ctx.branch.clone();
                event.isolation_scope = ctx.isolation_scope.clone();
                finalize_model_response_event(&mut event, &response);
                Some(event)
            })
            .collect()
    }

    /// C0149 (partial): one model turn — preprocess, call the model,
    /// postprocess. No multi-step loop (see the module doc).
    pub async fn run_one_step(
        &self,
        ctx: &mut InvocationContext,
    ) -> Result<Vec<Event>, LlmFlowError> {
        ctx.increment_llm_call_count()?;
        let request = self.preprocess(ctx).await?;
        let responses = self.call_model(&request).await?;
        Ok(self.postprocess(ctx, responses))
    }
}

impl AgentBehavior for LlmFlow {
    fn run_async_impl<'a>(
        &'a self,
        ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async move { self.run_one_step(ctx).await.map_err(AgentRunError::from) })
    }

    fn run_live_impl<'a>(
        &'a self,
        _ctx: &'a mut InvocationContext,
    ) -> BoxFuture<'a, Result<Vec<Event>, AgentRunError>> {
        Box::pin(async { Err(AgentRunError::from(LlmFlowError::LiveNotImplemented)) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::base_agent::BaseAgent;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::llm_agent::ModelRef;
    use adk_agents::session::Session;
    use adk_genai::content::{Content, Part};
    use std::future::Future;
    use std::pin::Pin;

    struct FakeLlm {
        responses: Vec<LlmResponse>,
    }

    impl BaseLlm for FakeLlm {
        fn model(&self) -> &str {
            "fake-model"
        }

        fn type_name(&self) -> &'static str {
            "FakeLlm"
        }

        fn generate_content_async<'a>(
            &'a self,
            _llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let responses = self.responses.clone();
            Box::pin(async move { Ok(responses) })
        }
    }

    fn flow_with_response(response: LlmResponse) -> LlmFlow {
        // `basic::build_basic_request` (inside `preprocess`) independently
        // resolves `llm_agent.model` through the real, process-wide model
        // registry (just to read off `.model()`/default config — no
        // network call happens there) — so this must name a model the
        // real registry actually resolves. The `FakeLlm` injected below is
        // what `call_model` actually invokes, so no real network call
        // happens anywhere in this test.
        let llm_agent = LlmAgent::new(ModelRef::Name("gemini-2.0-flash".to_string()));
        LlmFlow::with_model(
            llm_agent,
            Arc::new(FakeLlm {
                responses: vec![response],
            }),
        )
    }

    fn ctx_for(agent_name: &str) -> InvocationContext {
        let agent = BaseAgent::new(agent_name, adk_agents::base_agent::NoopBehavior).unwrap();
        InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1"))
            .agent(agent)
            .build()
    }

    /// Same as [`ctx_for`], but the session already carries one prior
    /// event from `agent_name` with `interaction_id` set — the shape
    /// [`find_previous_interaction_state`] scans for.
    fn ctx_with_prior_interaction(agent_name: &str, interaction_id: &str) -> InvocationContext {
        let mut session = Session::new("app", "user", "s1");
        let mut prior_event = Event::new("inv-0", agent_name, NodeInfo::new("root"));
        prior_event.interaction_id = Some(interaction_id.to_string());
        session.events.push(prior_event);

        let agent = BaseAgent::new(agent_name, adk_agents::base_agent::NoopBehavior).unwrap();
        InvocationContextBuilder::new("inv-1", session)
            .agent(agent)
            .build()
    }

    // --- finalize_model_response_event ---

    #[test]
    fn finalize_copies_content_and_leaves_other_fields_alone_when_absent() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        let response = LlmResponse {
            content: Some(Content::user_text("hello")),
            ..Default::default()
        };
        finalize_model_response_event(&mut event, &response);
        assert_eq!(
            event.content.unwrap().parts[0].text.as_deref(),
            Some("hello")
        );
        assert!(event.partial.is_none());
    }

    #[test]
    fn finalize_copies_every_populated_field() {
        let mut event = Event::new("inv-1", "agent", NodeInfo::new("root"));
        let response = LlmResponse {
            partial: Some(true),
            turn_complete: Some(false),
            finish_reason: Some(rusty_serde::value::Value::String("STOP".to_string())),
            error_code: Some("E1".to_string()),
            error_message: Some("boom".to_string()),
            interrupted: Some(true),
            avg_logprobs: Some(-0.5),
            interaction_id: Some("interaction-1".to_string()),
            environment_id: Some("env-1".to_string()),
            ..Default::default()
        };
        finalize_model_response_event(&mut event, &response);
        assert_eq!(event.partial, Some(true));
        assert_eq!(event.turn_complete, Some(false));
        assert_eq!(event.finish_reason.as_deref(), Some("STOP"));
        assert_eq!(event.error_code.as_deref(), Some("E1"));
        assert_eq!(event.error_message.as_deref(), Some("boom"));
        assert_eq!(event.interrupted, Some(true));
        assert_eq!(event.avg_logprobs, Some(-0.5));
        assert_eq!(event.interaction_id.as_deref(), Some("interaction-1"));
        assert_eq!(event.environment_id.as_deref(), Some("env-1"));
    }

    // --- interactions_processor gating (C0174) ---

    #[rusty_tokio::test]
    async fn preprocess_sets_previous_interaction_id_for_a_gemini_using_interactions_api() {
        let mut gemini = adk_models::gemini::Gemini::new("gemini-2.0-flash");
        gemini.use_interactions_api = true;
        let llm_agent = LlmAgent::new(ModelRef::Name("gemini-2.0-flash".to_string()));
        let flow = LlmFlow::with_model(llm_agent, Arc::new(gemini));
        let ctx = ctx_with_prior_interaction("my_agent", "prev-interaction");

        let request = flow.preprocess(&ctx).await.unwrap();
        assert_eq!(
            request.previous_interaction_id.as_deref(),
            Some("prev-interaction")
        );
    }

    #[rusty_tokio::test]
    async fn preprocess_leaves_previous_interaction_id_unset_when_use_interactions_api_is_false() {
        let gemini = adk_models::gemini::Gemini::new("gemini-2.0-flash");
        let llm_agent = LlmAgent::new(ModelRef::Name("gemini-2.0-flash".to_string()));
        let flow = LlmFlow::with_model(llm_agent, Arc::new(gemini));
        let ctx = ctx_with_prior_interaction("my_agent", "prev-interaction");

        let request = flow.preprocess(&ctx).await.unwrap();
        assert_eq!(request.previous_interaction_id, None);
    }

    #[rusty_tokio::test]
    async fn preprocess_leaves_previous_interaction_id_unset_for_a_non_gemini_model() {
        // FakeLlm doesn't downcast to Gemini at all, regardless of any
        // field it might otherwise have.
        let flow = flow_with_response(LlmResponse::default());
        let ctx = ctx_with_prior_interaction("my_agent", "prev-interaction");

        let request = flow.preprocess(&ctx).await.unwrap();
        assert_eq!(request.previous_interaction_id, None);
    }

    // --- postprocess: MODEL_RETURNED_NO_CONTENT + empty-event skip (C0156) ---

    fn stop_response(content: Option<Content>) -> LlmResponse {
        LlmResponse {
            content,
            finish_reason: Some(rusty_serde::value::Value::String("STOP".to_string())),
            ..Default::default()
        }
    }

    #[test]
    fn apply_no_content_error_converts_a_non_streaming_stop_with_no_content() {
        let mut response = stop_response(None);
        apply_no_content_error(&mut response, &RunConfig::default());
        assert_eq!(response.error_code.as_deref(), Some(NO_CONTENT_ERROR_CODE));
        assert_eq!(
            response.error_message.as_deref(),
            Some(NO_CONTENT_ERROR_MESSAGE)
        );
    }

    #[test]
    fn apply_no_content_error_converts_stop_with_empty_parts_the_same_as_no_content() {
        let mut response = stop_response(Some(Content::new("model", vec![])));
        apply_no_content_error(&mut response, &RunConfig::default());
        assert_eq!(response.error_code.as_deref(), Some(NO_CONTENT_ERROR_CODE));
    }

    #[test]
    fn apply_no_content_error_preserves_an_existing_error_message() {
        let mut response = stop_response(None);
        response.error_message = Some("already set".to_string());
        apply_no_content_error(&mut response, &RunConfig::default());
        assert_eq!(response.error_message.as_deref(), Some("already set"));
    }

    #[test]
    fn apply_no_content_error_is_excluded_for_sse_streaming() {
        let mut response = stop_response(None);
        let run_config = RunConfig {
            streaming_mode: StreamingMode::Sse,
            ..Default::default()
        };
        apply_no_content_error(&mut response, &run_config);
        assert!(response.error_code.is_none());
    }

    #[test]
    fn apply_no_content_error_leaves_a_partial_response_alone() {
        let mut response = stop_response(None);
        response.partial = Some(true);
        apply_no_content_error(&mut response, &RunConfig::default());
        assert!(response.error_code.is_none());
    }

    #[test]
    fn apply_no_content_error_leaves_a_non_stop_finish_reason_alone() {
        let mut response = LlmResponse {
            finish_reason: Some(rusty_serde::value::Value::String("MAX_TOKENS".to_string())),
            ..Default::default()
        };
        apply_no_content_error(&mut response, &RunConfig::default());
        assert!(response.error_code.is_none());
    }

    #[test]
    fn apply_no_content_error_leaves_content_bearing_responses_alone() {
        let mut response = stop_response(Some(Content::user_text("hi")));
        apply_no_content_error(&mut response, &RunConfig::default());
        assert!(response.error_code.is_none());
    }

    #[test]
    fn should_skip_empty_response_is_true_for_a_bare_default_response() {
        assert!(should_skip_empty_response(&LlmResponse::default()));
    }

    #[test]
    fn should_skip_empty_response_is_false_once_content_is_present() {
        let response = LlmResponse {
            content: Some(Content::user_text("hi")),
            ..Default::default()
        };
        assert!(!should_skip_empty_response(&response));
    }

    #[test]
    fn should_skip_empty_response_is_false_when_an_error_code_is_set() {
        let response = LlmResponse {
            error_code: Some("E1".to_string()),
            ..Default::default()
        };
        assert!(!should_skip_empty_response(&response));
    }

    #[test]
    fn should_skip_empty_response_is_false_when_interrupted() {
        let response = LlmResponse {
            interrupted: Some(true),
            ..Default::default()
        };
        assert!(!should_skip_empty_response(&response));
    }

    #[test]
    fn should_skip_empty_response_is_false_when_grounding_metadata_is_present() {
        let response = LlmResponse {
            grounding_metadata: Some(rusty_serde::value::Value::String("g".to_string())),
            ..Default::default()
        };
        assert!(!should_skip_empty_response(&response));
    }

    #[test]
    fn postprocess_skips_a_response_with_no_content_and_no_error() {
        let flow = flow_with_response(LlmResponse::default());
        let ctx = ctx_for("my_agent");
        let events = flow.postprocess(&ctx, vec![LlmResponse::default()]);
        assert!(events.is_empty());
    }

    #[test]
    fn postprocess_yields_a_no_content_error_event_instead_of_skipping() {
        // The no-content-error conversion runs before the skip check, so a
        // STOP-with-no-content response becomes a visible error event
        // rather than vanishing silently.
        let flow = flow_with_response(LlmResponse::default());
        let ctx = ctx_for("my_agent");
        let events = flow.postprocess(&ctx, vec![stop_response(None)]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].error_code.as_deref(), Some(NO_CONTENT_ERROR_CODE));
    }

    // --- LlmFlow turn ---

    #[rusty_tokio::test]
    async fn run_one_step_calls_the_model_and_produces_a_finalized_event() {
        let response = LlmResponse {
            content: Some(Content::new("model", vec![Part::text("hi there")])),
            ..Default::default()
        };
        let flow = flow_with_response(response);
        let mut ctx = ctx_for("my_agent");

        let events = flow.run_one_step(&mut ctx).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].author, "my_agent");
        assert_eq!(events[0].invocation_id, "inv-1");
        assert_eq!(
            events[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("hi there")
        );
    }

    #[rusty_tokio::test]
    async fn run_one_step_increments_the_llm_call_count() {
        let flow = flow_with_response(LlmResponse::default());
        let mut ctx = ctx_for("my_agent");
        ctx.run_config = Some(RunConfig {
            max_llm_calls: 1,
            ..Default::default()
        });

        flow.run_one_step(&mut ctx).await.unwrap();
        let err = flow.run_one_step(&mut ctx).await.unwrap_err();
        assert!(matches!(
            err,
            LlmFlowError::InvocationContext(InvocationContextError::LlmCallsLimitExceeded(1))
        ));
    }

    #[rusty_tokio::test]
    async fn run_async_impl_drives_a_full_agent_run_through_base_agent() {
        let response = LlmResponse {
            content: Some(Content::new("model", vec![Part::text("via base agent")])),
            ..Default::default()
        };
        let flow = flow_with_response(response);
        let base_agent = BaseAgent::new("my_agent", flow).unwrap();
        let parent_ctx =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();

        let events = base_agent.run_async(&parent_ctx).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].content.as_ref().unwrap().parts[0].text.as_deref(),
            Some("via base agent")
        );
    }

    #[rusty_tokio::test]
    async fn run_live_impl_reports_not_implemented() {
        let flow = flow_with_response(LlmResponse::default());
        let mut ctx = ctx_for("my_agent");
        let err = AgentBehavior::run_live_impl(&flow, &mut ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("live mode isn't implemented"));
    }
}
