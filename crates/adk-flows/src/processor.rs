//! Capability C0147: `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor`,
//! ported from `google.adk.flows.llm_flows._base_llm_processor`.
//!
//! **Adaptation**: the source methods are `async def run_async(...) ->
//! AsyncGenerator[Event, None]`. Rust has no native async-generator
//! equivalent, so — matching the same adaptation `BaseLlm::generate_content_async`
//! and `BaseLlmConnection` already made in Phase 3 — `run_async` returns a
//! boxed future resolving to `Result<Vec<Event>, ProcessorError>`: every
//! event the processor would have yielded, collected in order.

use std::future::Future;
use std::pin::Pin;

use adk_agents::invocation_context::InvocationContext;
use adk_events::Event;
use adk_models::llm_request::LlmRequest;
use adk_models::llm_response::LlmResponse;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum ProcessorError {
    #[error("{0}")]
    Failed(String),
}

/// C0147: `BaseLlmRequestProcessor` — every named request processor
/// (`basic`, `identity`, `instructions`, `agent_transfer`, ...) implements
/// this. Mutates `llm_request` in place, matching the source.
pub trait BaseLlmRequestProcessor: Send + Sync {
    fn run_async<'a>(
        &'a self,
        invocation_context: &'a mut InvocationContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, Result<Vec<Event>, ProcessorError>>;
}

/// C0147: `BaseLlmResponseProcessor` — every named response processor
/// (`nl_planning`, `code_execution`, ...) implements this. Mutates
/// `llm_response` in place, matching the source.
pub trait BaseLlmResponseProcessor: Send + Sync {
    fn run_async<'a>(
        &'a self,
        invocation_context: &'a mut InvocationContext,
        llm_response: &'a mut LlmResponse,
    ) -> BoxFuture<'a, Result<Vec<Event>, ProcessorError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    struct NoOpRequestProcessor;

    impl BaseLlmRequestProcessor for NoOpRequestProcessor {
        fn run_async<'a>(
            &'a self,
            _invocation_context: &'a mut InvocationContext,
            llm_request: &'a mut LlmRequest,
        ) -> BoxFuture<'a, Result<Vec<Event>, ProcessorError>> {
            Box::pin(async move {
                llm_request.model = Some("touched-by-processor".to_string());
                Ok(Vec::new())
            })
        }
    }

    struct FailingResponseProcessor;

    impl BaseLlmResponseProcessor for FailingResponseProcessor {
        fn run_async<'a>(
            &'a self,
            _invocation_context: &'a mut InvocationContext,
            _llm_response: &'a mut LlmResponse,
        ) -> BoxFuture<'a, Result<Vec<Event>, ProcessorError>> {
            Box::pin(async move { Err(ProcessorError::Failed("boom".to_string())) })
        }
    }

    #[rusty_tokio::test]
    async fn a_request_processor_can_mutate_the_request_and_yield_no_events() {
        let mut ctx =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let mut request = LlmRequest::new("original-model");
        let events = NoOpRequestProcessor
            .run_async(&mut ctx, &mut request)
            .await
            .unwrap();
        assert!(events.is_empty());
        assert_eq!(request.model.as_deref(), Some("touched-by-processor"));
    }

    #[rusty_tokio::test]
    async fn a_response_processor_can_fail() {
        let mut ctx =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let mut response = LlmResponse::default();
        let err = FailingResponseProcessor
            .run_async(&mut ctx, &mut response)
            .await
            .unwrap_err();
        assert!(matches!(err, ProcessorError::Failed(message) if message == "boom"));
    }
}
