//! Capabilities C0101, C0103-C0104: `BaseLlm`, ported from
//! `google.adk.models.base_llm`.
//!
//! **Deferred**: C0102 (the dual-mode streaming contract itself —
//! `stream=False` yields one response, `stream=True` yields partials then a
//! final aggregate) is a behavioral contract every concrete backend must
//! satisfy; nothing implements `generate_content_async` for real yet (the
//! native Gemini backend is Phase 3 batch 2), so there is no concrete
//! streaming behavior to test — only the abstract signature (this file)
//! exists so far.

use std::future::Future;
use std::pin::Pin;

use adk_genai::content::Content;

use crate::base_llm_connection::{BaseLlmConnection, ConnectionError};
use crate::capabilities::{legacy_output_schema_and_tools, LlmCapabilities};
use crate::llm_request::LlmRequest;
use crate::llm_response::LlmResponse;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, rusty_err::Error)]
pub enum BaseLlmError {
    #[error("Async generation is not supported for {0}.")]
    GenerationNotSupported(String),
    #[error("Live connection is not supported for {0}.")]
    LiveConnectionNotSupported(String),
}

/// The base trait for an LLM. `model`/`type_name` back
/// [`BaseLlm::capabilities`]'s default (deprecated) name-based fallback —
/// see the module doc for why `generate_content_async`'s real streaming
/// behavior isn't exercised yet.
pub trait BaseLlm: Send + Sync {
    fn model(&self) -> &str;

    /// The Rust type name backing this model — the source's
    /// `type(self).__name__`, used only by the deprecated fallback's
    /// warning message.
    fn type_name(&self) -> &'static str;

    /// C0105: this model instance's capabilities. The default falls back to
    /// deprecated name-based detection ([`legacy_output_schema_and_tools`]);
    /// override to self-report instead.
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities {
            output_schema_and_tools: legacy_output_schema_and_tools(self.model(), self.type_name()),
        }
    }

    /// Regexes matching model names this class supports, for `LLMRegistry`.
    /// Defaults to none.
    fn supported_models() -> Vec<&'static str>
    where
        Self: Sized,
    {
        Vec::new()
    }

    /// C0101/C0102: generates content for a single model turn.
    /// `NotImplementedError`-equivalent by default; see the module doc for
    /// the streaming-contract deferral.
    fn generate_content_async<'a>(
        &'a self,
        _llm_request: &'a LlmRequest,
        _stream: bool,
    ) -> BoxFuture<'a, Result<Vec<LlmResponse>, BaseLlmError>> {
        let model = self.model().to_string();
        Box::pin(async move { Err(BaseLlmError::GenerationNotSupported(model)) })
    }

    /// C0103: creates a live connection to the LLM. `NotImplementedError`-
    /// equivalent by default; only `Gemini` (Phase 3 batch 2) overrides it.
    fn connect(
        &self,
        _llm_request: &LlmRequest,
    ) -> Result<Box<dyn BaseLlmConnection>, BaseLlmError> {
        Err(BaseLlmError::LiveConnectionNotSupported(
            self.model().to_string(),
        ))
    }
}

/// C0104: appends a user content so the model can continue to output —
/// standalone (doesn't need `&self`, matching the source's own method,
/// which never reads `self` either).
pub fn maybe_append_user_content(llm_request: &mut LlmRequest) {
    if llm_request.contents.is_empty() {
        llm_request.contents.push(Content::new(
            "user",
            vec![adk_genai::content::Part::text(
                "Handle the requests as specified in the System Instruction.",
            )],
        ));
        return;
    }

    let last_is_user = llm_request.contents.last().and_then(|c| c.role.as_deref()) == Some("user");
    if !last_is_user {
        llm_request.contents.push(Content::new(
            "user",
            vec![adk_genai::content::Part::text(
                "Continue processing previous requests as instructed. Exit or provide a \
                 summary if no more outputs are needed.",
            )],
        ));
    }
}

/// A live-connection abstract-contract error surfaces as [`ConnectionError`]
/// downstream — re-exported here so callers of `connect()` don't need to
/// import `base_llm_connection` separately just for the error type.
pub type LiveConnectionError = ConnectionError;

#[cfg(test)]
mod tests {
    use super::*;

    struct StubLlm {
        model: String,
    }

    impl BaseLlm for StubLlm {
        fn model(&self) -> &str {
            &self.model
        }

        fn type_name(&self) -> &'static str {
            "StubLlm"
        }
    }

    fn stub(model: &str) -> StubLlm {
        StubLlm {
            model: model.to_string(),
        }
    }

    #[rusty_tokio::test]
    async fn generate_content_async_is_unsupported_by_default() {
        let llm = stub("stub-model");
        let request = LlmRequest::new("stub-model");
        let err = llm
            .generate_content_async(&request, false)
            .await
            .unwrap_err();
        assert!(
            matches!(err, BaseLlmError::GenerationNotSupported(model) if model == "stub-model")
        );
    }

    #[test]
    fn connect_is_unsupported_by_default() {
        let llm = stub("stub-model");
        let request = LlmRequest::new("stub-model");
        match llm.connect(&request) {
            Err(BaseLlmError::LiveConnectionNotSupported(model)) => assert_eq!(model, "stub-model"),
            _ => panic!("expected LiveConnectionNotSupported"),
        }
    }

    #[test]
    fn supported_models_defaults_to_empty() {
        assert!(StubLlm::supported_models().is_empty());
    }

    #[test]
    fn maybe_append_user_content_adds_a_hint_turn_when_contents_are_empty() {
        let mut request = LlmRequest::new("m");
        maybe_append_user_content(&mut request);
        assert_eq!(request.contents.len(), 1);
        assert_eq!(request.contents[0].role.as_deref(), Some("user"));
    }

    #[test]
    fn maybe_append_user_content_appends_a_continue_turn_when_the_last_isnt_user() {
        let mut request = LlmRequest::new("m");
        request.contents.push(Content::new("model", vec![]));
        maybe_append_user_content(&mut request);
        assert_eq!(request.contents.len(), 2);
        assert_eq!(request.contents[1].role.as_deref(), Some("user"));
    }

    #[test]
    fn maybe_append_user_content_is_a_noop_when_the_last_turn_is_already_user() {
        let mut request = LlmRequest::new("m");
        request.contents.push(Content::user_text("hi"));
        maybe_append_user_content(&mut request);
        assert_eq!(request.contents.len(), 1);
    }
}
