//! C0642: `optimization._gepa_utils`, ported from
//! `google.adk.optimization._gepa_utils`.
//!
//! Shared preconditions and the reflection-model call helper both GEPA
//! optimizers (C0640/C0641) will need — those optimizers themselves stay
//! deferred, since they need the optional `gepa` third-party package this
//! port has no reason to add yet (a new-dependency stop-and-ask trigger);
//! this pure, dependency-free helper pair doesn't.
//!
//! **`GEPAPrompt`, narrowed to `&str`**: the source's `GEPAPrompt: TypeAlias
//! = str | list[dict[str, Any]]` is GEPA's own public `LanguageModel`
//! input contract, but `generate_reflection_response` itself immediately
//! does `cast(str, prompt)` — i.e. only the `str` case is ever actually
//! handled at this layer, the `list[dict]` half is a type-checker fiction
//! for callers this port doesn't have yet. [`generate_reflection_response`]
//! takes `&str` directly rather than introducing an unused enum for the
//! other half.
//!
//! **`require_static_instruction`, takes `&Instruction` not `&LlmAgent`**:
//! the source's `require_static_instruction(agent: Agent)` only ever
//! reads `agent.instruction` — narrowed here to the sub-shape actually
//! used, avoiding an unused `&LlmAgent` parameter.

use adk_agents::llm_agent::Instruction;
use adk_genai::content::{Content, Part};
use adk_models::base_llm::{BaseLlm, BaseLlmError};
use adk_models::llm_request::{GenerateContentConfigStub, LlmRequest};

const REFLECTION_PROMPT_AUTHOR: &str = "user";

/// `_gepa_utils.require_static_instruction` — see the module doc for the
/// `&Instruction` narrowing.
pub fn require_static_instruction(instruction: &Instruction) -> Result<String, String> {
    match instruction {
        Instruction::Static(s) => Ok(s.clone()),
        Instruction::Provider(_) => Err(
            "GEPA optimization requires initial_agent.instruction to be a static string; \
             request-scoped instruction providers cannot be resolved without an invocation \
             context."
                .to_string(),
        ),
    }
}

/// `_gepa_utils.generate_reflection_response` — runs one GEPA reflection
/// request and returns all non-thought text. See the module doc for the
/// `GEPAPrompt` narrowing.
pub async fn generate_reflection_response(
    llm: &dyn BaseLlm,
    model: impl Into<String>,
    config: GenerateContentConfigStub,
    prompt: &str,
) -> Result<String, BaseLlmError> {
    let mut llm_request = LlmRequest::new(model);
    llm_request.config = config;
    llm_request.contents = vec![Content::new(
        REFLECTION_PROMPT_AUTHOR,
        vec![Part::text(prompt)],
    )];

    let responses = llm.generate_content_async(&llm_request, false).await?;
    // Only one yield expected, matching the source's single `__anext__()`.
    let Some(response) = responses.into_iter().next() else {
        return Ok(String::new());
    };
    let Some(content) = response.content else {
        return Ok(String::new());
    };
    Ok(content
        .parts
        .into_iter()
        .filter_map(|part| {
            if part.thought == Some(true) {
                return None;
            }
            part.text.filter(|text| !text.is_empty())
        })
        .collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_models::llm_response::LlmResponse;
    use std::future::Future;
    use std::pin::Pin;

    #[test]
    fn require_static_instruction_returns_the_static_string() {
        let instruction = Instruction::Static("be helpful".to_string());
        assert_eq!(
            require_static_instruction(&instruction),
            Ok("be helpful".to_string())
        );
    }

    #[test]
    fn require_static_instruction_rejects_a_provider() {
        let instruction = Instruction::Provider(std::sync::Arc::new(|_ctx| "dynamic".to_string()));
        assert!(require_static_instruction(&instruction).is_err());
    }

    struct FixedResponseLlm {
        model: String,
        responses: Vec<LlmResponse>,
    }

    impl BaseLlm for FixedResponseLlm {
        fn model(&self) -> &str {
            &self.model
        }
        fn type_name(&self) -> &'static str {
            "FixedResponseLlm"
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

    #[rusty_tokio::test]
    async fn generate_reflection_response_concatenates_non_thought_text() {
        let mut thought_part = Part::text("(thinking...)");
        thought_part.thought = Some(true);
        let llm = FixedResponseLlm {
            model: "gemini-2.5-flash".to_string(),
            responses: vec![LlmResponse {
                content: Some(Content::new(
                    "model",
                    vec![thought_part, Part::text("final answer")],
                )),
                ..Default::default()
            }],
        };

        let result = generate_reflection_response(
            &llm,
            "gemini-2.5-flash",
            GenerateContentConfigStub::default(),
            "reflect on this",
        )
        .await
        .unwrap();

        assert_eq!(result, "final answer");
    }

    #[rusty_tokio::test]
    async fn generate_reflection_response_returns_empty_string_with_no_responses() {
        let llm = FixedResponseLlm {
            model: "gemini-2.5-flash".to_string(),
            responses: vec![],
        };

        let result = generate_reflection_response(
            &llm,
            "gemini-2.5-flash",
            GenerateContentConfigStub::default(),
            "reflect on this",
        )
        .await
        .unwrap();

        assert_eq!(result, "");
    }
}
