//! Capability C0384: `BuiltInCodeExecutor`, ported from
//! `google.adk.code_executors.built_in_code_executor`.
//!
//! **`execute_code`, adapted**: the source's override actually returns
//! `None` at runtime — "execution is delegated to the model, so there is
//! nothing to run here" — even though its own declared return type is
//! the non-`Optional` `CodeExecutionResult` (the source even marks the
//! method `# type: ignore[empty-body]`, acknowledging this mismatch
//! itself rather than fixing it). This port's `BaseCodeExecutor::execute_code`
//! trait method keeps the honest, non-`Option` contract every *other*
//! executor (e.g. a future `UnsafeLocalCodeExecutor`) truly satisfies,
//! rather than widening it to `Option<CodeExecutionResult>` for every
//! implementor just to accommodate this one delegate-to-model case — so
//! `BuiltInCodeExecutor::execute_code` here returns
//! `CodeExecutionResult::default()` (empty stdout/stderr/output_files)
//! as the closest-fitting sentinel for "nothing to run here", a
//! disclosed adaptation rather than a claimed exact match to the
//! source's `None`.

use adk_agents::invocation_context::InvocationContext;
use adk_models::capabilities::is_gemini_model;
use adk_models::llm_request::LlmRequest;

use crate::append_tools::append_built_in_tool_marker;
use crate::base_code_executor::{BaseCodeExecutor, CodeExecutorConfig};
use crate::code_execution_utils::{CodeExecutionInput, CodeExecutionResult};
use crate::model_name_utils::is_gemini_model_id_check_disabled;

/// C0384: `BuiltInCodeExecutor` — a code executor that uses the model's
/// own built-in code execution tool. Currently only supports Gemini
/// models.
#[derive(Default)]
pub struct BuiltInCodeExecutor {
    config: CodeExecutorConfig,
}

impl BuiltInCodeExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    /// `BuiltInCodeExecutor.process_llm_request` — appends the Gemini
    /// code-execution tool marker for a Gemini-compatible model. Returns
    /// `Err` for an unsupported model, matching the source's `raise
    /// ValueError`.
    pub fn process_llm_request(&self, llm_request: &mut LlmRequest) -> Result<(), String> {
        if is_gemini_model(llm_request.model.as_deref()) || is_gemini_model_id_check_disabled() {
            append_built_in_tool_marker(llm_request, "codeExecution");
            return Ok(());
        }
        Err(format!(
            "Gemini code execution tool is not supported for model {:?}",
            llm_request.model
        ))
    }
}

impl BaseCodeExecutor for BuiltInCodeExecutor {
    fn config(&self) -> &CodeExecutorConfig {
        &self.config
    }

    fn execute_code(
        &self,
        _invocation_context: &InvocationContext,
        _code_execution_input: &CodeExecutionInput,
    ) -> CodeExecutionResult {
        // Execution is delegated to the model -- see the module doc.
        CodeExecutionResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_llm_request_appends_the_code_execution_marker_for_a_gemini_model() {
        let executor = BuiltInCodeExecutor::new();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        executor.process_llm_request(&mut request).unwrap();
        let tools = request.config.tools.unwrap();
        let entries = tools.as_seq().unwrap();
        assert!(entries
            .iter()
            .any(|entry| entry.get("codeExecution").is_some()));
    }

    #[test]
    fn process_llm_request_errors_for_a_non_gemini_model() {
        let executor = BuiltInCodeExecutor::new();
        let mut request = LlmRequest::new("gpt-4");
        let result = executor.process_llm_request(&mut request);
        assert!(result.is_err());
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn execute_code_returns_an_empty_result() {
        let executor = BuiltInCodeExecutor::new();
        let session = adk_agents::session::Session::new("app", "user", "s1");
        let ctx =
            adk_agents::invocation_context::InvocationContextBuilder::new("inv-1", session).build();
        let result = executor.execute_code(&ctx, &CodeExecutionInput::default());
        assert_eq!(result, CodeExecutionResult::default());
    }
}
