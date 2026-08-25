//! Capability C0383: `BaseCodeExecutor`, ported from
//! `google.adk.code_executors.base_code_executor`.
//!
//! **Adaptation**: the source is a Pydantic `BaseModel` — its 6 config
//! fields are inherited, mutable attributes on every concrete subclass,
//! and `execute_code` is an `@abc.abstractmethod`. Rust traits can't
//! declare data fields, so this port splits the two: [`CodeExecutorConfig`]
//! holds the 6 fields (with the source's own defaults via its `Default`
//! impl) for a concrete executor to embed and expose via
//! [`BaseCodeExecutor::config`]; [`BaseCodeExecutor::execute_code`] is
//! the trait's one required method.

use crate::code_execution_utils::{CodeExecutionInput, CodeExecutionResult};
use adk_agents::invocation_context::InvocationContext;

/// Trait-object-safe `as_any` — the same mechanism
/// `adk-agents::base_agent::AsAny`/`adk-models::base_llm::AsAny` already
/// established for downcasting a type-erased trait object back onto a
/// concrete implementor. Needed here so `adk-flows::code_execution` can
/// detect whether a resolved `&dyn BaseCodeExecutor` is actually a
/// `BuiltInCodeExecutor` (its request/response handling is a completely
/// different branch from a general executor, matching the source's own
/// `isinstance(code_executor, BuiltInCodeExecutor)` checks) without
/// `adk-tools` needing to know about `adk-flows` or vice versa. Purely
/// additive: every existing `BaseCodeExecutor` implementor needs no
/// changes, since `AsAny` is blanket-implemented for every `'static` type.
pub trait AsAny: std::any::Any {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: std::any::Any> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// C0383: `BaseCodeExecutor`'s 6 config attributes, with the source's
/// own defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeExecutorConfig {
    /// If true, extract and process data files from the model request
    /// and attach them to the code executor. Supported data file MIME
    /// types are `text/csv`.
    pub optimize_data_file: bool,
    /// Whether the code executor is stateful.
    pub stateful: bool,
    /// The number of attempts to retry on consecutive code execution
    /// errors.
    pub error_retry_attempts: u32,
    /// The list of enclosing delimiters that identify code blocks.
    pub code_block_delimiters: Vec<(String, String)>,
    /// The delimiters used to format the code execution result.
    pub execution_result_delimiters: (String, String),
    /// The fallback timeout in seconds for the code execution.
    pub timeout_seconds: Option<u64>,
}

impl Default for CodeExecutorConfig {
    fn default() -> Self {
        Self {
            optimize_data_file: false,
            stateful: false,
            error_retry_attempts: 2,
            code_block_delimiters: vec![
                ("```tool_code\n".to_string(), "\n```".to_string()),
                ("```python\n".to_string(), "\n```".to_string()),
            ],
            execution_result_delimiters: ("```tool_output\n".to_string(), "\n```".to_string()),
            timeout_seconds: None,
        }
    }
}

/// C0383: `code_executors.base_code_executor.BaseCodeExecutor` — the
/// abstract interface every concrete code executor implements.
pub trait BaseCodeExecutor: AsAny {
    /// The executor's shared config attributes. See [`CodeExecutorConfig`].
    fn config(&self) -> &CodeExecutorConfig;

    /// Executes `code_execution_input.code` and returns the result.
    fn execute_code(
        &self,
        invocation_context: &InvocationContext,
        code_execution_input: &CodeExecutionInput,
    ) -> CodeExecutionResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_matches_the_source() {
        let config = CodeExecutorConfig::default();
        assert!(!config.optimize_data_file);
        assert!(!config.stateful);
        assert_eq!(config.error_retry_attempts, 2);
        assert_eq!(
            config.code_block_delimiters,
            vec![
                ("```tool_code\n".to_string(), "\n```".to_string()),
                ("```python\n".to_string(), "\n```".to_string()),
            ]
        );
        assert_eq!(
            config.execution_result_delimiters,
            ("```tool_output\n".to_string(), "\n```".to_string())
        );
        assert_eq!(config.timeout_seconds, None);
    }

    struct EchoExecutor {
        config: CodeExecutorConfig,
    }

    impl BaseCodeExecutor for EchoExecutor {
        fn config(&self) -> &CodeExecutorConfig {
            &self.config
        }

        fn execute_code(
            &self,
            _invocation_context: &InvocationContext,
            code_execution_input: &CodeExecutionInput,
        ) -> CodeExecutionResult {
            CodeExecutionResult {
                stdout: code_execution_input.code.clone(),
                ..Default::default()
            }
        }
    }

    #[test]
    fn a_concrete_executor_can_implement_the_trait() {
        let executor = EchoExecutor {
            config: CodeExecutorConfig::default(),
        };
        let session = adk_agents::session::Session::new("app", "user", "s1");
        let ctx =
            adk_agents::invocation_context::InvocationContextBuilder::new("inv-1", session).build();
        let input = CodeExecutionInput {
            code: "print(1)".to_string(),
            ..Default::default()
        };
        let result = executor.execute_code(&ctx, &input);
        assert_eq!(result.stdout, "print(1)");
    }
}
