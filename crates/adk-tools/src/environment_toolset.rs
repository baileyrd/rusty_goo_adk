//! Capability C0440: `EnvironmentToolset`, ported from
//! `google.adk.tools.environment._environment_toolset`.
//!
//! **`ENVIRONMENT_INSTRUCTION`/`DEFAULT_TIMEOUT` constants folded in**:
//! `tools/environment/_constants.py`'s `ENVIRONMENT_INSTRUCTION` lives
//! here (its only consumer); `DEFAULT_TIMEOUT`/`MAX_OUTPUT_CHARS` live in
//! `execute_tool.rs` (`MAX_OUTPUT_CHARS` re-exported `pub(crate)` for
//! `read_file_tool.rs`'s shared default) — no separate `constants`/
//! `utils`/`tools` module, since none of the three has more than one
//! real consumer in this port; `tools/environment/_tools.py` itself is a
//! pure backward-compatibility re-export shim in the source (no logic),
//! so it isn't ported at all.
//!
//! **Adaptation, disclosed**: [`crate::base_toolset::BaseToolset::get_tools`]/
//! `::process_llm_request` are infallible in this port's trait (already
//! shipped as C0403 — widening it to `Result`-returning here would be a
//! breaking change to an already-DONE public surface, out of scope for
//! this batch). The source's `await self._environment.initialize()`
//! inside both methods has no `try`/`except` either — an initialize
//! failure (e.g. a real IO error creating the working directory)
//! propagates as an uncaught exception. This port's equivalent is a
//! panic with a clear message — the same "uncaught exception becomes a
//! panic at an infallible trait boundary" translation, not a silently
//! swallowed error.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use adk_agents::readonly_context::ReadonlyContext;
use adk_models::llm_request::{Instructions, LlmRequest};

use crate::base_environment::BaseEnvironment;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::base_toolset::{BaseToolset, PrefixCache};
use crate::edit_file_tool::EditFileTool;
use crate::execute_tool::ExecuteTool;
use crate::read_file_tool::ReadFileTool;
use crate::tool_context::ToolContext;
use crate::write_file_tool::WriteFileTool;

const ENVIRONMENT_INSTRUCTION: &str = "\
Your environment is at {working_dir}/

# Environment Rules

DO:
- Chain sequential, dependent commands with `&&` in a single `Execute` call
- To read existing files, always use the `ReadFile` tool. Use `EditFile` to modify existing files.

DON'T:
- Use `Execute` to run cat, head, or tail when `ReadFile` tools can do the job
- Combine `EditFile` or `ReadFile` with `Execute` in the same response (Instead, call the file tool first, then `Execute` in the next turn)
- Use multiple `Execute` calls for dependent commands (they run in parallel)
";

/// C0440: bundles `Execute`/`ReadFile`/`EditFile`/`WriteFile` tools bound
/// to an injected [`BaseEnvironment`], and injects an environment-level
/// system instruction on each LLM call.
pub struct EnvironmentToolset {
    environment: Arc<dyn BaseEnvironment>,
    max_output_chars: Option<usize>,
    environment_initialized: AtomicBool,
    prefix_cache: Mutex<PrefixCache>,
}

impl EnvironmentToolset {
    pub fn new(environment: Arc<dyn BaseEnvironment>, max_output_chars: Option<usize>) -> Self {
        Self {
            environment,
            max_output_chars,
            environment_initialized: AtomicBool::new(false),
            prefix_cache: Mutex::new(PrefixCache::new()),
        }
    }

    async fn ensure_environment_initialized(&self) {
        if !self.environment_initialized.load(Ordering::SeqCst) {
            self.environment
                .initialize()
                .await
                .expect("EnvironmentToolset: failed to initialize environment");
            self.environment_initialized.store(true, Ordering::SeqCst);
        }
    }
}

impl BaseToolset for EnvironmentToolset {
    fn get_tools<'a>(
        &'a self,
        _readonly_context: Option<&'a ReadonlyContext>,
    ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
        Box::pin(async move {
            self.ensure_environment_initialized().await;
            vec![
                Arc::new(ExecuteTool::new(
                    self.environment.clone(),
                    self.max_output_chars,
                )) as Arc<dyn BaseTool>,
                Arc::new(ReadFileTool::new(
                    self.environment.clone(),
                    self.max_output_chars,
                )) as Arc<dyn BaseTool>,
                Arc::new(EditFileTool::new(self.environment.clone())) as Arc<dyn BaseTool>,
                Arc::new(WriteFileTool::new(self.environment.clone())) as Arc<dyn BaseTool>,
            ]
        })
    }

    fn prefix_cache(&self) -> &Mutex<PrefixCache> {
        &self.prefix_cache
    }

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.ensure_environment_initialized().await;
            let working_dir = self
                .environment
                .working_dir()
                .expect("EnvironmentToolset: working_dir unset after initialize()");
            let instruction = ENVIRONMENT_INSTRUCTION
                .replace("{working_dir}", &working_dir.display().to_string());
            llm_request.append_instructions(Instructions::Strings(vec![instruction]));
        })
    }

    fn close<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if self.environment_initialized.load(Ordering::SeqCst) {
                self.environment.close().await;
                self.environment_initialized.store(false, Ordering::SeqCst);
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_environment::LocalEnvironment;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn readonly_context() -> ReadonlyContext {
        ReadonlyContext::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn get_tools_returns_all_four_tools_and_initializes_the_environment() {
        let toolset = EnvironmentToolset::new(Arc::new(LocalEnvironment::new()), None);
        let ctx = readonly_context();
        let tools = toolset.get_tools(Some(&ctx)).await;
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, vec!["Execute", "ReadFile", "EditFile", "WriteFile"]);
        assert!(toolset.environment.is_initialized());
    }

    #[rusty_tokio::test]
    async fn process_llm_request_injects_the_working_dir_instruction() {
        let toolset = EnvironmentToolset::new(Arc::new(LocalEnvironment::new()), None);
        let mut ctx = adk_agents::context::Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        let mut request = LlmRequest::default();
        toolset.process_llm_request(&mut ctx, &mut request).await;
        let working_dir = toolset.environment.working_dir().unwrap();
        let system_instruction = request.config.system_instruction.unwrap();
        assert!(system_instruction.contains(&working_dir.display().to_string()));
        assert!(system_instruction.contains("Execute"));
        assert!(toolset.environment.is_initialized());
    }

    #[rusty_tokio::test]
    async fn close_releases_the_environment_once_initialized() {
        let toolset = EnvironmentToolset::new(Arc::new(LocalEnvironment::new()), None);
        let ctx = readonly_context();
        toolset.get_tools(Some(&ctx)).await;
        assert!(toolset.environment.is_initialized());
        toolset.close().await;
        assert!(!toolset.environment.is_initialized());
    }
}
