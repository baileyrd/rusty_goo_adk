//! Capability C0948: `BaseEnvironment`/`ExecutionResult`, ported from
//! `google.adk.environment._base_environment`.
//!
//! **Inventory gap, discovered this batch**: `environment/` is a
//! top-level source module with no manifest row at all prior to this
//! batch — the same shape of gap C0947 (`llm_as_judge_utils`) was, found
//! by a background scoping agent grepping the manifest for
//! `BaseEnvironment` and finding nothing, despite four existing rows
//! (C0410, C0440, C0775, C0776) already referencing it as a dependency.
//! Added as C0948 (this file) and C0949 (`local_environment`).
//!
//! **Adaptation**: same "class → trait, instance attributes → methods"
//! shape as [`crate::base_tool::BaseTool`]/[`crate::base_toolset::BaseToolset`]
//! before it. The source's `_is_initialized` class-level attribute (with
//! a property getter/setter every subclass inherits for free) has no
//! Rust equivalent — traits carry no data — so [`BaseEnvironment::is_initialized`]
//! is a required method instead; a concrete implementor backs it with its
//! own interior-mutable state (see `local_environment`'s `AtomicBool`).
//! This property genuinely matters, not just documentation: `skill_toolset.py`
//! (not built this batch) reads `self._env.is_initialized` before
//! deciding whether to call `initialize()`, and the not-yet-built
//! Daytona/E2B environments (C0775/C0776) set it directly too.
//!
//! **Adaptation**: `initialize()`/`close()` return `BoxFuture<'_, ()>`/
//! `BoxFuture<'_, ()>` by default (no-op), same as `BaseToolset::close`'s
//! own default — except `initialize()` here returns
//! `Result<(), EnvironmentError>` even in the trivial default case, since
//! a real implementor's `initialize()` (`LocalEnvironment`'s `os::create_dir_all`)
//! can genuinely fail on IO — the source lets that exception propagate
//! uncaught; this port surfaces it as an explicit `Result` instead, the
//! same "propagate via `Result` where Python propagates via an uncaught
//! exception" translation used throughout this port.
//!
//! **Adaptation**: `write_file(path, content: str | bytes)` collapses to
//! `write_file(path, content: &[u8])` — not a narrowing. The source's str
//! branch opens with `encoding='utf-8', newline=''`, which disables
//! newline translation entirely, so `content.encode('utf-8')` (str
//! branch) and the raw bytes (bytes branch) always produce byte-identical
//! output; a caller with a `String` just calls `.as_bytes()`.
//!
//! **Adaptation**: `path: Path` narrows to `&str` — every real caller in
//! this batch (`ExecuteTool`/`ReadFileTool`/`EditFileTool`/`WriteFileTool`,
//! all driven by JSON tool-call args) only ever has a path as a string,
//! matching `LocalEnvironment`'s own override signature (`path: str | Path`).

use std::path::PathBuf;
use std::time::Duration;

use crate::base_tool::BoxFuture;

/// `_base_environment.ExecutionResult` — the result of a shell command
/// execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// The error cases `BaseEnvironment`'s methods can surface — see the
/// module doc on why these are an explicit `Result` rather than a
/// propagating exception.
#[derive(Debug, Clone, PartialEq, Eq, rusty_err::Error)]
pub enum EnvironmentError {
    #[error("`working_dir` is not set. Call initialize() first.")]
    NotInitialized,
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Path escapes working directory: {0}")]
    PathEscapesWorkingDir(String),
    #[error("{0}")]
    Io(String),
}

/// C0948: the base trait for code execution environments. An environment
/// provides the ability to execute shell commands, read files, and write
/// files within a working directory. See the module doc for the
/// attribute-to-method and error-propagation adaptations.
///
/// Lifecycle: construct, call [`BaseEnvironment::initialize`] before
/// first use, use `execute`/`read_file`/`write_file`, call
/// [`BaseEnvironment::close`] when done.
pub trait BaseEnvironment: Send + Sync {
    /// Whether the environment has been initialized.
    fn is_initialized(&self) -> bool;

    /// The absolute path to the environment's working directory.
    fn working_dir(&self) -> Result<PathBuf, EnvironmentError>;

    /// Initialize the environment (e.g. create the working directory).
    /// Called before first use. The default implementation is a no-op.
    /// Implementors should ensure this method is idempotent.
    fn initialize(&self) -> BoxFuture<'_, Result<(), EnvironmentError>> {
        Box::pin(async { Ok(()) })
    }

    /// Release resources held by the environment. Called when the
    /// environment is no longer needed. The default implementation is a
    /// no-op. Implementors should ensure this method is idempotent.
    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    /// Execute a shell command in the working directory. `timeout` is the
    /// maximum execution time; `None` means no limit.
    fn execute<'a>(
        &'a self,
        command: &'a str,
        timeout: Option<Duration>,
    ) -> BoxFuture<'a, Result<ExecutionResult, EnvironmentError>>;

    /// Read a file from the environment filesystem. `path` is absolute or
    /// working-dir-relative.
    fn read_file<'a>(&'a self, path: &'a str) -> BoxFuture<'a, Result<Vec<u8>, EnvironmentError>>;

    /// Write `content` to a file in the environment's filesystem, creating
    /// parent directories automatically if they don't exist.
    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: &'a [u8],
    ) -> BoxFuture<'a, Result<(), EnvironmentError>>;
}
