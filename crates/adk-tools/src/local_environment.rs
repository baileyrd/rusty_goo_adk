//! Capability C0949: `LocalEnvironment`, ported from
//! `google.adk.environment._local_environment`.
//!
//! **Adaptation, interior mutability**: [`crate::base_environment::BaseEnvironment`]'s
//! methods take `&self` (needed so `Arc<dyn BaseEnvironment>` can be
//! shared across the four tools `EnvironmentToolset` hands out, all
//! wrapping the same environment instance, matching the source's normal
//! by-reference sharing). Python's plain instance-attribute mutation
//! (`self._working_dir = ...`) becomes a `Mutex<Option<PathBuf>>` +
//! `AtomicBool`s here instead.
//!
//! **Divergence, disclosed**: `execute` shells out via `sh -c <command>`
//! (matching the source's `asyncio.create_subprocess_shell`), not the
//! allowlist-checked `execve`-style argv path `bash_tool.rs`'s
//! `ExecuteBashTool` (C0418) uses — a different, lower-guardrail tool
//! than that one, with no confirmation gate and no `BashToolPolicy`
//! equivalent, matching the source exactly (no such policy exists on
//! `LocalEnvironment` either).
//!
//! **Disclosed narrowing (same one already established in `bash_tool.rs`)**:
//! the source re-invokes `proc.communicate()` after killing a timed-out
//! process, to drain whatever stdout/stderr was buffered before the
//! kill. This port's `Child::wait_with_output` consumes the child as one
//! unit — `Command::kill_on_drop(true)` still reliably kills the process
//! when the timed-out future is dropped, but the response carries no
//! partial output, same disclosed gap as `bash_tool.rs`.
//!
//! **`_resolve_path`, lexical, not `Path.resolve()`-based**: same "path
//! safety by construction, not by canonicalize" adaptation already
//! established in `file_artifact_service.rs` (C0268-C0269) — a candidate
//! path here can legitimately not exist yet (`write_file` creating a new
//! file), so `std::fs::canonicalize` (which requires the full path to
//! already exist) can't be used the way the source's `Path.resolve()`
//! can. [`resolve_path`] instead lexically normalizes `..`/`.` segments
//! (popping a preceding normal component for `..`, matching
//! `os.path.normpath`'s algorithm) without touching the filesystem, then
//! checks the result still starts with the (also lexically normalized)
//! working directory. Disclosed divergence: this doesn't follow a
//! symlink partway through the working directory the way the source's
//! real filesystem resolution would — only the traversal-prevention
//! property is replicated, not exact symlink-canonicalization behavior.

use std::collections::BTreeMap;
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use rusty_tokio::process::{Command, Stdio};

use crate::base_environment::{BaseEnvironment, EnvironmentError, ExecutionResult};
use crate::base_tool::BoxFuture;

/// Lexically normalizes `path` (`os.path.normpath`-equivalent): drops
/// `.` segments, and pops the preceding normal component for a `..`
/// segment rather than leaving it in place. Never touches the
/// filesystem.
fn lexically_normalize(path: &Path) -> PathBuf {
    let mut stack: Vec<Component> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(stack.last(), Some(Component::Normal(_))) {
                    stack.pop();
                } else {
                    stack.push(component);
                }
            }
            other => stack.push(other),
        }
    }
    stack.into_iter().collect()
}

/// Resolves `path` inside `working_dir` — see the module doc.
fn resolve_path(path: &str, working_dir: &Path) -> Result<PathBuf, EnvironmentError> {
    let candidate = Path::new(path);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        working_dir.join(candidate)
    };
    let resolved = lexically_normalize(&joined);
    let normalized_working_dir = lexically_normalize(working_dir);
    if !resolved.starts_with(&normalized_working_dir) {
        return Err(EnvironmentError::PathEscapesWorkingDir(path.to_string()));
    }
    Ok(resolved)
}

fn exit_code_from_status(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        code
    } else if let Some(signal) = status.signal() {
        -signal
    } else {
        0
    }
}

/// C0949: executes commands via local `rusty_tokio` subprocesses. When
/// `working_dir` is not specified at construction, a temporary directory
/// is created on [`BaseEnvironment::initialize`] and removed on
/// [`BaseEnvironment::close`].
pub struct LocalEnvironment {
    working_dir: Mutex<Option<PathBuf>>,
    env_vars: Option<BTreeMap<String, String>>,
    auto_created: AtomicBool,
    is_initialized: AtomicBool,
}

impl LocalEnvironment {
    /// A temporary directory is created during `initialize()`.
    pub fn new() -> Self {
        Self::with_options(None, None)
    }

    pub fn with_options(
        working_dir: Option<PathBuf>,
        env_vars: Option<BTreeMap<String, String>>,
    ) -> Self {
        Self {
            working_dir: Mutex::new(working_dir),
            env_vars,
            auto_created: AtomicBool::new(false),
            is_initialized: AtomicBool::new(false),
        }
    }
}

impl Default for LocalEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseEnvironment for LocalEnvironment {
    fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::SeqCst)
    }

    fn working_dir(&self) -> Result<PathBuf, EnvironmentError> {
        self.working_dir
            .lock()
            .unwrap()
            .clone()
            .ok_or(EnvironmentError::NotInitialized)
    }

    fn initialize(&self) -> BoxFuture<'_, Result<(), EnvironmentError>> {
        Box::pin(async move {
            let mut working_dir = self.working_dir.lock().unwrap();
            if working_dir.is_none() {
                let path = std::env::temp_dir()
                    .join(format!("adk_workspace_{}", adk_platform::uuid::new_uuid()));
                std::fs::create_dir_all(&path)
                    .map_err(|err| EnvironmentError::Io(err.to_string()))?;
                *working_dir = Some(path);
                self.auto_created.store(true, Ordering::SeqCst);
            } else {
                let path = working_dir.as_ref().unwrap();
                std::fs::create_dir_all(path)
                    .map_err(|err| EnvironmentError::Io(err.to_string()))?;
            }
            self.is_initialized.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if self.auto_created.load(Ordering::SeqCst) {
                let mut working_dir = self.working_dir.lock().unwrap();
                if let Some(path) = working_dir.take() {
                    let _ = std::fs::remove_dir_all(&path);
                }
            }
            self.is_initialized.store(false, Ordering::SeqCst);
        })
    }

    fn execute<'a>(
        &'a self,
        command: &'a str,
        timeout: Option<Duration>,
    ) -> BoxFuture<'a, Result<ExecutionResult, EnvironmentError>> {
        Box::pin(async move {
            let working_dir = self.working_dir()?;

            let mut cmd = Command::new("sh");
            cmd.arg("-c").arg(command);
            cmd.current_dir(&working_dir);
            if let Some(env_vars) = &self.env_vars {
                cmd.envs(env_vars);
            }
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
            cmd.kill_on_drop(true);

            let child = cmd
                .spawn()
                .map_err(|err| EnvironmentError::Io(err.to_string()))?;
            let output_future = child.wait_with_output();

            let output = match timeout {
                Some(duration) => match rusty_tokio::time::timeout(duration, output_future).await {
                    Ok(output) => output,
                    Err(_) => {
                        return Ok(ExecutionResult {
                            timed_out: true,
                            ..Default::default()
                        });
                    }
                },
                None => output_future.await,
            };
            let output = output.map_err(|err| EnvironmentError::Io(err.to_string()))?;

            Ok(ExecutionResult {
                exit_code: exit_code_from_status(output.status),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                timed_out: false,
            })
        })
    }

    fn read_file<'a>(&'a self, path: &'a str) -> BoxFuture<'a, Result<Vec<u8>, EnvironmentError>> {
        Box::pin(async move {
            let working_dir = self.working_dir()?;
            let resolved = resolve_path(path, &working_dir)?;
            match rusty_tokio::spawn_blocking(move || {
                std::fs::read(&resolved).map_err(|err| {
                    if err.kind() == std::io::ErrorKind::NotFound {
                        EnvironmentError::FileNotFound(resolved.display().to_string())
                    } else {
                        EnvironmentError::Io(err.to_string())
                    }
                })
            })
            .await
            {
                Ok(inner) => inner,
                Err(join_error) => Err(EnvironmentError::Io(join_error.to_string())),
            }
        })
    }

    fn write_file<'a>(
        &'a self,
        path: &'a str,
        content: &'a [u8],
    ) -> BoxFuture<'a, Result<(), EnvironmentError>> {
        Box::pin(async move {
            let working_dir = self.working_dir()?;
            let resolved = resolve_path(path, &working_dir)?;
            let content = content.to_vec();
            match rusty_tokio::spawn_blocking(move || {
                if let Some(parent) = resolved.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|err| EnvironmentError::Io(err.to_string()))?;
                }
                std::fs::write(&resolved, &content)
                    .map_err(|err| EnvironmentError::Io(err.to_string()))
            })
            .await
            {
                Ok(inner) => inner,
                Err(join_error) => Err(EnvironmentError::Io(join_error.to_string())),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_workspace() -> PathBuf {
        std::env::temp_dir().join(format!(
            "adk_local_env_test_{}",
            adk_platform::uuid::new_uuid()
        ))
    }

    #[test]
    fn lexically_normalize_pops_parent_dir_segments() {
        assert_eq!(
            lexically_normalize(Path::new("/a/b/../c")),
            PathBuf::from("/a/c")
        );
        assert_eq!(
            lexically_normalize(Path::new("/a/./b")),
            PathBuf::from("/a/b")
        );
    }

    #[test]
    fn resolve_path_allows_a_path_that_stays_inside_the_working_dir() {
        let working_dir = PathBuf::from("/workspace");
        assert_eq!(
            resolve_path("sub/../file.txt", &working_dir).unwrap(),
            PathBuf::from("/workspace/file.txt")
        );
    }

    #[test]
    fn resolve_path_rejects_an_escape() {
        let working_dir = PathBuf::from("/workspace");
        assert!(matches!(
            resolve_path("../outside.txt", &working_dir),
            Err(EnvironmentError::PathEscapesWorkingDir(_))
        ));
    }

    #[rusty_tokio::test]
    async fn working_dir_errors_before_initialize() {
        let env = LocalEnvironment::new();
        assert!(matches!(
            env.working_dir(),
            Err(EnvironmentError::NotInitialized)
        ));
        assert!(!env.is_initialized());
    }

    #[rusty_tokio::test]
    async fn initialize_auto_creates_a_temp_dir_and_close_removes_it() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let working_dir = env.working_dir().unwrap();
        assert!(working_dir.exists());
        assert!(env.is_initialized());

        env.close().await;
        assert!(!working_dir.exists());
        assert!(!env.is_initialized());
    }

    #[rusty_tokio::test]
    async fn initialize_creates_an_explicit_working_dir_but_close_does_not_remove_it() {
        let workspace = temp_workspace();
        let env = LocalEnvironment::with_options(Some(workspace.clone()), None);
        env.initialize().await.unwrap();
        assert!(workspace.exists());

        env.close().await;
        assert!(workspace.exists());
        std::fs::remove_dir_all(&workspace).unwrap();
    }

    #[rusty_tokio::test]
    async fn execute_captures_stdout_and_exit_code() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let result = env.execute("echo hello", None).await.unwrap();
        assert_eq!(result.stdout.trim(), "hello");
        assert_eq!(result.exit_code, 0);
        assert!(!result.timed_out);
        env.close().await;
    }

    #[rusty_tokio::test]
    async fn execute_reports_a_nonzero_exit_code() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let result = env.execute("exit 7", None).await.unwrap();
        assert_eq!(result.exit_code, 7);
        env.close().await;
    }

    #[rusty_tokio::test]
    async fn execute_times_out_a_long_running_command() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let result = env
            .execute("sleep 5", Some(Duration::from_millis(200)))
            .await
            .unwrap();
        assert!(result.timed_out);
        env.close().await;
    }

    #[rusty_tokio::test]
    async fn write_file_then_read_file_round_trips() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        env.write_file("nested/hello.txt", b"hello world")
            .await
            .unwrap();
        let content = env.read_file("nested/hello.txt").await.unwrap();
        assert_eq!(content, b"hello world");
        env.close().await;
    }

    #[rusty_tokio::test]
    async fn read_file_reports_a_missing_file() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let error = env.read_file("does-not-exist.txt").await.unwrap_err();
        assert!(matches!(error, EnvironmentError::FileNotFound(_)));
        env.close().await;
    }

    #[rusty_tokio::test]
    async fn read_file_rejects_an_escape_from_the_working_dir() {
        let env = LocalEnvironment::new();
        env.initialize().await.unwrap();
        let error = env.read_file("../escape.txt").await.unwrap_err();
        assert!(matches!(error, EnvironmentError::PathEscapesWorkingDir(_)));
        env.close().await;
    }
}
