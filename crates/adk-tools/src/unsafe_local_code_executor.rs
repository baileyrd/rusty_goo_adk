//! Capability C0385: `UnsafeLocalCodeExecutor`, ported from
//! `google.adk.code_executors.unsafe_local_code_executor`.
//!
//! **Python interpreter, still required, disclosed**: this executor's
//! whole job is to run *Python* source code extracted from a model
//! response — the source does this by re-invoking `sys.executable`
//! (itself, since the source runs inside a Python process). A Rust
//! process was never a Python interpreter, so there is no `sys.executable`
//! equivalent to fall back on: this port spawns a configurable command
//! (`python_executable`, defaulting to `"python3"` on `PATH`) instead.
//! Running this executor still genuinely requires a Python interpreter
//! installed on the host — that's the capability itself, not an
//! avoidable narrowing.
//!
//! **`PYTHONPATH`, narrowed**: the source forwards the *current
//! interpreter's* import path (`sys.path`) into the child's `PYTHONPATH`
//! so code importing an already-resolved application module keeps
//! working. This Rust process has no `sys.path`-equivalent to forward —
//! there is nothing analogous to translate. This port doesn't set
//! `PYTHONPATH` at all, leaving the child to inherit whatever value (if
//! any) is already in this process's own environment, rather than
//! fabricate one.
//!
//! **Process-group kill on timeout, narrowed — same disclosed gap as
//! `bash_tool.rs`**: the source signals the whole process group
//! (SIGTERM, a grace period, then SIGKILL) so a timed-out execution's
//! own children die too. This port has no `killpg`-equivalent (see
//! `bash_tool.rs`'s own module doc for the same gap and why); a timeout
//! here calls `Child::kill()` (SIGKILL) on the immediate child only,
//! with no grace period and no signaling of anything it spawned.
//!
//! **Partial output on timeout, narrowed**: the source's `communicate()`
//! after killing recovers whatever the process had already written.
//! This port's stdout/stderr are drained by dedicated reader threads
//! that run for the process's whole lifetime (not a drain-after-kill
//! step), so a timed-out run's `stdout`/`stderr` still reflect whatever
//! was written before the kill, up to whatever the reader threads
//! managed to read before the pipes closed — functionally equivalent
//! for the common case, though not derived the same way.
//!
//! **Sync by design, matching the source**: `BaseCodeExecutor::execute_code`
//! (and the source's own `execute_code`) is a synchronous method, not
//! `async`. This port therefore uses blocking `std::process::Command`,
//! with dedicated OS threads doing the stdin-write/stdout-read/stderr-read
//! concurrently (mirroring what Python's `subprocess.communicate()` does
//! internally) so a large child output can't deadlock against this
//! process still trying to finish writing stdin.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use adk_agents::invocation_context::InvocationContext;
use regex::Regex;

use crate::base_code_executor::{BaseCodeExecutor, CodeExecutorConfig};
use crate::code_execution_utils::{CodeExecutionInput, CodeExecutionResult};

const DEFAULT_PYTHON_EXECUTABLE: &str = "python3";

/// Runs one program in the child interpreter. Identical to the source's
/// `_RUNNER` script.
const RUNNER: &str = r#"
import sys, traceback

_run_name = sys.argv[1]
del sys.argv[1:]

_globals = {'__name__': _run_name} if _run_name else {}
_source = sys.stdin.buffer.read().decode('utf-8')

try:
  exec(compile(_source, '<code>', 'exec'), _globals, _globals)
except SystemExit:
  raise
except BaseException as exc:
  _tb = exc.__traceback__
  traceback.print_exception(
      type(exc), exc, _tb.tb_next if _tb else None, file=sys.stderr
  )
  sys.exit(1)
"#;

/// `_run_name` — the `__name__` the code should run under, or `""` for
/// none.
fn run_name(code: &str) -> &'static str {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r#"if\s+__name__\s*==\s*['"]__main__['"]"#).unwrap());
    if re.is_match(code) {
        "__main__"
    } else {
        ""
    }
}

#[cfg(unix)]
fn set_new_session(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: setsid() is async-signal-safe and takes no arguments that
    // could violate memory safety; it just puts the child in its own
    // session, the same as the source's `start_new_session=True`.
    unsafe {
        command.pre_exec(|| {
            libc_setsid();
            Ok(())
        });
    }
}

#[cfg(unix)]
fn libc_setsid() {
    // Avoids a `libc` crate dependency for one syscall: `setsid` takes no
    // arguments and returns the new session id (or -1 on error, which we
    // don't act on -- matching the source's own best-effort
    // `start_new_session`, not a hard requirement for correctness).
    extern "C" {
        fn setsid() -> i32;
    }
    unsafe {
        setsid();
    }
}

#[cfg(not(unix))]
fn set_new_session(_command: &mut Command) {}

fn exit_status_display(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(code) = status.code() {
            return code.to_string();
        }
        if let Some(signal) = status.signal() {
            return (-signal).to_string();
        }
    }
    status.code().map(|c| c.to_string()).unwrap_or_default()
}

/// C0385: `UnsafeLocalCodeExecutor` — a code executor that unsafely
/// executes code in the current local context (via a spawned Python
/// interpreter subprocess).
pub struct UnsafeLocalCodeExecutor {
    config: CodeExecutorConfig,
    python_executable: String,
}

impl UnsafeLocalCodeExecutor {
    /// `UnsafeLocalCodeExecutor()` — the source's `stateful`/
    /// `optimize_data_file` are frozen at `false` for this executor;
    /// [`Self::with_config`] returns `Err` (matching the source's
    /// `raise ValueError`) if a caller tries to set either to `true`.
    pub fn new() -> Self {
        Self {
            config: CodeExecutorConfig::default(),
            python_executable: DEFAULT_PYTHON_EXECUTABLE.to_string(),
        }
    }

    pub fn with_config(config: CodeExecutorConfig) -> Result<Self, String> {
        if config.stateful {
            return Err("Cannot set `stateful=True` in UnsafeLocalCodeExecutor.".to_string());
        }
        if config.optimize_data_file {
            return Err(
                "Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor.".to_string(),
            );
        }
        Ok(Self {
            config,
            python_executable: DEFAULT_PYTHON_EXECUTABLE.to_string(),
        })
    }

    /// Overrides the Python interpreter command — see the module doc for
    /// why this replaces the source's `sys.executable`.
    pub fn with_python_executable(mut self, python_executable: impl Into<String>) -> Self {
        self.python_executable = python_executable.into();
        self
    }
}

impl Default for UnsafeLocalCodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseCodeExecutor for UnsafeLocalCodeExecutor {
    fn config(&self) -> &CodeExecutorConfig {
        &self.config
    }

    fn execute_code(
        &self,
        _invocation_context: &InvocationContext,
        code_execution_input: &CodeExecutionInput,
    ) -> CodeExecutionResult {
        let mut command = Command::new(&self.python_executable);
        command
            .arg("-c")
            .arg(RUNNER)
            .arg(run_name(&code_execution_input.code))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONIOENCODING", "utf-8");
        set_new_session(&mut command);

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(e) => {
                return CodeExecutionResult {
                    stdout: String::new(),
                    stderr: format!("Failed to start code execution: {e}"),
                    output_files: Vec::new(),
                };
            }
        };

        let mut stdin = child.stdin.take().expect("stdin was piped");
        let code = code_execution_input.code.clone();
        let stdin_thread = std::thread::spawn(move || {
            let _ = stdin.write_all(code.as_bytes());
        });

        let mut stdout_pipe = child.stdout.take().expect("stdout was piped");
        let stdout_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout_pipe.read_to_end(&mut buf);
            buf
        });

        let mut stderr_pipe = child.stderr.take().expect("stderr was piped");
        let stderr_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr_pipe.read_to_end(&mut buf);
            buf
        });

        let timeout = self.config.timeout_seconds.map(Duration::from_secs);
        let start = Instant::now();
        let mut timed_out = false;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if timeout.is_some_and(|t| start.elapsed() >= t) {
                        timed_out = true;
                        let _ = child.kill();
                        let _ = child.wait();
                        break None;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => break None,
            }
        };

        let _ = stdin_thread.join();
        let stdout_bytes = stdout_thread.join().unwrap_or_default();
        let stderr_bytes = stderr_thread.join().unwrap_or_default();
        let output = String::from_utf8_lossy(&stdout_bytes).into_owned();
        let mut error = String::from_utf8_lossy(&stderr_bytes).into_owned();

        if timed_out {
            let note = match self.config.timeout_seconds {
                Some(seconds) => format!("Code execution timed out after {seconds} seconds."),
                None => "Code execution timed out.".to_string(),
            };
            error = if error.is_empty() {
                note
            } else {
                format!("{error}\n{note}")
            };
        } else if let Some(status) = status {
            if status.success() {
                error.clear();
            } else if error.is_empty() {
                error = format!(
                    "Code execution exited with status {}.",
                    exit_status_display(status)
                );
            }
        }

        CodeExecutionResult {
            stdout: output,
            stderr: error,
            output_files: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> InvocationContext {
        let session = adk_agents::session::Session::new("app", "user", "s1");
        adk_agents::invocation_context::InvocationContextBuilder::new("inv-1", session).build()
    }

    fn python_available() -> bool {
        Command::new(DEFAULT_PYTHON_EXECUTABLE)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn run_name_detects_a_main_guard() {
        assert_eq!(run_name("if __name__ == '__main__':\n  pass"), "__main__");
        assert_eq!(run_name("print(1)"), "");
    }

    #[test]
    fn with_config_rejects_stateful() {
        let config = CodeExecutorConfig {
            stateful: true,
            ..Default::default()
        };
        assert!(UnsafeLocalCodeExecutor::with_config(config).is_err());
    }

    #[test]
    fn with_config_rejects_optimize_data_file() {
        let config = CodeExecutorConfig {
            optimize_data_file: true,
            ..Default::default()
        };
        assert!(UnsafeLocalCodeExecutor::with_config(config).is_err());
    }

    #[test]
    fn execute_code_runs_python_and_captures_stdout() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let executor = UnsafeLocalCodeExecutor::new();
        let input = CodeExecutionInput {
            code: "print('hello from unsafe executor')".to_string(),
            ..Default::default()
        };
        let result = executor.execute_code(&ctx(), &input);
        assert!(result.stdout.contains("hello from unsafe executor"));
        assert!(result.stderr.is_empty());
    }

    #[test]
    fn execute_code_reports_a_traceback_on_error() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let executor = UnsafeLocalCodeExecutor::new();
        let input = CodeExecutionInput {
            code: "raise ValueError('boom')".to_string(),
            ..Default::default()
        };
        let result = executor.execute_code(&ctx(), &input);
        assert!(result.stderr.contains("boom"));
    }

    #[test]
    fn execute_code_times_out_a_long_running_program() {
        if !python_available() {
            eprintln!("skipping: no python3 interpreter on PATH");
            return;
        }
        let config = CodeExecutorConfig {
            timeout_seconds: Some(1),
            ..Default::default()
        };
        let executor = UnsafeLocalCodeExecutor::with_config(config).unwrap();
        let input = CodeExecutionInput {
            code: "import time; time.sleep(10)".to_string(),
            ..Default::default()
        };
        let result = executor.execute_code(&ctx(), &input);
        assert!(result.stderr.to_lowercase().contains("timed out"));
    }
}
