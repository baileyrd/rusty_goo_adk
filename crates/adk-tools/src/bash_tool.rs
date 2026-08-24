//! Capability C0418: `ExecuteBashTool`/`execute_bash`, ported from
//! `google.adk.tools.bash_tool`.
//!
//! **Disclosed narrowings** (this port has no equivalent primitive for
//! any of these — each is a real capability gap, not a silent one):
//! - **Resource limits**: the source's `BashToolPolicy.max_memory_bytes`/
//!   `max_file_size_bytes`/`max_child_processes` are enforced via a
//!   `preexec_fn` calling POSIX `setrlimit()` in the child before
//!   `execve`. This port has no `libc`/`setrlimit` binding (no
//!   `rusty_tokio::process::Command` hook exposes it, and adding a raw
//!   `libc` dependency for three `setrlimit` calls wasn't judged worth
//!   the new dependency for this batch), so those three policy fields
//!   don't exist on this port's `BashToolPolicy` at all — a config field
//!   with no enforcement behind it would be worse than no field.
//!   `RLIMIT_CORE` (core-dump suppression) is likewise not set.
//! - **Process-group kill on timeout**: the source's `start_new_session`
//!   plus `os.killpg(pid, SIGKILL)` kills the whole process group a
//!   timed-out command spawned (catching grandchildren the
//!   shell/interpreter itself forked). This port sets the child into its
//!   own process group (`process_group(0)`) but has no
//!   `killpg`-equivalent — `Child::kill` signals only the immediate
//!   child. `Command::kill_on_drop(true)` is used so a timeout still
//!   reliably kills *that* process, but a grandchild the command spawned
//!   can survive it.
//! - **Partial output on timeout**: the source re-invokes
//!   `process.communicate()` after killing to capture whatever was
//!   buffered before the kill. This port's `Child::wait_with_output`
//!   consumes the child as one unit (no drain-then-kill-then-drain-again
//!   split), so a timeout's response carries no partial stdout/stderr —
//!   disclosed, not silently empty-stringed to look like real output.
//! - **`shlex.split`**: replicated with a hand-rolled POSIX-ish word
//!   splitter (quotes + backslash escaping) rather than a full shlex
//!   grammar (no ANSI-C `$'...'` quoting, no `#` comment handling). This
//!   matches what actually matters here: the source's own
//!   `create_subprocess_exec` never runs a shell, so `|`/`;`/`&&`/`` ` ``
//!   in a command are never interpreted as shell syntax by the source
//!   either — they just become literal argv words, exactly as this port
//!   produces them too.

use std::collections::BTreeMap;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Output;
use std::time::Duration;

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;
use rusty_tokio::process::{Command, Stdio};

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_confirmation::ToolConfirmation;
use crate::tool_context::ToolContext;

/// Configuration for allowed bash commands and (where enforceable) a
/// wall-clock timeout. See the module doc for the resource-limit fields
/// this port doesn't carry.
#[derive(Debug, Clone)]
pub struct BashToolPolicy {
    /// `("*",)` (the default) allows any command.
    pub allowed_command_prefixes: Vec<String>,
    pub blocked_operators: Vec<String>,
    pub timeout_seconds: Option<u64>,
}

impl Default for BashToolPolicy {
    fn default() -> Self {
        Self {
            allowed_command_prefixes: vec!["*".to_string()],
            blocked_operators: Vec::new(),
            timeout_seconds: Some(30),
        }
    }
}

/// Validates a bash command against the permitted prefixes/blocked
/// operators. `None` means the command is allowed.
fn validate_command(command: &str, policy: &BashToolPolicy) -> Option<String> {
    let stripped = command.trim();
    if stripped.is_empty() {
        return Some("Command is required.".to_string());
    }

    for op in &policy.blocked_operators {
        if command.contains(op.as_str()) {
            return Some(format!("Command contains blocked operator: {op}"));
        }
    }

    if policy.allowed_command_prefixes.iter().any(|p| p == "*") {
        return None;
    }
    if policy
        .allowed_command_prefixes
        .iter()
        .any(|prefix| stripped.starts_with(prefix.as_str()))
    {
        return None;
    }

    let allowed = policy.allowed_command_prefixes.join(", ");
    Some(format!(
        "Command blocked. Permitted prefixes are: {allowed}"
    ))
}

#[derive(PartialEq)]
enum Quote {
    None,
    Single,
    Double,
}

/// See the module doc's disclosed narrowing versus Python's `shlex.split`.
fn shlex_split(command: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote = Quote::None;
    let mut chars = command.chars().peekable();

    while let Some(c) = chars.next() {
        match quote {
            Quote::None => match c {
                ' ' | '\t' | '\n' => {
                    if in_word {
                        words.push(std::mem::take(&mut current));
                        in_word = false;
                    }
                }
                '\'' => {
                    quote = Quote::Single;
                    in_word = true;
                }
                '"' => {
                    quote = Quote::Double;
                    in_word = true;
                }
                '\\' => match chars.next() {
                    Some(next) => {
                        current.push(next);
                        in_word = true;
                    }
                    None => return Err("trailing backslash".to_string()),
                },
                _ => {
                    current.push(c);
                    in_word = true;
                }
            },
            Quote::Single => {
                if c == '\'' {
                    quote = Quote::None;
                } else {
                    current.push(c);
                }
            }
            Quote::Double => match c {
                '"' => quote = Quote::None,
                '\\' => match chars.peek() {
                    Some('"') | Some('\\') | Some('$') | Some('`') => {
                        current.push(chars.next().unwrap());
                    }
                    _ => current.push('\\'),
                },
                _ => current.push(c),
            },
        }
    }

    if quote != Quote::None {
        return Err("unterminated quote".to_string());
    }
    if in_word {
        words.push(current);
    }
    Ok(words)
}

fn error_response(message: impl Into<String>) -> Value {
    Value::Map(vec![("error".to_string(), Value::String(message.into()))])
}

fn decode_or_placeholder(bytes: &[u8], placeholder: &str) -> String {
    if bytes.is_empty() {
        placeholder.to_string()
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Mirrors Python's `subprocess.returncode` convention: the exit code if
/// the process exited normally, or the negated signal number if it was
/// killed by one.
fn returncode_value(status: std::process::ExitStatus) -> Value {
    if let Some(code) = status.code() {
        Value::Int(code as i64)
    } else if let Some(signal) = status.signal() {
        Value::Int(-(signal as i64))
    } else {
        Value::Null
    }
}

fn success_response(output: &Output) -> Value {
    Value::Map(vec![
        (
            "stdout".to_string(),
            Value::String(decode_or_placeholder(
                &output.stdout,
                "<no stdout captured>",
            )),
        ),
        (
            "stderr".to_string(),
            Value::String(decode_or_placeholder(
                &output.stderr,
                "<no stderr captured>",
            )),
        ),
        ("returncode".to_string(), returncode_value(output.status)),
    ])
}

/// C0418: executes a validated bash command within a workspace directory.
pub struct ExecuteBashTool {
    workspace: PathBuf,
    policy: BashToolPolicy,
    description: String,
}

impl ExecuteBashTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_policy(workspace, BashToolPolicy::default())
    }

    pub fn with_policy(workspace: PathBuf, policy: BashToolPolicy) -> Self {
        let allowed_hint = if policy.allowed_command_prefixes.iter().any(|p| p == "*") {
            "any command".to_string()
        } else {
            format!(
                "commands matching prefixes: {}",
                policy.allowed_command_prefixes.join(", ")
            )
        };
        let description = format!(
            "Executes a bash command with the working directory set to the workspace. Allowed: {allowed_hint}. All commands require user confirmation."
        );
        Self {
            workspace,
            policy,
            description,
        }
    }

    async fn execute(&self, command: &str) -> Value {
        let argv = match shlex_split(command) {
            Ok(argv) if !argv.is_empty() => argv,
            Ok(_) => return error_response("Command is required."),
            Err(err) => return error_response(format!("Failed to parse command: {err}")),
        };

        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        cmd.current_dir(&self.workspace);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.process_group(0);
        cmd.kill_on_drop(true);

        let child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) => return error_response(format!("Execution failed: {err}")),
        };

        let output_future = child.wait_with_output();
        let output = match self.policy.timeout_seconds {
            Some(seconds) => {
                match rusty_tokio::time::timeout(Duration::from_secs(seconds), output_future).await
                {
                    Ok(result) => result,
                    Err(_) => {
                        return Value::Map(vec![(
                            "error".to_string(),
                            Value::String(format!("Command timed out after {seconds} seconds.")),
                        )]);
                    }
                }
            }
            None => output_future.await,
        };

        match output {
            Ok(output) => success_response(&output),
            Err(err) => error_response(format!("Execution failed: {err}")),
        }
    }
}

impl BaseTool for ExecuteBashTool {
    fn name(&self) -> &str {
        "execute_bash"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        Some(FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "command".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("string".to_string())),
                            (
                                "description".to_string(),
                                Value::String("The bash command to execute.".to_string()),
                            ),
                        ]),
                    )]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("command".to_string())]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let command = match args.get("command") {
                Some(Value::String(command)) if !command.is_empty() => command.clone(),
                _ => return Ok(error_response("Command is required.")),
            };

            if let Some(error) = validate_command(&command, &self.policy) {
                return Ok(error_response(error));
            }

            match tool_context.tool_confirmation() {
                None => {
                    let _ = tool_context.request_confirmation(
                        Some(format!(
                            "Please approve or reject the bash command: {command}"
                        )),
                        None,
                    );
                    tool_context.actions_mut().skip_summarization = true;
                    return Ok(error_response(
                        "This tool call requires confirmation, please approve or reject.",
                    ));
                }
                Some(confirmation_value) => {
                    let confirmed = rusty_serde::json::from_value::<ToolConfirmation>(
                        confirmation_value.clone(),
                    )
                    .map(|confirmation| confirmation.confirmed)
                    .unwrap_or(false);
                    if !confirmed {
                        return Ok(error_response("This tool call is rejected."));
                    }
                }
            }

            Ok(self.execute(&command).await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        let mut context = Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        context.set_function_call_id(Some("fc-1".to_string()));
        context
    }

    fn confirmed_ctx() -> Context {
        let mut context = ctx();
        context.set_tool_confirmation(Some(
            rusty_serde::json::to_value(&ToolConfirmation {
                hint: String::new(),
                confirmed: true,
                payload: None,
            })
            .unwrap(),
        ));
        context
    }

    #[test]
    fn validate_command_rejects_empty_commands() {
        let policy = BashToolPolicy::default();
        assert!(validate_command("", &policy).is_some());
        assert!(validate_command("   ", &policy).is_some());
    }

    #[test]
    fn validate_command_default_policy_allows_everything() {
        let policy = BashToolPolicy::default();
        assert!(validate_command("rm -rf /", &policy).is_none());
        assert!(validate_command("echo hello | grep h", &policy).is_none());
    }

    #[test]
    fn validate_command_restricted_policy_allows_prefixes() {
        let policy = BashToolPolicy {
            allowed_command_prefixes: vec!["ls".to_string(), "cat".to_string()],
            ..Default::default()
        };
        assert!(validate_command("ls -la", &policy).is_none());
        assert!(validate_command("cat file.txt", &policy).is_none());
    }

    #[test]
    fn validate_command_restricted_policy_blocks_others() {
        let policy = BashToolPolicy {
            allowed_command_prefixes: vec!["ls".to_string(), "cat".to_string()],
            ..Default::default()
        };
        let error = validate_command("rm -rf .", &policy).unwrap();
        assert!(error.contains("Permitted prefixes are: ls, cat"));
    }

    #[test]
    fn validate_command_blocked_operators() {
        let policy = BashToolPolicy {
            allowed_command_prefixes: vec!["*".to_string()],
            blocked_operators: vec!["|".to_string(), ";".to_string()],
            ..Default::default()
        };
        assert_eq!(
            validate_command("echo hello | grep h", &policy).unwrap(),
            "Command contains blocked operator: |"
        );
        assert_eq!(
            validate_command("ls ; rm -rf /", &policy).unwrap(),
            "Command contains blocked operator: ;"
        );
    }

    #[test]
    fn shlex_split_handles_quotes_and_escapes() {
        assert_eq!(
            shlex_split("echo 'a b' \"c d\" e\\ f").unwrap(),
            vec!["echo", "a b", "c d", "e f"]
        );
    }

    #[test]
    fn shlex_split_rejects_unterminated_quotes() {
        assert!(shlex_split("echo 'unterminated").is_err());
    }

    #[rusty_tokio::test]
    async fn requests_confirmation_on_first_call() {
        let tool = ExecuteBashTool::new(std::env::temp_dir());
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("ls".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(
                    matches!(&error.1, Value::String(s) if s.contains("requires confirmation"))
                );
            }
            other => panic!("expected an error map, got {other:?}"),
        }
        assert!(context.actions().skip_summarization);
    }

    #[rusty_tokio::test]
    async fn rejects_when_confirmation_is_denied() {
        let tool = ExecuteBashTool::new(std::env::temp_dir());
        let mut context = ctx();
        context.set_tool_confirmation(Some(
            rusty_serde::json::to_value(&ToolConfirmation {
                hint: String::new(),
                confirmed: false,
                payload: None,
            })
            .unwrap(),
        ));
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("ls".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        assert_eq!(
            result,
            Value::Map(vec![(
                "error".to_string(),
                Value::String("This tool call is rejected.".to_string())
            )])
        );
    }

    #[rusty_tokio::test]
    async fn blocks_disallowed_commands_without_requesting_confirmation() {
        let policy = BashToolPolicy {
            allowed_command_prefixes: vec!["ls".to_string()],
            ..Default::default()
        };
        let tool = ExecuteBashTool::with_policy(std::env::temp_dir(), policy);
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("rm -rf .".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(
                    matches!(&error.1, Value::String(s) if s.contains("Permitted prefixes are: ls"))
                );
            }
            other => panic!("expected an error map, got {other:?}"),
        }
        assert!(context.actions().requested_tool_confirmations.is_empty());
    }

    #[rusty_tokio::test]
    async fn executes_when_confirmed_and_captures_stdout() {
        let tool = ExecuteBashTool::new(std::env::temp_dir());
        let mut context = confirmed_ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "command".to_string(),
            Value::String("echo hello-from-bash-tool".to_string()),
        );

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                assert!(
                    matches!(&stdout.1, Value::String(s) if s.contains("hello-from-bash-tool"))
                );
                let returncode = fields.iter().find(|(k, _)| k == "returncode").unwrap();
                assert_eq!(returncode.1, Value::Int(0));
            }
            other => panic!("expected a result map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn nonzero_exit_code_is_reported() {
        let tool = ExecuteBashTool::new(std::env::temp_dir());
        let mut context = confirmed_ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "command".to_string(),
            Value::String("sh -c 'exit 42'".to_string()),
        );

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let returncode = fields.iter().find(|(k, _)| k == "returncode").unwrap();
                assert_eq!(returncode.1, Value::Int(42));
            }
            other => panic!("expected a result map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn cwd_is_the_workspace() {
        let workspace = std::env::temp_dir();
        let tool = ExecuteBashTool::new(workspace.clone());
        let mut context = confirmed_ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("pwd".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let stdout = fields.iter().find(|(k, _)| k == "stdout").unwrap();
                let canonical_workspace = workspace.canonicalize().unwrap();
                match &stdout.1 {
                    Value::String(s) => {
                        let printed = std::path::Path::new(s.trim())
                            .canonicalize()
                            .unwrap_or_else(|_| s.trim().into());
                        assert_eq!(printed, canonical_workspace);
                    }
                    other => panic!("expected a string, got {other:?}"),
                }
            }
            other => panic!("expected a result map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn times_out_a_long_running_command() {
        let policy = BashToolPolicy {
            timeout_seconds: Some(1),
            ..Default::default()
        };
        let tool = ExecuteBashTool::with_policy(std::env::temp_dir(), policy);
        let mut context = confirmed_ctx();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), Value::String("sleep 5".to_string()));

        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(
                    matches!(&error.1, Value::String(s) if s.to_lowercase().contains("timed out"))
                );
            }
            other => panic!("expected an error map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn no_command_returns_an_error() {
        let tool = ExecuteBashTool::new(std::env::temp_dir());
        let mut context = confirmed_ctx();

        let result = tool
            .run_async(&BTreeMap::new(), &mut context)
            .await
            .unwrap();
        match result {
            Value::Map(fields) => {
                let error = fields.iter().find(|(k, _)| k == "error").unwrap();
                assert!(
                    matches!(&error.1, Value::String(s) if s.to_lowercase().contains("required"))
                );
            }
            other => panic!("expected an error map, got {other:?}"),
        }
    }
}
