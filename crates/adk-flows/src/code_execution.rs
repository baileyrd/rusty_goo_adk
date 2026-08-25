//! Capabilities C0177/C0180 (partial): the `_code_execution` request/
//! response processors, ported from
//! `google.adk.flows.llm_flows._code_execution`.
//!
//! **Scope**: the `BuiltInCodeExecutor` branches of both processors
//! (`_run_pre_processor`/`_run_post_processor`'s early-return paths) plus
//! the request side's unconditional "convert executable-code/execution-
//! result parts to text" pass, which the source runs regardless of which
//! kind of executor is configured. Every primitive both branches need —
//! [`BaseCodeExecutor`], [`BuiltInCodeExecutor`], [`CodeExecutorContext`],
//! [`CodeExecutionUtils`]'s free functions — is already built (C0383/
//! C0384/C0390/C0391).
//!
//! **Adaptation, disclosed**: the source resolves `code_executor` itself,
//! off `agent.code_executor` (an `LlmAgent` attribute). This port's
//! `LlmAgent.code_executor` stays an opaque `Value` placeholder — the
//! same C0092 tree-fusion gap every other Phase 4 processor in this
//! crate already discloses (`agent_transfer.rs`, `request_confirmation.rs`,
//! `identity.rs`) — so [`apply_code_execution_request`]/
//! [`apply_code_execution_response`] take `code_executor: &dyn
//! BaseCodeExecutor` as a caller-supplied parameter instead, the same
//! "caller supplies the resolved bits" precedent. Not wired into
//! `LlmFlow::preprocess`/`postprocess` this batch — left ready for a
//! future C0092-unblocking batch, exactly like those other processors.
//!
//! **Not ported this batch, disclosed**:
//! - The general (non-built-in) executor's `optimize_data_file` data-file
//!   extraction/preprocessing path (`_extract_and_replace_inline_files`,
//!   `_get_data_file_preprocessing_code`, the pandas-prelude code
//!   template) — real additional surface area, not a blocker, left for a
//!   follow-up batch. Without `optimize_data_file` support,
//!   [`apply_code_execution_request`]'s non-built-in branch is
//!   consequently always a no-op (matches the source's own `if not
//!   code_executor.optimize_data_file: return` early-out when the
//!   feature isn't enabled).
//! - The general executor's code-execute-and-yield-events response path
//!   (`_run_post_processor`'s non-built-in branch: extract code, run it
//!   via `execute_code`, emit code + result events, clear
//!   `llm_response.content` to loop) — also real additional surface, not
//!   ported this batch. [`apply_code_execution_response`]'s non-built-in
//!   branch is a no-op for now.
//! - Tool-level plugin/canonical callback dispatch — not applicable here,
//!   this capability has none in the source either.

use adk_agents::invocation_context::InvocationContext;
use adk_events::node_info::NodeInfo;
use adk_events::{Event, EventActions};
use adk_genai::content::Part;
use adk_models::llm_request::LlmRequest;
use adk_models::llm_response::LlmResponse;
use adk_tools::base_code_executor::BaseCodeExecutor;
use adk_tools::built_in_code_executor::BuiltInCodeExecutor;
use adk_tools::code_execution_utils::convert_code_execution_parts;
use rusty_serde::value::Value;

#[derive(Debug, rusty_err::Error)]
pub enum CodeExecutionError {
    #[error("{0}")]
    ProcessLlmRequest(String),
    #[error("artifact service is not initialized")]
    ArtifactServiceUnset,
}

fn as_built_in(code_executor: &dyn BaseCodeExecutor) -> Option<&BuiltInCodeExecutor> {
    code_executor.as_any().downcast_ref::<BuiltInCodeExecutor>()
}

/// `_CodeExecutionRequestProcessor.run_async`/`_run_pre_processor`
/// (narrowed — see the module doc). For a [`BuiltInCodeExecutor`], calls
/// its `process_llm_request` (appends the Gemini code-execution tool
/// marker). For any executor, converts every content's trailing
/// executable-code/execution-result parts into plain text using the
/// executor's own delimiters — matching the source's unconditional
/// "Convert the code execution parts to text parts" pass, which runs
/// after the pre-processor regardless of which executor branch fired.
pub async fn apply_code_execution_request(
    llm_request: &mut LlmRequest,
    code_executor: &dyn BaseCodeExecutor,
) -> Result<(), CodeExecutionError> {
    if let Some(built_in) = as_built_in(code_executor) {
        built_in
            .process_llm_request(llm_request)
            .map_err(CodeExecutionError::ProcessLlmRequest)?;
    }
    // Non-built-in, `optimize_data_file`-driven preprocessing (data-file
    // extraction, explore_df code emission) is disclosed-not-ported this
    // batch — see the module doc.

    let code_block_delimiter = code_executor
        .config()
        .code_block_delimiters
        .first()
        .cloned()
        .unwrap_or_else(|| (String::new(), String::new()));
    let execution_result_delimiters = code_executor.config().execution_result_delimiters.clone();
    for content in &mut llm_request.contents {
        convert_code_execution_parts(content, &code_block_delimiter, &execution_result_delimiters);
    }

    Ok(())
}

/// `_CodeExecutionResponseProcessor.run_async`/`_run_post_processor`
/// (narrowed — see the module doc). Skips a partial (streaming) response,
/// matching the source's own top-level check. For a
/// [`BuiltInCodeExecutor`], saves every generated image part to the
/// artifact service and clears it from the response content, always
/// yielding exactly one event carrying the resulting `artifact_delta`
/// (matching the source's own unconditional yield, even with an empty
/// delta). The non-built-in "extract code, execute it, emit events"
/// path is disclosed-not-ported this batch — see the module doc.
pub async fn apply_code_execution_response(
    ctx: &InvocationContext,
    llm_response: &mut LlmResponse,
    code_executor: &dyn BaseCodeExecutor,
    agent_name: &str,
) -> Result<Vec<Event>, CodeExecutionError> {
    if llm_response.partial == Some(true) {
        return Ok(Vec::new());
    }
    if as_built_in(code_executor).is_none() {
        return Ok(Vec::new());
    }
    let Some(content) = llm_response.content.as_mut() else {
        return Ok(Vec::new());
    };

    let mut actions = EventActions::default();
    for part in content.parts.iter_mut() {
        save_generated_image_as_artifact(ctx, part, &mut actions).await?;
    }

    let mut event = Event::new(ctx.invocation_id.clone(), agent_name, NodeInfo::new(""));
    event.branch = ctx.branch.clone();
    event.actions = actions;
    Ok(vec![event])
}

/// If `part` carries inline image data, saves it to the artifact service,
/// records the resulting version in `actions.artifact_delta`, and
/// replaces the part's `inline_data` with a "Saved as artifact" text
/// note — mirroring the source's per-part loop inside the built-in
/// response branch exactly (including its `display_name`-or-timestamp
/// filename fallback).
async fn save_generated_image_as_artifact(
    ctx: &InvocationContext,
    part: &mut Part,
    actions: &mut EventActions,
) -> Result<(), CodeExecutionError> {
    let Some(inline_data) = part.inline_data.clone() else {
        return Ok(());
    };
    let is_image = inline_data
        .mime_type
        .as_deref()
        .unwrap_or("")
        .starts_with("image/");
    if !is_image {
        return Ok(());
    }

    let artifact_service = ctx
        .artifact_service
        .as_ref()
        .ok_or(CodeExecutionError::ArtifactServiceUnset)?;

    let file_name = inline_data
        .rest
        .as_ref()
        .and_then(|rest| rest.get("displayName"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            let extension = inline_data
                .mime_type
                .as_deref()
                .unwrap_or("image")
                .rsplit('/')
                .next()
                .unwrap_or("image");
            format!(
                "{}.{extension}",
                format_timestamp_compact(adk_platform::time::get_time())
            )
        });

    let artifact_part = Part {
        inline_data: Some(inline_data),
        ..Default::default()
    };
    let version = artifact_service.save_artifact(
        &ctx.session.app_name,
        &ctx.session.user_id,
        &ctx.session.id,
        &file_name,
        rusty_serde::json::to_value(&artifact_part).unwrap_or(Value::Null),
        None,
    );
    actions.artifact_delta.insert(file_name.clone(), version);
    part.inline_data = None;
    part.text = Some(format!("Saved as artifact: {file_name}. "));
    Ok(())
}

/// Formats a Unix timestamp as `YYYYMMDD_HHMMSS` in UTC — the source's
/// `now.strftime('%Y%m%d_%H%M%S')` fallback filename format. Same
/// UTC-not-local-time disclosed narrowing, and the same hand-rolled
/// `civil_from_days` calendar algorithm (Howard Hinnant, public domain),
/// as `in_memory_memory_service.rs::format_timestamp` — that function
/// isn't reused directly since its ISO-8601 output shape doesn't match
/// this compact filename format.
fn format_timestamp_compact(timestamp: f64) -> String {
    let total_seconds = timestamp.floor() as i64;
    let days = total_seconds.div_euclid(86400);
    let seconds_of_day = total_seconds.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = (seconds_of_day % 3600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}{month:02}{day:02}_{hour:02}{minute:02}{second:02}")
}

/// Days-since-epoch → (year, month, day) in the proleptic Gregorian
/// calendar. See `in_memory_memory_service.rs`'s identical algorithm for
/// the citation; duplicated here since that one is a private helper.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::services::ArtifactService;
    use adk_agents::session::Session;
    use adk_genai::content::{Content, MediaBlobStub};
    use adk_tools::base_code_executor::CodeExecutorConfig;
    use adk_tools::code_execution_utils::{CodeExecutionInput, CodeExecutionResult};
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    struct StubExecutor {
        config: CodeExecutorConfig,
    }
    impl BaseCodeExecutor for StubExecutor {
        fn config(&self) -> &CodeExecutorConfig {
            &self.config
        }
        fn execute_code(
            &self,
            _ctx: &InvocationContext,
            _input: &CodeExecutionInput,
        ) -> CodeExecutionResult {
            CodeExecutionResult::default()
        }
    }

    fn stub_executor() -> StubExecutor {
        StubExecutor {
            config: CodeExecutorConfig::default(),
        }
    }

    #[rusty_tokio::test]
    async fn built_in_process_llm_request_appends_the_tool_marker_for_a_gemini_model() {
        let mut request = LlmRequest::new("gemini-2.0-flash");
        let executor = BuiltInCodeExecutor::new();

        apply_code_execution_request(&mut request, &executor)
            .await
            .unwrap();

        assert!(
            request.config.tools.is_some(),
            "expected a tools entry, got {request:?}"
        );
    }

    #[rusty_tokio::test]
    async fn built_in_process_llm_request_errors_for_a_non_gemini_model() {
        let mut request = LlmRequest::new("gpt-4");
        let executor = BuiltInCodeExecutor::new();

        let err = apply_code_execution_request(&mut request, &executor)
            .await
            .unwrap_err();
        assert!(matches!(err, CodeExecutionError::ProcessLlmRequest(_)));
    }

    #[rusty_tokio::test]
    async fn convert_code_execution_parts_runs_regardless_of_executor_kind() {
        let mut request = LlmRequest::new("gemini-2.0-flash");
        request.contents = vec![Content::new(
            "model",
            vec![adk_tools::code_execution_utils::build_executable_code_part(
                "print(1)",
            )],
        )];
        let executor = stub_executor();

        apply_code_execution_request(&mut request, &executor)
            .await
            .unwrap();

        assert_eq!(
            request.contents[0].parts[0].text.as_deref(),
            Some("```tool_code\nprint(1)\n```")
        );
    }

    struct RecordingArtifactService {
        saved: Mutex<Vec<(String, String, String, String)>>,
    }
    impl ArtifactService for RecordingArtifactService {
        fn load_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<Value> {
            None
        }

        fn save_artifact(
            &self,
            app_name: &str,
            user_id: &str,
            session_id: &str,
            filename: &str,
            _artifact: Value,
            _custom_metadata: Option<BTreeMap<String, Value>>,
        ) -> i64 {
            self.saved.lock().unwrap().push((
                app_name.to_string(),
                user_id.to_string(),
                session_id.to_string(),
                filename.to_string(),
            ));
            self.saved.lock().unwrap().len() as i64 - 1
        }

        fn get_artifact_version(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
            _version: Option<i64>,
        ) -> Option<adk_agents::services::ArtifactVersion> {
            None
        }

        fn list_artifact_keys(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
        ) -> Vec<String> {
            Vec::new()
        }

        fn list_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<i64> {
            Vec::new()
        }

        fn list_artifact_versions(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) -> Vec<adk_agents::services::ArtifactVersion> {
            Vec::new()
        }

        fn delete_artifact(
            &self,
            _app_name: &str,
            _user_id: &str,
            _session_id: &str,
            _filename: &str,
        ) {
        }
    }

    fn ctx_with_artifact_service() -> (InvocationContext, Arc<RecordingArtifactService>) {
        let service = Arc::new(RecordingArtifactService {
            saved: Mutex::new(Vec::new()),
        });
        let mut ctx =
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        ctx.artifact_service = Some(service.clone());
        (ctx, service)
    }

    fn image_part(display_name: Option<&str>) -> Part {
        Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some("image/png".to_string()),
                rest: display_name.map(|name| {
                    Value::Map(vec![(
                        "displayName".to_string(),
                        Value::String(name.to_string()),
                    )])
                }),
            }),
            ..Default::default()
        }
    }

    #[rusty_tokio::test]
    async fn built_in_response_saves_a_generated_image_using_its_display_name() {
        let (ctx, service) = ctx_with_artifact_service();
        let mut response = LlmResponse {
            content: Some(Content::new("model", vec![image_part(Some("chart.png"))])),
            ..Default::default()
        };
        let executor = BuiltInCodeExecutor::new();

        let events = apply_code_execution_response(&ctx, &mut response, &executor, "my_agent")
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].actions.artifact_delta.get("chart.png"), Some(&0));
        let saved = service.saved.lock().unwrap();
        assert_eq!(saved[0].3, "chart.png");

        let content = response.content.unwrap();
        assert!(content.parts[0].inline_data.is_none());
        assert_eq!(
            content.parts[0].text.as_deref(),
            Some("Saved as artifact: chart.png. ")
        );
    }

    #[rusty_tokio::test]
    async fn built_in_response_generates_a_timestamped_name_without_a_display_name() {
        let (ctx, _service) = ctx_with_artifact_service();
        let mut response = LlmResponse {
            content: Some(Content::new("model", vec![image_part(None)])),
            ..Default::default()
        };
        let executor = BuiltInCodeExecutor::new();

        let events = apply_code_execution_response(&ctx, &mut response, &executor, "my_agent")
            .await
            .unwrap();

        let (name, _) = events[0].actions.artifact_delta.iter().next().unwrap();
        assert!(name.ends_with(".png"), "expected {name:?} to end with .png");
    }

    #[rusty_tokio::test]
    async fn built_in_response_yields_one_event_even_with_no_images() {
        let (ctx, _service) = ctx_with_artifact_service();
        let mut response = LlmResponse {
            content: Some(Content::new("model", vec![Part::text("no image here")])),
            ..Default::default()
        };
        let executor = BuiltInCodeExecutor::new();

        let events = apply_code_execution_response(&ctx, &mut response, &executor, "my_agent")
            .await
            .unwrap();

        assert_eq!(events.len(), 1);
        assert!(events[0].actions.artifact_delta.is_empty());
    }

    #[rusty_tokio::test]
    async fn response_processor_skips_a_partial_response() {
        let (ctx, _service) = ctx_with_artifact_service();
        let mut response = LlmResponse {
            content: Some(Content::new("model", vec![image_part(Some("x.png"))])),
            partial: Some(true),
            ..Default::default()
        };
        let executor = BuiltInCodeExecutor::new();

        let events = apply_code_execution_response(&ctx, &mut response, &executor, "my_agent")
            .await
            .unwrap();
        assert!(events.is_empty());
    }

    #[rusty_tokio::test]
    async fn response_processor_is_a_no_op_for_a_non_built_in_executor() {
        let (ctx, _service) = ctx_with_artifact_service();
        let mut response = LlmResponse {
            content: Some(Content::new("model", vec![image_part(Some("x.png"))])),
            ..Default::default()
        };
        let executor = stub_executor();

        let events = apply_code_execution_response(&ctx, &mut response, &executor, "my_agent")
            .await
            .unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn format_timestamp_compact_matches_a_known_calendar_date() {
        // 2024-01-02T03:04:05Z
        let ts = 1_704_164_645.0;
        assert_eq!(format_timestamp_compact(ts), "20240102_030405");
    }
}
