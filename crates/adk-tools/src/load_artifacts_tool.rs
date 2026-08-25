//! Capability C0425 (partial): `LoadArtifactsTool`/`load_artifacts_tool`,
//! ported from `google.adk.tools.load_artifacts_tool`.
//!
//! **Scope, disclosed**: `as_safe_part_for_llm`'s MIME
//! normalization/classification, base64 decoding, text-like decoding,
//! and binary-placeholder fallback are ported (now living in
//! `adk_genai::safe_part` — see that module's doc for why this port
//! relocates it out of this crate; re-exported here so this crate's own
//! callers and any external caller of `load_artifacts_tool::as_safe_part_for_llm`
//! see no path change). **Not** ported:
//! - DOCX text extraction (`_try_extract_docx_text`) — needs a zip
//!   reader; no zip-reading crate is a workspace dependency, and adding
//!   one for this single narrow use wasn't judged worth it for this
//!   batch. A `.docx` artifact falls through to the generic
//!   binary-placeholder response instead of extracted text.
//! - Spreadsheet parsing (`_parse_spreadsheet`) — needs a `pandas`
//!   equivalent this port has none of; disabled by default upstream too
//!   (`enable_spreadsheet_parsing=False`), so this is the same
//!   optional-dependency treatment the source itself gives it, not a
//!   narrowing unique to this port.
//! - `process_artifact` (the custom sync/async override callback) — not
//!   exposed; every artifact goes through the built-in safety conversion.
//!
//! `tool_context.load_artifact`/`list_artifacts` return an opaque
//! `Value` (`adk-agents`'s own disclosed placeholder shape for the
//! not-yet-built Phase 6 artifact backend) — parsed back into a typed
//! [`Part`] via its own `Deserialize` impl, the same pattern
//! `ExampleTool`/`PreloadMemoryTool` already use for `user_content`.

use std::collections::BTreeMap;

use adk_genai::content::{Content, FunctionDeclaration, Part};
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::tool_context::ToolContext;

pub use adk_genai::safe_part::{as_safe_part_for_llm, maybe_base64_to_bytes};

/// C0425 (partial): loads artifacts and adds them to the session.
pub struct LoadArtifactsTool;

impl LoadArtifactsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoadArtifactsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for LoadArtifactsTool {
    fn name(&self) -> &str {
        "load_artifacts"
    }

    fn description(&self) -> &str {
        "Loads artifacts into the session for this request.\n\nNOTE: Call when you need access to artifacts (for example, uploads saved by the\nweb UI)."
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
                        "artifact_names".to_string(),
                        Value::Map(vec![
                            ("type".to_string(), Value::String("array".to_string())),
                            (
                                "items".to_string(),
                                Value::Map(vec![(
                                    "type".to_string(),
                                    Value::String("string".to_string()),
                                )]),
                            ),
                        ]),
                    )]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, crate::base_tool::ToolError>> {
        Box::pin(async move {
            let artifact_names = args
                .get("artifact_names")
                .cloned()
                .unwrap_or(Value::Seq(Vec::new()));
            Ok(Value::Map(vec![
                ("artifact_names".to_string(), artifact_names),
                (
                    "status".to_string(),
                    Value::String(
                        "artifact contents temporarily inserted and removed. to access these artifacts, call load_artifacts tool again."
                            .to_string(),
                    ),
                ),
            ]))
        })
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(declaration) = self.get_declaration() {
                merge_declarations(llm_request, [(self.name().to_string(), declaration)]);
            }
            self.append_artifacts_to_llm_request(tool_context, llm_request)
                .await;
        })
    }
}

impl LoadArtifactsTool {
    async fn append_artifacts_to_llm_request(
        &self,
        tool_context: &mut ToolContext,
        llm_request: &mut LlmRequest,
    ) {
        let Ok(artifact_names) = tool_context.list_artifacts().await else {
            return;
        };
        if artifact_names.is_empty() {
            return;
        }

        let names_json = rusty_serde::json::to_string(&Value::Seq(
            artifact_names.iter().cloned().map(Value::String).collect(),
        ))
        .unwrap_or_default();
        let instruction_text = format!(
            "You have a list of artifacts:\n  {names_json}\n\n  When the user asks questions about any of the artifacts, you should call the\n  `load_artifacts` function to load the artifact. Always call load_artifacts\n  before answering questions related to the artifacts, regardless of whether the\n  artifacts have been loaded before. Do not depend on prior answers about the\n  artifacts.\n  "
        );
        llm_request.append_dynamic_instructions(&[instruction_text]);

        let Some(last_content) = llm_request.contents.last() else {
            return;
        };
        let Some(first_part) = last_content.parts.first() else {
            return;
        };
        let Some(function_response) = &first_part.function_response else {
            return;
        };
        if function_response.name.as_deref() != Some("load_artifacts") {
            return;
        }
        let requested_names: Vec<String> = function_response
            .response
            .as_ref()
            .and_then(|response| response.get("artifact_names"))
            .and_then(|value| match value {
                Value::Seq(items) => Some(
                    items
                        .iter()
                        .filter_map(|v| match v {
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        for artifact_name in requested_names {
            let mut artifact_value = tool_context
                .load_artifact(&artifact_name, None)
                .await
                .ok()
                .flatten();
            if artifact_value.is_none() && !artifact_name.starts_with("user:") {
                let prefixed_name = format!("user:{artifact_name}");
                artifact_value = tool_context
                    .load_artifact(&prefixed_name, None)
                    .await
                    .ok()
                    .flatten();
            }
            let Some(artifact_value) = artifact_value else {
                continue;
            };
            let Ok(artifact) = rusty_serde::json::from_value::<Part>(artifact_value) else {
                continue;
            };

            let artifact_part = as_safe_part_for_llm(&artifact, &artifact_name);

            llm_request.contents.push(Content::new(
                "user",
                vec![
                    Part::text(format!("Artifact {artifact_name} is:")),
                    artifact_part,
                ],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn run_async_echoes_requested_artifact_names_with_a_status() {
        let tool = LoadArtifactsTool::new();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "artifact_names".to_string(),
            Value::Seq(vec![Value::String("a.txt".to_string())]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert!(fields.iter().any(|(k, _)| k == "status"));
                let names = fields.iter().find(|(k, _)| k == "artifact_names").unwrap();
                assert_eq!(
                    names.1,
                    Value::Seq(vec![Value::String("a.txt".to_string())])
                );
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn process_llm_request_is_a_no_op_without_an_artifact_service() {
        let tool = LoadArtifactsTool::new();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert!(request.dynamic_instructions().is_empty());
    }
}
