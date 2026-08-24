//! Capability C0426 (partial): `LoadMcpResourceTool`/`load_mcp_resource`,
//! ported from `google.adk.tools.load_mcp_resource_tool`.
//!
//! **Not** ported: a real `McpToolset` — the actual MCP client speaking
//! stdio/SSE/streamable-HTTP is its own, much larger capability
//! (C0540-C0542), not built in this port. This batch defines a minimal
//! [`McpResourceProvider`] trait carrying just the two async operations
//! this tool actually calls (`list_resources`/`read_resource`), matching
//! the same "placeholder trait, forward-referencing a not-yet-built
//! phase" pattern `adk-agents::services` already uses for
//! `MemoryService`/`ArtifactService`. `LoadMcpResourceTool` itself is
//! fully real and testable against any provider implementing the trait,
//! even though no real MCP client exists yet to plug into it.
//!
//! Reuses `load_artifacts_tool`'s base64 decoder for the same
//! `content.blob` decode shape the source's own `_mcp_content_to_part`
//! uses (`base64.b64decode`, with a text-placeholder fallback on
//! failure).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use adk_genai::content::{Content, FunctionDeclaration, Part};
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::append_tools::merge_declarations;
use crate::base_tool::{BaseTool, BoxFuture};
use crate::load_artifacts_tool::maybe_base64_to_bytes;
use crate::tool_context::ToolContext;

type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One piece of content returned by reading an MCP resource — mirrors
/// the subset of MCP's `ResourceContents` the source's own
/// `_mcp_content_to_part` inspects (`text`/`blob`/`mimeType`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct McpResourceContent {
    pub text: Option<String>,
    /// Base64-encoded binary content, matching the MCP wire format.
    pub blob: Option<String>,
    pub mime_type: Option<String>,
}

/// See the module doc: a minimal stand-in for the real `McpToolset`'s
/// resource operations, not the full MCP client.
pub trait McpResourceProvider: Send + Sync {
    fn list_resources(&self) -> ProviderFuture<'_, Result<Vec<String>, String>>;

    fn read_resource<'a>(
        &'a self,
        resource_name: &'a str,
    ) -> ProviderFuture<'a, Result<Vec<McpResourceContent>, String>>;
}

fn mcp_content_to_part(content: &McpResourceContent, resource_name: &str) -> Part {
    if let Some(text) = &content.text {
        return Part::text(text.clone());
    }
    if let Some(blob) = &content.blob {
        return match maybe_base64_to_bytes(blob) {
            Some(_data) => {
                // This port's `Part` has no `from_bytes`/inline-data
                // constructor helper wired up for arbitrary binary
                // payloads outside the `MediaBlobStub` placeholder shape
                // (see `adk-genai`'s own disclosed scope); the decoded
                // bytes are re-encoded as inline data with the resource's
                // declared MIME type, matching the source's
                // `types.Part.from_bytes` call.
                let mime_type = content
                    .mime_type
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                Part {
                    inline_data: Some(adk_genai::content::MediaBlobStub {
                        mime_type: Some(mime_type),
                        rest: Some(Value::Map(vec![(
                            "data".to_string(),
                            Value::String(blob.clone()),
                        )])),
                    }),
                    ..Default::default()
                }
            }
            None => Part::text(format!(
                "[Binary content for {resource_name} could not be decoded]"
            )),
        };
    }
    Part::text(format!("[Unknown content type for {resource_name}]"))
}

/// C0426 (partial): loads MCP resources and adds them to the session.
pub struct LoadMcpResourceTool {
    provider: Box<dyn McpResourceProvider>,
}

impl LoadMcpResourceTool {
    pub fn new(provider: Box<dyn McpResourceProvider>) -> Self {
        Self { provider }
    }
}

impl BaseTool for LoadMcpResourceTool {
    fn name(&self) -> &str {
        "load_mcp_resource"
    }

    fn description(&self) -> &str {
        "Loads resources from the MCP server.\n\nNOTE: Call when you need access to resources."
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
                        "resource_names".to_string(),
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
            let resource_names = args
                .get("resource_names")
                .cloned()
                .unwrap_or(Value::Seq(Vec::new()));
            Ok(Value::Map(vec![
                ("resource_names".to_string(), resource_names),
                (
                    "status".to_string(),
                    Value::String(
                        "resource contents temporarily inserted and removed. to access these resources, call load_mcp_resource tool again."
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
            self.append_resources_to_llm_request(tool_context, llm_request)
                .await;
        })
    }
}

impl LoadMcpResourceTool {
    async fn append_resources_to_llm_request(
        &self,
        _tool_context: &mut ToolContext,
        llm_request: &mut LlmRequest,
    ) {
        if let Ok(resource_names) = self.provider.list_resources().await {
            if !resource_names.is_empty() {
                let names_json = rusty_serde::json::to_string(&Value::Seq(
                    resource_names.iter().cloned().map(Value::String).collect(),
                ))
                .unwrap_or_default();
                let instruction = format!(
                    "You have a list of MCP resources:\n{names_json}\n\nWhen the user asks questions about any of the resources, you should call the\n`load_mcp_resource` function to load the resource. Always call load_mcp_resource\nbefore answering questions related to the resources.\n"
                );
                llm_request.append_dynamic_instructions(&[instruction]);
            }
        }
        // Failure to list is logged (not raised) by the source; no
        // logging framework has been adopted by this workspace yet
        // (same disclosed omission as `contents.rs`'s
        // `drop_orphaned_function_responses`).

        let Some(last_content) = llm_request.contents.last() else {
            return;
        };
        let Some(first_part) = last_content.parts.first() else {
            return;
        };
        let Some(function_response) = &first_part.function_response else {
            return;
        };
        if function_response.name.as_deref() != Some(self.name()) {
            return;
        }
        let requested_names: Vec<String> = function_response
            .response
            .as_ref()
            .and_then(|response| response.get("resource_names"))
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

        for resource_name in requested_names {
            let Ok(contents) = self.provider.read_resource(&resource_name).await else {
                continue;
            };
            for content in &contents {
                let part = mcp_content_to_part(content, &resource_name);
                llm_request.contents.push(Content::new(
                    "user",
                    vec![Part::text(format!("Resource {resource_name} is:")), part],
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_genai::content::FunctionResponse;
    use std::sync::Mutex;

    struct StubProvider {
        resources: Vec<String>,
        contents: Mutex<BTreeMap<String, Vec<McpResourceContent>>>,
    }

    impl McpResourceProvider for StubProvider {
        fn list_resources(&self) -> ProviderFuture<'_, Result<Vec<String>, String>> {
            let resources = self.resources.clone();
            Box::pin(async move { Ok(resources) })
        }

        fn read_resource<'a>(
            &'a self,
            resource_name: &'a str,
        ) -> ProviderFuture<'a, Result<Vec<McpResourceContent>, String>> {
            let contents = self.contents.lock().unwrap().get(resource_name).cloned();
            Box::pin(async move { contents.ok_or_else(|| "not found".to_string()) })
        }
    }

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn run_async_echoes_requested_resource_names_with_a_status() {
        let tool = LoadMcpResourceTool::new(Box::new(StubProvider {
            resources: vec![],
            contents: Mutex::new(BTreeMap::new()),
        }));
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "resource_names".to_string(),
            Value::Seq(vec![Value::String("doc.txt".to_string())]),
        );
        let result = tool.run_async(&args, &mut context).await.unwrap();
        match result {
            Value::Map(fields) => {
                assert!(fields.iter().any(|(k, _)| k == "status"));
            }
            other => panic!("expected a map, got {other:?}"),
        }
    }

    #[rusty_tokio::test]
    async fn process_llm_request_injects_the_resource_list_instruction() {
        let tool = LoadMcpResourceTool::new(Box::new(StubProvider {
            resources: vec!["doc.txt".to_string()],
            contents: Mutex::new(BTreeMap::new()),
        }));
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert_eq!(request.dynamic_instructions().len(), 1);
        assert!(request.dynamic_instructions()[0].contains("doc.txt"));
    }

    #[rusty_tokio::test]
    async fn process_llm_request_is_a_no_op_when_no_resources_are_listed() {
        let tool = LoadMcpResourceTool::new(Box::new(StubProvider {
            resources: vec![],
            contents: Mutex::new(BTreeMap::new()),
        }));
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        tool.process_llm_request(&mut context, &mut request).await;
        assert!(request.dynamic_instructions().is_empty());
    }

    #[rusty_tokio::test]
    async fn process_llm_request_appends_text_content_from_a_requested_resource() {
        let mut contents = BTreeMap::new();
        contents.insert(
            "doc.txt".to_string(),
            vec![McpResourceContent {
                text: Some("hello from mcp".to_string()),
                blob: None,
                mime_type: None,
            }],
        );
        let tool = LoadMcpResourceTool::new(Box::new(StubProvider {
            resources: vec![],
            contents: Mutex::new(contents),
        }));
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        request.contents.push(Content::new(
            "user",
            vec![Part {
                function_response: Some(FunctionResponse {
                    id: None,
                    name: Some("load_mcp_resource".to_string()),
                    response: Some(BTreeMap::from([(
                        "resource_names".to_string(),
                        Value::Seq(vec![Value::String("doc.txt".to_string())]),
                    )])),
                }),
                ..Default::default()
            }],
        ));

        tool.process_llm_request(&mut context, &mut request).await;

        let appended = request.contents.last().unwrap();
        assert_eq!(appended.parts[1].text.as_deref(), Some("hello from mcp"));
    }

    #[rusty_tokio::test]
    async fn process_llm_request_skips_a_resource_that_fails_to_read() {
        let tool = LoadMcpResourceTool::new(Box::new(StubProvider {
            resources: vec![],
            contents: Mutex::new(BTreeMap::new()),
        }));
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        let contents_len_before = request.contents.len();
        request.contents.push(Content::new(
            "user",
            vec![Part {
                function_response: Some(FunctionResponse {
                    id: None,
                    name: Some("load_mcp_resource".to_string()),
                    response: Some(BTreeMap::from([(
                        "resource_names".to_string(),
                        Value::Seq(vec![Value::String("missing.txt".to_string())]),
                    )])),
                }),
                ..Default::default()
            }],
        ));

        tool.process_llm_request(&mut context, &mut request).await;

        assert_eq!(request.contents.len(), contents_len_before + 1);
    }

    #[test]
    fn mcp_content_to_part_falls_back_to_a_placeholder_for_undecodable_blobs() {
        let content = McpResourceContent {
            text: None,
            blob: Some("!!!@@@###$$$%%%^^^&&&".to_string()),
            mime_type: Some("application/octet-stream".to_string()),
        };
        let part = mcp_content_to_part(&content, "bad.bin");
        assert!(part.text.unwrap().contains("could not be decoded"));
    }

    #[test]
    fn mcp_content_to_part_reports_unknown_content() {
        let content = McpResourceContent::default();
        let part = mcp_content_to_part(&content, "mystery");
        assert!(part.text.unwrap().contains("Unknown content type"));
    }
}
