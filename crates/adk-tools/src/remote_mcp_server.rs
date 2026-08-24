//! Capability C0491: `RemoteMcpServer`, ported from
//! `google.adk.tools._remote_mcp_server`.
//!
//! A declarative model describing a server-side MCP server for the
//! Managed Agents API: `ManagedAgent` forwards the server's URL and
//! headers to `interactions.create`, and the Interactions backend opens
//! the MCP session and runs the tools — ADK itself never connects to
//! this server. Contrast with the client-side `McpToolset`
//! (C0540-C0542, unbuilt in this port — see `load_mcp_resource_tool.rs`'s
//! `McpResourceProvider` placeholder), where ADK opens the session and
//! executes tools itself.
//!
//! **Adaptation**: the source is a Pydantic `BaseModel` with
//! `extra='forbid'`, meaning an unrecognized constructor kwarg raises a
//! validation error. A Rust struct has no way to be constructed with
//! extra fields in the first place — the compiler rejects unknown
//! struct-literal fields — so `extra='forbid'` is satisfied trivially by
//! the language, not by any code in this port. `arbitrary_types_allowed`
//! (needed in Pydantic so `header_provider`'s `Callable` type validates)
//! has no Rust equivalent to port either — it's a Pydantic schema-
//! generation concern, not a runtime behavior.
//!
//! **Not built yet**: nothing in this port calls `RemoteMcpServer` —
//! `ManagedAgent`/the Managed Agents API `interactions.create` request
//! path is a separate, larger, unbuilt capability. This is the same
//! "the struct is real and tested, nothing yet produces or consumes a
//! real instance in a live turn" situation as `_node_tool.py`'s sibling
//! row (C0490), which is blocked outright on the unbuilt
//! `workflow::BaseNode` graph engine — `RemoteMcpServer` isn't blocked
//! the same way (it has no such missing dependency), it's simply ahead
//! of its own only caller.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::base_tool::BoxFuture;
use adk_agents::readonly_context::ReadonlyContext;

/// Runtime callback that mints headers (e.g. a fresh bearer token) at
/// request time. Invoked by `ManagedAgent` during resolution
/// (runner-driven), once per turn. Same contract as `LlmAgent`'s
/// `McpToolset.header_provider`.
pub type HeaderProvider = Arc<
    dyn for<'a> Fn(&'a ReadonlyContext) -> BoxFuture<'a, BTreeMap<String, String>> + Send + Sync,
>;

/// C0491: a remote MCP server executed server-side by the Managed
/// Agents API.
#[derive(Clone)]
pub struct RemoteMcpServer {
    /// Full URL of the remote MCP server endpoint (e.g.
    /// `"https://api.example.com/mcp"`). Maps to `MCPServerParam.url`.
    pub url: String,
    /// Optional server label. Maps to `MCPServerParam.name`.
    pub name: Option<String>,
    /// Static headers sent on every turn (e.g. a fixed API key). Merged
    /// with `header_provider` output; `header_provider` wins on key
    /// conflict.
    pub headers: Option<BTreeMap<String, String>>,
    /// Restrict which of the server's tools are exposed. Maps to
    /// `MCPServerParam.allowed_tools`.
    pub allowed_tools: Option<Vec<String>>,
    /// Runtime callback that mints headers at request time. See
    /// [`HeaderProvider`].
    pub header_provider: Option<HeaderProvider>,
}

impl RemoteMcpServer {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            name: None,
            headers: None,
            allowed_tools: None,
            header_provider: None,
        }
    }

    /// Merges `headers` with `header_provider`'s output for the given
    /// context, with the provider's keys winning on conflict —
    /// matching the source's documented merge order.
    pub async fn resolved_headers(
        &self,
        readonly_context: &ReadonlyContext,
    ) -> BTreeMap<String, String> {
        let mut merged = self.headers.clone().unwrap_or_default();
        if let Some(provider) = &self.header_provider {
            for (key, value) in provider(readonly_context).await {
                merged.insert(key, value);
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn readonly_ctx() -> ReadonlyContext {
        ReadonlyContext::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[test]
    fn new_sets_only_the_url_leaving_everything_else_unset() {
        let server = RemoteMcpServer::new("https://api.example.com/mcp");
        assert_eq!(server.url, "https://api.example.com/mcp");
        assert!(server.name.is_none());
        assert!(server.headers.is_none());
        assert!(server.allowed_tools.is_none());
        assert!(server.header_provider.is_none());
    }

    #[rusty_tokio::test]
    async fn resolved_headers_is_empty_without_headers_or_a_provider() {
        let server = RemoteMcpServer::new("https://api.example.com/mcp");
        let resolved = server.resolved_headers(&readonly_ctx()).await;
        assert!(resolved.is_empty());
    }

    #[rusty_tokio::test]
    async fn resolved_headers_returns_the_static_headers() {
        let mut server = RemoteMcpServer::new("https://api.example.com/mcp");
        server.headers = Some(BTreeMap::from([(
            "X-Api-Key".to_string(),
            "static-key".to_string(),
        )]));
        let resolved = server.resolved_headers(&readonly_ctx()).await;
        assert_eq!(resolved.get("X-Api-Key"), Some(&"static-key".to_string()));
    }

    #[rusty_tokio::test]
    async fn header_provider_wins_over_static_headers_on_key_conflict() {
        let mut server = RemoteMcpServer::new("https://api.example.com/mcp");
        server.headers = Some(BTreeMap::from([(
            "Authorization".to_string(),
            "stale-token".to_string(),
        )]));
        server.header_provider = Some(Arc::new(|_ctx| {
            Box::pin(async move {
                BTreeMap::from([("Authorization".to_string(), "fresh-token".to_string())])
            })
        }));
        let resolved = server.resolved_headers(&readonly_ctx()).await;
        assert_eq!(
            resolved.get("Authorization"),
            Some(&"fresh-token".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn header_provider_output_is_merged_with_static_headers() {
        let mut server = RemoteMcpServer::new("https://api.example.com/mcp");
        server.headers = Some(BTreeMap::from([(
            "X-Api-Key".to_string(),
            "static-key".to_string(),
        )]));
        server.header_provider = Some(Arc::new(|_ctx| {
            Box::pin(async move {
                BTreeMap::from([("Authorization".to_string(), "fresh-token".to_string())])
            })
        }));
        let resolved = server.resolved_headers(&readonly_ctx()).await;
        assert_eq!(resolved.get("X-Api-Key"), Some(&"static-key".to_string()));
        assert_eq!(
            resolved.get("Authorization"),
            Some(&"fresh-token".to_string())
        );
    }
}
