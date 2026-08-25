//! Capability C0451: `StdioConnectionParams`/`SseConnectionParams`/
//! `StreamableHTTPConnectionParams`, ported from
//! `google.adk.tools.mcp_tool.mcp_session_manager`.
//!
//! Plain connection-parameter data models — real, standalone structs
//! with no client-transport behavior attached. Nothing in this port
//! constructs a live MCP session from one of these yet: a real
//! client-side MCP transport (`McpToolset`, C0540-C0542 in the manifest,
//! itself unbuilt — see `load_mcp_resource_tool.rs`'s `McpResourceProvider`
//! placeholder for the established "no `mcp` client crate dependency"
//! stance this port already takes) stays out of scope; these are the
//! data shape a future `McpToolset` would accept as configuration.
//!
//! **`httpx_client_factory`, not ported, disclosed**: the source's
//! `SseConnectionParams`/`StreamableHTTPConnectionParams` both default
//! this field to `create_mcp_http_client`, a callable building a
//! `CheckableMcpHttpClientFactory`-typed HTTPX client. This port has no
//! HTTP-client-factory abstraction decided for MCP transports at all —
//! the same disclosed gap `_DebugHttpxClientFactory` (C0453) is left
//! blocked on — so it's omitted entirely rather than represented as a
//! dead placeholder field with nothing to construct or call it.
//!
//! **`StdioServerParameters`, a plain struct not the real `mcp` SDK
//! type**: the source's `StdioConnectionParams.server_params` is typed
//! `mcp.StdioServerParameters` (an upstream MCP Python SDK dataclass).
//! No `mcp` crate is a dependency of this port (nor should it become
//! one just for this data shape — see the module-level stance above),
//! so [`StdioServerParameters`] here is this port's own minimal
//! reimplementation of that SDK type's well-known public shape
//! (`command`/`args`/`env`/`cwd`), not a wrapper around a real
//! dependency.
//!
//! **`arbitrary_types_allowed`/`model_config`, no Rust equivalent**:
//! same disclosed non-issue `remote_mcp_server.rs`'s own doc already
//! establishes for `ConfigDict(arbitrary_types_allowed=True)` — a
//! Pydantic schema-generation concern with nothing to port at the
//! struct-shape level.

use std::collections::BTreeMap;

fn default_timeout() -> f64 {
    5.0
}

fn default_sse_read_timeout() -> f64 {
    60.0 * 5.0
}

/// `mcp.StdioServerParameters` — this port's own minimal reimplementation
/// of the upstream MCP Python SDK's dataclass shape. See the module doc
/// for why this isn't a real `mcp` crate dependency.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StdioServerParameters {
    pub command: String,
    pub args: Vec<String>,
    pub env: Option<BTreeMap<String, String>>,
    pub cwd: Option<String>,
}

impl StdioServerParameters {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: None,
            cwd: None,
        }
    }
}

/// `mcp_session_manager.StdioConnectionParams` — parameters for the MCP
/// Stdio connection.
#[derive(Debug, Clone, PartialEq)]
pub struct StdioConnectionParams {
    pub server_params: StdioServerParameters,
    pub timeout: f64,
}

impl StdioConnectionParams {
    pub fn new(server_params: StdioServerParameters) -> Self {
        Self {
            server_params,
            timeout: default_timeout(),
        }
    }
}

/// `mcp_session_manager.SseConnectionParams` — parameters for the MCP
/// SSE connection. See the module doc for why `httpx_client_factory`
/// isn't ported.
#[derive(Debug, Clone, PartialEq)]
pub struct SseConnectionParams {
    pub url: String,
    pub headers: Option<BTreeMap<String, String>>,
    pub timeout: f64,
    pub sse_read_timeout: f64,
}

impl SseConnectionParams {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: None,
            timeout: default_timeout(),
            sse_read_timeout: default_sse_read_timeout(),
        }
    }
}

/// `mcp_session_manager.StreamableHTTPConnectionParams` — parameters for
/// the MCP Streamable HTTP connection. See the module doc for why
/// `httpx_client_factory` isn't ported.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamableHttpConnectionParams {
    pub url: String,
    pub headers: Option<BTreeMap<String, String>>,
    pub timeout: f64,
    pub sse_read_timeout: f64,
    pub terminate_on_close: bool,
}

impl StreamableHttpConnectionParams {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: None,
            timeout: default_timeout(),
            sse_read_timeout: default_sse_read_timeout(),
            terminate_on_close: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdio_connection_params_defaults_a_five_second_timeout() {
        let params = StdioConnectionParams::new(StdioServerParameters::new("npx"));
        assert_eq!(params.timeout, 5.0);
        assert_eq!(params.server_params.command, "npx");
        assert!(params.server_params.args.is_empty());
        assert_eq!(params.server_params.env, None);
    }

    #[test]
    fn stdio_server_parameters_carries_args_and_env() {
        let mut env = BTreeMap::new();
        env.insert("OPENAPI_MCP_HEADERS".to_string(), "{}".to_string());
        let params = StdioServerParameters {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@notionhq/notion-mcp-server".to_string()],
            env: Some(env.clone()),
            cwd: None,
        };
        assert_eq!(params.args, vec!["-y", "@notionhq/notion-mcp-server"]);
        assert_eq!(params.env, Some(env));
    }

    #[test]
    fn sse_connection_params_defaults_timeout_and_sse_read_timeout() {
        let params = SseConnectionParams::new("https://example.com/sse");
        assert_eq!(params.url, "https://example.com/sse");
        assert_eq!(params.timeout, 5.0);
        assert_eq!(params.sse_read_timeout, 300.0);
        assert_eq!(params.headers, None);
    }

    #[test]
    fn streamable_http_connection_params_defaults_terminate_on_close_to_true() {
        let params = StreamableHttpConnectionParams::new("https://example.com/mcp");
        assert_eq!(params.url, "https://example.com/mcp");
        assert_eq!(params.timeout, 5.0);
        assert_eq!(params.sse_read_timeout, 300.0);
        assert!(params.terminate_on_close);
    }
}
