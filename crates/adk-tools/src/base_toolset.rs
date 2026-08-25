//! Capability C0403 (partial): `BaseToolset`, ported from
//! `google.adk.tools.base_toolset`.
//!
//! **Adaptation**: same trait-instead-of-class shape as [`crate::base_tool`]
//! — the source's `tool_filter`/`tool_name_prefix` instance attributes
//! become trait methods. The source's `_cached_invocation_id`/
//! `_cached_prefixed_tools` (mutated by `get_tools_with_prefix`, a normal
//! Python instance method) can't live on the trait itself — Rust traits
//! have no fields, and `&self` is a shared reference — so implementors own
//! that cache explicitly as a [`PrefixCache`] behind a [`std::sync::Mutex`]
//! and expose it via the required [`BaseToolset::prefix_cache`] method,
//! the same "concrete type stores its own data, trait method fetches it"
//! pattern `BaseTool::custom_metadata` already uses. `get_tools_with_prefix`
//! itself is then a normal default method built on top, using that cache —
//! not `#[final]` the way the source's is (Rust has no equivalent), but
//! disclosed rather than silently dropped since overriding it would be
//! unusual, not disallowed.
//!
//! **Adaptation**: `ToolPredicate` (a `@runtime_checkable Protocol`) plus
//! `tool_filter: Optional[Union[ToolPredicate, List[str]]]` — Python
//! distinguishes the two via `isinstance` at call time; Rust has no
//! runtime duck typing for this, so [`ToolFilter`] models the union
//! directly as an enum instead.
//!
//! **Adaptation**: the source's `_is_tool_selected` (a "private", `_`-
//! prefixed helper) is exposed as the public trait method
//! [`BaseToolset::is_tool_selected`] — Rust trait methods on a `pub trait`
//! are necessarily reachable from any implementor, so there's no
//! private/public distinction to preserve here.
//!
//! **Not** ported: `from_config` — `ToolArgsConfig`/`ToolConfig` (C0417,
//! `crate::tool_configs`) are real types now, but the dynamic-dispatch
//! resolution itself needs Python's `importlib`, genuinely inapplicable
//! in this port — the same disclosed-inapplicable gap
//! `BaseTool::from_config` (C0402) discloses.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use adk_agents::readonly_context::ReadonlyContext;
use adk_agents::services::AuthConfig;
use adk_genai::content::FunctionDeclaration;
use adk_models::capabilities::GoogleLlmVariant;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ResponseScheduling, ToolError};
use crate::tool_context::ToolContext;

/// `ToolPredicate` — decides whether a tool should be exposed to the LLM
/// under the current context.
pub type ToolPredicate = dyn Fn(&dyn BaseTool, Option<&ReadonlyContext>) -> bool + Send + Sync;

/// `Optional[Union[ToolPredicate, List[str]]]` — see the module doc for why
/// this is an enum rather than a `Union`-typed field with `isinstance`
/// dispatch.
pub enum ToolFilter {
    Predicate(Arc<ToolPredicate>),
    Names(Vec<String>),
}

/// The per-invocation cache `get_tools_with_prefix` reads and writes.
/// Implementors of [`BaseToolset`] own one of these (typically behind a
/// `Mutex`, since `get_tools_with_prefix` only has `&self`) and hand it
/// back via [`BaseToolset::prefix_cache`].
#[derive(Default)]
pub struct PrefixCache {
    invocation_id: Option<String>,
    prefixed_tools: Option<Vec<Arc<dyn BaseTool>>>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// C0403: the base trait for all toolsets — a collection of tools usable
/// by an agent. See the module doc for the attribute-to-method and
/// cache-ownership adaptations.
pub trait BaseToolset: Send + Sync {
    /// Returns all tools in the toolset for the given context.
    fn get_tools<'a>(
        &'a self,
        readonly_context: Option<&'a ReadonlyContext>,
    ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>>;

    /// The toolset's own cache storage for `get_tools_with_prefix` — see
    /// the module doc.
    fn prefix_cache(&self) -> &Mutex<PrefixCache>;

    fn tool_filter(&self) -> Option<&ToolFilter> {
        None
    }

    fn tool_name_prefix(&self) -> Option<&str> {
        None
    }

    /// Performs cleanup and releases resources held by the toolset.
    fn close<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// Processes the outgoing LLM request for this toolset, called before
    /// each of its tools processes the request.
    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        _llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// The auth config for this toolset, if authentication is configured.
    fn get_auth_config(&self) -> Option<AuthConfig> {
        None
    }

    fn is_tool_selected(
        &self,
        tool: &dyn BaseTool,
        readonly_context: Option<&ReadonlyContext>,
    ) -> bool {
        match self.tool_filter() {
            None => true,
            Some(ToolFilter::Names(names)) => names.iter().any(|name| name == tool.name()),
            Some(ToolFilter::Predicate(predicate)) => predicate(tool, readonly_context),
        }
    }

    /// Returns all tools, with `tool_name_prefix` applied to their names
    /// (and declaration names) if set, cached per invocation id.
    fn get_tools_with_prefix<'a>(
        &'a self,
        readonly_context: Option<&'a ReadonlyContext>,
    ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
        Box::pin(async move {
            let invocation_id = readonly_context.map(|ctx| ctx.invocation_id().to_string());

            {
                let cache = self.prefix_cache().lock().unwrap();
                if cache.invocation_id == invocation_id {
                    if let Some(tools) = &cache.prefixed_tools {
                        return tools.clone();
                    }
                }
            }

            let tools = self.get_tools(readonly_context).await;
            let result = match self.tool_name_prefix() {
                None => tools,
                Some(prefix) => tools
                    .into_iter()
                    .map(|tool| {
                        let prefixed_name = format!("{prefix}_{}", tool.name());
                        Arc::new(PrefixedTool::new(tool, prefixed_name)) as Arc<dyn BaseTool>
                    })
                    .collect(),
            };

            let mut cache = self.prefix_cache().lock().unwrap();
            cache.invocation_id = invocation_id;
            cache.prefixed_tools = Some(result.clone());
            result
        })
    }
}

/// The wrapper `get_tools_with_prefix` uses to give a tool a prefixed name
/// (and matching declaration name) without touching the original tool —
/// the Rust equivalent of the source's `copy.copy(tool)` +
/// attribute/closure rewrite, which has no direct translation onto a
/// trait object.
struct PrefixedTool {
    inner: Arc<dyn BaseTool>,
    name: String,
}

impl PrefixedTool {
    fn new(inner: Arc<dyn BaseTool>, name: String) -> Self {
        Self { inner, name }
    }
}

impl BaseTool for PrefixedTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn is_long_running(&self) -> bool {
        self.inner.is_long_running()
    }

    fn custom_metadata(&self) -> Option<&BTreeMap<String, Value>> {
        self.inner.custom_metadata()
    }

    fn response_scheduling(&self) -> Option<ResponseScheduling> {
        self.inner.response_scheduling()
    }

    fn api_variant(&self) -> GoogleLlmVariant {
        self.inner.api_variant()
    }

    fn get_declaration(&self) -> Option<FunctionDeclaration> {
        self.inner.get_declaration().map(|mut declaration| {
            declaration.name = Some(self.name.clone());
            declaration
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        self.inner.run_async(args, tool_context)
    }

    fn process_llm_request<'a>(
        &'a self,
        tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> BoxFuture<'a, ()> {
        self.inner.process_llm_request(tool_context, llm_request)
    }

    fn check_require_confirmation<'a>(
        &'a self,
        args: &'a BTreeMap<String, Value>,
        tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, bool> {
        self.inner.check_require_confirmation(args, tool_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    struct NamedTool {
        name: String,
    }

    impl BaseTool for NamedTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "a test tool"
        }
        fn get_declaration(&self) -> Option<FunctionDeclaration> {
            Some(FunctionDeclaration {
                name: Some(self.name.clone()),
                ..Default::default()
            })
        }
    }

    struct StaticToolset {
        tools: Vec<Arc<dyn BaseTool>>,
        tool_name_prefix: Option<String>,
        tool_filter: Option<ToolFilter>,
        cache: Mutex<PrefixCache>,
    }

    impl BaseToolset for StaticToolset {
        fn get_tools<'a>(
            &'a self,
            _readonly_context: Option<&'a ReadonlyContext>,
        ) -> BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
            let tools = self.tools.clone();
            Box::pin(async move { tools })
        }

        fn prefix_cache(&self) -> &Mutex<PrefixCache> {
            &self.cache
        }

        fn tool_filter(&self) -> Option<&ToolFilter> {
            self.tool_filter.as_ref()
        }

        fn tool_name_prefix(&self) -> Option<&str> {
            self.tool_name_prefix.as_deref()
        }
    }

    fn readonly_context(invocation_id: &str) -> ReadonlyContext {
        ReadonlyContext::new(
            InvocationContextBuilder::new(invocation_id, Session::new("app", "user", "s1")).build(),
        )
    }

    #[rusty_tokio::test]
    async fn get_tools_with_prefix_is_a_passthrough_without_a_prefix() {
        let toolset = StaticToolset {
            tools: vec![Arc::new(NamedTool {
                name: "a".to_string(),
            })],
            tool_name_prefix: None,
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        let ctx = readonly_context("inv-1");
        let tools = toolset.get_tools_with_prefix(Some(&ctx)).await;
        assert_eq!(tools[0].name(), "a");
    }

    #[rusty_tokio::test]
    async fn get_tools_with_prefix_renames_the_tool_and_its_declaration() {
        let toolset = StaticToolset {
            tools: vec![Arc::new(NamedTool {
                name: "a".to_string(),
            })],
            tool_name_prefix: Some("ns".to_string()),
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        let ctx = readonly_context("inv-1");
        let tools = toolset.get_tools_with_prefix(Some(&ctx)).await;
        assert_eq!(tools[0].name(), "ns_a");
        assert_eq!(
            tools[0].get_declaration().unwrap().name,
            Some("ns_a".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn get_tools_with_prefix_caches_by_invocation_id() {
        let toolset = StaticToolset {
            tools: vec![Arc::new(NamedTool {
                name: "a".to_string(),
            })],
            tool_name_prefix: Some("ns".to_string()),
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        let ctx = readonly_context("inv-1");
        let first = toolset.get_tools_with_prefix(Some(&ctx)).await;
        let second = toolset.get_tools_with_prefix(Some(&ctx)).await;
        assert!(Arc::ptr_eq(&first[0], &second[0]));
    }

    #[rusty_tokio::test]
    async fn get_tools_with_prefix_recomputes_for_a_new_invocation_id() {
        let toolset = StaticToolset {
            tools: vec![Arc::new(NamedTool {
                name: "a".to_string(),
            })],
            tool_name_prefix: Some("ns".to_string()),
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        let first = toolset
            .get_tools_with_prefix(Some(&readonly_context("inv-1")))
            .await;
        let second = toolset
            .get_tools_with_prefix(Some(&readonly_context("inv-2")))
            .await;
        assert!(!Arc::ptr_eq(&first[0], &second[0]));
    }

    #[test]
    fn is_tool_selected_defaults_to_true_without_a_filter() {
        let toolset = StaticToolset {
            tools: vec![],
            tool_name_prefix: None,
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        let tool = NamedTool {
            name: "a".to_string(),
        };
        assert!(toolset.is_tool_selected(&tool, None));
    }

    #[test]
    fn is_tool_selected_checks_names_list() {
        let toolset = StaticToolset {
            tools: vec![],
            tool_name_prefix: None,
            tool_filter: Some(ToolFilter::Names(vec!["a".to_string()])),
            cache: Mutex::new(PrefixCache::new()),
        };
        let a = NamedTool {
            name: "a".to_string(),
        };
        let b = NamedTool {
            name: "b".to_string(),
        };
        assert!(toolset.is_tool_selected(&a, None));
        assert!(!toolset.is_tool_selected(&b, None));
    }

    #[test]
    fn is_tool_selected_calls_the_predicate() {
        let toolset = StaticToolset {
            tools: vec![],
            tool_name_prefix: None,
            tool_filter: Some(ToolFilter::Predicate(Arc::new(|tool, _ctx| {
                tool.name() == "a"
            }))),
            cache: Mutex::new(PrefixCache::new()),
        };
        let a = NamedTool {
            name: "a".to_string(),
        };
        let b = NamedTool {
            name: "b".to_string(),
        };
        assert!(toolset.is_tool_selected(&a, None));
        assert!(!toolset.is_tool_selected(&b, None));
    }

    #[test]
    fn default_get_auth_config_is_none() {
        let toolset = StaticToolset {
            tools: vec![],
            tool_name_prefix: None,
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        assert!(toolset.get_auth_config().is_none());
    }

    #[rusty_tokio::test]
    async fn default_close_and_process_llm_request_are_no_ops() {
        let toolset = StaticToolset {
            tools: vec![],
            tool_name_prefix: None,
            tool_filter: None,
            cache: Mutex::new(PrefixCache::new()),
        };
        toolset.close().await;
        let mut ctx = adk_agents::context::Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        );
        let mut request = LlmRequest::default();
        toolset.process_llm_request(&mut ctx, &mut request).await;
        assert!(request.config.tools.is_none());
    }
}
