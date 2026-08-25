//! Capability C0482: `tools.retrieval.base_retrieval_tool.BaseRetrievalTool`,
//! ported from `google.adk.tools.retrieval.base_retrieval_tool`.
//!
//! **Class -> trait + free function**: the source overrides only
//! `_get_declaration`, and its concrete subclasses (`FilesRetrieval`
//! C0483, `LlamaIndexRetrieval` C0484, `VertexAiRagRetrieval` C0485 — each
//! blocked on its own not-yet-adopted dependency: LlamaIndex, Vertex AI
//! RAG) each get exactly the same declaration for free by inheritance.
//! `crate::base_tool::BaseTool::get_declaration` already has one default
//! implementation (`None`), defined once at the supertrait — a Rust
//! subtrait can't override an inherited default method for just its own
//! implementers, so instead [`retrieval_tool_declaration`] is the shared
//! construction logic as a free function (the same shape
//! `request_input_tool.rs`'s own `parameters_schema()` helper already
//! uses), and [`BaseRetrievalTool`] is a thin marker supertrait whose
//! default [`BaseRetrievalTool::retrieval_declaration`] method wraps it —
//! a future concrete retrieval tool implements `BaseTool` as usual and
//! has its own `get_declaration()` call
//! `self.retrieval_declaration()`.
//!
//! `BaseRetrievalTool` also gives a future port of
//! `cli.agent_graph`'s `isinstance(tool_or_agent, BaseRetrievalTool)`
//! checks (C0281, docs-only, deferred with the rest of `adk-cli`) a real
//! marker trait to check against once that lands — built ahead of that
//! still-blocked caller, the same "widen/build a placeholder ahead of a
//! still-blocked caller" precedent already used elsewhere in this port
//! (e.g. `runner::get_function_responses_from_content`).
//!
//! **`run_async` contract**: left abstract in the source (no default
//! body) — documented there, and here, as expecting `args["query"]`; not
//! implemented since [`crate::base_tool::BaseTool::run_async`] already
//! defaults to an error for exactly this "must be overridden" case.
//!
//! **Feature gate**: mirrors the source's
//! `is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)` branch
//! exactly — when on, the query parameter is described via
//! `parameters_json_schema` (a plain JSON-Schema-shaped `Value`); when
//! off, via the typed `parameters` field instead. Both branches build the
//! identical `{"type": "object", "properties": {"query": {"type":
//! "string", "description": "The query to retrieve."}}}` shape, just
//! addressed to a different [`adk_genai::content::FunctionDeclaration`]
//! field — this crate models both fields as opaque JSON `Value`s (see
//! `content.rs`), so both branches build the same literal either way.

use adk_features::feature_registry::{is_feature_enabled, FeatureName};
use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::BaseTool;

const QUERY_DESCRIPTION: &str = "The query to retrieve.";

fn query_schema() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("object".to_string())),
        (
            "properties".to_string(),
            Value::Map(vec![(
                "query".to_string(),
                Value::Map(vec![
                    ("type".to_string(), Value::String("string".to_string())),
                    (
                        "description".to_string(),
                        Value::String(QUERY_DESCRIPTION.to_string()),
                    ),
                ]),
            )]),
        ),
    ])
}

/// `BaseRetrievalTool._get_declaration` — builds the `query`-string
/// declaration shared by every retrieval tool, gated on
/// `FeatureName::JsonSchemaForFuncDecl` exactly as the source is.
pub fn retrieval_tool_declaration(name: &str, description: &str) -> FunctionDeclaration {
    let mut declaration = FunctionDeclaration {
        name: Some(name.to_string()),
        description: Some(description.to_string()),
        ..Default::default()
    };
    if is_feature_enabled(FeatureName::JsonSchemaForFuncDecl) {
        declaration.parameters_json_schema = Some(query_schema());
    } else {
        declaration.parameters = Some(query_schema());
    }
    declaration
}

/// `tools.retrieval.base_retrieval_tool.BaseRetrievalTool` — see the
/// module doc for why this is a thin marker supertrait around the shared
/// [`retrieval_tool_declaration`] free function rather than a default
/// override of [`BaseTool::get_declaration`] itself.
pub trait BaseRetrievalTool: BaseTool {
    /// The shared `query`-string declaration every retrieval tool
    /// exposes. A concrete implementer's own `get_declaration()` should
    /// return `Some(self.retrieval_declaration())`.
    fn retrieval_declaration(&self) -> FunctionDeclaration {
        retrieval_tool_declaration(self.name(), self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_features::feature_registry::TemporaryFeatureOverride;
    use std::sync::Mutex as StdMutex;

    // `TemporaryFeatureOverride` mutates process-wide state
    // (`adk_features::feature_registry`'s override map); serialize every
    // test in this module that touches `FeatureName::JsonSchemaForFuncDecl`
    // so they don't race each other under the default parallel test runner.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    struct StubRetrievalTool;

    impl BaseTool for StubRetrievalTool {
        fn name(&self) -> &str {
            "stub_retrieval"
        }

        fn description(&self) -> &str {
            "A stub retrieval tool."
        }

        fn get_declaration(&self) -> Option<FunctionDeclaration> {
            Some(self.retrieval_declaration())
        }
    }

    impl BaseRetrievalTool for StubRetrievalTool {}

    #[test]
    fn declaration_uses_parameters_json_schema_when_the_feature_is_enabled() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, true);
        let declaration = retrieval_tool_declaration("my_tool", "my description");

        assert_eq!(declaration.name.as_deref(), Some("my_tool"));
        assert_eq!(declaration.description.as_deref(), Some("my description"));
        assert!(declaration.parameters.is_none());
        assert_eq!(declaration.parameters_json_schema, Some(query_schema()));
    }

    #[test]
    fn declaration_uses_parameters_when_the_feature_is_disabled() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, false);
        let declaration = retrieval_tool_declaration("my_tool", "my description");

        assert!(declaration.parameters_json_schema.is_none());
        assert_eq!(declaration.parameters, Some(query_schema()));
    }

    #[test]
    fn declaration_describes_the_query_field_with_the_source_string() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, true);
        let declaration = retrieval_tool_declaration("t", "d");
        let schema = declaration.parameters_json_schema.unwrap();
        let Value::Map(entries) = &schema else {
            panic!("expected a Value::Map");
        };
        let properties = entries
            .iter()
            .find(|(key, _)| key == "properties")
            .map(|(_, value)| value)
            .expect("properties key");
        let Value::Map(properties) = properties else {
            panic!("expected a Value::Map");
        };
        let query = properties
            .iter()
            .find(|(key, _)| key == "query")
            .map(|(_, value)| value)
            .expect("query key");
        let Value::Map(query) = query else {
            panic!("expected a Value::Map");
        };
        let description = query
            .iter()
            .find(|(key, _)| key == "description")
            .map(|(_, value)| value)
            .expect("description key");
        assert_eq!(description, &Value::String(QUERY_DESCRIPTION.to_string()));
    }

    #[test]
    fn base_retrieval_tool_default_method_matches_the_free_function() {
        let _lock = TEST_LOCK.lock().unwrap();
        let _override = TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, true);
        let tool = StubRetrievalTool;
        let declaration = tool.get_declaration().expect("declaration");
        assert_eq!(
            declaration,
            retrieval_tool_declaration("stub_retrieval", "A stub retrieval tool.")
        );
    }

    #[test]
    fn toggling_the_feature_twice_round_trips_through_both_branches() {
        let _lock = TEST_LOCK.lock().unwrap();
        {
            let _override = TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, true);
            assert!(retrieval_tool_declaration("t", "d")
                .parameters_json_schema
                .is_some());
        }
        {
            let _override =
                TemporaryFeatureOverride::new(FeatureName::JsonSchemaForFuncDecl, false);
            assert!(retrieval_tool_declaration("t", "d").parameters.is_some());
        }
    }
}
