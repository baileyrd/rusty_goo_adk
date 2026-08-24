//! Part of C0455: `adk_to_mcp_tool_type`/`gemini_to_json_schema`, ported
//! from `google.adk.tools.mcp_tool.conversion_utils`. `session_context.py`
//! (`SessionContext`, real async `mcp.ClientSession` pooling) stays
//! `REQUIRED` — this port has no `mcp` crate dependency, and none of the
//! two functions here actually need one: `gemini_to_json_schema` is a
//! pure tree transform over schema-shaped data, and `adk_to_mcp_tool_type`
//! only needs a `{name, description, inputSchema}` triple, not the real
//! `mcp.types.Tool` model.

use adk_genai::content::FunctionDeclaration;
use rusty_serde::value::Value;

use crate::base_tool::BaseTool;

/// Narrowed stand-in for `mcp.types.Tool` — just the three fields
/// `adk_to_mcp_tool_type` populates. See the module doc for why this
/// port has no real `mcp` crate dependency to build the actual type.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// `conversion_utils.adk_to_mcp_tool_type` — converts an ADK tool
/// definition into its MCP-tool-shaped equivalent.
pub fn adk_to_mcp_tool_type(tool: &dyn BaseTool) -> McpTool {
    let input_schema = match tool.get_declaration() {
        None => Value::Map(Vec::new()),
        Some(FunctionDeclaration {
            parameters_json_schema: Some(json_schema),
            ..
        }) => json_schema,
        Some(FunctionDeclaration {
            parameters: Some(parameters),
            ..
        }) => gemini_to_json_schema(&parameters),
        Some(_) => Value::Map(Vec::new()),
    };
    McpTool {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        input_schema,
    }
}

fn direct_field(gemini_schema: &Value, key: &str) -> Option<Value> {
    gemini_schema.get(key).filter(|v| !v.is_null()).cloned()
}

/// `conversion_utils.gemini_to_json_schema` — converts a Gemini
/// `Schema`-shaped value into a JSON-Schema dictionary. Operates on
/// `Value` rather than a typed `Schema` struct — this workspace
/// represents a Gemini schema as `Value` throughout (see
/// `adk_genai::content::FunctionDeclaration::parameters`), the same
/// representation `gemini_schema_util::to_gemini_schema` produces.
pub fn gemini_to_json_schema(gemini_schema: &Value) -> Value {
    let mut json_schema: Vec<(String, Value)> = Vec::new();

    let gemini_type = gemini_schema
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_lowercase);
    let type_str = match gemini_type.as_deref() {
        Some(t) if t != "type_unspecified" => t.to_string(),
        _ => "null".to_string(),
    };
    json_schema.push(("type".to_string(), Value::String(type_str.clone())));

    if gemini_schema.get("nullable").and_then(Value::as_bool) == Some(true) {
        json_schema.push(("nullable".to_string(), Value::Bool(true)));
    }

    for (gemini_key, json_key) in [
        ("title", "title"),
        ("description", "description"),
        ("default", "default"),
        ("enum", "enum"),
        ("format", "format"),
        ("example", "example"),
    ] {
        if let Some(value) = direct_field(gemini_schema, gemini_key) {
            json_schema.push((json_key.to_string(), value));
        }
    }

    if type_str == "string" {
        for (gemini_key, json_key) in [
            ("pattern", "pattern"),
            ("min_length", "minLength"),
            ("max_length", "maxLength"),
        ] {
            if let Some(value) = direct_field(gemini_schema, gemini_key) {
                json_schema.push((json_key.to_string(), value));
            }
        }
    }

    if type_str == "number" || type_str == "integer" {
        for (gemini_key, json_key) in [("minimum", "minimum"), ("maximum", "maximum")] {
            if let Some(value) = direct_field(gemini_schema, gemini_key) {
                json_schema.push((json_key.to_string(), value));
            }
        }
    }

    if type_str == "array" {
        if let Some(items) = direct_field(gemini_schema, "items") {
            json_schema.push(("items".to_string(), gemini_to_json_schema(&items)));
        }
        for (gemini_key, json_key) in [("min_items", "minItems"), ("max_items", "maxItems")] {
            if let Some(value) = direct_field(gemini_schema, gemini_key) {
                json_schema.push((json_key.to_string(), value));
            }
        }
    }

    if type_str == "object" {
        if let Some(Value::Map(properties)) = direct_field(gemini_schema, "properties") {
            let converted: Vec<(String, Value)> = properties
                .into_iter()
                .map(|(name, schema)| (name, gemini_to_json_schema(&schema)))
                .collect();
            json_schema.push(("properties".to_string(), Value::Map(converted)));
        }
        // `property_ordering` is intentionally ignored -- it's not a
        // standard JSON Schema field, matching the source's own comment.
        for (gemini_key, json_key) in [
            ("required", "required"),
            ("min_properties", "minProperties"),
            ("max_properties", "maxProperties"),
        ] {
            if let Some(value) = direct_field(gemini_schema, gemini_key) {
                json_schema.push((json_key.to_string(), value));
            }
        }
    }

    if let Some(Value::Seq(any_of)) = direct_field(gemini_schema, "any_of") {
        let converted: Vec<Value> = any_of.iter().map(gemini_to_json_schema).collect();
        json_schema.push(("anyOf".to_string(), Value::Seq(converted)));
    }

    Value::Map(json_schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    struct StubTool {
        name: String,
        description: String,
        declaration: Option<FunctionDeclaration>,
    }

    impl BaseTool for StubTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn get_declaration(&self) -> Option<FunctionDeclaration> {
            self.declaration.clone()
        }
    }

    #[test]
    fn gemini_to_json_schema_maps_a_string_type() {
        let schema = map(vec![
            ("type", Value::String("STRING".into())),
            ("description", Value::String("a name".into())),
        ]);
        let result = gemini_to_json_schema(&schema);
        assert_eq!(result.get("type").and_then(Value::as_str), Some("string"));
        assert_eq!(
            result.get("description").and_then(Value::as_str),
            Some("a name")
        );
    }

    #[test]
    fn gemini_to_json_schema_defaults_an_unspecified_type_to_null() {
        let schema = map(vec![("type", Value::String("TYPE_UNSPECIFIED".into()))]);
        let result = gemini_to_json_schema(&schema);
        assert_eq!(result.get("type").and_then(Value::as_str), Some("null"));
    }

    #[test]
    fn gemini_to_json_schema_recurses_into_array_items() {
        let schema = map(vec![
            ("type", Value::String("ARRAY".into())),
            (
                "items",
                map(vec![("type", Value::String("INTEGER".into()))]),
            ),
        ]);
        let result = gemini_to_json_schema(&schema);
        assert_eq!(
            result
                .get("items")
                .unwrap()
                .get("type")
                .and_then(Value::as_str),
            Some("integer")
        );
    }

    #[test]
    fn gemini_to_json_schema_recurses_into_object_properties() {
        let schema = map(vec![
            ("type", Value::String("OBJECT".into())),
            (
                "properties",
                map(vec![(
                    "age",
                    map(vec![("type", Value::String("INTEGER".into()))]),
                )]),
            ),
            ("required", Value::Seq(vec![Value::String("age".into())])),
        ]);
        let result = gemini_to_json_schema(&schema);
        assert_eq!(
            result
                .get("properties")
                .unwrap()
                .get("age")
                .unwrap()
                .get("type")
                .and_then(Value::as_str),
            Some("integer")
        );
        match result.get("required").unwrap() {
            Value::Seq(items) => assert_eq!(items[0].as_str(), Some("age")),
            other => panic!("expected a list, got {other:?}"),
        }
    }

    #[test]
    fn gemini_to_json_schema_maps_any_of_recursively() {
        let schema = map(vec![(
            "any_of",
            Value::Seq(vec![
                map(vec![("type", Value::String("STRING".into()))]),
                map(vec![("type", Value::String("INTEGER".into()))]),
            ]),
        )]);
        let result = gemini_to_json_schema(&schema);
        match result.get("anyOf").unwrap() {
            Value::Seq(branches) => assert_eq!(branches.len(), 2),
            other => panic!("expected a list, got {other:?}"),
        }
    }

    #[test]
    fn adk_to_mcp_tool_type_prefers_the_json_schema_field() {
        let tool = StubTool {
            name: "get_weather".to_string(),
            description: "fetches the weather".to_string(),
            declaration: Some(FunctionDeclaration {
                name: Some("get_weather".to_string()),
                description: None,
                parameters: None,
                parameters_json_schema: Some(map(vec![("type", Value::String("object".into()))])),
                response: None,
                response_json_schema: None,
            }),
        };
        let mcp_tool = adk_to_mcp_tool_type(&tool);
        assert_eq!(mcp_tool.name, "get_weather");
        assert_eq!(mcp_tool.description, "fetches the weather");
        assert_eq!(
            mcp_tool.input_schema.get("type").and_then(Value::as_str),
            Some("object")
        );
    }

    #[test]
    fn adk_to_mcp_tool_type_falls_back_to_converting_the_gemini_schema() {
        let tool = StubTool {
            name: "get_weather".to_string(),
            description: "".to_string(),
            declaration: Some(FunctionDeclaration {
                name: None,
                description: None,
                parameters: Some(map(vec![("type", Value::String("OBJECT".into()))])),
                parameters_json_schema: None,
                response: None,
                response_json_schema: None,
            }),
        };
        let mcp_tool = adk_to_mcp_tool_type(&tool);
        assert_eq!(
            mcp_tool.input_schema.get("type").and_then(Value::as_str),
            Some("object")
        );
    }

    #[test]
    fn adk_to_mcp_tool_type_returns_an_empty_schema_with_no_declaration() {
        let tool = StubTool {
            name: "noop".to_string(),
            description: "".to_string(),
            declaration: None,
        };
        let mcp_tool = adk_to_mcp_tool_type(&tool);
        assert_eq!(mcp_tool.input_schema, Value::Map(Vec::new()));
    }
}
