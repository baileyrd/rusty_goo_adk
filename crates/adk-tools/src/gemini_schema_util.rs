//! Part of C0489: OpenAPI/JSON-Schema → Gemini-Schema conversion, ported
//! from `google.adk.tools._gemini_schema_util`.
//!
//! **Scope boundary, disclosed**: the source's `_to_gemini_schema` ends
//! by calling `google.genai.types.Schema.from_json_schema(...)` — a
//! ~380-line method belonging to the third-party `google-genai` SDK
//! (not `google.adk` itself), which re-derefs `$ref`s a second time and
//! applies its own stricter, per-JSON-Schema-type field allow-list on
//! top of this file's own sanitization. That method lives outside
//! `google/adk-python`'s own source tree — the boundary this migration
//! ports — and this workspace has no typed Gemini `Schema`/`JSONSchema`
//! struct to begin with (`adk_genai::content::FunctionDeclaration.parameters`
//! is already just an opaque `Value`, the same representation this port
//! returns). This port therefore covers everything `_gemini_schema_util.py`
//! itself does — snake-casing, `$ref` dereferencing (with circular-ref
//! guarding), and the module's own field/format sanitization — and stops
//! at the SDK boundary: the returned `Value` may retain a few fields the
//! real SDK's `from_json_schema` would additionally prune per JSON-Schema
//! type (e.g. an `enum` left on an object-typed branch), a narrow,
//! disclosed gap rather than a silent one.

use std::sync::OnceLock;

use regex::Regex;
use rusty_serde::value::Value;

fn non_alnum_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^a-zA-Z0-9]+").unwrap())
}

fn lower_upper_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([a-z0-9])([A-Z])").unwrap())
}

fn acronym_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([A-Z]+)([A-Z][a-z])").unwrap())
}

fn underscores_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"_+").unwrap())
}

/// `_gemini_schema_util._to_snake_case` — handles lowerCamelCase,
/// UpperCamelCase, space-separated case, acronyms ("REST API"), and
/// consecutive uppercase letters.
pub fn to_snake_case(text: &str) -> String {
    let text = non_alnum_re().replace_all(text, "_");
    let text = lower_upper_re().replace_all(&text, "${1}_${2}");
    let text = acronym_re().replace_all(&text, "${1}_${2}");
    let text = text.to_lowercase();
    let text = underscores_re().replace_all(&text, "_");
    text.trim_matches('_').to_string()
}

fn map_get<'a>(entries: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn map_set(entries: &mut Vec<(String, Value)>, key: &str, value: Value) {
    if let Some(entry) = entries.iter_mut().find(|(k, _)| k == key) {
        entry.1 = value;
    } else {
        entries.push((key.to_string(), value));
    }
}

/// `_gemini_schema_util._sanitize_schema_type`.
fn sanitize_schema_type(mut entries: Vec<(String, Value)>, preserve_null_type: bool) -> Value {
    if entries.is_empty() {
        map_set(&mut entries, "type", Value::String("object".to_string()));
    }

    match map_get(&entries, "type").cloned() {
        Some(Value::Seq(types)) => {
            let types_no_null: Vec<Value> = types
                .iter()
                .filter(|t| t.as_str() != Some("null"))
                .cloned()
                .collect();
            let nullable = types_no_null.len() != types.len();
            let non_null_type = if types_no_null.iter().any(|t| t.as_str() == Some("array")) {
                "array".to_string()
            } else {
                types_no_null
                    .first()
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| "object".to_string())
            };
            let new_type = if nullable {
                Value::Seq(vec![
                    Value::String(non_null_type),
                    Value::String("null".to_string()),
                ])
            } else {
                Value::String(non_null_type)
            };
            map_set(&mut entries, "type", new_type);
        }
        Some(Value::String(s)) if s == "null" && !preserve_null_type => {
            map_set(
                &mut entries,
                "type",
                Value::Seq(vec![
                    Value::String("object".to_string()),
                    Value::String("null".to_string()),
                ]),
            );
        }
        _ => {}
    }

    let schema_type = map_get(&entries, "type").cloned();
    let is_array = match &schema_type {
        Some(Value::String(s)) => s == "array",
        Some(Value::Seq(types)) => types.iter().any(|t| t.as_str() == Some("array")),
        _ => false,
    };
    if is_array && map_get(&entries, "items").is_none() {
        map_set(
            &mut entries,
            "items",
            Value::Map(vec![(
                "type".to_string(),
                Value::String("string".to_string()),
            )]),
        );
    }

    let effective_type: Option<String> = match &schema_type {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Seq(types)) => types
            .iter()
            .find(|t| t.as_str() != Some("null"))
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    };
    if effective_type.as_deref() == Some("string") {
        if let Some(Value::Seq(enum_values)) = map_get(&entries, "enum").cloned() {
            let sanitized: Vec<Value> = enum_values
                .into_iter()
                .filter(|v| !v.is_null())
                .map(|v| match v {
                    Value::String(_) => v,
                    other => {
                        Value::String(rusty_serde::json::to_string(&other).unwrap_or_default())
                    }
                })
                .collect();
            map_set(&mut entries, "enum", Value::Seq(sanitized));
        }
    }

    Value::Map(entries)
}

/// `_gemini_schema_util._dereference_schema` — resolves `$ref` pointers,
/// supporting both `$defs` (draft 2019-09+/2020-12) and `definitions`
/// (draft-07) keywords, with `$defs` taking precedence on key collision.
pub fn dereference_schema(schema: &Value) -> Value {
    let mut defs: Vec<(String, Value)> = Vec::new();
    if let Some(Value::Map(entries)) = schema.get("definitions") {
        for (k, v) in entries {
            map_set(&mut defs, k, v.clone());
        }
    }
    if let Some(Value::Map(entries)) = schema.get("$defs") {
        for (k, v) in entries {
            map_set(&mut defs, k, v.clone());
        }
    }

    let resolved = resolve_refs(schema, &defs, &[]);
    match resolved {
        Value::Map(mut entries) => {
            entries.retain(|(k, _)| k != "$defs" && k != "definitions");
            Value::Map(entries)
        }
        other => other,
    }
}

fn resolve_refs(sub_schema: &Value, defs: &[(String, Value)], path_refs: &[String]) -> Value {
    match sub_schema {
        Value::Map(entries) => {
            if let Some(ref_value) = map_get(entries, "$ref") {
                let ref_uri = ref_value.as_str().unwrap_or_default().to_string();
                let ref_key = ref_uri.rsplit('/').next().unwrap_or(&ref_uri).to_string();

                if path_refs.iter().any(|seen| seen == &ref_uri) {
                    return Value::Map(vec![
                        ("type".to_string(), Value::String("object".to_string())),
                        (
                            "description".to_string(),
                            Value::String(format!("Circular ref to {ref_key}")),
                        ),
                    ]);
                }

                let mut new_path = path_refs.to_vec();
                new_path.push(ref_uri.clone());

                match map_get(defs, &ref_key) {
                    Some(Value::Map(def_entries)) => {
                        let mut resolved_entries = def_entries.clone();
                        for (k, v) in entries {
                            if k != "$ref" {
                                map_set(&mut resolved_entries, k, v.clone());
                            }
                        }
                        resolve_refs(&Value::Map(resolved_entries), defs, &new_path)
                    }
                    _ => sub_schema.clone(),
                }
            } else {
                Value::Map(
                    entries
                        .iter()
                        .map(|(k, v)| (k.clone(), resolve_refs(v, defs, path_refs)))
                        .collect(),
                )
            }
        }
        Value::Seq(items) => Value::Seq(
            items
                .iter()
                .map(|item| resolve_refs(item, defs, path_refs))
                .collect(),
        ),
        other => other.clone(),
    }
}

const SUPPORTED_FIELDS: &[&str] = &[
    "any_of",
    "default",
    "defs",
    "description",
    "enum",
    "format",
    "items",
    "max_items",
    "max_length",
    "max_properties",
    "maximum",
    "min_items",
    "min_length",
    "min_properties",
    "minimum",
    "one_of",
    "pattern",
    "properties",
    "ref",
    "required",
    "title",
    "type",
    "unique_items",
    "property_ordering",
];

/// `_gemini_schema_util._sanitize_schema_formats_for_gemini` — filters a
/// dereferenced schema down to the fields Gemini's `Schema` supports,
/// snake-casing wire keys along the way.
pub fn sanitize_schema_formats_for_gemini(schema: &Value, preserve_null_type: bool) -> Value {
    match schema {
        Value::Seq(items) => Value::Seq(
            items
                .iter()
                .map(|item| sanitize_schema_formats_for_gemini(item, preserve_null_type))
                .collect(),
        ),
        // JSON Schema allows boolean schemas (`true`/`false`); Gemini has no
        // equivalent for either, so both map to an unconstrained object
        // schema as a safe fallback.
        Value::Bool(_) => Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]),
        Value::Map(entries) => {
            let mut snake_case_schema: Vec<(String, Value)> = Vec::new();
            for (raw_field_name, field_value) in entries {
                let field_name = to_snake_case(raw_field_name);
                match field_name.as_str() {
                    "items" => {
                        let sanitized = sanitize_schema_formats_for_gemini(field_value, false);
                        map_set(&mut snake_case_schema, "items", sanitized);
                    }
                    "any_of" | "one_of" => {
                        let should_preserve = true;
                        let mut sanitized_branches: Vec<Value> = match field_value {
                            Value::Seq(items) => items
                                .iter()
                                .map(|item| {
                                    sanitize_schema_formats_for_gemini(item, should_preserve)
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        // `one_of` widens to `any_of` (Gemini has no `one_of`
                        // and would otherwise silently drop it, leaving the
                        // property with no type at all); a schema may carry
                        // both keywords, in either order, so accumulate
                        // rather than letting whichever comes second win.
                        let existing = match map_get(&snake_case_schema, "any_of") {
                            Some(Value::Seq(items)) => items.clone(),
                            _ => Vec::new(),
                        };
                        let mut combined = existing;
                        combined.append(&mut sanitized_branches);
                        map_set(&mut snake_case_schema, "any_of", Value::Seq(combined));
                    }
                    "properties" | "defs" if !field_value.is_null() => {
                        let sanitized = match field_value {
                            Value::Map(inner) => Value::Map(
                                inner
                                    .iter()
                                    .map(|(k, v)| {
                                        (k.clone(), sanitize_schema_formats_for_gemini(v, false))
                                    })
                                    .collect(),
                            ),
                            other => other.clone(),
                        };
                        map_set(&mut snake_case_schema, &field_name, sanitized);
                    }
                    "format" if !field_value.is_null() => {
                        let current_type = map_get(entries, "type").and_then(Value::as_str);
                        let format_value = field_value.as_str().unwrap_or_default();
                        let keep = match current_type {
                            Some("integer") | Some("number") => {
                                matches!(format_value, "int32" | "int64")
                            }
                            Some("string") => matches!(format_value, "date-time" | "enum"),
                            _ => false,
                        };
                        if keep {
                            map_set(&mut snake_case_schema, "format", field_value.clone());
                        }
                    }
                    _ if SUPPORTED_FIELDS.contains(&field_name.as_str())
                        && !field_value.is_null() =>
                    {
                        map_set(&mut snake_case_schema, &field_name, field_value.clone());
                    }
                    _ => {}
                }
            }
            sanitize_schema_type(snake_case_schema, preserve_null_type)
        }
        other => other.clone(),
    }
}

/// C0489 (partial): `_gemini_schema_util._to_gemini_schema` — converts an
/// OpenAPI v3.1 schema value into a Gemini-compatible schema value. See
/// the module doc for the SDK-boundary narrowing.
pub fn to_gemini_schema(openapi_schema: &Value) -> Value {
    let dereferenced = dereference_schema(openapi_schema);
    sanitize_schema_formats_for_gemini(&dereferenced, false)
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

    #[test]
    fn to_snake_case_handles_lower_camel_case() {
        assert_eq!(to_snake_case("camelCase"), "camel_case");
    }

    #[test]
    fn to_snake_case_handles_upper_camel_case() {
        assert_eq!(to_snake_case("UpperCamelCase"), "upper_camel_case");
    }

    #[test]
    fn to_snake_case_handles_space_separated_text() {
        assert_eq!(to_snake_case("space separated"), "space_separated");
    }

    #[test]
    fn to_snake_case_handles_acronyms() {
        assert_eq!(to_snake_case("REST API"), "rest_api");
    }

    #[test]
    fn to_snake_case_strips_a_leading_dollar_sign() {
        assert_eq!(to_snake_case("$ref"), "ref");
        assert_eq!(to_snake_case("$defs"), "defs");
    }

    #[test]
    fn dereference_schema_resolves_a_defs_ref() {
        let schema = map(vec![
            (
                "$defs",
                map(vec![(
                    "Foo",
                    map(vec![("type", Value::String("string".into()))]),
                )]),
            ),
            (
                "properties",
                map(vec![(
                    "bar",
                    map(vec![("$ref", Value::String("#/$defs/Foo".into()))]),
                )]),
            ),
        ]);
        let result = dereference_schema(&schema);
        assert!(result.get("$defs").is_none());
        let bar = result.get("properties").unwrap().get("bar").unwrap();
        assert_eq!(bar.get("type").unwrap().as_str(), Some("string"));
    }

    #[test]
    fn dereference_schema_guards_against_circular_refs() {
        let schema = map(vec![(
            "$defs",
            map(vec![(
                "Node",
                map(vec![(
                    "properties",
                    map(vec![(
                        "child",
                        map(vec![("$ref", Value::String("#/$defs/Node".into()))]),
                    )]),
                )]),
            )]),
        )]);
        let referenced = map(vec![("$ref", Value::String("#/$defs/Node".into()))]);
        let with_ref = match schema {
            Value::Map(mut entries) => {
                entries.push(("root".to_string(), referenced));
                Value::Map(entries)
            }
            _ => unreachable!(),
        };
        let result = dereference_schema(&with_ref);
        let root = result.get("root").unwrap();
        let child = root.get("properties").unwrap().get("child").unwrap();
        assert_eq!(child.get("type").unwrap().as_str(), Some("object"));
        assert!(child
            .get("description")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Circular ref"));
    }

    #[test]
    fn sanitize_schema_type_defaults_an_empty_schema_to_object() {
        let result = sanitize_schema_type(Vec::new(), false);
        assert_eq!(result.get("type").unwrap().as_str(), Some("object"));
    }

    #[test]
    fn sanitize_schema_type_collapses_a_nullable_type_list() {
        let entries = vec![(
            "type".to_string(),
            Value::Seq(vec![
                Value::String("string".into()),
                Value::String("null".into()),
            ]),
        )];
        let result = sanitize_schema_type(entries, false);
        match result.get("type").unwrap() {
            Value::Seq(types) => {
                assert_eq!(types[0].as_str(), Some("string"));
                assert_eq!(types[1].as_str(), Some("null"));
            }
            other => panic!("expected a list, got {other:?}"),
        }
    }

    #[test]
    fn sanitize_schema_type_prefers_array_when_present_in_a_type_list() {
        let entries = vec![(
            "type".to_string(),
            Value::Seq(vec![
                Value::String("string".into()),
                Value::String("array".into()),
            ]),
        )];
        let result = sanitize_schema_type(entries, false);
        assert_eq!(result.get("type").unwrap().as_str(), Some("array"));
    }

    #[test]
    fn sanitize_schema_type_widens_a_null_type_unless_preserved() {
        let entries = vec![("type".to_string(), Value::String("null".to_string()))];
        let widened = sanitize_schema_type(entries.clone(), false);
        match widened.get("type").unwrap() {
            Value::Seq(types) => {
                assert_eq!(types[0].as_str(), Some("object"));
                assert_eq!(types[1].as_str(), Some("null"));
            }
            other => panic!("expected a list, got {other:?}"),
        }
        let preserved = sanitize_schema_type(entries, true);
        assert_eq!(preserved.get("type").unwrap().as_str(), Some("null"));
    }

    #[test]
    fn sanitize_schema_type_defaults_array_items_to_string() {
        let entries = vec![("type".to_string(), Value::String("array".to_string()))];
        let result = sanitize_schema_type(entries, false);
        assert_eq!(
            result.get("items").unwrap().get("type").unwrap().as_str(),
            Some("string")
        );
    }

    #[test]
    fn sanitize_schema_type_stringifies_non_string_enum_values_on_string_type() {
        let entries = vec![
            ("type".to_string(), Value::String("string".to_string())),
            (
                "enum".to_string(),
                Value::Seq(vec![Value::String("a".into()), Value::Int(1), Value::Null]),
            ),
        ];
        let result = sanitize_schema_type(entries, false);
        match result.get("enum").unwrap() {
            Value::Seq(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0].as_str(), Some("a"));
                assert_eq!(values[1].as_str(), Some("1"));
            }
            other => panic!("expected a list, got {other:?}"),
        }
    }

    #[test]
    fn sanitize_schema_formats_drops_additional_properties() {
        let schema = map(vec![
            ("type", Value::String("object".into())),
            ("additionalProperties", Value::Bool(false)),
        ]);
        let result = sanitize_schema_formats_for_gemini(&schema, false);
        assert!(result.get("additional_properties").is_none());
    }

    #[test]
    fn sanitize_schema_formats_widens_one_of_to_any_of() {
        let schema = map(vec![(
            "oneOf",
            Value::Seq(vec![
                map(vec![("type", Value::String("string".into()))]),
                map(vec![("type", Value::String("integer".into()))]),
            ]),
        )]);
        let result = sanitize_schema_formats_for_gemini(&schema, false);
        match result.get("any_of").unwrap() {
            Value::Seq(branches) => assert_eq!(branches.len(), 2),
            other => panic!("expected a list, got {other:?}"),
        }
        assert!(result.get("one_of").is_none());
    }

    #[test]
    fn sanitize_schema_formats_accumulates_any_of_and_one_of() {
        let schema = map(vec![
            (
                "anyOf",
                Value::Seq(vec![map(vec![("type", Value::String("string".into()))])]),
            ),
            (
                "oneOf",
                Value::Seq(vec![map(vec![("type", Value::String("integer".into()))])]),
            ),
        ]);
        let result = sanitize_schema_formats_for_gemini(&schema, false);
        match result.get("any_of").unwrap() {
            Value::Seq(branches) => assert_eq!(branches.len(), 2),
            other => panic!("expected a list, got {other:?}"),
        }
    }

    #[test]
    fn sanitize_schema_formats_keeps_only_gemini_supported_string_formats() {
        let date_time = map(vec![
            ("type", Value::String("string".into())),
            ("format", Value::String("date-time".into())),
        ]);
        assert_eq!(
            sanitize_schema_formats_for_gemini(&date_time, false)
                .get("format")
                .and_then(Value::as_str),
            Some("date-time")
        );

        let email = map(vec![
            ("type", Value::String("string".into())),
            ("format", Value::String("email".into())),
        ]);
        assert!(sanitize_schema_formats_for_gemini(&email, false)
            .get("format")
            .is_none());
    }

    #[test]
    fn sanitize_schema_formats_keeps_only_gemini_supported_numeric_formats() {
        let int32 = map(vec![
            ("type", Value::String("integer".into())),
            ("format", Value::String("int32".into())),
        ]);
        assert_eq!(
            sanitize_schema_formats_for_gemini(&int32, false)
                .get("format")
                .and_then(Value::as_str),
            Some("int32")
        );

        let float_format = map(vec![
            ("type", Value::String("number".into())),
            ("format", Value::String("float".into())),
        ]);
        assert!(sanitize_schema_formats_for_gemini(&float_format, false)
            .get("format")
            .is_none());
    }

    #[test]
    fn sanitize_schema_formats_maps_boolean_schemas_to_an_object_schema() {
        let allow_anything = Value::Bool(true);
        assert_eq!(
            sanitize_schema_formats_for_gemini(&allow_anything, false)
                .get("type")
                .and_then(Value::as_str),
            Some("object")
        );
        let reject_everything = Value::Bool(false);
        assert_eq!(
            sanitize_schema_formats_for_gemini(&reject_everything, false)
                .get("type")
                .and_then(Value::as_str),
            Some("object")
        );
    }

    #[test]
    fn sanitize_schema_formats_recurses_into_properties() {
        let schema = map(vec![
            ("type", Value::String("object".into())),
            (
                "properties",
                map(vec![(
                    "userName",
                    map(vec![("type", Value::String("string".into()))]),
                )]),
            ),
        ]);
        let result = sanitize_schema_formats_for_gemini(&schema, false);
        // Property keys themselves are NOT snake-cased -- only schema
        // keyword names are.
        assert_eq!(
            result
                .get("properties")
                .unwrap()
                .get("userName")
                .unwrap()
                .get("type")
                .unwrap()
                .as_str(),
            Some("string")
        );
    }

    #[test]
    fn to_gemini_schema_dereferences_then_sanitizes() {
        let schema = map(vec![
            (
                "$defs",
                map(vec![(
                    "Foo",
                    map(vec![("type", Value::String("string".into()))]),
                )]),
            ),
            (
                "properties",
                map(vec![(
                    "bar",
                    map(vec![
                        ("$ref", Value::String("#/$defs/Foo".into())),
                        ("extraneousField", Value::String("dropped".into())),
                    ]),
                )]),
            ),
        ]);
        let result = to_gemini_schema(&schema);
        assert!(result.get("defs").is_none());
        let bar = result.get("properties").unwrap().get("bar").unwrap();
        assert_eq!(bar.get("type").unwrap().as_str(), Some("string"));
        assert!(bar.get("extraneous_field").is_none());
    }

    /// End-to-end cross-check against the real `google.adk.tools.
    /// _gemini_schema_util` source (imported directly from the checked-out
    /// `google/adk-python` repo and run locally, not reconstructed from
    /// memory) — `_dereference_schema` piped into
    /// `_sanitize_schema_formats_for_gemini`, which is exactly this port's
    /// own `to_gemini_schema` boundary (see the module doc for why the
    /// SDK-internal `Schema.from_json_schema` step downstream of that
    /// isn't included). 11 fixtures spanning `$ref`/`$defs` resolution,
    /// circular refs (including the trickier case where the top-level
    /// schema itself is a `$ref` sibling to `$defs`), nullable type
    /// lists, `oneOf`→`anyOf` widening, camelCase key conversion,
    /// default array items, boolean sub-schemas, mixed-type enum
    /// coercion, and per-type format allow-listing.
    #[test]
    fn matches_the_real_gemini_schema_util_end_to_end() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "simple_object",
                r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}"#,
                r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}"#,
            ),
            (
                "with_ref",
                r##"{"$defs":{"Address":{"type":"object","properties":{"city":{"type":"string"}}}},"type":"object","properties":{"home":{"$ref":"#/$defs/Address"}}}"##,
                r#"{"type":"object","properties":{"home":{"type":"object","properties":{"city":{"type":"string"}}}}}"#,
            ),
            (
                "nullable_type_list",
                r#"{"type":["string","null"],"description":"a nullable string"}"#,
                r#"{"type":["string","null"],"description":"a nullable string"}"#,
            ),
            (
                "one_of_widened",
                r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#,
                r#"{"any_of":[{"type":"string"},{"type":"integer"}]}"#,
            ),
            (
                "camel_case_keys",
                r#"{"type":"object","properties":{"userName":{"type":"string","minLength":3,"maxLength":20}},"additionalProperties":false}"#,
                r#"{"type":"object","properties":{"userName":{"type":"string","min_length":3,"max_length":20}}}"#,
            ),
            (
                "array_with_items",
                r#"{"type":"array","items":{"type":"number","format":"float"}}"#,
                r#"{"type":"array","items":{"type":"number"}}"#,
            ),
            (
                "enum_mixed",
                r#"{"type":"string","enum":["a",1,null,"b"]}"#,
                r#"{"type":"string","enum":["a","1","b"]}"#,
            ),
            (
                "unsupported_format",
                r#"{"type":"string","format":"email"}"#,
                r#"{"type":"string"}"#,
            ),
            (
                "int32_format",
                r#"{"type":"integer","format":"int32"}"#,
                r#"{"type":"integer","format":"int32"}"#,
            ),
            (
                "circular",
                r##"{"$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}}}},"$ref":"#/$defs/Node"}"##,
                r#"{"type":"object","properties":{"next":{"type":"object","description":"Circular ref to Node"}}}"#,
            ),
            (
                "bool_schemas",
                r#"{"type":"object","properties":{"anything":true,"nothing":false}}"#,
                r#"{"type":"object","properties":{"anything":{"type":"object"},"nothing":{"type":"object"}}}"#,
            ),
        ];

        for (name, input, expected) in cases {
            let input: Value = rusty_serde::json::from_str(input).unwrap();
            let expected: Value = rusty_serde::json::from_str(expected).unwrap();
            let got = to_gemini_schema(&input);
            assert!(
                values_equal_ignoring_map_order(&got, &expected),
                "case {name:?}: got {got:?}, expected {expected:?}"
            );
        }
    }

    /// `Value::Map`'s derived `PartialEq` is order-sensitive (it's a
    /// `Vec<(String, Value)>`), but JSON object key order isn't
    /// semantically meaningful here -- this port's own map-manipulation
    /// order doesn't need to match the source's dict-insertion order key
    /// for key. Compares maps as unordered key sets instead.
    fn values_equal_ignoring_map_order(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Map(a_entries), Value::Map(b_entries)) => {
                a_entries.len() == b_entries.len()
                    && a_entries.iter().all(|(key, a_value)| {
                        b_entries
                            .iter()
                            .find(|(k, _)| k == key)
                            .is_some_and(|(_, b_value)| {
                                values_equal_ignoring_map_order(a_value, b_value)
                            })
                    })
            }
            (Value::Seq(a_items), Value::Seq(b_items)) => {
                a_items.len() == b_items.len()
                    && a_items
                        .iter()
                        .zip(b_items.iter())
                        .all(|(x, y)| values_equal_ignoring_map_order(x, y))
            }
            _ => a == b,
        }
    }
}
