//! Capability C0680: telemetry JSON serialization helpers, ported from
//! `google.adk.telemetry._serialization` (`safe_json_serialize`/
//! `serialize_content`) plus the sibling no-whitespace variant
//! duplicated in `telemetry/_experimental_semconv.py`
//! (`_safe_json_serialize_no_whitespaces`, used for span attributes
//! where a compact wire form matters). All three exist so a
//! span/log-attribute writer can serialize an arbitrary value for
//! telemetry output without ever raising — a failed serialization
//! degrades to a literal `"<not serializable>"` string.
//!
//! **`AnyValue`, narrowed**: the source's `serialize_content` returns
//! `opentelemetry.util.types.AnyValue` — no OTel span/attribute
//! machinery exists in this port (the same scope cut
//! `adk-agents::telemetry_context`/`schema_version`, C0651/C0652/C0679,
//! already discloses), so this returns [`rusty_serde::value::Value`]
//! instead, which is exactly the same JSON-shaped union `AnyValue`
//! restricts itself to.
//!
//! **Per-value `default` hook, not portable**: `safe_json_serialize`'s
//! `_default` callback lets `json.dumps` recover from an individual
//! non-serializable object nested *anywhere* inside an otherwise-normal
//! structure — the rest of the structure still serializes; only that
//! one leaf falls back. Rust's `Serialize` trait bound is a
//! compile-time guarantee that a type already knows how to serialize
//! itself in full, so there's no equivalent "serialize what you can,
//! substitute a fallback for what you can't" mid-traversal recovery —
//! [`safe_json_serialize`] can only fall back for the *whole* call, not
//! a nested leaf. In practice this rarely (if ever) fires for this
//! port's own derived `Serialize` impls, the same "the type system
//! already rules out what the source's runtime check exists to catch"
//! situation `session_util::make_json_safe_state` already discloses.
//!
//! **`safe_json_serialize` vs. `_safe_json_serialize_no_whitespaces`,
//! collapsed**: the source's default `json.dumps` separators include a
//! space after `,`/`:`; the no-whitespace variant passes
//! `separators=(',', ':')` for a compact form. `rusty_serde::json::to_string`
//! only ever emits the compact form (no `to_string_pretty`/custom-separator
//! option exists in this workspace's `rusty_serde`) — so both functions
//! are kept as separate, named call sites for API parity with the
//! source, but produce byte-identical output in this port. A disclosed,
//! low-severity divergence: telemetry payloads that expect the source's
//! whitespace-including default form will instead always see the
//! compact one.
//!
//! **`serialize_content`'s `BaseModel`/`ContentUnion` dispatch,
//! adapted**: the source dispatches on `isinstance(content, ...)` at
//! runtime across `types.Content | str | BaseModel | list | ...`. This
//! port takes an explicit [`ContentUnion`] enum instead, the same
//! "caller already knows which shape they hold" adaptation
//! `content_utils::ToUserContentInput` already established for an
//! identical union-dispatch problem. [`ContentUnion::Content`] maps
//! through `model_dump()`'s Rust analog (`rusty_serde::json::to_value`,
//! a structured [`Value`], not a JSON string); the source's final
//! "anything else" branch (`safe_json_serialize(content)`, which
//! returns a JSON *string*, not a structured value — the source's own
//! `AnyValue` return type is intentionally inconsistent this way) is
//! covered by [`ContentUnion::Value`].

use rusty_serde::value::Value;
use rusty_serde::Serialize;

use crate::content::Content;

/// The literal fallback both serialization helpers below return when
/// the underlying serializer fails, matching the source's own
/// `"<not serializable>"` string exactly.
const NOT_SERIALIZABLE: &str = "<not serializable>";

/// C0680: `_serialization.safe_json_serialize` — serializes `value` to
/// a JSON string, falling back to [`NOT_SERIALIZABLE`] rather than
/// propagating an error. See the module doc for the disclosed
/// per-value-fallback and whitespace narrowings.
pub fn safe_json_serialize<T: Serialize + ?Sized>(value: &T) -> String {
    rusty_serde::json::to_string(value).unwrap_or_else(|_| NOT_SERIALIZABLE.to_string())
}

/// C0680: `_experimental_semconv._safe_json_serialize_no_whitespaces` —
/// same fallback contract as [`safe_json_serialize`]. See the module
/// doc for why both collapse to identical output in this port.
pub fn safe_json_serialize_no_whitespaces<T: Serialize + ?Sized>(value: &T) -> String {
    safe_json_serialize(value)
}

/// `_serialization.serialize_content`'s input shape — see the module
/// doc for why this replaces the source's runtime `isinstance`
/// dispatch across `types.ContentUnion`.
pub enum ContentUnion {
    Content(Content),
    Text(String),
    List(Vec<ContentUnion>),
    /// Covers the source's `BaseModel`/anything-else fallback branch —
    /// serialized via [`safe_json_serialize`] into a JSON *string*, not
    /// a structured value, matching the source's own `AnyValue`-typed
    /// inconsistency between this branch and [`ContentUnion::Content`].
    Value(Value),
}

/// C0680: `_serialization.serialize_content` — serializes a
/// `ContentUnion`-shaped value into an OTel-attribute-friendly
/// [`Value`]. `None` is preserved; a [`ContentUnion::Content`] dumps
/// via `rusty_serde::json::to_value` (structured, not stringified); a
/// [`ContentUnion::Text`] is returned as-is; a [`ContentUnion::List`]
/// recurses over its elements; anything else falls back to
/// [`safe_json_serialize`], wrapped as [`Value::String`].
pub fn serialize_content(content: Option<ContentUnion>) -> Value {
    match content {
        None => Value::Null,
        Some(ContentUnion::Content(content)) => rusty_serde::json::to_value(&content)
            .unwrap_or_else(|_| Value::String(NOT_SERIALIZABLE.to_string())),
        Some(ContentUnion::Text(text)) => Value::String(text),
        Some(ContentUnion::List(items)) => Value::Seq(
            items
                .into_iter()
                .map(|item| serialize_content(Some(item)))
                .collect(),
        ),
        Some(ContentUnion::Value(value)) => Value::String(safe_json_serialize(&value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Part;

    #[test]
    fn safe_json_serialize_encodes_a_map_compactly() {
        let value = Value::Map(vec![("k".to_string(), Value::Int(1))]);
        assert_eq!(safe_json_serialize(&value), r#"{"k":1}"#);
    }

    #[test]
    fn safe_json_serialize_encodes_a_string() {
        assert_eq!(
            safe_json_serialize(&Value::String("hi".to_string())),
            "\"hi\""
        );
    }

    #[test]
    fn safe_json_serialize_no_whitespaces_matches_safe_json_serialize() {
        let value = Value::Seq(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(
            safe_json_serialize_no_whitespaces(&value),
            safe_json_serialize(&value)
        );
    }

    #[test]
    fn serialize_content_preserves_none() {
        assert_eq!(serialize_content(None), Value::Null);
    }

    #[test]
    fn serialize_content_returns_a_string_as_is() {
        assert_eq!(
            serialize_content(Some(ContentUnion::Text("hello".to_string()))),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn serialize_content_dumps_a_content_as_a_structured_value() {
        let content = Content::new("user", vec![Part::text("hi")]);
        let result = serialize_content(Some(ContentUnion::Content(content)));
        assert!(matches!(result, Value::Map(_)));
    }

    #[test]
    fn serialize_content_recurses_over_a_list() {
        let result = serialize_content(Some(ContentUnion::List(vec![
            ContentUnion::Text("a".to_string()),
            ContentUnion::Text("b".to_string()),
        ])));
        assert_eq!(
            result,
            Value::Seq(vec![
                Value::String("a".to_string()),
                Value::String("b".to_string())
            ])
        );
    }

    #[test]
    fn serialize_content_falls_back_to_a_serialized_string_for_the_value_variant() {
        let value = Value::Map(vec![("k".to_string(), Value::Int(1))]);
        let result = serialize_content(Some(ContentUnion::Value(value)));
        assert_eq!(result, Value::String(r#"{"k":1}"#.to_string()));
    }
}
