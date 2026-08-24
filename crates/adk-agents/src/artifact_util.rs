//! Capabilities C0262-C0264: artifact-URI parsing/construction and
//! path-safety validation, ported from
//! `google.adk.artifacts.artifact_util`.
//!
//! Pure string/regex logic, no I/O and no dependency on any concrete
//! artifact backend — nothing in this port yet constructs an
//! `InMemoryArtifactService` (C0265, still unbuilt) to exercise these
//! against, the same "the utility is real and tested, ahead of its
//! own only caller" situation as `remote_mcp_server.rs`. Builds on
//! `adk_errors::input_validation::InputValidationError`, already real.
//!
//! **`is_artifact_ref`, adapted**: the source reads `Part.file_data`
//! (a typed `FileData` with a `file_uri: str` field). This port's
//! `Part.file_data` is `Option<MediaBlobStub>`, whose fields beyond
//! `mime_type` are captured in an opaque flattened `rest: Option<Value>`
//! map (`adk-genai::content`'s own documented narrowing — no typed
//! `FileData.file_uri` field exists). `is_artifact_ref` reads
//! `"fileUri"` (the camelCase wire key `Part`'s `rename_all =
//! "camelCase"` produces) out of that map instead, the same
//! read-an-opaque-flattened-field pattern
//! `load_artifacts_tool.rs::maybe_base64_to_bytes`'s caller already
//! uses for `inline_data.rest.get("data")`.

use regex::Regex;
use std::sync::OnceLock;

use adk_errors::input_validation::InputValidationError;
use adk_genai::content::Part;

/// `artifact_util.ParsedArtifactUri`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedArtifactUri {
    pub app_name: String,
    pub user_id: String,
    pub session_id: Option<String>,
    pub filename: String,
    pub version: u64,
}

fn session_scoped_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^artifact://apps/([^/]+)/users/([^/]+)/sessions/([^/]+)/artifacts/(.+)/versions/(\d+)$",
        )
        .unwrap()
    })
}

fn user_scoped_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^artifact://apps/([^/]+)/users/([^/]+)/artifacts/(.+)/versions/(\d+)$")
            .unwrap()
    })
}

/// C0262: `artifact_util.parse_artifact_uri`.
pub fn parse_artifact_uri(uri: &str) -> Option<ParsedArtifactUri> {
    if uri.is_empty() || !uri.starts_with("artifact://") {
        return None;
    }

    if let Some(captures) = session_scoped_re().captures(uri) {
        return Some(ParsedArtifactUri {
            app_name: captures[1].to_string(),
            user_id: captures[2].to_string(),
            session_id: Some(captures[3].to_string()),
            filename: captures[4].to_string(),
            version: captures[5].parse().ok()?,
        });
    }

    if let Some(captures) = user_scoped_re().captures(uri) {
        return Some(ParsedArtifactUri {
            app_name: captures[1].to_string(),
            user_id: captures[2].to_string(),
            session_id: None,
            filename: captures[3].to_string(),
            version: captures[4].parse().ok()?,
        });
    }

    None
}

/// C0262: `artifact_util.get_artifact_uri`.
pub fn get_artifact_uri(
    app_name: &str,
    user_id: &str,
    filename: &str,
    version: u64,
    session_id: Option<&str>,
) -> String {
    match session_id {
        Some(session_id) if !session_id.is_empty() => {
            format!(
                "artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
            )
        }
        _ => format!(
            "artifact://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"
        ),
    }
}

/// C0262: `artifact_util.is_artifact_ref` — see the module doc for the
/// disclosed `file_uri`-via-flattened-`rest` adaptation.
pub fn is_artifact_ref(part: &Part) -> bool {
    part.file_data
        .as_ref()
        .and_then(|file_data| file_data.rest.as_ref())
        .and_then(|rest| rest.get("fileUri"))
        .and_then(|value| value.as_str())
        .is_some_and(|file_uri| file_uri.starts_with("artifact://"))
}

/// C0263: `artifact_util.validate_artifact_reference_scope` — the
/// security boundary preventing cross-tenant artifact-reference
/// escapes.
pub fn validate_artifact_reference_scope(
    app_name: &str,
    user_id: &str,
    session_id: Option<&str>,
    parsed_uri: &ParsedArtifactUri,
) -> Result<(), InputValidationError> {
    if parsed_uri.app_name != app_name || parsed_uri.user_id != user_id {
        return Err(InputValidationError::new(
            "Artifact references must stay within the same app and user scope.",
        ));
    }
    if let Some(uri_session_id) = &parsed_uri.session_id {
        if Some(uri_session_id.as_str()) != session_id {
            return Err(InputValidationError::new(
                "Session-scoped artifact references must stay within the same session scope.",
            ));
        }
    }
    Ok(())
}

fn is_drive_qualified(value: &str) -> bool {
    let mut chars = value.chars();
    match (chars.next(), chars.next()) {
        (Some(letter), Some(':')) => letter.is_ascii_alphabetic(),
        _ => false,
    }
}

/// C0264: `artifact_util.validate_path_segment` — rejects a
/// caller-supplied identifier that could alter the constructed path
/// (used by every backend for app/user/session identifiers).
pub fn validate_path_segment(value: &str, field_name: &str) -> Result<(), InputValidationError> {
    if value.is_empty() {
        return Err(InputValidationError::new(format!(
            "{field_name} must not be empty."
        )));
    }
    if value.contains('\0') {
        return Err(InputValidationError::new(format!(
            "{field_name} must not contain null bytes."
        )));
    }
    if value.starts_with('/') || value.starts_with('\\') {
        return Err(InputValidationError::new(format!(
            "{field_name} {value:?} must not be an absolute path or start with a slash."
        )));
    }
    if is_drive_qualified(value) {
        return Err(InputValidationError::new(format!(
            "{field_name} {value:?} must not be drive-qualified."
        )));
    }
    if value == "."
        || value == ".."
        || value
            .replace('\\', "/")
            .split('/')
            .any(|segment| segment == "..")
    {
        return Err(InputValidationError::new(format!(
            "{field_name} {value:?} must not contain traversal segments."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::MediaBlobStub;
    use rusty_serde::value::Value;

    #[test]
    fn parse_artifact_uri_parses_a_session_scoped_uri() {
        let parsed = parse_artifact_uri(
            "artifact://apps/myapp/users/user1/sessions/s1/artifacts/report.pdf/versions/3",
        )
        .unwrap();
        assert_eq!(parsed.app_name, "myapp");
        assert_eq!(parsed.user_id, "user1");
        assert_eq!(parsed.session_id.as_deref(), Some("s1"));
        assert_eq!(parsed.filename, "report.pdf");
        assert_eq!(parsed.version, 3);
    }

    #[test]
    fn parse_artifact_uri_parses_a_user_scoped_uri() {
        let parsed =
            parse_artifact_uri("artifact://apps/myapp/users/user1/artifacts/report.pdf/versions/0")
                .unwrap();
        assert_eq!(parsed.session_id, None);
        assert_eq!(parsed.version, 0);
    }

    #[test]
    fn parse_artifact_uri_returns_none_for_a_malformed_uri() {
        assert_eq!(parse_artifact_uri("not-an-artifact-uri"), None);
        assert_eq!(parse_artifact_uri(""), None);
        assert_eq!(
            parse_artifact_uri("artifact://apps/myapp/users/user1/artifacts/f/versions/notanumber"),
            None
        );
    }

    #[test]
    fn get_artifact_uri_round_trips_through_parse_artifact_uri() {
        let uri = get_artifact_uri("myapp", "user1", "report.pdf", 3, Some("s1"));
        assert_eq!(
            uri,
            "artifact://apps/myapp/users/user1/sessions/s1/artifacts/report.pdf/versions/3"
        );
        assert_eq!(parse_artifact_uri(&uri).unwrap().filename, "report.pdf");
    }

    #[test]
    fn get_artifact_uri_omits_the_session_segment_when_absent() {
        let uri = get_artifact_uri("myapp", "user1", "report.pdf", 3, None);
        assert_eq!(
            uri,
            "artifact://apps/myapp/users/user1/artifacts/report.pdf/versions/3"
        );
    }

    #[test]
    fn is_artifact_ref_is_true_for_an_artifact_scheme_file_uri() {
        let part = Part {
            file_data: Some(MediaBlobStub {
                mime_type: Some("application/pdf".to_string()),
                rest: Some(Value::Map(vec![(
                    "fileUri".to_string(),
                    Value::String(
                        "artifact://apps/myapp/users/user1/artifacts/f/versions/0".to_string(),
                    ),
                )])),
            }),
            ..Default::default()
        };
        assert!(is_artifact_ref(&part));
    }

    #[test]
    fn is_artifact_ref_is_false_for_a_non_artifact_file_uri() {
        let part = Part {
            file_data: Some(MediaBlobStub {
                mime_type: Some("application/pdf".to_string()),
                rest: Some(Value::Map(vec![(
                    "fileUri".to_string(),
                    Value::String("https://example.com/f.pdf".to_string()),
                )])),
            }),
            ..Default::default()
        };
        assert!(!is_artifact_ref(&part));
    }

    #[test]
    fn is_artifact_ref_is_false_without_file_data() {
        assert!(!is_artifact_ref(&Part::default()));
    }

    #[test]
    fn validate_artifact_reference_scope_rejects_a_cross_user_reference() {
        let parsed =
            parse_artifact_uri("artifact://apps/myapp/users/attacker/artifacts/f/versions/0")
                .unwrap();
        let err = validate_artifact_reference_scope("myapp", "victim", None, &parsed).unwrap_err();
        assert!(err.message.contains("same app and user scope"));
    }

    #[test]
    fn validate_artifact_reference_scope_rejects_a_cross_session_reference() {
        let parsed = parse_artifact_uri(
            "artifact://apps/myapp/users/user1/sessions/other-session/artifacts/f/versions/0",
        )
        .unwrap();
        let err = validate_artifact_reference_scope("myapp", "user1", Some("my-session"), &parsed)
            .unwrap_err();
        assert!(err.message.contains("same session scope"));
    }

    #[test]
    fn validate_artifact_reference_scope_accepts_a_matching_scope() {
        let parsed = parse_artifact_uri(
            "artifact://apps/myapp/users/user1/sessions/s1/artifacts/f/versions/0",
        )
        .unwrap();
        assert!(validate_artifact_reference_scope("myapp", "user1", Some("s1"), &parsed).is_ok());
    }

    #[test]
    fn validate_path_segment_accepts_a_plain_identifier() {
        assert!(validate_path_segment("user-123", "user_id").is_ok());
    }

    #[test]
    fn validate_path_segment_rejects_empty() {
        assert!(validate_path_segment("", "user_id").is_err());
    }

    #[test]
    fn validate_path_segment_rejects_null_bytes() {
        assert!(validate_path_segment("user\x001", "user_id").is_err());
    }

    #[test]
    fn validate_path_segment_rejects_absolute_paths() {
        assert!(validate_path_segment("/etc/passwd", "user_id").is_err());
        assert!(validate_path_segment("\\windows\\system32", "user_id").is_err());
    }

    #[test]
    fn validate_path_segment_rejects_drive_qualified_paths() {
        assert!(validate_path_segment("C:evil", "user_id").is_err());
    }

    #[test]
    fn validate_path_segment_rejects_traversal_segments() {
        assert!(validate_path_segment("..", "user_id").is_err());
        assert!(validate_path_segment(".", "user_id").is_err());
        assert!(validate_path_segment("a/../b", "user_id").is_err());
        assert!(validate_path_segment("a\\..\\b", "user_id").is_err());
    }
}
