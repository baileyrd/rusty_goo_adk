//! C0614: `EvalSetsManager` path-traversal hardening, ported from
//! `google.adk.evaluation._path_validation`. Applied to
//! `app_name`/`eval_set_id`/`eval_case_id`/`eval_set_result_id` before any
//! of those values touch a filesystem path.

use adk_errors::input_validation::InputValidationError;

/// `_path_validation.validate_path_segment` — rejects a value that could
/// alter a filesystem path.
///
/// **Adaptation**: the source raises a plain `ValueError`; this port uses
/// [`InputValidationError`] (`ValueErrorLike`), the same typed stand-in
/// this codebase already uses for the artifacts subsystem's own
/// path-traversal/null-byte rejections (`InputValidationError`'s own doc:
/// "widely used by the source's artifacts subsystem") — the identical
/// failure category, just reached from a different subsystem here.
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
    if value.contains('/') || value.contains('\\') {
        return Err(InputValidationError::new(format!(
            "{field_name} {value:?} must not contain path separators."
        )));
    }
    if value == "." || value == ".." {
        return Err(InputValidationError::new(format!(
            "{field_name} {value:?} must not contain traversal segments."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_plain_identifier() {
        assert!(validate_path_segment("my_app_123", "app_name").is_ok());
    }

    #[test]
    fn rejects_empty() {
        assert!(validate_path_segment("", "app_name").is_err());
    }

    #[test]
    fn rejects_null_bytes() {
        assert!(validate_path_segment("a\0b", "app_name").is_err());
    }

    #[test]
    fn rejects_forward_slash() {
        assert!(validate_path_segment("a/b", "app_name").is_err());
    }

    #[test]
    fn rejects_backslash() {
        assert!(validate_path_segment("a\\b", "app_name").is_err());
    }

    #[test]
    fn rejects_dot_traversal_segments() {
        assert!(validate_path_segment(".", "app_name").is_err());
        assert!(validate_path_segment("..", "app_name").is_err());
    }

    #[test]
    fn allows_a_value_that_merely_contains_dots() {
        // Only the exact segments "." and ".." are rejected -- a value
        // like "..hidden" or "v1.2" is a legitimate identifier.
        assert!(validate_path_segment("v1.2", "app_name").is_ok());
    }
}
