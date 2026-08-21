//! Capability C0010: user-input validation failure.

/// User input failed validation. Ports
/// `google.adk.errors.InputValidationError`, widely used by the source's
/// artifacts subsystem for path-traversal/null-byte/invalid-scope/
/// malformed-URI rejections. Subclasses `ValueError` in the source (see
/// [`crate::ValueErrorLike`]).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so
/// single-field structs like this one implement [`std::fmt::Display`] and
/// [`rusty_err::Error`] by hand rather than deriving them.
#[derive(Debug)]
pub struct InputValidationError {
    pub message: String,
}

impl InputValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for InputValidationError {
    /// Matches the source's default constructor message.
    fn default() -> Self {
        Self::new("Invalid input.")
    }
}

impl std::fmt::Display for InputValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for InputValidationError {}
impl crate::ValueErrorLike for InputValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_matches_source() {
        assert_eq!(
            InputValidationError::default().to_string(),
            "Invalid input."
        );
    }

    #[test]
    fn custom_message_overrides_default() {
        assert_eq!(
            InputValidationError::new("path escapes working directory: ../etc/passwd").to_string(),
            "path escapes working directory: ../etc/passwd"
        );
    }
}
