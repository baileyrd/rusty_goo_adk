//! Capability C0012: session lookup failure.

/// A session could not be found. Ports
/// `google.adk.errors.SessionNotFoundError`, used across all session
/// backends and the CLI/HTTP surface. Subclasses `ValueError` in the
/// source (see [`crate::ValueErrorLike`]).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so
/// single-field structs like this one implement [`std::fmt::Display`] and
/// [`rusty_err::Error`] by hand rather than deriving them.
#[derive(Debug)]
pub struct SessionNotFoundError {
    pub message: String,
}

impl SessionNotFoundError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for SessionNotFoundError {
    /// Matches the source's default constructor message.
    fn default() -> Self {
        Self::new("Session not found.")
    }
}

impl std::fmt::Display for SessionNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for SessionNotFoundError {}
impl crate::ValueErrorLike for SessionNotFoundError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_matches_source() {
        assert_eq!(
            SessionNotFoundError::default().to_string(),
            "Session not found."
        );
    }
}
