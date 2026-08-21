//! Capability C0011: generic "entity not found" error.

/// Generic "entity not found" error. Ports
/// `google.adk.errors.NotFoundError`, used by the source's evaluation
/// subsystem for eval-set/persona/result lookups. Distinct from
/// [`crate::session_not_found::SessionNotFoundError`] (a separate,
/// session-specific type). Subclasses plain `Exception` in the source, not
/// `ValueError` (see [`crate::ValueErrorLike`], which this type
/// deliberately does not implement).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so
/// single-field structs like this one implement [`std::fmt::Display`] and
/// [`rusty_err::Error`] by hand rather than deriving them.
#[derive(Debug)]
pub struct NotFoundError {
    pub message: String,
}

impl NotFoundError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for NotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for NotFoundError {}

impl Default for NotFoundError {
    /// Matches the source's default constructor message.
    fn default() -> Self {
        Self::new("The requested item was not found.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_matches_source() {
        assert_eq!(
            NotFoundError::default().to_string(),
            "The requested item was not found."
        );
    }
}
