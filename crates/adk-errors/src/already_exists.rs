//! Capability C0009: generic "entity already exists" error.

/// Generic "entity already exists" error. Ports
/// `google.adk.errors.AlreadyExistsError`, which subclasses plain
/// `Exception` in the source (not `ValueError` — see
/// [`crate::ValueErrorLike`], which this type deliberately does not
/// implement).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so
/// single-field structs like this one implement [`std::fmt::Display`] and
/// [`rusty_err::Error`] by hand rather than deriving them.
#[derive(Debug)]
pub struct AlreadyExistsError {
    pub message: String,
}

impl AlreadyExistsError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for AlreadyExistsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for AlreadyExistsError {}

impl Default for AlreadyExistsError {
    /// Matches the source's default constructor message.
    fn default() -> Self {
        Self::new("The resource already exists.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_message_matches_source() {
        assert_eq!(
            AlreadyExistsError::default().to_string(),
            "The resource already exists."
        );
    }

    #[test]
    fn custom_message_overrides_default() {
        assert_eq!(
            AlreadyExistsError::new("session already exists").to_string(),
            "session already exists"
        );
    }
}
