//! Capability C0008: optimistic-concurrency session-write conflict.

/// Raised when a session write loses an optimistic-concurrency race — the
/// stored session's update marker no longer matches what the writer last
/// read. Ports `google.adk.errors.StaleSessionError`, which the source
/// deliberately subclasses from `ValueError` for backward compatibility
/// (see [`crate::ValueErrorLike`]).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so
/// single-field structs like this one implement [`std::fmt::Display`] and
/// [`rusty_err::Error`] by hand rather than deriving them.
#[derive(Debug)]
pub struct StaleSessionError {
    pub message: String,
}

impl StaleSessionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for StaleSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for StaleSessionError {}
impl crate::ValueErrorLike for StaleSessionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_the_given_message() {
        let err = StaleSessionError::new("session abc123 was updated concurrently");
        assert_eq!(err.message, "session abc123 was updated concurrently");
        assert_eq!(err.to_string(), "session abc123 was updated concurrently");
    }
}
