//! Capabilities C0013 (`ToolErrorType`) and C0014 (`ToolExecutionError`).

/// Semantic HTTP-style error-type taxonomy for tool execution, aligned to
/// OpenTelemetry `error.type` semantics. Ports
/// `google.adk.errors.ToolErrorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorType {
    BadRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    RequestTimeout,
    Internal,
    BadGateway,
    ServiceUnavailable,
    GatewayTimeout,
}

impl ToolErrorType {
    /// The wire value used as the OTel `error.type` span attribute —
    /// SCREAMING_SNAKE_CASE, matching the source's string-enum values.
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolErrorType::BadRequest => "BAD_REQUEST",
            ToolErrorType::Unauthorized => "UNAUTHORIZED",
            ToolErrorType::Forbidden => "FORBIDDEN",
            ToolErrorType::NotFound => "NOT_FOUND",
            ToolErrorType::RequestTimeout => "REQUEST_TIMEOUT",
            ToolErrorType::Internal => "INTERNAL_SERVER_ERROR",
            ToolErrorType::BadGateway => "BAD_GATEWAY",
            ToolErrorType::ServiceUnavailable => "SERVICE_UNAVAILABLE",
            ToolErrorType::GatewayTimeout => "GATEWAY_TIMEOUT",
        }
    }
}

impl std::fmt::Display for ToolErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A tool call failed. Ports `google.adk.errors.ToolExecutionError`, used
/// to populate the `error.type` span attribute in tool-call telemetry.
/// Carries a message and an optional error type; the source also accepts
/// an arbitrary string in place of the enum (for error types the enum
/// doesn't cover) — [`ToolExecutionError::error_type_str`] mirrors that by
/// storing the raw string form regardless of which constructor was used.
/// Subclasses plain `Exception` in the source, not `ValueError` (see
/// [`crate::ValueErrorLike`], which this type deliberately does not
/// implement).
///
/// `rusty_err::Error`'s `#[derive]` only supports enums today, so this
/// struct implements [`std::fmt::Display`] and [`rusty_err::Error`] by
/// hand rather than deriving them.
#[derive(Debug)]
pub struct ToolExecutionError {
    pub message: String,
    error_type: Option<String>,
}

impl std::fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl rusty_err::Error for ToolExecutionError {}

impl ToolExecutionError {
    /// Constructs with a typed [`ToolErrorType`] (or `None`).
    pub fn new(message: impl Into<String>, error_type: Option<ToolErrorType>) -> Self {
        Self {
            message: message.into(),
            error_type: error_type.map(|t| t.as_str().to_string()),
        }
    }

    /// Constructs with an arbitrary error-type string not covered by
    /// [`ToolErrorType`] — mirrors the source accepting a plain `str` in
    /// place of the enum.
    pub fn with_error_type_str(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: Some(error_type.into()),
        }
    }

    /// The error-type string, if one was set — regardless of whether it
    /// came from a [`ToolErrorType`] or a raw string.
    pub fn error_type_str(&self) -> Option<&str> {
        self.error_type.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_error_type_stores_its_string_value() {
        let err = ToolExecutionError::new("timed out", Some(ToolErrorType::RequestTimeout));
        assert_eq!(err.error_type_str(), Some("REQUEST_TIMEOUT"));
        assert_eq!(err.to_string(), "timed out");
    }

    #[test]
    fn omitted_error_type_is_none() {
        let err = ToolExecutionError::new("boom", None);
        assert_eq!(err.error_type_str(), None);
    }

    #[test]
    fn arbitrary_string_error_type_is_preserved_verbatim() {
        let err = ToolExecutionError::with_error_type_str("boom", "SOMETHING_CUSTOM");
        assert_eq!(err.error_type_str(), Some("SOMETHING_CUSTOM"));
    }
}
