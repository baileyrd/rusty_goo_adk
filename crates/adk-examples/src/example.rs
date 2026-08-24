//! Capability C0829 (partial): `Example`, ported from
//! `google.adk.examples.example`.

use adk_genai::content::Content;

/// C0829: a few-shot example — an `input` [`Content`] and its expected
/// `output` [`Content`] sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub input: Content,
    pub output: Vec<Content>,
}

impl Example {
    pub fn new(input: Content, output: Vec<Content>) -> Self {
        Self { input, output }
    }
}
