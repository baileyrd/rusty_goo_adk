//! Capability C0829 (partial): `BaseExampleProvider`, ported from
//! `google.adk.examples.base_example_provider`.

use crate::example::Example;

/// C0829: the interface for providing examples for a given query.
pub trait BaseExampleProvider {
    fn get_examples(&self, query: &str) -> Vec<Example>;
}
