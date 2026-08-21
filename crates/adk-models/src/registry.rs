//! Capabilities C0107-C0108, C0110: `LLMRegistry`, ported from
//! `google.adk.models.registry`.
//!
//! **Deferred**: C0109 (the LiteLLM provider-list fallback — checking an
//! unmatched `provider/model` string against `litellm.provider_list`) needs
//! a LiteLLM-equivalent Rust crate that doesn't exist; `resolve` accepts an
//! optional known-provider list so that integration has a seam to plug
//! into later, but nothing populates it yet. C0111 (the real lazy provider
//! table — Gemini/Claude/LiteLlm/etc.) and C0112/C0113 (their
//! `models/__init__.py` lazy-`__getattr__`/registration-order details) are
//! about *registering real backends*, which are Phase 3 batch 2 (needs an
//! HTTP client decision) — this file is the dispatch *mechanism*, exercised
//! here with test-double factories, not real registrations.
//!
//! **Adaptation**: the source's `_register_lazy` defers a class's *module
//! import* until first resolved (so an optional dependency like
//! `anthropic`/`litellm` isn't imported at package-load time). Rust has no
//! runtime dynamic-module-loading equivalent, but it has something
//! stronger for the same goal: a Cargo feature flag can exclude the
//! backend's code from the build entirely unless enabled, rather than just
//! deferring an import that still has to succeed at some point. Real
//! backends (batch 2) are expected to register behind such a feature
//! rather than through a lazy-module-path string.

use std::sync::Arc;

use regex::Regex;

use crate::base_llm::BaseLlm;

pub type LlmFactory = Arc<dyn Fn(&str) -> Box<dyn BaseLlm> + Send + Sync>;

#[derive(Debug, rusty_err::Error)]
pub enum RegistryError {
    #[error("invalid model-name regex `{0}`: {1}")]
    InvalidPattern(String, String),
    #[error("Model {0} not found.")]
    ModelNotFound(String),
    #[error(
        "Model {0} not found.\n\nClaude models require the anthropic package.\nInstall it with: pip install google-adk[extensions]\nOr: pip install anthropic>=0.43.0"
    )]
    ClaudeModelNotFound(String),
    #[error(
        "Model {0} not found.\n\nProvider-style models (e.g., \"provider/model-name\") require the litellm package.\nInstall it with: pip install google-adk[extensions]\nOr: pip install litellm>=1.75.5\n\nSupported providers include: openai, groq, anthropic, and 100+ others.\nSee https://docs.litellm.ai/docs/providers for a full list."
    )]
    LiteLlmModelNotFound(String),
}

struct Entry {
    pattern: Regex,
    class_name: String,
    factory: LlmFactory,
}

/// Registry mapping a model name to a `BaseLlm` factory.
///
/// **Adaptation**: an instantiable struct rather than the source's
/// process-wide static dict + staticmethods — real global registration
/// (populating one shared registry with every built-in backend at
/// startup) is exactly what C0111 defers to batch 2; until then, an
/// instance per caller (or per test) avoids global mutable state that
/// would otherwise make tests interfere with each other's registrations.
#[derive(Default)]
pub struct LlmRegistry {
    entries: Vec<Entry>,
    /// Known `provider/model`-style providers a LiteLLM-equivalent would
    /// recognize — empty until such an integration exists (see the module
    /// doc's C0109 deferral).
    known_litellm_providers: Vec<String>,
}

impl LlmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// C0107: registers a factory for every regex `llm_cls_supported_models`
    /// reports — mirrors `LLMRegistry.register(llm_cls)`, taking the
    /// `supported_models()` list and factory explicitly since Rust has no
    /// runtime access to "all regexes a trait impl's `supported_models`
    /// reports" without a concrete type in hand.
    pub fn register(
        &mut self,
        model_name_regexes: &[&str],
        class_name: &str,
        factory: LlmFactory,
    ) -> Result<(), RegistryError> {
        for regex in model_name_regexes {
            self.register_one(regex, class_name, factory.clone())?;
        }
        Ok(())
    }

    fn register_one(
        &mut self,
        model_name_regex: &str,
        class_name: &str,
        factory: LlmFactory,
    ) -> Result<(), RegistryError> {
        let pattern = Regex::new(&format!("^(?:{model_name_regex})$")).map_err(|e| {
            RegistryError::InvalidPattern(model_name_regex.to_string(), e.to_string())
        })?;
        // Matches the source's "last registration for a regex wins" update
        // semantics.
        self.entries
            .retain(|e| e.pattern.as_str() != pattern.as_str());
        self.entries.push(Entry {
            pattern,
            class_name: class_name.to_string(),
            factory,
        });
        Ok(())
    }

    fn parse_model(model: &str) -> (Option<&str>, &str) {
        match model.split_once(':') {
            Some((prefix, actual)) => (Some(prefix), actual),
            None => (None, model),
        }
    }

    /// C0108: whether an explicit-class-override prefix (e.g. `"lite"` in
    /// `"lite:openai/gpt-4o"`) matches a class name — case-insensitive, and
    /// a trailing `"Llm"` suffix on the class name is stripped before
    /// comparing.
    fn match_prefix(prefix: &str, class_name: &str) -> bool {
        let prefix_lower = prefix.to_lowercase();
        let class_lower = class_name.to_lowercase();
        let stripped = class_lower.strip_suffix("llm").unwrap_or(&class_lower);
        prefix_lower == stripped || prefix_lower == class_lower
    }

    /// C0107/C0108: resolves `model` to a registered factory. Supports the
    /// `prefix:model` override syntax (bypassing regex resolution
    /// entirely), then falls through to regex full-match, then (if
    /// `known_litellm_providers` has been populated) a `provider/model`
    /// fallback.
    pub fn resolve(&self, model: &str) -> Result<&LlmFactory, RegistryError> {
        self.resolve_entry(model).map(|entry| &entry.factory)
    }

    fn resolve_entry(&self, model: &str) -> Result<&Entry, RegistryError> {
        let (prefix, _) = Self::parse_model(model);

        if let Some(prefix) = prefix {
            if let Some(entry) = self
                .entries
                .iter()
                .find(|e| Self::match_prefix(prefix, &e.class_name))
            {
                return Ok(entry);
            }
        }

        if let Some(entry) = self.entries.iter().find(|e| e.pattern.is_match(model)) {
            return Ok(entry);
        }

        if model.contains('/') {
            let provider = model.split_once('/').map(|(p, _)| p).unwrap_or(model);
            if self.known_litellm_providers.iter().any(|p| p == provider) {
                // A real LiteLLM-equivalent factory would be registered as
                // an ordinary entry above; reaching here with a *known*
                // provider but no matching entry means the integration
                // seam is populated but nothing is registered for it yet.
            }
        }

        Err(Self::not_found_error(model))
    }

    fn not_found_error(model: &str) -> RegistryError {
        if model.starts_with("claude-") {
            RegistryError::ClaudeModelNotFound(model.to_string())
        } else if model.contains('/') {
            RegistryError::LiteLlmModelNotFound(model.to_string())
        } else {
            RegistryError::ModelNotFound(model.to_string())
        }
    }

    /// C0107: creates a new LLM instance for `model`.
    pub fn new_llm(&self, model: &str) -> Result<Box<dyn BaseLlm>, RegistryError> {
        let (prefix, actual_model) = Self::parse_model(model);
        let entry = self.resolve_entry(model)?;

        if let Some(prefix) = prefix {
            if Self::match_prefix(prefix, &entry.class_name) {
                return Ok((entry.factory)(actual_model));
            }
        }
        Ok((entry.factory)(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubLlm {
        model: String,
    }

    impl BaseLlm for StubLlm {
        fn model(&self) -> &str {
            &self.model
        }

        fn type_name(&self) -> &'static str {
            "StubLlm"
        }
    }

    fn stub_factory() -> LlmFactory {
        Arc::new(|model: &str| {
            Box::new(StubLlm {
                model: model.to_string(),
            }) as Box<dyn BaseLlm>
        })
    }

    #[test]
    fn resolves_by_regex_full_match() {
        let mut registry = LlmRegistry::new();
        registry
            .register(&["gemini-.*"], "GeminiLlm", stub_factory())
            .unwrap();
        let llm = registry.new_llm("gemini-2.5-flash").unwrap();
        assert_eq!(llm.model(), "gemini-2.5-flash");
    }

    #[test]
    fn regex_match_is_a_full_match_not_a_substring_search() {
        let mut registry = LlmRegistry::new();
        registry
            .register(&["gemini-.*"], "GeminiLlm", stub_factory())
            .unwrap();
        assert!(registry.resolve("not-a-gemini-model-at-all").is_err());
    }

    #[test]
    fn prefix_override_bypasses_regex_resolution() {
        let mut registry = LlmRegistry::new();
        registry
            .register(&["gemini-.*"], "GeminiLlm", stub_factory())
            .unwrap();
        registry
            .register(&["openai/.*"], "OpenAILlm", stub_factory())
            .unwrap();
        // "lite:openai/gpt-4o" should resolve via the OpenAILlm-matching
        // prefix, not by testing "openai/gpt-4o" against every regex.
        let llm = registry.new_llm("openai:openai/gpt-4o").unwrap();
        assert_eq!(llm.model(), "openai/gpt-4o");
    }

    #[test]
    fn prefix_match_strips_a_trailing_llm_suffix_and_is_case_insensitive() {
        let mut registry = LlmRegistry::new();
        registry
            .register(&["gemini-.*"], "GeminiLlm", stub_factory())
            .unwrap();
        assert!(LlmRegistry::match_prefix("gemini", "GeminiLlm"));
        assert!(LlmRegistry::match_prefix("GEMINI", "GeminiLlm"));
        assert!(LlmRegistry::match_prefix("geminillm", "GeminiLlm"));
        assert!(!LlmRegistry::match_prefix("claude", "GeminiLlm"));
    }

    #[test]
    fn unresolvable_claude_model_gets_a_helpful_install_hint() {
        let registry = LlmRegistry::new();
        match registry.resolve("claude-3-opus") {
            Err(RegistryError::ClaudeModelNotFound(_)) => {}
            _ => panic!("expected ClaudeModelNotFound"),
        }
    }

    #[test]
    fn unresolvable_provider_style_model_gets_a_litellm_hint() {
        let registry = LlmRegistry::new();
        match registry.resolve("groq/llama3") {
            Err(RegistryError::LiteLlmModelNotFound(_)) => {}
            _ => panic!("expected LiteLlmModelNotFound"),
        }
    }

    #[test]
    fn unresolvable_plain_model_gets_a_generic_not_found_error() {
        let registry = LlmRegistry::new();
        match registry.resolve("totally-unknown-model") {
            Err(RegistryError::ModelNotFound(_)) => {}
            _ => panic!("expected ModelNotFound"),
        }
    }

    #[test]
    fn re_registering_the_same_regex_replaces_the_previous_entry() {
        let mut registry = LlmRegistry::new();
        registry
            .register(&["gemini-.*"], "First", stub_factory())
            .unwrap();
        registry
            .register(&["gemini-.*"], "Second", stub_factory())
            .unwrap();
        assert_eq!(registry.entries.len(), 1);
        assert_eq!(registry.entries[0].class_name, "Second");
    }
}
