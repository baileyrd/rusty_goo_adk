//! Capabilities C0080, C0090: `LlmAgent.canonical_model`/`canonical_live_model`,
//! ported from `google.adk.agents.llm_agent`.
//!
//! **Deferred, disclosed**: the source resolves an unset `model` (an empty
//! string, the field's default) by walking up `parent_agent` looking for
//! the nearest `LlmAgent` ancestor before falling back to the class-level
//! default. `LlmAgent` isn't wired into `BaseAgent`'s tree yet (see
//! `llm_agent.rs`'s own module doc — that wiring is blocked on Phase 3/4/8
//! landing, which is exactly what this crate is starting), so there is no
//! parent to walk to yet. An unset model resolves straight to the default
//! model/live model, skipping the ancestor-lookup step; once `LlmAgent`
//! gains real tree placement, that step slots in here without changing
//! this function's contract.
//!
//! **Deferred, disclosed**: [`ModelRef::Instance`] (the source's
//! `model: BaseLlm` case — an agent constructed with a live model
//! instance rather than a name) can't be resolved through this port.
//! `ModelRef` lives in `adk-agents`, which can't hold a real
//! `Arc<dyn BaseLlm>` without depending on `adk-models` — and `adk-models`
//! already depends on `adk-agents` (for `ContextCacheConfig`), so that
//! would make the two crates depend on each other (see this crate's own
//! module doc). [`CanonicalModelError::LiveInstanceNotSupported`] names
//! this explicitly rather than silently misresolving or panicking. Fixing
//! it for real means extracting `ContextCacheConfig` into a shared lower
//! crate both `adk-agents` and `adk-models` can depend on — a deliberate
//! restructuring, not something to do as a side effect of this batch.

use std::sync::Arc;

use adk_agents::llm_agent::{default_live_model, default_model, LlmAgent, ModelRef};
use adk_models::base_llm::BaseLlm;
use adk_models::registry::{default_registry, RegistryError};

#[derive(Debug, rusty_err::Error)]
pub enum CanonicalModelError {
    #[error("{0}")]
    Registry(#[from] RegistryError),
    /// See the module doc's disclosed deferral.
    #[error(
        "an LlmAgent constructed with a live BaseLlm instance (rather than a model name string) \
         can't be resolved yet in this port — ModelRef::Instance is an opaque placeholder \
         (see canonical_model.rs's module doc); pass the model by name instead"
    )]
    LiveInstanceNotSupported,
}

fn resolve(model_ref: &ModelRef) -> Result<Arc<dyn BaseLlm>, CanonicalModelError> {
    match model_ref {
        ModelRef::Instance(_) => Err(CanonicalModelError::LiveInstanceNotSupported),
        ModelRef::Name(name) => {
            let registry = default_registry()
                .read()
                .expect("default_registry lock is never held across a panic");
            registry.new_llm(name).map(Arc::from).map_err(Into::into)
        }
    }
}

/// C0080: `LlmAgent.canonical_model` — the resolved `self.model` field as a
/// real `BaseLlm`. See the module doc for what's deferred.
pub fn canonical_model(agent: &LlmAgent) -> Result<Arc<dyn BaseLlm>, CanonicalModelError> {
    match &agent.model {
        ModelRef::Name(name) if name.is_empty() => resolve(&default_model()),
        other => resolve(other),
    }
}

/// C0090: `LlmAgent.canonical_live_model` — the resolved `self.model` field
/// as a real `BaseLlm` for live mode. See the module doc for what's
/// deferred.
pub fn canonical_live_model(agent: &LlmAgent) -> Result<Arc<dyn BaseLlm>, CanonicalModelError> {
    match &agent.model {
        ModelRef::Name(name) if name.is_empty() => resolve(&default_live_model()),
        other => resolve(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_model_resolves_an_explicit_model_name() {
        let agent = LlmAgent::new(ModelRef::Name("gemini-2.5-flash".to_string()));
        let model = canonical_model(&agent).unwrap();
        assert_eq!(model.model(), "gemini-2.5-flash");
        assert_eq!(model.type_name(), "Gemini");
    }

    #[test]
    fn canonical_model_falls_back_to_the_default_model_when_unset() {
        let agent = LlmAgent::new(ModelRef::Name(String::new()));
        let model = canonical_model(&agent).unwrap();
        assert_eq!(model.type_name(), "Gemini");
    }

    #[test]
    fn canonical_model_resolves_an_ollama_model_name() {
        let agent = LlmAgent::new(ModelRef::Name("ollama/llama3.2".to_string()));
        let model = canonical_model(&agent).unwrap();
        assert_eq!(model.type_name(), "OllamaLlm");
    }

    #[test]
    fn canonical_model_errors_on_a_live_instance() {
        let agent = LlmAgent::new(ModelRef::Instance(rusty_serde::value::Value::Null));
        // `Arc<dyn BaseLlm>` isn't `Debug`, so `unwrap_err()` (which would
        // need to `Debug`-print the discarded `Ok` value) isn't available —
        // match directly instead.
        match canonical_model(&agent) {
            Err(CanonicalModelError::LiveInstanceNotSupported) => {}
            _ => panic!("expected LiveInstanceNotSupported"),
        }
    }

    #[test]
    fn canonical_model_surfaces_a_registry_not_found_error() {
        let agent = LlmAgent::new(ModelRef::Name("totally-unknown-model".to_string()));
        match canonical_model(&agent) {
            Err(CanonicalModelError::Registry(RegistryError::ModelNotFound(_))) => {}
            _ => panic!("expected Registry(ModelNotFound)"),
        }
    }

    #[test]
    fn canonical_live_model_resolves_an_explicit_model_name() {
        let agent = LlmAgent::new(ModelRef::Name(
            "gemini-live-2.5-flash-native-audio".to_string(),
        ));
        let model = canonical_live_model(&agent).unwrap();
        assert_eq!(model.type_name(), "Gemini");
    }

    #[test]
    fn canonical_live_model_falls_back_to_the_default_live_model_when_unset() {
        let agent = LlmAgent::new(ModelRef::Name(String::new()));
        let model = canonical_live_model(&agent).unwrap();
        assert_eq!(model.type_name(), "Gemini");
    }
}
