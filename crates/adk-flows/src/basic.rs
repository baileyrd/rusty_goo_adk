//! Capability C0168: the `basic` request processor's core logic, ported
//! from `google.adk.flows.llm_flows.basic`.
//!
//! **Scope, disclosed**: only `_build_basic_request`'s pure logic is
//! ported here — [`build_basic_request`] is a free function taking
//! `&LlmAgent`/`&RunConfig` directly, not yet wrapped in a real
//! [`crate::processor::BaseLlmRequestProcessor`] impl reading through
//! `InvocationContext`. The source's `as_llm_agent` narrows
//! `invocation_context.agent: BaseAgent` down to a concrete `LlmAgent` via
//! a (Python, runtime-no-op) `cast()`; this port's `InvocationContext.agent:
//! Option<BaseAgent>` has no equivalent downcast — `LlmAgent` doesn't
//! implement `AgentBehavior` yet (see `llm_agent.rs`'s own module doc:
//! that wiring needs `BaseLlmFlow`/`SingleFlow`/`AutoFlow`, which this
//! crate is only just starting). Wiring this into a real trait impl is
//! deferred to whichever batch gives `LlmAgent` real tree placement.
//!
//! **Adaptation, disclosed**: the source's `_copy_request_scoped_fields`
//! deep-copies every list/dict field of the agent's typed
//! `GenerateContentConfig` so request assembly can't write back into the
//! agent's own config object. This port's `LlmAgent.generate_content_config`
//! is an opaque, already-validated [`rusty_serde::value::Value`] (C0084);
//! [`build_basic_request`] deserializes it into a fresh
//! [`GenerateContentConfigStub`] instead of a field-by-field copy — the
//! deserialized value is already a new, unaliased owned struct, so there's
//! nothing left for a container-level deep copy to protect against.
//!
//! **Adaptation, disclosed**: `_merge_run_config_http_options`'s
//! `timeout`/`retry_options`/`extra_body` fields aren't modeled in
//! [`HttpOptionsStub`] yet (see `llm_request.rs`'s module doc) — only the
//! `headers` merge (RunConfig's preserved headers win on key conflict) is
//! ported. `RunConfig.session_resumption` is deserialized best-effort into
//! [`SessionResumptionStub`]; unlike `generate_content_config` (a
//! user-authored, already-validated shape this function can afford to
//! fail loudly on), `RunConfig`'s per-field opaque values were never
//! validated against this port's narrower stub types — a shape mismatch
//! just doesn't get copied forward rather than erroring out of request
//! assembly entirely.

use adk_agents::llm_agent::{AgentMode, LlmAgent};
use adk_agents::run_config::RunConfig;
use adk_models::capabilities::is_gemini_3_x_live;
use adk_models::llm_request::{
    GenerateContentConfigStub, HttpOptionsStub, LlmRequest, SessionResumptionStub,
};

use crate::canonical_model::{canonical_live_model, canonical_model, CanonicalModelError};

#[derive(Debug, rusty_err::Error)]
pub enum BasicRequestError {
    #[error("{0}")]
    Model(#[from] CanonicalModelError),
    #[error("failed to interpret LlmAgent.generate_content_config: {0}")]
    InvalidGenerateContentConfig(String),
}

fn merge_run_config_http_options(
    config: &mut GenerateContentConfigStub,
    preserved: HttpOptionsStub,
) {
    match &mut config.http_options {
        None => config.http_options = Some(preserved),
        Some(http_options) => {
            if let Some(preserved_headers) = preserved.headers {
                http_options
                    .headers
                    .get_or_insert_with(Default::default)
                    .extend(preserved_headers);
            }
            // `timeout`/`retry_options`/`extra_body` aren't modeled in
            // `HttpOptionsStub` yet — see the module doc.
        }
    }
}

/// C0168: `_build_basic_request` — populates `llm_request`'s
/// model/config/output_schema/live-connect fields from the agent's
/// canonical settings and `RunConfig`. See the module doc for what's
/// deferred (real `BaseLlmRequestProcessor` wiring).
pub fn build_basic_request(
    agent: &LlmAgent,
    run_config: &RunConfig,
    llm_request: &mut LlmRequest,
) -> Result<(), BasicRequestError> {
    let model = canonical_model(agent)?;
    llm_request.model = Some(model.model().to_string());

    // Preserved across the agent-config overwrite below, then merged back
    // — matches the source exactly (see the module doc's http_options note).
    let preserved_http_options = llm_request.config.http_options.take();

    llm_request.config = rusty_serde::json::from_value(agent.generate_content_config.clone())
        .map_err(|e| BasicRequestError::InvalidGenerateContentConfig(e.to_string()))?;

    if let Some(preserved_http_options) = preserved_http_options {
        merge_run_config_http_options(&mut llm_request.config, preserved_http_options);
    }

    if let Some(labels) = &run_config.labels {
        llm_request
            .config
            .labels
            .get_or_insert_with(Default::default)
            .extend(labels.clone());
    }

    // Only set output_schema if no tools are specified — models don't
    // support output_schema and tools together yet; task-mode agents
    // collect structured output via the finish_task tool schema instead
    // (not ported — needs BaseTool, Phase 8).
    if agent.mode != Some(AgentMode::Task) {
        if let Some(output_schema) = &agent.output_schema {
            if agent.tools.is_empty() || model.capabilities().output_schema_and_tools {
                llm_request.set_output_schema(output_schema.clone());
            }
        }
    }

    let live_connect_config = llm_request
        .live_connect_config
        .get_or_insert_with(Default::default);
    live_connect_config.response_modalities = run_config.response_modalities.clone();
    live_connect_config.speech_config = run_config.speech_config.clone();
    live_connect_config.output_audio_transcription = run_config.output_audio_transcription.clone();
    live_connect_config.input_audio_transcription = run_config.input_audio_transcription.clone();
    live_connect_config.realtime_input_config = run_config.realtime_input_config.clone();
    live_connect_config.explicit_vad_signal = run_config.explicit_vad_signal;
    live_connect_config.translation_config = run_config.translation_config.clone();

    let active_model_name = canonical_live_model(agent)
        .ok()
        .map(|m| m.model().to_string())
        .or_else(|| llm_request.model.clone());
    let is_gemini_3_x = is_gemini_3_x_live(active_model_name.as_deref());
    live_connect_config.enable_affective_dialog = if is_gemini_3_x {
        None
    } else {
        run_config.enable_affective_dialog
    };
    live_connect_config.proactivity = if is_gemini_3_x {
        None
    } else {
        run_config.proactivity.clone()
    };
    live_connect_config.session_resumption = run_config
        .session_resumption
        .clone()
        .and_then(|value| rusty_serde::json::from_value::<SessionResumptionStub>(value).ok());
    live_connect_config.history_config = run_config.history_config.clone();
    live_connect_config.context_window_compression = run_config.context_window_compression.clone();
    live_connect_config.avatar_config = run_config.avatar_config.clone();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::llm_agent::ModelRef;
    use rusty_serde::value::Value;

    fn agent(model: &str) -> LlmAgent {
        LlmAgent::new(ModelRef::Name(model.to_string()))
    }

    #[test]
    fn sets_the_resolved_model_name() {
        let agent = agent("gemini-2.5-flash");
        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap();
        assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
    }

    #[test]
    fn errors_when_the_model_cannot_be_resolved() {
        let agent = agent("totally-unknown-model");
        let mut request = LlmRequest::new("placeholder");
        let err = build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap_err();
        assert!(matches!(err, BasicRequestError::Model(_)));
    }

    #[test]
    fn copies_generate_content_config_fields() {
        let mut agent = agent("gemini-2.5-flash");
        agent.generate_content_config = Value::Map(vec![(
            "response_mime_type".to_string(),
            Value::String("application/json".to_string()),
        )]);
        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap();
        assert_eq!(
            request.config.response_mime_type.as_deref(),
            Some("application/json")
        );
    }

    #[test]
    fn merges_run_config_labels_into_config_labels() {
        let mut agent = agent("gemini-2.5-flash");
        agent.generate_content_config = Value::Map(vec![(
            "labels".to_string(),
            Value::Map(vec![("team".to_string(), Value::String("a".to_string()))]),
        )]);
        let mut labels = std::collections::BTreeMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        let run_config = RunConfig {
            labels: Some(labels),
            ..RunConfig::default()
        };

        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &run_config, &mut request).unwrap();

        let config_labels = request.config.labels.unwrap();
        assert_eq!(config_labels.get("team").map(String::as_str), Some("a"));
        assert_eq!(config_labels.get("env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn preserves_and_merges_pre_existing_http_options_headers() {
        let agent = agent("gemini-2.5-flash");
        let mut request = LlmRequest::new("placeholder");
        let mut preserved_headers = std::collections::BTreeMap::new();
        preserved_headers.insert("x-run-config".to_string(), "1".to_string());
        request.config.http_options = Some(HttpOptionsStub {
            headers: Some(preserved_headers),
            api_version: None,
        });

        build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap();

        let headers = request.config.http_options.unwrap().headers.unwrap();
        assert_eq!(headers.get("x-run-config").map(String::as_str), Some("1"));
    }

    #[test]
    fn sets_output_schema_when_no_tools_are_present() {
        let mut agent = agent("gemini-2.5-flash");
        agent.output_schema = Some(Value::Map(vec![(
            "type".to_string(),
            Value::String("object".to_string()),
        )]));
        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap();
        assert!(request.config.response_schema.is_some());
        assert_eq!(
            request.config.response_mime_type.as_deref(),
            Some("application/json")
        );
    }

    #[test]
    fn skips_output_schema_for_a_task_mode_agent() {
        let mut agent = agent("gemini-2.5-flash");
        agent.mode = Some(AgentMode::Task);
        agent.output_schema = Some(Value::Map(vec![]));
        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &RunConfig::default(), &mut request).unwrap();
        assert!(request.config.response_schema.is_none());
    }

    #[test]
    fn forwards_run_config_live_connect_fields() {
        let agent = agent("gemini-2.5-flash");
        let run_config = RunConfig {
            explicit_vad_signal: Some(true),
            speech_config: Some(Value::String("some-voice".to_string())),
            ..RunConfig::default()
        };

        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &run_config, &mut request).unwrap();

        let live = request.live_connect_config.unwrap();
        assert_eq!(live.explicit_vad_signal, Some(true));
        assert_eq!(
            live.speech_config,
            Some(Value::String("some-voice".to_string()))
        );
    }

    #[test]
    fn suppresses_affective_dialog_and_proactivity_on_gemini_3_x_live_models() {
        let agent = agent("gemini-3.0-live-preview");
        let run_config = RunConfig {
            enable_affective_dialog: Some(true),
            proactivity: Some(Value::String("eager".to_string())),
            ..RunConfig::default()
        };

        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &run_config, &mut request).unwrap();

        let live = request.live_connect_config.unwrap();
        assert_eq!(live.enable_affective_dialog, None);
        assert_eq!(live.proactivity, None);
    }

    #[test]
    fn keeps_affective_dialog_and_proactivity_on_a_non_gemini_3_x_live_model() {
        let agent = agent("gemini-2.5-flash");
        let run_config = RunConfig {
            enable_affective_dialog: Some(true),
            proactivity: Some(Value::String("eager".to_string())),
            ..RunConfig::default()
        };

        let mut request = LlmRequest::new("placeholder");
        build_basic_request(&agent, &run_config, &mut request).unwrap();

        let live = request.live_connect_config.unwrap();
        assert_eq!(live.enable_affective_dialog, Some(true));
        assert_eq!(live.proactivity, Some(Value::String("eager".to_string())));
    }
}
