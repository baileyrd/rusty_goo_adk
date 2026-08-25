//! The wire shape of the Live API's initial `BidiGenerateContent` setup
//! message — what [`crate::gemini::Gemini::connect`] (C0131's handshake
//! half) sends as the first frame after opening the WebSocket.
//!
//! **Reconstructed from the third-party SDK, disclosed confidence
//! caveat**: same treatment `live_server_message.rs` already carries for
//! the receive side. In the source, this JSON is built entirely inside
//! the third-party `google-genai` SDK's `live.py`/`_live_converters.py`
//! (`_LiveConnectConfig_to_mldev`, the mldev/Gemini-Developer-API-key
//! branch specifically — Vertex AI has its own sibling converter, not
//! ported since [`crate::gemini::GeminiCallError::VertexAiAuthNotSupported`]
//! already blocks that backend from reaching `connect()` at all). The
//! field-to-wire-path map here was read directly out of the *installed*
//! `google-genai` package (not guessed), so it should be faithful — but
//! it is still Google's undocumented wire protocol, not ADK's own code,
//! and unverified against a live endpoint.
//!
//! **Field coverage matches [`crate::llm_request::LiveConnectConfigStub`]
//! exactly**: every field this port models on that struct has a
//! corresponding wire path here (`speech_config`/`tools`/
//! `thinking_config`/`safety_settings`/etc. stay opaque [`Value`]
//! placeholders, serialized verbatim — same as `LiveConnectConfigStub`
//! itself already discloses). Fields the mldev converter maps but this
//! port's `LiveConnectConfigStub` never modeled (`temperature`/`top_p`/
//! `top_k`/`max_output_tokens`/`media_resolution`/`seed` — all
//! `GenerateContentConfig`-level fields the source's `LiveConnectConfig`
//! also carries but this port's config stub never gained) are not
//! covered — the same pre-existing narrowing as
//! `generate_content_request.rs`'s `config.tools`.
//!
//! **`explicit_vad_signal`, ported as a hard error**: the source's own
//! mldev converter raises `ValueError` unconditionally for this field —
//! "only supported in Gemini Enterprise Agent Platform mode, not in
//! Gemini Developer API mode" — matching this port's already-Gemini-API-only
//! `connect()` (Vertex AI isn't reachable here at all yet, see
//! `GeminiCallError::VertexAiAuthNotSupported`), so this check is
//! unconditional here too, not backend-gated.

use adk_genai::content::Content;
use rusty_serde::value::Value;
use rusty_serde::Serialize;

use crate::llm_request::{LiveConnectConfigStub, SessionResumptionStub};

/// `setup.generationConfig` — see the module doc for which
/// `LiveConnectConfigStub` fields nest here on the wire vs. directly
/// under `setup`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LiveGenerationConfigBody {
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_affective_dialog: Option<bool>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub translation_config: Option<Value>,
}

/// `setup` — the body of the initial `BidiGenerateContent` client
/// message.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LiveClientSetupBody {
    pub model: String,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<LiveGenerationConfigBody>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub session_resumption: Option<SessionResumptionStub>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub input_audio_transcription: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub output_audio_transcription: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub realtime_input_config: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window_compression: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub proactivity: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub history_config: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub avatar_config: Option<Value>,
    #[rusty_serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Value>,
}

/// The top-level `{"setup": {...}}` envelope sent as the first WebSocket
/// text frame.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LiveClientSetupMessage {
    pub setup: LiveClientSetupBody,
}

/// Returns `true` if `config.explicit_vad_signal` is set — see the
/// module doc for why this is always an error in this port (Vertex AI,
/// the one backend that supports it, isn't reachable via `connect()`
/// yet).
pub fn explicit_vad_signal_requested(config: &LiveConnectConfigStub) -> bool {
    config.explicit_vad_signal.unwrap_or(false)
}

/// Builds the setup envelope from a model name and the (already
/// `Gemini::prepare_live_connect_config`-processed) live-connect config.
pub fn build_live_setup_message(
    model: &str,
    config: Option<&LiveConnectConfigStub>,
) -> LiveClientSetupMessage {
    let Some(config) = config else {
        return LiveClientSetupMessage {
            setup: LiveClientSetupBody {
                model: model.to_string(),
                ..Default::default()
            },
        };
    };

    let generation_config = if config.response_modalities.is_some()
        || config.speech_config.is_some()
        || config.thinking_config.is_some()
        || config.enable_affective_dialog.is_some()
        || config.translation_config.is_some()
    {
        Some(LiveGenerationConfigBody {
            response_modalities: config.response_modalities.clone().map(Value::Seq),
            speech_config: config.speech_config.clone(),
            thinking_config: config.thinking_config.clone(),
            enable_affective_dialog: config.enable_affective_dialog,
            translation_config: config.translation_config.clone(),
        })
    } else {
        None
    };

    LiveClientSetupMessage {
        setup: LiveClientSetupBody {
            model: model.to_string(),
            generation_config,
            system_instruction: config.system_instruction.clone(),
            tools: config.tools.clone(),
            session_resumption: config.session_resumption.clone(),
            input_audio_transcription: config.input_audio_transcription.clone(),
            output_audio_transcription: config.output_audio_transcription.clone(),
            realtime_input_config: config.realtime_input_config.clone(),
            context_window_compression: config.context_window_compression.clone(),
            proactivity: config.proactivity.clone(),
            history_config: config.history_config.clone(),
            avatar_config: config.avatar_config.clone(),
            safety_settings: config.safety_settings.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::Part;

    #[test]
    fn builds_a_minimal_setup_message_without_a_config() {
        let message = build_live_setup_message("gemini-2.5-flash", None);
        assert_eq!(message.setup.model, "gemini-2.5-flash");
        assert!(message.setup.generation_config.is_none());
        let json = rusty_serde::json::to_string(&message).unwrap();
        assert_eq!(json, r#"{"setup":{"model":"gemini-2.5-flash"}}"#);
    }

    #[test]
    fn carries_the_system_instruction_and_session_resumption() {
        let config = LiveConnectConfigStub {
            system_instruction: Some(Content {
                role: Some("system".to_string()),
                parts: vec![Part::text("be helpful")],
            }),
            session_resumption: Some(SessionResumptionStub {
                transparent: Some(true),
            }),
            ..Default::default()
        };
        let message = build_live_setup_message("gemini-2.5-flash", Some(&config));
        let json = rusty_serde::json::to_string(&message).unwrap();
        assert!(json.contains(r#""systemInstruction""#));
        assert!(json.contains(r#""sessionResumption":{"transparent":true}"#));
    }

    #[test]
    fn nests_speech_config_and_thinking_config_under_generation_config() {
        let config = LiveConnectConfigStub {
            speech_config: Some(Value::String("a-voice".to_string())),
            thinking_config: Some(Value::Bool(true)),
            enable_affective_dialog: Some(true),
            ..Default::default()
        };
        let message = build_live_setup_message("gemini-2.5-flash", Some(&config));
        let json = rusty_serde::json::to_string(&message).unwrap();
        let generation_config = message.setup.generation_config.unwrap();
        assert_eq!(
            generation_config.speech_config,
            Some(Value::String("a-voice".to_string()))
        );
        assert_eq!(generation_config.thinking_config, Some(Value::Bool(true)));
        assert_eq!(generation_config.enable_affective_dialog, Some(true));
        assert!(json.contains(r#""generationConfig":{"#));
        assert!(!json.contains(r#""setup":{"speechConfig""#));
    }

    #[test]
    fn omits_absent_fields_from_the_wire_json() {
        let message =
            build_live_setup_message("gemini-2.5-flash", Some(&LiveConnectConfigStub::default()));
        let json = rusty_serde::json::to_string(&message).unwrap();
        assert_eq!(json, r#"{"setup":{"model":"gemini-2.5-flash"}}"#);
    }

    #[test]
    fn explicit_vad_signal_requested_reflects_the_config() {
        assert!(!explicit_vad_signal_requested(
            &LiveConnectConfigStub::default()
        ));
        let config = LiveConnectConfigStub {
            explicit_vad_signal: Some(true),
            ..Default::default()
        };
        assert!(explicit_vad_signal_requested(&config));
    }
}
