//! C0630: `evaluation.simulation._llm_audio_user_simulator`, ported from
//! `google.adk.evaluation.simulation._llm_audio_user_simulator`.
//!
//! [`LlmAudioUserSimulator`] is a **decorator** `UserSimulator`: it
//! delegates text generation to a wrapped inner `UserSimulator` (treated
//! as a black box — only its `get_next_user_message` output is
//! consumed), then feeds the resulting text to a second `BaseLlm`
//! (resolved by name via [`crate::user_simulator`]'s shared
//! `LlmRegistry`) to produce audio bytes, and wraps the result into a
//! live-input-ready `Content`.
//!
//! **`UserSimulatorProvider` composition, now wired (C0627)**: this
//! decorator's own `new()` takes an already-built
//! `Box<dyn UserSimulator + Send + Sync>` directly, matching the
//! source's own constructor shape exactly. The composition that builds
//! that inner simulator from this config's own text-generation fields —
//! and registers the `"llm_audio"` discriminator — lives in
//! `user_simulator.rs`'s `registry()` and
//! `UserSimulatorProvider::provide()` (see those for the scenario- and
//! static-path wiring); this module only provides the pieces
//! (`LlmAudioUserSimulatorConfig`/`LlmAudioUserSimulator`) that
//! composition assembles.
//!
//! **`GenerateContentConfigStub::speech_config`, newly added**: this
//! config's `audio_model_configuration` default needs a `speech_config`
//! field `GenerateContentConfigStub` didn't have yet — added there as an
//! opaque `Value` placeholder (see that struct's own doc), the same
//! "widen a placeholder once its first real consumer needs the shape"
//! precedent used throughout this port.
//!
//! **Custom base64 codec, duplicated not shared**: `adk-eval` can't
//! reach `adk-agents::file_artifact_service`'s hand-rolled base64
//! helpers (the crate-dependency direction runs the other way), so this
//! module hand-rolls its own tiny encode/decode pair — the same
//! "duplicate across an unreachable crate boundary" precedent that
//! file's own doc already establishes for why `adk-tools` can't reuse it
//! either.
//!
//! **`_generate_audio`'s streaming loop, narrowed to a materialized
//! `Vec<LlmResponse>`**: the source iterates an async generator of
//! partial responses, concatenating each yielded chunk's audio bytes.
//! This port's `BaseLlm::generate_content_async` already returns a
//! fully-materialized `Vec<LlmResponse>`, not a stream (see
//! `base_llm.rs`'s own module doc on the deferred streaming contract),
//! so [`LlmAudioUserSimulator::generate_audio`] iterates that vector
//! instead — behaviorally identical for any backend that already
//! collects internally before returning.
//!
//! **`get_simulation_evaluator`, `unimplemented!()` not `None`**: the
//! source's override is a literal `raise NotImplementedError()` — ported
//! as a panic, the direct Rust analogue for "deliberately unimplemented,
//! callers must not reach this," not a silent `None` (which would imply
//! "this simulator legitimately has no evaluator").
//!
//! **Not ported**: `_CloudTTSLlm` (C0631) — the default `"cloud_tts"`
//! audio-model backend, needing live Google Cloud TTS network/credentials
//! — stays unregistered in `LlmRegistry`. Constructing this simulator
//! with the default `audio_model` therefore fails at resolution time
//! today, the same as resolving any other unregistered model name; a
//! caller supplying an already-registered `audio_model` (or a test
//! double) can exercise this simulator end-to-end right now.

use adk_genai::content::{Content, MediaBlobStub, Part};
use adk_models::base_llm::BaseLlm;
use adk_models::llm_request::{GenerateContentConfigStub, LlmRequest};
use adk_models::registry::{default_registry, RegistryError};
use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::llm_backed_user_simulator::is_valid_user_simulator_template;
use crate::user_simulator::{
    parse_simulator_config, BoxFuture, NextUserMessage, Status, UserSimulator,
};

const AUTHOR_USER: &str = "user";
const DEFAULT_AUDIO_MODEL: &str = "cloud_tts";

fn default_simulator_type() -> String {
    "llm_audio".to_string()
}

fn default_text_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_text_model_configuration() -> GenerateContentConfigStub {
    GenerateContentConfigStub {
        thinking_config: Some(Value::Map(vec![
            ("includeThoughts".to_string(), Value::Bool(true)),
            ("thinkingBudget".to_string(), Value::UInt(10240)),
        ])),
        ..Default::default()
    }
}

fn default_max_allowed_invocations() -> i64 {
    20
}

fn default_audio_model() -> String {
    DEFAULT_AUDIO_MODEL.to_string()
}

fn default_audio_model_configuration() -> GenerateContentConfigStub {
    GenerateContentConfigStub {
        speech_config: Some(Value::Map(vec![
            (
                "voiceConfig".to_string(),
                Value::Map(vec![(
                    "prebuiltVoiceConfig".to_string(),
                    Value::Map(vec![(
                        "voiceName".to_string(),
                        Value::String("en-US-Studio-O".to_string()),
                    )]),
                )]),
            ),
            (
                "languageCode".to_string(),
                Value::String("en-US".to_string()),
            ),
        ])),
        ..Default::default()
    }
}

fn default_include_text_with_audio() -> bool {
    true
}

/// `_llm_audio_user_simulator.LlmAudioUserSimulatorConfig`.
///
/// The `model`/`model_configuration`/`max_allowed_invocations`/
/// `custom_instructions`/`include_function_calls` fields configure the
/// *wrapped* text simulator this decorator will eventually receive —
/// see the module doc for why building that inner simulator from these
/// fields (the `UserSimulatorProvider` composition) isn't wired in this
/// batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase")]
pub struct LlmAudioUserSimulatorConfig {
    #[rusty_serde(rename = "type", default = "default_simulator_type")]
    pub simulator_type: String,
    #[rusty_serde(default = "default_text_model")]
    pub model: String,
    #[rusty_serde(default = "default_text_model_configuration")]
    pub model_configuration: GenerateContentConfigStub,
    #[rusty_serde(default = "default_max_allowed_invocations")]
    pub max_allowed_invocations: i64,
    #[rusty_serde(default)]
    pub custom_instructions: Option<String>,
    #[rusty_serde(default)]
    pub include_function_calls: bool,
    #[rusty_serde(default = "default_audio_model")]
    pub audio_model: String,
    #[rusty_serde(default = "default_audio_model_configuration")]
    pub audio_model_configuration: GenerateContentConfigStub,
    #[rusty_serde(default = "default_include_text_with_audio")]
    pub include_text_with_audio: bool,
}

impl Default for LlmAudioUserSimulatorConfig {
    fn default() -> Self {
        Self {
            simulator_type: default_simulator_type(),
            model: default_text_model(),
            model_configuration: default_text_model_configuration(),
            max_allowed_invocations: default_max_allowed_invocations(),
            custom_instructions: None,
            include_function_calls: false,
            audio_model: default_audio_model(),
            audio_model_configuration: default_audio_model_configuration(),
            include_text_with_audio: default_include_text_with_audio(),
        }
    }
}

impl LlmAudioUserSimulatorConfig {
    /// `@field_validator("custom_instructions")` — same placeholder
    /// requirements `LlmBackedUserSimulatorConfig::validate` already
    /// enforces.
    pub fn validate(&self) -> Result<(), LlmAudioUserSimulatorError> {
        let Some(custom_instructions) = &self.custom_instructions else {
            return Ok(());
        };
        if is_valid_user_simulator_template(
            custom_instructions,
            &["stop_signal", "conversation_plan", "conversation_history"],
        ) {
            Ok(())
        } else {
            Err(LlmAudioUserSimulatorError::InvalidCustomInstructions)
        }
    }
}

/// Error type for [`LlmAudioUserSimulator::new`]/
/// [`LlmAudioUserSimulator::generate_audio`].
#[derive(Debug, rusty_err::Error)]
pub enum LlmAudioUserSimulatorError {
    #[error(
        "custom_instructions must contain each of the following formatting placeholders using \
         Jinja syntax: {{{{ stop_signal }}}}, {{{{ conversation_plan }}}}, {{{{ conversation_history }}}}"
    )]
    InvalidCustomInstructions,
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Registry(#[from] RegistryError),
    #[error("{0}")]
    Generation(#[from] adk_models::base_llm::BaseLlmError),
    #[error("Audio generation failed: {0}")]
    AudioGenerationFailed(String),
    #[error("Audio model returned no audio data")]
    NoAudioData,
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied();
        let b2 = chunk.get(2).copied();
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as usize] as char);
        out.push(match b1 {
            Some(b1) => ALPHABET[(((b1 & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as usize] as char,
            None => '=',
        });
        out.push(match b2 {
            Some(b2) => ALPHABET[(b2 & 0x3f) as usize] as char,
            None => '=',
        });
    }
    out
}

fn base64_decode_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn base64_decode(data: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for &byte in data.as_bytes() {
        if byte == b'=' {
            break;
        }
        let value = base64_decode_value(byte)?;
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push((buffer >> bits) as u8);
        }
    }
    Some(bytes)
}

fn blob_data(blob: &MediaBlobStub) -> Option<Vec<u8>> {
    match &blob.rest {
        Some(Value::Map(entries)) => entries
            .iter()
            .find(|(key, _)| key == "data")
            .and_then(|(_, value)| value.as_str())
            .and_then(base64_decode),
        _ => None,
    }
}

/// C0630: `_llm_audio_user_simulator._LlmAudioUserSimulator`.
pub struct LlmAudioUserSimulator {
    config: LlmAudioUserSimulatorConfig,
    text_simulator: Box<dyn UserSimulator + Send + Sync>,
    audio_llm: Box<dyn BaseLlm>,
}

impl LlmAudioUserSimulator {
    /// `_LlmAudioUserSimulator.__init__`. `text_simulator` is the
    /// already-constructed inner simulator to wrap — see the module doc
    /// for why building it from `config`'s own text-generation fields
    /// (the `UserSimulatorProvider` composition) isn't wired here.
    pub fn new(
        config: &Value,
        text_simulator: Box<dyn UserSimulator + Send + Sync>,
    ) -> Result<Self, LlmAudioUserSimulatorError> {
        let config: LlmAudioUserSimulatorConfig =
            parse_simulator_config(config).map_err(LlmAudioUserSimulatorError::Config)?;
        config.validate()?;
        let audio_llm = default_registry()
            .read()
            .expect("llm registry lock poisoned")
            .new_llm(&config.audio_model)?;
        Ok(Self {
            config,
            text_simulator,
            audio_llm,
        })
    }

    /// `_LlmAudioUserSimulator._generate_audio` — see the module doc for
    /// the streaming-to-materialized-`Vec` narrowing.
    async fn generate_audio(
        &self,
        text: &str,
    ) -> Result<(Vec<u8>, String), LlmAudioUserSimulatorError> {
        let mut llm_request = LlmRequest::new(self.config.audio_model.clone());
        llm_request.config = self.config.audio_model_configuration.clone();
        llm_request.contents = vec![Content::new(AUTHOR_USER, vec![Part::text(text)])];

        let responses = self
            .audio_llm
            .generate_content_async(&llm_request, false)
            .await?;

        let mut audio_bytes = Vec::new();
        let mut mime_type = "audio/pcm".to_string();
        for llm_response in &responses {
            if let Some(error_code) = &llm_response.error_code {
                return Err(LlmAudioUserSimulatorError::AudioGenerationFailed(format!(
                    "{error_code} — {}",
                    llm_response.error_message.as_deref().unwrap_or("")
                )));
            }
            let Some(content) = &llm_response.content else {
                continue;
            };
            for part in &content.parts {
                let Some(inline_data) = &part.inline_data else {
                    continue;
                };
                if let Some(bytes) = blob_data(inline_data) {
                    audio_bytes.extend(bytes);
                    if let Some(part_mime_type) = &inline_data.mime_type {
                        mime_type = part_mime_type.clone();
                    }
                }
            }
        }

        if audio_bytes.is_empty() {
            return Err(LlmAudioUserSimulatorError::NoAudioData);
        }
        Ok((audio_bytes, mime_type))
    }

    /// `_LlmAudioUserSimulator.to_audio_content` — the single, reusable
    /// audio-generation entry point.
    pub async fn to_audio_content(
        &self,
        text: &str,
    ) -> Result<Content, LlmAudioUserSimulatorError> {
        let mut parts = Vec::new();
        if self.config.include_text_with_audio {
            parts.push(Part::text(text));
        }

        let (audio_bytes, mime_type) = self.generate_audio(text).await?;
        let live_audio_bytes = crate::audio_utils::to_live_input(&audio_bytes, Some(&mime_type));
        parts.push(Part {
            inline_data: Some(MediaBlobStub {
                mime_type: Some(crate::audio_utils::LIVE_INPUT_MIME_TYPE.to_string()),
                rest: Some(Value::Map(vec![(
                    "data".to_string(),
                    Value::String(base64_encode(&live_audio_bytes)),
                )])),
            }),
            ..Default::default()
        });

        Ok(Content::new(AUTHOR_USER, parts))
    }
}

impl UserSimulator for LlmAudioUserSimulator {
    fn get_next_user_message<'a>(
        &'a mut self,
        events: &'a [adk_events::Event],
    ) -> BoxFuture<'a, Result<NextUserMessage, String>> {
        Box::pin(async move {
            let text_result = self
                .text_simulator
                .get_next_user_message(events)
                .await
                .map_err(|error| error.to_string())?;

            if text_result.status != Status::Success {
                return Ok(text_result);
            }

            let mut text = String::new();
            if let Some(user_message) = &text_result.user_message {
                for part in &user_message.parts {
                    if let Some(part_text) = &part.text {
                        text.push_str(part_text);
                    }
                }
            }

            if text.is_empty() {
                return Ok(text_result);
            }

            let user_message = self
                .to_audio_content(&text)
                .await
                .map_err(|error| error.to_string())?;
            Ok(NextUserMessage {
                status: Status::Success,
                user_message: Some(user_message),
            })
        })
    }

    /// `_LlmAudioUserSimulator.get_simulation_evaluator` — see the module
    /// doc for why this panics rather than returning `None`.
    fn get_simulation_evaluator(&self) -> Option<Box<dyn crate::evaluator::Evaluator>> {
        unimplemented!(
            "_LlmAudioUserSimulator.get_simulation_evaluator is `raise NotImplementedError()` \
             in the source"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_models::base_llm::BaseLlmError;
    use adk_models::llm_response::LlmResponse;
    use std::future::Future;
    use std::pin::Pin;

    #[test]
    fn base64_round_trips() {
        let data = b"hello world audio bytes!!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    struct FixedAudioLlm {
        model: String,
        response: LlmResponse,
    }

    impl BaseLlm for FixedAudioLlm {
        fn model(&self) -> &str {
            &self.model
        }
        fn type_name(&self) -> &'static str {
            "FixedAudioLlm"
        }
        fn generate_content_async<'a>(
            &'a self,
            _llm_request: &'a LlmRequest,
            _stream: bool,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<LlmResponse>, BaseLlmError>> + Send + 'a>>
        {
            let response = self.response.clone();
            Box::pin(async move { Ok(vec![response]) })
        }
    }

    fn audio_response(bytes: &[u8], mime_type: &str) -> LlmResponse {
        LlmResponse {
            content: Some(Content::new(
                "model",
                vec![Part {
                    inline_data: Some(MediaBlobStub {
                        mime_type: Some(mime_type.to_string()),
                        rest: Some(Value::Map(vec![(
                            "data".to_string(),
                            Value::String(base64_encode(bytes)),
                        )])),
                    }),
                    ..Default::default()
                }],
            )),
            ..Default::default()
        }
    }

    struct FixedTextSimulator {
        text: String,
        used: bool,
    }

    impl UserSimulator for FixedTextSimulator {
        fn get_next_user_message<'a>(
            &'a mut self,
            _events: &'a [adk_events::Event],
        ) -> BoxFuture<'a, Result<NextUserMessage, String>> {
            let status = if self.used {
                Status::TurnLimitReached
            } else {
                self.used = true;
                Status::Success
            };
            let user_message = if status == Status::Success {
                Some(Content::new(
                    AUTHOR_USER,
                    vec![Part::text(self.text.clone())],
                ))
            } else {
                None
            };
            Box::pin(async move {
                Ok(NextUserMessage {
                    status,
                    user_message,
                })
            })
        }
        fn get_simulation_evaluator(&self) -> Option<Box<dyn crate::evaluator::Evaluator>> {
            None
        }
    }

    fn simulator_with(
        text: &str,
        response: LlmResponse,
        include_text_with_audio: bool,
    ) -> LlmAudioUserSimulator {
        LlmAudioUserSimulator {
            config: LlmAudioUserSimulatorConfig {
                include_text_with_audio,
                ..Default::default()
            },
            text_simulator: Box::new(FixedTextSimulator {
                text: text.to_string(),
                used: false,
            }),
            audio_llm: Box::new(FixedAudioLlm {
                model: "test-audio-model".to_string(),
                response,
            }),
        }
    }

    #[rusty_tokio::test]
    async fn to_audio_content_includes_text_and_audio_parts_by_default() {
        let simulator = simulator_with("hello", audio_response(b"PCMDATA", "audio/pcm"), true);
        let content = simulator.to_audio_content("hello").await.unwrap();
        assert_eq!(content.parts.len(), 2);
        assert_eq!(content.parts[0].text.as_deref(), Some("hello"));
        assert!(content.parts[1].inline_data.is_some());
    }

    #[rusty_tokio::test]
    async fn to_audio_content_omits_text_when_disabled() {
        let simulator = simulator_with("hello", audio_response(b"PCMDATA", "audio/pcm"), false);
        let content = simulator.to_audio_content("hello").await.unwrap();
        assert_eq!(content.parts.len(), 1);
        assert!(content.parts[0].inline_data.is_some());
    }

    #[rusty_tokio::test]
    async fn to_audio_content_errors_when_no_audio_bytes_are_returned() {
        let simulator = simulator_with(
            "hello",
            LlmResponse {
                content: Some(Content::new("model", vec![Part::text("no audio here")])),
                ..Default::default()
            },
            true,
        );
        let result = simulator.to_audio_content("hello").await;
        assert!(matches!(
            result,
            Err(LlmAudioUserSimulatorError::NoAudioData)
        ));
    }

    #[rusty_tokio::test]
    async fn get_next_user_message_delegates_text_then_converts_to_audio() {
        let mut simulator = simulator_with("hi there", audio_response(b"BYTES", "audio/pcm"), true);
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::Success);
        let content = result.user_message.unwrap();
        assert_eq!(content.parts[0].text.as_deref(), Some("hi there"));
    }

    #[rusty_tokio::test]
    async fn get_next_user_message_passes_through_a_non_success_text_result() {
        let mut simulator = simulator_with("hi there", audio_response(b"BYTES", "audio/pcm"), true);
        // First call succeeds and marks the text simulator as used; the
        // second call returns TurnLimitReached, which must pass through
        // unconverted.
        let _ = simulator.get_next_user_message(&[]).await.unwrap();
        let result = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(result.status, Status::TurnLimitReached);
        assert!(result.user_message.is_none());
    }

    #[test]
    fn config_defaults_match_the_source() {
        let config = LlmAudioUserSimulatorConfig::default();
        assert_eq!(config.simulator_type, "llm_audio");
        assert_eq!(config.audio_model, "cloud_tts");
        assert_eq!(config.max_allowed_invocations, 20);
        assert!(config.include_text_with_audio);
        assert!(!config.include_function_calls);
    }

    #[test]
    fn validate_rejects_custom_instructions_missing_placeholders() {
        let config = LlmAudioUserSimulatorConfig {
            custom_instructions: Some("no placeholders here".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(LlmAudioUserSimulatorError::InvalidCustomInstructions)
        ));
    }

    #[test]
    fn validate_accepts_custom_instructions_with_all_placeholders() {
        let config = LlmAudioUserSimulatorConfig {
            custom_instructions: Some(
                "{{ stop_signal }} {{ conversation_plan }} {{ conversation_history }}".to_string(),
            ),
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }
}
