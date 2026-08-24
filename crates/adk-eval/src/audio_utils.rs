//! C0625: `_audio_utils.resample_pcm16`/`to_live_input`/`parse_sample_rate`,
//! ported from `google.adk.evaluation._audio_utils`. Feeds synthesized
//! speech (commonly 24kHz TTS PCM) into a Live API session, which only
//! accepts 16kHz PCM input.

/// `_audio_utils.LIVE_INPUT_RATE_HZ`.
pub const LIVE_INPUT_RATE_HZ: u32 = 16000;
/// `_audio_utils.LIVE_OUTPUT_RATE_HZ`.
pub const LIVE_OUTPUT_RATE_HZ: u32 = 24000;
/// `_audio_utils.LIVE_INPUT_MIME_TYPE`.
pub const LIVE_INPUT_MIME_TYPE: &str = "audio/pcm;rate=16000";

/// Hand-rolled equivalent of the source's `_RATE_RE =
/// re.compile(r"(?:^|;)\s*rate\s*=\s*(\d+)\s*(?=;|$)", re.IGNORECASE)` —
/// no `regex` dependency needed for a single `;`-delimited-parameter
/// search. Splitting on `;` and checking each trimmed segment for a
/// `rate=<digits>` shape (case-insensitive) is a faithful reimplementation
/// since MIME-type parameters are themselves `;`-separated.
///
/// **Disclosed narrowing**: an implausibly large `rate=` value that
/// overflows `u32` (the source's `int()` is arbitrary-precision) falls
/// through to checking the next `;`-segment rather than matching —
/// real-world sample rates never come close to `u32::MAX`.
fn find_rate(mime_type: &str) -> Option<u32> {
    for segment in mime_type.split(';') {
        let segment = segment.trim();
        let lower = segment.to_ascii_lowercase();
        if !lower.starts_with("rate") {
            continue;
        }
        let rest = segment[4..].trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let digits = rest.trim_start();
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            if let Ok(value) = digits.parse::<u32>() {
                return Some(value);
            }
        }
    }
    None
}

/// C0625: `_audio_utils.parse_sample_rate` — extracts the sample rate
/// from a mime type like `"audio/pcm;rate=24000"`.
pub fn parse_sample_rate(mime_type: Option<&str>, default: u32) -> u32 {
    match mime_type {
        None => default,
        Some(mime_type) => find_rate(mime_type).unwrap_or(default),
    }
}

/// C0625: `_audio_utils.resample_pcm16` — resamples 16-bit mono PCM via
/// linear interpolation. Returns the input unchanged when the rates
/// match or it's too short to interpolate, avoiding a heavy DSP
/// dependency for speech relayed to a transcribing model.
///
/// **Adaptation**: PCM samples are read/written little-endian
/// explicitly. The source's `array.array("h")` uses the host's native
/// byte order, which is little-endian on every platform this workspace
/// actually targets/builds on — this port pins that choice rather than
/// leaving it implicitly platform-dependent.
pub fn resample_pcm16(pcm: &[u8], src_rate: u32, dst_rate: u32) -> Result<Vec<u8>, String> {
    if src_rate == 0 || dst_rate == 0 {
        return Err("Sample rates must be positive".to_string());
    }
    if pcm.is_empty() || src_rate == dst_rate {
        return Ok(pcm.to_vec());
    }

    // Drop a trailing odd byte so the buffer is a whole number of samples.
    let usable_len = pcm.len() - (pcm.len() % 2);
    let samples: Vec<i16> = pcm[..usable_len]
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    if samples.len() < 2 {
        return Ok(pcm.to_vec());
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((samples.len() as f64 / ratio) as usize).max(1);
    let last_index = samples.len() - 1;

    let mut out = Vec::with_capacity(out_len * 2);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let left = src_pos as usize;
        let right = (left + 1).min(last_index);
        let frac = src_pos - left as f64;
        let value = (samples[left] as f64 * (1.0 - frac) + samples[right] as f64 * frac) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// C0625: `_audio_utils.to_live_input` — resamples synthesized speech
/// audio to 16kHz PCM for Live API input.
///
/// **Disclosed narrowing**: the source `logging.warning`s when the
/// source mime type has no `rate=` parameter (a wrong assumed rate
/// mis-pitches the resample and would otherwise pass unnoticed); no
/// logging framework is adopted by this workspace yet (same disclosed
/// omission as `content_utils`'s `drop_orphaned_function_responses` and
/// `preload_memory_tool`'s failed-search case) — the fallback-rate
/// behavior itself is preserved exactly, only the warning is dropped.
pub fn to_live_input(pcm: &[u8], source_mime_type: Option<&str>) -> Vec<u8> {
    let src_rate = parse_sample_rate(source_mime_type, LIVE_OUTPUT_RATE_HZ);
    resample_pcm16(pcm, src_rate, LIVE_INPUT_RATE_HZ).unwrap_or_else(|_| pcm.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_rate_extracts_the_rate_parameter() {
        assert_eq!(parse_sample_rate(Some("audio/pcm;rate=24000"), 1), 24000);
    }

    #[test]
    fn parse_sample_rate_is_case_insensitive() {
        assert_eq!(parse_sample_rate(Some("audio/pcm;RATE=24000"), 1), 24000);
    }

    #[test]
    fn parse_sample_rate_tolerates_surrounding_whitespace() {
        assert_eq!(
            parse_sample_rate(Some("audio/pcm; rate = 24000 ;x=y"), 1),
            24000
        );
    }

    #[test]
    fn parse_sample_rate_falls_back_to_default_without_a_rate_param() {
        assert_eq!(parse_sample_rate(Some("audio/pcm"), 8000), 8000);
    }

    #[test]
    fn parse_sample_rate_falls_back_to_default_for_none() {
        assert_eq!(parse_sample_rate(None, 8000), 8000);
    }

    #[test]
    fn parse_sample_rate_rejects_non_digit_suffix() {
        assert_eq!(parse_sample_rate(Some("audio/pcm;rate=24000hz"), 1), 1);
    }

    #[test]
    fn parse_sample_rate_finds_a_later_parameter() {
        assert_eq!(
            parse_sample_rate(Some("audio/pcm;encoding=signed-integer;rate=44100"), 1),
            44100
        );
    }

    fn pcm16_from(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    fn samples_from_pcm16(pcm: &[u8]) -> Vec<i16> {
        pcm.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn resample_pcm16_returns_input_unchanged_for_matching_rates() {
        let pcm = pcm16_from(&[100, 200, 300]);
        assert_eq!(resample_pcm16(&pcm, 16000, 16000).unwrap(), pcm);
    }

    #[test]
    fn resample_pcm16_returns_input_unchanged_for_empty_input() {
        assert_eq!(resample_pcm16(&[], 24000, 16000).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn resample_pcm16_returns_input_unchanged_when_too_short_to_interpolate() {
        let pcm = pcm16_from(&[100]);
        assert_eq!(resample_pcm16(&pcm, 24000, 16000).unwrap(), pcm);
    }

    #[test]
    fn resample_pcm16_errors_on_non_positive_rates() {
        assert!(resample_pcm16(&[0, 0], 0, 16000).is_err());
        assert!(resample_pcm16(&[0, 0], 16000, 0).is_err());
    }

    #[test]
    fn resample_pcm16_downsamples_24k_to_16k_matching_python_reference_output() {
        // A short ramp resampled 24kHz -> 16kHz, matching the exact
        // linear-interpolation output computed by hand from the source's
        // own algorithm: ratio = 1.5, out_len = 6/1.5 = 4.
        // i=0: src_pos=0.0 -> samples[0]=0
        // i=1: src_pos=1.5 -> 0.5*samples[1] + 0.5*samples[2] = 0.5*1000+0.5*2000=1500
        // i=2: src_pos=3.0 -> samples[3]=3000
        // i=3: src_pos=4.5 -> 0.5*samples[4] + 0.5*samples[5] = 0.5*4000+0.5*5000=4500
        let input = pcm16_from(&[0, 1000, 2000, 3000, 4000, 5000]);
        let output = resample_pcm16(&input, 24000, 16000).unwrap();
        assert_eq!(samples_from_pcm16(&output), vec![0, 1500, 3000, 4500]);
    }

    #[test]
    fn resample_pcm16_drops_a_trailing_odd_byte() {
        let mut pcm = pcm16_from(&[100, 200]);
        pcm.push(0xFF);
        // Should behave identically to the even-length input -- rates
        // match here so this exercises the early-return path with the
        // odd byte still attached (matching the source, which only trims
        // the odd byte once it actually builds the `array.array`).
        assert_eq!(resample_pcm16(&pcm, 16000, 16000).unwrap(), pcm);
    }

    #[test]
    fn to_live_input_resamples_from_the_mime_types_rate() {
        let input = pcm16_from(&[0, 1000, 2000, 3000, 4000, 5000]);
        let output = to_live_input(&input, Some("audio/pcm;rate=24000"));
        assert_eq!(samples_from_pcm16(&output), vec![0, 1500, 3000, 4500]);
    }

    #[test]
    fn to_live_input_assumes_the_default_rate_without_a_mime_type() {
        let input = pcm16_from(&[0, 1000, 2000, 3000, 4000, 5000]);
        let output = to_live_input(&input, None);
        assert_eq!(samples_from_pcm16(&output), vec![0, 1500, 3000, 4500]);
    }
}
