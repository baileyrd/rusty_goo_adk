//! Capability C0133 (partial): tracking headers, ported from
//! `google.adk.utils._google_client_headers` /
//! `google.adk.utils._client_labels_utils`.
//!
//! **Deferred**: the source's `client_label_context`/`EVAL_CLIENT_LABEL`
//! (a `contextvars`-scoped extra label pushed onto the header value, used by
//! the Evals surface to attribute its own calls) is omitted — nothing in
//! this port calls it yet, since Evals isn't built. `get_client_labels`
//! below always returns just the two unconditional labels; adding the
//! scoped third label is a small, self-contained addition to make once an
//! Evals-equivalent caller actually needs it.
//!
//! **Adaptation**: the source's `gl-python/{sys.version}` language label
//! reports the interpreter version backing the `google-adk` package. There
//! is no single top-level `adk` package version yet (this migration is
//! split across per-phase crates), and no portable way to read the
//! *rustc* version at runtime without a build script. `gl-rust` is used
//! as the language token with no version suffix — the label's job (telling
//! Google's server-side usage pipeline which client language made the
//! call) still holds; only the fine-grained version detail is dropped.

use std::env;

const ADK_LABEL: &str = "google-adk";
const LANGUAGE_LABEL: &str = "gl-rust";
const AGENT_ENGINE_TELEMETRY_TAG: &str = "remote_reasoning_engine";
const AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME: &str = "GOOGLE_CLOUD_AGENT_ENGINE_ID";

/// `_client_labels_utils._get_default_labels`. `framework_label` mirrors the
/// source's optional SemVer build-metadata suffix (e.g. `"managed_agent"`).
fn default_labels(framework_label: Option<&str>) -> Vec<String> {
    let mut framework_token = format!("{ADK_LABEL}/{}", env!("CARGO_PKG_VERSION"));
    if let Some(label) = framework_label {
        framework_token = format!("{framework_token}+{label}");
    } else if env::var(AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME).is_ok() {
        framework_token = format!("{framework_token}+{AGENT_ENGINE_TELEMETRY_TAG}");
    }
    vec![framework_token, LANGUAGE_LABEL.to_string()]
}

/// `_client_labels_utils.get_client_labels` (without the deferred
/// context-scoped label — see the module doc).
pub fn get_client_labels(framework_label: Option<&str>) -> Vec<String> {
    default_labels(framework_label)
}

/// `_google_client_headers.get_tracking_headers`.
pub fn get_tracking_headers(framework_label: Option<&str>) -> Vec<(String, String)> {
    let header_value = get_client_labels(framework_label).join(" ");
    vec![
        ("x-goog-api-client".to_string(), header_value.clone()),
        ("user-agent".to_string(), header_value),
    ]
}

/// `_google_client_headers.merge_tracking_headers` — merges tracking
/// headers into `headers`, appending onto (not replacing) any pre-existing
/// value for the same header, de-duplicating by space-separated token.
pub fn merge_tracking_headers(
    headers: &[(String, String)],
    framework_label: Option<&str>,
) -> Vec<(String, String)> {
    let mut merged: Vec<(String, String)> = headers.to_vec();

    for (key, tracking_value) in get_tracking_headers(framework_label) {
        match merged.iter_mut().find(|(k, _)| k == &key) {
            None => merged.push((key, tracking_value)),
            Some((_, existing)) if existing.is_empty() => *existing = tracking_value,
            Some((_, existing)) => {
                let mut parts: Vec<&str> = tracking_value.split(' ').collect();
                for custom_part in existing.split(' ') {
                    if !parts.contains(&custom_part) {
                        parts.push(custom_part);
                    }
                }
                *existing = parts.join(" ");
            }
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_client_labels_reports_the_adk_and_language_tokens() {
        let labels = get_client_labels(None);
        assert_eq!(labels.len(), 2);
        assert!(labels[0].starts_with("google-adk/"));
        assert_eq!(labels[1], "gl-rust");
    }

    #[test]
    fn get_client_labels_appends_a_framework_label_suffix() {
        let labels = get_client_labels(Some("managed_agent"));
        assert!(labels[0].ends_with("+managed_agent"));
    }

    #[test]
    fn get_tracking_headers_sets_both_header_names_to_the_same_value() {
        let headers = get_tracking_headers(None);
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].0, "x-goog-api-client");
        assert_eq!(headers[1].0, "user-agent");
        assert_eq!(headers[0].1, headers[1].1);
    }

    #[test]
    fn merge_tracking_headers_adds_tracking_headers_when_absent() {
        let merged = merge_tracking_headers(&[], None);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_tracking_headers_appends_onto_an_existing_value_without_duplicating() {
        let existing = vec![("x-goog-api-client".to_string(), "custom-token".to_string())];
        let merged = merge_tracking_headers(&existing, None);
        let (_, value) = merged
            .iter()
            .find(|(k, _)| k == "x-goog-api-client")
            .unwrap();
        assert!(value.starts_with("google-adk/"));
        assert!(value.ends_with(" custom-token"));

        // Merging again must not duplicate the tracking tokens.
        let merged_again = merge_tracking_headers(&merged, None);
        let (_, value_again) = merged_again
            .iter()
            .find(|(k, _)| k == "x-goog-api-client")
            .unwrap();
        assert_eq!(value, value_again);
    }

    #[test]
    fn merge_tracking_headers_leaves_unrelated_headers_untouched() {
        let existing = vec![("content-type".to_string(), "application/json".to_string())];
        let merged = merge_tracking_headers(&existing, None);
        assert!(merged
            .iter()
            .any(|(k, v)| k == "content-type" && v == "application/json"));
    }
}
