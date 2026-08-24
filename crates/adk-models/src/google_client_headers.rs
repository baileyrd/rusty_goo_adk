//! Capability C0133 (partial)/C0932: tracking headers, ported from
//! `google.adk.utils._google_client_headers` /
//! `google.adk.utils._client_labels_utils`.
//!
//! **C0932, no longer deferred**: the source's `client_label_context`/
//! `EVAL_CLIENT_LABEL` (a `contextvars`-scoped extra label pushed onto the
//! header value, used by the Evals surface to attribute its own calls) is
//! now ported below as [`ClientLabelScope`]/[`EVAL_CLIENT_LABEL`] — nothing
//! in this workspace *calls* it yet (Evals isn't built), same as
//! `SKIP_THOUGHT_SIGNATURE_VALIDATOR` (C0929): the capability is real and
//! independently testable even without its eventual caller. Rust has no
//! `contextvars` (implicit, task-local, propagates across `.await` points
//! without an explicit handle); this port uses a `thread_local!`
//! [`std::cell::RefCell`] instead, scoped by an RAII guard
//! ([`ClientLabelScope`]) rather than the source's `@contextmanager` —
//! the same "`Drop` replaces `try`/`finally`" pattern already used for
//! `adk-features::TemporaryFeatureOverride`. **Adaptation, disclosed**: a
//! `thread_local` is *thread*-scoped, not *task*-scoped — it won't follow
//! a value across an `.await` that resumes on a different worker thread
//! the way Python's `contextvars.ContextVar` (which the source explicitly
//! chose over a plain global specifically for async-safety) does. This
//! matters once real concurrent multi-task Evals-equivalent work exists in
//! this workspace; nothing here yet exercises it across an actual
//! thread-hop, so it's a disclosed gap rather than a proven bug.
//!
//! **Adaptation**: the source's `gl-python/{sys.version}` language label
//! reports the interpreter version backing the `google-adk` package. There
//! is no single top-level `adk` package version yet (this migration is
//! split across per-phase crates), and no portable way to read the
//! *rustc* version at runtime without a build script. `gl-rust` is used
//! as the language token with no version suffix — the label's job (telling
//! Google's server-side usage pipeline which client language made the
//! call) still holds; only the fine-grained version detail is dropped.

use std::cell::RefCell;
use std::env;

const ADK_LABEL: &str = "google-adk";
const LANGUAGE_LABEL: &str = "gl-rust";
const AGENT_ENGINE_TELEMETRY_TAG: &str = "remote_reasoning_engine";
const AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME: &str = "GOOGLE_CLOUD_AGENT_ENGINE_ID";

/// C0932: `_client_labels_utils.EVAL_CLIENT_LABEL` — the label used to
/// denote calls emerging to an external system as part of Evals.
///
/// **Adaptation**: the source interpolates its own package version
/// (`f"google-adk-eval/{version.__version__}"`); this port uses
/// `CARGO_PKG_VERSION`, the same substitution `default_labels` below
/// already makes for the `google-adk/{version}` framework token.
pub const EVAL_CLIENT_LABEL: &str = concat!("google-adk-eval/", env!("CARGO_PKG_VERSION"));

thread_local! {
    // C0932: `_client_labels_utils._LABEL_CONTEXT` — see the module doc for
    // why this port uses a thread-local `RefCell` rather than the source's
    // async-safe `contextvars.ContextVar`.
    static LABEL_CONTEXT: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// C0932: `_client_labels_utils.client_label_context` — an RAII guard that
/// scopes `client_label` for its lifetime, restoring the prior value (via
/// [`Drop`]) when it goes out of scope. Mirrors the source's
/// `@contextmanager`; see the module doc for the `Drop`-replaces-
/// `try`/`finally` convention this follows.
///
/// # Panics
///
/// Matches the source's `raise ValueError`: panics if a label is already
/// scoped on this thread — only one client label may be active at a time.
pub struct ClientLabelScope {
    previous: Option<String>,
}

impl ClientLabelScope {
    pub fn new(client_label: impl Into<String>) -> Self {
        let previous = LABEL_CONTEXT.with_borrow_mut(|current| {
            if current.is_some() {
                panic!("Client label already exists. You can only add one client label.");
            }
            current.replace(client_label.into())
        });
        Self { previous }
    }
}

impl Drop for ClientLabelScope {
    fn drop(&mut self) {
        LABEL_CONTEXT.with_borrow_mut(|current| *current = self.previous.take());
    }
}

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

/// `_client_labels_utils.get_client_labels` — now including the
/// [`ClientLabelScope`]-scoped label (C0932) when one is active on the
/// current thread.
pub fn get_client_labels(framework_label: Option<&str>) -> Vec<String> {
    let mut labels = default_labels(framework_label);
    if let Some(scoped) = LABEL_CONTEXT.with_borrow(|current| current.clone()) {
        labels.push(scoped);
    }
    labels
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

    #[test]
    fn client_label_scope_appends_the_scoped_label_while_active() {
        assert_eq!(get_client_labels(None).len(), 2);
        {
            let _scope = ClientLabelScope::new(EVAL_CLIENT_LABEL);
            let labels = get_client_labels(None);
            assert_eq!(labels.len(), 3);
            assert_eq!(labels[2], EVAL_CLIENT_LABEL);
        }
        // Dropping the scope restores the unscoped label set.
        assert_eq!(get_client_labels(None).len(), 2);
    }

    #[test]
    #[should_panic(expected = "Client label already exists")]
    fn client_label_scope_rejects_nesting() {
        let _outer = ClientLabelScope::new("outer");
        let _inner = ClientLabelScope::new("inner");
    }

    #[test]
    fn eval_client_label_matches_the_expected_shape() {
        assert!(EVAL_CLIENT_LABEL.starts_with("google-adk-eval/"));
    }
}
