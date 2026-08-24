//! Capabilities C0509/C0531 (partial)/C0532: small, self-contained
//! OAuth2/mTLS helper functions, ported from
//! `google.adk.auth.auth_handler::_normalize_oauth_scopes`,
//! `google.adk.utils._mtls_utils` (a narrow subset — see below), and
//! `google.adk.auth.oauth2_credential_util::update_credential_with_tokens`.
//!
//! **Why these three, together**: each is small and already testable
//! against real types this port has (`OAuth2Auth`/`AuthCredential`,
//! C0494-C0499) without needing `authlib` or `google-auth` — the two
//! third-party dependencies the rest of Phase 9's OAuth2 flow
//! (`create_oauth2_session`, C0530; `configure_session_for_mtls`,
//! part of C0531) is blocked on. No new dependency was added for this
//! batch: URL-host inspection/rewriting is done with plain string
//! operations (the `.googleapis.com`/`.mtls.googleapis.com` suffix
//! check and swap don't need a full URL-parsing crate), the same
//! "hand-roll a small self-contained algorithm rather than pull in a
//! dependency" precedent as `bash_tool.rs`'s `shlex_split`.
//!
//! **C0531, scoped to its portable half**: `_mtls_utils.py` has two
//! kinds of functions — pure env-var/URL-string logic
//! (`is_non_mtls_googleapis_endpoint`, `effective_googleapis_endpoint`,
//! the `GOOGLE_API_USE_MTLS_ENDPOINT` setting parse), and real
//! client-certificate loading/mounting (`configure_session_for_mtls`,
//! `MtlsClientCerts`, `get_api_endpoint`'s
//! `mtls.has_default_client_cert_source()` check) that needs
//! `google.auth.transport.mtls` — unported, no such crate is a
//! workspace dependency. This batch ports only the first kind.
//! [`use_client_cert_effective`] is a partial port of its own: the
//! source tries `mtls.should_use_client_cert()` first and only falls
//! back to the `GOOGLE_API_USE_CLIENT_CERTIFICATE` env var on
//! `ImportError`/`AttributeError`; this port always takes that
//! fallback branch, since `google.auth`'s own cert-availability probe
//! isn't available here — a real, disclosed behavior gap (this port
//! can report "the env var says use a cert" but never "a cert is
//! actually available"), not a cosmetic one. Because the real cert
//! step is unported, nothing in this port can complete the full
//! call-site gating the source performs (rewrite the token endpoint
//! only once a certificate is actually mounted) — that integration is
//! left with `create_oauth2_session` (C0530), still fully `REQUIRED`.

use std::collections::BTreeMap;

use rusty_serde::value::Value;

use crate::auth_credential::AuthCredential;

/// `auth.auth_handler._normalize_oauth_scopes`'s input shape —
/// `dict[str, str] | list[str] | None` in the source. Modeled as an
/// enum rather than `Option` alone since Rust needs the dict/list
/// distinction spelled out at the type level.
pub enum OAuthScopes {
    /// A docs-style `{scope: description}` mapping — only the keys
    /// matter.
    Described(BTreeMap<String, String>),
    List(Vec<String>),
}

/// C0509: normalizes OAuth scopes into the list shape `authlib`
/// expects (or, in this port, the shape any future OAuth2 client
/// layer would expect).
pub fn normalize_oauth_scopes(scopes: Option<&OAuthScopes>) -> Vec<String> {
    match scopes {
        None => Vec::new(),
        Some(OAuthScopes::Described(map)) if map.is_empty() => Vec::new(),
        Some(OAuthScopes::List(list)) if list.is_empty() => Vec::new(),
        Some(OAuthScopes::Described(map)) => map.keys().cloned().collect(),
        Some(OAuthScopes::List(list)) => list.clone(),
    }
}

const GOOGLEAPIS_SUFFIX: &str = ".googleapis.com";
const MTLS_GOOGLEAPIS_SUFFIX: &str = ".mtls.googleapis.com";

/// `utils._mtls_utils.MtlsEndpoint` — the `GOOGLE_API_USE_MTLS_ENDPOINT`
/// setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtlsEndpointSetting {
    Auto,
    Always,
    Never,
}

/// `utils._mtls_utils._mtls_endpoint_setting` — reads
/// `GOOGLE_API_USE_MTLS_ENDPOINT`, defaulting to (and falling back to
/// on any unrecognized value) `Auto`.
fn mtls_endpoint_setting() -> MtlsEndpointSetting {
    match std::env::var("GOOGLE_API_USE_MTLS_ENDPOINT")
        .unwrap_or_else(|_| "auto".to_string())
        .to_lowercase()
        .as_str()
    {
        "always" => MtlsEndpointSetting::Always,
        "never" => MtlsEndpointSetting::Never,
        _ => MtlsEndpointSetting::Auto,
    }
}

/// `utils._mtls_utils.use_client_cert_effective` — see the module doc
/// for why this port always takes the env-var-fallback branch.
pub fn use_client_cert_effective() -> bool {
    std::env::var("GOOGLE_API_USE_CLIENT_CERTIFICATE")
        .map(|value| value.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Extracts the hostname portion of an absolute `scheme://host[:port]/...`
/// URL — enough for the `.googleapis.com` suffix checks below without
/// pulling in a full URL-parsing crate (see the module doc).
fn hostname(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://")?.1;
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    Some(host.split(':').next().unwrap_or(host))
}

/// `utils._mtls_utils.is_non_mtls_googleapis_endpoint` — C0531.
pub fn is_non_mtls_googleapis_endpoint(url: &str) -> bool {
    if url.is_empty() {
        return false;
    }
    match hostname(url) {
        Some(host) => host.ends_with(GOOGLEAPIS_SUFFIX) && !host.contains(MTLS_GOOGLEAPIS_SUFFIX),
        None => false,
    }
}

/// `utils._mtls_utils.effective_googleapis_endpoint` — C0531. Rewrites
/// a `*.googleapis.com` URL to its `.mtls.googleapis.com` variant,
/// honoring `GOOGLE_API_USE_MTLS_ENDPOINT=never` as an opt-out. See
/// the module doc: nothing calls this gated on an actually-mounted
/// client certificate in this port yet, since that step is unported.
pub fn effective_googleapis_endpoint(url: &str) -> String {
    if !is_non_mtls_googleapis_endpoint(url)
        || mtls_endpoint_setting() == MtlsEndpointSetting::Never
    {
        return url.to_string();
    }
    let host = hostname(url).unwrap_or_default();
    let new_host = format!(
        "{}{MTLS_GOOGLEAPIS_SUFFIX}",
        &host[..host.len() - GOOGLEAPIS_SUFFIX.len()]
    );
    url.replacen(host, &new_host, 1)
}

/// C0532: `auth.oauth2_credential_util.update_credential_with_tokens`.
/// `tokens` stands in for `authlib`'s `OAuth2Token` (a dict subclass
/// read only via `.get(...)` in the source) — modeled as the same
/// `BTreeMap<String, Value>` shape this port already uses wherever an
/// opaque dict-like value crosses a boundary (e.g.
/// `preload_memory_tool.rs`'s parsed `user_content`).
pub fn update_credential_with_tokens(
    auth_credential: &mut AuthCredential,
    tokens: &BTreeMap<String, Value>,
) {
    let Some(oauth2) = auth_credential.oauth2.as_mut() else {
        return;
    };
    if tokens.is_empty() {
        return;
    }
    oauth2.access_token = string_field(tokens, "access_token");
    oauth2.refresh_token = string_field(tokens, "refresh_token");
    oauth2.id_token = string_field(tokens, "id_token");
    oauth2.expires_at = int_field(tokens, "expires_at");
    oauth2.expires_in = int_field(tokens, "expires_in");
}

fn string_field(tokens: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    match tokens.get(key) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn int_field(tokens: &BTreeMap<String, Value>, key: &str) -> Option<i64> {
    match tokens.get(key) {
        Some(Value::Int(value)) => Some(*value),
        Some(Value::UInt(value)) => Some(*value as i64),
        Some(Value::Float(value)) => Some(*value as i64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::{AuthCredentialTypes, OAuth2Auth};

    #[test]
    fn normalize_oauth_scopes_returns_empty_for_none() {
        assert_eq!(normalize_oauth_scopes(None), Vec::<String>::new());
    }

    #[test]
    fn normalize_oauth_scopes_returns_empty_for_an_empty_dict_or_list() {
        assert_eq!(
            normalize_oauth_scopes(Some(&OAuthScopes::Described(BTreeMap::new()))),
            Vec::<String>::new()
        );
        assert_eq!(
            normalize_oauth_scopes(Some(&OAuthScopes::List(Vec::new()))),
            Vec::<String>::new()
        );
    }

    #[test]
    fn normalize_oauth_scopes_extracts_keys_from_a_dict() {
        let scopes = OAuthScopes::Described(BTreeMap::from([
            ("read".to_string(), "Read access".to_string()),
            ("write".to_string(), "Write access".to_string()),
        ]));
        assert_eq!(
            normalize_oauth_scopes(Some(&scopes)),
            vec!["read".to_string(), "write".to_string()]
        );
    }

    #[test]
    fn normalize_oauth_scopes_passes_a_list_through() {
        let scopes = OAuthScopes::List(vec!["read".to_string(), "write".to_string()]);
        assert_eq!(
            normalize_oauth_scopes(Some(&scopes)),
            vec!["read".to_string(), "write".to_string()]
        );
    }

    #[test]
    fn is_non_mtls_googleapis_endpoint_matches_a_plain_googleapis_host() {
        assert!(is_non_mtls_googleapis_endpoint(
            "https://oauth2.googleapis.com/token"
        ));
    }

    #[test]
    fn is_non_mtls_googleapis_endpoint_rejects_an_already_mtls_host() {
        assert!(!is_non_mtls_googleapis_endpoint(
            "https://oauth2.mtls.googleapis.com/token"
        ));
    }

    #[test]
    fn is_non_mtls_googleapis_endpoint_rejects_a_non_google_host() {
        assert!(!is_non_mtls_googleapis_endpoint(
            "https://example.com/oauth/token"
        ));
    }

    #[test]
    fn is_non_mtls_googleapis_endpoint_rejects_an_empty_url() {
        assert!(!is_non_mtls_googleapis_endpoint(""));
    }

    #[test]
    fn effective_googleapis_endpoint_rewrites_the_host() {
        assert_eq!(
            effective_googleapis_endpoint("https://oauth2.googleapis.com/token"),
            "https://oauth2.mtls.googleapis.com/token"
        );
    }

    #[test]
    fn effective_googleapis_endpoint_leaves_non_google_hosts_unchanged() {
        assert_eq!(
            effective_googleapis_endpoint("https://example.com/oauth/token"),
            "https://example.com/oauth/token"
        );
    }

    #[test]
    fn effective_googleapis_endpoint_honors_the_never_opt_out() {
        unsafe {
            std::env::set_var("GOOGLE_API_USE_MTLS_ENDPOINT", "never");
        }
        let result = effective_googleapis_endpoint("https://oauth2.googleapis.com/token");
        unsafe {
            std::env::remove_var("GOOGLE_API_USE_MTLS_ENDPOINT");
        }
        assert_eq!(result, "https://oauth2.googleapis.com/token");
    }

    #[test]
    fn use_client_cert_effective_reads_the_env_var() {
        unsafe {
            std::env::set_var("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true");
        }
        let result = use_client_cert_effective();
        unsafe {
            std::env::remove_var("GOOGLE_API_USE_CLIENT_CERTIFICATE");
        }
        assert!(result);
    }

    #[test]
    fn use_client_cert_effective_defaults_to_false() {
        unsafe {
            std::env::remove_var("GOOGLE_API_USE_CLIENT_CERTIFICATE");
        }
        assert!(!use_client_cert_effective());
    }

    #[test]
    fn update_credential_with_tokens_is_a_no_op_without_an_oauth2_credential() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        let tokens =
            BTreeMap::from([("access_token".to_string(), Value::String("abc".to_string()))]);
        update_credential_with_tokens(&mut credential, &tokens);
        assert!(credential.oauth2.is_none());
    }

    #[test]
    fn update_credential_with_tokens_copies_the_expected_fields() {
        let mut credential = AuthCredential {
            oauth2: Some(OAuth2Auth::new()),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        };
        let tokens = BTreeMap::from([
            (
                "access_token".to_string(),
                Value::String("access-123".to_string()),
            ),
            (
                "refresh_token".to_string(),
                Value::String("refresh-456".to_string()),
            ),
            ("id_token".to_string(), Value::String("id-789".to_string())),
            ("expires_at".to_string(), Value::Int(1_700_000_000)),
            ("expires_in".to_string(), Value::Int(3600)),
        ]);
        update_credential_with_tokens(&mut credential, &tokens);
        let oauth2 = credential.oauth2.unwrap();
        assert_eq!(oauth2.access_token.as_deref(), Some("access-123"));
        assert_eq!(oauth2.refresh_token.as_deref(), Some("refresh-456"));
        assert_eq!(oauth2.id_token.as_deref(), Some("id-789"));
        assert_eq!(oauth2.expires_at, Some(1_700_000_000));
        assert_eq!(oauth2.expires_in, Some(3600));
    }

    #[test]
    fn update_credential_with_tokens_leaves_missing_fields_none() {
        let mut credential = AuthCredential {
            oauth2: Some(OAuth2Auth::new()),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        };
        let tokens = BTreeMap::from([(
            "access_token".to_string(),
            Value::String("access-123".to_string()),
        )]);
        update_credential_with_tokens(&mut credential, &tokens);
        let oauth2 = credential.oauth2.unwrap();
        assert_eq!(oauth2.access_token.as_deref(), Some("access-123"));
        assert_eq!(oauth2.refresh_token, None);
        assert_eq!(oauth2.expires_at, None);
    }

    #[test]
    fn update_credential_with_tokens_is_a_no_op_for_empty_tokens() {
        let mut credential = AuthCredential {
            oauth2: Some(OAuth2Auth {
                access_token: Some("existing".to_string()),
                ..OAuth2Auth::new()
            }),
            ..AuthCredential::new(AuthCredentialTypes::OAuth2)
        };
        update_credential_with_tokens(&mut credential, &BTreeMap::new());
        assert_eq!(
            credential.oauth2.unwrap().access_token.as_deref(),
            Some("existing")
        );
    }
}
