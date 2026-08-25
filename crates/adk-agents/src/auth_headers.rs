//! Capability C0522: `auth._auth_headers`, ported from
//! `google.adk.auth._auth_headers` — conversion of a resolved
//! [`AuthCredential`] into HTTP headers.
//!
//! **`logger.warning`, adapted**: the source logs a warning for a
//! non-header API-key location; this port uses `eprintln!` instead, the
//! same disclosed "no logging framework" adaptation already established
//! for `preload_memory_tool.rs`/`feature_registry.rs`'s own notices.
//!
//! **API-key header-name resolution, adapted**: the source reads
//! `auth_scheme.in_`/`auth_scheme.name` via `hasattr`, since `AuthScheme`
//! is a union and only `APIKey` declares those fields — the source's own
//! comment discloses this as an unchecked read the extraction preserved
//! as-is. This port's [`AuthScheme::Security`] wraps a
//! [`SecurityScheme`](crate::auth_schemes::SecurityScheme) enum, so the
//! equivalent is a `match` against [`SecurityScheme::ApiKey`] instead of
//! `hasattr` — same behavior (only the API-key variant carries
//! `in_`/`name`), checked structurally rather than reflectively.

use std::collections::BTreeMap;

use crate::auth_credential::AuthCredential;
use crate::auth_schemes::{ApiKeyIn, AuthScheme, SecurityScheme};

/// C0522: `build_auth_headers` — builds the HTTP headers that carry an
/// exchanged credential. Returns `None` if the credential is absent or
/// can't be expressed as headers.
pub fn build_auth_headers(
    credential: Option<&AuthCredential>,
    auth_scheme: Option<&AuthScheme>,
) -> Option<BTreeMap<String, String>> {
    let credential = credential?;
    let mut headers: Option<BTreeMap<String, String>> = None;

    if let Some(oauth2) = &credential.oauth2 {
        // The token is checked here the same way it is on the HTTP bearer
        // branch below. A failed exchange returns the credential with no
        // access token, and without this the header would be the literal
        // string "Bearer None".
        if let Some(access_token) = &oauth2.access_token {
            headers = Some(BTreeMap::from([(
                "Authorization".to_string(),
                format!("Bearer {access_token}"),
            )]));
        }
    } else if let Some(http) = &credential.http {
        if http.scheme.eq_ignore_ascii_case("bearer") {
            if let Some(token) = &http.credentials.token {
                headers = Some(BTreeMap::from([(
                    "Authorization".to_string(),
                    format!("Bearer {token}"),
                )]));
            }
        } else if http.scheme.eq_ignore_ascii_case("basic") {
            if let (Some(username), Some(password)) =
                (&http.credentials.username, &http.credentials.password)
            {
                let credentials_str = format!("{username}:{password}");
                let encoded = base64_encode(credentials_str.as_bytes());
                headers = Some(BTreeMap::from([(
                    "Authorization".to_string(),
                    format!("Basic {encoded}"),
                )]));
            }
        } else if let Some(token) = &http.credentials.token {
            let scheme = &http.scheme;
            headers = Some(BTreeMap::from([(
                "Authorization".to_string(),
                format!("{scheme} {token}"),
            )]));
        }

        if let Some(additional_headers) = &http.additional_headers {
            let mut merged = headers.unwrap_or_default();
            merged.extend(additional_headers.clone());
            headers = Some(merged);
        }
    } else if let Some(api_key) = &credential.api_key {
        // For API key, use the auth scheme to determine the header name.
        let api_key_scheme = match auth_scheme {
            Some(AuthScheme::Security(security)) => match security.as_ref() {
                SecurityScheme::ApiKey(scheme) => Some(scheme),
                _ => None,
            },
            _ => None,
        };
        if let Some(scheme) = api_key_scheme {
            if scheme.in_ == ApiKeyIn::Header {
                headers = Some(BTreeMap::from([(scheme.name.clone(), api_key.clone())]));
            } else {
                eprintln!(
                    "auth: only header-based API key authentication is supported. Configured \
                     location: {:?}",
                    scheme.in_
                );
            }
        }
        // The source's `else` fallback (`auth_scheme.name`, no `hasattr`
        // guard) reads a field only `APIKey` declares — for any other
        // scheme it's an unchecked attribute access the source's own
        // `# type: ignore[union-attr]` comment flags as unsound, and
        // would raise `AttributeError` at runtime if actually hit. This
        // port's `match` can't express that unsound read at all (every
        // non-`ApiKey` variant has no `name` field to reach for); falling
        // through to `None` here is a disclosed, strictly safer
        // narrowing of a source branch that was already a latent crash.
    }

    headers
}

/// Minimal base64 (standard alphabet, `=` padding) encoder — this port
/// has no `base64` crate dependency yet. `pub`, not `pub(crate)`: reused
/// by `adk-tools::computer_use_tool` (C0447) for the same
/// `base64.b64encode` shape the source's `ComputerUseTool.run_async`
/// uses for its screenshot payload, rather than a second hand-rolled
/// copy — `adk-tools` already depends on `adk-agents`, so this is a
/// pure reuse-across-an-already-satisfied-dependency, not a new edge.
pub fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied();
        let b2 = chunk.get(2).copied();

        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1.unwrap_or(0) >> 4)) as usize] as char);
        if let Some(b1) = b1 {
            out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2.unwrap_or(0) >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if let Some(b2) = b2 {
            out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::{AuthCredentialTypes, HttpAuth, HttpCredentials, OAuth2Auth};
    use crate::auth_schemes::{ApiKeyScheme, HttpScheme};

    #[test]
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64_encode(b"user:pass"), "dXNlcjpwYXNz");
    }

    #[test]
    fn returns_none_for_no_credential() {
        assert_eq!(build_auth_headers(None, None), None);
    }

    #[test]
    fn oauth2_credential_without_an_access_token_yields_no_headers() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth::default());
        assert_eq!(build_auth_headers(Some(&credential), None), None);
    }

    #[test]
    fn oauth2_credential_with_an_access_token_yields_a_bearer_header() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth {
            access_token: Some("abc123".to_string()),
            ..OAuth2Auth::default()
        });
        let headers = build_auth_headers(Some(&credential), None).expect("expected headers");
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer abc123".to_string())
        );
    }

    #[test]
    fn http_bearer_credential_yields_a_bearer_header() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::Http);
        credential.http = Some(HttpAuth::new(
            "bearer",
            HttpCredentials {
                token: Some("xyz".to_string()),
                ..HttpCredentials::default()
            },
        ));
        let headers = build_auth_headers(Some(&credential), None).expect("expected headers");
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer xyz".to_string())
        );
    }

    #[test]
    fn http_basic_credential_yields_a_base64_encoded_header() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::Http);
        credential.http = Some(HttpAuth::new(
            "basic",
            HttpCredentials {
                username: Some("user".to_string()),
                password: Some("pass".to_string()),
                ..HttpCredentials::default()
            },
        ));
        let headers = build_auth_headers(Some(&credential), None).expect("expected headers");
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Basic dXNlcjpwYXNz".to_string())
        );
    }

    #[test]
    fn http_other_scheme_with_a_token_uses_the_scheme_name() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::Http);
        credential.http = Some(HttpAuth::new(
            "digest",
            HttpCredentials {
                token: Some("tok".to_string()),
                ..HttpCredentials::default()
            },
        ));
        let headers = build_auth_headers(Some(&credential), None).expect("expected headers");
        assert_eq!(
            headers.get("Authorization"),
            Some(&"digest tok".to_string())
        );
    }

    #[test]
    fn http_additional_headers_are_merged_in() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::Http);
        let mut http = HttpAuth::new(
            "bearer",
            HttpCredentials {
                token: Some("xyz".to_string()),
                ..HttpCredentials::default()
            },
        );
        http.additional_headers = Some(BTreeMap::from([(
            "X-Trace-Id".to_string(),
            "abc".to_string(),
        )]));
        credential.http = Some(http);
        let headers = build_auth_headers(Some(&credential), None).expect("expected headers");
        assert_eq!(headers.get("X-Trace-Id"), Some(&"abc".to_string()));
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer xyz".to_string())
        );
    }

    #[test]
    fn api_key_credential_with_a_header_scheme_uses_the_schemes_name() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        credential.api_key = Some("secret".to_string());
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Header,
            name: "X-Api-Key".to_string(),
        })));
        let headers =
            build_auth_headers(Some(&credential), Some(&scheme)).expect("expected headers");
        assert_eq!(headers.get("X-Api-Key"), Some(&"secret".to_string()));
    }

    #[test]
    fn api_key_credential_with_a_non_header_scheme_yields_no_headers() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        credential.api_key = Some("secret".to_string());
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::ApiKey(ApiKeyScheme {
            description: None,
            in_: ApiKeyIn::Query,
            name: "api_key".to_string(),
        })));
        assert_eq!(build_auth_headers(Some(&credential), Some(&scheme)), None);
    }

    #[test]
    fn api_key_credential_with_no_scheme_yields_no_headers() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        credential.api_key = Some("secret".to_string());
        assert_eq!(build_auth_headers(Some(&credential), None), None);
    }

    #[test]
    fn api_key_credential_with_a_non_api_key_scheme_yields_no_headers() {
        let mut credential = AuthCredential::new(AuthCredentialTypes::ApiKey);
        credential.api_key = Some("secret".to_string());
        let scheme = AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "bearer".to_string(),
            bearer_format: None,
        })));
        assert_eq!(build_auth_headers(Some(&credential), Some(&scheme)), None);
    }
}
