//! Capability C0504: `auth.auth_tool`, ported from `google.adk.auth.auth_tool`.
//!
//! **`_stable_model_digest`, adapted**: the source dumps a pydantic model
//! (`by_alias=True, exclude_none=True, mode="json"`), canonicalizes with
//! `json.dumps(..., sort_keys=True, separators=(",", ":"))`, then SHA-256
//! hashes and truncates to 16 hex chars. This port's [`stable_digest`]
//! does the equivalent over a `rusty_serde::value::Value` tree — dump via
//! `rusty_serde::json::to_value` (already alias-respecting, since this
//! port's structs declare their wire names directly via
//! `#[rusty_serde(rename...)]`), then [`canonicalize`] recursively drops
//! `Value::Null` map entries (`exclude_none`) and sorts each map's
//! entries by key (`sort_keys`), then serializes compactly (this port's
//! `to_string` is already compact, no `indent=2` — same established
//! default as `local_eval_sets_manager`, C0613) and hashes. **Not
//! byte-identical to the source's digest** — different languages'
//! JSON serializers escape/format differently (this port doesn't try to
//! replicate `ensure_ascii=False` byte-for-byte) — but the actual
//! behavioral contract (`_stable_model_digest`'s own docstring: stable
//! across hash seeds, dict-ordering differences, and `model_extra`
//! values) holds identically here, which is what `credential_key`
//! actually needs: a deterministic, collision-resistant function of the
//! model's declared fields.
//!
//! **`AuthConfig.__init__`'s `model_extra` scan, narrowed**: the source
//! checks both `raw_auth_credential.model_extra` and
//! `auth_scheme.model_extra` for an explicit `credential_key`/
//! `credentialKey` override. This port's `AuthCredential` (C0494-499,
//! already shipped) has no `extra` field at all — that batch's own doc
//! discloses `extra="allow"` as "dropped, not preserved" for every
//! credential struct — so only `auth_scheme` (specifically its
//! [`crate::auth_schemes::AuthScheme::Custom`] variant, the one branch
//! that does carry `extra`, see that module's doc) can supply this
//! override here.

use sha2::{Digest, Sha256};

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

use crate::auth_credential::AuthCredential;
use crate::auth_schemes::AuthScheme;

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Map(entries) => {
            let mut filtered: Vec<(String, Value)> = entries
                .into_iter()
                .filter(|(_, v)| !matches!(v, Value::Null))
                .map(|(k, v)| (k, canonicalize(v)))
                .collect();
            filtered.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Map(filtered)
        }
        Value::Seq(items) => Value::Seq(items.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

/// C0504: `_stable_model_digest` — see the module doc.
pub fn stable_digest<T: Serialize>(value: &T) -> String {
    let dumped = rusty_serde::json::to_value(value).expect("value tree always serializes");
    let canonical = canonicalize(dumped);
    let canonical_json =
        rusty_serde::json::to_string(&canonical).expect("canonicalized value always serializes");
    let digest = Sha256::digest(canonical_json.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    hex[..16].to_string()
}

/// `auth.auth_tool.AuthConfig` — the auth config sent by a tool asking
/// the client to collect auth credentials; ADK and the client fill in
/// the response together.
///
/// `Serialize`/`Deserialize` added in the credential-service batch
/// (C0527-C0529) — `Context::request_credential` needs to store this in
/// `EventActions.requested_auth_configs` (`adk-events`, `Value`-typed and
/// out of scope to widen here) as an opaque `Value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthConfig {
    /// The auth scheme used to collect credentials.
    pub auth_scheme: AuthScheme,
    /// The raw auth credential used to collect credentials — used by
    /// schemes that need to exchange credentials (OAuth2/OIDC); `None`
    /// otherwise.
    pub raw_auth_credential: Option<AuthCredential>,
    /// The exchanged auth credential; ADK and the client fill this in
    /// together as the OAuth2/OIDC flow progresses.
    pub exchanged_auth_credential: Option<AuthCredential>,
    /// A user-specified key used to load/save this credential in a
    /// credential service.
    pub credential_key: Option<String>,
}

impl AuthConfig {
    /// C0504: the source's `__init__` — auto-derives `credential_key`
    /// when not explicitly given, first checking `auth_scheme`'s `extra`
    /// for an override (see the module doc), then falling back to
    /// [`AuthConfig::get_credential_key`].
    pub fn new(
        auth_scheme: AuthScheme,
        raw_auth_credential: Option<AuthCredential>,
        exchanged_auth_credential: Option<AuthCredential>,
        credential_key: Option<String>,
    ) -> Self {
        let mut config = AuthConfig {
            auth_scheme,
            raw_auth_credential,
            exchanged_auth_credential,
            credential_key: None,
        };

        if let Some(key) = credential_key.filter(|k| !k.is_empty()) {
            config.credential_key = Some(key);
            return config;
        }

        if let Some(extra) = config.auth_scheme.extra() {
            for key in ["credential_key", "credentialKey"] {
                if let Some(Value::String(value)) = extra.get(key) {
                    if !value.is_empty() {
                        config.credential_key = Some(value.clone());
                        return config;
                    }
                }
            }
        }

        config.credential_key = Some(config.get_credential_key());
        config
    }

    /// Builds a stable key based on `auth_scheme` and
    /// `raw_auth_credential`, used to save/load credentials to/from a
    /// credential service when `credential_key` isn't explicitly
    /// provided. Deprecated in the source in favor of `credential_key`
    /// directly — ported anyway since [`AuthConfig::new`] still calls it
    /// internally.
    pub fn get_credential_key(&self) -> String {
        let digestable_scheme = self.auth_scheme.without_extra();
        let scheme_name = format!(
            "{}_{}",
            self.auth_scheme.type_name(),
            stable_digest(&digestable_scheme)
        );

        let credential_name = match &self.raw_auth_credential {
            Some(credential) => {
                let mut sanitized = credential.clone();
                if let Some(oauth2) = &mut sanitized.oauth2 {
                    oauth2.auth_uri = None;
                    oauth2.state = None;
                    oauth2.auth_response_uri = None;
                    oauth2.auth_code = None;
                    oauth2.access_token = None;
                    oauth2.refresh_token = None;
                    oauth2.expires_at = None;
                    oauth2.expires_in = None;
                    oauth2.redirect_uri = None;
                }
                format!(
                    "{}_{}",
                    credential.auth_type.as_str(),
                    stable_digest(&sanitized)
                )
            }
            None => String::new(),
        };

        format!("adk_{scheme_name}_{credential_name}")
    }
}

/// `auth.auth_tool.AuthToolArguments` — arguments for the special
/// long-running function tool used to request end-user credentials.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthToolArguments {
    pub function_call_id: String,
    pub auth_config: AuthConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_credential::{AuthCredentialTypes, OAuth2Auth};
    use crate::auth_schemes::{CustomAuthScheme, HttpScheme, SecurityScheme};

    fn http_scheme() -> AuthScheme {
        AuthScheme::Security(Box::new(SecurityScheme::Http(HttpScheme {
            description: None,
            scheme: "bearer".to_string(),
            bearer_format: None,
        })))
    }

    #[test]
    fn stable_digest_is_deterministic_across_calls() {
        let scheme = http_scheme();
        assert_eq!(stable_digest(&scheme), stable_digest(&scheme));
    }

    #[test]
    fn stable_digest_ignores_map_key_insertion_order() {
        let a = Value::Map(vec![
            ("a".to_string(), Value::Bool(true)),
            ("b".to_string(), Value::Bool(false)),
        ]);
        let b = Value::Map(vec![
            ("b".to_string(), Value::Bool(false)),
            ("a".to_string(), Value::Bool(true)),
        ]);
        assert_eq!(stable_digest(&a), stable_digest(&b));
    }

    #[test]
    fn stable_digest_ignores_null_valued_entries() {
        let with_null = Value::Map(vec![
            ("a".to_string(), Value::Bool(true)),
            ("b".to_string(), Value::Null),
        ]);
        let without_null = Value::Map(vec![("a".to_string(), Value::Bool(true))]);
        assert_eq!(stable_digest(&with_null), stable_digest(&without_null));
    }

    #[test]
    fn stable_digest_differs_for_different_content() {
        let a = Value::Map(vec![("a".to_string(), Value::Bool(true))]);
        let b = Value::Map(vec![("a".to_string(), Value::Bool(false))]);
        assert_ne!(stable_digest(&a), stable_digest(&b));
    }

    #[test]
    fn auth_config_new_uses_the_given_credential_key_when_present() {
        let config = AuthConfig::new(http_scheme(), None, None, Some("explicit-key".to_string()));
        assert_eq!(config.credential_key.as_deref(), Some("explicit-key"));
    }

    #[test]
    fn auth_config_new_reads_credential_key_from_a_custom_schemes_extra() {
        let scheme = AuthScheme::Custom(CustomAuthScheme {
            type_: "my_scheme".to_string(),
            extra: Some(Value::Map(vec![(
                "credential_key".to_string(),
                Value::String("from-extra".to_string()),
            )])),
        });
        let config = AuthConfig::new(scheme, None, None, None);
        assert_eq!(config.credential_key.as_deref(), Some("from-extra"));
    }

    #[test]
    fn auth_config_new_reads_credential_key_from_the_camel_case_extra_key() {
        let scheme = AuthScheme::Custom(CustomAuthScheme {
            type_: "my_scheme".to_string(),
            extra: Some(Value::Map(vec![(
                "credentialKey".to_string(),
                Value::String("from-camel-extra".to_string()),
            )])),
        });
        let config = AuthConfig::new(scheme, None, None, None);
        assert_eq!(config.credential_key.as_deref(), Some("from-camel-extra"));
    }

    #[test]
    fn auth_config_new_synthesizes_a_credential_key_absent_any_override() {
        let config = AuthConfig::new(http_scheme(), None, None, None);
        let key = config.credential_key.expect("expected a synthesized key");
        assert!(key.starts_with("adk_http_"));
    }

    #[test]
    fn get_credential_key_is_stable_across_dynamic_oauth2_fields() {
        let scheme = http_scheme();
        let mut credential = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential.oauth2 = Some(OAuth2Auth {
            client_id: Some("client".to_string()),
            access_token: Some("token-a".to_string()),
            ..OAuth2Auth::default()
        });
        let config_a = AuthConfig::new(scheme.clone(), Some(credential.clone()), None, None);

        let mut credential_b = credential;
        credential_b.oauth2.as_mut().unwrap().access_token = Some("token-b".to_string());
        let config_b = AuthConfig::new(scheme, Some(credential_b), None, None);

        assert_eq!(config_a.credential_key, config_b.credential_key);
    }

    #[test]
    fn get_credential_key_differs_for_different_client_ids() {
        let scheme = http_scheme();
        let mut credential_a = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential_a.oauth2 = Some(OAuth2Auth {
            client_id: Some("client-a".to_string()),
            ..OAuth2Auth::default()
        });
        let mut credential_b = AuthCredential::new(AuthCredentialTypes::OAuth2);
        credential_b.oauth2 = Some(OAuth2Auth {
            client_id: Some("client-b".to_string()),
            ..OAuth2Auth::default()
        });

        let config_a = AuthConfig::new(scheme.clone(), Some(credential_a), None, None);
        let config_b = AuthConfig::new(scheme, Some(credential_b), None, None);

        assert_ne!(config_a.credential_key, config_b.credential_key);
    }
}
