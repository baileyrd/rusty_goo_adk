//! Capabilities C0533-C0535: `auth.oauth2_discovery`, ported from
//! `google.adk.auth.oauth2_discovery`.
//!
//! [`AuthorizationServerMetadata`] (RFC8414) and
//! [`ProtectedResourceMetadata`] (RFC9728) are the two response shapes;
//! [`OAuth2DiscoveryManager`] does the actual `.well-known` candidate-path
//! traversal and issuer/resource mix-up-attack validation (C0534/C0535).
//! `reqwest::blocking` (already adopted at the workspace level —
//! `gemini.rs`/`load_web_page.rs`) does the fetching, wrapped in
//! `rusty_tokio::spawn_blocking` for the async public entrypoints — the
//! same bridging pattern `load_web_page.rs` already established.
//!
//! **Wire shape, not camelCase**: unlike this crate's Google-genai-facing
//! types (`Part`/`FunctionResponse`/etc., which use `alias_generator=to_camel`
//! to match Google's own API convention), these two models mirror RFC8414/
//! RFC9728's actual JSON field names verbatim — both RFCs specify
//! snake_case field names (`authorization_endpoint`, `scopes_supported`,
//! ...) — so no `rename_all` is applied here.
//!
//! **`@experimental`, same precedent as `ResumabilityConfig`**: the
//! source decorates both classes with `@experimental`, which wraps
//! `__init__` to warn on every *construction* (confirmed by reading
//! `_create_decorator`'s class-decorating branch) — not on class
//! definition, and not on deserialization specifically (pydantic's
//! `model_validate` also calls `__init__`, so the source warns on parse
//! too; this port's derive-based `Deserialize` bypasses any constructor
//! function, same already-accepted narrowing `ResumabilityConfig::new`
//! established: the explicit constructor warns, deserialization doesn't).

use rusty_serde::{Deserialize, Serialize};

use adk_features::legacy_feature_decorator::warn_experimental;

/// `auth.oauth2_discovery.AuthorizationServerMetadata` — the OAuth2
/// authorization server metadata document per RFC8414.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[rusty_serde(default)]
    pub scopes_supported: Option<Vec<String>>,
    #[rusty_serde(default)]
    pub registration_endpoint: Option<String>,
}

impl AuthorizationServerMetadata {
    /// Constructs the metadata, emitting the source's `@experimental`
    /// warning — see the module doc.
    pub fn new(
        issuer: impl Into<String>,
        authorization_endpoint: impl Into<String>,
        token_endpoint: impl Into<String>,
        scopes_supported: Option<Vec<String>>,
        registration_endpoint: Option<String>,
    ) -> Self {
        warn_experimental("AuthorizationServerMetadata", None);
        Self {
            issuer: issuer.into(),
            authorization_endpoint: authorization_endpoint.into(),
            token_endpoint: token_endpoint.into(),
            scopes_supported,
            registration_endpoint,
        }
    }
}

/// `auth.oauth2_discovery.ProtectedResourceMetadata` — the OAuth2
/// protected resource metadata document per RFC9728.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtectedResourceMetadata {
    pub resource: String,
    #[rusty_serde(default)]
    pub authorization_servers: Vec<String>,
}

impl ProtectedResourceMetadata {
    /// Constructs the metadata, emitting the source's `@experimental`
    /// warning — see the module doc.
    pub fn new(resource: impl Into<String>, authorization_servers: Vec<String>) -> Self {
        warn_experimental("ProtectedResourceMetadata", None);
        Self {
            resource: resource.into(),
            authorization_servers,
        }
    }
}

/// `auth.oauth2_discovery.OAuth2DiscoveryManager` — implements metadata
/// discovery for OAuth2 following RFC8414 and RFC9728.
#[derive(Debug, Clone, Copy)]
pub struct OAuth2DiscoveryManager;

impl Default for OAuth2DiscoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl OAuth2DiscoveryManager {
    pub fn new() -> Self {
        warn_experimental("OAuth2DiscoveryManager", None);
        Self
    }

    /// `discover_auth_server_metadata`: tries the standard `.well-known`
    /// candidate endpoints for `issuer_url`, in order, returning the
    /// first one whose `issuer` matches `issuer_url` (trailing `/`
    /// stripped) — the explicit defense against OAuth "IdP mix-up"
    /// attacks. Swallows per-candidate HTTP/parse errors and tries the
    /// next; `None` if every candidate fails, errors, or mismatches.
    pub async fn discover_auth_server_metadata(
        &self,
        issuer_url: &str,
    ) -> Option<AuthorizationServerMetadata> {
        let issuer_url = issuer_url.to_string();
        rusty_tokio::spawn_blocking(move || discover_auth_server_metadata_blocking(&issuer_url))
            .await
            .ok()
            .flatten()
    }

    /// `discover_resource_metadata`: same mix-up-defense pattern as
    /// [`Self::discover_auth_server_metadata`], for the single protected-
    /// resource `.well-known` candidate.
    pub async fn discover_resource_metadata(
        &self,
        resource_url: &str,
    ) -> Option<ProtectedResourceMetadata> {
        let resource_url = resource_url.to_string();
        rusty_tokio::spawn_blocking(move || discover_resource_metadata_blocking(&resource_url))
            .await
            .ok()
            .flatten()
    }
}

/// The `.well-known` candidate endpoints for an authorization-server
/// issuer URL, in the source's exact priority order: path-inserted
/// OAuth2 metadata, path-inserted OpenID Connect discovery, then
/// path-appended OpenID Connect discovery — or, when the issuer URL
/// carries no path, the same first two without path handling at all.
fn auth_server_metadata_candidates(base_url: &str, path: &str) -> Vec<String> {
    if path != "/" {
        vec![
            format!("{base_url}/.well-known/oauth-authorization-server{path}"),
            format!("{base_url}/.well-known/openid-configuration{path}"),
            format!("{base_url}{path}/.well-known/openid-configuration"),
        ]
    } else {
        vec![
            format!("{base_url}/.well-known/oauth-authorization-server"),
            format!("{base_url}/.well-known/openid-configuration"),
        ]
    }
}

fn well_known_client() -> Option<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()
}

fn discover_auth_server_metadata_blocking(issuer_url: &str) -> Option<AuthorizationServerMetadata> {
    let parsed = reqwest::Url::parse(issuer_url).ok()?;
    let base_url = parsed.origin().ascii_serialization();
    let expected_issuer = issuer_url.trim_end_matches('/');
    let client = well_known_client()?;

    for endpoint in auth_server_metadata_candidates(&base_url, parsed.path()) {
        let Ok(response) = client.get(&endpoint).send() else {
            continue;
        };
        let Ok(response) = response.error_for_status() else {
            continue;
        };
        let Ok(text) = response.text() else {
            continue;
        };
        let Ok(metadata) = rusty_serde::json::from_str::<AuthorizationServerMetadata>(&text) else {
            continue;
        };
        if metadata.issuer == expected_issuer {
            return Some(metadata);
        }
    }
    None
}

fn discover_resource_metadata_blocking(resource_url: &str) -> Option<ProtectedResourceMetadata> {
    let parsed = reqwest::Url::parse(resource_url).ok()?;
    let base_url = parsed.origin().ascii_serialization();
    let path = parsed.path();
    let endpoint = if path != "/" {
        format!("{base_url}/.well-known/oauth-protected-resource{path}")
    } else {
        format!("{base_url}/.well-known/oauth-protected-resource")
    };
    let expected_resource = resource_url.trim_end_matches('/');
    let client = well_known_client()?;

    let response = client.get(&endpoint).send().ok()?;
    let response = response.error_for_status().ok()?;
    let text = response.text().ok()?;
    let metadata = rusty_serde::json::from_str::<ProtectedResourceMetadata>(&text).ok()?;
    if metadata.resource == expected_resource {
        Some(metadata)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_server_metadata_round_trips_through_json() {
        let metadata = AuthorizationServerMetadata::new(
            "https://issuer.example.com",
            "https://issuer.example.com/authorize",
            "https://issuer.example.com/token",
            Some(vec!["openid".to_string(), "email".to_string()]),
            Some("https://issuer.example.com/register".to_string()),
        );
        let json = rusty_serde::json::to_string(&metadata).unwrap();
        let round_tripped: AuthorizationServerMetadata =
            rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(round_tripped, metadata);
        assert!(json.contains("authorization_endpoint"));
        assert!(!json.contains("authorizationEndpoint"));
    }

    #[test]
    fn authorization_server_metadata_defaults_optional_fields_when_absent() {
        let json = r#"{"issuer":"i","authorization_endpoint":"a","token_endpoint":"t"}"#;
        let metadata: AuthorizationServerMetadata = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(metadata.scopes_supported, None);
        assert_eq!(metadata.registration_endpoint, None);
    }

    #[test]
    fn protected_resource_metadata_round_trips_through_json() {
        let metadata = ProtectedResourceMetadata::new(
            "https://resource.example.com",
            vec!["https://issuer.example.com".to_string()],
        );
        let json = rusty_serde::json::to_string(&metadata).unwrap();
        let round_tripped: ProtectedResourceMetadata = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(round_tripped, metadata);
    }

    #[test]
    fn protected_resource_metadata_defaults_authorization_servers_to_an_empty_list() {
        let json = r#"{"resource":"r"}"#;
        let metadata: ProtectedResourceMetadata = rusty_serde::json::from_str(json).unwrap();
        assert_eq!(metadata.authorization_servers, Vec::<String>::new());
    }

    /// A tiny multi-connection mock HTTP server: `routes_fn` is handed the
    /// server's own base URL (so a test can build response bodies that
    /// self-reference the mock host) and returns a path -> (status, body)
    /// table; unmatched paths get a 404. Serves `accept_count` connections
    /// in sequence, one per accepted `TcpStream` — unlike `gemini.rs`'s
    /// `spawn_one_shot_server` (single connection only), the candidate-
    /// fallback logic here needs a server that can answer several distinct
    /// requests against the same mock host within one test.
    fn spawn_mock_server(
        accept_count: usize,
        routes_fn: impl FnOnce(&str) -> std::collections::HashMap<String, (u16, String)>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        let routes = routes_fn(&base_url);
        let handle = std::thread::spawn(move || {
            use std::io::{Read, Write};
            for _ in 0..accept_count {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut received = Vec::new();
                let mut buf = [0u8; 4096];
                let headers = loop {
                    let n = stream.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break String::new();
                    }
                    received.extend_from_slice(&buf[..n]);
                    if let Some(header_end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                        break String::from_utf8_lossy(&received[..header_end]).to_string();
                    }
                };
                let path = headers
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/")
                    .to_string();
                let (status, body) = routes
                    .get(&path)
                    .cloned()
                    .unwrap_or((404, "not found".to_string()));
                let status_line = if status == 200 {
                    "HTTP/1.1 200 OK"
                } else {
                    "HTTP/1.1 404 Not Found"
                };
                let response = format!(
                    "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        (base_url, handle)
    }

    #[rusty_tokio::test]
    async fn discover_auth_server_metadata_succeeds_on_the_first_candidate() {
        let (base_url, server) = spawn_mock_server(1, |base_url| {
            let mut routes = std::collections::HashMap::new();
            routes.insert(
                "/.well-known/oauth-authorization-server".to_string(),
                (
                    200,
                    format!(
                        r#"{{"issuer":"{base_url}","authorization_endpoint":"{base_url}/authorize","token_endpoint":"{base_url}/token"}}"#
                    ),
                ),
            );
            routes
        });

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_auth_server_metadata(&base_url).await;

        server.join().unwrap();
        let metadata = metadata.expect("expected metadata from the first candidate");
        assert_eq!(metadata.issuer, base_url);
    }

    #[rusty_tokio::test]
    async fn discover_auth_server_metadata_falls_through_to_the_second_candidate() {
        let (base_url, server) = spawn_mock_server(2, |base_url| {
            let mut routes = std::collections::HashMap::new();
            // Deliberately no route for the first candidate
            // (`/.well-known/oauth-authorization-server`) — it 404s, and
            // discovery must fall through to the second.
            routes.insert(
                "/.well-known/openid-configuration".to_string(),
                (
                    200,
                    format!(
                        r#"{{"issuer":"{base_url}","authorization_endpoint":"{base_url}/authorize","token_endpoint":"{base_url}/token"}}"#
                    ),
                ),
            );
            routes
        });

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_auth_server_metadata(&base_url).await;

        server.join().unwrap();
        let metadata = metadata.expect("expected metadata from the fallback candidate");
        assert_eq!(metadata.issuer, base_url);
    }

    #[rusty_tokio::test]
    async fn discover_auth_server_metadata_returns_none_when_every_candidate_fails() {
        let (base_url, server) = spawn_mock_server(2, |_| std::collections::HashMap::new());

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_auth_server_metadata(&base_url).await;

        server.join().unwrap();
        assert!(metadata.is_none());
    }

    #[rusty_tokio::test]
    async fn discover_auth_server_metadata_with_a_path_succeeds_via_the_third_candidate() {
        let (base_url, server) = spawn_mock_server(3, |base_url| {
            let mut routes = std::collections::HashMap::new();
            let issuer_url = format!("{base_url}/tenant1");
            // Only the third (path-appended) candidate is wired up — the
            // first two (path-inserted) 404, proving the exact 3-candidate
            // priority order the source constructs for a non-root issuer.
            routes.insert(
                "/tenant1/.well-known/openid-configuration".to_string(),
                (
                    200,
                    format!(
                        r#"{{"issuer":"{issuer_url}","authorization_endpoint":"{issuer_url}/authorize","token_endpoint":"{issuer_url}/token"}}"#
                    ),
                ),
            );
            routes
        });
        let issuer_url = format!("{base_url}/tenant1");

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_auth_server_metadata(&issuer_url).await;

        server.join().unwrap();
        let metadata = metadata.expect("expected metadata from the path-appended candidate");
        assert_eq!(metadata.issuer, issuer_url);
    }

    #[rusty_tokio::test]
    async fn discover_auth_server_metadata_returns_none_for_a_malformed_issuer_url() {
        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_auth_server_metadata("not a url").await;
        assert!(metadata.is_none());
    }

    #[rusty_tokio::test]
    async fn discover_resource_metadata_succeeds_on_its_candidate() {
        let (base_url, server) = spawn_mock_server(1, |base_url| {
            let mut routes = std::collections::HashMap::new();
            routes.insert(
                "/.well-known/oauth-protected-resource".to_string(),
                (
                    200,
                    format!(
                        r#"{{"resource":"{base_url}","authorization_servers":["{base_url}"]}}"#
                    ),
                ),
            );
            routes
        });

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_resource_metadata(&base_url).await;

        server.join().unwrap();
        let metadata = metadata.expect("expected resource metadata");
        assert_eq!(metadata.resource, base_url);
    }

    #[rusty_tokio::test]
    async fn discover_resource_metadata_rejects_a_mismatched_resource_field() {
        let (base_url, server) = spawn_mock_server(1, |_| {
            let mut routes = std::collections::HashMap::new();
            routes.insert(
                "/.well-known/oauth-protected-resource".to_string(),
                (
                    200,
                    r#"{"resource":"https://someone-else.example.com"}"#.to_string(),
                ),
            );
            routes
        });

        let manager = OAuth2DiscoveryManager::new();
        let metadata = manager.discover_resource_metadata(&base_url).await;

        server.join().unwrap();
        assert!(metadata.is_none());
    }
}
