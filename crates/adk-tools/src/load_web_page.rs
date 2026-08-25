//! Capability C0427: `load_web_page`, ported from
//! `google.adk.tools.load_web_page`.
//!
//! **The SSRF-hardening core is ported in full**: URL scheme/host/port
//! validation, `localhost`/`*.localhost` rejection, DNS resolution with
//! every resolved address vetted before any connection is attempted,
//! IP-literal vetting, IPv4/IPv6 global-reachability classification
//! (transcribed field-for-field from CPython 3.11's `ipaddress` module —
//! see the `ip_classification` submodule doc for the exact source
//! consulted and why), the embedded-IPv4-in-IPv6 checks (mapped/6to4/
//! NAT64/deprecated-compatible forms), IP-pinned connection with the
//! original Host header preserved (via `reqwest`'s `ClientBuilder::resolve`,
//! not a hand-rolled connection adapter), and disabled redirects.
//!
//! **Disclosed narrowings**:
//! - **No proxy-aware branching.** The source detects an environment-
//!   configured proxy for the target URL and, when present, skips
//!   IP-pinning entirely (trusting the proxy to resolve, only vetting
//!   literal IPs/localhost locally) so egress can go through a corporate
//!   proxy. This port always does the direct, IP-pinned fetch — it never
//!   reads `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`. This is strictly more
//!   restrictive, not a safety regression (a proxied fetch bypasses this
//!   port's own pinning, so simply not supporting proxies at all avoids
//!   reintroducing the DNS-rebinding window pinning defends against), but
//!   is a real behavior gap: an environment that relies on an egress
//!   proxy won't have its traffic routed through it here.
//! - **HTML text extraction has no HTML5 parser behind it** — no
//!   `BeautifulSoup`/`lxml` equivalent is a workspace dependency. A
//!   regex-based extractor (already-adopted `regex` crate) strips
//!   `<script>`/`<style>` element contents, strips all remaining tags,
//!   and decodes a handful of common named entities
//!   (`&amp;`/`&lt;`/`&gt;`/`&quot;`/`&apos;`/`&#39;`/`&nbsp;`) — not the
//!   full HTML5 entity table, and with no DOM-aware whitespace
//!   collapsing. The same short/long-line filter as the source
//!   (`len(line.split()) > 3`) is applied afterward.
//! - **The live-fetch path (`fetch_response`, the real `reqwest` GET) has
//!   no automated test in this environment** — it depends on outbound
//!   network/DNS this sandboxed test run may not have, and the tool's own
//!   safety design correctly rejects `127.0.0.1`/`localhost` before any
//!   request is attempted, so a local mock server can't stand in without
//!   weakening the real check being tested. Every other piece (URL
//!   parsing/validation, hostname blocking, the full IP-classification
//!   battery, HTML extraction) is covered.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use regex::Regex;
use rusty_serde::value::Value;

use crate::base_tool::{BaseTool, BoxFuture, ToolError};
use crate::tool_context::ToolContext;

const DEFAULT_TIMEOUT_SECONDS: u64 = 30;

/// IPv4/IPv6 global-reachability classification, transcribed
/// field-for-field from CPython 3.11's `ipaddress.py`
/// (`_IPv4Constants`/`_IPv6Constants`, consulted locally in this
/// environment at `/usr/lib/python3.11/ipaddress.py`) — this is the exact
/// reference the source's own `is_global` calls resolve to, so matching
/// it here is matching the source's actual behavior, not an
/// approximation of it.
pub(crate) mod ip_classification {
    use super::*;

    const fn v4(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr::new(a, b, c, d)
    }

    #[allow(clippy::too_many_arguments)]
    const fn v6(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Ipv6Addr {
        Ipv6Addr::new(a, b, c, d, e, f, g, h)
    }

    fn ipv4_in(addr: Ipv4Addr, network: Ipv4Addr, prefix: u8) -> bool {
        let mask: u32 = if prefix == 0 {
            0
        } else {
            !0u32 << (32 - prefix)
        };
        (u32::from(addr) & mask) == (u32::from(network) & mask)
    }

    fn ipv6_in(addr: Ipv6Addr, network: Ipv6Addr, prefix: u8) -> bool {
        let mask: u128 = if prefix == 0 {
            0
        } else {
            !0u128 << (128 - prefix)
        };
        (u128::from(addr) & mask) == (u128::from(network) & mask)
    }

    // https://www.iana.org/assignments/iana-ipv4-special-registry
    const V4_PRIVATE_NETWORKS: &[(Ipv4Addr, u8)] = &[
        (v4(0, 0, 0, 0), 8),
        (v4(10, 0, 0, 0), 8),
        (v4(127, 0, 0, 0), 8),
        (v4(169, 254, 0, 0), 16),
        (v4(172, 16, 0, 0), 12),
        (v4(192, 0, 0, 0), 24),
        (v4(192, 0, 0, 170), 31),
        (v4(192, 0, 2, 0), 24),
        (v4(192, 168, 0, 0), 16),
        (v4(198, 18, 0, 0), 15),
        (v4(198, 51, 100, 0), 24),
        (v4(203, 0, 113, 0), 24),
        (v4(240, 0, 0, 0), 4),
        (v4(255, 255, 255, 255), 32),
    ];
    const V4_PRIVATE_EXCEPTIONS: &[(Ipv4Addr, u8)] =
        &[(v4(192, 0, 0, 9), 32), (v4(192, 0, 0, 10), 32)];
    const V4_PUBLIC_NETWORK: (Ipv4Addr, u8) = (v4(100, 64, 0, 0), 10);

    pub fn ipv4_is_private(addr: Ipv4Addr) -> bool {
        V4_PRIVATE_NETWORKS
            .iter()
            .any(|&(net, prefix)| ipv4_in(addr, net, prefix))
            && V4_PRIVATE_EXCEPTIONS
                .iter()
                .all(|&(net, prefix)| !ipv4_in(addr, net, prefix))
    }

    pub fn ipv4_is_global(addr: Ipv4Addr) -> bool {
        !ipv4_in(addr, V4_PUBLIC_NETWORK.0, V4_PUBLIC_NETWORK.1) && !ipv4_is_private(addr)
    }

    // https://www.iana.org/assignments/iana-ipv6-special-registry
    const V6_PRIVATE_NETWORKS: &[(Ipv6Addr, u8)] = &[
        (v6(0, 0, 0, 0, 0, 0, 0, 1), 128),
        (v6(0, 0, 0, 0, 0, 0, 0, 0), 128),
        (v6(0, 0, 0, 0, 0, 0xffff, 0, 0), 96),
        (v6(0x64, 0xff9b, 1, 0, 0, 0, 0, 0), 48),
        (v6(0x100, 0, 0, 0, 0, 0, 0, 0), 64),
        (v6(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
        (v6(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32),
        (v6(0x2002, 0, 0, 0, 0, 0, 0, 0), 16),
        (v6(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
        (v6(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
        (v6(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
    ];
    const V6_PRIVATE_EXCEPTIONS: &[(Ipv6Addr, u8)] = &[
        (v6(0x2001, 1, 0, 0, 0, 0, 0, 1), 128),
        (v6(0x2001, 1, 0, 0, 0, 0, 0, 2), 128),
        (v6(0x2001, 3, 0, 0, 0, 0, 0, 0), 32),
        (v6(0x2001, 4, 0x112, 0, 0, 0, 0, 0), 48),
        (v6(0x2001, 0x20, 0, 0, 0, 0, 0, 0), 28),
        (v6(0x2001, 0x30, 0, 0, 0, 0, 0, 0), 28),
    ];

    fn ipv6_is_private_raw(addr: Ipv6Addr) -> bool {
        V6_PRIVATE_NETWORKS
            .iter()
            .any(|&(net, prefix)| ipv6_in(addr, net, prefix))
            && V6_PRIVATE_EXCEPTIONS
                .iter()
                .all(|&(net, prefix)| !ipv6_in(addr, net, prefix))
    }

    /// Matches `IPv6Address.is_private`: delegates to the embedded IPv4's
    /// own classification only for the true `::ffff:a.b.c.d` mapped form
    /// (`to_ipv4_mapped`) — 6to4/NAT64/compatible forms are NOT delegated
    /// here, matching the source exactly (see [`embedded_ipv4`] and the
    /// module doc for why the source layers a *second*, tool-specific
    /// check on top of this for those forms). Exposed for API symmetry
    /// with the source's own public `is_private` property, and exercised
    /// directly in tests; `is_blocked_address` itself only needs
    /// `is_global`.
    #[allow(dead_code)]
    pub fn ipv6_is_private(addr: Ipv6Addr) -> bool {
        match addr.to_ipv4_mapped() {
            Some(mapped) => ipv4_is_private(mapped),
            None => ipv6_is_private_raw(addr),
        }
    }

    pub fn ipv6_is_global(addr: Ipv6Addr) -> bool {
        match addr.to_ipv4_mapped() {
            Some(mapped) => ipv4_is_global(mapped),
            None => !ipv6_is_private_raw(addr),
        }
    }

    pub fn is_global(addr: IpAddr) -> bool {
        match addr {
            IpAddr::V4(a) => ipv4_is_global(a),
            IpAddr::V6(a) => ipv6_is_global(a),
        }
    }

    const NAT64_PREFIX: Ipv6Addr = v6(0x0064, 0xff9b, 0, 0, 0, 0, 0, 0);

    /// Returns the IPv4 address embedded in an IPv6 address, if any —
    /// `is_global`/`is_private` above do not reflect the reachability of
    /// the embedded IPv4 target for IPv4-mapped, IPv4-compatible, 6to4,
    /// and NAT64 forms. See the module doc: this is a real gap in the
    /// plain `is_global` check the source's own `_embedded_ipv4` helper
    /// exists to close.
    pub fn embedded_ipv4(addr: Ipv6Addr) -> Option<Ipv4Addr> {
        if let Some(mapped) = addr.to_ipv4_mapped() {
            return Some(mapped);
        }
        let segments = addr.segments();
        // 6to4: 2002:WWXX:YYZZ::/48 embeds WW.XX.YY.ZZ.
        if segments[0] == 0x2002 {
            return Some(Ipv4Addr::new(
                (segments[1] >> 8) as u8,
                segments[1] as u8,
                (segments[2] >> 8) as u8,
                segments[2] as u8,
            ));
        }
        // NAT64: 64:ff9b::/96 embeds the low 32 bits.
        if ipv6_in(addr, NAT64_PREFIX, 96) {
            return Some(Ipv4Addr::new(
                (segments[6] >> 8) as u8,
                segments[6] as u8,
                (segments[7] >> 8) as u8,
                segments[7] as u8,
            ));
        }
        // Deprecated IPv4-compatible ::a.b.c.d: top 96 bits zero, low 32
        // bits a non-trivial IPv4 (excluding :: and ::1).
        let packed = u128::from(addr);
        if packed >> 32 == 0 {
            let low32 = (packed & 0xFFFF_FFFF) as u32;
            if low32 != 0 && low32 != 1 {
                return Some(Ipv4Addr::from(low32));
            }
        }
        None
    }

    /// `_is_blocked_address`: `true` if this address must not be
    /// connected to.
    pub fn is_blocked_address(addr: IpAddr) -> bool {
        if !is_global(addr) {
            return true;
        }
        if let IpAddr::V6(v6_addr) = addr {
            if let Some(embedded) = embedded_ipv4(v6_addr) {
                return !ipv4_is_global(embedded);
            }
        }
        false
    }
}

pub(crate) fn is_blocked_hostname(hostname: &str) -> bool {
    let normalized = hostname.trim_end_matches('.').to_lowercase();
    normalized == "localhost" || normalized.ends_with(".localhost")
}

fn failed_to_fetch_message(url: &str) -> String {
    format!("Failed to fetch url: {url}")
}

pub(crate) struct RequestTarget {
    pub(crate) url: reqwest::Url,
    pub(crate) host: url::Host,
}

impl RequestTarget {
    /// The hostname as a plain string, suitable for
    /// [`is_blocked_hostname`]/[`resolve_direct_addresses`] — an IP
    /// literal renders as its own string form (`to_socket_addrs`, and
    /// thus [`resolve_addresses`], accepts an IP-literal string directly,
    /// no separate short-circuit needed on this port's side).
    pub(crate) fn hostname(&self) -> String {
        match &self.host {
            url::Host::Domain(domain) => domain.clone(),
            url::Host::Ipv4(addr) => addr.to_string(),
            url::Host::Ipv6(addr) => addr.to_string(),
        }
    }
}

/// Formats what the Host header would read for `url` — used only by
/// tests: the actual request never sets this header manually (see the
/// module doc — `reqwest`'s `resolve()` keeps the URL pointed at the
/// original hostname, so it sends the correct Host header on its own).
#[cfg(test)]
fn host_header_for_test(url_str: &str) -> String {
    let parsed = reqwest::Url::parse(url_str).unwrap();
    let scheme = parsed.scheme();
    let default_port = if scheme == "http" { 80 } else { 443 };
    let host_str = parsed.host_str().unwrap().to_string();
    match parsed.port() {
        Some(port) if port != default_port => format!("{host_str}:{port}"),
        _ => host_str,
    }
}

pub(crate) fn parse_request_target(url_str: &str) -> Result<RequestTarget, String> {
    let parsed =
        reqwest::Url::parse(url_str).map_err(|_| format!("Unsupported url scheme: {url_str}"))?;
    let scheme = parsed.scheme().to_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(format!("Unsupported url scheme: {url_str}"));
    }
    let Some(host) = parsed.host() else {
        return Err(format!("URL is missing a hostname: {url_str}"));
    };
    let host = host.to_owned();
    Ok(RequestTarget { url: parsed, host })
}

pub(crate) fn resolve_addresses(hostname: &str) -> Result<Vec<IpAddr>, String> {
    use std::net::ToSocketAddrs;
    let addrs = (hostname, 0u16)
        .to_socket_addrs()
        .map_err(|_| format!("Unable to resolve host: {hostname}"))?;
    let mut resolved: Vec<IpAddr> = Vec::new();
    for addr in addrs {
        let ip = addr.ip();
        if !resolved.contains(&ip) {
            resolved.push(ip);
        }
    }
    if resolved.is_empty() {
        return Err(format!("Unable to resolve host: {hostname}"));
    }
    Ok(resolved)
}

/// `load_web_page._resolve_direct_addresses` — resolves `hostname` (an
/// IP literal short-circuits inside [`resolve_addresses`]'s own
/// `to_socket_addrs` call, the same way the source's
/// `_resolve_host_addresses` special-cases `_parse_ip_literal` first)
/// and rejects if any resolved address isn't globally reachable.
/// Reused by `computer_use_toolset.rs`'s `navigate` SSRF validator — see
/// that module's doc for why the source's separate raw-backslash-netloc
/// check isn't ported (verified structurally unreachable under this
/// port's WHATWG-spec-compliant URL parser).
pub(crate) fn resolve_direct_addresses(hostname: &str) -> Result<Vec<IpAddr>, String> {
    let addresses = resolve_addresses(hostname)?;
    if addresses
        .iter()
        .any(|addr| ip_classification::is_blocked_address(*addr))
    {
        return Err(format!("Blocked host: {hostname}"));
    }
    Ok(addresses)
}

fn fetch_response(url_str: &str) -> Result<(u16, Vec<u8>), String> {
    let target = parse_request_target(url_str)?;

    let hostname_for_blocklist = match &target.host {
        url::Host::Domain(domain) => domain.clone(),
        url::Host::Ipv4(addr) => addr.to_string(),
        url::Host::Ipv6(addr) => addr.to_string(),
    };
    if is_blocked_hostname(&hostname_for_blocklist) {
        return Err(format!("Blocked host: {hostname_for_blocklist}"));
    }

    let candidate_addresses: Vec<IpAddr> = match &target.host {
        url::Host::Ipv4(addr) => vec![IpAddr::V4(*addr)],
        url::Host::Ipv6(addr) => vec![IpAddr::V6(*addr)],
        url::Host::Domain(domain) => resolve_addresses(domain)?,
    };

    if candidate_addresses
        .iter()
        .any(|addr| ip_classification::is_blocked_address(*addr))
    {
        return Err(format!("Blocked host: {hostname_for_blocklist}"));
    }

    let connect_host = match &target.host {
        url::Host::Domain(domain) => domain.as_str(),
        _ => hostname_for_blocklist.as_str(),
    };

    let mut last_error: Option<String> = None;
    for address in &candidate_addresses {
        let socket_addr =
            SocketAddr::new(*address, target.url.port_or_known_default().unwrap_or(0));
        let client = match reqwest::blocking::Client::builder()
            .resolve(connect_host, socket_addr)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS))
            .build()
        {
            Ok(client) => client,
            Err(err) => {
                last_error = Some(err.to_string());
                continue;
            }
        };
        match client.get(target.url.clone()).send() {
            Ok(response) => {
                let status = response.status().as_u16();
                let body = response.bytes().map(|b| b.to_vec()).unwrap_or_default();
                return Ok((status, body));
            }
            Err(err) => last_error = Some(err.to_string()),
        }
    }

    Err(last_error.unwrap_or_else(|| format!("Unable to fetch url: {url_str}")))
}

fn strip_element_content(html: &str, tag: &str) -> String {
    let pattern = format!(
        r"(?is)<{tag}\b[^>]*>.*?</{tag}\s*>",
        tag = regex::escape(tag)
    );
    Regex::new(&pattern)
        .expect("valid regex")
        .replace_all(html, "")
        .into_owned()
}

fn decode_common_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// See the module doc for what this narrows relative to
/// `BeautifulSoup(...).get_text(separator='\n', strip=True)`.
fn extract_text(html: &str) -> String {
    let without_scripts = strip_element_content(html, "script");
    let without_style = strip_element_content(&without_scripts, "style");
    let tag_pattern = Regex::new(r"<[^>]*>").expect("valid regex");
    let untagged = tag_pattern.replace_all(&without_style, "\n");
    let decoded = decode_common_entities(&untagged);

    decoded
        .lines()
        .map(str::trim)
        .filter(|line| line.split_whitespace().count() > 3)
        .collect::<Vec<_>>()
        .join("\n")
}

/// C0427: fetches the content at `url` and returns the text in it.
pub fn load_web_page(url: &str) -> String {
    match fetch_response(url) {
        Ok((200, body)) => extract_text(&String::from_utf8_lossy(&body)),
        Ok(_) => failed_to_fetch_message(url),
        Err(_) => failed_to_fetch_message(url),
    }
}

/// C0427: `FunctionTool`-compatible async wrapper — offloads the blocking
/// DNS/HTTP work to `rusty_tokio::spawn_blocking`, matching `gemini.rs`'s
/// own `reqwest::blocking` bridging pattern.
pub async fn load_web_page_async(url: String) -> String {
    rusty_tokio::spawn_blocking(move || load_web_page(&url))
        .await
        .unwrap_or_else(|_| "Failed to fetch url: task panicked".to_string())
}

pub struct LoadWebPageTool;

impl LoadWebPageTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoadWebPageTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseTool for LoadWebPageTool {
    fn name(&self) -> &str {
        "load_web_page"
    }

    fn description(&self) -> &str {
        "Fetches the content in the url and returns the text in it."
    }

    fn get_declaration(&self) -> Option<adk_genai::content::FunctionDeclaration> {
        Some(adk_genai::content::FunctionDeclaration {
            name: Some(self.name().to_string()),
            description: Some(self.description().to_string()),
            parameters: Some(Value::Map(vec![
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    Value::Map(vec![(
                        "url".to_string(),
                        Value::Map(vec![(
                            "type".to_string(),
                            Value::String("string".to_string()),
                        )]),
                    )]),
                ),
                (
                    "required".to_string(),
                    Value::Seq(vec![Value::String("url".to_string())]),
                ),
            ])),
            ..Default::default()
        })
    }

    fn run_async<'a>(
        &'a self,
        args: &'a std::collections::BTreeMap<String, Value>,
        _tool_context: &'a mut ToolContext,
    ) -> BoxFuture<'a, Result<Value, ToolError>> {
        Box::pin(async move {
            let url = match args.get("url") {
                Some(Value::String(url)) => url.clone(),
                _ => return Ok(Value::String("Missing required argument: url".to_string())),
            };
            Ok(Value::String(load_web_page_async(url).await))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ip_classification::*;
    use super::*;

    #[test]
    fn ipv4_global_reachability_matches_iana_registry() {
        let global = ["8.8.8.8", "1.1.1.1", "192.0.0.9", "192.0.0.10"];
        let blocked = [
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "127.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.100.100.100",
            "192.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "198.51.100.5",
            "203.0.113.5",
            "192.0.2.5",
            "198.18.0.1",
        ];
        for ip in global {
            let addr: Ipv4Addr = ip.parse().unwrap();
            assert!(ipv4_is_global(addr), "{ip} should be global");
        }
        for ip in blocked {
            let addr: Ipv4Addr = ip.parse().unwrap();
            assert!(!ipv4_is_global(addr), "{ip} should not be global");
        }
    }

    #[test]
    fn ipv6_global_reachability_matches_iana_registry() {
        let global = [
            "2001:4860:4860::8888",
            "::ffff:8.8.8.8",
            "2001:1::1",
            "2001:20::1",
        ];
        let blocked = [
            "::1",
            "::",
            "::ffff:127.0.0.1",
            "fe80::1",
            "fc00::1",
            "2001:db8::1",
            "3fff::1",
            "2002:0808:0808::1",
        ];
        for ip in global {
            let addr: Ipv6Addr = ip.parse().unwrap();
            assert!(ipv6_is_global(addr), "{ip} should be global");
        }
        for ip in blocked {
            let addr: Ipv6Addr = ip.parse().unwrap();
            assert!(!ipv6_is_global(addr), "{ip} should not be global");
        }
    }

    #[test]
    fn is_blocked_address_catches_embedded_ipv4_that_raw_is_global_misses() {
        // Raw stdlib is_global says these are "global" (NAT64/IPv4-compatible
        // aren't in the IPv6 private-network list), but the embedded IPv4
        // target is link-local -- the whole point of `_embedded_ipv4`.
        let nat64_metadata: Ipv6Addr = "64:ff9b::169.254.169.254".parse().unwrap();
        assert!(ipv6_is_global(nat64_metadata));
        assert!(is_blocked_address(IpAddr::V6(nat64_metadata)));

        let compat_metadata: Ipv6Addr = "::169.254.169.254".parse().unwrap();
        assert!(ipv6_is_global(compat_metadata));
        assert!(is_blocked_address(IpAddr::V6(compat_metadata)));

        let nat64_public: Ipv6Addr = "64:ff9b::8.8.8.8".parse().unwrap();
        assert!(!is_blocked_address(IpAddr::V6(nat64_public)));

        let compat_public: Ipv6Addr = "::8.8.8.8".parse().unwrap();
        assert!(!is_blocked_address(IpAddr::V6(compat_public)));
    }

    #[test]
    fn embedded_ipv4_extracts_mapped_sixtofour_nat64_and_compatible_forms() {
        assert_eq!(
            embedded_ipv4("::ffff:8.8.8.8".parse().unwrap()),
            Some(Ipv4Addr::new(8, 8, 8, 8))
        );
        assert_eq!(
            embedded_ipv4("2002:0808:0808::1".parse().unwrap()),
            Some(Ipv4Addr::new(8, 8, 8, 8))
        );
        assert_eq!(
            embedded_ipv4("64:ff9b::8.8.8.8".parse().unwrap()),
            Some(Ipv4Addr::new(8, 8, 8, 8))
        );
        assert_eq!(
            embedded_ipv4("::8.8.8.8".parse().unwrap()),
            Some(Ipv4Addr::new(8, 8, 8, 8))
        );
        assert_eq!(embedded_ipv4("::1".parse().unwrap()), None);
        assert_eq!(embedded_ipv4("2001:db8::1".parse().unwrap()), None);
    }

    #[test]
    fn is_blocked_hostname_matches_localhost_and_subdomains() {
        assert!(is_blocked_hostname("localhost"));
        assert!(is_blocked_hostname("LOCALHOST"));
        assert!(is_blocked_hostname("localhost."));
        assert!(is_blocked_hostname("foo.localhost"));
        assert!(!is_blocked_hostname("example.com"));
        assert!(!is_blocked_hostname("notlocalhost.com"));
    }

    #[test]
    fn parse_request_target_rejects_unsupported_schemes() {
        assert!(parse_request_target("ftp://example.com/").is_err());
        assert!(parse_request_target("javascript:alert(1)").is_err());
    }

    #[test]
    fn parse_request_target_accepts_http_and_https() {
        let target = parse_request_target("https://example.com/page").unwrap();
        assert!(matches!(target.host, url::Host::Domain(ref d) if d == "example.com"));
        assert_eq!(
            host_header_for_test("https://example.com/page"),
            "example.com"
        );
    }

    #[test]
    fn host_header_includes_a_non_default_port() {
        assert_eq!(
            host_header_for_test("http://example.com:8080/"),
            "example.com:8080"
        );
    }

    #[test]
    fn host_header_omits_the_default_port() {
        assert_eq!(
            host_header_for_test("http://example.com:80/"),
            "example.com"
        );
    }

    #[test]
    fn ipv6_is_private_matches_is_global_negation_outside_the_100_64_carveout() {
        assert!(ipv6_is_private("fe80::1".parse().unwrap()));
        assert!(!ipv6_is_private("2001:4860:4860::8888".parse().unwrap()));
    }

    #[test]
    fn fetch_response_rejects_loopback_and_private_targets() {
        assert!(fetch_response("http://127.0.0.1/").is_err());
        assert!(fetch_response("http://localhost/").is_err());
        assert!(fetch_response("http://10.0.0.1/").is_err());
        assert!(fetch_response("http://169.254.169.254/").is_err());
    }

    #[test]
    fn load_web_page_reports_a_failure_message_for_a_blocked_host() {
        let result = load_web_page("http://127.0.0.1/");
        assert_eq!(result, "Failed to fetch url: http://127.0.0.1/");
    }

    #[test]
    fn extract_text_strips_tags_scripts_and_styles() {
        let html = "<html><head><style>.a{}</style><script>evil()</script></head><body><p>This is a real sentence with words.</p><p>Hi</p></body></html>";
        let text = extract_text(html);
        assert!(text.contains("This is a real sentence with words."));
        assert!(!text.contains("evil()"));
        assert!(!text.contains(".a{}"));
        assert!(!text.contains("Hi"));
    }

    #[test]
    fn extract_text_decodes_common_entities() {
        let html = "<p>Tom &amp; Jerry said &quot;hello&quot; to everyone in the room.</p>";
        let text = extract_text(html);
        assert!(text.contains("Tom & Jerry said \"hello\" to everyone in the room."));
    }

    #[rusty_tokio::test]
    async fn tool_run_async_reports_a_failure_message_for_a_blocked_host() {
        let tool = LoadWebPageTool::new();
        let ic = adk_agents::invocation_context::InvocationContextBuilder::new(
            "inv-1",
            adk_agents::session::Session::new("app", "user", "s1"),
        )
        .build();
        let mut ctx = adk_agents::context::Context::new(ic);
        let mut args = std::collections::BTreeMap::new();
        args.insert(
            "url".to_string(),
            Value::String("http://127.0.0.1/".to_string()),
        );
        let result = tool.run_async(&args, &mut ctx).await.unwrap();
        assert_eq!(
            result,
            Value::String("Failed to fetch url: http://127.0.0.1/".to_string())
        );
    }
}
