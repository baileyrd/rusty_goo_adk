//! Capability C0415: `ToolContext`, ported from
//! `google.adk.tools.tool_context`.
//!
//! The source's `ToolContext = Context` is a bare type alias (`Context`
//! already carries everything a tool needs — state, artifacts, the
//! invocation). This port already has a real `Context` in `adk-agents`
//! (Phase 2), so this alias is the whole capability.
//!
//! **`AuthCredential`/`AuthHandler`/`AuthConfig` back-compat re-exports**:
//! the source lazily re-exports these three names via `__getattr__`
//! (`tool_context.py`'s own `_lazy_imports` map) so old code importing
//! them from `tools.tool_context` keeps working. All three types now
//! exist in `adk-agents` (`auth_credential.rs`, `auth_handler.rs`,
//! `auth_tool.rs`), which `adk-tools` already depends on — ported as
//! plain `pub use` re-exports, since this port has no lazy-attribute
//! mechanism to mirror `__getattr__` with (the same "static instead of
//! lazy" adaptation already established elsewhere in this port).

/// `ToolContext = Context` — see the module doc.
pub type ToolContext = adk_agents::context::Context;

/// Back-compat re-export — see the module doc.
pub use adk_agents::auth_credential::AuthCredential;
/// Back-compat re-export — see the module doc.
pub use adk_agents::auth_handler::AuthHandler;
/// Back-compat re-export — see the module doc.
pub use adk_agents::auth_tool::AuthConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_credential_re_export_is_the_same_type_as_the_source() {
        let _: AuthCredential =
            adk_agents::auth_credential::AuthCredential::api_key("k".to_string());
    }

    #[test]
    fn auth_config_re_export_is_the_same_type_as_the_source() {
        fn accepts_source_auth_config(_: adk_agents::auth_tool::AuthConfig) {}
        fn build() -> AuthConfig {
            let scheme = adk_agents::auth_schemes::AuthScheme::Custom(
                adk_agents::auth_schemes::CustomAuthScheme {
                    type_: "test".to_string(),
                    extra: None,
                },
            );
            AuthConfig::new(scheme, None, None, Some("key".to_string()))
        }
        accepts_source_auth_config(build());
    }

    #[test]
    fn auth_handler_re_export_constructs_from_the_re_exported_auth_config() {
        let scheme = adk_agents::auth_schemes::AuthScheme::Custom(
            adk_agents::auth_schemes::CustomAuthScheme {
                type_: "test".to_string(),
                extra: None,
            },
        );
        let auth_config = AuthConfig::new(scheme, None, None, Some("key".to_string()));
        let _: AuthHandler = AuthHandler::new(auth_config);
    }
}
