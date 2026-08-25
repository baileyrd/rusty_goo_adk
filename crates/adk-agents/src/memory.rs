//! Capability C0246: the `memory` package's public export surface,
//! ported from `google.adk.memory.__init__`.
//!
//! **Eager vs. lazy has no Rust equivalent, so it's dropped**: same
//! precedent `agents.rs`/`plugins.rs` already established for their own
//! source packages — the source's `_LAZY_MEMBERS`/`__getattr__` split
//! exists only to avoid importing heavy backend modules at package-load
//! time; a Rust `pub use` has no such cost to defer, so every name below
//! is a plain re-export regardless of which list it was on in the
//! source.
//!
//! **`BaseMemoryService` → [`crate::services::MemoryService`]**: this
//! port's memory-service trait already lives in `services.rs` (C0243)
//! under its own name; re-exported here under the source's own name too
//! so a caller reaching for `memory::BaseMemoryService` finds it.
//!
//! **Not re-exported here, disclosed — genuinely unbuilt in this
//! port**: `VertexAiMemoryBankService`/`VertexAiRagMemoryService`, both
//! GCP/Vertex-backed and blocked on the same undecided GCP-SDK
//! dependency as every other Vertex-backed service in this port. This
//! module re-exports only what actually exists; revisit once either
//! backend lands.

pub use crate::in_memory_memory_service::InMemoryMemoryService;
pub use crate::services::MemoryService as BaseMemoryService;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_memory_service_is_reachable_through_the_facade() {
        let _service_type_check: Option<InMemoryMemoryService> = None;
    }

    #[test]
    fn base_memory_service_is_reachable_through_the_facade() {
        struct StubMemoryService;
        impl BaseMemoryService for StubMemoryService {
            fn add_session_to_memory(&self, _session: &crate::session::Session) {}

            fn add_events_to_memory(
                &self,
                _app_name: &str,
                _user_id: &str,
                _session_id: &str,
                _events: &[adk_events::Event],
                _custom_metadata: Option<
                    &std::collections::BTreeMap<String, rusty_serde::value::Value>,
                >,
            ) {
            }

            fn add_memory(
                &self,
                _app_name: &str,
                _user_id: &str,
                _memories: &[crate::services::MemoryEntry],
                _custom_metadata: Option<
                    &std::collections::BTreeMap<String, rusty_serde::value::Value>,
                >,
            ) {
            }

            fn search_memory(
                &self,
                _app_name: &str,
                _user_id: &str,
                _query: &str,
            ) -> crate::services::SearchMemoryResponse {
                crate::services::SearchMemoryResponse::default()
            }
        }
        let _service = StubMemoryService;
    }
}
