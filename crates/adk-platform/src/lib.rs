//! Runtime-abstraction primitives ported from `google.adk.platform`.
//!
//! Mirrors the source's own asymmetric export shape (capability C0001):
//! only [`random::get_random`]/[`random::set_random_provider`]/
//! [`random::reset_random_provider`] are re-exported at the crate root
//! (matching the source's `platform/__init__.py`, which re-exports only
//! its `_random` module's functions); [`time`], [`uuid`], and [`thread`]
//! must be reached via their own submodules.
//!
//! **Forward-pull**: [`visual_builder_context`] (C0936) is from
//! `utils/_telemetry_context.py`, and [`telemetry_config`] (C0942) is from
//! `utils/_telemetry_config.py` — neither is `platform/` — both are
//! pulled in here anyway since they're the same shape of thing this crate
//! already exists for: a small, runtime-scoped primitive with no natural
//! home in any higher-level crate.

pub mod random;
pub mod telemetry_config;
pub mod thread;
pub mod time;
pub mod uuid;
pub mod visual_builder_context;

pub use random::{get_random, reset_random_provider, set_random_provider};
