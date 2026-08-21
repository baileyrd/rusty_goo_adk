//! Runtime-abstraction primitives ported from `google.adk.platform`.
//!
//! Mirrors the source's own asymmetric export shape (capability C0001):
//! only [`random::get_random`]/[`random::set_random_provider`]/
//! [`random::reset_random_provider`] are re-exported at the crate root
//! (matching the source's `platform/__init__.py`, which re-exports only
//! its `_random` module's functions); [`time`], [`uuid`], and [`thread`]
//! must be reached via their own submodules.

pub mod random;
pub mod thread;
pub mod time;
pub mod uuid;

pub use random::{get_random, reset_random_provider, set_random_provider};
