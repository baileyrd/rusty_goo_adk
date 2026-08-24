//! Event model ported from `google.adk.events`.
//!
//! Mirrors the source's export asymmetry (capability C0016): the source's
//! `events/__init__.py` re-exports only `Event`, `EventActions`, and
//! `RequestInput` at the package root — [`NodeInfo`], [`EventCompaction`],
//! [`UiWidget`], [`branch_path::BranchPath`], and
//! [`node_path_builder::NodePathBuilder`] are all reachable only via their
//! own submodules, matching `_BranchPath`/`_NodePathBuilder`/`EventCompaction`/
//! `UiWidget` not being re-exported from the source's package root either.

mod event;
mod event_actions;
mod request_input;

pub mod branch_path;
pub mod debug_output;
pub mod event_compaction;
pub mod json_safe;
pub mod node_info;
pub mod node_path_builder;
pub mod rewind;
pub mod ui_widget;

// `event` and `event_actions` are private modules specifically so that
// `Event`/`EventActions`'s own internal cross-references (e.g. `Event`
// using `NodeInfo`/`EventCompaction`) can still reach those types, while
// external callers only get them via the crate-root re-export below —
// matching the source's package-root re-export list exactly.
pub use event::Event;
pub use event_actions::EventActions;
pub use request_input::RequestInput;
