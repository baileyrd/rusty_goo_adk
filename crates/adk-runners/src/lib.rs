//! The core execution engine, ported from `google.adk.runners` (94
//! capability rows, C0833-C0926).
//!
//! See [`runner`]'s module doc for the batch-by-batch scoping plan —
//! most of `runners.py` depends on infrastructure this port doesn't have
//! yet (an `App` type, the workflow/node/task-delegation engine, a real
//! plugin system), so this crate builds up the buildable "legacy" (plain
//! `BaseAgent`, no node/task/live/rewind/debug) slice first.

pub mod runner;
