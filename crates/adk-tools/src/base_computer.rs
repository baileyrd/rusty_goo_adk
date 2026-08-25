//! Capability C0445: `BaseComputer`/`ComputerState`/`ComputerEnvironment`,
//! ported from `google.adk.tools.computer_use.base_computer`.
//!
//! **Adaptation**: the source is an `abc.ABC` with `async def` methods;
//! Rust traits have no native `async fn` in a `dyn`-safe trait, so every
//! method returns [`BoxFuture`] instead, the same manual idiom
//! [`crate::base_toolset::BaseToolset`] already established.
//!
//! **`@experimental(FeatureName.COMPUTER_USE)`**: gates every class in
//! this module in the source (each one's `__init__`/construction raises
//! while the feature is off). This trait itself has no constructor to
//! gate — only a concrete implementor does — so the feature check lives
//! at [`crate::computer_use_toolset::ComputerUseToolset::new`] and
//! [`crate::computer_use_tool::ComputerUseTool::new`] instead, the same
//! "gate once, at the real entry-point constructor" collapsing
//! `environment_simulation_config.rs` already established for its own
//! multi-type cluster. [`ComputerState`]/[`ComputerEnvironment`] stay
//! plain, ungated types.
//!
//! **`screen_size`/`environment` excluded from tool generation**: both
//! are abstract on the source's `BaseComputer` (real methods a concrete
//! computer must implement), but `ComputerUseToolset.EXCLUDED_METHODS`
//! keeps them out of the auto-generated tool set — they're
//! implementation queries, not model-callable actions. See
//! `computer_use_toolset.rs`.

use std::future::Future;
use std::pin::Pin;

use crate::tool_context::ToolContext;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `base_computer.ComputerEnvironment` — case-insensitive in the source
/// (a `str` subclass `Enum`); this port models only the two real
/// members, matched case-sensitively (no caller in this port constructs
/// one from an arbitrary case-varied string yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputerEnvironment {
    /// Defaults to browser (server-side interpretation — this port does
    /// not itself promote this to `Browser`, matching the source: see
    /// `computer_use_toolset.rs`'s doc for why the `getattr(...,
    /// default=BROWSER)` fallback never actually fires for this variant
    /// either).
    Unspecified,
    Browser,
}

impl ComputerEnvironment {
    /// The wire enum member name (`types.Environment`'s member names
    /// mirror `ComputerEnvironment`'s own, see `computer_use_toolset.rs`).
    pub fn wire_name(&self) -> &'static str {
        match self {
            ComputerEnvironment::Unspecified => "ENVIRONMENT_UNSPECIFIED",
            ComputerEnvironment::Browser => "ENVIRONMENT_BROWSER",
        }
    }
}

/// `base_computer.ComputerState` — `screenshot` stays `Vec<u8>` (already
/// decoded bytes, matching the source's `bytes` field) rather than a
/// base64 string; base64-encoding only happens at the
/// [`crate::computer_use_tool::ComputerUseTool`] wire boundary, matching
/// the source (`ComputerUseTool.run_async` is the one place that calls
/// `base64.b64encode`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ComputerState {
    pub screenshot: Vec<u8>,
    pub url: Option<String>,
}

/// `Literal["up", "down", "left", "right"]` — the `scroll_document`/
/// `scroll_at` direction parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

impl ScrollDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScrollDirection::Up => "up",
            ScrollDirection::Down => "down",
            ScrollDirection::Left => "left",
            ScrollDirection::Right => "right",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "up" => Some(ScrollDirection::Up),
            "down" => Some(ScrollDirection::Down),
            "left" => Some(ScrollDirection::Left),
            "right" => Some(ScrollDirection::Right),
            _ => None,
        }
    }
}

/// C0445: `base_computer.BaseComputer` — the full browser-automation
/// contract. See the module doc for the `async fn` → [`BoxFuture`]
/// adaptation and the feature-gating collapse.
///
/// `x`/`y`/`destination_x`/`destination_y`/`magnitude`/`seconds` are
/// `i64` (the source's plain `int`, no upper/lower bound of its own —
/// [`crate::computer_use_tool::ComputerUseTool`] is what clamps a
/// caller-supplied `x`/`y` into `[0, screen_size - 1]` before a trait
/// method here ever sees it, matching the source's own
/// `ComputerUseTool._normalize_x`/`_normalize_y`).
pub trait BaseComputer: Send + Sync {
    /// Called before each tool invocation — override to bind
    /// session-level resources (sandbox, tokens, etc.) via
    /// `tool_context`. Default no-op, matching the source.
    fn prepare<'a>(&'a self, _tool_context: &'a mut ToolContext) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// Initializes the computer. Default no-op, matching the source.
    fn initialize(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    /// Releases resources held by the computer. Default no-op, matching
    /// the source.
    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }

    /// Returns `(width, height)` in pixels.
    fn screen_size(&self) -> BoxFuture<'_, (u32, u32)>;

    fn environment(&self) -> BoxFuture<'_, ComputerEnvironment>;

    fn open_web_browser(&self) -> BoxFuture<'_, ComputerState>;

    fn click_at(&self, x: i64, y: i64) -> BoxFuture<'_, ComputerState>;

    fn hover_at(&self, x: i64, y: i64) -> BoxFuture<'_, ComputerState>;

    fn type_text_at<'a>(
        &'a self,
        x: i64,
        y: i64,
        text: &'a str,
        press_enter: bool,
        clear_before_typing: bool,
    ) -> BoxFuture<'a, ComputerState>;

    fn scroll_document(&self, direction: ScrollDirection) -> BoxFuture<'_, ComputerState>;

    fn scroll_at(
        &self,
        x: i64,
        y: i64,
        direction: ScrollDirection,
        magnitude: i64,
    ) -> BoxFuture<'_, ComputerState>;

    fn wait(&self, seconds: i64) -> BoxFuture<'_, ComputerState>;

    fn go_back(&self) -> BoxFuture<'_, ComputerState>;

    fn go_forward(&self) -> BoxFuture<'_, ComputerState>;

    fn search(&self) -> BoxFuture<'_, ComputerState>;

    fn navigate<'a>(&'a self, url: &'a str) -> BoxFuture<'a, ComputerState>;

    fn key_combination<'a>(&'a self, keys: &'a [String]) -> BoxFuture<'a, ComputerState>;

    fn drag_and_drop(
        &self,
        x: i64,
        y: i64,
        destination_x: i64,
        destination_y: i64,
    ) -> BoxFuture<'_, ComputerState>;

    fn current_state(&self) -> BoxFuture<'_, ComputerState>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_environment_wire_name_matches_the_source() {
        assert_eq!(
            ComputerEnvironment::Unspecified.wire_name(),
            "ENVIRONMENT_UNSPECIFIED"
        );
        assert_eq!(
            ComputerEnvironment::Browser.wire_name(),
            "ENVIRONMENT_BROWSER"
        );
    }

    #[test]
    fn scroll_direction_round_trips_through_its_wire_string() {
        for direction in [
            ScrollDirection::Up,
            ScrollDirection::Down,
            ScrollDirection::Left,
            ScrollDirection::Right,
        ] {
            assert_eq!(ScrollDirection::parse(direction.as_str()), Some(direction));
        }
        assert_eq!(ScrollDirection::parse("diagonal"), None);
    }
}
