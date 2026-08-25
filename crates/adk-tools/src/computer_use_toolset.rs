//! Capability C0446: `ComputerUseToolset`, ported from
//! `google.adk.tools.computer_use.computer_use_toolset`.
//!
//! **`dir(BaseComputer)`-reflection → a fixed table, disclosed**: the
//! source enumerates every non-underscore, non-excluded method on the
//! `BaseComputer` *class* at runtime to build its tool list. Rust has no
//! equivalent runtime introspection over a trait's method set, so this
//! port hard-codes the same 15-entry list [`get_tools`] iterates —
//! matching `function_tool.rs`'s own established "static instead of
//! dynamic" adaptation for the identical class of problem.
//!
//! **`initialize` leaks into the tool set — a confirmed source quirk,
//! ported faithfully**: `EXCLUDED_METHODS = {"screen_size", "environment",
//! "close", "prepare"}` does *not* list `initialize`, and `initialize` is
//! a public, non-underscore, zero-argument method on `BaseComputer` — so
//! the source's own `dir()`-based enumeration genuinely exposes it as a
//! 15th, model-callable tool alongside the 14 real actions. Verified by
//! reading `EXCLUDED_METHODS`' literal contents, not assumed. This looks
//! like an oversight (a lifecycle hook leaking into the model-facing tool
//! surface) rather than a documented feature, but per this migration's
//! standing rule a confirmed source quirk is replicated, not silently
//! "fixed" — see [`build_tool_table`].
//!
//! **`session_state` dead-code check, not ported**: the source's
//! `get_tools` also skips a method literally named `"session_state"` —
//! no such attribute exists anywhere on `BaseComputer` in this file
//! (vestigial, likely left over from a `AgentEngineSandboxComputer`-shaped
//! subclass outside this batch's scope, see C0786). Since this port
//! builds its tool table from a fixed list rather than enumerating real
//! attributes, there is nothing for this dead check to filter — dropped
//! as inapplicable, not silently narrowed.
//!
//! **`navigate`'s raw-backslash-in-netloc check, not ported (verified
//! unreachable, not assumed)**: the source refuses any `navigate` URL
//! whose `urlparse(...).netloc` contains a literal `\`, because Python's
//! stdlib `urlparse` and a real browser's URL parser (WHATWG-spec) can
//! disagree about where the authority ends when a raw backslash appears
//! — `http://169.254.169.254\@example.com/` parses to host `example.com`
//! under `urlparse` but host `169.254.169.254` under Chrome. This port's
//! `parse_request_target` (`load_web_page.rs`, C0427) is built on
//! `reqwest::Url`/the `url` crate, which *is* WHATWG-spec-compliant — the
//! same spec real browsers implement. Verified empirically (not assumed):
//! parsing `http://169.254.169.254\@example.com/` through `url::Url`
//! yields host `169.254.169.254`, matching Chrome, because the `url`
//! crate treats a raw backslash as a path/authority separator the same
//! way a browser does. The exact host-parsing mismatch this check exists
//! to prevent cannot occur through this port's URL parser, so the
//! resulting dangerous host is instead caught by the ordinary
//! blocked-hostname/blocked-address path — see
//! `navigate_refuses_a_backslash_authority_that_resolves_to_a_link_local_host`'s
//! test coverage.
//!
//! **`adapt_computer_use_tool`, not ported — disclosed blocker**: the
//! source's static `adapt_computer_use_tool` mutates
//! `llm_request.tools_dict[name]` (swaps in a new callable-bearing tool
//! by name). This port's `LlmRequest` has no `tools_dict` — `adk-tools`
//! (where `BaseTool` lives) already depends on `adk-models` (where
//! `LlmRequest` lives), the reverse of the source's own `models` →
//! `tools` import direction, so `LlmRequest` cannot hold
//! `Arc<dyn BaseTool>` without either a crate-graph cycle or moving
//! `BaseTool` itself down to `adk-genai` — a breaking-change-shaped
//! restructuring of an already-widely-used trait, not a same-batch,
//! no-sign-off change. This narrows `gemini.rs::Gemini::preprocess_request`'s
//! (C0132) own disclosed gap from "the entire toolset doesn't exist" down
//! to exactly this one function.

use std::collections::BTreeMap;
use std::sync::Arc;

use adk_agents::readonly_context::ReadonlyContext;
use adk_features::feature_decorator::{check_feature_enabled, FeatureNotEnabledError};
use adk_features::feature_registry::FeatureName;
use adk_genai::content::FunctionDeclaration;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::append_tools::{append_built_in_tool_marker_with_fields, has_built_in_tool_marker};
use crate::base_computer::{BaseComputer, ComputerState, ScrollDirection};
use crate::base_tool::BaseTool;
use crate::base_toolset::{BaseToolset, PrefixCache};
use crate::computer_use_tool::ComputerUseTool;
use crate::function_tool::ToolFn;
use crate::tool_context::ToolContext;

const COMPUTER_USE_TOOL_KEY: &str = "computerUse";
const URL_REFUSED_ERROR: &str = "navigate refused: url must be http(s) and must not target a \
     private or link-local address.";

fn computer_state_to_value(state: ComputerState) -> Value {
    Value::Map(vec![
        (
            "image".to_string(),
            Value::Map(vec![
                (
                    "mimetype".to_string(),
                    Value::String("image/png".to_string()),
                ),
                (
                    "data".to_string(),
                    Value::String(adk_agents::auth_headers::base64_encode(&state.screenshot)),
                ),
            ]),
        ),
        (
            "url".to_string(),
            state.url.map(Value::String).unwrap_or(Value::Null),
        ),
    ])
}

fn arg_i64(args: &BTreeMap<String, Value>, key: &str) -> Option<i64> {
    match args.get(key) {
        Some(Value::Int(i)) => Some(*i),
        Some(Value::UInt(u)) => Some(*u as i64),
        Some(Value::Float(f)) => Some(*f as i64),
        _ => None,
    }
}

fn arg_string(args: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    match args.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn arg_bool_or(args: &BTreeMap<String, Value>, key: &str, default: bool) -> bool {
    match args.get(key) {
        Some(Value::Bool(b)) => *b,
        _ => default,
    }
}

fn arg_direction(args: &BTreeMap<String, Value>, key: &str) -> Option<ScrollDirection> {
    arg_string(args, key).and_then(|s| ScrollDirection::parse(&s))
}

fn arg_string_list(args: &BTreeMap<String, Value>, key: &str) -> Option<Vec<String>> {
    match args.get(key) {
        Some(Value::Seq(items)) => Some(
            items
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn missing_arg_error(name: &str) -> Value {
    Value::Map(vec![(
        "error".to_string(),
        Value::String(format!("missing or invalid argument for {name}")),
    )])
}

fn object_schema(properties: Vec<(&str, Value)>, required: &[&str]) -> Value {
    let mut fields = vec![("type".to_string(), Value::String("object".to_string()))];
    if !properties.is_empty() {
        fields.push((
            "properties".to_string(),
            Value::Map(
                properties
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ),
        ));
    }
    if !required.is_empty() {
        fields.push((
            "required".to_string(),
            Value::Seq(
                required
                    .iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
        ));
    }
    Value::Map(fields)
}

fn int_property(description: &str) -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("integer".to_string())),
        (
            "description".to_string(),
            Value::String(description.to_string()),
        ),
    ])
}

fn string_property(description: &str) -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("string".to_string())),
        (
            "description".to_string(),
            Value::String(description.to_string()),
        ),
    ])
}

fn bool_property(description: &str) -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("boolean".to_string())),
        (
            "description".to_string(),
            Value::String(description.to_string()),
        ),
    ])
}

fn direction_property() -> Value {
    Value::Map(vec![
        ("type".to_string(), Value::String("string".to_string())),
        (
            "enum".to_string(),
            Value::Seq(vec![
                Value::String("up".to_string()),
                Value::String("down".to_string()),
                Value::String("left".to_string()),
                Value::String("right".to_string()),
            ]),
        ),
        (
            "description".to_string(),
            Value::String("The direction to scroll.".to_string()),
        ),
    ])
}

fn xy_properties(x_desc: &str, y_desc: &str) -> Vec<(&'static str, Value)> {
    vec![("x", int_property(x_desc)), ("y", int_property(y_desc))]
}

#[derive(Debug, rusty_err::Error)]
pub enum ComputerUseToolsetError {
    #[error("{0}")]
    FeatureNotEnabled(#[from] FeatureNotEnabledError),
}

/// C0446: `computer_use_toolset.ComputerUseToolset`. See the module doc
/// for the `dir()`-reflection-to-fixed-table adaptation, the `initialize`
/// quirk, the unported raw-backslash-netloc check (verified
/// unreachable), and the disclosed `adapt_computer_use_tool` gap.
pub struct ComputerUseToolset {
    computer: Arc<dyn BaseComputer>,
    excluded_predefined_functions: Option<Vec<String>>,
    allow_private_network_access: bool,
    initialized: rusty_tokio::sync::Mutex<bool>,
    tools: rusty_tokio::sync::Mutex<Option<Vec<Arc<dyn BaseTool>>>>,
    prefix_cache: std::sync::Mutex<PrefixCache>,
}

impl ComputerUseToolset {
    /// `ComputerUseToolset.__init__`.
    pub fn new(
        computer: Arc<dyn BaseComputer>,
        excluded_predefined_functions: Option<Vec<String>>,
        allow_private_network_access: bool,
    ) -> Result<Self, ComputerUseToolsetError> {
        check_feature_enabled(FeatureName::ComputerUse)?;
        Ok(Self {
            computer,
            excluded_predefined_functions,
            allow_private_network_access,
            initialized: rusty_tokio::sync::Mutex::new(false),
            tools: rusty_tokio::sync::Mutex::new(None),
            prefix_cache: std::sync::Mutex::new(PrefixCache::new()),
        })
    }

    async fn ensure_initialized(&self) {
        let mut initialized = self.initialized.lock().await;
        if !*initialized {
            self.computer.initialize().await;
            *initialized = true;
        }
    }

    fn is_excluded(&self, name: &str) -> bool {
        self.excluded_predefined_functions
            .as_ref()
            .is_some_and(|excluded| excluded.iter().any(|n| n == name))
    }

    /// `ComputerUseToolset._wrap_navigate_with_url_validation`, minus the
    /// unreachable-under-this-port's-URL-parser backslash check — see
    /// the module doc.
    async fn navigate_action(
        computer: Arc<dyn BaseComputer>,
        allow_private_network_access: bool,
        args: &BTreeMap<String, Value>,
    ) -> Value {
        let Some(url) = arg_string(args, "url") else {
            return missing_arg_error("navigate");
        };

        let refused = crate::load_web_page::parse_request_target(&url)
            .map_err(|_| ())
            .and_then(|target| {
                if allow_private_network_access {
                    return Ok(());
                }
                let hostname = target.hostname();
                if crate::load_web_page::is_blocked_hostname(&hostname) {
                    return Err(());
                }
                crate::load_web_page::resolve_direct_addresses(&hostname)
                    .map(|_| ())
                    .map_err(|_| ())
            })
            .is_err();

        if refused {
            let state = computer.current_state().await;
            return Value::Map(vec![
                (
                    "error".to_string(),
                    Value::String(URL_REFUSED_ERROR.to_string()),
                ),
                (
                    "url".to_string(),
                    state.url.map(Value::String).unwrap_or(Value::Null),
                ),
            ]);
        }

        computer_state_to_value(computer.navigate(&url).await)
    }

    /// The fixed 15-entry action table — see the module doc for why this
    /// replaces the source's `dir(BaseComputer)` reflection, and for the
    /// `initialize` quirk. Each entry pairs a tool name/declaration with
    /// a closure implementing state-binding (`computer.prepare`) plus the
    /// real action; `navigate` additionally gets the SSRF wrapper.
    fn build_tool_table(&self, screen_size: (u32, u32)) -> Vec<Arc<dyn BaseTool>> {
        let computer = self.computer.clone();
        let allow_private_network_access = self.allow_private_network_access;

        let mut entries: Vec<(&'static str, &'static str, FunctionDeclaration, ToolFn)> = vec![
            (
                "open_web_browser",
                "Opens the web browser.",
                FunctionDeclaration {
                    name: Some("open_web_browser".to_string()),
                    description: Some("Opens the web browser.".to_string()),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, _args| {
                    Box::pin(async move { Ok(computer.open_web_browser().await) })
                }),
            ),
            (
                "click_at",
                "Clicks at a specific x, y coordinate on the webpage.",
                FunctionDeclaration {
                    name: Some("click_at".to_string()),
                    description: Some(
                        "Clicks at a specific x, y coordinate on the webpage. The 'x' and 'y' \
                         values are absolute values, scaled to the height and width of the \
                         screen."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        xy_properties(
                            "The x-coordinate to click at.",
                            "The y-coordinate to click at.",
                        ),
                        &["x", "y"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let (Some(x), Some(y)) = (arg_i64(args, "x"), arg_i64(args, "y")) else {
                            return Err(missing_arg_error("click_at"));
                        };
                        Ok(computer.click_at(x, y).await)
                    })
                }),
            ),
            (
                "hover_at",
                "Hovers at a specific x, y coordinate on the webpage.",
                FunctionDeclaration {
                    name: Some("hover_at".to_string()),
                    description: Some(
                        "Hovers at a specific x, y coordinate on the webpage. May be used to \
                         explore sub-menus that appear on hover. The 'x' and 'y' values are \
                         absolute values, scaled to the height and width of the screen."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        xy_properties(
                            "The x-coordinate to hover at.",
                            "The y-coordinate to hover at.",
                        ),
                        &["x", "y"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let (Some(x), Some(y)) = (arg_i64(args, "x"), arg_i64(args, "y")) else {
                            return Err(missing_arg_error("hover_at"));
                        };
                        Ok(computer.hover_at(x, y).await)
                    })
                }),
            ),
            (
                "type_text_at",
                "Types text at a specific x, y coordinate.",
                FunctionDeclaration {
                    name: Some("type_text_at".to_string()),
                    description: Some(
                        "Types text at a specific x, y coordinate. The system automatically \
                         presses ENTER after typing unless press_enter is false. The system \
                         automatically clears any existing content before typing unless \
                         clear_before_typing is false. The 'x' and 'y' values are absolute \
                         values, scaled to the height and width of the screen."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![
                            ("x", int_property("The x-coordinate to type at.")),
                            ("y", int_property("The y-coordinate to type at.")),
                            ("text", string_property("The text to type.")),
                            (
                                "press_enter",
                                bool_property("Whether to press ENTER after typing."),
                            ),
                            (
                                "clear_before_typing",
                                bool_property("Whether to clear existing content before typing."),
                            ),
                        ],
                        &["x", "y", "text"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let (Some(x), Some(y), Some(text)) = (
                            arg_i64(args, "x"),
                            arg_i64(args, "y"),
                            arg_string(args, "text"),
                        ) else {
                            return Err(missing_arg_error("type_text_at"));
                        };
                        let press_enter = arg_bool_or(args, "press_enter", true);
                        let clear_before_typing = arg_bool_or(args, "clear_before_typing", true);
                        Ok(computer
                            .type_text_at(x, y, &text, press_enter, clear_before_typing)
                            .await)
                    })
                }),
            ),
            (
                "scroll_document",
                "Scrolls the entire webpage up, down, left or right.",
                FunctionDeclaration {
                    name: Some("scroll_document".to_string()),
                    description: Some(
                        "Scrolls the entire webpage \"up\", \"down\", \"left\" or \"right\" \
                         based on direction."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![("direction", direction_property())],
                        &["direction"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let Some(direction) = arg_direction(args, "direction") else {
                            return Err(missing_arg_error("scroll_document"));
                        };
                        Ok(computer.scroll_document(direction).await)
                    })
                }),
            ),
            (
                "scroll_at",
                "Scrolls up, down, right, or left at a x, y coordinate by magnitude.",
                FunctionDeclaration {
                    name: Some("scroll_at".to_string()),
                    description: Some(
                        "Scrolls up, down, right, or left at a x, y coordinate by magnitude. \
                         The 'x' and 'y' values are absolute values, scaled to the height and \
                         width of the screen."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![
                            ("x", int_property("The x-coordinate to scroll at.")),
                            ("y", int_property("The y-coordinate to scroll at.")),
                            ("direction", direction_property()),
                            ("magnitude", int_property("The amount to scroll.")),
                        ],
                        &["x", "y", "direction", "magnitude"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let (Some(x), Some(y), Some(direction), Some(magnitude)) = (
                            arg_i64(args, "x"),
                            arg_i64(args, "y"),
                            arg_direction(args, "direction"),
                            arg_i64(args, "magnitude"),
                        ) else {
                            return Err(missing_arg_error("scroll_at"));
                        };
                        Ok(computer.scroll_at(x, y, direction, magnitude).await)
                    })
                }),
            ),
            (
                "wait",
                "Waits for n seconds to allow unfinished webpage processes to complete.",
                FunctionDeclaration {
                    name: Some("wait".to_string()),
                    description: Some(
                        "Waits for n seconds to allow unfinished webpage processes to complete."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![("seconds", int_property("The number of seconds to wait."))],
                        &["seconds"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let Some(seconds) = arg_i64(args, "seconds") else {
                            return Err(missing_arg_error("wait"));
                        };
                        Ok(computer.wait(seconds).await)
                    })
                }),
            ),
            (
                "go_back",
                "Navigates back to the previous webpage in the browser history.",
                FunctionDeclaration {
                    name: Some("go_back".to_string()),
                    description: Some(
                        "Navigates back to the previous webpage in the browser history."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, _args| {
                    Box::pin(async move { Ok(computer.go_back().await) })
                }),
            ),
            (
                "go_forward",
                "Navigates forward to the next webpage in the browser history.",
                FunctionDeclaration {
                    name: Some("go_forward".to_string()),
                    description: Some(
                        "Navigates forward to the next webpage in the browser history.".to_string(),
                    ),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, _args| {
                    Box::pin(async move { Ok(computer.go_forward().await) })
                }),
            ),
            (
                "search",
                "Directly jumps to a search engine home page.",
                FunctionDeclaration {
                    name: Some("search".to_string()),
                    description: Some(
                        "Directly jumps to a search engine home page. Used when you need to \
                         start with a search."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, _args| {
                    Box::pin(async move { Ok(computer.search().await) })
                }),
            ),
            (
                "key_combination",
                "Presses keyboard keys and combinations, such as \"control+c\" or \"enter\".",
                FunctionDeclaration {
                    name: Some("key_combination".to_string()),
                    description: Some(
                        "Presses keyboard keys and combinations, such as \"control+c\" or \
                         \"enter\"."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![(
                            "keys",
                            Value::Map(vec![
                                ("type".to_string(), Value::String("array".to_string())),
                                (
                                    "items".to_string(),
                                    Value::Map(vec![(
                                        "type".to_string(),
                                        Value::String("string".to_string()),
                                    )]),
                                ),
                                (
                                    "description".to_string(),
                                    Value::String(
                                        "List of keys to press in combination.".to_string(),
                                    ),
                                ),
                            ]),
                        )],
                        &["keys"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let Some(keys) = arg_string_list(args, "keys") else {
                            return Err(missing_arg_error("key_combination"));
                        };
                        Ok(computer.key_combination(&keys).await)
                    })
                }),
            ),
            (
                "drag_and_drop",
                "Drag and drop an element from a x, y coordinate to a destination coordinate.",
                FunctionDeclaration {
                    name: Some("drag_and_drop".to_string()),
                    description: Some(
                        "Drag and drop an element from a x, y coordinate to a destination \
                         destination_y, destination_x coordinate. All values are absolute, \
                         scaled to the height and width of the screen."
                            .to_string(),
                    ),
                    parameters: Some(object_schema(
                        vec![
                            (
                                "x",
                                int_property("The x-coordinate to start dragging from."),
                            ),
                            (
                                "y",
                                int_property("The y-coordinate to start dragging from."),
                            ),
                            (
                                "destination_x",
                                int_property("The x-coordinate to drop at."),
                            ),
                            (
                                "destination_y",
                                int_property("The y-coordinate to drop at."),
                            ),
                        ],
                        &["x", "y", "destination_x", "destination_y"],
                    )),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, args| {
                    Box::pin(async move {
                        let (Some(x), Some(y), Some(destination_x), Some(destination_y)) = (
                            arg_i64(args, "x"),
                            arg_i64(args, "y"),
                            arg_i64(args, "destination_x"),
                            arg_i64(args, "destination_y"),
                        ) else {
                            return Err(missing_arg_error("drag_and_drop"));
                        };
                        Ok(computer
                            .drag_and_drop(x, y, destination_x, destination_y)
                            .await)
                    })
                }),
            ),
            (
                "current_state",
                "Returns the current state of the current webpage.",
                FunctionDeclaration {
                    name: Some("current_state".to_string()),
                    description: Some(
                        "Returns the current state of the current webpage.".to_string(),
                    ),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                make_action(computer.clone(), |computer, _args| {
                    Box::pin(async move { Ok(computer.current_state().await) })
                }),
            ),
            (
                "initialize",
                "Initialize the computer.",
                FunctionDeclaration {
                    name: Some("initialize".to_string()),
                    description: Some("Initialize the computer.".to_string()),
                    parameters: Some(object_schema(vec![], &[])),
                    ..Default::default()
                },
                {
                    // Not routed through `make_action`: `initialize`
                    // returns `None` in the source, not a `ComputerState`
                    // — so there's nothing to convert to an image dict —
                    // but it's still one of the `dir()`-enumerated
                    // methods, so it still gets the same state-binding
                    // `prepare()` call every other entry gets (the
                    // source's loop wraps every surviving method the
                    // same way, `navigate` alone gets an *additional*
                    // wrap).
                    let computer = computer.clone();
                    Arc::new(
                        move |_args: &BTreeMap<String, Value>, tool_context: &mut ToolContext| {
                            let computer = computer.clone();
                            Box::pin(async move {
                                computer.prepare(tool_context).await;
                                computer.initialize().await;
                                Value::Null
                            }) as crate::base_tool::BoxFuture<'_, Value>
                        },
                    )
                },
            ),
        ];

        // `navigate` is built separately: it needs the SSRF wrapper, not
        // just the state-binding one every other entry gets via
        // `make_action`.
        let navigate_computer = computer.clone();
        let navigate_fn: ToolFn = Arc::new(move |args, _ctx| {
            let computer = navigate_computer.clone();
            let args = args.clone();
            Box::pin(async move {
                ComputerUseToolset::navigate_action(computer, allow_private_network_access, &args)
                    .await
            })
        });
        entries.push((
            "navigate",
            "Navigates directly to a specified URL.",
            FunctionDeclaration {
                name: Some("navigate".to_string()),
                description: Some("Navigates directly to a specified URL.".to_string()),
                parameters: Some(object_schema(
                    vec![("url", string_property("The URL to navigate to."))],
                    &["url"],
                )),
                ..Default::default()
            },
            navigate_fn,
        ));

        entries
            .into_iter()
            .filter(|(name, _, _, _)| !self.is_excluded(name))
            .filter_map(|(name, description, declaration, func)| {
                ComputerUseTool::new(
                    name,
                    description,
                    declaration,
                    Vec::new(),
                    func,
                    screen_size,
                    (1000, 1000),
                )
                .ok()
                .map(|tool| Arc::new(tool) as Arc<dyn BaseTool>)
            })
            .collect()
    }
}

/// Wraps a `computer.<action>(...)` call with the state-binding step
/// (`computer.prepare(tool_context)`, run before every call when a
/// `tool_context` is available — see the source's own
/// `_wrap_method_with_state_binding`) and converts its `ComputerState`
/// result to the wire `Value`. `handler` extracts args and calls the
/// real trait method; `Err` from `handler` is an argument-extraction
/// failure (surfaced as an error dict, not a panic).
fn make_action(
    computer: Arc<dyn BaseComputer>,
    handler: impl for<'a> Fn(
            &'a Arc<dyn BaseComputer>,
            &'a BTreeMap<String, Value>,
        ) -> crate::base_computer::BoxFuture<'a, Result<ComputerState, Value>>
        + Send
        + Sync
        + 'static,
) -> ToolFn {
    let handler = Arc::new(handler);
    Arc::new(move |args, tool_context| {
        let computer = computer.clone();
        let handler = handler.clone();
        let args = args.clone();
        Box::pin(async move {
            computer.prepare(tool_context).await;
            match handler(&computer, &args).await {
                Ok(state) => computer_state_to_value(state),
                Err(error) => error,
            }
        })
    })
}

impl BaseToolset for ComputerUseToolset {
    fn get_tools<'a>(
        &'a self,
        _readonly_context: Option<&'a ReadonlyContext>,
    ) -> crate::base_tool::BoxFuture<'a, Vec<Arc<dyn BaseTool>>> {
        Box::pin(async move {
            {
                let cached = self.tools.lock().await;
                if let Some(tools) = &*cached {
                    return tools.clone();
                }
            }
            self.ensure_initialized().await;
            let screen_size = self.computer.screen_size().await;
            let built = self.build_tool_table(screen_size);
            *self.tools.lock().await = Some(built.clone());
            built
        })
    }

    fn prefix_cache(&self) -> &std::sync::Mutex<PrefixCache> {
        &self.prefix_cache
    }

    fn close<'a>(&'a self) -> crate::base_tool::BoxFuture<'a, ()> {
        Box::pin(async move { self.computer.close().await })
    }

    fn process_llm_request<'a>(
        &'a self,
        _tool_context: &'a mut ToolContext,
        llm_request: &'a mut LlmRequest,
    ) -> crate::base_tool::BoxFuture<'a, ()> {
        Box::pin(async move {
            let tools = self.get_tools(None).await;
            for tool in &tools {
                if let Some(declaration) = tool.get_declaration() {
                    crate::append_tools::merge_declarations(
                        llm_request,
                        [(tool.name().to_string(), declaration)],
                    );
                }
            }

            if has_built_in_tool_marker(llm_request, COMPUTER_USE_TOOL_KEY) {
                return;
            }

            let environment = self.computer.environment().await;
            let mut fields = vec![(
                "environment".to_string(),
                Value::String(environment.wire_name().to_string()),
            )];
            if let Some(excluded) = &self.excluded_predefined_functions {
                fields.push((
                    "excludedPredefinedFunctions".to_string(),
                    Value::Seq(excluded.iter().map(|s| Value::String(s.clone())).collect()),
                ));
            }
            append_built_in_tool_marker_with_fields(
                llm_request,
                COMPUTER_USE_TOOL_KEY,
                Value::Map(fields),
            );
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_computer::ComputerEnvironment;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use std::sync::Mutex as StdMutex;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    struct FakeComputer {
        prepared: StdMutex<u32>,
        current_state_url: Option<String>,
    }

    impl FakeComputer {
        fn new() -> Self {
            Self {
                prepared: StdMutex::new(0),
                current_state_url: Some("https://example.com".to_string()),
            }
        }
    }

    fn state(url: &str) -> ComputerState {
        ComputerState {
            screenshot: vec![1, 2, 3],
            url: Some(url.to_string()),
        }
    }

    impl BaseComputer for FakeComputer {
        fn prepare<'a>(
            &'a self,
            _tool_context: &'a mut ToolContext,
        ) -> crate::base_computer::BoxFuture<'a, ()> {
            Box::pin(async move {
                *self.prepared.lock().unwrap() += 1;
            })
        }

        fn screen_size(&self) -> crate::base_computer::BoxFuture<'_, (u32, u32)> {
            Box::pin(async { (1920, 1080) })
        }

        fn environment(&self) -> crate::base_computer::BoxFuture<'_, ComputerEnvironment> {
            Box::pin(async { ComputerEnvironment::Browser })
        }

        fn open_web_browser(&self) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://start.example.com") })
        }

        fn click_at(&self, _x: i64, _y: i64) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://clicked.example.com") })
        }

        fn hover_at(&self, _x: i64, _y: i64) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://hover.example.com") })
        }

        fn type_text_at<'a>(
            &'a self,
            _x: i64,
            _y: i64,
            _text: &'a str,
            _press_enter: bool,
            _clear_before_typing: bool,
        ) -> crate::base_computer::BoxFuture<'a, ComputerState> {
            Box::pin(async { state("https://typed.example.com") })
        }

        fn scroll_document(
            &self,
            _direction: ScrollDirection,
        ) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://scrolled.example.com") })
        }

        fn scroll_at(
            &self,
            _x: i64,
            _y: i64,
            _direction: ScrollDirection,
            _magnitude: i64,
        ) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://scrolled-at.example.com") })
        }

        fn wait(&self, _seconds: i64) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://waited.example.com") })
        }

        fn go_back(&self) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://back.example.com") })
        }

        fn go_forward(&self) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://forward.example.com") })
        }

        fn search(&self) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://search.example.com") })
        }

        fn navigate<'a>(
            &'a self,
            url: &'a str,
        ) -> crate::base_computer::BoxFuture<'a, ComputerState> {
            Box::pin(async move { state(url) })
        }

        fn key_combination<'a>(
            &'a self,
            _keys: &'a [String],
        ) -> crate::base_computer::BoxFuture<'a, ComputerState> {
            Box::pin(async { state("https://keys.example.com") })
        }

        fn drag_and_drop(
            &self,
            _x: i64,
            _y: i64,
            _destination_x: i64,
            _destination_y: i64,
        ) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            Box::pin(async { state("https://dragged.example.com") })
        }

        fn current_state(&self) -> crate::base_computer::BoxFuture<'_, ComputerState> {
            let url = self.current_state_url.clone();
            Box::pin(async move {
                ComputerState {
                    screenshot: vec![],
                    url,
                }
            })
        }
    }

    fn toolset(allow_private_network_access: bool) -> ComputerUseToolset {
        // SAFETY (test-only): the feature check requires
        // ADK_ENABLE_COMPUTER_USE=1 in the real environment; tests set it
        // directly rather than depending on process-wide env state.
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        ComputerUseToolset::new(
            Arc::new(FakeComputer::new()),
            None,
            allow_private_network_access,
        )
        .unwrap()
    }

    #[rusty_tokio::test]
    async fn get_tools_builds_15_tools_including_the_leaked_initialize() {
        let toolset = toolset(true);
        let tools = toolset.get_tools(None).await;
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 15);
        assert!(names.contains(&"initialize"));
        assert!(!names.contains(&"screen_size"));
        assert!(!names.contains(&"environment"));
        assert!(!names.contains(&"close"));
        assert!(!names.contains(&"prepare"));
    }

    #[rusty_tokio::test]
    async fn get_tools_is_memoized() {
        let toolset = toolset(true);
        let first = toolset.get_tools(None).await;
        let second = toolset.get_tools(None).await;
        assert_eq!(first.len(), second.len());
    }

    #[rusty_tokio::test]
    async fn get_tools_honors_excluded_predefined_functions() {
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(
            Arc::new(FakeComputer::new()),
            Some(vec!["wait".to_string(), "go_back".to_string()]),
            true,
        )
        .unwrap();
        let tools = toolset.get_tools(None).await;
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"wait"));
        assert!(!names.contains(&"go_back"));
        assert_eq!(names.len(), 13);
    }

    #[rusty_tokio::test]
    async fn click_at_prepares_the_computer_before_running() {
        let computer = Arc::new(FakeComputer::new());
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(computer.clone(), None, true).unwrap();
        let tools = toolset.get_tools(None).await;
        let click = tools.iter().find(|t| t.name() == "click_at").unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert("x".to_string(), Value::Int(500));
        args.insert("y".to_string(), Value::Int(500));
        let result = click.run_async(&args, &mut context).await.unwrap();
        assert!(matches!(result, Value::Map(_)));
        assert_eq!(*computer.prepared.lock().unwrap(), 1);
    }

    #[rusty_tokio::test]
    async fn navigate_refuses_a_url_targeting_a_link_local_address() {
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(Arc::new(FakeComputer::new()), None, false).unwrap();
        let tools = toolset.get_tools(None).await;
        let navigate = tools.iter().find(|t| t.name() == "navigate").unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "url".to_string(),
            Value::String("http://169.254.169.254/".to_string()),
        );
        let result = navigate.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        assert!(fields.iter().any(|(k, _)| k == "error"));
        let url = fields.iter().find(|(k, _)| k == "url").unwrap();
        assert_eq!(url.1, Value::String("https://example.com".to_string()));
    }

    #[rusty_tokio::test]
    async fn navigate_refuses_a_backslash_authority_that_resolves_to_a_link_local_host() {
        // See the module doc: this port's WHATWG-compliant URL parser
        // resolves the backslash-confused authority to the *same*
        // dangerous host a real browser would (169.254.169.254), so the
        // ordinary blocked-address path catches it without needing the
        // source's separate netloc-backslash special case.
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(Arc::new(FakeComputer::new()), None, false).unwrap();
        let tools = toolset.get_tools(None).await;
        let navigate = tools.iter().find(|t| t.name() == "navigate").unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "url".to_string(),
            Value::String("http://169.254.169.254\\@example.com/".to_string()),
        );
        let result = navigate.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        assert!(fields.iter().any(|(k, _)| k == "error"));
    }

    #[rusty_tokio::test]
    async fn navigate_allows_a_private_address_when_configured_to() {
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(Arc::new(FakeComputer::new()), None, true).unwrap();
        let tools = toolset.get_tools(None).await;
        let navigate = tools.iter().find(|t| t.name() == "navigate").unwrap();
        let mut context = ctx();
        let mut args = BTreeMap::new();
        args.insert(
            "url".to_string(),
            Value::String("http://127.0.0.1/".to_string()),
        );
        let result = navigate.run_async(&args, &mut context).await.unwrap();
        let Value::Map(fields) = result else {
            panic!("expected a map");
        };
        assert!(!fields.iter().any(|(k, _)| k == "error"));
        assert!(fields.iter().any(|(k, _)| k == "image"));
    }

    #[rusty_tokio::test]
    async fn process_llm_request_adds_declarations_and_the_computer_use_marker() {
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(Arc::new(FakeComputer::new()), None, true).unwrap();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        toolset
            .process_llm_request(&mut context, &mut request)
            .await;

        let Some(Value::Seq(entries)) = &request.config.tools else {
            panic!("expected config.tools to be populated");
        };
        let has_function_declarations = entries.iter().any(|entry| match entry {
            Value::Map(fields) => fields.iter().any(|(k, _)| k == "functionDeclarations"),
            _ => false,
        });
        assert!(has_function_declarations);
        assert!(has_built_in_tool_marker(&request, COMPUTER_USE_TOOL_KEY));

        let marker = entries
            .iter()
            .find_map(|entry| match entry {
                Value::Map(fields) => fields
                    .iter()
                    .find(|(k, _)| k == COMPUTER_USE_TOOL_KEY)
                    .map(|(_, v)| v),
                _ => None,
            })
            .unwrap();
        let Value::Map(marker_fields) = marker else {
            panic!("expected the marker to be a map");
        };
        let environment = marker_fields
            .iter()
            .find(|(k, _)| k == "environment")
            .unwrap();
        assert_eq!(
            environment.1,
            Value::String("ENVIRONMENT_BROWSER".to_string())
        );
    }

    #[rusty_tokio::test]
    async fn process_llm_request_does_not_double_add_the_marker() {
        std::env::set_var("ADK_ENABLE_COMPUTER_USE", "1");
        let toolset = ComputerUseToolset::new(Arc::new(FakeComputer::new()), None, true).unwrap();
        let mut context = ctx();
        let mut request = LlmRequest::new("gemini-2.5-flash");
        toolset
            .process_llm_request(&mut context, &mut request)
            .await;
        toolset
            .process_llm_request(&mut context, &mut request)
            .await;

        let Some(Value::Seq(entries)) = &request.config.tools else {
            panic!("expected config.tools to be populated");
        };
        let marker_count = entries
            .iter()
            .filter(|entry| match entry {
                Value::Map(fields) => fields.iter().any(|(k, _)| k == COMPUTER_USE_TOOL_KEY),
                _ => false,
            })
            .count();
        assert_eq!(marker_count, 1);
    }
}
