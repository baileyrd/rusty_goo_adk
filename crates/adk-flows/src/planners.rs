//! Capabilities C0200-C0203: `planners.py` (`BasePlanner`, `BuiltInPlanner`,
//! `PlanReActPlanner`), ported from `google.adk.planners`.
//!
//! `BasePlanner::process_planning_response` takes a `&mut CallbackContext`
//! parameter for interface parity with the source, but neither
//! `BuiltInPlanner` nor `PlanReActPlanner` actually reads or mutates it —
//! matching the source, where the parameter exists on the abstract method
//! for future planner implementations, not because either concrete
//! implementation needs it today.
//!
//! Wired into the `_nl_planning` request/response processors (C0176/C0179)
//! — see `nl_planning.rs`. The blocker this module doc previously
//! claimed ("needs `InvocationContext.agent` to resolve a concrete
//! `LlmAgent`") was stale: `LlmFlow` already owns its `LlmAgent` and
//! resolved model directly (the same "caller supplies the resolved bits"
//! shape already established for `tools`/`tools_dict`, C0151), so
//! `nl_planning.rs`'s free functions just take `Option<&dyn BasePlanner>`
//! as a plain parameter — the same pattern every sibling processor in
//! `llm_flow.rs` already uses.

use adk_agents::context::CallbackContext;
use adk_agents::readonly_context::ReadonlyContext;
use adk_genai::content::Part;
use adk_models::base_llm::AsAny;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

/// C0200: `BasePlanner(ABC)`. `AsAny + Send + Sync` (widened for this
/// batch): `Send + Sync` so `Arc<dyn BasePlanner>` is usable from async
/// code (matching `BaseTool`'s own bound); `AsAny` so `nl_planning.rs`
/// can downcast to the concrete `BuiltInPlanner` — see that module's doc
/// for why (the source's "is this method literally
/// `BuiltInPlanner.process_planning_response`, unbound" identity check
/// has no Rust equivalent without it). Additive: this trait's only two
/// impls live in this same file, and it has no consumer anywhere else in
/// the workspace yet (verified — `llm_agent.rs`'s own `planner` field is
/// still an opaque, unread `Value` placeholder), so widening its bounds
/// breaks no shipped call site.
pub trait BasePlanner: AsAny + Send + Sync {
    fn build_planning_instruction(
        &self,
        readonly_context: &ReadonlyContext,
        llm_request: &LlmRequest,
    ) -> Option<String>;

    fn process_planning_response(
        &self,
        callback_context: &mut CallbackContext,
        response_parts: Vec<Part>,
    ) -> Option<Vec<Part>>;
}

/// C0201: `BuiltInPlanner` — wraps a model's native thinking features. Both
/// hooks are no-ops (the model handles thinking itself); `thinking_config`
/// is applied to the request separately via `apply_thinking_config`, not
/// through `build_planning_instruction`.
///
/// `thinking_config` is an opaque [`Value`] placeholder, matching
/// `LlmRequest.config.thinking_config`'s own opaque-`Value` field — no
/// typed `ThinkingConfig` exists in this port.
pub struct BuiltInPlanner {
    pub thinking_config: Value,
}

impl BuiltInPlanner {
    pub fn new(thinking_config: Value) -> Self {
        Self { thinking_config }
    }

    /// C0201: `apply_thinking_config` — sets `llm_request.config.thinking_config`.
    pub fn apply_thinking_config(&self, llm_request: &mut LlmRequest) {
        llm_request.config.thinking_config = Some(self.thinking_config.clone());
    }
}

impl BasePlanner for BuiltInPlanner {
    fn build_planning_instruction(
        &self,
        _readonly_context: &ReadonlyContext,
        _llm_request: &LlmRequest,
    ) -> Option<String> {
        None
    }

    fn process_planning_response(
        &self,
        _callback_context: &mut CallbackContext,
        _response_parts: Vec<Part>,
    ) -> Option<Vec<Part>> {
        None
    }
}

const PLANNING_TAG: &str = "/*PLANNING*/";
const REPLANNING_TAG: &str = "/*REPLANNING*/";
const REASONING_TAG: &str = "/*REASONING*/";
const ACTION_TAG: &str = "/*ACTION*/";
const FINAL_ANSWER_TAG: &str = "/*FINAL_ANSWER*/";

const PLANNING_TAGS: [&str; 4] = [PLANNING_TAG, REASONING_TAG, ACTION_TAG, REPLANNING_TAG];

/// C0202/C0203: `PlanReActPlanner` — model-agnostic prompted Plan-Re-Act.
/// Requires no config; injects a 5-tag NL instruction and splits the
/// model's tagged response back into thought/answer/tool-call parts.
pub struct PlanReActPlanner;

impl PlanReActPlanner {
    fn split_by_last_pattern(text: &str, separator: &str) -> (String, String) {
        match text.rfind(separator) {
            None => (text.to_string(), String::new()),
            Some(index) => (
                text[..index + separator.len()].to_string(),
                text[index + separator.len()..].to_string(),
            ),
        }
    }

    fn strip_planning_tags(text: &str) -> String {
        let mut stripped = text.to_string();
        for tag in PLANNING_TAGS {
            stripped = stripped.replace(tag, "");
        }
        stripped
    }

    fn mark_as_thought(part: &mut Part) {
        if part.text.is_some() {
            part.thought = Some(true);
        }
    }

    fn handle_non_function_call_part(mut part: Part, preserved_parts: &mut Vec<Part>) {
        if let Some(text) = part.text.clone().filter(|t| t.contains(FINAL_ANSWER_TAG)) {
            let (mut reasoning_text, final_answer_text) =
                Self::split_by_last_pattern(&text, FINAL_ANSWER_TAG);
            if let Some(stripped) = reasoning_text.strip_suffix(FINAL_ANSWER_TAG) {
                reasoning_text = stripped.to_string();
            }
            let reasoning_text = Self::strip_planning_tags(&reasoning_text);
            if !reasoning_text.is_empty() {
                let mut reasoning_part = Part {
                    text: Some(reasoning_text),
                    ..Default::default()
                };
                Self::mark_as_thought(&mut reasoning_part);
                preserved_parts.push(reasoning_part);
            }
            if !final_answer_text.is_empty() {
                preserved_parts.push(Part {
                    text: Some(final_answer_text),
                    ..Default::default()
                });
            }
            return;
        }

        let response_text = part.text.clone().unwrap_or_default();
        if !response_text.is_empty()
            && PLANNING_TAGS
                .iter()
                .any(|tag| response_text.starts_with(tag))
        {
            part.text = Some(Self::strip_planning_tags(&response_text));
            Self::mark_as_thought(&mut part);
        }
        preserved_parts.push(part);
    }

    fn build_nl_planner_instruction() -> String {
        let high_level_preamble = format!(
            "\nWhen answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.\n\nFollow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.\n\nFollow this format when answering the question: (1) The planning part should be under {PLANNING_TAG}. (2) The tool code snippets should be under {ACTION_TAG}, and the reasoning parts should be under {REASONING_TAG}. (3) The final answer part should be under {FINAL_ANSWER_TAG}.\n"
        );

        let planning_preamble = format!(
            "\nBelow are the requirements for the planning:\nThe plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.\nIf the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be under {REPLANNING_TAG}. Then use tools to follow the new plan.\n"
        );

        let reasoning_preamble = "\nBelow are the requirements for the reasoning:\nThe reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.\n";

        let final_answer_preamble = "\nBelow are the requirements for the final answer:\nThe final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.\n";

        let tool_code_without_python_libraries_preamble = "\nBelow are the requirements for the tool code:\n\n**Custom Tools:** The available tools are described in the context and can be directly used.\n- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.\n- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.\n- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.\n- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().\n- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.\n";

        let user_input_preamble = "\nVERY IMPORTANT instruction that you MUST follow in addition to the above instructions:\n\nYou should ask for clarification if you need more information to answer the question.\nYou should prefer using the information available in the context instead of repeated tool use.\n";

        [
            high_level_preamble.as_str(),
            planning_preamble.as_str(),
            reasoning_preamble,
            final_answer_preamble,
            tool_code_without_python_libraries_preamble,
            user_input_preamble,
        ]
        .join("\n\n")
    }
}

impl BasePlanner for PlanReActPlanner {
    fn build_planning_instruction(
        &self,
        _readonly_context: &ReadonlyContext,
        _llm_request: &LlmRequest,
    ) -> Option<String> {
        Some(Self::build_nl_planner_instruction())
    }

    fn process_planning_response(
        &self,
        _callback_context: &mut CallbackContext,
        response_parts: Vec<Part>,
    ) -> Option<Vec<Part>> {
        if response_parts.is_empty() {
            return None;
        }

        let n = response_parts.len();
        let mut preserved_parts: Vec<Part> = Vec::new();
        let mut first_fc_index: Option<usize> = None;

        let mut i = 0;
        while i < n {
            let function_call = response_parts[i].function_call.as_ref();
            if let Some(function_call) = function_call {
                let name_is_empty = function_call
                    .name
                    .as_deref()
                    .map(str::is_empty)
                    .unwrap_or(true);
                if name_is_empty {
                    i += 1;
                    continue;
                }
                preserved_parts.push(response_parts[i].clone());
                first_fc_index = Some(i);
                break;
            }
            Self::handle_non_function_call_part(response_parts[i].clone(), &mut preserved_parts);
            i += 1;
        }

        if let Some(first_fc_index) = first_fc_index {
            let mut j = first_fc_index + 1;
            while j < n && response_parts[j].function_call.is_some() {
                preserved_parts.push(response_parts[j].clone());
                j += 1;
            }
        }

        Some(preserved_parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;
    use adk_genai::content::FunctionCall;

    fn callback_context() -> CallbackContext {
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        CallbackContext::new(ic)
    }

    fn function_call_part(name: &str) -> Part {
        Part {
            function_call: Some(FunctionCall {
                partial_args: None,
                id: None,
                name: Some(name.to_string()),
                args: Some(std::collections::BTreeMap::new()),
                will_continue: None,
            }),
            ..Default::default()
        }
    }

    fn function_call_names(parts: &[Part]) -> Vec<&str> {
        parts
            .iter()
            .filter_map(|p| p.function_call.as_ref())
            .filter_map(|fc| fc.name.as_deref())
            .collect()
    }

    #[test]
    fn built_in_planner_hooks_are_both_no_ops() {
        let planner = BuiltInPlanner::new(Value::String("thinking".to_string()));
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let readonly = ReadonlyContext::new(ic);
        let request = LlmRequest::new("gemini-3-pro");
        assert!(planner
            .build_planning_instruction(&readonly, &request)
            .is_none());

        let mut ctx = callback_context();
        assert!(planner
            .process_planning_response(&mut ctx, vec![Part::default()])
            .is_none());
    }

    #[test]
    fn built_in_planner_apply_thinking_config_sets_the_request_field() {
        let planner = BuiltInPlanner::new(Value::String("budget-128".to_string()));
        let mut request = LlmRequest::new("gemini-3-pro");
        planner.apply_thinking_config(&mut request);
        assert_eq!(
            request.config.thinking_config,
            Some(Value::String("budget-128".to_string()))
        );
    }

    #[test]
    fn plan_re_act_build_planning_instruction_contains_all_five_tags() {
        let planner = PlanReActPlanner;
        let ic = InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build();
        let readonly = ReadonlyContext::new(ic);
        let request = LlmRequest::new("gemini-3-pro");
        let instruction = planner
            .build_planning_instruction(&readonly, &request)
            .unwrap();
        for tag in [
            PLANNING_TAG,
            REPLANNING_TAG,
            REASONING_TAG,
            ACTION_TAG,
            FINAL_ANSWER_TAG,
        ] {
            assert!(instruction.contains(tag), "missing {tag}");
        }
    }

    #[test]
    fn strips_planning_tag_from_thought_part_and_preserves_signature() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![
            Part {
                text: Some("/*PLANNING*/Step 1: look it up.".to_string()),
                thought_signature: Some(Value::String("sig1".to_string())),
                ..Default::default()
            },
            Part {
                text: Some("/*REASONING*/I need to call the tool.".to_string()),
                thought_signature: Some(Value::String("sig2".to_string())),
                ..Default::default()
            },
            function_call_part("lookup"),
        ];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        let text_parts: Vec<&Part> = result.iter().filter(|p| p.text.is_some()).collect();
        for part in &text_parts {
            let text = part.text.as_deref().unwrap();
            assert!(!text.contains(PLANNING_TAG));
            assert!(!text.contains(REASONING_TAG));
            assert_eq!(part.thought, Some(true));
        }
        assert_eq!(
            text_parts[0].thought_signature,
            Some(Value::String("sig1".to_string()))
        );
        assert_eq!(
            text_parts[1].thought_signature,
            Some(Value::String("sig2".to_string()))
        );
        assert_eq!(function_call_names(&result), vec!["lookup"]);
    }

    #[test]
    fn strips_final_answer_tag_boundary() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![Part {
            text: Some("/*REASONING*/Some reasoning./*FINAL_ANSWER*/The answer is 42.".to_string()),
            ..Default::default()
        }];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        let combined = result
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!combined.contains(FINAL_ANSWER_TAG));
        assert!(!combined.contains(REASONING_TAG));
        assert!(combined.contains("The answer is 42."));
    }

    #[test]
    fn strips_multiple_planning_tags() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![Part {
            text: Some(
                "/*PLANNING*/Initial plan.\n/*REASONING*/Some reasoning.\n/*FINAL_ANSWER*/The answer is 42."
                    .to_string(),
            ),
            ..Default::default()
        }];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        let combined = result
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!combined.contains(PLANNING_TAG));
        assert!(!combined.contains(REASONING_TAG));
        assert!(!combined.contains(FINAL_ANSWER_TAG));
        assert!(combined.contains("Initial plan."));
        assert!(combined.contains("Some reasoning."));
        assert!(combined.contains("The answer is 42."));
    }

    #[test]
    fn part_without_leading_tag_is_not_marked_as_thought() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![Part {
            text: Some("Here is the answer /*PLANNING*/ with stray tag.".to_string()),
            ..Default::default()
        }];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_ne!(result[0].thought, Some(true));
        assert_eq!(
            result[0].text.as_deref(),
            Some("Here is the answer /*PLANNING*/ with stray tag.")
        );
    }

    #[test]
    fn bare_tag_part_is_marked_as_thought() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![
            Part {
                text: Some(ACTION_TAG.to_string()),
                ..Default::default()
            },
            function_call_part("lookup"),
        ];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text.as_deref(), Some(""));
        assert_eq!(result[0].thought, Some(true));
        assert_eq!(function_call_names(&result), vec!["lookup"]);
    }

    #[test]
    fn sole_bare_tag_part_is_marked_as_thought() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![Part {
            text: Some(ACTION_TAG.to_string()),
            ..Default::default()
        }];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text.as_deref(), Some(""));
        assert_eq!(result[0].thought, Some(true));
    }

    #[test]
    fn preserves_all_leading_parallel_function_calls() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![
            function_call_part("get_weather"),
            function_call_part("get_time"),
        ];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(
            function_call_names(&result),
            vec!["get_weather", "get_time"]
        );
    }

    #[test]
    fn preserves_parallel_function_calls_after_leading_text() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![
            Part {
                text: Some("Let me look that up.".to_string()),
                ..Default::default()
            },
            function_call_part("get_weather"),
            function_call_part("get_time"),
        ];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(
            function_call_names(&result),
            vec!["get_weather", "get_time"]
        );
    }

    #[test]
    fn process_planning_response_returns_none_for_an_empty_list() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        assert!(planner
            .process_planning_response(&mut ctx, vec![])
            .is_none());
    }

    #[test]
    fn a_function_call_with_an_empty_name_is_skipped_not_stopped_on() {
        let planner = PlanReActPlanner;
        let mut ctx = callback_context();
        let response_parts = vec![
            Part {
                function_call: Some(FunctionCall {
                    partial_args: None,
                    id: None,
                    name: Some(String::new()),
                    args: None,
                    will_continue: None,
                }),
                ..Default::default()
            },
            function_call_part("lookup"),
        ];

        let result = planner
            .process_planning_response(&mut ctx, response_parts)
            .unwrap();

        assert_eq!(function_call_names(&result), vec!["lookup"]);
    }
}
