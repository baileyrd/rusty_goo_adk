//! Capability C0116: `LlmRequest.append_tools`, ported from
//! `google.adk.models.llm_request`.
//!
//! See this crate's own module doc for why this is a free function
//! (`append_tools`) rather than a real method on `LlmRequest` — the
//! crate-graph cycle `append_tools` would otherwise create.
//!
//! **Adaptation, disclosed**: `LlmRequest.config.tools` stays the
//! pre-existing opaque `Value` placeholder (`adk-models`'s own module doc)
//! rather than being narrowed to a real `Vec<Tool>` in this batch —
//! `append_tools` builds the correct camelCase JSON shape
//! (`[{"functionDeclarations": [...]}, ...]`) directly into it. Narrowing
//! `config.tools` to a real typed field is a natural follow-up once more
//! of Phase 8's tool ecosystem exists to justify the change (the same
//! "narrow when a capability needs it" pattern this migration already
//! used for `EventCompaction.compacted_content`).
//!
//! **Adaptation, disclosed**: the source's `tools_dict` (a `name ->
//! BaseTool` map used later to resolve which tool answers a function
//! call) isn't tracked here — it isn't serialized in the source either
//! (`exclude=True`), and nothing in this port yet consumes it (function-
//! call execution, C0191, needs `BaseTool` dispatch this batch doesn't
//! build). The source only *logs* a warning on a duplicate tool name,
//! rather than failing; this port returns the shadowed names instead of
//! logging them, the same "no logging framework adopted yet" adaptation
//! `functions_utils.rs`/`contents.rs` already disclose.

use std::collections::HashSet;

use adk_genai::content::FunctionDeclaration;
use adk_models::llm_request::LlmRequest;
use rusty_serde::value::Value;

use crate::base_tool::BaseTool;

const FUNCTION_DECLARATIONS_KEY: &str = "functionDeclarations";

/// `LlmRequest.append_tools`: appends each tool's declaration (if any) to
/// `llm_request.config.tools`, finding-or-creating the one `Tool` entry
/// that carries `functionDeclarations` (as opposed to a built-in-tool
/// marker entry like Google Search). Returns the names of any tools whose
/// declaration shadowed an earlier one with the same name in this same
/// call — both declarations are still advertised to the model, but only
/// the survivor's name maps back to a distinguishable tool.
pub fn append_tools(llm_request: &mut LlmRequest, tools: &[&dyn BaseTool]) -> Vec<String> {
    let entries = tools
        .iter()
        .filter_map(|tool| tool.get_declaration().map(|d| (tool.name().to_string(), d)));
    merge_declarations(llm_request, entries)
}

/// The object-free core of [`append_tools`]: merges `(name, declaration)`
/// pairs into `llm_request.config.tools`. Split out so that
/// [`crate::base_tool::BaseTool`]'s default `process_llm_request` method
/// can call it directly on `(self.name(), self.get_declaration())` without
/// ever needing to coerce `&Self` into `&dyn BaseTool` — a coercion that
/// isn't legal from inside a default trait method (`Self: Sized` would be
/// required, which conflicts with calling this method through a trait
/// object at all).
pub fn merge_declarations(
    llm_request: &mut LlmRequest,
    entries: impl IntoIterator<Item = (String, FunctionDeclaration)>,
) -> Vec<String> {
    let mut declarations = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut shadowed_names = Vec::new();
    for (name, declaration) in entries {
        declarations.push(rusty_serde::json::to_value(&declaration).unwrap_or(Value::Null));
        if !seen_names.insert(name.clone()) {
            shadowed_names.push(name);
        }
    }
    if declarations.is_empty() {
        return shadowed_names;
    }

    if !matches!(llm_request.config.tools, Some(Value::Seq(_))) {
        llm_request.config.tools = Some(Value::Seq(Vec::new()));
    }
    let Some(Value::Seq(entries)) = &mut llm_request.config.tools else {
        unreachable!("just ensured config.tools is Some(Value::Seq(_))");
    };

    let existing_declarations = entries.iter_mut().find_map(|entry| match entry {
        Value::Map(fields) => fields
            .iter_mut()
            .find(|(key, _)| key == FUNCTION_DECLARATIONS_KEY)
            .map(|(_, value)| value),
        _ => None,
    });

    match existing_declarations {
        Some(Value::Seq(existing)) => existing.extend(declarations),
        Some(other) => *other = Value::Seq(declarations),
        None => entries.push(Value::Map(vec![(
            FUNCTION_DECLARATIONS_KEY.to_string(),
            Value::Seq(declarations),
        )])),
    }

    shadowed_names
}

/// Appends a built-in-tool marker entry (e.g. `{"googleSearch": {}}`) to
/// `llm_request.config.tools` — the shape every built-in Gemini grounding
/// tool (`GoogleSearchTool`/`UrlContextTool`/`EnterpriseWebSearchTool`/
/// `GoogleMapsGroundingTool`, C0428/C0430-C0432) appends instead of a
/// `functionDeclarations` entry.
pub fn append_built_in_tool_marker(llm_request: &mut LlmRequest, key: &str) {
    if !matches!(llm_request.config.tools, Some(Value::Seq(_))) {
        llm_request.config.tools = Some(Value::Seq(Vec::new()));
    }
    let Some(Value::Seq(entries)) = &mut llm_request.config.tools else {
        unreachable!("just ensured config.tools is Some(Value::Seq(_))");
    };
    entries.push(Value::Map(vec![(key.to_string(), Value::Map(Vec::new()))]));
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::FunctionDeclaration;

    struct DeclaringTool {
        name: &'static str,
    }

    impl BaseTool for DeclaringTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "a test tool"
        }
        fn get_declaration(&self) -> Option<FunctionDeclaration> {
            Some(FunctionDeclaration {
                name: Some(self.name.to_string()),
                ..Default::default()
            })
        }
    }

    struct NonDeclaringTool;

    impl BaseTool for NonDeclaringTool {
        fn name(&self) -> &str {
            "non_declaring"
        }
        fn description(&self) -> &str {
            "no declaration"
        }
    }

    fn declarations_seq(request: &LlmRequest) -> &[Value] {
        let Some(Value::Seq(entries)) = &request.config.tools else {
            panic!("expected config.tools to be a Value::Seq");
        };
        for entry in entries {
            if let Value::Map(fields) = entry {
                if let Some((_, Value::Seq(declarations))) =
                    fields.iter().find(|(k, _)| k == "functionDeclarations")
                {
                    return declarations;
                }
            }
        }
        panic!("expected a functionDeclarations entry")
    }

    #[test]
    fn is_a_no_op_for_an_empty_tool_list() {
        let mut request = LlmRequest::default();
        let shadowed = append_tools(&mut request, &[]);
        assert!(shadowed.is_empty());
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn a_tool_with_no_declaration_is_skipped() {
        let mut request = LlmRequest::default();
        let tool = NonDeclaringTool;
        append_tools(&mut request, &[&tool]);
        assert!(request.config.tools.is_none());
    }

    #[test]
    fn creates_a_new_function_declarations_entry() {
        let mut request = LlmRequest::default();
        let tool = DeclaringTool { name: "tool_a" };
        append_tools(&mut request, &[&tool]);
        let declarations = declarations_seq(&request);
        assert_eq!(declarations.len(), 1);
    }

    #[test]
    fn appends_to_an_existing_function_declarations_entry_across_calls() {
        let mut request = LlmRequest::default();
        let tool_a = DeclaringTool { name: "tool_a" };
        let tool_b = DeclaringTool { name: "tool_b" };
        append_tools(&mut request, &[&tool_a]);
        append_tools(&mut request, &[&tool_b]);

        let Some(Value::Seq(entries)) = &request.config.tools else {
            panic!("expected Value::Seq");
        };
        assert_eq!(
            entries.len(),
            1,
            "a second call should extend the existing entry, not create a second one"
        );
        assert_eq!(declarations_seq(&request).len(), 2);
    }

    #[test]
    fn reports_a_shadowed_duplicate_name_within_one_call() {
        let mut request = LlmRequest::default();
        let tool_a = DeclaringTool { name: "dup" };
        let tool_b = DeclaringTool { name: "dup" };
        let shadowed = append_tools(&mut request, &[&tool_a, &tool_b]);
        assert_eq!(shadowed, vec!["dup".to_string()]);
        // Both declarations are still advertised to the model.
        assert_eq!(declarations_seq(&request).len(), 2);
    }
}
