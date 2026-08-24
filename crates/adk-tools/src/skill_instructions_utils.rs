//! Part of capability C0401: a local duplicate of
//! `adk_flows::instructions_utils::inject_session_state` (C0170), needed
//! by `skill_toolset::LoadSkillTool` for `Frontmatter.metadata["adk_inject_state"]`
//! interpolation.
//!
//! **Why a duplicate, not a dependency**: `adk-flows` already depends on
//! `adk-tools` (for `BaseTool`/`ToolContext`), so `adk-tools` depending
//! back on `adk-flows` for this one function would be a crate-graph
//! cycle. This is the same "duplicate locally to avoid a cycle" pattern
//! already used by `adk-examples::example_util::value_to_display_string`
//! (duplicated from `adk-flows`) — this file duplicates the same
//! function, plus its private helpers, verbatim from
//! `adk-flows/src/instructions_utils.rs`. See that module's own doc for
//! the adaptations already disclosed there (ASCII-only identifier check,
//! `value_to_display_string`'s non-byte-identical `str()` approximation,
//! Jinja2 scope) — they apply here unchanged, since this is a literal
//! copy, not a re-derivation.

use std::sync::OnceLock;

use adk_agents::readonly_context::ReadonlyContext;
use regex::Regex;
use rusty_serde::value::Value;

const APP_PREFIX: &str = "app:";
const USER_PREFIX: &str = "user:";
const TEMP_PREFIX: &str = "temp:";

#[derive(Debug, rusty_err::Error)]
pub enum InjectSessionStateError {
    #[error("Artifact service is not initialized.")]
    NoArtifactService,
    #[error("Artifact {0} not found.")]
    ArtifactNotFound(String),
    #[error("Context variable not found: `{0}`.")]
    VariableNotFound(String),
}

fn template_var_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\{+[^{}]*\}+").expect("valid regex"))
}

fn is_ascii_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    !s.is_empty() && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn is_valid_state_name(var_name: &str) -> bool {
    match var_name.split_once(':') {
        None => is_ascii_identifier(var_name),
        Some((prefix, rest)) => {
            let prefix_with_colon = format!("{prefix}:");
            matches!(
                prefix_with_colon.as_str(),
                APP_PREFIX | USER_PREFIX | TEMP_PREFIX
            ) && is_ascii_identifier(rest)
        }
    }
}

fn value_to_display_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::UInt(u) => u.to_string(),
        Value::Float(_) | Value::Seq(_) | Value::Map(_) => {
            rusty_serde::json::to_string(value).unwrap_or_default()
        }
    }
}

fn replace_match(matched: &str, ctx: &ReadonlyContext) -> Result<String, InjectSessionStateError> {
    let inner = matched.trim_start_matches('{').trim_end_matches('}').trim();
    let (var_name, optional) = match inner.strip_suffix('?') {
        Some(stripped) => (stripped, true),
        None => (inner, false),
    };

    if let Some(artifact_name) = var_name.strip_prefix("artifact.") {
        let service = ctx
            .artifact_service()
            .ok_or(InjectSessionStateError::NoArtifactService)?;
        let session = ctx.session();
        let artifact = service.load_artifact(
            &session.app_name,
            &session.user_id,
            &session.id,
            artifact_name,
            None,
        );
        return match artifact {
            Some(value) => Ok(value_to_display_string(&value)),
            None if optional => Ok(String::new()),
            None => Err(InjectSessionStateError::ArtifactNotFound(
                artifact_name.to_string(),
            )),
        };
    }

    if !is_valid_state_name(var_name) {
        return Ok(matched.to_string());
    }

    match ctx.state().get(var_name) {
        Some(Value::Null) => Ok(String::new()),
        Some(value) => Ok(value_to_display_string(value)),
        None if optional => Ok(String::new()),
        None => Err(InjectSessionStateError::VariableNotFound(
            var_name.to_string(),
        )),
    }
}

/// See this module's own doc for why this is a duplicate of the C0170
/// port in `adk-flows`, not a shared dependency.
pub fn inject_session_state(
    template: &str,
    ctx: &ReadonlyContext,
) -> Result<String, InjectSessionStateError> {
    if !template.contains('{') {
        return Ok(template.to_string());
    }

    let mut result = String::new();
    let mut last_end = 0;
    for m in template_var_pattern().find_iter(template) {
        result.push_str(&template[last_end..m.start()]);
        result.push_str(&replace_match(m.as_str(), ctx)?);
        last_end = m.end();
    }
    result.push_str(&template[last_end..]);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx_with_state(state: Vec<(&str, Value)>) -> ReadonlyContext {
        let mut session = Session::new("app", "user", "s1");
        for (k, v) in state {
            session.state.insert(k.to_string(), v);
        }
        let ic = InvocationContextBuilder::new("inv-1", session).build();
        ReadonlyContext::new(ic)
    }

    #[test]
    fn a_template_without_braces_is_returned_unchanged() {
        let ctx = ctx_with_state(vec![]);
        assert_eq!(
            inject_session_state("no vars here", &ctx).unwrap(),
            "no vars here"
        );
    }

    #[test]
    fn substitutes_a_plain_state_variable() {
        let ctx = ctx_with_state(vec![("user_name", Value::String("Ada".to_string()))]);
        assert_eq!(
            inject_session_state("Hello {user_name}!", &ctx).unwrap(),
            "Hello Ada!"
        );
    }

    #[test]
    fn an_optional_missing_variable_becomes_empty() {
        let ctx = ctx_with_state(vec![]);
        assert_eq!(
            inject_session_state("Hello {missing?}!", &ctx).unwrap(),
            "Hello !"
        );
    }

    #[test]
    fn missing_required_variable_errors() {
        let ctx = ctx_with_state(vec![]);
        let err = inject_session_state("Hello {missing}!", &ctx).unwrap_err();
        assert!(
            matches!(err, InjectSessionStateError::VariableNotFound(name) if name == "missing")
        );
    }
}
