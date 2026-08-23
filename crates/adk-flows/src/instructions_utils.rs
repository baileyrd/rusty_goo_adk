//! Part of capability C0170: `inject_session_state`, ported from
//! `google.adk.utils.instructions_utils` — the regex-based instruction
//! template engine `instructions.rs`'s `build_instructions` renders every
//! agent instruction string through.
//!
//! **Scope, disclosed**: only `_render_with_regex` (the default engine) is
//! ported. `_render_with_jinja2` is an explicit opt-in
//! (`use_jinja2=True`) needing the optional `jinja2` Python package; no
//! Jinja2-equivalent Rust crate decision has been made for this
//! workspace, and nothing in this port's own request processors ever
//! passes `use_jinja2=True`, so there is no observable capability gap
//! yet — only a documented one, the same "optional dependency, not
//! silently dropped" treatment already given to LiteLLM/Claude in
//! `adk-models`'s `registry.rs`.
//!
//! **Adaptation, disclosed**: `_is_valid_state_name` uses Python's
//! `str.isidentifier()` (full Unicode identifier rules). This port checks
//! ASCII `[A-Za-z_][A-Za-z0-9_]*` instead — narrower, but state variable
//! names in practice are always plain ASCII identifiers; a name using
//! non-ASCII identifier characters is the only case this diverges on.
//!
//! **Adaptation, disclosed**: `str(value)` (Python's generic stringification,
//! used to render a resolved state/artifact value into the template) is
//! approximated by [`value_to_display_string`] — exact for the scalar
//! cases instruction templates actually interpolate (strings verbatim,
//! Python's `True`/`False`/`None` spellings for bool/null, integers), but
//! Python's `float.__str__` (`1.0`, not `1`) and `dict`/`list.__str__`
//! (single-quoted Python literal syntax) aren't reproduced —
//! `rusty_serde::json::to_string`'s JSON formatting is used for those
//! instead, which is a reasonable, if not byte-identical, stand-in.

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

/// `_is_valid_state_name` — a bare identifier, or `<app|user|temp>:<identifier>`.
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

/// See the module doc's disclosed adaptation for what this doesn't
/// reproduce byte-for-byte against Python's `str()`.
pub fn value_to_display_string(value: &Value) -> String {
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

/// C0170 (part of the `instructions` request processor): populates state/
/// artifact values into an instruction template. See the module doc for
/// the Jinja2 scope decision.
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
    fn missing_required_variable_errors() {
        let ctx = ctx_with_state(vec![]);
        let err = inject_session_state("Hello {missing}!", &ctx).unwrap_err();
        assert!(
            matches!(err, InjectSessionStateError::VariableNotFound(name) if name == "missing")
        );
    }

    #[test]
    fn missing_optional_variable_becomes_empty_string() {
        let ctx = ctx_with_state(vec![]);
        assert_eq!(
            inject_session_state("Hello {missing?}!", &ctx).unwrap(),
            "Hello !"
        );
    }

    #[test]
    fn an_invalid_variable_name_is_left_untouched() {
        let ctx = ctx_with_state(vec![]);
        assert_eq!(
            inject_session_state("literal {not a var} text", &ctx).unwrap(),
            "literal {not a var} text"
        );
    }

    #[test]
    fn a_null_state_value_renders_as_an_empty_string() {
        let ctx = ctx_with_state(vec![("k", Value::Null)]);
        assert_eq!(inject_session_state("[{k}]", &ctx).unwrap(), "[]");
    }

    #[test]
    fn a_bool_state_value_renders_with_python_capitalization() {
        let ctx = ctx_with_state(vec![("flag", Value::Bool(true))]);
        assert_eq!(inject_session_state("{flag}", &ctx).unwrap(), "True");
    }

    #[test]
    fn app_and_user_and_temp_prefixed_names_are_valid() {
        let ctx = ctx_with_state(vec![
            ("app:setting", Value::String("x".to_string())),
            ("user:pref", Value::String("y".to_string())),
            ("temp:scratch", Value::String("z".to_string())),
        ]);
        assert_eq!(
            inject_session_state("{app:setting}-{user:pref}-{temp:scratch}", &ctx).unwrap(),
            "x-y-z"
        );
    }

    #[test]
    fn a_missing_artifact_service_errors() {
        let ctx = ctx_with_state(vec![]);
        let err = inject_session_state("{artifact.report.pdf}", &ctx).unwrap_err();
        assert!(matches!(err, InjectSessionStateError::NoArtifactService));
    }

    #[test]
    fn is_valid_state_name_rejects_a_bad_prefix() {
        assert!(!is_valid_state_name("weird:name"));
        assert!(is_valid_state_name("app:name"));
        assert!(is_valid_state_name("plain_name"));
        assert!(!is_valid_state_name("123bad"));
    }
}
