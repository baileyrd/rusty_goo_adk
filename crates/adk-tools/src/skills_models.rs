//! C0393/C0394: `skills.models` — data models for Agent Skills.
//!
//! Three layers, matching the source's own L1/L2/L3 naming:
//! - L1 ([`Frontmatter`]): metadata parsed from a `SKILL.md` YAML
//!   frontmatter block, used for skill discovery.
//! - L2: the markdown instruction body (a plain `String`, held directly on
//!   [`Skill::instructions`] — no dedicated type in the source either).
//! - L3 ([`Resources`]): additional reference/asset/script content, loaded
//!   on demand.
//!
//! **Adaptation**: the source's `@field_validator`s run automatically on
//! every construction, including deserialization (pydantic). This port
//! keeps [`Frontmatter`]'s fields plainly `pub`/deserializable (the same
//! "plain fields + explicit `validate()`" pattern already established for
//! `adk_eval::eval_case::EvalCase`) and exposes the checks as
//! [`Frontmatter::validate`] instead — deserializing an invalid payload
//! succeeds structurally; call `validate()` to enforce what the source
//! enforces automatically. `Frontmatter::normalize_name` is split out
//! separately (rather than folded into `validate`) because the source's
//! own validator *mutates* `name` (NFKC-normalizes it, then validates the
//! normalized form) — a Rust validator can't rewrite the struct it's
//! borrowing, so callers normalize first (typically right after
//! deserializing, before further use) and then validate.
//!
//! **Disclosed narrowing**: the source's `model_config = ConfigDict(extra
//! ="allow")` keeps any unrecognized inbound frontmatter field accessible
//! on the model (e.g. a client-specific YAML key this port doesn't know
//! about); this port has no `deny_unknown_fields` (so an unrecognized
//! field no longer rejects the payload, matching "allow" rather than
//! "forbid") but, unlike pydantic, doesn't capture the extra field
//! anywhere — it's silently dropped rather than preserved. Same narrowing
//! already disclosed for `adk_eval::eval_case::SessionInput`.

use std::collections::HashMap;
use std::sync::OnceLock;

use adk_features::feature_registry::{is_feature_enabled, FeatureName};
use regex::Regex;
use rusty_serde::value::Value;
use unicode_normalization::UnicodeNormalization;

fn kebab_name_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-z0-9]+(-[a-z0-9]+)*$").unwrap())
}

fn snake_or_kebab_name_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([a-z0-9]+(-[a-z0-9]+)*|[a-z0-9]+(_[a-z0-9]+)*)$").unwrap())
}

/// C0393: `skills.models.Frontmatter` — L1 skill content: metadata parsed
/// from `SKILL.md` for skill discovery.
#[derive(Debug, Clone, Default, PartialEq, rusty_serde::Serialize, rusty_serde::Deserialize)]
pub struct Frontmatter {
    pub name: String,
    pub description: String,
    #[rusty_serde(default)]
    pub license: Option<String>,
    #[rusty_serde(default)]
    pub compatibility: Option<String>,
    /// Accepts either wire name on the way in (`allowed-tools` or
    /// `allowed_tools`, matching the source's `populate_by_name=True`);
    /// serializes back out as `allowed-tools` (the source's
    /// `serialization_alias`).
    #[rusty_serde(rename = "allowed-tools", alias = "allowed_tools", default)]
    pub allowed_tools: Option<String>,
    #[rusty_serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// A [`Frontmatter::validate`] failure, mirroring the source's
/// `ValueError` messages field-for-field so a caller surfacing this to a
/// skill author sees the same guidance the source gives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontmatterValidationError(pub String);

impl std::fmt::Display for FrontmatterValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FrontmatterValidationError {}

fn err(message: impl Into<String>) -> FrontmatterValidationError {
    FrontmatterValidationError(message.into())
}

impl Frontmatter {
    /// `_validate_name`'s NFKC-normalization step, split out from
    /// [`Self::validate`] per this module's top-level doc — call this
    /// first (typically right after deserializing) so `self.name` holds
    /// the normalized form before validating or using it.
    pub fn normalize_name(&mut self) {
        self.name = self.name.nfkc().collect();
    }

    /// Runs every `@field_validator` the source declares, in the source's
    /// own declaration order (`metadata`, `name`, `description`,
    /// `compatibility`). Assumes [`Self::normalize_name`] has already been
    /// called — this does not normalize `name` itself, only checks its
    /// length and pattern against whatever `self.name` currently holds.
    pub fn validate(&self) -> Result<(), FrontmatterValidationError> {
        if let Some(tools) = self.metadata.get("adk_additional_tools") {
            if !matches!(tools, Value::Seq(_)) {
                return Err(err("adk_additional_tools must be a list of strings"));
            }
        }
        if let Some(inject_state) = self.metadata.get("adk_inject_state") {
            if !matches!(inject_state, Value::Bool(_)) {
                return Err(err("adk_inject_state must be a bool"));
            }
        }

        if self.name.chars().count() > 64 {
            return Err(err("name must be at most 64 characters"));
        }
        let (pattern, message) = if is_feature_enabled(FeatureName::SnakeCaseSkillName) {
            (
                snake_or_kebab_name_pattern(),
                "name must be lowercase kebab-case (a-z, 0-9, hyphens) or snake_case (a-z, \
                 0-9, underscores), with no leading, trailing, or consecutive delimiters. \
                 Mixing hyphens and underscores is not allowed.",
            )
        } else {
            (
                kebab_name_pattern(),
                "name must be lowercase kebab-case (a-z, 0-9, hyphens), with no leading, \
                 trailing, or consecutive delimiters",
            )
        };
        if !pattern.is_match(&self.name) {
            return Err(err(message));
        }

        if self.description.is_empty() {
            return Err(err("description must not be empty"));
        }
        let description_len = self.description.chars().count();
        if description_len > 1024 {
            return Err(err(format!(
                "description must be at most 1024 characters. Description length: \
                 {description_len}"
            )));
        }

        if let Some(compatibility) = &self.compatibility {
            if compatibility.chars().count() > 500 {
                return Err(err("compatibility must be at most 500 characters"));
            }
        }

        Ok(())
    }
}

/// C0394: `skills.models.Script` — wrapper for script content.
#[derive(Debug, Clone, Default, PartialEq, rusty_serde::Serialize, rusty_serde::Deserialize)]
pub struct Script {
    pub src: String,
}

impl std::fmt::Display for Script {
    /// `Script.__str__` — returns the script content directly, so any
    /// script type can be interpolated into a prompt or written to disk
    /// without a separate accessor.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.src)
    }
}

/// `dict[str, str | bytes]`'s value union, now that `LoadSkillResourceTool`
/// (C0409, `adk-tools::skill_toolset`) is a real consumer needing the
/// binary branch — widened from the placeholder `String`-only shape this
/// module's own doc previously disclosed, the same "widen a placeholder
/// once a real consumer needs the structure" pattern already used
/// elsewhere in this port (e.g. `adk_eval::evaluator::PerInvocationResult
/// ::rubric_scores`).
#[derive(Debug, Clone, PartialEq, rusty_serde::Serialize, rusty_serde::Deserialize)]
#[rusty_serde(untagged)]
pub enum ResourceContent {
    Text(String),
    Bytes(Vec<u8>),
}

/// C0394: `skills.models.Resources` — L3 skill content: additional
/// instructions, assets, and scripts.
#[derive(Debug, Clone, Default, PartialEq, rusty_serde::Serialize, rusty_serde::Deserialize)]
pub struct Resources {
    #[rusty_serde(default)]
    pub references: HashMap<String, ResourceContent>,
    #[rusty_serde(default)]
    pub assets: HashMap<String, ResourceContent>,
    #[rusty_serde(default)]
    pub scripts: HashMap<String, Script>,
}

impl Resources {
    /// `Resources.get_reference`.
    pub fn get_reference(&self, reference_id: &str) -> Option<&ResourceContent> {
        self.references.get(reference_id)
    }

    /// `Resources.get_asset`.
    pub fn get_asset(&self, asset_id: &str) -> Option<&ResourceContent> {
        self.assets.get(asset_id)
    }

    /// `Resources.get_script`.
    pub fn get_script(&self, script_id: &str) -> Option<&Script> {
        self.scripts.get(script_id)
    }

    /// `Resources.list_references`.
    pub fn list_references(&self) -> Vec<&String> {
        self.references.keys().collect()
    }

    /// `Resources.list_assets`.
    pub fn list_assets(&self) -> Vec<&String> {
        self.assets.keys().collect()
    }

    /// `Resources.list_scripts`.
    pub fn list_scripts(&self) -> Vec<&String> {
        self.scripts.keys().collect()
    }
}

/// C0394: `skills.models.Skill` — complete skill representation combining
/// frontmatter (L1), instructions (L2), and resources (L3).
#[derive(Debug, Clone, Default, PartialEq, rusty_serde::Serialize, rusty_serde::Deserialize)]
pub struct Skill {
    pub frontmatter: Frontmatter,
    pub instructions: String,
    #[rusty_serde(default)]
    pub resources: Resources,
    /// `Skill._uri` — location the skill was loaded from, used for
    /// telemetry. Should be RFC-3986-compliant. Private in the source
    /// (leading underscore, a pydantic `PrivateAttr` in all but name —
    /// excluded from the model's own field set, serialization, and
    /// validation); this port keeps it `pub(crate)` for the same reason:
    /// it's set by the loader that constructs a `Skill`, not by
    /// deserializing skill content itself.
    #[rusty_serde(skip)]
    pub(crate) uri: Option<String>,
}

impl Skill {
    /// `Skill.name` — convenience property to access the skill name.
    pub fn name(&self) -> &str {
        &self.frontmatter.name
    }

    /// `Skill.description` — convenience property to access the skill
    /// description.
    pub fn description(&self) -> &str {
        &self.frontmatter.description
    }

    /// `Skill._uri`'s only source-side write path is direct attribute
    /// assignment after construction (the loader sets it once a skill has
    /// been read from disk); this is that path's Rust equivalent.
    pub fn set_uri(&mut self, uri: Option<String>) {
        self.uri = uri;
    }

    /// `Skill._uri`'s read path.
    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_frontmatter() -> Frontmatter {
        Frontmatter {
            name: "my-skill".to_string(),
            description: "Does a thing.".to_string(),
            license: None,
            compatibility: None,
            allowed_tools: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn accepts_a_valid_kebab_case_name() {
        assert!(valid_frontmatter().validate().is_ok());
    }

    #[test]
    fn rejects_a_name_over_64_characters() {
        let mut fm = valid_frontmatter();
        fm.name = "a".repeat(65);
        let err = fm.validate().unwrap_err();
        assert_eq!(err.0, "name must be at most 64 characters");
    }

    #[test]
    fn rejects_snake_case_name_when_the_feature_is_disabled() {
        // SNAKE_CASE_SKILL_NAME defaults to off (Experimental, default_on:
        // false) -- only kebab-case is accepted unless a caller opts in.
        let mut fm = valid_frontmatter();
        fm.name = "my_skill".to_string();
        let err = fm.validate().unwrap_err();
        assert!(err.0.contains("kebab-case"));
    }

    #[test]
    fn accepts_snake_case_name_once_the_feature_is_enabled() {
        let _override = adk_features::feature_registry::TemporaryFeatureOverride::new(
            FeatureName::SnakeCaseSkillName,
            true,
        );
        let mut fm = valid_frontmatter();
        fm.name = "my_skill".to_string();
        assert!(fm.validate().is_ok());

        // Still rejects mixed hyphen/underscore delimiters.
        fm.name = "my-skill_name".to_string();
        assert!(fm.validate().is_err());
    }

    #[test]
    fn rejects_uppercase_or_leading_delimiter_names() {
        for bad in ["My-Skill", "-my-skill", "my-skill-", "my--skill", ""] {
            let mut fm = valid_frontmatter();
            fm.name = bad.to_string();
            assert!(fm.validate().is_err(), "expected {bad:?} to be rejected");
        }
    }

    #[test]
    fn nfkc_normalizes_the_name_before_validation() {
        // U+FF0D (fullwidth hyphen-minus) NFKC-normalizes to U+002D ('-').
        let mut fm = valid_frontmatter();
        fm.name = "my\u{FF0D}skill".to_string();
        fm.normalize_name();
        assert_eq!(fm.name, "my-skill");
        assert!(fm.validate().is_ok());
    }

    #[test]
    fn rejects_an_empty_description() {
        let mut fm = valid_frontmatter();
        fm.description = String::new();
        let err = fm.validate().unwrap_err();
        assert_eq!(err.0, "description must not be empty");
    }

    #[test]
    fn rejects_a_description_over_1024_characters() {
        let mut fm = valid_frontmatter();
        fm.description = "a".repeat(1025);
        let err = fm.validate().unwrap_err();
        assert_eq!(
            err.0,
            "description must be at most 1024 characters. Description length: 1025"
        );
    }

    #[test]
    fn rejects_a_compatibility_string_over_500_characters() {
        let mut fm = valid_frontmatter();
        fm.compatibility = Some("a".repeat(501));
        let err = fm.validate().unwrap_err();
        assert_eq!(err.0, "compatibility must be at most 500 characters");
    }

    #[test]
    fn rejects_a_non_list_adk_additional_tools() {
        let mut fm = valid_frontmatter();
        fm.metadata.insert(
            "adk_additional_tools".to_string(),
            Value::String("nope".to_string()),
        );
        let err = fm.validate().unwrap_err();
        assert_eq!(err.0, "adk_additional_tools must be a list of strings");
    }

    #[test]
    fn accepts_a_list_adk_additional_tools() {
        let mut fm = valid_frontmatter();
        fm.metadata.insert(
            "adk_additional_tools".to_string(),
            Value::Seq(vec![Value::String("tool_a".to_string())]),
        );
        assert!(fm.validate().is_ok());
    }

    #[test]
    fn rejects_a_non_bool_adk_inject_state() {
        let mut fm = valid_frontmatter();
        fm.metadata.insert(
            "adk_inject_state".to_string(),
            Value::String("true".to_string()),
        );
        let err = fm.validate().unwrap_err();
        assert_eq!(err.0, "adk_inject_state must be a bool");
    }

    #[test]
    fn accepts_a_bool_adk_inject_state() {
        let mut fm = valid_frontmatter();
        fm.metadata
            .insert("adk_inject_state".to_string(), Value::Bool(true));
        assert!(fm.validate().is_ok());
    }

    #[test]
    fn allowed_tools_deserializes_from_either_wire_name() {
        let via_alias: Frontmatter =
            rusty_serde::json::from_str(r#"{"name":"x","description":"d","allowed_tools":"a b"}"#)
                .unwrap();
        assert_eq!(via_alias.allowed_tools, Some("a b".to_string()));

        let via_primary: Frontmatter =
            rusty_serde::json::from_str(r#"{"name":"x","description":"d","allowed-tools":"a b"}"#)
                .unwrap();
        assert_eq!(via_primary.allowed_tools, Some("a b".to_string()));
    }

    #[test]
    fn allowed_tools_serializes_as_the_hyphenated_wire_name() {
        let mut fm = valid_frontmatter();
        fm.allowed_tools = Some("a b".to_string());
        let json = rusty_serde::json::to_string(&fm).unwrap();
        assert!(json.contains("\"allowed-tools\":\"a b\""));
        assert!(!json.contains("allowed_tools"));
    }

    #[test]
    fn resource_content_untagged_round_trips_both_variants() {
        let text = ResourceContent::Text("hello".to_string());
        let json = rusty_serde::json::to_string(&text).unwrap();
        assert_eq!(json, "\"hello\"");
        assert_eq!(
            rusty_serde::json::from_str::<ResourceContent>(&json).unwrap(),
            text
        );

        let bytes = ResourceContent::Bytes(vec![1, 2, 3]);
        let json = rusty_serde::json::to_string(&bytes).unwrap();
        assert_eq!(
            rusty_serde::json::from_str::<ResourceContent>(&json).unwrap(),
            bytes
        );
    }

    #[test]
    fn script_display_returns_its_source() {
        let script = Script {
            src: "echo hi".to_string(),
        };
        assert_eq!(script.to_string(), "echo hi");
    }

    #[test]
    fn resources_accessors_look_up_by_id() {
        let mut resources = Resources::default();
        resources.references.insert(
            "ref.md".to_string(),
            ResourceContent::Text("content".to_string()),
        );
        resources.assets.insert(
            "schema.json".to_string(),
            ResourceContent::Text("{}".to_string()),
        );
        resources.scripts.insert(
            "run.sh".to_string(),
            Script {
                src: "echo hi".to_string(),
            },
        );

        assert_eq!(
            resources.get_reference("ref.md"),
            Some(&ResourceContent::Text("content".to_string()))
        );
        assert_eq!(resources.get_reference("missing"), None);
        assert_eq!(
            resources.get_asset("schema.json"),
            Some(&ResourceContent::Text("{}".to_string()))
        );
        assert_eq!(resources.get_script("run.sh").unwrap().src, "echo hi");
        assert_eq!(resources.list_references(), vec![&"ref.md".to_string()]);
        assert_eq!(resources.list_assets(), vec![&"schema.json".to_string()]);
        assert_eq!(resources.list_scripts(), vec![&"run.sh".to_string()]);
    }

    #[test]
    fn skill_name_and_description_delegate_to_frontmatter() {
        let skill = Skill {
            frontmatter: valid_frontmatter(),
            instructions: "Do the thing.".to_string(),
            resources: Resources::default(),
            uri: None,
        };
        assert_eq!(skill.name(), "my-skill");
        assert_eq!(skill.description(), "Does a thing.");
    }

    #[test]
    fn skill_uri_is_settable_after_construction() {
        let mut skill = Skill {
            frontmatter: valid_frontmatter(),
            instructions: String::new(),
            resources: Resources::default(),
            uri: None,
        };
        assert_eq!(skill.uri(), None);
        skill.set_uri(Some("file:///skills/my-skill/SKILL.md".to_string()));
        assert_eq!(skill.uri(), Some("file:///skills/my-skill/SKILL.md"));
    }
}
