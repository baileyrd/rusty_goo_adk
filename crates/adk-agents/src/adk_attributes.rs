//! C0669: `telemetry._adk_attributes`, ported from
//! `google.adk.telemetry._adk_attributes` — ADK-owned span attribute
//! names.
//!
//! These attributes are defined by ADK itself; they are not part of any
//! OpenTelemetry semantic convention (neither the stable one this port's
//! `adk-models::stable_semconv` covers nor the experimental one in the
//! still-unported `_experimental_semconv.py`).
//!
//! Everything named `adk.experimental.*` is emitted only when
//! experimental telemetry is enabled and carries no compatibility
//! guarantee — an attribute may be renamed, restructured, or removed in
//! any release. No consumer in this port yet: the span-emission
//! machinery that would set these on a live span needs the still-
//! unbuilt OTel SDK integration; this row is scoped to just the
//! attribute *names* themselves, per its own manifest description —
//! same "declare ahead of a blocked consumer" precedent as C0671/C0672/
//! C0673 (`schema_version.rs`).

pub const ADK_EXPERIMENTAL_SKILL_NAME: &str = "adk.experimental.skill.name";
pub const ADK_EXPERIMENTAL_SKILL_SOURCE_TYPE: &str = "adk.experimental.skill.source.type";
pub const ADK_EXPERIMENTAL_SKILL_DESCRIPTION: &str = "adk.experimental.skill.description";
pub const ADK_EXPERIMENTAL_SKILL_ADDITIONAL_TOOLS: &str = "adk.experimental.skill.additional_tools";
pub const ADK_EXPERIMENTAL_SKILL_SOURCE_URI: &str = "adk.experimental.skill.source.uri";
pub const ADK_EXPERIMENTAL_SKILL_RESOURCE_PATH: &str = "adk.experimental.skill.resource.path";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_attribute_name_is_under_the_adk_experimental_skill_namespace() {
        for name in [
            ADK_EXPERIMENTAL_SKILL_NAME,
            ADK_EXPERIMENTAL_SKILL_SOURCE_TYPE,
            ADK_EXPERIMENTAL_SKILL_DESCRIPTION,
            ADK_EXPERIMENTAL_SKILL_ADDITIONAL_TOOLS,
            ADK_EXPERIMENTAL_SKILL_SOURCE_URI,
            ADK_EXPERIMENTAL_SKILL_RESOURCE_PATH,
        ] {
            assert!(name.starts_with("adk.experimental.skill."));
        }
    }
}
