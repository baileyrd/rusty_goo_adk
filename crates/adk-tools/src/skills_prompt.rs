//! C0400: `format_skills_as_xml`, ported from `google.adk.skills.prompt`.
//!
//! **Disclosed narrowing**: the source accepts
//! `list[Union[Frontmatter, Skill]]` — the full skill metadata/content
//! models (C0393/C0394, both still `REQUIRED`; loading skills from disk
//! needs a YAML crate decision, C0396). This function only ever reads
//! `.name`/`.description` off either shape, so this port accepts a
//! minimal `SkillSummary{name, description}` instead of depending on
//! either unbuilt model — the same "narrow to what's actually read"
//! convention used throughout this migration. Widen the parameter type
//! once `Frontmatter`/`Skill` land and a real caller needs to pass one
//! directly.

/// Just the two fields `format_skills_as_xml` reads off a
/// `Frontmatter`/`Skill`. See the module doc.
pub struct SkillSummary<'a> {
    pub name: &'a str,
    pub description: &'a str,
}

fn html_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// C0400: `skills.prompt.format_skills_as_xml` — renders skills into an
/// `<available_skills>` XML block for LLM instructions, HTML-escaped.
pub fn format_skills_as_xml(skills: &[SkillSummary]) -> String {
    if skills.is_empty() {
        return "<available_skills>\n</available_skills>".to_string();
    }

    let mut lines: Vec<String> = vec!["<available_skills>".to_string()];
    for skill in skills {
        lines.push("<skill>".to_string());
        lines.push("<name>".to_string());
        lines.push(html_escape(skill.name));
        lines.push("</name>".to_string());
        lines.push("<description>".to_string());
        lines.push(html_escape(skill.description));
        lines.push("</description>".to_string());
        lines.push("</skill>".to_string());
    }
    lines.push("</available_skills>".to_string());

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list_renders_an_empty_block() {
        assert_eq!(
            format_skills_as_xml(&[]),
            "<available_skills>\n</available_skills>"
        );
    }

    #[test]
    fn renders_one_skill() {
        let skills = [SkillSummary {
            name: "roll-dice",
            description: "Rolls an N-sided die.",
        }];
        assert_eq!(
            format_skills_as_xml(&skills),
            "<available_skills>\n\
             <skill>\n\
             <name>\n\
             roll-dice\n\
             </name>\n\
             <description>\n\
             Rolls an N-sided die.\n\
             </description>\n\
             </skill>\n\
             </available_skills>"
        );
    }

    #[test]
    fn renders_multiple_skills_in_order() {
        let skills = [
            SkillSummary {
                name: "a",
                description: "first",
            },
            SkillSummary {
                name: "b",
                description: "second",
            },
        ];
        let result = format_skills_as_xml(&skills);
        assert!(result.find("a").unwrap() < result.find("b").unwrap());
    }

    #[test]
    fn html_escapes_name_and_description() {
        let skills = [SkillSummary {
            name: "<script>",
            description: "danger & \"quotes\" 'here'",
        }];
        let result = format_skills_as_xml(&skills);
        assert!(result.contains("&lt;script&gt;"));
        assert!(result.contains("danger &amp; &quot;quotes&quot; &#x27;here&#x27;"));
        assert!(!result.contains("<script>"));
    }
}
