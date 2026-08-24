//! Part of capabilities C0423/C0424: `_memory_entry_utils.py`, ported from
//! `google.adk.tools._memory_entry_utils`.

use adk_agents::services::MemoryEntry;

/// Extracts the text from the memory entry, joining every text part with a
/// single space — matches the source's default `splitter=' '`.
pub fn extract_text(memory: &MemoryEntry) -> String {
    extract_text_with_splitter(memory, " ")
}

/// `extract_text`, with an explicit joining separator.
pub fn extract_text_with_splitter(memory: &MemoryEntry, splitter: &str) -> String {
    if memory.content.parts.is_empty() {
        return String::new();
    }
    memory
        .content
        .parts
        .iter()
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join(splitter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::services::MemoryEntry;
    use adk_genai::content::{Content, Part};

    fn entry(parts: Vec<Part>) -> MemoryEntry {
        MemoryEntry {
            content: Content::new("user", parts),
            custom_metadata: Default::default(),
            id: None,
            author: None,
            timestamp: None,
        }
    }

    #[test]
    fn joins_text_parts_with_a_space_by_default() {
        let memory = entry(vec![Part::text("hello"), Part::text("world")]);
        assert_eq!(extract_text(&memory), "hello world");
    }

    #[test]
    fn skips_non_text_parts() {
        let memory = entry(vec![
            Part::text("hello"),
            Part::default(),
            Part::text("world"),
        ]);
        assert_eq!(extract_text(&memory), "hello world");
    }

    #[test]
    fn returns_empty_string_for_no_parts() {
        let memory = entry(vec![]);
        assert_eq!(extract_text(&memory), "");
    }

    #[test]
    fn uses_a_custom_splitter() {
        let memory = entry(vec![Part::text("a"), Part::text("b")]);
        assert_eq!(extract_text_with_splitter(&memory, ", "), "a, b");
    }
}
