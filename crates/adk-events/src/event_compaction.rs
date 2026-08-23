//! Capability C0027: `EventCompaction`, ported from
//! `google.adk.events.event_actions`.

use adk_genai::content::Content;
use rusty_serde::{Deserialize, Serialize};

/// Records that a range of session events (`start_timestamp..=end_timestamp`)
/// was summarized into `compacted_content`. This is the schema
/// `VertexAiSessionService` smuggles through `custom_metadata['_compaction']`
/// (see the sessions phase), and that `DatabaseSessionService`/
/// `SqliteSessionService` persist inline as part of the event JSON.
///
/// **Adaptation, narrowed in Phase 4 batch 7**: `compacted_content` was
/// originally a placeholder JSON [`rusty_serde::value::Value`] (the source's
/// `Content` type belonged to the not-yet-built models/ phase). Phase 3
/// landed a real `adk_genai::content::Content`, and `adk-events` already
/// depends on `adk-genai` for `Event.content` itself, so this narrows to
/// the same real type — needed as a genuine `Content` (not an opaque blob)
/// by `_content_compaction.rs`'s `process_compaction_events`, which
/// constructs a synthetic `Event` whose `content` *is* this field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EventCompaction {
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub compacted_content: Content,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let c = EventCompaction {
            start_timestamp: 1.0,
            end_timestamp: 2.0,
            compacted_content: Content::user_text("summary text"),
        };
        let json = rusty_serde::json::to_string(&c).unwrap();
        let back: EventCompaction = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
