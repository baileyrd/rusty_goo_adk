//! Capability C0027: `EventCompaction`, ported from
//! `google.adk.events.event_actions`.

use rusty_serde::value::Value;
use rusty_serde::{Deserialize, Serialize};

/// Records that a range of session events (`start_timestamp..=end_timestamp`)
/// was summarized into `compacted_content`. This is the schema
/// `VertexAiSessionService` smuggles through `custom_metadata['_compaction']`
/// (see the sessions phase), and that `DatabaseSessionService`/
/// `SqliteSessionService` persist inline as part of the event JSON.
///
/// **Adaptation**: `compacted_content` is the source's `Content` type
/// (a `google.genai.types.Content` — text/media parts), which belongs to
/// the models/ phase and isn't built yet. It's typed as a plain JSON
/// [`Value`] here as a placeholder; this narrows to a concrete `Content`
/// type once that phase lands, without changing the wire shape (a
/// `Content` value serializes to the same JSON either way).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[rusty_serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EventCompaction {
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub compacted_content: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusty_serde::value::Value;

    #[test]
    fn round_trips_through_json() {
        let c = EventCompaction {
            start_timestamp: 1.0,
            end_timestamp: 2.0,
            compacted_content: Value::String("summary text".to_string()),
        };
        let json = rusty_serde::json::to_string(&c).unwrap();
        let back: EventCompaction = rusty_serde::json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
