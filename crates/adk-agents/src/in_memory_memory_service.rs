//! Capabilities C0245/C0247/C0248/C0249: `InMemoryMemoryService`, ported
//! from `google.adk.memory.in_memory_memory_service`, plus
//! `memory._utils.format_timestamp` (C0245). This is the first real
//! implementation of the `MemoryService` trait (a placeholder since
//! Phase 6 — see `services.rs`'s module doc) — the same "first real
//! backend narrows a placeholder trait to its actual contract" moment
//! `InMemorySessionService` was for `SessionService`.
//!
//! **`format_timestamp`, disclosed**: the source's
//! `datetime.fromtimestamp(timestamp).isoformat()` uses the *local*
//! system timezone (no explicit `tzinfo`) — the manifest's own C0245
//! row flags this as worth parity attention. Porting local-timezone
//! conversion faithfully would need a full IANA timezone
//! database/DST-rules crate (`chrono-tz` or similar) — a new
//! dependency this batch doesn't add. [`format_timestamp`] instead
//! formats in UTC: the epoch-seconds→calendar-date math
//! (`civil_from_days`, Howard Hinnant's well-known public-domain
//! algorithm) is simple and unambiguous with no DST-database risk,
//! unlike full local-time conversion — the same "hand-roll only what's
//! genuinely simple and get a real dependency for what's genuinely
//! complex" reasoning `load_web_page.rs`'s IP classification already
//! established for this port. **Real, disclosed narrowing**: the ISO
//! 8601 string this port produces for a given Unix timestamp will
//! differ from the source's wall-clock hour/date whenever the host
//! isn't running in UTC — since `MemoryEntry.timestamp` is documented
//! as forwarded to the LLM verbatim rather than parsed back, this is a
//! real but low-severity divergence (the LLM sees a different clock
//! time, not wrong or unparseable data).
//!
//! **`add_memory`, disclosed**: the source's `BaseMemoryService`
//! gives `add_memory` a default body that raises `NotImplementedError`
//! unless a backend overrides it; `InMemoryMemoryService` doesn't
//! override it. This port's `MemoryService` trait method has no
//! `Result` to return "unsupported" through (it predates this batch —
//! see `services.rs`), so [`InMemoryMemoryService::add_memory`] calls
//! `unimplemented!()` — a Rust panic is the closest behavioral analog
//! to an uncaught Python exception (both abort the call rather than
//! returning), even though callers can't recover from it via a
//! `Result` the way a caught `NotImplementedError` could be.
//!
//! **`add_events_to_memory`'s `session_id`, already narrowed**: the
//! source's `session_id: str | None = None` (falling back to the
//! `__unknown_session_id__` sentinel) isn't representable through this
//! port's pre-existing `MemoryService::add_events_to_memory` trait
//! signature, which takes `session_id: &str` (no `Option`) — a
//! narrowing already baked into that trait and its sole caller,
//! `Context::add_events_to_memory` (built in an earlier batch, before
//! any real `MemoryService` implementation existed to exercise it),
//! not something newly introduced here. [`UNKNOWN_SESSION_ID`] is kept
//! as a public constant so a future batch correcting that trait
//! signature has the exact sentinel string to restore.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use adk_events::Event;
use rusty_serde::value::Value;

use crate::services::{MemoryEntry, MemoryService, SearchMemoryResponse};
use crate::session::Session;

/// `memory.in_memory_memory_service._UNKNOWN_SESSION_ID` — see the
/// module doc for why this port's `add_events_to_memory` can't
/// currently reach this fallback through `Context`.
pub const UNKNOWN_SESSION_ID: &str = "__unknown_session_id__";

const MAX_SEARCH_RESULTS: usize = 10;

/// Formats a Unix timestamp as an ISO 8601 string in UTC — see the
/// module doc for the disclosed UTC-vs-local-time narrowing from the
/// source's `datetime.fromtimestamp(...).isoformat()`.
pub fn format_timestamp(timestamp: f64) -> String {
    let total_seconds = timestamp.floor() as i64;
    let fractional = timestamp - timestamp.floor();
    let days = total_seconds.div_euclid(86400);
    let seconds_of_day = total_seconds.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = (seconds_of_day % 3600) / 60;
    let second = seconds_of_day % 60;
    let micros = (fractional * 1_000_000.0).round() as i64;
    if micros == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}")
    }
}

/// Days-since-epoch → (year, month, day) in the proleptic Gregorian
/// calendar. Howard Hinnant's `civil_from_days` algorithm (public
/// domain): <https://howardhinnant.github.io/date_algorithms.html>.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Extracts Unicode-aware lowercase word tokens from `text`, matching
/// the source's `re.findall(r'\w+', text)` (Python's `\w` and the
/// `regex` crate's `\w` are both Unicode-aware by default).
fn extract_words_lower(text: &str) -> std::collections::HashSet<String> {
    static WORD_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = WORD_RE.get_or_init(|| regex::Regex::new(r"\w+").unwrap());
    re.find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

/// Keyed by `(app_name, user_id)`; values map `session_id` to that
/// session's retained events.
type SessionEventsByUser = HashMap<(String, String), HashMap<String, Vec<Event>>>;

/// C0247-C0249: an in-memory, in-process keyword-search memory
/// service — for prototyping only, per the source's own docstring.
/// Uses keyword matching instead of semantic search: a search returns
/// at most [`MAX_SEARCH_RESULTS`] memories, the ones sharing the most
/// words with the query. Thread-safe (guarded by a single [`Mutex`]),
/// matching the source's own `threading.Lock`.
pub struct InMemoryMemoryService {
    session_events: Mutex<SessionEventsByUser>,
}

impl InMemoryMemoryService {
    pub fn new() -> Self {
        Self {
            session_events: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryMemoryService {
    fn default() -> Self {
        Self::new()
    }
}

fn has_content(event: &Event) -> bool {
    event
        .content
        .as_ref()
        .is_some_and(|content| !content.parts.is_empty())
}

impl MemoryService for InMemoryMemoryService {
    fn add_session_to_memory(&self, session: &Session) {
        let user_key = (session.app_name.clone(), session.user_id.clone());
        let retained: Vec<Event> = session
            .events
            .iter()
            .filter(|event| has_content(event))
            .cloned()
            .collect();

        let mut sessions = self.session_events.lock().unwrap();
        sessions
            .entry(user_key)
            .or_default()
            .insert(session.id.clone(), retained);
    }

    fn add_events_to_memory(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        events: &[Event],
        _custom_metadata: Option<&BTreeMap<String, Value>>,
    ) {
        let user_key = (app_name.to_string(), user_id.to_string());
        let events_to_add: Vec<&Event> = events.iter().filter(|event| has_content(event)).collect();

        let mut sessions = self.session_events.lock().unwrap();
        let session_map = sessions.entry(user_key).or_default();
        let existing = session_map.entry(session_id.to_string()).or_default();
        let mut existing_ids: std::collections::HashSet<String> =
            existing.iter().map(|event| event.id.clone()).collect();
        for event in events_to_add {
            if existing_ids.insert(event.id.clone()) {
                existing.push(event.clone());
            }
        }
    }

    fn add_memory(
        &self,
        _app_name: &str,
        _user_id: &str,
        _memories: &[MemoryEntry],
        _custom_metadata: Option<&BTreeMap<String, Value>>,
    ) {
        // See the module doc: the source never overrides this default,
        // which raises `NotImplementedError` — `unimplemented!()` is
        // the closest Rust analog available through this trait's
        // current (non-`Result`) signature.
        unimplemented!(
            "InMemoryMemoryService does not support direct memory writes; \
             call add_events_to_memory(...) or add_session_to_memory(session) instead."
        )
    }

    fn search_memory(&self, app_name: &str, user_id: &str, query: &str) -> SearchMemoryResponse {
        let user_key = (app_name.to_string(), user_id.to_string());

        // Snapshot under the lock, score outside it -- iterating a live
        // reference outside the lock would race with concurrent writers
        // mutating the same maps/vecs, matching the source's own comment.
        let session_event_lists: Vec<Vec<Event>> = {
            let sessions = self.session_events.lock().unwrap();
            sessions
                .get(&user_key)
                .map(|by_session| by_session.values().cloned().collect())
                .unwrap_or_default()
        };

        let words_in_query = extract_words_lower(query);
        let mut scored_memories: Vec<(usize, MemoryEntry)> = Vec::new();

        for session_events in &session_event_lists {
            for event in session_events {
                let Some(content) = &event.content else {
                    continue;
                };
                if content.parts.is_empty() {
                    continue;
                }
                let event_text = content
                    .parts
                    .iter()
                    .filter_map(|part| part.text.as_deref())
                    .collect::<Vec<_>>()
                    .join(" ");
                let words_in_event = extract_words_lower(&event_text);
                if words_in_event.is_empty() {
                    continue;
                }

                let event_text_lower = event_text.to_lowercase();
                let matched_words = words_in_query
                    .iter()
                    .filter(|query_word| {
                        words_in_event.contains(*query_word)
                            || (!query_word.is_ascii() && event_text_lower.contains(*query_word))
                    })
                    .count();

                if matched_words > 0 {
                    scored_memories.push((
                        matched_words,
                        MemoryEntry {
                            content: content.clone(),
                            custom_metadata: BTreeMap::new(),
                            id: None,
                            author: Some(event.author.clone()),
                            timestamp: Some(format_timestamp(event.timestamp)),
                        },
                    ));
                }
            }
        }

        // Stable sort on the count alone (matching the source's own sort
        // key) keeps equally-scored memories in insertion order.
        scored_memories.sort_by_key(|(count, _)| std::cmp::Reverse(*count));
        SearchMemoryResponse {
            memories: scored_memories
                .into_iter()
                .take(MAX_SEARCH_RESULTS)
                .map(|(_, memory)| memory)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_genai::content::{Content, Part};

    fn text_event(id: &str, author: &str, text: &str, timestamp: f64) -> Event {
        let mut event = Event::new("inv", author, adk_events::node_info::NodeInfo::new("root"));
        event.id = id.to_string();
        event.timestamp = timestamp;
        event.content = Some(Content::new(
            "user",
            vec![Part {
                text: Some(text.to_string()),
                ..Default::default()
            }],
        ));
        event
    }

    fn empty_content_event(id: &str) -> Event {
        let mut event = Event::new("inv", "user", adk_events::node_info::NodeInfo::new("root"));
        event.id = id.to_string();
        event.timestamp = 0.0;
        event
    }

    #[test]
    fn format_timestamp_formats_a_whole_second_without_a_fraction() {
        assert_eq!(format_timestamp(0.0), "1970-01-01T00:00:00");
    }

    #[test]
    fn format_timestamp_includes_microseconds_when_present() {
        assert_eq!(format_timestamp(0.5), "1970-01-01T00:00:00.500000");
    }

    #[test]
    fn format_timestamp_handles_a_known_calendar_date() {
        // 2024-01-15T10:30:00 UTC.
        assert_eq!(format_timestamp(1_705_314_600.0), "2024-01-15T10:30:00");
    }

    #[test]
    fn add_session_to_memory_retains_only_events_with_content() {
        let service = InMemoryMemoryService::new();
        let mut session = Session::new("app", "user", "s1");
        session.events = vec![
            text_event("e1", "user", "hello world", 1.0),
            empty_content_event("e2"),
        ];
        service.add_session_to_memory(&session);

        let result = service.search_memory("app", "user", "hello");
        assert_eq!(result.memories.len(), 1);
    }

    #[test]
    fn add_session_to_memory_overwrites_the_previous_entry_for_the_same_session_id() {
        let service = InMemoryMemoryService::new();
        let mut session = Session::new("app", "user", "s1");
        session.events = vec![text_event("e1", "user", "first version", 1.0)];
        service.add_session_to_memory(&session);

        session.events = vec![text_event("e2", "user", "second version", 2.0)];
        service.add_session_to_memory(&session);

        let result = service.search_memory("app", "user", "first");
        assert!(result.memories.is_empty());
        let result = service.search_memory("app", "user", "second");
        assert_eq!(result.memories.len(), 1);
    }

    #[test]
    fn add_events_to_memory_is_additive_and_dedups_by_event_id() {
        let service = InMemoryMemoryService::new();
        service.add_events_to_memory(
            "app",
            "user",
            "s1",
            &[text_event("e1", "user", "alpha", 1.0)],
            None,
        );
        service.add_events_to_memory(
            "app",
            "user",
            "s1",
            &[
                text_event("e1", "user", "alpha duplicate ignored", 1.0),
                text_event("e2", "user", "beta", 2.0),
            ],
            None,
        );

        let result = service.search_memory("app", "user", "alpha");
        assert_eq!(result.memories.len(), 1);
        let result = service.search_memory("app", "user", "beta");
        assert_eq!(result.memories.len(), 1);
    }

    #[test]
    fn search_memory_scopes_by_app_and_user() {
        let service = InMemoryMemoryService::new();
        service.add_events_to_memory(
            "app",
            "alice",
            "s1",
            &[text_event("e1", "alice", "alice's memory", 1.0)],
            None,
        );

        assert!(service
            .search_memory("app", "bob", "memory")
            .memories
            .is_empty());
        assert_eq!(
            service
                .search_memory("app", "alice", "memory")
                .memories
                .len(),
            1
        );
    }

    #[test]
    fn search_memory_ranks_by_matched_word_count_and_caps_at_ten() {
        let service = InMemoryMemoryService::new();
        let events: Vec<Event> = (0..15)
            .map(|i| text_event(&format!("e{i}"), "user", "cat", i as f64))
            .collect();
        service.add_events_to_memory("app", "user", "s1", &events, None);
        service.add_events_to_memory(
            "app",
            "user",
            "s1",
            &[text_event("best", "user", "cat dog bird", 99.0)],
            None,
        );

        let result = service.search_memory("app", "user", "cat dog bird");
        assert_eq!(result.memories.len(), MAX_SEARCH_RESULTS);
        assert_eq!(result.memories[0].author.as_deref(), Some("user"));
        assert_eq!(
            result.memories[0].content.parts[0].text.as_deref(),
            Some("cat dog bird")
        );
    }

    #[test]
    fn search_memory_matches_non_ascii_query_words_via_substring() {
        let service = InMemoryMemoryService::new();
        service.add_events_to_memory(
            "app",
            "user",
            "s1",
            &[text_event("e1", "user", "东京タワー is a landmark", 1.0)],
            None,
        );

        let result = service.search_memory("app", "user", "东京");
        assert_eq!(result.memories.len(), 1);
    }

    #[test]
    fn search_memory_returns_the_formatted_timestamp() {
        let service = InMemoryMemoryService::new();
        service.add_events_to_memory(
            "app",
            "user",
            "s1",
            &[text_event("e1", "user", "hello", 0.0)],
            None,
        );
        let result = service.search_memory("app", "user", "hello");
        assert_eq!(
            result.memories[0].timestamp.as_deref(),
            Some("1970-01-01T00:00:00")
        );
    }

    #[test]
    #[should_panic]
    fn add_memory_is_unimplemented() {
        let service = InMemoryMemoryService::new();
        service.add_memory("app", "user", &[], None);
    }
}
