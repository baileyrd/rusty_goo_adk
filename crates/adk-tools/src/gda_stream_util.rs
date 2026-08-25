//! Part of C0489: the pure accumulator/parsing half of
//! `_gda_stream_util.py` (Gemini Data Analytics streaming responses),
//! ported from `google.adk.tools._gda_stream_util`.
//!
//! **Scope boundary, disclosed**: `get_gda_endpoint`/`get_gda_session`
//! (lines 37-89 of the source) build a live mTLS-aware `AuthorizedSession`
//! off `google.auth.credentials.Credentials` and issue the actual
//! streaming HTTP POST — this workspace has no `google-auth`-equivalent
//! dependency (the same disclosed gap `base_authenticated_tool.rs`
//! already names for `GoogleTool`/`_google_credentials.py`, C0413/C0414),
//! so neither function is ported. What *is* portable without any
//! network/credential dependency is everything downstream of an already-
//! received response body: [`get_stream`] takes the decoded response
//! lines directly (in place of the source's `requests.Session`/`url`/
//! `ca_payload`/`headers` parameters, which exist only to perform the
//! live POST this port can't make), plus [`extract_data_result`]/
//! [`format_data_retrieved`] unchanged in behavior.
//!
//! **Built ahead of its own caller**: `_gda_stream_util.py`'s sole
//! caller in the source, `data_agent_tool.py`'s `DataAgentToolset`
//! (C0480/C0481), is itself GCP-blocked — same "port the pure logic
//! ahead of a still-blocked concrete consumer" precedent already used
//! for `remote_mcp_server.rs`/`resolve_authorization_endpoint_and_scopes`
//! (C0508).

use rusty_serde::value::Value;

/// `_gda_stream_util.get_stream`'s accumulator/parsing loop — see the
/// module doc for why this takes already-decoded response lines rather
/// than a live `requests.Session`/URL/payload/headers.
///
/// Buffers lines until a complete JSON value is assembled (handling the
/// streaming API's `"[{"`/`"}]"`/`","` line-framing), classifies each
/// parsed object via [`extract_data_result`], and replaces the previous
/// "Data Retrieved" entry with an "Intermediate result omitted"
/// placeholder whenever a newer one supersedes it.
pub fn get_stream<'a>(
    lines: impl IntoIterator<Item = &'a str>,
    max_query_result_rows: i64,
) -> Vec<Value> {
    let mut accumulator = String::new();
    let mut messages: Vec<Value> = Vec::new();
    let mut data_msg_idx: Option<usize> = None;

    for line in lines {
        if line.is_empty() {
            continue;
        }

        match line {
            "[{" => accumulator = "{".to_string(),
            "}]" => accumulator.push('}'),
            "," => continue,
            _ => accumulator.push_str(line),
        }

        let data_json = match rusty_serde::json::from_str::<Value>(&accumulator) {
            Ok(value) => value,
            Err(_) => continue,
        };
        accumulator.clear();

        if !matches!(data_json, Value::Map(_)) {
            messages.push(data_json);
            continue;
        }

        let processed_msg = if let Some(data_result) = extract_data_result(&data_json) {
            let formatted = format_data_retrieved(&data_result, max_query_result_rows);
            if let Some(idx) = data_msg_idx {
                messages[idx] = intermediate_result_omitted();
            }
            data_msg_idx = Some(messages.len());
            formatted
        } else if matches!(data_json.get("systemMessage"), Some(Value::Map(_))) {
            data_json.get("systemMessage").cloned().unwrap()
        } else {
            data_json
        };

        messages.push(processed_msg);
    }

    messages
}

fn intermediate_result_omitted() -> Value {
    let mut wrapper = Value::Map(Vec::new());
    wrapper.insert(
        "Data Retrieved",
        Value::String("Intermediate result omitted".to_string()),
    );
    wrapper
}

/// `_gda_stream_util._extract_data_result` — attempts to find
/// `systemMessage.data.result` deep inside the generic message dict,
/// returning it only when `result.data` is itself a list.
pub fn extract_data_result(msg: &Value) -> Option<Value> {
    let system_message = msg.get("systemMessage")?;
    if !matches!(system_message, Value::Map(_)) {
        return None;
    }
    let data = system_message.get("data")?;
    if !matches!(data, Value::Map(_)) {
        return None;
    }
    let result = data.get("result")?;
    if !matches!(result, Value::Map(_)) {
        return None;
    }
    match result.get("data") {
        Some(Value::Seq(_)) => Some(result.clone()),
        _ => None,
    }
}

/// `_gda_stream_util._format_data_retrieved` — transforms the raw
/// `result` dict into the simplified `{headers, rows, summary}` shape,
/// wrapped under a `"Data Retrieved"` key.
pub fn format_data_retrieved(result: &Value, max_rows: i64) -> Value {
    let raw_data: Vec<Value> = match result.get("data") {
        Some(Value::Seq(items)) => items.clone(),
        _ => Vec::new(),
    };

    let mut headers: Vec<String> = result
        .get("schema")
        .filter(|schema| matches!(schema, Value::Map(_)))
        .and_then(|schema| schema.get("fields"))
        .and_then(Value::as_seq)
        .map(|fields| {
            fields
                .iter()
                .filter_map(|field| field.get("name")?.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    if headers.is_empty() {
        if let Some(Value::Map(entries)) = raw_data.first() {
            headers = entries.iter().map(|(key, _)| key.clone()).collect();
        }
    }

    let total_rows = raw_data.len() as i64;
    let num_to_display = total_rows.min(max_rows).max(0) as usize;

    let rows: Vec<Value> = raw_data
        .iter()
        .take(num_to_display)
        .filter(|row| matches!(row, Value::Map(_)))
        .map(|row| {
            Value::Seq(
                headers
                    .iter()
                    .map(|header| row.get(header).cloned().unwrap_or(Value::Null))
                    .collect(),
            )
        })
        .collect();

    let summary = if total_rows > max_rows {
        format!("Showing the first {num_to_display} of {total_rows} total rows.")
    } else {
        format!("Showing all {total_rows} rows.")
    };

    let mut data_retrieved = Value::Map(Vec::new());
    data_retrieved.insert(
        "headers",
        Value::Seq(headers.into_iter().map(Value::String).collect()),
    );
    data_retrieved.insert("rows", Value::Seq(rows));
    data_retrieved.insert("summary", Value::String(summary));

    let mut wrapper = Value::Map(Vec::new());
    wrapper.insert("Data Retrieved", data_retrieved);
    wrapper
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_empty_lines() {
        let messages = get_stream(["", "\"hello\"", ""], 10);
        assert_eq!(messages, vec![Value::String("hello".to_string())]);
    }

    #[test]
    fn assembles_a_bracketed_object_stream() {
        let messages = get_stream(["[{", "\"foo\":1", "}]"], 10);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].get("foo").and_then(Value::as_i64), Some(1));
    }

    #[test]
    fn a_comma_separator_line_is_skipped() {
        let messages = get_stream(["[{", "\"foo\":1", "}]", ",", "[{", "\"bar\":2", "}]"], 10);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].get("bar").and_then(Value::as_i64), Some(2));
    }

    #[test]
    fn a_malformed_line_is_buffered_until_valid_json_completes() {
        let messages = get_stream(["{\"a\":", "1}"], 10);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].get("a").and_then(Value::as_i64), Some(1));
    }

    #[test]
    fn a_non_dict_message_is_passed_through_directly() {
        let messages = get_stream(["42"], 10);
        assert_eq!(messages, vec![Value::Int(42)]);
    }

    #[test]
    fn a_system_message_dict_is_unwrapped() {
        let messages = get_stream(["{\"systemMessage\": {\"text\": \"hi\"}}"], 10);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].get("text").and_then(Value::as_str), Some("hi"));
    }

    #[test]
    fn extract_data_result_requires_a_list_shaped_data_field() {
        let msg: Value = rusty_serde::json::from_str(
            r#"{"systemMessage": {"data": {"result": {"data": [1, 2]}}}}"#,
        )
        .unwrap();
        assert!(extract_data_result(&msg).is_some());

        let not_a_list: Value = rusty_serde::json::from_str(
            r#"{"systemMessage": {"data": {"result": {"data": "nope"}}}}"#,
        )
        .unwrap();
        assert!(extract_data_result(&not_a_list).is_none());
    }

    #[test]
    fn a_data_result_message_replaces_the_previous_one_with_a_placeholder() {
        let messages = get_stream(
            [
                "{\"systemMessage\": {\"data\": {\"result\": {\"data\": [{\"a\": 1}]}}}}",
                "{\"systemMessage\": {\"data\": {\"result\": {\"data\": [{\"a\": 2}]}}}}",
            ],
            10,
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("Data Retrieved").and_then(Value::as_str),
            Some("Intermediate result omitted")
        );
        assert!(messages[1].get("Data Retrieved").is_some());
    }

    #[test]
    fn format_data_retrieved_uses_schema_field_names_as_headers() {
        let result: Value = rusty_serde::json::from_str(
            r#"{"schema": {"fields": [{"name": "id"}, {"name": "name"}]}, "data": [{"id": 1, "name": "a"}]}"#,
        )
        .unwrap();
        let formatted = format_data_retrieved(&result, 10);
        let data_retrieved = formatted.get("Data Retrieved").unwrap();
        let headers = data_retrieved
            .get("headers")
            .and_then(Value::as_seq)
            .unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].as_str(), Some("id"));
        assert_eq!(headers[1].as_str(), Some("name"));
    }

    #[test]
    fn format_data_retrieved_falls_back_to_first_row_keys_when_no_schema() {
        let result: Value = rusty_serde::json::from_str(r#"{"data": [{"x": 1, "y": 2}]}"#).unwrap();
        let formatted = format_data_retrieved(&result, 10);
        let headers = formatted
            .get("Data Retrieved")
            .unwrap()
            .get("headers")
            .and_then(Value::as_seq)
            .unwrap();
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn format_data_retrieved_truncates_rows_and_reports_the_total() {
        let result: Value =
            rusty_serde::json::from_str(r#"{"data": [{"id": 1}, {"id": 2}, {"id": 3}]}"#).unwrap();
        let formatted = format_data_retrieved(&result, 2);
        let data_retrieved = formatted.get("Data Retrieved").unwrap();
        let rows = data_retrieved.get("rows").and_then(Value::as_seq).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            data_retrieved.get("summary").and_then(Value::as_str),
            Some("Showing the first 2 of 3 total rows.")
        );
    }

    #[test]
    fn format_data_retrieved_reports_all_rows_when_under_the_limit() {
        let result: Value = rusty_serde::json::from_str(r#"{"data": [{"id": 1}]}"#).unwrap();
        let formatted = format_data_retrieved(&result, 10);
        assert_eq!(
            formatted
                .get("Data Retrieved")
                .unwrap()
                .get("summary")
                .and_then(Value::as_str),
            Some("Showing all 1 rows.")
        );
    }
}
