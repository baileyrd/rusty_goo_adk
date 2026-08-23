//! Runnable demo of the `Gemini` backend end to end: builds an
//! `LlmRequest`, sends it, and prints the model's reply.
//!
//! - If `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set, this makes a real
//!   call to the Gemini Developer API with `GEMINI_MODEL` (default
//!   `gemini-2.5-flash`).
//! - Otherwise it starts a local one-shot mock server that speaks the
//!   same `generateContent` response shape (see `gemini.rs`'s own tests)
//!   and points `Gemini` at it via an injected client, so this runs with
//!   no API key or network access required.
//!
//! Run with:
//!   cargo run -p adk-models --example gemini_demo
//!   GOOGLE_API_KEY=... GEMINI_MODEL=gemini-2.5-pro cargo run -p adk-models --example gemini_demo

use std::io::{Read, Write};
use std::sync::Arc;

use adk_genai::content::Content;
use adk_models::gemini::{Gemini, GeminiApiClient};
use adk_models::llm_request::LlmRequest;

#[rusty_tokio::main]
async fn main() {
    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let has_real_key =
        std::env::var("GOOGLE_API_KEY").is_ok() || std::env::var("GEMINI_API_KEY").is_ok();

    let (gemini, mock_server) = if has_real_key {
        println!("Found an API key in the environment — calling the real Gemini API.");
        (Gemini::new(&model), None)
    } else {
        println!(
            "No GOOGLE_API_KEY/GEMINI_API_KEY set — starting a local mock Gemini server instead."
        );
        let (base_url, handle) = spawn_mock_gemini_server();
        let mock_client = Arc::new(GeminiApiClient {
            http: reqwest::blocking::Client::new(),
            base_url: Some(base_url),
            api_version: None,
            headers: Vec::new(),
            retry_options: None,
            enterprise: false,
        });
        (Gemini::new(&model).with_client(mock_client), Some(handle))
    };

    let mut request = LlmRequest::new(&model);
    request
        .contents
        .push(Content::user_text("Say hello in one short sentence."));

    println!("Sending request to model {model}...");
    match gemini.generate_content(&request).await {
        Ok(response) => {
            let text = response
                .content
                .as_ref()
                .and_then(|c| c.parts.first())
                .and_then(|p| p.text.as_deref())
                .unwrap_or("<no text in response>");
            println!("Response: {text}");
        }
        Err(e) => println!("Error: {e}"),
    }

    if let Some(handle) = mock_server {
        handle.join().unwrap();
    }
}

/// A one-shot local HTTP server returning a canned, wire-accurate
/// `GenerateContentResponse` body — enough to exercise `Gemini`'s real
/// request/response path without a live Gemini endpoint. Same
/// dependency-free pattern as `gemini.rs`'s own tests.
fn spawn_mock_gemini_server() -> (String, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut received = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                break;
            }
            received.extend_from_slice(&buf[..n]);
            if let Some(header_end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&received[..header_end]);
                let content_length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                if received.len() >= header_end + 4 + content_length {
                    break;
                }
            }
        }

        let body = r#"{"modelVersion":"gemini-2.5-flash-mock","candidates":[{"content":{"role":"model","parts":[{"text":"Hello! (this reply came from the local mock Gemini server)"}]},"finishReason":"STOP"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    });
    (format!("http://{addr}"), handle)
}
