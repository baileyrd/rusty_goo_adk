//! Runnable demo of the `OllamaLlm` backend end to end: builds an
//! `LlmRequest`, sends it, and prints the model's reply.
//!
//! - If a real Ollama server answers at `OLLAMA_HOST` (default
//!   `http://localhost:11434`), this makes a real call using
//!   `OLLAMA_MODEL` (default `llama3.2` — pull it first with
//!   `ollama pull llama3.2`, or set `OLLAMA_MODEL` to a model you already
//!   have).
//! - Otherwise it starts a local one-shot mock server that speaks
//!   Ollama's documented `/api/chat` response shape, so this runs with no
//!   Ollama install required.
//!
//! Run with:
//!   cargo run -p adk-models --example ollama_demo
//!   OLLAMA_MODEL=mistral cargo run -p adk-models --example ollama_demo

use std::io::{Read, Write};
use std::time::Duration;

use adk_genai::content::Content;
use adk_models::llm_request::LlmRequest;
use adk_models::ollama::OllamaLlm;

#[rusty_tokio::main]
async fn main() {
    let default_base_url =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());

    let (base_url, model, mock_server) = if probe_real_ollama(&default_base_url) {
        println!("Found a real Ollama server at {default_base_url}.");
        let tag = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
        (default_base_url, format!("ollama/{tag}"), None)
    } else {
        println!(
            "No Ollama server reachable at {default_base_url} — starting a local mock server instead."
        );
        let (mock_base_url, handle) = spawn_mock_ollama_server();
        (mock_base_url, "ollama/mock-model".to_string(), Some(handle))
    };

    let llm = OllamaLlm::new(&model).with_base_url(base_url);
    let mut request = LlmRequest::new(&model);
    request
        .contents
        .push(Content::user_text("Say hello in one short sentence."));

    println!("Sending request to model {model}...");
    match llm.generate_content(&request).await {
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

/// A quick, short-timeout probe against Ollama's own `/api/tags`
/// (list-models) endpoint — cheap enough to call on every demo run
/// without noticeably slowing down the no-server case.
fn probe_real_ollama(base_url: &str) -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    client
        .get(format!("{}/api/tags", base_url.trim_end_matches('/')))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// A one-shot local HTTP server returning a canned, wire-accurate
/// `/api/chat` response body — enough to exercise `OllamaLlm`'s real
/// request/response path without a live Ollama install. Same
/// dependency-free pattern as `ollama.rs`'s own tests.
fn spawn_mock_ollama_server() -> (String, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
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

        let body = r#"{"model":"mock-model","message":{"role":"assistant","content":"Hello! (this reply came from the local mock Ollama server)"},"done":true,"prompt_eval_count":8,"eval_count":12}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    });
    (format!("http://{addr}"), handle)
}
