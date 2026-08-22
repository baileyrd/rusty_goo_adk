//! The WebSocket transport primitive for the Gemini Live API — the
//! low-level counterpart to `gemini.rs`'s `GeminiApiClient` (the REST
//! transport).
//!
//! **Dependency decision, Phase 3 batch 5**: `tungstenite` — checked every
//! sibling Rusty-Mill repo under the platform directory (same list as the
//! `reqwest` decision in `gemini.rs`) for a WebSocket candidate; none
//! exists. Adopted `tungstenite` rather than `tokio-tungstenite`: it's the
//! synchronous, runtime-agnostic core `tokio-tungstenite` itself wraps, so
//! it has the exact same "doesn't need a real tokio reactor" property that
//! made `reqwest::blocking` the right fit for the REST transport (see
//! `gemini.rs`'s load-bearing adaptation note) — `rusty_tokio` (this
//! workspace's from-scratch, independent async runtime) still can't share a
//! reactor with anything that assumes real tokio underneath, so a
//! synchronous WebSocket library bridged via `rusty_tokio::spawn_blocking`
//! is the same fix applied to a second transport. `rustls-tls-webpki-roots`
//! keeps TLS pure-Rust, matching the `reqwest` decision (no system
//! OpenSSL).
//!
//! **Scope**: this module is the transport primitive only — open a
//! connection, send/receive text frames, close. It's tested end-to-end
//! against a local `tungstenite`-based test server (dependency-free, same
//! pattern as `gemini.rs`'s local HTTP test server for the REST transport).
//! The actual Gemini Live API wire protocol (the `BidiGenerateContent*`
//! message envelopes `GeminiLlmConnection` builds and sends over this
//! transport) is a separate concern — see `gemini_llm_connection.rs`.

use std::net::TcpStream;
use std::sync::Mutex;

use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

#[derive(Debug, rusty_err::Error)]
pub enum LiveWsError {
    #[error("failed to connect to the Live WebSocket endpoint: {0}")]
    Connect(String),
    #[error("failed to send a Live WebSocket message: {0}")]
    Send(String),
    #[error("failed to receive a Live WebSocket message: {0}")]
    Receive(String),
    #[error("failed to close the Live WebSocket connection: {0}")]
    Close(String),
}

/// A single Live WebSocket connection. All methods take `&self` (interior
/// mutability via a [`Mutex`]) so this can back a `BaseLlmConnection` trait
/// object, whose methods take `&self`.
pub struct LiveWsConnection {
    socket: Mutex<WebSocket<MaybeTlsStream<TcpStream>>>,
}

impl LiveWsConnection {
    /// Opens a blocking WebSocket connection to `url` (`ws://` or `wss://`).
    /// Genuinely blocks the calling thread — callers on a `rusty_tokio`
    /// async worker must run this via `rusty_tokio::spawn_blocking`.
    pub fn connect(url: &str) -> Result<Self, LiveWsError> {
        let (socket, _response) =
            tungstenite::connect(url).map_err(|e| LiveWsError::Connect(e.to_string()))?;
        Ok(Self {
            socket: Mutex::new(socket),
        })
    }

    /// Sends a text frame. Blocks until the write completes.
    pub fn send_text(&self, text: String) -> Result<(), LiveWsError> {
        self.socket
            .lock()
            .expect("Live WebSocket mutex poisoned")
            .send(Message::Text(text.into()))
            .map_err(|e| LiveWsError::Send(e.to_string()))
    }

    /// Blocks until the next text frame arrives. Silently skips
    /// ping/pong/binary/raw frames (nothing in the Live API protocol this
    /// migration models needs them) and returns `Ok(None)` on a clean
    /// close.
    pub fn receive_text(&self) -> Result<Option<String>, LiveWsError> {
        let mut socket = self.socket.lock().expect("Live WebSocket mutex poisoned");
        loop {
            match socket.read() {
                Ok(Message::Text(text)) => return Ok(Some(text.to_string())),
                Ok(Message::Close(_)) => return Ok(None),
                Ok(_) => continue,
                Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                    return Ok(None)
                }
                Err(e) => return Err(LiveWsError::Receive(e.to_string())),
            }
        }
    }

    /// Closes the connection with a normal-closure frame.
    pub fn close(&self) -> Result<(), LiveWsError> {
        self.socket
            .lock()
            .expect("Live WebSocket mutex poisoned")
            .close(None)
            .map_err(|e| LiveWsError::Close(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawns a one-shot local WebSocket server: accepts a single
    /// connection, echoes back every text frame it receives until the
    /// client closes.
    fn spawn_echo_server() -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            loop {
                match socket.read() {
                    Ok(Message::Text(text)) => {
                        if socket.send(Message::Text(text)).is_err() {
                            break;
                        }
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
        });
        (format!("ws://{addr}"), handle)
    }

    #[test]
    fn sends_and_receives_a_text_frame_round_trip() {
        let (url, server) = spawn_echo_server();
        let connection = LiveWsConnection::connect(&url).unwrap();
        connection.send_text("hello".to_string()).unwrap();
        let reply = connection.receive_text().unwrap();
        connection.close().unwrap();
        server.join().unwrap();
        assert_eq!(reply.as_deref(), Some("hello"));
    }

    #[test]
    fn receive_text_returns_none_after_the_server_closes() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            let _ = socket.close(None);
        });

        let connection = LiveWsConnection::connect(&format!("ws://{addr}")).unwrap();
        let reply = connection.receive_text().unwrap();
        server.join().unwrap();
        assert_eq!(reply, None);
    }

    #[test]
    fn connect_fails_against_an_address_nothing_is_listening_on() {
        // Port 1 requires privileges no test process has, so nothing is
        // ever listening there.
        let result = LiveWsConnection::connect("ws://127.0.0.1:1");
        assert!(result.is_err());
    }
}
