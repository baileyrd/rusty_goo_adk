# Architecture

## Overview
A Rust reimplementation of [google/adk-python](https://github.com/google/adk-python),
an agent framework covering agent/flow orchestration, LLM model backends, tool
execution (including sandboxed code execution), session/memory/artifact
persistence, evaluation, auth, and a CLI. This repo mirrors that capability
surface rather than the source's module layout — Rust idiom decides the shape,
not a 1:1 file-for-file port. Not a goal: matching adk-python's internal Python
class hierarchy where Rust's trait/ownership model suggests something different;
behavior parity is the bar, not implementation-detail parity.

Migration is tracked capability-by-capability in `capability-manifest.md`
(rust-migration skill) against one GitHub issue per capability.

## Boundaries
<!-- Domain logic vs. I/O and framework details (ports-and-adapters).
     List the ports (interfaces) and the adapters that implement them. -->

| Port | Adapter(s) | Notes |
| ---- | ---------- | ----- |
|      |            |       |

No code has landed yet, so this table stays empty rather than pre-declaring
ports that don't exist. Expected shape, once ported: an LLM-backend port (one
adapter per model provider), a session/memory/artifact-store port (in-memory
first, persistent backends after), and a tool-execution port (native tools vs.
sandboxed code execution) — filled in for real as each lands, not asserted here
in advance.

## Structure
Modular monolith to start: one Cargo workspace, crates split along the source's
own capability boundaries (agents, tools, flows, sessions, models, ...) as they're
ported, not a premature split into separate services. Ports-and-adapters keeps
domain/orchestration logic free of I/O specifics (LLM HTTP calls, storage
backends, sandboxed execution) — per repo-config's generic greenfield default;
no more specific requirement was found in `rusty_foundation_akb` or
`Atlas_Engineering_Standards_Library` at time of writing (both are early-stage
per their own docs — re-check as they mature). A component gets split into its
own service only for a concrete forcing function (independent scaling, a
team/language boundary, hard fault isolation) — none identified yet.

## Data flow
<!-- Diagram or short walkthrough of a request/event through the system -->
Fills in once the first agent-invocation path is ported end-to-end.

## Key decisions
See [docs/adr/](./docs/adr/) for the record of individual decisions and their tradeoffs.

## Non-goals
- Byte-for-byte transliteration of adk-python's Python source — idiomatic Rust
  (ownership, `Result`/`?`, traits over duck typing) is how capabilities are
  reimplemented, not a constraint to work around.
- Anything explicitly marked OUT-OF-SCOPE in `capability-manifest.md`, each with
  a written, user-attributed reason (see the rust-migration skill's boundary
  contract) — nothing is dropped by default or by omission.
