# Changelog

All notable changes to this repo are documented here.
Format: Added / Changed / Deprecated / Removed / Fixed / Security, newest first.

## [Unreleased]
### Added
- Standard governance file set (repo-config): README, ARCHITECTURE, CONTRIBUTING,
  CODE_OF_CONDUCT, SECURITY, RELEASE_NOTES, ADR seed, PR/issue templates,
  `.gitattributes`.
- `capability-manifest.md`: full 831-row capability roadmap for the
  google/adk-python migration, grouped into 17 dependency-ordered phases.
- Cargo workspace + Phase 1 crates (`adk-platform`, `adk-errors`,
  `adk-events`), implementing capabilities C0001-C0033 (minus C0022/C0023,
  partially blocked on Phase 3) against `rusty_uuid`/`rusty_time`/
  `rusty_err`/`rusty_serde` sibling crates.
- `adk-agents` crate (Phase 2 batch 1): `BaseAgent`, `Context`/
  `ReadonlyContext`/`InvocationContext`, `RunConfig`, `LiveRequestQueue`,
  and supporting types (C0035-C0078 minus C0043/C0047/C0059/C0060/C0066/
  C0069/C0071, deferred pending later phases). Adopts `rusty_tokio` as the
  async runtime.
- 94 capability rows (C0833-C0926) filling the `runners.py` inventory gap
  flagged at row C0788.
### Changed
### Fixed
### Security

<!-- ## [0.1.0] - YYYY-MM-DD
### Added
- Initial release -->
