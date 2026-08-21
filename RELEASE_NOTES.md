# Release Notes

<!--
Two variants, pick the one that fits this repo's actual unit of change:

1. No version tags yet (pre-1.0, nothing published) — track by PR instead, same way
   AISF does it: one entry per merged PR against main, reverse chronological, each
   linking to its PR and (where one exists) to the doc that covers the change in full
   detail. Use "## PR #N — <summary>" headers.

2. Actual version tags exist — use "## vX.Y.Z - YYYY-MM-DD" headers instead, each
   linking to the PRs it shipped and a compare link to the previous tag. Add an
   "### Upgrade notes" subsection under any entry with a breaking change.

Either way, keep the tone AISF's file uses: bolded category tags inline in the
bullet (**Added:** / **Changed:** / **Fixed:**), not separate subheaders per
category — and state known limitations or deliberate scope cuts plainly instead of
leaving them implied.
-->

Notable changes to this repo, one entry per merged PR against `main`, newest first.

---

## PR #TBD — Bootstrap governance files; begin rust-migration of google/adk-python
**2026-08-21** · (link added once this PR is opened)

- **Added:** standard governance file set (repo-config) — README, ARCHITECTURE,
  CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG, RELEASE_NOTES, ADR seed,
  PR/issue templates, `.gitattributes` (`eol=lf`).
- **Known limitation:** no Cargo workspace exists yet, so no CI workflow was
  added (repo-config skips CI generation when there's no stack manifest to run
  against) and README's Getting Started section is a placeholder.
- This is the first PR of an ongoing rust-migration-skill loop porting
  [google/adk-python](https://github.com/google/adk-python) to Rust; see
  `capability-manifest.md` (added in a follow-up PR) for the tracked capability
  list.
