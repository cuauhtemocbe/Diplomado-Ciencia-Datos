---
title: Fix stale spec statuses and Python version references
status: completed
created: 2026-08-06
updated: 2026-08-06
issue: #35
---

# Fix stale spec statuses and Python version references

## Objective

Make spec frontmatter and documented Python version claims match reality:
close out two specs whose issues already shipped, and align
`README.md`/`pyproject.toml` with the Python 3.13 runtime `Dockerfile.dev`
already uses.

## Context

- `specs/data-analysis-octopus-refactor.md` (issues #19, #21, #22 — all
  closed) and `specs/poetry-lock-versioning.md` (issue #4 — closed) still
  have `status: approved`; every other spec (`branch-protection`,
  `ci-actions-pinning`, `dependabot-config`, `makefile`,
  `python-3.13-upgrade`) correctly uses `status: completed` once its issue
  closed.
- `README.md:5` says "Python 3.12.3"; `Dockerfile.dev:1` uses
  `python:3.13-slim` (issue #18, closed); `pyproject.toml:9` still pins
  `python = "^3.12"` — which technically permits 3.13 (Poetry caret =
  `>=3.12,<4.0`) but doesn't explicitly reflect it as the supported/minimum
  version, and issue #35's acceptance criteria asks for the constraint to
  "permit it explicitly (not just via an open-ended caret range)".

## Requirements

### Functional Requirements

- [ ] `specs/data-analysis-octopus-refactor.md` frontmatter: `status:
      approved` → `status: completed`
- [ ] `specs/poetry-lock-versioning.md` frontmatter: `status: approved` →
      `status: completed`
- [ ] `README.md`: "Python 3.12.3" → "Python 3.13" (matching
      `Dockerfile.dev`'s `python:3.13-slim` major.minor)
- [ ] `pyproject.toml`: `python = "^3.12"` → `python = "^3.13"` (moves the
      floor to match the actual runtime, still permits future 3.14+ same as
      before — narrower, more accurate lower bound, not a new upper bound)

### Non-Functional Requirements

- [ ] New `tests/test_docs_consistency.py` automates all three checks so
      this can't silently drift again

## Architecture

### Components

- `specs/data-analysis-octopus-refactor.md`, `specs/poetry-lock-versioning.md`
  (frontmatter edits)
- `README.md` (one line)
- `pyproject.toml` (one line)
- `tests/test_docs_consistency.py` (new)

### External Dependencies

None.

## User Stories

Full Gherkin acceptance criteria (3 scenarios: spec statuses, README
version, pyproject constraint) live in GitHub Issue **#35**.

## Testing Strategy

### Unit Tests

New `tests/test_docs_consistency.py`, runnable in `core` (no extra deps
needed — just file parsing):

- Parse YAML frontmatter of every `specs/*.md` file that isn't a
  `*-plan.md`; for each, if a linked GitHub issue is closed... **not
  checkable offline** — instead: assert
  `data-analysis-octopus-refactor.md` and `poetry-lock-versioning.md`
  specifically have `status: completed` (the two names are hardcoded in
  the test since "issue is closed" requires live GitHub API access this
  repo's test suite doesn't have — matching this issue's own scope of
  "these two specific specs")
- Read `Dockerfile.dev`, extract the base image's Python major.minor via
  regex on the `FROM python:X.Y-slim` line; read `README.md`, assert the
  same major.minor string appears
  - Read `pyproject.toml`, assert the `python` constraint's version
  matches (or is compatible with) the same major.minor

## Boundaries & Constraints

### In Scope

- The two named spec frontmatter fixes
- The three named metadata inconsistencies (README, pyproject, and the
  new regression test)

### Out of Scope

- Auditing every other spec file for staleness beyond the two named ones
- Any behavior change — this is metadata/documentation only
- CI changes (already on 3.13 per issue #18)

### Technical Constraints

- Test runs via `make test` (core group — PyYAML or manual frontmatter
  parsing must not require a new dependency; Python's stdlib can split on
  the `---` delimiters and use a minimal manual parse, or reuse whatever
  YAML lib is already available — check before adding a new dependency)

## Success Criteria

- [ ] All 3 Gherkin scenarios have passing automated tests
- [ ] `poetry run pytest` and `poetry run pylint $(git ls-files '*.py')`
      pass inside `core`
- [ ] Issue #35 closed with evidence

## Implementation Plan

See `specs/docs-metadata-consistency-plan.md`.
