---
title: Version poetry.lock for reproducible builds
status: approved
created: 2026-08-05
updated: 2026-08-05
issue: #4
---

# Version poetry.lock for reproducible builds

## Objective

Track `poetry.lock` in git and add a `poetry check --lock` CI step, so every
`poetry install` (local, Docker, CI) resolves the exact same dependency
versions instead of re-resolving from scratch each time.

## Context

`.gitignore` currently excludes `poetry.lock` (`git ls-files | grep poetry`
returns nothing) — it has never been committed. This repo's dependency
stack is heavy and breaking-change-sensitive (`tensorflow`, `keras`,
`scikit-learn`, `sentence-transformers`, the CPU-only `torch` pin), so an
unlocked `poetry install` run today vs. next week can silently resolve
different versions and break a notebook or `app_clustering` without any
code change. This also blocks issue #9 (Dependabot): Dependabot's Python
ecosystem support tracks `poetry.lock` for version-bump PRs — without a
committed lockfile, it has nothing to open PRs against.

## Requirements

### Functional Requirements

- [ ] `poetry.lock` is removed from `.gitignore` and committed to the repo.
- [ ] The committed lockfile is generated (`poetry lock`) against the
      current `pyproject.toml`, covering the main group plus all five
      optional groups (tensorflow, nlp, geo, bio, explain) and dev.
- [ ] `.github/workflows/pylint.yml` gains a `poetry check --lock` step
      that fails CI if the lockfile drifts out of sync with
      `pyproject.toml`.

### Non-Functional Requirements

- [ ] Reproducibility: two `poetry install` runs against the same commit
      resolve identical versions (verified by re-running install twice and
      diffing the resolved environment).

## Architecture

### Components

- `poetry.lock` (tracked file, ~generated, not hand-written)
- `.gitignore` (one line removed)
- `.github/workflows/pylint.yml` (one step added)

### External Dependencies

None new — `poetry check --lock` is a built-in Poetry command.

## User Stories

Full Gap description and suggested steps live in GitHub Issue **#4**
("poetry.lock no está versionado (está en .gitignore)"), filed by the
2026-08-04 dev-standards audit. This is a `dev-standards-gap` issue, not a
Gherkin-style user story — it's evaluated by a direct reproducibility check
rather than persona-driven acceptance criteria.

## Testing Strategy

### Verification (no new automated test suite — infra change)

- `poetry check --lock` run locally inside the core Docker container must
  exit 0 immediately after generating and committing the lockfile.
- CI (`.github/workflows/pylint.yml`) must fail if `poetry.lock` and
  `pyproject.toml` are put out of sync (verified by a throwaway local
  edit + revert, not committed).
- `poetry install --no-root` (main + dev) and, separately, each
  `--with <group>` install must succeed against the committed lockfile
  inside their respective Docker targets.

## Boundaries & Constraints

### In Scope

- Committing `poetry.lock`.
- Adding the `poetry check --lock` CI step.

### Out of Scope

- Adding a `make lock-check` Makefile target that wraps the same command
  inside Docker — noted in #4 as a nice-to-have tied to the Makefile issue
  (#7, already completed); can be added later without re-opening this
  spec.
- Enabling Dependabot's `pip` ecosystem (#9) — #4 is an explicit
  prerequisite for #9, not the same change.

### Technical Constraints

- Lockfile generation must happen inside the Docker dev container (`make
  shell` / `diplomado-core`), not on the host — the host's Poetry/Python
  combo is known to crash on sdist extraction (see project memory:
  "Never run poetry bare on host").

## Success Criteria

- [ ] `git ls-files | grep poetry.lock` returns the file.
- [ ] `poetry check --lock` passes locally inside the core container.
- [ ] CI's `poetry check --lock` step is present and green on the next PR.
- [ ] Issue #4 is closed with evidence (commit + CI run).

## Implementation Plan

See `specs/poetry-lock-versioning-plan.md`.
