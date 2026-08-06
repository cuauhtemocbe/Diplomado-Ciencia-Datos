---
title: Upgrade Python base image from frozen 3.12.6-slim to 3.13-slim
status: completed
created: 2026-08-05
updated: 2026-08-05
issue: #18
---

# Upgrade Python base image from frozen 3.12.6-slim to 3.13-slim

## Objective

Move the dev Docker image off the frozen `python:3.12.6-slim` tag onto the
actively-maintained `python:3.13-slim`, and align CI's Python matrix to
match, so the repo keeps receiving OS security patches.

## Context

`Dockerfile.dev` line 1 is pinned to `python:3.12.6-slim` — a frozen patch
tag that stops receiving OS security updates once superseded.
`pyproject.toml` already declares `python = "^3.12"` (Poetry caret =
`>=3.12,<4.0`), so no constraint change is needed to allow 3.13.
`.github/workflows/pylint.yml` line 10 pins `python-version: ["3.12"]` and
must move to `"3.13"` so CI lints against the same version the container
runs.

Note: the issue body also mentions a production `Dockerfile`
(gunicorn/prod image) — that file no longer exists (removed along with
`docker-compose.prod.yml`, since `app_clustering` is local-only, never
deployed; see project memory). Only `Dockerfile.dev` is in scope.

Highest risk in this repo vs. a plain upgrade: five optional Poetry groups
pull in heavy/native-wheel-sensitive packages (tensorflow, torch via
sentence-transformers, geopandas, wfdb, neurokit2, shap), plus xgboost
unconditionally in main. Any of these may not yet publish 3.13 wheels.

**Resolved during implementation**: the `core` build initially failed —
not an optional-group issue, but `numpy = "^1.25.0"` resolving to
`numpy==1.26.4`, which has no `cp313` wheel and fails to compile from
source (no C compiler in the slim image). Fixed by bumping the
`pyproject.toml` constraint to `numpy = "^2.0.0"` and regenerating
`poetry.lock` — numpy 2.4.6 resolved cleanly, and as a side effect
collapsed several duplicate per-python-version marker branches that were
already latent in the old lockfile (it had been silently carrying
`python_version >= "3.13"` alternatives that could never actually
install). `explain-env` (shap 0.46.0, no cp313 wheel, needs a C++
compiler not present in the image) is the one group deferred to a
follow-up per the Out of Scope clause below — see issue #23.

## Requirements

### Functional Requirements

- [x] `Dockerfile.dev` line 1 → `FROM python:3.13-slim AS base`
- [x] `.github/workflows/pylint.yml` matrix → `python-version: ["3.13"]`
- [x] `core` target builds and `poetry install --no-root` succeeds
      (main + dev groups, including unconditional xgboost) — required a
      `numpy` bump to `^2.0.0` (see Context)
- [x] `tensorflow-env`, `nlp-env`, `geo-env`, `bio-env` build and install
      cleanly
- [ ] `explain-env` — **deferred to issue #23** (shap 0.46.0 has no cp313
      wheel, needs a C++ toolchain not present in the image)

### Non-Functional Requirements

- [x] No regression: `poetry run pytest` (40 passed, 10 skipped) and
      `poetry run pylint` (10.00/10) pass inside the `core` image on 3.13

## Architecture

### Components

- `Dockerfile.dev` (one line changed: base image tag)
- `.github/workflows/pylint.yml` (matrix version)

### External Dependencies

None new — verifying existing dependencies resolve on the new interpreter.

## User Stories

Full user story and Gherkin acceptance criteria live in GitHub Issue
**#18**.

## Testing Strategy

### Verification (infra change, no new test suite)

- `docker build -f Dockerfile.dev --target core .` succeeds
- `poetry run pytest` and `poetry run pylint $(git ls-files '*.py')`
  pass inside the built `core` image
- `docker build -f Dockerfile.dev --target <group>-env .` succeeds for
  each of tensorflow, nlp, geo, bio, explain
- CI run on `.github/workflows/pylint.yml` (3.13 matrix) is green

## Boundaries & Constraints

### In Scope

- `Dockerfile.dev` base image bump
- CI Python matrix bump
- Verifying all six build targets (core + 5 groups) still resolve/import

### Out of Scope

- Any change to `pyproject.toml` version constraints (already permits 3.13)
- Adding a HEALTHCHECK to the Dockerfile (not requested, no current probe
  either)
- If a specific optional group's wheels aren't available yet for 3.13,
  fixing that group is a follow-up issue, not a blocker for the groups
  that do work

### Technical Constraints

- Must build/verify via Docker, not bare host Poetry (host Poetry/Python
  crashes on sdist extraction — see project memory)

## Success Criteria

- [x] `Dockerfile.dev` and CI both reference `3.13`
- [x] `core` target + 4 of 5 optional group targets build successfully
      (`explain-env` deferred to #23)
- [x] `poetry run pytest` passes in `core`
- [ ] CI green on the updated matrix (verify on next push)
- [ ] Issue #18 closed with evidence

## Implementation Plan

See `specs/python-3.13-upgrade-plan.md`.
