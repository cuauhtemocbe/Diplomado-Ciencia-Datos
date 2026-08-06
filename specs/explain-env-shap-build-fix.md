---
title: Fix explain group (shap) cp313 wheel gap on Python 3.13
status: completed
created: 2026-08-06
updated: 2026-08-06
issue: #23
---

# Fix explain group (shap) cp313 wheel gap on Python 3.13

## Objective

Make the `explain-env` Docker target build cleanly on `python:3.13-slim` by
resolving `shap` to a release that ships a wheel compatible with Python
3.13, instead of falling back to a from-source compile that fails for lack
of a C++ toolchain in the image.

## Context

Issue #18 (Python 3.13 upgrade) moved `Dockerfile.dev`'s base image to
`python:3.13-slim` and got `core`, `tensorflow-env`, `nlp-env`, `geo-env`,
and `bio-env` all building — `explain-env` was explicitly deferred because
`shap = "^0.46.0"` (`pyproject.toml`) resolves to `shap==0.46.0`, which has
no `cp313` wheel on PyPI. Poetry falls back to building from the sdist, and
the image has no C++ compiler (`Dockerfile.dev` only installs `git` via
apt), so the build fails with `error: command 'g++' failed: No such file
or directory`.

Two options were on the table (per the issue): add `build-essential` to the
`explain-env` stage, or bump the `shap` pin to a version with cp313 wheels.
Checked PyPI directly (`pypi.org/pypi/shap/<version>/json`):

- `shap` 0.47.0–0.47.2 and 0.52.0 (latest as of this writing): no `cp313`
  wheel on Linux — 0.52.0 ships `cp312-abi3` wheels instead (forward
  -compatible with 3.13 via the stable ABI, so it installs fine on 3.13
  without needing a `cp313`-tagged wheel).
- `shap` 0.48.0, 0.50.0, 0.51.0: ship explicit `cp313-cp313-manylinux*`
  wheels for x86_64/aarch64.
- `shap`'s own dependency on `numba>=0.54` (unconstrained upper bound)
  resolves to `numba==0.66.0`, which does publish a `cp313` manylinux
  wheel — so the transitive dependency chain isn't a blocker either.

A version bump avoids adding a compiler toolchain to the image (smaller,
matches the pattern already used to dodge `explain-env` bloat) and is
consistent with how the numpy/cp313 gap was resolved during #18 (bump the
pin, don't add build tooling).

## Requirements

### Functional Requirements

- [ ] `pyproject.toml`'s `shap` pin moves off `^0.46.0` to a version whose
      PyPI release ships a Linux wheel installable on Python 3.13 (either
      an explicit `cp313` wheel or a forward-compatible `abi3` wheel)
- [ ] `poetry.lock` is regenerated (via Docker, not bare host Poetry — see
      Technical Constraints) to reflect the new `shap` resolution
- [ ] `docker build -f Dockerfile.dev --target explain-env .` completes
      successfully with no compiler-toolchain workaround needed

### Non-Functional Requirements

- [ ] No `build-essential`/compiler toolchain added to any Dockerfile.dev
      stage (avoids the image-size cost across all 6 targets that sharing
      `core-deps` would otherwise impose)

## Architecture

### Components

- `pyproject.toml` (`shap` version constraint under `[tool.poetry.group.explain.dependencies]`)
- `poetry.lock` (regenerated lockfile)

### External Dependencies

- `shap`: bumped from `^0.46.0` to a cp313-wheel-compatible version

## User Stories

Full Gherkin acceptance criteria live in GitHub Issue **#23**.

## Testing Strategy

### Verification (infra change, no new test suite)

- `docker build -f Dockerfile.dev --target explain-env .` succeeds
- `poetry run pytest` and `poetry run pylint $(git ls-files '*.py')` still
  pass inside the `core` target (regression check — the shap bump only
  touches the `explain` group, but the lockfile regen touches the shared
  file)
- `import shap` and `import nltk` succeed inside a container built from
  `explain-env`

## Boundaries & Constraints

### In Scope

- `shap` version bump in `pyproject.toml` + `poetry.lock` regen
- Verifying `explain-env` builds and both its packages (`shap`, `nltk`)
  import cleanly

### Out of Scope

- Adding `build-essential`/gcc/g++ to any Dockerfile.dev stage
- Any change to notebook 10 (`10-Proyecto-Hipertension-Mexico.ipynb`) or
  notebook 6 (`6-Whatsapp.ipynb`) code itself — this issue only unblocks
  their Docker environment, doesn't touch their content
- Re-running/re-verifying notebooks 6 or 10 end-to-end (no test coverage
  requirement for notebook *output* in this repo — see CLAUDE.md)

### Technical Constraints

- Must regenerate `poetry.lock` via Docker, not bare host Poetry (host
  Poetry/Python crashes on sdist extraction — see project memory
  `feedback_no_local_poetry`)

## Success Criteria

- [ ] `docker build -f Dockerfile.dev --target explain-env .` succeeds
- [ ] `core` target's `poetry run pytest` + `poetry run pylint` still pass
      after the lockfile regen
- [ ] Issue #23 closed with build evidence

## Implementation Plan

See `specs/explain-env-shap-build-fix-plan.md`.
