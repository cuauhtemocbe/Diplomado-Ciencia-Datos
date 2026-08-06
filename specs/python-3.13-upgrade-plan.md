# Implementation Plan: Python 3.13 upgrade

**Spec**: `specs/python-3.13-upgrade.md`
**Created**: 2026-08-05
**Status**: approved

## Components

### 1. Base image + CI matrix bump
- **Purpose**: Point `Dockerfile.dev` and CI at Python 3.13
- **Files**: `Dockerfile.dev`, `.github/workflows/pylint.yml`
- **Effort**: XS

### 2. Build verification (core + 5 optional groups)
- **Purpose**: Confirm every Poetry group still resolves/imports on 3.13
- **Files**: none (verification only)
- **Effort**: M (network/build-time bound, not complexity)

## Dependencies

### Build Order

1. Bump `Dockerfile.dev` base image
2. Build + verify `core` target (fast fail if main group breaks)
3. Bump CI matrix (only once core is known-good)
4. Build + verify each optional group target (tensorflow, nlp, geo, bio,
   explain) — independent of each other, can run in any order

### External Dependencies

None — verifying existing pinned versions resolve on the new interpreter.

## Risks & Assumptions

### Risks

- **Risk**: tensorflow/torch (via sentence-transformers) or another
  native-wheel-heavy package has no 3.13 wheel yet. **Mitigation**: build
  `core` first; if a specific optional group fails, file a narrow
  follow-up issue for that group rather than blocking the whole upgrade
  (explicitly allowed by issue #18's acceptance criteria).
- **Risk**: heavy image builds (tensorflow-env, nlp-env) are slow/large
  downloads. **Mitigation**: run sequentially in the background, one-time
  cost.

### Assumptions

- `pyproject.toml`'s `python = "^3.12"` constraint needs no change (already
  permits 3.13).

## Milestones

- [ ] Milestone 1: `core` target builds, `poetry run pytest` +
      `poetry run pylint` pass on 3.13
- [ ] Milestone 2: all 5 optional group targets build successfully
- [ ] Milestone 3: CI green on the 3.13 matrix

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Bump `Dockerfile.dev` line 1 to `python:3.13-slim`
  - **Acceptance**: `docker build --target core` succeeds, `poetry run
    pytest` + `poetry run pylint $(git ls-files '*.py')` pass inside
  - **Files**: `Dockerfile.dev`
  - **Tests**: manual docker build + in-container pytest/pylint
  - **Effort**: XS

### Features (Build Second)

- [ ] **Task 2**: Build and verify each optional group target
  - **Acceptance**: `docker build --target <group>-env` succeeds for
    tensorflow, nlp, geo, bio, explain
  - **Files**: none
  - **Tests**: manual docker build per target
  - **Effort**: M

### Integration (Build Third)

- [ ] **Task 3**: Bump CI matrix to 3.13
  - **Acceptance**: `.github/workflows/pylint.yml` matrix is `["3.13"]`,
    next CI run green
  - **Files**: `.github/workflows/pylint.yml`
  - **Tests**: CI itself
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: ~0.5 day (mostly build/download wait time)

| Phase | Effort |
|-------|--------|
| Foundation | 0.1 day |
| Features | 0.3 day |
| Integration | 0.1 day |
