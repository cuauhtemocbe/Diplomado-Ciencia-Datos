# Implementation Plan: Version poetry.lock

**Spec**: `specs/poetry-lock-versioning.md`
**Created**: 2026-08-05
**Status**: approved

## Components

### 1. Untrack-and-generate lockfile
- **Purpose**: Remove `poetry.lock` from `.gitignore`, generate it fresh
  against current `pyproject.toml`, commit it.
- **Files**: `.gitignore`, `poetry.lock` (new)
- **Effort**: XS

### 2. CI lock-check step
- **Purpose**: Fail CI if `poetry.lock` drifts from `pyproject.toml`.
- **Files**: `.github/workflows/pylint.yml`
- **Effort**: XS

## Dependencies

### Build Order

1. Generate and commit `poetry.lock` first (must exist before CI can check
   it).
2. Add the CI step second.

### External Dependencies

None.

## Risks & Assumptions

### Risks

- **Risk**: `poetry lock` run inside the core container only resolves the
  main+dev groups by default; optional groups (tensorflow/nlp/geo/bio/
  explain) need to be included in the same lock resolution or their
  constraints won't be captured. **Mitigation**: `poetry lock` locks all
  groups declared in `pyproject.toml` regardless of which are installed
  locally — it resolves the full dependency graph, not just installed
  groups. Verify by checking the generated lockfile's `[metadata]` section
  lists all group's packages.
- **Risk**: A 12GB+ dependency graph (tensorflow, torch, sentence-
  transformers) can make `poetry lock` slow. **Mitigation**: accepted
  one-time cost; not a recurring problem since lock only re-runs on
  `pyproject.toml` changes going forward.

### Assumptions

- The current `pyproject.toml` state (CPU-only torch pin, xgboost-cpu,
  etc. — all already merged) is the correct target to lock against.

## Milestones

- [ ] Milestone 1: `poetry.lock` committed, `poetry check --lock` passes
      locally.
- [ ] Milestone 2: CI step added, verified green on a real push.

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Remove `poetry.lock` from `.gitignore`
  - **Acceptance**: line removed, `git status` would track the file once
    generated
  - **Files**: `.gitignore`
  - **Tests**: n/a
  - **Effort**: XS

- [ ] **Task 2**: Generate `poetry.lock` inside the core Docker container
  - **Acceptance**: `poetry lock` exits 0; `poetry check --lock` exits 0
    immediately after
  - **Files**: `poetry.lock` (new)
  - **Tests**: `poetry check --lock` (manual verification, not a pytest)
  - **Effort**: S (network-bound, not complexity-bound)

### Features (Build Second)

- [ ] **Task 3**: Commit `poetry.lock`
  - **Acceptance**: `git ls-files | grep poetry.lock` returns the file
  - **Files**: `poetry.lock`
  - **Tests**: n/a
  - **Effort**: XS

### Integration (Build Third)

- [ ] **Task 4**: Add `poetry check --lock` step to
      `.github/workflows/pylint.yml`
  - **Acceptance**: step present, positioned after `poetry install
    --no-root --with nlp`; CI run shows it passing
  - **Files**: `.github/workflows/pylint.yml`
  - **Tests**: CI itself is the test
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: < 0.5 day

| Phase | Effort |
|-------|--------|
| Foundation | 0.2 day |
| Features | 0.1 day |
| Integration | 0.1 day |
