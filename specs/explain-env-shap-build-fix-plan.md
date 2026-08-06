# Implementation Plan: Fix explain group (shap) cp313 wheel gap

**Spec**: `specs/explain-env-shap-build-fix.md`
**Created**: 2026-08-06
**Status**: approved

## Components

### 1. Bump `shap` pin + regenerate lockfile
- **Purpose**: Point Poetry at a `shap` release with a Python-3.13-installable wheel
- **Files**: `pyproject.toml`, `poetry.lock`
- **Effort**: XS

### 2. Build verification
- **Purpose**: Confirm `explain-env` builds and `core` hasn't regressed
- **Files**: none (verification only)
- **Effort**: S

## Dependencies

### Build Order

1. Bump `shap = "^0.46.0"` to `shap = "^0.48.0"` in `pyproject.toml`
   (0.48.0 is the oldest cp313-wheeled release — narrowest possible diff
   from the current pin, per repo's existing minimal-version-bump pattern)
2. Regenerate `poetry.lock` inside the `core` container (`poetry lock`)
3. Build `explain-env` target, verify `shap`/`nltk` import
4. Re-verify `core` target still builds + passes pytest/pylint (lockfile
   regen touches the shared file, same risk class as the #18 numpy bump)

### External Dependencies

None new — re-pinning an existing dependency.

## Risks & Assumptions

### Risks

- **Risk**: `shap==0.48.0`'s transitive deps (`numba>=0.54`, `slicer==0.0.8`,
  `cloudpickle`, etc.) might not all resolve cleanly against the rest of the
  `explain` group (`nltk`) or main-group pins. **Mitigation**: if `poetry
  lock` fails to resolve, try `^0.50.0` or `^0.51.0` next (also confirmed
  cp313-wheeled) before falling back to option 1 (build-essential).
- **Risk**: lockfile regen could shift versions of unrelated packages
  (already happened with the numpy bump in #18). **Mitigation**: re-run
  `core` target's pytest/pylint after regen to catch regressions.

### Assumptions

- No shap API used by notebook 10/6 (or any src/ code) changed between
  0.46.0 and 0.48.0 in a way that would break existing (untested) notebook
  cells — reasonable given shap's stable public API (`shap.Explainer`,
  `shap.TreeExplainer`, etc.) across these two minor releases.

## Milestones

- [ ] Milestone 1: `poetry lock` resolves with the new `shap` pin
- [ ] Milestone 2: `explain-env` target builds successfully
- [ ] Milestone 3: `core` target pytest/pylint still green after regen

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Bump `shap` pin and regenerate `poetry.lock`
  - **Acceptance**: `pyproject.toml` shows `shap = "^0.48.0"` (or the
    working alternative from the risk mitigation), `poetry lock` completes
    without errors inside the `core` container
  - **Files**: `pyproject.toml`, `poetry.lock`
  - **Tests**: manual `poetry lock` run
  - **Effort**: XS

### Integration (Build Second)

- [ ] **Task 2**: Build and verify `explain-env`
  - **Acceptance**: `docker build -f Dockerfile.dev --target explain-env .`
    succeeds; `import shap; import nltk` works inside the built image
  - **Files**: none
  - **Tests**: manual docker build + import check
  - **Effort**: XS

- [ ] **Task 3**: Regression-check `core`
  - **Acceptance**: `docker build -f Dockerfile.dev --target core .`
    succeeds; `poetry run pytest` and `poetry run pylint $(git ls-files
    '*.py')` pass inside it
  - **Files**: none
  - **Tests**: `make test`, `make lint`
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: ~0.1 day (mostly build wait time)

| Phase | Effort |
|-------|--------|
| Foundation | 0.05 day |
| Integration | 0.05 day |
