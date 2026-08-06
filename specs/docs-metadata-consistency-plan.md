# Implementation Plan: Fix stale spec statuses and Python version references

**Spec**: `specs/docs-metadata-consistency.md`
**Created**: 2026-08-06
**Status**: approved

## Components

### 1. Spec frontmatter fixes
- **Purpose**: Mark two completed specs as `completed`
- **Files**: `specs/data-analysis-octopus-refactor.md`, `specs/poetry-lock-versioning.md`
- **Effort**: XS

### 2. README/pyproject version alignment
- **Purpose**: Match documented Python version to the actual runtime
- **Files**: `README.md`, `pyproject.toml`
- **Effort**: XS

### 3. Regression test
- **Purpose**: Prevent this from drifting silently again
- **Files**: `tests/test_docs_consistency.py` (new)
- **Effort**: S

## Dependencies

### Build Order

All three are independent; order doesn't matter, but writing the test
first (against the *current*, stale state) then watching it pass after
the fixes is the cleanest TDD sequence:

1. Write `tests/test_docs_consistency.py` (fails against current state)
2. Fix spec frontmatter
3. Fix README/pyproject
4. Confirm test passes

### External Dependencies

`PyYAML` — already resolved transitively in the lockfile (pulled in by
`jupyterlab`/`nbformat` in the `dev` group, which installs alongside
`main` by default). No new dependency needed.

## Risks & Assumptions

### Risks

- **Risk**: `pyproject.toml`'s `python = "^3.13"` bump could theoretically
  narrow what's installable if anyone still runs this on 3.12.
  **Mitigation**: negligible — `Dockerfile.dev` (the only supported
  environment per CLAUDE.md) is already on `python:3.13-slim`; nobody runs
  bare-host Poetry successfully anyway (see project memory).

### Assumptions

- "Python major.minor matches" means the digits after `python:` and before
  `-slim` in `Dockerfile.dev` line 1 equal the digits Poetry's caret
  constraint starts from, and equal the version string in `README.md`.

## Milestones

- [ ] Milestone 1: new test fails against current (stale) state
- [ ] Milestone 2: all three fixes applied, test passes
- [ ] Milestone 3: `poetry run pytest` full suite green

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Write `tests/test_docs_consistency.py`
  - **Acceptance**: 3 tests (spec statuses, README version, pyproject
    constraint), all initially failing against the current repo state
  - **Files**: `tests/test_docs_consistency.py`
  - **Tests**: itself
  - **Effort**: S

### Features (Build Second)

- [ ] **Task 2**: Fix the two spec frontmatters
  - **Acceptance**: both files' `status:` field reads `completed`
  - **Files**: `specs/data-analysis-octopus-refactor.md`, `specs/poetry-lock-versioning.md`
  - **Tests**: Task 1's spec-status test
  - **Effort**: XS

- [ ] **Task 3**: Fix README + pyproject Python version
  - **Acceptance**: `README.md` says "Python 3.13"; `pyproject.toml` has
    `python = "^3.13"`
  - **Files**: `README.md`, `pyproject.toml`
  - **Tests**: Task 1's README/pyproject tests
  - **Effort**: XS

### Integration (Build Third)

- [ ] **Task 4**: Full suite + lint pass
  - **Acceptance**: `poetry run pytest` and `poetry run pylint $(git
    ls-files '*.py')` both green inside `core`
  - **Files**: none
  - **Tests**: `make test`, `make lint`
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: ~0.1 day

| Phase | Effort |
|-------|--------|
| Foundation | 0.05 day |
| Features | 0.03 day |
| Integration | 0.02 day |
