# Implementation Plan: Branch protection on main

**Spec**: `specs/branch-protection.md`
**Created**: 2026-08-05
**Status**: approved

## Components

### 1. Branch protection API call
- **Purpose**: Require the Pylint build check before merging into `main`
- **Files**: none (GitHub repo settings, not a versioned file)
- **Effort**: XS

## Dependencies

### Build Order

**Must run last**, after #18 (Python 3.13 upgrade) lands, so the required
check context reflects the real job name (`build (3.13)`) instead of the
stale `build (3.12)`.

### External Dependencies

`gh api` with sufficient repo admin permissions.

## Risks & Assumptions

### Risks

- **Risk**: this is a live change to shared repo settings (not a file in
  the PR diff), so it isn't reviewable the way a code change is.
  **Mitigation**: confirm the exact check context name against the actual
  post-#18 CI run before calling the API, and report the result back
  clearly (this is the one task in the batch that isn't "just a commit").

### Assumptions

- No other collaborators currently need PR review gating (solo repo) —
  revisit `required_pull_request_reviews` if that changes.

## Milestones

- [ ] Milestone 1: protection active, confirmed via `gh api .../
      protection` GET

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Confirm the exact check context name from a real CI run
      after #18's matrix bump (`build (3.13)`)
  - **Acceptance**: name confirmed via `gh api repos/.../commits/HEAD/
    check-runs`, not assumed
  - **Files**: none
  - **Tests**: n/a
  - **Effort**: XS

### Integration (Build Second)

- [ ] **Task 2**: `gh api ... branches/main/protection --method PUT`
      with `required_status_checks.contexts = ["build (3.13)"]`,
      `strict: true`, `enforce_admins: false`,
      `required_pull_request_reviews: null`
  - **Acceptance**: subsequent GET on the same endpoint returns the
    configured protection instead of 404
  - **Files**: none
  - **Tests**: `gh api .../protection` GET
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: < 0.1 day
