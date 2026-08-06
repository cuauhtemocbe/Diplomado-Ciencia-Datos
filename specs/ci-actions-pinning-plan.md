# Implementation Plan: CI actions pinning + permissions

**Spec**: `specs/ci-actions-pinning.md`
**Created**: 2026-08-05
**Status**: approved

## Components

### 1. SHA-pin actions + permissions block
- **Purpose**: Replace floating tags with commit SHAs, add minimal
  `permissions:`
- **Files**: `.github/workflows/pylint.yml`
- **Effort**: XS

## Dependencies

### Build Order

Single-file change, no ordering concerns. Independent of #18's matrix
bump (different lines), can land in the same commit.

### External Dependencies

None.

## Risks & Assumptions

### Risks

- **Risk**: SHA goes stale as soon as a new tag version ships.
  **Mitigation**: accepted — issue #9 (Dependabot `github-actions`
  ecosystem) keeps it current going forward via automated PRs.

### Assumptions

- SHAs quoted in issue #8 (`actions/checkout` → `v4.4.0`,
  `actions/setup-python` → `v5`) are re-verified at implementation time
  via `gh api repos/actions/<repo>/git/ref/tags/<tag>` rather than trusted
  blindly from the issue text.

## Milestones

- [ ] Milestone 1: workflow file updated, CI run green

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Verify current SHAs for `actions/checkout@v4` and
      `actions/setup-python@v5`
  - **Acceptance**: SHA confirmed via `gh api`, not copied blindly
  - **Files**: none
  - **Tests**: n/a
  - **Effort**: XS

### Features (Build Second)

- [ ] **Task 2**: Replace tags with pinned SHAs + version comments, add
      `permissions: contents: read`
  - **Acceptance**: workflow diff matches issue #8's suggested shape
  - **Files**: `.github/workflows/pylint.yml`
  - **Tests**: n/a
  - **Effort**: XS

### Integration (Build Third)

- [ ] **Task 3**: Push and confirm CI run passes with pinned actions
  - **Acceptance**: CI green
  - **Files**: none
  - **Tests**: CI itself
  - **Effort**: XS

## Effort Estimate

**Total Estimated Days**: < 0.1 day
