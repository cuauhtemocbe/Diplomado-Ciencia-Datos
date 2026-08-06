---
title: "CI: pin third-party actions by SHA and add explicit permissions"
status: approved
created: 2026-08-05
updated: 2026-08-05
issue: #8
---

# CI: pin third-party actions by SHA and add explicit permissions

## Objective

Pin `actions/checkout` and `actions/setup-python` in
`.github/workflows/pylint.yml` to immutable commit SHAs (not floating
version tags), and add an explicit minimal `permissions:` block, closing
the same supply-chain vector as the 2025 `tj-actions/changed-files`
compromise (a retagged, malicious action).

## Context

`.github/workflows/pylint.yml` currently uses `actions/checkout@v4` and
`actions/setup-python@v5` — mutable tags that can be repointed to
different (potentially malicious) code without consumers noticing. The
workflow also has no `permissions:` block, so `GITHUB_TOKEN` runs with the
repo's default (potentially broad) permissions instead of the minimum
pylint actually needs (read-only).

## Requirements

### Functional Requirements

- [ ] `actions/checkout` pinned to its `v4.4.0` commit SHA, with a
      `# v4.4.0` trailing comment
- [ ] `actions/setup-python` pinned to its `v5` commit SHA, with a
      trailing version comment
- [ ] Workflow-level `permissions: contents: read` added

### Non-Functional Requirements

- [ ] Security: no floating tags for third-party actions in this workflow

## Architecture

### Components

- `.github/workflows/pylint.yml` (SHA pins + permissions block)

### External Dependencies

None.

## User Stories

Full Gap description and suggested diff live in GitHub Issue **#8**.

## Testing Strategy

### Verification (infra change)

- Next CI run on `.github/workflows/pylint.yml` completes successfully
  with the pinned SHAs
- `permissions: contents: read` is sufficient — checkout + setup-python +
  pytest/pylint need no write access

## Boundaries & Constraints

### In Scope

- Pinning the two actions currently used in `pylint.yml`
- Adding the workflow-level `permissions:` block

### Out of Scope

- Setting up Dependabot to keep the SHA pins current — that's issue #9,
  a coordinated follow-up, not part of this change
- Job-level permission overrides (single job, workflow-level is
  sufficient)

### Technical Constraints

- SHAs must resolve to the exact tag noted in the trailing comment at
  time of pinning (verified via `gh api repos/actions/checkout/git/
  ref/tags/<tag>` or the GitHub UI) — do not hand-guess a SHA

## Success Criteria

- [ ] Both actions pinned by SHA with version comments
- [ ] `permissions: contents: read` present at workflow level
- [ ] CI run green after the change
- [ ] Issue #8 closed with evidence

## Implementation Plan

See `specs/ci-actions-pinning-plan.md`.
