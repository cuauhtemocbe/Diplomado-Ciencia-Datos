---
title: Enable branch protection on main
status: approved
created: 2026-08-05
updated: 2026-08-05
issue: #10
---

# Enable branch protection on main

## Objective

Enable branch protection on `main` requiring the Pylint CI check to pass
before merge, so a failing check can no longer be merged around (which is
possible today — `main` has no protection at all).

## Context

`gh api repos/cuauhtemocbe/Diplomado-Ciencia-Datos/branches/main/
protection` currently 404s ("Branch not protected"). The Pylint workflow
runs on every push/PR but doesn't block merges. The originally-suggested
required check context in issue #10 was `build (3.12)`, derived from the
job name (`build`) + matrix (`python-version: ["3.12"]`) in
`pylint.yml` — **this repo's `python-3.13-upgrade` work (issue #18) bumps
that matrix to `3.13`**, so the actual context to require is
`build (3.13)`, confirmed against the CI run after that change lands.

Single-maintainer repo: `enforce_admins: false` (owner can still push
directly when it makes sense) and `required_pull_request_reviews: null`
(no other collaborators to approve) are deliberate, documented exceptions
— not oversights.

## Requirements

### Functional Requirements

- [ ] `main` has branch protection enabled requiring the current Pylint
      build check context (`build (3.13)`, post-#18) to pass
- [ ] `strict: true` (branch must be up to date with `main` before
      merging)
- [ ] `enforce_admins: false`, `required_pull_request_reviews: null`
      (documented single-maintainer exceptions)

### Non-Functional Requirements

None beyond the functional scope.

## Architecture

### Components

- GitHub repo settings (branch protection API), not a versioned file

### External Dependencies

None — uses `gh api`.

## User Stories

Full Gap description and suggested `gh api` call live in GitHub Issue
**#10**.

## Testing Strategy

### Verification (infra change, no automated test)

- `gh api repos/cuauhtemocbe/Diplomado-Ciencia-Datos/branches/main/
  protection` returns the configured protection (no longer 404)
- A PR with a failing Pylint check cannot be merged (verified by
  inspecting the merge button state on an existing/test PR, not by
  deliberately breaking `main`)

## Boundaries & Constraints

### In Scope

- Enabling protection on `main` with the required status check

### Out of Scope

- Requiring PR reviews (no other collaborators)
- Enforcing admins (explicit single-maintainer exception)

### Technical Constraints

- Must run **after** issue #18 lands so the required check context
  (`build (3.13)`) matches the actual CI job name instead of the
  now-stale `build (3.12)`

## Success Criteria

- [ ] Branch protection active on `main`, required check =
      `build (3.13)`
- [ ] `gh api .../protection` no longer 404s
- [ ] Issue #10 closed with evidence

## Implementation Plan

See `specs/branch-protection-plan.md`.
