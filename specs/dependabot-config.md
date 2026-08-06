---
title: Add Dependabot configuration
status: completed
created: 2026-08-05
updated: 2026-08-05
issue: #9
---

# Add Dependabot configuration

## Objective

Add `.github/dependabot.yml` so the repo's three dependency ecosystems
(`pip` via Poetry, `github-actions`, `docker`) get automated weekly
update PRs, instead of the SHA/digest pins added by #8 becoming silent
technical debt nobody updates by hand.

## Context

No `.github/dependabot.yml` exists today. Issue #9's original suggestion
left the `pip` ecosystem commented out pending issue #4 (`poetry.lock` not
versioned) — **#4 is now closed** (`poetry.lock` is committed and checked
in CI, per commit `e2e8aaa`), so `pip` can be enabled directly rather than
left commented.

## Requirements

### Functional Requirements

- [ ] `.github/dependabot.yml` created with `github-actions`, `docker`,
      and `pip` ecosystems, all weekly
- [ ] `pip` ecosystem enabled (not commented out) since `poetry.lock` is
      now tracked

### Non-Functional Requirements

None beyond the functional scope — this is a static config file.

## Architecture

### Components

- `.github/dependabot.yml` (new file)

### External Dependencies

None — Dependabot is a native GitHub feature, no app install required for
public/default-enabled repos.

## User Stories

Full Gap description lives in GitHub Issue **#9**.

## Testing Strategy

### Verification (infra change, no automated test)

- YAML validates (`python -c "import yaml; yaml.safe_load(open('.github/
  dependabot.yml'))"` or GitHub's own config parse on push)
- GitHub's "Insights → Dependency graph → Dependabot" tab shows all three
  ecosystems registered after the file lands on the default branch

## Boundaries & Constraints

### In Scope

- The three ecosystems listed above, weekly interval, root directory

### Out of Scope

- Auto-merge rules for Dependabot PRs (not requested, would need
  additional CI gating from #10 to be safe anyway)
- Per-group update grouping/schedule tuning — plain weekly is enough for
  a single-maintainer repo

### Technical Constraints

- `pip` ecosystem requires `poetry.lock` present (satisfied — #4 closed)

## Success Criteria

- [ ] `.github/dependabot.yml` committed with all three ecosystems active
- [ ] Dependabot tab shows the ecosystems registered
- [ ] Issue #9 closed with evidence

## Implementation Plan

See `specs/dependabot-config-plan.md`.
