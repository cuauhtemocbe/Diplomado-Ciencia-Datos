---
title: data_analysis_octopus notebook-import hygiene, test coverage, and module split
status: approved
created: 2026-08-05
updated: 2026-08-05
issue: #19, #21, #22
---

# data_analysis_octopus notebook-import hygiene, test coverage, and module split

## Objective

Make `src/data_analysis_octopus.py` — the shared helper imported by 12 of the
repo's 20 notebooks — safer to change: standardize how notebooks import it,
add unit tests for its pure statistical helpers (currently zero coverage),
then extract its self-contained plotting class (`DataViz`) into its own
submodule, all without breaking any of the 12 dependent notebooks.

## Context

`data_analysis_octopus.py` is a 967-line, single-file grab-bag mixing a
plotting class (`DataViz`), pure stats/outlier helpers, and sklearn
model-training helpers. Three problems compound each other:

1. Notebooks import it three different, inconsistent ways (wildcard
   `import *`, `import ... as dao/octo`, explicit tuple import), making it
   impossible to tell what a notebook actually depends on without executing
   it.
2. The module has no test coverage at all — a regression in shared logic
   silently changes results across all 12 dependent notebooks.
3. The file mixes visualization, stats, and modeling concerns in one place,
   which is itself a byproduct of (1) and (2) never having been fixed: no
   test safety net has existed to refactor against.

This spec sequences three GitHub issues, already fully specified with
Gherkin acceptance criteria, as one arc: fix the imports first (#19), add
the test safety net second (#21), then perform the first safe extraction
slice of the module third (#22). Each step was chosen, via a Dry-Run
Review Gate against the real code, to be the smallest safe unit of change —
see each issue for the full verification detail (line numbers, grep
results, confirmed runtime behavior).

## Requirements

### Functional Requirements

- [ ] Every notebook importing `data_analysis_octopus` uses a single,
      explicit `import data_analysis_octopus as dao` statement — no
      wildcard imports remain (issue #19).
- [ ] `detect_outliers_iqr`, `transform_outliers`, `process_outliers`,
      `count_percentage`, `create_feature_dataframe`, and
      `get_information_value` each have unit tests with known
      inputs/outputs (issue #21).
- [ ] `DataViz` is extracted into `src/data_analysis_octopus/viz.py`,
      re-exported from `src/data_analysis_octopus/__init__.py`, with zero
      changes required to any notebook's import statement (issue #22).

### Non-Functional Requirements

- [ ] Reproducibility: none of the three changes may alter the runtime
      behavior of any of the 12 dependent notebooks — verified per-notebook
      by executing (or import-smoke-testing) each one inside its own
      Poetry-group Docker container.
- [ ] Test isolation: new tests must skip (not fail) when a notebook's
      Poetry group isn't installed in the current container, matching the
      existing convention in `tests/test_notebook_dependencies.py`.

## Architecture

### Components

- `tests/test_notebook_imports.py` (new, #19) — parses each notebook's
  code cells (same JSON-cell-scanning approach as
  `tests/test_notebook_dependencies.py`) to assert no wildcard import
  remains and that converted notebooks execute cleanly via `nbconvert`.
- `tests/test_data_analysis_octopus.py` (new, #21) — unit tests for the six
  target pure functions, using small hand-built `pandas.DataFrame` fixtures.
- `src/data_analysis_octopus/__init__.py` + `src/data_analysis_octopus/viz.py`
  (new, #22) — converts the single file into a package, following the
  `src/app_clustering/` precedent (package under `src/`, no `pyproject.toml`
  packaging changes needed since Poetry package-mode isn't used here).

### Data Model

Not applicable — no persisted data model changes; all changes are to
Python source, notebook cells, and test files.

### External Dependencies

None new. `nbconvert` (used by #19's execution scenarios) ships with the
already-installed `jupyterlab` dev dependency.

## User Stories

Full User Story, Technical Context, Gherkin Acceptance Criteria, and
Definition of Done for each piece live in their GitHub Issues (created via
`/user-stories`, already dry-run-reviewed against the real code):

- **#19** — Standardize data_analysis_octopus imports across notebooks
- **#21** — Add unit tests for data_analysis_octopus's outlier and stats helpers
- **#22** — Extract DataViz into its own submodule within data_analysis_octopus

This spec does not repeat that content; see the Implementation Plan
(`specs/data-analysis-octopus-refactor-plan.md`) for how the three are
sequenced and verified together.

## Testing Strategy

### Unit Tests

`tests/test_data_analysis_octopus.py` covers the six pure functions per
#21's Gherkin scenarios (happy path, IQR boundary, zero-variance column,
percentage-sums-to-100, first-row-only characterization, zero-event
category).

### Integration Tests

`tests/test_notebook_imports.py` covers #19's import-hygiene scenarios
(no wildcard, no duplicate import statement, calls go through `dao.`
namespace) plus notebook-execution scenarios via `nbconvert`.

### Regression Tests

For #22, the existing `tests/test_notebook_dependencies.py` suite plus a
new identity check (`data_analysis_octopus.DataViz is
data_analysis_octopus.viz.DataViz`) serve as the regression net proving the
extraction didn't change any public import surface.

### Manual Verification

Every change in this arc is additionally verified by running the full test
suite inside all six `docker-compose.dev.yml` services (core, bio, geo,
explain, nlp, tensorflow) — not just core — since notebook coverage spans
every Poetry group.

## Boundaries & Constraints

### In Scope

- Import-statement standardization in the 12 notebooks that import
  `data_analysis_octopus` (#19).
- Unit tests for the 6 pure stats/outlier functions listed above (#21).
- Extracting `DataViz` only — the smallest safe slice (#22).

### Out of Scope

- `15-AirBnb`'s `!wget` fetch of the module from GitHub raw (a separate
  Colab-portability issue, not touched by #19).
- `4-Restaurantes`'s local redefinition of `count_percentage`/
  `create_feature_dataframe` shadowing its own import (pre-existing
  duplication bug, not addressed here).
- Splitting the remaining stats/modeling functions into further submodules
  (`stats.py`/`modeling.py`) — noted in #22 as a follow-up story, not
  created yet, and depends on #21 landing first.
- Fixing `get_information_value`'s `inf`/`NaN` behavior on a degenerate
  category — #21 characterizes and locks in current behavior, does not
  change it.

### Technical Constraints

- No `pyproject.toml` changes required for #22 (no `packages =` directive
  in use).
- `xgboost`/`varclushi` remain unconditional module-level imports — this is
  a prior, deliberate decision (documented in `CLAUDE.md` and
  `pyproject.toml`) and is not revisited by this spec.

## Success Criteria

- [ ] All Gherkin scenarios in #19, #21, and #22 have passing automated
      tests.
- [ ] `poetry run pytest` passes (or correctly skips) cleanly in all six
      Docker Compose services.
- [ ] `poetry run pylint $(git ls-files '*.py')`, `black --check`, and
      `isort --check` are green.
- [ ] All three GitHub issues are closed with evidence comments referencing
      the commits/test runs that resolved them.

## Implementation Plan

See `specs/data-analysis-octopus-refactor-plan.md`.
