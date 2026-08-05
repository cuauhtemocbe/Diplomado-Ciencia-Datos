# Implementation Plan: data_analysis_octopus refactor arc

**Spec**: `specs/data-analysis-octopus-refactor.md`
**Created**: 2026-08-05
**Status**: approved

## Components

### 1. Notebook import standardization (#19)
- **Purpose**: Replace wildcard/mixed imports of `data_analysis_octopus`
  across 5 notebooks with `import data_analysis_octopus as dao`, and
  prefix call sites with `dao.`.
- **Files**: `notebooks/2-Cardiotocography.ipynb`,
  `notebooks/6-Whatsapp.ipynb`,
  `notebooks/7-Clasificación-clientes.ipynb`,
  `notebooks/9-Electrocardiograma.ipynb`,
  `notebooks/10-Proyecto-Hipertension-Mexico.ipynb`,
  `tests/test_notebook_imports.py` (new)
- **Effort**: M

### 2. Unit tests for pure stats helpers (#21)
- **Purpose**: Add `tests/test_data_analysis_octopus.py` covering
  `detect_outliers_iqr`, `transform_outliers`, `process_outliers`,
  `count_percentage`, `create_feature_dataframe`, `get_information_value`.
- **Files**: `tests/test_data_analysis_octopus.py` (new)
- **Effort**: M

### 3. Extract DataViz submodule (#22)
- **Purpose**: Convert `src/data_analysis_octopus.py` into a package with
  `DataViz` moved to `viz.py`, re-exported from `__init__.py`.
- **Files**: `src/data_analysis_octopus/__init__.py` (new, replaces the
  file), `src/data_analysis_octopus/viz.py` (new)
- **Effort**: M

## Dependencies

### Build Order

1. **#19** (imports) — independent of #21/#22; touches notebooks only, no
   changes to `data_analysis_octopus.py` itself.
2. **#21** (unit tests) — independent of #19's notebook edits; adds a test
   file only. Sequenced second so it exists as a regression net *before*
   #22 reorganizes the module.
3. **#22** (module split) — depends on #21 being in place first (per its
   own Technical Context: "depends on the unit-test coverage from #21
   landing first" for the *follow-up* story, but the extraction itself is
   safer done with #21's tests already passing against the pre-split
   module, so they can be re-run unchanged against the post-split module
   as the regression check).

There is no hard technical dependency between #19 and #21/#22 — they touch
disjoint files (notebooks vs. `src/data_analysis_octopus.py` vs. new test
files). Sequencing is for risk management (test safety net before module
surgery), not because the code requires it.

### External Dependencies

None beyond what's already in `pyproject.toml` (`jupyterlab` for
`nbconvert`, `pytest`, `nbformat`).

## Risks & Assumptions

### Risks

- **Risk**: Converting a notebook's wildcard-imported call sites to
  `dao.`-prefixed calls requires enumerating every function name actually
  used per notebook — missing one causes a silent `NameError` at execution
  time, not a lint error. **Mitigation**: the execution scenario in #19's
  Gherkin (`nbconvert --execute`) catches this directly; run it for every
  converted notebook, not just the two in the Examples table.
- **Risk**: `src/data_analysis_octopus.py` → package conversion could
  break an import if any of the 12 dependent notebooks imports a symbol in
  a way not covered by the three tested styles. **Mitigation**: #22's
  Gherkin already parametrizes execution across notebooks from all three
  import styles; extend that check to all 12 dependent notebooks, not just
  the 3 in the Examples table, before considering #22 done.
- **Risk**: Heavier Poetry groups (bio, explain, geo, nlp, tensorflow)
  needed to actually execute several of the affected notebooks — building
  all six Docker images is slow. **Mitigation**: accepted cost, already
  paid once in #20's implementation; images are cached locally after first
  build.

### Assumptions

- `nbconvert` (bundled with `jupyterlab`) is sufficient for #19's execution
  scenarios without adding `nbval`/`papermill`/`testbook` — confirmed: no
  new dependency needed.
- Poetry package-mode is not in use for `src/` (no `packages =` directive
  in `pyproject.toml`), so #22's package conversion needs no build-config
  change — confirmed by inspecting `pyproject.toml`.

## Milestones

- [ ] Milestone 1: #19 done — no wildcard imports remain; all 5 converted
      notebooks execute cleanly via `nbconvert` in their respective
      containers; `tests/test_notebook_imports.py` passes.
- [ ] Milestone 2: #21 done — `tests/test_data_analysis_octopus.py` passes
      with all 6 Gherkin scenarios covered.
- [ ] Milestone 3: #22 done — `DataViz` importable from
      `data_analysis_octopus.viz`; all three legacy import styles still
      resolve; all 12 dependent notebooks' import cells execute without
      error across their respective containers.

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Enumerate exact `dao.`-prefixed call sites needed for
      each of the 5 wildcard-import notebooks
  - **Acceptance**: A per-notebook list of every `data_analysis_octopus`
    function/class name actually referenced (bare, pre-conversion)
  - **Files**: none (analysis only, informs Task 2)
  - **Tests**: n/a
  - **Effort**: S

### Features (Build Second)

- [ ] **Task 2**: Convert the 5 wildcard-import notebooks to
      `import data_analysis_octopus as dao` + `dao.`-prefixed calls
  - **Acceptance**: #19 Gherkin scenarios 1–3 pass
  - **Files**: the 5 notebooks listed in Component 1
  - **Tests**: `tests/test_notebook_imports.py` (new)
  - **Effort**: M

- [ ] **Task 3**: Write `tests/test_notebook_imports.py`
  - **Acceptance**: no-wildcard scenario, no-duplicate-import scenario,
    `dao.`-prefix scenario, and nbconvert-execution scenario all pass
  - **Files**: `tests/test_notebook_imports.py`
  - **Tests**: is itself the test file
  - **Effort**: S

- [ ] **Task 4**: Write `tests/test_data_analysis_octopus.py`
  - **Acceptance**: all 6 Gherkin scenarios from #21 pass
  - **Files**: `tests/test_data_analysis_octopus.py`
  - **Tests**: is itself the test file
  - **Effort**: M

### Integration (Build Third)

- [ ] **Task 5**: Convert `src/data_analysis_octopus.py` into a package
      (`__init__.py` + `viz.py`)
  - **Acceptance**: all 5 Gherkin scenarios from #22 pass; #21's tests
    still pass unchanged against the new package layout
  - **Files**: `src/data_analysis_octopus/__init__.py`,
    `src/data_analysis_octopus/viz.py`
  - **Tests**: extend `tests/test_data_analysis_octopus.py` with the
    `DataViz` identity check; verify all 12 dependent notebooks' import
    cells across their containers
  - **Effort**: M

### Verification (Build Fourth)

- [ ] **Task 6**: Full-suite verification across all six Docker services
  - **Acceptance**: `poetry run pytest` passes/skips cleanly in core, bio,
    geo, explain, nlp, and tensorflow containers; lint/black/isort green
  - **Files**: none
  - **Tests**: full `tests/` suite
  - **Effort**: S

## Effort Estimate

**Total Estimated Days**: 4–6 days (matches the M+M+M effort already
assigned per-issue)

| Phase | Effort |
|-------|--------|
| Foundation (Task 1) | 0.5 day |
| Features (Tasks 2–4) | 3–4 days |
| Integration (Task 5) | 1–1.5 days |
| Verification (Task 6) | 0.5 day |
