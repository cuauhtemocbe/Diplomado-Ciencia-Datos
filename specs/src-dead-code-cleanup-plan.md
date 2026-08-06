# Implementation Plan: Remove dead code and misplaced constants in src/

**Spec**: `specs/src-dead-code-cleanup.md`
**Created**: 2026-08-06
**Status**: approved

## Components

### 1. Remove `process_outliers` + its tests
- **Purpose**: Drop an unreferenced function and its dedicated tests
- **Files**: `src/data_analysis_octopus/__init__.py`, `tests/test_data_analysis_octopus.py`
- **Effort**: XS

### 2. Remove duplicate `RANDOM_STATE` constants
- **Purpose**: Drop two unused module-level constants
- **Files**: `src/app_clustering/app.py`, `src/app_clustering/clustering.py`
- **Effort**: XS

### 3. Remove `plot_k_distance`
- **Purpose**: Drop the one genuinely-dead plotting function from `clustering.py`
- **Files**: `src/app_clustering/clustering.py`
- **Effort**: XS

### 4. Move `heatmap`/`plot_heatmap_clusters` into `viz.py`
- **Purpose**: Co-locate visualization code with the rest of `DataViz`,
  re-export for backward compatibility
- **Files**: `src/data_analysis_octopus/__init__.py`, `src/data_analysis_octopus/viz.py`, `tests/test_data_analysis_octopus_package.py`
- **Effort**: S

## Dependencies

### Build Order

All four components are independent of each other (different functions,
non-overlapping file regions except both touching `__init__.py` — apply
sequentially to avoid conflicting edits to the same file):

1. Remove `process_outliers` + tests
2. Remove `RANDOM_STATE` from both `app_clustering` files
3. Remove `plot_k_distance`
4. Move `heatmap`/`plot_heatmap_clusters` to `viz.py` + re-export + tests

### External Dependencies

None.

## Risks & Assumptions

### Risks

- **Risk**: `heatmap`/`plot_heatmap_clusters` need imports (`plotly.express`,
  `plotly.graph_objects`, `matplotlib.pyplot`, `seaborn`,
  `sklearn.preprocessing.MinMaxScaler`) not currently in `viz.py`.
  **Mitigation**: add exactly the imports both functions need; don't touch
  `DataViz`'s existing imports.
- **Risk**: moving functions could shift `__init__.py`'s re-export
  ordering in a way that breaks `from data_analysis_octopus import *`.
  **Mitigation**: mirror the exact `from .viz import DataViz` pattern
  already used for the #22 split; add
  `from .viz import DataViz, heatmap, plot_heatmap_clusters`.

### Assumptions

- No other file in `src/` imports `process_outliers`, `RANDOM_STATE`, or
  `plot_k_distance` directly (only via the grep already run against
  notebooks + src/ + tests — confirmed zero hits beyond the definition
  site and, for `process_outliers`, its own tests).

## Milestones

- [ ] Milestone 1: all four removals/moves applied, `poetry run pytest`
      green inside `core`
- [ ] Milestone 2: `poetry run pylint $(git ls-files '*.py')` still 10.00/10
- [ ] Milestone 3: notebook 13/15 spot-checked unaffected

## Tasks

### Foundation (Build First)

- [ ] **Task 1**: Remove `process_outliers` and its two tests
  - **Acceptance**: `dao.process_outliers` no longer exists;
    `detect_outliers_iqr`/`transform_outliers` tests still pass
  - **Files**: `src/data_analysis_octopus/__init__.py`, `tests/test_data_analysis_octopus.py`
  - **Tests**: `poetry run pytest tests/test_data_analysis_octopus.py`
  - **Effort**: XS

- [ ] **Task 2**: Remove both `RANDOM_STATE` constants
  - **Acceptance**: neither `app.py` nor `clustering.py` defines
    `RANDOM_STATE`; `grep -rn RANDOM_STATE src/` returns nothing
  - **Files**: `src/app_clustering/app.py`, `src/app_clustering/clustering.py`
  - **Tests**: none (no test references the constant); lint pass is the check
  - **Effort**: XS

- [ ] **Task 3**: Remove `plot_k_distance`
  - **Acceptance**: `clustering.plot_k_distance` no longer exists;
    `grep -rn plot_k_distance` (repo-wide) returns nothing
  - **Files**: `src/app_clustering/clustering.py`
  - **Tests**: none exist for this function today
  - **Effort**: XS

### Features (Build Second)

- [ ] **Task 4**: Move `heatmap`/`plot_heatmap_clusters` to `viz.py`
  - **Acceptance**: both functions defined in `viz.py` with needed
    imports added; `__init__.py` re-exports them; `dao.heatmap is
    viz.heatmap` and `dao.plot_heatmap_clusters is
    viz.plot_heatmap_clusters` both True
  - **Files**: `src/data_analysis_octopus/__init__.py`, `src/data_analysis_octopus/viz.py`
  - **Tests**: new assertions in `tests/test_data_analysis_octopus_package.py`
  - **Effort**: S

## Effort Estimate

**Total Estimated Days**: ~0.15 day

| Phase | Effort |
|-------|--------|
| Foundation | 0.1 day |
| Features | 0.05 day |
