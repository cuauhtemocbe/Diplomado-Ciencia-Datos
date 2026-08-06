---
title: Remove dead code and misplaced constants in src/
status: completed
created: 2026-08-06
updated: 2026-08-06
issue: #34
---

# Remove dead code and misplaced constants in src/

## Objective

Remove code in `src/` that has zero remaining references anywhere in the
repo (notebooks included), and relocate visualization helpers stranded in
`data_analysis_octopus/__init__.py` after the `viz.py` split (#22), so the
module surface only contains code that's actually live.

## Context

Issue #34's original technical context asserted four functions in
`src/app_clustering/clustering.py` — `plot_wordcloud`,
`plot_sentiment_global`, `create_3d_umap_plot`, `plot_k_distance` — were
"defined but never called from `app.py`" and proposed deleting all four.
That's true but incomplete: it only checked `app.py`, not the notebooks.

**Verified during spec review** (grepping all of `notebooks/*.ipynb`, not
just `src/`): `notebooks/13-Agrupamiento-texto.ipynb` imports `from
app_clustering import clustering` and actively calls
`clustering.plot_wordcloud(...)`, `clustering.plot_sentiment_global(df)`,
and `clustering.create_3d_umap_plot(umap_df)`. Deleting those three would
break that notebook. Only `plot_k_distance` has zero references anywhere
in the repo (not in `app.py`, not in any notebook) — it's the one function
in that list that's actually dead.

**Scope was narrowed accordingly** (user decision, 2026-08-06): only
`plot_k_distance` is removed from `clustering.py`. The other three items
from the original issue stand as verified:

- `process_outliers` (`src/data_analysis_octopus/__init__.py:137-152`):
  zero references across all 22 notebooks (confirmed by grep) — every
  notebook calls `detect_outliers_iqr`+`transform_outliers` separately.
  Note: `tests/test_data_analysis_octopus.py` currently has two tests
  against `process_outliers` (`test_process_outliers_clips_values_above_upper_bound`,
  `test_process_outliers_leaves_values_within_bounds_unchanged`) that must
  be removed alongside the function.
- `RANDOM_STATE = 333` duplicated in `src/app_clustering/app.py:20` and
  `clustering.py:31` — referenced nowhere in either file;
  `transform_embeddings` hardcodes `random_seed=42` instead
  (`clustering.py:881`).
- `heatmap`/`plot_heatmap_clusters` (`__init__.py:704-774`) are
  visualization code stranded in `__init__.py` after `DataViz` was
  extracted to `viz.py` in #22 — both are actively used by
  `notebooks/15-AirBnb.ipynb` (`dao.heatmap(...)`,
  `dao.plot_heatmap_clusters(...)`), so this is a *move*, not a deletion,
  re-using the re-export pattern #22 already established for `DataViz`.

## Requirements

### Functional Requirements

- [ ] `process_outliers` removed from `src/data_analysis_octopus/__init__.py`
- [ ] Its two dedicated tests removed from `tests/test_data_analysis_octopus.py`
- [ ] `RANDOM_STATE = 333` removed from both `src/app_clustering/app.py`
      and `src/app_clustering/clustering.py`
- [ ] `plot_k_distance` removed from `src/app_clustering/clustering.py`
      (only this one function from the original issue's list — see
      Context)
- [ ] `heatmap` and `plot_heatmap_clusters` moved from
      `data_analysis_octopus/__init__.py` to `data_analysis_octopus/viz.py`,
      re-exported from `__init__.py` so `dao.heatmap` /
      `dao.plot_heatmap_clusters` keep resolving unchanged

### Non-Functional Requirements

- [ ] No behavior change to `app_clustering`'s Flask app output
- [ ] No behavior change to notebook 13 or notebook 15 — both keep working
      unmodified against the post-cleanup module surface

## Architecture

### Components

- `src/data_analysis_octopus/__init__.py` (remove `process_outliers`;
  remove `heatmap`/`plot_heatmap_clusters` definitions, add re-export)
- `src/data_analysis_octopus/viz.py` (add `heatmap`, `plot_heatmap_clusters`
  — both need `plotly.express`/`plotly.graph_objects`/`matplotlib.pyplot`/
  `seaborn`/`sklearn.preprocessing.MinMaxScaler` imports added; `viz.py`
  currently only imports matplotlib/pandas/seaborn)
- `src/app_clustering/app.py` (remove unused `RANDOM_STATE`)
- `src/app_clustering/clustering.py` (remove unused `RANDOM_STATE`,
  remove `plot_k_distance`)
- `tests/test_data_analysis_octopus.py` (remove the two `process_outliers`
  tests)
- `tests/test_data_analysis_octopus_package.py` (extend to assert
  `heatmap`/`plot_heatmap_clusters` resolve from `viz.py` and are
  re-exported from `dao`, matching the existing `DataViz` regression-test
  pattern)

### External Dependencies

None new.

## User Stories

Full Gherkin acceptance criteria live in GitHub Issue **#34** — note the
`plot_wordcloud`/`plot_sentiment_global`/`create_3d_umap_plot` scenario is
superseded by the Context section above (scope narrowed to
`plot_k_distance` only).

## Testing Strategy

### Unit Tests

- `tests/test_data_analysis_octopus.py`: remove the two `process_outliers`
  tests; existing `detect_outliers_iqr`/`transform_outliers` tests
  untouched (they don't depend on `process_outliers`)
- `tests/test_data_analysis_octopus_package.py`: add assertions that
  `heatmap`/`plot_heatmap_clusters` are defined in `viz` and that
  `dao.heatmap is viz.heatmap` / `dao.plot_heatmap_clusters is
  viz.plot_heatmap_clusters` (mirrors the existing `dao.DataViz is
  viz.DataViz` check)

### Manual Verification

- `docker compose exec diplomado-core poetry run pytest` (core group —
  covers `data_analysis_octopus` changes)
- Notebook 13/15 continue to import and call the unaffected functions
  without modification (no automated notebook-execution test exists in
  this repo; visual/manual spot-check only, consistent with existing
  coverage gaps documented in CLAUDE.md)

## Boundaries & Constraints

### In Scope

- Deleting `process_outliers`, its tests, both `RANDOM_STATE` constants,
  and `plot_k_distance`
- Moving `heatmap`/`plot_heatmap_clusters` into `viz.py` with a re-export

### Out of Scope

- `plot_wordcloud`, `plot_sentiment_global`, `create_3d_umap_plot` — kept
  as-is; they're used by notebook 13 (see Context)
- Any change to notebook 13 or notebook 15 content
- Any further `src/` dead-code audit beyond the four items named in issue
  #34

### Technical Constraints

- Tests run via Docker (`make test`), not bare host Poetry

## Success Criteria

- [ ] `process_outliers`, its tests, both `RANDOM_STATE` constants, and
      `plot_k_distance` no longer exist in the repo
- [ ] `dao.heatmap` / `dao.plot_heatmap_clusters` still resolve correctly,
      now sourced from `viz.py`
- [ ] `poetry run pytest` and `poetry run pylint $(git ls-files '*.py')`
      pass inside `core`
- [ ] Notebook 13 and notebook 15 unaffected (imports/calls unchanged)
- [ ] Issue #34 closed, with a comment noting the scope was narrowed from
      4 to 1 function in `clustering.py` and why

## Implementation Plan

See `specs/src-dead-code-cleanup-plan.md`.
