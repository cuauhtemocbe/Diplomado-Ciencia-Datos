# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Coursework repo for the Diplomado en Ciencia de Datos (2024–2025): Jupyter notebooks for each module/project plus a small Flask app (`app_clustering`) that productionizes one of them (YouTube comment clustering).

## Commands

Dependencies are managed with Poetry; the intended environment is the dev container/Docker image, not a bare local install.

- Install deps: `poetry install`
- Run all tests: `poetry run pytest`
- Run a single test: `poetry run pytest tests/test_hello_world.py::test_print_hello`
- Lint (must pass in CI, see `.github/workflows/pylint.yml`): `poetry run pylint $(git ls-files '*.py')`
- Format: `poetry run black .` and `poetry run isort .`

Pylint's disabled-checks list in `pyproject.toml` is deliberate for this repo's exploratory/notebook-support nature (see the comment block above `[tool.pylint."messages control"]`) — don't re-enable those checks or fight them locally.

### Docker

- Jupyter Lab dev environment: `docker compose -f docker-compose.dev.yml up -d`, then `docker exec -it diplomado-ds bash` and `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''`. Notebooks live at `http://localhost:8889/lab/tree/notebooks`. Requires a `.env` file in the repo root.
- Clustering API (prod-shaped): `docker compose -f docker-compose.prod.yml up -d`, served via gunicorn on port 5000 (`src.app_clustering.app:app`).

## Architecture

- `notebooks/` — one notebook per module/project, numbered roughly in course order (e.g. `2-Cardiotocography.ipynb`, `9-Electrocardiograma.ipynb`). These are the primary deliverables of the repo; most work happens here rather than in `src/`.
- `data/` — raw datasets consumed by the notebooks, one subfolder/file per project, matching notebook names/numbers.
- `results/` — per-project trained artifacts (pickled models, preprocessors, train/test splits) and rendered outputs (e.g. `10-Proyecto-Hipertension-Mexico.pdf`), saved by their corresponding notebook.
- `resouces/` — static images referenced by specific notebooks (sic — matches the actual directory name, do not "fix" the typo without checking notebook references to it).
- `src/` — the one project promoted out of notebook form into an importable/deployable app:
  - `src/app_clustering/clustering.py` — YouTube comment fetching (Google API), sentence-embedding, UMAP dimensionality reduction, and clustering/plotting logic.
  - `src/app_clustering/app.py` — Flask front end over `clustering.py`; single `/` route takes a YouTube URL and renders comment-cluster/sentiment visualizations. Reads `youtube_api_key` from env (`.env` locally, real env vars when `RAILWAY_ENVIRONMENT` is set, e.g. on deploy).
  - `pythonpath` in `pyproject.toml` includes `src`, so notebooks/tests import it as `from app_clustering import clustering`, not `from src.app_clustering import ...`.
- `tests/` — currently a placeholder (`test_hello_world.py`); no real coverage of `app_clustering` yet.
