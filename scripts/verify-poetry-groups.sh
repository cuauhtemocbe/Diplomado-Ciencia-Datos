#!/usr/bin/env bash
# Verifies the acceptance criteria from GitHub issue #13:
#   - `poetry install --no-root` (no --with) satisfies the core-only notebooks
#     and does NOT pull in tensorflow/torch/geopandas
#   - each optional group, installed on top of core, satisfies its notebooks' imports
#   - src/data_analysis_octopus.py (shared helper) only needs core
#
# Each case runs in a fresh python:3.12.6-slim container (matching
# Dockerfile.dev's base) so the check reflects true install-time isolation,
# not whatever happens to already be on this machine or in a cached image.
#
# tensorflow/nlp build and are checked by default; set SKIP_HEAVY_GROUPS=1
# to skip them for a faster iteration loop (multi-GB downloads).
set -euo pipefail

cd "$(dirname "$0")/.."

run_case() {
  local name="$1" with_flag="$2" import_check="$3"
  echo "==> [$name] poetry install --no-root ${with_flag:+--with $with_flag}"
  if docker run --rm -v "$(pwd):/workspace:ro" -w /tmp/build \
      -e PYTHONPATH=/tmp/build/src \
      python:3.12.6-slim bash -euc "
        cp /workspace/pyproject.toml /workspace/poetry.lock .
        cp -r /workspace/src .
        pip install -q poetry
        poetry config virtualenvs.create false
        poetry install --no-root ${with_flag:+--with $with_flag} -q
        python -c \"$import_check\"
      " > "/tmp/verify-group-${name}.log" 2>&1; then
    echo "PASS: $name"
  else
    echo "FAIL: $name — see /tmp/verify-group-${name}.log" >&2
    tail -30 "/tmp/verify-group-${name}.log" >&2
    exit 1
  fi
}

run_case "core" "" "
import pandas, numpy, scipy, sklearn, plotly, matplotlib, seaborn, ipywidgets
import bs4, cufflinks, unidecode, phik, yellowbrick, statsmodels, xgboost
import data_analysis_octopus  # unconditionally imports xgboost + varclushi at module level
import importlib.util
for mod in ('tensorflow', 'torch', 'geopandas', 'dash'):
    assert importlib.util.find_spec(mod) is None, f'{mod} must not be installed with core alone'
print('core imports OK, heavy extras correctly absent')
"

if [ "${SKIP_HEAVY_GROUPS:-}" != "1" ]; then
  run_case "tensorflow" "tensorflow" "
import tensorflow, keras, tf_keras, h5py
print('tensorflow group imports OK')
"

  run_case "nlp" "nlp" "
import sentence_transformers, umap, googleapiclient, flask, gunicorn, wordcloud, transformers
from app_clustering import clustering
print('nlp group imports OK')
"
else
  echo "==> Skipping tensorflow/nlp (SKIP_HEAVY_GROUPS=1)"
fi

run_case "geo" "geo" "
import geopandas, pyproj
print('geo group imports OK')
"

run_case "bio" "bio" "
import wfdb, neurokit2
print('bio group imports OK')
"

run_case "explain" "explain" "
import shap, nltk
print('explain group imports OK')
"

echo "All Poetry group checks passed."
