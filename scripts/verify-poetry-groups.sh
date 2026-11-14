#!/usr/bin/env bash
# Verifies the acceptance criteria from GitHub issue #13:
#   - `poetry install --no-root` (no --with) satisfies the core-only notebooks
#     and does NOT pull in tensorflow/torch/geopandas
#   - each optional group, installed on top of core, satisfies its notebooks' imports
#   - src/data_analysis_octopus.py (shared helper) only needs core
#
# Each case runs in a fresh container on the same pinned Python 3.14 image as
# Dockerfile.dev's base (patch tag + digest) so the check reflects true
# install-time isolation, not whatever happens to already be on this machine
# or in a cached image. The tensorflow case runs on the pinned Python 3.13
# image instead, matching Dockerfile.dev's tensorflow-env target (no stable
# cp314 tensorflow wheel yet). Keep both strings identical to Dockerfile.dev's.
#
# tensorflow/nlp build and are checked by default; set SKIP_HEAVY_GROUPS=1
# to skip them for a faster iteration loop (multi-GB downloads).
set -euo pipefail

PYTHON_IMAGE="python:3.14.10-slim-trixie@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f"
TENSORFLOW_PYTHON_IMAGE="python:3.13.18-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90"

cd "$(dirname "$0")/.."

run_case() {
  local name="$1" with_flag="$2" import_check="$3" image="${4:-$PYTHON_IMAGE}"
  echo "==> [$name] ($image) poetry install --no-root ${with_flag:+--with $with_flag}"
  if docker run --rm -v "$(pwd):/workspace:ro" -w /tmp/build \
      -e PYTHONPATH=/tmp/build/src \
      "$image" bash -euc "
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
" "$TENSORFLOW_PYTHON_IMAGE"

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
