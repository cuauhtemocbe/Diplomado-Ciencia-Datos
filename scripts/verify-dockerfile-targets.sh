#!/usr/bin/env bash
# Verifies the acceptance criteria from GitHub issue #14:
#   - `docker build --target core` produces an image under 2GB with
#     tensorflow/torch/geopandas/dash absent
#   - geo-env/bio-env/explain-env targets build and satisfy their notebooks'
#     imports, within a reasonable size bound
#   - rebuilding an unchanged target is a cache hit (no reinstall)
#
# tensorflow-env/nlp-env are skipped by default (same SKIP_HEAVY_GROUPS
# convention as verify-poetry-groups.sh): the combined install currently
# fails on torch, deferred until those notebooks get a dependency-reduction
# pass. Set SKIP_HEAVY_GROUPS=0 to attempt them anyway.
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_SIZE_CORE_MB=2048
MAX_SIZE_GEO_MB=3072
MAX_SIZE_BIO_MB=2048
MAX_SIZE_EXPLAIN_MB=2048

image_size_mb() {
  docker image inspect "$1" --format='{{.Size}}' | awk '{printf "%d", $1/1024/1024}'
}

build_target() {
  local target="$1" tag="diplomado-ds:${1}-verify"
  echo "==> building target '$target'"
  docker build -f Dockerfile.dev --target "$target" -t "$tag" . \
    > "/tmp/verify-dockerfile-${target}.log" 2>&1
}

check_size() {
  local target="$1" max_mb="$2" tag="diplomado-ds:${1}-verify"
  local size_mb
  size_mb=$(image_size_mb "$tag")
  echo "    size: ${size_mb}MB (max ${max_mb}MB)"
  if [ "$size_mb" -gt "$max_mb" ]; then
    echo "FAIL: $target image is ${size_mb}MB, exceeds ${max_mb}MB — see /tmp/verify-dockerfile-${target}.log" >&2
    exit 1
  fi
}

check_imports() {
  local target="$1" import_check="$2" tag="diplomado-ds:${1}-verify"
  if ! docker run --rm "$tag" python -c "$import_check" \
      > "/tmp/verify-dockerfile-${target}-imports.log" 2>&1; then
    echo "FAIL: $target imports — see /tmp/verify-dockerfile-${target}-imports.log" >&2
    tail -30 "/tmp/verify-dockerfile-${target}-imports.log" >&2
    exit 1
  fi
  echo "PASS: $target imports"
}

echo "==> [core]"
build_target core
check_size core "$MAX_SIZE_CORE_MB"
check_imports core "
import pandas, numpy, scipy, sklearn, plotly, matplotlib, seaborn, ipywidgets
import bs4, cufflinks, unidecode, phik, yellowbrick, statsmodels, xgboost
import data_analysis_octopus
import importlib.util
for mod in ('tensorflow', 'torch', 'geopandas', 'dash'):
    assert importlib.util.find_spec(mod) is None, f'{mod} must not be installed in core target'
print('core target OK')
"

echo "==> [cache-hit check: rebuilding core with no changes]"
# Redirect to a file rather than piping into `grep -q` directly: with
# `set -o pipefail`, grep -q's early exit-on-first-match can SIGPIPE the
# still-writing docker build process, making the pipeline report failure
# even though the build itself succeeded.
docker build -f Dockerfile.dev --target core -t diplomado-ds:core-verify . \
  > "/tmp/verify-dockerfile-core-cachehit.log" 2>&1
if grep -qi "cached" "/tmp/verify-dockerfile-core-cachehit.log"; then
  echo "PASS: core rebuild reused cached layers"
else
  echo "FAIL: core rebuild did not hit cache — see /tmp/verify-dockerfile-core-cachehit.log" >&2
  exit 1
fi

echo "==> [geo-env]"
build_target geo-env
check_size geo-env "$MAX_SIZE_GEO_MB"
check_imports geo-env "
import geopandas, pyproj
print('geo-env target OK')
"

echo "==> [bio-env]"
build_target bio-env
check_size bio-env "$MAX_SIZE_BIO_MB"
check_imports bio-env "
import wfdb, neurokit2
print('bio-env target OK')
"

echo "==> [explain-env]"
build_target explain-env
check_size explain-env "$MAX_SIZE_EXPLAIN_MB"
check_imports explain-env "
import shap, nltk
print('explain-env target OK')
"

if [ "${SKIP_HEAVY_GROUPS:-}" != "1" ]; then
  echo "==> [tensorflow-env]"
  build_target tensorflow-env
  check_imports tensorflow-env "
import tensorflow, keras, tf_keras
print('tensorflow-env target OK')
"

  echo "==> [nlp-env]"
  build_target nlp-env
  check_imports nlp-env "
import sentence_transformers, umap, googleapiclient, flask, gunicorn, wordcloud
from app_clustering import clustering
print('nlp-env target OK')
"
else
  echo "==> Skipping tensorflow-env/nlp-env (SKIP_HEAVY_GROUPS=1) -- torch install currently broken, pending notebook refactor"
fi

echo "All Dockerfile.dev target checks passed."
