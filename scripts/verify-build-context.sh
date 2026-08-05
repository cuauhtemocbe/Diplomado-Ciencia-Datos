#!/usr/bin/env bash
# Verifies the acceptance criteria from GitHub issue #12:
#   - build context stays small (.dockerignore excludes data/results/.git/etc.)
#   - changing a notebook doesn't invalidate the poetry install layer
#   - .env is never baked into the image
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=diplomado-ds:verify-build-context
MAX_CONTEXT_KB=5120 # 5MB

echo "==> Building once to warm the cache"
docker build -f Dockerfile.dev --target core -t "$IMAGE" . >/tmp/build-context-verify-1.log 2>&1

context_raw=$(grep -Eo 'transferring context: [0-9.]+[kKmMgG]?B' /tmp/build-context-verify-1.log | tail -1 | grep -Eo '[0-9.]+[kKmMgG]?B')
if [ -z "$context_raw" ]; then
  echo "FAIL: could not find a 'transferring context' line in the build log" >&2
  exit 1
fi
echo "==> Reported build context transfer: $context_raw"

if [[ "$context_raw" =~ ^([0-9.]+)([kKmMgG]?)B$ ]]; then
  number="${BASH_REMATCH[1]}"
  unit=$(echo "${BASH_REMATCH[2]}" | tr '[:upper:]' '[:lower:]')
else
  echo "FAIL: could not parse build context size from '$context_raw'" >&2
  exit 1
fi

case "$unit" in
  g) context_kb=$(awk -v n="$number" 'BEGIN { print n * 1024 * 1024 }') ;;
  m) context_kb=$(awk -v n="$number" 'BEGIN { print n * 1024 }') ;;
  k) context_kb="$number" ;;
  *) context_kb=$(awk -v n="$number" 'BEGIN { print n / 1024 }') ;;
esac

if awk -v kb="$context_kb" -v max="$MAX_CONTEXT_KB" 'BEGIN { exit !(kb <= max) }'; then
  echo "PASS: build context ($context_kb KB) is under the ${MAX_CONTEXT_KB}KB limit"
else
  echo "FAIL: build context ($context_kb KB) exceeds the ${MAX_CONTEXT_KB}KB limit" >&2
  exit 1
fi

echo "==> Touching a notebook and rebuilding to check the poetry install layer is cached"
touch notebooks/0-Hello-Pandas.ipynb
docker build -f Dockerfile.dev --target core -t "$IMAGE" . >/tmp/build-context-verify-2.log 2>&1

if grep -q "RUN poetry install --no-root" /tmp/build-context-verify-2.log && \
   grep -A1 "RUN poetry install --no-root" /tmp/build-context-verify-2.log | grep -qi "CACHED"; then
  echo "PASS: poetry install layer was cached after a notebook-only change"
else
  echo "FAIL: poetry install layer rebuilt after a notebook-only change" >&2
  exit 1
fi

echo "==> Checking .env is not present inside the built image"
if docker run --rm "$IMAGE" test -f /workspace/.env; then
  echo "FAIL: .env was found inside the image" >&2
  exit 1
else
  echo "PASS: .env is absent from the image"
fi

echo "All build-context checks passed."
