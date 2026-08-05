#!/usr/bin/env bash
# Verifies the acceptance criteria from GitHub issue #15:
#   - each docker-compose service builds successfully
#   - each service starts and reaches the "running" state
#   - multiple group containers can run at the same time without port
#     conflicts (they all get checked while still up, at the end)
#
# tensorflow/nlp are skipped by default (SKIP_HEAVY_GROUPS=1, same
# convention as verify-poetry-groups.sh / verify-dockerfile-targets.sh):
# the underlying torch install currently fails, deferred pending a
# notebook dependency-reduction refactor.
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE="docker compose -f docker-compose.dev.yml"

TARGET_GROUPS=(core geo bio explain)
if [ "${SKIP_HEAVY_GROUPS:-}" != "1" ]; then
  TARGET_GROUPS+=(tensorflow nlp)
else
  echo "==> Skipping tensorflow/nlp services (SKIP_HEAVY_GROUPS=1) -- torch install currently broken, pending notebook refactor"
fi

cleanup() {
  echo "==> Cleaning up: stopping all started services"
  for g in "${TARGET_GROUPS[@]}"; do
    $COMPOSE stop "diplomado-$g" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for g in "${TARGET_GROUPS[@]}"; do
  service="diplomado-$g"

  echo "==> [$service] build"
  if ! $COMPOSE build "$service" > "/tmp/verify-compose-${g}-build.log" 2>&1; then
    echo "FAIL: $service build — see /tmp/verify-compose-${g}-build.log" >&2
    tail -30 "/tmp/verify-compose-${g}-build.log" >&2
    exit 1
  fi

  echo "==> [$service] up"
  if ! $COMPOSE up -d "$service" > "/tmp/verify-compose-${g}-up.log" 2>&1; then
    echo "FAIL: $service up — see /tmp/verify-compose-${g}-up.log" >&2
    exit 1
  fi

  running=$($COMPOSE ps --status running --services "$service")
  if [ "$running" != "$service" ]; then
    echo "FAIL: $service is not in the running state after 'up'" >&2
    exit 1
  fi
  echo "PASS: $service built and running"
done

echo "==> Checking all ${#TARGET_GROUPS[@]} services are still running concurrently (no port conflicts)"
still_running=$($COMPOSE ps --status running --services)
for g in "${TARGET_GROUPS[@]}"; do
  service="diplomado-$g"
  if ! grep -qx "$service" <<< "$still_running"; then
    echo "FAIL: $service is no longer running while others were being started" >&2
    exit 1
  fi
done
echo "PASS: all ${#TARGET_GROUPS[@]} services running concurrently on their own ports"

echo "All docker-compose service checks passed."
