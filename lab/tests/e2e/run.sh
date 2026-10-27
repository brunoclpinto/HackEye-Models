#!/usr/bin/env bash
# Browser E2E tests for the Models Lab (lab/PLAN.md, "Tests").
#
#   lab/tests/e2e/run.sh                 # all tests
#   lab/tests/e2e/run.sh -k editor -x    # extra args go to pytest
#   PW_TAG=v1.63.0-noble lab/tests/e2e/run.sh
#
# What it does: builds the fixture (20 real frames, /runs/lab-e2e/HackEye-fixture), starts a throwaway lab container
# `lab-e2e` on 127.0.0.1:8767 (LAB_DATASET=fixture, exports into /runs/lab-e2e/datasets), starts the `sam` service if it is
# not running (tests that need SAM skip when /api/meta says it is unavailable), runs Playwright (Chromium) in
# mcr.microsoft.com/playwright/python, then stops the throwaway lab, removes the fixture, the e2e-* maps it wrote into
# taxonomies/, and the sam service if this script started it.
# The Playwright image is ~3.5 GB; `docker pull mcr.microsoft.com/playwright/python:v1.63.0-noble` once.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
TAG="${PW_TAG:-v1.63.0-noble}"
PWV="${TAG#v}"; PWV="${PWV%%-*}"
RUNS="${HACKEYE_RUNS_DIR:-$REPO/../../media/hackeye-models/runs}"
RUNS="$(cd "$RUNS" && pwd)"
PORT=8767
NAME=lab-e2e
cd "$REPO/docker"
COMPOSE=(docker compose)

STARTED_SAM=0
cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1
  rm -rf "$RUNS/lab-e2e"
  rm -f "$REPO"/taxonomies/e2e-*.yaml
  if [ "$STARTED_SAM" = 1 ]; then "${COMPOSE[@]}" rm -sf sam >/dev/null 2>&1; fi
}
trap cleanup EXIT

docker image inspect hackeye-models-lab >/dev/null 2>&1 || "${COMPOSE[@]}" build lab || exit 1
docker rm -f "$NAME" >/dev/null 2>&1
rm -rf "$RUNS/lab-e2e"

# SAM is optional: start it only if its image exists.
if docker image inspect hackeye-models-sam >/dev/null 2>&1; then
  if [ -z "$("${COMPOSE[@]}" ps -q --status running sam 2>/dev/null)" ]; then
    "${COMPOSE[@]}" up -d sam && STARTED_SAM=1
  fi
fi

"${COMPOSE[@]}" run --rm --no-deps --entrypoint python lab /app/lab/tests/e2e/make_fixture.py || exit 1

"${COMPOSE[@]}" run -d --rm --no-deps --name "$NAME" -p "127.0.0.1:$PORT:8766" \
  -e LAB_DATASET=/runs/lab-e2e/HackEye-fixture -e LAB_DATASETS_ROOT=/runs/lab-e2e/datasets \
  -e SAM_URL=http://sam:8000 lab >/dev/null || exit 1
# `run --no-deps` still attaches to the compose network, so http://sam:8000 resolves.
for i in $(seq 60); do
  curl -fs "http://127.0.0.1:$PORT/api/meta" >/dev/null && break
  sleep 1
done
curl -fs "http://127.0.0.1:$PORT/api/meta" >/dev/null || { echo "throwaway lab did not come up"; docker logs "$NAME" | tail -30; exit 1; }
if [ "$STARTED_SAM" = 1 ]; then sleep 3; fi   # let sam's first health answer settle

docker run --rm --network host --ipc host \
  -e LAB_URL="http://127.0.0.1:$PORT" -e E2E_DATA=/e2e-data \
  -e PYTHONDONTWRITEBYTECODE=1 -e PWV="$PWV" \
  -v "$RUNS/lab-e2e:/e2e-data:ro" -v "$HERE:/e2e:ro" -w /e2e \
  "mcr.microsoft.com/playwright/python:$TAG" \
  bash -c 'pip install -q --root-user-action=ignore pytest playwright==$PWV pillow numpy opencv-python-headless 2>&1 | tail -2; exec python -m pytest -p no:cacheprovider -v "$@"' _ "$@"
RC=$?
exit $RC
