#!/usr/bin/env bash
# Run after every change: tests first, then the dashboard (if not running) and the live replay
# with the current code.
#
#   bash scripts/check_and_run.sh                       # Test_1 replay, context mode, YOLO-World + CLIP
#   bash scripts/check_and_run.sh --mode improved       # any extra live.py options are passed on
#   VIDEO=data/videos/other.mp4 bash scripts/check_and_run.sh
#   NOTEST=1 bash scripts/check_and_run.sh              # skip the tests
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python                     # Mac / Linux
VIDEO="${VIDEO:-data/videos/Test_1.mp4}"

if [ -z "${NOTEST:-}" ]; then
  echo "== tests"
  "$PY" -m pytest -q || { echo "Tests failed - fix them before running the demo."; exit 1; }
fi

if ! curl -s -o /dev/null http://127.0.0.1:8765/; then
  echo "== starting dashboard -> http://localhost:8765"
  "$PY" scripts/dashboard.py > results/dashboard.log 2>&1 &
  sleep 2
else
  echo "== dashboard already running (restart it yourself if you changed dashboard.py)"
fi

echo "== live replay of $VIDEO  (space pause, j/l -/+10 s, +/- speed, k pose, q quit)"
"$PY" scripts/live.py --mode context --video "$VIDEO" --start-point T1 \
  --object-detector world --imgsz 640 --det-imgsz 1280 --det-skip 2 --stride 2 "$@"
