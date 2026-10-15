#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${REASONING_GEOMETRY_ROOT:-/home/ai/chenchen/projects/reasoning-geometry-pilot}"
PILOT_PYTHON="${B23_PYTHON:-/home/ai/.miniconda3/envs/chenchen_grounding_mechprobe/bin/python}"
PILOT_OUT="$PROJECT_ROOT/artifacts/stage_b2_3_budget_pilot_20261013"
PILOT_SCRIPT="$PROJECT_ROOT/scripts/stage-b2_3_budget_pilot_20261013/run_budget_pilot.py"

cd "$PROJECT_ROOT"
mkdir -p "$PILOT_OUT/logs"
if [[ -f "$PILOT_OUT/launcher.pid" ]]; then
  OLD_RUN_PID="$(cat "$PILOT_OUT/launcher.pid")"
  if [[ "$OLD_RUN_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_RUN_PID" 2>/dev/null; then
    printf 'Existing experiment process is still alive: %s\n' "$OLD_RUN_PID" >&2
    exit 1
  fi
fi

PILOT_LOG="$PILOT_OUT/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$PILOT_PYTHON" -u "$PILOT_SCRIPT" \
  --root "$PROJECT_ROOT" \
  --manifest "$PILOT_OUT/sample_manifest.json" \
  --out "$PILOT_OUT" \
  --budgets 8192 16384 32768 --gpu 0 \
  > "$PILOT_LOG" 2>&1 < /dev/null &
NEW_RUN_PID=$!
printf '%s\n' "$NEW_RUN_PID" > "$PILOT_OUT/launcher.pid"
printf '%s\n' "$PILOT_LOG" > "$PILOT_OUT/latest_log_path.txt"
sleep 2
if kill -0 "$NEW_RUN_PID" 2>/dev/null; then
  printf 'STARTED pid=%s log=%s\n' "$NEW_RUN_PID" "$PILOT_LOG"
else
  tail -n 30 "$PILOT_LOG"
  exit 1
fi
