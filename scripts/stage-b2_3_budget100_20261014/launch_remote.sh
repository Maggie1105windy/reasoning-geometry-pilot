#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${REASONING_GEOMETRY_ROOT:-/home/ai/chenchen/projects/reasoning-geometry-pilot}"
BUDGET100_PYTHON="${B23_PYTHON:-/home/ai/.miniconda3/envs/chenchen_grounding_mechprobe/bin/python}"
BUDGET100_OUT="$PROJECT_ROOT/artifacts/stage_b2_3_budget100_20261014"
BUDGET100_SCRIPT="$PROJECT_ROOT/scripts/stage-b2_3_budget100_20261014/run_budget100.py"

cd "$PROJECT_ROOT"
mkdir -p "$BUDGET100_OUT/logs"
if [[ -f "$BUDGET100_OUT/launcher.pid" ]]; then
  PREVIOUS_RUN_PID="$(cat "$BUDGET100_OUT/launcher.pid")"
  if [[ "$PREVIOUS_RUN_PID" =~ ^[0-9]+$ ]] && kill -0 "$PREVIOUS_RUN_PID" 2>/dev/null; then
    printf 'Existing experiment process is still alive: %s\n' "$PREVIOUS_RUN_PID" >&2
    exit 1
  fi
fi

BUDGET100_LOG="$BUDGET100_OUT/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$BUDGET100_PYTHON" -u "$BUDGET100_SCRIPT" \
  --root "$PROJECT_ROOT" \
  --manifest "$BUDGET100_OUT/sample_manifest.json" \
  --out "$BUDGET100_OUT" \
  --budgets 8192 16384 32768 --gpu 0 \
  > "$BUDGET100_LOG" 2>&1 < /dev/null &
CURRENT_RUN_PID=$!
printf '%s\n' "$CURRENT_RUN_PID" > "$BUDGET100_OUT/launcher.pid"
printf '%s\n' "$BUDGET100_LOG" > "$BUDGET100_OUT/latest_log_path.txt"
sleep 2
if kill -0 "$CURRENT_RUN_PID" 2>/dev/null; then
  printf 'STARTED pid=%s log=%s\n' "$CURRENT_RUN_PID" "$BUDGET100_LOG"
else
  tail -n 30 "$BUDGET100_LOG"
  exit 1
fi
