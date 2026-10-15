#!/usr/bin/env bash
set -euo pipefail

ROOT="${REASONING_GEOMETRY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${B22_PYTHON:-/home/ai/.miniconda3/envs/chenchen_grounding_mechprobe/bin/python}"
MODEL_PATH="${QWEN3_8B_PATH:-}"
BATCH_SIZE="${B22_BATCH_SIZE:-4}"
SHARD_ROWS="${B22_SHARD_ROWS:-32}"
SCRIPT_DIR="$ROOT/scripts/stage-b2_2"
OUT="${STAGE_B22_OUT:-$ROOT/artifacts/stage_b2_2_qwen3}"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

if [[ -n "$MODEL_PATH" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/audit_and_freeze_b22.py" --phase preflight --model-path "$MODEL_PATH" 2>&1 | tee "$LOG_DIR/phase_0_preflight.log"
else
  "$PYTHON_BIN" "$SCRIPT_DIR/audit_and_freeze_b22.py" --phase preflight 2>&1 | tee "$LOG_DIR/phase_0_preflight.log"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/load_smoke_b22.py" 2>&1 | tee "$LOG_DIR/phase_1_model_load_smoke.log"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_and_freeze_b22.py" --phase boundary 2>&1 | tee "$LOG_DIR/phase_2_boundary_and_freeze.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b22.py" --mode smoke --protocol all --batch-size "$BATCH_SIZE" --shard-rows 15 2>&1 | tee "$LOG_DIR/phase_3_behavior_smoke.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b22.py" --mode full --protocol all --batch-size "$BATCH_SIZE" --shard-rows "$SHARD_ROWS" 2>&1 | tee "$LOG_DIR/phases_4_to_5_full.log"
"$PYTHON_BIN" "$SCRIPT_DIR/finalize_b22.py" 2>&1 | tee "$LOG_DIR/phase_6_finalize.log"
"$PYTHON_BIN" "$SCRIPT_DIR/validate_b22.py" 2>&1 | tee "$LOG_DIR/phase_7_validate.log"

echo "Stage B2.2 completed. Report: $OUT/STAGE_B2_2_QWEN3_REPORT.md"
