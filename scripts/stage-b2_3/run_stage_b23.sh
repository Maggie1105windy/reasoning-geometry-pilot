#!/usr/bin/env bash
set -euo pipefail

ROOT="${REASONING_GEOMETRY_ROOT:-/home/ai/chenchen/projects/reasoning-geometry-pilot}"
PYTHON_BIN="${B23_PYTHON:-/home/ai/.miniconda3/envs/chenchen_grounding_mechprobe/bin/python}"
SCRIPT_DIR="$ROOT/scripts/stage-b2_3"
LOG_DIR="$ROOT/artifacts/stage_b2_3_qwen3_generation/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/audit_b23.py" 2>&1 | tee "$LOG_DIR/phase_0_audit.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase smoke 2>&1 | tee "$LOG_DIR/phase_1_smoke.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase batch-smoke 2>&1 | tee "$LOG_DIR/phase_1b_batch_smoke.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase pilot 2>&1 | tee "$LOG_DIR/phase_3_pilot.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase matched 2>&1 | tee "$LOG_DIR/phase_4_matched.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase full-fgn 2>&1 | tee "$LOG_DIR/phase_5_full_fgn.log"
"$PYTHON_BIN" "$SCRIPT_DIR/run_b23.py" --phase full-fgt 2>&1 | tee "$LOG_DIR/phase_6_full_fgt.log"
"$PYTHON_BIN" "$SCRIPT_DIR/finalize_b23.py" 2>&1 | tee "$LOG_DIR/phase_7_9_finalize.log"
"$PYTHON_BIN" "$SCRIPT_DIR/validate_b23.py" 2>&1 | tee "$LOG_DIR/phase_8_validation.log"

