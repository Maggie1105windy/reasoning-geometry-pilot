# Stage B2.3 — Qwen3-8B Free-Generation / Thinking Gate

This directory implements the frozen Stage B2.3 protocol without modifying B0–B2.2 artifacts.

Execution order:

1. `audit_b23.py` — recover B2.2, hash the frozen data/model/protected artifacts, freeze prompt/parser/pilot/config.
2. `run_b23.py --phase smoke` — BF16 generation smoke and truncation-only resolution of 1024 vs 2048 tokens.
3. `run_b23.py --phase batch-smoke` — accuracy-blind BF16 memory/throughput smoke for conservative batching.
4. `run_b23.py --phase pilot` — 250 rows × FG-N/FG-T × three fixed seeds.
5. `run_b23.py --phase matched` — matched-decoding sensitivity on the same 250 rows.
6. `run_b23.py --phase full-fgn` — 2,063-row FG-N primary run with seed 20260924.
7. `run_b23.py --phase full-fgt` — 2,063-row FG-T primary run with seed 20260924.
8. `finalize_b23.py` — metrics, bootstrap, error transitions, figures, decisions, and report.
9. `validate_b23.py` — independent recomputation from raw JSONL and protected-artifact verification.

Every generated row is first written atomically to a separate JSON shard. Re-running a phase validates existing shards and resumes only missing rows. A conflicting shard causes a hard stop rather than an overwrite.

Run the complete sequence with:

```bash
bash scripts/stage-b2_3/run_stage_b23.sh
```

After Phase 0/1/1b have passed, run the long remaining sequence with:

```bash
bash scripts/stage-b2_3/run_remaining_b23.sh
```

