# Stage B2.2 — Qwen3-8B Model Selection Gate

This directory extends the frozen Stage B2.1 behavior protocol without modifying any B0/B1/B2/B2.1 artifact. The execution order is deliberately gated:

1. `audit_and_freeze_b22.py --phase preflight` — frozen-data, environment, model-manifest, and protected-file audit.
2. `load_smoke_b22.py` — BF16 model load, `eval()` check, Raw/Native forward smoke, and non-thinking render check.
3. `audit_and_freeze_b22.py --phase boundary` — tokenizer/boundary and equal-length gates, then protocol freeze.
4. `run_b22.py --mode smoke --protocol all` — deterministic 3 rows per hop.
5. `run_b22.py --mode full --protocol all` — resumable sharded inference.
6. `finalize_b22.py` — metrics, bootstrap comparisons, figures, decision, and report.
7. `validate_b22.py` — independent row-level reconstruction and protected-file verification.

The scripts default to the repository containing them. On the research server, run with the existing Stage B2.1 environment (`transformers==4.57.6`) and either place the complete model under `models/Qwen3-8B` or pass `--model-path`/`QWEN3_8B_PATH`.

No script enables Qwen thinking mode. Native prompts always use the tokenizer's own chat template with `add_generation_prompt=True, enable_thinking=False`.

For a gated end-to-end server run:

```bash
export REASONING_GEOMETRY_ROOT="$HOME/projects/reasoning-geometry-pilot"
export QWEN3_8B_PATH=/absolute/path/to/Qwen3-8B  # optional; local caches are searched
bash scripts/stage-b2_2/run_stage_b22.sh
```

Rerunning the command validates and reuses completed shards. An invalid existing shard or failed data/boundary gate stops the pipeline instead of overwriting evidence.
