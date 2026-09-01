#!/usr/bin/env python3
"""Finalize an already-complete B1 extraction after exact-batch replay integrity."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_b1 import ACT_DIR, HIDDEN_SIZE, MODEL_PATH, N_LAYERS, OUT, SEED, sha256, write_json
from extract_b1 import integrity_reforward, load_rows


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rows = load_rows()
    sanity = json.loads((OUT / "activation_sanity.json").read_text(encoding="utf-8"))
    model_audit = json.loads((OUT / "model_audit.json").read_text(encoding="utf-8"))
    if sanity["status"] != "PASS" or model_audit["status"] != "PASS":
        raise RuntimeError("Upstream model/sanity Gate not PASS")
    row_ids_path = ACT_DIR / "row_ids.npy"
    if not row_ids_path.is_file():
        # The first extraction completed all layer files but stopped before this
        # small row-order artifact was written by the original code path.
        np.save(row_ids_path, rows["row_id"].to_numpy(dtype=np.int64))
    expected_ids = rows["row_id"].to_numpy(dtype=np.int64)
    row_ids = np.load(row_ids_path)
    per_layer = {}
    layer_files = {}
    for layer in range(N_LAYERS):
        path = ACT_DIR / f"layer_{layer:02d}.npy"
        matrix = np.load(path, mmap_mode="r")
        per_layer[str(layer)] = {
            "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "nan_count": int(np.isnan(matrix).sum()), "inf_count": int(np.isinf(matrix).sum()),
        }
        layer_files[str(layer)] = {"path": str(path), "sha256": sha256(path)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    reforward = integrity_reforward(model, tokenizer, rows, extraction_batch_size=32)
    checks = {
        "row_ids_exact_order": bool(np.array_equal(row_ids, expected_ids)),
        "row_ids_unique": len(np.unique(row_ids)) == len(row_ids),
        "all_32_layers_present": len(per_layer) == N_LAYERS,
        "all_shapes_exact": all(value["shape"] == [len(rows), HIDDEN_SIZE] for value in per_layer.values()),
        "all_dtypes_float16": all(value["dtype"] == "float16" for value in per_layer.values()),
        "all_nan_inf_zero": all(value["nan_count"] == 0 and value["inf_count"] == 0 for value in per_layer.values()),
        "same_batch_reforward_pass": reforward["status"] == "PASS",
    }
    integrity = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "activation_definition": sanity["activation_definition"],
        "union_manifest": str(OUT / "union_manifest.csv"),
        "checks": checks,
        "extraction": {
            "rows": len(rows), "layers": N_LAYERS, "hidden_size": HIDDEN_SIZE,
            "dtype": "float16", "batch_size": 32,
            "runtime_seconds": None,
            "runtime_note": "first extraction completed before integrity threshold stop; progress log reached 12,800 at 349.83s; exact final runtime was not persisted",
            "row_ids_sha256": sha256(row_ids_path), "layer_files": layer_files,
        },
        "per_layer": per_layer, "reforward": reforward,
    }
    write_json(OUT / "activation_integrity.json", integrity)
    write_json(ACT_DIR / "metadata.json", {
        "status": integrity["status"], "model_audit": model_audit,
        "activation_definition": sanity["activation_definition"],
        "row_count": len(rows), "layers": list(range(N_LAYERS)), "hidden_size": HIDDEN_SIZE,
        "batch_size": 32, "extraction": integrity["extraction"],
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__).resolve()),
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": integrity["status"], "rows": len(rows),
        "minimum_same_batch_cosine": reforward["minimum_cosine"],
        "mean_same_batch_mae": reforward["mean_mae"],
        "batch1_non_gate_minimum_cosine": reforward["batch1_non_gate_diagnostic"]["minimum_cosine"],
        "failed_checks": [key for key, value in checks.items() if not value],
    }, indent=2))
    if integrity["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

