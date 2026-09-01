#!/usr/bin/env python3
"""Stage B1 model audit, activation sanity, union extraction, and integrity."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common_b1 import (
    ACT_DIR, HIDDEN_SIZE, MODEL_PATH, MODEL_REVISION, N_LAYERS, OUT, SEED,
    TOKENIZER_REVISION, ensure_output_dirs, sha256, stable_int, write_json,
)


INDEX_LAYERS = [0, 15, 31]
INTEGRITY_LAYERS = [0, 7, 15, 23, 31]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "union_manifest.csv")
    frame["token_ids_list"] = frame["token_ids"].map(json.loads)
    return frame


def padded_batch(rows: pd.DataFrame, tokenizer, device: str = "cuda"):
    sequences = rows["token_ids_list"].tolist()
    max_len = max(len(ids) for ids in sequences)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, ids in enumerate(sequences):
        length = len(ids)
        input_ids[idx, :length] = torch.tensor(ids, dtype=torch.long)
        attention[idx, :length] = 1
    readout = torch.tensor(rows["readout_index"].astype(int).tolist(), dtype=torch.long)
    assert torch.equal(readout, attention.sum(1) - 1)
    return input_ids.to(device), attention.to(device), readout.to(device)


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    delta = (a - b).abs()
    return {
        "cosine_similarity": float(F.cosine_similarity(a[None], b[None], dim=1).item()),
        "mean_absolute_error": float(delta.mean()),
        "max_absolute_error": float(delta.max()),
    }


def forward_capture(model, tokenizer, rows: pd.DataFrame, layers: list[int]) -> dict[int, torch.Tensor]:
    ids, attention, readout = padded_batch(rows, tokenizer)
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            captures[layer] = hidden[batch, readout].detach().float().cpu()
        return hook

    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=attention, use_cache=False, output_hidden_states=False)
        assert set(captures) == set(layers)
        return captures
    finally:
        for handle in handles:
            handle.remove()
        del ids, attention, readout


def batch_trial(model, tokenizer, rows: pd.DataFrame, batch_size: int) -> dict:
    sample = rows.nlargest(batch_size, "sequence_length")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        captures = forward_capture(model, tokenizer, sample, list(range(N_LAYERS)))
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        del captures
        return {
            "batch_size": batch_size, "status": "PASS", "runtime_seconds": runtime,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
            "sequence_length_max": int(sample["sequence_length"].max()),
        }
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size, "status": "OOM", "runtime_seconds": time.perf_counter() - start,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
            "sequence_length_max": int(sample["sequence_length"].max()),
        }


def indexing_check(model, tokenizer, rows: pd.DataFrame) -> pd.DataFrame:
    sample = rows.iloc[[0, len(rows) // 2]].copy()
    ids, attention, readout = padded_batch(sample, tokenizer)
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            captures[layer] = hidden[batch, readout].detach()
        return hook

    for layer in INDEX_LAYERS:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            result = model(
                input_ids=ids, attention_mask=attention, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )
        assert len(result.hidden_states) == 33
        batch = torch.arange(len(sample), device="cuda")
        records = []
        for sample_idx, row in enumerate(sample.itertuples(index=False)):
            for layer in INDEX_LAYERS:
                raw = captures[layer][sample_idx]
                if layer < 31:
                    tuple_vec = result.hidden_states[layer + 1][batch, readout][sample_idx]
                    records.append({
                        "row_id": int(row.row_id), "layer": layer,
                        "comparison": "raw_hook_vs_hidden_states_layer_plus_1",
                        "hf_index": layer + 1, **vector_metrics(raw, tuple_vec),
                    })
                else:
                    post_norm = result.hidden_states[32][batch, readout][sample_idx]
                    records.append({
                        "row_id": int(row.row_id), "layer": 31,
                        "comparison": "raw_hook_vs_hidden_states_32_post_final_RMSNorm",
                        "hf_index": 32, **vector_metrics(raw, post_norm),
                    })
                    renorm = model.model.norm(raw[None])[0]
                    records.append({
                        "row_id": int(row.row_id), "layer": 31,
                        "comparison": "final_RMSNorm_raw_hook_vs_hidden_states_32",
                        "hf_index": 32, **vector_metrics(renorm, post_norm),
                    })
        return pd.DataFrame(records)
    finally:
        for handle in handles:
            handle.remove()
        del ids, attention, readout


def sanity_gate(model, tokenizer, rows: pd.DataFrame) -> tuple[dict, list[dict]]:
    sanity = rows.assign(_order=[stable_int("activation-sanity", value) for value in rows["row_id"]]).nsmallest(32, "_order")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    captures = forward_capture(model, tokenizer, sanity, list(range(N_LAYERS)))
    torch.cuda.synchronize()
    layer_records = []
    for layer in range(N_LAYERS):
        tensor = captures[layer]
        layer_records.append({
            "layer": layer, "shape": list(tensor.shape), "hidden_size": int(tensor.shape[1]),
            "nan_count": int(torch.isnan(tensor).sum()), "inf_count": int(torch.isinf(tensor).sum()),
            "mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False)),
            "mean_l2": float(torch.linalg.vector_norm(tensor, dim=1).mean()),
        })
    indexing = indexing_check(model, tokenizer, sanity)
    indexing.to_csv(OUT / "activation_indexing_check.csv", index=False)
    raw31 = indexing[indexing["comparison"] == "raw_hook_vs_hidden_states_32_post_final_RMSNorm"]
    renorm = indexing[indexing["comparison"] == "final_RMSNorm_raw_hook_vs_hidden_states_32"]
    direct = indexing[indexing["comparison"] == "raw_hook_vs_hidden_states_layer_plus_1"]
    checks = {
        "samples_32": len(sanity) == 32,
        "all_32_layers": set(captures) == set(range(N_LAYERS)),
        "all_shapes_32x4096": all(record["shape"] == [32, HIDDEN_SIZE] for record in layer_records),
        "nan_inf_zero": all(record["nan_count"] == 0 and record["inf_count"] == 0 for record in layer_records),
        "layers_0_15_match_hf_tuple": bool(direct["cosine_similarity"].min() > 0.99999 and direct["max_absolute_error"].max() < 1e-6),
        "raw_layer31_differs_from_final_norm": bool(raw31["max_absolute_error"].min() > 1e-3 and raw31["mean_absolute_error"].min() > 1e-5),
        "normed_raw_layer31_matches_hidden_states_32": bool(renorm["cosine_similarity"].min() > 0.99999),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_row_ids": sanity["row_id"].astype(int).tolist(),
        "activation_definition": "raw decoder block output from model.model.layers[layer] forward hook before final model RMSNorm",
        "runtime_seconds": time.perf_counter() - start,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
        "checks": checks,
        "layer_statistics": layer_records,
        "indexing_records": indexing.to_dict(orient="records"),
    }
    write_json(OUT / "activation_sanity.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"Activation sanity Gate failed: {[key for key, value in checks.items() if not value]}")
    del captures
    return payload, layer_records


def extract_union(model, tokenizer, rows: pd.DataFrame, batch_size: int) -> dict:
    n_rows = len(rows)
    maps = {
        layer: np.lib.format.open_memmap(
            ACT_DIR / f"layer_{layer:02d}.npy", mode="w+", dtype=np.float16,
            shape=(n_rows, HIDDEN_SIZE),
        )
        for layer in range(N_LAYERS)
    }
    state = {"readout": None, "start": 0, "end": 0}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[batch, state["readout"]].detach().to(torch.float16).cpu().numpy()
            maps[layer][state["start"]:state["end"]] = selected
        return hook

    for layer in range(N_LAYERS):
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    try:
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch_rows = rows.iloc[start:end]
            ids, attention, readout = padded_batch(batch_rows, tokenizer)
            state.update({"readout": readout, "start": start, "end": end})
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=attention, use_cache=False, output_hidden_states=False)
            del ids, attention, readout
            if start and start % (batch_size * 100) == 0:
                elapsed = time.perf_counter() - start_time
                print(json.dumps({"extracted": start, "total": n_rows, "elapsed_seconds": elapsed}), flush=True)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    for layer, mmap in maps.items():
        mmap.flush()
        if tuple(mmap.shape) != (n_rows, HIDDEN_SIZE) or not np.isfinite(mmap).all():
            raise RuntimeError(f"Invalid saved activation layer {layer}")
    row_ids = rows["row_id"].to_numpy(dtype=np.int64)
    np.save(ACT_DIR / "row_ids.npy", row_ids)
    return {
        "rows": n_rows, "layers": N_LAYERS, "hidden_size": HIDDEN_SIZE,
        "dtype": "float16", "batch_size": batch_size,
        "runtime_seconds": time.perf_counter() - start_time,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
        "row_ids_sha256": sha256(ACT_DIR / "row_ids.npy"),
        "layer_files": {
            str(layer): {"path": str(ACT_DIR / f"layer_{layer:02d}.npy"), "sha256": sha256(ACT_DIR / f"layer_{layer:02d}.npy")}
            for layer in range(N_LAYERS)
        },
    }


def integrity_reforward(model, tokenizer, rows: pd.DataFrame, extraction_batch_size: int = 32) -> dict:
    ordered = rows.assign(_order=[stable_int("integrity-reforward", value) for value in rows["row_id"]]).nsmallest(10, "_order")
    records = []
    batch1_records = []
    for row_index, row in ordered.iterrows():
        batch_start = (int(row_index) // extraction_batch_size) * extraction_batch_size
        batch_end = min(batch_start + extraction_batch_size, len(rows))
        replay = forward_capture(model, tokenizer, rows.iloc[batch_start:batch_end], INTEGRITY_LAYERS)
        batch1 = forward_capture(model, tokenizer, rows.loc[[row_index]], INTEGRITY_LAYERS)
        local_index = int(row_index) - batch_start
        for layer in INTEGRITY_LAYERS:
            saved = torch.from_numpy(np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")[row_index].copy()).float()
            metrics = vector_metrics(replay[layer][local_index], saved)
            records.append({
                "row_id": int(row.row_id), "union_index": int(row_index), "layer": layer,
                "comparison": "same_extraction_batch_replay_vs_saved_float16",
                "batch_start": batch_start, "batch_end": batch_end, **metrics,
            })
            batch1_records.append({
                "row_id": int(row.row_id), "union_index": int(row_index), "layer": layer,
                "comparison": "batch1_reforward_vs_saved_float16_non_gate_diagnostic",
                **vector_metrics(batch1[layer][0], saved),
            })
    frame = pd.DataFrame(records)
    batch1_frame = pd.DataFrame(batch1_records)
    checks = {
        "comparisons_10x5": len(frame) == 50,
        "minimum_cosine_at_least_0_9999": bool(frame["cosine_similarity"].min() >= 0.9999),
        "all_errors_finite": bool(np.isfinite(frame[["mean_absolute_error", "max_absolute_error"]]).all().all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "minimum_cosine": float(frame["cosine_similarity"].min()),
        "mean_mae": float(frame["mean_absolute_error"].mean()),
        "max_absolute_error": float(frame["max_absolute_error"].max()),
        "records": frame.to_dict(orient="records"),
        "batch1_non_gate_diagnostic": {
            "minimum_cosine": float(batch1_frame["cosine_similarity"].min()),
            "mean_mae": float(batch1_frame["mean_absolute_error"].mean()),
            "max_absolute_error": float(batch1_frame["max_absolute_error"].max()),
            "interpretation": "BF16 kernel/batch-context numerical sensitivity; not storage/alignment Gate",
            "records": batch1_frame.to_dict(orient="records"),
        },
    }


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    ensure_output_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    token_gate = json.loads((OUT / "tokenization_audit.json").read_text(encoding="utf-8"))
    checksum_gate = json.loads((OUT / "split_checksum_verification.json").read_text(encoding="utf-8"))
    if token_gate["status"] != "PASS" or checksum_gate["status"] != "PASS":
        raise RuntimeError("Pre-model Gate is not PASS")
    rows = load_rows()
    started = utc_now()
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    if config.num_hidden_layers != N_LAYERS or config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError("Unexpected model architecture")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 8030261248 or len(model.model.layers) != N_LAYERS:
        raise RuntimeError(f"Unexpected parameter/layer count: {parameter_count}")

    audit = {
        "status": "PASS", "model": "Meta-Llama-3.1-8B Base", "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
        "parameter_count": parameter_count, "layers": N_LAYERS, "hidden_size": HIDDEN_SIZE,
        "model_dtype": str(next(model.parameters()).dtype), "activation_storage_dtype": "float16",
        "transformers_version": __import__("transformers").__version__, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": float(torch.cuda.get_device_properties(0).total_memory / 2**30),
        "config_sha256": sha256(MODEL_PATH / "config.json"),
        "tokenizer_json_sha256": sha256(MODEL_PATH / "tokenizer.json"),
        "model_load_seconds": load_seconds, "started_utc": started,
    }
    write_json(OUT / "model_audit.json", audit)

    trials = []
    for batch_size in [1, 2, 4, 8, 16, 32]:
        record = batch_trial(model, tokenizer, rows, batch_size)
        trials.append(record)
        print(json.dumps({"batch_trial": record}), flush=True)
        if record["status"] != "PASS":
            break
    passed = [record for record in trials if record["status"] == "PASS"]
    if not passed:
        raise RuntimeError("Batch size 1 failed")
    chosen_batch = max(record["batch_size"] for record in passed if record["peak_reserved_gib"] < 23.5)
    sanity, _ = sanity_gate(model, tokenizer, rows)
    extraction = extract_union(model, tokenizer, rows, chosen_batch)
    reforward = integrity_reforward(model, tokenizer, rows)
    if reforward["status"] != "PASS":
        raise RuntimeError("Saved activation re-forward integrity failed")

    per_layer = {}
    row_ids = np.load(ACT_DIR / "row_ids.npy")
    expected_ids = rows["row_id"].to_numpy(dtype=np.int64)
    for layer in range(N_LAYERS):
        matrix = np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")
        per_layer[str(layer)] = {
            "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "nan_count": int(np.isnan(matrix).sum()), "inf_count": int(np.isinf(matrix).sum()),
        }
    integrity_checks = {
        "row_ids_exact_order": bool(np.array_equal(row_ids, expected_ids)),
        "row_ids_unique": len(np.unique(row_ids)) == len(row_ids),
        "all_32_layers_present": len(per_layer) == N_LAYERS,
        "all_shapes_exact": all(value["shape"] == [len(rows), HIDDEN_SIZE] for value in per_layer.values()),
        "all_dtypes_float16": all(value["dtype"] == "float16" for value in per_layer.values()),
        "all_nan_inf_zero": all(value["nan_count"] == 0 and value["inf_count"] == 0 for value in per_layer.values()),
        "reforward_pass": reforward["status"] == "PASS",
    }
    integrity = {
        "status": "PASS" if all(integrity_checks.values()) else "FAIL",
        "activation_definition": sanity["activation_definition"],
        "union_manifest": str(OUT / "union_manifest.csv"),
        "checks": integrity_checks, "batch_trials": trials,
        "extraction": extraction, "per_layer": per_layer, "reforward": reforward,
    }
    write_json(OUT / "activation_integrity.json", integrity)
    write_json(ACT_DIR / "metadata.json", {
        "status": integrity["status"], "model_audit": audit,
        "activation_definition": sanity["activation_definition"],
        "row_count": len(rows), "layers": list(range(N_LAYERS)), "hidden_size": HIDDEN_SIZE,
        "batch_size": chosen_batch, "extraction": extraction,
        "ended_utc": utc_now(), "script_sha256": sha256(__import__("pathlib").Path(__file__).resolve()),
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": integrity["status"], "sanity": sanity["status"],
        "rows": len(rows), "batch_size": chosen_batch,
        "runtime_seconds": extraction["runtime_seconds"],
        "peak_reserved_gib": extraction["peak_reserved_gib"],
        "minimum_reforward_cosine": reforward["minimum_cosine"],
    }, indent=2))


if __name__ == "__main__":
    main()
