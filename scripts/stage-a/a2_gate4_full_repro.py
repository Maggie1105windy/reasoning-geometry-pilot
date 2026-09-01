#!/usr/bin/env python3
"""Stage A2 Gate 4: resumable full World x 32-layer Pythia-2.8B probe.

Activation semantics are delegated to the released get_layer_activations_tl()
used in Gates 2 and 3. This wrapper adds atomic per-shard persistence, resume,
integrity checks, full-data fixed-alpha probes, and requested controls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import transformer_lens.loading_from_pretrained as tl_loading


PROJECT = Path("/home/ai/projects/linear-probe-repro")
UPSTREAM = PROJECT / "external" / "world-models"
MODEL_PATH = PROJECT / "models" / "pythia-2.8b-step143000"
PROMPT_PATH = UPSTREAM / "data" / "prompt_datasets" / "pythia" / "world_place" / "coords"
ENTITY_CSV = UPSTREAM / "data" / "entity_datasets" / "world_place.csv"
RESULTS = PROJECT / "results"
ACTIVATION_DIR = PROJECT / "activations" / "full" / "pythia-2.8b" / "world" / "coords"
SHARDS = ACTIVATION_DIR / "shards"

SEED = 42
N_ROWS = 39585
N_TRAIN = 31668
N_TEST = 7917
N_LAYERS = 32
HIDDEN_DIM = 2560
LAYERS = list(range(N_LAYERS))
CONTROL_LAYERS = [15, 31]
LIVE_LAYERS = [0, 7, 15, 23, 31]
FIXED_ALPHA = 2560.0
BATCH_SIZE = 16
SHARD_SIZE = 512
MODEL_COMMIT = "dbe7ae300a54abcdc475a33907b3dff81d25709f"
MODEL_WEIGHT_SHA256 = "a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf"
UPSTREAM_COMMIT = "a572f162948ee185e9c842eb5fa15e23aad3d218"

sys.path.insert(0, str(UPSTREAM))
from save_activations import get_layer_activations_tl  # noqa: E402
from utils import adjust_precision  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload):
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(UPSTREAM), *args], text=True).strip()


def frame_text(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


@contextmanager
def local_tl_config():
    original = tl_loading.AutoConfig.from_pretrained

    def local_auto_config(_model_name, *args, **kwargs):
        return original(str(MODEL_PATH), local_files_only=True)

    tl_loading.AutoConfig.from_pretrained = local_auto_config
    try:
        yield
    finally:
        tl_loading.AutoConfig.from_pretrained = original


def shard_specs() -> list[dict]:
    specs = []
    for shard_id, start in enumerate(range(0, N_ROWS, SHARD_SIZE)):
        end = min(start + SHARD_SIZE, N_ROWS)
        specs.append({"shard_id": shard_id, "start": start, "end": end, "n_rows": end - start})
    return specs


def shard_dir(spec: dict) -> Path:
    return SHARDS / f"shard_{spec['shard_id']:05d}_rows_{spec['start']:05d}_{spec['end'] - 1:05d}"


def validate_quantized_tensor(tensor, n_rows: int) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if tuple(tensor.shape) != (n_rows, HIDDEN_DIM):
        return False
    if not tensor.is_quantized or tensor.dtype != torch.quint8:
        return False
    # The released function passes floating-point zero points, so current PyTorch
    # reports per_channel_affine_float_qparams rather than per_channel_affine.
    if tensor.qscheme() != torch.per_channel_affine_float_qparams or tensor.q_per_channel_axis() != 1:
        return False
    scales = tensor.q_per_channel_scales()
    if scales.numel() != HIDDEN_DIM or not torch.isfinite(scales).all() or not (scales > 0).all():
        return False
    return bool(torch.isfinite(tensor.dequantize()).all())


def validate_shard(spec: dict, tensor_check: bool = True) -> tuple[bool, str, dict | None]:
    directory = shard_dir(spec)
    complete_path = directory / "complete.json"
    if not complete_path.exists():
        return False, "missing complete.json", None
    try:
        record = json.loads(complete_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid complete.json: {type(exc).__name__}", None
    required = {
        "shard_id": spec["shard_id"], "start": spec["start"],
        "end": spec["end"], "n_rows": spec["n_rows"],
    }
    if any(record.get(key) != value for key, value in required.items()):
        return False, "shard bounds metadata mismatch", record
    files = record.get("layer_files", {})
    if set(files) != {str(layer) for layer in LAYERS}:
        return False, "layer file index mismatch", record
    try:
        for layer in LAYERS:
            item = files[str(layer)]
            path = directory / item["name"]
            if not path.exists() or path.stat().st_size != item["size_bytes"]:
                return False, f"layer {layer} missing/size mismatch", record
            if sha256_file(path) != item["sha256"]:
                return False, f"layer {layer} checksum mismatch", record
            if tensor_check:
                tensor = torch.load(path, map_location="cpu", weights_only=False)
                if not validate_quantized_tensor(tensor, spec["n_rows"]):
                    return False, f"layer {layer} tensor validation failed", record
                del tensor
        return True, "complete", record
    except Exception as exc:
        return False, f"validation exception: {type(exc).__name__}: {exc}", record


def make_manifest(entity_df: pd.DataFrame, prompt_dataset) -> pd.DataFrame:
    assert len(entity_df) == N_ROWS and len(prompt_dataset) == N_ROWS
    manifest = entity_df.copy()
    manifest.insert(0, "row_id", np.arange(N_ROWS, dtype=np.int64))
    manifest.insert(1, "entity", list(prompt_dataset["entity"]))
    manifest.insert(2, "raw_entity", manifest["name"])
    manifest["original_category"] = manifest["entity_type"]
    manifest = manifest[[
        "row_id", "entity", "raw_entity", "longitude", "latitude", "is_test",
        "original_category", "entity_subtype", "country",
    ]]
    flags = manifest["is_test"].astype(bool)
    assert int((~flags).sum()) == N_TRAIN and int(flags.sum()) == N_TEST
    assert manifest.row_id.tolist() == list(range(N_ROWS))
    assert manifest.row_id.is_unique
    return manifest


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    hf_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, use_safetensors=False,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    with local_tl_config():
        model = HookedTransformer.from_pretrained(
            "pythia-2.8b", hf_model=hf_cpu, tokenizer=tokenizer,
            device="cuda", dtype=torch.float16, default_prepend_bos=True,
            default_padding_side="right",
        )
    model.eval()
    assert model.cfg.n_layers == N_LAYERS and model.cfg.d_model == HIDDEN_DIM
    del hf_cpu
    gc.collect()
    return model, tokenizer


def extract_one_shard(model, full_prompt_dataset, spec: dict) -> dict:
    directory = shard_dir(spec)
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "complete.json"
    complete_path.unlink(missing_ok=True)
    for partial in directory.glob("*.partial"):
        partial.unlink(missing_ok=True)

    subset = full_prompt_dataset.select(range(spec["start"], spec["end"]))
    subset.set_format(type="torch", columns=["input_ids", "entity_mask"], output_all_columns=True)
    args = SimpleNamespace(activation_aggregation="last", batch_size=BATCH_SIZE)
    torch.cuda.reset_peak_memory_stats()
    forward_start = time.perf_counter()
    with torch.inference_mode():
        activations = get_layer_activations_tl(
            args, model, subset, layers=LAYERS, device=torch.device("cuda:0")
        )
    torch.cuda.synchronize()
    forward_runtime = time.perf_counter() - forward_start

    quant_start = time.perf_counter()
    layer_files = {}
    for layer in LAYERS:
        activation = activations.pop(layer)
        assert tuple(activation.shape) == (spec["n_rows"], HIDDEN_DIM)
        assert activation.dtype == torch.float16 and torch.isfinite(activation).all()
        quantized = adjust_precision(activation.float(), 8, per_channel=True)
        assert validate_quantized_tensor(quantized, spec["n_rows"])
        path = directory / f"layer_{layer:02d}.pt"
        atomic_torch_save(path, quantized)
        layer_files[str(layer)] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        del activation, quantized
    quant_runtime = time.perf_counter() - quant_start
    record = {
        **spec,
        "status": "complete",
        "layers": LAYERS,
        "shape_per_layer": [spec["n_rows"], HIDDEN_DIM],
        "storage_dtype": "torch.quint8",
        "qscheme": "torch.per_channel_affine_float_qparams",
        "quantization_axis": 1,
        "quantization_scope": "per shard, per layer",
        "official_function": "utils.adjust_precision(tensor.float(), 8, per_channel=True)",
        "batch_size": BATCH_SIZE,
        "forward_runtime_seconds": forward_runtime,
        "quantize_and_save_runtime_seconds": quant_runtime,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "completed_utc": utc_now(),
        "layer_files": layer_files,
    }
    atomic_json(complete_path, record)
    ok, reason, _ = validate_shard(spec, tensor_check=True)
    if not ok:
        complete_path.unlink(missing_ok=True)
        raise RuntimeError(f"post-write validation failed for shard {spec['shard_id']}: {reason}")
    return record


def aggregate_shard_records(specs: list[dict]) -> list[dict]:
    records = []
    for spec in specs:
        ok, reason, record = validate_shard(spec, tensor_check=False)
        if not ok:
            raise RuntimeError(f"incomplete shard {spec['shard_id']}: {reason}")
        records.append(record)
    return records


def load_layer(layer: int, specs: list[dict]) -> np.ndarray:
    matrix = torch.empty((N_ROWS, HIDDEN_DIM), dtype=torch.float32)
    for spec in specs:
        record = json.loads((shard_dir(spec) / "complete.json").read_text(encoding="utf-8"))
        path = shard_dir(spec) / record["layer_files"][str(layer)]["name"]
        quantized = torch.load(path, map_location="cpu", weights_only=False)
        matrix[spec["start"]:spec["end"]] = quantized.dequantize()
        del quantized
    result = matrix.numpy()
    assert result.shape == (N_ROWS, HIDDEN_DIM) and np.isfinite(result).all()
    return result


def full_integrity(specs: list[dict], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    start = time.perf_counter()
    rows = []
    for layer in LAYERS:
        nan_count = 0
        inf_count = 0
        row_count = 0
        for spec in specs:
            record = json.loads((shard_dir(spec) / "complete.json").read_text(encoding="utf-8"))
            path = shard_dir(spec) / record["layer_files"][str(layer)]["name"]
            quantized = torch.load(path, map_location="cpu", weights_only=False)
            assert validate_quantized_tensor(quantized, spec["n_rows"])
            restored = quantized.dequantize()
            nan_count += int(torch.isnan(restored).sum())
            inf_count += int(torch.isinf(restored).sum())
            row_count += restored.shape[0]
            del quantized, restored
        rows.append({
            "layer": layer,
            "hook": f"blocks.{layer}.hook_resid_post",
            "shape": f"[{row_count}, {HIDDEN_DIM}]",
            "row_count": row_count,
            "hidden_dim": HIDDEN_DIM,
            "storage_dtype": "torch.quint8",
            "dequantized_dtype": "torch.float32",
            "nan_count": nan_count,
            "inf_count": inf_count,
        })
    frame = pd.DataFrame(rows)
    checks = {
        "manifest_rows_39585": len(manifest) == N_ROWS,
        "activation_rows_39585_each_layer": bool(frame.row_count.eq(N_ROWS).all()),
        "label_rows_39585": len(manifest[["longitude", "latitude"]]) == N_ROWS,
        "layer_count_32": len(frame) == N_LAYERS,
        "hidden_dim_2560": bool(frame.hidden_dim.eq(HIDDEN_DIM).all()),
        "train_count_31668": int((~manifest.is_test.astype(bool)).sum()) == N_TRAIN,
        "test_count_7917": int(manifest.is_test.astype(bool).sum()) == N_TEST,
        "no_nan": int(frame.nan_count.sum()) == 0,
        "no_inf": int(frame.inf_count.sum()) == 0,
        "row_id_original_order": manifest.row_id.tolist() == list(range(N_ROWS)),
    }
    checks["all_pre_forward_integrity_checks"] = all(checks.values())
    checks["integrity_scan_runtime_seconds"] = time.perf_counter() - start
    return frame, checks


def live_forward_check(model, tokenizer, full_prompt_dataset, manifest, specs) -> pd.DataFrame:
    sample_ids = np.random.default_rng(SEED).choice(N_ROWS, size=10, replace=False).astype(int)
    subset = full_prompt_dataset.select(sample_ids.tolist())
    subset.set_format(type="torch", columns=["input_ids", "entity_mask"], output_all_columns=True)
    args = SimpleNamespace(activation_aggregation="last", batch_size=BATCH_SIZE)
    with torch.inference_mode():
        live = get_layer_activations_tl(
            args, model, subset, layers=LIVE_LAYERS, device=torch.device("cuda:0")
        )

    spec_by_id = {spec["shard_id"]: spec for spec in specs}
    cache = {}
    rows = []
    for sample_order, row_id in enumerate(sample_ids):
        prompt_row = subset[sample_order]
        positions = torch.where(prompt_row["entity_mask"].bool())[0]
        last = int(positions[-1])
        prompt = tokenizer.decode(
            prompt_row["input_ids"][:last + 1].tolist(), skip_special_tokens=True
        )
        entity = str(manifest.iloc[row_id]["entity"])
        assert prompt.startswith("What are the lat/lon coordinates of ") and entity in prompt
        shard_id = row_id // SHARD_SIZE
        spec = spec_by_id[shard_id]
        local_row = row_id - spec["start"]
        for layer in LIVE_LAYERS:
            key = (shard_id, layer)
            if key not in cache:
                record = json.loads((shard_dir(spec) / "complete.json").read_text(encoding="utf-8"))
                path = shard_dir(spec) / record["layer_files"][str(layer)]["name"]
                cache[key] = torch.load(path, map_location="cpu", weights_only=False).dequantize()
            saved = cache[key][local_row].float()
            current = live[layer][sample_order].float()
            delta = (saved - current).abs()
            rows.append({
                "sample_order": sample_order,
                "row_id": row_id,
                "entity": entity,
                "prompt": prompt,
                "entity_last_token_index": last,
                "entity_last_token": tokenizer.convert_ids_to_tokens(int(prompt_row["input_ids"][last])),
                "layer": layer,
                "mae": float(delta.mean()),
                "max_absolute_error": float(delta.max()),
                "cosine_similarity": float(F.cosine_similarity(saved[None], current[None]).item()),
            })
    return pd.DataFrame(rows)


def scores(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2_overall": float(r2_score(target, pred)),
        "r2_longitude": float(r2_score(target[:, 0], pred[:, 0])),
        "r2_latitude": float(r2_score(target[:, 1], pred[:, 1])),
    }


def run_probes(specs: list[dict], manifest: pd.DataFrame):
    is_test = manifest.is_test.astype(bool).to_numpy()
    target = manifest[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    train_target = target[~is_test]
    test_target = target[is_test]
    mean = train_target.mean(axis=0)
    std = train_target.std(axis=0)
    normalized_train = (train_target - mean) / std
    scaler = {
        "target_order": ["longitude", "latitude"], "ddof": 0,
        "fit_split": "all official is_test=False rows", "fit_rows": N_TRAIN,
        "longitude_mean": float(mean[0]), "longitude_std": float(std[0]),
        "latitude_mean": float(mean[1]), "latitude_std": float(std[1]),
    }
    atomic_json(RESULTS / "a2_4_target_scaler.json", scaler)
    baseline = scores(test_target, np.broadcast_to(mean, test_target.shape))
    atomic_json(RESULTS / "a2_4_mean_baseline.json", baseline)

    probe_rows = []
    control_rows = []
    permutation = np.random.default_rng(SEED).permutation(N_TRAIN)
    with threadpool_limits(limits=min(16, os.cpu_count() or 1)):
        for layer in LAYERS:
            load_start = time.perf_counter()
            matrix = load_layer(layer, specs)
            load_runtime = time.perf_counter() - load_start
            x_train = matrix[~is_test]
            x_test = matrix[is_test]
            assert x_train.shape == (N_TRAIN, HIDDEN_DIM)
            assert x_test.shape == (N_TEST, HIDDEN_DIM)
            fit_start = time.perf_counter()
            probe = Ridge(alpha=FIXED_ALPHA)
            probe.fit(x_train, normalized_train)
            pred = probe.predict(x_test) * std + mean
            fit_runtime = time.perf_counter() - fit_start
            result = scores(test_target, pred)
            coef = np.asarray(probe.coef_)
            intercept = np.asarray(probe.intercept_)
            assert coef.shape == (2, HIDDEN_DIM) and intercept.shape == (2,)
            assert np.isfinite(coef).all() and np.isfinite(intercept).all()
            probe_rows.append({
                "layer": layer, "relative_depth": layer / 31.0,
                "n_train": N_TRAIN, "n_test": N_TEST,
                "hidden_dim": HIDDEN_DIM, "alpha": FIXED_ALPHA,
                **result,
                "load_dequantize_runtime_seconds": load_runtime,
                "fit_runtime_seconds": fit_runtime,
                "coef_l2_norm": float(np.linalg.norm(coef)),
                "coef_l2_longitude": float(np.linalg.norm(coef[0])),
                "coef_l2_latitude": float(np.linalg.norm(coef[1])),
                "intercept_l2_norm": float(np.linalg.norm(intercept)),
                "coef_shape": str(list(coef.shape)),
                "intercept_shape": str(list(intercept.shape)),
                "coef_nan_count": int(np.isnan(coef).sum()),
                "coef_inf_count": int(np.isinf(coef).sum()),
            })

            if layer in CONTROL_LAYERS:
                control_start = time.perf_counter()
                shuffled = Ridge(alpha=FIXED_ALPHA)
                shuffled.fit(x_train, normalized_train[permutation])
                shuffled_pred = shuffled.predict(x_test) * std + mean
                control_runtime = time.perf_counter() - control_start
                shuffled_result = scores(test_target, shuffled_pred)
                control_rows.append({
                    "layer": layer, "seed": SEED, "alpha": FIXED_ALPHA,
                    "shuffle_scope": "training labels only", "test_labels_unchanged": True,
                    "normal_r2_overall": result["r2_overall"],
                    "random_label_r2_overall": shuffled_result["r2_overall"],
                    "r2_difference_normal_minus_random": result["r2_overall"] - shuffled_result["r2_overall"],
                    "random_label_r2_longitude": shuffled_result["r2_longitude"],
                    "random_label_r2_latitude": shuffled_result["r2_latitude"],
                    "fit_runtime_seconds": control_runtime,
                })
            del matrix, x_train, x_test, probe, pred
            gc.collect()

    probe_df = pd.DataFrame(probe_rows)
    control_df = pd.DataFrame(control_rows)
    probe_df.to_csv(RESULTS / "a2_4_full_layer_r2.csv", index=False)
    control_df.to_csv(RESULTS / "a2_4_random_label_control.csv", index=False)
    if not control_df.r2_difference_normal_minus_random.gt(0.1).all():
        raise RuntimeError("random-label control did not materially degrade; stop for investigation")
    return probe_df, control_df, scaler, baseline


def make_plots(probe_df: pd.DataFrame, baseline: dict):
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    ax.plot(probe_df.layer, probe_df.r2_overall, marker="o", markersize=3.5, linewidth=1.8)
    ax.axhline(baseline["r2_overall"], color="0.45", linestyle="--", label="Train-mean baseline")
    ax.set(xlabel="Transformer layer", ylabel="Held-out overall R²",
           title="Pythia-2.8B World coords full 32-layer probe")
    ax.set_xticks(range(0, 32, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "a2_4_full_layer_r2.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    ax.plot(probe_df.layer, probe_df.r2_longitude, marker="o", markersize=3.5, label="Longitude R²")
    ax.plot(probe_df.layer, probe_df.r2_latitude, marker="s", markersize=3.5, label="Latitude R²")
    ax.set(xlabel="Transformer layer", ylabel="Held-out R²",
           title="Pythia-2.8B World coords: longitude vs latitude")
    ax.set_xticks(range(0, 32, 2))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "a2_4_lat_lon_layer_r2.png", bbox_inches="tight")
    plt.close(fig)


def mini_comparison(probe_df: pd.DataFrame) -> pd.DataFrame:
    mini = pd.read_csv(RESULTS / "a2_3_mini_probe_fixed_alpha.csv")
    mini = mini[mini.layer.isin(LIVE_LAYERS)][["layer", "r2_overall", "r2_longitude", "r2_latitude"]]
    mini = mini.rename(columns={c: f"mini_{c}" for c in mini.columns if c != "layer"})
    full = probe_df[probe_df.layer.isin(LIVE_LAYERS)][["layer", "r2_overall", "r2_longitude", "r2_latitude"]]
    full = full.rename(columns={c: f"full_{c}" for c in full.columns if c != "layer"})
    comparison = full.merge(mini, on="layer", validate="one_to_one")
    for coord in ["overall", "longitude", "latitude"]:
        comparison[f"delta_full_minus_mini_{coord}"] = (
            comparison[f"full_r2_{coord}"] - comparison[f"mini_r2_{coord}"]
        )
    comparison.to_csv(RESULTS / "a2_4_gate3_comparison.csv", index=False)
    return comparison


def write_integrity_report(stats_df, checks, live_df, shard_records):
    live_ok = bool(live_df.cosine_similarity.min() > 0.999)
    lines = [
        "# Stage A2 Gate 4 — Full Activation Integrity",
        "", f"Status: **{'PASS' if checks['all_pre_forward_integrity_checks'] and live_ok else 'FAIL'}**",
        "", "## Storage", "",
        "- 78 atomic/resumable shards, each with separate layer files",
        "- Official `adjust_precision(..., output_precision=8, per_channel=True)`",
        "- Storage dtype/qscheme: `torch.quint8` / `torch.per_channel_affine_float_qparams`, axis 1",
        "- Quantization parameters are fit independently per shard and layer",
        "- Probe inputs are dequantized to float32",
        "", "## Shape and finite-value checks", "", frame_text(stats_df),
        "", "## Ten-row live-forward comparison", "", frame_text(live_df),
        "", f"- Minimum cosine similarity: {live_df.cosine_similarity.min():.9f}",
        f"- Mean MAE: {live_df.mae.mean():.9g}",
        f"- Maximum absolute error: {live_df.max_absolute_error.max():.9g}",
        "", "## Checklist", "",
    ]
    for key, value in checks.items():
        if isinstance(value, bool):
            lines.append(f"- [{'x' if value else ' '}] {key}")
    lines.append(f"- [{'x' if live_ok else ' '}] live_forward_min_cosine_gt_0.999")
    (RESULTS / "a2_4_full_activation_integrity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not checks["all_pre_forward_integrity_checks"] or not live_ok:
        raise RuntimeError("full activation integrity gate failed; probes must not run")


def write_preliminary_report(probe_df, controls, scaler, baseline, comparison, metadata):
    peak = probe_df.loc[probe_df.r2_overall.idxmax()]
    lines = [
        "# Stage A2 Full Reproduction Report",
        "", "Status: **COMPUTATION PASS; PAPER COMPARISON PENDING FINAL SOURCE AUDIT**",
        "", "## 1. Goal", "",
        "Best-effort methodological reproduction of the full World × 32-layer Pythia-2.8B spatial linear probing curve.",
        "", "## 2. Paper setup", "",
        "Gurnee & Tegmark, *Language Models Represent Space and Time*, Figure 2 World experiment.",
        "", "## 3. Reproduction scope", "",
        "All 39,585 released World rows, all 32 Pythia-2.8B transformer blocks, coords prompt, official `is_test` split.",
        "", "## 4. Known unavoidable deviations", "",
        "- The paper/released code does not pin the Pythia revision; this run pins `step143000` at its recorded commit.",
        "- Released `RIDGE_ALPHAS` has no Pythia entry. The primary probe uses the released fallback `Ridge(alpha=activation_dimension)`, hence alpha 2560.",
        "- For resume support, official per-channel quint8 quantization is applied independently per shard/layer rather than after assembling a complete layer matrix.",
        "", "## 5. Model", "",
        f"`EleutherAI/pythia-2.8b`, `step143000`, commit `{MODEL_COMMIT}`; 32 blocks, hidden size 2560.",
        "", "## 6. Dataset", "",
        "39,585 total; 31,668 official train; 7,917 official test; original row order preserved.",
        "", "## 7. Prompt/tokenization", "",
        "`What are the lat/lon coordinates of <entity>`; released Pythia tokenization and entity mask.",
        "", "## 8. Activation definition", "",
        "Entity-last-token activation from `blocks.{layer}.hook_resid_post`; embedding excluded.",
        "", "## 9. Probe definition", "",
        "A new independent `Ridge(alpha=2560)` per layer. Targets `[longitude, latitude]` standardized using train-only statistics and unscaled before held-out R².",
        "", "## 10. Full-data results", "", frame_text(probe_df),
        "", f"Peak overall R²: **{peak.r2_overall:.6f}** at layer **{int(peak.layer)}**.",
        "", "## 11. Layer-wise curve", "",
        "See `results/a2_4_full_layer_r2.png`; raw points only, no smoothing or interpolation.",
        "", "## 12. Longitude vs latitude", "",
        "See `results/a2_4_lat_lon_layer_r2.png` and the coordinate-specific columns above.",
        "", "## 13. Controls", "",
        f"Train-mean baseline: `{json.dumps(baseline)}`", "", frame_text(controls),
        "", "## 14. Comparison with Gate 3", "", frame_text(comparison),
        "", "## 15. Comparison with Figure 2", "",
        "Pending final audit of released notebooks and the paper figure. No visual estimate is treated as an official numeric series.",
        "", "## 16. Reproduction assessment", "",
        "Pending the Figure 2 source audit. The computational and control gates pass.",
        "", "## 17. Limitations", "",
        "Linear decodability does not establish that the model uses the representation, grounding, symbolic reasoning, or a causal mechanism.",
        "", "## 18. Exact artifact paths", "",
        "- `results/a2_4_full_layer_r2.csv`",
        "- `results/a2_4_full_layer_r2.png`",
        "- `results/a2_4_lat_lon_layer_r2.png`",
        "- `results/a2_4_random_label_control.csv`",
        "- `results/a2_4_full_activation_integrity.md`",
        "- `activations/full/pythia-2.8b/world/coords/metadata.json`",
        "- `activations/full/pythia-2.8b/world/coords/integrity_report.json`",
    ]
    (RESULTS / "A2_FULL_REPRODUCTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-shards", type=int, default=0,
                        help="0 means unlimited; positive values support benchmark/resume testing")
    args_cli = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 16 * 2**30:
        raise RuntimeError(f"only {free_bytes / 2**30:.2f} GiB GPU memory free; stop rather than compete")
    if git_output("rev-parse", "HEAD") != UPSTREAM_COMMIT or git_output("status", "--porcelain") != "":
        raise RuntimeError("external/world-models revision/cleanliness precondition failed")

    RESULTS.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    script_path = Path(__file__).resolve()
    specs = shard_specs()
    assert len(specs) == 78 and sum(spec["n_rows"] for spec in specs) == N_ROWS

    full_prompt_dataset = load_from_disk(str(PROMPT_PATH))
    full_prompt_dataset.reset_format()
    entity_df = pd.read_csv(ENTITY_CSV)
    manifest = make_manifest(entity_df, full_prompt_dataset)
    manifest.to_csv(RESULTS / "a2_4_full_sample_manifest.csv", index=False)
    manifest.to_csv(ACTIVATION_DIR / "sample_manifest.csv", index=False)

    initial = []
    for spec in specs:
        ok, reason, record = validate_shard(spec, tensor_check=True)
        initial.append({**spec, "valid": ok, "reason": reason})
    pending = [spec for spec, state in zip(specs, initial) if not state["valid"]]
    limit = len(pending) if args_cli.max_new_shards <= 0 else min(args_cli.max_new_shards, len(pending))
    selected = pending[:limit]

    model = tokenizer = None
    if selected:
        model, tokenizer = load_model_and_tokenizer()
        for number, spec in enumerate(selected, start=1):
            print(f"EXTRACT shard {spec['shard_id']} ({number}/{len(selected)}) rows {spec['start']}:{spec['end']}", flush=True)
            record = extract_one_shard(model, full_prompt_dataset, spec)
            progress = {
                "timestamp_utc": utc_now(), "total_shards": len(specs),
                "new_shards_requested_this_run": limit,
                "last_completed_shard": spec["shard_id"],
                "last_forward_runtime_seconds": record["forward_runtime_seconds"],
            }
            atomic_json(RESULTS / "a2_4_extraction_progress.json", progress)

    completion_states = [validate_shard(spec, tensor_check=False)[0] for spec in specs]
    if not all(completion_states):
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps({
            "status": "INCOMPLETE_RESUMABLE",
            "complete_shards": int(sum(completion_states)),
            "total_shards": len(specs),
            "remaining_shards": len(specs) - int(sum(completion_states)),
            "rerun_command": str(script_path),
        }, indent=2))
        return

    shard_records = aggregate_shard_records(specs)
    stats_df, integrity_checks = full_integrity(specs, manifest)
    stats_df.to_csv(RESULTS / "a2_4_activation_stats.csv", index=False)
    if model is None:
        model, tokenizer = load_model_and_tokenizer()
    live_df = live_forward_check(model, tokenizer, full_prompt_dataset, manifest, specs)
    live_df.to_csv(RESULTS / "a2_4_live_forward_comparison.csv", index=False)
    write_integrity_report(stats_df, integrity_checks, live_df, shard_records)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    probe_df, controls, scaler, baseline = run_probes(specs, manifest)
    make_plots(probe_df, baseline)
    comparison = mini_comparison(probe_df)

    extraction_forward_seconds = sum(float(x["forward_runtime_seconds"]) for x in shard_records)
    extraction_save_seconds = sum(float(x["quantize_and_save_runtime_seconds"]) for x in shard_records)
    peak_allocated = max(int(x["peak_allocated_bytes"]) for x in shard_records)
    peak_reserved = max(int(x["peak_reserved_bytes"]) for x in shard_records)
    storage_bytes = sum(
        int(item["size_bytes"])
        for record in shard_records for item in record["layer_files"].values()
    )
    metadata = {
        "status": "PASS",
        "experiment_nature": "Best-effort methodological reproduction",
        "model_name": "EleutherAI/pythia-2.8b",
        "model_revision": "step143000", "hf_commit": MODEL_COMMIT,
        "model_path": str(MODEL_PATH), "model_weight_file": "pytorch_model.bin",
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "dataset": "world_place", "dataset_row_count": N_ROWS,
        "train_count": N_TRAIN, "test_count": N_TEST,
        "prompt_type": "coords", "prompt_template": "What are the lat/lon coordinates of <entity>",
        "activation_hook": "blocks.{layer}.hook_resid_post",
        "entity_position": "last entity token selected by released entity_mask aggregation",
        "n_layers": N_LAYERS, "hidden_size": HIDDEN_DIM,
        "storage_dtype": "torch.quint8", "dequantized_dtype": "torch.float32",
        "quantization": {
            "function": "official utils.adjust_precision", "bits": 8,
            "per_channel": True, "qscheme": "torch.per_channel_affine_float_qparams", "axis": 1,
            "scope": "independently per shard and layer",
        },
        "shard_size": SHARD_SIZE, "shard_count": len(specs),
        "batch_size": BATCH_SIZE, "seed": SEED,
        "extraction_script": str(script_path), "extraction_script_sha256": sha256_file(script_path),
        "external_world_models_commit": UPSTREAM_COMMIT,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "transformer_lens_version": importlib.metadata.version("transformer-lens"),
        "transformers_version": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "start_time_utc_this_completion_run": started_utc, "end_time_utc": utc_now(),
        "aggregate_forward_runtime_seconds": extraction_forward_seconds,
        "aggregate_quantize_save_runtime_seconds": extraction_save_seconds,
        "samples_per_forward_second": N_ROWS / extraction_forward_seconds,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "activation_layer_files_total_bytes": storage_bytes,
        "integrity_checks": integrity_checks,
        "live_forward_min_cosine": float(live_df.cosine_similarity.min()),
        "live_forward_mean_mae": float(live_df.mae.mean()),
        "probe_alpha": FIXED_ALPHA,
        "upstream_clean_at_completion": git_output("status", "--porcelain") == "",
    }
    atomic_json(ACTIVATION_DIR / "metadata.json", metadata)
    atomic_json(ACTIVATION_DIR / "integrity_report.json", {
        "checks": integrity_checks,
        "live_forward_min_cosine": float(live_df.cosine_similarity.min()),
        "live_forward_mean_mae": float(live_df.mae.mean()),
        "live_forward_max_absolute_error": float(live_df.max_absolute_error.max()),
        "shards": [{
            "shard_id": x["shard_id"], "start": x["start"], "end": x["end"],
            "n_rows": x["n_rows"], "status": x["status"],
        } for x in shard_records],
    })
    write_preliminary_report(probe_df, controls, scaler, baseline, comparison, metadata)

    peak = probe_df.loc[probe_df.r2_overall.idxmax()]
    print(json.dumps({
        "status": "PASS",
        "sample_count": N_ROWS, "train_count": N_TRAIN, "test_count": N_TEST,
        "layer_count": N_LAYERS, "shape_per_layer": [N_ROWS, HIDDEN_DIM],
        "aggregate_forward_runtime_seconds": extraction_forward_seconds,
        "samples_per_second": N_ROWS / extraction_forward_seconds,
        "peak_allocated_gib": peak_allocated / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
        "stored_activation_gib": storage_bytes / 2**30,
        "live_forward_min_cosine": float(live_df.cosine_similarity.min()),
        "peak_layer": int(peak.layer), "peak_r2": float(peak.r2_overall),
        "baseline": baseline,
        "random_label_controls": controls.to_dict(orient="records"),
        "upstream_clean": metadata["upstream_clean_at_completion"],
    }, indent=2))


if __name__ == "__main__":
    main()
