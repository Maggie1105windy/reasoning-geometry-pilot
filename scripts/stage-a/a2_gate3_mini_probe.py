#!/usr/bin/env python3
"""Stage A2 Gate 3: mini Pythia-2.8B world/coords layer-wise probe.

This intentionally stops at 4,000 official-train and 1,000 official-test rows
and five predeclared layers. It imports the released activation extractor rather
than reimplementing entity-last-token aggregation.
"""

from __future__ import annotations

import gc
import json
import os
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
from datasets import load_from_disk
from sklearn.linear_model import Ridge, RidgeCV
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
ACTIVATION_DIR = PROJECT / "activations" / "mini" / "pythia-2.8b" / "world" / "coords"

SEED = 42
N_TRAIN = 4000
N_TEST = 1000
LAYERS = [0, 7, 15, 23, 31]
HIDDEN_DIM = 2560
FIXED_ALPHA = 2560.0
MODEL_COMMIT = "dbe7ae300a54abcdc475a33907b3dff81d25709f"
RIDGECV_ALPHAS = np.array(
    sorted(set(np.logspace(1, 5, 17).tolist() + [FIXED_ALPHA])), dtype=np.float64
)

sys.path.insert(0, str(UPSTREAM))
from save_activations import get_layer_activations_tl  # noqa: E402


@contextmanager
def local_tl_config():
    """Redirect TransformerLens 1.14 config lookup to the pinned local model."""
    original = tl_loading.AutoConfig.from_pretrained

    def local_auto_config(_model_name, *args, **kwargs):
        return original(str(MODEL_PATH), local_files_only=True)

    tl_loading.AutoConfig.from_pretrained = local_auto_config
    try:
        yield
    finally:
        tl_loading.AutoConfig.from_pretrained = original


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), *args], text=True
    ).strip()


def frame_text(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def gpu_peak_record(phase: str, runtime_seconds: float, batch_size: int, status: str) -> dict:
    return {
        "phase": phase,
        "batch_size": batch_size,
        "status": status,
        "runtime_seconds": runtime_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
    }


def choose_manifest(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Sample independently inside each released is_test partition."""
    rng = np.random.default_rng(SEED)
    flags = entity_df["is_test"].astype(bool)
    train_pool = entity_df.index[~flags].to_numpy(dtype=np.int64)
    test_pool = entity_df.index[flags].to_numpy(dtype=np.int64)
    train_rows = rng.choice(train_pool, size=N_TRAIN, replace=False)
    test_rows = rng.choice(test_pool, size=N_TEST, replace=False)
    row_ids = np.concatenate([train_rows, test_rows])

    manifest = entity_df.loc[row_ids].copy()
    manifest.insert(0, "sample_order", np.arange(len(manifest), dtype=np.int64))
    manifest.insert(1, "row_id", manifest.index.astype(np.int64))
    manifest.insert(2, "entity", manifest["name"])
    manifest["split"] = np.where(manifest["is_test"].astype(bool), "test", "train")
    manifest["original_category"] = manifest["entity_type"]
    manifest = manifest[[
        "sample_order", "row_id", "entity", "latitude", "longitude", "is_test",
        "split", "original_category", "entity_subtype", "country",
    ]].reset_index(drop=True)

    assert len(manifest) == N_TRAIN + N_TEST
    assert manifest.iloc[:N_TRAIN]["is_test"].astype(bool).eq(False).all()
    assert manifest.iloc[N_TRAIN:]["is_test"].astype(bool).eq(True).all()
    assert manifest["row_id"].is_unique
    assert set(train_rows).isdisjoint(set(test_rows))
    return manifest


def prepare_subset(manifest: pd.DataFrame):
    full = load_from_disk(str(PROMPT_PATH))
    full.reset_format()
    row_ids = manifest["row_id"].astype(int).tolist()
    subset = full.select(row_ids)
    prompt_entities = list(subset["entity"])
    manifest = manifest.copy()
    manifest["raw_entity"] = manifest["entity"]
    manifest["entity"] = prompt_entities
    subset.set_format(
        type="torch", columns=["input_ids", "entity_mask"], output_all_columns=True
    )
    assert tuple(subset["input_ids"].shape)[0] == N_TRAIN + N_TEST
    assert tuple(subset["input_ids"].shape) == tuple(subset["entity_mask"].shape)
    return manifest, subset


def make_alignment_checks(manifest: pd.DataFrame, subset, tokenizer) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    indices = rng.choice(len(manifest), size=10, replace=False)
    records = []
    for sample_index in indices:
        row = subset[int(sample_index)]
        ids = row["input_ids"]
        mask = row["entity_mask"].bool()
        positions = torch.where(mask)[0]
        assert len(positions) > 0
        last = int(positions[-1])
        first = int(positions[0])
        prompt = tokenizer.decode(ids[: last + 1].tolist(), skip_special_tokens=True)
        expected_entity = str(manifest.iloc[sample_index]["entity"])
        assert prompt.startswith("What are the lat/lon coordinates of ")
        assert expected_entity in prompt
        records.append({
            "sample_index": int(sample_index),
            "row_id": int(manifest.iloc[sample_index]["row_id"]),
            "split": str(manifest.iloc[sample_index]["split"]),
            "entity": expected_entity,
            "prompt": prompt,
            "entity_first_token_index": first,
            "entity_last_token_index": last,
            "entity_last_token_id": int(ids[last]),
            "entity_last_token": tokenizer.convert_ids_to_tokens(int(ids[last])),
            "entity_mask_true_count": int(mask.sum()),
            "row_alignment_ok": True,
        })
    return pd.DataFrame(records)


def run_batch_trial(model, subset, batch_size: int, n_rows: int) -> tuple[dict, dict[int, torch.Tensor] | None]:
    trial = subset.select(range(n_rows))
    trial.set_format(
        type="torch", columns=["input_ids", "entity_mask"], output_all_columns=True
    )
    args = SimpleNamespace(activation_aggregation="last", batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            acts = get_layer_activations_tl(
                args, model, trial, layers=LAYERS, device=torch.device("cuda:0")
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        for layer in LAYERS:
            assert tuple(acts[layer].shape) == (n_rows, HIDDEN_DIM)
            assert acts[layer].dtype == torch.float16
        return gpu_peak_record(
            f"batch_trial_{batch_size}", elapsed, batch_size, "PASS"
        ), acts
    except torch.cuda.OutOfMemoryError:
        elapsed = time.perf_counter() - start
        model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()
        return gpu_peak_record(
            f"batch_trial_{batch_size}", elapsed, batch_size, "OOM"
        ), None


def extract_activations(model, subset, batch_size: int) -> tuple[dict[int, torch.Tensor], dict]:
    args = SimpleNamespace(activation_aggregation="last", batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        activations = get_layer_activations_tl(
            args, model, subset, layers=LAYERS, device=torch.device("cuda:0")
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for layer in LAYERS:
        tensor = activations[layer]
        assert tuple(tensor.shape) == (N_TRAIN + N_TEST, HIDDEN_DIM)
        assert tensor.dtype == torch.float16
        assert torch.isfinite(tensor).all()
    return activations, gpu_peak_record(
        "formal_extraction_5000", elapsed, batch_size, "PASS"
    )


def score_predictions(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Match released score_place_probe x/y/overall R2 semantics."""
    return {
        "r2_overall": float(r2_score(target, pred)),
        "r2_longitude": float(r2_score(target[:, 0], pred[:, 0])),
        "r2_latitude": float(r2_score(target[:, 1], pred[:, 1])),
    }


def fit_and_score(probe, x_train, x_test, y_train_norm, y_test, mean, std):
    start = time.perf_counter()
    probe.fit(x_train, y_train_norm)
    pred = probe.predict(x_test) * std + mean
    elapsed = time.perf_counter() - start
    scores = score_predictions(y_test, pred)
    coef = np.asarray(probe.coef_)
    intercept = np.asarray(probe.intercept_)
    assert coef.shape == (2, HIDDEN_DIM)
    assert intercept.shape == (2,)
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    return probe, scores, elapsed, {
        "coef_shape": str(list(coef.shape)),
        "intercept_shape": str(list(intercept.shape)),
        "coef_l2_longitude": float(np.linalg.norm(coef[0])),
        "coef_l2_latitude": float(np.linalg.norm(coef[1])),
        "intercept_l2": float(np.linalg.norm(intercept)),
        "coef_nan_count": int(np.isnan(coef).sum()),
        "coef_inf_count": int(np.isinf(coef).sum()),
    }


def plot_results(fixed_df: pd.DataFrame, cv_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    ax.plot(
        fixed_df["layer"], fixed_df["r2_overall"], marker="o", linewidth=2,
        label="Ridge(alpha=2560; primary)",
    )
    ax.plot(
        cv_df["layer"], cv_df["r2_overall"], marker="s", linewidth=2,
        label="RidgeCV (sensitivity)",
    )
    ax.axhline(
        fixed_df["baseline_r2_overall"].iloc[0], color="0.45", linestyle="--",
        label="Train-mean baseline",
    )
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Transformer block (hook_resid_post)")
    ax.set_ylabel("Held-out R² (longitude/latitude uniform average)")
    ax.set_title("Pythia-2.8B World coords mini probe (4,000 train / 1,000 test)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    upstream_commit = git_output("rev-parse", "HEAD")
    assert upstream_commit == "a572f162948ee185e9c842eb5fa15e23aad3d218"
    assert git_output("status", "--porcelain") == ""

    entity_df = pd.read_csv(ENTITY_CSV)
    manifest = choose_manifest(entity_df)
    manifest, subset = prepare_subset(manifest)
    manifest.to_csv(RESULTS / "a2_3_sample_manifest.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    alignment_df = make_alignment_checks(manifest, subset, tokenizer)
    alignment_df.to_csv(RESULTS / "a2_3_alignment_check.csv", index=False)

    hf_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    with local_tl_config():
        model = HookedTransformer.from_pretrained(
            "pythia-2.8b",
            hf_model=hf_cpu,
            tokenizer=tokenizer,
            device="cuda",
            dtype=torch.float16,
            default_prepend_bos=True,
            default_padding_side="right",
        )
    model.eval()
    assert model.cfg.n_layers == 32
    assert model.cfg.d_model == HIDDEN_DIM
    del hf_cpu
    gc.collect()

    batch_trials = []
    trial8, acts8 = run_batch_trial(model, subset, batch_size=8, n_rows=8)
    batch_trials.append(trial8)
    del acts8
    trial16, acts16 = run_batch_trial(model, subset, batch_size=16, n_rows=16)
    batch_trials.append(trial16)
    del acts16
    chosen_batch = 16 if trial16["status"] == "PASS" else 8
    if trial8["status"] != "PASS":
        raise RuntimeError("Batch size 8 failed; Gate 3 must stop")
    if trial16["status"] == "PASS" and trial16["peak_reserved_gib"] > 22.0:
        chosen_batch = 8

    activations, extraction_record = extract_activations(model, subset, chosen_batch)
    batch_trials.append(extraction_record)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    activation_shapes = {}
    activation_stats = []
    for layer in LAYERS:
        tensor = activations[layer]
        activation_shapes[str(layer)] = {
            "combined": list(tensor.shape),
            "train": list(tensor[:N_TRAIN].shape),
            "test": list(tensor[N_TRAIN:].shape),
            "dtype": str(tensor.dtype),
        }
        tf = tensor.float()
        activation_stats.append({
            "layer": layer,
            "hook": f"blocks.{layer}.hook_resid_post",
            "combined_shape": str(list(tensor.shape)),
            "train_shape": str(list(tensor[:N_TRAIN].shape)),
            "test_shape": str(list(tensor[N_TRAIN:].shape)),
            "dtype": str(tensor.dtype),
            "nan_count": int(torch.isnan(tf).sum()),
            "inf_count": int(torch.isinf(tf).sum()),
            "mean": float(tf.mean()),
            "std": float(tf.std(unbiased=False)),
        })
    pd.DataFrame(activation_stats).to_csv(
        RESULTS / "a2_3_activation_stats.csv", index=False
    )
    torch.save({layer: activations[layer] for layer in LAYERS}, ACTIVATION_DIR / "activations.pt")

    # Released target order is [longitude, latitude]. Standardization is train-only.
    targets = manifest[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    y_train = targets[:N_TRAIN]
    y_test = targets[N_TRAIN:]
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)
    assert np.all(target_std > 0)
    y_train_norm = (y_train - target_mean) / target_std
    scaler = {
        "target_order": ["longitude", "latitude"],
        "ddof": 0,
        "fit_rows": N_TRAIN,
        "fit_split": "official is_test=False only",
        "longitude_mean": float(target_mean[0]),
        "longitude_std": float(target_std[0]),
        "latitude_mean": float(target_mean[1]),
        "latitude_std": float(target_std[1]),
    }
    (RESULTS / "a2_3_target_scaler.json").write_text(
        json.dumps(scaler, indent=2) + "\n", encoding="utf-8"
    )

    baseline_pred = np.broadcast_to(target_mean, y_test.shape)
    baseline_scores = score_predictions(y_test, baseline_pred)
    fixed_rows = []
    cv_rows = []
    coefficient_rows = []
    x_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    with threadpool_limits(limits=min(16, os.cpu_count() or 1)):
        for layer in LAYERS:
            x_train = activations[layer][:N_TRAIN].float().numpy()
            x_test = activations[layer][N_TRAIN:].float().numpy()
            assert x_train.shape == (N_TRAIN, HIDDEN_DIM)
            assert x_test.shape == (N_TEST, HIDDEN_DIM)
            assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
            x_cache[layer] = (x_train, x_test)

            fixed, scores, runtime, coef_info = fit_and_score(
                Ridge(alpha=FIXED_ALPHA), x_train, x_test, y_train_norm,
                y_test, target_mean, target_std,
            )
            fixed_rows.append({
                "layer": layer,
                "relative_depth": layer / 31.0,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "hidden_dim": HIDDEN_DIM,
                "alpha": FIXED_ALPHA,
                **scores,
                "baseline_r2_overall": baseline_scores["r2_overall"],
                "baseline_r2_longitude": baseline_scores["r2_longitude"],
                "baseline_r2_latitude": baseline_scores["r2_latitude"],
                "fit_predict_runtime_seconds": runtime,
            })
            coefficient_rows.append({"analysis": "fixed_alpha", "layer": layer, **coef_info})

            cv, cv_scores, cv_runtime, cv_coef_info = fit_and_score(
                RidgeCV(alphas=RIDGECV_ALPHAS, store_cv_values=True),
                x_train, x_test, y_train_norm, y_test, target_mean, target_std,
            )
            selected_alpha = float(np.asarray(cv.alpha_).reshape(-1)[0])
            cv_rows.append({
                "layer": layer,
                "relative_depth": layer / 31.0,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "hidden_dim": HIDDEN_DIM,
                "selected_alpha": selected_alpha,
                "alpha_grid": json.dumps(RIDGECV_ALPHAS.tolist()),
                **cv_scores,
                "baseline_r2_overall": baseline_scores["r2_overall"],
                "baseline_r2_longitude": baseline_scores["r2_longitude"],
                "baseline_r2_latitude": baseline_scores["r2_latitude"],
                "fit_predict_runtime_seconds": cv_runtime,
                "analysis_label": "our robustness/sensitivity analysis",
            })
            coefficient_rows.append({"analysis": "ridgecv_sensitivity", "layer": layer, **cv_coef_info})

        # Negative control: shuffle training targets only; test labels are untouched.
        x15_train, x15_test = x_cache[15]
        permutation = np.random.default_rng(SEED).permutation(N_TRAIN)
        shuffled, shuffled_scores, shuffled_runtime, shuffled_coef = fit_and_score(
            Ridge(alpha=FIXED_ALPHA), x15_train, x15_test,
            y_train_norm[permutation], y_test, target_mean, target_std,
        )
        coefficient_rows.append({
            "analysis": "layer15_shuffled_train_labels", "layer": 15, **shuffled_coef
        })

    fixed_df = pd.DataFrame(fixed_rows)
    cv_df = pd.DataFrame(cv_rows)
    coef_df = pd.DataFrame(coefficient_rows)
    fixed_df.to_csv(RESULTS / "a2_3_mini_probe_fixed_alpha.csv", index=False)
    cv_df.to_csv(RESULTS / "a2_3_mini_probe_ridgecv.csv", index=False)
    coef_df.to_csv(RESULTS / "a2_3_probe_coefficients.csv", index=False)

    layer15_r2 = float(fixed_df.loc[fixed_df.layer == 15, "r2_overall"].iloc[0])
    shuffled_drop = layer15_r2 - shuffled_scores["r2_overall"]
    controls = {
        "baseline": baseline_scores,
        "random_label_control": {
            "layer": 15,
            "seed": SEED,
            "shuffle_scope": "training targets only",
            "test_targets_unchanged": True,
            "alpha": FIXED_ALPHA,
            **shuffled_scores,
            "normal_layer15_r2_overall": layer15_r2,
            "r2_drop_vs_normal": shuffled_drop,
            "fit_predict_runtime_seconds": shuffled_runtime,
        },
    }
    (RESULTS / "a2_3_controls.json").write_text(
        json.dumps(controls, indent=2) + "\n", encoding="utf-8"
    )

    plot_results(fixed_df, cv_df, RESULTS / "a2_3_mini_probe.png")

    checks = {
        "official_split_sampled_separately": bool(
            manifest.iloc[:N_TRAIN].is_test.astype(bool).eq(False).all()
            and manifest.iloc[N_TRAIN:].is_test.astype(bool).eq(True).all()
        ),
        "no_train_test_row_overlap": manifest.row_id.is_unique,
        "row_alignment_10_of_10": bool(alignment_df.row_alignment_ok.all()),
        "only_requested_five_layers": set(activations) == set(LAYERS),
        "all_shapes_5000x2560_float16": all(
            tuple(activations[layer].shape) == (5000, HIDDEN_DIM)
            and activations[layer].dtype == torch.float16 for layer in LAYERS
        ),
        "no_activation_nan_inf": all(
            torch.isfinite(activations[layer]).all().item() for layer in LAYERS
        ),
        "train_only_target_scaling": scaler["fit_split"] == "official is_test=False only",
        "target_order_longitude_latitude": scaler["target_order"] == ["longitude", "latitude"],
        "fixed_alpha_2560_all_layers": bool(fixed_df.alpha.eq(FIXED_ALPHA).all()),
        "ridgecv_grid_contains_2560": bool(FIXED_ALPHA in RIDGECV_ALPHAS),
        "all_probe_outputs_finite": bool(
            np.isfinite(fixed_df[["r2_overall", "r2_longitude", "r2_latitude"]]).all().all()
            and np.isfinite(cv_df[["r2_overall", "r2_longitude", "r2_latitude"]]).all().all()
        ),
        "coefficient_shapes_and_values_valid": bool(
            coef_df.coef_shape.eq("[2, 2560]").all()
            and coef_df.intercept_shape.eq("[2]").all()
            and coef_df.coef_nan_count.eq(0).all()
            and coef_df.coef_inf_count.eq(0).all()
        ),
        "layers_not_numerically_identical": bool(fixed_df.r2_overall.nunique() > 1),
        "random_labels_materially_degrade_layer15": bool(shuffled_drop > 0.05),
        "upstream_clean": git_output("status", "--porcelain") == "",
    }
    gate_pass = all(checks.values())

    metadata = {
        "status": "PASS" if gate_pass else "FAIL",
        "model_name": "EleutherAI/pythia-2.8b",
        "revision": "step143000",
        "huggingface_commit": MODEL_COMMIT,
        "local_model_path": str(MODEL_PATH),
        "external_world_models_git_commit": upstream_commit,
        "seed": SEED,
        "dataset": "world_place",
        "prompt": "coords",
        "prompt_template": "What are the lat/lon coordinates of <entity>",
        "official_split": "is_test",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "sample_order": "4,000 official train rows followed by 1,000 official test rows",
        "layers": LAYERS,
        "hooks": [f"blocks.{layer}.hook_resid_post" for layer in LAYERS],
        "activation_aggregation": "official last (entity last token)",
        "activation_shapes": activation_shapes,
        "model_dtype": "torch.float16",
        "activation_storage_dtype": "torch.float16",
        "sklearn_input_dtype": "float32",
        "batch_trials": batch_trials,
        "chosen_batch_size": chosen_batch,
        "target_scaler": scaler,
        "fixed_alpha": FIXED_ALPHA,
        "ridgecv_alphas": RIDGECV_ALPHAS.tolist(),
        "ridgecv_label": "our robustness/sensitivity analysis",
        "r2_semantics": "sklearn r2_score; overall default multioutput uniform_average",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": time.perf_counter() - total_start,
        "checks": checks,
    }
    (ACTIVATION_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(batch_trials).to_csv(RESULTS / "a2_3_batch_memory.csv", index=False)

    report = [
        "# Stage A2 Gate 3 — Mini Layer-wise Linear Probe",
        "",
        f"Status: **{'PASS' if gate_pass else 'FAIL'}**  ",
        "This is a bounded methodological mini-reproduction, not a quantitative reproduction of the paper's full Figure 2 curve.",
        "",
        "## Scope and provenance",
        "",
        "- Model: `EleutherAI/pythia-2.8b`, revision `step143000`",
        f"- Hugging Face commit: `{MODEL_COMMIT}`",
        f"- Released world-models commit: `{upstream_commit}`",
        "- Prompt: `What are the lat/lon coordinates of <entity>`",
        "- Split: released `is_test`; 4,000 train and 1,000 test sampled independently with seed 42",
        "- Layers/hooks: 0, 7, 15, 23, 31 at `blocks.<layer>.hook_resid_post`",
        "- No full World extraction and no other layers/prompts/models were run",
        "",
        "## Activation extraction",
        "",
        f"- Chosen batch size: {chosen_batch}",
        "- Saved dtype: float16; sklearn conversion: float32",
        "- Every layer: combined `[5000, 2560]`, train `[4000, 2560]`, test `[1000, 2560]`",
        "",
        frame_text(pd.DataFrame(batch_trials)),
        "",
        "## Ten row/token alignment checks",
        "",
        frame_text(alignment_df),
        "",
        "## Target preprocessing and metric",
        "",
        "- Released target order: `[longitude, latitude]`",
        "- Mean and population standard deviation (`ddof=0`) computed from official train rows only",
        "- Predictions unstandardized using the same train-only parameters",
        "- Overall R² uses sklearn's default uniform average over longitude and latitude",
        "",
        "```json",
        json.dumps(scaler, indent=2),
        "```",
        "",
        "## Primary probe: Ridge(alpha=2560)",
        "",
        frame_text(fixed_df),
        "",
        "## RidgeCV sensitivity analysis (ours, not claimed as the paper's Pythia setting)",
        "",
        f"Grid: `{RIDGECV_ALPHAS.tolist()}`",
        "",
        frame_text(cv_df.drop(columns=["alpha_grid"])),
        "",
        "## Baseline and negative control",
        "",
        "```json",
        json.dumps(controls, indent=2),
        "```",
        "",
        "## Coefficient sanity checks",
        "",
        frame_text(coef_df),
        "",
        "## Gate checklist",
        "",
    ]
    for name, value in checks.items():
        report.append(f"- [{'x' if value else ' '}] {name}")
    report += [
        "",
        "## Known irreducible reproduction differences",
        "",
        "- The paper/released code does not pin a Pythia revision; this run pins current `step143000` and records its commit.",
        "- Current `main_probe_experiment` publishes RidgeCV grids only for Llama models, not Pythia. Therefore Ridge(alpha=2560) is the released `probe=None` rule; the RidgeCV result above is explicitly our sensitivity analysis.",
        "- `place_probe_experiment` also computes an O(n²) proximity matrix. Gate 3 requests R² only, so this bounded run reproduces its train-only scaling, fitting, unscaling, and exact R² calls without allocating that unrelated matrix.",
        "",
        f"Final Gate 3 decision: **{'PASS' if gate_pass else 'FAIL'}**",
    ]
    (RESULTS / "a2_3_mini_probe_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "gate_pass": gate_pass,
        "checks": checks,
        "chosen_batch_size": chosen_batch,
        "batch_trials": batch_trials,
        "fixed_alpha_results": fixed_df.to_dict(orient="records"),
        "ridgecv_results": cv_df.drop(columns=["alpha_grid"]).to_dict(orient="records"),
        "controls": controls,
        "total_runtime_seconds": metadata["total_runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
