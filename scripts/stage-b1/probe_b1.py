#!/usr/bin/env python3
"""Fit all Stage B1 linear probes, controls, summaries, and figures."""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b1 import (
    ACT_DIR, B02, B02C, C_GRID, HIDDEN_SIZE, HOPS, N_LAYERS, OUT,
    RELATIONS, SEED, SPLITS, macro_accuracy_baseline, stable_int, write_json,
)


CHECKPOINTS = OUT / "probe_checkpoints"
FIGURES = OUT / "figures"
SCALERS = OUT / "scalers"
MODELS = OUT / "probe_models"
N_JOBS = 8

EXPERIMENTS = {
    "primary": {
        "column": "primary_split", "hops": HOPS, "class_weight": "balanced",
        "output": "primary_probe.csv",
    },
    "balanced": {
        "column": "balanced_split", "hops": HOPS, "class_weight": None,
        "output": "balanced_probe.csv",
    },
    "composition": {
        "column": "composition_split", "hops": [2, 3, 4, 5], "class_weight": "balanced",
        "output": "composition_heldout_probe.csv",
    },
    "template": {
        "column": "template_split", "hops": HOPS, "class_weight": "balanced",
        "output": "template_test_probe.csv",
    },
}


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def predict_from_params(x: np.ndarray, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray) -> np.ndarray:
    logits = x @ coef.T + intercept
    return classes[np.argmax(logits, axis=1)]


def fit_one(
    matrix: np.ndarray, union: pd.DataFrame, experiment: str, hop: int,
    class_weight, random_label: bool = False,
) -> dict:
    split_col = EXPERIMENTS[experiment]["column"]
    positions = {}
    for split in SPLITS:
        mask = union[split_col].eq(split) & union["computed_hop"].eq(hop)
        positions[split] = np.flatnonzero(mask.to_numpy())
    if any(len(positions[split]) == 0 for split in SPLITS):
        raise RuntimeError(f"Empty split: {experiment=} {hop=} { {k:len(v) for k,v in positions.items()} }")
    x_train = np.asarray(matrix[positions["train"]], dtype=np.float32)
    x_val = np.asarray(matrix[positions["val"]], dtype=np.float32)
    x_test = np.asarray(matrix[positions["test"]], dtype=np.float32)
    y_train = union.iloc[positions["train"]]["canonical_relation"].to_numpy(dtype=object)
    y_val = union.iloc[positions["val"]]["canonical_relation"].to_numpy(dtype=object)
    y_test = union.iloc[positions["test"]]["canonical_relation"].to_numpy(dtype=object)
    if set(y_train) != set(RELATIONS) or set(y_val) != set(RELATIONS) or set(y_test) != set(RELATIONS):
        raise RuntimeError(f"Class coverage failure: {experiment=} {hop=}")
    scaler = StandardScaler(copy=True).fit(x_train)
    train_scaled = scaler.transform(x_train).astype(np.float32, copy=False)
    val_scaled = scaler.transform(x_val).astype(np.float32, copy=False)
    test_scaled = scaler.transform(x_test).astype(np.float32, copy=False)
    del x_train, x_val, x_test

    train_labels = y_train.copy()
    analysis = "normal"
    if random_label:
        analysis = "random_train_labels"
        rng = np.random.default_rng(SEED + hop * 1009 + int(union.attrs.get("layer", 0)) * 9176)
        train_labels = train_labels[rng.permutation(len(train_labels))]

    classifier = LogisticRegression(
        C=C_GRID[0], solver="lbfgs", multi_class="multinomial", max_iter=400,
        tol=1e-4, class_weight=class_weight, warm_start=True,
        random_state=SEED,
    )
    best = None
    trace = []
    task_start = time.perf_counter()
    for c_value in C_GRID:
        classifier.C = c_value
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(train_scaled, train_labels)
        val_pred = classifier.predict(val_scaled)
        val_metrics = metrics(y_val, val_pred)
        convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        record = {
            "experiment": experiment, "analysis": analysis, "hop": hop,
            "C": c_value, "validation_macro_f1": val_metrics["macro_f1"],
            "validation_accuracy": val_metrics["accuracy"],
            "fit_seconds": time.perf_counter() - fit_start,
            "n_iter_max": int(np.max(classifier.n_iter_)),
            "convergence_warning": convergence_warnings > 0,
            "test_accessed_for_selection": False,
        }
        trace.append(record)
        candidate = (
            val_metrics["macro_f1"], -float(c_value),
            classifier.coef_.astype(np.float32, copy=True),
            classifier.intercept_.astype(np.float32, copy=True),
            classifier.classes_.astype(object, copy=True),
            val_metrics,
            convergence_warnings > 0,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    selected_f1, neg_c, coef, intercept, classes, val_metrics, selected_warning = best
    selected_c = -neg_c
    # Test is first accessed only after validation has frozen C and parameters.
    test_pred = predict_from_params(test_scaled, coef, intercept, classes)
    test_metrics = metrics(y_test, test_pred)
    result = {
        "experiment": experiment, "analysis": analysis, "hop": hop,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "hidden_size": HIDDEN_SIZE, "selected_C": selected_c,
        "validation_macro_f1": val_metrics["macro_f1"],
        "validation_accuracy": val_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "selected_fit_convergence_warning": bool(selected_warning),
        "task_seconds": time.perf_counter() - task_start,
        "scaler_fit_split": "train_only", "C_selection_split": "validation_only",
        "test_used_for_selection": False,
        "class_weight": "balanced" if class_weight == "balanced" else "none",
    }
    prediction = pd.DataFrame({
        "row_id": union.iloc[positions["test"]]["row_id"].astype(int).to_numpy(),
        "story_group_id": union.iloc[positions["test"]]["story_group_id"].astype(str).to_numpy(),
        "hop": hop, "true_relation": y_test, "predicted_relation": test_pred,
    })
    return {
        "result": result, "trace": trace, "prediction": prediction,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coef": coef, "intercept": intercept,
        "classes": np.asarray(classes, dtype=str),
    }


def layer_checkpoint_complete(layer: int) -> bool:
    marker = CHECKPOINTS / f"layer_{layer:02d}_complete.json"
    return marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "PASS"


def run_layer(layer: int, union: pd.DataFrame) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if layer_checkpoint_complete(layer):
        print(json.dumps({"layer": layer, "status": "SKIP_COMPLETE"}), flush=True)
        return
    matrix = np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")
    if tuple(matrix.shape) != (len(union), HIDDEN_SIZE):
        raise RuntimeError(f"Layer shape mismatch: {layer} {matrix.shape}")
    union.attrs["layer"] = layer
    tasks = []
    for experiment, spec in EXPERIMENTS.items():
        for hop in spec["hops"]:
            tasks.append((experiment, hop, spec["class_weight"], False))
    if layer in [0, 7, 15, 23, 31]:
        for hop in [1, 3, 5]:
            tasks.append(("primary", hop, "balanced", True))
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        outputs = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=0)(
            delayed(fit_one)(matrix, union, experiment, hop, weight, random_label)
            for experiment, hop, weight, random_label in tasks
        )
    normal_results, random_results, trace_rows, primary_predictions = [], [], [], []
    scaler_payload, model_payload = {}, {}
    for output in outputs:
        result = output["result"]
        result["layer"] = layer
        for record in output["trace"]:
            record["layer"] = layer
        trace_rows.extend(output["trace"])
        key = f"{result['experiment']}_h{result['hop']}_{result['analysis']}"
        scaler_payload[f"{key}_mean"] = output["scaler_mean"]
        scaler_payload[f"{key}_scale"] = output["scaler_scale"]
        model_payload[f"{key}_coef"] = output["coef"]
        model_payload[f"{key}_intercept"] = output["intercept"]
        model_payload[f"{key}_classes"] = output["classes"]
        prediction = output["prediction"].copy()
        prediction["layer"] = layer
        if result["analysis"] == "normal":
            normal_results.append(result)
            if result["experiment"] == "primary":
                primary_predictions.append(prediction)
        else:
            random_results.append(result)
    pd.DataFrame(normal_results).to_csv(CHECKPOINTS / f"layer_{layer:02d}_results.csv", index=False)
    pd.DataFrame(random_results).to_csv(CHECKPOINTS / f"layer_{layer:02d}_random.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(CHECKPOINTS / f"layer_{layer:02d}_c_trace.csv", index=False)
    pd.concat(primary_predictions, ignore_index=True).to_csv(
        CHECKPOINTS / f"layer_{layer:02d}_primary_predictions.csv", index=False
    )
    np.savez(CHECKPOINTS / f"layer_{layer:02d}_scalers.npz", **scaler_payload)
    np.savez_compressed(CHECKPOINTS / f"layer_{layer:02d}_models.npz", **model_payload)
    write_json(CHECKPOINTS / f"layer_{layer:02d}_complete.json", {
        "status": "PASS", "layer": layer, "tasks": len(tasks),
        "runtime_seconds": time.perf_counter() - start,
        "normal_results": len(normal_results), "random_results": len(random_results),
    })
    print(json.dumps({
        "layer": layer, "status": "PASS", "tasks": len(tasks),
        "runtime_seconds": time.perf_counter() - start,
    }), flush=True)


def consolidate(union: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames, trace_frames, random_frames, prediction_frames = [], [], [], []
    for layer in range(N_LAYERS):
        if not layer_checkpoint_complete(layer):
            raise RuntimeError(f"Missing complete checkpoint layer {layer}")
        result_frames.append(pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_results.csv"))
        trace_frames.append(pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_c_trace.csv"))
        random_path = CHECKPOINTS / f"layer_{layer:02d}_random.csv"
        if random_path.stat().st_size > 1:
            try:
                random_frames.append(pd.read_csv(random_path))
            except pd.errors.EmptyDataError:
                pass
        prediction_frames.append(pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_primary_predictions.csv"))
    results = pd.concat(result_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    for experiment, spec in EXPERIMENTS.items():
        part = results[results["experiment"].eq(experiment)].sort_values(["layer", "hop"])
        expected = N_LAYERS * len(spec["hops"])
        if len(part) != expected:
            raise RuntimeError(f"Incomplete {experiment}: {len(part)} != {expected}")
        part.to_csv(OUT / spec["output"], index=False)
    random_df = pd.concat(random_frames, ignore_index=True).sort_values(["layer", "hop"])
    random_df.to_csv(OUT / "random_label_control.csv", index=False)
    traces.sort_values(["experiment", "analysis", "layer", "hop", "C"]).to_csv(
        OUT / "c_selection_trace.csv", index=False
    )
    predictions.sort_values(["layer", "hop", "row_id"]).to_csv(OUT / "primary_test_predictions.csv", index=False)

    # Collect per-layer scaler/model artifacts into explicit final directories.
    SCALERS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for layer in range(N_LAYERS):
        (SCALERS / f"layer_{layer:02d}.npz").write_bytes(
            (CHECKPOINTS / f"layer_{layer:02d}_scalers.npz").read_bytes()
        )
        (MODELS / f"layer_{layer:02d}.npz").write_bytes(
            (CHECKPOINTS / f"layer_{layer:02d}_models.npz").read_bytes()
        )
    return results, predictions


def target_absent_eval(union: pd.DataFrame, predictions: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    target_ids = set(union.loc[union["target_absent_split"].eq("test"), "row_id"].astype(int))
    rows = []
    for layer in range(N_LAYERS):
        for hop in [2, 3, 4, 5]:
            pred = predictions[
                predictions["layer"].eq(layer) & predictions["hop"].eq(hop)
                & predictions["row_id"].isin(target_ids)
            ]
            if pred.empty:
                raise RuntimeError(f"Empty target-absent eval {layer=} {hop=}")
            score = metrics(pred["true_relation"].to_numpy(), pred["predicted_relation"].to_numpy())
            iid = primary[primary["layer"].eq(layer) & primary["hop"].eq(hop)].iloc[0]
            rows.append({
                "layer": layer, "hop": hop, "n_test": len(pred),
                "macro_f1": score["macro_f1"], "accuracy": score["accuracy"],
                "derived_all_macro_f1": float(iid.test_macro_f1),
                "delta_macro_f1_target_absent_minus_all": score["macro_f1"] - float(iid.test_macro_f1),
                "probe_source": "Primary trained probe; no target-absent retraining",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "target_absent_eval.csv", index=False)
    return frame


def baselines(union: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, spec in EXPERIMENTS.items():
        for hop in spec["hops"]:
            labels = union.loc[
                union[spec["column"]].eq("test") & union["computed_hop"].eq(hop),
                "canonical_relation",
            ].to_numpy(dtype=object)
            rows.append({"experiment": experiment, "hop": hop, **macro_accuracy_baseline(labels)})
    for hop in [2, 3, 4, 5]:
        labels = union.loc[
            union["target_absent_split"].eq("test") & union["computed_hop"].eq(hop),
            "canonical_relation",
        ].to_numpy(dtype=object)
        rows.append({"experiment": "target_absent", "hop": hop, **macro_accuracy_baseline(labels)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "baselines.csv", index=False)
    return frame


def bootstrap_peak(primary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for hop in HOPS:
        curve = primary[primary["hop"].eq(hop)]
        peak_value = curve["test_macro_f1"].max()
        peak_layer = int(curve.loc[curve["test_macro_f1"].eq(peak_value), "layer"].min())
        pred = predictions[(predictions["hop"].eq(hop)) & (predictions["layer"].eq(peak_layer))].reset_index(drop=True)
        groups = pred["story_group_id"].unique()
        group_to_rows = {group: np.flatnonzero(pred["story_group_id"].eq(group).to_numpy()) for group in groups}
        rng = np.random.default_rng(SEED + hop)
        values = []
        for _ in range(1000):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            indices = np.concatenate([group_to_rows[group] for group in sampled_groups])
            values.append(metrics(
                pred.iloc[indices]["true_relation"].to_numpy(),
                pred.iloc[indices]["predicted_relation"].to_numpy(),
            )["macro_f1"])
        records.append({
            "hop": hop, "peak_layer": peak_layer, "point_macro_f1": float(peak_value),
            "bootstrap_unit": "story_group_id", "bootstrap_resamples": 1000,
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "primary_peak_bootstrap.csv", index=False)
    return frame


def layer_summary(primary: pd.DataFrame, baseline_df: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        curve = primary[primary["hop"].eq(hop)].sort_values("layer").copy()
        baseline = float(baseline_df[(baseline_df["experiment"].eq("primary")) & (baseline_df["hop"].eq(hop))]["macro_f1"].iloc[0])
        peak = float(curve["test_macro_f1"].max())
        peak_layer = int(curve.loc[curve["test_macro_f1"].eq(peak), "layer"].min())
        denom = peak - baseline
        curve["normalized_gain"] = (curve["test_macro_f1"] - baseline) / denom if denom > 0 else np.nan
        l50_candidates = curve.loc[curve["normalized_gain"].ge(0.5), "layer"]
        l50 = None if l50_candidates.empty else int(l50_candidates.min())
        l90 = None
        gains = dict(zip(curve["layer"].astype(int), curve["normalized_gain"]))
        for layer in curve["layer"].astype(int):
            future = list(range(layer, min(31, layer + 2) + 1))
            if gains[layer] >= 0.9 and all(gains[item] >= 0.85 for item in future):
                l90 = layer
                break
        ci = bootstrap[bootstrap["hop"].eq(hop)].iloc[0]
        rows.append({
            "hop": hop, "relation_type": "Explicit" if hop == 1 else "Derived",
            "majority_macro_f1": baseline, "peak_layer": peak_layer, "peak_macro_f1": peak,
            "peak_ci95_lower": float(ci.ci95_lower), "peak_ci95_upper": float(ci.ci95_upper),
            "L50": l50, "L90": l90,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "layer_summary.csv", index=False)
    return frame


def style_axis(ax, title: str, subtitle: str):
    ax.set_title(title, loc="left", fontsize=12, color="#202124", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color="#5f6368", va="bottom")
    ax.grid(True, axis="y", color="#dadce0", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa0a6")
    ax.set_xlim(0, 31)
    ax.set_xticks([0, 7, 15, 23, 31])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Transformer layer (raw decoder block output)")
    ax.set_ylabel("Held-out Macro-F1")


PALETTE = {1: "#2F6BFF", 2: "#D97706", 3: "#6B7D2A", 4: "#C04A8B", 5: "#B88A00"}
STYLES = {1: "-", 2: "--", 3: "-.", 4: ":", 5: (0, (5, 2))}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}


def plot_hop_curves(frame: pd.DataFrame, path: Path, title: str, subtitle: str, hops: list[int]):
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=170)
    for hop in hops:
        curve = frame[frame["hop"].eq(hop)].sort_values("layer")
        ax.plot(curve["layer"], curve["test_macro_f1"], color=PALETTE[hop], linestyle=STYLES[hop],
                marker=MARKERS[hop], markevery=[0, 7, 15, 23, 31], markersize=4,
                linewidth=1.9, label=f"h={hop} ({'Explicit' if hop == 1 else 'Derived'})")
    style_axis(ax, title, subtitle)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(primary, balanced, target, composition, template):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_hop_curves(primary, FIGURES / "figure1_primary_macro_f1.png",
                    "Primary relation decodability by layer and hop",
                    "Held-out Primary test; five hop-specific multinomial linear probes", HOPS)
    plot_hop_curves(balanced, FIGURES / "figure2_balanced_macro_f1.png",
                    "Balanced-control relation decodability by layer and hop",
                    "Exactly 56 samples per relation×hop across train/validation/test", HOPS)
    plot_hop_curves(composition, FIGURES / "figure4_composition_heldout.png",
                    "Composition-held-out derived-relation decodability",
                    "Held-out unordered composition signatures; h=2–5", [2, 3, 4, 5])
    plot_hop_curves(template, FIGURES / "figure5_template_test_only.png",
                    "Template test-only relation decodability",
                    "Examples containing reserved Mid-level families unseen in probe train/validation", HOPS)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=170, sharex=True, sharey=True)
    for ax, hop in zip(axes.flat, [2, 3, 4, 5]):
        iid = primary[primary["hop"].eq(hop)].sort_values("layer")
        absent = target[target["hop"].eq(hop)].sort_values("layer")
        ax.plot(iid.layer, iid.test_macro_f1, color="#2F6BFF", linewidth=1.8, label="Derived-All")
        ax.plot(absent.layer, absent.macro_f1, color="#D97706", linestyle="--", linewidth=1.8, label="Target-Absent")
        ax.set_title(f"h={hop}", loc="left")
        ax.grid(True, axis="y", color="#dadce0", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 31); ax.set_xticks([0, 7, 15, 23, 31])
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Derived-All and Target-Absent Macro-F1", x=0.08, ha="left", fontsize=13)
    fig.supxlabel("Transformer layer (raw decoder block output)")
    fig.supylabel("Held-out Macro-F1")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_target_absent.png", bbox_inches="tight")
    plt.close(fig)

    heat = primary.pivot(index="hop", columns="layer", values="test_macro_f1").reindex(index=HOPS, columns=range(N_LAYERS))
    fig, ax = plt.subplots(figsize=(11.2, 3.8), dpi=170)
    image = ax.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Primary Macro-F1 across layer and graph distance", loc="left", pad=14)
    ax.set_xlabel("Transformer layer (raw decoder block output)"); ax.set_ylabel("Hop")
    ax.set_xticks([0, 7, 15, 23, 31]); ax.set_yticks(range(5), labels=HOPS)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02); colorbar.set_label("Macro-F1")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6_primary_layer_hop_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    write_json(OUT / "chart_map.json", {
        "delivery_surface": "static PNG figures embedded/referenced by the technical Markdown report",
        "palette_policy": "relaxed multi-category up to five restrained roots; line style and marker duplicate color encoding",
        "charts": [
            {"file": "figure1_primary_macro_f1.png", "family": "Trend", "variant": "five-series line", "claim": "Primary layer profiles by hop"},
            {"file": "figure2_balanced_macro_f1.png", "family": "Trend", "variant": "five-series line", "claim": "sample/class-balanced layer profiles"},
            {"file": "figure3_target_absent.png", "family": "Uncertainty & Benchmark", "variant": "faceted two-series line", "claim": "Derived-All versus Target-Absent"},
            {"file": "figure4_composition_heldout.png", "family": "Trend", "variant": "four-series line", "claim": "composition generalization layer profiles"},
            {"file": "figure5_template_test_only.png", "family": "Trend", "variant": "five-series line", "claim": "reserved-template-family examples"},
            {"file": "figure6_primary_layer_hop_heatmap.png", "family": "Matrix & Cohort", "variant": "heatmap", "claim": "joint layer×hop pattern"},
        ],
    })


def main() -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    integrity = json.loads((OUT / "activation_integrity.json").read_text(encoding="utf-8"))
    if integrity["status"] != "PASS":
        raise RuntimeError("Activation integrity Gate is not PASS")
    union = pd.read_csv(OUT / "union_manifest.csv")
    union["computed_hop"] = union["computed_hop"].astype(int)
    row_ids = np.load(ACT_DIR / "row_ids.npy")
    if not np.array_equal(row_ids, union["row_id"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Activation row order mismatch before probing")
    for layer in range(N_LAYERS):
        run_layer(layer, union)
    results, predictions = consolidate(union)
    primary = pd.read_csv(OUT / "primary_probe.csv")
    balanced = pd.read_csv(OUT / "balanced_probe.csv")
    composition = pd.read_csv(OUT / "composition_heldout_probe.csv")
    template = pd.read_csv(OUT / "template_test_probe.csv")
    target = target_absent_eval(union, predictions, primary)
    baseline_df = baselines(union)
    bootstrap = bootstrap_peak(primary, predictions)
    summary = layer_summary(primary, baseline_df, bootstrap)
    make_figures(primary, balanced, target, composition, template)
    print(json.dumps({
        "status": "PASS", "primary_rows": len(primary), "balanced_rows": len(balanced),
        "composition_rows": len(composition), "template_rows": len(template),
        "target_absent_rows": len(target), "random_label_rows": len(pd.read_csv(OUT / "random_label_control.csv")),
        "layer_summary": summary.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
