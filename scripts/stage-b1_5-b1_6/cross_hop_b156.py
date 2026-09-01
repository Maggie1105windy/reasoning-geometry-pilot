#!/usr/bin/env python3
"""Stage B1.6 cross-hop transfer probes using frozen Stage B1 activations."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b156 import (
    ACT_DIR, B1, CHECKPOINTS, CROSS, C_GRID, HOPS, LAYERS, OUT, RELATIONS, SEED,
    ensure_dirs, load_union, metric_dict, read_json, write_json,
)


RANDOM_LAYERS = [0, 7, 15, 17, 23, 31]


def predict_params(x: np.ndarray, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray) -> np.ndarray:
    logits = x @ coef.T + intercept
    return classes[np.argmax(logits, axis=1)]


def fit_transfer(
    matrix: np.ndarray,
    union: pd.DataFrame,
    layer: int,
    analysis: str,
    split_col: str,
    source_hops: list[int],
    class_weight,
    random_labels: bool = False,
) -> dict:
    source_train_mask = union[split_col].eq("train") & union["computed_hop"].isin(source_hops)
    source_val_mask = union[split_col].eq("val") & union["computed_hop"].isin(source_hops)
    train_pos = np.flatnonzero(source_train_mask.to_numpy())
    val_pos = np.flatnonzero(source_val_mask.to_numpy())
    if not len(train_pos) or not len(val_pos):
        raise RuntimeError(f"Empty source split: {analysis}")
    x_train = np.asarray(matrix[train_pos], dtype=np.float32)
    x_val = np.asarray(matrix[val_pos], dtype=np.float32)
    y_train = union.iloc[train_pos]["canonical_relation"].to_numpy(dtype=object)
    y_val = union.iloc[val_pos]["canonical_relation"].to_numpy(dtype=object)
    if set(y_train) != set(RELATIONS) or set(y_val) != set(RELATIONS):
        raise RuntimeError(f"Class coverage failure: {analysis} layer={layer}")
    scaler = StandardScaler(copy=True).fit(x_train)
    train_scaled = scaler.transform(x_train).astype(np.float32, copy=False)
    val_scaled = scaler.transform(x_val).astype(np.float32, copy=False)
    del x_train, x_val

    fit_labels = y_train.copy()
    if random_labels:
        rng = np.random.default_rng(SEED + layer * 9176 + 601)
        fit_labels = fit_labels[rng.permutation(len(fit_labels))]

    classifier = LogisticRegression(
        C=C_GRID[0], solver="lbfgs", multi_class="multinomial", max_iter=400,
        tol=1e-4, class_weight=class_weight, warm_start=True, random_state=SEED,
    )
    best = None
    trace = []
    started = time.perf_counter()
    with threadpool_limits(limits=3):
        for c_value in C_GRID:
            classifier.C = c_value
            fit_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(train_scaled, fit_labels)
            pred = classifier.predict(val_scaled)
            score = metric_dict(y_val, pred)
            warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            trace.append({
                "analysis": analysis, "layer": layer, "source_hops": "+".join(map(str, source_hops)),
                "C": c_value, "validation_macro_f1": score["macro_f1"],
                "validation_accuracy": score["accuracy"], "n_iter_max": int(np.max(classifier.n_iter_)),
                "convergence_warning": warned, "fit_seconds": time.perf_counter() - fit_start,
                "test_accessed_for_selection": False,
            })
            candidate = (
                score["macro_f1"], -float(c_value), classifier.coef_.astype(np.float32, copy=True),
                classifier.intercept_.astype(np.float32, copy=True),
                classifier.classes_.astype(object, copy=True), score, warned,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    selected_f1, neg_c, coef, intercept, classes, val_score, selected_warning = best
    selected_c = -neg_c

    results, relationwise = [], []
    for target_hop in HOPS:
        test_mask = union[split_col].eq("test") & union["computed_hop"].eq(target_hop)
        test_pos = np.flatnonzero(test_mask.to_numpy())
        x_test = np.asarray(matrix[test_pos], dtype=np.float32)
        test_scaled = scaler.transform(x_test).astype(np.float32, copy=False)
        y_test = union.iloc[test_pos]["canonical_relation"].to_numpy(dtype=object)
        pred = predict_params(test_scaled, coef, intercept, classes)
        score = metric_dict(y_test, pred)
        results.append({
            "analysis": analysis, "layer": layer, "source_hops": "+".join(map(str, source_hops)),
            "target_hop": target_hop, "n_train": len(y_train), "n_val": len(y_val),
            "n_test": len(y_test), "hidden_size": matrix.shape[1], "selected_C": selected_c,
            "validation_macro_f1": val_score["macro_f1"], "validation_accuracy": val_score["accuracy"],
            "test_macro_f1": score["macro_f1"], "test_accuracy": score["accuracy"],
            "selected_fit_convergence_warning": bool(selected_warning),
            "scaler_fit_population": f"{split_col}:train:hops={source_hops}",
            "C_selection_population": f"{split_col}:val:hops={source_hops}",
            "target_test_used_for_fitting_or_selection": False,
            "class_weight": "balanced" if class_weight == "balanced" else "none",
            "random_train_labels": random_labels,
            "task_seconds": time.perf_counter() - started,
        })
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, pred, labels=RELATIONS, zero_division=0,
        )
        for idx, relation in enumerate(RELATIONS):
            relationwise.append({
                "analysis": analysis, "layer": layer, "source_hops": "+".join(map(str, source_hops)),
                "target_hop": target_hop, "relation": relation, "support": int(support[idx]),
                "precision": float(precision[idx]), "recall": float(recall[idx]), "f1": float(f1[idx]),
            })
        del x_test, test_scaled
    return {
        "results": results, "relationwise": relationwise, "trace": trace,
        "model": {
            "coef": coef, "intercept": intercept, "classes": np.asarray(classes, dtype=str),
            "scaler_mean": scaler.mean_.astype(np.float32), "scaler_scale": scaler.scale_.astype(np.float32),
            "selected_C": np.asarray([selected_c], dtype=np.float64),
        },
    }


def checkpoint_complete(layer: int) -> bool:
    path = CHECKPOINTS / f"layer_{layer:02d}_complete.json"
    return path.is_file() and read_json(path).get("status") == "PASS"


def run_layer(layer: int) -> None:
    ensure_dirs()
    if read_json(OUT / "input_checksum_verification.json")["status"] != "PASS":
        raise RuntimeError("Input checksum Gate is not PASS")
    if checkpoint_complete(layer):
        print(json.dumps({"layer": layer, "status": "SKIP_COMPLETE"}), flush=True)
        return
    union = load_union()
    row_ids = np.load(ACT_DIR / "row_ids.npy")
    if not np.array_equal(row_ids, union["row_id"].to_numpy(np.int64)):
        raise RuntimeError("Activation row alignment failed")
    matrix = np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")
    if tuple(matrix.shape) != (13758, 4096) or matrix.dtype != np.float16:
        raise RuntimeError(f"Unexpected activation matrix {layer}: {matrix.shape} {matrix.dtype}")
    tasks = [
        ("primary_h1", "primary_split", [1], "balanced", False),
        ("primary_derived_pooled", "primary_split", [2, 3, 4, 5], "balanced", False),
        ("balanced_h1", "balanced_split", [1], None, False),
    ]
    if layer in RANDOM_LAYERS:
        tasks.append(("primary_h1_random_labels", "primary_split", [1], "balanced", True))
    results, relationwise, traces, models = [], [], [], {}
    started = time.perf_counter()
    for analysis, split_col, source_hops, class_weight, random_labels in tasks:
        output = fit_transfer(
            matrix, union, layer, analysis, split_col, source_hops,
            class_weight, random_labels=random_labels,
        )
        results.extend(output["results"])
        relationwise.extend(output["relationwise"])
        traces.extend(output["trace"])
        for key, value in output["model"].items():
            models[f"{analysis}__{key}"] = value
    pd.DataFrame(results).to_csv(CHECKPOINTS / f"layer_{layer:02d}_results.csv", index=False)
    pd.DataFrame(relationwise).to_csv(CHECKPOINTS / f"layer_{layer:02d}_relationwise.csv", index=False)
    pd.DataFrame(traces).to_csv(CHECKPOINTS / f"layer_{layer:02d}_trace.csv", index=False)
    np.savez_compressed(CHECKPOINTS / f"layer_{layer:02d}_models.npz", **models)
    write_json(CHECKPOINTS / f"layer_{layer:02d}_complete.json", {
        "status": "PASS", "layer": layer, "tasks": [task[0] for task in tasks],
        "result_rows": len(results), "relationwise_rows": len(relationwise),
        "trace_rows": len(traces), "runtime_seconds": time.perf_counter() - started,
        "activation_reextracted": False,
    })
    print(json.dumps({"layer": layer, "status": "PASS", "tasks": len(tasks),
                      "runtime_seconds": time.perf_counter() - started}, indent=2), flush=True)


def consolidate() -> None:
    ensure_dirs()
    for layer in LAYERS:
        if not checkpoint_complete(layer):
            raise RuntimeError(f"Incomplete layer {layer}")
    results = pd.concat([
        pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_results.csv") for layer in LAYERS
    ], ignore_index=True)
    relationwise = pd.concat([
        pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_relationwise.csv") for layer in LAYERS
    ], ignore_index=True)
    trace = pd.concat([
        pd.read_csv(CHECKPOINTS / f"layer_{layer:02d}_trace.csv") for layer in LAYERS
    ], ignore_index=True)
    outputs = {
        "primary_h1": "h1_to_all_layers.csv",
        "primary_derived_pooled": "derived_pooled_to_all_layers.csv",
        "balanced_h1": "balanced_h1_to_all_layers.csv",
        "primary_h1_random_labels": "random_label_transfer.csv",
    }
    for analysis, filename in outputs.items():
        part = results[results["analysis"].eq(analysis)].sort_values(["layer", "target_hop"])
        expected = (len(RANDOM_LAYERS) if "random" in analysis else 32) * 5
        if len(part) != expected:
            raise RuntimeError(f"Unexpected {analysis} rows: {len(part)} != {expected}")
        part.to_csv(CROSS / filename, index=False)
    relationwise.sort_values(["analysis", "layer", "target_hop", "relation"]).to_csv(
        CROSS / "relationwise_transfer.csv", index=False
    )
    trace.sort_values(["analysis", "layer", "C"]).to_csv(CROSS / "c_selection_trace.csv", index=False)

    within = pd.read_csv(B1 / "primary_probe.csv").rename(columns={"hop": "target_hop"})
    transfer = results[results["analysis"].eq("primary_h1")].copy()
    baseline = pd.read_csv(B1 / "baselines.csv")
    base = baseline[baseline["experiment"].eq("primary")][["hop", "macro_f1"]].rename(
        columns={"hop": "target_hop", "macro_f1": "majority_macro_f1"}
    )
    gap = transfer.merge(
        within[["layer", "target_hop", "test_macro_f1", "test_accuracy"]].rename(columns={
            "test_macro_f1": "within_hop_macro_f1", "test_accuracy": "within_hop_accuracy",
        }), on=["layer", "target_hop"], validate="one_to_one",
    ).merge(base, on="target_hop", validate="many_to_one")
    gap = gap.rename(columns={
        "test_macro_f1": "h1_transfer_macro_f1", "test_accuracy": "h1_transfer_accuracy",
    })
    gap["absolute_transfer_gap"] = gap["within_hop_macro_f1"] - gap["h1_transfer_macro_f1"]
    gap["retention_ratio"] = np.where(
        gap["within_hop_macro_f1"] > gap["majority_macro_f1"] + 0.02,
        gap["h1_transfer_macro_f1"] / gap["within_hop_macro_f1"], np.nan,
    )
    gap["retention_interpretable"] = gap["within_hop_macro_f1"] > gap["majority_macro_f1"] + 0.02
    gap.to_csv(CROSS / "transfer_gap.csv", index=False)

    summary_rows = []
    for hop in [2, 3, 4, 5]:
        part = gap[gap["target_hop"].eq(hop)].sort_values("layer")
        transfer_peak = part.loc[part["h1_transfer_macro_f1"].idxmax()]
        within_peak = part.loc[part["within_hop_macro_f1"].idxmax()]
        summary_rows.append({
            "target_hop": hop,
            "peak_transfer_layer": int(transfer_peak["layer"]),
            "peak_transfer_macro_f1": float(transfer_peak["h1_transfer_macro_f1"]),
            "transfer_gap_at_peak_transfer_layer": float(transfer_peak["absolute_transfer_gap"]),
            "within_hop_peak_layer": int(within_peak["layer"]),
            "within_hop_peak_macro_f1": float(within_peak["within_hop_macro_f1"]),
            "h1_transfer_at_within_hop_peak_layer": float(within_peak["h1_transfer_macro_f1"]),
            "gap_at_within_hop_peak_layer": float(within_peak["absolute_transfer_gap"]),
            "retention_at_within_hop_peak_layer": float(within_peak["retention_ratio"]),
            "mean_layerwise_gap": float(part["absolute_transfer_gap"].mean()),
            "late_layer_16_31_mean_gap": float(part[part["layer"].between(16, 31)]["absolute_transfer_gap"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CROSS / "transfer_summary.csv", index=False)

    random = results[results["analysis"].eq("primary_h1_random_labels")]
    matched_normal = results[results["analysis"].eq("primary_h1")].merge(
        random[["layer", "target_hop"]], on=["layer", "target_hop"], how="inner"
    )
    write_json(CROSS / "cross_hop_summary.json", {
        "status": "PASS", "late_layer_definition": "layers 16-31; frozen before results",
        "primary_rows": int(len(transfer)), "derived_pooled_rows": int((results["analysis"] == "primary_derived_pooled").sum()),
        "balanced_rows": int((results["analysis"] == "balanced_h1").sum()),
        "random_rows": int(len(random)),
        "random_mean_accuracy": float(random["test_accuracy"].mean()),
        "random_mean_macro_f1": float(random["test_macro_f1"].mean()),
        "matched_normal_mean_accuracy": float(matched_normal["test_accuracy"].mean()),
        "matched_normal_mean_macro_f1": float(matched_normal["test_macro_f1"].mean()),
        "test_accessed_for_selection_any": bool(trace["test_accessed_for_selection"].astype(bool).any()),
        "selected_fit_convergence_warnings": int(results.drop_duplicates(["analysis", "layer"])["selected_fit_convergence_warning"].sum()),
        "activation_reextracted": False,
    })
    print(json.dumps({
        "status": "PASS", "result_rows": len(results), "trace_rows": len(trace),
        "summary": summary.to_dict("records"),
        "random_mean_accuracy": float(random["test_accuracy"].mean()),
        "random_mean_macro_f1": float(random["test_macro_f1"].mean()),
        "selected_warnings": int(results.drop_duplicates(["analysis", "layer"])["selected_fit_convergence_warning"].sum()),
    }, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["layer", "consolidate"])
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    if args.command == "layer":
        if args.layer is None:
            raise SystemExit("--layer is required")
        run_layer(args.layer)
    else:
        consolidate()
