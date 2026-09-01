#!/usr/bin/env python3
"""Aggregate Stage B1.5 behavioral metrics, controls, and bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common_b156 import (
    BEHAVIOR, B1, HOPS, RELATIONS, SEED, ensure_dirs, load_union,
    metric_dict, read_json, write_json,
)


def bootstrap_story_groups(frame: pd.DataFrame, pred_col: str, seed: int, n_resamples: int = 1000) -> dict:
    groups = frame["story_group_id"].astype(str).unique()
    group_to_idx = {
        group: np.flatnonzero(frame["story_group_id"].astype(str).eq(group).to_numpy())
        for group in groups
    }
    rng = np.random.default_rng(seed)
    accuracy, macro_f1 = [], []
    truth = frame["canonical_relation"].to_numpy(dtype=object)
    pred = frame[pred_col].to_numpy(dtype=object)
    for _ in range(n_resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_to_idx[group] for group in sampled])
        score = metric_dict(truth[indices], pred[indices])
        accuracy.append(score["accuracy"])
        macro_f1.append(score["macro_f1"])
    return {
        "bootstrap_unit": "story_group_id", "bootstrap_resamples": n_resamples,
        "accuracy_ci95_low": float(np.quantile(accuracy, 0.025)),
        "accuracy_ci95_high": float(np.quantile(accuracy, 0.975)),
        "macro_f1_ci95_low": float(np.quantile(macro_f1, 0.025)),
        "macro_f1_ci95_high": float(np.quantile(macro_f1, 0.975)),
    }


def majority_baseline(labels: np.ndarray) -> dict:
    values, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    majority = sorted(values[counts == max_count].tolist())[0]
    pred = np.full(len(labels), majority, dtype=object)
    return {"majority_relation": majority, **metric_dict(labels, pred)}


def aggregate_dataset(union: pd.DataFrame, predictions: pd.DataFrame, dataset: str, split_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = set(union.loc[union[split_col].eq("test"), "row_id"].astype(int))
    frame = predictions[predictions["row_id"].isin(ids)].copy()
    expected = union[union[split_col].eq("test")][
        ["row_id", "story_group_id", "computed_hop", "canonical_relation"]
    ].copy()
    if set(frame["row_id"].astype(int)) != ids or len(frame) != len(ids):
        raise RuntimeError(f"Behavior row mismatch for {dataset}")
    frame = frame.drop(columns=["story_group_id", "computed_hop", "canonical_relation"]).merge(
        expected, on="row_id", validate="one_to_one",
    )
    rows, relation_rows = [], []
    confusion_dir = BEHAVIOR / "behavior_confusions"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for hop in HOPS:
        part = frame[frame["computed_hop"].eq(hop)].reset_index(drop=True)
        labels = part["canonical_relation"].to_numpy(dtype=object)
        baseline = majority_baseline(labels)
        record = {
            "dataset": dataset, "hop": hop, "n_test": len(part),
            "n_story_groups": part["story_group_id"].nunique(),
            "chance_accuracy": 0.125,
            "majority_relation": baseline["majority_relation"],
            "majority_accuracy": baseline["accuracy"],
            "majority_macro_f1": baseline["macro_f1"],
        }
        for method, pred_col in [("sum", "predicted_relation_sum"), ("mean", "predicted_relation_mean")]:
            pred = part[pred_col].to_numpy(dtype=object)
            score = metric_dict(labels, pred)
            ci = bootstrap_story_groups(part, pred_col, SEED + hop * 100 + (0 if method == "sum" else 1) + (0 if dataset == "primary" else 10000))
            record.update({
                f"accuracy_{method}": score["accuracy"],
                f"macro_f1_{method}": score["macro_f1"],
                f"accuracy_{method}_ci95_low": ci["accuracy_ci95_low"],
                f"accuracy_{method}_ci95_high": ci["accuracy_ci95_high"],
                f"macro_f1_{method}_ci95_low": ci["macro_f1_ci95_low"],
                f"macro_f1_{method}_ci95_high": ci["macro_f1_ci95_high"],
            })
            cm = confusion_matrix(labels, pred, labels=RELATIONS)
            pd.DataFrame(cm, index=RELATIONS, columns=RELATIONS).rename_axis("true_relation").to_csv(
                confusion_dir / f"{dataset}_h{hop}_{method}.csv"
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                labels, pred, labels=RELATIONS, zero_division=0,
            )
            for idx, relation in enumerate(RELATIONS):
                relation_rows.append({
                    "dataset": dataset, "scoring": method, "hop": hop,
                    "relation": relation, "support": int(support[idx]),
                    "precision": float(precision[idx]), "recall": float(recall[idx]),
                    "f1": float(f1[idx]), "accuracy_one_vs_rest": float(np.mean((labels == relation) == (pred == relation))),
                })
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(relation_rows)


def main() -> None:
    ensure_dirs()
    forward = read_json(BEHAVIOR / "behavior_forward_metadata.json")
    if forward["status"] != "PASS":
        raise RuntimeError("Behavior forward metadata is not PASS")
    union = load_union()
    predictions = pd.read_csv(BEHAVIOR / "all_behavior_predictions.csv")
    primary, primary_relation = aggregate_dataset(union, predictions, "primary", "primary_split")
    balanced, balanced_relation = aggregate_dataset(union, predictions, "balanced", "balanced_split")
    primary.to_csv(BEHAVIOR / "primary_behavior.csv", index=False)
    balanced.to_csv(BEHAVIOR / "balanced_behavior.csv", index=False)
    pd.concat([primary_relation, balanced_relation], ignore_index=True).to_csv(
        BEHAVIOR / "behavior_relationwise.csv", index=False
    )

    layer_summary = pd.read_csv(B1 / "layer_summary.csv")
    behavior_probe = primary.merge(
        layer_summary[["hop", "peak_layer", "peak_macro_f1"]], on="hop", validate="one_to_one",
    ).rename(columns={
        "accuracy_sum": "behavior_accuracy", "macro_f1_sum": "behavior_macro_f1",
        "accuracy_sum_ci95_low": "behavior_ci_low", "accuracy_sum_ci95_high": "behavior_ci_high",
        "peak_macro_f1": "probe_peak_macro_f1", "peak_layer": "probe_peak_layer",
    })
    keep = [
        "hop", "n_test", "behavior_accuracy", "behavior_macro_f1",
        "behavior_ci_low", "behavior_ci_high", "probe_peak_macro_f1", "probe_peak_layer",
        "accuracy_mean", "macro_f1_mean", "chance_accuracy", "majority_accuracy", "majority_macro_f1",
    ]
    behavior_probe[keep].to_csv(BEHAVIOR / "behavior_vs_probe.csv", index=False)
    rho_acc, p_acc = spearmanr(behavior_probe["behavior_accuracy"], behavior_probe["probe_peak_macro_f1"])
    rho_f1, p_f1 = spearmanr(behavior_probe["behavior_macro_f1"], behavior_probe["probe_peak_macro_f1"])
    primary_monotonic = bool(np.all(np.diff(primary.sort_values("hop")["accuracy_sum"].to_numpy()) <= 0))
    balanced_monotonic = bool(np.all(np.diff(balanced.sort_values("hop")["accuracy_sum"].to_numpy()) <= 0))
    summary = {
        "status": "PASS",
        "primary_sum_accuracy": dict(zip(primary.hop.astype(str), primary.accuracy_sum.astype(float))),
        "primary_sum_macro_f1": dict(zip(primary.hop.astype(str), primary.macro_f1_sum.astype(float))),
        "primary_mean_accuracy": dict(zip(primary.hop.astype(str), primary.accuracy_mean.astype(float))),
        "balanced_sum_accuracy": dict(zip(balanced.hop.astype(str), balanced.accuracy_sum.astype(float))),
        "balanced_mean_accuracy": dict(zip(balanced.hop.astype(str), balanced.accuracy_mean.astype(float))),
        "primary_accuracy_monotonic_nonincreasing": primary_monotonic,
        "balanced_accuracy_monotonic_nonincreasing": balanced_monotonic,
        "behavior_probe_spearman_n5": {
            "behavior_accuracy_vs_probe_peak_f1": {"rho": float(rho_acc), "p_value": float(p_acc)},
            "behavior_macro_f1_vs_probe_peak_f1": {"rho": float(rho_f1), "p_value": float(p_f1)},
            "interpretation": "descriptive only; n=5 hops",
        },
        "bootstrap": {"unit": "story_group_id", "resamples": 1000},
    }
    write_json(BEHAVIOR / "behavior_summary.json", summary)
    print(json.dumps({
        "status": "PASS",
        "primary": primary[["hop", "n_test", "accuracy_sum", "macro_f1_sum", "accuracy_mean", "macro_f1_mean", "majority_accuracy"]].to_dict("records"),
        "balanced": balanced[["hop", "n_test", "accuracy_sum", "macro_f1_sum", "accuracy_mean", "macro_f1_mean", "majority_accuracy"]].to_dict("records"),
        "primary_accuracy_monotonic": primary_monotonic,
        "behavior_probe_spearman_rho_accuracy": float(rho_acc),
    }, indent=2))


if __name__ == "__main__":
    main()
