#!/usr/bin/env python3
"""Fixed-C convergence sensitivity for selected Stage B1.6 probe fits.

Only tasks whose selected warm-start fit raised a convergence warning at
max_iter=400 are refit.  C remains frozen at the validation-selected value;
the test sets are evaluated only after fitting and never affect selection.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b156 import ACT_DIR, CROSS, HOPS, RELATIONS, SEED, ensure_dirs, load_union, metric_dict, write_json


MAX_ITER_SENSITIVITY = 2000


def _fit_one(layer: int, analysis: str, selected_c: float) -> list[dict]:
    union = load_union()
    matrix = np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")
    if analysis == "primary_h1":
        split_col, source_hops, class_weight, random_labels = "primary_split", [1], "balanced", False
    elif analysis == "primary_derived_pooled":
        split_col, source_hops, class_weight, random_labels = "primary_split", [2, 3, 4, 5], "balanced", False
    elif analysis == "balanced_h1":
        split_col, source_hops, class_weight, random_labels = "balanced_split", [1], None, False
    elif analysis == "primary_h1_random_labels":
        split_col, source_hops, class_weight, random_labels = "primary_split", [1], "balanced", True
    else:
        raise ValueError(analysis)

    train_mask = union[split_col].eq("train") & union["computed_hop"].isin(source_hops)
    val_mask = union[split_col].eq("val") & union["computed_hop"].isin(source_hops)
    train_pos = np.flatnonzero(train_mask.to_numpy())
    val_pos = np.flatnonzero(val_mask.to_numpy())
    x_train = np.asarray(matrix[train_pos], dtype=np.float32)
    x_val = np.asarray(matrix[val_pos], dtype=np.float32)
    y_train = union.iloc[train_pos]["canonical_relation"].to_numpy(dtype=object)
    y_val = union.iloc[val_pos]["canonical_relation"].to_numpy(dtype=object)
    scaler = StandardScaler(copy=True).fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32, copy=False)
    x_val = scaler.transform(x_val).astype(np.float32, copy=False)
    fit_labels = y_train.copy()
    if random_labels:
        rng = np.random.default_rng(SEED + layer * 9176 + 601)
        fit_labels = fit_labels[rng.permutation(len(fit_labels))]

    clf = LogisticRegression(
        C=float(selected_c), solver="lbfgs", multi_class="multinomial",
        max_iter=MAX_ITER_SENSITIVITY, tol=1e-4, class_weight=class_weight,
        warm_start=False, random_state=SEED,
    )
    with threadpool_limits(limits=3), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, fit_labels)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    val_score = metric_dict(y_val, clf.predict(x_val))

    rows = []
    for hop in HOPS:
        test_mask = union[split_col].eq("test") & union["computed_hop"].eq(hop)
        test_pos = np.flatnonzero(test_mask.to_numpy())
        x_test = np.asarray(matrix[test_pos], dtype=np.float32)
        x_test = scaler.transform(x_test).astype(np.float32, copy=False)
        y_test = union.iloc[test_pos]["canonical_relation"].to_numpy(dtype=object)
        score = metric_dict(y_test, clf.predict(x_test))
        rows.append({
            "analysis": analysis, "layer": layer, "target_hop": hop,
            "selected_C_frozen": float(selected_c),
            "max_iter_sensitivity": MAX_ITER_SENSITIVITY,
            "n_iter_max": int(np.max(clf.n_iter_)),
            "convergence_warning_after_refit": bool(warned),
            "validation_macro_f1_refit": val_score["macro_f1"],
            "validation_accuracy_refit": val_score["accuracy"],
            "test_macro_f1_refit": score["macro_f1"],
            "test_accuracy_refit": score["accuracy"],
            "test_used_for_C_selection": False,
            "scaler_fit_population": f"{split_col}:train:hops={source_hops}",
            "C_selection_population": f"{split_col}:val:hops={source_hops}",
        })
    return rows


def main() -> None:
    ensure_dirs()
    files = {
        "primary_h1": CROSS / "h1_to_all_layers.csv",
        "primary_derived_pooled": CROSS / "derived_pooled_to_all_layers.csv",
        "balanced_h1": CROSS / "balanced_h1_to_all_layers.csv",
        "primary_h1_random_labels": CROSS / "random_label_transfer.csv",
    }
    originals = pd.concat([pd.read_csv(path) for path in files.values()], ignore_index=True)
    selected = originals.drop_duplicates(["analysis", "layer"])
    selected = selected[selected["selected_fit_convergence_warning"].astype(bool)]
    tasks = [(int(row.layer), str(row.analysis), float(row.selected_C)) for row in selected.itertuples()]
    nested = Parallel(n_jobs=8, verbose=10)(delayed(_fit_one)(*task) for task in tasks)
    refit = pd.DataFrame([row for rows in nested for row in rows])
    if len(refit):
        merged = refit.merge(
            originals[["analysis", "layer", "target_hop", "validation_macro_f1",
                       "validation_accuracy", "test_macro_f1", "test_accuracy"]],
            on=["analysis", "layer", "target_hop"], validate="one_to_one",
        )
        for metric in ["validation_macro_f1", "validation_accuracy", "test_macro_f1", "test_accuracy"]:
            merged[f"delta_{metric}"] = merged[f"{metric}_refit"] - merged[metric]
    else:
        merged = refit
    out_csv = CROSS / "convergence_sensitivity.csv"
    merged.sort_values(["analysis", "layer", "target_hop"]).to_csv(out_csv, index=False)
    summary = {
        "status": "PASS" if not len(merged) or not merged["convergence_warning_after_refit"].any() else "REVIEW",
        "selection_rule_unchanged": True,
        "selected_C_frozen": True,
        "test_used_for_C_selection": False,
        "max_iter_original": 400,
        "max_iter_sensitivity": MAX_ITER_SENSITIVITY,
        "warning_tasks_refit": len(tasks),
        "remaining_warning_tasks": int(merged.drop_duplicates(["analysis", "layer"])["convergence_warning_after_refit"].sum()) if len(merged) else 0,
        "max_abs_test_macro_f1_delta": float(merged["delta_test_macro_f1"].abs().max()) if len(merged) else 0.0,
        "mean_abs_test_macro_f1_delta": float(merged["delta_test_macro_f1"].abs().mean()) if len(merged) else 0.0,
        "output_csv": str(out_csv),
    }
    write_json(CROSS / "convergence_sensitivity.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
