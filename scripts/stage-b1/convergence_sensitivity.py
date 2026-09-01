#!/usr/bin/env python3
"""Refit only selected Stage B1 models that warned at max_iter=400.

This is a numerical sensitivity audit.  It does not reselect C and therefore
does not access test data until after the validation-selected C is frozen.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b1 import ACT_DIR, OUT, RELATIONS, SEED
from probe_b1 import EXPERIMENTS


FILES = [
    "primary_probe.csv",
    "balanced_probe.csv",
    "composition_heldout_probe.csv",
    "template_test_probe.csv",
    "random_label_control.csv",
]


def score(y_true, y_pred):
    return (
        float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
        float(accuracy_score(y_true, y_pred)),
    )


def refit(row: dict, union: pd.DataFrame) -> dict:
    experiment = str(row["experiment"])
    analysis = str(row["analysis"])
    layer = int(row["layer"])
    hop = int(row["hop"])
    split_col = EXPERIMENTS[experiment]["column"]
    positions = {
        split: np.flatnonzero(
            (union[split_col].eq(split) & union["computed_hop"].eq(hop)).to_numpy()
        )
        for split in ["train", "val", "test"]
    }
    matrix = np.load(ACT_DIR / f"layer_{layer:02d}.npy", mmap_mode="r")
    xs = {
        split: np.asarray(matrix[pos], dtype=np.float32)
        for split, pos in positions.items()
    }
    ys = {
        split: union.iloc[pos]["canonical_relation"].to_numpy(dtype=object)
        for split, pos in positions.items()
    }
    scaler = StandardScaler(copy=True).fit(xs["train"])
    xs = {key: scaler.transform(value).astype(np.float32, copy=False) for key, value in xs.items()}
    train_labels = ys["train"].copy()
    if analysis == "random_train_labels":
        rng = np.random.default_rng(SEED + hop * 1009 + layer * 9176)
        train_labels = train_labels[rng.permutation(len(train_labels))]
    classifier = LogisticRegression(
        C=float(row["selected_C"]), solver="lbfgs", multi_class="multinomial",
        max_iter=2000, tol=1e-4,
        class_weight=("balanced" if row["class_weight"] == "balanced" else None),
        random_state=SEED,
    )
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(xs["train"], train_labels)
    warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    val_f1, val_acc = score(ys["val"], classifier.predict(xs["val"]))
    # Test is evaluated only after C has remained fixed from the original validation selection.
    test_f1, test_acc = score(ys["test"], classifier.predict(xs["test"]))
    return {
        "experiment": experiment,
        "analysis": analysis,
        "layer": layer,
        "hop": hop,
        "selected_C_frozen": float(row["selected_C"]),
        "original_validation_macro_f1": float(row["validation_macro_f1"]),
        "refit_validation_macro_f1": val_f1,
        "delta_validation_macro_f1": val_f1 - float(row["validation_macro_f1"]),
        "original_test_macro_f1": float(row["test_macro_f1"]),
        "refit_test_macro_f1": test_f1,
        "delta_test_macro_f1": test_f1 - float(row["test_macro_f1"]),
        "original_test_accuracy": float(row["test_accuracy"]),
        "refit_test_accuracy": test_acc,
        "delta_test_accuracy": test_acc - float(row["test_accuracy"]),
        "refit_n_iter_max": int(np.max(classifier.n_iter_)),
        "refit_convergence_warning": bool(warning),
        "max_iter_original": 400,
        "max_iter_refit": 2000,
        "C_reselected": False,
        "test_used_for_selection": False,
    }


def main() -> None:
    union = pd.read_csv(OUT / "union_manifest.csv")
    union["computed_hop"] = union["computed_hop"].astype(int)
    selected = pd.concat([pd.read_csv(OUT / name) for name in FILES], ignore_index=True)
    selected = selected[selected["selected_fit_convergence_warning"].astype(bool)].copy()
    rows = Parallel(n_jobs=8, backend="loky", verbose=10)(
        delayed(refit)(row, union) for row in selected.to_dict("records")
    )
    frame = pd.DataFrame(rows).sort_values(["experiment", "analysis", "layer", "hop"])
    frame.to_csv(OUT / "convergence_refit_sensitivity.csv", index=False)
    print(frame.groupby(["experiment", "analysis"]).agg(
        n=("hop", "size"), remaining_warnings=("refit_convergence_warning", "sum"),
        max_abs_delta_test_f1=("delta_test_macro_f1", lambda x: float(np.max(np.abs(x)))),
        mean_abs_delta_test_f1=("delta_test_macro_f1", lambda x: float(np.mean(np.abs(x)))),
    ).to_string())


if __name__ == "__main__":
    main()
