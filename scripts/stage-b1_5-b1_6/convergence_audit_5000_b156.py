#!/usr/bin/env python3
"""Escalated fixed-C convergence check for tasks still warning at 2000 iterations."""

from __future__ import annotations

import json

import pandas as pd
from joblib import Parallel, delayed

import convergence_audit_b156 as base
from common_b156 import CROSS, write_json


def _fit_at_5000(layer: int, analysis: str, selected_c: float):
    # Loky workers import modules in fresh processes, so the override must be
    # applied inside each worker rather than only in the parent process.
    base.MAX_ITER_SENSITIVITY = 5000
    return base._fit_one(layer, analysis, selected_c)


def main() -> None:
    previous = pd.read_csv(CROSS / "convergence_sensitivity.csv")
    remaining = previous[previous["convergence_warning_after_refit"].astype(bool)].drop_duplicates(["analysis", "layer"])
    tasks = [(int(row.layer), str(row.analysis), float(row.selected_C_frozen)) for row in remaining.itertuples()]
    nested = Parallel(n_jobs=6, verbose=10)(delayed(_fit_at_5000)(*task) for task in tasks)
    refit = pd.DataFrame([row for rows in nested for row in rows])
    original_rows = previous[["analysis", "layer", "target_hop", "test_macro_f1_refit", "test_accuracy_refit"]].rename(columns={
        "test_macro_f1_refit": "test_macro_f1_at_2000", "test_accuracy_refit": "test_accuracy_at_2000",
    })
    merged = refit.merge(original_rows, on=["analysis", "layer", "target_hop"], validate="one_to_one")
    merged["delta_test_macro_f1_vs_2000"] = merged["test_macro_f1_refit"] - merged["test_macro_f1_at_2000"]
    merged["delta_test_accuracy_vs_2000"] = merged["test_accuracy_refit"] - merged["test_accuracy_at_2000"]
    out_csv = CROSS / "convergence_sensitivity_5000.csv"
    merged.sort_values(["analysis", "layer", "target_hop"]).to_csv(out_csv, index=False)
    summary = {
        "status": "PASS" if not len(merged) or not merged["convergence_warning_after_refit"].any() else "REVIEW",
        "selection_rule_unchanged": True,
        "selected_C_frozen": True,
        "test_used_for_C_selection": False,
        "max_iter": 5000,
        "tasks_escalated_from_2000": len(tasks),
        "remaining_warning_tasks": int(merged.drop_duplicates(["analysis", "layer"])["convergence_warning_after_refit"].sum()) if len(merged) else 0,
        "max_abs_test_macro_f1_delta_vs_2000": float(merged["delta_test_macro_f1_vs_2000"].abs().max()) if len(merged) else 0.0,
        "mean_abs_test_macro_f1_delta_vs_2000": float(merged["delta_test_macro_f1_vs_2000"].abs().mean()) if len(merged) else 0.0,
        "output_csv": str(out_csv),
    }
    write_json(CROSS / "convergence_sensitivity_5000.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
