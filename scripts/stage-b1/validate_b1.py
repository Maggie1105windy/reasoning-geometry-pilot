#!/usr/bin/env python3
"""Independent, fail-closed validation of every Stage B1 completion gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_b1 import ACT_DIR, C_GRID, HIDDEN_SIZE, N_LAYERS, OUT, RELATIONS, sha256, write_json


def main() -> None:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    split = json.loads((OUT / "split_checksum_verification.json").read_text())
    recomputed = []
    for item in split["files"]:
        actual = sha256(Path(item["path"]))
        recomputed.append({"path": item["path"], "expected": item["expected_sha256"], "actual": actual,
                           "match": actual == item["expected_sha256"]})
    checks["frozen_split_checksums_recomputed"] = all(x["match"] for x in recomputed)
    checks["frozen_snapshot_checks_pass"] = bool(
        split["all_formal_manifest_files_match"]
        and split["protected_b0_2_match_b0_2c_snapshot"]
        and split["broad_restricted_match_b0_2c_snapshot"]
    )
    details["split_files_recomputed"] = len(recomputed)

    token = json.loads((OUT / "tokenization_audit.json").read_text())
    checks["tokenization_readout_audit"] = token["status"] == "PASS" and all(token["checks"].values())
    checks["token_audit_sample_and_cell_coverage"] = (
        token["audit_sample_count"] >= 100 and token["coverage_cells"] == 40 and token["union_rows"] == 13758
    )

    model = json.loads((OUT / "model_audit.json").read_text())
    checks["model_is_llama31_base_only"] = (
        model["status"] == "PASS"
        and model["model"] == "Meta-Llama-3.1-8B Base"
        and model["model_path"] == "/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B"
        and model["layers"] == 32 and model["hidden_size"] == 4096
        and "tulu" not in json.dumps(model).lower()
    )
    sanity = json.loads((OUT / "activation_sanity.json").read_text())
    checks["raw_decoder_layer_activation_definition"] = sanity["status"] == "PASS" and all(sanity["checks"].values())
    integrity = json.loads((OUT / "activation_integrity.json").read_text())
    checks["recorded_activation_integrity"] = integrity["status"] == "PASS" and all(integrity["checks"].values())

    union = pd.read_csv(OUT / "union_manifest.csv")
    row_ids = np.load(ACT_DIR / "row_ids.npy")
    checks["row_alignment_recomputed"] = np.array_equal(row_ids, union["row_id"].to_numpy(np.int64))
    checks["row_ids_unique"] = len(row_ids) == len(np.unique(row_ids)) == 13758
    layer_scan = []
    for layer in range(N_LAYERS):
        path = ACT_DIR / f"layer_{layer:02d}.npy"
        if not path.is_file():
            layer_scan.append({"layer": layer, "present": False})
            continue
        matrix = np.load(path, mmap_mode="r")
        finite = bool(np.isfinite(matrix).all())
        layer_scan.append({"layer": layer, "present": True, "shape": list(matrix.shape),
                           "dtype": str(matrix.dtype), "finite": finite})
    checks["all_32_activation_files"] = len(layer_scan) == 32 and all(x.get("present") for x in layer_scan)
    checks["activation_shapes_dtypes_finite_recomputed"] = all(
        x.get("shape") == [13758, HIDDEN_SIZE] and x.get("dtype") == "float16" and x.get("finite")
        for x in layer_scan
    )
    details["activation_scan"] = layer_scan

    files = {
        "primary": ("primary_probe.csv", 160, set(range(1, 6))),
        "balanced": ("balanced_probe.csv", 160, set(range(1, 6))),
        "composition": ("composition_heldout_probe.csv", 128, set(range(2, 6))),
        "template": ("template_test_probe.csv", 160, set(range(1, 6))),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (filename, expected_rows, hops) in files.items():
        frame = pd.read_csv(OUT / filename)
        frames[name] = frame
        combos = set(zip(frame["layer"].astype(int), frame["hop"].astype(int)))
        expected = {(layer, hop) for layer in range(32) for hop in hops}
        checks[f"{name}_complete"] = len(frame) == expected_rows and combos == expected
        checks[f"{name}_metrics_valid"] = bool(
            np.isfinite(frame[["validation_macro_f1", "test_macro_f1", "test_accuracy"]]).all().all()
            and frame[["validation_macro_f1", "test_macro_f1", "test_accuracy"]].ge(0).all().all()
            and frame[["validation_macro_f1", "test_macro_f1", "test_accuracy"]].le(1).all().all()
        )
        checks[f"{name}_protocol_columns"] = bool(
            frame["scaler_fit_split"].eq("train_only").all()
            and frame["C_selection_split"].eq("validation_only").all()
            and (~frame["test_used_for_selection"].astype(bool)).all()
            and frame["selected_C"].isin(C_GRID).all()
        )

    target = pd.read_csv(OUT / "target_absent_eval.csv")
    checks["target_absent_complete"] = (
        len(target) == 128
        and set(zip(target["layer"].astype(int), target["hop"].astype(int)))
        == {(layer, hop) for layer in range(32) for hop in range(2, 6)}
        and target["probe_source"].eq("Primary trained probe; no target-absent retraining").all()
    )

    random = pd.read_csv(OUT / "random_label_control.csv")
    expected_random = {(layer, hop) for layer in [0, 7, 15, 23, 31] for hop in [1, 3, 5]}
    checks["random_label_complete"] = (
        len(random) == 15
        and set(zip(random["layer"].astype(int), random["hop"].astype(int))) == expected_random
        and (~random["test_used_for_selection"].astype(bool)).all()
    )
    matched = random.merge(frames["primary"], on=["layer", "hop"], suffixes=("_random", "_normal"))
    random_mean_acc = float(random["test_accuracy"].mean())
    random_mean_f1 = float(random["test_macro_f1"].mean())
    normal_mean_f1 = float(matched["test_macro_f1_normal"].mean())
    drop = float((matched["test_macro_f1_normal"] - matched["test_macro_f1_random"]).mean())
    checks["random_label_clearly_degraded"] = abs(random_mean_acc - 0.125) <= 0.02 and drop >= 0.04
    details["random_label"] = {"mean_accuracy": random_mean_acc, "mean_macro_f1": random_mean_f1,
                               "matched_normal_mean_macro_f1": normal_mean_f1, "mean_macro_f1_drop": drop}

    trace = pd.read_csv(OUT / "c_selection_trace.csv")
    trace_keys = ["experiment", "analysis", "layer", "hop"]
    checks["C_trace_has_nine_candidates"] = bool(trace.groupby(trace_keys).size().eq(9).all())
    checks["test_never_used_for_C_selection"] = not bool(trace["test_accessed_for_selection"].astype(bool).any())
    selected_all = pd.concat([*frames.values(), random], ignore_index=True)
    selection_ok = True
    for key, group in trace.groupby(trace_keys):
        max_f1 = group["validation_macro_f1"].max()
        expected_c = float(group.loc[group["validation_macro_f1"].eq(max_f1), "C"].min())
        exp, analysis, layer, hop = key
        found = selected_all[
            selected_all["experiment"].eq(exp) & selected_all["analysis"].eq(analysis)
            & selected_all["layer"].eq(layer) & selected_all["hop"].eq(hop)
        ]
        if len(found) != 1 or not np.isclose(float(found.iloc[0]["selected_C"]), expected_c):
            selection_ok = False
            break
    checks["validation_selection_and_smaller_C_tie_rule"] = selection_ok

    sensitivity = pd.read_csv(OUT / "convergence_refit_sensitivity.csv")
    normal_sens = sensitivity[sensitivity["analysis"].eq("normal")]
    random_sens = sensitivity[sensitivity["analysis"].eq("random_train_labels")]
    checks["normal_selected_warning_refits_converged"] = (
        len(normal_sens) == 41 and not bool(normal_sens["refit_convergence_warning"].astype(bool).any())
        and normal_sens["C_reselected"].eq(False).all()
        and normal_sens["test_used_for_selection"].eq(False).all()
    )
    details["convergence_sensitivity"] = {
        "normal_refits": len(normal_sens),
        "normal_remaining_warnings": int(normal_sens["refit_convergence_warning"].sum()),
        "normal_max_abs_test_macro_f1_delta": float(normal_sens["delta_test_macro_f1"].abs().max()),
        "normal_mean_abs_test_macro_f1_delta": float(normal_sens["delta_test_macro_f1"].abs().mean()),
        "random_refits": len(random_sens),
        "random_remaining_warnings": int(random_sens["refit_convergence_warning"].sum()),
        "random_max_abs_test_macro_f1_delta": float(random_sens["delta_test_macro_f1"].abs().max()),
    }

    baselines = pd.read_csv(OUT / "baselines.csv")
    checks["majority_baselines_complete"] = len(baselines) == 23 and np.isfinite(baselines[["accuracy", "macro_f1"]]).all().all()
    bootstrap = pd.read_csv(OUT / "primary_peak_bootstrap.csv")
    checks["story_group_bootstrap_complete"] = (
        len(bootstrap) == 5 and bootstrap["bootstrap_unit"].eq("story_group_id").all()
        and bootstrap["bootstrap_resamples"].eq(1000).all()
    )
    summary = pd.read_csv(OUT / "layer_summary.csv")
    checks["layer_dynamics_summary_complete"] = len(summary) == 5 and set(summary["hop"]) == set(range(1, 6))
    predictions = pd.read_csv(OUT / "primary_test_predictions.csv")
    expected_predictions = 32 * int(frames["primary"].groupby("hop")["n_test"].first().sum())
    checks["primary_predictions_complete"] = (
        len(predictions) == expected_predictions
        and set(predictions["true_relation"]).issubset(set(RELATIONS))
        and set(predictions["predicted_relation"]).issubset(set(RELATIONS))
    )

    checks["all_scaler_checkpoints"] = len(list((OUT / "scalers").glob("layer_*.npz"))) == 32
    checks["all_probe_model_checkpoints"] = len(list((OUT / "probe_models").glob("layer_*.npz"))) == 32
    figure_paths = sorted((OUT / "figures").glob("figure*.png"))
    checks["all_six_figures_present"] = len(figure_paths) == 6 and all(x.stat().st_size > 10000 for x in figure_paths)
    checks["no_post_trained_checkpoint_used"] = all(
        "tulu" not in path.name.lower() for path in OUT.rglob("*")
    ) and "tulu" not in json.dumps(model).lower()

    # Normalize numpy boolean scalars before the JSON audit artifact is written.
    checks = {name: bool(passed) for name, passed in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "stage": "B1",
        "validator": "independent_fail_closed_post_run_audit",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failed_checks": failures,
        "details": details,
    }
    write_json(OUT / "validation_pre_report.json", payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": failures,
                      "random_label": details["random_label"],
                      "convergence": details["convergence_sensitivity"]}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
