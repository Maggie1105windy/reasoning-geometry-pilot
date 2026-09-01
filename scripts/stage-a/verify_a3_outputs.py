#!/usr/bin/env python3
"""Independent post-run verification for bounded Stage A3 artifacts."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import torch


PROJECT = Path("/home/ai/projects/linear-probe-repro")
RESULTS = PROJECT / "results"
ACTIVATIONS = PROJECT / "activations/a3_migration/llama-3.1-8b/world/coords"
LAYERS = [0, 7, 15, 23, 31]


def main():
    tensors = torch.load(
        ACTIVATIONS / "activations.pt", map_location="cpu", weights_only=True
    )
    metadata = json.loads((ACTIVATIONS / "metadata.json").read_text())
    sample = pd.read_csv(RESULTS / "a3_sample_manifest.csv")
    a2 = pd.read_csv(RESULTS / "a2_3_sample_manifest.csv")
    indexing = pd.read_csv(RESULTS / "a3_hidden_state_indexing.csv")
    sanity = pd.read_csv(RESULTS / "a3_activation_sanity_stats.csv")
    fixed = pd.read_csv(RESULTS / "a3_llama31_mini_probe_fixed_alpha.csv")
    cv = pd.read_csv(RESULTS / "a3_llama31_mini_probe_ridgecv.csv")
    controls = json.loads((RESULTS / "a3_controls.json").read_text())
    required = [
        RESULTS / "a3_llama31_model_audit.md",
        RESULTS / "a3_activation_sanity_report.md",
        RESULTS / "a3_llama31_mini_probe_fixed_alpha.csv",
        RESULTS / "a3_llama31_mini_probe_ridgecv.csv",
        RESULTS / "a3_llama31_layer_r2.png",
        RESULTS / "A3_LLAMA31_MIGRATION_REPORT.md",
        ACTIVATIONS / "activations.pt",
        ACTIVATIONS / "metadata.json",
    ]
    checks = {
        "all_required_files_nonempty": all(p.is_file() and p.stat().st_size > 0 for p in required),
        "saved_layers_exact": sorted(tensors) == LAYERS,
        "saved_shapes_exact": all(tuple(tensors[k].shape) == (5000, 4096) for k in LAYERS),
        "saved_dtypes_float16": all(tensors[k].dtype == torch.float16 for k in LAYERS),
        "saved_values_finite": all(torch.isfinite(tensors[k]).all().item() for k in LAYERS),
        "metadata_pass": metadata["status"] == "PASS",
        "metadata_checks_pass": all(metadata["checks"].values()),
        "official_split_4000_1000": sample.is_test.value_counts().to_dict() == {False: 4000, True: 1000},
        "a2_row_ids_exact": sample.row_id.tolist() == a2.row_id.tolist(),
        "train_test_row_ids_disjoint": set(sample.loc[~sample.is_test, "row_id"]).isdisjoint(
            set(sample.loc[sample.is_test, "row_id"])
        ),
        "sanity_32_layers": sanity.layer.tolist() == list(range(32)),
        "sanity_shapes": sanity["shape"].eq("[16, 4096]").all(),
        "sanity_finite": sanity.nan_count.eq(0).all() and sanity.inf_count.eq(0).all(),
        "indexing_two_samples": indexing.sample_index.nunique() == 2,
        "fixed_five_layers": fixed.layer.tolist() == LAYERS,
        "cv_five_layers": cv.layer.tolist() == LAYERS,
        "baseline_near_zero": abs(controls["baseline"]["r2_overall"]) < 0.05,
        "random_label_degraded": controls["random_label"]["drop_vs_normal"] > 0.05,
        "upstream_clean": subprocess.check_output(
            ["git", "-C", str(PROJECT / "external/world-models"), "status", "--porcelain"],
            text=True,
        ).strip() == "",
    }
    print(json.dumps({
        "status": "PASS" if all(checks.values()) else "FAIL",
        "failed_checks": [key for key, value in checks.items() if not value],
        "checks": checks,
        "saved_shapes": {str(key): list(value.shape) for key, value in tensors.items()},
        "saved_dtypes": {str(key): str(value.dtype) for key, value in tensors.items()},
        "entity_alignment": metadata["entity_alignment"],
        "fixed_r2": dict(zip(fixed.layer.astype(str), fixed.r2_overall)),
        "ridgecv_r2": dict(zip(cv.layer.astype(str), cv.r2_overall)),
        "ridgecv_alpha": dict(zip(cv.layer.astype(str), cv.selected_alpha)),
        "controls": controls,
    }, indent=2, default=lambda value: value.item()))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
