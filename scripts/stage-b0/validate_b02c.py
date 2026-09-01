#!/usr/bin/env python3
"""Independent validation for frozen Stage B0.2c artifacts."""

from __future__ import annotations

import json

import pandas as pd

from common_b02c import (
    GROUP_COLS, OUT, RELATIONS, SPLITS, aggregate_hash, broad_hashes,
    candidate_data, groups, json_read, protected_hashes, sha256, write_json,
)


def main() -> None:
    protocol = json_read(OUT / "template_test_protocol.json")
    checksum_payload = json_read(OUT / "template_test_checksums.json")
    grouping = protocol["grouping"]["name"]
    group_col = GROUP_COLS[grouping]
    heldout = set(protocol["grouping"]["heldout_test_groups"])
    candidates, excluded, _ = candidate_data()
    source = candidates.set_index("row_id", drop=False)

    parts = []
    for split in SPLITS:
        part = pd.read_csv(OUT / f"mid_template_test_{split}.csv")
        part["split"] = split
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    checks: dict[str, bool] = {}
    checks["formal_status_pass"] = protocol["status"] == "PASS" and protocol["adopted_control"].startswith("Mid-level")
    checks["surface_not_formally_run"] = protocol["surface_formal_attempt"] == "NOT_RUN_AFTER_MID_PASS"
    checks["all_candidate_rows_partitioned_once"] = (
        len(frame) == len(candidates) and frame["row_id"].is_unique and set(frame["row_id"]) == set(candidates["row_id"])
    )
    checks["exact_duplicates_excluded_zero"] = len(excluded) == 0
    aligned = source.loc[frame["row_id"]]
    checks["source_fields_match"] = bool(
        (aligned["story_group_id"].astype(str).to_numpy() == frame["story_group_id"].astype(str).to_numpy()).all()
        and (aligned["exact_duplicate_group"].astype(str).to_numpy() == frame["exact_duplicate_group"].astype(str).to_numpy()).all()
        and (aligned["computed_hop"].astype(int).to_numpy() == frame["computed_hop"].astype(int).to_numpy()).all()
        and (aligned["canonical_relation"].to_numpy() == frame["canonical_relation"].to_numpy()).all()
        and (aligned[group_col].astype(str).to_numpy() == frame[group_col].astype(str).to_numpy()).all()
    )
    checks["full_ok"] = bool(frame["parsing_status"].eq("FULL_OK").all())
    checks["hops_exact"] = set(frame["computed_hop"].astype(int)) == {1, 2, 3, 4, 5}
    checks["relations_exact"] = set(frame["canonical_relation"]) == set(RELATIONS)
    checks["suspicious_templates_zero"] = bool((~aligned["contains_suspicious_template"].astype(bool)).all())

    expected_test = candidates[group_col].map(lambda value: bool(groups(value) & heldout))
    actual_test_ids = set(frame.loc[frame["split"].eq("test"), "row_id"])
    checks["test_is_exactly_all_rows_containing_heldout"] = actual_test_ids == set(candidates.loc[expected_test, "row_id"])
    checks["contains_flag_exact"] = bool(
        (frame["contains_heldout_test_group"].astype(bool).to_numpy()
         == frame[group_col].map(lambda value: bool(groups(value) & heldout)).to_numpy()).all()
    )
    observed = {
        split: {group for value in frame.loc[frame["split"].eq(split), group_col] for group in groups(value)}
        for split in SPLITS
    }
    checks["heldout_absent_train"] = not (heldout & observed["train"])
    checks["heldout_absent_val"] = not (heldout & observed["val"])
    checks["all_heldout_observed_test"] = heldout <= observed["test"]

    for key in ["story_group_id", "exact_duplicate_group"]:
        sets = {split: set(frame.loc[frame["split"].eq(split), key]) for split in SPLITS}
        checks[f"{key}_overlap_zero"] = not (
            sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]
        )
    counts = frame.groupby(["split", "computed_hop", "canonical_relation"], observed=True).size()
    for split in SPLITS:
        checks[f"{split}_40_cells"] = len(counts.loc[split]) == 40 and bool((counts.loc[split] > 0).all())
    checks["test_min_cell_gate"] = int(counts.loc["test"].min()) >= 5
    checks["test_min_cell_priority_target"] = int(counts.loc["test"].min()) >= 10
    checks["train_larger_than_test"] = int((frame["split"] == "train").sum()) > int((frame["split"] == "test").sum())
    checks["train_larger_than_val"] = int((frame["split"] == "train").sum()) > int((frame["split"] == "val").sum())

    manifest_hashes = {path.name: sha256(path) for path in sorted(OUT.glob("mid_template_test_*.csv"))}
    checks["manifest_hashes_match"] = manifest_hashes == checksum_payload["adopted_manifests"]
    checks["manifest_aggregate_matches"] = aggregate_hash(manifest_hashes) == checksum_payload["adopted_manifest_aggregate_sha256"]
    checks["protected_b0_2_unchanged"] = (
        checksum_payload["protected_b0_2_before"] == protected_hashes() == checksum_payload["protected_b0_2_after"]
    )
    checks["broad_restricted_unchanged"] = (
        checksum_payload["broad_restricted_before"] == broad_hashes() == checksum_payload["broad_restricted_after"]
    )
    checks["no_model_work"] = not any(protocol["prohibitions_observed"][key] for key in ["model_loaded", "activations_extracted", "probe_trained", "new_taxonomy_created"])

    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "failed_checks": [key for key, value in checks.items() if not value],
        "checks": checks,
        "observed": {
            "rows": frame["split"].value_counts().reindex(SPLITS).astype(int).to_dict(),
            "min_cells": {split: int(counts.loc[split].min()) for split in SPLITS},
            "heldout_groups": len(heldout),
            "heldout_leakage_train": len(heldout & observed["train"]),
            "heldout_leakage_val": len(heldout & observed["val"]),
            "story_overlap_total": 0 if checks["story_group_id_overlap_zero"] else None,
        },
    }
    write_json(OUT / "validation_summary.json", payload)
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

