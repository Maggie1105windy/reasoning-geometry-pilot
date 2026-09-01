#!/usr/bin/env python3
"""Independent B0.2b validation, including protected-file immutability."""

from __future__ import annotations

import json

import pandas as pd

from common_b02b import (
    B02, OUT, PROVISIONAL, RELATIONS, SPLITS, add_parsing_status, load_metadata,
    protected_hashes, sha256, split_overlap, write_json,
)


def load_candidate(prefix: str) -> pd.DataFrame:
    parts = []
    for split in SPLITS:
        part = pd.read_csv(PROVISIONAL / f"{prefix}_{split}.csv")
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    before = json.loads((OUT / "protected_split_hashes_before.json").read_text(encoding="utf-8"))
    after = protected_hashes()
    protocol = json.loads((OUT / "template_split_protocol.json").read_text(encoding="utf-8"))
    source = add_parsing_status(load_metadata()).set_index("row_id", drop=False)
    checks = {
        "protected_b0_2_files_unchanged": before == after,
        "formal_status_fail": protocol["status"] == "FAIL" and protocol["adopted_control"] is None,
    }
    for split in SPLITS:
        original = pd.read_csv(B02 / f"template_heldout_{split}.csv")
        archive = pd.read_csv(OUT / f"broad_template_heldout_{split}.csv")
        checks[f"broad_archive_{split}_exact"] = original.equals(archive)

    for prefix, group_col in [
        ("surface_template_heldout", "query_path_surface_skeletons"),
        ("mid_template_heldout", "query_path_mid_families"),
    ]:
        frame = load_candidate(prefix)
        checks[f"{prefix}_row_unique"] = bool(frame["row_id"].is_unique)
        checks[f"{prefix}_relations"] = bool(frame["canonical_relation"].isin(RELATIONS).all())
        checks[f"{prefix}_hops"] = bool(frame["computed_hop"].isin([1, 2, 3, 4, 5]).all())
        checks[f"{prefix}_full_ok"] = bool(frame["parsing_status"].eq("FULL_OK").all())
        source_rows = source.loc[frame["row_id"]]
        checks[f"{prefix}_source_fields_match"] = bool(
            (source_rows["story_group_id"].astype(str).to_numpy() == frame["story_group_id"].astype(str).to_numpy()).all()
            and (source_rows["exact_duplicate_group"].astype(str).to_numpy() == frame["exact_duplicate_group"].astype(str).to_numpy()).all()
            and (source_rows["canonical_relation"].to_numpy() == frame["canonical_relation"].to_numpy()).all()
            and (source_rows["computed_hop"].astype(int).to_numpy() == frame["computed_hop"].astype(int).to_numpy()).all()
        )
        checks[f"{prefix}_suspicious_zero"] = bool((~source_rows["contains_suspicious_template"].astype(bool)).all())
        checks[f"{prefix}_template_overlap_zero"] = all(v == 0 for v in split_overlap(frame, group_col).values())
        story_sets = {s: set(frame.loc[frame.split.eq(s), "story_group_id"]) for s in SPLITS}
        checks[f"{prefix}_story_overlap_zero"] = not (
            story_sets["train"] & story_sets["val"]
            or story_sets["train"] & story_sets["test"]
            or story_sets["val"] & story_sets["test"]
        )
        exact_sets = {s: set(frame.loc[frame.split.eq(s), "exact_duplicate_group"]) for s in SPLITS}
        checks[f"{prefix}_exact_duplicate_overlap_zero"] = not (
            exact_sets["train"] & exact_sets["val"]
            or exact_sets["train"] & exact_sets["test"]
            or exact_sets["val"] & exact_sets["test"]
        )
        counts = frame.groupby(["split", "computed_hop", "canonical_relation"], observed=True).size()
        checks[f"{prefix}_all_40_each"] = all(len(counts.loc[s]) == 40 for s in SPLITS)

    support = pd.read_csv(OUT / "broad_template_common_support.csv")
    checks["broad_common_support_flag_correct"] = bool(
        (support["common_supported"] == support[SPLITS].gt(0).all(axis=1)).all()
    )
    checksums = json.loads((OUT / "template_split_checksums.json").read_text(encoding="utf-8"))
    checks["checksum_protection_match"] = checksums["protected_unchanged"] and before == checksums["protected_b0_2_after"]
    checks["broad_archive_checksums_match"] = all(
        sha256(OUT / name) == expected
        for name, expected in checksums["frozen_broad_archive"].items()
    )
    checks["provisional_checksums_match"] = all(
        sha256(PROVISIONAL / name) == expected
        for name, expected in checksums["provisional_not_adopted"].items()
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"status": status, "failed_checks": [k for k, v in checks.items() if not v], "checks": checks}
    write_json(OUT / "validation_summary.json", payload)
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
