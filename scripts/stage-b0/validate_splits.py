#!/usr/bin/env python3
"""Independent validation of frozen Stage B0.2 manifests and checksums."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pandas as pd

from common import HOPS, OUT, RELATIONS, ROOT, SPLITS, add_parsing_status, load_metadata, split_group_overlap, write_json


DATASETS = ["primary", "balanced", "template_heldout", "composition_heldout", "target_absent"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(name: str) -> pd.DataFrame:
    parts = []
    for split in SPLITS:
        part = pd.read_csv(OUT / f"{name}_{split}.csv")
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    source = add_parsing_status(load_metadata()).set_index("row_id", drop=False)
    checks = {}
    loaded = {name: load_dataset(name) for name in DATASETS}
    for name, frame in loaded.items():
        checks[f"{name}_row_ids_unique"] = bool(frame["row_id"].is_unique)
        checks[f"{name}_row_ids_exist"] = bool(frame["row_id"].isin(source.index).all())
        joined = source.loc[frame["row_id"]]
        checks[f"{name}_fields_match_source"] = bool(
            (joined["story_group_id"].astype(str).to_numpy() == frame["story_group_id"].astype(str).to_numpy()).all()
            and (joined["canonical_relation"].to_numpy() == frame["canonical_relation"].to_numpy()).all()
            and (joined["computed_hop"].astype(int).to_numpy() == frame["computed_hop"].astype(int).to_numpy()).all()
        )
        checks[f"{name}_full_ok"] = bool(joined["parsing_status"].eq("FULL_OK").all())
        checks[f"{name}_template_clean"] = bool((~joined["contains_suspicious_template"].astype(bool)).all())
        checks[f"{name}_relations_legal"] = bool(frame["canonical_relation"].isin(RELATIONS).all())
        expected_hops = [2, 3, 4, 5] if name in {"composition_heldout", "target_absent"} else HOPS
        checks[f"{name}_hops_legal"] = bool(frame["computed_hop"].isin(expected_hops).all())
        checks[f"{name}_story_overlap_zero"] = all(v == 0 for v in split_group_overlap(frame, "story_group_id").values())
        checks[f"{name}_exact_overlap_zero"] = all(v == 0 for v in split_group_overlap(frame, "exact_duplicate_group").values())
        split_sets = {s: set(frame.loc[frame["split"].eq(s), "row_id"]) for s in SPLITS}
        checks[f"{name}_row_split_overlap_zero"] = not (
            split_sets["train"] & split_sets["val"]
            or split_sets["train"] & split_sets["test"]
            or split_sets["val"] & split_sets["test"]
        )

    balanced = loaded["balanced"]
    expected = {"train": 34, "val": 11, "test": 11}
    checks["balanced_exact_cell_quotas"] = all(
        set(
            balanced.loc[balanced["split"].eq(split)]
            .groupby(["computed_hop", "canonical_relation"], observed=True).size()
        ) == {quota}
        for split, quota in expected.items()
    )

    template = loaded["template_heldout"]
    template_sets = {
        s: {f for value in template.loc[template["split"].eq(s), "heldout_template_families"] for f in str(value).split("|") if f}
        for s in SPLITS
    }
    checks["template_family_overlap_zero"] = not (
        template_sets["train"] & template_sets["val"]
        or template_sets["train"] & template_sets["test"]
        or template_sets["val"] & template_sets["test"]
    )

    composition = loaded["composition_heldout"]
    signature_sets = {
        s: set(composition.loc[composition["split"].eq(s), "unordered_composition_signature"].astype(str))
        for s in SPLITS
    }
    checks["composition_signature_overlap_zero"] = not (
        signature_sets["train"] & signature_sets["val"]
        or signature_sets["train"] & signature_sets["test"]
        or signature_sets["val"] & signature_sets["test"]
    )
    checks["composition_all_32_cells"] = all(
        len(
            composition.loc[composition["split"].eq(s)]
            .groupby(["computed_hop", "canonical_relation"], observed=True)
        ) == 32 for s in SPLITS
    )

    primary = loaded["primary"]
    target = loaded["target_absent"]
    expected_target_ids = set(
        primary.loc[
            primary["computed_hop"].between(2, 5)
            & source.loc[primary["row_id"], "target_absent_anywhere"].astype(bool).to_numpy(),
            "row_id",
        ]
    )
    checks["target_absent_exact_primary_subset"] = set(target["row_id"]) == expected_target_ids
    primary_membership = primary.set_index("row_id")["split"].to_dict()
    checks["target_absent_membership_preserved"] = all(primary_membership[r.row_id] == r.split for r in target.itertuples())

    checksum_payload = json.loads((OUT / "split_checksums.json").read_text(encoding="utf-8"))
    checks["all_frozen_checksums_match"] = all(
        sha256(OUT / name) == expected for name, expected in checksum_payload["per_file"].items()
    )
    raw_status = subprocess.run(
        ["git", "-C", str(ROOT / "third_party" / "StepGame"), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    checks["raw_repository_clean"] = raw_status == ""
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"status": status, "failed_checks": [k for k, v in checks.items() if not v], "checks": checks}
    write_json(OUT / "validation_summary.json", payload)
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
