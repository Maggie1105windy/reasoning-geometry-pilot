#!/usr/bin/env python3
"""Audit template granularity and save an enriched candidate table."""

from __future__ import annotations

import json

import pandas as pd

from common_b02b import INVENTORY, OUT, PROVISIONAL, candidate_data, protected_hashes, split_group_string, surface_skeleton, write_json


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROVISIONAL.mkdir(parents=True, exist_ok=True)
    candidates, excluded, inventory = candidate_data()
    full_inventory = pd.read_csv(INVENTORY)
    full_inventory["surface_skeleton"] = full_inventory["template_text_or_pattern"].map(surface_skeleton)
    baseline = protected_hashes()
    write_json(OUT / "protected_split_hashes_before.json", baseline)

    inventory.to_csv(OUT / "template_granularity_inventory.csv", index=False)
    candidates.to_parquet(PROVISIONAL / "template_candidates.parquet", index=False)

    mappings = {
        "template_id": "query_path_template_ids",
        "surface_skeleton": "query_path_surface_skeletons",
        "mid_level_family": "query_path_mid_families",
        "broad_family": "query_path_broad_families",
    }
    support_rows = []
    for grain, col in mappings.items():
        exploded = candidates[["row_id", "computed_hop", "canonical_relation", col]].copy()
        exploded["template_group"] = exploded[col].map(split_group_string)
        exploded = exploded.explode("template_group")
        relation_counts = (
            exploded.groupby(["canonical_relation", "template_group"], observed=True).size().rename("rows").reset_index()
        )
        relation_counts.insert(0, "grain", grain)
        support_rows.append(relation_counts)
        hop_counts = (
            exploded.groupby(["computed_hop", "canonical_relation", "template_group"], observed=True)
            .size().rename("rows").reset_index()
        )
        hop_counts.insert(0, "grain", grain)
        hop_counts.to_csv(OUT / f"{grain}_hop_relation_template_support.csv", index=False)
    pd.concat(support_rows, ignore_index=True).to_csv(OUT / "relation_template_support.csv", index=False)

    audit = {
        "primary_candidates": int(len(candidates)),
        "exact_duplicates_excluded": int(len(excluded)),
        "template_definitions_all": int(len(full_inventory)),
        "surface_skeletons_all": int(full_inventory["surface_skeleton"].nunique()),
        "template_definitions_ok": int(len(inventory)),
        "surface_skeletons_all_ok": int(inventory["surface_skeleton"].nunique()),
        "mid_level_families_all_ok": int(inventory["mid_level_family"].nunique()),
        "broad_families_all_ok": int(inventory["template_family"].nunique()),
        "query_path_groups_in_candidates": {
            grain: int(len({item for value in candidates[col] for item in split_group_string(value)}))
            for grain, col in mappings.items()
        },
        "story_groups_unique_in_candidate": bool(candidates["story_group_id"].is_unique),
        "protected_file_count": len(baseline),
    }
    write_json(OUT / "template_granularity_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
