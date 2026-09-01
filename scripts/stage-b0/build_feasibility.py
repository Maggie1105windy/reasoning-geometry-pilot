#!/usr/bin/env python3
"""Construct provisional Stage B0.2 splits and the pre-freeze feasibility report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_composition_heldout import build_composition_heldout
from build_primary_split import build_balanced, build_primary
from build_target_absent import build_target_absent
from build_template_heldout import build_template_heldout
from common import (
    DATASET_COMMIT, HOPS, OUT, PROVISIONAL, RELATIONS, SEED, SPLITS,
    add_parsing_status, distribution, load_metadata, manifest, markdown_table,
    primary_candidates, write_distribution, write_json,
)


def pivot_table(dist: pd.DataFrame) -> str:
    pivot = dist.pivot_table(
        index=["split", "computed_hop"], columns="canonical_relation",
        values="count", fill_value=0, aggfunc="sum"
    ).reset_index()
    pivot.columns.name = None
    return markdown_table(pivot)


def save_manifests(name: str, df: pd.DataFrame, extra: list[str]) -> None:
    for split in SPLITS:
        part = manifest(df.loc[df["split"].eq(split)], extra)
        part.to_csv(PROVISIONAL / f"{name}_{split}.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROVISIONAL.mkdir(parents=True, exist_ok=True)
    df = add_parsing_status(load_metadata())
    candidates, duplicate_excluded = primary_candidates(df)

    primary, primary_info = build_primary(candidates)
    balanced, balanced_info = build_balanced(candidates)
    template, template_info, template_strict_info, template_query_info = build_template_heldout(candidates)
    composition, composition_info = build_composition_heldout(candidates)
    target_absent, target_info = build_target_absent(primary)

    save_manifests("primary", primary, ["target_absent_anywhere"])
    save_manifests("balanced", balanced, ["target_absent_anywhere"])
    save_manifests(
        "template_heldout", template,
        ["template_holdout_definition", "heldout_template_families", "target_absent_anywhere"],
    )
    save_manifests(
        "composition_heldout", composition,
        ["unordered_composition_signature", "composition_component", "target_absent_anywhere"],
    )
    save_manifests("target_absent", target_absent, ["target_absent_anywhere"])
    duplicate_excluded[["row_id", "exact_duplicate_group", "story_group_id", "excluded_reason"]].to_csv(
        OUT / "exact_duplicates_excluded.csv", index=False
    )

    primary_dist = write_distribution(primary, PROVISIONAL / "primary_distribution.csv")
    balanced_dist = write_distribution(balanced, PROVISIONAL / "balanced_distribution.csv")
    template_dist = write_distribution(template, PROVISIONAL / "template_heldout_distribution.csv")
    composition_dist = write_distribution(
        composition, PROVISIONAL / "composition_heldout_distribution.csv", [2, 3, 4, 5]
    )
    target_dist = write_distribution(
        target_absent, PROVISIONAL / "target_absent_distribution.csv", [2, 3, 4, 5]
    )

    parsing_counts = {k: int(v) for k, v in df["parsing_status"].value_counts().items()}
    metrics = {
        "dataset_commit": DATASET_COMMIT,
        "seed": SEED,
        "parsing_status": parsing_counts,
        "primary_before_exact_dedup": int(len(candidates) + len(duplicate_excluded)),
        "exact_duplicates_removed": int(len(duplicate_excluded)),
        "primary": primary_info,
        "balanced": balanced_info,
        "template_adopted": template_info,
        "template_strict_story": template_strict_info,
        "template_query_path": template_query_info,
        "composition": composition_info,
        "target_absent": target_info,
    }
    write_json(OUT / "feasibility_metrics.json", metrics)

    hard_gate_checks = {
        "primary_all_40_cells": all(v > 0 for v in primary_info["min_cells"].values()),
        "primary_zero_story_leakage": all(v == 0 for v in primary_info["story_group_overlap"].values()),
        "balanced_eval_cells_at_least_10": min(balanced_info["per_split_cell"]["val"], balanced_info["per_split_cell"]["test"]) >= 10,
        "balanced_zero_story_leakage": all(v == 0 for v in balanced_info["story_group_overlap"].values()),
        "template_all_splits_nonempty": all(v > 0 for v in template_info["sizes"].values()),
        "template_zero_family_overlap": all(v == 0 for v in template_info["family_overlap"].values()),
        "template_zero_story_leakage": all(v == 0 for v in template_info["story_group_overlap"].values()),
        "composition_all_32_cells": all(v == 32 for v in composition_info["covered_cells"].values()),
        "composition_zero_signature_overlap": all(v == 0 for v in composition_info["signature_overlap"].values()),
        "composition_zero_story_leakage": all(v == 0 for v in composition_info["story_group_overlap"].values()),
    }
    limitation_checks = {
        "template_all_40_cells": all(v == 40 for v in template_info["covered_cells"].values()),
        "template_eval_min_cell_at_least_10": min(template_info["min_cells"]["val"], template_info["min_cells"]["test"]) >= 10,
        "primary_eval_min_cell_at_least_20": min(primary_info["min_cells"]["val"], primary_info["min_cells"]["test"]) >= 20,
    }
    hard_pass = all(hard_gate_checks.values())
    gate_status = "PASS" if hard_pass and all(limitation_checks.values()) else ("PASS_WITH_LIMITATION" if hard_pass else "FAIL")
    write_json(
        OUT / "feasibility_gate.json",
        {"status": gate_status, "hard_checks": hard_gate_checks, "limitation_checks": limitation_checks},
    )

    report = f"""# StepGame Stage B0.2 Split Feasibility Report

## Technical summary

The frozen-data design is **{gate_status}** under FULL_OK parsing, five-sentence stories, eight non-overlap relations, independently computed hop, template-clean filtering, story-level grouping, and exact-duplicate representative retention. This report is generated before final manifests are frozen.

- Primary retains **{len(primary):,}** candidates after excluding **{len(duplicate_excluded):,}** repeated exact rows.
- Balanced control reaches **{balanced_info['feasible_n']} samples/class/hop**, with per-cell train/val/test quotas of **{balanced_info['per_split_cell']['train']}/{balanced_info['per_split_cell']['val']}/{balanced_info['per_split_cell']['test']}**.
- Template held-out adopts **{template_info['definition']}**; strict story-level feasibility at the >=10 evaluation-cell gate is **{template_strict_info['feasible_at_10_per_eval_cell']}**.
- Composition held-out contains **{composition_info['signature_count']:,}** signatures in **{composition_info['component_count']:,}** leakage-safe components.

## Scope and parsing policy

`FULL_OK` requires a complete story/question parse, query entities, canonical explicit edges, template metadata, a reconstructable graph path, and path metadata. `PATH_ONLY` has a query path but lacks at least one complete Primary requirement. Primary uses FULL_OK only.

| status | rows |
| --- | ---: |
| FULL_OK | {parsing_counts.get('FULL_OK', 0):,} |
| PATH_ONLY | {parsing_counts.get('PATH_ONLY', 0):,} |
| FAILED | {parsing_counts.get('FAILED', 0):,} |

## Primary natural-distribution split

- Candidate rows: **{len(primary):,}**
- Sizes: train **{primary_info['sizes']['train']:,}**, val **{primary_info['sizes']['val']:,}**, test **{primary_info['sizes']['test']:,}**
- Minimum cells: train **{primary_info['min_cells']['train']}**, val **{primary_info['min_cells']['val']}**, test **{primary_info['min_cells']['test']}**
- Maximum absolute per-cell share deviation from 70/15/15: **{primary_info['max_absolute_cell_share_deviation']:.4f}**
- Story-group overlap: `{primary_info['story_group_overlap']}`

{pivot_table(primary_dist)}

## Strict balanced control

- B0.1 theoretical ceiling after current gates: **{balanced_info['theoretical_n']} samples/class/hop**
- Feasible ceiling under group and split constraints: **{balanced_info['feasible_n']} samples/class/hop**
- Exact per-cell train/val/test: **{balanced_info['per_split_cell']['train']}/{balanced_info['per_split_cell']['val']}/{balanced_info['per_split_cell']['test']}**
- Total: **{balanced_info['total']:,}**

{pivot_table(balanced_dist)}

## Template-family held-out feasibility

### Strict story-level definition

- Included rows: **{template_strict_info['included_rows']:,}**
- Split sizes: `{template_strict_info['sizes']}`
- Covered cells: `{template_strict_info['covered_cells']}`
- Minimum cells: `{template_strict_info['min_cells']}`
- Meets >=10 val/test cell gate: **{template_strict_info['feasible_at_10_per_eval_cell']}**

### Query-path definition

- Included rows: **{template_query_info['included_rows']:,}**
- Split sizes: `{template_query_info['sizes']}`
- Families: `{template_query_info['assigned_families']}`
- Covered cells: `{template_query_info['covered_cells']}`
- Minimum cells: `{template_query_info['min_cells']}`
- Family overlap: `{template_query_info['family_overlap']}`

Adopted definition: **{template_info['definition']}**. Distractor templates may recur only under the query-path definition; query-path family sets remain disjoint.

{pivot_table(template_dist)}

## Composition-held-out feasibility

- Split sizes: `{composition_info['sizes']}`
- Signature counts: `{composition_info['signature_counts']}`
- Signature overlap: `{composition_info['signature_overlap']}`
- Covered hop×relation cells: `{composition_info['covered_cells']}`
- Minimum cells: `{composition_info['min_cells']}`
- Maximum relation share: `{composition_info['max_relation_share']}`

{pivot_table(composition_dist)}

## Derived Target-Absent subset preserves Primary membership

- Total: **{target_info['total']:,}**
- Hop totals: `{target_info['hop_counts']}`
- Minimum overall hop×relation cell: **{target_info['min_relation_cell_overall']}**
- Per-split minimum cells: `{target_info['min_cells_by_split']}`
- h=5 split sizes: `{target_info['h5_sizes_by_split']}`

{pivot_table(target_dist)}

## Gate and limitations

Gate status: **{gate_status}**. Hard checks: `{hard_gate_checks}`. Limitation checks: `{limitation_checks}`.

- Template families are the seven broad B0.1 semantic/surface families, not individual template IDs.
- Query-path held-out does not claim that distractor sentence families are unseen.
- Composition components merge signatures linked by repeated-story groups; this is stricter than row-level signature assignment.
- Primary validation/test bottleneck cells below 20 are reported, not oversampled.

## Recommended next step

If the feasibility gate is accepted, freeze exactly these provisional row IDs, calculate SHA-256 checksums, and run an independent validation pass. Do not use model outputs to revise any assignment.
"""
    (OUT / "B0_2_FEASIBILITY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({"gate": gate_status, "hard_checks": hard_gate_checks, "limitation_checks": limitation_checks}, indent=2))


if __name__ == "__main__":
    main()
