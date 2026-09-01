#!/usr/bin/env python3
"""Freeze accepted provisional manifests, protocol, checksums, and split report."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import DATASET_COMMIT, HOPS, METADATA_PATH, OUT, PROVISIONAL, RELATIONS, SEED, SPLITS, write_json


DATASETS = ["primary", "balanced", "template_heldout", "composition_heldout", "target_absent"]
DISTRIBUTIONS = [
    "primary_distribution.csv", "balanced_distribution.csv",
    "template_heldout_distribution.csv", "composition_heldout_distribution.csv",
    "target_absent_distribution.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(items: list[tuple[str, str]]) -> str:
    payload = "".join(f"{name}:{value}\n" for name, value in sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    gate = json.loads((OUT / "feasibility_gate.json").read_text(encoding="utf-8"))
    if gate["status"] not in {"PASS", "PASS_WITH_LIMITATION"}:
        raise RuntimeError(f"Refusing freeze: feasibility gate is {gate['status']}")
    metrics = json.loads((OUT / "feasibility_metrics.json").read_text(encoding="utf-8"))

    frozen_files = []
    for dataset in DATASETS:
        for split in SPLITS:
            name = f"{dataset}_{split}.csv"
            source = PROVISIONAL / name
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, OUT / name)
            frozen_files.append(name)
    for name in DISTRIBUTIONS:
        shutil.copy2(PROVISIONAL / name, OUT / name)

    protocol = {
        "stage": "B0.2",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": str(METADATA_PATH),
        "source_metadata_sha256": sha256(METADATA_PATH),
        "dataset_commit": DATASET_COMMIT,
        "random_seed": SEED,
        "primary_definition": {
            "dataset": "StepGame CompleteVersion/clean",
            "parsing_status": "FULL_OK",
            "story_len": 5,
            "computed_hop": HOPS,
            "canonical_relation": RELATIONS,
            "contains_suspicious_template": False,
            "recorded_hop_or_hop_match_used": False,
        },
        "group_key": "story_group_id",
        "relation_classes": RELATIONS,
        "hop_values": HOPS,
        "template_policy": "exclude all B0.1 ERROR/UNKNOWN templates from every cohort; template held-out uses query-path broad template_family sets and permits distractor-family reuse",
        "duplicate_policy": "retain one representative per exact_duplicate_group; keep same-story/different-query rows together",
        "primary_split_method": "deterministic 70/15/15 hop×relation×target-absent secondary stratification; audited Primary story groups are singleton after filtering",
        "balanced_split_method": "maximize common N/class/hop; group assignment plus capacity repair; exact 34/11/11 rows per cell",
        "template_heldout_method": "exhaustive three-way partition of seven broad B0.1 families; maximize common cell coverage; exclude cross-partition query paths; story groups indivisible",
        "composition_heldout_method": "union unordered signatures linked by a story; deterministic component stratification plus coverage repair",
        "target_absent_method": "subset of frozen Primary membership; computed_hop>=2 and target_relation_present_anywhere=False",
        "feasibility_gate": gate,
        "known_limitations": [
            "Broad-family template held-out cannot cover all 40 hop×relation cells in all three splits.",
            "Primary validation/test minimum cells are below 20 and are not oversampled.",
        ],
    }
    write_json(OUT / "split_protocol.json", protocol)

    per_file = {name: sha256(OUT / name) for name in sorted(frozen_files)}
    aggregate = {}
    for dataset in DATASETS:
        items = [(name, value) for name, value in per_file.items() if name.startswith(dataset + "_")]
        aggregate[dataset] = aggregate_hash(items)
    checksum_payload = {
        "algorithm": "SHA-256",
        "per_file": per_file,
        "aggregate_by_dataset": aggregate,
    }
    write_json(OUT / "split_checksums.json", checksum_payload)

    p = metrics["primary"]
    b = metrics["balanced"]
    t = metrics["template_adopted"]
    c = metrics["composition"]
    a = metrics["target_absent"]
    report = f"""# StepGame Stage B0.2 Frozen Split Report

## Technical summary

The B0.2 manifests are frozen with seed **{SEED}** and status **{gate['status']}**. Primary, balanced, and composition splits satisfy their coverage and zero-leakage gates. Template held-out is frozen as a query-path broad-family diagnostic with zero family/story leakage, but it cannot support a complete 40-cell comparison.

## Frozen Primary split preserves the natural candidate distribution

- Candidates: **{p['candidates']:,}**
- Train/val/test: **{p['sizes']['train']:,}/{p['sizes']['val']:,}/{p['sizes']['test']:,}**
- Minimum hop×relation cells: **{p['min_cells']['train']}/{p['min_cells']['val']}/{p['min_cells']['test']}**
- Story-group and exact-group overlaps: `{p['story_group_overlap']}` / `{p['exact_group_overlap']}`

Primary keeps every eligible row after the candidate filter because no exact-duplicate group survives twice within this cohort. Evaluation cells below 20 remain unchanged and must be reflected in uncertainty estimates downstream.

## Balanced control reaches the theoretical ceiling

- Feasible N/class/hop: **{b['feasible_n']}**
- Exact per-cell train/val/test: **{b['per_split_cell']['train']}/{b['per_split_cell']['val']}/{b['per_split_cell']['test']}**
- Total: **{b['total']:,}**
- Leakage: `{b['story_group_overlap']}`

## Template held-out is restricted-support, not a complete factorial control

- Definition: **{t['definition']}**
- Families: `{t['assigned_families']}`
- Rows: **{t['included_rows']:,}**; split sizes `{t['sizes']}`
- Covered cells: `{t['covered_cells']}`; minimum cells `{t['min_cells']}`
- Family overlap: `{t['family_overlap']}`

The seven B0.1 broad families cannot be partitioned three ways while preserving all 40 hop×relation cells. Results from this manifest must therefore be compared only on shared supported cells, and must be labeled query-path template-family held-out; distractor families may repeat.

## Composition held-out preserves all derived cells

- Unordered signatures: **{c['signature_count']:,}** in **{c['component_count']:,}** leakage-safe components
- Split signatures: `{c['signature_counts']}`
- Split rows: `{c['sizes']}`
- Minimum hop×relation cells: `{c['min_cells']}`
- Signature overlap: `{c['signature_overlap']}`

## Target-Absent remains a Primary-membership subset

- Hop totals: `{a['hop_counts']}`
- Split rows: `{a['sizes']}`
- Minimum overall relation cell: **{a['min_relation_cell_overall']}**
- Per-split minimum cells: `{a['min_cells_by_split']}`

## Parsing, grouping, and reproducibility contract

- FULL_OK/PATH_ONLY/FAILED: **{metrics['parsing_status']['FULL_OK']:,}/{metrics['parsing_status']['PATH_ONLY']:,}/{metrics['parsing_status']['FAILED']:,}**
- PATH_ONLY is excluded because complete explicit-edge and template/path metadata are required by held-out controls.
- Every manifest uses `story_group_id` as the minimum grouping unit; exact duplicates cannot cross a split.
- Checksums are stored in `split_checksums.json`; Stage B1 must verify them before use.

## Limitations and next decision

Primary test cells have a minimum of **{p['min_cells']['test']}**, and the Target-Absent test minimum is **{a['min_cells_by_split']['test']}**. The Template-held-out manifest has unsupported cells. These limitations do not invalidate Primary/Balanced/Composition, but they require an explicit human decision before Stage B1.
"""
    (OUT / "B0_2_SPLIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"frozen_files": len(frozen_files), "aggregate_checksums": aggregate}, indent=2))


if __name__ == "__main__":
    main()
