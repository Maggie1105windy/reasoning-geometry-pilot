#!/usr/bin/env python3
"""Archive broad control, evaluate candidates, and emit the B0.2b report/protocol."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone

import pandas as pd

from common_b02b import (
    B02, DATASET_COMMIT, HOPS, OUT, PROVISIONAL, RELATIONS, SEED, SPLITS,
    protected_hashes, sha256, write_json,
)


def aggregate_hash(items: dict[str, str]) -> str:
    payload = "".join(f"{k}:{v}\n" for k, v in sorted(items.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_info(name: str) -> dict:
    return json.loads((OUT / f"{name}_feasibility.json").read_text(encoding="utf-8"))


def main() -> None:
    baseline = json.loads((OUT / "protected_split_hashes_before.json").read_text(encoding="utf-8"))
    current = protected_hashes()
    if baseline != current:
        changed = sorted(k for k in set(baseline) | set(current) if baseline.get(k) != current.get(k))
        raise RuntimeError(f"Protected B0.2 files changed: {changed}")

    broad_files = []
    for split in SPLITS:
        src = B02 / f"template_heldout_{split}.csv"
        dst = OUT / f"broad_template_heldout_{split}.csv"
        shutil.copy2(src, dst)
        broad_files.append(dst)
    shutil.copy2(B02 / "template_heldout_distribution.csv", OUT / "broad_template_heldout_distribution.csv")

    broad_dist = pd.read_csv(OUT / "broad_template_heldout_distribution.csv")
    support = (
        broad_dist.pivot_table(
            index=["computed_hop", "canonical_relation"], columns="split",
            values="count", fill_value=0, aggfunc="sum",
        ).reset_index()
    )
    for split in SPLITS:
        if split not in support:
            support[split] = 0
    support["common_supported"] = support[SPLITS].gt(0).all(axis=1)
    support.to_csv(OUT / "broad_template_common_support.csv", index=False)
    common_support = support.loc[support["common_supported"]].copy()

    surface = load_info("surface_template_heldout")
    mid = load_info("mid_template_heldout")
    broad_metrics = {
        "families": 7,
        "sizes": {
            split: int(len(pd.read_csv(OUT / f"broad_template_heldout_{split}.csv"))) for split in SPLITS
        },
        "covered_cells": {
            split: int((broad_dist.loc[broad_dist["split"].eq(split), "count"] > 0).sum()) for split in SPLITS
        },
        "common_supported_cells": int(len(common_support)),
        "missing_common_cells": int(40 - len(common_support)),
        "usable_hops": sorted(common_support["computed_hop"].astype(int).unique().tolist()),
    }

    def passes(info):
        return (
            info["covered_cells"]["train"] == 40
            and info["covered_cells"]["val"] == 40
            and info["covered_cells"]["test"] == 40
            and info["min_cells"]["val"] >= 5
            and info["min_cells"]["test"] >= 5
            and all(v == 0 for v in info["template_overlap"].values())
            and all(v == 0 for v in info["story_overlap"].values())
        )

    adopted = None
    if passes(mid):
        adopted = "mid_template_heldout"
    elif passes(surface):
        adopted = "surface_template_heldout"
    gate_status = "PASS" if adopted else "FAIL"

    protocol = {
        "stage": "B0.2b",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_commit": DATASET_COMMIT,
        "source_metadata": str(B02.parent / "stage_b0_1" / "stepgame_metadata.parquet"),
        "seed": SEED,
        "candidate_definition": {
            "parsing_status": "FULL_OK", "story_len": 5, "computed_hop": HOPS,
            "relations": RELATIONS, "contains_suspicious_template": False,
        },
        "group_key": "story_group_id",
        "heldout_scope": "query shortest-path templates; distractor templates are not constrained",
        "surface_skeleton_rule": "official template text with AA/BB entity placeholders normalized to <E>; duplicate skeletons are one group",
        "mid_level_rule": "broad family plus direction-lexeme-normalized syntactic frame; clock values, compass/directional phrases and degree adverbs are normalized without using model outcomes",
        "formal_gate": {
            "story_overlap": 0, "query_path_group_overlap": 0, "exact_duplicate_leakage": 0,
            "relations": RELATIONS, "hops": HOPS, "suspicious_templates": 0,
            "all_train_val_test_cells_positive": True,
            "minimum_val_cell": 5, "minimum_test_cell": 5,
        },
        "status": gate_status,
        "adopted_control": adopted,
        "surface_result": surface,
        "mid_result": mid,
        "broad_result": broad_metrics,
        "solver_note": "Finite deterministic local search found full nonzero support; fixed >=5 feasibility MILPs timed out without a feasible seed and without proving infeasibility.",
        "protected_b0_2_hashes_unchanged": True,
    }
    write_json(OUT / "template_split_protocol.json", protocol)

    checksum_paths = broad_files + [OUT / "broad_template_common_support.csv"]
    provisional_paths = sorted(PROVISIONAL.glob("surface_template_heldout_*.csv")) + sorted(PROVISIONAL.glob("mid_template_heldout_*.csv"))
    per_file = {path.name: sha256(path) for path in checksum_paths}
    provisional = {path.name: sha256(path) for path in provisional_paths}
    provisional_aggregate = {
        prefix: aggregate_hash({k: v for k, v in provisional.items() if k.startswith(prefix + "_")})
        for prefix in ["surface_template_heldout", "mid_template_heldout"]
    }
    checksums = {
        "algorithm": "SHA-256",
        "formal_status": gate_status,
        "frozen_broad_archive": per_file,
        "frozen_broad_archive_aggregate": aggregate_hash(per_file),
        "provisional_not_adopted": provisional,
        "provisional_aggregate_by_candidate": provisional_aggregate,
        "protected_b0_2_before": baseline,
        "protected_b0_2_after": current,
        "protected_unchanged": baseline == current,
    }
    write_json(OUT / "template_split_checksums.json", checksums)

    report = f"""# StepGame Stage B0.2b Template-Held-Out Audit

## Technical summary

No new fine/mid template control passes the requested minimum **5 validation and 5 test samples in every hop×relation cell**. Surface skeleton and mid-level syntactic-frame partitions both achieve nonzero 40-cell support with zero template/story leakage, but their observed minimum cells are only **{surface['min_cells']['val']}/{surface['min_cells']['test']}** and **{mid['min_cells']['val']}/{mid['min_cells']['test']}** respectively. The formal B0.2b status is therefore **FAIL**, and no new official template manifest is frozen.

## The existing broad control remains restricted-support

- Seven broad families; split sizes `{broad_metrics['sizes']}`.
- Covered cells by split: `{broad_metrics['covered_cells']}`.
- Common train/val/test support: **{broad_metrics['common_supported_cells']} / 40 cells**; usable hops `{broad_metrics['usable_hops']}`.
- The original B0.2 template files are unchanged; exact copies are archived under `broad_template_heldout_*.csv`.

Broad results may only be evaluated on `broad_template_common_support.csv`; missing cells must never be encoded as zero performance.

## Surface skeleton is the strongest available nonzero-support candidate

- Query-path groups: **{surface['template_groups']}**.
- Split group counts: `{surface['assigned_group_counts']}`; rows `{surface['sizes']}`.
- Covered cells: `{surface['covered_cells']}`; minimum cells `{surface['min_cells']}`.
- Template overlap `{surface['template_overlap']}`; story overlap `{surface['story_overlap']}`.

This grouping genuinely holds out normalized official surface skeletons, including duplicate template IDs with the same skeleton. It covers every cell but misses the requested minimum count, so its manifests remain provisional.

## Mid-level syntactic frames are stronger but no more feasible

- Query-path groups: **{mid['template_groups']}**.
- Definition: broad semantic family plus a direction-lexeme-normalized syntactic frame. Direction words, clock values, compass phrases, and degree adverbs are normalized; grammatical construction words remain.
- Split group counts: `{mid['assigned_group_counts']}`; rows `{mid['sizes']}`.
- Covered cells: `{mid['covered_cells']}`; minimum cells `{mid['min_cells']}`.
- Template overlap `{mid['template_overlap']}`; story overlap `{mid['story_overlap']}`.

The rule is based only on language structure and was fixed before evaluating split support. Mid-level is a stronger paraphrase control than exact skeleton, but it also fails the minimum-count Gate.

## Scope, methods, and robustness

The candidate cohort is the unchanged B0.2 Primary definition: FULL_OK, five-sentence stories, computed hop 1–5, eight non-overlap relations, and no suspicious templates. Query paths are unique in this cohort. Every template group receives one global split owner; rows whose path spans owners are excluded. No model, tokenizer, activation, or probe result was accessed.

Two optimization routes were tested: deterministic multi-start group assignment and sparse fixed-quota MILP feasibility. The latter timed out without a feasible >=5 seed and without proving mathematical infeasibility. Therefore the report states an empirical maximum found, not a proof of the global ceiling.

## Gate decision and next step

- Structural checks pass for the provisional surface and mid candidates: zero story/template leakage, legal relations/hops, and no suspicious templates.
- The quantitative minimum-count Gate fails.
- Keep Primary, Balanced, Composition, and Target-Absent frozen exactly as before.
- Do not enter Stage B1 with a claimed full-support template robustness control until the protocol either accepts a lower minimum, restricts the evaluated cells, or authorizes a different linguistic grouping definition.
"""
    (OUT / "B0_2B_TEMPLATE_HELDOUT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": gate_status, "adopted": adopted, "broad": broad_metrics, "surface": surface, "mid": mid}, indent=2))


if __name__ == "__main__":
    main()
