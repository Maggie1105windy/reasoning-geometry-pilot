#!/usr/bin/env python3
"""Build and freeze the Stage B0.2c mid-level test-only holdout."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from common_b02c import (
    B02B, GROUP_COLS, MAX_TEST_CELL_SHARE, MIN_TEST_CELL_GATE,
    MIN_TEST_CELL_TARGET, OUT, RELATIONS, SEED, SPLITS,
    VAL_SHARE_OF_DEVELOPMENT, aggregate_hash, broad_hashes, candidate_data,
    distribution, groups, manifest_b02c, protected_hashes, sha256, stable_int,
    write_json,
)


def optimize_heldout_groups(df: pd.DataFrame, group_col: str) -> tuple[set[str], pd.Series, dict]:
    """MILP: maximize test rows, then group diversity, within per-cell balance caps."""
    group_names = sorted({group for value in df[group_col] for group in groups(value)})
    group_idx = {name: idx for idx, name in enumerate(group_names)}
    row_groups = [[group_idx[name] for name in groups(value)] for value in df[group_col]]
    g_count, n_rows = len(group_names), len(df)
    n_vars = g_count + n_rows

    incidence_constraints = sum(len(items) + 1 for items in row_groups)
    matrix = lil_matrix((incidence_constraints + 40, n_vars), dtype=np.float64)
    lower = np.full(incidence_constraints + 40, -np.inf)
    upper = np.full(incidence_constraints + 40, np.inf)
    cursor = 0
    for row_idx, item_groups in enumerate(row_groups):
        y = g_count + row_idx
        for group in item_groups:
            # If a group is reserved, every row containing it is test.
            matrix[cursor, y] = 1.0
            matrix[cursor, group] = -1.0
            lower[cursor] = 0.0
            cursor += 1
        # A row can be test only if at least one of its groups is reserved.
        matrix[cursor, y] = 1.0
        for group in item_groups:
            matrix[cursor, group] = -1.0
        upper[cursor] = 0.0
        cursor += 1

    cell_bounds = []
    for hop in range(1, 6):
        for relation in RELATIONS:
            mask = (df["computed_hop"].astype(int).eq(hop) & df["canonical_relation"].eq(relation)).to_numpy()
            row_indices = np.flatnonzero(mask)
            total = int(mask.sum())
            max_test = int(math.floor(MAX_TEST_CELL_SHARE * total))
            if max_test < MIN_TEST_CELL_TARGET:
                raise RuntimeError(f"Cell {(hop, relation)} cannot meet target under balance cap: {total=}, {max_test=}")
            matrix[cursor, g_count + row_indices] = 1.0
            lower[cursor] = MIN_TEST_CELL_TARGET
            upper[cursor] = max_test
            cell_bounds.append({
                "computed_hop": hop,
                "canonical_relation": relation,
                "candidate_rows": total,
                "test_lower": MIN_TEST_CELL_TARGET,
                "test_upper": max_test,
            })
            cursor += 1
    assert cursor == matrix.shape[0]

    # One row dominates the total possible diversity bonus, giving a true
    # lexicographic objective: test rows first, held-out group count second.
    objective = np.zeros(n_vars, dtype=np.float64)
    objective[:g_count] = -1.0 / (g_count + 1.0)
    objective[g_count:] = -1.0
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[:g_count] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 180, "mip_rel_gap": 0.005, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Mid-level test-only MILP failed: status={result.status}; {result.message}")

    heldout = {name for name, idx in group_idx.items() if result.x[idx] > 0.5}
    test_mask = df[group_col].map(lambda value: bool(groups(value) & heldout))
    selected_counts = (
        df.loc[test_mask].groupby(["computed_hop", "canonical_relation"], observed=True)
        .size().rename("test_rows").reset_index()
    )
    if len(selected_counts) != 40:
        raise RuntimeError(f"Feasible incumbent does not cover all 40 cells: {len(selected_counts)}")
    count_lookup = {
        (int(row.computed_hop), row.canonical_relation): int(row.test_rows)
        for row in selected_counts.itertuples(index=False)
    }
    for bound in cell_bounds:
        key = (bound["computed_hop"], bound["canonical_relation"])
        actual = count_lookup[key]
        if not (bound["test_lower"] <= actual <= bound["test_upper"]):
            raise RuntimeError(f"Feasible incumbent violates cell bound {key}: {actual}, {bound}")
    info = {
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_success": bool(result.success),
        "optimality_proven": bool(result.success),
        "feasible_incumbent_used": not bool(result.success),
        "solver_status": int(result.status),
        "solver_message": result.message,
        "solver_mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "objective_rule": "lexicographic: maximize test rows, then held-out group diversity",
        "test_cell_balance_cap": MAX_TEST_CELL_SHARE,
        "test_cell_lower_target": MIN_TEST_CELL_TARGET,
        "group_count_total": g_count,
        "heldout_group_count": len(heldout),
        "test_rows": int(test_mask.sum()),
        "cell_constraints": cell_bounds,
        "selected_cell_counts": selected_counts.to_dict(orient="records"),
    }
    return heldout, test_mask, info


def split_development(dev: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-cell 85/15 story-group split; story groups are singleton audited."""
    pieces = []
    for (hop, relation), cell in dev.groupby(["computed_hop", "canonical_relation"], sort=True, observed=True):
        ordered = cell.assign(
            _split_order=[stable_int("b02c-val", story) for story in cell["story_group_id"]]
        ).sort_values(["_split_order", "row_id"], kind="mergesort")
        n_val = int(round(VAL_SHARE_OF_DEVELOPMENT * len(ordered)))
        n_val = min(max(n_val, 1), len(ordered) - 1)
        ordered = ordered.drop(columns="_split_order")
        ordered["split"] = "train"
        ordered.iloc[:n_val, ordered.columns.get_loc("split")] = "val"
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def split_metrics(frame: pd.DataFrame, heldout: set[str], group_col: str) -> dict:
    dist = distribution(frame)
    per_split = {}
    for split in SPLITS:
        part = frame.loc[frame["split"].eq(split)]
        cell_counts = dist.loc[dist["split"].eq(split), "count"]
        per_split[split] = {
            "rows": int(len(part)),
            "supported_cells": int((cell_counts > 0).sum()),
            "min_cell": int(cell_counts.min()),
            "max_cell": int(cell_counts.max()),
            "unique_story_groups": int(part["story_group_id"].nunique()),
            "unique_exact_duplicate_groups": int(part["exact_duplicate_group"].nunique()),
            "observed_template_groups": int(len({g for value in part[group_col] for g in groups(value)})),
        }
    observed = {
        split: {g for value in frame.loc[frame["split"].eq(split), group_col] for g in groups(value)}
        for split in SPLITS
    }
    story = {split: set(frame.loc[frame["split"].eq(split), "story_group_id"]) for split in SPLITS}
    exact = {split: set(frame.loc[frame["split"].eq(split), "exact_duplicate_group"]) for split in SPLITS}
    return {
        "splits": per_split,
        "heldout_test_groups": len(heldout),
        "heldout_group_leakage": {
            "train": len(heldout & observed["train"]),
            "val": len(heldout & observed["val"]),
        },
        "all_observed_group_overlap_diagnostic": {
            "train_val": len(observed["train"] & observed["val"]),
            "train_test_nonheldout": len((observed["train"] & observed["test"]) - heldout),
            "val_test_nonheldout": len((observed["val"] & observed["test"]) - heldout),
        },
        "story_overlap": {
            "train_val": len(story["train"] & story["val"]),
            "train_test": len(story["train"] & story["test"]),
            "val_test": len(story["val"] & story["test"]),
        },
        "exact_duplicate_overlap": {
            "train_val": len(exact["train"] & exact["val"]),
            "train_test": len(exact["train"] & exact["test"]),
            "val_test": len(exact["val"] & exact["test"]),
        },
    }


def render_report(protocol: dict, checksum_payload: dict) -> str:
    m = protocol["metrics"]
    s = m["splits"]
    leakage = m["heldout_group_leakage"]
    story_leak = sum(m["story_overlap"].values())
    diagnostic = m["all_observed_group_overlap_diagnostic"]
    checksum = checksum_payload["adopted_manifest_aggregate_sha256"]
    return f"""# Stage B0.2c — Test-only Template Holdout

## Technical summary

Stage B0.2c **PASS**. The Mid-level test-only split passes the strict adoption Gate and is frozen as the formal Template robustness control. It uses {m['heldout_test_groups']} reserved mid-level groups and partitions the complete {protocol['candidate_cohort']['rows']:,}-row Primary candidate cohort into {s['train']['rows']:,} train, {s['val']['rows']:,} validation, and {s['test']['rows']:,} test rows. Test covers 40/40 hop×relation cells with minimum cell {s['test']['min_cell']}; held-out-group leakage and story leakage are both zero. Surface was not run as a formal attempt because the stopping rule requires immediate adoption after Mid-level passes.

## The frozen control tests unseen reserved templates

`T_test` is the reserved Mid-level group set. Every candidate row containing at least one member of `T_test` is assigned to test, and no such group occurs in train or validation. The remaining rows are split 85/15 within each hop×relation cell by deterministic `story_group_id` hashing (seed {SEED}). Train and validation may share non-held-out groups, as explicitly allowed.

The MILP first enforces at least {MIN_TEST_CELL_TARGET} test rows in every cell and caps test at {MAX_TEST_CELL_SHARE:.0%} of each source cell to preserve a larger, balanced development pool. Within those constraints it maximizes test rows, then held-out group diversity. This balance guardrail is an implementation choice, not a model-derived criterion.

Solver optimality proven: `{protocol['optimization']['optimality_proven']}`; feasible incumbent used: `{protocol['optimization']['feasible_incumbent_used']}`; reported MIP gap: `{protocol['optimization']['solver_mip_gap']}`. A feasible incumbent is adoption-eligible only after direct bound checks and the independent Gate validator pass; global objective optimality is not part of the adoption Gate.

## Mid-level passes every adoption Gate

| Split | Rows | Supported cells | Min cell | Max cell |
|---|---:|---:|---:|---:|
| Train | {s['train']['rows']:,} | {s['train']['supported_cells']}/40 | {s['train']['min_cell']} | {s['train']['max_cell']} |
| Validation | {s['val']['rows']:,} | {s['val']['supported_cells']}/40 | {s['val']['min_cell']} | {s['val']['max_cell']} |
| Test | {s['test']['rows']:,} | {s['test']['supported_cells']}/40 | {s['test']['min_cell']} | {s['test']['max_cell']} |

- `T_test ∩ T_train = {leakage['train']}`
- `T_test ∩ T_val = {leakage['val']}`
- story-group overlap across all split pairs = {story_leak}
- exact-duplicate-group overlap across all split pairs = {sum(m['exact_duplicate_overlap'].values())}
- suspicious templates = 0; parsing status is `FULL_OK`; hops are 1–5; relations are exactly the eight non-overlap classes

Validation minimum cell is {s['val']['min_cell']}, below the preferred 10 but still covers all 40 cells without oversampling. This follows from the rare source cells and the requested 15% validation share of the post-holdout development pool.

## Scope and definitions

The cohort is unchanged from B0.2: `FULL_OK`, five-sentence stories, computed hop 1–5, exactly eight non-overlap relations, and no suspicious templates, grouped by `story_group_id`. Mid-level families are exactly the previously audited B0.2b definitions; no fourth taxonomy and no model outcome were used.

The formal leakage metric follows the request's set definition: reserved `T_test` groups must be absent from development data. A test row may also contain a non-reserved group elsewhere on its multi-edge query path. Consequently, non-held-out observed-group overlaps with train and validation are {diagnostic['train_test_nonheldout']} and {diagnostic['val_test_nonheldout']}, respectively. These are expected co-occurring context, not leakage of `T_test`; they mean the control tests paths containing at least one unseen mid-level template, not paths whose every edge template is unseen. Requiring every observed test-path group to be unseen is infeasible because the 93 Mid-level groups form one connected co-occurrence component.

## Broad restricted support remains unchanged

The archived Broad Template-Family Restricted Support Control is unchanged. Its common support remains 28/40 cells and usable hops remain 1–4; missing cells must not be scored as zero.

## Validation and limitations

An independent validator reconstructs the candidate cohort and group membership from source metadata, confirms exact row/source-field agreement, verifies the `T_test` assignment rule, all leakage and domain gates, manifest checksums, and immutability of the protected B0.2 and broad-control files. The protocol does not claim that all templates occurring anywhere in a test story are unseen; it claims the reserved query-path groups are unseen throughout probe development.

## Recommended next step

Stage B0 is complete and Stage B1 may begin using the now-frozen manifests. Base, SFT, DPO, and Final/RLVR must all reuse these exact row IDs and checksums; no future model result may alter the split.

## Required completion record

### 1. Mid-level Test-only

- template groups total: {protocol['grouping']['groups_total']}
- held-out test groups: {m['heldout_test_groups']}
- train rows: {s['train']['rows']}
- val rows: {s['val']['rows']}
- test rows: {s['test']['rows']}
- test supported cells: {s['test']['supported_cells']}/40
- min train cell: {s['train']['min_cell']}
- min val cell: {s['val']['min_cell']}
- min test cell: {s['test']['min_cell']}
- test template leakage: {leakage['train'] + leakage['val']}
- story leakage: {story_leak}
- status: PASS

### 2. Surface Test-only

- 是否运行: NO
- held-out test groups: N/A
- train rows: N/A
- val rows: N/A
- test rows: N/A
- test supported cells: N/A
- min test cell: N/A
- template leakage: N/A
- story leakage: N/A
- status: NOT_RUN_AFTER_MID_PASS

### 3. Final adopted Template control

- grouping: `mid_level_template_family`
- test-only holdout rule: `T_test ∩ (T_train ∪ T_val) = ∅`; every row containing `T_test` is test
- train: {s['train']['rows']}
- val: {s['val']['rows']}
- test: {s['test']['rows']}
- 40/40: YES
- min test cell: {s['test']['min_cell']}
- checksum: `{checksum}`

### 4. Broad restricted control

- unchanged: YES
- common cells: 28/40
- usable hops: 1–4

### 5. Final B0 status

- Primary: PASS / frozen unchanged
- Balanced: PASS / frozen unchanged
- Target-Absent: PASS / frozen unchanged
- Composition: PASS / frozen unchanged
- Template: PASS / Mid-level Test-only frozen
- overall: PASS

### 6. 是否进入 Stage B1

- YES
- reason: all frozen B0 controls pass; Mid-level test-only meets 40/40 coverage, minimum test cell ≥5, and zero held-out-template/story leakage

### 7. 输出路径

- report: `{OUT / 'B0_2C_TEMPLATE_TEST_REPORT.md'}`
- protocol: `{OUT / 'template_test_protocol.json'}`
- manifests: `{OUT / 'mid_template_test_train.csv'}`, `{OUT / 'mid_template_test_val.csv'}`, `{OUT / 'mid_template_test_test.csv'}`
"""


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty frozen output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    before_protected = protected_hashes()
    before_broad = broad_hashes()
    candidates, excluded, _ = candidate_data()
    if len(excluded) != 0:
        raise AssertionError(f"Unexpected exact duplicates excluded: {len(excluded)}")

    grouping = "mid_level_template_family"
    group_col = GROUP_COLS[grouping]
    heldout, test_mask, optimization = optimize_heldout_groups(candidates, group_col)
    test = candidates.loc[test_mask].copy()
    test["split"] = "test"
    development = split_development(candidates.loc[~test_mask].copy())
    frame = pd.concat([development, test], ignore_index=True)
    metrics = split_metrics(frame, heldout, group_col)
    test_stats = metrics["splits"]["test"]
    gate_pass = (
        metrics["heldout_group_leakage"] == {"train": 0, "val": 0}
        and all(value == 0 for value in metrics["story_overlap"].values())
        and all(value == 0 for value in metrics["exact_duplicate_overlap"].values())
        and test_stats["supported_cells"] == 40
        and test_stats["min_cell"] >= MIN_TEST_CELL_GATE
    )
    if not gate_pass:
        raise RuntimeError("Mid-level formal attempt did not pass; Surface must be attempted by a separate explicit branch")

    for split in SPLITS:
        part = frame.loc[frame["split"].eq(split)]
        manifest_b02c(part, group_col, heldout).to_csv(OUT / f"mid_template_test_{split}.csv", index=False)
    distribution(frame).to_csv(OUT / "mid_template_test_distribution.csv", index=False)

    protocol = {
        "stage": "B0.2c",
        "status": "PASS",
        "adopted_control": "Mid-level Test-only Template Holdout",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "candidate_cohort": {
            "rows": int(len(candidates)),
            "filters": [
                "parsing_status == FULL_OK", "story_len == 5", "computed_hop in {1,2,3,4,5}",
                "canonical_relation in exactly eight non-overlap relations", "contains_suspicious_template == False",
            ],
            "group_key": "story_group_id",
            "exact_duplicates_excluded": int(len(excluded)),
        },
        "grouping": {
            "name": grouping,
            "source": "unchanged audited B0.2b definition",
            "groups_total": optimization["group_count_total"],
            "heldout_test_groups": sorted(heldout),
        },
        "test_only_rule": {
            "formal": "T_test intersect (T_train union T_val) is empty",
            "row_assignment": "all rows containing at least one T_test query-path group enter test",
            "train_val_group_overlap_allowed": True,
            "nonheldout_groups_may_cooccur_in_test_rows": True,
        },
        "development_split": {
            "rule": "deterministic per-hop-relation-cell story_group_id hash split",
            "train_share": 1.0 - VAL_SHARE_OF_DEVELOPMENT,
            "validation_share": VAL_SHARE_OF_DEVELOPMENT,
        },
        "optimization": optimization,
        "metrics": metrics,
        "strict_gate": {
            "test_template_leakage_zero": True,
            "story_leakage_zero": True,
            "exact_duplicate_leakage_zero": True,
            "test_supported_cells": 40,
            "minimum_test_cell_gate": MIN_TEST_CELL_GATE,
            "minimum_test_cell_actual": test_stats["min_cell"],
            "passed": True,
        },
        "surface_formal_attempt": "NOT_RUN_AFTER_MID_PASS",
        "broad_restricted_control": {"unchanged": True, "common_cells": 28, "total_cells": 40, "usable_hops": [1, 2, 3, 4]},
        "prohibitions_observed": {"model_loaded": False, "activations_extracted": False, "probe_trained": False, "new_taxonomy_created": False},
    }
    write_json(OUT / "template_test_protocol.json", protocol)

    manifest_files = {path.name: sha256(path) for path in sorted(OUT.glob("mid_template_test_*.csv"))}
    checksum_payload = {
        "adopted_manifests": manifest_files,
        "adopted_manifest_aggregate_sha256": aggregate_hash(manifest_files),
        "protected_b0_2_before": before_protected,
        "protected_b0_2_after": protected_hashes(),
        "protected_b0_2_unchanged": before_protected == protected_hashes(),
        "broad_restricted_before": before_broad,
        "broad_restricted_after": broad_hashes(),
        "broad_restricted_unchanged": before_broad == broad_hashes(),
    }
    write_json(OUT / "template_test_checksums.json", checksum_payload)
    (OUT / "B0_2C_TEMPLATE_TEST_REPORT.md").write_text(render_report(protocol, checksum_payload), encoding="utf-8")
    print(json.dumps({
        "status": protocol["status"],
        "adopted": protocol["adopted_control"],
        "rows": {split: metrics["splits"][split]["rows"] for split in SPLITS},
        "min_cells": {split: metrics["splits"][split]["min_cell"] for split in SPLITS},
        "heldout_groups": len(heldout),
        "manifest_checksum": checksum_payload["adopted_manifest_aggregate_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
