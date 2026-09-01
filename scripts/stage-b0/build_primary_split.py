#!/usr/bin/env python3
"""Primary natural-distribution and strict balanced-control builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    HOPS, RELATIONS, SEED, SPLITS, apply_assignment, distribution,
    greedy_group_assignment, manifest, min_cells, split_group_overlap, stable_int,
)


def build_primary(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if candidates["story_group_id"].is_unique:
        # In the audited Primary cohort each retained story group contributes one row.
        # Add Target-Absent as a secondary stratum so its membership-preserving
        # evaluation subset does not acquire avoidable empty cells.
        pieces = []
        work = candidates.copy()
        work["_secondary_stratum"] = (
            work["computed_hop"].astype(str) + "|" + work["canonical_relation"]
            + "|" + work["target_absent_anywhere"].astype(int).astype(str)
        )
        for stratum, part in work.groupby("_secondary_stratum", sort=True, observed=True):
            part = part.copy()
            part["_order"] = part["row_id"].map(
                lambda row_id: stable_int("primary-row", stratum, row_id)
            )
            part = part.sort_values(["_order", "row_id"], kind="mergesort")
            n = len(part)
            n_train = int(round(0.70 * n))
            n_val = int(round(0.15 * n))
            labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
            part["split"] = labels
            pieces.append(part.drop(columns=["_order", "_secondary_stratum"]))
        split_df = pd.concat(pieces, ignore_index=True)
    else:
        assignment = greedy_group_assignment(
            candidates, "story_group_id", (0.70, 0.15, 0.15), HOPS, SEED
        )
        split_df = apply_assignment(candidates, "story_group_id", assignment)
    dist = distribution(split_df)
    global_dist = distribution(candidates).rename(columns={"count": "global_count"})
    merged = dist.merge(global_dist, on=["computed_hop", "canonical_relation"])
    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    merged["share_within_global_cell"] = merged["count"] / merged["global_count"]
    merged["absolute_share_deviation"] = merged.apply(
        lambda r: abs(r["share_within_global_cell"] - split_ratios[r["split"]]), axis=1
    )
    info = {
        "candidates": int(len(candidates)),
        "sizes": {s: int(split_df["split"].eq(s).sum()) for s in SPLITS},
        "actual_ratios": {s: float(split_df["split"].eq(s).mean()) for s in SPLITS},
        "min_cells": min_cells(dist),
        "max_cells": {
            s: int(dist.loc[dist["split"].eq(s), "count"].max()) for s in SPLITS
        },
        "max_min_ratio": {
            s: float(
                dist.loc[dist["split"].eq(s), "count"].max()
                / dist.loc[dist["split"].eq(s), "count"].min()
            ) for s in SPLITS
        },
        "max_absolute_cell_share_deviation": float(merged["absolute_share_deviation"].max()),
        "story_group_overlap": split_group_overlap(split_df, "story_group_id"),
        "exact_group_overlap": split_group_overlap(split_df, "exact_duplicate_group"),
    }
    return split_df, info


def _availability(df: pd.DataFrame, assignment: dict[str, str]) -> dict[str, np.ndarray]:
    tmp = apply_assignment(df, "story_group_id", assignment)
    result = {}
    for split in SPLITS:
        table = (
            tmp.loc[tmp["split"].eq(split)]
            .groupby(["computed_hop", "canonical_relation"], observed=True)
            .size().reindex(
                pd.MultiIndex.from_product([HOPS, RELATIONS]), fill_value=0
            )
        )
        result[split] = table.to_numpy(dtype=int).copy()
    return result


def _group_vectors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = pd.MultiIndex.from_product([HOPS, RELATIONS])
    table = (
        df.groupby(["story_group_id", "computed_hop", "canonical_relation"], observed=True)
        .size().unstack(["computed_hop", "canonical_relation"], fill_value=0)
        .reindex(columns=idx, fill_value=0)
    )
    return {str(k): row.to_numpy(dtype=int) for k, row in table.iterrows()}


def _repair_assignment(
    df: pd.DataFrame,
    assignment: dict[str, str],
    quotas: dict[str, int],
) -> tuple[dict[str, str], bool]:
    vectors = _group_vectors(df)
    availability = _availability(df, assignment)
    q = {s: np.full(len(HOPS) * len(RELATIONS), quotas[s], dtype=int) for s in SPLITS}
    for _ in range(10000):
        deficits = [
            (q[s][c] - availability[s][c], s, c)
            for s in SPLITS for c in range(len(q[s])) if availability[s][c] < q[s][c]
        ]
        if not deficits:
            return assignment, True
        _, receiver, cell = max(deficits)
        moves = []
        for group, donor in assignment.items():
            if donor == receiver:
                continue
            v = vectors[group]
            if v[cell] <= 0:
                continue
            if np.any(availability[donor] - v < q[donor]):
                continue
            gain = min(v[cell], q[receiver][cell] - availability[receiver][cell])
            overshoot = np.maximum(availability[receiver] + v - q[receiver], 0).sum()
            moves.append((-gain, overshoot, int(v.sum()), stable_int("repair", group, receiver), group, donor))
        if not moves:
            return assignment, False
        _, _, _, _, group, donor = min(moves)
        v = vectors[group]
        assignment[group] = receiver
        availability[donor] -= v
        availability[receiver] += v
    return assignment, False


def _select_exact_quotas(
    df: pd.DataFrame,
    assignment: dict[str, str],
    quotas: dict[str, int],
) -> pd.DataFrame:
    assigned = apply_assignment(df, "story_group_id", assignment)
    chosen = []
    for split in SPLITS:
        for hop in HOPS:
            for relation in RELATIONS:
                cell = assigned.loc[
                    assigned["split"].eq(split)
                    & assigned["computed_hop"].eq(hop)
                    & assigned["canonical_relation"].eq(relation)
                ].copy()
                cell["_order"] = cell["row_id"].map(
                    lambda row_id: stable_int("balanced-row", split, hop, relation, row_id)
                )
                cell = cell.sort_values(["_order", "row_id"], kind="mergesort")
                if len(cell) < quotas[split]:
                    raise AssertionError((split, hop, relation, len(cell), quotas[split]))
                chosen.append(cell.head(quotas[split]).drop(columns="_order"))
    out = pd.concat(chosen, ignore_index=True)
    assert out["row_id"].is_unique
    return out


def build_balanced(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    theoretical = int(distribution(candidates)["count"].min())
    selected = None
    selected_n = None
    selected_quotas = None
    for n in range(theoretical, 19, -1):
        eval_q = max(10, int(round(0.20 * n)))
        train_q = n - 2 * eval_q
        if train_q <= 0:
            continue
        quotas = {"train": train_q, "val": eval_q, "test": eval_q}
        ratios = tuple(quotas[s] / n for s in SPLITS)
        assignment = greedy_group_assignment(
            candidates, "story_group_id", ratios, HOPS, SEED + 101 + n
        )
        assignment, feasible = _repair_assignment(candidates, assignment, quotas)
        if not feasible:
            continue
        try:
            selected = _select_exact_quotas(candidates, assignment, quotas)
        except AssertionError:
            continue
        selected_n = n
        selected_quotas = quotas
        break
    if selected is None:
        raise RuntimeError("No balanced solution with validation/test cell >=10")
    dist = distribution(selected)
    for split in SPLITS:
        assert set(dist.loc[dist["split"].eq(split), "count"]) == {selected_quotas[split]}
    info = {
        "theoretical_n": theoretical,
        "feasible_n": int(selected_n),
        "per_split_cell": selected_quotas,
        "total": int(len(selected)),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "story_group_overlap": split_group_overlap(selected, "story_group_id"),
        "exact_group_overlap": split_group_overlap(selected, "exact_duplicate_group"),
    }
    return selected, info
