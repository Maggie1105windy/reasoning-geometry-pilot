#!/usr/bin/env python3
"""Composition-held-out split with zero signature and story-group leakage."""

from __future__ import annotations

import pandas as pd

from common import (
    RELATIONS, SEED, SPLITS, apply_assignment, distribution,
    greedy_group_assignment, split_group_overlap, stable_int,
)


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _component_vectors(df: pd.DataFrame) -> dict[str, list[int]]:
    idx = pd.MultiIndex.from_product([[2, 3, 4, 5], RELATIONS])
    table = (
        df.groupby(["composition_component", "computed_hop", "canonical_relation"], observed=True)
        .size().unstack(["computed_hop", "canonical_relation"], fill_value=0)
        .reindex(columns=idx, fill_value=0)
    )
    return {str(k): row.astype(int).tolist() for k, row in table.iterrows()}


def _repair_coverage(df: pd.DataFrame, assignment: dict[str, str]) -> tuple[dict[str, str], bool]:
    vectors = _component_vectors(df)
    counts = {s: [0] * 32 for s in SPLITS}
    for component, split in assignment.items():
        counts[split] = [a + b for a, b in zip(counts[split], vectors[component])]
    for _ in range(10000):
        missing = [(s, c) for s in SPLITS for c, value in enumerate(counts[s]) if value == 0]
        if not missing:
            return assignment, True
        receiver, cell = min(
            missing,
            key=lambda x: (
                sum(1 for component, split in assignment.items() if split != x[0] and vectors[component][x[1]] > 0),
                SPLITS.index(x[0]), x[1],
            ),
        )
        moves = []
        for component, donor in assignment.items():
            if donor == receiver or vectors[component][cell] == 0:
                continue
            vector = vectors[component]
            if any(value > 0 and counts[donor][i] - value <= 0 for i, value in enumerate(vector)):
                continue
            new_cells = sum(1 for i, value in enumerate(vector) if value > 0 and counts[receiver][i] == 0)
            moves.append((-new_cells, sum(vector), stable_int("composition-repair", component, receiver), component, donor))
        if not moves:
            return assignment, False
        _, _, _, component, donor = min(moves)
        vector = vectors[component]
        assignment[component] = receiver
        counts[donor] = [a - b for a, b in zip(counts[donor], vector)]
        counts[receiver] = [a + b for a, b in zip(counts[receiver], vector)]
    return assignment, False


def build_composition_heldout(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    derived = candidates.loc[candidates["computed_hop"].between(2, 5)].copy()
    if derived["unordered_composition_signature"].isna().any():
        raise AssertionError("Derived FULL_OK candidate has null unordered signature")
    signatures = sorted(derived["unordered_composition_signature"].astype(str).unique())
    uf = UnionFind(signatures)
    for _, group in derived.groupby("story_group_id", observed=True):
        values = sorted(group["unordered_composition_signature"].astype(str).unique())
        for other in values[1:]:
            uf.union(values[0], other)
    signature_component = {s: uf.find(s) for s in signatures}
    derived["composition_component"] = derived["unordered_composition_signature"].astype(str).map(signature_component)
    assignment = greedy_group_assignment(
        derived, "composition_component", (0.70, 0.15, 0.15), [2, 3, 4, 5], SEED + 303
    )
    assignment, coverage_feasible = _repair_coverage(derived, assignment)
    split_df = apply_assignment(derived, "composition_component", assignment)
    dist = distribution(split_df, [2, 3, 4, 5])
    observed = {
        split: set(split_df.loc[split_df["split"].eq(split), "unordered_composition_signature"].astype(str))
        for split in SPLITS
    }
    overlaps = {
        "train_val": len(observed["train"] & observed["val"]),
        "train_test": len(observed["train"] & observed["test"]),
        "val_test": len(observed["val"] & observed["test"]),
    }
    label_share = (
        split_df.groupby(["split", "canonical_relation"], observed=True).size()
        .groupby(level=0).transform(lambda x: x / x.sum())
    )
    max_label_share = {
        split: float(label_share.loc[split].max()) for split in SPLITS
    }
    info = {
        "candidate_rows": int(len(derived)),
        "included_rows": int(len(split_df)),
        "signature_count": int(len(signatures)),
        "component_count": int(split_df["composition_component"].nunique()),
        "coverage_repair_feasible": bool(coverage_feasible),
        "sizes": {s: int(split_df["split"].eq(s).sum()) for s in SPLITS},
        "signature_counts": {s: int(len(observed[s])) for s in SPLITS},
        "signature_overlap": overlaps,
        "story_group_overlap": split_group_overlap(split_df, "story_group_id"),
        "min_cells": {
            s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS
        },
        "covered_cells": {
            s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS
        },
        "max_relation_share": max_label_share,
    }
    assert all(v == 0 for v in overlaps.values())
    assert all(v == 0 for v in info["story_group_overlap"].values())
    return split_df, info
