#!/usr/bin/env python3
"""Strict story-level and query-path template-family held-out feasibility."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from common import HOPS, RELATIONS, SPLITS, all_cell_index, distribution, json_list, split_group_overlap


def add_template_family_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    template_to_family: dict[str, str] = {}
    for ids_raw, families_raw in zip(candidates["template_ids"], candidates["template_families"]):
        ids = json_list(ids_raw)
        families = json_list(families_raw)
        for template_id, family in zip(ids, families):
            if family is not None:
                previous = template_to_family.setdefault(str(template_id), str(family))
                if previous != str(family):
                    raise AssertionError((template_id, previous, family))

    def row_values(row) -> tuple[str, str]:
        story_families = sorted({str(x) for x in json_list(row.template_families) if x is not None})
        path_entities = [str(x) for x in json_list(row.path_entities)]
        edges = json_list(row.parsed_edges)
        path_families: set[str] = set()
        for left, right in zip(path_entities[:-1], path_entities[1:]):
            matching = [
                edge for edge in edges
                if {str(edge.get("arg1")), str(edge.get("arg2"))} == {left, right}
            ]
            if len(matching) != 1:
                raise AssertionError(
                    f"row {row.row_id}: expected one edge for {left}-{right}, got {len(matching)}"
                )
            template_id = str(matching[0]["template_id"])
            family = template_to_family.get(template_id)
            if family is None:
                raise AssertionError(f"row {row.row_id}: no family for {template_id}")
            path_families.add(family)
        if not path_families:
            raise AssertionError(f"row {row.row_id}: empty query path family set")
        return "|".join(story_families), "|".join(sorted(path_families))

    values = [row_values(row) for row in candidates.itertuples(index=False)]
    out = candidates.copy()
    out["story_template_family_set"] = [x[0] for x in values]
    out["query_path_template_family_set"] = [x[1] for x in values]
    return out


def _group_family_vectors(df: pd.DataFrame, family_col: str):
    idx = all_cell_index(HOPS)
    vectors = (
        df.groupby(["story_group_id", "computed_hop", "canonical_relation"], observed=True)
        .size().unstack(["computed_hop", "canonical_relation"], fill_value=0)
        .reindex(columns=idx, fill_value=0)
    )
    family_sets = (
        df.groupby("story_group_id", observed=True)[family_col]
        .apply(lambda values: frozenset(f for value in values for f in str(value).split("|") if f))
    )
    rows_by_group = df.groupby("story_group_id", observed=True).indices
    groups = vectors.index.astype(str).tolist()
    return groups, vectors.to_numpy(dtype=int), {str(k): family_sets.loc[k] for k in vectors.index}, rows_by_group


def _evaluate_partition(groups, vectors, family_sets, family_assignment):
    counts = np.zeros((3, len(HOPS) * len(RELATIONS)), dtype=int)
    group_split: dict[str, str] = {}
    excluded = 0
    for group, vector in zip(groups, vectors):
        destinations = {family_assignment[f] for f in family_sets[group]}
        if len(destinations) != 1:
            excluded += int(vector.sum())
            continue
        split_index = next(iter(destinations))
        counts[split_index] += vector
        group_split[group] = SPLITS[split_index]
    sizes = counts.sum(axis=1)
    coverage = np.count_nonzero(counts, axis=1)
    mins = counts.min(axis=1)
    total = int(sizes.sum())
    shares = sizes / total if total else np.zeros(3)
    ratio_error = float(np.abs(shares - np.asarray([0.70, 0.15, 0.15])).sum())
    score = (
        int(coverage.min()),
        int(coverage.sum()),
        int(mins.min()),
        int(min(mins[1], mins[2])),
        total,
        -ratio_error,
    )
    return score, counts, group_split, excluded


def best_family_partition(df: pd.DataFrame, family_col: str, definition: str) -> tuple[pd.DataFrame, dict]:
    groups, vectors, family_sets, _ = _group_family_vectors(df, family_col)
    families = sorted({family for values in family_sets.values() for family in values})
    if len(families) > 12:
        raise RuntimeError(f"Unexpected family count for exhaustive search: {len(families)}")
    best = None
    for labels in itertools.product(range(3), repeat=len(families)):
        if set(labels) != {0, 1, 2}:
            continue
        family_assignment = dict(zip(families, labels))
        evaluated = _evaluate_partition(groups, vectors, family_sets, family_assignment)
        score = evaluated[0]
        tie = tuple(labels)
        if best is None or (score, tuple(-x for x in tie)) > (best[0], tuple(-x for x in best[1])):
            best = (score, labels, evaluated)
    if best is None:
        raise RuntimeError("No three-way family partition")
    _, labels, (_, counts, group_split, excluded) = best
    assignment = dict(zip(families, labels))
    selected = df.loc[df["story_group_id"].astype(str).isin(group_split)].copy()
    selected["split"] = selected["story_group_id"].astype(str).map(group_split)
    selected["template_holdout_definition"] = definition
    selected["heldout_template_families"] = selected[family_col]
    dist = distribution(selected)
    assigned_families = {
        SPLITS[i]: sorted(f for f, s in assignment.items() if s == i) for i in range(3)
    }
    observed_families = {
        split: sorted({
            f for value in selected.loc[selected["split"].eq(split), family_col]
            for f in str(value).split("|") if f
        }) for split in SPLITS
    }
    overlaps = {
        "train_val": len(set(observed_families["train"]) & set(observed_families["val"])),
        "train_test": len(set(observed_families["train"]) & set(observed_families["test"])),
        "val_test": len(set(observed_families["val"]) & set(observed_families["test"])),
    }
    info = {
        "definition": definition,
        "family_column": family_col,
        "candidate_rows": int(len(df)),
        "included_rows": int(len(selected)),
        "excluded_cross_partition_rows": int(excluded),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "assigned_families": assigned_families,
        "observed_families": observed_families,
        "family_overlap": overlaps,
        "min_cells": {
            s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS
        },
        "covered_cells": {
            s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS
        },
        "story_group_overlap": split_group_overlap(selected, "story_group_id"),
        "feasible_at_10_per_eval_cell": bool(
            all(v == 40 for v in {
                s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS
            }.values())
            and min(
                int(dist.loc[dist["split"].eq("val"), "count"].min()),
                int(dist.loc[dist["split"].eq("test"), "count"].min()),
            ) >= 10
        ),
    }
    assert all(v == 0 for v in overlaps.values())
    assert all(v == 0 for v in info["story_group_overlap"].values())
    return selected, info


def build_template_heldout(candidates: pd.DataFrame):
    enriched = add_template_family_metadata(candidates)
    strict, strict_info = best_family_partition(
        enriched, "story_template_family_set", "strict_story_level_template_family"
    )
    query, query_info = best_family_partition(
        enriched, "query_path_template_family_set", "query_path_template_family"
    )
    adopted = strict if strict_info["feasible_at_10_per_eval_cell"] else query
    adopted_info = strict_info if strict_info["feasible_at_10_per_eval_cell"] else query_info
    adopted_info = dict(adopted_info)
    adopted_info["adopted_because"] = (
        "strict story-level met the >=10 validation/test cell gate"
        if strict_info["feasible_at_10_per_eval_cell"]
        else "strict story-level failed the >=10 validation/test cell gate; query-path definition adopted"
    )
    return adopted, adopted_info, strict_info, query_info
