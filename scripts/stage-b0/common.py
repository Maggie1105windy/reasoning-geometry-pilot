#!/usr/bin/env python3
"""Shared deterministic utilities for StepGame Stage B0.2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260823
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
HOPS = [1, 2, 3, 4, 5]
SPLITS = ["train", "val", "test"]
ROOT = Path(__file__).resolve().parents[2]
B01 = ROOT / "artifacts" / "stage_b0_1"
OUT = ROOT / "artifacts" / "stage_b0_2"
PROVISIONAL = OUT / "_provisional"
METADATA_PATH = B01 / "stepgame_metadata.parquet"
DATASET_COMMIT = "5e6aff1563a6d7f46ee1f1aeff98b94e68c29005"


def stable_int(*parts: object, seed: int = SEED) -> int:
    payload = "|".join([str(seed), *(str(x) for x in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def json_list(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    return pd.read_parquet(METADATA_PATH)


def add_parsing_status(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "story", "question", "query_arg1", "query_arg2", "canonical_relation",
        "computed_hop", "explicit_relations", "parsed_edges", "template_ids",
        "template_families", "path_entities", "path_relations",
    ]
    complete = df[required].notna().all(axis=1)
    full = (
        df["parse_success"].astype(bool)
        & df["graph_success"].astype(bool)
        & df["query_entities_in_graph"].astype(bool)
        & complete
    )
    path_only = (~full) & df["graph_success"].astype(bool) & df["computed_hop"].notna()
    out = df.copy()
    out["parsing_status"] = np.select(
        [full, path_only], ["FULL_OK", "PATH_ONLY"], default="FAILED"
    )
    return out


def primary_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        df["parsing_status"].eq("FULL_OK")
        & df["story_len"].eq(5)
        & df["computed_hop"].isin(HOPS)
        & df["canonical_relation"].isin(RELATIONS)
        & ~df["contains_suspicious_template"].astype(bool)
    )
    raw = df.loc[mask].copy()
    raw["computed_hop"] = raw["computed_hop"].astype(int)
    raw = raw.sort_values("row_id", kind="mergesort")
    duplicate = raw.duplicated("exact_duplicate_group", keep="first")
    excluded = raw.loc[duplicate].copy()
    excluded["excluded_reason"] = "EXACT_DUPLICATE"
    kept = raw.loc[~duplicate].copy().reset_index(drop=True)
    assert kept["exact_duplicate_group"].is_unique
    return kept, excluded


def stratum_key(df: pd.DataFrame) -> pd.Series:
    return df["computed_hop"].astype(int).astype(str) + "|" + df["canonical_relation"]


def all_cell_index(hops=HOPS) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [list(hops), RELATIONS], names=["computed_hop", "canonical_relation"]
    )


def distribution(df: pd.DataFrame, hops=HOPS) -> pd.DataFrame:
    idx = all_cell_index(hops)
    if "split" not in df.columns:
        count = (
            df.groupby(["computed_hop", "canonical_relation"], observed=True)
            .size().reindex(idx, fill_value=0).rename("count").reset_index()
        )
        return count
    pieces = []
    for split in SPLITS:
        part = (
            df.loc[df["split"].eq(split)]
            .groupby(["computed_hop", "canonical_relation"], observed=True)
            .size().reindex(idx, fill_value=0).rename("count").reset_index()
        )
        part.insert(0, "split", split)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def group_vectors(
    df: pd.DataFrame,
    group_key: str,
    hops=HOPS,
) -> tuple[list[str], np.ndarray, pd.MultiIndex]:
    idx = all_cell_index(hops)
    table = (
        df.groupby([group_key, "computed_hop", "canonical_relation"], observed=True)
        .size().unstack(["computed_hop", "canonical_relation"], fill_value=0)
        .reindex(columns=idx, fill_value=0)
    )
    return table.index.astype(str).tolist(), table.to_numpy(dtype=np.int64), idx


def greedy_group_assignment(
    df: pd.DataFrame,
    group_key: str,
    ratios: tuple[float, float, float],
    hops=HOPS,
    seed: int = SEED,
) -> dict[str, str]:
    groups, vectors, _ = group_vectors(df, group_key, hops)
    total = vectors.sum(axis=0).astype(float)
    targets = np.outer(np.asarray(ratios, dtype=float), total)
    rarity = (vectors / np.maximum(total, 1.0)).sum(axis=1)
    sizes = vectors.sum(axis=1)
    order = sorted(
        range(len(groups)),
        key=lambda i: (-rarity[i], -sizes[i], stable_int("group-order", groups[i], seed=seed)),
    )
    current = np.zeros_like(targets)
    assignment: dict[str, str] = {}
    target_rows = np.asarray(ratios) * sizes.sum()
    current_rows = np.zeros(3, dtype=float)
    for i in order:
        v = vectors[i].astype(float)
        scores = []
        for s in range(3):
            after = current[s] + v
            scale = np.sqrt(targets[s] + 1.0)
            before_cell_error = np.square((current[s] - targets[s]) / scale).sum()
            after_cell_error = np.square((after - targets[s]) / scale).sum()
            before_over = np.square(np.maximum(current[s] - targets[s], 0.0) / scale).sum()
            after_over = np.square(np.maximum(after - targets[s], 0.0) / scale).sum()
            row_scale = np.sqrt(target_rows[s] + 1.0)
            before_row_error = ((current_rows[s] - target_rows[s]) / row_scale) ** 2
            after_row_error = ((current_rows[s] + sizes[i] - target_rows[s]) / row_scale) ** 2
            scores.append(
                (after_cell_error - before_cell_error)
                + 2.0 * (after_over - before_over)
                + 0.25 * (after_row_error - before_row_error)
            )
        best_score = min(scores)
        tied = [s for s, score in enumerate(scores) if abs(score - best_score) < 1e-12]
        chosen = min(tied, key=lambda s: stable_int("split-tie", groups[i], SPLITS[s], seed=seed))
        assignment[groups[i]] = SPLITS[chosen]
        current[chosen] += v
        current_rows[chosen] += sizes[i]
    return assignment


def apply_assignment(df: pd.DataFrame, group_key: str, assignment: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out[group_key].astype(str).map(assignment)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), group_key].astype(str).unique()[:5]
        raise AssertionError(f"Missing group assignment: {missing}")
    return out


def split_group_overlap(df: pd.DataFrame, key: str) -> dict[str, int]:
    sets = {s: set(df.loc[df["split"].eq(s), key].astype(str)) for s in SPLITS}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def manifest(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    columns = [
        "row_id", "story_group_id", "exact_duplicate_group", "computed_hop",
        "canonical_relation", "source_split", "parsing_status",
    ]
    for col in extra or []:
        if col in df.columns and col not in columns:
            columns.append(col)
    return df[columns].sort_values("row_id", kind="mergesort").reset_index(drop=True)


def write_distribution(df: pd.DataFrame, path: Path, hops=HOPS) -> pd.DataFrame:
    dist = distribution(df, hops)
    path.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(path, index=False)
    return dist


def min_cells(dist: pd.DataFrame) -> dict[str, int]:
    return {
        split: int(dist.loc[dist["split"].eq(split), "count"].min())
        for split in SPLITS
    }


def max_min_ratio(dist: pd.DataFrame) -> dict[str, float]:
    result = {}
    for split in SPLITS:
        values = dist.loc[dist["split"].eq(split), "count"]
        mn = int(values.min())
        result[split] = float(values.max() / mn) if mn else float("inf")
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)
