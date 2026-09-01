#!/usr/bin/env python3
"""Shared definitions for the Stage B0.2b template-held-out revision."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stage_b0_2"))
from common import (  # noqa: E402
    B01, DATASET_COMMIT, HOPS, RELATIONS, SEED, SPLITS,
    add_parsing_status, load_metadata, manifest, primary_candidates,
)

B02 = ROOT / "artifacts" / "stage_b0_2"
OUT = ROOT / "artifacts" / "stage_b0_2b"
PROVISIONAL = OUT / "_provisional"
INVENTORY = B01 / "template_inventory.csv"
PROTECTED_PREFIXES = ["primary", "balanced", "composition_heldout", "target_absent"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    files = []
    for prefix in PROTECTED_PREFIXES:
        files.extend(sorted(B02.glob(f"{prefix}_*.csv")))
    return {path.name: sha256(path) for path in sorted(set(files))}


def stable_int(*parts: object, seed: int = SEED) -> int:
    payload = "|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def json_list(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def surface_skeleton(text: str) -> str:
    return str(text).replace("AA", "<E>").replace("BB", "<E>")


def mid_level_signature(skeleton: str, broad_family: str) -> str:
    """Direction-lexeme-normalized syntactic frame; no outcome information."""
    text = skeleton.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    # Clock values encode direction, while the construction itself is retained.
    text = re.sub(r"\b(?:1[0-2]|[1-9])(?::00)?\b", "<dir>", text)
    # Normalize multi-token compass/diagonal and relative-direction phrases first.
    phrases = [
        r"upper\s+and\s+slightly\s+to\s+the\s+(?:left|right)",
        r"above\s+and\s+(?:slightly\s+)?to\s+the\s+(?:left|right)",
        r"below\s+and\s+(?:slightly\s+)?to\s+the\s+(?:left|right)",
        r"top[- ]?(?:left|right)", r"bottom[- ]?(?:left|right)",
        r"upper[- ]?(?:left|right)", r"lower[- ]?(?:left|right)",
        r"north[- ]?(?:east|west)", r"south[- ]?(?:east|west)",
        r"(?:left|right)[- ]?(?:upper|lower)",
    ]
    for pattern in phrases:
        text = re.sub(pattern, "<dir>", text)
    text = re.sub(
        r"\b(?:left|right|above|below|upper|lower|top|bottom|north|south|east|west|over|under)\b",
        "<dir>", text,
    )
    text = re.sub(r"\b(?:directly|slightly)\b", "<degree>", text)
    text = re.sub(r"\b(?:45)\s+degree(?:s)?\b", "<angle>", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return f"{broad_family}|{text}"


def build_template_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str], pd.DataFrame]:
    inventory = pd.read_csv(INVENTORY)
    inventory = inventory.loc[inventory["semantic_status"].eq("OK")].copy()
    inventory["surface_skeleton"] = inventory["template_text_or_pattern"].map(surface_skeleton)
    inventory["mid_level_family"] = inventory.apply(
        lambda row: mid_level_signature(row["surface_skeleton"], row["template_family"]), axis=1
    )
    surface = inventory.set_index("template_id")["surface_skeleton"].to_dict()
    mid = inventory.set_index("template_id")["mid_level_family"].to_dict()
    broad = inventory.set_index("template_id")["template_family"].to_dict()
    return surface, mid, broad, inventory


def add_query_path_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surface_map, mid_map, broad_map, inventory = build_template_maps()

    def groups_for_row(row):
        path = [str(x) for x in json_list(row.path_entities)]
        edges = json_list(row.parsed_edges)
        template_ids = []
        for left, right in zip(path[:-1], path[1:]):
            matches = [
                edge for edge in edges
                if {str(edge.get("arg1")), str(edge.get("arg2"))} == {left, right}
            ]
            if len(matches) != 1:
                raise AssertionError((row.row_id, left, right, len(matches)))
            template_ids.append(str(matches[0]["template_id"]))
        try:
            surfaces = sorted({surface_map[x] for x in template_ids})
            mids = sorted({mid_map[x] for x in template_ids})
            broads = sorted({broad_map[x] for x in template_ids})
        except KeyError as exc:
            raise AssertionError(f"Non-OK template leaked into candidate row {row.row_id}: {exc}")
        return (
            "||".join(template_ids),
            "||".join(surfaces),
            "||".join(mids),
            "||".join(broads),
        )

    values = [groups_for_row(row) for row in df.itertuples(index=False)]
    out = df.copy()
    out["query_path_template_ids"] = [x[0] for x in values]
    out["query_path_surface_skeletons"] = [x[1] for x in values]
    out["query_path_mid_families"] = [x[2] for x in values]
    out["query_path_broad_families"] = [x[3] for x in values]
    return out, inventory


def candidate_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = add_parsing_status(load_metadata())
    candidates, excluded = primary_candidates(full)
    enriched, inventory = add_query_path_groups(candidates)
    assert enriched["story_group_id"].is_unique, "B0.2b optimizer assumes audited singleton candidate story groups"
    return enriched, excluded, inventory


def split_group_string(value: str) -> list[str]:
    return [part for part in str(value).split("||") if part]


def cell_index(df: pd.DataFrame) -> np.ndarray:
    relation_index = {relation: i for i, relation in enumerate(RELATIONS)}
    return np.asarray([
        (int(hop) - 1) * len(RELATIONS) + relation_index[relation]
        for hop, relation in zip(df["computed_hop"], df["canonical_relation"])
    ], dtype=np.int16)


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([SPLITS, HOPS, RELATIONS], names=["split", "computed_hop", "canonical_relation"])
    return (
        df.groupby(["split", "computed_hop", "canonical_relation"], observed=True)
        .size().reindex(idx, fill_value=0).rename("count").reset_index()
    )


def split_overlap(df: pd.DataFrame, col: str, delimiter: str = "||") -> dict[str, int]:
    sets = {
        split: {
            item for value in df.loc[df["split"].eq(split), col]
            for item in str(value).split(delimiter) if item
        }
        for split in SPLITS
    }
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def manifest_b02b(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    cols = [
        "row_id", "story_group_id", "exact_duplicate_group", "computed_hop",
        "canonical_relation", "source_split", "parsing_status", group_col,
    ]
    return df[cols].sort_values("row_id", kind="mergesort").reset_index(drop=True)

