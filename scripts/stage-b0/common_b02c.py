#!/usr/bin/env python3
"""Shared definitions for Stage B0.2c test-only template holdout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stage_b0_2b"))
from common_b02b import (  # noqa: E402
    B02, OUT as B02B, PROTECTED_PREFIXES, RELATIONS, SEED, SPLITS,
    candidate_data, distribution, manifest_b02b, protected_hashes,
    sha256, split_group_string, stable_int, write_json,
)

OUT = ROOT / "artifacts" / "stage_b0_2c"
GROUP_COLS = {
    "mid_level_template_family": "query_path_mid_families",
    "surface_skeleton": "query_path_surface_skeletons",
}
MAX_TEST_CELL_SHARE = 0.25
VAL_SHARE_OF_DEVELOPMENT = 0.15
MIN_TEST_CELL_TARGET = 10
MIN_TEST_CELL_GATE = 5


def aggregate_hash(items: dict[str, str]) -> str:
    payload = "".join(f"{name}:{items[name]}\n" for name in sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def broad_hashes() -> dict[str, str]:
    names = [
        "broad_template_heldout_train.csv",
        "broad_template_heldout_val.csv",
        "broad_template_heldout_test.csv",
        "broad_template_heldout_distribution.csv",
        "broad_template_common_support.csv",
    ]
    return {name: sha256(B02B / name) for name in names}


def groups(value: str) -> set[str]:
    return set(split_group_string(value))


def manifest_b02c(df: pd.DataFrame, group_col: str, heldout: set[str]) -> pd.DataFrame:
    frame = manifest_b02b(df, group_col).copy()
    frame["contains_heldout_test_group"] = frame[group_col].map(lambda value: bool(groups(value) & heldout))
    return frame


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

