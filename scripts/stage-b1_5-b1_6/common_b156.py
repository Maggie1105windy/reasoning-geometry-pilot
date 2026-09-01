#!/usr/bin/env python3
"""Shared constants and helpers for Stage B1.5-B1.6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ai/projects/reasoning-geometry-pilot")
B01 = ROOT / "artifacts" / "stage_b0_1"
B02 = ROOT / "artifacts" / "stage_b0_2"
B02C = ROOT / "artifacts" / "stage_b0_2c"
B1 = ROOT / "artifacts" / "stage_b1"
OUT = ROOT / "artifacts" / "stage_b1_5_1_6"
BEHAVIOR = OUT / "behavioral"
CROSS = OUT / "cross_hop"
FIGURES = OUT / "figures"
CHECKPOINTS = CROSS / "checkpoints"
ACT_DIR = ROOT / "activations" / "stage_b1" / "llama31_base" / "stepgame"
MODEL_PATH = ROOT / "models" / "Meta-Llama-3.1-8B"

SEED = 20260825
LAYERS = list(range(32))
HOPS = [1, 2, 3, 4, 5]
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# Frozen before candidate scoring.  The leading space is part of the causal
# continuation after a prompt ending in '?'.  Hyphenation matches StepGame's
# canonical directional verbalization.
CANDIDATES = {
    "ABOVE": " above",
    "BELOW": " below",
    "LEFT": " left",
    "RIGHT": " right",
    "UPPER_LEFT": " upper-left",
    "UPPER_RIGHT": " upper-right",
    "LOWER_LEFT": " lower-left",
    "LOWER_RIGHT": " lower-right",
}


def ensure_dirs() -> None:
    for path in [OUT, BEHAVIOR, BEHAVIOR / "behavior_confusions", CROSS, FIGURES, CHECKPOINTS]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_union() -> pd.DataFrame:
    frame = pd.read_csv(B1 / "union_manifest.csv")
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    return frame


def split_positions(union: pd.DataFrame, split_column: str, split: str, hop: int | None = None) -> np.ndarray:
    mask = union[split_column].eq(split)
    if hop is not None:
        mask &= union["computed_hop"].eq(hop)
    return np.flatnonzero(mask.to_numpy())


def metric_dict(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
    }

