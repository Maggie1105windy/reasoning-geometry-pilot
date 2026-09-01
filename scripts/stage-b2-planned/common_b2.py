#!/usr/bin/env python3
"""Frozen constants and shared helpers for Stage B2."""

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
B156 = ROOT / "artifacts" / "stage_b1_5_1_6"
OUT = ROOT / "artifacts" / "stage_b2"
ACT_ROOT = ROOT / "activations" / "stage_b2"
BASE_ACT = ROOT / "activations" / "stage_b1" / "llama31_base" / "stepgame"

MODELS = {
    "base": {
        "label": "Base",
        "path": ROOT / "models" / "Meta-Llama-3.1-8B",
        "activation_path": BASE_ACT,
        "file_revision": "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea",
        "tokenizer_file_revision": "c52792af35fe806a76530f3028f037d1af4e3490",
    },
    "sft": {
        "label": "SFT",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B-SFT",
        "activation_path": ACT_ROOT / "tulu_sft",
        "file_revision": "aa9a78450ea2c6916824501e276cb97b7caf47e6",
        "tokenizer_file_revision": "aa9a78450ea2c6916824501e276cb97b7caf47e6",
    },
    "dpo": {
        "label": "DPO",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B-DPO",
        "activation_path": ACT_ROOT / "tulu_dpo",
        "file_revision": "2301bf3b5e8733aad822a59b93381369da1692eb",
        "tokenizer_file_revision": "2301bf3b5e8733aad822a59b93381369da1692eb",
    },
    "final": {
        "label": "Final",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B",
        "activation_path": ACT_ROOT / "tulu_final",
        "file_revision": "c5d2996f1e01989a69bd740d8d034edf672cc1e0",
        "tokenizer_file_revision": "191d2c80ef01945c2dee535a263a29e923b6002c",
    },
}

SEED = 20260901
N_LAYERS = 32
HIDDEN_SIZE = 4096
HOPS = [1, 2, 3, 4, 5]
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
REPRESENTATIVE_LAYERS = [0, 7, 15, 23, 31]
LATE_LAYERS = list(range(16, 32))
RANDOM_HOPS = [1, 3, 5]
BOOTSTRAP_RESAMPLES = 1000
STAGE_PAIRS = [
    ("base", "sft"), ("sft", "dpo"), ("dpo", "final"), ("base", "final")
]

CANDIDATES = {
    "ABOVE": " above", "BELOW": " below", "LEFT": " left", "RIGHT": " right",
    "UPPER_LEFT": " upper-left", "UPPER_RIGHT": " upper-right",
    "LOWER_LEFT": " lower-left", "LOWER_RIGHT": " lower-right",
}


def ensure_dirs() -> None:
    paths = [OUT, ACT_ROOT, OUT / "behavior", OUT / "within_hop", OUT / "cross_hop",
             OUT / "figures", OUT / "checkpoints"]
    for control in ["balanced", "target_absent", "composition", "template", "random_label"]:
        paths.append(OUT / "controls" / control)
    for spec in MODELS.values():
        paths.append(spec["activation_path"])
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object, seed: int = SEED) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_union() -> pd.DataFrame:
    frame = pd.read_csv(B1 / "union_manifest.csv")
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    return frame


def metric_dict(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
    }
