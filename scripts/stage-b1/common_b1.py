#!/usr/bin/env python3
"""Shared definitions for Stage B1 Llama-3.1-8B relation probing."""

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
OUT = ROOT / "artifacts" / "stage_b1"
RESULT_ALIAS = ROOT / "results" / "stage_b1"
ACT_DIR = ROOT / "activations" / "stage_b1" / "llama31_base" / "stepgame"
MODEL_PATH = ROOT / "models" / "Meta-Llama-3.1-8B"
METADATA_PATH = B01 / "stepgame_metadata.parquet"

SEED = 20260824
N_LAYERS = 32
HIDDEN_SIZE = 4096
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
HOPS = [1, 2, 3, 4, 5]
SPLITS = ["train", "val", "test"]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
MODEL_REVISION = "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea"
TOKENIZER_REVISION = "c52792af35fe806a76530f3028f037d1af4e3490"


DATASETS = {
    "primary": (B02, "primary"),
    "balanced": (B02, "balanced"),
    "composition": (B02, "composition_heldout"),
    "target_absent": (B02, "target_absent"),
    "template": (B02C, "mid_template_test"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object, seed: int = SEED) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_ALIAS.parent.mkdir(parents=True, exist_ok=True)
    if RESULT_ALIAS.exists() or RESULT_ALIAS.is_symlink():
        if RESULT_ALIAS.resolve() != OUT.resolve():
            raise RuntimeError(f"Unexpected existing results/stage_b1 target: {RESULT_ALIAS}")
    else:
        RESULT_ALIAS.symlink_to(Path("../artifacts/stage_b1"), target_is_directory=True)


def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(METADATA_PATH)


def load_manifest(dataset: str, split: str) -> pd.DataFrame:
    directory, prefix = DATASETS[dataset]
    return pd.read_csv(directory / f"{prefix}_{split}.csv")


def story_sentences(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Invalid story JSON: {value!r}")
    return parsed


def build_prompt(story_json: str, question: str) -> str:
    story = "\n".join(story_sentences(story_json))
    return f"Story:\n{story}\n\nQuestion:\n{str(question).strip()}"


def macro_accuracy_baseline(labels: np.ndarray) -> dict[str, float | str | int]:
    from sklearn.metrics import accuracy_score, f1_score

    values, counts = np.unique(labels, return_counts=True)
    max_count = int(counts.max())
    majority = sorted(values[counts == max_count].tolist())[0]
    prediction = np.full(len(labels), majority, dtype=object)
    return {
        "majority_class": str(majority),
        "n_test": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=RELATIONS, average="macro", zero_division=0)),
    }
