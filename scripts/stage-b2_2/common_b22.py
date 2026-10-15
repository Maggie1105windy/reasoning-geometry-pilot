#!/usr/bin/env python3
"""Shared, frozen helpers for Stage B2.2 Qwen3-8B behavior gating."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path(os.environ.get("REASONING_GEOMETRY_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = Path(os.environ.get("STAGE_B22_OUT", ROOT / "artifacts" / "stage_b2_2_qwen3")).resolve()


def _first_existing(*relative_paths: str) -> Path:
    for relative in relative_paths:
        path = ROOT / relative
        if path.exists():
            return path
    return ROOT / relative_paths[0]


B02 = _first_existing("artifacts/stage_b0_2", "results/stage-b0/b0_2")
B02C = _first_existing("artifacts/stage_b0_2c", "results/stage-b0/b0_2c")
B1 = _first_existing("artifacts/stage_b1", "results/stage-b1")
B156 = _first_existing("artifacts/stage_b1_5_1_6", "results/stage-b1_5-b1_6")
B2 = _first_existing("artifacts/stage_b2", "results/stage-b2")
B21 = _first_existing("artifacts/stage_b2_1", "results/stage-b2_1")

SEED = 20260910
BOOTSTRAP_RESAMPLES = 2000
HOPS = [1, 2, 3, 4, 5]
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
CARDINAL = {"ABOVE", "BELOW", "LEFT", "RIGHT"}
DIAGONAL = set(RELATIONS) - CARDINAL
NATURAL_CANDIDATES = {
    "ABOVE": " above",
    "BELOW": " below",
    "LEFT": " left",
    "RIGHT": " right",
    "UPPER_LEFT": " upper-left",
    "UPPER_RIGHT": " upper-right",
    "LOWER_LEFT": " lower-left",
    "LOWER_RIGHT": " lower-right",
}
DISPLAY_WORDS = {relation: candidate.strip() for relation, candidate in NATURAL_CANDIDATES.items()}
PREFERRED_LABELS = dict(zip(RELATIONS, list("ABCDEFGH")))
EXPECTED_ROWS = 2063
EXPECTED_HOP_COUNTS = {1: 712, 2: 505, 3: 422, 4: 282, 5: 142}
EXPECTED_ROW_HASH = "dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950"
EXPECTED_B1_MANIFEST_SHA256 = "83e6db8374b70ca8a86b466626475a174129a633f3b9a29db2d29672a7eaa50c"
EXPECTED_B02_PRIMARY_SHA256 = "daa08915843ae9441c6343a7da47a5d142beb9782f0765c065e0f7b30fbd5049"
MODEL_NAME = "Qwen/Qwen3-8B"

# These definitions are frozen before any Qwen test accuracy is computed.
GATE_THRESHOLDS = {
    "collapse_cardinal_or_diagonal_frequency": 0.95,
    "collapse_single_relation_frequency": 0.80,
    "high_sensitivity_derived_accuracy_range": 0.20,
    "high_sensitivity_weak_protocol_accuracy": 0.15,
    "native_stronger_requires_ci_low_above_zero_for": ["accuracy", "macro_f1", "balanced_accuracy"],
    "strong_gate_hops": [2, 3, 4, 5],
}


def ensure_dirs() -> None:
    directories = [
        OUT / "audit",
        OUT / "smoke",
        OUT / "raw_sum" / "shards",
        OUT / "native_sum" / "shards",
        OUT / "scoring_sensitivity" / "raw_mean",
        OUT / "scoring_sensitivity" / "native_mean",
        OUT / "scoring_sensitivity" / "equal_length_raw" / "shards",
        OUT / "scoring_sensitivity" / "equal_length_native" / "shards",
        OUT / "statistics",
        OUT / "figures",
        OUT / "validation",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def stable_json_hash(payload) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default))


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def row_ids_checksum(values: Iterable[int]) -> str:
    return sha256_text("\n".join(str(int(value)) for value in values) + "\n")


def canonical_value_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = []
    for row in frame[columns].itertuples(index=False, name=None):
        values.append(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default))
    return sha256_text("\n".join(values) + "\n")


def story_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("story is not a JSON list")
    return [str(item) for item in parsed]


def build_raw_prompt(row) -> str:
    return "Story:\n" + "\n".join(story_list(row.story)) + "\n\nQuestion:\n" + str(row.question).strip()


def native_natural_user(row) -> str:
    choices = "\n".join(DISPLAY_WORDS[relation] for relation in RELATIONS)
    return build_raw_prompt(row) + "\n\nAnswer with exactly one of:\n" + choices


def mapping_block(labels: dict[str, str]) -> str:
    return "\n".join(f"{labels[relation]} = {DISPLAY_WORDS[relation]}" for relation in RELATIONS)


def equal_raw_prompt(row, labels: dict[str, str]) -> str:
    return build_raw_prompt(row) + "\n\nAnswer using exactly one label:\n\n" + mapping_block(labels) + "\n\nAnswer:"


def equal_native_user(row, labels: dict[str, str]) -> str:
    return equal_raw_prompt(row, labels)


def load_primary() -> pd.DataFrame:
    path = B1 / "union_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen B1 manifest not found: {path}")
    frame = pd.read_csv(path)
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    frame = frame.loc[frame.primary_split.eq("test")].sort_values("row_id").reset_index(drop=True)
    return frame


def primary_invariants(frame: pd.DataFrame) -> dict:
    counts = {int(key): int(value) for key, value in frame.groupby("computed_hop").size().to_dict().items()}
    row_hash = row_ids_checksum(frame.row_id)
    prompt_exact = frame.apply(lambda row: build_raw_prompt(row) == row.prompt, axis=1)
    checks = {
        "rows_exact": len(frame) == EXPECTED_ROWS,
        "hop_counts_exact": counts == EXPECTED_HOP_COUNTS,
        "row_ids_unique": bool(frame.row_id.is_unique),
        "row_ids_sha256_exact": row_hash == EXPECTED_ROW_HASH,
        "story_group_ids_present": bool(frame.story_group_id.notna().all()),
        "relations_exact": set(frame.canonical_relation.unique()) == set(RELATIONS),
        "computed_hops_exact": set(frame.computed_hop.unique()) == set(HOPS),
        "raw_prompt_exact": bool(prompt_exact.all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(frame),
        "hop_counts": counts,
        "row_ids_sha256": row_hash,
        "mismatched_raw_prompt_rows": frame.loc[~prompt_exact, "row_id"].astype(int).tolist(),
        "content_hashes": {
            "gold_relation": canonical_value_hash(frame, ["row_id", "canonical_relation"]),
            "story_question": canonical_value_hash(frame, ["row_id", "story", "question"]),
            "raw_prompt": canonical_value_hash(frame, ["row_id", "prompt"]),
        },
    }


def deterministic_audit_rows(primary: pd.DataFrame, rows_per_hop: int = 3) -> pd.DataFrame:
    return pd.concat(
        [primary.loc[primary.computed_hop.eq(hop)].head(rows_per_hop) for hop in HOPS],
        ignore_index=True,
    )


def metric_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
    }


def reconstruct_predictions(scores: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {"row_id", "candidate_relation", score_column}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if not np.isfinite(scores[score_column].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in {score_column}")
    order = {relation: index for index, relation in enumerate(RELATIONS)}
    ordered = scores.assign(_candidate_order=scores.candidate_relation.map(order)).sort_values(
        ["row_id", score_column, "_candidate_order"], ascending=[True, False, True]
    )
    chosen = ordered.drop_duplicates("row_id", keep="first")[["row_id", "candidate_relation"]]
    return chosen.rename(columns={"candidate_relation": "prediction"}).sort_values("row_id").reset_index(drop=True)


def attach_truth(predictions: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    truth = primary[["row_id", "story_group_id", "computed_hop", "canonical_relation"]]
    merged = truth.merge(predictions, on="row_id", how="left", validate="one_to_one")
    if merged.prediction.isna().any():
        raise ValueError("Missing predictions after truth join")
    merged["correct"] = merged.prediction.eq(merged.canonical_relation)
    return merged


def metrics_by_hop(predictions: pd.DataFrame, model: str, protocol: str) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        part = predictions.loc[predictions.computed_hop.eq(hop)]
        metrics = metric_dict(part.canonical_relation, part.prediction)
        recalls = recall_score(part.canonical_relation, part.prediction, labels=RELATIONS, average=None, zero_division=0)
        item = {
            "model": model,
            "protocol": protocol,
            "hop": hop,
            "n_test": len(part),
            "n_story_groups": int(part.story_group_id.nunique()),
            "majority_accuracy": float(part.canonical_relation.value_counts(normalize=True).max()),
            "uniform_chance": 0.125,
            **metrics,
        }
        item.update({f"recall_{relation}": float(value) for relation, value in zip(RELATIONS, recalls)})
        rows.append(item)
    return pd.DataFrame(rows)


def confusion_frames(predictions: pd.DataFrame, hop: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = predictions.loc[predictions.computed_hop.eq(hop)]
    matrix = confusion_matrix(part.canonical_relation, part.prediction, labels=RELATIONS)
    raw = pd.DataFrame(matrix, index=RELATIONS, columns=RELATIONS)
    denominator = raw.sum(axis=1).replace(0, np.nan)
    normalized = raw.div(denominator, axis=0).fillna(0.0)
    return raw, normalized


def protocol_payloads(labels: dict[str, str]) -> dict:
    base = {
        "stage": "B2.2",
        "model": MODEL_NAME,
        "frozen_before_qwen_test_accuracy": True,
        "primary_test_rows": EXPECTED_ROWS,
        "row_ids_sha256": EXPECTED_ROW_HASH,
        "relation_order": RELATIONS,
        "bootstrap_unit": "story_group_id",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "uniform_chance": 0.125,
        "derived_metric": "equal-weight arithmetic mean of hop-specific h2,h3,h4,h5 values",
        "gate_thresholds": GATE_THRESHOLDS,
        "no_test_set_tuning": True,
    }
    return {
        "raw_natural": {
            **base,
            "protocols": ["Raw-Sum", "Raw-Mean"],
            "prompt": "exact frozen Story/Question prompt; no system/chat/CoT/few-shot/answer instruction/mapping",
            "bos_policy": "one explicit model config bos_token_id, matching the B2.1 explicit-BOS policy even if tokenizer.bos_token is null",
            "candidate_strings": NATURAL_CANDIDATES,
            "scores": ["sum teacher-forced continuation log probability", "arithmetic mean over candidate tokens"],
        },
        "native_natural": {
            **base,
            "protocols": ["Native-Sum", "Native-Mean"],
            "prompt": "B2.1 native natural user content rendered by tokenizer.apply_chat_template",
            "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
            "candidate_strings": NATURAL_CANDIDATES,
            "scores": ["sum teacher-forced continuation log probability", "arithmetic mean over candidate tokens"],
            "comparison_scope": "own-native-interface task fitness, not identical-token-sequence mechanistic comparison",
        },
        "equal_length": {
            **base,
            "protocols": ["EqualLength-Raw", "EqualLength-Native"],
            "relation_to_label": labels,
            "score": "sum teacher-forced label continuation log probability",
            "interpretation": "measurement-control / sensitivity diagnostic; not a gold-standard performance protocol",
            "label_selection_uses": "tokenizer only; no model probabilities or gold-conditioned accuracy",
        },
    }
