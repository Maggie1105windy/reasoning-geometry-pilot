#!/usr/bin/env python3
"""Shared constants and helpers for Stage B2.1 model-task fitness diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path("/home/ai/chenchen/projects/reasoning-geometry-pilot")
OUT = ROOT / "artifacts" / "stage_b2_1"
B02 = ROOT / "artifacts" / "stage_b0_2"
B02C = ROOT / "artifacts" / "stage_b0_2c"
B1 = ROOT / "artifacts" / "stage_b1"
B156 = ROOT / "artifacts" / "stage_b1_5_1_6"
B2 = ROOT / "artifacts" / "stage_b2"

PYTHON_ENV = "/home/ai/.miniconda3/envs/chenchen_grounding_mechprobe/bin/python"
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
DISPLAY_WORDS = {
    "ABOVE": "above",
    "BELOW": "below",
    "LEFT": "left",
    "RIGHT": "right",
    "UPPER_LEFT": "upper-left",
    "UPPER_RIGHT": "upper-right",
    "LOWER_LEFT": "lower-left",
    "LOWER_RIGHT": "lower-right",
}
LABEL_MAP = dict(zip(RELATIONS, list("ABCDEFGH")))
MODEL_SPECS = {
    "base": {
        "label": "Base",
        "path": ROOT / "models" / "Meta-Llama-3.1-8B",
        "revision": "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea",
    },
    "sft": {
        "label": "SFT",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B-SFT",
        "revision": "aa9a78450ea2c6916824501e276cb97b7caf47e6",
    },
    "dpo": {
        "label": "DPO",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B-DPO",
        "revision": "2301bf3b5e8733aad822a59b93381369da1692eb",
    },
    "final": {
        "label": "Final",
        "path": ROOT / "models" / "Llama-3.1-Tulu-3-8B",
        "revision": "c5d2996f1e01989a69bd740d8d034edf672cc1e0",
    },
}


def ensure_dirs() -> None:
    paths = [
        OUT / "audit",
        OUT / "raw_reproduction" / "confusion_matrices",
        OUT / "native_chat" / "confusion_matrices",
        OUT / "native_chat" / "per_model",
        OUT / "scoring_sensitivity" / "confusion_sum",
        OUT / "scoring_sensitivity" / "confusion_mean",
        OUT / "equal_length_verbalizer" / "confusion_matrices",
        OUT / "equal_length_verbalizer" / "per_model",
        OUT / "statistics",
        OUT / "figures",
        OUT / "validation",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


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


def stable_json_hash(payload) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default))


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def row_ids_checksum(values) -> str:
    ordered = [int(value) for value in values]
    return sha256_text("\n".join(map(str, ordered)) + "\n")


def build_raw_prompt(row) -> str:
    story = row.story if isinstance(row.story, list) else json.loads(row.story)
    return "Story:\n" + "\n".join(story) + "\n\nQuestion:\n" + str(row.question).strip()


def native_natural_user(row) -> str:
    choices = "\n".join(DISPLAY_WORDS[relation] for relation in RELATIONS)
    return build_raw_prompt(row) + "\n\nAnswer with exactly one of:\n" + choices


def mapping_block(labels: dict[str, str] | None = None) -> str:
    labels = labels or LABEL_MAP
    return "\n".join(f"{labels[relation]} = {DISPLAY_WORDS[relation]}" for relation in RELATIONS)


def equal_raw_prompt(row, labels: dict[str, str] | None = None) -> str:
    return (
        build_raw_prompt(row)
        + "\n\nAnswer using exactly one label:\n\n"
        + mapping_block(labels)
        + "\n\nAnswer:"
    )


def equal_native_user(row, labels: dict[str, str] | None = None) -> str:
    return equal_raw_prompt(row, labels)


def load_primary() -> pd.DataFrame:
    frame = pd.read_csv(B1 / "union_manifest.csv")
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    frame = frame.loc[frame.primary_split.eq("test")].sort_values("row_id").reset_index(drop=True)
    return frame


def primary_invariants(frame: pd.DataFrame) -> dict:
    counts = frame.groupby("computed_hop").size().to_dict()
    checks = {
        "rows_2063": len(frame) == 2063,
        "hop_counts_exact": counts == {1: 712, 2: 505, 3: 422, 4: 282, 5: 142},
        "row_ids_unique": frame.row_id.is_unique,
        "story_group_ids_present": frame.story_group_id.notna().all(),
        "relations_exact": set(frame.canonical_relation.unique()) == set(RELATIONS),
        "computed_hops_exact": set(frame.computed_hop.unique()) == set(HOPS),
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "rows": len(frame),
        "hop_counts": {str(k): int(v) for k, v in counts.items()},
        "row_ids_sha256": row_ids_checksum(frame.row_id),
    }


def metric_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def reconstruct_predictions(scores: pd.DataFrame, scoring_column: str) -> pd.DataFrame:
    required = {"row_id", "candidate_relation", scoring_column}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if scores[scoring_column].isna().any() or not np.isfinite(scores[scoring_column]).all():
        raise ValueError(f"Non-finite values in {scoring_column}")
    chosen = scores.loc[
        scores.groupby("row_id", sort=False)[scoring_column].idxmax(),
        ["row_id", "candidate_relation"],
    ].rename(columns={"candidate_relation": "prediction"})
    return chosen.sort_values("row_id").reset_index(drop=True)


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
        majority = float(part.canonical_relation.value_counts(normalize=True).max())
        recalls = recall_score(
            part.canonical_relation,
            part.prediction,
            labels=RELATIONS,
            average=None,
            zero_division=0,
        )
        row = {
            "model": model,
            "protocol": protocol,
            "hop": hop,
            "n_test": len(part),
            "n_story_groups": part.story_group_id.nunique(),
            "majority_accuracy": majority,
            **metrics,
        }
        row.update({f"recall_{relation}": float(value) for relation, value in zip(RELATIONS, recalls)})
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metrics(predictions: pd.DataFrame, model: str, protocol: str, resamples: int = BOOTSTRAP_RESAMPLES) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        part = predictions.loc[predictions.computed_hop.eq(hop)].reset_index(drop=True)
        groups = np.array(sorted(part.story_group_id.unique()))
        rng = np.random.default_rng(SEED + 1000 * hop + int(sha256_text(model + protocol)[:8], 16))
        rel_index = {relation: index for index, relation in enumerate(RELATIONS)}
        group_index = {group: index for index, group in enumerate(groups)}
        group_confusions = np.zeros((len(groups), len(RELATIONS), len(RELATIONS)), dtype=np.int64)
        for item in part.itertuples(index=False):
            group_confusions[group_index[item.story_group_id], rel_index[item.canonical_relation], rel_index[item.prediction]] += 1
        weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=resamples)
        matrices = np.tensordot(weights, group_confusions, axes=(1, 0))
        totals = matrices.sum(axis=(1, 2))
        accuracy_values = np.diagonal(matrices, axis1=1, axis2=2).sum(axis=1) / totals
        true_positive = np.diagonal(matrices, axis1=1, axis2=2)
        false_positive = matrices.sum(axis=1) - true_positive
        false_negative = matrices.sum(axis=2) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator != 0)
        macro_values = per_class_f1.mean(axis=1)
        point = metric_dict(part.canonical_relation, part.prediction)
        rows.append({
            "model": model,
            "protocol": protocol,
            "hop": hop,
            "n_test": len(part),
            "accuracy": point["accuracy"],
            "accuracy_ci95_low": float(np.quantile(accuracy_values, 0.025)),
            "accuracy_ci95_high": float(np.quantile(accuracy_values, 0.975)),
            "macro_f1": point["macro_f1"],
            "macro_f1_ci95_low": float(np.quantile(macro_values, 0.025)),
            "macro_f1_ci95_high": float(np.quantile(macro_values, 0.975)),
            "bootstrap_unit": "story_group_id",
            "bootstrap_resamples": resamples,
            "seed": SEED,
        })
    return pd.DataFrame(rows)


def write_confusions(predictions: pd.DataFrame, directory: Path, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for hop in HOPS:
        part = predictions.loc[predictions.computed_hop.eq(hop)]
        matrix = confusion_matrix(part.canonical_relation, part.prediction, labels=RELATIONS)
        pd.DataFrame(matrix, index=RELATIONS, columns=RELATIONS).to_csv(directory / f"{prefix}_h{hop}.csv")


def candidate_suffix_from_full(tok, prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant-content rendering does not preserve generation-boundary prefix")
    suffix = list(full_ids[len(prompt_ids) :])
    while suffix and suffix[-1] in set(tok.all_special_ids):
        suffix.pop()
    if not suffix:
        raise ValueError("Empty candidate suffix")
    return suffix
