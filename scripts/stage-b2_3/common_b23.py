#!/usr/bin/env python3
"""Frozen definitions and shared helpers for Stage B2.3."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path(os.environ.get("REASONING_GEOMETRY_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = Path(os.environ.get("STAGE_B23_OUT", ROOT / "artifacts" / "stage_b2_3_qwen3_generation")).resolve()
B22 = ROOT / "artifacts" / "stage_b2_2_qwen3"


def _first_existing(*relative_paths: str) -> Path:
    for relative in relative_paths:
        path = ROOT / relative
        if path.exists():
            return path
    return ROOT / relative_paths[0]


B02 = _first_existing("artifacts/stage_b0_2", "results/stage-b0/b0_2")
B1 = _first_existing("artifacts/stage_b1", "results/stage-b1")

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_PATH = ROOT / "models" / "Qwen3-8B"
HOPS = [1, 2, 3, 4, 5]
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
RELATION_WORDS = {
    "ABOVE": "above",
    "BELOW": "below",
    "LEFT": "left",
    "RIGHT": "right",
    "UPPER_LEFT": "upper-left",
    "UPPER_RIGHT": "upper-right",
    "LOWER_LEFT": "lower-left",
    "LOWER_RIGHT": "lower-right",
}
WORD_TO_RELATION = {value: key for key, value in RELATION_WORDS.items()}
CARDINAL = {"ABOVE", "BELOW", "LEFT", "RIGHT"}
DIAGONAL = set(RELATIONS) - CARDINAL

EXPECTED_ROWS = 2063
EXPECTED_HOP_COUNTS = {1: 712, 2: 505, 3: 422, 4: 282, 5: 142}
EXPECTED_ROW_HASH = "dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950"
EXPECTED_B1_MANIFEST_SHA256 = "83e6db8374b70ca8a86b466626475a174129a633f3b9a29db2d29672a7eaa50c"
EXPECTED_B02_PRIMARY_SHA256 = "daa08915843ae9441c6343a7da47a5d142beb9782f0765c065e0f7b30fbd5049"
EXPECTED_B22_NATIVE_SUM_DERIVED_ACCURACY = 0.1853
EXPECTED_B22_VALIDATION = "PASS"

PILOT_SAMPLE_SEED = 20260924
GENERATION_SEEDS = [20260924, 20260925, 20260926]
PRIMARY_GENERATION_SEED = 20260924
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 2000

OUTPUT_FORMAT_INSTRUCTION = """End your response with exactly one final line in the following format:

Final answer: <relation>

where <relation> must be one of:
above, below, left, right, upper-left, upper-right, lower-left, lower-right."""

PARSER_SPEC = {
    "version": "stage_b2_3_v1_frozen_before_generation",
    "case_sensitive": False,
    "selection": "last valid line matching Final answer: <relation>",
    "allowed_labels": list(RELATION_WORDS.values()),
    "normalization": "case, surrounding whitespace, and separators space/_/- for diagonal labels only",
    "semantic_synonyms": {},
    "invalid_policy": "INVALID counts as wrong in primary metrics",
}

OFFICIAL_CONFIGS = {
    "fg_nonthinking": {
        "protocol": "FG-N",
        "enable_thinking": False,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "fg_thinking": {
        "protocol": "FG-T",
        "enable_thinking": True,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
}
MATCHED_CONFIGS = {
    "fg_nonthinking_matched": {
        "protocol": "FG-N-Matched",
        "enable_thinking": False,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
    "fg_thinking_matched": {
        "protocol": "FG-T-Matched",
        "enable_thinking": True,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
}

DECISION_THRESHOLDS = {
    "prediction_collapse_cardinal_or_diagonal_frequency": 0.95,
    "prediction_collapse_single_relation_frequency": 0.80,
    "strong_derived_accuracy": 0.35,
    "strong_derived_macro_f1": 0.30,
    "strong_derived_balanced_accuracy": 0.30,
    "strong_parse_success": 0.98,
    "meaningful_relation_recall": 0.05,
    "minimum_meaningful_relations": 5,
    "weak_derived_accuracy": 0.30,
    "weak_derived_macro_f1": 0.20,
    "weak_derived_balanced_accuracy": 0.20,
    "uninterpretable_parse_success": 0.95,
    "underestimation_min_accuracy_gain": 0.10,
    "underestimation_no_effect_margin": 0.03,
    "pilot_stochasticity_low_max_range": 0.03,
    "pilot_stochasticity_moderate_max_range": 0.07,
    "smoke_truncation_switch_rate": 0.20,
}


def ensure_dirs() -> None:
    directories = [
        OUT / "audit",
        OUT / "pilot" / "fg_nonthinking",
        OUT / "pilot" / "fg_thinking",
        OUT / "pilot" / "matched_decoding",
        OUT / "full" / "fg_nonthinking",
        OUT / "full" / "fg_thinking",
        OUT / "comparisons",
        OUT / "statistics",
        OUT / "figures",
        OUT / "validation",
        OUT / "logs",
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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    value = "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows)
    write_text_atomic(path, value)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def row_ids_checksum(values: Iterable[int]) -> str:
    return sha256_text("\n".join(str(int(value)) for value in values) + "\n")


def canonical_value_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        for row in frame[columns].itertuples(index=False, name=None)
    ]
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


def generation_user_content(row) -> str:
    original = build_raw_prompt(row)
    if hasattr(row, "prompt") and str(row.prompt) != original:
        raise ValueError(f"Frozen prompt mismatch for row {row.row_id}")
    return original + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


def load_primary() -> pd.DataFrame:
    path = B1 / "union_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen B1 manifest not found: {path}")
    frame = pd.read_csv(path)
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    return frame.loc[frame.primary_split.eq("test")].sort_values("row_id").reset_index(drop=True)


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


def deterministic_pilot_subset(primary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PILOT_SAMPLE_SEED)
    parts = []
    for hop in HOPS:
        part = primary.loc[primary.computed_hop.eq(hop)].sort_values("row_id")
        positions = np.sort(rng.choice(len(part), size=50, replace=False))
        parts.append(part.iloc[positions])
    return pd.concat(parts, ignore_index=True).sort_values(["computed_hop", "row_id"]).reset_index(drop=True)


def smoke_subset(primary: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [primary.loc[primary.computed_hop.eq(hop)].sort_values("row_id").head(2) for hop in HOPS],
        ignore_index=True,
    )


_FINAL_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*Final[ \t]+answer[ \t]*:[ \t]*"
    r"(above|below|left|right|upper(?:[ _-]+)left|upper(?:[ _-]+)right|"
    r"lower(?:[ _-]+)left|lower(?:[ _-]+)right)[ \t]*$"
)


def normalize_answer_word(value: str) -> str:
    normalized = re.sub(r"[ _-]+", "-", value.strip().lower())
    return normalized


def parse_final_answer(text: str) -> dict:
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return {
            "parsed_relation": "INVALID",
            "parsed_word": None,
            "parse_status": "INVALID_NO_VALID_FINAL_ANSWER_LINE",
            "n_valid_final_answer_lines": 0,
        }
    parsed_word = normalize_answer_word(matches[-1].group(1))
    relation = WORD_TO_RELATION.get(parsed_word, "INVALID")
    return {
        "parsed_relation": relation,
        "parsed_word": parsed_word if relation != "INVALID" else None,
        "parse_status": "PARSED_LAST_VALID_FINAL_ANSWER_LINE" if relation != "INVALID" else "INVALID_LABEL",
        "n_valid_final_answer_lines": len(matches),
    }


def child_seed(generation_seed: int, row_id: int) -> int:
    payload = f"{int(generation_seed)}:{int(row_id)}"
    return int(sha256_text(payload)[:8], 16) % (2**31 - 1)


def metric_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=RELATIONS, average="macro", zero_division=0)),
    }


def metrics_for_frame(frame: pd.DataFrame) -> dict:
    result = metric_dict(frame.gold_relation, frame.parsed_relation)
    recalls = recall_score(
        frame.gold_relation, frame.parsed_relation, labels=RELATIONS, average=None, zero_division=0
    )
    result.update({f"recall_{relation}": float(value) for relation, value in zip(RELATIONS, recalls)})
    result["n"] = int(len(frame))
    result["parse_success_rate"] = float(frame.parsed_relation.ne("INVALID").mean())
    prediction_frequency = {
        relation: float(frame.parsed_relation.eq(relation).mean()) for relation in RELATIONS
    }
    result.update({f"prediction_frequency_{relation}": value for relation, value in prediction_frequency.items()})
    result["invalid_frequency"] = float(frame.parsed_relation.eq("INVALID").mean())
    result["cardinal_prediction_frequency"] = float(frame.parsed_relation.isin(CARDINAL).mean())
    result["diagonal_prediction_frequency"] = float(frame.parsed_relation.isin(DIAGONAL).mean())
    result["max_single_relation_frequency"] = max(prediction_frequency.values())
    result["dominant_relation"] = max(prediction_frequency, key=prediction_frequency.get)
    return result


def prediction_collapse(metrics: dict) -> bool:
    return bool(
        metrics["cardinal_prediction_frequency"] >= DECISION_THRESHOLDS["prediction_collapse_cardinal_or_diagonal_frequency"]
        or metrics["diagonal_prediction_frequency"] >= DECISION_THRESHOLDS["prediction_collapse_cardinal_or_diagonal_frequency"]
        or metrics["max_single_relation_frequency"] >= DECISION_THRESHOLDS["prediction_collapse_single_relation_frequency"]
    )


def confusion_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = confusion_matrix(frame.gold_relation, frame.parsed_relation, labels=RELATIONS)
    raw = pd.DataFrame(matrix, index=RELATIONS, columns=RELATIONS)
    normalized = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return raw, normalized


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    subset = frame.loc[:, columns].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: f"{value:.{decimals}f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])

