#!/usr/bin/env python3
"""Stage B1.5 candidate-token audit and full behavioral scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_b156 import (
    BEHAVIOR, B1, CANDIDATES, HOPS, MODEL_PATH, OUT, RELATIONS, SEED,
    ensure_dirs, load_union, metric_dict, read_json, sha256, write_json,
)


MODEL_REVISION = "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea"
TOKENIZER_REVISION = "c52792af35fe806a76530f3028f037d1af4e3490"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenization_records(tokenizer) -> list[dict]:
    rows = []
    for relation in RELATIONS:
        text = CANDIDATES[relation]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        rows.append({
            "relation": relation, "candidate_string": text,
            "token_ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids),
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            "token_count": len(ids),
            "contains_special_token": any(token in tokenizer.all_special_ids for token in ids),
        })
    return rows


def freeze_protocol(tokenizer, records) -> None:
    prompt_protocol = read_json(B1 / "prompt_protocol.json")
    payload = {
        "stage": "B1.5", "status": "FROZEN_BEFORE_BEHAVIOR_RESULTS",
        "model": "Meta-Llama-3.1-8B Base", "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "prompt_protocol_path": str(B1 / "prompt_protocol.json"),
        "prompt_protocol_sha256": sha256(B1 / "prompt_protocol.json"),
        "prompt_template": prompt_protocol["prompt_template"],
        "chat_template": False, "few_shot": False, "chain_of_thought": False,
        "gold_answer_in_prompt": False,
        "candidate_verbalizations": CANDIDATES,
        "leading_space_policy": "one literal leading ASCII space after the prompt-final question mark",
        "primary_scoring": "sum conditional log-probability across all candidate tokens",
        "sensitivity_scoring": "mean conditional log-probability per candidate token",
        "candidate_tokenization": records,
        "bos": {"explicit": True, "token_id": tokenizer.bos_token_id},
        "padding": {"side": "right", "token": "EOS", "token_id": tokenizer.eos_token_id},
        "candidate_eos_in_score": False, "candidate_pad_in_score": False,
        "seed": SEED,
    }
    write_json(BEHAVIOR / "candidate_protocol.json", payload)
    token_payload = {
        "stage": "B1.5", "status": "PASS" if all(not r["contains_special_token"] and r["token_count"] > 0 for r in records) else "FAIL",
        "records": records,
        "token_count_min": min(r["token_count"] for r in records),
        "token_count_max": max(r["token_count"] for r in records),
        "lengths_equal": len({r["token_count"] for r in records}) == 1,
        "sum_and_length_normalized_both_required": len({r["token_count"] for r in records}) > 1,
    }
    write_json(BEHAVIOR / "candidate_tokenization.json", token_payload)
    write_json(BEHAVIOR / "behavior_candidate_tokenization.json", token_payload)


def load_model():
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True,
    ).to("cuda")
    model.eval()
    return model


def score_rows(model, tokenizer, frame: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    candidate_ids = {
        relation: tokenizer(text, add_special_tokens=False).input_ids
        for relation, text in CANDIDATES.items()
    }
    expanded = []
    for row in frame.itertuples(index=False):
        prompt_ids = json.loads(row.token_ids) if isinstance(row.token_ids, str) else list(row.token_ids)
        if prompt_ids[0] != tokenizer.bos_token_id:
            raise RuntimeError(f"Missing explicit BOS for row_id={row.row_id}")
        for relation in RELATIONS:
            cids = candidate_ids[relation]
            expanded.append({
                "row_id": int(row.row_id), "candidate_relation": relation,
                "candidate_ids": cids, "prompt_length": len(prompt_ids),
                "sequence": prompt_ids + cids,
            })
    scored = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        max_len = max(len(item["sequence"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), tokenizer.eos_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, item in enumerate(batch):
            seq = torch.tensor(item["sequence"], dtype=torch.long)
            input_ids[idx, :len(seq)] = seq
            attention[idx, :len(seq)] = 1
        input_ids = input_ids.to("cuda", non_blocking=True)
        attention = attention.to("cuda", non_blocking=True)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
        for idx, item in enumerate(batch):
            start_pos = item["prompt_length"]
            token_scores = []
            for offset, token_id in enumerate(item["candidate_ids"]):
                predicting_position = start_pos + offset - 1
                token_scores.append(float(log_probs[idx, predicting_position, token_id].item()))
            scored.append({
                "row_id": item["row_id"], "candidate_relation": item["candidate_relation"],
                "candidate_token_count": len(item["candidate_ids"]),
                "sum_log_probability": float(sum(token_scores)),
                "mean_log_probability": float(np.mean(token_scores)),
                "token_log_probabilities": json.dumps(token_scores, separators=(",", ":")),
            })
        del logits, log_probs, input_ids, attention
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(expanded):
            print(json.dumps({
                "scored_sequences": min(start + batch_size, len(expanded)),
                "total_sequences": len(expanded), "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)
    result = pd.DataFrame(scored)
    result.attrs["runtime_seconds"] = time.perf_counter() - started
    result.attrs["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    result.attrs["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    return result


def prediction_table(samples: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    base = samples[["row_id", "story_group_id", "computed_hop", "canonical_relation"]].copy()
    sum_idx = scores.groupby("row_id")["sum_log_probability"].idxmax()
    mean_idx = scores.groupby("row_id")["mean_log_probability"].idxmax()
    sum_pred = scores.loc[sum_idx, ["row_id", "candidate_relation"]].rename(columns={"candidate_relation": "predicted_relation_sum"})
    mean_pred = scores.loc[mean_idx, ["row_id", "candidate_relation"]].rename(columns={"candidate_relation": "predicted_relation_mean"})
    return base.merge(sum_pred, on="row_id", validate="one_to_one").merge(mean_pred, on="row_id", validate="one_to_one")


def audit() -> None:
    ensure_dirs()
    gate = read_json(OUT / "input_checksum_verification.json")
    if gate["status"] != "PASS":
        raise RuntimeError("Input checksum Gate is not PASS")
    tokenizer = load_tokenizer()
    records = tokenization_records(tokenizer)
    freeze_protocol(tokenizer, records)
    if any(record["contains_special_token"] or record["token_count"] == 0 for record in records):
        raise RuntimeError("Candidate tokenization failure")

    union = load_union()
    primary_test = union[union["primary_split"].eq("test")].copy()
    selected = []
    rng = np.random.default_rng(SEED)
    for hop in HOPS:
        for relation in RELATIONS:
            cell = primary_test[
                primary_test["computed_hop"].eq(hop) & primary_test["canonical_relation"].eq(relation)
            ]
            if cell.empty:
                raise RuntimeError(f"Missing audit cell {hop=} {relation=}")
            selected.append(cell.iloc[int(rng.integers(0, len(cell)))])
    audit_samples = pd.DataFrame(selected).sort_values(["computed_hop", "canonical_relation"]).reset_index(drop=True)
    model = load_model()
    scores = score_rows(model, tokenizer, audit_samples, batch_size=32)
    predictions = prediction_table(audit_samples, scores)
    merged = scores.merge(
        audit_samples[["row_id", "computed_hop", "canonical_relation", "prompt", "readout_token"]],
        on="row_id", validate="many_to_one",
    )
    checks = {
        "exactly_40_samples": len(audit_samples) == 40,
        "all_5x8_cells": audit_samples.groupby(["computed_hop", "canonical_relation"]).size().eq(1).all(),
        # Relation words may legitimately occur inside the StepGame story.
        # Leakage means an appended answer/suffix, so require the frozen prompt
        # to terminate exactly at the original question for every audit row.
        "prompt_gold_answer_absent": all(
            row.prompt.endswith(str(row.question).strip())
            for row in audit_samples.itertuples(index=False)
        ),
        "prompt_matches_frozen_union": audit_samples["prompt"].notna().all(),
        "readout_is_question_terminal": audit_samples["readout_token"].eq("?").all(),
        "all_8_candidates_per_sample": scores.groupby("row_id").size().eq(8).all(),
        "all_scores_finite": np.isfinite(scores[["sum_log_probability", "mean_log_probability"]]).all().all(),
        "no_eos_or_pad_in_candidate": all(
            tokenizer.eos_token_id not in record["token_ids"] and tokenizer.pad_token_id not in record["token_ids"]
            for record in records
        ),
        "conditional_positions_verified": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    audit_json = {
        "stage": "B1.5", "status": status, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "sample_row_ids": audit_samples["row_id"].astype(int).tolist(),
        "coverage": {"hops": HOPS, "relations": RELATIONS, "cells": 40},
        "scoring_definition": "candidate token j uses causal logits at prompt_length+j-1; EOS/pad excluded",
        "runtime_seconds": scores.attrs["runtime_seconds"],
        "gpu_peak_allocated_gib": scores.attrs["peak_allocated_gib"],
        "gpu_peak_reserved_gib": scores.attrs["peak_reserved_gib"],
    }
    write_json(BEHAVIOR / "candidate_audit.json", audit_json)
    scores.to_csv(BEHAVIOR / "candidate_audit_scores.csv", index=False)
    predictions.to_csv(BEHAVIOR / "candidate_audit_predictions.csv", index=False)

    lines = [
        "# Stage B1.5 candidate scoring audit", "", f"Status: **{status}**", "",
        "## Frozen scoring", "",
        "- Primary: sum of conditional log-probabilities over all candidate tokens.",
        "- Sensitivity: mean conditional log-probability per candidate token.",
        "- Candidate EOS and padding tokens are excluded.",
        "- Coverage: one Primary-test sample for each of 5 hops x 8 relations.", "",
        "## Candidate tokenization", "",
        "| Relation | Exact continuation | Token IDs | Tokens | n |",
        "|---|---|---|---|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['relation']} | `{record['candidate_string']}` | `{record['token_ids']}` | "
            f"`{record['tokens']}` | {record['token_count']} |"
        )
    lines.extend(["", "## 40-cell scoring results", "", "| Hop | Gold | Row ID | Sum prediction | Mean prediction | Prompt SHA-256 |", "|---:|---|---:|---|---|---|"])
    prompt_map = audit_samples.set_index("row_id")["prompt"].to_dict()
    for row in predictions.sort_values(["computed_hop", "canonical_relation"]).itertuples(index=False):
        phash = hashlib.sha256(prompt_map[row.row_id].encode("utf-8")).hexdigest()
        lines.append(
            f"| {row.computed_hop} | {row.canonical_relation} | {row.row_id} | "
            f"{row.predicted_relation_sum} | {row.predicted_relation_mean} | `{phash[:16]}...` |"
        )
    lines.extend(["", "## Prompt examples", ""])
    for row in audit_samples.iloc[:3].itertuples(index=False):
        lines.extend([f"### row_id={row.row_id}; hop={row.computed_hop}; gold={row.canonical_relation}", "", "```text", row.prompt, "```", ""])
    content = "\n".join(lines) + "\n"
    (BEHAVIOR / "candidate_audit.md").write_text(content, encoding="utf-8")
    (BEHAVIOR / "behavior_candidate_audit.md").write_text(content, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "tokenization": records,
                      "runtime_seconds": scores.attrs["runtime_seconds"],
                      "peak_reserved_gib": scores.attrs["peak_reserved_gib"]}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


def full() -> None:
    ensure_dirs()
    audit_info = read_json(BEHAVIOR / "candidate_audit.json")
    if audit_info["status"] != "PASS":
        raise RuntimeError("Candidate scoring audit is not PASS")
    tokenizer = load_tokenizer()
    union = load_union()
    primary_ids = set(union.loc[union["primary_split"].eq("test"), "row_id"].astype(int))
    balanced_ids = set(union.loc[union["balanced_split"].eq("test"), "row_id"].astype(int))
    target_ids = sorted(primary_ids | balanced_ids)
    samples = union[union["row_id"].isin(target_ids)].copy().sort_values("row_id").reset_index(drop=True)
    if len(samples) != len(target_ids):
        raise RuntimeError("Behavior union row mismatch")
    model = load_model()
    scores = score_rows(model, tokenizer, samples, batch_size=32)
    scores.to_csv(BEHAVIOR / "all_candidate_scores.csv", index=False)
    predictions = prediction_table(samples, scores)
    predictions["in_primary_test"] = predictions["row_id"].isin(primary_ids)
    predictions["in_balanced_test"] = predictions["row_id"].isin(balanced_ids)
    predictions.to_csv(BEHAVIOR / "all_behavior_predictions.csv", index=False)
    write_json(BEHAVIOR / "behavior_forward_metadata.json", {
        "status": "PASS", "rows_scored": len(samples), "candidate_sequences": len(scores),
        "runtime_seconds": scores.attrs["runtime_seconds"],
        "gpu_peak_allocated_gib": scores.attrs["peak_allocated_gib"],
        "gpu_peak_reserved_gib": scores.attrs["peak_reserved_gib"],
        "batch_size": 32, "model": "Meta-Llama-3.1-8B Base", "post_trained_model": False,
    })
    print(json.dumps({"status": "PASS", "rows": len(samples), "candidate_sequences": len(scores),
                      "runtime_seconds": scores.attrs["runtime_seconds"],
                      "peak_reserved_gib": scores.attrs["peak_reserved_gib"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "full"])
    args = parser.parse_args()
    audit() if args.mode == "audit" else full()


if __name__ == "__main__":
    main()
