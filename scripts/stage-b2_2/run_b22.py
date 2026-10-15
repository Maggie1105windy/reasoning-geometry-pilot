#!/usr/bin/env python3
"""Resumable Qwen3-8B teacher-forced scoring for Stage B2.2."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common_b22 import (
    HOPS, NATURAL_CANDIDATES, OUT, RELATIONS, build_raw_prompt,
    deterministic_audit_rows, ensure_dirs, equal_native_user, equal_raw_prompt,
    load_primary, native_natural_user, read_json, reconstruct_predictions,
    sha256_file, stable_json_hash, write_csv_atomic, write_json_atomic,
)


COMPONENTS = ["raw_natural", "native_natural", "equal_raw", "equal_native"]
OUTPUTS = {
    "raw_natural": OUT / "raw_sum",
    "native_natural": OUT / "native_sum",
    "equal_raw": OUT / "scoring_sensitivity" / "equal_length_raw",
    "equal_native": OUT / "scoring_sensitivity" / "equal_length_native",
}


def native_render(tok, user: str) -> tuple[str, list[int]]:
    kwargs = {
        "conversation": [{"role": "user", "content": user}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    text = tok.apply_chat_template(tokenize=False, **kwargs)
    ids = list(tok.apply_chat_template(tokenize=True, **kwargs))
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if any(content.strip() for content in think_blocks):
        raise RuntimeError("enable_thinking=False unexpectedly rendered non-empty thinking content")
    if list(tok(text, add_special_tokens=False).input_ids) != ids:
        raise RuntimeError("Native rendered-text/token boundary changed after audit")
    return text, ids


def contextual_suffix(tok, prompt_text: str, prompt_ids: list[int], candidate: str, explicit_bos: bool, raw_bos_id: int) -> list[int]:
    prefix = ([raw_bos_id] if explicit_bos else []) + list(tok(prompt_text, add_special_tokens=False).input_ids)
    if prompt_ids != prefix:
        raise RuntimeError("Prompt IDs violate frozen boundary policy")
    joint = ([raw_bos_id] if explicit_bos else []) + list(tok(prompt_text + candidate, add_special_tokens=False).input_ids)
    if joint[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Candidate changes tokenization before the frozen boundary")
    suffix = joint[len(prompt_ids):]
    if not suffix:
        raise RuntimeError("Empty candidate continuation")
    return suffix


def prepare_one(tok, row, component: str, labels: dict[str, str], raw_bos_id: int) -> dict:
    if component == "raw_natural":
        prompt_text = build_raw_prompt(row)
        prompt_ids = [raw_bos_id] + list(tok(prompt_text, add_special_tokens=False).input_ids)
        candidates = {
            relation: contextual_suffix(tok, prompt_text, prompt_ids, NATURAL_CANDIDATES[relation], True, raw_bos_id)
            for relation in RELATIONS
        }
    elif component == "native_natural":
        prompt_text, prompt_ids = native_render(tok, native_natural_user(row))
        candidates = {
            relation: contextual_suffix(tok, prompt_text, prompt_ids, NATURAL_CANDIDATES[relation], False, raw_bos_id)
            for relation in RELATIONS
        }
    elif component == "equal_raw":
        prompt_text = equal_raw_prompt(row, labels)
        prompt_ids = [raw_bos_id] + list(tok(prompt_text, add_special_tokens=False).input_ids)
        candidates = {
            relation: contextual_suffix(tok, prompt_text, prompt_ids, " " + labels[relation], True, raw_bos_id)
            for relation in RELATIONS
        }
    elif component == "equal_native":
        prompt_text, prompt_ids = native_render(tok, equal_native_user(row, labels))
        candidates = {
            relation: contextual_suffix(tok, prompt_text, prompt_ids, " " + labels[relation], False, raw_bos_id)
            for relation in RELATIONS
        }
    else:
        raise ValueError(component)
    return {
        "row_id": int(row.row_id),
        "story_group_id": row.story_group_id,
        "hop": int(row.computed_hop),
        "gold_relation": row.canonical_relation,
        "prompt_ids": prompt_ids,
        "prompt_token_length": len(prompt_ids),
        "candidate_ids": candidates,
    }


def _candidate_log_prob(logits: torch.Tensor, token_id: int) -> float:
    values = logits.float()
    return float((values[token_id] - torch.logsumexp(values, dim=-1)).item())


def score_expanded(model, tok, prepared: list[dict], batch_size: int) -> pd.DataFrame:
    expanded = []
    for item in prepared:
        for relation in RELATIONS:
            candidate_ids = item["candidate_ids"][relation]
            expanded.append((item, relation, item["prompt_ids"] + candidate_ids, candidate_ids))
    output = []
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        maximum = max(len(entry[2]) for entry in batch)
        input_ids = torch.full((len(batch), maximum), tok.pad_token_id, dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        for index, (_, _, sequence, _) in enumerate(batch):
            input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
            attention[index, :len(sequence)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        for index, (item, relation, _, candidate_ids) in enumerate(batch):
            prompt_length = item["prompt_token_length"]
            token_values = [
                _candidate_log_prob(logits[index, prompt_length + offset - 1], token_id)
                for offset, token_id in enumerate(candidate_ids)
            ]
            output.append({
                "row_id": item["row_id"], "story_group_id": item["story_group_id"],
                "hop": item["hop"], "gold_relation": item["gold_relation"],
                "candidate_relation": relation,
                "candidate_token_ids": json.dumps(candidate_ids, separators=(",", ":")),
                "candidate_token_count": len(candidate_ids),
                "prompt_token_length": prompt_length,
                "sum_log_probability": float(sum(token_values)),
                "mean_log_probability": float(np.mean(token_values)),
                "token_log_probabilities": json.dumps(token_values, separators=(",", ":")),
            })
        del input_ids, attention, logits
    return pd.DataFrame(output)


def score_single_token(model, tok, prepared: list[dict], batch_size: int) -> pd.DataFrame:
    if not all(len(ids) == 1 for item in prepared for ids in item["candidate_ids"].values()):
        raise RuntimeError("Equal-length shared-forward optimization received a non-single-token candidate")
    output = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        maximum = max(len(item["prompt_ids"]) for item in batch)
        input_ids = torch.full((len(batch), maximum), tok.pad_token_id, dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        for index, item in enumerate(batch):
            sequence = item["prompt_ids"]
            input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
            attention[index, :len(sequence)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        for index, item in enumerate(batch):
            final_logits = logits[index, item["prompt_token_length"] - 1]
            for relation in RELATIONS:
                candidate_ids = item["candidate_ids"][relation]
                value = _candidate_log_prob(final_logits, candidate_ids[0])
                output.append({
                    "row_id": item["row_id"], "story_group_id": item["story_group_id"],
                    "hop": item["hop"], "gold_relation": item["gold_relation"],
                    "candidate_relation": relation,
                    "candidate_token_ids": json.dumps(candidate_ids, separators=(",", ":")),
                    "candidate_token_count": 1, "prompt_token_length": item["prompt_token_length"],
                    "sum_log_probability": value, "mean_log_probability": value,
                    "token_log_probabilities": json.dumps([value], separators=(",", ":")),
                })
        del input_ids, attention, logits
    return pd.DataFrame(output)


def score_with_backoff(model, tok, prepared: list[dict], requested_batch_size: int, single_token: bool) -> tuple[pd.DataFrame, int, list[dict]]:
    batch_size = requested_batch_size
    failures = []
    while True:
        try:
            scorer = score_single_token if single_token else score_expanded
            return scorer(model, tok, prepared, batch_size), batch_size, failures
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"batch_size": batch_size, "error": str(exc)})
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise RuntimeError("CUDA OOM at batch size 1; quantization is prohibited without a new decision") from exc
            batch_size = max(1, batch_size // 2)


def validate_score_frame(frame: pd.DataFrame, expected_ids: list[int]) -> None:
    required = {
        "row_id", "story_group_id", "hop", "gold_relation", "candidate_relation",
        "candidate_token_count", "prompt_token_length", "sum_log_probability", "mean_log_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Score shard missing columns: {sorted(missing)}")
    if sorted(frame.row_id.astype(int).unique().tolist()) != sorted(expected_ids):
        raise RuntimeError("Score shard row IDs differ from the expected frozen shard")
    if not frame.groupby("row_id").size().eq(8).all():
        raise RuntimeError("Score shard does not have exactly 8 candidates per row")
    relation_sets = frame.groupby("row_id").candidate_relation.apply(set)
    if not all(value == set(RELATIONS) for value in relation_sets):
        raise RuntimeError("Score shard candidate relation set mismatch")
    if not np.isfinite(frame[["sum_log_probability", "mean_log_probability"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Score shard contains NaN/Inf")
    if (frame.prompt_token_length <= 0).any() or (frame.candidate_token_count <= 0).any():
        raise RuntimeError("Score shard contains empty/truncated prompt or candidate")


def shard_ranges(frame: pd.DataFrame, shard_rows: int):
    for start in range(0, len(frame), shard_rows):
        yield start // shard_rows, frame.iloc[start:start + shard_rows]


def output_directory(component: str, mode: str) -> Path:
    return (OUT / "smoke" / component) if mode == "smoke" else OUTPUTS[component]


def run_component(
    model, tok, config, primary: pd.DataFrame, labels: dict[str, str], raw_bos_id: int,
    component: str, mode: str, batch_size: int, shard_rows: int,
    protocol_sha256: str, model_identity_sha256: str,
) -> dict:
    target = output_directory(component, mode)
    shard_dir = target / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "run_manifest.json"
    manifest = {
        "status": "RUNNING", "component": component, "mode": mode,
        "rows_expected": len(primary), "row_ids": primary.row_id.astype(int).tolist(),
        "batch_size_requested": batch_size, "shard_rows": shard_rows,
        "protocol_sha256": protocol_sha256, "model_identity_sha256": model_identity_sha256,
        "enable_thinking": False if "native" in component else None,
        "no_truncation": True, "shards": {}, "oom_backoffs": [],
    }
    if manifest_path.is_file():
        old = read_json(manifest_path)
        if (
            old.get("row_ids") != manifest["row_ids"]
            or old.get("component") != component
            or old.get("shard_rows") != shard_rows
            or old.get("protocol_sha256") != protocol_sha256
            or old.get("model_identity_sha256") != model_identity_sha256
        ):
            raise RuntimeError(f"Existing manifest conflicts with the frozen run: {manifest_path}")
        manifest["shards"] = old.get("shards", {})
        manifest["oom_backoffs"] = old.get("oom_backoffs", [])

    max_position = getattr(config, "max_position_embeddings", None)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for shard_index, shard_frame in shard_ranges(primary, shard_rows):
        shard_name = f"shard_{shard_index:04d}.csv"
        shard_path = shard_dir / shard_name
        expected_ids = shard_frame.row_id.astype(int).tolist()
        if shard_path.is_file():
            existing = pd.read_csv(shard_path)
            validate_score_frame(existing, expected_ids)
            checksum = sha256_file(shard_path)
            recorded = manifest["shards"].get(shard_name, {}).get("sha256")
            if recorded and recorded != checksum:
                raise RuntimeError(f"Existing shard checksum mismatch: {shard_path}")
            manifest["shards"][shard_name] = {"status": "PASS", "rows": len(expected_ids), "row_ids": expected_ids, "sha256": checksum, "resumed": True}
            write_json_atomic(manifest_path, manifest)
            continue
        prepared = [prepare_one(tok, row, component, labels, raw_bos_id) for row in shard_frame.itertuples(index=False)]
        maximum = max(item["prompt_token_length"] + max(len(ids) for ids in item["candidate_ids"].values()) for item in prepared)
        if max_position and maximum > max_position:
            raise RuntimeError(f"Prompt truncation would be required: {maximum} > {max_position}")
        single_token = component.startswith("equal_")
        scores, used_batch, failures = score_with_backoff(model, tok, prepared, batch_size, single_token)
        validate_score_frame(scores, expected_ids)
        scores["component"] = component
        write_csv_atomic(shard_path, scores)
        manifest["oom_backoffs"].extend(failures)
        manifest["shards"][shard_name] = {
            "status": "PASS", "rows": len(expected_ids), "row_ids": expected_ids,
            "candidate_rows": len(scores), "batch_size_used": used_batch,
            "max_total_tokens": maximum, "sha256": sha256_file(shard_path), "resumed": False,
        }
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({"component": component, "shard": shard_name, "status": "PASS", "rows": len(expected_ids), "batch_size": used_batch}), flush=True)

    parts = []
    for shard_index, shard_frame in shard_ranges(primary, shard_rows):
        path = shard_dir / f"shard_{shard_index:04d}.csv"
        part = pd.read_csv(path)
        validate_score_frame(part, shard_frame.row_id.astype(int).tolist())
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True).sort_values(["row_id", "candidate_relation"]).reset_index(drop=True)
    validate_score_frame(combined, primary.row_id.astype(int).tolist())
    sum_predictions = reconstruct_predictions(combined, "sum_log_probability").rename(columns={"prediction": "prediction_sum"})
    mean_predictions = reconstruct_predictions(combined, "mean_log_probability").rename(columns={"prediction": "prediction_mean"})
    combined = combined.merge(sum_predictions, on="row_id", validate="many_to_one").merge(mean_predictions, on="row_id", validate="many_to_one")
    combined["correct_sum"] = combined.prediction_sum.eq(combined.gold_relation)
    combined["correct_mean"] = combined.prediction_mean.eq(combined.gold_relation)
    output_path = target / "row_scores.csv"
    write_csv_atomic(output_path, combined)
    manifest.update({
        "status": "PASS", "rows": int(combined.row_id.nunique()), "candidate_rows": len(combined),
        "row_scores": str(output_path), "row_scores_sha256": sha256_file(output_path),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })
    write_json_atomic(manifest_path, manifest)
    return manifest


def verify_frozen_protocols() -> None:
    checksums = read_json(OUT / "audit" / "protocol_checksums.json")
    for name, expected in checksums.items():
        payload = read_json(OUT / "audit" / f"protocol_{name}.json")
        stored = payload.pop("protocol_sha256")
        if stored != expected or stable_json_hash(payload) != expected:
            raise RuntimeError(f"Frozen protocol checksum mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["all", *COMPONENTS], default="all")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-rows", type=int, default=32)
    args = parser.parse_args()
    ensure_dirs()
    audit = read_json(OUT / "audit" / "audit_summary.json")
    if audit.get("status") != "PASS":
        raise RuntimeError("Audit/boundary gate is not PASS")
    verify_frozen_protocols()
    model_manifest = read_json(OUT / "audit" / "model_manifest.json")
    model_path = Path(model_manifest["local_path"])
    labels = read_json(OUT / "audit" / "equal_length_gate.json")["relation_to_label"]
    protocol_checksums = read_json(OUT / "audit" / "protocol_checksums.json")
    component_protocol = {
        "raw_natural": "raw_natural", "native_natural": "native_natural",
        "equal_raw": "equal_length", "equal_native": "equal_length",
    }
    primary = load_primary()
    if args.mode == "smoke":
        primary = deterministic_audit_rows(primary)
    components = COMPONENTS if args.protocol == "all" else [args.protocol]

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; main-experiment quantization/fallback is intentionally not automatic")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    raw_bos_id = model_manifest.get("raw_explicit_bos_token_id")
    if raw_bos_id is None:
        raise RuntimeError("Frozen Raw-Sum requires the audited explicit config/tokenizer BOS")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
    ).cuda().eval()
    if model.training:
        raise RuntimeError("model.eval() did not take effect")
    run = {
        "status": "RUNNING", "mode": args.mode, "model_path": str(model_path),
        "model_identity_sha256": model_manifest["model_identity_sha256"],
        "dtype": str(next(model.parameters()).dtype), "load_seconds": time.perf_counter() - load_started,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "components": {},
    }
    run_metadata_path = OUT / "audit" / f"run_{args.mode}_metadata.json"
    if run_metadata_path.is_file():
        previous = read_json(run_metadata_path)
        if (
            previous.get("mode") != args.mode
            or previous.get("model_path") != str(model_path)
            or previous.get("model_identity_sha256") != model_manifest["model_identity_sha256"]
        ):
            raise RuntimeError(f"Existing run metadata conflicts with this invocation: {run_metadata_path}")
        run["components"] = previous.get("components", {})
    write_json_atomic(run_metadata_path, run)
    for component in components:
        run["components"][component] = run_component(
            model, tok, config, primary, labels, int(raw_bos_id), component, args.mode,
            args.batch_size, args.shard_rows, protocol_checksums[component_protocol[component]],
            model_manifest["model_identity_sha256"],
        )
        write_json_atomic(run_metadata_path, run)
    run["status"] = "PASS"
    run["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    run["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    write_json_atomic(run_metadata_path, run)
    print(json.dumps({"stage": "B2.2", "phase": args.mode, "status": "PASS", "components": components}, indent=2))


if __name__ == "__main__":
    main()
