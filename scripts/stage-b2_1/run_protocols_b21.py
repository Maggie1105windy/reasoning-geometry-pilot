#!/usr/bin/env python3
"""Run preregistered native-chat and equal-length Stage B2.1 protocols."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_b21 import (
    LABEL_MAP, MODEL_SPECS, NATURAL_CANDIDATES, OUT, RELATIONS,
    candidate_suffix_from_full, ensure_dirs, equal_native_user, equal_raw_prompt,
    load_primary, native_natural_user, read_json, sha256_file, write_json,
)


def rendered(tok, user: str) -> list[int]:
    return list(tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True,
    ))


def rendered_with_answer(tok, user: str, answer: str) -> list[int]:
    return list(tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False,
    ))


def score_expanded(model, tok, prompts: list[tuple[int, list[int]]], candidates: dict[str, list[int]], batch_size: int = 32):
    expanded = []
    for row_id, prompt in prompts:
        for relation in RELATIONS:
            cids = candidates[relation]
            expanded.append((row_id, relation, prompt + cids, len(prompt), cids))
    output = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        max_len = max(len(item[2]) for item in batch)
        input_ids = torch.full((len(batch), max_len), tok.eos_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, (_, _, seq, _, _) in enumerate(batch):
            input_ids[index, :len(seq)] = torch.tensor(seq)
            attention[index, :len(seq)] = 1
        input_ids = input_ids.cuda(non_blocking=True)
        attention = attention.cuda(non_blocking=True)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
        for index, (row_id, relation, _, prompt_len, cids) in enumerate(batch):
            values = [float(logp[index, prompt_len + offset - 1, token_id]) for offset, token_id in enumerate(cids)]
            output.append({
                "row_id": row_id, "candidate_relation": relation, "candidate_token_count": len(cids),
                "sum_log_probability": float(sum(values)), "mean_log_probability": float(np.mean(values)),
                "token_log_probabilities": json.dumps(values, separators=(",", ":")),
            })
        del input_ids, attention, logits, logp
        if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(expanded):
            print(json.dumps({"scored_sequences": min(start + batch_size, len(expanded)), "total_sequences": len(expanded), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    meta = {
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "candidate_sequences": len(expanded), "batch_size": batch_size,
    }
    return pd.DataFrame(output), meta


def score_single_token(model, tok, prompts: list[tuple[int, list[int]]], candidates: dict[str, list[int]], batch_size: int = 32):
    if not all(len(value) == 1 for value in candidates.values()):
        return score_expanded(model, tok, prompts, candidates, batch_size)
    output = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        max_len = max(len(item[1]) for item in batch)
        input_ids = torch.full((len(batch), max_len), tok.eos_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, (_, seq) in enumerate(batch):
            input_ids[index, :len(seq)] = torch.tensor(seq)
            attention[index, :len(seq)] = 1
        input_ids = input_ids.cuda(non_blocking=True)
        attention = attention.cuda(non_blocking=True)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
        for index, (row_id, seq) in enumerate(batch):
            for relation in RELATIONS:
                value = float(logp[index, len(seq) - 1, candidates[relation][0]])
                output.append({
                    "row_id": row_id, "candidate_relation": relation, "candidate_token_count": 1,
                    "sum_log_probability": value, "mean_log_probability": value,
                    "token_log_probabilities": json.dumps([value], separators=(",", ":")),
                })
        del input_ids, attention, logits, logp
        if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(prompts):
            print(json.dumps({"scored_prompts": min(start + batch_size, len(prompts)), "total_prompts": len(prompts), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    meta = {
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "candidate_sequences": len(prompts) * 8, "forward_prompt_count": len(prompts),
        "single_token_shared_forward_optimization": True, "batch_size": batch_size,
    }
    return pd.DataFrame(output), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    ensure_dirs()
    model_key = args.model
    if read_json(OUT / "raw_reproduction" / "reproduction_check.json")["status"] != "PASS":
        raise RuntimeError("Gate A is not PASS")
    equal_protocol = read_json(OUT / "equal_length_verbalizer" / "protocol.json")
    labels = equal_protocol["relation_to_label"]
    primary = load_primary()
    spec = MODEL_SPECS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    tasks = []
    if model_key != "base":
        natural_prompts = []
        natural_candidates = None
        max_prompt = 0
        for row in primary.itertuples(index=False):
            user = native_natural_user(row)
            prompt = rendered(tok, user)
            max_prompt = max(max_prompt, len(prompt))
            natural_prompts.append((int(row.row_id), prompt))
            if natural_candidates is None:
                natural_candidates = {
                    relation: candidate_suffix_from_full(tok, prompt, rendered_with_answer(tok, user, candidate))
                    for relation, candidate in NATURAL_CANDIDATES.items()
                }
        tasks.append(("native_natural", natural_prompts, natural_candidates, "expanded"))

    raw_prompts = []
    raw_candidates = {relation: tok(" " + labels[relation], add_special_tokens=False).input_ids for relation in RELATIONS}
    for row in primary.itertuples(index=False):
        content = tok(equal_raw_prompt(row, labels), add_special_tokens=False).input_ids
        prompt = [tok.bos_token_id] + content
        raw_prompts.append((int(row.row_id), prompt))
    tasks.append(("equal_raw", raw_prompts, raw_candidates, "single"))

    if model_key != "base":
        native_equal_prompts = []
        native_equal_candidates = None
        for row in primary.itertuples(index=False):
            user = equal_native_user(row, labels)
            prompt = rendered(tok, user)
            native_equal_prompts.append((int(row.row_id), prompt))
            if native_equal_candidates is None:
                native_equal_candidates = {
                    relation: candidate_suffix_from_full(tok, prompt, rendered_with_answer(tok, user, " " + labels[relation]))
                    for relation in RELATIONS
                }
        tasks.append(("equal_native", native_equal_prompts, native_equal_candidates, "single"))

    limits = {
        name: {
            "max_prompt_tokens": max(len(seq) for _, seq in prompts),
            "max_total_tokens": max(len(seq) for _, seq in prompts) + max(len(ids) for ids in candidates.values()),
        } for name, prompts, candidates, _ in tasks
    }
    config_limit = getattr(tok, "model_max_length", None)
    if config_limit and config_limit < 10**9 and any(item["max_total_tokens"] > config_limit for item in limits.values()):
        raise RuntimeError(f"Prompt truncation would be required: {limits}, tokenizer limit={config_limit}")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
    ).cuda().eval()
    metadata = {
        "status": "RUNNING", "model": model_key, "model_path": str(spec["path"]),
        "revision": spec["revision"], "rows": len(primary), "batch_size": args.batch_size,
        "no_truncation": True, "prompt_limits": limits, "tasks": {},
    }
    for name, prompts, candidates, method in tasks:
        print(json.dumps({"model": model_key, "task": name, "status": "START"}), flush=True)
        if method == "single":
            scores, meta = score_single_token(model, tok, prompts, candidates, args.batch_size)
        else:
            scores, meta = score_expanded(model, tok, prompts, candidates, args.batch_size)
        scores["model"] = model_key
        scores["protocol_component"] = name
        if name == "native_natural":
            target = OUT / "native_chat" / "per_model" / f"{model_key}_scores.csv"
        else:
            target = OUT / "equal_length_verbalizer" / "per_model" / f"{model_key}_{name}_scores.csv"
        scores.to_csv(target, index=False)
        meta.update({
            "output": str(target), "output_sha256": sha256_file(target),
            "all_finite": bool(np.isfinite(scores[["sum_log_probability", "mean_log_probability"]].to_numpy()).all()),
            "rows": int(scores.row_id.nunique()), "candidate_rows": len(scores),
            "candidate_token_ids": candidates,
        })
        metadata["tasks"][name] = meta
        write_json(OUT / "audit" / f"run_metadata_{model_key}.json", metadata)
        print(json.dumps({"model": model_key, "task": name, "status": "PASS", "runtime_seconds": round(meta["runtime_seconds"], 1)}), flush=True)
    metadata["status"] = "PASS"
    metadata["total_runtime_seconds"] = time.perf_counter() - started
    write_json(OUT / "audit" / f"run_metadata_{model_key}.json", metadata)
    print(json.dumps({"model": model_key, "status": "PASS", "total_runtime_seconds": round(metadata["total_runtime_seconds"], 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
