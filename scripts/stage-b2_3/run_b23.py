#!/usr/bin/env python3
"""Resumable BF16 free-generation runner for Stage B2.3."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_b23 import (
    DECISION_THRESHOLDS,
    GENERATION_SEEDS,
    MATCHED_CONFIGS,
    MODEL_PATH,
    OFFICIAL_CONFIGS,
    OUT,
    PARSER_SPEC,
    PRIMARY_GENERATION_SEED,
    child_seed,
    deterministic_pilot_subset,
    ensure_dirs,
    generation_user_content,
    load_primary,
    parse_final_answer,
    read_json,
    row_ids_checksum,
    sha256_file,
    sha256_text,
    smoke_subset,
    stable_json_hash,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


def load_model_and_tokenizer():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; quantization and CPU fallback are forbidden")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    if model.training or next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Model is not in eval() BF16 mode")
    return model, tokenizer, time.perf_counter() - started


def special_token_id(tokenizer, token: str) -> int:
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    if token_id < 0 or token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Tokenizer does not expose required Qwen thinking token: {token}")
    return token_id


def prompt_bundle(tokenizer, row, enable_thinking: bool) -> dict:
    user_content = generation_user_content(row)
    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_ids = list(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    ))
    roundtrip = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    if prompt_ids != roundtrip:
        raise RuntimeError(f"Chat-template text/token roundtrip mismatch for row {row.row_id}")
    return {
        "user_content": user_content,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "prompt_text_hash": sha256_text(prompt_text),
        "prompt_token_length": len(prompt_ids),
    }


def split_thinking(tokenizer, prompt_ids: list[int], generated_ids: list[int], enable_thinking: bool) -> dict:
    open_id = special_token_id(tokenizer, "<think>")
    close_id = special_token_id(tokenizer, "</think>")
    prompt_has_open = open_id in prompt_ids
    prompt_has_close = close_id in prompt_ids
    generated_has_open = open_id in generated_ids
    generated_has_close = close_id in generated_ids

    if not enable_thinking:
        thinking_ids = []
        final_ids = list(generated_ids)
        think_parse_success = prompt_has_open and prompt_has_close and not generated_has_open
    else:
        start = generated_ids.index(open_id) + 1 if generated_has_open else 0
        if close_id in generated_ids[start:]:
            close_position = generated_ids.index(close_id, start)
            thinking_ids = generated_ids[start:close_position]
            final_ids = generated_ids[close_position + 1:]
            think_parse_success = prompt_has_open and generated_has_close
        else:
            thinking_ids = generated_ids[start:]
            final_ids = list(generated_ids)
            think_parse_success = False
    return {
        "has_think_open": bool(prompt_has_open or generated_has_open),
        "has_think_close": bool(prompt_has_close or generated_has_close),
        "prompt_has_think_open": prompt_has_open,
        "prompt_has_think_close": prompt_has_close,
        "generated_has_think_open": generated_has_open,
        "generated_has_think_close": generated_has_close,
        "think_parse_success": think_parse_success,
        "thinking_text": tokenizer.decode(thinking_ids, skip_special_tokens=True),
        "thinking_token_count": len(thinking_ids),
        "final_response_text": tokenizer.decode(final_ids, skip_special_tokens=True),
        "final_response_token_count": len(final_ids),
    }


def eos_ids(tokenizer, model) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def generate_row(model, tokenizer, row, protocol_key: str, config: dict, generation_seed: int,
                 max_new_tokens: int, config_hash: str) -> dict:
    bundle = prompt_bundle(tokenizer, row, bool(config["enable_thinking"]))
    input_ids = torch.tensor([bundle["prompt_ids"]], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    row_seed = child_seed(generation_seed, int(row.row_id))
    generation_kwargs = {
        "do_sample": bool(config["do_sample"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "top_k": int(config["top_k"]),
        "min_p": float(config["min_p"]),
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": model.generation_config.eos_token_id,
        "use_cache": True,
    }
    torch.manual_seed(row_seed)
    torch.cuda.manual_seed_all(row_seed)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                **generation_kwargs,
            )
    except torch.cuda.OutOfMemoryError:
        del input_ids, attention_mask
        torch.cuda.empty_cache()
        bundle = prompt_bundle(tokenizer, row, bool(config["enable_thinking"]))
        input_ids = torch.tensor([bundle["prompt_ids"]], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        torch.manual_seed(row_seed)
        torch.cuda.manual_seed_all(row_seed)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                **generation_kwargs,
            )
    latency = time.perf_counter() - started
    sequence = output.sequences[0].detach().cpu().tolist()
    generated_ids = [int(item) for item in sequence[len(bundle["prompt_ids"]):]]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    split = split_thinking(
        tokenizer,
        bundle["prompt_ids"],
        generated_ids,
        bool(config["enable_thinking"]),
    )
    parse = parse_final_answer(split["final_response_text"])
    eos = eos_ids(tokenizer, model)
    eos_terminated = bool(generated_ids and generated_ids[-1] in eos)
    hit_max = len(generated_ids) >= int(max_new_tokens)
    correct = parse["parsed_relation"] == str(row.canonical_relation)
    effective_config = {
        "do_sample": generation_kwargs["do_sample"],
        "temperature": generation_kwargs["temperature"],
        "top_p": generation_kwargs["top_p"],
        "top_k": generation_kwargs["top_k"],
        "min_p": generation_kwargs["min_p"],
        "max_new_tokens": generation_kwargs["max_new_tokens"],
        "pad_token_id": generation_kwargs["pad_token_id"],
        "eos_token_id": generation_kwargs["eos_token_id"],
    }
    result = {
        "row_id": int(row.row_id),
        "story_group_id": str(row.story_group_id),
        "hop": int(row.computed_hop),
        "gold_relation": str(row.canonical_relation),
        "protocol": config["protocol"],
        "protocol_key": protocol_key,
        "generation_seed": int(generation_seed),
        "row_rng_seed": int(row_seed),
        "prompt_text_hash": bundle["prompt_text_hash"],
        "prompt_token_length": bundle["prompt_token_length"],
        "enable_thinking": bool(config["enable_thinking"]),
        "generation_config": effective_config,
        "generation_config_hash": config_hash,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        **split,
        **parse,
        "correct": bool(correct),
        "generation_token_count": len(generated_ids),
        "hit_max_new_tokens": hit_max,
        "eos_terminated": eos_terminated,
        "latency_seconds": latency,
        "tokens_per_second": len(generated_ids) / latency if latency > 0 else None,
    }
    del input_ids, attention_mask, output
    return result


def canonical_batch_seed(generation_seed: int, row_ids: list[int]) -> int:
    payload = f"{int(generation_seed)}:" + ",".join(str(int(row_id)) for row_id in row_ids)
    return int(sha256_text(payload)[:8], 16) % (2**31 - 1)


def generate_batch(model, tokenizer, rows: list, protocol_key: str, config: dict,
                   generation_seed: int, max_new_tokens: int, config_hash: str) -> list[dict]:
    bundles = [prompt_bundle(tokenizer, row, bool(config["enable_thinking"])) for row in rows]
    width = max(len(bundle["prompt_ids"]) for bundle in bundles)
    pad_id = int(tokenizer.pad_token_id)
    padded = [[pad_id] * (width - len(bundle["prompt_ids"])) + bundle["prompt_ids"] for bundle in bundles]
    masks = [[0] * (width - len(bundle["prompt_ids"])) + [1] * len(bundle["prompt_ids"]) for bundle in bundles]
    input_ids = torch.tensor(padded, dtype=torch.long, device="cuda")
    attention_mask = torch.tensor(masks, dtype=torch.long, device="cuda")
    row_ids = [int(row.row_id) for row in rows]
    batch_seed = canonical_batch_seed(generation_seed, row_ids)
    generation_kwargs = {
        "do_sample": bool(config["do_sample"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "top_k": int(config["top_k"]),
        "min_p": float(config["min_p"]),
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": pad_id,
        "eos_token_id": model.generation_config.eos_token_id,
        "use_cache": True,
    }
    torch.manual_seed(batch_seed)
    torch.cuda.manual_seed_all(batch_seed)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            **generation_kwargs,
        )
    batch_latency = time.perf_counter() - started
    sequences = output.sequences.detach().cpu().tolist()
    eos = eos_ids(tokenizer, model)
    effective_config = {
        "do_sample": generation_kwargs["do_sample"],
        "temperature": generation_kwargs["temperature"],
        "top_p": generation_kwargs["top_p"],
        "top_k": generation_kwargs["top_k"],
        "min_p": generation_kwargs["min_p"],
        "max_new_tokens": generation_kwargs["max_new_tokens"],
        "pad_token_id": generation_kwargs["pad_token_id"],
        "eos_token_id": generation_kwargs["eos_token_id"],
        "batch_size": len(rows),
    }
    results = []
    for row, bundle, sequence in zip(rows, bundles, sequences):
        generated_ids = [int(item) for item in sequence[width:]]
        eos_positions = [index for index, token_id in enumerate(generated_ids) if token_id in eos]
        if eos_positions:
            generated_ids = generated_ids[:eos_positions[0] + 1]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        split = split_thinking(
            tokenizer, bundle["prompt_ids"], generated_ids, bool(config["enable_thinking"])
        )
        parse = parse_final_answer(split["final_response_text"])
        eos_terminated = bool(generated_ids and generated_ids[-1] in eos)
        hit_max = len(generated_ids) >= int(max_new_tokens)
        correct = parse["parsed_relation"] == str(row.canonical_relation)
        attributed_latency = batch_latency / len(rows)
        results.append({
            "row_id": int(row.row_id),
            "story_group_id": str(row.story_group_id),
            "hop": int(row.computed_hop),
            "gold_relation": str(row.canonical_relation),
            "protocol": config["protocol"],
            "protocol_key": protocol_key,
            "generation_seed": int(generation_seed),
            "row_rng_seed": int(batch_seed),
            "batch_rng_seed": int(batch_seed),
            "batch_row_ids": row_ids,
            "batch_size": len(rows),
            "prompt_text_hash": bundle["prompt_text_hash"],
            "prompt_token_length": bundle["prompt_token_length"],
            "enable_thinking": bool(config["enable_thinking"]),
            "generation_config": effective_config,
            "generation_config_hash": config_hash,
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            **split,
            **parse,
            "correct": bool(correct),
            "generation_token_count": len(generated_ids),
            "hit_max_new_tokens": hit_max,
            "eos_terminated": eos_terminated,
            "latency_seconds": attributed_latency,
            "batch_wall_latency_seconds": batch_latency,
            "latency_definition": "equal share of deterministic batch wall time",
            "tokens_per_second": len(generated_ids) / attributed_latency if attributed_latency > 0 else None,
        })
    del input_ids, attention_mask, output
    return results


def row_metrics_frame(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "row_id", "story_group_id", "hop", "gold_relation", "protocol", "protocol_key",
        "generation_seed", "row_rng_seed", "prompt_text_hash", "prompt_token_length",
        "enable_thinking", "parsed_relation", "parse_status", "correct",
        "has_think_open", "has_think_close", "think_parse_success",
        "thinking_token_count", "final_response_token_count", "generation_token_count",
        "hit_max_new_tokens", "eos_terminated", "latency_seconds", "tokens_per_second",
        "generation_config_hash",
    ]
    return pd.DataFrame([{column: row.get(column) for column in columns} for row in rows])


def verify_existing_shard(payload: dict, row, protocol_key: str, config_hash: str,
                          generation_seed: int, expected_prompt_hash: str) -> None:
    checks = {
        "row_id": payload.get("row_id") == int(row.row_id),
        "protocol_key": payload.get("protocol_key") == protocol_key,
        "generation_seed": payload.get("generation_seed") == int(generation_seed),
        "generation_config_hash": payload.get("generation_config_hash") == config_hash,
        "prompt_text_hash": payload.get("prompt_text_hash") == expected_prompt_hash,
        "generated_token_ids": isinstance(payload.get("generated_token_ids"), list),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Existing shard integrity conflict for row {row.row_id}: {checks}")


def run_one(model, tokenizer, frame: pd.DataFrame, protocol_key: str, config: dict,
            generation_seed: int, output_dir: Path, max_new_tokens: int,
            model_identity_sha256: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    parser_hash = read_json(OUT / "audit" / "parser_spec.json")["parser_spec_sha256"]
    config_payload = {
        "protocol_key": protocol_key,
        "config": config,
        "max_new_tokens": int(max_new_tokens),
        "parser_spec_sha256": parser_hash,
        "model_identity_sha256": model_identity_sha256,
    }
    config_hash = stable_json_hash(config_payload)
    manifest_path = output_dir / "run_manifest.json"
    expected_ids = frame.row_id.astype(int).tolist()
    manifest = {
        "status": "RUNNING",
        "protocol_key": protocol_key,
        "protocol": config["protocol"],
        "generation_seed": int(generation_seed),
        "row_rng_policy": "sha256(generation_seed:row_id) child seed",
        "rows_expected": len(frame),
        "row_ids": expected_ids,
        "row_ids_sha256": row_ids_checksum(expected_ids),
        "generation_config": config_payload,
        "generation_config_hash": config_hash,
        "model_identity_sha256": model_identity_sha256,
        "completed_row_ids": [],
        "shards": {},
    }
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        for key in ["protocol_key", "generation_seed", "row_ids", "generation_config_hash", "model_identity_sha256"]:
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Existing run manifest conflicts on {key}: {manifest_path}")
        manifest["completed_row_ids"] = existing.get("completed_row_ids", [])
        manifest["shards"] = existing.get("shards", {})

    completed = set()
    rows_by_id = {int(row.row_id): row for row in frame.itertuples(index=False)}
    for row_id, row in rows_by_id.items():
        shard_path = shard_dir / f"row_{row_id}.json"
        if not shard_path.is_file():
            continue
        payload = read_json(shard_path)
        expected_prompt_hash = prompt_bundle(tokenizer, row, bool(config["enable_thinking"]))["prompt_text_hash"]
        verify_existing_shard(payload, row, protocol_key, config_hash, generation_seed, expected_prompt_hash)
        checksum = sha256_file(shard_path)
        recorded = manifest.get("shards", {}).get(str(row_id), {}).get("sha256")
        if recorded is not None and recorded != checksum:
            raise RuntimeError(f"Existing shard checksum conflict for row {row_id}")
        completed.add(row_id)
        manifest["shards"][str(row_id)] = {"sha256": checksum, "resumed": True}
    manifest["completed_row_ids"] = sorted(completed)
    write_json_atomic(manifest_path, manifest)

    started = time.perf_counter()
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        row_id = int(row.row_id)
        if row_id in completed:
            continue
        payload = generate_row(
            model, tokenizer, row, protocol_key, config, generation_seed,
            max_new_tokens, config_hash,
        )
        shard_path = shard_dir / f"row_{row_id}.json"
        write_json_atomic(shard_path, payload)
        manifest["shards"][str(row_id)] = {"sha256": sha256_file(shard_path), "resumed": False}
        completed.add(row_id)
        manifest["completed_row_ids"] = sorted(completed)
        manifest["last_completed_position"] = position
        manifest["last_completed_row_id"] = row_id
        manifest["elapsed_generation_seconds_this_invocation"] = time.perf_counter() - started
        write_json_atomic(manifest_path, manifest)
        print(
            json.dumps({
                "protocol": config["protocol"], "seed": generation_seed,
                "position": position, "rows": len(frame), "row_id": row_id,
                "tokens": payload["generation_token_count"],
                "parsed": payload["parsed_relation"],
            }),
            flush=True,
        )

    rows = [read_json(shard_dir / f"row_{row_id}.json") for row_id in expected_ids]
    if len(rows) != len(frame) or len({row["row_id"] for row in rows}) != len(frame):
        raise RuntimeError("Combined run does not contain every expected row exactly once")
    write_jsonl_atomic(output_dir / "row_generations.jsonl", rows)
    write_csv_atomic(output_dir / "metrics.csv", row_metrics_frame(rows))
    total_tokens = sum(int(row["generation_token_count"]) for row in rows)
    total_latency = sum(float(row["latency_seconds"]) for row in rows)
    manifest.update({
        "status": "PASS",
        "completed_row_ids": expected_ids,
        "row_generations_sha256": sha256_file(output_dir / "row_generations.jsonl"),
        "metrics_sha256": sha256_file(output_dir / "metrics.csv"),
        "total_generated_tokens": total_tokens,
        "summed_row_latency_seconds": total_latency,
        "tokens_per_second_summed": total_tokens / total_latency if total_latency > 0 else None,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })
    write_json_atomic(manifest_path, manifest)
    return manifest


def run_one_batched(model, tokenizer, frame: pd.DataFrame, protocol_key: str, config: dict,
                    generation_seed: int, output_dir: Path, max_new_tokens: int,
                    model_identity_sha256: str, batch_size: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    parser_hash = read_json(OUT / "audit" / "parser_spec.json")["parser_spec_sha256"]
    seed_policy = (
        "canonical sorted batches; batch_seed = first 8 hex of "
        "sha256(generation_seed:comma-separated-row-ids) modulo 2^31-1"
    )
    config_payload = {
        "protocol_key": protocol_key,
        "config": config,
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "seed_policy": seed_policy,
        "parser_spec_sha256": parser_hash,
        "model_identity_sha256": model_identity_sha256,
    }
    config_hash = stable_json_hash(config_payload)
    manifest_path = output_dir / "run_manifest.json"
    expected_ids = frame.row_id.astype(int).tolist()
    manifest = {
        "status": "RUNNING",
        "protocol_key": protocol_key,
        "protocol": config["protocol"],
        "generation_seed": int(generation_seed),
        "row_rng_policy": seed_policy,
        "batch_size": int(batch_size),
        "rows_expected": len(frame),
        "row_ids": expected_ids,
        "row_ids_sha256": row_ids_checksum(expected_ids),
        "generation_config": config_payload,
        "generation_config_hash": config_hash,
        "model_identity_sha256": model_identity_sha256,
        "completed_row_ids": [],
        "shards": {},
    }
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        for key in ["protocol_key", "generation_seed", "row_ids", "generation_config_hash", "model_identity_sha256", "batch_size"]:
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"Existing batched manifest conflicts on {key}: {manifest_path}")
        manifest["completed_row_ids"] = existing.get("completed_row_ids", [])
        manifest["shards"] = existing.get("shards", {})

    completed = set()
    for row in frame.itertuples(index=False):
        row_id = int(row.row_id)
        shard_path = shard_dir / f"row_{row_id}.json"
        if not shard_path.is_file():
            continue
        payload = read_json(shard_path)
        expected_prompt_hash = prompt_bundle(tokenizer, row, bool(config["enable_thinking"]))["prompt_text_hash"]
        verify_existing_shard(payload, row, protocol_key, config_hash, generation_seed, expected_prompt_hash)
        checksum = sha256_file(shard_path)
        recorded = manifest.get("shards", {}).get(str(row_id), {}).get("sha256")
        if recorded is not None and recorded != checksum:
            raise RuntimeError(f"Existing batched shard checksum conflict for row {row_id}")
        completed.add(row_id)
        manifest["shards"][str(row_id)] = {"sha256": checksum, "resumed": True}
    manifest["completed_row_ids"] = sorted(completed)
    write_json_atomic(manifest_path, manifest)

    frame_rows = list(frame.itertuples(index=False))
    batches = [frame_rows[start:start + batch_size] for start in range(0, len(frame_rows), batch_size)]
    started = time.perf_counter()
    for batch_position, batch_rows in enumerate(batches, start=1):
        batch_ids = [int(row.row_id) for row in batch_rows]
        if all(row_id in completed for row_id in batch_ids):
            continue
        payloads = generate_batch(
            model, tokenizer, batch_rows, protocol_key, config,
            generation_seed, max_new_tokens, config_hash,
        )
        for payload in payloads:
            row_id = int(payload["row_id"])
            shard_path = shard_dir / f"row_{row_id}.json"
            if shard_path.is_file():
                existing_payload = read_json(shard_path)
                for key in ["generated_token_ids", "prompt_text_hash", "generation_config_hash", "batch_rng_seed"]:
                    if existing_payload.get(key) != payload.get(key):
                        raise RuntimeError(f"Deterministic batch replay conflict for row {row_id}, field {key}")
            else:
                write_json_atomic(shard_path, payload)
            manifest["shards"][str(row_id)] = {"sha256": sha256_file(shard_path), "resumed": row_id in completed}
            completed.add(row_id)
        manifest["completed_row_ids"] = sorted(completed)
        manifest["last_completed_batch"] = batch_position
        manifest["last_completed_row_ids"] = batch_ids
        manifest["elapsed_generation_seconds_this_invocation"] = time.perf_counter() - started
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({
            "protocol": config["protocol"], "seed": generation_seed,
            "batch": batch_position, "batches": len(batches), "row_ids": batch_ids,
            "tokens": [payload["generation_token_count"] for payload in payloads],
            "parsed": [payload["parsed_relation"] for payload in payloads],
        }), flush=True)

    rows = [read_json(shard_dir / f"row_{row_id}.json") for row_id in expected_ids]
    if len(rows) != len(frame) or len({row["row_id"] for row in rows}) != len(frame):
        raise RuntimeError("Combined batched run does not contain every expected row exactly once")
    write_jsonl_atomic(output_dir / "row_generations.jsonl", rows)
    write_csv_atomic(output_dir / "metrics.csv", row_metrics_frame(rows))
    total_tokens = sum(int(row["generation_token_count"]) for row in rows)
    attributed_latency = sum(float(row["latency_seconds"]) for row in rows)
    manifest.update({
        "status": "PASS",
        "completed_row_ids": expected_ids,
        "row_generations_sha256": sha256_file(output_dir / "row_generations.jsonl"),
        "metrics_sha256": sha256_file(output_dir / "metrics.csv"),
        "total_generated_tokens": total_tokens,
        "attributed_latency_seconds": attributed_latency,
        "tokens_per_second_attributed": total_tokens / attributed_latency if attributed_latency > 0 else None,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    })
    write_json_atomic(manifest_path, manifest)
    return manifest


def freeze_max_new_tokens(smoke_rows: dict[str, list[dict]]) -> dict:
    generation_path = OUT / "audit" / "generation_config.json"
    generation = read_json(generation_path)
    thinking = smoke_rows["fg_thinking"]
    truncation_rate = sum(bool(row["hit_max_new_tokens"]) for row in thinking) / len(thinking)
    resolved = 2048 if truncation_rate >= DECISION_THRESHOLDS["smoke_truncation_switch_rate"] else 1024
    generation["resolved_max_new_tokens"] = resolved
    generation["smoke_thinking_truncation_rate"] = truncation_rate
    generation["smoke_resolution_reason"] = (
        "switched uniformly to 2048 because the preregistered smoke truncation threshold was reached"
        if resolved == 2048
        else "retained 1024 because the preregistered smoke truncation threshold was not reached"
    )
    generation["status"] = "FROZEN"
    generation["resolved_config_sha256"] = stable_json_hash({
        key: value for key, value in generation.items() if key != "resolved_config_sha256"
    })
    write_json_atomic(generation_path, generation)
    return generation


def run_smoke(model, tokenizer, model_identity_sha256: str, load_seconds: float) -> None:
    primary = smoke_subset(load_primary())
    smoke_rows = {}
    for protocol_key in ["fg_nonthinking", "fg_thinking"]:
        config = OFFICIAL_CONFIGS[protocol_key]
        output_dir = OUT / "audit" / "smoke" / protocol_key
        run_one(
            model, tokenizer, primary, protocol_key, config,
            PRIMARY_GENERATION_SEED, output_dir, 1024, model_identity_sha256,
        )
        from common_b23 import read_jsonl
        smoke_rows[protocol_key] = read_jsonl(output_dir / "row_generations.jsonl")
    generation = freeze_max_new_tokens(smoke_rows)
    nonthinking_nonempty = [
        row["row_id"] for row in smoke_rows["fg_nonthinking"] if row["thinking_text"].strip()
    ]
    tokenizer_template_checks = {
        "nonthinking_prompts_have_closed_empty_think_scaffold": all(
            row["prompt_has_think_open"] and row["prompt_has_think_close"]
            for row in smoke_rows["fg_nonthinking"]
        ),
        "thinking_mode_has_official_open_boundary": all(
            row["has_think_open"] for row in smoke_rows["fg_thinking"]
        ),
        "nonthinking_has_no_nonempty_thinking": not nonthinking_nonempty,
    }
    smoke = {
        "status": "PASS" if all(tokenizer_template_checks.values()) else "FAIL",
        "model_eval": not model.training,
        "dtype": str(next(model.parameters()).dtype),
        "inference_mode": True,
        "load_seconds": load_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "template_checks": tokenizer_template_checks,
        "nonthinking_nonempty_thinking_row_ids": nonthinking_nonempty,
        "resolved_max_new_tokens": generation["resolved_max_new_tokens"],
        "thinking_truncation_rate": generation["smoke_thinking_truncation_rate"],
        "thinking_boundary_failures": [
            row["row_id"] for row in smoke_rows["fg_thinking"] if not row["think_parse_success"]
        ],
    }
    write_json_atomic(OUT / "audit" / "model_load_smoke.json", smoke)
    write_json_atomic(OUT / "audit" / "truncation_audit.json", {
        "status": "PASS",
        "initial_max_new_tokens": 1024,
        "thinking_smoke_rows": len(smoke_rows["fg_thinking"]),
        "thinking_rows_hit_max": sum(row["hit_max_new_tokens"] for row in smoke_rows["fg_thinking"]),
        "thinking_truncation_rate": generation["smoke_thinking_truncation_rate"],
        "switch_threshold": DECISION_THRESHOLDS["smoke_truncation_switch_rate"],
        "resolved_max_new_tokens": generation["resolved_max_new_tokens"],
        "accuracy_was_not_used_for_resolution": True,
    })
    if smoke["status"] != "PASS":
        raise RuntimeError("Model/chat-template smoke audit failed")


def run_batch_smoke(model, tokenizer, model_identity_sha256: str) -> None:
    generation_path = OUT / "audit" / "generation_config.json"
    generation = read_json(generation_path)
    if generation.get("status") != "FROZEN":
        raise RuntimeError("Length-resolved generation config is not frozen")
    max_new_tokens = int(generation["resolved_max_new_tokens"])
    primary = smoke_subset(load_primary())
    proposals = {"fg_nonthinking": 8, "fg_thinking": 8}
    resolved = {}
    attempts = []
    for protocol_key, proposal in proposals.items():
        candidates = []
        value = proposal
        while value >= 1:
            if value not in candidates:
                candidates.append(value)
            value //= 2
        for batch_size in candidates:
            subset = primary.head(batch_size)
            output_dir = OUT / "audit" / "batch_smoke" / protocol_key / f"batch_{batch_size}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                manifest = run_one_batched(
                    model, tokenizer, subset, protocol_key, OFFICIAL_CONFIGS[protocol_key],
                    PRIMARY_GENERATION_SEED, output_dir, max_new_tokens,
                    model_identity_sha256, batch_size,
                )
                attempts.append({
                    "protocol_key": protocol_key,
                    "batch_size": batch_size,
                    "status": "PASS",
                    "max_allocated_gib": manifest["max_allocated_gib"],
                    "max_reserved_gib": manifest["max_reserved_gib"],
                    "wall_seconds": manifest["wall_seconds_this_invocation"],
                    "accuracy_not_used": True,
                })
                resolved[protocol_key] = batch_size
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                attempts.append({
                    "protocol_key": protocol_key,
                    "batch_size": batch_size,
                    "status": "OOM_RETRY_SMALLER_BATCH",
                    "accuracy_not_used": True,
                })
        if protocol_key not in resolved:
            raise RuntimeError(f"No viable BF16 batch size for {protocol_key}")
    generation["batch_sizes"] = resolved
    generation["batch_seed_policy"] = (
        "canonical sorted batches; batch seed is sha256(generation_seed:comma-separated-row-ids); "
        "a partial resumed batch is deterministically replayed and completed shards are never overwritten"
    )
    generation["batch_size_resolution"] = "memory/throughput smoke only; accuracy was not inspected or used"
    generation["resolved_config_sha256"] = stable_json_hash({
        key: value for key, value in generation.items() if key != "resolved_config_sha256"
    })
    write_json_atomic(generation_path, generation)
    write_json_atomic(OUT / "audit" / "batch_smoke.json", {
        "status": "PASS",
        "resolved_max_new_tokens": max_new_tokens,
        "proposals": proposals,
        "resolved_batch_sizes": resolved,
        "attempts": attempts,
        "accuracy_was_not_used": True,
    })


def run_phase(phase: str) -> None:
    ensure_dirs()
    preflight = read_json(OUT / "audit" / "preflight_summary.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("Phase 0 preflight is not PASS")
    model_manifest = read_json(OUT / "audit" / "model_manifest.json")
    model, tokenizer, load_seconds = load_model_and_tokenizer()
    model_identity = model_manifest["model_identity_sha256"]
    if phase == "smoke":
        run_smoke(model, tokenizer, model_identity, load_seconds)
        return

    if phase == "batch-smoke":
        run_batch_smoke(model, tokenizer, model_identity)
        return

    generation = read_json(OUT / "audit" / "generation_config.json")
    if generation.get("status") != "FROZEN":
        raise RuntimeError("Generation config is not frozen by the smoke audit")
    max_new_tokens = int(generation["resolved_max_new_tokens"])
    batch_sizes = generation.get("batch_sizes")
    if not batch_sizes:
        raise RuntimeError("Batch sizes are not frozen by the memory/throughput smoke")
    primary = load_primary()
    if phase == "pilot":
        subset = deterministic_pilot_subset(primary)
        for protocol_key in ["fg_nonthinking", "fg_thinking"]:
            for seed in GENERATION_SEEDS:
                run_one_batched(
                    model, tokenizer, subset, protocol_key, OFFICIAL_CONFIGS[protocol_key],
                    seed, OUT / "pilot" / protocol_key / f"seed_{seed}",
                    max_new_tokens, model_identity, int(batch_sizes[protocol_key]),
                )
    elif phase == "matched":
        subset = deterministic_pilot_subset(primary)
        for protocol_key in ["fg_nonthinking_matched", "fg_thinking_matched"]:
            short_name = protocol_key.replace("_matched", "")
            run_one_batched(
                model, tokenizer, subset, protocol_key, MATCHED_CONFIGS[protocol_key],
                PRIMARY_GENERATION_SEED,
                OUT / "pilot" / "matched_decoding" / short_name,
                max_new_tokens, model_identity, int(batch_sizes[short_name]),
            )
    elif phase == "full-fgn":
        run_one_batched(
            model, tokenizer, primary, "fg_nonthinking", OFFICIAL_CONFIGS["fg_nonthinking"],
            PRIMARY_GENERATION_SEED, OUT / "full" / "fg_nonthinking",
            max_new_tokens, model_identity, int(batch_sizes["fg_nonthinking"]),
        )
    elif phase == "full-fgt":
        run_one_batched(
            model, tokenizer, primary, "fg_thinking", OFFICIAL_CONFIGS["fg_thinking"],
            PRIMARY_GENERATION_SEED, OUT / "full" / "fg_thinking",
            max_new_tokens, model_identity, int(batch_sizes["fg_thinking"]),
        )
    else:
        raise ValueError(phase)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "batch-smoke", "pilot", "matched", "full-fgn", "full-fgt"], required=True)
    args = parser.parse_args()
    run_phase(args.phase)
    print(json.dumps({"stage": "B2.3", "phase": args.phase, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()

