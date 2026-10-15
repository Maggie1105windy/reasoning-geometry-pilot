#!/usr/bin/env python3
"""Phase 0: audit B2.2, freeze B2.3 protocols, and stop on any mismatch."""

from __future__ import annotations

import json
import platform
import socket
import sys
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from common_b23 import (
    B02,
    B1,
    B22,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLDS,
    EXPECTED_B02_PRIMARY_SHA256,
    EXPECTED_B1_MANIFEST_SHA256,
    EXPECTED_B22_NATIVE_SUM_DERIVED_ACCURACY,
    EXPECTED_B22_VALIDATION,
    GENERATION_SEEDS,
    HOPS,
    MATCHED_CONFIGS,
    MODEL_NAME,
    MODEL_PATH,
    OFFICIAL_CONFIGS,
    OUT,
    OUTPUT_FORMAT_INSTRUCTION,
    PARSER_SPEC,
    PILOT_SAMPLE_SEED,
    PRIMARY_GENERATION_SEED,
    RELATIONS,
    ROOT,
    canonical_value_hash,
    deterministic_pilot_subset,
    ensure_dirs,
    generation_user_content,
    load_primary,
    primary_invariants,
    read_json,
    row_ids_checksum,
    sha256_file,
    sha256_text,
    stable_json_hash,
    write_json_atomic,
)


TOKENIZER_FILES = [
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def current_weight_manifest(model_path: Path) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing weight index: {index_path}")
    index = read_json(index_path)
    declared = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in declared if not (model_path / name).is_file()]
    shards = [
        {
            "file": name,
            "bytes": (model_path / name).stat().st_size,
            "sha256": sha256_file(model_path / name),
        }
        for name in declared
        if (model_path / name).is_file()
    ]
    return {
        "format": "indexed",
        "index_file": index_path.name,
        "index_sha256": sha256_file(index_path),
        "declared_shards": len(declared),
        "missing_shards": missing,
        "complete": not missing and len(shards) == len(declared),
        "shards": shards,
    }


def audit_model() -> dict:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Required local model missing: {MODEL_PATH}")
    previous = read_json(B22 / "audit" / "model_manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer_files = {
        name: {
            "bytes": (MODEL_PATH / name).stat().st_size,
            "sha256": sha256_file(MODEL_PATH / name),
        }
        for name in TOKENIZER_FILES
        if (MODEL_PATH / name).is_file()
    }
    chat_template = tokenizer.chat_template or ""
    current = {
        "canonical_model_name": MODEL_NAME,
        "local_path": str(MODEL_PATH),
        "config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "tokenizer_files": tokenizer_files,
        "chat_template_present": bool(chat_template),
        "chat_template_sha256": sha256_text(chat_template) if chat_template else None,
        "weight_shard_manifest": current_weight_manifest(MODEL_PATH),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "tokenizer_vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    checks = {
        "canonical_model_name_exact": previous.get("canonical_model_name") == MODEL_NAME,
        "config_sha256_exact_b22": current["config_sha256"] == previous.get("config_sha256"),
        "tokenizer_files_exact_b22": current["tokenizer_files"] == previous.get("tokenizer_files"),
        "chat_template_sha256_exact_b22": current["chat_template_sha256"] == previous.get("chat_template_sha256"),
        "weights_manifest_exact_b22": current["weight_shard_manifest"] == previous.get("weight_shard_manifest"),
        "weights_complete": current["weight_shard_manifest"]["complete"],
        "architecture_exact_b22": current["architectures"] == previous.get("architectures"),
        "layer_count_exact_b22": current["num_hidden_layers"] == previous.get("num_hidden_layers"),
        "hidden_size_exact_b22": current["hidden_size"] == previous.get("hidden_size"),
    }
    current["checks"] = checks
    current["status"] = "PASS" if all(checks.values()) else "FAIL"
    current["b22_model_identity_sha256"] = previous.get("model_identity_sha256")
    current["model_identity_sha256"] = stable_json_hash({
        key: current[key]
        for key in [
            "canonical_model_name", "config_sha256", "tokenizer_files",
            "chat_template_sha256", "weight_shard_manifest", "architectures",
            "num_hidden_layers", "hidden_size", "tokenizer_vocab_size",
        ]
    })
    return current


def audit_environment() -> dict:
    values = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    checks = {
        "transformers_4_57_6": values["transformers"] == "4.57.6",
        "torch_2_4_0_cu124": values["torch"] == "2.4.0+cu124",
        "cuda_12_4": values["torch_cuda"] == "12.4",
        "cuda_available": values["cuda_available"],
        "bf16_supported": values["bf16_supported"],
        "rtx_3090": values["gpu"] == "NVIDIA GeForce RTX 3090",
    }
    values["checks"] = checks
    values["status"] = "PASS" if all(checks.values()) else "FAIL"
    return values


def audit_b22_reference(primary: pd.DataFrame) -> dict:
    validation = read_json(B22 / "validation_summary.json")
    decision = read_json(B22 / "fitness_decision.json")
    metrics = pd.read_csv(B22 / "statistics" / "all_hop_metrics.csv")
    qwen = metrics.loc[metrics.model.eq("qwen3_8b")].copy()
    recovered = {}
    for protocol in ["Raw-Sum", "Native-Sum", "Native-Mean", "EqualLength-Native"]:
        part = qwen.loc[qwen.protocol.eq(protocol)].set_index("hop")
        recovered[protocol] = {
            "accuracy_by_hop": {str(hop): float(part.loc[hop, "accuracy"]) for hop in HOPS if hop in part.index},
            "derived_accuracy_h2_h5": float(part.loc[[2, 3, 4, 5], "accuracy"].mean()) if set([2, 3, 4, 5]).issubset(part.index) else None,
            "derived_macro_f1_h2_h5": float(part.loc[[2, 3, 4, 5], "macro_f1"].mean()) if set([2, 3, 4, 5]).issubset(part.index) else None,
            "derived_balanced_accuracy_h2_h5": float(part.loc[[2, 3, 4, 5], "balanced_accuracy"].mean()) if set([2, 3, 4, 5]).issubset(part.index) else None,
        }
    native_scores = pd.read_csv(B22 / "native_sum" / "row_scores.csv")
    row_level = native_scores[["row_id", "story_group_id", "hop", "gold_relation", "prediction_sum"]].drop_duplicates("row_id")
    checks = {
        "validation_pass": validation.get("status") == EXPECTED_B22_VALIDATION,
        "strong_target_uncertain": decision.get("strong_reasoning_target") == "UNCERTAIN",
        "native_sum_derived_accuracy_reference": abs(recovered["Native-Sum"]["derived_accuracy_h2_h5"] - EXPECTED_B22_NATIVE_SUM_DERIVED_ACCURACY) < 5e-5,
        "row_score_coverage_exact": sorted(row_level.row_id.astype(int).tolist()) == primary.row_id.astype(int).tolist(),
        "row_score_unique": bool(row_level.row_id.is_unique),
        "row_score_truth_exact": bool(row_level.set_index("row_id").loc[primary.row_id, "gold_relation"].to_numpy().tolist() == primary.canonical_relation.tolist()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "validation_status": validation.get("status"),
        "strong_reasoning_target": decision.get("strong_reasoning_target"),
        "recovered_metrics": recovered,
        "native_sum_row_prediction_hash": canonical_value_hash(row_level.sort_values("row_id"), ["row_id", "prediction_sum"]),
    }


def freeze_protected_snapshot() -> dict:
    previous = read_json(B22 / "audit" / "protected_files_audit.json")
    files = {}
    mismatches = []
    for relative, expected in previous["files"].items():
        path = ROOT / relative
        ok = path.is_file() and path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"]
        if not ok:
            mismatches.append(relative)
        files[relative] = expected
    b22_files = sorted(path for path in B22.rglob("*") if path.is_file())
    for path in b22_files:
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    result = {
        "status": "PASS" if not mismatches else "FAIL",
        "algorithm": "sha256",
        "protected_roots": [*previous["protected_roots"], "artifacts/stage_b2_2_qwen3"],
        "file_count": len(files),
        "files": files,
        "preexisting_mismatches": mismatches,
    }
    result["snapshot_sha256"] = stable_json_hash(files)
    return result


def freeze_prompt_parser_and_subset(primary: pd.DataFrame) -> tuple[dict, dict, dict]:
    prompts = [
        json.dumps([int(row.row_id), generation_user_content(row)], ensure_ascii=False, separators=(",", ":"))
        for row in primary.itertuples(index=False)
    ]
    prompt_audit = {
        "status": "PASS",
        "original_content_source": "frozen row Story/Question reconstructed exactly as Stage B2.2 raw prompt",
        "only_appended_text": OUTPUT_FORMAT_INSTRUCTION,
        "output_format_instruction_sha256": sha256_text(OUTPUT_FORMAT_INSTRUCTION),
        "all_user_messages_sha256": sha256_text("\n".join(prompts) + "\n"),
        "rows": len(prompts),
        "forbidden_additions": [],
    }
    parser = {**PARSER_SPEC, "parser_spec_sha256": stable_json_hash(PARSER_SPEC), "status": "FROZEN"}
    pilot = deterministic_pilot_subset(primary)
    subset_manifest = {
        "status": "FROZEN_BEFORE_GENERATION",
        "sampling_seed": PILOT_SAMPLE_SEED,
        "method": "sort rows within hop, draw 50 positions per hop with one sequential numpy.default_rng(seed), then sort by hop,row_id",
        "rows": len(pilot),
        "rows_per_hop": {str(key): int(value) for key, value in pilot.groupby("computed_hop").size().to_dict().items()},
        "row_ids": pilot.row_id.astype(int).tolist(),
        "row_ids_sha256": row_ids_checksum(pilot.row_id),
        "gold_hash": canonical_value_hash(pilot, ["row_id", "canonical_relation"]),
    }
    return prompt_audit, parser, subset_manifest


def freeze_generation_config() -> dict:
    default_generation = GenerationConfig().to_dict()
    min_p_supported = "min_p" in default_generation
    payload = {
        "status": "FROZEN_PENDING_SMOKE_TRUNCATION_AUDIT",
        "stage": "B2.3",
        "model": MODEL_NAME,
        "official_use": OFFICIAL_CONFIGS,
        "matched_decoding": MATCHED_CONFIGS,
        "initial_max_new_tokens": 1024,
        "resolved_max_new_tokens": None,
        "allowed_smoke_resolution": [1024, 2048],
        "min_p_api_supported": min_p_supported,
        "generation_seeds": GENERATION_SEEDS,
        "primary_generation_seed": PRIMARY_GENERATION_SEED,
        "pilot_sampling_seed": PILOT_SAMPLE_SEED,
        "row_rng_policy": "torch manual_seed(child_seed) before every row; child_seed = first 8 hex of sha256(generation_seed:row_id) modulo 2^31-1",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "story_group_id"},
        "decision_thresholds": DECISION_THRESHOLDS,
        "dtype": "torch.bfloat16",
        "quantization": None,
        "model_eval": True,
        "inference_mode": True,
        "batch_size": 1,
    }
    payload["pre_smoke_config_sha256"] = stable_json_hash(payload)
    return payload


def main() -> None:
    ensure_dirs()
    primary = load_primary()
    frozen = primary_invariants(primary)
    frozen["b1_manifest"] = {
        "path": str((B1 / "union_manifest.csv").relative_to(ROOT)),
        "sha256": sha256_file(B1 / "union_manifest.csv"),
        "expected_sha256": EXPECTED_B1_MANIFEST_SHA256,
    }
    frozen["b02_primary"] = {
        "path": str((B02 / "primary_test.csv").relative_to(ROOT)),
        "sha256": sha256_file(B02 / "primary_test.csv"),
        "expected_sha256": EXPECTED_B02_PRIMARY_SHA256,
    }
    frozen["checks"]["b1_manifest_sha256_exact"] = frozen["b1_manifest"]["sha256"] == EXPECTED_B1_MANIFEST_SHA256
    frozen["checks"]["b02_primary_sha256_exact"] = frozen["b02_primary"]["sha256"] == EXPECTED_B02_PRIMARY_SHA256
    frozen["status"] = "PASS" if all(frozen["checks"].values()) else "FAIL"
    write_json_atomic(OUT / "audit" / "frozen_data_audit.json", frozen)

    environment = audit_environment()
    write_json_atomic(OUT / "audit" / "environment.json", environment)
    model = audit_model()
    write_json_atomic(OUT / "audit" / "model_manifest.json", model)
    b22_reference = audit_b22_reference(primary)
    write_json_atomic(OUT / "audit" / "b22_reference_audit.json", b22_reference)
    protected = freeze_protected_snapshot()
    write_json_atomic(OUT / "audit" / "protected_snapshot.json", protected)
    prompt, parser, pilot = freeze_prompt_parser_and_subset(primary)
    write_json_atomic(OUT / "audit" / "prompt_format_audit.json", prompt)
    write_json_atomic(OUT / "audit" / "parser_spec.json", parser)
    write_json_atomic(OUT / "audit" / "pilot_subset_manifest.json", pilot)
    generation = freeze_generation_config()
    write_json_atomic(OUT / "audit" / "generation_config.json", generation)

    checks = {
        "frozen_data_pass": frozen["status"] == "PASS",
        "environment_pass": environment["status"] == "PASS",
        "model_manifest_pass": model["status"] == "PASS",
        "b22_reference_pass": b22_reference["status"] == "PASS",
        "protected_snapshot_pass": protected["status"] == "PASS",
        "prompt_format_frozen": prompt["status"] == "PASS",
        "parser_frozen": parser["status"] == "FROZEN",
        "pilot_subset_frozen_250": pilot["rows"] == 250,
        "min_p_supported": generation["min_p_api_supported"],
    }
    summary = {
        "stage": "B2.3",
        "phase": 0,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": [key for key, value in checks.items() if not value],
    }
    write_json_atomic(OUT / "audit" / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

