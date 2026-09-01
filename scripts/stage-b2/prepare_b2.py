#!/usr/bin/env python3
"""Stage B2.0 fail-closed lineage, frozen-input, protocol, and tokenizer audit."""

from __future__ import annotations

import json
import math
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from common_b2 import (
    ACT_ROOT, BASE_ACT, B1, B156, BOOTSTRAP_RESAMPLES, CANDIDATES, C_GRID,
    HIDDEN_SIZE, HOPS, LATE_LAYERS, MODELS, N_LAYERS, OUT, RANDOM_HOPS,
    RELATIONS, REPRESENTATIVE_LAYERS, SEED, STAGE_PAIRS, ensure_dirs,
    load_union, read_json, sha256, stable_int, write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_safetensor_parameters(path: Path) -> tuple[int, list[str], int]:
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing safetensor shards at {path}: {missing}")
    count = 0
    tensor_names = set()
    for shard in shards:
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_names:
                    raise RuntimeError(f"Duplicate tensor {name} in {path}")
                tensor_names.add(name)
                count += math.prod(handle.get_slice(name).get_shape())
    if set(index["weight_map"]) != tensor_names:
        missing_index = set(index["weight_map"]) - tensor_names
        extra = tensor_names - set(index["weight_map"])
        raise RuntimeError(f"Index/header mismatch {path}: missing={missing_index}, extra={extra}")
    return count, shards, len(tensor_names)


def read_evidence(path: Path, patterns: list[str]) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    output = []
    for number, line in enumerate(lines, 1):
        if any(re.search(pattern, line, flags=re.I) for pattern in patterns):
            output.append({"line": number, "text": line.strip()[:500]})
    return output[:30]


def lineage_audit() -> dict:
    records = []
    for key, spec in MODELS.items():
        raw_path = Path(spec["path"])
        resolved = raw_path.resolve()
        config = AutoConfig.from_pretrained(raw_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(raw_path, local_files_only=True, use_fast=True)
        params, shards, tensors = count_safetensor_parameters(resolved)
        readme = resolved / "README.md"
        patterns = ([r"SFT", r"DPO", r"Final Models", r"RLVR", r"finetuned from", r"PPO settings"]
                    if key != "base" else [r"Model Information", r"pretrained", r"Base"])
        evidence = read_evidence(readme, patterns)
        final_rlvr_confirmed = key == "final" and any("RLVR" in item["text"] for item in evidence)
        model_files = {
            name: {"bytes": (resolved / name).stat().st_size, "sha256": sha256(resolved / name)}
            for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
        }
        records.append({
            "model_key": key, "model_name": spec["label"], "model_path": str(raw_path),
            "resolved_model_path": str(resolved), "path_is_symlink": raw_path.is_symlink(),
            "model_revision": spec["file_revision"],
            "tokenizer_revision": spec["tokenizer_file_revision"],
            "revision_provenance": "ModelScope local .msc file-level revision recovered during B2 audit; not claimed as Hugging Face repository commit",
            "architecture": list(getattr(config, "architectures", [])),
            "model_type": config.model_type, "parameters": int(params),
            "layers": int(config.num_hidden_layers), "hidden_size": int(config.hidden_size),
            "attention_heads": int(config.num_attention_heads),
            "key_value_heads": int(config.num_key_value_heads),
            "intermediate_size": int(config.intermediate_size),
            "dtype": str(getattr(config, "torch_dtype", None)),
            "vocab_size_config": int(config.vocab_size), "vocab_size_tokenizer": int(len(tokenizer)),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "transformers_version_config": getattr(config, "transformers_version", None),
            "transformers_version_runtime": __import__("transformers").__version__,
            "safetensor_shards": shards, "tensor_count": tensors,
            "weight_loading_policy": "use_safetensors=True; local_files_only=True",
            "model_files": model_files, "readme_evidence": evidence,
            "final_rlvr_locally_confirmed": bool(final_rlvr_confirmed),
        })
    architecture_keys = [
        (r["architecture"], r["model_type"], r["layers"], r["hidden_size"],
         r["attention_heads"], r["key_value_heads"], r["intermediate_size"])
        for r in records
    ]
    checks = {
        "four_models_present": len(records) == 4,
        "all_32_decoder_blocks": all(r["layers"] == N_LAYERS for r in records),
        "all_hidden_size_4096": all(r["hidden_size"] == HIDDEN_SIZE for r in records),
        "architecture_dimensions_compatible": len({json.dumps(x, sort_keys=True) for x in architecture_keys}) == 1,
        "parameter_counts_match_base_and_tulu_vocab_extension": (
            {r["model_key"]: r["parameters"] for r in records}
            == {"base": 8030261248, "sft": 8030326784, "dpo": 8030326784, "final": 8030326784}
        ),
        "all_safetensor_manifests_complete": all(len(r["safetensor_shards"]) == 4 for r in records),
        "all_bos_eos_match": len({(r["bos_token_id"], r["eos_token_id"]) for r in records}) == 1,
    }
    payload = {
        "stage": "B2.0", "checked_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "models": records,
        "lineage_naming": {
            "formal_labels": ["Base", "SFT", "DPO", "Final"],
            "final_rlvr_statement": "Local Final README explicitly identifies Final Models (RLVR), states finetuned from DPO, and documents PPO settings for RLVR. Formal tables retain label Final; report may parenthetically state locally documented RLVR.",
        },
        "runtime_python": platform.python_version(),
    }
    write_json(OUT / "model_lineage_audit.json", payload)
    lines = ["# Stage B2.0 Model Lineage Audit", "", f"Status: **{payload['status']}**", "",
             "> Revisions below are local ModelScope file-level revisions recovered from `.msc`; they are not represented as Hugging Face repository commit SHAs.", "",
             "| Model | Path | Resolved path | Params | Layers | Hidden | Config vocab | Tokenizer vocab | Model file revision |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in records:
        lines.append(f"| {r['model_name']} | `{r['model_path']}` | `{r['resolved_model_path']}` | {r['parameters']:,} | {r['layers']} | {r['hidden_size']} | {r['vocab_size_config']} | {r['vocab_size_tokenizer']} | `{r['model_revision']}` |")
    lines += ["", "## Comparability", ""] + [f"- {k}: **{v}**" for k, v in checks.items()]
    lines += ["", "## Final-stage local evidence", "", payload["lineage_naming"]["final_rlvr_statement"], ""]
    (OUT / "model_lineage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError(f"Lineage audit failed: {[k for k,v in checks.items() if not v]}")
    return payload


def frozen_input_audit() -> dict:
    b1_artifacts = read_json(B1 / "stage_b1_artifact_checksums.json")
    artifact_rows = []
    for item in b1_artifacts["files"]:
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        artifact_rows.append({"path": str(path), "expected": item["sha256"], "actual": actual, "match": actual == item["sha256"]})
    split = read_json(B1 / "split_checksum_verification.json")
    manifest_rows = []
    for item in split["files"]:
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        manifest_rows.append({"dataset": item["dataset"], "file": item["file"], "path": str(path),
                              "expected": item["expected_sha256"], "actual": actual,
                              "match": actual == item["expected_sha256"]})
    union = load_union()
    base_integrity = read_json(B1 / "activation_integrity.json")
    saved_ids = np.load(BASE_ACT / "row_ids.npy")
    base_ids_hash = sha256(BASE_ACT / "row_ids.npy")
    prior_gate = read_json(B156 / "input_checksum_verification.json")
    prompt = read_json(B1 / "prompt_protocol.json")
    candidate = read_json(B156 / "behavioral" / "candidate_protocol.json")
    manifest_files = {r["file"] for r in manifest_rows}
    required_prefixes = ["primary_", "balanced_", "composition_heldout_", "target_absent_", "mid_template_test_"]
    checks = {
        "all_stage_b1_artifact_hashes_match": all(r["match"] for r in artifact_rows),
        "all_formal_manifest_hashes_match": all(r["match"] for r in manifest_rows),
        "all_required_datasets_covered": all(any(name.startswith(prefix) for name in manifest_files) for prefix in required_prefixes),
        "base_activation_row_ids_hash_match": base_ids_hash == base_integrity["extraction"]["row_ids_sha256"],
        "base_activation_row_ids_exact_union_order": np.array_equal(saved_ids, union["row_id"].to_numpy(np.int64)),
        "base_union_exactly_13758_unique_rows": len(saved_ids) == len(np.unique(saved_ids)) == len(union) == 13758,
        "prior_b1_pass": read_json(B1 / "stage_b1_gate.json")["status"] == "PASS",
        "prior_b1_5_b1_6_pass": read_json(B156 / "stage_b1_5_1_6_gate.json")["status"] == "PASS",
        "prior_independent_input_audit_pass": prior_gate["status"] == "PASS",
        "raw_prompt_frozen_no_chat_cot_fewshot": prompt["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION" and not prompt["chat_template"] and not prompt["chain_of_thought"] and not prompt["few_shot"],
        "candidate_protocol_frozen": candidate["status"] == "FROZEN_BEFORE_BEHAVIOR_RESULTS" and candidate["candidate_verbalizations"] == CANDIDATES,
    }
    payload = {"stage": "B2.0", "checked_utc": utc_now(), "status": "PASS" if all(checks.values()) else "FAIL",
               "checks": checks, "formal_manifests": manifest_rows, "stage_b1_artifacts": artifact_rows,
               "base_row_ids": {"path": str(BASE_ACT / "row_ids.npy"), "sha256": base_ids_hash, "rows": len(saved_ids)},
               "frozen_protocol_hashes": {"prompt_protocol": sha256(B1 / "prompt_protocol.json"),
                                            "probe_protocol": sha256(B1 / "probe_protocol.json"),
                                            "candidate_protocol": sha256(B156 / "behavioral" / "candidate_protocol.json")}}
    write_json(OUT / "frozen_input_verification.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"Frozen-input audit failed: {[k for k,v in checks.items() if not v]}")
    return payload


def freeze_protocol() -> dict:
    payload = {
        "stage": "B2", "status": "FROZEN_BEFORE_FIRST_TULU_MODEL_LOAD", "frozen_utc": utc_now(),
        "seed": SEED, "execution_order": ["B2.0", "B2.1 SFT", "B2.2 DPO", "B2.3 Final", "B2.4 behavior", "B2.5 within-hop", "B2.6 cross-hop", "B2.7 controls", "B2.8 paired bootstrap/report"],
        "research_variables": {"story_len": 5, "hop_definition": "independently computed BFS graph distance", "explicit_hops": [1], "derived_hops": [2,3,4,5], "classes": RELATIONS, "overlap_excluded": True, "qaK_not_used_as_hop": True},
        "prompt": {"source": str(B1 / "prompt_protocol.json"), "identical_raw_prompt_all_models": True, "chat_template": False, "instruction_wrapper": False, "assistant_prompt": False, "cot": False, "few_shot": False},
        "activation": {"definition": "raw decoder block output before final model RMSNorm", "readout": "question-final non-special token ?", "layers": list(range(32)), "hidden_size": 4096, "storage_dtype": "float16", "union_rows": 13758, "base_not_reextracted": True, "integrity_layers": REPRESENTATIVE_LAYERS, "integrity_rows": 10},
        "behavior": {"candidate_strings": CANDIDATES, "primary_scoring": "summed conditional log-probability", "sensitivity_scoring": "mean conditional log-probability", "test_sets": ["primary", "balanced"], "bootstrap_unit": "story_group_id", "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
        "probe": {"classifier": "Multinomial LogisticRegression", "solver": "lbfgs", "primary_max_iter": 400, "sensitivity_max_iter": [2000,5000], "standard_scaler": "fit train only per model/layer/task", "C_grid": C_GRID, "selection": "max validation Macro-F1; exact ties smaller C; test never used", "primary_class_weight": "balanced", "balanced_class_weight": None, "independent_probe_per_checkpoint": True, "cross_model_probe_transfer": False},
        "summaries": {"all_layer_mean": list(range(32)), "late_layer_mean": LATE_LAYERS, "peak": "descriptive secondary only"},
        "stage_pairs": [{"earlier": a, "later": b} for a,b in STAGE_PAIRS],
        "paired_probe_bootstrap": {"unit": "story_group_id", "resamples": BOOTSTRAP_RESAMPLES, "primary_hops": HOPS, "late_layers": LATE_LAYERS, "base_peak_layers_frozen_from_B1": {"1":17,"2":19,"3":31,"4":30,"5":1}, "representative_layers_frozen": REPRESENTATIVE_LAYERS, "same_group_resample_across_all_models": True},
        "cross_hop": {"primary": "h1 train/val -> h1..h5 test", "secondary": "pooled h2-5 train/val -> h1..h5 test", "primary_comparison": "absolute transfer gap", "retention_interpretability_threshold": "within-hop Macro-F1 > majority baseline + 0.02"},
        "controls": {"balanced": True, "target_absent_reuses_primary_probe": True, "composition_heldout": True, "template_test_only": True, "random_label": {"new_checkpoints_only": ["sft","dpo","final"], "layers": REPRESENTATIVE_LAYERS, "hops": RANDOM_HOPS, "shuffle": "train labels only"}},
        "interpretation_boundary": "Associational linear decodability/behavior/transfer only; no causal, grounding, symbolic-code, or reasoning-layer claim.",
    }
    write_json(OUT / "frozen_b2_protocol.json", payload)
    return payload


def tokenizer_audit() -> dict:
    union = load_union()
    primary = union[union["primary_split"].ne("")].copy()
    selected = []
    for hop in HOPS:
        for relation in RELATIONS:
            cell = primary[(primary["computed_hop"] == hop) & (primary["canonical_relation"] == relation)].copy()
            cell["_order"] = [stable_int("tokenizer-cross-audit", row_id) for row_id in cell["row_id"]]
            selected.extend(cell.nsmallest(3, "_order").index.tolist())
    sample = union.loc[sorted(set(selected))].sort_values(["computed_hop", "canonical_relation", "row_id"]).reset_index(drop=True)
    if len(sample) < 120:
        raise RuntimeError(f"Tokenizer audit has only {len(sample)} rows")
    tokenizers = {}
    for key, spec in MODELS.items():
        tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, use_fast=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        tokenizers[key] = tok
    records = []
    for row in sample.itertuples(index=False):
        per = {}
        for key, tok in tokenizers.items():
            prompt_ids = [int(x) for x in tok(row.prompt, add_special_tokens=False).input_ids]
            ids = [int(tok.bos_token_id)] + prompt_ids
            per[key] = {"token_ids": ids, "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
                        "readout_index": len(ids)-1, "readout_token_id": ids[-1],
                        "readout_token": tok.convert_ids_to_tokens(ids[-1]), "readout_decoded": tok.decode([ids[-1]], skip_special_tokens=False),
                        "sequence_length": len(ids), "readout_is_special": ids[-1] in set(tok.all_special_ids)}
        base = per["base"]
        exact = {key: values["token_ids"] == base["token_ids"] for key, values in per.items()}
        records.append({"row_id": int(row.row_id), "hop": int(row.computed_hop), "relation": row.canonical_relation,
                        "prompt_sha256": __import__("hashlib").sha256(row.prompt.encode()).hexdigest(),
                        "models": per, "token_ids_match_base": exact})
    checks = {
        "at_least_120_rows": len(records) >= 120,
        "all_40_hop_relation_cells_covered": sample.groupby(["computed_hop","canonical_relation"]).ngroups == 40,
        "exact_token_ids_all_checkpoints": all(all(r["token_ids_match_base"].values()) for r in records),
        "bos_identical": len({r["models"][k]["bos_token_id"] for r in records for k in MODELS}) == 1,
        "readout_indices_identical": all(len({v["readout_index"] for v in r["models"].values()}) == 1 for r in records),
        "sequence_lengths_identical": all(len({v["sequence_length"] for v in r["models"].values()}) == 1 for r in records),
        "question_end_is_question_mark": all(v["readout_decoded"] == "?" for r in records for v in r["models"].values()),
        "readout_not_eos_or_special": all(not v["readout_is_special"] and v["readout_token_id"] != v["eos_token_id"] for r in records for v in r["models"].values()),
        "no_truncation": True,
        "no_gold_answer_appended": all(str(row.prompt).endswith(str(row.question).strip()) for row in sample.itertuples(index=False)),
    }
    revisions_differ = len({spec["tokenizer_file_revision"] for spec in MODELS.values()}) > 1
    payload = {"stage": "B2.0", "checked_utc": utc_now(), "status": "PASS" if all(checks.values()) else "FAIL",
               "checks": checks, "sample_count": len(records), "coverage_cells": 40,
               "tokenizer_revisions_differ": revisions_differ,
               "interpretation": "Tokenizer files/revisions and vocabulary sizes differ, but every frozen raw prompt has identical token IDs, BOS, sequence length, readout index, and question-final readout token across all four checkpoints." if all(checks.values()) else "Cross-checkpoint tokenization mismatch detected.",
               "records": records}
    write_json(OUT / "tokenization_cross_checkpoint_audit.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"Tokenizer audit failed: {[k for k,v in checks.items() if not v]}")
    return payload


def main() -> None:
    ensure_dirs()
    lineage = lineage_audit()
    frozen = frozen_input_audit()
    protocol = freeze_protocol()
    token = tokenizer_audit()
    gate = {"stage": "B2.0", "status": "PASS", "checked_utc": utc_now(),
            "lineage": lineage["status"], "frozen_inputs": frozen["status"],
            "protocol": protocol["status"], "tokenization": token["status"],
            "large_model_loaded": False, "safe_to_start_b2_1_sft": True}
    write_json(OUT / "b2_0_gate.json", gate)
    print(json.dumps({"status": "PASS", "models": len(lineage["models"]),
                      "manifest_files": len(frozen["formal_manifests"]),
                      "tokenizer_rows": token["sample_count"],
                      "token_ids_exact": token["checks"]["exact_token_ids_all_checkpoints"],
                      "safe_to_start_b2_1_sft": True}, indent=2))


if __name__ == "__main__":
    main()
