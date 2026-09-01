#!/usr/bin/env python3
"""Pre-model Stage B1 gates: checksums, prompt freeze, union, tokenization."""

from __future__ import annotations

import json
import platform
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common_b1 import (
    B02, B02C, C_GRID, DATASETS, HOPS, MODEL_PATH, MODEL_REVISION, OUT,
    RELATIONS, SEED, SPLITS, TOKENIZER_REVISION, build_prompt,
    ensure_output_dirs, load_manifest, load_metadata, read_json, sha256,
    stable_int, write_json,
)


def verify_checksums() -> dict:
    b02_expected = read_json(B02 / "split_checksums.json")
    b02c_expected = read_json(B02C / "template_test_checksums.json")
    required_b02_prefixes = ["primary_", "balanced_", "composition_heldout_", "target_absent_"]
    records = []
    for name, expected in sorted(b02_expected["per_file"].items()):
        if not any(name.startswith(prefix) for prefix in required_b02_prefixes):
            continue
        path = B02 / name
        actual = sha256(path)
        records.append({
            "dataset": name.split("_")[0], "path": str(path), "file": name,
            "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected,
        })
    for name, expected in sorted(b02c_expected["adopted_manifests"].items()):
        path = B02C / name
        actual = sha256(path)
        records.append({
            "dataset": "template_test_only", "path": str(path), "file": name,
            "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected,
        })
    protected_current = {
        name: sha256(B02 / name)
        for name in sorted(b02c_expected["protected_b0_2_before"])
    }
    broad_current = {
        name: sha256(B02C.parent / "stage_b0_2b" / name)
        for name in sorted(b02c_expected["broad_restricted_before"])
    }
    payload = {
        "stage": "B1 pre-model checksum gate",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA-256",
        "files": records,
        "all_formal_manifest_files_match": all(record["match"] for record in records),
        "protected_b0_2_match_b0_2c_snapshot": protected_current == b02c_expected["protected_b0_2_before"],
        "broad_restricted_match_b0_2c_snapshot": broad_current == b02c_expected["broad_restricted_before"],
    }
    payload["status"] = "PASS" if all([
        payload["all_formal_manifest_files_match"],
        payload["protected_b0_2_match_b0_2c_snapshot"],
        payload["broad_restricted_match_b0_2c_snapshot"],
    ]) else "FAIL"
    write_json(OUT / "split_checksum_verification.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("Split checksum verification failed; model work is forbidden")
    return payload


def freeze_protocols(tokenizer) -> None:
    prompt_protocol = {
        "stage": "B1",
        "status": "FROZEN_BEFORE_ACTIVATION_EXTRACTION",
        "model_scope": "Meta-Llama-3.1-8B Base only",
        "prompt_template": "Story:\n{one original StepGame sentence per line}\n\nQuestion:\n{original StepGame question}",
        "story_handling": "parse official JSON sentence list; preserve sentence text and order; join with newline",
        "question_handling": "preserve official question text; strip only outer whitespace",
        "chat_template": False,
        "few_shot": False,
        "chain_of_thought": False,
        "answer_choices": False,
        "gold_relation_in_prompt": False,
        "assistant_prefix": False,
        "tokenization": {
            "add_special_tokens": False,
            "explicit_bos": True,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_as_pad": True,
            "pad_token_id": tokenizer.eos_token_id,
            "padding_side": "right",
            "truncation": False,
        },
        "readout": {
            "definition": "last non-special token at the end of the complete question",
            "position": "len([BOS] + tokenizer(prompt, add_special_tokens=False).input_ids) - 1",
            "answer_tokens_seen": False,
            "gold_label_seen": False,
            "pooling": "none",
        },
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "seed": SEED,
    }
    probe_protocol = {
        "status": "FROZEN_BEFORE_PROBE_RESULTS",
        "classifier": "sklearn LogisticRegression multinomial loss, solver=lbfgs",
        "feature_scaling": "per experiment/layer StandardScaler fit on train only",
        "C_grid": C_GRID,
        "C_selection": "maximum validation Macro-F1; exact ties choose smaller C; test never used",
        "primary_class_weight": "balanced",
        "balanced_class_weight": None,
        "composition_class_weight": "balanced",
        "template_class_weight": "balanced",
        "metrics": ["macro_f1", "accuracy"],
        "random_label": {
            "scope": "Primary train labels only",
            "validation_and_test_labels": "real and unchanged",
            "layers": [0, 7, 15, 23, 31],
            "hops": [1, 3, 5],
            "seed": SEED,
        },
        "layer_summary": {
            "control": "per-hop Primary test majority-class Macro-F1",
            "normalized_gain": "(F_layer - F_control) / (F_peak - F_control)",
            "L50": "first layer with normalized_gain >= 0.50",
            "L90": "first layer with normalized_gain >= 0.90 and that layer plus next two available layers all >= 0.85; through layer 31 when fewer than two remain",
            "rules_frozen_before_results": True,
        },
        "bootstrap": {
            "unit": "story_group_id",
            "resamples": 1000,
            "confidence": 0.95,
            "scope": "Primary peak layer for each hop",
            "seed": SEED,
        },
        "interpretation_boundary": "linear decodability only; no causal, grounding-layer, symbolic-reasoning, or step-localization claim",
    }
    write_json(OUT / "prompt_protocol.json", prompt_protocol)
    write_json(OUT / "probe_protocol.json", probe_protocol)


def build_union(tokenizer) -> tuple[pd.DataFrame, dict]:
    metadata = load_metadata().set_index("row_id", drop=False)
    memberships: dict[str, dict[int, str]] = {}
    dataset_counts = {}
    for dataset in DATASETS:
        split_map = {}
        split_counts = {}
        for split in SPLITS:
            manifest = load_manifest(dataset, split)
            split_counts[split] = int(len(manifest))
            for row_id in manifest["row_id"].astype(int):
                if row_id in split_map:
                    raise AssertionError(f"Duplicate row within {dataset}: {row_id}")
                split_map[int(row_id)] = split
        memberships[dataset] = split_map
        dataset_counts[dataset] = split_counts

    extraction_datasets = ["primary", "balanced", "composition", "template"]
    union_ids = sorted(set().union(*(set(memberships[name]) for name in extraction_datasets)))
    target_ids = set(memberships["target_absent"])
    if not target_ids <= set(memberships["primary"]):
        raise AssertionError("Target-Absent is not a Primary subset")
    source = metadata.loc[union_ids].copy().reset_index(drop=True)
    frame = source[[
        "row_id", "story_group_id", "exact_duplicate_group", "computed_hop",
        "canonical_relation", "story", "question", "target_relation_present_anywhere",
        "target_absent_anywhere", "unordered_composition_signature",
    ]].copy()
    frame["computed_hop"] = frame["computed_hop"].astype(int)
    for dataset in DATASETS:
        frame[f"{dataset}_split"] = frame["row_id"].map(memberships[dataset]).fillna("")

    prompts, token_ids, readout_indices, readout_ids, readout_tokens, seq_lengths = [], [], [], [], [], []
    audit_flags = []
    offset_terminal_exact = []
    for row in frame.itertuples(index=False):
        prompt = build_prompt(row.story, row.question)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(tokenizer.bos_token_id)] + [int(value) for value in encoded["input_ids"]]
        offsets = [(0, 0)] + [tuple(map(int, value)) for value in encoded["offset_mapping"]]
        readout = len(ids) - 1
        readout_id = ids[readout]
        question_text = str(row.question).strip()
        last_offset = offsets[readout]
        decoded_prompt = tokenizer.decode(ids[1:], skip_special_tokens=False)
        normalize = lambda value: unicodedata.normalize("NFKC", value).strip()
        decoded_readout = tokenizer.decode([readout_id], skip_special_tokens=False)
        flags = {
            "readout_last": readout == len(ids) - 1,
            "readout_not_bos": readout_id != tokenizer.bos_token_id,
            "readout_not_eos": readout_id != tokenizer.eos_token_id,
            "readout_not_pad": readout_id != tokenizer.pad_token_id,
            "readout_not_special": readout_id not in set(tokenizer.all_special_ids),
            "readout_in_question": prompt.endswith(question_text) and decoded_readout == question_text[-1:],
            "no_truncation": normalize(decoded_prompt) == normalize(prompt),
            "gold_answer_absent": prompt.endswith(question_text) and not prompt.endswith(str(row.canonical_relation)),
        }
        prompts.append(prompt)
        token_ids.append(json.dumps(ids, separators=(",", ":")))
        readout_indices.append(readout)
        readout_ids.append(readout_id)
        readout_tokens.append(tokenizer.convert_ids_to_tokens(readout_id))
        seq_lengths.append(len(ids))
        audit_flags.append(flags)
        offset_terminal_exact.append(last_offset[1] == len(prompt))
    frame["prompt"] = prompts
    frame["token_ids"] = token_ids
    frame["readout_index"] = readout_indices
    frame["readout_token_id"] = readout_ids
    frame["readout_token"] = readout_tokens
    frame["sequence_length"] = seq_lengths
    frame.to_csv(OUT / "union_manifest.csv", index=False)

    flag_frame = pd.DataFrame(audit_flags)
    primary = frame.loc[frame["primary_split"].ne("")].copy()
    sample_indices = []
    for hop in HOPS:
        for relation in RELATIONS:
            cell = primary.loc[
                primary["computed_hop"].eq(hop) & primary["canonical_relation"].eq(relation)
            ].copy()
            cell["_audit_order"] = [stable_int("token-audit", row_id) for row_id in cell["row_id"]]
            sample_indices.extend(cell.nsmallest(3, "_audit_order").index.tolist())
    sample_indices = sorted(set(sample_indices))
    audit_sample_flags = flag_frame.loc[sample_indices]
    checks = {column: bool(audit_sample_flags[column].all()) for column in audit_sample_flags.columns}
    coverage = frame.loc[sample_indices].groupby(["computed_hop", "canonical_relation"]).size()
    checks["sample_count_at_least_100"] = len(sample_indices) >= 100
    checks["all_40_cells_covered"] = len(coverage) == 40 and bool((coverage > 0).all())
    checks["all_union_rows_tokenize_safely"] = bool(flag_frame.all(axis=None))
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": SEED,
        "audit_sample_count": len(sample_indices),
        "coverage_cells": len(coverage),
        "checks": checks,
        "union_rows": int(len(frame)),
        "dataset_split_counts": dataset_counts,
        "target_absent_is_primary_subset": True,
        "sequence_length": {
            "min": int(frame["sequence_length"].min()),
            "median": float(frame["sequence_length"].median()),
            "p95": float(frame["sequence_length"].quantile(0.95)),
            "max": int(frame["sequence_length"].max()),
            "model_max_length": int(tokenizer.model_max_length),
        },
        "readout_token_counts": frame.loc[sample_indices, "readout_token"].value_counts().to_dict(),
        "offset_mapping_terminal_exact_rows": int(sum(offset_terminal_exact)),
        "offset_mapping_terminal_inexact_rows": int(len(offset_terminal_exact) - sum(offset_terminal_exact)),
        "offset_mapping_role": "diagnostic only; readout/no-truncation gates use token-ID decode equality because Llama fast-tokenizer terminal offsets can be zero-length",
        "audit_row_ids": frame.loc[sample_indices, "row_id"].astype(int).tolist(),
    }
    write_json(OUT / "tokenization_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Tokenization Gate failed: {[key for key, value in checks.items() if not value]}")

    examples = frame.loc[sample_indices].sort_values(
        ["computed_hop", "canonical_relation", "row_id"], kind="mergesort"
    ).groupby("computed_hop", sort=True).head(4)
    lines = [
        "# Stage B1 Tokenization Examples", "", "Status: **PASS**", "",
        f"Audited {len(sample_indices)} rows: three deterministic samples per hop×relation cell.", "",
    ]
    for row in examples.itertuples(index=False):
        ids = json.loads(row.token_ids)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        lines += [
            f"## row {row.row_id} — h={row.computed_hop}, {row.canonical_relation}", "",
            "```text", row.prompt, "```", "",
            f"- sequence length: {row.sequence_length}",
            f"- readout index: {row.readout_index}",
            f"- readout token: `{row.readout_token}` (id={row.readout_token_id})",
            f"- token IDs: `{ids}`",
            f"- tokens: `{tokens}`", "",
        ]
    (OUT / "tokenization_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame, audit


def main() -> None:
    ensure_output_dirs()
    checksum = verify_checksums()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer lacks BOS/EOS")
    freeze_protocols(tokenizer)
    union, token_audit = build_union(tokenizer)
    print(json.dumps({
        "status": "PASS",
        "checksum_gate": checksum["status"],
        "tokenization_gate": token_audit["status"],
        "union_rows": len(union),
        "max_sequence_length": token_audit["sequence_length"]["max"],
        "runtime_python": platform.python_version(),
    }, indent=2))


if __name__ == "__main__":
    main()
