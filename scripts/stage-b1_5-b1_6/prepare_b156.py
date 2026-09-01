#!/usr/bin/env python3
"""Fail-closed Stage B1.5-B1.6 input checksum and row-identity gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common_b156 import ACT_DIR, B02, B1, OUT, ensure_dirs, load_union, read_json, sha256, write_json


def main() -> None:
    ensure_dirs()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checksum_manifest = read_json(B1 / "stage_b1_artifact_checksums.json")
    artifact_rows = []
    for item in checksum_manifest["files"]:
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        artifact_rows.append({
            "path": str(path), "expected_sha256": item["sha256"],
            "actual_sha256": actual, "match": actual == item["sha256"],
        })
    checks["all_stage_b1_checksum_manifest_entries_match"] = all(row["match"] for row in artifact_rows)
    details["stage_b1_artifacts"] = artifact_rows

    split_manifest = read_json(B1 / "split_checksum_verification.json")
    split_rows = []
    for item in split_manifest["files"]:
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        split_rows.append({
            "dataset": item["dataset"], "path": str(path),
            "expected_sha256": item["expected_sha256"], "actual_sha256": actual,
            "match": actual == item["expected_sha256"],
        })
    checks["all_formal_manifest_hashes_recomputed_match"] = all(row["match"] for row in split_rows)
    details["formal_manifests"] = split_rows

    integrity = read_json(B1 / "activation_integrity.json")
    extraction = integrity["extraction"]
    activation_rows = []
    for layer in range(32):
        item = extraction["layer_files"][str(layer)]
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        activation_rows.append({
            "layer": layer, "path": str(path), "expected_sha256": item["sha256"],
            "actual_sha256": actual, "match": actual == item["sha256"],
        })
    checks["all_32_activation_shards_match"] = len(activation_rows) == 32 and all(row["match"] for row in activation_rows)
    details["activation_shards"] = activation_rows

    union = load_union()
    row_ids_path = ACT_DIR / "row_ids.npy"
    saved_ids = np.load(row_ids_path)
    row_ids_hash = sha256(row_ids_path)
    checks["activation_row_ids_hash_match"] = row_ids_hash == extraction["row_ids_sha256"]
    checks["activation_row_ids_exact_union_order"] = np.array_equal(saved_ids, union["row_id"].to_numpy(np.int64))
    checks["activation_row_ids_unique"] = len(saved_ids) == len(np.unique(saved_ids)) == len(union) == 13758
    details["row_ids"] = {
        "path": str(row_ids_path), "expected_sha256": extraction["row_ids_sha256"],
        "actual_sha256": row_ids_hash, "rows": int(len(saved_ids)),
    }

    row_id_checks = []
    for dataset, column in [("primary", "primary_split"), ("balanced", "balanced_split")]:
        for split in ["train", "val", "test"]:
            manifest = pd.read_csv(B02 / f"{dataset}_{split}.csv")
            expected_ids = manifest["row_id"].astype(np.int64).to_numpy()
            actual_ids = union.loc[union[column].eq(split), "row_id"].astype(np.int64).to_numpy()
            exact_set = set(expected_ids.tolist()) == set(actual_ids.tolist())
            no_duplicates = len(expected_ids) == len(set(expected_ids.tolist()))
            row_id_checks.append({
                "dataset": dataset, "split": split, "manifest_rows": int(len(expected_ids)),
                "union_rows": int(len(actual_ids)), "exact_row_id_set": exact_set,
                "manifest_unique": no_duplicates,
            })
    checks["primary_balanced_row_id_sets_exact"] = all(
        row["exact_row_id_set"] and row["manifest_unique"] for row in row_id_checks
    )
    details["row_id_split_checks"] = row_id_checks

    prompt_protocol = read_json(B1 / "prompt_protocol.json")
    checks["prompt_protocol_frozen"] = (
        prompt_protocol["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION"
        and prompt_protocol["chat_template"] is False
        and prompt_protocol["few_shot"] is False
        and prompt_protocol["chain_of_thought"] is False
        and prompt_protocol["gold_relation_in_prompt"] is False
        and prompt_protocol["tokenization"]["explicit_bos"] is True
        and prompt_protocol["tokenization"]["add_special_tokens"] is False
    )
    checks["stage_b1_gate_pass"] = read_json(B1 / "stage_b1_gate.json")["status"] == "PASS"
    checks["stage_b1_validation_pass"] = read_json(B1 / "validation_pre_report.json")["status"] == "PASS"

    checks = {name: bool(value) for name, value in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "stage": "B1.5-B1.6", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL", "checks": checks,
        "failed_checks": failures, "details": details,
    }
    write_json(OUT / "input_checksum_verification.json", payload)
    print(json.dumps({
        "status": payload["status"], "checks": len(checks), "failed_checks": failures,
        "stage_b1_artifacts": len(artifact_rows), "formal_manifests": len(split_rows),
        "activation_shards": len(activation_rows), "row_id_split_checks": row_id_checks,
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
