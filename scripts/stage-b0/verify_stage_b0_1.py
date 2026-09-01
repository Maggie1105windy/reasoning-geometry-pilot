#!/usr/bin/env python3
"""Independent artifact-level QA for the completed Stage B0.1 run."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, STEPGAME


REQUIRED = [
    "data_provenance.json", "schema_examples.md", "relation_mapping.csv",
    "template_inventory.csv", "suspicious_templates.csv", "hop_mismatches.csv",
    "parse_failures.csv", "story_len_by_hop.csv", "hop_relation_counts.csv",
    "composition_counts.csv", "target_absent_counts.csv", "primary_candidate_counts.csv",
    "stepgame_metadata.parquet", "stepgame_metadata.csv", "B0_1_DATA_AUDIT_REPORT.md",
]


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def main():
    metadata_csv = pd.read_csv(
        ARTIFACTS / "stepgame_metadata.csv",
        usecols=["row_id", "source_file", "source_record_id", "canonical_relation"],
        low_memory=False,
    )
    metadata_parquet = pd.read_parquet(
        ARTIFACTS / "stepgame_metadata.parquet",
        columns=["row_id", "source_file", "source_record_id", "canonical_relation"],
    )
    relation = pd.read_csv(ARTIFACTS / "relation_mapping.csv")
    templates = pd.read_csv(ARTIFACTS / "template_inventory.csv")
    suspicious = pd.read_csv(ARTIFACTS / "suspicious_templates.csv")
    hop_relation = pd.read_csv(ARTIFACTS / "hop_relation_counts.csv")
    summary = json.loads((ARTIFACTS / "summary_metrics.json").read_text())
    report = (ARTIFACTS / "B0_1_DATA_AUDIT_REPORT.md").read_text()

    checks = {
        "required_files_nonempty": all((ARTIFACTS / name).is_file() and (ARTIFACTS / name).stat().st_size > 0 for name in REQUIRED),
        "csv_rows_210000": len(metadata_csv) == 210000,
        "parquet_rows_210000": len(metadata_parquet) == 210000,
        "csv_parquet_key_columns_equal": metadata_csv.equals(metadata_parquet),
        "row_ids_unique": metadata_csv.row_id.is_unique,
        "relation_counts_sum": int(relation["count"].sum()) == 210000,
        "template_definitions_214": len(templates) == 214,
        "non_ok_templates_11": len(suspicious) == 11,
        "parse_failure_rows_match": csv_rows(ARTIFACTS / "parse_failures.csv") == summary["parse_failure"],
        "hop_mismatch_rows_match": csv_rows(ARTIFACTS / "hop_mismatches.csv") == summary["hop_mismatch"],
        "hop_relation_computable_sum": int(hop_relation["count"].sum()) == summary["graph_success"],
        "all_summary_validations_pass": all(summary["validations"].values()),
        "report_has_protocol_warning": "qaK` is not the actual query hop" in report,
        "report_has_balanced_ceiling": "template-clean eight-class balanced ceiling" in report,
        "raw_repository_clean": subprocess.check_output(
            ["git", "-C", str(STEPGAME), "status", "--porcelain"], text=True
        ).strip() == "",
    }
    artifact_manifest = []
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file():
            artifact_manifest.append({
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "status": status,
        "failed_checks": [key for key, value in checks.items() if not value],
        "checks": checks,
        "artifact_count": len(artifact_manifest),
        "total_artifact_bytes": sum(item["size_bytes"] for item in artifact_manifest),
        "artifacts": artifact_manifest,
    }
    (ARTIFACTS / "artifact_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
