#!/usr/bin/env python3
"""Freeze StepGame provenance, real file counts, and human-readable schema examples."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACTS, COMPLETE, STEPGAME, TEMPLATE_SOURCE, load_json_records, parse_source_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_json(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(STEPGAME), "rev-parse", "HEAD"], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "-C", str(STEPGAME), "remote", "get-url", "origin"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(STEPGAME), "status", "--porcelain"], text=True
    ).strip()
    assert status == ""

    inventory_rows = []
    provenance_files = []
    complete_counts = defaultdict(lambda: Counter())
    for path in sorted(COMPLETE.glob("*/*.json")):
        variant = path.parent.name
        recorded_hop, split = parse_source_file(path)
        count = count_json(path)
        complete_counts[variant][split] += count
        complete_counts[variant]["total"] += count
        row = {
            "dataset_variant": "CompleteVersion",
            "clean_or_noise": variant,
            "source_split": split,
            "recorded_hop": recorded_hop,
            "path": str(path.relative_to(STEPGAME)),
            "n_records": count,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        inventory_rows.append(row)
        provenance_files.append({
            "path": row["path"], "size_bytes": row["size_bytes"],
            "sha256": row["sha256"], "n_records": count,
        })

    train_version = STEPGAME / "Dataset/TrainVersion"
    train_inventory = []
    for path in sorted(train_version.glob("*.json")):
        count = count_json(path)
        train_inventory.append({
            "path": str(path.relative_to(STEPGAME)),
            "n_records": count,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    for path in [TEMPLATE_SOURCE, STEPGAME / "Code/parameterized_step_game_8relation.py", STEPGAME / "README.md"]:
        provenance_files.append({
            "path": str(path.relative_to(STEPGAME)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    with (ARTIFACTS / "dataset_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    provenance = {
        "dataset": "StepGame",
        "source": "Official StepGame GitHub repository; 2024-02 regenerated dataset",
        "repository_url": remote,
        "git_commit": commit,
        "git_clean_at_audit": True,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "dataset_variant": "CompleteVersion",
        "clean_or_noise": "clean (Primary); noise inventoried but not used for Primary metadata",
        "official_readme_latest_dataset_note": "2024-03 README recommends Hugging Face; HF exposes TrainVersion-sized consolidated splits, while repository also carries CompleteVersion.",
        "actual_complete_version_counts": {
            variant: dict(counts) for variant, counts in complete_counts.items()
        },
        "train_version_json_inventory": train_inventory,
        "files": provenance_files,
    }
    (ARTIFACTS / "data_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    example_specs = [
        ("clean/qa1_train.json", "0"),
        ("clean/qa2_valid.json", "1"),
        ("clean/qa3_test.json", "2"),
        ("clean/qa5_train.json", "3"),
        ("clean/qa5_train.json", "6"),
        ("clean/qa7_valid.json", "4"),
        ("clean/qa10_test.json", "5"),
        ("noise/qa5_train.json", "0"),
    ]
    lines = [
        "# StepGame schema examples", "",
        "## Schema interpretation", "",
        "- Each source file is a JSON object keyed by a stringified local record number.",
        "- `story` is a list of natural-language relation sentences, not one string.",
        "- `question` asks for the relation of its first named agent (`query_arg1`) to its second (`query_arg2`).",
        "- `label` is one of nine raw answer labels.",
        "- No per-record hop or template metadata field exists in the repository JSON.",
        "- `recorded_hop` is therefore recovered from `qaK_*`; in clean files K also equals story length, but it must not be assumed to equal query shortest-path distance.",
        "- `computed_hop`, explicit edges, template identity, path metadata, inverse label, and grouping identifiers are audit-derived fields.", "",
    ]
    for number, (relative, local_id) in enumerate(example_specs, 1):
        path = COMPLETE / relative
        payload = json.loads(path.read_text(encoding="utf-8"))[local_id]
        recorded_hop, split = parse_source_file(path)
        lines += [
            f"## Example {number}: `{relative}` record `{local_id}`", "",
            f"- Source split: `{split}`",
            f"- Filename K (`recorded_hop`): `{recorded_hop}`",
            f"- Story type/length: `list` / `{len(payload['story'])}`",
            f"- Question: `{payload['question']}`",
            f"- Raw label: `{payload['label']}`",
            "- Story:", "",
        ]
        for index, sentence in enumerate(payload["story"], 1):
            lines.append(f"  {index}. {sentence}")
        lines.append("")
    (ARTIFACTS / "schema_examples.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "commit": commit,
        "complete_counts": {key: dict(value) for key, value in complete_counts.items()},
        "files_hashed": len(provenance_files),
        "schema_examples": len(example_specs),
    }, indent=2))


if __name__ == "__main__":
    main()
