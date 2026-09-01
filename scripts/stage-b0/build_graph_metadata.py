#!/usr/bin/env python3
"""Build full CompleteVersion-clean graph/path metadata without modifying raw data."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter

from common import (
    ARTIFACTS, INVERSE, RELATIONS, extract_template_specs, load_json_records,
    parse_source_file, sorted_dataset_files, template_index,
)
from parse_stepgame import parse_record


FAILURE_FIELDS = [
    "row_id", "source_file", "source_record_id", "story", "question", "raw_label",
    "recorded_hop", "computed_hop", "parse_failure_reasons", "parsed_edges",
]
MISMATCH_FIELDS = [
    "row_id", "source_file", "source_record_id", "story", "question", "raw_label",
    "recorded_hop", "computed_hop", "hop_match", "parsed_edges", "parse_failure_reasons",
]


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    specs = extract_template_specs()
    index = template_index(specs)
    metadata_path = ARTIFACTS / "stepgame_metadata.csv"
    parse_failures_path = ARTIFACTS / "parse_failures.csv"
    hop_mismatch_path = ARTIFACTS / "hop_mismatches.csv"
    counters = Counter()
    start = time.perf_counter()
    fieldnames = None
    row_id = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_stream, \
         parse_failures_path.open("w", newline="", encoding="utf-8") as failure_stream, \
         hop_mismatch_path.open("w", newline="", encoding="utf-8") as mismatch_stream:
        metadata_writer = None
        failure_writer = csv.DictWriter(failure_stream, fieldnames=FAILURE_FIELDS, extrasaction="ignore")
        mismatch_writer = csv.DictWriter(mismatch_stream, fieldnames=MISMATCH_FIELDS, extrasaction="ignore")
        failure_writer.writeheader()
        mismatch_writer.writeheader()

        for path in sorted_dataset_files():
            recorded_hop, split = parse_source_file(path)
            source_file = str(path.relative_to(path.parents[1]))
            for source_record_id, record in load_json_records(path):
                context = {
                    "row_id": row_id,
                    "source_split": split,
                    "source_file": source_file,
                    "source_record_id": source_record_id,
                    "recorded_hop": recorded_hop,
                }
                parsed = parse_record(record, context, index)
                if metadata_writer is None:
                    fieldnames = list(parsed)
                    metadata_writer = csv.DictWriter(
                        metadata_stream, fieldnames=fieldnames, extrasaction="raise"
                    )
                    metadata_writer.writeheader()
                metadata_writer.writerow(parsed)
                counters["input_rows"] += 1
                counters["parse_success"] += int(parsed["parse_success"])
                counters["parse_failure"] += int(not parsed["parse_success"])
                counters["graph_success"] += int(parsed["graph_success"])
                counters["hop_match"] += int(parsed["hop_match"])
                counters["hop_mismatch"] += int(not parsed["hop_match"])
                counters["label_path_match"] += int(parsed["label_matches_computed_relation"] is True)
                counters["contains_non_ok_template"] += int(parsed["contains_suspicious_template"])
                if not parsed["parse_success"] or not parsed["graph_success"]:
                    failure_writer.writerow(parsed)
                if not parsed["hop_match"]:
                    mismatch_writer.writerow(parsed)

                assert parsed["row_id"] == row_id
                assert parsed["canonical_relation"] in RELATIONS
                assert parsed["inverse_relation"] == INVERSE[parsed["canonical_relation"]]
                if parsed["parse_success"]:
                    assert parsed["query_entities_in_graph"]
                if parsed["computed_hop"] is not None:
                    assert parsed["computed_hop"] >= 1
                if parsed["computed_hop"] == 1:
                    assert json.loads(parsed["path_relations"])[0] in RELATIONS
                assert parsed["primary_candidate_reason"]
                row_id += 1

    assert counters["input_rows"] == row_id
    assert counters["parse_success"] + counters["parse_failure"] == row_id
    summary = {
        **counters,
        "parse_success_rate": counters["parse_success"] / row_id,
        "hop_match_rate": counters["hop_match"] / row_id,
        "runtime_seconds": time.perf_counter() - start,
        "metadata_columns": fieldnames,
        "metadata_path": str(metadata_path),
    }
    (ARTIFACTS / "graph_build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
