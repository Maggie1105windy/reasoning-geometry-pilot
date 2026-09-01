#!/usr/bin/env python3
"""Extract official templates, audit semantics, and measure real dataset matches."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict

from common import (
    ARTIFACTS, COMPLETE, SEED, extract_template_specs, load_json_records,
    match_sentence, sorted_dataset_files, template_index,
)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    specs = extract_template_specs()
    index = template_index(specs)
    by_id = {spec.template_id: spec for spec in specs}
    possible_counts = defaultdict(Counter)
    unambiguous_counts = defaultdict(Counter)
    suspicious_samples = defaultdict(lambda: defaultdict(set))
    unmatched_rows = []
    ambiguous_rows = []
    unmatched_by_variant = Counter()
    ambiguous_by_variant = Counter()
    validation_pool = []
    total_sentences = Counter()
    matched_sentences = Counter()
    usable_sentences = Counter()

    for variant in ["clean", "noise"]:
        for path in sorted_dataset_files(COMPLETE / variant):
            relative = str(path.relative_to(COMPLETE))
            for source_record_id, record in load_json_records(path):
                sample_key = f"{relative}::{source_record_id}"
                for sentence_index, sentence in enumerate(record["story"]):
                    total_sentences[variant] += 1
                    result = match_sentence(sentence, index)
                    if result["matched"]:
                        matched_sentences[variant] += 1
                    if result["usable"]:
                        usable_sentences[variant] += 1
                    candidate_ids = result.get("candidate_template_ids", [])
                    for template_id in candidate_ids:
                        possible_counts[variant][template_id] += 1
                        spec = by_id[template_id]
                        if spec.semantic_status != "OK":
                            suspicious_samples[variant][template_id].add(sample_key)
                    if result.get("template_id") and not result["ambiguous"]:
                        unambiguous_counts[variant][result["template_id"]] += 1
                    row = {
                        "source_file": relative,
                        "source_record_id": source_record_id,
                        "sentence_index": sentence_index,
                        "sentence": sentence,
                        "reason": result["reason"],
                        "candidate_template_ids": json.dumps(candidate_ids),
                    }
                    if not result["matched"]:
                        unmatched_rows.append(row)
                        unmatched_by_variant[variant] += 1
                    if result["ambiguous"]:
                        ambiguous_rows.append(row)
                        ambiguous_by_variant[variant] += 1
                    if len(validation_pool) < 20000 and result["matched"]:
                        validation_pool.append((sentence, result))

    inventory_rows = []
    for spec in specs:
        inventory_rows.append({
            "template_id": spec.template_id,
            "relation": spec.declared_relation,
            "semantic_relation": spec.semantic_relation,
            "template_text_or_pattern": spec.source_text,
            "template_family": spec.template_family,
            "source_file": "Code/template.py",
            "line_or_function": f"line {spec.source_line}; {spec.function_name}",
            "semantic_status": spec.semantic_status,
            "notes": spec.notes,
            "complete_clean_possible_matches": possible_counts["clean"][spec.template_id],
            "complete_clean_unambiguous_matches": unambiguous_counts["clean"][spec.template_id],
            "complete_noise_possible_matches": possible_counts["noise"][spec.template_id],
            "complete_noise_unambiguous_matches": unambiguous_counts["noise"][spec.template_id],
        })
    inventory_fields = list(inventory_rows[0])
    write_csv(ARTIFACTS / "template_inventory.csv", inventory_rows, inventory_fields)

    suspicious_rows = []
    for row in inventory_rows:
        if row["semantic_status"] == "OK":
            continue
        template_id = row["template_id"]
        suspicious_rows.append({
            "template_id": template_id,
            "declared_relation": row["relation"],
            "semantic_relation": row["semantic_relation"],
            "semantic_status": row["semantic_status"],
            "reason": row["notes"],
            "template_text_or_pattern": row["template_text_or_pattern"],
            "affected_clean_sentence_count": row["complete_clean_possible_matches"],
            "affected_clean_sample_count": len(suspicious_samples["clean"][template_id]),
            "affected_noise_sentence_count": row["complete_noise_possible_matches"],
            "affected_noise_sample_count": len(suspicious_samples["noise"][template_id]),
        })
    write_csv(
        ARTIFACTS / "suspicious_templates.csv", suspicious_rows,
        list(suspicious_rows[0]) if suspicious_rows else ["template_id"],
    )
    write_csv(
        ARTIFACTS / "template_unmatched_sentences.csv", unmatched_rows,
        ["source_file", "source_record_id", "sentence_index", "sentence", "reason", "candidate_template_ids"],
    )
    write_csv(
        ARTIFACTS / "template_ambiguous_sentences.csv", ambiguous_rows,
        ["source_file", "source_record_id", "sentence_index", "sentence", "reason", "candidate_template_ids"],
    )

    rng = random.Random(SEED)
    validation_sample = rng.sample(validation_pool, min(500, len(validation_pool)))
    validation_failures = []
    for sentence, result in validation_sample:
        for template_id in result["candidate_template_ids"]:
            if not by_id[template_id].regex.fullmatch(sentence):
                validation_failures.append({"sentence": sentence, "template_id": template_id})
    assert not validation_failures
    summary = {
        "template_definitions": len(specs),
        "unique_surface_skeletons": len(index),
        "non_ok_template_definitions": len(suspicious_rows),
        "semantic_status_counts": dict(Counter(spec.semantic_status for spec in specs)),
        "templates_per_declared_relation": dict(Counter(spec.declared_relation for spec in specs)),
        "total_sentences": dict(total_sentences),
        "matched_sentences": dict(matched_sentences),
        "usable_sentences": dict(usable_sentences),
        "unmatched_sentence_count": len(unmatched_rows),
        "unmatched_sentences_by_variant": dict(unmatched_by_variant),
        "ambiguous_sentence_count": len(ambiguous_rows),
        "ambiguous_sentences_by_variant": dict(ambiguous_by_variant),
        "validation_sample_size": len(validation_sample),
        "validation_failure_count": len(validation_failures),
        "primary_clean_non_ok_affected_sample_union": len(set().union(*suspicious_samples["clean"].values())),
    }
    (ARTIFACTS / "template_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
