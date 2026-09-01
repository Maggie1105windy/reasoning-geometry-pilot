#!/usr/bin/env python3
"""Parse StepGame records into explicit graph/path metadata; includes small-sample QA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import (
    ARTIFACTS, INVERSE, NON_OVERLAP, RAW_TO_CANONICAL, RELATIONS, SEED,
    all_shortest_paths, build_adjacency, composed_relation, extract_template_specs,
    json_or_none, load_json_records, match_sentence, normalize_entities,
    ordered_signature, parse_source_file, query_entities, sorted_dataset_files,
    stable_hash, template_index, unordered_signature,
)


def parse_record(record: dict, context: dict, index) -> dict:
    story = record["story"]
    question = record["question"]
    raw_label = record["label"]
    if raw_label not in RAW_TO_CANONICAL:
        raise ValueError(f"unknown raw label: {raw_label}")
    canonical = RAW_TO_CANONICAL[raw_label]
    arg1, arg2 = query_entities(question)
    matches = [match_sentence(sentence, index) for sentence in story]
    edges = []
    for sentence_index, (sentence, match) in enumerate(zip(story, matches)):
        if match["usable"]:
            edges.append({
                "sentence_index": sentence_index,
                "sentence": sentence,
                "arg1": match["arg1"],
                "arg2": match["arg2"],
                "relation": match["relation"],
                "template_id": match["template_id"],
            })
    adjacency = build_adjacency(edges)
    paths, paths_truncated = all_shortest_paths(adjacency, arg1, arg2)
    n_paths = len(paths)
    computed_hop = len(paths[0][1]) if paths else None
    path_unique = bool(n_paths == 1 and not paths_truncated)
    path_entities = paths[0][0] if path_unique else None
    path_relations = paths[0][1] if path_unique else None
    path_composed = [composed_relation(relations) for _nodes, relations in paths]
    computed_relation = path_composed[0] if path_composed and len(set(path_composed)) == 1 else None
    stated_relations = [edge["relation"] for edge in edges]
    all_path_relations = [relations for _nodes, relations in paths]
    target_present_anywhere = canonical in stated_relations
    target_present_on_path = any(
        canonical in relations for relations in all_path_relations
    ) if paths else None
    parse_success = all(match["usable"] for match in matches)
    query_entities_in_graph = arg1 in adjacency and arg2 in adjacency
    graph_success = bool(paths)
    failure_reasons = []
    if not parse_success:
        failure_reasons.extend(
            f"sentence_{i}:{match['reason']}"
            for i, match in enumerate(matches) if not match["usable"]
        )
    if not query_entities_in_graph:
        failure_reasons.append("QUERY_ENTITY_MISSING_FROM_PARSED_GRAPH")
    if not graph_success:
        failure_reasons.append("NO_QUERY_PATH_IN_PARSED_GRAPH")
    if paths_truncated:
        failure_reasons.append("SHORTEST_PATH_ENUMERATION_TRUNCATED")

    template_ids = [
        match.get("template_id") or "|".join(match.get("candidate_template_ids", []))
        for match in matches
    ]
    template_families = [match.get("template_family") for match in matches]
    statuses = [status for match in matches for status in match.get("candidate_statuses", [])]
    contains_error = "ERROR" in statuses
    contains_unknown = "UNKNOWN" in statuses
    contains_non_ok = any(status != "OK" for status in statuses)

    entities = sorted(set(
        entity for sentence in story for entity in __import__("re").findall(r"\b[A-Z]\b", sentence)
    ) | {arg1, arg2})
    normalized_story, normalized_question = normalize_entities(story, question)
    story_group_id = stable_hash(story, "story::")
    exact_duplicate_group = stable_hash([story, question, raw_label], "exact::")
    normalized_duplicate_group = stable_hash(
        [normalized_story, normalized_question, raw_label], "normalized::"
    )
    graph_template_payload = sorted(
        (edge["template_id"], edge["relation"], edge["arg1"], edge["arg2"])
        for edge in edges
    )
    graph_template_group = stable_hash(graph_template_payload, "graph_template_heuristic::")

    primary_scope = bool(
        len(story) == 5 and computed_hop in {1, 2, 3, 4, 5}
        and canonical in NON_OVERLAP and parse_success and graph_success
    )
    structural_primary_candidate = bool(primary_scope and not contains_error and not contains_unknown)
    primary_candidate = bool(structural_primary_candidate and context["recorded_hop"] == computed_hop)
    candidate_reasons = []
    if len(story) != 5:
        candidate_reasons.append("story_len_not_5")
    if computed_hop not in {1, 2, 3, 4, 5}:
        candidate_reasons.append("computed_hop_not_1_to_5")
    if canonical not in NON_OVERLAP:
        candidate_reasons.append("overlap_target")
    if not parse_success:
        candidate_reasons.append("parse_failure")
    if not graph_success:
        candidate_reasons.append("no_query_path")
    if contains_error or contains_unknown:
        candidate_reasons.append("non_ok_template")
    if context["recorded_hop"] != computed_hop:
        candidate_reasons.append("recorded_hop_differs_from_query_distance")
    if not candidate_reasons:
        candidate_reasons.append("eligible")

    ordered = ordered_signature(path_relations) if path_unique and computed_hop and computed_hop >= 2 else None
    unordered = unordered_signature(path_relations) if path_unique and computed_hop and computed_hop >= 2 else None
    return {
        **context,
        "story": json.dumps(story, ensure_ascii=False),
        "question": question,
        "raw_label": raw_label,
        "canonical_relation": canonical,
        "story_len": len(story),
        "computed_hop": computed_hop,
        "hop_match": context["recorded_hop"] == computed_hop if computed_hop is not None else False,
        "query_arg1": arg1,
        "query_arg2": arg2,
        "inverse_relation": INVERSE[canonical],
        "n_entities": len(entities),
        "n_edges": len(edges),
        "story_group_id": story_group_id,
        "path_unique": path_unique,
        "n_shortest_paths": n_paths,
        "paths_truncated": paths_truncated,
        "path_entities": json_or_none(path_entities),
        "path_relations": json_or_none(path_relations),
        "all_shortest_path_entities": json_or_none([nodes for nodes, _relations in paths]),
        "all_shortest_path_relations": json_or_none(all_path_relations),
        "computed_relation_from_path": computed_relation,
        "label_matches_computed_relation": canonical == computed_relation if computed_relation else None,
        "ordered_composition_signature": ordered,
        "unordered_composition_signature": unordered,
        "target_relation_present_anywhere": target_present_anywhere,
        "target_relation_present_on_query_path": target_present_on_path,
        "target_absent_anywhere": not target_present_anywhere,
        "contains_overlap": canonical == "OVERLAP" or "OVERLAP" in stated_relations,
        "explicit_relations": json.dumps(stated_relations),
        "parsed_edges": json.dumps(edges, ensure_ascii=False),
        "template_ids": json.dumps(template_ids),
        "template_families": json.dumps(template_families),
        "contains_suspicious_template": contains_non_ok,
        "contains_error_template": contains_error,
        "contains_unknown_template": contains_unknown,
        "template_match_ambiguous": any(match["ambiguous"] for match in matches),
        "parse_success": parse_success,
        "graph_success": graph_success,
        "query_entities_in_graph": query_entities_in_graph,
        "parse_failure_reasons": ";".join(failure_reasons),
        "exact_duplicate_group": exact_duplicate_group,
        "normalized_duplicate_group": normalized_duplicate_group,
        "graph_template_group": graph_template_group,
        "primary_candidate": primary_candidate,
        "structural_primary_candidate": structural_primary_candidate,
        "primary_template_clean": bool(primary_scope and not contains_non_ok),
        "primary_target_absent": bool(primary_scope and computed_hop >= 2 and not target_present_anywhere),
        "primary_target_absent_template_clean": bool(
            primary_scope and computed_hop >= 2 and not target_present_anywhere and not contains_non_ok
        ),
        "primary_candidate_reason": ";".join(candidate_reasons),
    }


def small_validation(n: int = 500):
    specs = extract_template_specs()
    index = template_index(specs)
    candidates = []
    rng = random.Random(SEED)
    for path in sorted_dataset_files():
        recorded_hop, split = parse_source_file(path)
        records = list(load_json_records(path))
        for source_record_id, record in rng.sample(records, min(20, len(records))):
            candidates.append((path, recorded_hop, split, source_record_id, record))
    sample = rng.sample(candidates, n)
    parsed = []
    for validation_id, (path, recorded_hop, split, source_record_id, record) in enumerate(sample):
        parsed.append(parse_record(record, {
            "row_id": validation_id,
            "source_split": split,
            "source_file": str(path.relative_to(path.parents[1])),
            "source_record_id": source_record_id,
            "recorded_hop": recorded_hop,
        }, index))
    summary = {
        "sample_size": len(parsed),
        "parse_success": sum(row["parse_success"] for row in parsed),
        "graph_success": sum(row["graph_success"] for row in parsed),
        "hop_match": sum(row["hop_match"] for row in parsed),
        "label_path_match": sum(row["label_matches_computed_relation"] is True for row in parsed),
        "non_ok_template_samples": sum(row["contains_suspicious_template"] for row in parsed),
        "representative_rows": parsed[:20],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "small_sample_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    summary = small_validation(args.sample_size)
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "representative_rows"},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
