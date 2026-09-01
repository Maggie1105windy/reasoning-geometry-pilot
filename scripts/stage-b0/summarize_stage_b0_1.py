#!/usr/bin/env python3
"""Generate Stage B0.1 audit tables, feasibility ceilings, validations, and report."""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from common import ARTIFACTS, INVERSE, NON_OVERLAP, RELATIONS, STEPGAME


def md_table(frame: pd.DataFrame) -> str:
    display = frame.copy().fillna("")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def full_grid_counts(frame, mask, hops, relations):
    index = pd.MultiIndex.from_product([hops, relations], names=["computed_hop", "canonical_relation"])
    counts = frame.loc[mask].groupby(["computed_hop", "canonical_relation"]).size()
    return counts.reindex(index, fill_value=0).rename("count").reset_index()


def balanced_summary(counts: pd.DataFrame, name: str):
    minimum = int(counts["count"].min())
    bottlenecks = counts[counts["count"] == minimum][
        ["computed_hop", "canonical_relation", "count"]
    ].to_dict(orient="records")
    return {
        "subset": name,
        "n_min": minimum,
        "balanced_total_ceiling": minimum * counts.computed_hop.nunique() * counts.canonical_relation.nunique(),
        "bottleneck_cells": bottlenecks,
    }


def main():
    frame = pd.read_csv(ARTIFACTS / "stepgame_metadata.csv", low_memory=False)
    provenance = json.loads((ARTIFACTS / "data_provenance.json").read_text())
    template_summary = json.loads((ARTIFACTS / "template_audit_summary.json").read_text())
    duplicate_summary = json.loads((ARTIFACTS / "duplicate_summary.json").read_text())
    graph_summary = json.loads((ARTIFACTS / "graph_build_summary.json").read_text())
    inventory = pd.read_csv(ARTIFACTS / "template_inventory.csv")
    suspicious = pd.read_csv(ARTIFACTS / "suspicious_templates.csv")

    for column in [
        "hop_match", "path_unique", "target_relation_present_anywhere",
        "target_absent_anywhere",
        "contains_overlap", "contains_suspicious_template", "contains_error_template",
        "contains_unknown_template", "parse_success", "graph_success",
        "query_entities_in_graph", "primary_candidate", "structural_primary_candidate",
        "primary_template_clean", "primary_target_absent",
        "primary_target_absent_template_clean",
    ]:
        assert frame[column].dtype == bool, (column, frame[column].dtype)
    assert frame.target_relation_present_on_query_path.dropna().isin([True, False]).all()

    relation_counts = frame.groupby(["raw_label", "canonical_relation"]).size().rename("count").reset_index()
    relation_rows = []
    for row in relation_counts.itertuples(index=False):
        relation_rows.append({
            "raw_label": row.raw_label,
            "canonical_label": row.canonical_relation,
            "inverse_relation": INVERSE[row.canonical_relation],
            "count": int(row.count),
            "notes": "Direct canonical mapping; raw values exhaustively enumerated from Primary clean data.",
        })
    pd.DataFrame(relation_rows).to_csv(ARTIFACTS / "relation_mapping.csv", index=False)

    story_hop = pd.crosstab(
        frame.story_len,
        frame.computed_hop.fillna(-1).astype(int),
    ).rename(columns={-1: "UNCOMPUTABLE"})
    story_hop.to_csv(ARTIFACTS / "story_len_by_hop.csv")

    hop_relation = full_grid_counts(frame, frame.computed_hop.notna(), range(1, 11), RELATIONS)
    hop_relation.to_csv(ARTIFACTS / "hop_relation_counts.csv", index=False)

    derived = frame[
        frame.computed_hop.ge(2) & frame.path_unique
        & frame.ordered_composition_signature.notna()
    ]
    composition_counts = derived.groupby([
        "computed_hop", "ordered_composition_signature",
        "unordered_composition_signature", "canonical_relation",
    ]).size().rename("count").reset_index()
    composition_counts.to_csv(ARTIFACTS / "composition_counts.csv", index=False)

    target_absent_counts = frame[frame.computed_hop.ge(2)].groupby([
        "computed_hop", "canonical_relation", "target_absent_anywhere",
    ]).size().rename("count").reset_index()
    target_absent_counts.to_csv(ARTIFACTS / "target_absent_counts.csv", index=False)

    primary_scope = (
        frame.story_len.eq(5) & frame.computed_hop.isin([1, 2, 3, 4, 5])
        & frame.canonical_relation.isin(NON_OVERLAP)
        & frame.parse_success & frame.graph_success
    )
    primary_masks = {
        "Strict-Hop-Match": frame.primary_candidate,
        "Primary-All": primary_scope,
        "Primary-Target-Absent": primary_scope & frame.computed_hop.ge(2) & frame.target_absent_anywhere,
        "Primary-Template-Clean": primary_scope & ~frame.contains_suspicious_template,
        "Primary-Target-Absent+Template-Clean": (
            primary_scope & frame.computed_hop.ge(2) & frame.target_absent_anywhere
            & ~frame.contains_suspicious_template
        ),
    }
    primary_tables = []
    for subset, mask in primary_masks.items():
        table = full_grid_counts(frame, mask, range(1, 6), NON_OVERLAP)
        table.insert(0, "subset", subset)
        primary_tables.append(table)
    primary_counts = pd.concat(primary_tables, ignore_index=True)
    primary_counts.to_csv(ARTIFACTS / "primary_candidate_counts.csv", index=False)

    all_counts = full_grid_counts(frame, primary_masks["Primary-All"], range(1, 6), NON_OVERLAP)
    clean_counts = full_grid_counts(frame, primary_masks["Primary-Template-Clean"], range(1, 6), NON_OVERLAP)
    strict_counts = full_grid_counts(frame, primary_masks["Strict-Hop-Match"], range(1, 6), NON_OVERLAP)
    target_absent_derived_counts = full_grid_counts(
        frame, primary_masks["Primary-Target-Absent+Template-Clean"], range(2, 6), NON_OVERLAP
    )
    ceilings = {
        "strict_hop_match": balanced_summary(strict_counts, "Strict-Hop-Match"),
        "all_parsed": balanced_summary(all_counts, "Primary-All"),
        "template_clean": balanced_summary(clean_counts, "Primary-Template-Clean"),
        "target_absent_derived_template_clean": balanced_summary(
            target_absent_derived_counts, "Primary-Target-Absent+Template-Clean h=2..5"
        ),
    }
    (ARTIFACTS / "balanced_ceiling.json").write_text(
        json.dumps(ceilings, indent=2) + "\n", encoding="utf-8"
    )

    hop5 = frame[frame.story_len.eq(5)]
    story5_hops = hop5.groupby("computed_hop").size().reindex(range(1, 6), fill_value=0).rename("samples").reset_index()
    story5_hops["computed_hop"] = story5_hops.computed_hop.astype(int)
    primary_hop_counts = frame.loc[primary_scope].groupby("computed_hop").size().reindex(range(1, 6), fill_value=0)
    target_absent_by_hop = []
    target_absent_template_clean_by_hop = []
    for hop in range(2, 6):
        subset = frame.loc[primary_scope & frame.computed_hop.eq(hop)]
        count = int(subset.target_absent_anywhere.sum())
        target_absent_by_hop.append({
            "computed_hop": hop,
            "target_absent": count,
            "eligible_total": int(len(subset)),
            "rate": count / len(subset) if len(subset) else None,
        })
        clean_subset = frame.loc[
            primary_scope & frame.computed_hop.eq(hop)
            & ~frame.contains_suspicious_template
        ]
        clean_count = int(clean_subset.target_absent_anywhere.sum())
        target_absent_template_clean_by_hop.append({
            "computed_hop": hop,
            "target_absent": clean_count,
            "eligible_total": int(len(clean_subset)),
            "rate": clean_count / len(clean_subset) if len(clean_subset) else None,
        })
    target_absent_hop_df = pd.DataFrame(target_absent_by_hop)

    overlap_by_hop = frame[frame.canonical_relation.eq("OVERLAP")].groupby("computed_hop").size()
    overlap_earliest = int(overlap_by_hop.index.min()) if len(overlap_by_hop) else None
    overlap_h1 = int(overlap_by_hop.get(1, 0))

    primary_derived = frame.loc[primary_scope & frame.computed_hop.ge(2) & frame.path_unique]
    ordered_frequency = primary_derived.ordered_composition_signature.value_counts()
    unordered_frequency = primary_derived.unordered_composition_signature.value_counts()
    composition_summary = {
        "primary_ordered_signature_count": int(ordered_frequency.size),
        "primary_unordered_signature_count": int(unordered_frequency.size),
        "ordered_signatures_ge_50": int((ordered_frequency >= 50).sum()),
        "ordered_signatures_ge_100": int((ordered_frequency >= 100).sum()),
        "unordered_signatures_ge_50": int((unordered_frequency >= 50).sum()),
        "unordered_signatures_ge_100": int((unordered_frequency >= 100).sum()),
        "ordered_singletons": int((ordered_frequency == 1).sum()),
        "unordered_singletons": int((unordered_frequency == 1).sum()),
    }

    mismatch_computable = int((frame.graph_success & ~frame.hop_match).sum())
    mismatch_uncomputable = int((~frame.graph_success).sum())
    recorded_greater = int((frame.recorded_hop > frame.computed_hop).fillna(False).sum())
    recorded_less = int((frame.recorded_hop < frame.computed_hop).fillna(False).sum())
    label_computable = int(frame.computed_relation_from_path.notna().sum())
    label_match = int(frame.label_matches_computed_relation.eq(True).sum())
    label_mismatch = int(frame.label_matches_computed_relation.eq(False).sum())
    label_quality = frame.groupby("contains_suspicious_template").agg(
        samples=("row_id", "size"),
        computable=("computed_relation_from_path", lambda x: int(x.notna().sum())),
        label_match=("label_matches_computed_relation", lambda x: int(x.eq(True).sum())),
        label_mismatch=("label_matches_computed_relation", lambda x: int(x.eq(False).sum())),
    ).reset_index()

    template_clean_diversity = inventory[
        inventory.semantic_status.eq("OK") & inventory.complete_clean_possible_matches.gt(0)
    ].groupby("relation").size().rename("clean_OK_templates").reindex(NON_OVERLAP).reset_index()

    parse_reason_counts = Counter()
    for reasons in frame.loc[~frame.parse_success, "parse_failure_reasons"].fillna(""):
        for reason in reasons.split(";"):
            if reason:
                parse_reason_counts[reason.split(":", 1)[-1]] += 1

    validations = {
        "row_id_unique": bool(frame.row_id.is_unique),
        "row_id_contiguous": frame.row_id.tolist() == list(range(len(frame))),
        "canonical_relations_legal": set(frame.canonical_relation) == set(RELATIONS),
        "query_entities_in_graph_when_parse_success": bool(
            frame.loc[frame.parse_success, "query_entities_in_graph"].all()
        ),
        "computed_hop_at_least_one": bool(frame.computed_hop.dropna().ge(1).all()),
        "hop1_has_direct_path": bool(
            frame.loc[frame.computed_hop.eq(1), "path_relations"].notna().all()
        ),
        "inverse_mapping_complete": bool(
            frame.apply(lambda row: INVERSE[row.canonical_relation] == row.inverse_relation, axis=1).all()
        ),
        "no_silent_template_unmatched": template_summary["unmatched_sentence_count"] == 0,
        "metadata_rows_equal_input": len(frame) == graph_summary["input_rows"] == 210000,
        "candidate_reason_complete": bool(frame.primary_candidate_reason.notna().all()),
        "raw_repository_clean": subprocess.check_output(
            ["git", "-C", str(STEPGAME), "status", "--porcelain"], text=True
        ).strip() == "",
        "parquet_exists": (ARTIFACTS / "stepgame_metadata.parquet").is_file(),
    }
    assert all(validations.values()), [key for key, value in validations.items() if not value]

    summary = {
        "dataset_rows": len(frame),
        "split_counts": frame.source_split.value_counts().to_dict(),
        "story_length_counts": frame.story_len.value_counts().sort_index().to_dict(),
        "relation_counts": frame.canonical_relation.value_counts().reindex(RELATIONS).to_dict(),
        "parse_success": int(frame.parse_success.sum()),
        "parse_failure": int((~frame.parse_success).sum()),
        "parse_success_rate": float(frame.parse_success.mean()),
        "graph_success": int(frame.graph_success.sum()),
        "hop_match": int(frame.hop_match.sum()),
        "hop_mismatch": int((~frame.hop_match).sum()),
        "hop_match_rate": float(frame.hop_match.mean()),
        "hop_mismatch_computable": mismatch_computable,
        "hop_mismatch_uncomputable": mismatch_uncomputable,
        "recorded_hop_greater_than_computed": recorded_greater,
        "recorded_hop_less_than_computed": recorded_less,
        "label_path_computable": label_computable,
        "label_path_match": label_match,
        "label_path_mismatch": label_mismatch,
        "overlap_h1": overlap_h1,
        "overlap_earliest_hop": overlap_earliest,
        "overlap_by_hop": {str(int(key)): int(value) for key, value in overlap_by_hop.items()},
        "story_len5_hops": {str(int(row.computed_hop)): int(row.samples) for row in story5_hops.itertuples()},
        "primary_scope_hops": {str(int(key)): int(value) for key, value in primary_hop_counts.items()},
        "target_absent_by_hop": target_absent_by_hop,
        "target_absent_template_clean_by_hop": target_absent_template_clean_by_hop,
        "composition": composition_summary,
        "template_clean_diversity": dict(zip(
            template_clean_diversity.relation,
            template_clean_diversity.clean_OK_templates.astype(int),
        )),
        "ceilings": ceilings,
        "parse_failure_reason_occurrences": dict(parse_reason_counts),
        "duplicate_summary": duplicate_summary,
        "validations": validations,
    }
    (ARTIFACTS / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (ARTIFACTS / "validation_summary.json").write_text(
        json.dumps(validations, indent=2) + "\n", encoding="utf-8"
    )

    relation_display = pd.DataFrame(relation_rows)[
        ["raw_label", "canonical_label", "inverse_relation", "count"]
    ]
    hop_relation_primary = full_grid_counts(frame, primary_scope, range(1, 6), NON_OVERLAP)
    hop_relation_pivot = hop_relation_primary.pivot(
        index="computed_hop", columns="canonical_relation", values="count"
    ).reset_index()
    story5_display = story5_hops.copy()
    story5_display["share"] = story5_display.samples / story5_display.samples.sum()
    target_display = target_absent_hop_df.copy()
    clean_target_display = pd.DataFrame(target_absent_template_clean_by_hop).rename(columns={
        "target_absent": "template_clean_target_absent",
        "eligible_total": "template_clean_eligible_total",
        "rate": "template_clean_rate",
    })
    target_display = target_display.rename(columns={
        "target_absent": "all_parsed_target_absent",
        "eligible_total": "all_parsed_eligible_total",
        "rate": "all_parsed_rate",
    }).merge(clean_target_display, on="computed_hop")
    target_display["all_parsed_rate"] = target_display.all_parsed_rate.map(lambda value: f"{value:.2%}")
    target_display["template_clean_rate"] = target_display.template_clean_rate.map(lambda value: f"{value:.2%}")
    template_status_display = pd.DataFrame(
        list(template_summary["semantic_status_counts"].items()),
        columns=["semantic_status", "template_definitions"],
    )

    clean_count = provenance["actual_complete_version_counts"]["clean"]
    noise_count = provenance["actual_complete_version_counts"]["noise"]
    exact = duplicate_summary["exact_duplicate_group"]
    stories = duplicate_summary["story_group_id"]
    lines = [
        "# StepGame Stage B0.1 Data Audit", "",
        "## Technical summary", "",
        f"CompleteVersion clean contains **{len(frame):,}** records, not the 610,000 implied by the current README's 30k/1k/30k-per-K prose. The actual checked-in files contain 10k train, 1k valid, and 10k test examples per K for K=1..10.", "",
        f"The parser matches every one of the {template_summary['total_sentences']['clean']:,} clean sentences to official template source. **{int(frame.parse_success.sum()):,} / {len(frame):,} ({frame.parse_success.mean():.2%})** records have every sentence mapped to a defensible two-entity semantic edge; **{int(frame.graph_success.sum()):,}** still provide a query path.", "",
        f"`qaK` is not the actual query hop: only **{int(frame.hop_match.sum()):,} / {len(frame):,} ({frame.hop_match.mean():.2%})** records have K equal to the independently computed query distance. This is explained by the official generator constructing a K-edge chain and then sampling any entity pair. Treating the mismatch as corruption would be incorrect.", "",
        f"A fixed five-sentence comparison is data-feasible only after redefining K as source story length and using `computed_hop` as the experimental hop. Under the literal requested `hop_match=True` rule, h=1..4 cells are zero; under the audited semantics, the template-clean eight-class balanced ceiling is **{ceilings['template_clean']['n_min']:,} per relation per hop**.", "",
        "## The official source is frozen, but README size prose disagrees with files", "",
        f"- Repository: `{provenance['repository_url']}`",
        f"- Commit: `{provenance['git_commit']}`",
        "- Primary: `Dataset/CompleteVersion/clean`",
        f"- Actual CompleteVersion clean: train {clean_count['train']:,}, valid {clean_count['valid']:,}, test {clean_count['test']:,}; total {clean_count['total']:,}",
        f"- Actual CompleteVersion noise: train {noise_count['train']:,}, valid {noise_count['valid']:,}, test {noise_count['test']:,}; total {noise_count['total']:,}",
        "- The 2024 repository is preferred over third-party repacks because it includes the corrected official templates and exact source provenance. Noise is inventoried but excluded from Primary metadata.", "",
        "## Schema and metric definitions", "",
        "The record grain is one story–question–label example. `story_len` is the number of sentences. `recorded_hop` is K recovered from `qaK_*`; no hop field exists inside repository records. `computed_hop` is the BFS shortest-path edge count between query entities in the independently parsed semantic graph. A target-absent record has no explicit sentence whose stated canonical relation equals the target label.", "",
        md_table(relation_display), "",
        "## K and actual query distance are different variables", "",
        f"Only {summary['hop_match']:,} records match. Of the {summary['hop_mismatch']:,} mismatches, {mismatch_computable:,} have a valid shorter query path and {mismatch_uncomputable:,} lack a complete path because of non-defensible templates. `recorded_hop > computed_hop` occurs {recorded_greater:,} times; `recorded_hop < computed_hop` occurs {recorded_less:,} times.", "",
        "For `story_len=5`, the actual query-distance distribution is:", "",
        md_table(story5_display.assign(share=story5_display.share.map(lambda value: f"{value:.2%}"))), "",
        "This supports controlling story length while varying actual hop, but requires dropping `hop_match=True` as an exclusion because that predicate encodes the wrong interpretation of K.", "",
        "## Template source contains ten errors and one unresolved meaning", "",
        md_table(template_status_display), "",
        f"All clean and noise sentences match official template skeletons. Equivalent/duplicate template ambiguity affects {template_summary['ambiguous_sentences_by_variant'].get('clean', 0):,} clean and {template_summary['ambiguous_sentences_by_variant'].get('noise', 0):,} noise sentences; unmatched count is zero in both. Eleven non-OK definitions affect {template_summary['primary_clean_non_ok_affected_sample_union']:,} clean samples. Ten have a direction or argument error; one uses `front right`, which does not uniquely define upper-right in the canonical 2D system. Original source remains unchanged.", "",
        "Template-clean, actually used definitions by relation:", "",
        md_table(template_clean_diversity), "",
        "The available clean pattern diversity is sufficient to attempt template-held-out splits, but ambiguous duplicate surface forms must be grouped by normalized surface skeleton rather than source template ID.", "",
        "## Five-sentence eight-class feasibility", "",
        "Counts below use parsed five-sentence records, actual query distance, and eight non-overlap targets. They intentionally do not enforce the misnamed K field.", "",
        md_table(hop_relation_pivot), "",
        f"- All parsed balanced ceiling: **{ceilings['all_parsed']['n_min']:,}** per relation/hop.",
        f"- Template-clean balanced ceiling: **{ceilings['template_clean']['n_min']:,}** per relation/hop.",
        f"- Literal strict-hop-match ceiling: **{ceilings['strict_hop_match']['n_min']:,}** because h=1..4 are empty.", "",
        "## Target-absent reasoning controls remain sizeable", "",
        md_table(target_display), "",
        f"The template-clean Derived-Target-Absent ceiling across h=2..5 and eight relations is **{ceilings['target_absent_derived_template_clean']['n_min']:,}** per cell. h=1 is reported separately and is not treated as a target-absent derived condition.", "",
        "## Composition-held-out is plausible but must be group-optimized in B0.2", "",
        f"Within the five-sentence parsed derived scope there are **{composition_summary['primary_ordered_signature_count']:,} ordered** and **{composition_summary['primary_unordered_signature_count']:,} unordered** signatures. {composition_summary['unordered_signatures_ge_100']:,} unordered signatures have at least 100 samples; {composition_summary['unordered_singletons']:,} are singletons. This is enough diversity to attempt an unseen-composition split, but B0.2 must jointly enforce signature grouping, target balance, hop balance, and story grouping before declaring a final split feasible.", "",
        "## Overlap is derived, not directly stated", "",
        f"`computed_hop=1` has **{overlap_h1:,}** OVERLAP targets. The earliest observed OVERLAP target is hop **{overlap_earliest}**. Counts by hop are `{summary['overlap_by_hop']}`. The generator has eight explicit edge relations and no OVERLAP edge template; overlap targets arise from vector composition.", "",
        "## Label/path agreement exposes the practical impact of template errors", "",
        f"Among {label_computable:,} records with a unique, composition-computable path, {label_match:,} labels agree and {label_mismatch:,} disagree with the meaning of the actual story. Disagreement is segmented below; this is why non-OK templates must remain flagged rather than silently interpreted from their function names.", "",
        md_table(label_quality), "",
        "## Duplicate and leakage audit", "",
        f"There are {exact['duplicate_groups']:,} exact-duplicate groups affecting {exact['rows_in_duplicate_groups']:,} rows ({exact['excess_duplicate_rows']:,} excess rows). There are {stories['duplicate_groups']:,} repeated-story groups affecting {stories['rows_in_duplicate_groups']:,} rows; {duplicate_summary['story_groups_with_multiple_queries']:,} story groups contain multiple questions. Future splits must group by `story_group_id`; inverse/query variants must additionally remain in the same story group. `normalized_duplicate_group` and `graph_template_group` are useful secondary leakage controls.", "",
        "## Robustness, limitations, and exclusions", "",
        f"- Full record parse failures: {int((~frame.parse_success).sum()):,}; no sentence is unmatched, but some matched templates lack a defensible two-entity canonical meaning.",
        f"- Query path unavailable: {int((~frame.graph_success).sum()):,} records.",
        f"- Multiple/ambiguous source-template matches: {template_summary['ambiguous_sentence_count']:,} sentences; equivalent matches are retained and flagged.",
        "- `graph_template_group` is a deterministic heuristic, not a complete graph-isomorphism canonicalizer.",
        "- No raw data or official template source was modified; exclusions are flags only.", "",
        "## Recommended B0 protocol adjustment", "",
        "1. Rename `recorded_hop` conceptually to `source_K` or `source_story_len` for clean data.",
        "2. Use independently computed BFS distance as the only experimental hop.",
        "3. Make the eight-class Primary subset `story_len=5`, `computed_hop in 1..5`, parsed path available, and template-clean; do not require `source_K==computed_hop`.",
        "4. Keep OVERLAP outside Primary; retain it as a derived diagnostic class.",
        "5. In B0.2, group by `story_group_id`, normalized template skeleton, and composition signature before selecting train/validation/test rows.", "",
        "## Further questions for B0.2", "",
        "- Should all ten ERROR templates be excluded, or should semantically corrected edges be retained only for a sensitivity set?",
        "- Should target-absent mean lexical/stated-relation absence (current definition) or absence after adding inverse entailments?",
        "- Which minimum cell size should the composition-held-out optimizer enforce?", "",
    ]
    (ARTIFACTS / "B0_1_DATA_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    source_notes = {
        "audience": "technical",
        "delivery": "Markdown requested explicitly by user",
        "report_structure": [
            "technical summary", "key findings", "scope and definitions", "method evidence",
            "limitations and validations", "recommended next steps", "further questions",
        ],
        "visual_omission_reason": "Exact audit crosstabs and cell ceilings are the decision surface; charts would not improve lookup or protocol freezing.",
        "sources": ["data_provenance.json", "stepgame_metadata.parquet", "template_inventory.csv"],
    }
    (ARTIFACTS / "report_source_notes.json").write_text(
        json.dumps(source_notes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
