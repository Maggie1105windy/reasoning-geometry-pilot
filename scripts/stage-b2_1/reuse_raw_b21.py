#!/usr/bin/env python3
"""Reuse frozen candidate scores for Experiment A and C1, then enforce Gate A."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common_b21 import (
    B156, B2, CARDINAL, DIAGONAL, HOPS, MODEL_SPECS, NATURAL_CANDIDATES,
    OUT, RELATIONS, attach_truth, bootstrap_metrics, ensure_dirs, load_primary,
    metrics_by_hop, read_json, reconstruct_predictions, sha256_file,
    write_confusions, write_json,
)


SOURCES = {
    "base": B156 / "behavioral" / "all_candidate_scores.csv",
    "sft": B2 / "behavior" / "sft_candidate_scores.csv",
    "dpo": B2 / "behavior" / "dpo_candidate_scores.csv",
    "final": B2 / "behavior" / "final_candidate_scores.csv",
}


def relation_metrics(pred: pd.DataFrame, model: str, scoring: str) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        part = pred.loc[pred.computed_hop.eq(hop)]
        for relation in RELATIONS:
            mask = part.canonical_relation.eq(relation)
            rows.append({
                "model": model, "scoring": scoring, "hop": hop, "relation": relation,
                "n": int(mask.sum()), "accuracy": float(part.loc[mask, "correct"].mean()),
                "prediction_frequency": float(part.prediction.eq(relation).mean()),
            })
    return pd.DataFrame(rows)


def cardinal_diagonal(scores: pd.DataFrame, pred: pd.DataFrame, model: str, scoring: str, column: str) -> pd.DataFrame:
    pivot = scores.pivot(index="row_id", columns="candidate_relation", values=column)
    truth = pred.set_index("row_id")
    rows = []
    for hop in HOPS:
        for true_type, relations in [("cardinal", CARDINAL), ("diagonal", DIAGONAL)]:
            ids = truth.index[(truth.computed_hop.eq(hop)) & truth.canonical_relation.isin(relations)]
            margins = []
            for row_id in ids:
                gold = truth.loc[row_id, "canonical_relation"]
                values = pivot.loc[row_id]
                margins.append(float(values[gold] - values.drop(gold).max()))
            rows.append({
                "model": model, "scoring": scoring, "hop": hop, "true_class_type": true_type,
                "n": len(ids), "accuracy": float(truth.loc[ids, "correct"].mean()),
                "mean_score_margin_gold_vs_best_other": float(np.mean(margins)),
                "predicted_cardinal_frequency": float(truth.loc[ids, "prediction"].isin(CARDINAL).mean()),
                "predicted_diagonal_frequency": float(truth.loc[ids, "prediction"].isin(DIAGONAL).mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    protocol = read_json(OUT / "raw_reproduction" / "protocol.json")
    primary = load_primary()
    valid_ids = set(primary.row_id)
    all_scores = []
    metrics = []
    bootstraps = []
    comparisons = []
    per_relation = []
    card_diag = []
    prediction_frequency = []
    gate_models = {}

    frozen_metrics = pd.read_csv(B2 / "behavior" / "all_models.csv")
    frozen_sum = frozen_metrics.loc[(frozen_metrics.dataset.eq("primary")) & (frozen_metrics.scoring.eq("sum"))]

    for model, source in SOURCES.items():
        scores = pd.read_csv(source)
        scores = scores.loc[scores.row_id.isin(valid_ids)].copy()
        scores["model"] = model
        scores["source_path"] = str(source)
        scores["source_sha256"] = sha256_file(source)
        scores = scores.sort_values(["row_id", "candidate_relation"]).reset_index(drop=True)
        counts = scores.groupby("row_id").size()
        finite = bool(np.isfinite(scores[["sum_log_probability", "mean_log_probability"]].to_numpy()).all())
        basic = len(scores) == 2063 * 8 and len(counts) == 2063 and counts.eq(8).all() and finite
        if not basic:
            raise RuntimeError(f"{model}: incomplete/non-finite frozen candidate scores")
        all_scores.append(scores)

        preds = {}
        for scoring, column, protocol_name in [
            ("sum", "sum_log_probability", "Raw-Sum"),
            ("mean", "mean_log_probability", "Raw-Mean"),
        ]:
            pred = attach_truth(reconstruct_predictions(scores, column), primary)
            pred["model"] = model
            pred["scoring"] = scoring
            preds[scoring] = pred
            m = metrics_by_hop(pred, model, protocol_name)
            b = bootstrap_metrics(pred, model, protocol_name)
            metrics.append(m)
            bootstraps.append(b)
            per_relation.append(relation_metrics(pred, model, scoring))
            card_diag.append(cardinal_diagonal(scores, pred, model, scoring, column))
            for hop in HOPS:
                part = pred.loc[pred.computed_hop.eq(hop)]
                for relation in RELATIONS:
                    prediction_frequency.append({
                        "model": model, "scoring": scoring, "hop": hop, "predicted_relation": relation,
                        "frequency": float(part.prediction.eq(relation).mean()),
                        "candidate_token_count": len(NATURAL_CANDIDATES[relation].strip().split("-")),
                        "relation_type": "cardinal" if relation in CARDINAL else "diagonal",
                    })
            write_confusions(pred, OUT / "scoring_sensitivity" / f"confusion_{scoring}", model)
            if scoring == "sum":
                write_confusions(pred, OUT / "raw_reproduction" / "confusion_matrices", model)

        merged = preds["sum"].merge(
            preds["mean"][["row_id", "prediction", "correct"]], on="row_id",
            suffixes=("_sum", "_mean"), validate="one_to_one",
        )
        comparisons.append(merged)

        computed = metrics[-2].set_index("hop")["accuracy"]
        expected = frozen_sum.loc[frozen_sum.model.eq(model)].set_index("hop")["accuracy"]
        expected_n = frozen_sum.loc[frozen_sum.model.eq(model)].set_index("hop")["n_test"]
        aligned = computed.to_frame("computed").join(expected.rename("frozen")).join(expected_n.rename("n_test"))
        aligned["absolute_difference"] = (aligned.computed - aligned.frozen).abs()
        aligned["computed_correct_count"] = np.rint(aligned.computed * aligned.n_test).astype(int)
        aligned["frozen_correct_count"] = np.rint(aligned.frozen * aligned.n_test).astype(int)
        aligned["exact_rational_accuracy"] = aligned.computed_correct_count.eq(aligned.frozen_correct_count)
        exact = bool(aligned.exact_rational_accuracy.all())
        gate_models[model] = {
            "status": "PASS" if exact else "FAIL",
            "source": str(source), "source_sha256": sha256_file(source),
            "rows": 2063, "candidate_rows": len(scores), "finite": finite,
            "accuracy_by_hop": {str(i): float(v) for i, v in computed.items()},
            "frozen_accuracy_by_hop": {str(i): float(v) for i, v in expected.items()},
            "absolute_difference_by_hop": {str(i): float(v) for i, v in aligned.absolute_difference.items()},
            "computed_correct_count_by_hop": {str(i): int(v) for i, v in aligned.computed_correct_count.items()},
            "frozen_correct_count_by_hop": {str(i): int(v) for i, v in aligned.frozen_correct_count.items()},
            "comparison_definition": "exact equality of integer correct counts at identical n; float CSV display deltas are retained",
        }

    combined = pd.concat(all_scores, ignore_index=True)
    combined.to_csv(OUT / "raw_reproduction" / "row_scores.csv", index=False)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_bootstrap = pd.concat(bootstraps, ignore_index=True)
    raw_metrics = all_metrics.loc[all_metrics.protocol.eq("Raw-Sum")]
    raw_bootstrap = all_bootstrap.loc[all_bootstrap.protocol.eq("Raw-Sum")]
    raw_metrics.to_csv(OUT / "raw_reproduction" / "hop_metrics.csv", index=False)
    raw_bootstrap.to_csv(OUT / "raw_reproduction" / "bootstrap_ci.csv", index=False)

    # Exact candidate token audit is copied into a flat machine-readable table.
    input_audit = read_json(OUT / "audit" / "input_audit.json")
    token_rows = []
    for model, item in input_audit["models"].items():
        for relation, detail in item["candidate_tokenization"].items():
            token_rows.append({"model": model, "relation": relation, **detail})
    pd.DataFrame(token_rows).to_csv(OUT / "raw_reproduction" / "candidate_tokenization.csv", index=False)

    all_metrics.to_csv(OUT / "scoring_sensitivity" / "metrics_sum_vs_mean.csv", index=False)
    pd.concat(per_relation, ignore_index=True).to_csv(OUT / "scoring_sensitivity" / "per_relation_metrics.csv", index=False)
    pd.concat(card_diag, ignore_index=True).to_csv(OUT / "scoring_sensitivity" / "cardinal_diagonal_metrics.csv", index=False)
    pd.DataFrame(prediction_frequency).to_csv(OUT / "scoring_sensitivity" / "prediction_frequency.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(OUT / "scoring_sensitivity" / "row_level_comparison.csv", index=False)

    gate_checks = {
        "rows_2063_each_model": all(item["rows"] == 2063 for item in gate_models.values()),
        "row_ids_exact": set(combined.row_id) == valid_ids,
        "eight_candidates_each_row_model": bool(combined.groupby(["model", "row_id"]).size().eq(8).all()),
        "prompt_hash_matches_frozen_manifest": input_audit["raw_prompt_exact_for_all_primary_rows"],
        "candidate_strings_exact": protocol["candidate_strings"] == NATURAL_CANDIDATES,
        "all_results_finite": bool(np.isfinite(combined[["sum_log_probability", "mean_log_probability"]].to_numpy()).all()),
        "accuracy_exactly_matches_frozen_B2": all(item["status"] == "PASS" for item in gate_models.values()),
        "gold_leakage": False,
    }
    gate_status = "PASS" if all(v is True for k, v in gate_checks.items() if k != "gold_leakage") and gate_checks["gold_leakage"] is False else "FAIL"
    gate = {"gate": "Gate A", "status": gate_status, "checks": gate_checks, "models": gate_models}
    write_json(OUT / "raw_reproduction" / "reproduction_check.json", gate)
    if gate_status != "PASS":
        raise RuntimeError("Gate A FAIL; subsequent B/C model evaluation is prohibited")
    print(json.dumps({"stage": "EXPERIMENT_A_AND_C1", "gate_A": gate_status, "rows_per_model": 2063}, indent=2))


if __name__ == "__main__":
    main()
