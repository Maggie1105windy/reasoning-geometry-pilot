#!/usr/bin/env python3
"""Phases 7 and 9: metrics, bootstrap, figures, report, and decisions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_b23 import (
    B22,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CARDINAL,
    DECISION_THRESHOLDS,
    DIAGONAL,
    EXPECTED_B22_NATIVE_SUM_DERIVED_ACCURACY,
    GENERATION_SEEDS,
    HOPS,
    OUT,
    PRIMARY_GENERATION_SEED,
    RELATIONS,
    RELATION_WORDS,
    confusion_frames,
    ensure_dirs,
    markdown_table,
    metric_dict,
    metrics_for_frame,
    prediction_collapse,
    read_json,
    read_jsonl,
    sha256_text,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


COLORS = {
    "B2.2 Native-Sum": "#777777",
    "FG-N": "#4C78A8",
    "FG-T": "#E39C37",
}


def generation_frame(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No rows in {path}")
    frame["row_id"] = frame.row_id.astype(int)
    frame["hop"] = frame.hop.astype(int)
    frame["correct"] = frame.correct.astype(bool)
    return frame.sort_values("row_id").reset_index(drop=True)


def metrics_by_hop(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        part = frame.loc[frame.hop.eq(hop)]
        metrics = metrics_for_frame(part)
        metrics.update({
            "protocol": protocol,
            "scope": f"h{hop}",
            "hop": hop,
            "n_story_groups": int(part.story_group_id.nunique()),
            "majority_accuracy": float(part.gold_relation.value_counts(normalize=True).max()),
            "prediction_collapse": prediction_collapse(metrics),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def derived_row(hop_metrics: pd.DataFrame, protocol: str) -> dict:
    part = hop_metrics.loc[hop_metrics.hop.isin([2, 3, 4, 5])]
    average_columns = [
        "accuracy", "macro_f1", "balanced_accuracy", "parse_success_rate",
        *[f"recall_{relation}" for relation in RELATIONS],
        *[f"prediction_frequency_{relation}" for relation in RELATIONS],
        "invalid_frequency", "cardinal_prediction_frequency",
        "diagonal_prediction_frequency", "max_single_relation_frequency",
    ]
    result = {"protocol": protocol, "scope": "h2-h5-derived", "hop": "derived", "n": int(part.n.sum())}
    result.update({column: float(part[column].mean()) for column in average_columns})
    result["prediction_collapse"] = bool(part.prediction_collapse.any())
    return result


def overall_row(frame: pd.DataFrame, protocol: str) -> dict:
    result = metrics_for_frame(frame)
    result.update({
        "protocol": protocol,
        "scope": "all",
        "hop": "all",
        "n_story_groups": int(frame.story_group_id.nunique()),
        "majority_accuracy": float(frame.gold_relation.value_counts(normalize=True).max()),
        "prediction_collapse": prediction_collapse(result),
    })
    return result


def generation_stats(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for hop in HOPS:
        part = frame.loc[frame.hop.eq(hop)]
        rows.append({
            "protocol": protocol,
            "hop": hop,
            "n": len(part),
            "mean_generated_tokens": float(part.generation_token_count.mean()),
            "median_generated_tokens": float(part.generation_token_count.median()),
            "mean_thinking_tokens": float(part.thinking_token_count.mean()),
            "median_thinking_tokens": float(part.thinking_token_count.median()),
            "p25_thinking_tokens": float(part.thinking_token_count.quantile(0.25)),
            "p75_thinking_tokens": float(part.thinking_token_count.quantile(0.75)),
            "p90_thinking_tokens": float(part.thinking_token_count.quantile(0.90)),
            "truncation_rate": float(part.hit_max_new_tokens.mean()),
            "eos_termination_rate": float(part.eos_terminated.mean()),
            "think_parse_success_rate": float(part.think_parse_success.mean()),
            "mean_latency_seconds": float(part.latency_seconds.mean()),
            "summed_latency_seconds": float(part.latency_seconds.sum()),
            "tokens_per_second_summed": float(part.generation_token_count.sum() / part.latency_seconds.sum()),
        })
    return pd.DataFrame(rows)


def metric_value(frame: pd.DataFrame, metric: str, prediction_column: str = "parsed_relation") -> float:
    return metric_dict(frame.gold_relation, frame[prediction_column])[metric]


def resample_groups(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = np.array(sorted(frame.story_group_id.unique()))
    sampled = rng.choice(groups, size=len(groups), replace=True)
    parts = [frame.loc[frame.story_group_id.eq(group)] for group in sampled]
    return pd.concat(parts, ignore_index=True)


def bootstrap_single(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    for scope in [*HOPS, "derived"]:
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
            point = (
                metric_value(frame.loc[frame.hop.eq(scope)], metric)
                if scope != "derived"
                else float(np.mean([metric_value(frame.loc[frame.hop.eq(hop)], metric) for hop in [2, 3, 4, 5]]))
            )
            rng = np.random.default_rng(BOOTSTRAP_SEED + int(sha256_text(f"single:{protocol}:{scope}:{metric}")[:8], 16))
            values = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                if scope == "derived":
                    values.append(float(np.mean([
                        metric_value(resample_groups(frame.loc[frame.hop.eq(hop)], rng), metric)
                        for hop in [2, 3, 4, 5]
                    ])))
                else:
                    values.append(metric_value(resample_groups(frame.loc[frame.hop.eq(scope)], rng), metric))
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append({
                "protocol": protocol,
                "scope": f"h{scope}" if scope != "derived" else "h2-h5-derived",
                "metric": metric,
                "point": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "story_group_id within hop",
            })
    return pd.DataFrame(rows)


def paired_bootstrap(left: pd.DataFrame, right: pd.DataFrame, comparison: str,
                     left_label: str, right_label: str) -> pd.DataFrame:
    merged = left[["row_id", "story_group_id", "hop", "gold_relation", "parsed_relation"]].merge(
        right[["row_id", "parsed_relation"]],
        on="row_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"Paired comparison coverage mismatch: {comparison}")
    rows = []
    for scope in [*HOPS, "derived"]:
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
            def difference(part: pd.DataFrame) -> float:
                left_value = metric_dict(part.gold_relation, part.parsed_relation_left)[metric]
                right_value = metric_dict(part.gold_relation, part.parsed_relation_right)[metric]
                return right_value - left_value

            point = (
                difference(merged.loc[merged.hop.eq(scope)])
                if scope != "derived"
                else float(np.mean([difference(merged.loc[merged.hop.eq(hop)]) for hop in [2, 3, 4, 5]]))
            )
            rng = np.random.default_rng(BOOTSTRAP_SEED + int(sha256_text(f"paired:{comparison}:{scope}:{metric}")[:8], 16))
            values = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                if scope == "derived":
                    values.append(float(np.mean([
                        difference(resample_groups(merged.loc[merged.hop.eq(hop)], rng))
                        for hop in [2, 3, 4, 5]
                    ])))
                else:
                    values.append(difference(resample_groups(merged.loc[merged.hop.eq(scope)], rng)))
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append({
                "comparison": comparison,
                "left": left_label,
                "right": right_label,
                "difference_definition": f"{right_label} minus {left_label}",
                "scope": f"h{scope}" if scope != "derived" else "h2-h5-derived",
                "metric": metric,
                "point_difference": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired story_group_id within hop; derived is equal mean of h2,h3,h4,h5",
            })
    return pd.DataFrame(rows)


def candidate_frame() -> pd.DataFrame:
    scores = pd.read_csv(B22 / "native_sum" / "row_scores.csv")
    frame = scores[["row_id", "story_group_id", "hop", "gold_relation", "prediction_sum"]].drop_duplicates("row_id")
    frame = frame.rename(columns={"prediction_sum": "parsed_relation"}).sort_values("row_id").reset_index(drop=True)
    frame["protocol"] = "B2.2 Native-Sum"
    return frame


def pilot_analysis() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    for protocol_key, label in [("fg_nonthinking", "FG-N"), ("fg_thinking", "FG-T")]:
        for seed in GENERATION_SEEDS:
            frame = generation_frame(OUT / "pilot" / protocol_key / f"seed_{seed}" / "row_generations.jsonl")
            hop_metrics = metrics_by_hop(frame, label)
            for item in hop_metrics.to_dict("records"):
                item["generation_seed"] = seed
                item["decoding"] = "official-use"
                rows.append(item)
            derived = derived_row(hop_metrics, label)
            derived["generation_seed"] = seed
            derived["decoding"] = "official-use"
            rows.append(derived)
    pilot_metrics = pd.DataFrame(rows)
    sensitivity_rows = []
    max_range = 0.0
    for protocol in ["FG-N", "FG-T"]:
        values = pilot_metrics.loc[
            pilot_metrics.protocol.eq(protocol) & pilot_metrics.scope.eq("h2-h5-derived"),
            "accuracy",
        ]
        value_range = float(values.max() - values.min())
        max_range = max(max_range, value_range)
        sensitivity_rows.append({
            "protocol": protocol,
            "metric": "h2-h5 derived accuracy",
            "min": float(values.min()),
            "max": float(values.max()),
            "range": value_range,
            "seeds": json.dumps(GENERATION_SEEDS),
        })
    if max_range <= DECISION_THRESHOLDS["pilot_stochasticity_low_max_range"]:
        classification = "LOW"
    elif max_range <= DECISION_THRESHOLDS["pilot_stochasticity_moderate_max_range"]:
        classification = "MODERATE"
    else:
        classification = "HIGH"
    return pilot_metrics, pd.DataFrame(sensitivity_rows), classification


def matched_analysis() -> pd.DataFrame:
    rows = []
    for protocol_key, label in [("fg_nonthinking", "FG-N-Matched"), ("fg_thinking", "FG-T-Matched")]:
        frame = generation_frame(OUT / "pilot" / "matched_decoding" / protocol_key / "row_generations.jsonl")
        hop_metrics = metrics_by_hop(frame, label)
        rows.extend(hop_metrics.to_dict("records"))
        rows.append(derived_row(hop_metrics, label))
    return pd.DataFrame(rows)


def transition_outputs(candidate: pd.DataFrame, fgn: pd.DataFrame, fgt: pd.DataFrame) -> None:
    merged = candidate[["row_id", "story_group_id", "hop", "gold_relation", "parsed_relation"]].rename(
        columns={"parsed_relation": "candidate_prediction"}
    ).merge(
        fgn[["row_id", "parsed_relation"]].rename(columns={"parsed_relation": "fgn_prediction"}),
        on="row_id", validate="one_to_one",
    ).merge(
        fgt[["row_id", "parsed_relation"]].rename(columns={"parsed_relation": "fgt_prediction"}),
        on="row_id", validate="one_to_one",
    )
    merged["candidate_correct"] = merged.candidate_prediction.eq(merged.gold_relation)
    merged["fgn_correct"] = merged.fgn_prediction.eq(merged.gold_relation)
    merged["fgt_correct"] = merged.fgt_prediction.eq(merged.gold_relation)
    merged["relation_type"] = np.where(merged.gold_relation.isin(CARDINAL), "cardinal", "diagonal")
    merged["cl_wrong_to_fgn_correct"] = ~merged.candidate_correct & merged.fgn_correct
    merged["cl_correct_to_fgn_wrong"] = merged.candidate_correct & ~merged.fgn_correct
    merged["fgn_wrong_to_fgt_correct"] = ~merged.fgn_correct & merged.fgt_correct
    merged["fgn_correct_to_fgt_wrong"] = merged.fgn_correct & ~merged.fgt_correct
    merged["all_wrong"] = ~merged.candidate_correct & ~merged.fgn_correct & ~merged.fgt_correct
    merged["all_correct"] = merged.candidate_correct & merged.fgn_correct & merged.fgt_correct
    transition_columns = [
        "cl_wrong_to_fgn_correct", "cl_correct_to_fgn_wrong",
        "fgn_wrong_to_fgt_correct", "fgn_correct_to_fgt_wrong",
        "all_wrong", "all_correct",
    ]
    write_csv_atomic(OUT / "comparisons" / "error_transitions.csv", merged)
    write_csv_atomic(
        OUT / "comparisons" / "candidate_vs_generation.csv",
        merged[[
            "row_id", "story_group_id", "hop", "gold_relation", "relation_type",
            "candidate_prediction", "fgn_prediction", "candidate_correct", "fgn_correct",
            "cl_wrong_to_fgn_correct", "cl_correct_to_fgn_wrong",
        ]],
    )
    write_csv_atomic(
        OUT / "comparisons" / "nonthinking_vs_thinking.csv",
        merged[[
            "row_id", "story_group_id", "hop", "gold_relation", "relation_type",
            "fgn_prediction", "fgt_prediction", "fgn_correct", "fgt_correct",
            "fgn_wrong_to_fgt_correct", "fgn_correct_to_fgt_wrong",
        ]],
    )
    summaries = []
    for dimensions in [["hop"], ["gold_relation"], ["relation_type"]]:
        for keys, part in merged.groupby(dimensions):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = {"grouping": "+".join(dimensions), **dict(zip(dimensions, keys)), "n": len(part)}
            base.update({column: int(part[column].sum()) for column in transition_columns})
            summaries.append(base)
    write_csv_atomic(OUT / "comparisons" / "error_transition_summary.csv", pd.DataFrame(summaries))


def make_figures(candidate: pd.DataFrame, fgn: pd.DataFrame, fgt: pd.DataFrame,
                 metrics: pd.DataFrame, derived: pd.DataFrame,
                 paired: pd.DataFrame, generation: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for protocol in ["B2.2 Native-Sum", "FG-N", "FG-T"]:
        part = metrics.loc[metrics.protocol.eq(protocol) & metrics.hop.isin(HOPS)]
        ax.plot(part.hop.astype(int), part.accuracy, marker="o", linewidth=2.4,
                label=protocol, color=COLORS[protocol])
    ax.set(xlabel="Hop", ylabel="Accuracy", xticks=HOPS, ylim=(0, 1), title="Behavior by hop")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_1_behavior_by_hop.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot = derived.melt(
        id_vars=["protocol"], value_vars=["accuracy", "macro_f1", "balanced_accuracy"],
        var_name="metric", value_name="value",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot, x="metric", y="value", hue="protocol",
                palette=COLORS, ax=ax, edgecolor="#333333")
    ax.set(xlabel="Metric (h2-h5 equal-weight mean)", ylabel="Value", ylim=(0, 1), title="Derived multi-hop metrics")
    ax.legend(frameon=False, title=None)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_2_derived_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gain = paired.loc[
        paired.comparison.eq("FG-T minus FG-N") & paired.metric.eq("accuracy") & paired.scope.ne("h2-h5-derived")
    ].copy()
    gain["hop"] = gain.scope.str.replace("h", "", regex=False).astype(int)
    gain = gain.sort_values("hop")
    fig, ax = plt.subplots(figsize=(10, 6))
    error = np.vstack([
        gain.point_difference - gain.ci95_low,
        gain.ci95_high - gain.point_difference,
    ])
    ax.errorbar(gain.hop, gain.point_difference, yerr=error, fmt="o-", color=COLORS["FG-T"],
                capsize=5, linewidth=2.2)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(xlabel="Hop", ylabel="Accuracy difference (FG-T - FG-N)", xticks=HOPS,
           title="Thinking gain by hop with paired 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_3_thinking_gain_by_hop.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True, sharey=True)
    for ax, frame, protocol in zip(axes, [fgn, fgt], ["FG-N", "FG-T"]):
        _, normalized = confusion_frames(frame)
        sns.heatmap(normalized, vmin=0, vmax=1, cmap="Blues", annot=True, fmt=".2f",
                    xticklabels=[RELATION_WORDS[item] for item in RELATIONS],
                    yticklabels=[RELATION_WORDS[item] for item in RELATIONS], ax=ax, cbar=ax is axes[-1])
        ax.set(title=protocol, xlabel="Predicted", ylabel="Gold")
        ax.tick_params(axis="x", rotation=50)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Row-normalized confusion matrices", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_4_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    distributions = []
    for frame, protocol in [(candidate, "B2.2 Native-Sum"), (fgn, "FG-N"), (fgt, "FG-T")]:
        for relation in RELATIONS:
            distributions.append({
                "protocol": protocol,
                "relation": RELATION_WORDS[relation],
                "frequency": float(frame.parsed_relation.eq(relation).mean()),
                "type": "cardinal" if relation in CARDINAL else "diagonal",
            })
    dist = pd.DataFrame(distributions)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 1]})
    sns.barplot(data=dist, x="relation", y="frequency", hue="protocol", palette=COLORS, ax=axes[0])
    axes[0].set(xlabel="Predicted relation", ylabel="Frequency", title="Eight-relation prediction distribution")
    axes[0].tick_params(axis="x", rotation=45)
    type_dist = dist.groupby(["protocol", "type"], as_index=False).frequency.sum()
    sns.barplot(data=type_dist, x="type", y="frequency", hue="protocol", palette=COLORS, ax=axes[1])
    axes[1].set(xlabel="Prediction type", ylabel="Frequency", title="Cardinal vs diagonal")
    axes[0].legend(frameon=False, title=None)
    axes[1].legend_.remove()
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_5_prediction_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    thinking = generation.loc[generation.protocol.eq("FG-T")].sort_values("hop")
    fig, ax = plt.subplots(figsize=(10, 6))
    yerr = np.vstack([
        thinking.median_thinking_tokens - thinking.p25_thinking_tokens,
        thinking.p75_thinking_tokens - thinking.median_thinking_tokens,
    ])
    ax.errorbar(thinking.hop, thinking.median_thinking_tokens, yerr=yerr, fmt="o-",
                color=COLORS["FG-T"], capsize=5, linewidth=2.2, label="Median and IQR")
    ax.scatter(thinking.hop, thinking.p90_thinking_tokens, marker="D", color="#8C564B", label="p90")
    ax.set(xlabel="Hop", ylabel="Thinking tokens", xticks=HOPS, title="Thinking-token length by hop")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_6_reasoning_token_length.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def strong_gate(derived: dict, hop_metrics: pd.DataFrame, overall: dict) -> tuple[bool, dict]:
    meaningful = sum(derived[f"recall_{relation}"] >= DECISION_THRESHOLDS["meaningful_relation_recall"] for relation in RELATIONS)
    hop_above_majority = bool((hop_metrics.loc[hop_metrics.hop.isin([2, 3, 4, 5]), "accuracy"] > hop_metrics.loc[hop_metrics.hop.isin([2, 3, 4, 5]), "majority_accuracy"]).all())
    checks = {
        "derived_accuracy": derived["accuracy"] >= DECISION_THRESHOLDS["strong_derived_accuracy"],
        "derived_macro_f1": derived["macro_f1"] >= DECISION_THRESHOLDS["strong_derived_macro_f1"],
        "derived_balanced_accuracy": derived["balanced_accuracy"] >= DECISION_THRESHOLDS["strong_derived_balanced_accuracy"],
        "parse_success": derived["parse_success_rate"] >= DECISION_THRESHOLDS["strong_parse_success"],
        "no_prediction_collapse": not bool(overall["prediction_collapse"]),
        "all_h2_h5_above_majority": hop_above_majority,
        "majority_relations_meaningful_recall": meaningful >= DECISION_THRESHOLDS["minimum_meaningful_relations"],
    }
    return all(checks.values()), {**checks, "meaningful_relation_count": meaningful}


def decision_payload(full_metrics: pd.DataFrame, derived_frame: pd.DataFrame,
                     overall_frame: pd.DataFrame, paired: pd.DataFrame,
                     pilot_stochasticity: str, generation: pd.DataFrame) -> dict:
    derived = {row.protocol: row._asdict() for row in derived_frame.itertuples(index=False)}
    overall = {row.protocol: row._asdict() for row in overall_frame.itertuples(index=False)}
    fgn_hops = full_metrics.loc[full_metrics.protocol.eq("FG-N")]
    fgt_hops = full_metrics.loc[full_metrics.protocol.eq("FG-T")]
    fgn_strong, fgn_checks = strong_gate(derived["FG-N"], fgn_hops, overall["FG-N"])
    fgt_strong, fgt_checks = strong_gate(derived["FG-T"], fgt_hops, overall["FG-T"])
    fgt = derived["FG-T"]
    if fgt_strong:
        normal_fitness = "STRONG"
    elif (
        fgt["accuracy"] < DECISION_THRESHOLDS["weak_derived_accuracy"]
        or fgt["macro_f1"] < DECISION_THRESHOLDS["weak_derived_macro_f1"]
        or fgt["balanced_accuracy"] < DECISION_THRESHOLDS["weak_derived_balanced_accuracy"]
        or overall["FG-T"]["prediction_collapse"]
    ):
        normal_fitness = "WEAK"
    else:
        normal_fitness = "BORDERLINE"
    if normal_fitness == "STRONG":
        strong_target = "YES"
    elif normal_fitness == "BORDERLINE":
        strong_target = "BORDERLINE"
    elif fgt["parse_success_rate"] >= DECISION_THRESHOLDS["strong_parse_success"]:
        strong_target = "NO"
    else:
        strong_target = "UNCERTAIN"

    thinking = paired.loc[
        paired.comparison.eq("FG-T minus FG-N")
        & paired.scope.eq("h2-h5-derived")
        & paired.metric.eq("accuracy")
    ].iloc[0]
    if thinking.ci95_low > 0:
        thinking_benefit = "YES"
    elif thinking.ci95_high <= 0:
        thinking_benefit = "NO"
    else:
        thinking_benefit = "UNCERTAIN"

    free = paired.loc[
        paired.comparison.eq("FG-N minus B2.2 Native-Sum")
        & paired.scope.eq("h2-h5-derived")
        & paired.metric.eq("accuracy")
    ].iloc[0]
    if (
        free.point_difference >= DECISION_THRESHOLDS["underestimation_min_accuracy_gain"]
        and free.ci95_low > 0
        and derived["FG-N"]["parse_success_rate"] >= DECISION_THRESHOLDS["strong_parse_success"]
        and not overall["FG-N"]["prediction_collapse"]
    ):
        underestimates = "YES"
    elif abs(free.point_difference) <= DECISION_THRESHOLDS["underestimation_no_effect_margin"] and free.ci95_low <= 0 <= free.ci95_high:
        underestimates = "NO"
    else:
        underestimates = "UNCERTAIN"

    if fgn_strong:
        branch = "A"
        recommendation = "Use a non-thinking free-generation-compatible mechanistic design; do not switch models yet."
    elif fgt_strong:
        branch = "B"
        recommendation = "Redesign the mechanism question around prompt, generated reasoning trajectory, and answer states."
    else:
        branch = "C"
        recommendation = "Do not extract Qwen3-8B activations; behavior-gate Qwen/Qwen3.5-9B next."

    matched = paired.loc[
        paired.comparison.eq("FG-T-Matched minus FG-N-Matched")
        & paired.scope.eq("h2-h5-derived")
        & paired.metric.eq("accuracy")
    ]
    return {
        "candidate_likelihood_underestimates_behavior": underestimates,
        "thinking_improves_free_generation": thinking_benefit,
        "qwen3_normal_use_task_fitness": {"label": {"STRONG": "YES", "BORDERLINE": "BORDERLINE", "WEAK": "NO"}[normal_fitness], "display": normal_fitness},
        "normal_use_task_fitness": normal_fitness,
        "strong_reasoning_target": strong_target,
        "recommended_next_step": branch,
        "recommendation": recommendation,
        "fg_n_strong_gate": {"pass": fgn_strong, "checks": fgn_checks},
        "fg_t_strong_gate": {"pass": fgt_strong, "checks": fgt_checks},
        "pilot_stochasticity": pilot_stochasticity,
        "thinking_primary_evidence": thinking.to_dict(),
        "free_generation_descriptive_evidence": free.to_dict(),
        "matched_decoding_accuracy_evidence": matched.iloc[0].to_dict() if len(matched) else None,
        "decision_thresholds_frozen_before_generation": DECISION_THRESHOLDS,
        "max_full_truncation_rate": float(generation.truncation_rate.max()),
        "validation_status": "PENDING",
    }


def readiness_note() -> str:
    return """# Qwen3.5-9B Readiness Note

Stage B2.3 selected Case C. No Qwen3.5 model was downloaded or run.

- Candidate model: `Qwen/Qwen3.5-9B`
- Candidate base: `Qwen/Qwen3.5-9B-Base`
- Re-audit tokenizer and chat template.
- Re-confirm reasoning controls and generation defaults.
- Re-audit model architecture; do not assume the Qwen3-8B 36-layer standard Transformer layout.
- Run a 24 GB RTX 3090 memory smoke test before any full experiment.
- Apply a behavior gate before activation extraction.
"""


def make_report(full_metrics: pd.DataFrame, derived: pd.DataFrame, overall: pd.DataFrame,
                pilot: pd.DataFrame, pilot_sensitivity: pd.DataFrame, matched: pd.DataFrame,
                paired: pd.DataFrame, generation: pd.DataFrame, decision: dict) -> str:
    fgn = full_metrics.loc[full_metrics.protocol.eq("FG-N")]
    fgt = full_metrics.loc[full_metrics.protocol.eq("FG-T")]
    fgnder = derived.loc[derived.protocol.eq("FG-N")].iloc[0]
    fgtder = derived.loc[derived.protocol.eq("FG-T")].iloc[0]
    think = paired.loc[
        paired.comparison.eq("FG-T minus FG-N") & paired.scope.eq("h2-h5-derived") & paired.metric.eq("accuracy")
    ].iloc[0]
    matched_derived = matched.loc[matched.scope.eq("h2-h5-derived")]
    config = read_json(OUT / "audit" / "generation_config.json")
    model = read_json(OUT / "audit" / "model_manifest.json")
    env = read_json(OUT / "audit" / "environment.json")
    pilot_summary = pilot.loc[pilot.scope.eq("h2-h5-derived"), ["protocol", "generation_seed", "accuracy", "macro_f1", "balanced_accuracy", "parse_success_rate"]]
    thinking_stats = generation.loc[generation.protocol.eq("FG-T"), ["hop", "median_thinking_tokens", "p25_thinking_tokens", "p75_thinking_tokens", "p90_thinking_tokens", "truncation_rate"]]
    return f"""# Stage B2.3 Qwen3-8B Free-Generation / Thinking Gate

Validation: **<!-- VALIDATION_STATUS -->PENDING**

## 1. Research Question

Does candidate likelihood underestimate Qwen3-8B StepGame task fitness, and does official Qwen3 thinking mode further improve multi-hop free-generation behavior?

## 2. Frozen Reference

- Primary test: 2,063 frozen rows; h1–h5 = 712 / 505 / 422 / 282 / 142.
- Row-ID SHA-256: `dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950`.
- B2.2 Qwen Native-Sum h2–h5 derived accuracy: `{EXPECTED_B22_NATIVE_SUM_DERIVED_ACCURACY:.4f}`.
- B2.2 strong-reasoning target: `UNCERTAIN`; B2.2 validation: `PASS`.

## 3. Model and Environment Audit

- Model: `Qwen/Qwen3-8B`, local BF16 weights, 36 layers, hidden size 4096.
- Config SHA-256: `{model['config_sha256']}`.
- Chat-template SHA-256: `{model['chat_template_sha256']}`.
- Weight and tokenizer manifests match B2.2 exactly: **{model['status']}**.
- Transformers `{env['transformers']}`; PyTorch `{env['torch']}`; CUDA `{env['torch_cuda']}`; GPU `{env['gpu']}`.

## 4. Generation Protocols

- FG-N: Native chat template, free generation, `enable_thinking=False`, temperature 0.7, top-p 0.8, top-k 20, min-p 0.
- FG-T: Native chat template, free generation, `enable_thinking=True`, temperature 0.6, top-p 0.95, top-k 20, min-p 0.
- Matched sensitivity: both modes use temperature 0.6, top-p 0.95, top-k 20, min-p 0 on the frozen 250-row pilot subset.
- Resolved `max_new_tokens`: `{config['resolved_max_new_tokens']}` based only on the preregistered smoke truncation audit.
- Full run seed: `{PRIMARY_GENERATION_SEED}`. Per-row child seeds are deterministic hashes of generation seed and row ID, which makes resume invariant to interruption.

## 5. Parser and Output Format Audit

The only prompt addition was the exact frozen output-format instruction. The deterministic parser selects the last valid `Final answer: <relation>` line, normalizing only case, whitespace, and `_`/space/`-` separators for diagonal labels. No semantic synonyms are accepted. `INVALID` counts as wrong. Parser and prompt hashes were frozen before generation.

## 6. Stability Pilot

{markdown_table(pilot_summary, ['protocol','generation_seed','accuracy','macro_f1','balanced_accuracy','parse_success_rate'])}

Stochasticity classification: **{decision['pilot_stochasticity']}**. Accuracy ranges were preregistered as LOW ≤ 0.03, MODERATE ≤ 0.07, and HIGH > 0.07.

{markdown_table(pilot_sensitivity, ['protocol','min','max','range'])}

## 7. FG-N Results

{markdown_table(fgn, ['hop','n','accuracy','macro_f1','balanced_accuracy','parse_success_rate','prediction_collapse'])}

h2–h5 equal-weight means: accuracy `{fgnder.accuracy:.4f}`, Macro-F1 `{fgnder.macro_f1:.4f}`, balanced accuracy `{fgnder.balanced_accuracy:.4f}`, parse rate `{fgnder.parse_success_rate:.4f}`.

## 8. FG-T Results

{markdown_table(fgt, ['hop','n','accuracy','macro_f1','balanced_accuracy','parse_success_rate','prediction_collapse'])}

h2–h5 equal-weight means: accuracy `{fgtder.accuracy:.4f}`, Macro-F1 `{fgtder.macro_f1:.4f}`, balanced accuracy `{fgtder.balanced_accuracy:.4f}`, parse rate `{fgtder.parse_success_rate:.4f}`.

## 9. Free Generation vs Candidate Likelihood

Candidate likelihood and free generation are different answer-production protocols. The paired row description estimates FG-N minus B2.2 Native-Sum derived accuracy as `{decision['free_generation_descriptive_evidence']['point_difference']:.4f}` with a descriptive 95% cluster-bootstrap interval `[{decision['free_generation_descriptive_evidence']['ci95_low']:.4f}, {decision['free_generation_descriptive_evidence']['ci95_high']:.4f}]`. This supports only the statement that free-generation behavior differs from candidate-likelihood behavior; it does not establish that one protocol mathematically corrects the other.

![Behavior by hop](figures/figure_1_behavior_by_hop.png)

## 10. Thinking Effect

FG-T minus FG-N h2–h5 derived accuracy is `{think.point_difference:.4f}`, 95% paired cluster-bootstrap CI `[{think.ci95_low:.4f}, {think.ci95_high:.4f}]`. Thinking benefit: **{decision['thinking_improves_free_generation']}**.

Matched-decoding sensitivity:

{markdown_table(matched_derived, ['protocol','accuracy','macro_f1','balanced_accuracy','parse_success_rate'])}

Because the official-use configurations differ, only the matched subset isolates mode while holding sampling parameters fixed.

## 11. Hop Interaction

The per-hop gain is shown with paired 95% intervals. Any larger multi-hop gain remains descriptive evidence and is not evidence of an internal graph representation.

![Thinking gain by hop](figures/figure_3_thinking_gain_by_hop.png)

## 12. Relation / Confusion Analysis

Prediction-collapse gates use cardinal or diagonal frequency ≥ 0.95, or a single relation frequency ≥ 0.80. Full-run status: FG-N `{bool(overall.loc[overall.protocol.eq('FG-N'), 'prediction_collapse'].iloc[0])}`, FG-T `{bool(overall.loc[overall.protocol.eq('FG-T'), 'prediction_collapse'].iloc[0])}`.

![Confusion matrices](figures/figure_4_confusion_matrices.png)

![Prediction distribution](figures/figure_5_prediction_distribution.png)

## 13. Generation Length Analysis

{markdown_table(thinking_stats, ['hop','median_thinking_tokens','p25_thinking_tokens','p75_thinking_tokens','p90_thinking_tokens','truncation_rate'], decimals=2)}

![Thinking-token length](figures/figure_6_reasoning_token_length.png)

Longer thinking is descriptive computation amount, not evidence of causal benefit.

## 14. Statistical Comparisons

All intervals use 2,000 resamples, seed 20260910, with paired `story_group_id` sampling. Derived comparisons compute each hop separately and then take the equal-weight mean of h2–h5.

![Derived metrics](figures/figure_2_derived_metrics.png)

## 15. Fitness Decision

- `candidate_likelihood_underestimates_behavior`: **{decision['candidate_likelihood_underestimates_behavior']}**
- `thinking_improves_free_generation`: **{decision['thinking_improves_free_generation']}**
- `normal_use_task_fitness`: **{decision['normal_use_task_fitness']}**
- `strong_reasoning_target`: **{decision['strong_reasoning_target']}**

## 16. Consequence for Stage B

Branch **{decision['recommended_next_step']}**: {decision['recommendation']}

## 17. Interpretation Boundaries

- Free-generation improvement is not hidden-mechanism improvement.
- Thinking improvement is not proof of symbolic reasoning.
- Thinking text is not assumed to be a faithful hidden-computation trace.
- Longer thinking is not assumed to causally improve reasoning.
- StepGame behavior does not establish graph formation.
- Candidate likelihood and free generation are different behavioral measurements.
- This stage does not justify graph, activation, probe, patching, or steering claims.
"""


def main() -> None:
    ensure_dirs()
    required_manifests = [
        OUT / "full" / "fg_nonthinking" / "run_manifest.json",
        OUT / "full" / "fg_thinking" / "run_manifest.json",
    ]
    if any(read_json(path).get("status") != "PASS" for path in required_manifests):
        raise RuntimeError("Full generation manifests are not PASS")

    fgn = generation_frame(OUT / "full" / "fg_nonthinking" / "row_generations.jsonl")
    fgt = generation_frame(OUT / "full" / "fg_thinking" / "row_generations.jsonl")
    candidate = candidate_frame()
    if not (fgn.row_id.tolist() == fgt.row_id.tolist() == candidate.row_id.tolist()):
        raise RuntimeError("Full-run and B2.2 row coverage/order differ")

    fgn_metrics = metrics_by_hop(fgn, "FG-N")
    fgt_metrics = metrics_by_hop(fgt, "FG-T")
    candidate_metrics = metrics_by_hop(candidate, "B2.2 Native-Sum")
    full_metrics = pd.concat([candidate_metrics, fgn_metrics, fgt_metrics], ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "full_metrics_by_hop.csv", full_metrics)
    derived = pd.DataFrame([
        derived_row(candidate_metrics, "B2.2 Native-Sum"),
        derived_row(fgn_metrics, "FG-N"),
        derived_row(fgt_metrics, "FG-T"),
    ])
    write_csv_atomic(OUT / "statistics" / "derived_summary.csv", derived)
    overall = pd.DataFrame([
        overall_row(candidate, "B2.2 Native-Sum"),
        overall_row(fgn, "FG-N"),
        overall_row(fgt, "FG-T"),
    ])
    write_csv_atomic(OUT / "statistics" / "overall_metrics.csv", overall)

    generation = pd.concat([
        generation_stats(fgn, "FG-N"), generation_stats(fgt, "FG-T")
    ], ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "generation_stats.csv", generation)

    pilot, pilot_sensitivity, pilot_class = pilot_analysis()
    write_csv_atomic(OUT / "statistics" / "pilot_metrics.csv", pilot)
    write_csv_atomic(OUT / "statistics" / "pilot_stochasticity.csv", pilot_sensitivity)
    matched = matched_analysis()
    write_csv_atomic(OUT / "statistics" / "matched_decoding_metrics.csv", matched)

    matched_fgn = generation_frame(OUT / "pilot" / "matched_decoding" / "fg_nonthinking" / "row_generations.jsonl")
    matched_fgt = generation_frame(OUT / "pilot" / "matched_decoding" / "fg_thinking" / "row_generations.jsonl")
    paired = pd.concat([
        paired_bootstrap(fgn, fgt, "FG-T minus FG-N", "FG-N", "FG-T"),
        paired_bootstrap(candidate, fgn, "FG-N minus B2.2 Native-Sum", "B2.2 Native-Sum", "FG-N"),
        paired_bootstrap(candidate, fgt, "FG-T minus B2.2 Native-Sum", "B2.2 Native-Sum", "FG-T"),
        paired_bootstrap(matched_fgn, matched_fgt, "FG-T-Matched minus FG-N-Matched", "FG-N-Matched", "FG-T-Matched"),
    ], ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "paired_bootstrap.csv", paired)
    intervals = pd.concat([
        bootstrap_single(fgn, "FG-N"), bootstrap_single(fgt, "FG-T")
    ], ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "bootstrap_ci.csv", intervals)

    transition_outputs(candidate, fgn, fgt)
    for protocol, frame in [("fg_nonthinking", fgn), ("fg_thinking", fgt)]:
        raw, normalized = confusion_frames(frame)
        directory = OUT / "statistics" / "confusion_matrices" / protocol
        directory.mkdir(parents=True, exist_ok=True)
        raw.to_csv(directory / "all_count.csv")
        normalized.to_csv(directory / "all_row_normalized.csv")
        for hop in HOPS:
            hop_raw, hop_norm = confusion_frames(frame.loc[frame.hop.eq(hop)])
            hop_raw.to_csv(directory / f"h{hop}_count.csv")
            hop_norm.to_csv(directory / f"h{hop}_row_normalized.csv")

    decision = decision_payload(full_metrics, derived, overall, paired, pilot_class, generation)
    write_json_atomic(OUT / "fitness_decision.json", decision)
    if decision["recommended_next_step"] == "C":
        write_text_atomic(OUT / "QWEN35_READINESS_NOTE.md", readiness_note())
    make_figures(candidate, fgn, fgt, full_metrics, derived, paired, generation)
    report = make_report(full_metrics, derived, overall, pilot, pilot_sensitivity, matched, paired, generation, decision)
    write_text_atomic(OUT / "STAGE_B2_3_QWEN3_GENERATION_REPORT.md", report)
    print(json.dumps({
        "stage": "B2.3", "status": "FINALIZED_PENDING_INDEPENDENT_VALIDATION",
        "normal_use_task_fitness": decision["normal_use_task_fitness"],
        "strong_reasoning_target": decision["strong_reasoning_target"],
        "recommended_next_step": decision["recommended_next_step"],
    }, indent=2))


if __name__ == "__main__":
    main()

