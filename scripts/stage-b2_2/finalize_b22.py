#!/usr/bin/env python3
"""Finalize Stage B2.2 metrics, comparisons, figures, decision, and report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_b22 import (
    B21, BOOTSTRAP_RESAMPLES, CARDINAL, DIAGONAL, GATE_THRESHOLDS, HOPS,
    NATURAL_CANDIDATES, OUT, RELATIONS, SEED, attach_truth, confusion_frames,
    ensure_dirs, load_primary, metric_dict, metrics_by_hop, read_json,
    reconstruct_predictions, sha256_text, write_csv_atomic, write_json_atomic,
)


PROTOCOLS = ["Raw-Sum", "Raw-Mean", "Native-Sum", "Native-Mean", "EqualLength-Raw", "EqualLength-Native"]
SCORE_SOURCES = {
    "Raw-Sum": (OUT / "raw_sum" / "row_scores.csv", "sum_log_probability"),
    "Raw-Mean": (OUT / "raw_sum" / "row_scores.csv", "mean_log_probability"),
    "Native-Sum": (OUT / "native_sum" / "row_scores.csv", "sum_log_probability"),
    "Native-Mean": (OUT / "native_sum" / "row_scores.csv", "mean_log_probability"),
    "EqualLength-Raw": (OUT / "scoring_sensitivity" / "equal_length_raw" / "row_scores.csv", "sum_log_probability"),
    "EqualLength-Native": (OUT / "scoring_sensitivity" / "equal_length_native" / "row_scores.csv", "sum_log_probability"),
}
COLORS = {"Qwen3-8B": "#4C78A8", "Tulu Final": "#E45756"}


def load_qwen_predictions(primary: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    predictions, score_frames = {}, {}
    for protocol, (path, column) in SCORE_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        scores = pd.read_csv(path)
        score_frames[protocol] = scores
        prediction = attach_truth(reconstruct_predictions(scores, column), primary)
        prediction["model"] = "qwen3_8b"
        prediction["protocol"] = protocol
        predictions[protocol] = prediction
    return predictions, score_frames


def locate_tulu_scores() -> tuple[Path, Path]:
    raw = B21 / "raw_reproduction" / "row_scores.csv"
    native_options = [
        B21 / "native_chat" / "per_model" / "final_scores.csv",
        B21 / "native_chat" / "row_scores_sum.csv",
    ]
    native = next((path for path in native_options if path.is_file()), native_options[0])
    if not raw.is_file() or not native.is_file():
        raise FileNotFoundError(
            "B2.1 row-level Tulu Final scores are required for paired comparison. "
            f"Expected {raw} and one of {native_options}"
        )
    return raw, native


def load_tulu_predictions(primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_path, native_path = locate_tulu_scores()
    raw = pd.read_csv(raw_path)
    if "model" in raw.columns:
        raw = raw.loc[raw.model.eq("final")].copy()
    native = pd.read_csv(native_path)
    if "model" in native.columns:
        native = native.loc[native.model.eq("final")].copy()
    output = {}
    for protocol, scores in [("Raw-Sum", raw), ("Native-Sum", native)]:
        pred = attach_truth(reconstruct_predictions(scores, "sum_log_probability"), primary)
        pred["model"] = "tulu_final"
        pred["protocol"] = protocol
        output[protocol] = pred
    return output


def matrices_by_group(frame: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
    relation_index = {relation: index for index, relation in enumerate(RELATIONS)}
    group_index = {group: index for index, group in enumerate(groups)}
    result = np.zeros((len(groups), len(RELATIONS), len(RELATIONS)), dtype=np.int64)
    for row in frame.itertuples(index=False):
        result[group_index[row.story_group_id], relation_index[row.canonical_relation], relation_index[row.prediction]] += 1
    return result


def metrics_from_matrices(matrices: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    total = matrices.sum(axis=(1, 2))
    accuracy = np.divide(diagonal.sum(axis=1), total, out=np.zeros(len(matrices), dtype=float), where=total != 0)
    false_positive = matrices.sum(axis=1) - diagonal
    false_negative = matrices.sum(axis=2) - diagonal
    f1_denominator = 2 * diagonal + false_positive + false_negative
    per_class_f1 = np.divide(2 * diagonal, f1_denominator, out=np.zeros_like(diagonal, dtype=float), where=f1_denominator != 0)
    true_counts = matrices.sum(axis=2)
    recalls = np.divide(diagonal, true_counts, out=np.zeros_like(diagonal, dtype=float), where=true_counts != 0)
    return {"accuracy": accuracy, "macro_f1": per_class_f1.mean(axis=1), "balanced_accuracy": recalls.mean(axis=1)}


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return metric_dict(frame.canonical_relation, frame.prediction)


def bootstrap_one(prediction: pd.DataFrame, model: str, protocol: str) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    rows = []
    distributions = {}
    for hop in HOPS:
        part = prediction.loc[prediction.computed_hop.eq(hop)].reset_index(drop=True)
        groups = np.array(sorted(part.story_group_id.unique()))
        matrices = matrices_by_group(part, groups)
        seed_offset = int(sha256_text(model + protocol)[:8], 16)
        rng = np.random.default_rng(SEED + seed_offset + hop * 1009)
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=BOOTSTRAP_RESAMPLES)
        metrics = metrics_from_matrices(np.tensordot(weights, matrices, axes=(1, 0)))
        distributions[hop] = metrics
        points = point_metrics(part)
        for metric, values in metrics.items():
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append({
                "model": model, "protocol": protocol, "scope": f"h{hop}", "hop": str(hop),
                "metric": metric, "point": points[metric], "ci95_low": float(low), "ci95_high": float(high),
                "n_rows": len(part), "n_bootstrap_units": len(groups),
                "bootstrap_unit": "story_group_id", "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED,
            })
    point_hops = {
        hop: point_metrics(prediction.loc[prediction.computed_hop.eq(hop)])
        for hop in [2, 3, 4, 5]
    }
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
        values = np.mean(np.stack([distributions[hop][metric] for hop in [2, 3, 4, 5]]), axis=0)
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({
            "model": model, "protocol": protocol, "scope": "h2_h5_derived", "hop": "derived",
            "metric": metric, "point": float(np.mean([point_hops[hop][metric] for hop in [2, 3, 4, 5]])),
            "ci95_low": float(low), "ci95_high": float(high),
            "n_rows": int(prediction.computed_hop.isin([2, 3, 4, 5]).sum()),
            "n_bootstrap_units": int(sum(prediction.loc[prediction.computed_hop.eq(hop), "story_group_id"].nunique() for hop in [2, 3, 4, 5])),
            "bootstrap_unit": "story_group_id within hop, then equal-weight mean across h2-h5",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED,
        })
    return pd.DataFrame(rows), distributions


def paired_bootstrap(qwen: pd.DataFrame, tulu: pd.DataFrame, protocol: str) -> pd.DataFrame:
    merged = qwen[["row_id", "story_group_id", "computed_hop", "canonical_relation", "prediction"]].merge(
        tulu[["row_id", "prediction"]], on="row_id", suffixes=("_qwen", "_tulu"), validate="one_to_one"
    )
    rows = []
    distributions = {}
    points = {}
    for hop in HOPS:
        part = merged.loc[merged.computed_hop.eq(hop)].reset_index(drop=True)
        groups = np.array(sorted(part.story_group_id.unique()))
        q = part.rename(columns={"prediction_qwen": "prediction"})
        t = part.rename(columns={"prediction_tulu": "prediction"})
        qm = matrices_by_group(q, groups)
        tm = matrices_by_group(t, groups)
        rng = np.random.default_rng(SEED + int(sha256_text("paired" + protocol)[:8], 16) + hop * 1013)
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=BOOTSTRAP_RESAMPLES)
        q_metrics = metrics_from_matrices(np.tensordot(weights, qm, axes=(1, 0)))
        t_metrics = metrics_from_matrices(np.tensordot(weights, tm, axes=(1, 0)))
        distributions[hop] = {metric: q_metrics[metric] - t_metrics[metric] for metric in q_metrics}
        points[hop] = {
            metric: point_metrics(q)[metric] - point_metrics(t)[metric]
            for metric in ["accuracy", "macro_f1", "balanced_accuracy"]
        }
        for metric, values in distributions[hop].items():
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append({
                "comparison": "Qwen3-8B minus Tulu Final", "protocol": protocol,
                "scope": f"h{hop}", "hop": str(hop), "metric": metric,
                "point_difference": points[hop][metric], "ci95_low": float(low), "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0), "n_rows": len(part),
                "n_bootstrap_units": len(groups), "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_unit": "paired story_group_id", "seed": SEED,
            })
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
        values = np.mean(np.stack([distributions[hop][metric] for hop in [2, 3, 4, 5]]), axis=0)
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({
            "comparison": "Qwen3-8B minus Tulu Final", "protocol": protocol,
            "scope": "h2_h5_derived", "hop": "derived", "metric": metric,
            "point_difference": float(np.mean([points[hop][metric] for hop in [2, 3, 4, 5]])),
            "ci95_low": float(low), "ci95_high": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "n_rows": int(merged.computed_hop.isin([2, 3, 4, 5]).sum()),
            "n_bootstrap_units": int(sum(merged.loc[merged.computed_hop.eq(hop), "story_group_id"].nunique() for hop in [2, 3, 4, 5])),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "paired story_group_id within hop, then equal-weight mean across h2-h5", "seed": SEED,
        })
    return pd.DataFrame(rows)


def prediction_frequencies(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for protocol, frame in predictions.items():
        for scope, part in [("all", frame), *[(f"h{hop}", frame.loc[frame.computed_hop.eq(hop)]) for hop in HOPS]]:
            frequencies = part.prediction.value_counts(normalize=True)
            gold = part.canonical_relation.value_counts(normalize=True)
            row = {
                "model": "qwen3_8b", "protocol": protocol, "scope": scope, "n": len(part),
                "predicted_cardinal_frequency": float(part.prediction.isin(CARDINAL).mean()),
                "predicted_diagonal_frequency": float(part.prediction.isin(DIAGONAL).mean()),
                "gold_cardinal_frequency": float(part.canonical_relation.isin(CARDINAL).mean()),
                "gold_diagonal_frequency": float(part.canonical_relation.isin(DIAGONAL).mean()),
                "max_single_relation_frequency": float(frequencies.max()),
                "dominant_relation": str(frequencies.idxmax()),
            }
            row.update({f"predicted_{relation}": float(frequencies.get(relation, 0.0)) for relation in RELATIONS})
            row.update({f"gold_{relation}": float(gold.get(relation, 0.0)) for relation in RELATIONS})
            rows.append(row)
    return pd.DataFrame(rows)


def derived_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, protocol), frame in metrics.groupby(["model", "protocol"]):
        derived = frame.loc[frame.hop.isin([2, 3, 4, 5])]
        rows.append({
            "model": model, "protocol": protocol,
            "accuracy_h2_h5": float(derived.accuracy.mean()),
            "macro_f1_h2_h5": float(derived.macro_f1.mean()),
            "balanced_accuracy_h2_h5": float(derived.balanced_accuracy.mean()),
        })
    return pd.DataFrame(rows)


def verify_tulu_reference(metrics: pd.DataFrame) -> dict:
    reference_path = B21 / "statistics" / "all_hop_metrics.csv"
    reference = pd.read_csv(reference_path)
    details = []
    for protocol in ["Raw-Sum", "Native-Sum"]:
        current = metrics.loc[(metrics.model.eq("tulu_final")) & (metrics.protocol.eq(protocol))].set_index("hop")
        for hop in HOPS:
            expected = reference.loc[
                (reference.model.eq("final")) & (reference.protocol.eq(protocol)) & (reference.hop.eq(hop))
            ].iloc[0]
            current_correct = int(round(float(current.loc[hop, "accuracy"]) * int(current.loc[hop, "n_test"])))
            expected_correct = int(round(float(expected["accuracy"]) * int(current.loc[hop, "n_test"])))
            details.append({
                "protocol": protocol, "hop": hop, "current_correct": current_correct,
                "expected_correct": expected_correct, "pass": current_correct == expected_correct,
            })
    return {"status": "PASS" if all(item["pass"] for item in details) else "FAIL", "checks": details, "source": str(reference_path)}


def save_confusions(predictions: dict[str, pd.DataFrame]) -> None:
    for protocol, frame in predictions.items():
        directory = OUT / "statistics" / "confusion_matrices" / protocol.lower().replace("-", "_")
        directory.mkdir(parents=True, exist_ok=True)
        for hop in HOPS:
            raw, normalized = confusion_frames(frame, hop)
            raw.to_csv(directory / f"h{hop}_count.csv")
            normalized.to_csv(directory / f"h{hop}_row_normalized.csv")


def collapse_status(frequency: pd.DataFrame, protocol: str) -> tuple[bool, list[dict]]:
    rows = frequency.loc[frequency.protocol.eq(protocol)]
    flagged = rows.loc[
        (rows.predicted_cardinal_frequency > GATE_THRESHOLDS["collapse_cardinal_or_diagonal_frequency"])
        | (rows.predicted_diagonal_frequency > GATE_THRESHOLDS["collapse_cardinal_or_diagonal_frequency"])
        | (rows.max_single_relation_frequency > GATE_THRESHOLDS["collapse_single_relation_frequency"])
    ]
    return not flagged.empty, flagged[["scope", "predicted_cardinal_frequency", "predicted_diagonal_frequency", "max_single_relation_frequency", "dominant_relation"]].to_dict("records")


def make_decision(metrics: pd.DataFrame, derived: pd.DataFrame, paired: pd.DataFrame, frequency: pd.DataFrame) -> dict:
    native_comparison = paired.loc[(paired.protocol.eq("Native-Sum")) & (paired.scope.eq("h2_h5_derived"))]
    comparison = {row.metric: row for row in native_comparison.itertuples(index=False)}
    stronger_components = {
        metric: "YES" if comparison[metric].ci95_low > 0 else ("NO" if comparison[metric].ci95_high <= 0 else "UNCERTAIN")
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]
    }
    native_stronger = "YES" if all(value == "YES" for value in stronger_components.values()) else (
        "NO" if any(value == "NO" for value in stronger_components.values()) else "UNCERTAIN"
    )
    qnative = metrics.loc[(metrics.model.eq("qwen3_8b")) & (metrics.protocol.eq("Native-Sum")) & (metrics.hop.isin([2, 3, 4, 5]))]
    above_uniform = "YES" if bool((qnative.accuracy > 0.125).all()) else "NO"
    above_majority = "YES" if bool((qnative.accuracy > qnative.majority_accuracy).all()) else "NO"
    raw_collapse, raw_flags = collapse_status(frequency, "Raw-Sum")
    native_collapse, native_flags = collapse_status(frequency, "Native-Sum")
    protocol_collapse = {}
    for protocol in ["Native-Sum", "Native-Mean", "EqualLength-Native"]:
        protocol_collapse[protocol] = collapse_status(frequency, protocol)[0]
    native_values = derived.loc[
        (derived.model.eq("qwen3_8b")) & derived.protocol.isin(["Native-Sum", "Native-Mean", "EqualLength-Native"]),
        "accuracy_h2_h5",
    ]
    accuracy_range = float(native_values.max() - native_values.min())
    high_sensitivity = (
        accuracy_range > GATE_THRESHOLDS["high_sensitivity_derived_accuracy_range"]
        and float(native_values.min()) <= GATE_THRESHOLDS["high_sensitivity_weak_protocol_accuracy"]
    ) or any(protocol_collapse.values())
    audits = [
        read_json(OUT / "audit" / "frozen_data_audit.json")["status"],
        read_json(OUT / "audit" / "tokenizer_audit.json")["status"],
        read_json(OUT / "audit" / "equal_length_gate.json")["status"],
    ]
    data_integrity = "PASS" if all(value == "PASS" for value in audits) else "FAIL"
    if data_integrity == "FAIL" or native_stronger == "NO":
        target = "NO"
    elif all([native_stronger == "YES", above_uniform == "YES", above_majority == "YES", not native_collapse, not high_sensitivity]):
        target = "YES"
    else:
        target = "UNCERTAIN"

    raw_comparison = paired.loc[(paired.protocol.eq("Raw-Sum")) & (paired.scope.eq("h2_h5_derived")) & (paired.metric.eq("accuracy"))].iloc[0]
    raw_strong = raw_comparison.ci95_low > 0
    native_strong = comparison["accuracy"].ci95_low > 0
    if target == "YES" and raw_strong and native_strong:
        consequence = "A. Qwen strong gate PASS: retain Tulu as weak control and Qwen as strong target; Raw common-interface hidden-state comparison is preferred."
    elif native_strong and not raw_strong:
        consequence = "B. Qwen Native strong but Raw weak: do not compare Tulu Raw states with Qwen Native states; extract Native states for both models."
    else:
        consequence = "C. Qwen remains weak or uncertain: do not start full activation extraction; examine free generation/thinking-mode evaluation or a stronger model in a future preregistered stage."
    return {
        "data_integrity": data_integrity,
        "tokenizer_boundary_audit": read_json(OUT / "audit" / "tokenizer_audit.json")["status"],
        "native_behavior_stronger_than_tulu": native_stronger,
        "above_uniform_chance": above_uniform,
        "above_majority_baseline": above_majority,
        "macro_f1_improved": stronger_components["macro_f1"],
        "balanced_accuracy_improved": stronger_components["balanced_accuracy"],
        "prediction_collapse": {
            "status": "YES" if native_collapse else "NO", "raw": "YES" if raw_collapse else "NO",
            "native": "YES" if native_collapse else "NO", "raw_flags": raw_flags, "native_flags": native_flags,
        },
        "protocol_sensitivity": {
            "status": "HIGH" if high_sensitivity else "ACCEPTABLE",
            "native_protocol_derived_accuracy_range": accuracy_range,
            "protocol_collapse": protocol_collapse,
        },
        "strong_reasoning_target": target,
        "recommended_next_step": consequence,
        "gate_thresholds_frozen_before_accuracy": GATE_THRESHOLDS,
        "native_comparison_evidence": {
            metric: {
                "point_difference": float(comparison[metric].point_difference),
                "ci95_low": float(comparison[metric].ci95_low),
                "ci95_high": float(comparison[metric].ci95_high),
            } for metric in comparison
        },
    }


def draw_figures(metrics: pd.DataFrame, derived: pd.DataFrame, predictions: dict[str, pd.DataFrame], tulu_predictions: dict[str, pd.DataFrame], frequency: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, protocols in [("qwen3_8b", ["Raw-Sum", "Native-Sum"]), ("tulu_final", ["Raw-Sum", "Native-Sum"])]:
        for protocol in protocols:
            part = metrics.loc[(metrics.model.eq(model)) & (metrics.protocol.eq(protocol))]
            label = ("Qwen3-8B" if model == "qwen3_8b" else "Tulu Final") + " " + protocol
            ax.plot(part.hop, part.accuracy, marker="o", lw=2, label=label, linestyle="-" if protocol == "Native-Sum" else "--")
    ax.axhline(0.125, color="black", ls=":", lw=1, label="Uniform chance")
    ax.set(xlabel="Hop", ylabel="Accuracy", title="Figure 1. Raw-Sum vs Native-Sum by hop", xticks=HOPS, ylim=(0, 1))
    ax.legend(ncol=2); fig.tight_layout(); fig.savefig(figure_dir / "figure_1_raw_vs_native_by_hop.png", dpi=220); plt.close(fig)

    native = metrics.loc[metrics.protocol.eq("Native-Sum")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1", "balanced_accuracy"], ["Accuracy", "Macro-F1", "Balanced accuracy"]):
        for model in ["qwen3_8b", "tulu_final"]:
            part = native.loc[native.model.eq(model)]
            ax.plot(part.hop, part[metric], marker="o", lw=2, label="Qwen3-8B" if model == "qwen3_8b" else "Tulu Final", color=COLORS["Qwen3-8B" if model == "qwen3_8b" else "Tulu Final"])
        ax.set(title=title, xlabel="Hop", ylabel=metric, xticks=HOPS, ylim=(0, 1)); ax.axhline(.125, color="black", ls=":", lw=1)
    axes[-1].legend(); fig.suptitle("Figure 2. Qwen3-8B vs Tulu Final Native-Sum", y=1.02); fig.tight_layout(); fig.savefig(figure_dir / "figure_2_native_metrics_qwen_vs_tulu.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    overall = frequency.loc[(frequency.scope.eq("all")) & frequency.protocol.isin(["Raw-Sum", "Raw-Mean", "Native-Sum", "Native-Mean"])]
    melted = overall.melt(id_vars=["protocol"], value_vars=["predicted_cardinal_frequency", "predicted_diagonal_frequency"], var_name="type", value_name="frequency")
    melted["type"] = melted.type.map({"predicted_cardinal_frequency": "Cardinal", "predicted_diagonal_frequency": "Diagonal"})
    fig, ax = plt.subplots(figsize=(10, 5.2)); sns.barplot(data=melted, x="protocol", y="frequency", hue="type", ax=ax, errorbar=None)
    ax.set(title="Figure 3. Qwen cardinal/diagonal prediction frequency", xlabel="Protocol", ylabel="Prediction frequency", ylim=(0, 1)); fig.tight_layout(); fig.savefig(figure_dir / "figure_3_cardinal_diagonal_frequency.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, protocol in zip(axes, ["Raw-Sum", "Native-Sum", "EqualLength-Native"]):
        frame = predictions[protocol]
        matrix = pd.crosstab(frame.canonical_relation, frame.prediction, normalize="index").reindex(index=RELATIONS, columns=RELATIONS, fill_value=0)
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=True)
        ax.set(title=protocol, xlabel="Predicted", ylabel="Gold")
    fig.suptitle("Figure 4. Qwen3-8B row-normalized confusion matrices", y=1.02); fig.tight_layout(); fig.savefig(figure_dir / "figure_4_qwen_confusion_matrices.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    qwen = derived.loc[derived.model.eq("qwen3_8b"), ["protocol", "accuracy_h2_h5"]].assign(Model="Qwen3-8B")
    tulu_core_path = B21 / "statistics" / "core_summary.csv"
    tulu_core = pd.read_csv(tulu_core_path)
    tulu = tulu_core.loc[tulu_core.model.eq("final"), ["Protocol", "Derived mean"]].rename(columns={"Protocol": "protocol", "Derived mean": "accuracy_h2_h5"}).assign(Model="Tulu Final")
    plot = pd.concat([qwen, tulu], ignore_index=True)
    plot = plot.loc[plot.protocol.isin(PROTOCOLS)]
    fig, ax = plt.subplots(figsize=(12, 5.5)); sns.barplot(data=plot, x="protocol", y="accuracy_h2_h5", hue="Model", order=PROTOCOLS, ax=ax, errorbar=None)
    ax.axhline(.125, color="black", ls="--", lw=1); ax.set(title="Figure 5. h2-h5 equal-weight derived accuracy", xlabel="Protocol", ylabel="Derived accuracy", ylim=(0, 1)); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(figure_dir / "figure_5_derived_protocol_comparison.png", dpi=220); plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def write_report(metrics: pd.DataFrame, derived: pd.DataFrame, paired: pd.DataFrame, frequency: pd.DataFrame, decision: dict) -> None:
    audit = read_json(OUT / "audit" / "audit_summary.json")
    model = read_json(OUT / "audit" / "model_manifest.json")
    environment = read_json(OUT / "audit" / "environment.json")
    labels = read_json(OUT / "audit" / "equal_length_gate.json")
    qwen = metrics.loc[metrics.model.eq("qwen3_8b")]
    tulu = metrics.loc[metrics.model.eq("tulu_final")]
    native_compare = paired.loc[paired.protocol.eq("Native-Sum")]
    report = f"""# Stage B2.2 Qwen3-8B Model Selection Gate

## 1. Research Question

Qwen3-8B 是否在冻结 StepGame task-fitness protocol 下表现出足够强且稳定的 multi-hop spatial reasoning behavior，从而适合作为后续 mechanistic comparison 的 strong-reasoning target？

## 2. Frozen Reference

- Primary frozen test: 2063 rows; h1–h5 = 712/505/422/282/142.
- Row-ID SHA-256: `{audit['row_ids_sha256']}`.
- Frozen source: `{read_json(OUT/'audit'/'frozen_data_audit.json')['frozen_manifest']['path']}`.
- Tülu Final Raw-Sum h2–h5 derived = 0.1112; Native-Sum = 0.1829.
- Frozen-data audit: **{read_json(OUT/'audit'/'frozen_data_audit.json')['status']}**.

## 3. Qwen Model Audit

- Exact model: `{model['canonical_model_name']}`; local path `{model['local_path']}`; revision/snapshot `{model['revision_or_snapshot_id']}`.
- Architecture: `{model['architectures']}`; layers={model['num_hidden_layers']}; hidden size={model['hidden_size']}; vocab={model['config_vocab_size']}.
- Tokenizer BOS/EOS/PAD IDs: {model['bos_token_id']}/{model['eos_token_id']}/{model['pad_token_id']}; model-config/raw explicit BOS ID: {model['config_bos_token_id']}/{model['raw_explicit_bos_token_id']}.
- Config SHA-256: `{model['config_sha256']}`; chat-template SHA-256: `{model['chat_template_sha256']}`.
- Main dtype: BF16; transformers={environment['transformers']}; torch={environment['torch']}; CUDA={environment['torch_cuda']}.
- Native protocol uses the tokenizer's own template with `add_generation_prompt=True, enable_thinking=False`. If that exact template renders an empty `<think>\n\n</think>` scaffold, it is retained; non-empty thinking content is a hard failure.

## 4. Tokenizer and Boundary Audit

Natural verbalizers are the frozen B2.1 strings `{json.dumps(NATURAL_CANDIDATES, ensure_ascii=False)}`. Raw and Native each received 120 deterministic contextual-boundary checks. Scoring uses the audited contextual continuation token IDs even when standalone tokenization differs. Equal-length mapping `{json.dumps(labels['relation_to_label'])}` passed exactly-one-token continuation checks at Raw and Native boundaries; it remains a measurement-control/sensitivity diagnostic.

## 5. Experiment A — Raw-Sum

{markdown_table(qwen.loc[qwen.protocol.eq('Raw-Sum')], ['hop','n_test','accuracy','macro_f1','balanced_accuracy','majority_accuracy','uniform_chance'])}

![Figure 1](figures/figure_1_raw_vs_native_by_hop.png)

## 6. Experiment B — Native-Sum

{markdown_table(qwen.loc[qwen.protocol.eq('Native-Sum')], ['hop','n_test','accuracy','macro_f1','balanced_accuracy','majority_accuracy','uniform_chance'])}

Native is an own-native-interface task-fitness protocol. Qwen Native and Tülu Native do not have identical token sequences.

## 7. Experiment C — Scoring Sensitivity

{markdown_table(derived.loc[(derived.model.eq('qwen3_8b')) & derived.protocol.isin(['Raw-Sum','Raw-Mean','Native-Sum','Native-Mean'])], ['protocol','accuracy_h2_h5','macro_f1_h2_h5','balanced_accuracy_h2_h5'])}

![Figure 3](figures/figure_3_cardinal_diagonal_frequency.png)

Mean-token is a sensitivity analysis, not an unbiased or mathematically privileged correction.

## 8. Experiment D — Equal-Length Control

{markdown_table(derived.loc[(derived.model.eq('qwen3_8b')) & derived.protocol.str.startswith('EqualLength')], ['protocol','accuracy_h2_h5','macro_f1_h2_h5','balanced_accuracy_h2_h5'])}

Equal-length changes the output mapping and adds an explicit mapping instruction; it is not a gold-standard replacement for natural verbalizers.

## 9. Qwen vs Tülu Final

{markdown_table(pd.concat([qwen.loc[qwen.protocol.eq('Native-Sum')], tulu.loc[tulu.protocol.eq('Native-Sum')]]), ['model','hop','accuracy','macro_f1','balanced_accuracy'])}

![Figure 2](figures/figure_2_native_metrics_qwen_vs_tulu.png)

## 10. Statistical Comparisons

Paired bootstrap resamples `story_group_id` 2,000 times with seed 20260910. Derived differences first compute h2/h3/h4/h5 separately and then take an equal-weight mean.

{markdown_table(native_compare, ['scope','metric','point_difference','ci95_low','ci95_high','ci_excludes_zero'])}

## 11. Fitness Gate

- Data integrity: **{decision['data_integrity']}**
- Tokenizer boundary audit: **{decision['tokenizer_boundary_audit']}**
- Native behavior stronger than Tülu: **{decision['native_behavior_stronger_than_tulu']}**
- Above uniform chance at each h2–h5: **{decision['above_uniform_chance']}**
- Above hop-specific majority baseline at each h2–h5: **{decision['above_majority_baseline']}**
- Macro-F1 improved: **{decision['macro_f1_improved']}**
- Balanced accuracy improved: **{decision['balanced_accuracy_improved']}**
- Prediction collapse: **{decision['prediction_collapse']['status']}**
- Protocol sensitivity: **{decision['protocol_sensitivity']['status']}**
- Strong-reasoning target: **{decision['strong_reasoning_target']}**

![Figure 4](figures/figure_4_qwen_confusion_matrices.png)

## 12. Consequence for Stage B

{decision['recommended_next_step']}

![Figure 5](figures/figure_5_derived_protocol_comparison.png)

## 13. Interpretation Boundaries

- Behavior ≠ hidden mechanism.
- Chat improvement ≠ representation improvement.
- Equal-length ≠ pure length correction.
- Above chance ≠ successful reasoning.
- External literature accuracy ≠ accuracy under this frozen protocol.
- This stage does not establish graphs, symbolic reasoning, representational structure, or post-training causal mechanisms.
"""
    (OUT / "STAGE_B2_2_QWEN3_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    full_meta = read_json(OUT / "audit" / "run_full_metadata.json")
    if full_meta.get("status") != "PASS" or not all(name in full_meta.get("components", {}) for name in ["raw_natural", "native_natural", "equal_raw", "equal_native"]):
        raise RuntimeError("All four full-run components must PASS before finalization")
    primary = load_primary()
    qwen_predictions, _ = load_qwen_predictions(primary)
    tulu_predictions = load_tulu_predictions(primary)

    metric_frames = [metrics_by_hop(frame, "qwen3_8b", protocol) for protocol, frame in qwen_predictions.items()]
    metric_frames += [metrics_by_hop(frame, "tulu_final", protocol) for protocol, frame in tulu_predictions.items()]
    metrics = pd.concat(metric_frames, ignore_index=True)
    tulu_reference = verify_tulu_reference(metrics)
    write_json_atomic(OUT / "statistics" / "tulu_reference_reproduction.json", tulu_reference)
    if tulu_reference["status"] != "PASS":
        raise RuntimeError("Tulu Final row-level reconstruction does not reproduce the frozen B2.1 reference")
    write_csv_atomic(OUT / "statistics" / "all_hop_metrics.csv", metrics)
    write_csv_atomic(OUT / "raw_sum" / "metrics.csv", metrics.loc[(metrics.model.eq("qwen3_8b")) & (metrics.protocol.eq("Raw-Sum"))])
    write_csv_atomic(OUT / "native_sum" / "metrics.csv", metrics.loc[(metrics.model.eq("qwen3_8b")) & (metrics.protocol.eq("Native-Sum"))])

    bootstrap_frames = []
    for protocol, frame in qwen_predictions.items():
        bootstrap_frames.append(bootstrap_one(frame, "qwen3_8b", protocol)[0])
    for protocol, frame in tulu_predictions.items():
        bootstrap_frames.append(bootstrap_one(frame, "tulu_final", protocol)[0])
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "bootstrap_ci.csv", bootstrap)

    paired = pd.concat([
        paired_bootstrap(qwen_predictions[protocol], tulu_predictions[protocol], protocol)
        for protocol in ["Raw-Sum", "Native-Sum"]
    ], ignore_index=True)
    write_csv_atomic(OUT / "statistics" / "paired_bootstrap.csv", paired)
    derived = derived_summary(metrics)
    write_csv_atomic(OUT / "statistics" / "derived_summary.csv", derived)
    frequency = prediction_frequencies(qwen_predictions)
    write_csv_atomic(OUT / "scoring_sensitivity" / "cardinal_diagonal_metrics.csv", frequency)
    save_confusions(qwen_predictions)
    decision = make_decision(metrics, derived, paired, frequency)
    write_json_atomic(OUT / "fitness_decision.json", decision)
    draw_figures(metrics, derived, qwen_predictions, tulu_predictions, frequency)
    write_report(metrics, derived, paired, frequency, decision)
    summary = {
        "status": "PASS", "metrics_sha256": sha256_text((OUT / "statistics" / "all_hop_metrics.csv").read_text(encoding="utf-8")),
        "paired_bootstrap_sha256": sha256_text((OUT / "statistics" / "paired_bootstrap.csv").read_text(encoding="utf-8")),
        "figures": sorted(path.name for path in (OUT / "figures").glob("figure_*.png")),
        "strong_reasoning_target": decision["strong_reasoning_target"],
    }
    write_json_atomic(OUT / "statistics" / "finalization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
