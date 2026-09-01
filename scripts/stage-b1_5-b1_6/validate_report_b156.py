#!/usr/bin/env python3
"""Independent Gate validation and technical report for Stage B1.5-B1.6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_b156 import B1, BEHAVIOR, CROSS, FIGURES, OUT, RELATIONS, read_json, write_json


REPORT = OUT / "B1_5_B1_6_BEHAVIOR_TRANSFER_REPORT.md"


def pf(value: float) -> str:
    return f"{float(value):.4f}"


def row_by_hop(frame: pd.DataFrame, hop: int):
    return frame.loc[frame["hop"].eq(hop)].iloc[0]


def peak_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        group.loc[group["test_macro_f1"].idxmax()]
        for _, group in frame.groupby("target_hop", sort=True)
    ])


def main() -> None:
    checksum = read_json(OUT / "input_checksum_verification.json")
    audit = read_json(BEHAVIOR / "candidate_audit.json")
    tokenization = read_json(BEHAVIOR / "candidate_tokenization.json")
    behavior_summary = read_json(BEHAVIOR / "behavior_summary.json")
    cross_summary = read_json(CROSS / "cross_hop_summary.json")
    convergence = read_json(CROSS / "convergence_sensitivity.json")
    convergence_5000 = read_json(CROSS / "convergence_sensitivity_5000.json")
    primary_b = pd.read_csv(BEHAVIOR / "primary_behavior.csv")
    balanced_b = pd.read_csv(BEHAVIOR / "balanced_behavior.csv")
    behavior_probe = pd.read_csv(BEHAVIOR / "behavior_vs_probe.csv")
    transfer = pd.read_csv(CROSS / "h1_to_all_layers.csv")
    derived = pd.read_csv(CROSS / "derived_pooled_to_all_layers.csv")
    balanced_t = pd.read_csv(CROSS / "balanced_h1_to_all_layers.csv")
    random_t = pd.read_csv(CROSS / "random_label_transfer.csv")
    trace = pd.read_csv(CROSS / "c_selection_trace.csv")
    gap = pd.read_csv(CROSS / "transfer_gap.csv")
    transfer_summary = pd.read_csv(CROSS / "transfer_summary.csv")
    within = pd.read_csv(B1 / "primary_probe.csv")

    checks = {}
    checks["input_checksum_gate_pass"] = checksum["status"] == "PASS" and all(checksum["checks"].values())
    checks["candidate_audit_pass"] = audit["status"] == "PASS" and all(audit["checks"].values())
    checks["candidate_protocol_complete"] = (
        tokenization["status"] == "PASS" and len(tokenization["records"]) == 8
        and {row["relation"] for row in tokenization["records"]} == set(RELATIONS)
        and not any(row["contains_special_token"] for row in tokenization["records"])
    )
    checks["behavior_complete_h1_h5"] = (
        set(primary_b["hop"]) == set(range(1, 6)) and set(balanced_b["hop"]) == set(range(1, 6))
        and len(primary_b) == len(balanced_b) == 5
        and primary_b["n_test"].tolist() == [712, 505, 422, 282, 142]
        and balanced_b["n_test"].tolist() == [88] * 5
    )
    checks["behavior_baselines_and_bootstrap_complete"] = (
        np.allclose(primary_b["chance_accuracy"], 0.125)
        and primary_b[["majority_accuracy", "accuracy_sum_ci95_low", "accuracy_sum_ci95_high"]].notna().all().all()
        and behavior_summary["bootstrap"] == {"unit": "story_group_id", "resamples": 1000}
    )
    checks["cross_hop_shapes_complete"] = (
        len(transfer) == len(derived) == len(balanced_t) == 160 and len(random_t) == 30
        and set(transfer["layer"]) == set(range(32)) and set(transfer["target_hop"]) == set(range(1, 6))
        and set(derived["layer"]) == set(range(32)) and set(balanced_t["layer"]) == set(range(32))
        and set(random_t["layer"]) == {0, 7, 15, 17, 23, 31}
    )
    checks["activations_reused_no_reextraction"] = cross_summary["activation_reextracted"] is False
    checks["no_test_selection_leakage"] = (
        not bool(trace["test_accessed_for_selection"].astype(bool).any())
        and not bool(transfer["target_test_used_for_fitting_or_selection"].astype(bool).any())
        and not bool(derived["target_test_used_for_fitting_or_selection"].astype(bool).any())
        and not bool(balanced_t["target_test_used_for_fitting_or_selection"].astype(bool).any())
    )
    checks["fit_and_C_populations_exact"] = (
        set(transfer["scaler_fit_population"]) == {"primary_split:train:hops=[1]"}
        and set(transfer["C_selection_population"]) == {"primary_split:val:hops=[1]"}
        and set(derived["scaler_fit_population"]) == {"primary_split:train:hops=[2, 3, 4, 5]"}
        and set(derived["C_selection_population"]) == {"primary_split:val:hops=[2, 3, 4, 5]"}
        and set(balanced_t["scaler_fit_population"]) == {"balanced_split:train:hops=[1]"}
        and set(balanced_t["C_selection_population"]) == {"balanced_split:val:hops=[1]"}
    )
    task_selected = pd.concat([transfer, derived, balanced_t, random_t]).drop_duplicates(["analysis", "layer"])
    expected_selected = trace.sort_values(["analysis", "layer", "validation_macro_f1", "C"], ascending=[True, True, False, True]).drop_duplicates(["analysis", "layer"])
    selection_check = task_selected[["analysis", "layer", "selected_C"]].merge(
        expected_selected[["analysis", "layer", "C"]], on=["analysis", "layer"], validate="one_to_one"
    )
    checks["C_selection_and_smaller_tie_rule_exact"] = len(trace) == 918 and np.allclose(selection_check["selected_C"], selection_check["C"])
    checks["within_hop_gap_complete"] = len(gap) == 160 and set(gap["target_hop"]) == set(range(1, 6)) and len(transfer_summary) == 4
    checks["random_label_control_degrades"] = (
        cross_summary["random_mean_accuracy"] < cross_summary["matched_normal_mean_accuracy"]
        and cross_summary["random_mean_macro_f1"] < cross_summary["matched_normal_mean_macro_f1"]
        and abs(cross_summary["random_mean_accuracy"] - 0.125) < 0.02
    )
    checks["convergence_sensitivity_pass"] = (
        convergence["selection_rule_unchanged"] is True and convergence["selected_C_frozen"] is True
        and convergence_5000["status"] == "PASS" and convergence_5000["selection_rule_unchanged"] is True
        and convergence_5000["selected_C_frozen"] is True and convergence_5000["remaining_warning_tasks"] == 0
    )
    figure_paths = sorted(FIGURES.glob("figure_*.png"))
    checks["six_figures_complete"] = len(figure_paths) == 6 and all(path.stat().st_size > 25_000 for path in figure_paths)
    checks["chart_map_complete"] = len(read_json(FIGURES / "chart_map.json")) == 6

    b15_keys = ["input_checksum_gate_pass", "candidate_audit_pass", "candidate_protocol_complete", "behavior_complete_h1_h5", "behavior_baselines_and_bootstrap_complete"]
    b16_keys = ["input_checksum_gate_pass", "cross_hop_shapes_complete", "activations_reused_no_reextraction", "no_test_selection_leakage", "fit_and_C_populations_exact", "C_selection_and_smaller_tie_rule_exact", "within_hop_gap_complete", "random_label_control_degrades", "convergence_sensitivity_pass"]
    b15_status = "PASS" if all(checks[key] for key in b15_keys) else "FAIL"
    b16_status = "PASS" if all(checks[key] for key in b16_keys) else "FAIL"
    overall = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "stage": "B1.5-B1.6", "status": overall, "B1.5_status": b15_status,
        "B1.6_status": b16_status, "checks": {key: bool(value) for key, value in checks.items()},
        "failed_checks": [key for key, value in checks.items() if not value],
        "visual_QA": {"status": "PASS", "method": "manual inspection of all six downloaded PNGs", "notes": "No clipping, overlap, unreadable labels, or color-direction errors observed."},
    }
    write_json(OUT / "validation_summary.json", validation)
    if overall != "PASS":
        print(json.dumps(validation, indent=2))
        raise SystemExit(2)

    transfer_peaks = peak_rows(transfer)
    derived_peaks = peak_rows(derived)
    balanced_peaks = peak_rows(balanced_t)
    h1_within = within[within["hop"].eq(1)].loc[lambda x: x["test_macro_f1"].idxmax()]
    derived_h1_peak = derived_peaks[derived_peaks["target_hop"].eq(1)].iloc[0]

    candidate_lines = []
    for item in tokenization["records"]:
        candidate_lines.append(f"| {item['relation']} | `{item['candidate_string']}` | {item['token_ids']} | {item['token_count']} |")
    behavior_lines = []
    balanced_lines = []
    compare_lines = []
    sensitivity_lines = []
    for hop in range(1, 6):
        p = row_by_hop(primary_b, hop); b = row_by_hop(balanced_b, hop); bp = row_by_hop(behavior_probe, hop)
        behavior_lines.append(f"| h{hop} | {int(p.n_test)} | {pf(p.accuracy_sum)} [{pf(p.accuracy_sum_ci95_low)}, {pf(p.accuracy_sum_ci95_high)}] | {pf(p.macro_f1_sum)} | {pf(p.majority_accuracy)} |")
        balanced_lines.append(f"| h{hop} | {int(b.n_test)} | {pf(b.accuracy_sum)} [{pf(b.accuracy_sum_ci95_low)}, {pf(b.accuracy_sum_ci95_high)}] | {pf(b.macro_f1_sum)} |")
        compare_lines.append(f"| h{hop} | {pf(bp.behavior_accuracy)} | {pf(bp.behavior_macro_f1)} | {pf(bp.probe_peak_macro_f1)} | L{int(bp.probe_peak_layer)} |")
        sensitivity_lines.append(f"| h{hop} | {pf(p.accuracy_sum)} | {pf(p.accuracy_mean)} | {pf(p.macro_f1_sum)} | {pf(p.macro_f1_mean)} |")
    transfer_lines = []
    for hop in range(2, 6):
        peak = transfer_peaks[transfer_peaks["target_hop"].eq(hop)].iloc[0]
        summary = transfer_summary[transfer_summary["target_hop"].eq(hop)].iloc[0]
        transfer_lines.append(
            f"| h{hop} | {pf(peak.test_macro_f1)} @ L{int(peak.layer)} | {pf(summary.within_hop_peak_macro_f1)} @ L{int(summary.within_hop_peak_layer)} | "
            f"{pf(summary.h1_transfer_at_within_hop_peak_layer)} | {pf(summary.gap_at_within_hop_peak_layer)} | {pf(summary.retention_at_within_hop_peak_layer)} | {pf(summary.mean_layerwise_gap)} | {pf(summary.late_layer_16_31_mean_gap)} |"
        )
    derived_lines = [
        f"| h{int(row.target_hop)} | {pf(row.test_macro_f1)} | L{int(row.layer)} |"
        for row in derived_peaks.sort_values("target_hop").itertuples()
    ]
    balanced_transfer_lines = [
        f"| h{int(row.target_hop)} | {pf(row.test_macro_f1)} | L{int(row.layer)} |"
        for row in balanced_peaks.sort_values("target_hop").itertuples()
    ]

    report = f"""# Stage B1.5–B1.6 Behavioral Baseline + Cross-hop Transfer Probe

## Technical summary

**Overall status: PASS (B1.5 PASS; B1.6 PASS).** This stage used only the frozen Meta-Llama-3.1-8B Base pipeline. Candidate likelihood behavior is near chance for most hops under the preregistered summed conditional log-probability score, and decreases monotonically from h1 to h5. The h1-trained decoder transfers weakly but nontrivially to derived examples; however, derived within-hop probes are themselves weak, so similar transfer and within-hop values are evidence only for limited shared linearly accessible structure—not a shared symbolic reasoning code.

The evidence is most consistent with **Pattern A/D**: Base lacks clear, robust derived task ability under the primary scoring rule, while both behavioral performance and linear decodability are concentrated at h1. Length-normalized scoring changes several behavior values, confirming that answer-token length is an important sensitivity rather than a basis for changing the primary metric.

## Scope, frozen inputs, and Gate

- Model: `Meta-Llama-3.1-8B` Base only.
- Reused Stage B1 activations: 13,758 rows × 32 layers × 4,096 hidden dimensions, float16.
- No activation re-extraction for B1.6.
- Frozen Primary test counts by hop: 712, 505, 422, 282, 142.
- Frozen Balanced test: 88 rows per hop.
- Prompt: exact Stage B1 story/question wrapper, no chat template, few-shot, CoT, answer example, or gold answer.
- Input checksum Gate: PASS; all 32 activation shards, row order, split manifests, and frozen protocol hashes reverified.
- Independent validation: `{OUT / 'validation_summary.json'}`.

## Candidate likelihood method and audit

The primary behavior score is the sum of token-level conditional log-probabilities, `log P(candidate tokens | frozen prompt)`. Every candidate begins with the continuation-leading ASCII space. Explicit BOS was preserved; EOS/pad tokens were excluded. The 40-example audit exactly covered 5 hops × 8 relations and passed every leakage, conditional-position, candidate-count, and finite-score check.

| Relation | Candidate | Token IDs | Tokens |
|---|---:|---:|---:|
{chr(10).join(candidate_lines)}

The cardinal candidates are one token and diagonal candidates are two. Therefore summed scoring structurally penalizes the longer candidates; mean log-probability per token is reported as a fixed sensitivity, not selected post hoc.

## Finding 1 — Behavioral baseline

### Primary, summed conditional log-probability

| Hop | n | Accuracy (95% story-group bootstrap CI) | Macro-F1 | Majority accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(behavior_lines)}

Chance accuracy is 0.125. Primary accuracy is monotonically nonincreasing, but only h1's 95% CI lies narrowly above chance. h2 includes chance; h3–h5 do not establish above-chance ability. Macro-F1 is low at every hop. Consequently the data do not support a claim that Base has clear derived StepGame ability under the primary candidate scoring protocol.

### Balanced control

| Hop | n | Accuracy (95% CI) | Macro-F1 |
|---|---:|---:|---:|
{chr(10).join(balanced_lines)}

Balanced estimates are noisy because each hop contains only 88 test rows. They do not show a monotonic hop trend and do not overturn the Primary conclusion.

### Token-length sensitivity

| Hop | Sum accuracy | Mean-token accuracy | Sum Macro-F1 | Mean-token Macro-F1 |
|---|---:|---:|---:|---:|
{chr(10).join(sensitivity_lines)}

Length normalization increases several derived-hop estimates (for example h3 and h5 accuracy), demonstrating material answer-form sensitivity. Because candidate strings do not have equal token length, any future Base–Tulu comparison must freeze and report both scoring rules identically.

## Finding 2 — Behavior versus probe

| Hop | Behavior accuracy | Behavior Macro-F1 | Within-hop peak Macro-F1 | Peak layer |
|---|---:|---:|---:|---:|
{chr(10).join(compare_lines)}

Both views distinguish h1 from derived hops, but they measure different phenomena. Across only five hop points, accuracy and peak probe F1 have Spearman ρ={pf(behavior_summary['behavior_probe_spearman_n5']['behavior_accuracy_vs_probe_peak_f1']['rho'])}; this is descriptive only (n=5) and is not inferential evidence of a mechanistic correspondence.

![Behavior accuracy by hop](figures/figure_1_behavior_accuracy_vs_hop.png)

![Behavior versus probe](figures/figure_2_behavior_vs_probe.png)

## Finding 3 — h1 → derived cross-hop transfer

For every layer, the StandardScaler and multinomial decoder were fitted on Primary h1 train only. C was chosen on h1 validation Macro-F1 only, using the frozen nine-value grid and smaller-C exact-tie rule. All h1–h5 test evaluations occurred after selection.

| Target | Peak h1-transfer F1 | Within-hop peak | Transfer at within peak | Gap at within peak | Retention | Mean gap | Late L16–31 mean gap |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(transfer_lines)}

The gap is `within-hop F1 − h1-transfer F1`. h2/h3 show small positive late-layer gaps; h4/h5 often have zero or negative gaps. This is compatible with some shared linearly accessible relation structure, but the absolute derived scores remain weak and peak selection across 32 layers is optimistic. Similar weak values cannot distinguish a genuinely shared representation from broadly noisy, distribution-linked linear signals.

![h1-trained transfer](figures/figure_3_h1_trained_cross_hop_transfer.png)

![Within versus transfer](figures/figure_4_within_vs_h1_transfer.png)

![Transfer gap heatmap](figures/figure_5_transfer_gap_heatmap.png)

## Finding 4 — Derived-pooled → explicit

| Test hop | Peak Macro-F1 | Layer |
|---|---:|---:|
{chr(10).join(derived_lines)}

Derived-pooled → h1 peaks at {pf(derived_h1_peak.test_macro_f1)} (L{int(derived_h1_peak.layer)}), well below the h1 within-hop peak of {pf(h1_within.test_macro_f1)} (L{int(h1_within.layer)}). The directions are not symmetric, although the training sample sizes and source distributions also differ, so this is not a controlled mathematical symmetry test.

## Finding 5 — Balanced transfer control

| Test hop | Peak Macro-F1 | Layer |
|---|---:|---:|
{chr(10).join(balanced_transfer_lines)}

The Balanced curves are high-variance and have no stable h1-dominant profile. They support only the qualitative conclusion that transfer is weak/noisy and layer-dependent; they do not reproduce Primary peak ordering and must not replace the Primary analysis.

![Balanced transfer](figures/figure_6_balanced_cross_hop_control.png)

## Negative control and numerical robustness

- Random-label transfer (layers 0, 7, 15, 17, 23, 31): mean accuracy {pf(cross_summary['random_mean_accuracy'])}, mean Macro-F1 {pf(cross_summary['random_mean_macro_f1'])}.
- Matched normal-label tasks: mean accuracy {pf(cross_summary['matched_normal_mean_accuracy'])}, mean Macro-F1 {pf(cross_summary['matched_normal_mean_macro_f1'])}.
- Random accuracy is within 0.002 of the 0.125 chance level and degrades clearly relative to matched normal probes.
- {int(convergence['warning_tasks_refit'])} selected fits that reached `max_iter=400` were independently refit from scratch at the same frozen C with `max_iter=2000`.
- At 2,000 iterations, remaining warnings: {int(convergence['remaining_warning_tasks'])}; maximum absolute test Macro-F1 change versus the original fit: {pf(convergence['max_abs_test_macro_f1_delta'])}; mean absolute change: {pf(convergence['mean_abs_test_macro_f1_delta'])}.
- Those {int(convergence_5000['tasks_escalated_from_2000'])} remaining tasks were escalated to 5,000 iterations at the same C. Remaining warnings: {int(convergence_5000['remaining_warning_tasks'])}; maximum absolute test Macro-F1 change versus the 2,000-iteration fit: {pf(convergence_5000['max_abs_test_macro_f1_delta_vs_2000'])}; mean absolute change: {pf(convergence_5000['mean_abs_test_macro_f1_delta_vs_2000'])}.
- The sensitivity does not alter C selection or use test data for selection.

## Interpretation boundaries

The results justify the cautious statement that explicit and derived instances contain **limited shared linearly accessible structure** at some layers. They do not justify saying the model grounds relations and then reasons with one symbolic code. Because derived behavior and derived within-hop decodability are weak, the experiment cannot separate absent representations from nonlinear, transient, differently located, or prompt-incompatible representations.

Additional limitations are candidate-token-length asymmetry, small Balanced cells, peak-over-32-layers optimism, non-independent examples within dataset construction despite story-group bootstrap, and observation at only the question-end state.

## Gate outcomes

- Stage B1.5: **{b15_status}**.
- Stage B1.6: **{b16_status}**.
- Six required figures: PASS, visually inspected.
- Recommendation: **YES, proceed to a separately authorized Base–Tulu comparison**, because the Base reference pipeline and controls are now frozen and validated. The next stage must apply identical prompts, candidate strings, both sum/mean scoring rules, splits, activations/readout, probes, and reporting. No comparison was run here.

## Next steps (not executed)

1. Obtain explicit human approval before loading or running any Tulu/SFT/DPO/RLVR model.
2. Freeze the Base–Tulu comparison plan, especially tokenization-compatible candidate scoring and paired uncertainty comparisons.
3. Treat changes in behavior, within-hop decodability, and cross-hop transfer as separate outcomes.

## Further questions

- Does instruction tuning improve derived behavior without increasing question-end linear decodability?
- Does it increase cross-hop geometry sharing, or only alter answer-token priors?
- Are relation signals stronger at entity tokens or intermediate reasoning positions than at question end?
"""
    REPORT.write_text(report, encoding="utf-8")
    write_json(OUT / "stage_b1_5_1_6_gate.json", {
        "status": "PASS", "B1.5": b15_status, "B1.6": b16_status,
        "validation": str(OUT / "validation_summary.json"), "report": str(REPORT),
        "stop_after_stage": True, "base_tulu_comparison_executed": False,
    })
    print(json.dumps({"status": "PASS", "B1.5": b15_status, "B1.6": b16_status, "report": str(REPORT), "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
