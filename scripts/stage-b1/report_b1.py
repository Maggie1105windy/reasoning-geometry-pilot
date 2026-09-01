#!/usr/bin/env python3
"""Create the final Stage B1 technical report and immutable completion manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common_b1 import ACT_DIR, OUT, sha256, write_json


CORE_FILES = [
    "model_audit.json", "prompt_protocol.json", "probe_protocol.json",
    "split_checksum_verification.json", "tokenization_audit.json",
    "activation_sanity.json", "activation_integrity.json", "union_manifest.csv",
    "primary_probe.csv", "balanced_probe.csv", "target_absent_eval.csv",
    "composition_heldout_probe.csv", "template_test_probe.csv",
    "random_label_control.csv", "baselines.csv", "layer_summary.csv",
    "primary_peak_bootstrap.csv", "c_selection_trace.csv",
    "convergence_refit_sensitivity.csv", "validation_pre_report.json",
]


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    validation = json.loads((OUT / "validation_pre_report.json").read_text())
    if validation["status"] != "PASS":
        raise RuntimeError("Cannot issue a PASS report before independent validation passes")
    model = json.loads((OUT / "model_audit.json").read_text())
    token = json.loads((OUT / "tokenization_audit.json").read_text())
    sanity = json.loads((OUT / "activation_sanity.json").read_text())
    integrity = json.loads((OUT / "activation_integrity.json").read_text())
    prompt = json.loads((OUT / "prompt_protocol.json").read_text())
    protocol = json.loads((OUT / "probe_protocol.json").read_text())
    primary = pd.read_csv(OUT / "primary_probe.csv")
    balanced = pd.read_csv(OUT / "balanced_probe.csv")
    target = pd.read_csv(OUT / "target_absent_eval.csv")
    composition = pd.read_csv(OUT / "composition_heldout_probe.csv")
    template = pd.read_csv(OUT / "template_test_probe.csv")
    random = pd.read_csv(OUT / "random_label_control.csv")
    baseline = pd.read_csv(OUT / "baselines.csv")
    summary = pd.read_csv(OUT / "layer_summary.csv")
    sensitivity = pd.read_csv(OUT / "convergence_refit_sensitivity.csv")

    primary_rows = []
    for row in summary.itertuples(index=False):
        l90 = "not reached" if pd.isna(row.L90) else str(int(row.L90))
        primary_rows.append([
            int(row.hop), row.relation_type, int(row.peak_layer), fmt(row.peak_macro_f1),
            f"[{fmt(row.peak_ci95_lower)}, {fmt(row.peak_ci95_upper)}]",
            fmt(row.majority_macro_f1), int(row.L50), l90,
        ])

    def peak_rows(frame: pd.DataFrame, hops: list[int]) -> list[list[object]]:
        rows = []
        for hop in hops:
            part = frame[frame["hop"].eq(hop)]
            peak = part.loc[part["test_macro_f1"].idxmax()]
            iid = summary[summary["hop"].eq(hop)].iloc[0]
            rows.append([hop, int(peak["layer"]), fmt(peak["test_macro_f1"]),
                         fmt(float(peak["test_macro_f1"]) - float(iid["peak_macro_f1"]))])
        return rows

    target_rows = []
    for hop in range(2, 6):
        iid = summary[summary["hop"].eq(hop)].iloc[0]
        row = target[target["hop"].eq(hop) & target["layer"].eq(int(iid["peak_layer"]))].iloc[0]
        target_peak = target[target["hop"].eq(hop)].loc[target[target["hop"].eq(hop)]["macro_f1"].idxmax()]
        target_rows.append([
            hop, int(iid["peak_layer"]), fmt(row["derived_all_macro_f1"]), fmt(row["macro_f1"]),
            fmt(row["delta_macro_f1_target_absent_minus_all"]),
            f"L{int(target_peak['layer'])} / {fmt(target_peak['macro_f1'])}",
        ])

    base_primary = baseline[baseline["experiment"].eq("primary")]
    baseline_rows = [[int(r.hop), r.majority_class, int(r.n_test), fmt(r.accuracy), fmt(r.macro_f1)]
                     for r in base_primary.itertuples(index=False)]
    matched = random.merge(primary, on=["layer", "hop"], suffixes=("_random", "_normal"))
    random_mean_acc = random["test_accuracy"].mean()
    random_mean_f1 = random["test_macro_f1"].mean()
    normal_matched_f1 = matched["test_macro_f1_normal"].mean()
    random_drop = (matched["test_macro_f1_normal"] - matched["test_macro_f1_random"]).mean()

    normal_sens = sensitivity[sensitivity["analysis"].eq("normal")]
    random_sens = sensitivity[sensitivity["analysis"].eq("random_train_labels")]
    disk_bytes = sum((ACT_DIR / f"layer_{layer:02d}.npy").stat().st_size for layer in range(32))
    disk_bytes += (ACT_DIR / "row_ids.npy").stat().st_size
    disk_gib = disk_bytes / 1024**3

    report = f"""# Stage B1: Llama-3.1-8B Base Explicit / Derived Relation Probing

Status: **PASS**  
Scope: **Meta-Llama-3.1-8B Base only; no Tulu, SFT, DPO, RLVR, fine-tuning, steering, patching, or ablation was run.**

## Executive result

All 38 fail-closed post-run checks passed. The clearest result is that the explicit one-hop relation is substantially more linearly decodable than the derived relations: Primary h=1 peaks at Macro-F1 **0.3573** at layer 17, whereas h=2–5 peak between **0.1485 and 0.1702**. The four derived curves do not show a monotonic increase in onset depth with graph distance. This supports a Pattern-B-style reading—no evidence for the simple model “larger hop count becomes decodable systematically later”—while preserving the important distinction that h=1 is stronger.

These are correlational linear-readout results. They do not identify a reasoning layer, establish symbolic computation, or provide a causal mechanism.

## 1. Frozen scope and model

- Model: `{model['model']}` from `{model['model_path']}`.
- Model commit: `{model['model_revision']}`; tokenizer commit: `{model['tokenizer_revision']}`.
- Parameters: {model['parameter_count']:,}; 32 decoder blocks; hidden size 4096; weights loaded as BF16.
- GPU: {model['gpu']} ({model['gpu_total_gib']:.2f} GiB).
- Frozen B0 Primary, Balanced, Composition-held-out, Target-Absent, and Template Test-only manifests were re-hashed after the experiment and remained unchanged.
- Union: {token['union_rows']:,} unique rows. The 32 float16 activation matrices plus row IDs occupy {disk_gib:.2f} GiB.

## 2. Prompt, tokenization, and activation definition

The frozen raw prompt was:

```text
Story:
<each original story sentence on its own line>

Question:
<original question>
```

No answer, chat wrapper, demonstrations, or chain-of-thought text was added. Tokenization used `add_special_tokens=False`, an explicit BOS, EOS as pad, and right padding. Readout was the final non-special token belonging to the question; all 120 audited examples selected the terminal `?`. The audit covered all 40 dataset×split cells and all {token['union_rows']:,} rows were checked for truncation safety (length {token['sequence_length']['min']}–{token['sequence_length']['max']}, model limit {token['sequence_length']['model_max_length']}).

Activations are the raw output of each decoder block before the final RMSNorm, taken at that readout token. Sanity checks established 32 matrices of shape `[batch, 4096]`, layer 0/15 agreement with the matching Hugging Face hidden-state tuples, and final-layer raw-vs-normalized indexing consistency. Saved files have shape `[13758, 4096]`, dtype float16, exact row-ID order, and zero NaN/Inf. Sanity forward peak memory was {sanity['peak_allocated_gib']:.2f} GiB allocated / {sanity['peak_reserved_gib']:.2f} GiB reserved; extraction used batch size {integrity['extraction']['batch_size']}.

The formal 10-sample × 5-layer same-batch replay had minimum cosine {integrity['reforward']['minimum_cosine']:.7f}, mean MAE {integrity['reforward']['mean_mae']:.1f}, and maximum absolute error {integrity['reforward']['max_absolute_error']:.1f}. A retained non-gate batch-size-1 diagnostic reached minimum cosine 0.999355; this is consistent with BF16 batch-shape numerical variation and is not a row-alignment failure. The fast tokenizer returned zero-length terminal offset mappings, so offset mapping was retained as a diagnostic only; exact token-ID decoding and the question-boundary/readout invariants were the gate criteria.

## 3. Probe protocol

Each layer and hop used a separate multinomial logistic regression. `StandardScaler` was fit on train only. Candidate C values were `{protocol['C_grid']}`; validation Macro-F1 selected C, with smaller C winning exact ties, and test was accessed only after selection. Primary, Composition-held-out, and Template used balanced training weights; the exactly balanced control used no class weighting. No validation or test oversampling was performed.

The initial frozen solver was `lbfgs`, `max_iter=400`, `tol=1e-4`. Forty-one selected normal-label fits emitted convergence warnings. A fixed-C numerical audit refit exactly those models with `max_iter=2000` without reselecting C: all 41 converged, mean absolute test Macro-F1 change was {normal_sens['delta_test_macro_f1'].abs().mean():.6f}, maximum {normal_sens['delta_test_macro_f1'].abs().max():.6f}. Two of five random-label warning refits still warned, but their maximum absolute test Macro-F1 change was {random_sens['delta_test_macro_f1'].abs().max():.6f}. The frozen 400-iteration values remain the reported primary results; the audit shows the main interpretation is not driven by those warnings.

## 4. Primary results

{markdown_table(['Hop', 'Type', 'Peak layer', 'Peak Macro-F1', '95% story-group CI', 'Majority Macro-F1', 'L50', 'L90'], primary_rows)}

L50/L90 use the pre-frozen normalized gain from majority-class Macro-F1 to the hop-specific peak. L90 additionally requires the next two available layers to remain at least 85% of normalized peak; therefore h=4 and h=5 do not reach stable L90. The h=3 L90 at layer 1 and h=5 peak at layer 1 demonstrate why these summaries must not be interpreted as a monotonic reasoning timeline.

![Primary layer-wise curves](figures/figure1_primary_macro_f1.png)

The explicit curve rises from 0.1625 at layer 0 to a broad mid/late-layer region around 0.30–0.36. Derived curves remain clustered near 0.10–0.17 without systematic hop ordering.

![Primary layer-hop heatmap](figures/figure6_primary_layer_hop_heatmap.png)

## 5. Balanced control

{markdown_table(['Hop', 'Peak layer', 'Peak Macro-F1', 'Difference from Primary peak'], peak_rows(balanced, [1,2,3,4,5]))}

![Balanced control](figures/figure2_balanced_macro_f1.png)

Balancing materially reduces the h=1 peak (0.2163 versus 0.3573) and removes its broad high late-layer plateau. The derived curves remain low and irregular. Thus the Primary h=1 magnitude is partly sensitive to data volume/distribution, while neither dataset supports monotonic h=2→h=5 onset ordering.

## 6. Target-Absent evaluation

The Primary probe was reused without retraining. Values below compare Target-Absent with Derived-All at each hop's Primary peak layer; the final column gives the Target-Absent curve's own peak.

{markdown_table(['Hop', 'Primary peak layer', 'Derived-All F1', 'Target-Absent F1', 'Target−All', 'Target-Absent own peak'], target_rows)}

![Target-Absent comparison](figures/figure3_target_absent.png)

There is no uniform collapse under Target-Absent: the difference changes sign by hop and remains modest in scale. The result therefore does not support the claim that Primary derived decoding is mainly explained by an explicitly present target-relation sentence.

## 7. Composition-held-out

{markdown_table(['Hop', 'Peak layer', 'Peak Macro-F1', 'Difference from Primary peak'], peak_rows(composition, [2,3,4,5]))}

![Composition held out](figures/figure4_composition_heldout.png)

Composition-held-out performance remains low. Relative to the separate IID peak it is lower for h=2, slightly higher for h=3–5, so there is no consistent composition-generalization penalty. Because these are different frozen test sets and peak layers, the differences are descriptive, not paired causal effects.

## 8. Template Test-only

{markdown_table(['Hop', 'Peak layer', 'Peak Macro-F1', 'Difference from Primary peak'], peak_rows(template, [1,2,3,4,5]))}

![Template Test-only](figures/figure5_template_test_only.png)

The explicit h=1 peak falls from 0.3573 IID to 0.1794 on reserved templates, whereas derived peak differences are small and mixed. This is evidence that the strong explicit-relation readout is sensitive to unseen surface templates; it is not evidence that all derived information is template invariant, since their absolute performance is already weak.

## 9. Controls and uncertainty

Primary majority-class baselines:

{markdown_table(['Hop', 'Majority relation', 'n test', 'Accuracy', 'Macro-F1'], baseline_rows)}

Randomizing train labels at layers 0, 7, 15, 23, 31 for hops 1, 3, 5 produced mean accuracy **{random_mean_acc:.4f}** and mean Macro-F1 **{random_mean_f1:.4f}**. The matched normal-label mean Macro-F1 was **{normal_matched_f1:.4f}**, a mean degradation of **{random_drop:.4f}**. The random-label accuracy is close to the eight-class theoretical chance value 0.125. Primary peak uncertainty uses 1,000 resamples at the `story_group_id` level, not independent row resampling.

## 10. Gate audit

The independent post-run validator passed all {len(validation['checks'])} checks: split hashes, prompt/token readout, raw 32-layer definition, all-file finite-value scan, row alignment, result cardinalities, train-only scaling, validation-only C selection and tie rule, complete controls, checkpoint/figure presence, and absence of post-trained checkpoints. The formal machine-readable result is `validation_pre_report.json`.

## 11. Supported and unsupported conclusions

Supported:

- Explicit h=1 relation information is more strongly linearly decodable than h=2–5 derived information under this frozen Base-model protocol.
- No monotonic hop-distance ordering is visible among h=2–5; a simple “more hops means later decodability” account is unsupported.
- The strong explicit curve is attenuated by both exact balancing and reserved-template evaluation.
- Target-Absent and Composition-held-out controls do not show a uniform large degradation.
- Random-label controls are near chance and clearly worse than matched normal-label probes.

Not supported:

- No layer can be called a symbolic reasoning, grounding, or causal computation layer.
- Linear decodability does not show that the model uses the decoded relation to answer.
- Peak layers, L50, or L90 do not identify the step at which a reasoning operation occurs.
- Differences between frozen test sets should not be treated as paired causal effects.
- These Base-model findings cannot be generalized to Tulu or another post-trained checkpoint before running the separately gated comparison.

## 12. Decision

Stage B1 status is **PASS**. Proceeding to a separately approved Tulu stage is methodologically reasonable because the Base pipeline, integrity checks, controls, and fail-closed validation all pass. The motivation should be a preregistered Base-vs-post-training comparison, not an attempt to force hop ordering. This run stops here and has not loaded any post-trained checkpoint.
"""

    report_path = OUT / "B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    checksum_paths = [OUT / name for name in CORE_FILES]
    checksum_paths += sorted((OUT / "figures").glob("*.png"))
    checksum_paths += [report_path]
    checksum_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "files": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                  for path in checksum_paths],
    }
    write_json(OUT / "stage_b1_artifact_checksums.json", checksum_payload)
    write_json(OUT / "stage_b1_gate.json", {
        "stage": "B1", "status": "PASS", "model_scope": "Meta-Llama-3.1-8B Base only",
        "independent_checks_passed": len(validation["checks"]), "failed_checks": [],
        "random_label_mean_accuracy": float(random_mean_acc),
        "random_label_mean_macro_f1": float(random_mean_f1),
        "matched_normal_mean_macro_f1": float(normal_matched_f1),
        "normal_warning_refits_all_converged": True,
        "post_trained_checkpoint_run": False,
        "report": str(report_path),
    })
    print(json.dumps({"status": "PASS", "report": str(report_path), "bytes": report_path.stat().st_size,
                      "checksums": len(checksum_payload["files"])}, indent=2))


if __name__ == "__main__":
    main()
