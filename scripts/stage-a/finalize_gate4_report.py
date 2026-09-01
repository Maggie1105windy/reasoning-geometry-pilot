#!/usr/bin/env python3
"""Finalize Gate 4 paper comparison, report, and experiment log."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT = Path("/home/ai/projects/linear-probe-repro")
RESULTS = PROJECT / "results"
ACTIVATION_DIR = PROJECT / "activations/full/pythia-2.8b/world/coords"
UPSTREAM = PROJECT / "external/world-models"
PAPER_URL = "https://arxiv.org/html/2310.02207"
REPO_URL = "https://github.com/wesg52/world-models"


def frame_text(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def atomic_json(path: Path, data: dict):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


probe = pd.read_csv(RESULTS / "a2_4_full_layer_r2.csv")
controls = pd.read_csv(RESULTS / "a2_4_random_label_control.csv")
comparison = pd.read_csv(RESULTS / "a2_4_gate3_comparison.csv")
live = pd.read_csv(RESULTS / "a2_4_live_forward_comparison.csv")
stats = pd.read_csv(RESULTS / "a2_4_activation_stats.csv")
manifest = pd.read_csv(RESULTS / "a2_4_full_sample_manifest.csv")
scaler = json.loads((RESULTS / "a2_4_target_scaler.json").read_text())
baseline = json.loads((RESULTS / "a2_4_mean_baseline.json").read_text())
metadata = json.loads((ACTIVATION_DIR / "metadata.json").read_text())

assert len(probe) == 32 and probe.layer.tolist() == list(range(32))
assert len(manifest) == 39585
assert int(manifest.is_test.astype(bool).sum()) == 7917
assert stats.nan_count.sum() == 0 and stats.inf_count.sum() == 0
assert len(live) == 50 and live.cosine_similarity.min() > 0.999
assert controls.r2_difference_normal_minus_random.gt(0.1).all()

peak = probe.loc[probe.r2_overall.idxmax()]
pearson_mini = float(pearsonr(comparison.full_r2_overall, comparison.mini_r2_overall).statistic)
spearman_mini = float(spearmanr(comparison.full_r2_overall, comparison.mini_r2_overall).statistic)
mean_abs_delta_mini = float(comparison.delta_full_minus_mini_overall.abs().mean())

visual_reference = {
    "source": "Gurnee & Tegmark Figure 2, World Map panel",
    "source_url": PAPER_URL,
    "series": "Pythia-2.8B dashed purple curve",
    "status": "approximate visual comparison only",
    "official_raw_layer_values_found": False,
    "reason_no_numeric_curve_metrics": "The released repository contains no Pythia raw layer-wise result artifact; visual estimates are not treated as exact values.",
    "approximate_ranges": {
        "depth_0": "roughly 0.30-0.35",
        "depth_0.25": "roughly 0.48-0.52",
        "depth_0.5": "roughly 0.60-0.65",
        "late_plateau": "roughly 0.70-0.73",
    },
    "paper_qualitative_description": "quality rises smoothly through the first half of layers before reaching a plateau",
    "our_values": {
        "layer_0_depth_0": float(probe.loc[probe.layer == 0, "r2_overall"].iloc[0]),
        "layer_8_depth_0.258": float(probe.loc[probe.layer == 8, "r2_overall"].iloc[0]),
        "layer_16_depth_0.516": float(probe.loc[probe.layer == 16, "r2_overall"].iloc[0]),
        "layer_19_depth_0.613": float(probe.loc[probe.layer == 19, "r2_overall"].iloc[0]),
        "peak_layer": int(peak.layer),
        "peak_r2": float(peak.r2_overall),
        "layer_31": float(probe.loc[probe.layer == 31, "r2_overall"].iloc[0]),
    },
    "assessment": "Strong best-effort reproduction",
}
atomic_json(RESULTS / "a2_4_figure2_visual_reference.json", visual_reference)

audit = f"""# Gate 4 Official-result Source Audit

## Sources checked

- [Official paper / arXiv HTML]({PAPER_URL}), especially Figure 2 and Section 3.1.
- [Official world-models repository]({REPO_URL}) at local commit `{metadata['external_world_models_commit']}`.
- All 17 notebooks under `external/world-models/paper_results`.
- Repository branches/tags and numeric artifacts under `paper_results`.

## Finding

No official Pythia layer-wise numeric series was found. The 17 released notebooks contain no `Pythia` string, and `paper_results` contains no CSV, JSON, pickle, NumPy, or similar numeric result artifact. The shallow clone exposes only the official `main` branch at the pinned commit and no tag carrying result data.

Figure 2 is therefore the highest-priority available source for this comparison. The World Map panel visibly identifies Pythia-2.8B as the dashed purple curve. All values mentioned from that curve are explicitly approximate visual ranges, not reconstructed official measurements.

## Paper-level comparison

The paper describes representations as increasing smoothly through the first half of layers before reaching a plateau. The visually estimated Pythia-2.8B World curve begins around 0.30-0.35, reaches roughly 0.60-0.65 near half depth, and plateaus around 0.70-0.73 late in the model. Our curve begins at `{probe.iloc[0].r2_overall:.6f}`, reaches `{probe.loc[probe.layer == 16, 'r2_overall'].iloc[0]:.6f}` at layer 16, and peaks at `{peak.r2_overall:.6f}` at layer {int(peak.layer)}.

Because the comparison source is visual, no layer-wise MAE or Pearson/Spearman statistic against the paper curve is reported. Computing such metrics would imply false precision.
"""
(RESULTS / "a2_4_official_comparison_audit.md").write_text(audit, encoding="utf-8")

compact_probe = probe[[
    "layer", "relative_depth", "n_train", "n_test", "hidden_dim", "alpha",
    "r2_overall", "r2_longitude", "r2_latitude", "fit_runtime_seconds",
    "coef_l2_norm",
]]

report = [
    "# Stage A2 Full Reproduction Report",
    "",
    "Status: **PASS — Strong best-effort reproduction**",
    "",
    "## 1. Goal",
    "",
    "Complete the full World × 32-layer Pythia-2.8B spatial linear probing experiment and compare its layer-wise held-out R² curve with Gurnee & Tegmark Figure 2.",
    "",
    "## 2. Paper setup",
    "",
    f"The [official paper]({PAPER_URL}) reports out-of-sample R² for every layer/model/dataset in Figure 2 and describes spatial/temporal decodability as rising through early layers before reaching a plateau.",
    "",
    "## 3. Reproduction scope",
    "",
    "- Best-effort methodological reproduction; not an exact checkpoint/hyperparameter reproduction",
    "- Full 39,585-row released World dataset: 31,668 official train and 7,917 official test",
    "- All 32 Pythia-2.8B transformer blocks",
    "- `coords` prompt and original row ordering",
    "- One independent fixed-alpha Ridge probe per layer",
    "",
    "## 4. Known unavoidable deviations",
    "",
    "- The paper/released code does not pin the Pythia revision. This run pins current non-deduplicated `step143000` at commit `dbe7ae300a54abcdc475a33907b3dff81d25709f`.",
    "- The paper states that lambda is tuned by efficient leave-one-out cross-validation, but current released `RIDGE_ALPHAS` publishes no Pythia grid. Following the pre-registered Gate 4 policy, this run uses the released `probe=None` fallback `Ridge(alpha=activation_dimension)=Ridge(alpha=2560)`.",
    "- TransformerLens 1.14.0 and the current model conversion may differ from the authors' unpinned environment.",
    "- Official per-channel quint8 quantization is applied per shard/layer to support recovery, rather than after assembling an entire layer.",
    "",
    "## 5. Model",
    "",
    "- `EleutherAI/pythia-2.8b`, revision `step143000`",
    "- Hugging Face commit: `dbe7ae300a54abcdc475a33907b3dff81d25709f`",
    "- Weight SHA-256: `a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf`",
    "- 32 blocks; hidden size 2560; frozen float16 inference",
    "",
    "## 6. Dataset",
    "",
    "The original 39,585 row order and row IDs are preserved. The released `is_test` mask yields exactly 31,668 training rows and 7,917 testing rows; no resampling, filtering, or repartitioning was performed.",
    "",
    "## 7. Prompt/tokenization",
    "",
    "`What are the lat/lon coordinates of <entity>` with released Pythia tokenization, explicit BOS, EOS padding, right padding, and released `entity_mask`.",
    "",
    "## 8. Activation definition",
    "",
    "Entity-last-token activation from `blocks.{layer}.hook_resid_post`, layers 0-31; embedding excluded. Extraction imports the released `get_layer_activations_tl()` path already validated in Gates 2 and 3.",
    "",
    f"- 78 resumable atomic shards; every layer reconstructs to `[39585, 2560]`",
    f"- Official per-channel quint8 storage; dequantized to float32 before sklearn",
    f"- Aggregate forward runtime: {metadata['aggregate_forward_runtime_seconds']:.3f} seconds ({metadata['samples_per_forward_second']:.3f} samples/s)",
    f"- Quantize/save runtime: {metadata['aggregate_quantize_save_runtime_seconds']:.3f} seconds",
    f"- Peak allocated/reserved VRAM: {metadata['peak_allocated_bytes']/2**30:.3f} / {metadata['peak_reserved_bytes']/2**30:.3f} GiB",
    f"- Stored layer files: {metadata['activation_layer_files_total_bytes']/2**30:.3f} GiB",
    f"- Ten-row × five-layer live check: min cosine {live.cosine_similarity.min():.9f}, mean MAE {live.mae.mean():.9g}, max absolute error {live.max_absolute_error.max():.9g}",
    "",
    "## 9. Probe definition",
    "",
    "Targets are `[longitude, latitude]`. Mean/std are computed using all and only the 31,668 official training rows. Each layer creates a new `Ridge(alpha=2560)`; test labels are used only for final scoring. Overall R² is sklearn's default uniform average across the two outputs.",
    "",
    "```json",
    json.dumps(scaler, indent=2),
    "```",
    "",
    "## 10. Full-data results",
    "",
    frame_text(compact_probe),
    "",
    f"Peak overall R² is **{peak.r2_overall:.6f} at layer {int(peak.layer)}**. Layer 31 is {probe.loc[probe.layer == 31, 'r2_overall'].iloc[0]:.6f}.",
    "",
    "## 11. Layer-wise curve",
    "",
    "The curve starts at 0.295, rises in several early/middle steps, reaches 0.710 at layer 19, and then gains slow substantially. The descriptive plateau onset is therefore around layers 19-21 (relative depth 0.61-0.68); this is not presented as a formal paper threshold.",
    "",
    "See `results/a2_4_full_layer_r2.png`. No smoothing, interpolation, or layer deletion was applied.",
    "",
    "## 12. Longitude vs latitude",
    "",
    f"At the peak-overall layer {int(peak.layer)}, longitude R² is {peak.r2_longitude:.6f} and latitude R² is {peak.r2_latitude:.6f}. Across the late plateau longitude remains modestly more decodable than latitude. See `results/a2_4_lat_lon_layer_r2.png`.",
    "",
    "## 13. Controls",
    "",
    f"Train-mean baseline overall/longitude/latitude R²: `{baseline['r2_overall']:.9g}` / `{baseline['r2_longitude']:.9g}` / `{baseline['r2_latitude']:.9g}`.",
    "",
    frame_text(controls),
    "",
    "Both random-label controls are negative and materially below their normal probes; the test labels were not shuffled.",
    "",
    "## 14. Comparison with Gate 3",
    "",
    frame_text(comparison),
    "",
    f"The five shared points retain the same rank order (Spearman {spearman_mini:.3f}) and have Pearson correlation {pearson_mini:.6f}. Full-data R² is higher at all five points by 0.014-0.098 (mean absolute difference {mean_abs_delta_mini:.6f}), while the early-to-late trend is stable.",
    "",
    "## 15. Comparison with Figure 2",
    "",
    f"The [official repository]({REPO_URL}) does not publish a Pythia raw layer-wise numeric series: all 17 released `paper_results` notebooks and numeric artifact types were audited, with no Pythia result found. The comparison is therefore explicitly **approximate visual comparison**.",
    "",
    "In Figure 2's World Map panel, the dashed purple Pythia-2.8B curve begins around 0.30-0.35, rises to roughly 0.60-0.65 near half depth, and reaches a late plateau around 0.70-0.73. Our corresponding values are 0.295 at layer 0, 0.632 at layer 16, and a 0.743 peak at layer 30. Thus the curve shape, numerical scale, peak region, and plateau timing are broadly consistent; our late plateau appears slightly higher by visual inspection.",
    "",
    "No paper-curve MAE/Pearson/Spearman is computed because visual estimates are not official numbers and would create false precision.",
    "",
    "## 16. Reproduction assessment",
    "",
    "**Strong best-effort reproduction.** The qualitative layer-wise trend agrees with Figure 2, the R² scale and late plateau are close, the full integrity and negative-control gates pass, and every known non-identical or unpinned setting is recorded. This classification does not mean exact checkpoint- or hyperparameter-level reproduction.",
    "",
    "## 17. Limitations",
    "",
    "These results support only that spatial information is linearly decodable from Pythia hidden representations and that decodability changes with depth. They do not establish that the model uses this representation, grounding, symbolic reasoning, or a causal mechanism.",
    "",
    "## 18. Exact artifact paths",
    "",
    "- `/home/ai/projects/linear-probe-repro/results/A2_FULL_REPRODUCTION_REPORT.md`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_full_sample_manifest.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_full_layer_r2.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_full_layer_r2.png`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_lat_lon_layer_r2.png`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_random_label_control.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_mean_baseline.json`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_target_scaler.json`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_activation_stats.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_live_forward_comparison.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_full_activation_integrity.md`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_gate3_comparison.csv`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_official_comparison_audit.md`",
    "- `/home/ai/projects/linear-probe-repro/results/a2_4_figure2_visual_reference.json`",
    "- `/home/ai/projects/linear-probe-repro/activations/full/pythia-2.8b/world/coords/metadata.json`",
    "- `/home/ai/projects/linear-probe-repro/activations/full/pythia-2.8b/world/coords/integrity_report.json`",
    "- `/home/ai/projects/linear-probe-repro/scripts/gate4_a2_full_repro.py`",
    "",
    f"`external/world-models` clean at finalization: **{subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'], text=True).strip() == ''}**.",
]
(RESULTS / "A2_FULL_REPRODUCTION_REPORT.md").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)

metadata["paper_comparison"] = visual_reference
metadata["gate4_assessment"] = "Strong best-effort reproduction"
metadata["gate4_report_finalized"] = True
metadata["upstream_clean_at_report_finalization"] = (
    subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip() == ""
)
atomic_json(ACTIVATION_DIR / "metadata.json", metadata)

experiment = PROJECT / "EXPERIMENT.md"
text = experiment.read_text(encoding="utf-8")
marker = "## Stage A2 Gate 4: full World × 32-layer probe"
if marker not in text:
    text += f"""

{marker}

Status: **PASS — Strong best-effort reproduction**.

- Full released World data: 39,585 rows in original order; 31,668 official train and 7,917 official test
- Activations: all 32 `blocks.{{layer}}.hook_resid_post` matrices, entity-last-token, shape `[39585, 2560]`
- Storage: 78 resumable shards, official per-channel quint8 function applied per shard/layer; 3.072 GiB
- Extraction: batch 16, 286.770 seconds aggregate forward time, 138.037 samples/s
- Peak CUDA allocated/reserved: 5.351 / 5.588 GiB
- Integrity: zero NaN/Inf; 10 rows × 5 layers re-forwarded; minimum saved-vs-live cosine 0.999870837
- Primary probes: 32 independent `Ridge(alpha=2560)` fits with full train-only target standardization
- Peak held-out overall R²: 0.743157 at layer 30; layer 31: 0.741863
- Train-mean baseline overall R²: -0.000013542
- Random-label overall R²: layer 15 = -0.043735; layer 31 = -0.074465
- Figure 2 comparison: matching early-rise/late-plateau shape and close numerical scale; official Pythia raw numeric curve was not published, so comparison is explicitly visual and approximate
- Assessment: Strong best-effort reproduction, not exact checkpoint/hyperparameter reproduction

Detailed outputs are recorded in `results/A2_FULL_REPRODUCTION_REPORT.md` and `results/a2_4_full_activation_integrity.md`. No post-Gate-4 experiment was started.
"""
    experiment.write_text(text, encoding="utf-8")

print(json.dumps({
    "status": "PASS",
    "assessment": "Strong best-effort reproduction",
    "peak_layer": int(peak.layer),
    "peak_r2": float(peak.r2_overall),
    "mini_pearson": pearson_mini,
    "mini_spearman": spearman_mini,
    "paper_numeric_series_available": False,
    "paper_comparison": "approximate visual comparison",
    "upstream_clean": metadata["upstream_clean_at_report_finalization"],
}, indent=2))
