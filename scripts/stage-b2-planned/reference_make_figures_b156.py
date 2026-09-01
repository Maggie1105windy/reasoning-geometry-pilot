#!/usr/bin/env python3
"""Create the six preregistered Stage B1.5-B1.6 static figures."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_b156 import B1, BEHAVIOR, CROSS, FIGURES, ensure_dirs, write_json


BLUE = "#2F6B9A"
ORANGE = "#D17A22"
GRAY = "#666666"
HOP_COLORS = {1: "#2F6B9A", 2: "#D17A22", 3: "#3C8D73", 4: "#A5546A", 5: "#7563A8"}
HOP_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}


def style_axis(ax, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def main():
    ensure_dirs()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 9, "figure.titlesize": 14,
    })
    primary = pd.read_csv(BEHAVIOR / "primary_behavior.csv")
    balanced = pd.read_csv(BEHAVIOR / "balanced_behavior.csv")
    behavior_probe = pd.read_csv(BEHAVIOR / "behavior_vs_probe.csv")
    transfer = pd.read_csv(CROSS / "h1_to_all_layers.csv")
    balanced_transfer = pd.read_csv(CROSS / "balanced_h1_to_all_layers.csv")
    gap = pd.read_csv(CROSS / "transfer_gap.csv")
    within = pd.read_csv(B1 / "primary_probe.csv")
    chart_map = []

    # Figure 1: grouped point estimates with bootstrap confidence intervals.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for data, offset, color, label in [(primary, -0.08, BLUE, "Primary"), (balanced, 0.08, ORANGE, "Balanced")]:
        x = data["hop"].to_numpy() + offset
        y = data["accuracy_sum"].to_numpy()
        yerr = np.vstack([y - data["accuracy_sum_ci95_low"], data["accuracy_sum_ci95_high"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, capsize=3, linewidth=1.4, markersize=6, label=label)
    ax.axhline(0.125, color=GRAY, linestyle="--", linewidth=1.3, label="Chance (0.125)")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 0.34)
    style_axis(ax, "Hop", "Candidate-score accuracy")
    ax.set_title("Behavioral accuracy by reasoning hop")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.text(0.01, 0.01, "Primary score: summed conditional log-probability. Error bars: 95% story-group bootstrap CI.", fontsize=8, color=GRAY)
    f1 = save(fig, "figure_1_behavior_accuracy_vs_hop.png")
    chart_map.append({"figure": f1, "question": "Does behavioral accuracy decrease with hop?", "chart_family": "grouped point estimates with error bars", "fields": ["hop", "accuracy_sum", "accuracy_sum_ci95_low", "accuracy_sum_ci95_high"], "claim": "Primary accuracy declines from h1 to h5 but is near chance throughout; Balanced is noisier.", "source": [str(BEHAVIOR / "primary_behavior.csv"), str(BEHAVIOR / "balanced_behavior.csv")], "qa_notes": "Zero baseline shown; chance line explicit; no smoothing."})

    # Figure 2: vertically separated panels avoid a misleading dual y-axis.
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    axes[0].errorbar(behavior_probe["hop"], behavior_probe["behavior_accuracy"],
                     yerr=np.vstack([behavior_probe["behavior_accuracy"] - behavior_probe["behavior_ci_low"], behavior_probe["behavior_ci_high"] - behavior_probe["behavior_accuracy"]]),
                     fmt="o-", color=BLUE, capsize=3, linewidth=1.6)
    axes[0].axhline(0.125, color=GRAY, linestyle="--", linewidth=1.2)
    axes[0].set_ylim(0, 0.25)
    style_axis(axes[0], ylabel="Behavior accuracy")
    axes[0].set_title("Behavior (summed candidate likelihood)")
    axes[1].plot(behavior_probe["hop"], behavior_probe["probe_peak_macro_f1"], "s-", color=ORANGE, linewidth=1.6, markersize=6)
    axes[1].set_ylim(0, 0.42)
    axes[1].set_xticks(range(1, 6))
    style_axis(axes[1], "Hop", "Peak probe Macro-F1")
    axes[1].set_title("Within-hop linear probe decodability")
    fig.suptitle("Behavior and probe decodability answer different questions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.01, 0.01, "Separate panels preserve each metric's scale; no dual y-axis.", fontsize=8, color=GRAY)
    f2 = save(fig, "figure_2_behavior_vs_probe.png")
    chart_map.append({"figure": f2, "question": "How do behavioral ability and linear decodability vary by hop?", "chart_family": "aligned small multiples", "fields": ["hop", "behavior_accuracy", "probe_peak_macro_f1"], "claim": "Both are strongest at h1; derived-hop behavioral accuracy is near chance while probe F1 is weak.", "source": [str(BEHAVIOR / "behavior_vs_probe.csv")], "qa_notes": "Separate y scales in separate panels; no dual axis."})

    # Figure 3: all target hops across layers.
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for hop in range(1, 6):
        part = transfer[transfer["target_hop"].eq(hop)].sort_values("layer")
        ax.plot(part["layer"], part["test_macro_f1"], color=HOP_COLORS[hop], marker=HOP_MARKERS[hop], markevery=4, linewidth=1.7, markersize=5, label=f"Test h{hop}")
    ax.axhline(0.125, color=GRAY, linestyle="--", linewidth=1.1, label="Chance reference")
    ax.set_xlim(0, 31); ax.set_ylim(0, 0.40); ax.set_xticks(range(0, 32, 4))
    style_axis(ax, "Transformer block output", "Macro-F1")
    ax.set_title("h1-trained relation decoder transferred across hops")
    ax.legend(frameon=False, ncol=3)
    f3 = save(fig, "figure_3_h1_trained_cross_hop_transfer.png")
    chart_map.append({"figure": f3, "question": "Can an h1-trained linear decoder read h2-h5 examples across layers?", "chart_family": "multi-series layer profile", "fields": ["layer", "target_hop", "test_macro_f1"], "claim": "h1 self-decoding is clearly strongest; derived transfer is modest and layer-dependent.", "source": [str(CROSS / "h1_to_all_layers.csv")], "qa_notes": "All 32 layers shown; discrete block outputs; chance is a reference, not a formal Macro-F1 null."})

    # Figure 4: within-hop and h1 transfer, faceted by derived target hop.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, hop in zip(axes.flat, [2, 3, 4, 5]):
        w = within[within["hop"].eq(hop)].sort_values("layer")
        t = transfer[transfer["target_hop"].eq(hop)].sort_values("layer")
        ax.plot(w["layer"], w["test_macro_f1"], color=ORANGE, linewidth=1.7, label="Within-hop")
        ax.plot(t["layer"], t["test_macro_f1"], color=BLUE, linewidth=1.7, linestyle="--", label="h1-trained")
        ax.axhline(0.125, color=GRAY, linestyle=":", linewidth=1.0)
        ax.set_title(f"Target h{hop}")
        ax.set_xlim(0, 31); ax.set_ylim(0, 0.26); ax.set_xticks(range(0, 32, 8))
        style_axis(ax)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.supxlabel("Transformer block output"); fig.supylabel("Macro-F1")
    fig.suptitle("Within-hop probes versus h1-trained transfer")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    f4 = save(fig, "figure_4_within_vs_h1_transfer.png")
    chart_map.append({"figure": f4, "question": "How much does h1 transfer differ from independently trained within-hop probes?", "chart_family": "faceted layer profiles", "fields": ["layer", "hop", "test_macro_f1"], "claim": "Differences are small relative to the uniformly weak derived-hop decodability; their sign varies by hop and layer.", "source": [str(B1 / "primary_probe.csv"), str(CROSS / "h1_to_all_layers.csv")], "qa_notes": "Shared axes across all four facets; no peak-only comparison."})

    # Figure 5: absolute gap heatmap, within minus transfer.
    matrix = gap[gap["target_hop"].isin([2, 3, 4, 5])].pivot(index="target_hop", columns="layer", values="absolute_transfer_gap").sort_index()
    lim = max(0.05, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_yticks(range(4), [f"h{h}" for h in matrix.index])
    ax.set_xticks(range(0, 32, 2), range(0, 32, 2))
    ax.set_xlabel("Transformer block output"); ax.set_ylabel("Target hop")
    ax.set_title("Transfer gap = within-hop Macro-F1 − h1-trained Macro-F1")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Absolute transfer gap")
    fig.tight_layout()
    f5 = save(fig, "figure_5_transfer_gap_heatmap.png")
    chart_map.append({"figure": f5, "question": "Where across layers is within-hop decoding stronger or weaker than h1 transfer?", "chart_family": "diverging heatmap", "fields": ["layer", "target_hop", "absolute_transfer_gap"], "claim": "Gap signs vary; h2/h3 favor within-hop slightly in late layers, whereas h4/h5 often do not.", "source": [str(CROSS / "transfer_gap.csv")], "qa_notes": "Symmetric color limits around zero; negative means transfer exceeds within-hop."})

    # Figure 6: balanced control.
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for hop in range(1, 6):
        part = balanced_transfer[balanced_transfer["target_hop"].eq(hop)].sort_values("layer")
        ax.plot(part["layer"], part["test_macro_f1"], color=HOP_COLORS[hop], marker=HOP_MARKERS[hop], markevery=4, linewidth=1.6, markersize=5, label=f"Test h{hop}")
    ax.axhline(0.125, color=GRAY, linestyle="--", linewidth=1.1, label="Chance reference")
    ax.set_xlim(0, 31); ax.set_ylim(0, 0.34); ax.set_xticks(range(0, 32, 4))
    style_axis(ax, "Transformer block output", "Macro-F1")
    ax.set_title("Balanced-split h1-trained transfer (small-sample control)")
    ax.legend(frameon=False, ncol=3)
    f6 = save(fig, "figure_6_balanced_cross_hop_control.png")
    chart_map.append({"figure": f6, "question": "Does the balanced small-sample control qualitatively support Primary transfer?", "chart_family": "multi-series layer profile", "fields": ["layer", "target_hop", "test_macro_f1"], "claim": "Balanced curves remain noisy and weak; they do not overturn the Primary conclusion.", "source": [str(CROSS / "balanced_h1_to_all_layers.csv")], "qa_notes": "All hops have identical 88-row test cells; interpreted as robustness only."})

    write_json(FIGURES / "chart_map.json", chart_map)
    print(json.dumps({"status": "PASS", "figures": len(chart_map), "paths": [item["figure"] for item in chart_map]}, indent=2))


if __name__ == "__main__":
    main()
