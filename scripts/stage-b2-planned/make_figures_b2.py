#!/usr/bin/env python3
"""Create the ten pre-registered Stage B2 figures with shared scales."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ai/projects/reasoning-geometry-pilot")
OUT = ROOT / "artifacts/stage_b2"
FIG = OUT / "figures"
MODELS = ("base", "sft", "dpo", "final")
LABEL = {"base": "Base", "sft": "SFT", "dpo": "DPO", "final": "Final"}
COLORS = {"base": "#4C78A8", "sft": "#F58518", "dpo": "#54A24B", "final": "#B279A2"}
HOP_COLORS = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B", 4: "#E45756", 5: "#B279A2"}
STAGE_PAIRS = (("base", "sft"), ("sft", "dpo"), ("dpo", "final"), ("base", "final"))
PAIR_COLOR = {"sft-base": "#F58518", "dpo-sft": "#54A24B", "final-dpo": "#B279A2", "final-base": "#4C78A8"}
GRAY = "#666666"


def style(ax, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=.65, alpha=.7)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def read_control(name):
    frames=[]
    for model in MODELS:
        f=pd.read_csv(OUT/"controls"/name/f"{model}.csv")
        f["model"]=model
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8,"figure.titlesize":14})
    behavior=pd.read_csv(OUT/"behavior/all_models.csv")
    bdelta=pd.read_csv(OUT/"behavior/stagewise_delta.csv")
    within=pd.concat([pd.read_csv(OUT/"within_hop"/f"{m}_primary.csv").assign(model=m) for m in MODELS],ignore_index=True)
    summaries=pd.read_csv(OUT/"within_hop/summaries.csv")
    ldelta=pd.read_csv(OUT/"within_hop/layerwise_delta.csv")
    transfer=pd.concat([pd.read_csv(OUT/"cross_hop"/f"{m}.csv").assign(model=m) for m in MODELS],ignore_index=True)
    gap=pd.read_csv(OUT/"cross_hop/transfer_gap.csv")
    template=read_control("template")
    composition=read_control("composition")
    chart=[]

    # Figure 1
    fig,ax=plt.subplots(figsize=(8,4.7))
    for m in MODELS:
        p=behavior[(behavior.model==m)&(behavior.dataset=="primary")&(behavior.scoring=="sum")].sort_values("hop")
        y=p.accuracy.to_numpy(); err=np.vstack([y-p.accuracy_ci95_low,p.accuracy_ci95_high-y])
        ax.errorbar(p.hop,y,yerr=err,marker="o",capsize=2.5,lw=1.7,color=COLORS[m],label=LABEL[m])
    ax.axhline(.125,color=GRAY,ls="--",lw=1.1,label="Chance (0.125)")
    ax.set_xticks(range(1,6)); ax.set_ylim(0,.26); style(ax,"Hop","Accuracy")
    ax.set_title("Behavioral accuracy by hop (summed candidate log-probability)"); ax.legend(frameon=False,ncol=5)
    f=save(fig,"figure_01_behavior_accuracy_by_hop.png"); chart.append({"figure":f,"source":"behavior/all_models.csv","claim":"Checkpoint-wise Primary behavioral accuracy with paired rows and independent CIs."})

    # Figure 2
    fig,ax=plt.subplots(figsize=(8,4.7))
    p=bdelta[(bdelta.dataset=="primary")&(bdelta.scoring=="sum")&(bdelta.metric=="accuracy")]
    for earlier,later in STAGE_PAIRS:
        name=f"{later}-{earlier}"; q=p[p.comparison==name].sort_values("hop"); y=q.point_estimate.to_numpy(); err=np.vstack([y-q.ci95_low,q.ci95_high-y])
        ax.errorbar(q.hop,y,yerr=err,marker="o",capsize=2.5,lw=1.6,color=PAIR_COLOR[name],label=name)
    ax.axhline(0,color=GRAY,ls="--",lw=1); ax.set_xticks(range(1,6)); style(ax,"Hop","Paired ΔAccuracy")
    ax.set_title("Behavioral change by post-training stage"); ax.legend(frameon=False,ncol=2)
    f=save(fig,"figure_02_behavior_delta_by_hop.png"); chart.append({"figure":f,"source":"behavior/stagewise_delta.csv","claim":"Paired story-group bootstrap deltas; intervals crossing zero are not treated as improvements."})

    # Figure 3: four same-scale heatmaps
    mats=[]
    for m in MODELS: mats.append(within[within.model==m].pivot(index="hop",columns="layer",values="test_macro_f1").sort_index())
    vmin=min(x.to_numpy().min() for x in mats); vmax=max(x.to_numpy().max() for x in mats)
    fig,axes=plt.subplots(2,2,figsize=(12,6.2),sharex=True,sharey=True)
    im=None
    for ax,m,mat in zip(axes.flat,MODELS,mats):
        im=ax.imshow(mat.to_numpy(),aspect="auto",origin="lower",cmap="viridis",vmin=vmin,vmax=vmax,interpolation="nearest")
        ax.set_title(LABEL[m]); ax.set_yticks(range(5),[f"h{i}" for i in range(1,6)]); ax.set_xticks(range(0,32,4),range(0,32,4))
    fig.supxlabel("Transformer block output"); fig.supylabel("Hop"); fig.suptitle("Within-hop relation decodability (shared color scale)")
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),pad=.02); cb.set_label("Macro-F1")
    fig.subplots_adjust(left=.08,right=.89,bottom=.1,top=.88,wspace=.12,hspace=.25)
    f=save(fig,"figure_03_within_hop_heatmaps.png"); chart.append({"figure":f,"source":"within_hop/*_primary.csv","claim":"All 32×5 cells are shown on a common scale; no peak-only selection."})

    # Figure 4
    fig,ax=plt.subplots(figsize=(8,4.8))
    for m in MODELS:
        q=summaries[(summaries.model==m)&(summaries.experiment=="primary")&summaries.hop.isin([2,3,4,5])].sort_values("hop")
        ax.plot(q.hop,q.late_layer_mean,marker="o",lw=1.8,color=COLORS[m],label=LABEL[m])
    ax.axhline(.125,color=GRAY,ls="--",lw=1); ax.set_xticks([2,3,4,5]); style(ax,"Derived target hop","Late-layer mean Macro-F1 (L16–31)")
    ax.set_title("Derived relation late-layer decodability"); ax.legend(frameon=False,ncol=4)
    f=save(fig,"figure_04_derived_late_layer_mean.png"); chart.append({"figure":f,"source":"within_hop/summaries.csv","claim":"Pre-registered L16–31 mean by derived hop and checkpoint."})

    # Figure 5
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,hop in zip(axes.flat,[2,3,4,5]):
        for earlier,later in STAGE_PAIRS:
            name=f"{later}-{earlier}"; q=ldelta[(ldelta.experiment=="primary")&(ldelta.hop==hop)&(ldelta.comparison==name)].sort_values("layer")
            ax.plot(q.layer,q.delta_macro_f1,lw=1.45,color=PAIR_COLOR[name],label=name)
        ax.axhline(0,color=GRAY,ls="--",lw=.9); ax.set_title(f"h{hop}"); ax.set_xlim(0,31); ax.set_xticks(range(0,32,4)); style(ax)
    axes[0,0].legend(frameon=False,ncol=2); fig.supxlabel("Transformer block output"); fig.supylabel("ΔMacro-F1")
    fig.suptitle("Stagewise within-hop decodability change across all layers"); fig.tight_layout(rect=[.03,.03,1,.94])
    f=save(fig,"figure_05_stagewise_delta_layer_profiles.png"); chart.append({"figure":f,"source":"within_hop/layerwise_delta.csv","claim":"Full derived-hop delta profiles, including consecutive-stage and Final−Base contrasts."})

    # Figure 6
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,m in zip(axes.flat,MODELS):
        for hop in [2,3,4,5]:
            q=transfer[(transfer.model==m)&(transfer.target_hop==hop)].sort_values("layer")
            ax.plot(q.layer,q.test_macro_f1,lw=1.45,color=HOP_COLORS[hop],label=f"h1→h{hop}")
        ax.axhline(.125,color=GRAY,ls="--",lw=.9); ax.set_title(LABEL[m]); ax.set_xlim(0,31); ax.set_xticks(range(0,32,4)); style(ax)
    axes[0,0].legend(frameon=False,ncol=2); fig.supxlabel("Transformer block output"); fig.supylabel("Transfer Macro-F1")
    fig.suptitle("Explicit-relation (h1) decoder transferred to derived hops"); fig.tight_layout(rect=[.03,.03,1,.94])
    f=save(fig,"figure_06_h1_to_derived_transfer.png"); chart.append({"figure":f,"source":"cross_hop/{model}.csv","claim":"All 32-layer h1→h2–h5 transfer curves for each checkpoint."})

    # Figure 7
    late=gap[(gap.target_hop.isin([2,3,4,5]))&(gap.layer.between(16,31))].groupby(["model","target_hop"],as_index=False).absolute_transfer_gap.mean()
    fig,ax=plt.subplots(figsize=(8,4.8))
    for m in MODELS:
        q=late[late.model==m].sort_values("target_hop"); ax.plot(q.target_hop,q.absolute_transfer_gap,marker="o",lw=1.8,color=COLORS[m],label=LABEL[m])
    ax.axhline(0,color=GRAY,ls="--",lw=1); ax.set_xticks([2,3,4,5]); style(ax,"Derived target hop","Late mean gap: within − h1 transfer")
    ax.set_title("Explicit–derived transfer gap by checkpoint"); ax.legend(frameon=False,ncol=4)
    f=save(fig,"figure_07_transfer_gap_late_mean.png"); chart.append({"figure":f,"source":"cross_hop/transfer_gap.csv","claim":"Pre-registered absolute transfer gap averaged over L16–31."})

    # Figure 8: no dual axis
    beh=behavior[(behavior.dataset=="primary")&(behavior.scoring=="sum")&behavior.hop.isin([2,3,4,5])][["model","hop","accuracy"]]
    dec=summaries[(summaries.experiment=="primary")&summaries.hop.isin([2,3,4,5])][["model","hop","late_layer_mean"]]
    q=beh.merge(dec,on=["model","hop"])
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for m in MODELS:
        p=q[q.model==m].sort_values("hop"); axes[0].plot(p.hop,p.accuracy,marker="o",color=COLORS[m],label=LABEL[m]); axes[1].plot(p.hop,p.late_layer_mean,marker="o",color=COLORS[m],label=LABEL[m])
    axes[0].axhline(.125,color=GRAY,ls="--",lw=1); axes[1].axhline(.125,color=GRAY,ls="--",lw=1)
    axes[0].set_xticks([2,3,4,5]); axes[1].set_xticks([2,3,4,5]); style(axes[0],"Derived hop","Behavior accuracy"); style(axes[1],"Derived hop","Late mean probe Macro-F1")
    axes[0].legend(frameon=False,ncol=2); fig.suptitle("Behavior and derived decodability shown on separate axes"); fig.tight_layout(rect=[0,0,1,.93])
    f=save(fig,"figure_08_behavior_vs_decodability.png"); chart.append({"figure":f,"source":"behavior/all_models.csv; within_hop/summaries.csv","claim":"Separate aligned panels avoid implying comparable scales or causality."})

    # Figure 9
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True)
    for m in MODELS:
        q=template[(template.model==m)&(template.hop==1)].sort_values("layer"); axes[0].plot(q.layer,q.test_macro_f1,lw=1.5,color=COLORS[m],label=LABEL[m])
        q=template[(template.model==m)&template.hop.isin([2,3,4,5])].groupby("layer",as_index=False).test_macro_f1.mean(); axes[1].plot(q.layer,q.test_macro_f1,lw=1.5,color=COLORS[m],label=LABEL[m])
    for ax,title in zip(axes,["Explicit h1","Derived h2–h5 mean"]): ax.axhline(.125,color=GRAY,ls="--",lw=1); ax.set_title(title); ax.set_xlim(0,31); ax.set_xticks(range(0,32,4)); style(ax,"Transformer block output","Macro-F1")
    axes[0].legend(frameon=False,ncol=2); fig.suptitle("Mid-level Template Test-only comparison"); fig.tight_layout(rect=[0,0,1,.93])
    f=save(fig,"figure_09_template_test_only.png"); chart.append({"figure":f,"source":"controls/template/{model}.csv","claim":"Reserved Mid-level template families unseen during probe train/validation; h1 and derived trends separated."})

    # Figure 10
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,hop in zip(axes.flat,[2,3,4,5]):
        for m in MODELS:
            q=composition[(composition.model==m)&(composition.hop==hop)].sort_values("layer"); ax.plot(q.layer,q.test_macro_f1,lw=1.45,color=COLORS[m],label=LABEL[m])
        ax.axhline(.125,color=GRAY,ls="--",lw=.9); ax.set_title(f"h{hop}"); ax.set_xlim(0,31); ax.set_xticks(range(0,32,4)); style(ax)
    axes[0,0].legend(frameon=False,ncol=2); fig.supxlabel("Transformer block output"); fig.supylabel("Macro-F1")
    fig.suptitle("Composition-held-out decodability by checkpoint"); fig.tight_layout(rect=[.03,.03,1,.94])
    f=save(fig,"figure_10_composition_heldout.png"); chart.append({"figure":f,"source":"controls/composition/{model}.csv","claim":"Generalization to frozen unseen unordered composition signatures across all layers."})

    with open(FIG/"chart_map.json","w",encoding="utf-8") as fp: json.dump(chart,fp,indent=2,ensure_ascii=False)
    print(json.dumps({"status":"PASS","figures":len(chart),"paths":[x["figure"] for x in chart]},indent=2))

if __name__=="__main__": main()
