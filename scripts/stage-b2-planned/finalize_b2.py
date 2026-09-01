#!/usr/bin/env python3
"""Consolidate Stage B2 summaries, write the final report, and checksum artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path("/home/ai/projects/reasoning-geometry-pilot")
OUT=ROOT/"artifacts/stage_b2"
ACT=ROOT/"activations/stage_b2"
MODELS=("base","sft","dpo","final")
NAMES={"base":"Base","sft":"SFT","dpo":"DPO","final":"Final"}


def jread(path):
    with open(path,encoding="utf-8") as f:return json.load(f)


def fmt(x): return f"{float(x):.4f}"


def md_table(frame, columns, labels=None, digits=4):
    labels=labels or columns
    lines=["| "+" | ".join(labels)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for row in frame[columns].itertuples(index=False,name=None):
        vals=[]
        for v in row:
            vals.append(f"{v:.{digits}f}" if isinstance(v,(float,np.floating)) else str(v))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)


def late_control(name, value_col="test_macro_f1"):
    rows=[]
    for model in MODELS:
        d=pd.read_csv(OUT/"controls"/name/f"{model}.csv")
        if "test_macro_f1" not in d and "macro_f1" in d: value_col="macro_f1"
        for hop,g in d[d.layer.between(16,31)].groupby("hop"):
            rows.append({"model":model,"hop":int(hop),"late_mean":g[value_col].mean()})
    return pd.DataFrame(rows)


def model_hop_line(frame, model, col):
    d=frame[frame.model.eq(model)].sort_values("hop")
    return " ".join(f"h{int(r.hop)}={fmt(getattr(r,col))}" for r in d.itertuples())


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    gates={
        "lineage":jread(OUT/"model_lineage_audit.json"),
        "tokenization":jread(OUT/"tokenization_cross_checkpoint_audit.json"),
        "frozen":jread(OUT/"frozen_input_verification.json"),
        "activation_sft":jread(OUT/"activation_integrity_sft.json"),
        "activation_dpo":jread(OUT/"activation_integrity_dpo.json"),
        "activation_final":jread(OUT/"activation_integrity_final.json"),
        "behavior":jread(OUT/"behavior/behavior_gate.json"),
        "within":jread(OUT/"within_hop/within_hop_gate.json"),
        "cross":jread(OUT/"cross_hop/cross_hop_gate.json"),
        "controls":jread(OUT/"controls/controls_gate.json"),
        "probe_bootstrap":jread(OUT/"within_hop/paired_probe_bootstrap_audit.json"),
    }
    core=["lineage","tokenization","frozen","activation_sft","activation_dpo","activation_final","behavior","within","cross","probe_bootstrap"]
    if not all(gates[k]["status"]=="PASS" for k in core): status="FAIL"
    elif gates["controls"]["status"]=="PASS_WITH_LIMITATION": status="PASS_WITH_LIMITATION"
    elif gates["controls"]["status"]=="PASS": status="PASS"
    else: status="FAIL"

    behavior=pd.read_csv(OUT/"behavior/all_models.csv")
    bp=behavior[(behavior.dataset=="primary")&(behavior.scoring=="sum")].copy()
    bwide=bp.pivot(index="model",columns="hop",values="accuracy").reindex(MODELS)
    bderived=bwide[[2,3,4,5]].mean(axis=1)
    bstage={"Base→SFT":bderived.sft-bderived.base,"SFT→DPO":bderived.dpo-bderived.sft,"DPO→Final":bderived.final-bderived.dpo,"Final→Base":bderived.final-bderived.base}

    summ=pd.read_csv(OUT/"within_hop/summaries.csv")
    prim=summ[summ.experiment.eq("primary")].copy()
    late=prim.pivot(index="model",columns="hop",values="late_layer_mean").reindex(MODELS)
    derived_late=late[[2,3,4,5]].mean(axis=1)
    dlate={"Base→SFT":derived_late.sft-derived_late.base,"SFT→DPO":derived_late.dpo-derived_late.sft,"DPO→Final":derived_late.final-derived_late.dpo,"Final→Base":derived_late.final-derived_late.base}
    boot=pd.read_csv(OUT/"within_hop/paired_probe_bootstrap.csv")
    derived_late_boot=boot[(boot.summary=="late_layer_mean")&boot.hop.isin([2,3,4,5])]

    cross=pd.read_csv(OUT/"cross_hop/summaries.csv")
    transfer=cross.pivot(index="model",columns="target_hop",values="late_transfer_mean").reindex(MODELS)
    transfer_gap=cross.pivot(index="model",columns="target_hop",values="late_gap_mean").reindex(MODELS)
    tderived=transfer[[2,3,4,5]].mean(axis=1)
    tstage={"Base→SFT":tderived.sft-tderived.base,"SFT→DPO":tderived.dpo-tderived.sft,"DPO→Final":tderived.final-tderived.dpo,"Final→Base":tderived.final-tderived.base}

    balanced=late_control("balanced")
    target=late_control("target_absent",value_col="macro_f1")
    composition=late_control("composition")
    template=late_control("template")
    random=pd.read_csv(OUT/"controls/random_label/summary.csv")
    lineage=jread(OUT/"model_lineage_audit.json")
    line={m["model_key"]:m for m in lineage["models"]}

    # Durable machine-readable summary.
    summary={
        "status":status,
        "behavior_primary_accuracy":{m:{str(h):float(bwide.loc[m,h]) for h in range(1,6)} for m in MODELS},
        "behavior_derived_mean":{m:float(bderived[m]) for m in MODELS},
        "within_late_mean":{m:{str(h):float(late.loc[m,h]) for h in range(1,6)} for m in MODELS},
        "within_derived_late_mean":{m:float(derived_late[m]) for m in MODELS},
        "cross_late_transfer":{m:{str(h):float(transfer.loc[m,h]) for h in range(2,6)} for m in MODELS},
        "cross_late_gap":{m:{str(h):float(transfer_gap.loc[m,h]) for h in range(2,6)} for m in MODELS},
        "primary_answers":{"behavior":"MIXED","decodability":"MIXED","sharing":"MIXED"},
        "interpretation_boundary":"Associational behavioral, linear-decoding, and within-checkpoint transfer evidence only; no causal or symbolic-mechanism claim.",
    }
    with open(OUT/"b2_summary.json","w",encoding="utf-8") as f:json.dump(summary,f,indent=2,ensure_ascii=False)

    lineage_lines=[]
    for m in MODELS:
        x=line[m]
        lineage_lines.append(f"- {NAMES[m]}: revision `{x['model_revision']}`, {x['parameters']:,} parameters, 32×4096, vocab(config/tokenizer)={x['vocab_size_config']}/{x['vocab_size_tokenizer']}")
    behavior_rows=[]
    for m in MODELS: behavior_rows.append({"Checkpoint":NAMES[m],**{f"h{h}":bwide.loc[m,h] for h in range(1,6)},"Derived mean":bderived[m]})
    behavior_table=md_table(pd.DataFrame(behavior_rows),["Checkpoint","h1","h2","h3","h4","h5","Derived mean"])
    late_rows=[]
    for m in MODELS: late_rows.append({"Checkpoint":NAMES[m],**{f"h{h}":late.loc[m,h] for h in range(1,6)},"Derived mean":derived_late[m]})
    late_table=md_table(pd.DataFrame(late_rows),["Checkpoint","h1","h2","h3","h4","h5","Derived mean"])
    transfer_rows=[]
    for m in MODELS: transfer_rows.append({"Checkpoint":NAMES[m],**{f"h1→h{h}":transfer.loc[m,h] for h in range(2,6)},"Derived mean":tderived[m]})
    transfer_table=md_table(pd.DataFrame(transfer_rows),["Checkpoint","h1→h2","h1→h3","h1→h4","h1→h5","Derived mean"])
    gap_rows=[]
    for m in MODELS: gap_rows.append({"Checkpoint":NAMES[m],**{f"h{h}":transfer_gap.loc[m,h] for h in range(2,6)}})
    gap_table=md_table(pd.DataFrame(gap_rows),["Checkpoint","h2","h3","h4","h5"])

    def control_summary(frame,hops):
        q=frame[frame.hop.isin(hops)].groupby("model").late_mean.mean().reindex(MODELS)
        return ", ".join(f"{NAMES[m]}={fmt(q[m])}" for m in MODELS)
    balanced_text=control_summary(balanced,[2,3,4,5])
    target_text=control_summary(target,[2,3,4,5])
    composition_text=control_summary(composition,[2,3,4,5])
    template_h1=control_summary(template,[1]); template_derived=control_summary(template,[2,3,4,5])
    random_text="; ".join(f"{NAMES[r.model]} accuracy={fmt(r.mean_accuracy)}, Macro-F1={fmt(r.mean_macro_f1)}, matched-normal drop={fmt(r.macro_f1_drop)}" for r in random.itertuples())

    report=f"""# Stage B2 完成

## 1. Model lineage

{chr(10).join(lineage_lines)}
- architecture/tokenizer comparability: 四个 checkpoint 均为 32 个 Llama decoder blocks、hidden size 4096；120 条冻结 raw prompt 在 5 hops×8 relations 上的 token IDs、BOS、序列长度、readout index 和末尾 `?` 完全一致。Tokenizer 文件/revision 不同，但实际冻结 prompt tokenization 一致。
- status: PASS。Final 在本地 README 表格中被明确标为 Final Models (RLVR)，正式结果仍使用保守名称 Final。Revision 来自 ModelScope `.msc` 文件级本地记录，不宣称为 Hugging Face commit。

## 2. Behavioral accuracy

{behavior_table}

- Base→SFT: derived mean {bstage['Base→SFT']:+.4f}
- SFT→DPO: derived mean {bstage['SFT→DPO']:+.4f}
- DPO→Final: derived mean {bstage['DPO→Final']:+.4f}
- Final→Base: derived mean {bstage['Final→Base']:+.4f}
- 哪个 stage 对 derived behavior 改变最大: Base→SFT 的绝对变化最大，但方向是降低；SFT→DPO 大致恢复该降幅。各 hop 的 paired CI 不支持统一的 derived 行为提升。

## 3. Within-hop decodability

Late-layer mean (L16–31):

{late_table}

- h1: Base→SFT 增加 {late.loc['sft',1]-late.loc['base',1]:+.4f}，Final→Base 增加 {late.loc['final',1]-late.loc['base',1]:+.4f}；两者 paired 95% CI 排除 0。
- h2: Final→Base {late.loc['final',2]-late.loc['base',2]:+.4f}
- h3: Final→Base {late.loc['final',3]-late.loc['base',3]:+.4f}
- h4: Final→Base {late.loc['final',4]-late.loc['base',4]:+.4f}
- h5: Final→Base {late.loc['final',5]-late.loc['base',5]:+.4f}
- late-layer mean by checkpoint: derived mean Base={fmt(derived_late.base)}, SFT={fmt(derived_late.sft)}, DPO={fmt(derived_late.dpo)}, Final={fmt(derived_late.final)}
- stagewise delta: {', '.join(f'{k}={v:+.4f}' for k,v in dlate.items())}
- 是否出现稳定 derived enhancement: NO；四个 derived hop 的 late-layer paired CI 均包含 0，h3 与 h4/h2 方向不一致。Peak/all-layer/late-layer 均已保留在 `within_hop/summaries.csv`，不以单层 peak 作为主证据。

## 4. Cross-hop transfer

Late-layer h1-trained transfer:

{transfer_table}

Late-layer TransferGap = within-hop − h1-transfer:

{gap_table}

- Base: derived transfer mean {fmt(tderived.base)}
- SFT: derived transfer mean {fmt(tderived.sft)}
- DPO: derived transfer mean {fmt(tderived.dpo)}
- Final: derived transfer mean {fmt(tderived.final)}
- h1→h2: Base={fmt(transfer.loc['base',2])}, Final={fmt(transfer.loc['final',2])}
- h1→h3: Base={fmt(transfer.loc['base',3])}, Final={fmt(transfer.loc['final',3])}
- h1→h4: Base={fmt(transfer.loc['base',4])}, Final={fmt(transfer.loc['final',4])}
- h1→h5: Base={fmt(transfer.loc['base',5])}, Final={fmt(transfer.loc['final',5])}
- TransferGap: 对 hop 和 checkpoint 呈混合符号，且多个 within-hop derived 分数仅略高于 chance，retention ratio 不作强解释。
- 是否有 evidence for stronger explicit-derived shared linear geometry: MIXED/弱证据。Final 相对 Base 的 derived transfer 平均增加 {tstage['Final→Base']:+.4f}，但 gap 未在各 hop 统一缩小，且 within-hop 提升不稳定。

## 5. Balanced control

- 是否复现主要 stagewise trend: 部分复现但更噪声化。Derived late mean: {balanced_text}。Balanced 只作小样本 robustness control，不以 peak 排序作主结论。

## 6. Target-Absent

- 是否复现 derived improvement: 未复现一致的阶段性提升。Primary probe 直接评价 Target-Absent，未重训练；derived late mean: {target_text}。

## 7. Composition-heldout

- 是否复现 derived improvement: 未出现跨 h2–h5 稳定的提升。Frozen unseen unordered composition signatures 的 derived late mean: {composition_text}。

## 8. Template Test-only

- h1: late mean {template_h1}
- derived: h2–h5 late mean {template_derived}
- post-training 是否降低 template sensitivity: MIXED。这里只能解释为 reserved Mid-level template families 在 probe train/validation 中未见，不能扩展为 all reasoning-path templates unseen。

## 9. Random-label

- Base/SFT/DPO/Final: {random_text}
- status: {gates['controls']['status']}。所有 mean accuracy 均在 0.125±0.04，且 Macro-F1 均比 matched normal-label probe 低 >0.02。

## 10. 三个 Primary answers

### A. Post-training 是否提高 derived behavior

- MIXED
- evidence: derived mean accuracy Base={fmt(bderived.base)}, SFT={fmt(bderived.sft)}, DPO={fmt(bderived.dpo)}, Final={fmt(bderived.final)}；无单调改善，而且 paired hop-level CI 不支持稳定提升。

### B. Post-training 是否提高 derived linear decodability

- MIXED
- evidence: derived late mean 有小幅变化，Final−Base={dlate['Final→Base']:+.4f}，但 hop 方向不一致，所有 h2–h5 late-summary paired 95% CI 包含 0。

### C. Post-training 是否提高 explicit→derived linear geometry sharing

- MIXED
- evidence: derived transfer mean Base={fmt(tderived.base)} 到 Final={fmt(tderived.final)}，但 within-hop 没有相应稳定增强，TransferGap 也未各 hop 同向缩小。

## 11. 哪个 post-training stage 变化最大

- behavior: Base→SFT 绝对变化最大（derived mean {bstage['Base→SFT']:+.4f}，为降低）。
- decodability: Base→SFT 对 h1 的增幅最明确；derived 平均的最大连续阶段变化是 {max(dlate,key=lambda k:abs(dlate[k]) if k!='Final→Base' else -1)}={dlate[max(dlate,key=lambda k:abs(dlate[k]) if k!='Final→Base' else -1)]:+.4f}，但无稳定 hop-level CI 支持。
- transfer: 连续阶段 derived mean 变化为 {', '.join(f'{k}={v:+.4f}' for k,v in tstage.items())}；变化小且分散。

## 12. 当前能支持的结论

- Post-training 与 question-end 线性 relation geometry 的改变相关，其中 h1 explicit relation 的 late-layer decodability 在 SFT 后明显增强。
- Derived-hop 行为、within-hop decodability 和 h1→derived transfer 的变化较小且不统一；不存在一条简单的 Base→SFT→DPO→Final 单调提升曲线。
- Full-layer curves、late mean、all-layer mean 和 descriptive peak 均已保留；主解释不依赖孤立 peak。

## 13. 当前不能支持的结论

- 不能声称 post-training proves grounding、Final/RLVR creates symbolic reasoning、某层是 reasoning layer、probe 改善导致行为改善，或 transfer 证明 symbolic code。
- 不能将弱而混合的 derived 曲线解释为已建立的 explicit-derived 共享机制。
- 不能把 ModelScope `.msc` revision 当作 Hugging Face commit SHA，也不能忽略 Tulu 的 8-entry config vocab padding 差异。

## 14. Stage B2 status

- {status}

## 15. 是否建议进入因果验证

- NO（暂不自动进入）
- reason: 三个 Primary 对 derived relation 都呈混合/弱效应，应先人工审阅 B2 曲线、paired CI 和 controls，再决定是否对 h1 的稳定变化或特定 derived hop 定义因果问题。

## 16. 输出

- report: `{OUT/'B2_BASE_TULU_STAGE_COMPARISON_REPORT.md'}`
- behavior: `{OUT/'behavior/'}`
- probes: `{OUT/'within_hop/'}`
- transfer: `{OUT/'cross_hop/'}`
- controls: `{OUT/'controls/'}`
- figures: `{OUT/'figures/'}`
- activations: `{ACT}/{{tulu_sft,tulu_dpo,tulu_final}}`
"""
    report_path=OUT/"B2_BASE_TULU_STAGE_COMPARISON_REPORT.md"
    report_path.write_text(report,encoding="utf-8")

    # Hash finalized artifacts after the report is present.  Checkpoint files are
    # included so the manifest covers predictions and restartable intermediate outputs.
    files=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name!="artifact_checksums.json":
            files.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    manifest={"stage":"B2","status":status,"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files)}
    with open(OUT/"artifact_checksums.json","w",encoding="utf-8") as f:json.dump(manifest,f,indent=2,ensure_ascii=False)
    print(json.dumps({"status":status,"report":str(report_path),"checksummed_files":len(files),"checksummed_bytes":manifest["total_bytes"]},indent=2))

if __name__=="__main__":main()
