#!/usr/bin/env python3
"""Independent Stage B2.1 validator; deliberately imports no main-run helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer


ROOT = Path("/home/ai/chenchen/projects/reasoning-geometry-pilot")
OUT = ROOT / "artifacts" / "stage_b2_1"
RELATIONS = ["ABOVE","BELOW","LEFT","RIGHT","UPPER_LEFT","UPPER_RIGHT","LOWER_LEFT","LOWER_RIGHT"]
HOPS = [1,2,3,4,5]
MODELS = ["base","sft","dpo","final"]
MODEL_PATHS = {
    "base": ROOT/"models"/"Meta-Llama-3.1-8B",
    "sft": ROOT/"models"/"Llama-3.1-Tulu-3-8B-SFT",
    "dpo": ROOT/"models"/"Llama-3.1-Tulu-3-8B-DPO",
    "final": ROOT/"models"/"Llama-3.1-Tulu-3-8B",
}


def digest(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def stable(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def raw_prompt(row):
    story=json.loads(row.story)
    return "Story:\n"+"\n".join(story)+"\n\nQuestion:\n"+str(row.question).strip()


def metric_check(score_path: Path, metric_path: Path, protocols: list[str], primary: pd.DataFrame, errors: list):
    scores=pd.read_csv(score_path)
    if "model" not in scores: errors.append(f"{score_path}: missing model"); return
    if not np.isfinite(scores[["sum_log_probability","mean_log_probability"]].to_numpy()).all(): errors.append(f"{score_path}: non-finite")
    if not scores.groupby(["model","row_id"]).size().eq(8).all(): errors.append(f"{score_path}: not 8 candidates/row")
    stored=pd.read_csv(metric_path)
    for model in scores.model.unique():
        frame=scores[scores.model==model]
        for protocol in protocols:
            col="mean_log_probability" if protocol.endswith("Mean") else "sum_log_probability"
            idx=frame.groupby("row_id")[col].idxmax()
            pred=frame.loc[idx,["row_id","candidate_relation"]].rename(columns={"candidate_relation":"prediction"})
            merged=primary.merge(pred,on="row_id",validate="one_to_one")
            for hop in HOPS:
                part=merged[merged.computed_hop==hop]
                acc=accuracy_score(part.canonical_relation,part.prediction)
                mf1=f1_score(part.canonical_relation,part.prediction,labels=RELATIONS,average="macro",zero_division=0)
                got=stored[(stored.model==model)&(stored.protocol==protocol)&(stored.hop==hop)]
                if len(got)!=1 or abs(float(got.iloc[0].accuracy)-acc)>1e-15 or abs(float(got.iloc[0].macro_f1)-mf1)>1e-15:
                    errors.append(f"metric mismatch {model}/{protocol}/h{hop}")


def main():
    errors=[]; warnings=[]; checks={}
    union=pd.read_csv(ROOT/"artifacts"/"stage_b1"/"union_manifest.csv")
    primary=union[union.primary_split=="test"].sort_values("row_id")[["row_id","story_group_id","computed_hop","canonical_relation","story","question","prompt"]].reset_index(drop=True)
    b02=pd.read_csv(ROOT/"artifacts"/"stage_b0_2"/"primary_test.csv").sort_values("row_id")
    checks["primary_rows_2063"]=len(primary)==2063
    checks["hop_counts_exact"]=primary.groupby("computed_hop").size().to_dict()=={1:712,2:505,3:422,4:282,5:142}
    checks["row_ids_exact_b0_vs_b1"]=primary.row_id.tolist()==b02.row_id.astype(int).tolist()
    checks["row_ids_unique"]=primary.row_id.is_unique
    checks["relations_exact"]=set(primary.canonical_relation)==set(RELATIONS)
    checks["story_groups_present"]=primary.story_group_id.notna().all()
    checks["raw_prompts_exact"]=all(raw_prompt(row)==row.prompt for row in primary.itertuples(index=False))

    protected=json.load(open(OUT/"audit"/"protected_checksums.json",encoding="utf-8"))
    changed=[]
    for rel, item in protected["files"].items():
        path=ROOT/rel
        if not path.is_file() or digest(path)!=item["sha256"]: changed.append(rel)
    checks["protected_B0_to_B2_unchanged"]=not changed
    if changed: errors.append(f"protected files changed: {changed[:10]}")

    protocol_paths=[OUT/"raw_reproduction"/"protocol.json",OUT/"native_chat"/"protocol.json",OUT/"scoring_sensitivity"/"protocol.json",OUT/"equal_length_verbalizer"/"protocol.json"]
    protocol_ok=True
    for path in protocol_paths:
        p=json.load(open(path,encoding="utf-8")); expected=p.pop("protocol_sha256")
        protocol_ok &= stable(p)==expected and p["seed"]==20260910 and p["bootstrap_resamples"]==2000
    checks["protocol_hashes_and_seed_valid"]=protocol_ok

    boundary=pd.read_csv(OUT/"native_chat"/"token_boundary_audit.csv")
    checks["native_boundary_audit_at_least_120_each"]=all(len(boundary[boundary.model==m])>=120 for m in ["sft","dpo","final"])
    checks["native_boundary_matches_standalone"]=boundary.matches_standalone.astype(bool).all()
    labels=json.load(open(OUT/"equal_length_verbalizer"/"candidate_selection_audit.json",encoding="utf-8"))
    label_tokens=pd.read_csv(OUT/"equal_length_verbalizer"/"label_tokenization.csv")
    checks["equal_length_gate"]=labels["status"]=="PASS" and label_tokens.raw_token_count.nunique()==1 and label_tokens.raw_token_count.iloc[0]==1 and label_tokens.dropna(subset=["native_token_count"]).native_token_count.nunique()==1
    checks["candidate_selection_score_independent"]=labels["selection_used_no_model_scores_or_gold_conditioned_performance"] is True

    # Independently render one deterministic prompt/checkpoint and confirm the audited assistant boundary.
    sample=primary.groupby("computed_hop",sort=True).head(1).iloc[0]
    words=["above","below","left","right","upper-left","upper-right","lower-left","lower-right"]
    user=raw_prompt(sample)+"\n\nAnswer with exactly one of:\n"+"\n".join(words)
    render_ok=True
    for model in ["sft","dpo","final"]:
        tok=AutoTokenizer.from_pretrained(MODEL_PATHS[model],local_files_only=True,use_fast=True)
        prompt=list(tok.apply_chat_template([{"role":"user","content":user}],tokenize=True,add_generation_prompt=True))
        full=list(tok.apply_chat_template([{"role":"user","content":user},{"role":"assistant","content":" above"}],tokenize=True,add_generation_prompt=False))
        suffix=full[len(prompt):]
        while suffix and suffix[-1] in set(tok.all_special_ids): suffix.pop()
        render_ok &= full[:len(prompt)]==prompt and suffix==tok(" above",add_special_tokens=False).input_ids
    checks["independent_chat_render_boundary"]=render_ok

    expected_files=[
        OUT/"raw_reproduction"/"reproduction_check.json",OUT/"native_chat"/"row_scores_sum.csv",OUT/"native_chat"/"row_scores_mean.csv",
        OUT/"native_chat"/"metrics_sum.csv",OUT/"native_chat"/"metrics_mean.csv",OUT/"scoring_sensitivity"/"metrics_sum_vs_mean.csv",
        OUT/"scoring_sensitivity"/"cardinal_diagonal_metrics.csv",OUT/"equal_length_verbalizer"/"raw_row_scores.csv",
        OUT/"equal_length_verbalizer"/"native_row_scores.csv",OUT/"statistics"/"paired_bootstrap.csv",
        OUT/"statistics"/"core_summary.csv",OUT/"equal_length_verbalizer"/"prompt_examples.md",
        OUT/"STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md",
    ]+[next(iter((OUT/"figures").glob(f"figure_{i}_*.png")),OUT/"figures"/f"MISSING_{i}") for i in range(1,9)]
    missing=[str(p) for p in expected_files if not p.is_file()]
    checks["all_expected_outputs_present"]=not missing
    if missing: errors.append(f"missing outputs: {missing}")

    metric_check(OUT/"native_chat"/"row_scores_sum.csv",OUT/"statistics"/"all_hop_metrics.csv",["Native-Sum","Native-Mean"],primary,errors)
    metric_check(OUT/"equal_length_verbalizer"/"raw_row_scores.csv",OUT/"statistics"/"all_hop_metrics.csv",["EqualLength-Raw"],primary,errors)
    metric_check(OUT/"equal_length_verbalizer"/"native_row_scores.csv",OUT/"statistics"/"all_hop_metrics.csv",["EqualLength-Native"],primary,errors)
    checks["independent_argmax_and_metric_reconstruction"]=not any("metric mismatch" in e for e in errors)

    run_meta=[json.load(open(OUT/"audit"/f"run_metadata_{m}.json",encoding="utf-8")) for m in MODELS]
    checks["all_model_runs_pass"]=all(x["status"]=="PASS" for x in run_meta)
    checks["no_truncation"]=all(x["no_truncation"] is True for x in run_meta)
    figures=sorted((OUT/"figures").glob("figure_*.png"))
    dimensions={p.name:Image.open(p).size for p in figures}
    checks["eight_nonempty_figures"]=len(figures)==8 and all(w>=1200 and h>=700 for w,h in dimensions.values())
    checks["gate_A_pass"]=json.load(open(OUT/"raw_reproduction"/"reproduction_check.json"))["status"]=="PASS"

    checks={key:bool(value) for key,value in checks.items()}
    if not all(checks.values()):
        errors += [f"failed check: {k}" for k,v in checks.items() if not bool(v)]
    audit=json.load(open(OUT/"audit"/"input_audit.json",encoding="utf-8"))
    if audit["environment"]["transformers"]!="4.44.2":
        warnings.append(f"New diagnostic protocols ran with transformers {audit['environment']['transformers']}; frozen B2 metadata used 4.44.2. Gate A exact reproduction mitigates but does not eliminate interface-version risk.")
    status="FAIL" if errors else ("PASS_WITH_LIMITATION" if warnings else "PASS")
    summary={"status":status,"checks":checks,"errors":errors,"limitations":warnings,"figure_dimensions":dimensions,"protected_changed_files":changed}
    (OUT/"validation").mkdir(parents=True,exist_ok=True)
    (OUT/"validation"/"validation_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage B2.1 independent validation","",f"Final status: **{status}**","","## Checks",""]+[f"- {'PASS' if bool(v) else 'FAIL'}: `{k}`" for k,v in checks.items()]
    if warnings: lines += ["","## Limitations",""]+[f"- {x}" for x in warnings]
    if errors: lines += ["","## Errors",""]+[f"- {x}" for x in errors]
    (OUT/"validation"/"validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if errors: raise SystemExit("Independent validation FAIL: "+"; ".join(errors))

    core=pd.read_csv(OUT/"statistics"/"core_summary.csv")
    decisions=json.load(open(OUT/"statistics"/"fitness_decision.json",encoding="utf-8"))
    def row(model,protocol): return core[(core.model==model)&(core.Protocol==protocol)].iloc[0]
    best=core[core.model=="final"].sort_values("Derived mean",ascending=False).iloc[0]
    labels_text=", ".join(labels["selected_labels"])
    print(f"""# Stage B2.1 Complete

1. Raw reproduction
- status: PASS
- exact behavioral reproduction: YES (integer correct counts, every model/hop)
- rows: 2063

2. Native chat
- SFT derived mean: {row('sft','Native-Sum')['Derived mean']:.4f}
- DPO derived mean: {row('dpo','Native-Sum')['Derived mean']:.4f}
- Final derived mean: {row('final','Native-Sum')['Derived mean']:.4f}
- Final vs Raw delta: {row('final','Native-Sum')['Derived mean']-row('final','Raw-Sum')['Derived mean']:+.4f}
- status: PASS

3. Mean-token sensitivity
- Base derived mean sum: {row('base','Raw-Sum')['Derived mean']:.4f}
- Base derived mean mean-token: {row('base','Raw-Mean')['Derived mean']:.4f}
- Final derived mean sum: {row('final','Raw-Sum')['Derived mean']:.4f}
- Final derived mean mean-token: {row('final','Raw-Mean')['Derived mean']:.4f}
- status: PASS

4. Equal-length verbalizer
- candidate labels: {labels_text}
- tokens per candidate: 1
- Base derived mean: {row('base','EqualLength-Raw')['Derived mean']:.4f}
- SFT derived mean: {row('sft','EqualLength-Native')['Derived mean']:.4f}
- DPO derived mean: {row('dpo','EqualLength-Native')['Derived mean']:.4f}
- Final derived mean: {row('final','EqualLength-Native')['Derived mean']:.4f}
- status: PASS

5. Strongest task-fitness protocol
- protocol: {best.Protocol}
- Final h1: {best.h1:.4f}
- Final h2: {best.h2:.4f}
- Final h3: {best.h3:.4f}
- Final h4: {best.h4:.4f}
- Final h5: {best.h5:.4f}
- Final derived mean: {best['Derived mean']:.4f}

6. Fitness decision
- Is low raw accuracy mainly prompt-related?: {decisions['native_improves']}
- Is low raw accuracy materially token-length-related?: {decisions['token_length_affects']}
- Does Final demonstrate robust derived StepGame ability?: {decisions['final_h2_h5_robust_above_chance']}
- Is current model suitable as successful multi-hop mechanism target?: {decisions['successful_mechanism_target']}

7. Stage decision
- recommended next step: {'方案 B——保留当前 Llama/Tülu 作为 weak-reasoning control，新增一个在同一冻结协议下明显更强的模型。' if decisions['final_h2_h5_robust_above_chance']=='NO' else '按报告 Section 9 区分 frozen raw 与 native task-solving protocol。'}
- DO NOT automatically start causal intervention.

8. Report
- path: {OUT/'STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md'}

9. Validation
- status: {status}
- path: {OUT/'validation'/'validation_report.md'}
""")


if __name__=="__main__":main()
