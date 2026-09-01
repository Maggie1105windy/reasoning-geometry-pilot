#!/usr/bin/env python3
"""Frozen Stage B2 behavioral candidate scoring and paired stagewise deltas."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_b2 import (
    B156, BOOTSTRAP_RESAMPLES, CANDIDATES, HOPS, MODELS, OUT, RELATIONS,
    SEED, STAGE_PAIRS, ensure_dirs, load_union, metric_dict, read_json,
    sha256, write_json,
)


BDIR = OUT / "behavior"


def tokenizer(key):
    tok=AutoTokenizer.from_pretrained(MODELS[key]["path"],local_files_only=True,use_fast=True)
    tok.pad_token=tok.eos_token;tok.padding_side="right";return tok


def candidate_records(tok):
    return [{"relation":r,"string":CANDIDATES[r],"token_ids":tok(CANDIDATES[r],add_special_tokens=False).input_ids,
             "tokens":tok.convert_ids_to_tokens(tok(CANDIDATES[r],add_special_tokens=False).input_ids)} for r in RELATIONS]


def audit():
    frozen=read_json(OUT/"frozen_b2_protocol.json")
    prior=read_json(B156/"behavioral"/"candidate_protocol.json")
    records={key:candidate_records(tokenizer(key)) for key in MODELS}
    base={x["relation"]:x["token_ids"] for x in records["base"]}
    checks={"b2_protocol_frozen":frozen["status"]=="FROZEN_BEFORE_FIRST_TULU_MODEL_LOAD",
            "candidate_strings_exact_b1_5":prior["candidate_verbalizations"]==CANDIDATES,
            "candidate_token_ids_identical_all_models":all({x["relation"]:x["token_ids"] for x in rec}==base for rec in records.values()),
            "no_empty_candidates":all(x["token_ids"] for rec in records.values() for x in rec),
            "primary_sum_sensitivity_mean":frozen["behavior"]["primary_scoring"]=="summed conditional log-probability" and frozen["behavior"]["sensitivity_scoring"]=="mean conditional log-probability"}
    payload={"stage":"B2.4","status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"candidate_tokenization":records,
             "source_candidate_protocol":str(B156/"behavioral"/"candidate_protocol.json"),"source_sha256":sha256(B156/"behavioral"/"candidate_protocol.json")}
    write_json(BDIR/"candidate_cross_checkpoint_audit.json",payload)
    print(json.dumps(payload,indent=2))
    if payload["status"]!="PASS":raise SystemExit(2)


def score_rows(model,tok,frame,batch_size=32):
    cids={r:tok(CANDIDATES[r],add_special_tokens=False).input_ids for r in RELATIONS}
    expanded=[]
    for row in frame.itertuples(index=False):
        prompt=json.loads(row.token_ids)
        if prompt[0]!=tok.bos_token_id:raise RuntimeError(f"BOS mismatch row {row.row_id}")
        for relation in RELATIONS:
            expanded.append((int(row.row_id),relation,prompt+cids[relation],len(prompt),cids[relation]))
    output=[];torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for start in range(0,len(expanded),batch_size):
        batch=expanded[start:start+batch_size];max_len=max(len(x[2]) for x in batch)
        ids=torch.full((len(batch),max_len),tok.eos_token_id,dtype=torch.long);attention=torch.zeros_like(ids)
        for i,(_,_,seq,_,_) in enumerate(batch):ids[i,:len(seq)]=torch.tensor(seq);attention[i,:len(seq)]=1
        ids=ids.cuda(non_blocking=True);attention=attention.cuda(non_blocking=True)
        with torch.inference_mode():
            logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits.float()
            logp=torch.log_softmax(logits,dim=-1)
        for i,(row_id,relation,_seq,prompt_len,candidate) in enumerate(batch):
            values=[float(logp[i,prompt_len+j-1,token_id]) for j,token_id in enumerate(candidate)]
            output.append({"row_id":row_id,"candidate_relation":relation,"candidate_token_count":len(candidate),
                           "sum_log_probability":float(sum(values)),"mean_log_probability":float(np.mean(values)),
                           "token_log_probabilities":json.dumps(values,separators=(",",":"))})
        del ids,attention,logits,logp
        if (start//batch_size+1)%50==0 or start+batch_size>=len(expanded):
            print(json.dumps({"scored_sequences":min(start+batch_size,len(expanded)),"total_sequences":len(expanded),"elapsed_seconds":time.perf_counter()-started}),flush=True)
    return pd.DataFrame(output),{"runtime_seconds":time.perf_counter()-started,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}


def predictions(samples,scores):
    base=samples[["row_id","story_group_id","computed_hop","canonical_relation"]].copy()
    a=scores.loc[scores.groupby("row_id").sum_log_probability.idxmax(),["row_id","candidate_relation"]].rename(columns={"candidate_relation":"predicted_relation_sum"})
    b=scores.loc[scores.groupby("row_id").mean_log_probability.idxmax(),["row_id","candidate_relation"]].rename(columns={"candidate_relation":"predicted_relation_mean"})
    return base.merge(a,on="row_id",validate="one_to_one").merge(b,on="row_id",validate="one_to_one")


def bootstrap_summary(pred,key):
    rows=[]
    for dataset,flag in [("primary","in_primary_test"),("balanced","in_balanced_test")]:
        for hop in HOPS:
            part=pred[pred[flag]&pred.computed_hop.eq(hop)].reset_index(drop=True)
            groups=part.story_group_id.unique();lookup={g:np.flatnonzero(part.story_group_id.eq(g).to_numpy()) for g in groups}
            for scoring,col in [("sum","predicted_relation_sum"),("mean","predicted_relation_mean")]:
                point=metric_dict(part.canonical_relation.to_numpy(),part[col].to_numpy());vals=[]
                rng=np.random.default_rng(SEED+hop*100+len(dataset)*13+(0 if scoring=="sum" else 1))
                for _ in range(BOOTSTRAP_RESAMPLES):
                    chosen=rng.choice(groups,size=len(groups),replace=True);idx=np.concatenate([lookup[g] for g in chosen])
                    vals.append(metric_dict(part.iloc[idx].canonical_relation.to_numpy(),part.iloc[idx][col].to_numpy()))
                rows.append({"model":key,"dataset":dataset,"hop":hop,"scoring":scoring,"n_test":len(part),"n_story_groups":len(groups),
                             "accuracy":point["accuracy"],"macro_f1":point["macro_f1"],
                             "accuracy_ci95_low":float(np.quantile([x["accuracy"] for x in vals],.025)),"accuracy_ci95_high":float(np.quantile([x["accuracy"] for x in vals],.975)),
                             "macro_f1_ci95_low":float(np.quantile([x["macro_f1"] for x in vals],.025)),"macro_f1_ci95_high":float(np.quantile([x["macro_f1"] for x in vals],.975)),
                             "bootstrap_unit":"story_group_id","bootstrap_resamples":BOOTSTRAP_RESAMPLES})
    return pd.DataFrame(rows)


def score(key):
    if key not in ["sft","dpo","final"]:raise ValueError(key)
    if read_json(BDIR/"candidate_cross_checkpoint_audit.json")["status"]!="PASS":raise RuntimeError("candidate audit not PASS")
    if read_json(OUT/f"activation_integrity_{key}.json")["status"]!="PASS":raise RuntimeError("activation integrity not PASS")
    union=load_union();primary=set(union.loc[union.primary_split.eq("test"),"row_id"]);balanced=set(union.loc[union.balanced_split.eq("test"),"row_id"])
    samples=union[union.row_id.isin(primary|balanced)].sort_values("row_id").reset_index(drop=True)
    tok=tokenizer(key);started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(MODELS[key]["path"],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,attn_implementation="sdpa").cuda().eval()
    scores,meta=score_rows(model,tok,samples,32);scores.to_csv(BDIR/f"{key}_candidate_scores.csv",index=False)
    pred=predictions(samples,scores);pred["in_primary_test"]=pred.row_id.isin(primary);pred["in_balanced_test"]=pred.row_id.isin(balanced);pred.to_csv(BDIR/f"{key}_predictions.csv",index=False)
    summary=bootstrap_summary(pred,key);summary.to_csv(BDIR/f"{key}.csv",index=False)
    meta.update({"status":"PASS","model":key,"rows":len(samples),"candidate_sequences":len(scores),"batch_size":32,"total_seconds":time.perf_counter()-started})
    write_json(BDIR/f"{key}_forward_metadata.json",meta);print(json.dumps(meta,indent=2))


def copy_base():
    pred=pd.read_csv(B156/"behavioral"/"all_behavior_predictions.csv")
    pred.to_csv(BDIR/"base_predictions.csv",index=False)
    summary=bootstrap_summary(pred,"base");summary.to_csv(BDIR/"base.csv",index=False)
    # Preserve exact B1.5 source artifacts and provenance rather than rescoring Base.
    write_json(BDIR/"base_reuse.json",{"status":"PASS","base_rescored":False,"predictions_source":str(B156/"behavioral"/"all_behavior_predictions.csv"),
                                       "predictions_sha256":sha256(B156/"behavioral"/"all_behavior_predictions.csv"),
                                       "candidate_scores_source":str(B156/"behavioral"/"all_candidate_scores.csv"),
                                       "candidate_scores_sha256":sha256(B156/"behavioral"/"all_candidate_scores.csv")})


def consolidate():
    copy_base();summaries=[];preds={}
    for key in MODELS:
        path=BDIR/f"{key}.csv"
        if not path.is_file():raise RuntimeError(f"Missing {path}")
        summaries.append(pd.read_csv(path));preds[key]=pd.read_csv(BDIR/f"{key}_predictions.csv")
    all_summary=pd.concat(summaries,ignore_index=True);all_summary.to_csv(BDIR/"all_models.csv",index=False)
    delta_rows=[]
    for dataset,flag in [("primary","in_primary_test"),("balanced","in_balanced_test")]:
      for hop in HOPS:
       merged=None
       for key,frame in preds.items():
        part=frame[frame[flag]&frame.computed_hop.eq(hop)][["row_id","story_group_id","canonical_relation","predicted_relation_sum","predicted_relation_mean"]].copy()
        part=part.rename(columns={"predicted_relation_sum":f"sum_{key}","predicted_relation_mean":f"mean_{key}"})
        if merged is None:merged=part
        else:merged=merged.merge(part.drop(columns=["story_group_id","canonical_relation"]),on="row_id",validate="one_to_one")
       groups=merged.story_group_id.unique();lookup={g:np.flatnonzero(merged.story_group_id.eq(g).to_numpy()) for g in groups}
       for scoring in ["sum","mean"]:
        point={key:metric_dict(merged.canonical_relation,merged[f"{scoring}_{key}"]) for key in MODELS}
        rng=np.random.default_rng(SEED+hop*1000+len(dataset)*31+(0 if scoring=="sum" else 1));dist={pair:[] for pair in STAGE_PAIRS}
        for _ in range(BOOTSTRAP_RESAMPLES):
            chosen=rng.choice(groups,size=len(groups),replace=True);idx=np.concatenate([lookup[g] for g in chosen]);sample=merged.iloc[idx]
            vals={key:metric_dict(sample.canonical_relation,sample[f"{scoring}_{key}"]) for key in MODELS}
            for pair in STAGE_PAIRS:dist[pair].append({m:vals[pair[1]][m]-vals[pair[0]][m] for m in ["accuracy","macro_f1"]})
        for earlier,later in STAGE_PAIRS:
         for metric in ["accuracy","macro_f1"]:
            values=[x[metric] for x in dist[(earlier,later)]];delta_rows.append({"dataset":dataset,"hop":hop,"scoring":scoring,"metric":metric,
                "earlier":earlier,"later":later,"comparison":f"{later}-{earlier}","point_estimate":point[later][metric]-point[earlier][metric],
                "ci95_low":float(np.quantile(values,.025)),"ci95_high":float(np.quantile(values,.975)),"bootstrap_unit":"story_group_id",
                "bootstrap_resamples":BOOTSTRAP_RESAMPLES,"same_groups_all_models":True,"ci_excludes_zero":bool(np.quantile(values,.025)>0 or np.quantile(values,.975)<0)})
    delta=pd.DataFrame(delta_rows);delta.to_csv(BDIR/"stagewise_delta.csv",index=False);delta.to_csv(BDIR/"behavior_stagewise_delta.csv",index=False);delta.to_csv(BDIR/"paired_bootstrap.csv",index=False)
    write_json(BDIR/"behavior_gate.json",{"status":"PASS","models":list(MODELS),"summary_rows":len(all_summary),"delta_rows":len(delta),
                                          "primary_scoring":"sum","sensitivity_scoring":"mean","paired_bootstrap_complete":True})
    print(json.dumps({"status":"PASS","summary_rows":len(all_summary),"delta_rows":len(delta),
                      "primary_sum_accuracy":all_summary[(all_summary.dataset=="primary")&(all_summary.scoring=="sum")].pivot(index="model",columns="hop",values="accuracy").to_dict("index")},indent=2))


def main():
    ensure_dirs();p=argparse.ArgumentParser();p.add_argument("command",choices=["audit","score","consolidate"]);p.add_argument("--model",choices=["sft","dpo","final"]);a=p.parse_args()
    if a.command=="audit":audit()
    elif a.command=="score":
        if not a.model:raise SystemExit("--model required")
        score(a.model)
    else:consolidate()


if __name__=="__main__":main()
