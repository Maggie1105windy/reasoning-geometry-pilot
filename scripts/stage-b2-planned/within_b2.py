#!/usr/bin/env python3
"""B2.5 independently trained Primary and Balanced within-hop probes."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b2 import (
    B1, C_GRID, HIDDEN_SIZE, HOPS, MODELS, N_LAYERS, OUT, RELATIONS,
    SEED, STAGE_PAIRS, ensure_dirs, load_union, metric_dict, read_json,
    sha256, write_json,
)


WDIR=OUT/"within_hop";N_JOBS=8
# Set once before any worker thread starts.  Per-thread warning contexts are
# unsafe because Python's filter list is process-global.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
EXPERIMENTS={"primary":{"column":"primary_split","class_weight":"balanced"},
             "balanced":{"column":"balanced_split","class_weight":None}}


def predict(x,coef,intercept,classes):return classes[np.argmax(x@coef.T+intercept,axis=1)]


def fit_one(matrix,union,experiment,hop,layer):
    spec=EXPERIMENTS[experiment];positions={s:np.flatnonzero((union[spec["column"]].eq(s)&union.computed_hop.eq(hop)).to_numpy()) for s in ["train","val","test"]}
    if any(len(x)==0 for x in positions.values()):raise RuntimeError(f"empty {experiment} h{hop}")
    xs={s:np.asarray(matrix[pos],dtype=np.float32) for s,pos in positions.items()};ys={s:union.iloc[pos].canonical_relation.to_numpy(object) for s,pos in positions.items()}
    if any(set(y)!=set(RELATIONS) for y in ys.values()):raise RuntimeError(f"class coverage {experiment} h{hop}")
    scaler=StandardScaler().fit(xs["train"]);scaled={s:scaler.transform(x).astype(np.float32,copy=False) for s,x in xs.items()};del xs
    clf=LogisticRegression(C=C_GRID[0],solver="lbfgs",multi_class="multinomial",max_iter=400,tol=1e-4,class_weight=spec["class_weight"],warm_start=True,random_state=SEED)
    best=None;trace=[];started=time.perf_counter()
    for c in C_GRID:
        clf.C=c;fit_start=time.perf_counter()
        # Record the deterministic sklearn iteration-limit diagnostic instead
        # of relying on process-global warning capture in parallel threads.
        clf.fit(scaled["train"],ys["train"])
        warning=bool(np.max(clf.n_iter_) >= 400);vp=clf.predict(scaled["val"]);vm=metric_dict(ys["val"],vp)
        trace.append({"model":None,"experiment":experiment,"layer":layer,"hop":hop,"C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],
                      "n_iter_max":int(np.max(clf.n_iter_)),"convergence_warning":warning,"fit_seconds":time.perf_counter()-fit_start,"test_accessed_for_selection":False})
        candidate=(vm["macro_f1"],-float(c),clf.coef_.astype(np.float32,copy=True),clf.intercept_.astype(np.float32,copy=True),clf.classes_.astype(object,copy=True),vm,warning)
        if best is None or candidate[:2]>best[:2]:best=candidate
    _,neg_c,coef,intercept,classes,vm,selected_warning=best;c=-neg_c
    tp=predict(scaled["test"],coef,intercept,classes);tm=metric_dict(ys["test"],tp)
    result={"model":None,"experiment":experiment,"layer":layer,"hop":hop,"n_train":len(ys["train"]),"n_val":len(ys["val"]),"n_test":len(ys["test"]),
            "hidden_size":HIDDEN_SIZE,"selected_C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"test_macro_f1":tm["macro_f1"],
            "test_accuracy":tm["accuracy"],"selected_fit_convergence_warning":bool(selected_warning),"task_seconds":time.perf_counter()-started,
            "scaler_fit_split":"train_only","C_selection_split":"validation_only","test_used_for_selection":False,"class_weight":"balanced" if spec["class_weight"] else "none","primary_max_iter":400}
    pred=pd.DataFrame({"row_id":union.iloc[positions["test"]].row_id.astype(int).to_numpy(),"story_group_id":union.iloc[positions["test"]].story_group_id.astype(str).to_numpy(),
                       "experiment":experiment,"layer":layer,"hop":hop,"true_relation":ys["test"],"predicted_relation":tp})
    model={"coef":coef,"intercept":intercept,"classes":np.asarray(classes,str),"scaler_mean":scaler.mean_.astype(np.float32),"scaler_scale":scaler.scale_.astype(np.float32)}
    return result,trace,pred,model


def checkpoint_dir(key):return WDIR/"checkpoints"/key
def complete(key,layer):
    p=checkpoint_dir(key)/f"layer_{layer:02d}_complete.json"
    return p.is_file() and read_json(p).get("status")=="PASS" and read_json(p).get("convergence_diagnostic")=="n_iter_at_limit_v1"


def run_layer(key,layer):
    ensure_dirs();cp=checkpoint_dir(key);cp.mkdir(parents=True,exist_ok=True)
    if complete(key,layer):print(json.dumps({"model":key,"layer":layer,"status":"SKIP_COMPLETE"}));return
    if read_json(OUT/f"activation_integrity_{key}.json")["status"]!="PASS":raise RuntimeError("activation gate not pass")
    union=load_union();act=MODELS[key]["activation_path"];ids=np.load(act/"row_ids.npy")
    if not np.array_equal(ids,union.row_id.to_numpy(np.int64)):raise RuntimeError("row mismatch")
    matrix=np.load(act/f"layer_{layer:02d}.npy",mmap_mode="r")
    tasks=[(e,h) for e in EXPERIMENTS for h in HOPS];started=time.perf_counter()
    with threadpool_limits(limits=1):
        outputs=Parallel(n_jobs=N_JOBS,prefer="threads")(delayed(fit_one)(matrix,union,e,h,layer) for e,h in tasks)
    results=[];traces=[];preds=[];models={}
    for result,trace,pred,model in outputs:
        result["model"]=key;results.append(result)
        for row in trace:row["model"]=key
        traces.extend(trace);pred["model"]=key;preds.append(pred)
        prefix=f"{result['experiment']}_h{result['hop']}"
        for name,value in model.items():models[f"{prefix}__{name}"]=value
    pd.DataFrame(results).to_csv(cp/f"layer_{layer:02d}_results.csv",index=False);pd.DataFrame(traces).to_csv(cp/f"layer_{layer:02d}_trace.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(cp/f"layer_{layer:02d}_predictions.csv",index=False);np.savez_compressed(cp/f"layer_{layer:02d}_models.npz",**models)
    write_json(cp/f"layer_{layer:02d}_complete.json",{"status":"PASS","model":key,"layer":layer,"tasks":len(tasks),"seconds":time.perf_counter()-started,
                                                      "convergence_diagnostic":"n_iter_at_limit_v1"})
    print(json.dumps({"model":key,"layer":layer,"status":"PASS","tasks":len(tasks),"seconds":time.perf_counter()-started}),flush=True)


def sensitivity_refit(key,results,union):
    warnings_rows=results[results.selected_fit_convergence_warning.astype(bool)];records=[];act=MODELS[key]["activation_path"]
    for row in warnings_rows.itertuples(index=False):
        spec=EXPERIMENTS[row.experiment];matrix=np.load(act/f"layer_{int(row.layer):02d}.npy",mmap_mode="r")
        pos={s:np.flatnonzero((union[spec["column"]].eq(s)&union.computed_hop.eq(int(row.hop))).to_numpy()) for s in ["train","test"]}
        xtrain=np.asarray(matrix[pos["train"]],np.float32);xtest=np.asarray(matrix[pos["test"]],np.float32);ytrain=union.iloc[pos["train"]].canonical_relation.to_numpy(object);ytest=union.iloc[pos["test"]].canonical_relation.to_numpy(object)
        scaler=StandardScaler().fit(xtrain);xtrain=scaler.transform(xtrain).astype(np.float32);xtest=scaler.transform(xtest).astype(np.float32)
        final=None
        for max_iter in [2000,5000]:
            clf=LogisticRegression(C=float(row.selected_C),solver="lbfgs",multi_class="multinomial",max_iter=max_iter,tol=1e-4,class_weight=spec["class_weight"],random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",ConvergenceWarning);clf.fit(xtrain,ytrain)
            warned=any(issubclass(w.category,ConvergenceWarning) for w in caught);score=metric_dict(ytest,clf.predict(xtest))
            final={"model":key,"experiment":row.experiment,"layer":int(row.layer),"hop":int(row.hop),"selected_C":float(row.selected_C),"primary_max_iter":400,
                   "sensitivity_max_iter":max_iter,"sensitivity_warning":warned,"primary_macro_f1":float(row.test_macro_f1),"sensitivity_macro_f1":score["macro_f1"],
                   "delta_macro_f1":score["macro_f1"]-float(row.test_macro_f1),"abs_delta_gt_0_02":abs(score["macro_f1"]-float(row.test_macro_f1))>0.02,"sensitivity_accuracy":score["accuracy"]}
            if not warned:break
        records.append(final)
    return pd.DataFrame(records,columns=["model","experiment","layer","hop","selected_C","primary_max_iter","sensitivity_max_iter","sensitivity_warning","primary_macro_f1","sensitivity_macro_f1","delta_macro_f1","abs_delta_gt_0_02","sensitivity_accuracy"])


def consolidate_model(key):
    cp=checkpoint_dir(key)
    if any(not complete(key,l) for l in range(32)):raise RuntimeError(f"incomplete {key}")
    results=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_results.csv") for l in range(32)],ignore_index=True)
    traces=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_trace.csv") for l in range(32)],ignore_index=True)
    preds=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_predictions.csv") for l in range(32)],ignore_index=True)
    for exp in EXPERIMENTS:
        part=results[results.experiment.eq(exp)].sort_values(["layer","hop"])
        if len(part)!=160:raise RuntimeError(f"{key} {exp} {len(part)}")
        part.to_csv(WDIR/f"{key}_{exp}.csv",index=False)
        preds[preds.experiment.eq(exp)].sort_values(["layer","hop","row_id"]).to_csv(WDIR/f"{key}_{exp}_predictions.csv",index=False)
    traces.to_csv(WDIR/f"{key}_c_selection_trace.csv",index=False)
    sensitivity=sensitivity_refit(key,results,load_union());sensitivity.to_csv(WDIR/f"{key}_convergence_sensitivity.csv",index=False)
    gate={"status":"PASS_WITH_LIMITATION" if (not sensitivity.empty and sensitivity.abs_delta_gt_0_02.any()) else "PASS","model":key,
          "primary_rows":160,"balanced_rows":160,"selected_warnings":int(results.selected_fit_convergence_warning.sum()),
          "sensitivity_rows":len(sensitivity),"sensitivity_still_warning":int(sensitivity.sensitivity_warning.sum()) if len(sensitivity) else 0,
          "abs_delta_gt_0_02":int(sensitivity.abs_delta_gt_0_02.sum()) if len(sensitivity) else 0,"test_selection_leakage":bool(traces.test_accessed_for_selection.astype(bool).any())}
    write_json(WDIR/f"{key}_gate.json",gate);print(json.dumps(gate,indent=2))


def all_models():
    # Base is reused byte-for-byte; no re-extraction or re-fitting.
    base_primary=pd.read_csv(B1/"primary_probe.csv");base_balanced=pd.read_csv(B1/"balanced_probe.csv")
    base_primary.insert(0,"model","base");base_balanced.insert(0,"model","base")
    base_primary.to_csv(WDIR/"base_primary.csv",index=False);base_balanced.to_csv(WDIR/"base_balanced.csv",index=False)
    base_pred=pd.read_csv(B1/"primary_test_predictions.csv");base_pred.insert(0,"model","base");base_pred.insert(1,"experiment","primary");base_pred.to_csv(WDIR/"base_primary_predictions.csv",index=False)
    write_json(WDIR/"base_reuse.json",{"status":"PASS","refit":False,"primary_source":str(B1/"primary_probe.csv"),"primary_sha256":sha256(B1/"primary_probe.csv"),
                                       "balanced_source":str(B1/"balanced_probe.csv"),"balanced_sha256":sha256(B1/"balanced_probe.csv"),"predictions_source":str(B1/"primary_test_predictions.csv"),"predictions_sha256":sha256(B1/"primary_test_predictions.csv")})
    for key in ["sft","dpo","final"]:
        if read_json(WDIR/f"{key}_gate.json")["status"] not in ["PASS","PASS_WITH_LIMITATION"]:raise RuntimeError(key)
    summaries=[]
    for key in MODELS:
      for exp in EXPERIMENTS:
        frame=pd.read_csv(WDIR/f"{key}_{exp}.csv")
        for hop in HOPS:
            part=frame[frame.hop.eq(hop)];peak=part.loc[part.test_macro_f1.idxmax()]
            summaries.append({"model":key,"experiment":exp,"hop":hop,"all_layer_mean":part.test_macro_f1.mean(),"late_layer_mean":part[part.layer.between(16,31)].test_macro_f1.mean(),
                              "peak_macro_f1":peak.test_macro_f1,"peak_layer":int(peak.layer)})
    summary=pd.DataFrame(summaries);summary.to_csv(WDIR/"summaries.csv",index=False)
    deltas=[]
    for exp in EXPERIMENTS:
      frames={key:pd.read_csv(WDIR/f"{key}_{exp}.csv") for key in MODELS}
      for earlier,later in STAGE_PAIRS:
        m=frames[later].merge(frames[earlier][["layer","hop","test_macro_f1"]],on=["layer","hop"],suffixes=("_later","_earlier"))
        m["experiment"]=exp;m["earlier"]=earlier;m["later"]=later;m["comparison"]=f"{later}-{earlier}";m["delta_macro_f1"]=m.test_macro_f1_later-m.test_macro_f1_earlier
        deltas.append(m[["experiment","earlier","later","comparison","layer","hop","test_macro_f1_earlier","test_macro_f1_later","delta_macro_f1"]])
    delta=pd.concat(deltas,ignore_index=True);delta.to_csv(WDIR/"layerwise_delta.csv",index=False)
    write_json(WDIR/"within_hop_gate.json",{"status":"PASS_WITH_LIMITATION" if any(read_json(WDIR/f"{k}_gate.json")["status"]=="PASS_WITH_LIMITATION" for k in ["sft","dpo","final"]) else "PASS",
                                             "models":list(MODELS),"primary_rows_per_model":160,"balanced_rows_per_model":160,"summary_rows":len(summary),"layerwise_delta_rows":len(delta)})
    print(json.dumps({"status":read_json(WDIR/"within_hop_gate.json")["status"],"summary":summary[summary.experiment.eq("primary")].to_dict("records")},indent=2))


def main():
    ensure_dirs();p=argparse.ArgumentParser();p.add_argument("command",choices=["layer","consolidate-model","all-models"]);p.add_argument("--model",choices=["sft","dpo","final"]);p.add_argument("--layer",type=int);a=p.parse_args()
    if a.command=="layer":
        if a.model is None or a.layer is None:raise SystemExit("--model --layer required")
        run_layer(a.model,a.layer)
    elif a.command=="consolidate-model":
        if a.model is None:raise SystemExit("--model required")
        consolidate_model(a.model)
    else:all_models()


if __name__=="__main__":main()
