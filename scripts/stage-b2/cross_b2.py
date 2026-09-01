#!/usr/bin/env python3
"""B2.6 within-checkpoint h1 and derived-pooled cross-hop transfer."""

from __future__ import annotations

import argparse,json,time,warnings
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common_b2 import B156,C_GRID,HOPS,MODELS,OUT,RELATIONS,SEED,STAGE_PAIRS,ensure_dirs,load_union,metric_dict,read_json,sha256,write_json

CDIR=OUT/"cross_hop";warnings.filterwarnings("ignore",category=ConvergenceWarning)
TASKS={"primary_h1":{"split":"primary_split","source":[1],"class_weight":"balanced"},
       "primary_derived_pooled":{"split":"primary_split","source":[2,3,4,5],"class_weight":"balanced"},
       "balanced_h1":{"split":"balanced_split","source":[1],"class_weight":None}}

def pred(x,coef,intercept,classes):return classes[np.argmax(x@coef.T+intercept,axis=1)]

def fit_task(matrix,union,key,analysis,layer):
    spec=TASKS[analysis];train=np.flatnonzero((union[spec["split"]].eq("train")&union.computed_hop.isin(spec["source"])).to_numpy());val=np.flatnonzero((union[spec["split"]].eq("val")&union.computed_hop.isin(spec["source"])).to_numpy())
    xt=np.asarray(matrix[train],np.float32);xv=np.asarray(matrix[val],np.float32);yt=union.iloc[train].canonical_relation.to_numpy(object);yv=union.iloc[val].canonical_relation.to_numpy(object)
    scaler=StandardScaler().fit(xt);xt=scaler.transform(xt).astype(np.float32);xv=scaler.transform(xv).astype(np.float32)
    clf=LogisticRegression(C=C_GRID[0],solver="lbfgs",multi_class="multinomial",max_iter=400,tol=1e-4,class_weight=spec["class_weight"],warm_start=True,random_state=SEED)
    best=None;trace=[];started=time.perf_counter()
    for c in C_GRID:
        clf.C=c;fs=time.perf_counter();clf.fit(xt,yt);vp=clf.predict(xv);vm=metric_dict(yv,vp);warned=bool(np.max(clf.n_iter_)>=400)
        trace.append({"model":key,"analysis":analysis,"layer":layer,"source_hops":"+".join(map(str,spec["source"])),"C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"n_iter_max":int(np.max(clf.n_iter_)),"convergence_warning":warned,"fit_seconds":time.perf_counter()-fs,"test_accessed_for_selection":False})
        cand=(vm["macro_f1"],-float(c),clf.coef_.astype(np.float32,copy=True),clf.intercept_.astype(np.float32,copy=True),clf.classes_.astype(object,copy=True),vm,warned)
        if best is None or cand[:2]>best[:2]:best=cand
    _,neg_c,coef,intercept,classes,vm,warned=best;c=-neg_c;results=[];predictions=[]
    for hop in HOPS:
        pos=np.flatnonzero((union[spec["split"]].eq("test")&union.computed_hop.eq(hop)).to_numpy());x=np.asarray(matrix[pos],np.float32);x=scaler.transform(x).astype(np.float32);y=union.iloc[pos].canonical_relation.to_numpy(object);p=pred(x,coef,intercept,classes);score=metric_dict(y,p)
        results.append({"model":key,"analysis":analysis,"layer":layer,"source_hops":"+".join(map(str,spec["source"])),"target_hop":hop,"n_train":len(yt),"n_val":len(yv),"n_test":len(y),"hidden_size":matrix.shape[1],"selected_C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"test_macro_f1":score["macro_f1"],"test_accuracy":score["accuracy"],"selected_fit_convergence_warning":bool(warned),"scaler_fit_population":f"{spec['split']}:train:hops={spec['source']}","C_selection_population":f"{spec['split']}:val:hops={spec['source']}","target_test_used_for_fitting_or_selection":False,"class_weight":"balanced" if spec["class_weight"] else "none","task_seconds":time.perf_counter()-started})
        predictions.append(pd.DataFrame({"model":key,"analysis":analysis,"layer":layer,"target_hop":hop,"row_id":union.iloc[pos].row_id.astype(int).to_numpy(),"story_group_id":union.iloc[pos].story_group_id.astype(str).to_numpy(),"true_relation":y,"predicted_relation":p}))
    model={"coef":coef,"intercept":intercept,"classes":np.asarray(classes,str),"scaler_mean":scaler.mean_.astype(np.float32),"scaler_scale":scaler.scale_.astype(np.float32)}
    return results,trace,pd.concat(predictions,ignore_index=True),model

def cpdir(key):return CDIR/"checkpoints"/key
def complete(key,layer):
    p=cpdir(key)/f"layer_{layer:02d}_complete.json";return p.is_file() and read_json(p).get("status")=="PASS" and read_json(p).get("diagnostic")=="n_iter_at_limit_v2_blas3"

def run_layer(key,layer):
    ensure_dirs();cp=cpdir(key);cp.mkdir(parents=True,exist_ok=True)
    if complete(key,layer):print(json.dumps({"model":key,"layer":layer,"status":"SKIP_COMPLETE"}));return
    union=load_union();act=MODELS[key]["activation_path"];ids=np.load(act/"row_ids.npy")
    if not np.array_equal(ids,union.row_id.to_numpy(np.int64)):raise RuntimeError("row mismatch")
    matrix=np.load(act/f"layer_{layer:02d}.npy",mmap_mode="r");started=time.perf_counter()
    # B1.6 froze three BLAS threads for each L-BFGS fit.  Preserve that
    # computational setting while the three source-protocol tasks run together.
    with threadpool_limits(limits=3):outputs=Parallel(n_jobs=3,prefer="threads")(delayed(fit_task)(matrix,union,key,a,layer) for a in TASKS)
    results=[];traces=[];preds=[];models={}
    for (r,t,p,m),analysis in zip(outputs,TASKS):
        results+=r;traces+=t;preds.append(p)
        for name,value in m.items():models[f"{analysis}__{name}"]=value
    pd.DataFrame(results).to_csv(cp/f"layer_{layer:02d}_results.csv",index=False);pd.DataFrame(traces).to_csv(cp/f"layer_{layer:02d}_trace.csv",index=False);pd.concat(preds,ignore_index=True).to_csv(cp/f"layer_{layer:02d}_predictions.csv",index=False);np.savez_compressed(cp/f"layer_{layer:02d}_models.npz",**models)
    write_json(cp/f"layer_{layer:02d}_complete.json",{"status":"PASS","model":key,"layer":layer,"tasks":list(TASKS),"seconds":time.perf_counter()-started,"diagnostic":"n_iter_at_limit_v2_blas3"});print(json.dumps({"model":key,"layer":layer,"status":"PASS","seconds":time.perf_counter()-started}),flush=True)

def sensitivity(key,results,union):
    records=[];act=MODELS[key]["activation_path"]
    for row in results.drop_duplicates(["analysis","layer"])[lambda x:x.selected_fit_convergence_warning.astype(bool)].itertuples(index=False):
        spec=TASKS[row.analysis];matrix=np.load(act/f"layer_{int(row.layer):02d}.npy",mmap_mode="r");train=np.flatnonzero((union[spec["split"]].eq("train")&union.computed_hop.isin(spec["source"])).to_numpy());xt=np.asarray(matrix[train],np.float32);yt=union.iloc[train].canonical_relation.to_numpy(object);scaler=StandardScaler().fit(xt);xt=scaler.transform(xt).astype(np.float32)
        target_scores={}
        for max_iter in [2000,5000]:
            clf=LogisticRegression(C=float(row.selected_C),solver="lbfgs",multi_class="multinomial",max_iter=max_iter,tol=1e-4,class_weight=spec["class_weight"],random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:warnings.simplefilter("always",ConvergenceWarning);clf.fit(xt,yt)
            warned=any(issubclass(w.category,ConvergenceWarning) for w in caught);target_scores={}
            for hop in HOPS:
                pos=np.flatnonzero((union[spec["split"]].eq("test")&union.computed_hop.eq(hop)).to_numpy());x=scaler.transform(np.asarray(matrix[pos],np.float32));y=union.iloc[pos].canonical_relation.to_numpy(object);target_scores[hop]=metric_dict(y,clf.predict(x))["macro_f1"]
            if not warned:break
        primary=results[(results.analysis==row.analysis)&(results.layer==row.layer)].set_index("target_hop").test_macro_f1.to_dict();deltas=[target_scores[h]-primary[h] for h in HOPS]
        records.append({"model":key,"analysis":row.analysis,"layer":int(row.layer),"selected_C":float(row.selected_C),"sensitivity_max_iter":max_iter,"sensitivity_warning":warned,"max_abs_delta_macro_f1":max(map(abs,deltas)),"abs_delta_gt_0_02":max(map(abs,deltas))>0.02,"per_hop_delta_json":json.dumps(deltas)})
    return pd.DataFrame(records,columns=["model","analysis","layer","selected_C","sensitivity_max_iter","sensitivity_warning","max_abs_delta_macro_f1","abs_delta_gt_0_02","per_hop_delta_json"])

def consolidate_model(key):
    cp=cpdir(key)
    if any(not complete(key,l) for l in range(32)):raise RuntimeError("incomplete")
    results=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_results.csv") for l in range(32)],ignore_index=True);traces=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_trace.csv") for l in range(32)],ignore_index=True);preds=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_predictions.csv") for l in range(32)],ignore_index=True)
    for analysis,name in [("primary_h1",f"{key}.csv"),("primary_derived_pooled",f"{key}_derived_pooled.csv"),("balanced_h1",f"{key}_balanced.csv")]:
        part=results[results.analysis.eq(analysis)].sort_values(["layer","target_hop"])
        if len(part)!=160:raise RuntimeError(f"{analysis} {len(part)}")
        part.to_csv(CDIR/name,index=False)
    preds.to_csv(CDIR/f"{key}_predictions.csv",index=False);traces.to_csv(CDIR/f"{key}_c_selection_trace.csv",index=False)
    sens=sensitivity(key,results,load_union());sens.to_csv(CDIR/f"{key}_convergence_sensitivity.csv",index=False)
    gate={"status":"PASS_WITH_LIMITATION" if (len(sens) and sens.abs_delta_gt_0_02.any()) else "PASS","model":key,"rows_per_analysis":160,"selected_warnings":int(results.drop_duplicates(["analysis","layer"]).selected_fit_convergence_warning.sum()),"sensitivity_rows":len(sens),"sensitivity_still_warning":int(sens.sensitivity_warning.sum()) if len(sens) else 0,"abs_delta_gt_0_02":int(sens.abs_delta_gt_0_02.sum()) if len(sens) else 0,"test_selection_leakage":bool(traces.test_accessed_for_selection.astype(bool).any())}
    write_json(CDIR/f"{key}_gate.json",gate);print(json.dumps(gate,indent=2))

def all_models():
    sources={"base":B156/"cross_hop"/"h1_to_all_layers.csv","base_derived":B156/"cross_hop"/"derived_pooled_to_all_layers.csv","base_balanced":B156/"cross_hop"/"balanced_h1_to_all_layers.csv"}
    for name,path in sources.items():pd.read_csv(path).assign(model="base").to_csv(CDIR/f"{name}.csv" if name!="base" else CDIR/"base.csv",index=False)
    write_json(CDIR/"base_reuse.json",{"status":"PASS","refit":False,"sources":{k:{"path":str(v),"sha256":sha256(v)} for k,v in sources.items()}})
    gaps=[];summaries=[];deltas=[]
    for key in MODELS:
        transfer=pd.read_csv(CDIR/f"{key}.csv");within=pd.read_csv(OUT/"within_hop"/f"{key}_primary.csv").rename(columns={"hop":"target_hop"})
        gap=transfer.merge(within[["layer","target_hop","test_macro_f1"]].rename(columns={"test_macro_f1":"within_hop_macro_f1"}),on=["layer","target_hop"],validate="one_to_one").rename(columns={"test_macro_f1":"h1_transfer_macro_f1"})
        gap["model"]=key;gap["absolute_transfer_gap"]=gap.within_hop_macro_f1-gap.h1_transfer_macro_f1;gap["retention_ratio"]=gap.h1_transfer_macro_f1/gap.within_hop_macro_f1;gaps.append(gap)
        for hop in [2,3,4,5]:
            part=gap[gap.target_hop.eq(hop)];summaries.append({"model":key,"target_hop":hop,"late_transfer_mean":part[part.layer.between(16,31)].h1_transfer_macro_f1.mean(),"late_gap_mean":part[part.layer.between(16,31)].absolute_transfer_gap.mean(),"all_transfer_mean":part.h1_transfer_macro_f1.mean(),"all_gap_mean":part.absolute_transfer_gap.mean()})
    gapall=pd.concat(gaps,ignore_index=True);gapall.to_csv(CDIR/"transfer_gap.csv",index=False);summary=pd.DataFrame(summaries);summary.to_csv(CDIR/"summaries.csv",index=False)
    for earlier,later in STAGE_PAIRS:
        a=pd.read_csv(CDIR/f"{earlier}.csv");b=pd.read_csv(CDIR/f"{later}.csv");m=b.merge(a[["layer","target_hop","test_macro_f1"]],on=["layer","target_hop"],suffixes=("_later","_earlier"));m["earlier"]=earlier;m["later"]=later;m["comparison"]=f"{later}-{earlier}";m["delta_transfer_macro_f1"]=m.test_macro_f1_later-m.test_macro_f1_earlier;deltas.append(m[["earlier","later","comparison","layer","target_hop","test_macro_f1_earlier","test_macro_f1_later","delta_transfer_macro_f1"]])
    pd.concat(deltas,ignore_index=True).to_csv(CDIR/"stagewise_delta.csv",index=False)
    status="PASS_WITH_LIMITATION" if any(read_json(CDIR/f"{k}_gate.json")["status"]=="PASS_WITH_LIMITATION" for k in ["sft","dpo","final"]) else "PASS";write_json(CDIR/"cross_hop_gate.json",{"status":status,"models":list(MODELS),"primary_rows_per_model":160,"gap_rows":len(gapall),"summary_rows":len(summary)});print(json.dumps({"status":status,"summaries":summaries},indent=2))

def main():
    ensure_dirs();p=argparse.ArgumentParser();p.add_argument("command",choices=["layer","consolidate-model","all-models"]);p.add_argument("--model",choices=["sft","dpo","final"]);p.add_argument("--layer",type=int);a=p.parse_args()
    if a.command=="layer":run_layer(a.model,a.layer)
    elif a.command=="consolidate-model":consolidate_model(a.model)
    else:all_models()
if __name__=="__main__":main()
