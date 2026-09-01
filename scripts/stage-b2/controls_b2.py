#!/usr/bin/env python3
"""B2.7 Composition, Template, Target-Absent, Balanced, and random-label controls."""
from __future__ import annotations
import argparse,json,time,warnings
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common_b2 import B1,B156,C_GRID,HOPS,MODELS,OUT,RANDOM_HOPS,RELATIONS,REPRESENTATIVE_LAYERS,SEED,ensure_dirs,load_union,metric_dict,read_json,sha256,write_json

CTRL=OUT/"controls";warnings.filterwarnings("ignore",category=ConvergenceWarning)
SPECS={"composition":{"column":"composition_split","hops":[2,3,4,5],"weight":"balanced"},
       "template":{"column":"template_split","hops":HOPS,"weight":"balanced"}}

def predict(x,coef,intercept,classes):return classes[np.argmax(x@coef.T+intercept,axis=1)]
def fit(matrix,union,key,analysis,hop,layer,random_labels=False):
    spec={"column":"primary_split","weight":"balanced"} if random_labels else SPECS[analysis];pos={s:np.flatnonzero((union[spec["column"]].eq(s)&union.computed_hop.eq(hop)).to_numpy()) for s in ["train","val","test"]}
    xs={s:np.asarray(matrix[p],np.float32) for s,p in pos.items()};ys={s:union.iloc[p].canonical_relation.to_numpy(object) for s,p in pos.items()};scaler=StandardScaler().fit(xs["train"]);xs={s:scaler.transform(x).astype(np.float32) for s,x in xs.items()}
    train_y=ys["train"].copy()
    if random_labels:
        rng=np.random.default_rng(SEED+layer*9176+hop*1009);train_y=train_y[rng.permutation(len(train_y))]
    clf=LogisticRegression(C=C_GRID[0],solver="lbfgs",multi_class="multinomial",max_iter=400,tol=1e-4,class_weight=spec["weight"],warm_start=True,random_state=SEED);best=None;trace=[];started=time.perf_counter()
    for c in C_GRID:
        clf.C=c;fs=time.perf_counter();clf.fit(xs["train"],train_y);vp=clf.predict(xs["val"]);vm=metric_dict(ys["val"],vp);warned=bool(np.max(clf.n_iter_)>=400)
        trace.append({"model":key,"analysis":analysis,"random_train_labels":random_labels,"layer":layer,"hop":hop,"C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"n_iter_max":int(np.max(clf.n_iter_)),"convergence_warning":warned,"fit_seconds":time.perf_counter()-fs,"test_accessed_for_selection":False})
        candidate=(vm["macro_f1"],-float(c),clf.coef_.astype(np.float32,copy=True),clf.intercept_.astype(np.float32,copy=True),clf.classes_.astype(object,copy=True),vm,warned)
        if best is None or candidate[:2]>best[:2]:best=candidate
    _,neg_c,coef,intercept,classes,vm,warned=best;c=-neg_c;tp=predict(xs["test"],coef,intercept,classes);tm=metric_dict(ys["test"],tp)
    result={"model":key,"analysis":analysis,"random_train_labels":random_labels,"layer":layer,"hop":hop,"n_train":len(train_y),"n_val":len(ys["val"]),"n_test":len(ys["test"]),"selected_C":c,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"test_macro_f1":tm["macro_f1"],"test_accuracy":tm["accuracy"],"selected_fit_convergence_warning":bool(warned),"class_weight":"balanced","max_iter":400,"task_seconds":time.perf_counter()-started,"test_used_for_selection":False}
    pred=pd.DataFrame({"model":key,"analysis":analysis,"random_train_labels":random_labels,"layer":layer,"hop":hop,"row_id":union.iloc[pos["test"]].row_id.astype(int).to_numpy(),"story_group_id":union.iloc[pos["test"]].story_group_id.astype(str).to_numpy(),"true_relation":ys["test"],"predicted_relation":tp})
    return result,trace,pred

def cpdir(key):return CTRL/"checkpoints"/key
def complete(key,layer):
    p=cpdir(key)/f"layer_{layer:02d}_complete.json";return p.is_file() and read_json(p).get("status")=="PASS" and read_json(p).get("diagnostic")=="n_iter_at_limit_v1"
def run_layer(key,layer):
    ensure_dirs();cp=cpdir(key);cp.mkdir(parents=True,exist_ok=True)
    if complete(key,layer):print(json.dumps({"model":key,"layer":layer,"status":"SKIP_COMPLETE"}));return
    union=load_union();act=MODELS[key]["activation_path"];matrix=np.load(act/f"layer_{layer:02d}.npy",mmap_mode="r");tasks=[(a,h,False) for a,s in SPECS.items() for h in s["hops"]]
    if layer in REPRESENTATIVE_LAYERS:tasks += [("random_label",h,True) for h in RANDOM_HOPS]
    started=time.perf_counter()
    with threadpool_limits(limits=1):outputs=Parallel(n_jobs=8,prefer="threads")(delayed(fit)(matrix,union,key,a,h,layer,r) for a,h,r in tasks)
    results=[];traces=[];preds=[]
    for result,trace,pred in outputs:results.append(result);traces+=trace;preds.append(pred)
    pd.DataFrame(results).to_csv(cp/f"layer_{layer:02d}_results.csv",index=False);pd.DataFrame(traces).to_csv(cp/f"layer_{layer:02d}_trace.csv",index=False);pd.concat(preds,ignore_index=True).to_csv(cp/f"layer_{layer:02d}_predictions.csv",index=False)
    write_json(cp/f"layer_{layer:02d}_complete.json",{"status":"PASS","model":key,"layer":layer,"tasks":len(tasks),"seconds":time.perf_counter()-started,"diagnostic":"n_iter_at_limit_v1"});print(json.dumps({"model":key,"layer":layer,"status":"PASS","tasks":len(tasks),"seconds":time.perf_counter()-started}),flush=True)

def sensitivity(key,results,union):
    records=[];act=MODELS[key]["activation_path"]
    for row in results[results.selected_fit_convergence_warning.astype(bool)].itertuples(index=False):
        random_labels=bool(row.random_train_labels);spec={"column":"primary_split","weight":"balanced"} if random_labels else SPECS[row.analysis];matrix=np.load(act/f"layer_{int(row.layer):02d}.npy",mmap_mode="r");train=np.flatnonzero((union[spec["column"]].eq("train")&union.computed_hop.eq(int(row.hop))).to_numpy());test=np.flatnonzero((union[spec["column"]].eq("test")&union.computed_hop.eq(int(row.hop))).to_numpy());xt=np.asarray(matrix[train],np.float32);xx=np.asarray(matrix[test],np.float32);yt=union.iloc[train].canonical_relation.to_numpy(object);yx=union.iloc[test].canonical_relation.to_numpy(object);scaler=StandardScaler().fit(xt);xt=scaler.transform(xt);xx=scaler.transform(xx)
        if random_labels:
            rng=np.random.default_rng(SEED+int(row.layer)*9176+int(row.hop)*1009);yt=yt[rng.permutation(len(yt))]
        for max_iter in [2000,5000]:
            clf=LogisticRegression(C=float(row.selected_C),solver="lbfgs",multi_class="multinomial",max_iter=max_iter,tol=1e-4,class_weight="balanced",random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:warnings.simplefilter("always",ConvergenceWarning);clf.fit(xt,yt)
            warned=any(issubclass(w.category,ConvergenceWarning) for w in caught);score=metric_dict(yx,clf.predict(xx))
            if not warned:break
        delta=score["macro_f1"]-float(row.test_macro_f1);records.append({"model":key,"analysis":row.analysis,"random_train_labels":random_labels,"layer":int(row.layer),"hop":int(row.hop),"selected_C":float(row.selected_C),"sensitivity_max_iter":max_iter,"sensitivity_warning":warned,"primary_macro_f1":float(row.test_macro_f1),"sensitivity_macro_f1":score["macro_f1"],"delta_macro_f1":delta,"abs_delta_gt_0_02":abs(delta)>0.02})
    return pd.DataFrame(records,columns=["model","analysis","random_train_labels","layer","hop","selected_C","sensitivity_max_iter","sensitivity_warning","primary_macro_f1","sensitivity_macro_f1","delta_macro_f1","abs_delta_gt_0_02"])

def consolidate_model(key):
    cp=cpdir(key)
    if any(not complete(key,l) for l in range(32)):raise RuntimeError("incomplete")
    results=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_results.csv") for l in range(32)],ignore_index=True);traces=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_trace.csv") for l in range(32)],ignore_index=True);preds=pd.concat([pd.read_csv(cp/f"layer_{l:02d}_predictions.csv") for l in range(32)],ignore_index=True)
    for analysis,expected in [("composition",128),("template",160),("random_label",15)]:
        part=results[results.analysis.eq(analysis)].sort_values(["layer","hop"])
        if len(part)!=expected:raise RuntimeError(f"{analysis} {len(part)}")
        outdir=CTRL/("random_label" if analysis=="random_label" else analysis);part.to_csv(outdir/f"{key}.csv",index=False);preds[preds.analysis.eq(analysis)].to_csv(outdir/f"{key}_predictions.csv",index=False)
    traces.to_csv(CTRL/f"{key}_c_selection_trace.csv",index=False);sens=sensitivity(key,results,load_union());sens.to_csv(CTRL/f"{key}_convergence_sensitivity.csv",index=False)
    gate={"status":"PASS_WITH_LIMITATION" if (len(sens) and sens.abs_delta_gt_0_02.any()) else "PASS","model":key,"composition_rows":128,"template_rows":160,"random_rows":15,"selected_warnings":int(results.selected_fit_convergence_warning.sum()),"sensitivity_rows":len(sens),"sensitivity_still_warning":int(sens.sensitivity_warning.sum()) if len(sens) else 0,"abs_delta_gt_0_02":int(sens.abs_delta_gt_0_02.sum()) if len(sens) else 0,"test_selection_leakage":bool(traces.test_accessed_for_selection.astype(bool).any())}
    write_json(CTRL/f"{key}_gate.json",gate);print(json.dumps(gate,indent=2))

def target_absent(key):
    union=load_union();ids=set(union.loc[union.target_absent_split.eq("test"),"row_id"].astype(int));pred=pd.read_csv(OUT/"within_hop"/f"{key}_primary_predictions.csv");normal=pd.read_csv(OUT/"within_hop"/f"{key}_primary.csv");rows=[]
    for layer in range(32):
      for hop in [2,3,4,5]:
        part=pred[(pred.layer==layer)&(pred.hop==hop)&pred.row_id.isin(ids)];score=metric_dict(part.true_relation,part.predicted_relation);iid=float(normal[(normal.layer==layer)&(normal.hop==hop)].test_macro_f1.iloc[0]);rows.append({"model":key,"layer":layer,"hop":hop,"n_test":len(part),"macro_f1":score["macro_f1"],"accuracy":score["accuracy"],"derived_all_macro_f1":iid,"delta_macro_f1_target_absent_minus_all":score["macro_f1"]-iid,"probe_source":"Primary trained probe; no target-absent retraining"})
    pd.DataFrame(rows).to_csv(CTRL/"target_absent"/f"{key}.csv",index=False)

def all_models():
    # Reuse all exact Base controls.
    base_sources={"composition":B1/"composition_heldout_probe.csv","template":B1/"template_test_probe.csv","target_absent":B1/"target_absent_eval.csv","random_label":B1/"random_label_control.csv"}
    for name,path in base_sources.items():pd.read_csv(path).assign(model="base").to_csv(CTRL/name/"base.csv",index=False)
    # Balanced normal and transfer controls were completed in B2.5/B2.6.
    for key in MODELS:
        pd.read_csv(OUT/"within_hop"/f"{key}_balanced.csv").to_csv(CTRL/"balanced"/f"{key}_within.csv",index=False)
        src=OUT/"cross_hop"/("base_balanced.csv" if key=="base" else f"{key}_balanced.csv");pd.read_csv(src).to_csv(CTRL/"balanced"/f"{key}_h1_transfer.csv",index=False)
        if key!="base":target_absent(key)
    write_json(CTRL/"base_reuse.json",{"status":"PASS","sources":{k:{"path":str(v),"sha256":sha256(v)} for k,v in base_sources.items()}})
    # Random control gate against matched normal Primary probes.
    random_rows=[]
    for key in MODELS:
        rnd=pd.read_csv(CTRL/"random_label"/f"{key}.csv");normal=pd.read_csv(OUT/"within_hop"/f"{key}_primary.csv");matched=rnd.merge(normal[["layer","hop","test_accuracy","test_macro_f1"]],on=["layer","hop"],suffixes=("_random","_normal"));random_rows.append({"model":key,"mean_accuracy":rnd.test_accuracy.mean(),"mean_macro_f1":rnd.test_macro_f1.mean(),"matched_normal_macro_f1":matched.test_macro_f1_normal.mean(),"macro_f1_drop":matched.test_macro_f1_normal.mean()-rnd.test_macro_f1.mean(),"chance":0.125})
    random_summary=pd.DataFrame(random_rows);random_summary.to_csv(CTRL/"random_label"/"summary.csv",index=False)
    checks={"all_new_model_gates_pass":all(read_json(CTRL/f"{k}_gate.json")["status"] in ["PASS","PASS_WITH_LIMITATION"] for k in ["sft","dpo","final"]),"all_random_accuracy_near_chance":bool((random_summary.mean_accuracy.sub(.125).abs()<=.04).all()),"all_random_below_matched_normal":bool((random_summary.macro_f1_drop>.02).all()),"all_target_absent_128_rows":all(len(pd.read_csv(CTRL/"target_absent"/f"{k}.csv"))==128 for k in MODELS),"all_composition_128_rows":all(len(pd.read_csv(CTRL/"composition"/f"{k}.csv"))==128 for k in MODELS),"all_template_160_rows":all(len(pd.read_csv(CTRL/"template"/f"{k}.csv"))==160 for k in MODELS)}
    status="PASS_WITH_LIMITATION" if any(read_json(CTRL/f"{k}_gate.json")["status"]=="PASS_WITH_LIMITATION" for k in ["sft","dpo","final"]) else ("PASS" if all(checks.values()) else "FAIL")
    write_json(CTRL/"controls_gate.json",{"status":status,"checks":checks,"random_summary":random_rows});print(json.dumps({"status":status,"checks":checks,"random":random_rows},indent=2))

def main():
    ensure_dirs();p=argparse.ArgumentParser();p.add_argument("command",choices=["layer","consolidate-model","all-models"]);p.add_argument("--model",choices=["sft","dpo","final"]);p.add_argument("--layer",type=int);a=p.parse_args()
    if a.command=="layer":run_layer(a.model,a.layer)
    elif a.command=="consolidate-model":consolidate_model(a.model)
    else:all_models()
if __name__=="__main__":main()
