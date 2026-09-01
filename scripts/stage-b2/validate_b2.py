#!/usr/bin/env python3
"""Independent count/schema/gate validation for completed Stage B2 outputs."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path("/home/ai/projects/reasoning-geometry-pilot"); OUT=ROOT/"artifacts/stage_b2"
MODELS=("base","sft","dpo","final")

def readj(p):
    with open(p,encoding="utf-8") as f:return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def check(cond,name,checks):checks[name]=bool(cond)

def validate_core():
    c={}
    for name in ["model_lineage_audit","tokenization_cross_checkpoint_audit","frozen_input_verification"]:
        check(readj(OUT/f"{name}.json")["status"]=="PASS",f"{name}_pass",c)
    for m in ["sft","dpo","final"]:
        g=readj(OUT/f"activation_integrity_{m}.json");check(g["status"]=="PASS",f"activation_{m}_pass",c)
    b=pd.read_csv(OUT/"behavior/all_models.csv");check(len(b)==80,"behavior_80_rows",c);check(set(b.model)==set(MODELS),"behavior_four_models",c)
    bd=pd.read_csv(OUT/"behavior/stagewise_delta.csv");check(len(bd)==160,"behavior_delta_160_rows",c)
    for m in MODELS:
        w=pd.read_csv(OUT/"within_hop"/f"{m}_primary.csv");check(len(w)==160,f"within_{m}_160",c);check(not w.test_used_for_selection.astype(bool).any(),f"within_{m}_no_test_leakage",c)
        x=pd.read_csv(OUT/"cross_hop"/f"{m}.csv");check(len(x)==160,f"cross_{m}_160",c);check(not x.target_test_used_for_fitting_or_selection.astype(bool).any(),f"cross_{m}_no_test_leakage",c)
    check(readj(OUT/"within_hop/within_hop_gate.json")["status"]=="PASS","within_gate_pass",c)
    check(readj(OUT/"cross_hop/cross_hop_gate.json")["status"]=="PASS","cross_gate_pass",c)
    p=pd.read_csv(OUT/"within_hop/paired_probe_bootstrap.csv");check(len(p)==160,"probe_bootstrap_160",c);check((p.n_bootstrap==1000).all(),"probe_bootstrap_1000",c)
    for m in MODELS:
        for control,n in [("balanced",160),("target_absent",128),("composition",128),("template",160)]:
            suffix="_within" if control=="balanced" else ""
            q=pd.read_csv(OUT/"controls"/control/f"{m}{suffix}.csv");check(len(q)==n,f"{control}_{m}_{n}",c)
        r=pd.read_csv(OUT/"controls/random_label"/f"{m}.csv");check(len(r)==15,f"random_{m}_15",c)
    cg=readj(OUT/"controls/controls_gate.json");check(cg["status"] in ["PASS","PASS_WITH_LIMITATION"],"controls_gate_acceptable",c)
    figs=sorted((OUT/"figures").glob("figure_*.png"));check(len(figs)==10,"ten_figures",c);check(all(x.stat().st_size>20_000 for x in figs),"figures_nontrivial_size",c)
    check(len(readj(OUT/"figures/chart_map.json"))==10,"chart_map_ten",c)
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--verify-final",action="store_true");a=ap.parse_args();checks=validate_core()
    if a.verify_final:
        report=OUT/"B2_BASE_TULU_STAGE_COMPARISON_REPORT.md";text=report.read_text(encoding="utf-8")
        check(report.is_file() and report.stat().st_size>5000,"report_present",checks)
        check(all(f"## {i}." in text for i in range(1,17)),"report_16_sections",checks)
        manifest=readj(OUT/"artifact_checksums.json");bad=[]
        for row in manifest["files"]:
            p=ROOT/row["path"]
            if not p.is_file() or p.stat().st_size!=row["bytes"] or sha(p)!=row["sha256"]:bad.append(row["path"])
        check(not bad,"all_manifest_hashes_verify",checks)
        result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"manifest_bad":bad}
        print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="PASS" else 1)
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}
    with open(OUT/"B2_final_validation.json","w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="PASS" else 1)

if __name__=="__main__":main()
