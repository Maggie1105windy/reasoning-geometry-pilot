#!/usr/bin/env python3
"""Extract one Tulu checkpoint's frozen B2 activations and fail-closed integrity checks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common_b2 import (
    BASE_ACT, HIDDEN_SIZE, MODELS, N_LAYERS, OUT, REPRESENTATIVE_LAYERS,
    SEED, ensure_dirs, load_union, read_json, sha256, stable_int, write_json,
)


INDEX_LAYERS = [0, 15, 31]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def padded_batch(rows: pd.DataFrame, tokenizer):
    sequences = rows["token_ids_list"].tolist()
    max_len = max(map(len, sequences))
    ids = torch.full((len(rows), max_len), tokenizer.eos_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), max_len), dtype=torch.long)
    for index, seq in enumerate(sequences):
        ids[index, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[index, :len(seq)] = 1
    readout = torch.tensor(rows["readout_index"].astype(int).tolist(), dtype=torch.long)
    if not torch.equal(readout, attention.sum(1) - 1):
        raise RuntimeError("Readout index does not equal final attended token")
    return ids.cuda(non_blocking=True), attention.cuda(non_blocking=True), readout.cuda(non_blocking=True)


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    delta = (a - b).abs()
    return {"cosine_similarity": float(F.cosine_similarity(a[None], b[None], dim=1).item()),
            "mean_absolute_error": float(delta.mean()), "max_absolute_error": float(delta.max())}


def forward_capture(model, tokenizer, rows: pd.DataFrame, layers: list[int]) -> dict[int, torch.Tensor]:
    ids, attention, readout = padded_batch(rows, tokenizer)
    captures = {}; handles = []
    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            captures[layer] = hidden[batch, readout].detach().float().cpu()
        return hook
    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=attention, use_cache=False, output_hidden_states=False)
        return captures
    finally:
        for handle in handles: handle.remove()
        del ids, attention, readout


def batch_trial(model, tokenizer, rows, batch_size):
    sample = rows.nlargest(batch_size, "sequence_length")
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        values = forward_capture(model, tokenizer, sample, list(range(N_LAYERS)))
        torch.cuda.synchronize(); del values
        return {"batch_size": batch_size, "status": "PASS", "seconds": time.perf_counter()-started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "max_sequence_length": int(sample.sequence_length.max())}
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return {"batch_size": batch_size, "status": "OOM", "seconds": time.perf_counter()-started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "max_sequence_length": int(sample.sequence_length.max())}


def sanity(model, tokenizer, rows, key):
    sample = rows.assign(_order=[stable_int("b2-activation-sanity", value) for value in rows.row_id]).nsmallest(32, "_order")
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    captures = forward_capture(model, tokenizer, sample, list(range(N_LAYERS)))
    stats = []
    for layer in range(N_LAYERS):
        x = captures[layer]
        stats.append({"layer": layer, "shape": list(x.shape), "nan": int(torch.isnan(x).sum()),
                      "inf": int(torch.isinf(x).sum()), "mean": float(x.mean()),
                      "std": float(x.std(unbiased=False)), "mean_l2": float(torch.linalg.vector_norm(x,dim=1).mean())})
    check_rows = sample.iloc[[0, len(sample)//2]].copy()
    ids, attention, readout = padded_batch(check_rows, tokenizer)
    raw = {}; handles=[]
    def make_hook(layer):
        def hook(_m,_i,o):
            h=o[0] if isinstance(o,(tuple,list)) else o
            raw[layer]=h[torch.arange(h.shape[0],device=h.device),readout].detach()
        return hook
    for layer in INDEX_LAYERS: handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            result=model(input_ids=ids,attention_mask=attention,use_cache=False,output_hidden_states=True,return_dict=True)
        comparisons=[]; batch=torch.arange(len(check_rows),device="cuda")
        for i,row in enumerate(check_rows.itertuples(index=False)):
            for layer in [0,15]:
                comparisons.append({"row_id":int(row.row_id),"layer":layer,"comparison":"raw_hook_vs_hidden_states_layer_plus_1",**vector_metrics(raw[layer][i],result.hidden_states[layer+1][batch,readout][i])})
            comparisons.append({"row_id":int(row.row_id),"layer":31,"comparison":"raw_hook_vs_hidden_states_32_post_final_RMSNorm",**vector_metrics(raw[31][i],result.hidden_states[32][batch,readout][i])})
            comparisons.append({"row_id":int(row.row_id),"layer":31,"comparison":"final_RMSNorm_raw_hook_vs_hidden_states_32",**vector_metrics(model.model.norm(raw[31][i:i+1])[0],result.hidden_states[32][batch,readout][i])})
    finally:
        for h in handles: h.remove()
        del ids,attention,readout
    comp=pd.DataFrame(comparisons)
    direct=comp[comp.comparison.eq("raw_hook_vs_hidden_states_layer_plus_1")]
    raw31=comp[comp.comparison.eq("raw_hook_vs_hidden_states_32_post_final_RMSNorm")]
    norm31=comp[comp.comparison.eq("final_RMSNorm_raw_hook_vs_hidden_states_32")]
    checks={"samples_32":len(sample)==32,"all_32_layers":set(captures)==set(range(32)),
            "all_shapes_32x4096":all(s["shape"]==[32,4096] for s in stats),
            "nan_inf_zero":all(s["nan"]==0 and s["inf"]==0 for s in stats),
            "layers_0_15_match_hf_tuple":bool(direct.cosine_similarity.min()>0.99999 and direct.max_absolute_error.max()<1e-6),
            "raw_layer31_differs_from_final_norm":bool(raw31.max_absolute_error.min()>1e-3),
            "normed_raw_layer31_matches_hidden_states_32":bool(norm31.cosine_similarity.min()>0.99999)}
    payload={"stage":f"B2 activation sanity {key}","status":"PASS" if all(checks.values()) else "FAIL",
             "sample_row_ids":sample.row_id.astype(int).tolist(),"activation_definition":"raw decoder block output from decoder-layer forward hook before final model RMSNorm",
             "runtime_seconds":time.perf_counter()-started,"gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
             "gpu_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"checks":checks,"layer_statistics":stats,
             "indexing_records":comparisons}
    write_json(OUT/f"activation_sanity_{key}.json",payload)
    if payload["status"]!="PASS": raise RuntimeError(f"Sanity failed {key}: {[k for k,v in checks.items() if not v]}")
    del captures,result,raw
    return payload


def extract(model, tokenizer, rows, act_dir, batch_size):
    n=len(rows)
    maps={layer:np.lib.format.open_memmap(act_dir/f"layer_{layer:02d}.npy",mode="w+",dtype=np.float16,shape=(n,HIDDEN_SIZE)) for layer in range(N_LAYERS)}
    state={}; handles=[]
    def make_hook(layer):
        def hook(_m,_i,o):
            h=o[0] if isinstance(o,(tuple,list)) else o
            b=torch.arange(h.shape[0],device=h.device)
            maps[layer][state["start"]:state["end"]]=h[b,state["readout"]].detach().to(torch.float16).cpu().numpy()
        return hook
    for layer in range(N_LAYERS): handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        for start in range(0,n,batch_size):
            end=min(start+batch_size,n); ids,attention,readout=padded_batch(rows.iloc[start:end],tokenizer)
            state.update(start=start,end=end,readout=readout)
            with torch.inference_mode(): model(input_ids=ids,attention_mask=attention,use_cache=False,output_hidden_states=False)
            del ids,attention,readout
            if start and start%(batch_size*50)==0:
                print(json.dumps({"extracted":start,"total":n,"elapsed_seconds":time.perf_counter()-started}),flush=True)
        torch.cuda.synchronize()
    finally:
        for h in handles:h.remove()
    per={}
    for layer,m in maps.items():
        m.flush()
        per[str(layer)]={"path":str(act_dir/f"layer_{layer:02d}.npy"),"shape":list(m.shape),"dtype":str(m.dtype),
                         "nan_count":int(np.isnan(m).sum()),"inf_count":int(np.isinf(m).sum())}
    np.save(act_dir/"row_ids.npy",rows.row_id.to_numpy(np.int64))
    return {"rows":n,"layers":32,"hidden_size":4096,"dtype":"float16","batch_size":batch_size,
            "runtime_seconds":time.perf_counter()-started,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"per_layer":per,
            "row_ids_sha256":sha256(act_dir/"row_ids.npy")}


def reforward(model,tokenizer,rows,act_dir,batch_size):
    selected=rows.assign(_order=[stable_int("b2-integrity-reforward",value) for value in rows.row_id]).nsmallest(10,"_order")
    records=[]
    for idx,row in selected.iterrows():
        start=(int(idx)//batch_size)*batch_size;end=min(start+batch_size,len(rows))
        replay=forward_capture(model,tokenizer,rows.iloc[start:end],REPRESENTATIVE_LAYERS);local=int(idx)-start
        for layer in REPRESENTATIVE_LAYERS:
            saved=torch.from_numpy(np.load(act_dir/f"layer_{layer:02d}.npy",mmap_mode="r")[idx].copy()).float()
            records.append({"row_id":int(row.row_id),"union_index":int(idx),"layer":layer,"batch_start":start,"batch_end":end,
                            **vector_metrics(replay[layer][local],saved)})
    frame=pd.DataFrame(records)
    checks={"comparisons_10x5":len(frame)==50,"minimum_cosine_at_least_0_9999":bool(frame.cosine_similarity.min()>=0.9999),
            "all_errors_finite":bool(np.isfinite(frame[["mean_absolute_error","max_absolute_error"]]).all().all())}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"minimum_cosine":float(frame.cosine_similarity.min()),
            "mean_mae":float(frame.mean_absolute_error.mean()),"maximum_absolute_error":float(frame.max_absolute_error.max()),
            "records":records}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",choices=["sft","dpo","final"]);args=parser.parse_args();key=args.model
    os.environ["TOKENIZERS_PARALLELISM"]="false";torch.manual_seed(SEED);np.random.seed(SEED);torch.cuda.manual_seed_all(SEED)
    ensure_dirs()
    if read_json(OUT/"b2_0_gate.json")["status"]!="PASS":raise RuntimeError("B2.0 Gate not PASS")
    spec=MODELS[key];act_dir=spec["activation_path"]
    rows=load_union();rows["token_ids_list"]=rows.token_ids.map(json.loads)
    base_ids=np.load(BASE_ACT/"row_ids.npy")
    if not np.array_equal(base_ids,rows.row_id.to_numpy(np.int64)):raise RuntimeError("Base row IDs != union")
    config=AutoConfig.from_pretrained(spec["path"],local_files_only=True)
    if config.num_hidden_layers!=32 or config.hidden_size!=4096:raise RuntimeError("Architecture mismatch")
    tokenizer=AutoTokenizer.from_pretrained(spec["path"],local_files_only=True,use_fast=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side="right"
    started=utc_now();load_start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(spec["path"],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,attn_implementation="sdpa").cuda().eval()
    load_seconds=time.perf_counter()-load_start;params=sum(p.numel() for p in model.parameters())
    if params!=8030326784:raise RuntimeError(f"Unexpected parameters {params}")
    audit={"status":"PASS","model":spec["label"],"model_key":key,"path":str(spec["path"]),"resolved_path":str(spec["path"].resolve()),
           "model_file_revision":spec["file_revision"],"tokenizer_file_revision":spec["tokenizer_file_revision"],"parameters":params,"layers":32,"hidden_size":4096,
           "model_dtype":str(next(model.parameters()).dtype),"weight_source":"safetensors only","transformers_version":__import__("transformers").__version__,
           "torch_version":torch.__version__,"python":platform.python_version(),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"load_seconds":load_seconds,"started_utc":started}
    write_json(OUT/f"model_load_audit_{key}.json",audit)
    trials=[]
    for size in [32,16,8,4,2,1]:
        record=batch_trial(model,tokenizer,rows,size);trials.append(record);print(json.dumps({"model":key,"batch_trial":record}),flush=True)
        if record["status"]=="PASS":break
    passed=[r for r in trials if r["status"]=="PASS"]
    if not passed:raise RuntimeError("Batch 1 OOM")
    chosen=passed[0]["batch_size"]
    sanity_info=sanity(model,tokenizer,rows,key)
    extraction=extract(model,tokenizer,rows,act_dir,chosen)
    recheck=reforward(model,tokenizer,rows,act_dir,chosen)
    expected=rows.row_id.to_numpy(np.int64);saved=np.load(act_dir/"row_ids.npy")
    checks={"all_32_layers_exist":all((act_dir/f"layer_{x:02d}.npy").is_file() for x in range(32)),
            "all_shapes_13758x4096":all(v["shape"]==[13758,4096] for v in extraction["per_layer"].values()),
            "all_float16":all(v["dtype"]=="float16" for v in extraction["per_layer"].values()),
            "nan_inf_zero":all(v["nan_count"]==0 and v["inf_count"]==0 for v in extraction["per_layer"].values()),
            "row_ids_exact_union":np.array_equal(saved,expected),"row_ids_exact_base":np.array_equal(saved,base_ids),
            "reforward_pass":recheck["status"]=="PASS","layer31_indexing_pass":sanity_info["status"]=="PASS"}
    # Hash after all numerical gates pass; these hashes also support later artifact verification.
    hashes={str(layer):sha256(act_dir/f"layer_{layer:02d}.npy") for layer in range(32)}
    payload={"stage":f"B2 activation integrity {key}","status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
             "activation_definition":"raw decoder block output before final model RMSNorm","batch_trials":trials,"extraction":extraction,
             "reforward":recheck,"layer_sha256":hashes,"completed_utc":utc_now()}
    write_json(OUT/f"activation_integrity_{key}.json",payload)
    write_json(act_dir/"activation_integrity.json",payload)
    print(json.dumps({"model":key,"status":payload["status"],"batch_size":chosen,"runtime_seconds":extraction["runtime_seconds"],
                      "peak_reserved_gib":extraction["peak_reserved_gib"],"minimum_cosine":recheck["minimum_cosine"],"mean_mae":recheck["mean_mae"],
                      "maximum_absolute_error":recheck["maximum_absolute_error"]},indent=2),flush=True)
    if payload["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
