"""CPU-only independent audit of the frozen 100-input budget experiment.

Recomputes selection, hashes, tokenizer decoding, graph answers, parsing,
summaries and paired changes without calling the generation runner's helpers.
Graph answers use frozen B0 parsed-edge roles and an independent BFS: the
template inventory's AA/BB placeholders do not themselves specify edge roles.
Never writes to original result shards or model files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/stage_b2_3_budget100_20261014"
CAPS = (8192, 16384, 32768)
EXCLUDED = {85935, 94822, 85687, 94405, 86033, 90023, 101041, 92078, 94459, 93456}
REL = {"ABOVE": (0, 1), "BELOW": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0),
       "UPPER_LEFT": (-1, 1), "UPPER_RIGHT": (1, 1), "LOWER_LEFT": (-1, -1), "LOWER_RIGHT": (1, -1)}
LINE = re.compile(r"^\s*Final[ \t]+answer[ \t]*:[ \t]*(above|below|left|right|upper[ _-]+left|upper[ _-]+right|lower[ _-]+left|lower[ _-]+right)[ \t]*$", re.I)
INSTRUCTION = "End your response with exactly one final line in the following format:\n\nFinal answer: <relation>\n\nwhere <relation> must be one of:\nabove, below, left, right, upper-left, upper-right, lower-left, lower-right."


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def csvrows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def writejson(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def writecsv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    checks = []
    def check(name, ok, detail=None):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    plan, identity, manifest, status = [read(OUT/n) for n in
        ("experiment_plan.json", "runtime_identity.json", "sample_manifest.json", "status.json")]
    samples = manifest["samples"]
    sample_by = {int(s["row_id"]): s for s in samples}
    check("complete_status", status["status"] == "COMPLETE" and status["completed_runs"] == 300 and status["pending_runs"] == 0)
    check("plan_canonical_hash", digest({k:v for k,v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"])
    check("status_identity_plan_match", status["plan_sha256"] == identity["plan_sha256"] == plan["plan_sha256"])
    check("manifest_file_hash", filehash(OUT/"sample_manifest.json") == plan["sample_manifest_sha256"])
    check("runner_file_hash", filehash(Path(__file__).with_name("run_budget100.py")) == plan["runner_sha256"])
    check("common_file_hash", filehash(ROOT/"scripts/stage-b2_3/common_b23.py") == plan["frozen_common_sha256"])
    check("100_unique_equal_hop", len(sample_by) == 100 and Counter(int(s["computed_hop"]) for s in samples) == {h:20 for h in range(1,6)})
    check("exclude_previous_ten", not EXCLUDED.intersection(sample_by))
    source_paths = [ROOT/"results/stage-b0/b0_2/primary_test.csv", ROOT/"results/stage-b1/union_manifest.csv",
                    ROOT/"artifacts/stage_b2_3_budget_pilot_20261013/sample_manifest.json"]
    for source, recorded in zip(source_paths, manifest["source_files"]):
        check("source_hash:"+source.name, filehash(source) == recorded["sha256"])
    population = csvrows(source_paths[0])
    union = {int(r["row_id"]):r for r in csvrows(source_paths[1])}
    check("primary_test_2063", len(population) == 2063)
    selected = []
    for h in range(1,6):
        ids = [int(r["row_id"]) for r in population if int(r["computed_hop"]) == h and int(r["row_id"]) not in EXCLUDED]
        ids.sort(key=lambda n:(hashlib.sha256(f"stage-b2-3-budget100|20261014|{h}|{n}".encode()).hexdigest(),n))
        selected.extend(ids[:20])
    check("blind_selection_recomputed", selected == [int(s["row_id"]) for s in samples])
    check("plan_sample_order", selected == [int(s["row_id"]) for s in plan["sample_rows"]])

    model = identity["model_audit"]
    check("model_audit_canonical_hash", digest({k:v for k,v in model.items() if k != "model_audit_sha256"}) == model["model_audit_sha256"])
    check("full_weight_sha_recorded", model["weights_full_sha256_recomputed"] and model["compared_with_historical_manifest"])
    historical = read(ROOT/"artifacts/stage_b2_3_qwen3_generation/audit/model_manifest.json")
    check("weight_shards_match_historical", model["shards"] == historical["weight_shard_manifest"]["shards"])
    model_dir = OUT/"audit_support/model_files"
    for f in model_dir.iterdir():
        check("tokenizer_config_hash:"+f.name, filehash(f) == model["file_hashes"][f.name])
    tokenizer = Tokenizer.from_file(str(model_dir/"tokenizer.json"))
    tconfig, gconfig = read(model_dir/"tokenizer_config.json"), read(model_dir/"generation_config.json")
    eos = set(gconfig["eos_token_id"] if isinstance(gconfig["eos_token_id"],list) else [gconfig["eos_token_id"]])
    open_id, close_id = tokenizer.token_to_id("<think>"), tokenizer.token_to_id("</think>")
    check("thinking_token_identity", (open_id,close_id) == (151667,151668))
    template = ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(tconfig["chat_template"])

    template_specs = []
    for spec in csvrows(ROOT/"results/stage-b0/b0_1/template_inventory.csv"):
        if spec["semantic_status"] != "OK":
            continue
        text = re.escape(spec["template_text_or_pattern"])
        if "AA" not in text or "BB" not in text:
            continue
        text = text.replace("AA",r"(?P<a>[A-Z])",1).replace("AA",r"(?P=a)")
        text = text.replace("BB",r"(?P<b>[A-Z])",1).replace("BB",r"(?P=b)")
        template_specs.append((re.compile("^"+text+"$"),spec))
    graph_export = read(OUT/"audit_support/frozen_graph_subset.json")
    frozen_graph_rows = {int(r["row_id"]):r for r in graph_export["records"]}
    check("frozen_graph_subset_100", len(frozen_graph_rows)==100 and set(frozen_graph_rows)==set(sample_by))
    graphs = {}
    for s in samples:
        rid, h = int(s["row_id"]), int(s["computed_hop"])
        content = {k:s[k] for k in ("row_id","computed_hop","canonical_relation","story","question")}
        check(f"sample_content:{rid}", digest(content) == s["sample_content_sha256"])
        u = union[rid]
        check(f"source_content:{rid}", json.loads(u["story"]) == s["story"] and u["question"] == s["question"] and u["canonical_relation"] == s["canonical_relation"] and int(u["computed_hop"]) == h and u["primary_split"] == "test")
        metadata = frozen_graph_rows[rid]
        check(f"graph_metadata_content:{rid}",json.loads(metadata["story"])==s["story"] and metadata["question"]==s["question"])
        frozen_edges = {e["sentence_index"]:e for e in json.loads(metadata["parsed_edges"])}
        adjacency, edges = defaultdict(list), []
        for sentence_index,sentence in enumerate(s["story"]):
            candidates = []
            for pattern,spec in template_specs:
                match = pattern.fullmatch(sentence)
                if match:
                    candidates.append((match["a"],match["b"],spec["relation"],spec["template_id"]))
            unique = {(a,b,r) for a,b,r,_ in candidates}
            check(f"independent_sentence_template:{rid}:{len(edges)}", len(unique) == 1, sentence if len(unique)!=1 else None)
            if len(unique) != 1:
                continue
            captured_a,captured_b,r = next(iter(unique))
            frozen_edge = frozen_edges[sentence_index]
            a,b = frozen_edge["arg1"],frozen_edge["arg2"]
            check(f"frozen_edge_template_match:{rid}:{sentence_index}",frozen_edge["sentence"]==sentence and frozen_edge["relation"]==r and {a,b}=={captured_a,captured_b} and frozen_edge["template_id"] in [t for _,_,_,t in candidates])
            dx,dy = REL[r]
            adjacency[b].append((a,dx,dy,r)); adjacency[a].append((b,-dx,-dy,next(k for k,v in REL.items() if v == (-dx,-dy))))
            edges.append(dict(arg1=a,arg2=b,relation=r,sentence=sentence,template_ids=[t for _,_,_,t in candidates]))
        query_entities = re.findall(r"\b[A-Z]\b",s["question"])
        q1,q2 = query_entities[-2:]
        queue,visited = deque([(q2,[q2],[],0,0)]),{q2}
        solution = None
        while queue:
            node,path,relations,x,y = queue.popleft()
            if node == q1:
                solution = (path,relations,x,y); break
            for nxt,dx,dy,r in adjacency[node]:
                if nxt not in visited:
                    visited.add(nxt); queue.append((nxt,path+[nxt],relations+[r],x+dx,y+dy))
        if solution:
            path,relations,x,y = solution
            sign = ((x>0)-(x<0),(y>0)-(y<0))
            gold = next((k for k,v in REL.items() if v == sign),"OVERLAP")
            check(f"independent_graph_gold_hop:{rid}", gold == s["canonical_relation"] and len(relations)==h)
            graphs[str(rid)] = dict(query_arg1=q1,query_arg2=q2,path=path,path_relations=relations,dx=x,dy=y,gold=gold,edges=edges)
        else:
            check(f"independent_graph_gold_hop:{rid}",False)

    all_rows, row_metrics = {}, []
    for cap in CAPS:
        files = sorted((OUT/f"budget_{cap}/shards").glob("row_*.json"))
        check(f"100_shards:{cap}",len(files)==100)
        rows = {}
        for file in files:
            r = read(file); rid = int(r["row_id"]); s = sample_by[rid]
            ids, prompt_ids = r["generated_ids"],r["prompt_ids"]
            key = f"{cap}:{rid}"
            rows[rid] = r
            check("shard_identity:"+key, r["status"]=="COMPLETE" and r["max_new_tokens"]==cap and r["plan_sha256"]==plan["plan_sha256"] and r["model_audit_sha256"]==model["model_audit_sha256"] and r["content_sha256"]==s["sample_content_sha256"] and r["computed_hop"]==int(s["computed_hop"]) and r["gold_relation"]==s["canonical_relation"])
            check("tokens_hash_length:"+key, digest(ids)==r["generated_ids_sha256"] and len(ids)==r["generation_token_count"] and 0<len(ids)<=cap)
            seed = int(hashlib.sha256(f"20260924:{rid}".encode()).hexdigest()[:8],16)%(2**31-1)
            check("protocol_seed:"+key, r["row_seed"]==seed and r["generation_seed"]==20260924 and r["batch_size"]==1 and r["enable_thinking"] is True and r["decoding"]==plan["decoding"])
            raw = "Story:\n"+"\n".join(s["story"])+"\n\nQuestion:\n"+s["question"].strip()+"\n\n"+INSTRUCTION
            expected_prompt = template.render(messages=[{"role":"user","content":raw}],add_generation_prompt=True,enable_thinking=True,**{k:tconfig.get(k) for k in ("bos_token","eos_token","pad_token")})
            check("prompt_reconstructed:"+key, raw==r["user_content"] and expected_prompt==r["prompt_text"] and hashlib.sha256(expected_prompt.encode()).hexdigest()==r["prompt_sha256"] and tokenizer.encode(expected_prompt,add_special_tokens=False).ids==prompt_ids and len(prompt_ids)==r["prompt_token_count"])
            check("complete_text_redecoded:"+key, tokenizer.decode(ids,skip_special_tokens=False)==r["generated_text"])
            opens = [i for i,t in enumerate(ids) if t==open_id]
            closes = [i for i,t in enumerate(ids) if t==close_id]
            prompt_open = open_id in prompt_ids and (close_id not in prompt_ids or prompt_ids.index(open_id)>prompt_ids.index(close_id))
            start = opens[0]+1 if opens else 0
            boundary = bool((prompt_open or opens) and len(closes)==1 and closes[0]>=start and len(opens)<=1 and all(i<=closes[0] for i in opens))
            final_ids = ids[closes[0]+1:] if boundary else []
            thinking_ids = ids[start:closes[0]] if boundary else ids[start:]
            final_text = tokenizer.decode(final_ids,skip_special_tokens=True) if boundary else ""
            answers = [m.group(1) for line in final_text.splitlines() if (m:=LINE.fullmatch(line))]
            relation = re.sub(r"[ _-]+","_",answers[-1]).upper() if answers else "INVALID"
            end_eos = ids[-1] in eos
            parsed = relation != "INVALID"
            correct = relation == s["canonical_relation"]
            check("boundary_final_parse_correct:"+key, boundary==r["think_boundary_ok"] and final_ids==r["final_response_ids"] and final_text==r["final_response_text"] and relation==r["parsed_relation"] and parsed==r["parse_success"] and correct==r["correct"] and len(answers)==r["n_valid_final_answer_lines"] and opens==r["generated_think_open_positions"] and closes==r["generated_think_close_positions"] and len(thinking_ids)==r["thinking_token_count"] and tokenizer.decode(thinking_ids,skip_special_tokens=False)==r["thinking_text"])
            check("eos_and_cap:"+key, end_eos==r["eos_terminated"]==r["normal_completion"] and (len(ids)==cap)==r["reached_token_cap"] and (len(ids)==cap and not end_eos)==r["hit_max_new_tokens"] and not any(t in eos for t in ids[:-1]))
            seconds = r["wall_seconds"]
            delta = (datetime.fromisoformat(r["ended_utc"])-datetime.fromisoformat(r["started_utc"])).total_seconds()
            check("time_memory_context:"+key, seconds>0 and abs(delta-seconds)<1 and math.isclose(len(ids)/seconds,r["tokens_per_second"],rel_tol=1e-10) and 0<r["peak_allocated_gib"]<=r["peak_reserved_gib"]<=24 and len(ids)+len(prompt_ids)<=40960)
            check("gpu_identity_no_other_process:"+key, all(g["uuid"]==identity["gpu_uuid"] and g["name"]==identity["gpu_name"] and all(p["pid"]==str(status["pid"]) for p in g["compute_processes"]) for g in (r["gpu_before"],r["gpu_after"])))
            row_metrics.append(dict(row_id=rid,hop=r["computed_hop"],budget=cap,gold=s["canonical_relation"],parsed=relation,correct=correct,eos=end_eos,think_close=boundary,parse_success=parsed,truncated=len(ids)==cap and not end_eos,tokens=len(ids),thinking_tokens=len(thinking_ids),seconds=seconds,peak_allocated_gib=r["peak_allocated_gib"],peak_reserved_gib=r["peak_reserved_gib"],n_final_lines=len(answers)))
        all_rows[cap] = rows
        check(f"same_100_rows:{cap}",set(rows)==set(sample_by))
        jsonl = [json.loads(line) for line in (OUT/f"budget_{cap}/row_generations.jsonl").read_text(encoding="utf-8").splitlines()]
        check(f"jsonl_shards_agree:{cap}",len(jsonl)==100 and {r["row_id"]:digest(r) for r in jsonl}=={rid:digest(r) for rid,r in rows.items()})

    metrics, transitions = [], []
    summaries = {(int(r["max_new_tokens"]),r["hop"]):r for r in csvrows(OUT/"budget_summary.csv")}
    summary_json = read(OUT/"summary.json")
    json_summaries = {(r["max_new_tokens"],str(r["hop"])):r for r in summary_json["budget_summary"]}
    check("json_summary_scope", len(json_summaries)==18 and summary_json["budgets"]==list(CAPS))
    for cap in CAPS:
        for hop in ("all",1,2,3,4,5):
            group = [r for r in row_metrics if r["budget"]==cap and (hop=="all" or r["hop"]==hop)]
            n = len(group)
            m = dict(budget=cap,hop=hop,n=n,eos=sum(r["eos"] for r in group),think_close=sum(r["think_close"] for r in group),parsed=sum(r["parse_success"] for r in group),correct=sum(r["correct"] for r in group),truncated=sum(r["truncated"] for r in group),total_tokens=sum(r["tokens"] for r in group),mean_tokens=statistics.mean(r["tokens"] for r in group),median_tokens=statistics.median(r["tokens"] for r in group),max_tokens=max(r["tokens"] for r in group),seconds=sum(r["seconds"] for r in group),mean_seconds=statistics.mean(r["seconds"] for r in group),median_seconds=statistics.median(r["seconds"] for r in group),max_seconds=max(r["seconds"] for r in group),peak_allocated_gib=max(r["peak_allocated_gib"] for r in group),peak_reserved_gib=max(r["peak_reserved_gib"] for r in group))
            original = summaries[(cap,str(hop))]
            pairs = {"n_completed_runs":n,"eos_completion_rate":m["eos"]/n,"think_close_rate":m["think_close"]/n,"parse_success_rate":m["parsed"]/n,"accuracy":m["correct"]/n,"cap_truncation_rate":m["truncated"]/n,"total_generated_tokens":m["total_tokens"],"mean_generated_tokens":m["mean_tokens"],"median_generated_tokens":m["median_tokens"],"total_generation_wall_seconds":m["seconds"],"mean_generation_wall_seconds":m["mean_seconds"],"peak_allocated_gib":m["peak_allocated_gib"]}
            check(f"summary_independent:{cap}:{hop}",all(math.isclose(float(original[k]),v,rel_tol=1e-10,abs_tol=1e-9) for k,v in pairs.items()))
            check(f"json_summary_independent:{cap}:{hop}",all(math.isclose(float(json_summaries[(cap,str(hop))][k]),v,rel_tol=1e-10,abs_tol=1e-9) for k,v in pairs.items()))
            metrics.append(m)
    prefixes = []
    recorded_pairs = {(int(r["row_id"]),int(r["budget_a"]),int(r["budget_b"])):r for r in csvrows(OUT/"paired_budget_changes.csv")}
    json_pairs = {(r["row_id"],r["budget_a"],r["budget_b"]):r for r in summary_json["paired_budget_changes"]}
    check("json_paired_scope",len(json_pairs)==300)
    for a,b in ((8192,16384),(8192,32768),(16384,32768)):
        transition = Counter()
        for rid in selected:
            x,y = all_rows[a][rid],all_rows[b][rid]
            overlap = min(len(x["generated_ids"]),len(y["generated_ids"]))
            prefix = x["generated_ids"][:overlap]==y["generated_ids"][:overlap]
            entire = x["generated_ids"]==y["generated_ids"]
            original = recorded_pairs[(rid,a,b)]
            check(f"paired_record:{rid}:{a}:{b}",prefix==(original["prefix_equal"]=="True") and entire==(original["entire_outputs_equal"]=="True") and x["parsed_relation"]==original["parsed_a"] and y["parsed_relation"]==original["parsed_b"])
            jp = json_pairs[(rid,a,b)]
            check(f"json_paired_record:{rid}:{a}:{b}",jp["prefix_equal"]==prefix and jp["entire_outputs_equal"]==entire and jp["parsed_a"]==x["parsed_relation"] and jp["parsed_b"]==y["parsed_relation"] and jp["correct_a"]==x["correct"] and jp["correct_b"]==y["correct"] and jp["tokens_a"]==len(x["generated_ids"]) and jp["tokens_b"]==len(y["generated_ids"]))
            check(f"shared_prefix:{rid}:{a}:{b}",prefix and x["row_seed"]==y["row_seed"] and x["prompt_ids"]==y["prompt_ids"])
            prefixes.append(dict(row_id=rid,budget_a=a,budget_b=b,prefix_equal=prefix,entire_equal=entire,parsed_a=x["parsed_relation"],parsed_b=y["parsed_relation"],correct_a=x["correct"],correct_b=y["correct"],tokens_a=len(x["generated_ids"]),tokens_b=len(y["generated_ids"])))
            transition[(x["parsed_relation"],y["parsed_relation"])]+=1
        transitions.append(dict(budget_a=a,budget_b=b,unchanged_complete_outputs=sum(p["entire_equal"] for p in prefixes if p["budget_a"]==a and p["budget_b"]==b),correct_gain=sum(not all_rows[a][r]["correct"] and all_rows[b][r]["correct"] for r in selected),correct_loss=sum(all_rows[a][r]["correct"] and not all_rows[b][r]["correct"] for r in selected),new_parsed=sum(not all_rows[a][r]["parse_success"] and all_rows[b][r]["parse_success"] for r in selected),changed_answer_rows=[r for r in selected if all_rows[a][r]["parsed_relation"]!=all_rows[b][r]["parsed_relation"]]))
    chronology = sorted((datetime.fromisoformat(r["started_utc"]),datetime.fromisoformat(r["ended_utc"]),cap,rid) for cap,rows in all_rows.items() for rid,r in rows.items())
    check("300_independent_sequential_calls",[(rid,cap) for _,_,cap,rid in chronology]==[(rid,cap) for rid in selected for cap in CAPS] and all(chronology[i][1]<=chronology[i+1][0] for i in range(299)))
    error_files = list((OUT/"errors").glob("*.json")) if (OUT/"errors").exists() else []
    log = (OUT/"logs/run_20261013T162339Z.log").read_text(encoding="utf-8")
    check("no_errors_oom_traceback",not error_files and not any(x in log.lower() for x in ("traceback","out of memory","outofmemoryerror")))
    cells = Counter((int(s["computed_hop"]),s["canonical_relation"]) for s in samples)
    check("coverage_39_of_40",len(cells)==39 and cells[(3,"ABOVE")]==0)
    inventory = [dict(path=str(p.relative_to(OUT)).replace("\\","/"),bytes=p.stat().st_size,sha256=filehash(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and not p.name.startswith("audit_completion") and "audit_support" not in p.parts]
    writejson(OUT/"audit_completion_inventory.json",inventory)
    writecsv(OUT/"audit_completion_rows.csv",row_metrics)
    writecsv(OUT/"audit_completion_metrics.csv",metrics)
    writecsv(OUT/"audit_completion_pairs.csv",prefixes)
    writejson(OUT/"audit_completion_graphs.json",graphs)
    failed = [c for c in checks if not c["passed"]]
    result = dict(audited_utc=datetime.now(timezone.utc).isoformat(),audit_script_sha256=filehash(__file__),method="Independent CPU tokenizer decode and BFS over frozen B0 parsed edges; template inventory checked without assuming AA/BB semantic roles; no generation-runner helper imported",graph_source_sha256=graph_export["source_sha256"],weights_scope="Remote generation runner recomputed all weight SHA256; local audit checks recorded weight hashes against historical manifest without downloading weights",status="PASS" if not failed else "FAIL",checks_total=len(checks),checks_failed=failed,checks=checks,model_audit_sha256=model["model_audit_sha256"],plan_sha256=plan["plan_sha256"],metrics=metrics,transitions=transitions,prefix_pairs=len(prefixes),prefix_agree=sum(p["prefix_equal"] for p in prefixes),wall_seconds=(datetime.fromisoformat(status["ended_utc"])-datetime.fromisoformat(status["started_utc"])).total_seconds(),generation_seconds=sum(r["seconds"] for r in row_metrics),first_generation_utc=chronology[0][0].isoformat(),last_generation_utc=chronology[-1][1].isoformat(),error_files=[str(p) for p in error_files],coverage=dict(observed_cells=len(cells),missing_cells=[[h,k] for h in range(1,6) for k in REL if (h,k) not in cells]),full_cap_32k_rows=[r["row_id"] for r in row_metrics if r["budget"]==32768 and r["tokens"]==32768])
    writejson(OUT/"audit_completion.json",result)
    print(json.dumps({k:v for k,v in result.items() if k in ("status","checks_total","checks_failed","transitions","wall_seconds","generation_seconds","prefix_pairs","prefix_agree","full_cap_32k_rows")},ensure_ascii=False),flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
