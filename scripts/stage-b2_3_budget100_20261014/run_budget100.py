#!/usr/bin/env python3
"""Independent 100-input, batch-one Qwen3 thinking budget diagnostic.

The preparation and summary modes use only Python's standard library. Actual
generation requires local transformers/torch and an idle BF16-capable GPU.
Every row at every cap is generated from the original prompt with the same
row-specific RNG seed; longer runs never continue a shorter run's output.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

VERSION = "stage_b2_3_budget100_20261014_v1"
GENERATION_SEED = 20260924
SAMPLE_SIZE = 100
SAMPLES_PER_HOP = 20
SELECTION_SEED = 20261014
EXCLUDED_PILOT_ROW_IDS = (85935, 94822, 85687, 94405, 86033, 90023, 101041, 92078, 94459, 93456)
RELATIONS = ["ABOVE", "BELOW", "LEFT", "RIGHT", "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT"]
OUTPUT_FORMAT_INSTRUCTION = """End your response with exactly one final line in the following format:

Final answer: <relation>

where <relation> must be one of:
above, below, left, right, upper-left, upper-right, lower-left, lower-right."""
DECODING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, use_cache=True)
RNG_POLICY = (
    "Batch size 1; before each independent generate call, torch.manual_seed and "
    "torch.cuda.manual_seed_all use int(sha256(f'20260924:{row_id}')[:8],16) % (2**31-1). "
    "The budget is excluded from the seed. This row RNG strategy does not "
    "require agreement with the historical batch-eight generation prefixes. "
    "Cross-budget prefix equality is measured rather than assumed."
)
ANSWER_RE = re.compile(r"(?im)^[ \t]*Final[ \t]+answer[ \t]*:[ \t]*(above|below|left|right|upper[ _-]+left|upper[ _-]+right|lower[ _-]+left|lower[ _-]+right)[ \t]*$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path, data, immutable=False):
    """Publish a complete file atomically. Immutable files are never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                if os.name == "nt":
                    os.rename(temporary, path)  # Windows rename fails if destination exists
                else:
                    os.link(temporary, path)  # POSIX atomic create-if-absent
            except FileExistsError:
                if path.read_bytes() != data:
                    raise RuntimeError(f"Conflicting existing artifact; refusing overwrite: {path}")
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, immutable=False):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), immutable)


def update_status(out, status, **details):
    path = out / "status.json"
    previous = read_json(path) if path.exists() else {}
    previous.update(version=VERSION, status=status, pid=os.getpid(), host=socket.gethostname(), updated_utc=utc_now(), **details)
    write_json(path, previous)


def write_csv(path, rows):
    if not rows:
        atomic_bytes(path, b"")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def row_seed(row_id):
    return int(hashlib.sha256(f"{GENERATION_SEED}:{int(row_id)}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


def user_content(row):
    raw = "Story:\n" + "\n".join(row["story"]) + "\n\nQuestion:\n" + row["question"].strip()
    if row.get("prompt") is not None and row["prompt"] != raw:
        raise RuntimeError(f"Raw prompt mismatch for row {row['row_id']}")
    return raw + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


def load_samples(path):
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("selection_seed") != SELECTION_SEED:
        raise ValueError("This expansion requires the independently frozen selection seed 20261014")
    excluded = manifest.get("excluded_prior_diagnostic", {})
    if sorted(excluded.get("row_ids", [])) != sorted(EXCLUDED_PILOT_ROW_IDS) or excluded.get("rows") != 10:
        raise ValueError("Manifest must explicitly record exclusion of the original ten pilot inputs")
    rows = manifest.get("samples", manifest.get("rows")) if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest requires a nonempty samples array")
    result = []
    for raw in rows:
        story = raw["story"]
        if isinstance(story, str):
            story = json.loads(story)
        row = dict(raw, row_id=int(raw["row_id"]), computed_hop=int(raw.get("computed_hop", raw.get("hop", 0))),
                   canonical_relation=str(raw.get("canonical_relation", raw.get("gold", ""))), story=story)
        if not isinstance(story, list) or not all(isinstance(s, str) for s in story):
            raise ValueError("story must be an array of strings")
        if row["computed_hop"] not in range(1, 6) or row["canonical_relation"] not in RELATIONS:
            raise ValueError("Invalid hop or gold relation")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Question must be nonempty")
        frozen_content = {k: row[k] for k in ("row_id", "computed_hop", "canonical_relation", "story", "question")}
        row["content_sha256"] = digest(frozen_content)
        if raw.get("sample_content_sha256") != row["content_sha256"]:
            raise RuntimeError(f"Manifest content hash mismatch for row {row['row_id']}")
        selection_digest = hashlib.sha256(f"stage-b2-3-budget100|{SELECTION_SEED}|{row['computed_hop']}|{row['row_id']}".encode()).hexdigest()
        if raw.get("selection_digest") != selection_digest:
            raise RuntimeError(f"Manifest selection digest mismatch for row {row['row_id']}")
        result.append(row)
    if len({r["row_id"] for r in result}) != len(result):
        raise ValueError("Duplicate row_id")
    if len(result) != SAMPLE_SIZE or any(sum(r["computed_hop"] == h for r in result) != SAMPLES_PER_HOP for h in range(1, 6)):
        raise ValueError("This frozen expansion requires exactly 100 inputs: twenty samples in each of hops 1..5")
    overlap = sorted(set(EXCLUDED_PILOT_ROW_IDS) & {r["row_id"] for r in result})
    if overlap:
        raise ValueError(f"The 100-input expansion must exclude the original ten pilot inputs: {overlap}")
    return result


def prepare(root, manifest_path, out):
    rows = load_samples(manifest_path)
    manifest = read_json(manifest_path)
    common = root / "scripts" / "stage-b2_3" / "common_b23.py"
    if not common.is_file():
        raise FileNotFoundError(f"Frozen prompt definitions unavailable: {common}")
    constants = {}
    for node in ast.parse(common.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"OUTPUT_FORMAT_INSTRUCTION", "RELATIONS"}:
                    constants[target.id] = ast.literal_eval(node.value)
    if constants.get("OUTPUT_FORMAT_INSTRUCTION") != OUTPUT_FORMAT_INSTRUCTION or constants.get("RELATIONS") != RELATIONS:
        raise RuntimeError("Pilot prompt/labels differ from the frozen Stage B2.3 definitions")
    plan = dict(version=VERSION, sample_rows=[{k: row[k] for k in ("row_id", "computed_hop", "canonical_relation", "story", "question", "content_sha256")} for row in rows],
                generation_seed=GENERATION_SEED, decoding=DECODING, enable_thinking=True, batch_size=1,
                dtype="torch.bfloat16", attention="sdpa", quantization=None,
                rng_policy=RNG_POLICY, allowed_budgets=[8192, 16384, 32768],
                sample_size=SAMPLE_SIZE, samples_per_hop=SAMPLES_PER_HOP, excluded_pilot_row_ids=list(EXCLUDED_PILOT_ROW_IDS),
                sample_manifest_sha256=file_hash(manifest_path), selection_seed=SELECTION_SEED,
                selection_algorithm=manifest.get("selection_algorithm"), sample_source_files=manifest.get("source_files"),
                parser="Frozen final-line parser applied only AFTER an observed generated </think> token boundary; missing/ambiguous boundary is INVALID",
                independence="Every budget runs generate from the original prompt; no generated-prefix reuse",
                inference_boundary="100 equal-hop inputs, excluding the ten diagnostic pilot inputs; estimate conditional budget effects/completion/runtime, not the natural Primary hop mixture or seed variability",
                frozen_common_sha256=file_hash(common), runner_sha256=file_hash(__file__))
    plan["plan_sha256"] = digest(plan)
    write_json(out / "experiment_plan.json", plan, immutable=True)
    return rows, plan


def smi_snapshot(gpu):
    fields = ["index", "uuid", "name", "memory.total", "memory.used", "memory.free", "temperature.gpu", "power.draw", "pstate", "clocks.current.sm", "clocks.current.memory", "utilization.gpu"]
    command = ["nvidia-smi", "--id=" + str(gpu), "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    values = list(csv.reader(io.StringIO(result.stdout)))
    if len(values) != 1:
        raise RuntimeError("Exactly one selected GPU is required")
    status = dict(zip(fields, [v.strip() for v in values[0]]))
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
    processes = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    status["compute_processes"] = [dict(zip(["gpu_uuid", "pid", "process_name", "used_gpu_memory"], [v.strip() for v in row]))
                                   for row in csv.reader(io.StringIO(processes.stdout)) if row and row[0].strip() == status["uuid"]]
    status.update(captured_utc=utc_now(), raw_gpu_query=result.stdout, raw_process_query=processes.stdout)
    return status


def assert_gpu_available(snapshot, before_load=False):
    others = [p for p in snapshot["compute_processes"] if p["pid"] != str(os.getpid())]
    if others:
        raise RuntimeError(f"GPU_BUSY: other compute process(es) found; no process was stopped: {others}")
    if before_load and float(snapshot["memory.free"]) < 20 * 1024:
        raise RuntimeError(f"GPU_BUSY: only {snapshot['memory.free']} MiB free; at least 20480 MiB required before loading")
    if "3090" not in snapshot["name"]:
        raise RuntimeError(f"Expected the single RTX 3090; selected device is {snapshot['name']}")


def audit_model(root):
    model_path = root / "models" / "Qwen3-8B"
    config = read_json(model_path / "config.json")
    if config.get("model_type") != "qwen3":
        raise RuntimeError("Expected local Qwen3 model")
    index_path = model_path / "model.safetensors.index.json"
    index = read_json(index_path)
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (model_path / shard).is_file() for shard in shards):
        raise RuntimeError("Local model weight shards are missing")
    for shard in shards:
        if Path(shard).name != shard:
            raise RuntimeError("Unsafe weight shard entry")
    files = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json", "model.safetensors.index.json"]
    hashes = {name: file_hash(model_path / name) for name in files}
    weight_hashes = {name: file_hash(model_path / name) for name in shards}
    old_path = root / "artifacts" / "stage_b2_3_qwen3_generation" / "audit" / "model_manifest.json"
    if old_path.is_file():
        old = read_json(old_path)
        if hashes["config.json"] != old["config_sha256"]:
            raise RuntimeError("Model config differs from Stage B2.3")
        for name, old_file in old["tokenizer_files"].items():
            if name in hashes and hashes[name] != old_file["sha256"]:
                raise RuntimeError(f"Tokenizer/config differs from Stage B2.3: {name}")
        old_weight_manifest = old["weight_shard_manifest"]
        if hashes["model.safetensors.index.json"] != old_weight_manifest["index_sha256"]:
            raise RuntimeError("Model weight index differs from Stage B2.3")
        for old_shard in old_weight_manifest["shards"]:
            if (model_path / old_shard["file"]).stat().st_size != old_shard["bytes"]:
                raise RuntimeError("Model shard size differs from Stage B2.3")
            if weight_hashes[old_shard["file"]] != old_shard["sha256"]:
                raise RuntimeError(f"Model shard content differs from Stage B2.3: {old_shard['file']}")
    audit = dict(model_name="Qwen/Qwen3-8B", file_hashes=hashes,
                 shards=[dict(file=s, bytes=(model_path / s).stat().st_size, sha256=weight_hashes[s]) for s in shards],
                 weights_full_sha256_recomputed=True, compared_with_historical_manifest=old_path.is_file(),
                 max_position_embeddings=int(config["max_position_embeddings"]))
    audit["model_audit_sha256"] = digest(audit)
    return model_path, audit


def tokenizer_bundle(tokenizer, row):
    content = user_content(row)
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True))
    if ids != tokenizer(prompt, add_special_tokens=False).input_ids:
        raise RuntimeError("Qwen chat-template text/token roundtrip failed")
    return dict(user_content=content, prompt_text=prompt, prompt_ids=ids, prompt_token_count=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def parse_generated(tokenizer, prompt_ids, generated_ids, eos):
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    for name, token_id in (("<think>", open_id), ("</think>", close_id)):
        if token_id is None or token_id == tokenizer.unk_token_id or tokenizer.decode([token_id], skip_special_tokens=False) != name:
            raise RuntimeError(f"Required thinking special token unavailable: {name}")
    prompt_open = open_id in prompt_ids and (close_id not in prompt_ids or prompt_ids.index(open_id) > prompt_ids.index(close_id))
    generated_open = [i for i, v in enumerate(generated_ids) if v == open_id]
    generated_close = [i for i, v in enumerate(generated_ids) if v == close_id]
    start = generated_open[0] + 1 if generated_open else 0
    boundary_ok = bool((prompt_open or generated_open) and len(generated_close) == 1 and generated_close[0] >= start
                       and len(generated_open) <= 1 and not any(i > generated_close[0] for i in generated_open))
    close_position = generated_close[0] if boundary_ok else None
    thinking_ids = generated_ids[start:close_position] if boundary_ok else generated_ids[start:]
    final_ids = generated_ids[close_position + 1:] if boundary_ok else []
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True) if boundary_ok else ""
    matches = list(ANSWER_RE.finditer(final_text)) if boundary_ok else []
    word = re.sub(r"[ _-]+", "-", matches[-1].group(1).strip().lower()) if matches else None
    relation = word.upper().replace("-", "_") if word else "INVALID"
    return dict(think_boundary_ok=boundary_ok, has_generated_think_close=bool(generated_close),
                generated_think_open_positions=generated_open, generated_think_close_positions=generated_close,
                think_close_token_position=close_position, thinking_token_count=len(thinking_ids),
                thinking_text=tokenizer.decode(thinking_ids, skip_special_tokens=False),
                final_response_ids=final_ids, final_response_text=final_text,
                final_response_token_count=len(final_ids), parsed_relation=relation, parsed_word=word,
                parse_success=relation != "INVALID", n_valid_final_answer_lines=len(matches),
                parse_status="PARSED_LAST_VALID_FINAL_ANSWER_LINE" if matches else ("INVALID_NO_VALID_FINAL_ANSWER_LINE" if boundary_ok else "INVALID_NO_UNAMBIGUOUS_THINK_CLOSE"),
                eos_terminated=bool(generated_ids and generated_ids[-1] in eos))


def acquire_lock(out):
    lock = out / "run.lock"
    if lock.exists():
        previous = read_json(lock)
        stale_here = (previous.get("host") == socket.gethostname() and sys.platform.startswith("linux")
                      and not Path("/proc", str(previous.get("pid", ""))).exists())
        if stale_here:
            lock.unlink()
        else:
            raise RuntimeError(f"An active or unverified run lock exists: {lock}; {previous}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(host=socket.gethostname(), pid=os.getpid(), created_utc=utc_now()), handle)
    return lock


def validate_shard(row, budget, plan, model_audit, payload, bundle=None):
    expected = dict(version=VERSION, plan_sha256=plan["plan_sha256"], model_audit_sha256=model_audit["model_audit_sha256"],
                    row_id=row["row_id"], content_sha256=row["content_sha256"], max_new_tokens=budget, row_seed=row_seed(row["row_id"]))
    if any(payload.get(k) != v for k, v in expected.items()) or payload.get("status") != "COMPLETE":
        raise RuntimeError(f"Existing shard conflicts with the frozen run: row={row['row_id']}, budget={budget}")
    if payload.get("generated_ids_sha256") != digest(payload.get("generated_ids")):
        raise RuntimeError("Existing shard generated token integrity failure")
    metadata = dict(computed_hop=row["computed_hop"], gold_relation=row["canonical_relation"],
                    generation_seed=GENERATION_SEED, batch_size=1, enable_thinking=True, decoding=DECODING,
                    user_content=user_content(row))
    if any(payload.get(k) != v for k, v in metadata.items()):
        raise RuntimeError("Existing shard sample/decoding metadata conflict")
    generated = payload.get("generated_ids")
    if (not isinstance(generated, list) or not all(isinstance(v, int) for v in generated)
            or len(generated) != payload.get("generation_token_count") or not 0 < len(generated) <= budget):
        raise RuntimeError("Existing shard token count is invalid")
    if payload.get("prompt_token_count") != len(payload.get("prompt_ids", [])):
        raise RuntimeError("Existing shard prompt length conflict")
    if payload.get("prompt_sha256") != hashlib.sha256(payload.get("prompt_text", "").encode()).hexdigest():
        raise RuntimeError("Existing shard prompt text integrity failure")
    if bundle is not None and any(payload.get(k) != v for k, v in bundle.items()):
        raise RuntimeError("Existing shard prompt differs from the frozen runtime bundle")
    if payload.get("correct") != (payload.get("parsed_relation") == row["canonical_relation"]):
        raise RuntimeError("Existing shard correctness metadata conflict")
    if payload.get("parse_success") != (payload.get("parsed_relation") in RELATIONS):
        raise RuntimeError("Existing shard parser metadata conflict")


def run(root, out, rows, plan, budgets, gpu):
    initial = smi_snapshot(gpu)
    assert_gpu_available(initial, before_load=True)
    model_path, model_audit = audit_model(root)
    # Select the audited physical GPU before torch can initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = initial["uuid"]
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; CPU and quantized fallback are not allowed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bundles = {r["row_id"]: tokenizer_bundle(tokenizer, r) for r in rows}
    for row in rows:
        if bundles[row["row_id"]]["prompt_token_count"] + max(budgets) > model_audit["max_position_embeddings"]:
            raise RuntimeError(f"Context overflow: row {row['row_id']}, input + cap > max_position_embeddings")
    audit = dict(version=VERSION, plan_sha256=plan["plan_sha256"], model_audit=model_audit,
                 torch_version=torch.__version__, transformers_version=transformers.__version__,
                 python_version=sys.version, gpu_uuid=initial["uuid"], gpu_name=initial["name"],
                 max_input_token_count=max(b["prompt_token_count"] for b in bundles.values()),
                 declared_max_context=model_audit["max_position_embeddings"], thinking_open_id=tokenizer.convert_tokens_to_ids("<think>"),
                 thinking_close_id=tokenizer.convert_tokens_to_ids("</think>"), prompt_bundles=bundles)
    write_json(out / "runtime_identity.json", audit, immutable=True)
    write_json(out / "telemetry" / f"startup_{os.getpid()}_{time.time_ns()}.json", initial, immutable=True)
    pending = []
    for row in rows:
        for budget in budgets:
            shard = out / f"budget_{budget}" / "shards" / f"row_{row['row_id']}.json"
            if shard.exists():
                validate_shard(row, budget, plan, model_audit, read_json(shard), bundles[row["row_id"]])
            else:
                pending.append((budget, row, shard))
    if not pending:
        print(json.dumps(dict(status="ALREADY_COMPLETE", budgets=budgets)), flush=True)
        summarize(out, budgets)
        return
    assert_gpu_available(smi_snapshot(gpu), before_load=True)
    update_status(out, "RUNNING", phase="MODEL_LOADING", total_selected_runs=len(rows) * len(budgets),
                  pending_runs=len(pending), completed_runs=len(rows) * len(budgets) - len(pending))
    started_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa").cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started_load
    if model.training or next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Model is not eval BF16")
    eos_value = model.generation_config.eos_token_id
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    eos = {int(eos_value)} if isinstance(eos_value, int) else set(map(int, eos_value))
    completed_runs = len(rows) * len(budgets) - len(pending)
    for budget, row, shard in pending:
        before = smi_snapshot(gpu)
        assert_gpu_available(before)
        update_status(out, "RUNNING", phase="GENERATING", current_row_id=row["row_id"], current_hop=row["computed_hop"],
                      current_budget=budget, current_run_started_utc=utc_now(), model_load_seconds=load_seconds,
                      completed_runs=completed_runs)
        bundle = bundles[row["row_id"]]
        inputs = torch.tensor([bundle["prompt_ids"]], dtype=torch.long, device="cuda")
        mask = torch.ones_like(inputs)
        seed = row_seed(row["row_id"])
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        started_utc = utc_now()
        try:
            with torch.inference_mode():
                sequence = model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=budget,
                                          pad_token_id=int(tokenizer.pad_token_id), eos_token_id=eos_value,
                                          return_dict_in_generate=False, **DECODING)
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
            generated = list(map(int, sequence[0, len(bundle["prompt_ids"]):].cpu().tolist()))
        except Exception as exc:
            error = dict(version=VERSION, status="ERROR", row_id=row["row_id"], max_new_tokens=budget,
                         error_type=type(exc).__name__, error=str(exc), wall_seconds=time.perf_counter() - started,
                         gpu_before=before, started_utc=started_utc, ended_utc=utc_now())
            write_json(out / "errors" / f"budget_{budget}_row_{row['row_id']}_{time.time_ns()}.json", error, immutable=True)
            raise
        after = smi_snapshot(gpu)
        parsed = parse_generated(tokenizer, bundle["prompt_ids"], generated, eos)
        reached_cap = len(generated) >= budget
        payload = dict(version=VERSION, status="COMPLETE", plan_sha256=plan["plan_sha256"],
                       model_audit_sha256=model_audit["model_audit_sha256"], row_id=row["row_id"],
                       computed_hop=row["computed_hop"], gold_relation=row["canonical_relation"],
                       story_group_id=row.get("story_group_id"), content_sha256=row["content_sha256"],
                       generation_seed=GENERATION_SEED, row_seed=seed, rng_policy=RNG_POLICY,
                       max_new_tokens=budget, batch_size=1, enable_thinking=True, decoding=DECODING,
                       generated_ids=generated, generated_ids_sha256=digest(generated),
                       generated_text=tokenizer.decode(generated, skip_special_tokens=False),
                       generation_token_count=len(generated), reached_token_cap=reached_cap,
                       hit_max_new_tokens=reached_cap and not parsed["eos_terminated"],
                       normal_completion=parsed["eos_terminated"],
                       correct=parsed["parsed_relation"] == row["canonical_relation"],
                       started_utc=started_utc, ended_utc=utc_now(), wall_seconds=wall_seconds,
                       tokens_per_second=len(generated) / wall_seconds if wall_seconds else None,
                       model_load_seconds_this_invocation=load_seconds,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                       gpu_before=before, gpu_after=after, **bundle, **parsed)
        write_json(shard, payload, immutable=True)
        del inputs, mask, sequence
        print(json.dumps(dict(status="ROW_COMPLETE", row_id=row["row_id"], hop=row["computed_hop"], budget=budget,
                              tokens=len(generated), eos=parsed["eos_terminated"], think_close=parsed["think_boundary_ok"],
                              parsed=parsed["parsed_relation"], correct=payload["correct"], seconds=wall_seconds)), flush=True)
        summarize(out, budgets)
        completed_runs += 1
        update_status(out, "RUNNING", phase="BETWEEN_RUNS", completed_runs=completed_runs,
                      last_completed_row_id=row["row_id"], last_completed_budget=budget,
                      last_completed_tokens=len(generated), last_completed_seconds=wall_seconds,
                      pending_runs=len(rows) * len(budgets) - completed_runs)
        assert_gpu_available(after)
    summarize(out, budgets)


def summarize(out, budgets):
    plan_path = out / "experiment_plan.json"
    plan = read_json(plan_path) if plan_path.exists() else None
    expected = plan["sample_rows"] if plan else []
    expected_ids = {r["row_id"] for r in expected}
    expected_by_id = {r["row_id"]: r for r in expected}
    identity_path = out / "runtime_identity.json"
    identity = read_json(identity_path) if identity_path.exists() else None
    all_rows, summaries, changes = {}, [], []
    for budget in budgets:
        rows = [read_json(p) for p in sorted((out / f"budget_{budget}" / "shards").glob("row_*.json"))]
        for row in rows:
            if row.get("status") != "COMPLETE" or row.get("max_new_tokens") != budget or row.get("generated_ids_sha256") != digest(row.get("generated_ids")):
                raise RuntimeError("Invalid result shard during summary")
            if expected_ids and row["row_id"] not in expected_ids:
                raise RuntimeError("Unexpected row in budget directory")
            if not plan or not identity or identity["plan_sha256"] != plan["plan_sha256"]:
                raise RuntimeError("Result shards require matching frozen plan and runtime identity")
            validate_shard(expected_by_id[row["row_id"]], budget, plan, identity["model_audit"], row,
                           identity["prompt_bundles"][str(row["row_id"])])
        all_rows[budget] = {r["row_id"]: r for r in rows}
        for hop in ["all", 1, 2, 3, 4, 5]:
            group = rows if hop == "all" else [r for r in rows if r["computed_hop"] == hop]
            n = len(group)
            summaries.append(dict(max_new_tokens=budget, hop=hop, n_completed_runs=n,
                                  n_expected=len(expected) if hop == "all" else sum(r["computed_hop"] == hop for r in expected),
                                  eos_completion_rate=sum(r["eos_terminated"] for r in group) / n if n else None,
                                  think_close_rate=sum(r["think_boundary_ok"] for r in group) / n if n else None,
                                  parse_success_rate=sum(r["parse_success"] for r in group) / n if n else None,
                                  accuracy=sum(r["correct"] for r in group) / n if n else None,
                                  cap_truncation_rate=sum(r["hit_max_new_tokens"] for r in group) / n if n else None,
                                  total_generated_tokens=sum(r["generation_token_count"] for r in group),
                                  mean_generated_tokens=statistics.mean(r["generation_token_count"] for r in group) if n else None,
                                  median_generated_tokens=statistics.median(r["generation_token_count"] for r in group) if n else None,
                                  total_generation_wall_seconds=sum(r["wall_seconds"] for r in group),
                                  mean_generation_wall_seconds=statistics.mean(r["wall_seconds"] for r in group) if n else None,
                                  peak_allocated_gib=max((r["peak_allocated_gib"] for r in group), default=None)))
        if rows:
            atomic_bytes(out / f"budget_{budget}" / "row_generations.jsonl", b"".join(canonical_bytes(r) + b"\n" for r in rows))
    for position, small in enumerate(budgets):
        for large in budgets[position + 1:]:
            for row_id in sorted(set(all_rows[small]) & set(all_rows[large])):
                a, b = all_rows[small][row_id], all_rows[large][row_id]
                tokens_a, tokens_b = a["generated_ids"], b["generated_ids"]
                overlap = min(len(tokens_a), len(tokens_b))
                mismatch = next((i for i in range(overlap) if tokens_a[i] != tokens_b[i]), None)
                changes.append(dict(row_id=row_id, hop=a["computed_hop"], budget_a=small, budget_b=large,
                                    shared_seed=a["row_seed"] == b["row_seed"], compared_prefix_tokens=overlap,
                                    prefix_equal=mismatch is None, first_mismatch_position=mismatch,
                                    entire_outputs_equal=tokens_a == tokens_b,
                                    eos_a=a["eos_terminated"], eos_b=b["eos_terminated"],
                                    think_close_a=a["think_boundary_ok"], think_close_b=b["think_boundary_ok"],
                                    parsed_a=a["parsed_relation"], parsed_b=b["parsed_relation"],
                                    correct_a=a["correct"], correct_b=b["correct"],
                                    tokens_a=len(tokens_a), tokens_b=len(tokens_b),
                                    wall_seconds_a=a["wall_seconds"], wall_seconds_b=b["wall_seconds"]))
    write_csv(out / "budget_summary.csv", summaries)
    write_csv(out / "paired_budget_changes.csv", changes)
    write_json(out / "summary.json", dict(version=VERSION, updated_utc=utc_now(), budgets=budgets,
                                         inference_boundary="Partial denominators use completed runs only; 100 equal-hop paired inputs with one seed do not represent the natural Primary hop mixture or quantify seed variability",
                                         budget_summary=summaries, paired_budget_changes=changes))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index or UUID, default 0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "artifacts" / "stage_b2_3_budget100_20261014").resolve()
    manifest = (args.manifest or out / "sample_manifest.json").resolve()
    budgets = sorted(set(args.budgets))
    if not budgets or any(b not in [8192, 16384, 32768] for b in budgets):
        parser.error("Only the frozen budgets 8192, 16384 and 32768 are supported")
    if out == root or not out.is_relative_to(root / "artifacts"):
        parser.error("--out must be an independent directory under --root/artifacts")
    for protected_name in ("stage_b2_3_qwen3_generation", "stage_b2_3_budget_pilot_20261013"):
        protected = root / "artifacts" / protected_name
        if out == protected or out.is_relative_to(protected):
            parser.error("Historical Stage B2.3 and ten-input pilot outputs are protected; choose the new 100-input directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summaries = summarize(out, budgets)
        print(json.dumps(dict(status="SUMMARIZED", output=str(out), completed=[s for s in summaries if s["hop"] == "all"]), ensure_ascii=False), flush=True)
        return
    rows, plan = prepare(root, manifest, out)
    if args.prepare_only:
        print(json.dumps(dict(status="PREPARED_WITHOUT_TORCH_OR_GPU", plan_sha256=plan["plan_sha256"], rows=len(rows),
                              budgets=budgets, output=str(out), model_exists=(root / "models" / "Qwen3-8B" / "config.json").is_file(),
                              runtime_checks_pending=["idle GPU with >=20GiB free", "BF16", "local model audit", "prompt tokens + cap <= max_position_embeddings"]), ensure_ascii=False), flush=True)
        return
    lock = acquire_lock(out)
    try:
        update_status(out, "STARTED", started_utc=utc_now(), budgets=budgets, expected_rows=len(rows),
                      total_selected_runs=len(rows) * len(budgets), plan_sha256=plan["plan_sha256"])
        run(root, out, rows, plan, budgets, args.gpu)
        update_status(out, "COMPLETE", phase="FINISHED", completed_runs=len(rows) * len(budgets), pending_runs=0, ended_utc=utc_now())
    except Exception as error:
        update_status(out, "FAILED", ended_utc=utc_now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status="STOPPED_WITH_ERROR", error_type=type(error).__name__, error=str(error)), ensure_ascii=False), file=sys.stderr, flush=True)
        raise
