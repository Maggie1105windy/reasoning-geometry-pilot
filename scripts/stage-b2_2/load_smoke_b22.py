#!/usr/bin/env python3
"""Phase 1: load Qwen3-8B in BF16 and run two non-evaluative forward smokes."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common_b22 import OUT, build_raw_prompt, ensure_dirs, load_primary, native_natural_user, read_json, write_json_atomic


def main() -> None:
    ensure_dirs()
    preflight = read_json(OUT / "audit" / "preflight_summary.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("Phase 0 preflight is not PASS")
    manifest = read_json(OUT / "audit" / "model_manifest.json")
    model_path = Path(manifest["local_path"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; do not silently quantize the main model")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
    ).cuda().eval()
    if model.training:
        raise RuntimeError("model.eval() did not take effect")

    row = load_primary().iloc[0]
    raw_bos_id = manifest["raw_explicit_bos_token_id"]
    raw_ids = [int(raw_bos_id)] + list(tok(build_raw_prompt(row), add_special_tokens=False).input_ids)
    native_text = tok.apply_chat_template(
        [{"role": "user", "content": native_natural_user(row)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    native_ids = list(tok.apply_chat_template(
        [{"role": "user", "content": native_natural_user(row)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    think_blocks = re.findall(r"<think>(.*?)</think>", native_text, flags=re.DOTALL)
    if any(content.strip() for content in think_blocks):
        raise RuntimeError("enable_thinking=False produced non-empty thinking content")
    if list(tok(native_text, add_special_tokens=False).input_ids) != native_ids:
        raise RuntimeError("Native rendered text/token IDs do not round-trip")

    forward_checks = []
    for name, token_ids in [("raw", raw_ids), ("native_non_thinking", native_ids)]:
        tensor = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=tensor, use_cache=False).logits[:, -1, :]
        finite = bool(torch.isfinite(logits).all().item())
        forward_checks.append({"interface": name, "prompt_tokens": len(token_ids), "last_logits_finite": finite})
        del tensor, logits
        if not finite:
            raise RuntimeError(f"Non-finite logits in {name} smoke forward")
    result = {
        "status": "PASS", "model_path": str(model_path),
        "architecture": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "dtype": str(next(model.parameters()).dtype),
        "model_eval": not model.training, "inference_mode": True,
        "enable_thinking": False,
        "empty_think_scaffold_present": bool(re.search(r"<think>\s*</think>", native_text, flags=re.DOTALL)),
        "forward_checks": forward_checks,
        "load_and_smoke_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json_atomic(OUT / "audit" / "model_load_smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
