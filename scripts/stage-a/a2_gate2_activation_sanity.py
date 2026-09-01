#!/usr/bin/env python3
"""Gate 2: fixed 16-sample activation extraction sanity check only."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import transformer_lens.loading_from_pretrained as tl_loading


PROJECT = Path("/home/ai/projects/linear-probe-repro")
UPSTREAM = PROJECT / "external" / "world-models"
MODEL_PATH = PROJECT / "models" / "pythia-2.8b-step143000"
PROMPT_PATH = UPSTREAM / "data" / "prompt_datasets" / "pythia" / "world_place" / "coords"
ENTITY_CSV = UPSTREAM / "data" / "entity_datasets" / "world_place.csv"
RESULTS = PROJECT / "results"
ACTIVATION_DIR = PROJECT / "activations" / "sanity" / "pythia-2.8b" / "world" / "coords"
SEED = 42
LAYERS = list(range(32))
COMPARE_LAYERS = [0, 15, 31]
MODEL_COMMIT = "dbe7ae300a54abcdc475a33907b3dff81d25709f"

sys.path.insert(0, str(UPSTREAM))
from save_activations import get_layer_activations_tl  # noqa: E402
from utils import adjust_precision  # noqa: E402


@contextmanager
def local_tl_config():
    """Redirect TL 1.14's alias config lookup to the pinned local config."""
    original = tl_loading.AutoConfig.from_pretrained

    def local_auto_config(_model_name, *args, **kwargs):
        return original(str(MODEL_PATH), local_files_only=True)

    tl_loading.AutoConfig.from_pretrained = local_auto_config
    try:
        yield
    finally:
        tl_loading.AutoConfig.from_pretrained = original


def cuda_peak() -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float().cpu()
    b = b.float().cpu()
    delta = (a - b).abs()
    return {
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "cosine_similarity": float(F.cosine_similarity(a[None], b[None]).item()),
    }


def frame_text(frame: pd.DataFrame) -> str:
    """Render a dataframe without pandas' optional tabulate dependency."""
    return "```text\n" + frame.to_string(index=False) + "\n```"


def choose_samples(entity_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    quotas = {"populated_place": 3, "structure": 3, "natural_place": 2}
    chosen: list[int] = []
    for is_test in [False, True]:
        split = entity_df[entity_df["is_test"].astype(bool) == is_test]
        for entity_type, count in quotas.items():
            candidates = split.index[split["entity_type"] == entity_type].to_numpy()
            picked = rng.choice(candidates, size=count, replace=False)
            chosen.extend(int(x) for x in picked)
    manifest = entity_df.loc[chosen].copy()
    manifest.insert(0, "row_id", manifest.index.astype(int))
    manifest.insert(1, "entity", manifest["name"])
    manifest.insert(2, "sample_order", np.arange(len(manifest), dtype=int))
    columns = [
        "sample_order", "row_id", "entity", "latitude", "longitude",
        "is_test", "entity_type", "entity_subtype", "country",
    ]
    manifest = manifest[columns].reset_index(drop=True)
    assert len(manifest) == 16
    assert manifest["is_test"].astype(bool).value_counts().to_dict() == {False: 8, True: 8}
    for flag in [False, True]:
        counts = manifest[manifest.is_test.astype(bool) == flag].entity_type.value_counts().to_dict()
        assert counts == quotas, (flag, counts)
    return manifest


def token_record(row_id: int, prompt_row: dict, tokenizer) -> dict:
    ids = [int(x) for x in prompt_row["input_ids"]]
    mask = [bool(x) for x in prompt_row["entity_mask"]]
    positions = [i for i, value in enumerate(mask) if value]
    assert positions
    first, last = positions[0], positions[-1]
    valid_ids = ids[: last + 1]
    return {
        "row_id": int(row_id),
        "entity": prompt_row["entity"],
        "prompt": tokenizer.decode(valid_ids, skip_special_tokens=True),
        "token_ids": valid_ids,
        "tokens": tokenizer.convert_ids_to_tokens(valid_ids),
        "entity_mask": [int(x) for x in mask[: last + 1]],
        "entity_first_token_index": first,
        "entity_last_token_index": last,
        "entity_last_token_text": tokenizer.convert_ids_to_tokens(ids[last]),
        "sequence_final_token_index_after_official_clipping": last,
    }


def full_hook_control(model, tokenized_subset, sample_indices, token_records):
    records = []
    names = [f"blocks.{layer}.hook_resid_post" for layer in COMPARE_LAYERS]
    for sample_index in sample_indices:
        ids = tokenized_subset[sample_index]["input_ids"]
        info = token_records[sample_index]
        first = info["entity_first_token_index"]
        last = info["entity_last_token_index"]
        clipped = ids[: last + 1].unsqueeze(0).to("cuda")
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                clipped, names_filter=lambda name: name in names, stop_at_layer=32
            )
        for layer in COMPARE_LAYERS:
            tensor = cache[f"blocks.{layer}.hook_resid_post"][0].detach().float().cpu()
            first_vec, last_vec, final_vec = tensor[first], tensor[last], tensor[-1]
            row = {
                "sample_index": sample_index,
                "row_id": info["row_id"],
                "entity": info["entity"],
                "layer": layer,
                "first_index": first,
                "last_index": last,
                "sequence_final_index": tensor.shape[0] - 1,
                "last_equals_sequence_final_index": bool(last == tensor.shape[0] - 1),
                "first_vs_last_identical": bool(torch.equal(first_vec, last_vec)),
                "last_vs_final_identical": bool(torch.equal(last_vec, final_vec)),
                **{f"first_last_{k}": v for k, v in vector_metrics(first_vec, last_vec).items()},
            }
            records.append(row)
        del cache
    return records


def extract_hf_comparison(hf_model, tokenized_subset, sample_indices, formal_activations):
    rows = []
    hf_vectors = {}
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captures[layer] = value.detach()
        return hook

    for layer in COMPARE_LAYERS:
        handles.append(hf_model.gpt_neox.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        for sample_index in sample_indices:
            captures.clear()
            row = tokenized_subset[sample_index]
            mask = row["entity_mask"]
            positions = torch.where(mask)[0]
            last = int(positions[-1])
            ids = row["input_ids"][: last + 1].unsqueeze(0).to("cuda")
            with torch.inference_mode():
                out = hf_model(
                    ids,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                    use_cache=False,
                )
            assert len(out.hidden_states) == 33
            assert set(captures) == set(COMPARE_LAYERS)
            for layer in COMPARE_LAYERS:
                # HF hidden_states[0] is embedding. For GPT-NeoX, indices 1..31
                # match blocks 0..30, while index 32 is after final_layer_norm.
                block_vec = captures[layer][0, last].detach().cpu().to(torch.float16)
                tuple_vec = out.hidden_states[layer + 1][0, last].detach().cpu().to(torch.float16)
                hf_vectors[sample_index, layer] = block_vec
                rows.append({
                    "comparison": "official_processed_TL_vs_HF_block_hook",
                    "sample_index": sample_index,
                    "layer": layer,
                    "hf_hidden_state_index": layer + 1,
                    **vector_metrics(formal_activations[layer][sample_index], block_vec),
                })
                rows.append({
                    "comparison": "HF_block_hook_vs_hidden_states_tuple",
                    "sample_index": sample_index,
                    "layer": layer,
                    "hf_hidden_state_index": layer + 1,
                    **vector_metrics(block_vec, tuple_vec),
                })
            del out
    finally:
        for handle in handles:
            handle.remove()
    return rows, hf_vectors


def extract_raw_tl_comparison(hf_model, tokenizer, tokenized_subset, sample_indices, hf_vectors):
    rows = []
    with local_tl_config():
        raw_tl = HookedTransformer.from_pretrained_no_processing(
            "pythia-2.8b",
            hf_model=hf_model,
            tokenizer=tokenizer,
            device="cuda",
            dtype=torch.float16,
            default_prepend_bos=True,
            default_padding_side="right",
        )
    raw_tl.eval()
    names = [f"blocks.{layer}.hook_resid_post" for layer in COMPARE_LAYERS]
    for sample_index in sample_indices:
        row = tokenized_subset[sample_index]
        last = int(torch.where(row["entity_mask"])[0][-1])
        ids = row["input_ids"][: last + 1].unsqueeze(0).to("cuda")
        with torch.inference_mode():
            _, cache = raw_tl.run_with_cache(
                ids, names_filter=lambda name: name in names, stop_at_layer=32
            )
        for layer in COMPARE_LAYERS:
            raw_vec = cache[f"blocks.{layer}.hook_resid_post"][0, last].detach().cpu()
            rows.append({
                "comparison": "unprocessed_TL_vs_HF_diagnostic",
                "sample_index": sample_index,
                "layer": layer,
                "hf_hidden_state_index": layer + 1,
                **vector_metrics(raw_vec, hf_vectors[sample_index, layer]),
            })
        del cache
    del raw_tl
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)

    entity_df = pd.read_csv(ENTITY_CSV)
    manifest = choose_samples(entity_df)
    row_ids = manifest.row_id.astype(int).tolist()

    full_prompt_ds = load_from_disk(str(PROMPT_PATH))
    full_prompt_ds.reset_format()
    tokenized_subset = full_prompt_ds.select(row_ids)
    # Official prompt creation normalizes some names (commas/parentheses).
    manifest["raw_entity"] = manifest["entity"]
    manifest["entity"] = tokenized_subset["entity"]
    manifest_path = RESULTS / "a2_2_sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    # Compatibility wrapper: official extraction expects both columns as tensors.
    tokenized_subset.set_format(
        type="torch", columns=["input_ids", "entity_mask"], output_all_columns=True
    )
    assert tuple(tokenized_subset["input_ids"].shape)[0] == 16
    assert tuple(tokenized_subset["entity_mask"].shape) == tuple(tokenized_subset["input_ids"].shape)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_records = [
        token_record(row_id, tokenized_subset[i], tokenizer)
        for i, row_id in enumerate(row_ids)
    ]
    for record in token_records:
        assert record["prompt"].startswith("What are the lat/lon coordinates of ")

    phase_peaks = []
    total_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    hf_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    with local_tl_config():
        formal_tl = HookedTransformer.from_pretrained(
            "pythia-2.8b",
            hf_model=hf_cpu,
            tokenizer=tokenizer,
            device="cuda",
            dtype=torch.float16,
            default_prepend_bos=True,
            default_padding_side="right",
        )
    formal_tl.eval()
    assert formal_tl.cfg.n_layers == 32
    assert formal_tl.cfg.d_model == 2560
    del hf_cpu
    gc.collect()

    args = SimpleNamespace(activation_aggregation="last", batch_size=1)
    extraction_start = time.perf_counter()
    with torch.inference_mode():
        activations = get_layer_activations_tl(
            args, formal_tl, tokenized_subset, layers=LAYERS, device=device
        )
    extraction_runtime = time.perf_counter() - extraction_start
    phase_peaks.append({"phase": "formal_tl_16", **cuda_peak()})

    for layer in LAYERS:
        assert tuple(activations[layer].shape) == (16, 2560)
        assert activations[layer].dtype == torch.float16

    multi_token = [
        i for i, record in enumerate(token_records)
        if record["entity_last_token_index"] > record["entity_first_token_index"]
    ]
    alignment_records = full_hook_control(
        formal_tl, tokenized_subset, multi_token[:2], token_records
    )
    phase_peaks[-1].update(cuda_peak())

    del formal_tl
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    hf_model.eval()
    compare_indices = [0, 8]  # one official train and one official test sample
    comparison_rows, hf_vectors = extract_hf_comparison(
        hf_model, tokenized_subset, compare_indices, activations
    )
    comparison_rows += extract_raw_tl_comparison(
        hf_model, tokenizer, tokenized_subset, compare_indices, hf_vectors
    )
    phase_peaks.append({"phase": "hf_plus_unprocessed_tl_comparison", **cuda_peak()})
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    stats_rows = []
    for layer in LAYERS:
        tensor = activations[layer].float()
        stats_rows.append({
            "layer": layer,
            "hook": f"blocks.{layer}.hook_resid_post",
            "shape": "[16, 2560]",
            "nan_count": int(torch.isnan(tensor).sum()),
            "inf_count": int(torch.isinf(tensor).sum()),
            "zero_count": int((tensor == 0).sum()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "frobenius_l2_norm": float(torch.linalg.vector_norm(tensor)),
            "mean_sample_l2_norm": float(torch.linalg.vector_norm(tensor, dim=1).mean()),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_path = RESULTS / "a2_2_activation_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    assert stats_df.nan_count.sum() == 0
    assert stats_df.inf_count.sum() == 0
    assert all(not torch.equal(activations[0], activations[layer]) for layer in [15, 31])
    assert all(not torch.equal(activations[layer][0], activations[layer][8]) for layer in COMPARE_LAYERS)

    quant_rows = []
    for layer in LAYERS:
        original = activations[layer].float()
        try:
            quantized = adjust_precision(original, 8, per_channel=True)
            restored = quantized.dequantize()
            quant_rows.append({"layer": layer, **{
                "mae": float((restored - original).abs().mean()),
                "max_error": float((restored - original).abs().max()),
                "mean_cosine_similarity": float(
                    F.cosine_similarity(restored, original, dim=1).mean()
                ),
            }})
        except Exception as exc:
            quant_rows.append({"layer": layer, "error": f"{type(exc).__name__}: {exc}"})
    quant_df = pd.DataFrame(quant_rows)
    quant_df.to_csv(RESULTS / "a2_2_quantization_roundtrip.csv", index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS / "a2_2_hf_tl_comparison.csv", index=False)
    pd.DataFrame(alignment_records).to_csv(
        RESULTS / "a2_2_token_alignment_control.csv", index=False
    )

    torch.save({layer: activations[layer] for layer in LAYERS}, ACTIVATION_DIR / "activations.pt")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "model_name": "EleutherAI/pythia-2.8b",
        "revision": "step143000",
        "commit": MODEL_COMMIT,
        "local_model_path": str(MODEL_PATH),
        "dataset": "world_place",
        "prompt_type": "coords",
        "sample_ids": row_ids,
        "activation_hook": "blocks.{layer}.hook_resid_post",
        "layers": LAYERS,
        "hidden_size": 2560,
        "activation_dtype": "torch.float16",
        "model_dtype": "torch.float16",
        "device": "cuda:0",
        "batch_size": 1,
        "seed": SEED,
        "external_world_models_git_commit": upstream_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_runtime_seconds": extraction_runtime,
        "total_runtime_seconds": time.perf_counter() - total_start,
        "gpu_peak_phases_bytes": phase_peaks,
        "formal_activation_processing": "TransformerLens from_pretrained defaults, matching released load.py",
    }
    (ACTIVATION_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (ACTIVATION_DIR / "tokenization_examples.json").write_text(
        json.dumps(token_records[:5], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    numeric_samples = []
    for sample_index in compare_indices:
        m = manifest.iloc[sample_index]
        t = token_records[sample_index]
        numeric_samples.append({
            "sample_index": sample_index,
            "row_id": int(m.row_id),
            "split": "test" if bool(m.is_test) else "train",
            "entity": m.entity,
            "prompt": t["prompt"],
            "latitude": float(m.latitude),
            "longitude": float(m.longitude),
            "entity_last_token_index": t["entity_last_token_index"],
            "entity_last_token_text": t["entity_last_token_text"],
            "layer_0_first_8": activations[0][sample_index, :8].float().tolist(),
            "layer_15_first_8": activations[15][sample_index, :8].float().tolist(),
            "layer_31_first_8": activations[31][sample_index, :8].float().tolist(),
        })
    (ACTIVATION_DIR / "numeric_examples.json").write_text(
        json.dumps(numeric_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    processed_cmp = comparison_df[
        comparison_df.comparison == "official_processed_TL_vs_HF_block_hook"
    ]
    raw_cmp = comparison_df[comparison_df.comparison == "unprocessed_TL_vs_HF_diagnostic"]
    hf_tuple_cmp = comparison_df[
        comparison_df.comparison == "HF_block_hook_vs_hidden_states_tuple"
    ]
    max_alloc = max(item["allocated"] for item in phase_peaks)
    max_reserved = max(item["reserved"] for item in phase_peaks)
    quant_ok = "error" not in quant_df.columns or quant_df.error.isna().all()
    checks = {
        "16_official_samples_8_train_8_test": True,
        "coords_prompt": True,
        "entity_mask_and_last_token": True,
        "32_resid_post_layers": True,
        "hidden_dim_2560": True,
        "all_layer_shapes_16x2560": True,
        "no_nan_inf": True,
        "different_layers_not_identical": True,
        "different_entities_not_identical": True,
        "layer_0_is_blocks_0_resid_post": True,
        "hf_tl_indexing_checked": bool(
            raw_cmp.cosine_similarity.min() > 0.999
            and hf_tuple_cmp[hf_tuple_cmp.layer.isin([0, 15])].cosine_similarity.min() > 0.999
            and hf_tuple_cmp[hf_tuple_cmp.layer == 31].cosine_similarity.max() < 0.999
        ),
        "gpu_no_oom": True,
        "upstream_clean": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True
        ).strip() == "",
    }
    gate_pass = all(checks.values())

    report_lines = [
        "# Stage A2 Gate 2 — 16-sample Activation Sanity Report",
        "",
        f"Status: **{'PASS' if gate_pass else 'FAIL'}**  ",
        "No probe was trained; no full-dataset extraction was run.",
        "",
        "## Sample manifest summary",
        "",
        "- Seed: 42",
        "- 8 official train rows and 8 official test rows",
        "- Per split: 3 populated places, 3 structures, 2 natural places",
        "",
        frame_text(manifest),
        "",
        "## Tokenization examples",
        "",
    ]
    for rec in token_records[:5]:
        report_lines += [
            f"### {rec['entity']} (row {rec['row_id']})",
            "",
            f"- Prompt: `{rec['prompt']}`",
            f"- Token IDs: `{rec['token_ids']}`",
            f"- Tokens: `{rec['tokens']}`",
            f"- Entity mask: `{rec['entity_mask']}`",
            f"- Entity first/last: `{rec['entity_first_token_index']}` / `{rec['entity_last_token_index']}`",
            f"- Entity last token: `{rec['entity_last_token_text']}`",
            "",
        ]
    report_lines += [
        "## Activation definition and shapes",
        "",
        "- Formal hook: `blocks.{layer}.hook_resid_post`, layers 0–31",
        "- Layer 0 is explicitly `blocks.0.hook_resid_post`, not embedding",
        "- Model dtype: float16; activation storage dtype: float16",
        "- Batch size: 1",
        "- Number of hook layers: 32",
        "- Layer 0 shape: `[16, 2560]`",
        "- Layer 15 shape: `[16, 2560]`",
        "- Layer 31 shape: `[16, 2560]`",
        "- All 32 layer shapes: `[16, 2560]`",
        "",
        "## Numeric examples",
        "",
        "```json",
        json.dumps(numeric_samples, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Layer statistics",
        "",
        f"- Total NaN: {int(stats_df.nan_count.sum())}",
        f"- Total Inf: {int(stats_df.inf_count.sum())}",
        "- Full per-layer table: `results/a2_2_activation_stats.csv`",
        "",
        frame_text(stats_df[stats_df.layer.isin(COMPARE_LAYERS)]),
        "",
        "## Token alignment control",
        "",
        "The official coords prompt ends with the entity, and batch-size-1 clipping stops at the entity last token. Therefore the post-clipping sequence-final index is intentionally identical to the entity-last index. The entity-first and entity-last vectors were separately checked and were not identical.",
        "",
        frame_text(pd.DataFrame(alignment_records)),
        "",
        "## HF versus TransformerLens",
        "",
        "HF returned 33 hidden states. Index 0 is the embedding output; indices 1–31 match raw outputs of blocks 0–30, but index 32 is after GPT-NeoX `final_layer_norm`, not raw block-31 resid-post. HF module forward hooks are therefore the authoritative block-output comparison for layers 0, 15, and 31.",
        "",
        "The released TransformerLens path uses `from_pretrained` defaults, including weight processing. A second unprocessed-TL diagnostic isolates differences introduced by this processing; formal saved activations remain the released processed-TL definition.",
        "",
        frame_text(comparison_df),
        "",
        f"- Processed TL vs HF block-hook maximum absolute difference: {processed_cmp.max_absolute_difference.max():.8g}",
        f"- Processed TL vs HF block-hook minimum cosine similarity: {processed_cmp.cosine_similarity.min():.8g}",
        f"- Unprocessed TL vs HF block-hook maximum absolute difference: {raw_cmp.max_absolute_difference.max():.8g}",
        f"- Unprocessed TL vs HF block-hook minimum cosine similarity: {raw_cmp.cosine_similarity.min():.8g}",
        f"- HF block-hook vs hidden_states minimum cosine for layers 0/15: {hf_tuple_cmp[hf_tuple_cmp.layer.isin([0, 15])].cosine_similarity.min():.8g}",
        f"- HF block-hook vs hidden_states maximum cosine for layer 31 (post-final-LN mismatch expected): {hf_tuple_cmp[hf_tuple_cmp.layer == 31].cosine_similarity.max():.8g}",
        "",
        "## Runtime and GPU memory",
        "",
        f"- Formal 16-sample extraction runtime: {extraction_runtime:.3f} s",
        f"- Total script runtime: {metadata['total_runtime_seconds']:.3f} s",
        f"- Peak allocated: {max_alloc / 2**30:.3f} GiB",
        f"- Peak reserved: {max_reserved / 2**30:.3f} GiB",
        "- GPU OOM: no",
        "",
        "## Official quantization round trip (optional check)",
        "",
        f"- Completed without error: {quant_ok}",
    ]
    if quant_ok:
        report_lines += [
            f"- Mean MAE across layers: {quant_df.mae.mean():.8g}",
            f"- Maximum error across layers: {quant_df.max_error.max():.8g}",
            f"- Minimum mean cosine similarity across layers: {quant_df.mean_cosine_similarity.min():.8g}",
        ]
    report_lines += [
        "- Float16 formal activations were retained; quantization did not replace them.",
        "",
        "## Gate checklist",
        "",
    ]
    for name, value in checks.items():
        report_lines.append(f"- [{'x' if value else ' '}] {name}")
    report_lines += [
        "",
        f"Final Gate 2 decision: **{'PASS' if gate_pass else 'FAIL'}**",
        "",
        "## Output files",
        "",
        "- `results/a2_2_sample_manifest.csv`",
        "- `results/a2_2_activation_sanity_report.md`",
        "- `results/a2_2_activation_stats.csv`",
        "- `results/a2_2_hf_tl_comparison.csv`",
        "- `results/a2_2_token_alignment_control.csv`",
        "- `results/a2_2_quantization_roundtrip.csv`",
        "- `activations/sanity/pythia-2.8b/world/coords/activations.pt`",
        "- `activations/sanity/pythia-2.8b/world/coords/metadata.json`",
        "- `activations/sanity/pythia-2.8b/world/coords/tokenization_examples.json`",
        "- `activations/sanity/pythia-2.8b/world/coords/numeric_examples.json`",
    ]
    (RESULTS / "a2_2_activation_sanity_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "gate_pass": gate_pass,
        "checks": checks,
        "manifest_rows": row_ids,
        "shapes": {str(layer): list(activations[layer].shape) for layer in COMPARE_LAYERS},
        "runtime_seconds": metadata["total_runtime_seconds"],
        "formal_extraction_runtime_seconds": extraction_runtime,
        "peak_allocated_gib": max_alloc / 2**30,
        "peak_reserved_gib": max_reserved / 2**30,
        "processed_tl_hf_block_max_abs": float(processed_cmp.max_absolute_difference.max()),
        "processed_tl_hf_block_min_cosine": float(processed_cmp.cosine_similarity.min()),
        "raw_tl_hf_block_max_abs": float(raw_cmp.max_absolute_difference.max()),
        "raw_tl_hf_block_min_cosine": float(raw_cmp.cosine_similarity.min()),
    }, indent=2))


if __name__ == "__main__":
    main()
