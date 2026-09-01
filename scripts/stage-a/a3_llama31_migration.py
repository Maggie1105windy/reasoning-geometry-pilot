#!/usr/bin/env python3
"""Stage A3: Llama-3.1-8B cross-architecture probing migration validation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT = Path("/home/ai/projects/linear-probe-repro")
MODEL_PATH = Path("/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B")
ENTITY_CSV = PROJECT / "external/world-models/data/entity_datasets/world_place.csv"
RESULTS = PROJECT / "results"
ACTIVATION_DIR = PROJECT / "activations/a3_migration/llama-3.1-8b/world/coords"
A2_SANITY_MANIFEST = RESULTS / "a2_2_sample_manifest.csv"
A2_MINI_MANIFEST = RESULTS / "a2_3_sample_manifest.csv"

SEED = 42
N_LAYERS = 32
HIDDEN_DIM = 4096
LAYERS = [0, 7, 15, 23, 31]
INDEX_LAYERS = [0, 15, 31]
N_TRAIN = 4000
N_TEST = 1000
FIXED_ALPHA = 4096.0
ALPHAS = np.array(sorted(set(np.logspace(1, 5, 17).tolist() + [FIXED_ALPHA])))
PREFIX = "What are the lat/lon coordinates of "
MODEL_MANIFEST_REVISION = "24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea"
TOKENIZER_MANIFEST_REVISION = "c52792af35fe806a76530f3028f037d1af4e3490"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frame_text(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    delta = (a - b).abs()
    return {
        "max_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "cosine_similarity": float(F.cosine_similarity(a[None], b[None]).item()),
    }


def cuda_record(phase: str, batch_size: int, runtime: float, status: str = "PASS") -> dict:
    return {
        "phase": phase,
        "batch_size": batch_size,
        "status": status,
        "runtime_seconds": runtime,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
    }


def build_manifest(source_path: Path, entity_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    row_ids = source.row_id.astype(int).tolist()
    official = entity_df.iloc[row_ids].copy().reset_index(drop=True)
    manifest = pd.DataFrame({
        "sample_order": np.arange(len(source), dtype=np.int64),
        "row_id": row_ids,
        "entity": source.entity.astype(str),
        "raw_entity": official.name.astype(str),
        "longitude": official.longitude.astype(float),
        "latitude": official.latitude.astype(float),
        "is_test": official.is_test.astype(bool),
        "original_category": official.entity_type.astype(str),
        "entity_subtype": official.entity_subtype.astype(str),
        "country": official.country.astype(str),
    })
    assert manifest.row_id.tolist() == row_ids
    assert np.allclose(manifest.longitude, source.longitude)
    assert np.allclose(manifest.latitude, source.latitude)
    assert manifest.is_test.astype(bool).tolist() == source.is_test.astype(bool).tolist()
    manifest.to_csv(output_path, index=False)
    return manifest


def tokenize_manifest(manifest: pd.DataFrame, tokenizer):
    rows = []
    max_len = 0
    prefix_ids = [
        int(x) for x in tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ]
    for sample_order, entity in enumerate(manifest.entity.astype(str)):
        prompt = PREFIX + entity
        entity_start = len(PREFIX)
        encoded = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True,
            return_attention_mask=False,
        )
        prompt_ids = [int(x) for x in encoded["input_ids"]]
        common_prefix = 0
        for prefix_id, prompt_id in zip(prefix_ids, prompt_ids):
            if prefix_id != prompt_id:
                break
            common_prefix += 1
        # The entity occupies the token suffix after the fixed prompt prefix.
        # If BPE merges across the prefix/entity boundary, the first differing
        # token is deliberately included in the entity span.
        assert common_prefix < len(prompt_ids)
        entity_first_no_bos = common_prefix
        decoded_entity_suffix = tokenizer.decode(
            prompt_ids[entity_first_no_bos:], skip_special_tokens=False
        ).strip()
        canonical_entity = tokenizer.decode(
            tokenizer(entity, add_special_tokens=False)["input_ids"],
            skip_special_tokens=False,
        ).strip()
        normalize = lambda value: unicodedata.normalize("NFKC", value).strip()
        assert normalize(decoded_entity_suffix) == normalize(canonical_entity), (
            entity, canonical_entity, decoded_entity_suffix, common_prefix, len(prefix_ids)
        )

        ids = [tokenizer.bos_token_id] + prompt_ids
        offsets = [(0, 0)] + [tuple(map(int, x)) for x in encoded["offset_mapping"]]
        offset_mask = [False] + [
            end > entity_start and start < len(prompt) for start, end in offsets[1:]
        ]
        offset_positions = [i for i, value in enumerate(offset_mask) if value]
        entity_mask = (
            [False]
            + [False] * entity_first_no_bos
            + [True] * (len(prompt_ids) - entity_first_no_bos)
        )
        positions = [i for i, value in enumerate(entity_mask) if value]
        assert positions
        first, last = positions[0], positions[-1]
        assert last == len(ids) - 1  # prompt ends with entity, but extraction still uses the mask.
        rows.append({
            "sample_order": sample_order,
            "row_id": int(manifest.iloc[sample_order].row_id),
            "entity": entity,
            "prompt": prompt,
            "alignment_method": "fixed-prefix token-ID longest common prefix",
            "prefix_token_count": len(prefix_ids),
            "prefix_prompt_common_token_count": common_prefix,
            "boundary_token_merged": common_prefix < len(prefix_ids),
            "decoded_entity_suffix": decoded_entity_suffix,
            "canonical_tokenized_entity": canonical_entity,
            "decoded_entity_matches": (
                normalize(decoded_entity_suffix) == normalize(canonical_entity)
            ),
            "offset_mapping_usable": bool(offset_positions),
            "offset_mapping_agrees": bool(offset_positions == positions),
            "input_ids": ids,
            "offsets": offsets,
            "entity_mask": entity_mask,
            "entity_first_token_index": first,
            "entity_last_token_index": last,
            "entity_last_token_id": ids[last],
            "entity_last_token": tokenizer.convert_ids_to_tokens(ids[last]),
            "tokens": tokenizer.convert_ids_to_tokens(ids),
        })
        max_len = max(max_len, len(ids))

    input_ids = torch.full(
        (len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    entity_mask_tensor = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[i, :length] = 1
        entity_mask_tensor[i, :length] = torch.tensor(row["entity_mask"], dtype=torch.bool)
    assert torch.equal(entity_mask_tensor.sum(1) > 0, torch.ones(len(rows), dtype=torch.bool))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "entity_mask": entity_mask_tensor,
        "records": rows,
    }


def selected_vectors(hidden: torch.Tensor, last_indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, last_indices]


def extract_raw_hooks(model, tokenized, layers, batch_size, phase):
    n_rows = tokenized["input_ids"].shape[0]
    outputs = {
        layer: torch.empty((n_rows, HIDDEN_DIM), dtype=torch.float16) for layer in layers
    }
    state = {"last_indices": None, "captures": {}}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            state["captures"][layer] = selected_vectors(
                hidden, state["last_indices"]
            ).detach().cpu().to(torch.float16)
        return hook

    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        for start_row in range(0, n_rows, batch_size):
            end_row = min(start_row + batch_size, n_rows)
            ids = tokenized["input_ids"][start_row:end_row]
            attention = tokenized["attention_mask"][start_row:end_row]
            entity_mask = tokenized["entity_mask"][start_row:end_row]
            max_valid = int(attention.sum(1).max())
            ids = ids[:, :max_valid].to("cuda")
            attention = attention[:, :max_valid].to("cuda")
            entity_mask = entity_mask[:, :max_valid].to("cuda")
            positions = torch.arange(max_valid, device="cuda")[None, :].expand_as(entity_mask)
            state["last_indices"] = positions.masked_fill(~entity_mask, -1).max(1).values
            assert (state["last_indices"] >= 0).all()
            state["captures"] = {}
            with torch.inference_mode():
                model(
                    input_ids=ids, attention_mask=attention, use_cache=False,
                    output_hidden_states=False, return_dict=True,
                )
            assert set(state["captures"]) == set(layers)
            for layer in layers:
                outputs[layer][start_row:end_row] = state["captures"][layer]
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        for layer in layers:
            assert tuple(outputs[layer].shape) == (n_rows, HIDDEN_DIM)
            assert torch.isfinite(outputs[layer]).all()
        return outputs, cuda_record(phase, batch_size, runtime)
    finally:
        for handle in handles:
            handle.remove()


def batch_trial(model, tokenized, batch_size):
    subset = {
        key: value[:batch_size] if isinstance(value, torch.Tensor) else value[:batch_size]
        for key, value in tokenized.items()
    }
    try:
        acts, record = extract_raw_hooks(
            model, subset, INDEX_LAYERS, batch_size,
            f"batch_trial_{batch_size}",
        )
        del acts
        return record
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return cuda_record(f"batch_trial_{batch_size}", batch_size, 0.0, "OOM")


def indexing_validation(model, tokenized, manifest):
    sample_indices = [0, 8]
    ids = tokenized["input_ids"][sample_indices]
    attention = tokenized["attention_mask"][sample_indices]
    entity_mask = tokenized["entity_mask"][sample_indices]
    max_valid = int(attention.sum(1).max())
    ids = ids[:, :max_valid].to("cuda")
    attention = attention[:, :max_valid].to("cuda")
    entity_mask = entity_mask[:, :max_valid].to("cuda")
    positions = torch.arange(max_valid, device="cuda")[None, :].expand_as(entity_mask)
    last_indices = positions.masked_fill(~entity_mask, -1).max(1).values
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures[layer] = selected_vectors(hidden, last_indices).detach()
        return hook

    for layer in INDEX_LAYERS:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            result = model(
                input_ids=ids, attention_mask=attention, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )
        assert len(result.hidden_states) == 33
        rows = []
        for local_index, sample_index in enumerate(sample_indices):
            for layer in INDEX_LAYERS:
                raw = captures[layer][local_index]
                tuple_index = layer + 1
                tuple_vec = selected_vectors(
                    result.hidden_states[tuple_index], last_indices
                )[local_index]
                rows.append({
                    "sample_index": sample_index,
                    "row_id": int(manifest.iloc[sample_index].row_id),
                    "entity": manifest.iloc[sample_index].entity,
                    "layer": layer,
                    "comparison": "raw_decoder_hook_vs_hidden_states_tuple",
                    "hf_hidden_states_index": tuple_index,
                    **vector_metrics(raw, tuple_vec),
                })
                if layer == 31:
                    post_norm = selected_vectors(
                        result.hidden_states[-1], last_indices
                    )[local_index]
                    rows.append({
                        "sample_index": sample_index,
                        "row_id": int(manifest.iloc[sample_index].row_id),
                        "entity": manifest.iloc[sample_index].entity,
                        "layer": layer,
                        "comparison": "raw_decoder_hook_vs_post_final_RMSNorm",
                        "hf_hidden_states_index": 32,
                        **vector_metrics(raw, post_norm),
                    })
                    normalized_raw = model.model.norm(raw[None])[0]
                    rows.append({
                        "sample_index": sample_index,
                        "row_id": int(manifest.iloc[sample_index].row_id),
                        "entity": manifest.iloc[sample_index].entity,
                        "layer": layer,
                        "comparison": "RMSNorm_raw_hook_vs_hidden_states_32",
                        "hf_hidden_states_index": 32,
                        **vector_metrics(normalized_raw, post_norm),
                    })
        del result
        return pd.DataFrame(rows)
    finally:
        for handle in handles:
            handle.remove()


def score(target, pred):
    return {
        "r2_overall": float(r2_score(target, pred)),
        "r2_longitude": float(r2_score(target[:, 0], pred[:, 0])),
        "r2_latitude": float(r2_score(target[:, 1], pred[:, 1])),
    }


def fit_probe(probe, x_train, x_test, y_train_norm, y_test, mean, std):
    start = time.perf_counter()
    probe.fit(x_train, y_train_norm)
    pred = probe.predict(x_test) * std + mean
    runtime = time.perf_counter() - start
    coef = np.asarray(probe.coef_)
    intercept = np.asarray(probe.intercept_)
    assert coef.shape == (2, HIDDEN_DIM) and intercept.shape == (2,)
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    return score(y_test, pred), runtime, {
        "coef_shape": str(list(coef.shape)),
        "intercept_shape": str(list(intercept.shape)),
        "coef_l2_norm": float(np.linalg.norm(coef)),
        "coef_nan_count": int(np.isnan(coef).sum()),
        "coef_inf_count": int(np.isinf(coef).sum()),
    }


def plot_curves(fixed, cv, baseline):
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    ax.plot(fixed.layer, fixed.r2_overall, marker="o", linewidth=2,
            label="Ridge(alpha=4096; primary)")
    ax.plot(cv.layer, cv.r2_overall, marker="s", linewidth=2,
            label="RidgeCV (our sensitivity)")
    ax.axhline(baseline["r2_overall"], color="0.45", linestyle="--",
               label="Train-mean baseline")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Transformer layer (raw decoder block output)")
    ax.set_ylabel("Held-out overall R²")
    ax.set_title("Llama-3.1-8B World coords migration sanity probe")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "a3_llama31_layer_r2.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 20 * 2**30:
        raise RuntimeError(f"only {free_bytes / 2**30:.2f} GiB GPU memory free")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    started = utc_now()

    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    assert config.num_hidden_layers == N_LAYERS and config.hidden_size == HIDDEN_DIM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    entity_df = pd.read_csv(ENTITY_CSV)
    sanity_manifest = build_manifest(
        A2_SANITY_MANIFEST, entity_df, RESULTS / "a3_sanity_sample_manifest.csv"
    )
    assert sanity_manifest.is_test.astype(bool).value_counts().to_dict() == {False: 8, True: 8}
    mini_manifest = build_manifest(
        A2_MINI_MANIFEST, entity_df, RESULTS / "a3_sample_manifest.csv"
    )
    assert mini_manifest.row_id.tolist() == pd.read_csv(A2_MINI_MANIFEST).row_id.astype(int).tolist()
    assert mini_manifest.is_test.astype(bool).value_counts().to_dict() == {False: 4000, True: 1000}
    sanity_tokens = tokenize_manifest(sanity_manifest, tokenizer)
    mini_tokens = tokenize_manifest(mini_manifest, tokenizer)

    if os.environ.get("A3_TOKENIZE_ONLY") == "1":
        print(json.dumps({
            "rows": len(mini_tokens["records"]),
            "decoded_suffix_matches": sum(
                r["decoded_entity_matches"] for r in mini_tokens["records"]
            ),
            "offset_mapping_usable": sum(
                r["offset_mapping_usable"] for r in mini_tokens["records"]
            ),
            "offset_mapping_unusable": sum(
                not r["offset_mapping_usable"] for r in mini_tokens["records"]
            ),
            "offset_mapping_agrees": sum(
                r["offset_mapping_agrees"] for r in mini_tokens["records"]
            ),
            "boundary_token_merged": sum(
                r["boundary_token_merged"] for r in mini_tokens["records"]
            ),
        }, indent=2))
        return

    multi = [r for r in sanity_tokens["records"]
             if r["entity_last_token_index"] > r["entity_first_token_index"]]
    assert len(multi) >= 10
    token_examples = multi[:10]
    (RESULTS / "a3_tokenization_examples.json").write_text(
        json.dumps(token_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    model_load_runtime = time.perf_counter() - model_load_start
    assert sum(p.numel() for p in model.parameters()) == 8030261248
    assert len(model.model.layers) == N_LAYERS
    assert next(model.parameters()).dtype == torch.bfloat16

    memory_records = []
    for batch_size in [1, 2, 4]:
        record = batch_trial(model, sanity_tokens, batch_size)
        memory_records.append(record)
        if record["status"] != "PASS":
            if batch_size == 1:
                raise RuntimeError("BF16 batch size 1 OOM; stop A3")
            break
    passed_batches = [x["batch_size"] for x in memory_records if x["status"] == "PASS"]
    chosen_batch = max(passed_batches)
    if memory_records[-1]["peak_reserved_gib"] > 22.0:
        chosen_batch = min(chosen_batch, 2)
    pd.DataFrame(memory_records).to_csv(RESULTS / "a3_gpu_batch_trials.csv", index=False)

    indexing_df = indexing_validation(model, sanity_tokens, sanity_manifest)
    indexing_df.to_csv(RESULTS / "a3_hidden_state_indexing.csv", index=False)
    tuple_raw = indexing_df[indexing_df.comparison == "raw_decoder_hook_vs_hidden_states_tuple"]
    final_raw = indexing_df[indexing_df.comparison == "raw_decoder_hook_vs_post_final_RMSNorm"]
    renorm = indexing_df[indexing_df.comparison == "RMSNorm_raw_hook_vs_hidden_states_32"]
    assert tuple_raw[tuple_raw.layer.isin([0, 15])].cosine_similarity.min() > 0.99999
    # RMSNorm can preserve direction almost perfectly when learned weights are
    # near-uniform, so non-identity is established by absolute differences;
    # applying the model norm to the raw hook must reconstruct tuple index 32.
    assert final_raw.max_absolute_difference.min() > 1e-3
    assert final_raw.mean_absolute_difference.min() > 1e-5
    assert renorm.cosine_similarity.min() > 0.99999

    sanity_acts, sanity_record = extract_raw_hooks(
        model, sanity_tokens, list(range(N_LAYERS)), 1, "sanity_16_all_32_layers"
    )
    memory_records.append(sanity_record)
    sanity_stats = []
    for layer in range(N_LAYERS):
        tensor = sanity_acts[layer].float()
        sanity_stats.append({
            "layer": layer,
            "activation": f"model.layers.{layer} raw output before final RMSNorm",
            "shape": str(list(tensor.shape)),
            "nan_count": int(torch.isnan(tensor).sum()),
            "inf_count": int(torch.isinf(tensor).sum()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "frobenius_l2_norm": float(torch.linalg.vector_norm(tensor)),
            "mean_sample_l2_norm": float(torch.linalg.vector_norm(tensor, dim=1).mean()),
        })
    sanity_stats_df = pd.DataFrame(sanity_stats)
    sanity_stats_df.to_csv(RESULTS / "a3_activation_sanity_stats.csv", index=False)
    assert sanity_stats_df.nan_count.sum() == 0 and sanity_stats_df.inf_count.sum() == 0
    assert all(tuple(sanity_acts[layer].shape) == (16, HIDDEN_DIM) for layer in range(N_LAYERS))

    sanity_report = [
        "# Stage A3 — 16-sample Activation Sanity Report", "",
        "Status: **PASS**", "",
        "- 8 official train and 8 official test rows; Gate 2 row IDs reused",
        "- Llama-3.1 tokenizer with explicit BOS, EOS pad assignment, right padding",
        "- Explicit fixed-prefix token-boundary entity mask; extraction uses entity last-token index",
        "- Fast-tokenizer offsets are retained only as diagnostics because 12/5,000 rows expose zero-length entity offsets",
        "- Formal activation: raw output of every decoder block before final RMSNorm",
        "- All 32 shapes: `[16, 4096]`; float16 CPU storage",
        "- NaN/Inf: 0/0", "",
        "## Indexing validation", "", frame_text(indexing_df), "",
        "HF `hidden_states[0]` is embedding output; indices 1-31 are raw outputs of blocks 0-30; index 32 is raw block-31 after final RMSNorm. Raw block 31 is obtained by its decoder-layer hook. Formal activations use hooks for every layer, so no normalized/raw mixture occurs.",
        "", "## Ten multi-token entity examples", "",
    ]
    for row in token_examples:
        sanity_report += [
            f"### {row['entity']} (row {row['row_id']})", "",
            f"- Prompt: `{row['prompt']}`",
            f"- Tokens: `{row['tokens']}`",
            f"- Entity span: `{row['entity_first_token_index']}..{row['entity_last_token_index']}`",
            f"- Entity last token: `{row['entity_last_token']}`", "",
        ]
    sanity_report += [
        "## Representative activation statistics", "",
        frame_text(sanity_stats_df[sanity_stats_df.layer.isin(INDEX_LAYERS)]), "",
        "## GPU trials", "", frame_text(pd.DataFrame(memory_records)),
    ]
    (RESULTS / "a3_activation_sanity_report.md").write_text(
        "\n".join(sanity_report) + "\n", encoding="utf-8"
    )

    # Sanity gate has passed; only now run the bounded 4,000/1,000 extraction.
    mini_acts, extraction_record = extract_raw_hooks(
        model, mini_tokens, LAYERS, chosen_batch, "mini_extraction_5000_five_layers"
    )
    memory_records.append(extraction_record)
    torch.save({layer: mini_acts[layer] for layer in LAYERS}, ACTIVATION_DIR / "activations.pt")
    del model, sanity_acts
    gc.collect()
    torch.cuda.empty_cache()

    target = mini_manifest[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    is_test = mini_manifest.is_test.astype(bool).to_numpy()
    y_train, y_test = target[~is_test], target[is_test]
    mean, std = y_train.mean(0), y_train.std(0)
    y_train_norm = (y_train - mean) / std
    scaler = {
        "target_order": ["longitude", "latitude"], "ddof": 0,
        "fit_split": "mini official is_test=False only", "fit_rows": N_TRAIN,
        "longitude_mean": float(mean[0]), "longitude_std": float(std[0]),
        "latitude_mean": float(mean[1]), "latitude_std": float(std[1]),
    }
    (RESULTS / "a3_target_scaler.json").write_text(
        json.dumps(scaler, indent=2) + "\n", encoding="utf-8"
    )
    baseline = score(y_test, np.broadcast_to(mean, y_test.shape))
    fixed_rows, cv_rows, coef_rows = [], [], []
    x_cache = {}
    with threadpool_limits(limits=min(16, os.cpu_count() or 1)):
        for layer in LAYERS:
            matrix = mini_acts[layer].float().numpy()
            x_train, x_test = matrix[~is_test], matrix[is_test]
            assert x_train.shape == (N_TRAIN, HIDDEN_DIM)
            assert x_test.shape == (N_TEST, HIDDEN_DIM)
            x_cache[layer] = (x_train, x_test)
            fixed = Ridge(alpha=FIXED_ALPHA)
            fixed_scores, fixed_runtime, coef = fit_probe(
                fixed, x_train, x_test, y_train_norm, y_test, mean, std
            )
            fixed_rows.append({
                "layer": layer, "relative_depth": layer / 31.0,
                "n_train": N_TRAIN, "n_test": N_TEST,
                "hidden_dim": HIDDEN_DIM, "alpha": FIXED_ALPHA,
                **fixed_scores,
                "baseline_r2_overall": baseline["r2_overall"],
                "runtime_seconds": fixed_runtime,
            })
            coef_rows.append({"analysis": "fixed_alpha", "layer": layer, **coef})
            cv = RidgeCV(alphas=ALPHAS, store_cv_values=True)
            cv_scores, cv_runtime, cv_coef = fit_probe(
                cv, x_train, x_test, y_train_norm, y_test, mean, std
            )
            cv_rows.append({
                "layer": layer, "selected_alpha": float(cv.alpha_),
                "r2_overall": cv_scores["r2_overall"],
                "r2_longitude": cv_scores["r2_longitude"],
                "r2_latitude": cv_scores["r2_latitude"],
                "runtime_seconds": cv_runtime,
                "analysis_label": "OUR SENSITIVITY ANALYSIS",
            })
            coef_rows.append({"analysis": "ridgecv_sensitivity", "layer": layer, **cv_coef})

        permutation = np.random.default_rng(SEED).permutation(N_TRAIN)
        x15_train, x15_test = x_cache[15]
        shuffled = Ridge(alpha=FIXED_ALPHA)
        shuffled_scores, shuffled_runtime, shuffled_coef = fit_probe(
            shuffled, x15_train, x15_test, y_train_norm[permutation],
            y_test, mean, std,
        )
        coef_rows.append({"analysis": "layer15_shuffled_train_labels", "layer": 15, **shuffled_coef})

    fixed_df = pd.DataFrame(fixed_rows)
    cv_df = pd.DataFrame(cv_rows)
    coef_df = pd.DataFrame(coef_rows)
    fixed_df.to_csv(RESULTS / "a3_llama31_mini_probe_fixed_alpha.csv", index=False)
    cv_df.to_csv(RESULTS / "a3_llama31_mini_probe_ridgecv.csv", index=False)
    coef_df.to_csv(RESULTS / "a3_probe_coefficients.csv", index=False)
    control = {
        "baseline": baseline,
        "random_label": {
            "layer": 15, "seed": SEED, "alpha": FIXED_ALPHA,
            "shuffle_scope": "training labels only", "test_labels_unchanged": True,
            **shuffled_scores,
            "normal_r2_overall": float(fixed_df.loc[fixed_df.layer == 15, "r2_overall"].iloc[0]),
            "drop_vs_normal": float(fixed_df.loc[fixed_df.layer == 15, "r2_overall"].iloc[0] - shuffled_scores["r2_overall"]),
            "runtime_seconds": shuffled_runtime,
        },
    }
    (RESULTS / "a3_controls.json").write_text(
        json.dumps(control, indent=2) + "\n", encoding="utf-8"
    )
    plot_curves(fixed_df, cv_df, baseline)

    fixed_cv_spearman = float(spearmanr(fixed_df.r2_overall, cv_df.r2_overall).statistic)
    checks = {
        "model_config_audited": True,
        "layers_32_hidden_4096_bf16_parameters_exact": True,
        "hidden_state_indexing_explained": True,
        "raw_block_vs_final_RMSNorm_distinguished": True,
        "uniform_raw_hook_activation_definition": True,
        "entity_last_token_explicit_mask": True,
        "entity_span_prefix_boundary_validated": all(
            r["decoded_entity_matches"] for r in mini_tokens["records"]
        ),
        "ten_multi_token_examples": len(token_examples) == 10,
        "sanity_16_all_shapes_16x4096": True,
        "sanity_no_nan_inf": True,
        "gpu_no_oom_bf16": all(x["status"] == "PASS" for x in memory_records),
        "mini_exact_4000_1000_official_split": bool((~is_test).sum() == 4000 and is_test.sum() == 1000),
        "mini_row_ids_equal_a2_gate3": mini_manifest.row_id.tolist() == pd.read_csv(A2_MINI_MANIFEST).row_id.astype(int).tolist(),
        "five_independent_fixed_ridges": len(fixed_df) == 5 and fixed_df.layer.tolist() == LAYERS,
        "heldout_r2_finite": bool(np.isfinite(fixed_df[["r2_overall", "r2_longitude", "r2_latitude"]]).all().all()),
        "fixed_vs_ridgecv_trend_not_conflicting": fixed_cv_spearman > 0.8,
        "baseline_near_no_prediction": abs(baseline["r2_overall"]) < 0.05,
        "random_label_materially_degraded": control["random_label"]["drop_vs_normal"] > 0.05,
        "coefficient_shapes_and_values_valid": bool(
            coef_df.coef_shape.eq("[2, 4096]").all()
            and coef_df.intercept_shape.eq("[2]").all()
            and coef_df.coef_nan_count.eq(0).all()
            and coef_df.coef_inf_count.eq(0).all()
        ),
        "layers_not_all_identical": fixed_df.r2_overall.nunique() > 1,
        "external_world_models_clean": subprocess.check_output(
            ["git", "-C", str(PROJECT / "external/world-models"), "status", "--porcelain"], text=True
        ).strip() == "",
    }
    gate_pass = all(checks.values())
    if not gate_pass:
        raise RuntimeError(f"A3 gate failed: {[k for k,v in checks.items() if not v]}")

    pythia = pd.read_csv(RESULTS / "a2_3_mini_probe_fixed_alpha.csv")
    pythia = pythia[pythia.layer.isin(LAYERS)][["layer", "r2_overall"]].rename(
        columns={"r2_overall": "pythia_gate3_r2"}
    )
    migration_comparison = fixed_df[["layer", "r2_overall"]].rename(
        columns={"r2_overall": "llama31_a3_r2"}
    ).merge(pythia, on="layer")
    migration_comparison["difference_llama_minus_pythia"] = (
        migration_comparison.llama31_a3_r2 - migration_comparison.pythia_gate3_r2
    )
    migration_comparison.to_csv(RESULTS / "a3_pythia_sanity_reference.csv", index=False)

    all_memory = pd.DataFrame(memory_records)
    all_memory.to_csv(RESULTS / "a3_gpu_memory.csv", index=False)
    metadata = {
        "status": "PASS",
        "experiment": "Llama-3.1-8B linear-probe migration sanity experiment",
        "model_path": str(MODEL_PATH),
        "model_manifest_revision": MODEL_MANIFEST_REVISION,
        "tokenizer_manifest_revision": TOKENIZER_MANIFEST_REVISION,
        "config": config.to_dict(),
        "parameter_count": 8030261248,
        "model_storage_dtype": "BF16", "model_runtime_dtype": "torch.bfloat16",
        "tokenizer_class": type(tokenizer).__name__,
        "prompt": PREFIX + "<entity>",
        "entity_alignment": {
            "method": "fixed-prefix token-ID longest common prefix",
            "validation": "decoded token suffix equals entity for every row",
            "offset_mapping_usable_rows": int(sum(r["offset_mapping_usable"] for r in mini_tokens["records"])),
            "offset_mapping_unusable_rows": int(sum(not r["offset_mapping_usable"] for r in mini_tokens["records"])),
            "offset_mapping_agreement_rows": int(sum(r["offset_mapping_agrees"] for r in mini_tokens["records"])),
        },
        "activation_definition": "raw decoder-layer output after each block, before final model RMSNorm",
        "formal_extraction": "forward hooks on model.model.layers[layer] for every selected layer",
        "hf_hidden_state_indexing": {
            "0": "embedding output", "1..31": "raw blocks 0..30 outputs",
            "32": "raw block 31 output after final RMSNorm",
            "raw_block_31": "decoder-layer 31 forward hook",
        },
        "layers": LAYERS, "hidden_size": HIDDEN_DIM,
        "sample_count": 5000, "train_count": N_TRAIN, "test_count": N_TEST,
        "row_ids_equal_a2_gate3": True,
        "activation_storage_dtype": "torch.float16",
        "sklearn_input_dtype": "float32",
        "batch_size": chosen_batch, "seed": SEED,
        "model_load_runtime_seconds": model_load_runtime,
        "mini_extraction_runtime_seconds": extraction_record["runtime_seconds"],
        "gpu_records": memory_records,
        "fixed_alpha": FIXED_ALPHA, "ridgecv_alphas": ALPHAS.tolist(),
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "started_utc": started, "ended_utc": utc_now(),
        "total_runtime_seconds": time.perf_counter() - total_start,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks,
    }
    (ACTIVATION_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Stage A3 — Llama-3.1-8B Migration Report", "",
        "Status: **PASS**", "",
        "Experiment: **Llama-3.1-8B linear-probe migration sanity experiment**. This is a cross-architecture pipeline validation, not a Figure 2 reproduction.", "",
        "## Model", "",
        "- 32 decoder layers; hidden size 4096; 8,030,261,248 parameters",
        "- All stored weights BF16; runtime BF16; no weight quantization",
        f"- Read-only path: `{MODEL_PATH}`", "",
        "## Hidden-state indexing and formal activation", "",
        frame_text(indexing_df), "",
        "`hidden_states[0]` is embeddings. Indices 1-31 are raw outputs of blocks 0-30. Index 32 is block-31 output after final RMSNorm, so it cannot serve as raw block 31. Formal A3 activations use identical decoder forward hooks for all layers, yielding raw post-block/pre-final-RMSNorm representations.", "",
        "## Token/entity alignment examples", "",
        "The entity span is derived from the token-ID boundary between the fixed prompt prefix and the complete prompt. The decoded suffix was required to equal the entity on all 5,000 rows. Offset mappings are diagnostic only because a small set of single-token names receive zero-length offsets from this local tokenizer artifact.", "",
    ]
    for row in token_examples[:2]:
        report += [
            f"### {row['entity']}", "",
            f"- Prompt: `{row['prompt']}`",
            f"- Tokens: `{row['tokens']}`",
            f"- Entity token span: `{row['entity_first_token_index']}..{row['entity_last_token_index']}`",
            f"- Entity last token: `{row['entity_last_token']}`", "",
        ]
    report += [
        "## 16-sample sanity", "",
        "All 32 raw-block activation matrices have shape `[16, 4096]`, with zero NaN/Inf.", "",
        frame_text(sanity_stats_df[sanity_stats_df.layer.isin(INDEX_LAYERS)]), "",
        "## GPU and extraction", "",
        frame_text(all_memory), "",
        f"- Chosen mini batch size: {chosen_batch}",
        f"- 5,000-row five-layer extraction runtime: {extraction_record['runtime_seconds']:.3f} seconds",
        f"- Extraction peak allocated/reserved: {extraction_record['peak_allocated_gib']:.3f} / {extraction_record['peak_reserved_gib']:.3f} GiB", "",
        "## Primary Ridge(alpha=4096)", "",
        frame_text(fixed_df), "",
        "Alpha equals activation dimension as a unified migration rule; it is not a Gurnee & Tegmark Llama-3.1 setting.", "",
        "## RidgeCV — OUR SENSITIVITY ANALYSIS", "",
        frame_text(cv_df), "",
        f"Fixed-vs-CV five-point Spearman: {fixed_cv_spearman:.6f}", "",
        "## Controls", "", "```json", json.dumps(control, indent=2), "```", "",
        "## Pythia Gate 3 sanity reference", "",
        frame_text(migration_comparison), "",
        "The two architectures both support the same probing methodology and show depth-dependent decodability. Differences are not interpreted as a controlled model-capability or reasoning comparison because architecture, model scale, tokenizer, hidden size, and pretraining data differ.", "",
        "## PASS checklist", "",
    ]
    for key, value in checks.items():
        report.append(f"- [{'x' if value else ' '}] {key}")
    report += [
        "", "## Outputs", "",
        "- `results/a3_llama31_model_audit.md`",
        "- `results/a3_activation_sanity_report.md`",
        "- `results/a3_hidden_state_indexing.csv`",
        "- `results/a3_sample_manifest.csv`",
        "- `results/a3_target_scaler.json`",
        "- `results/a3_llama31_mini_probe_fixed_alpha.csv`",
        "- `results/a3_llama31_mini_probe_ridgecv.csv`",
        "- `results/a3_llama31_layer_r2.png`",
        "- `activations/a3_migration/llama-3.1-8b/world/coords/activations.pt`",
        "- `activations/a3_migration/llama-3.1-8b/world/coords/metadata.json`",
        "", "No Tulu, reasoning, grounding, or full 39,585 × 32-layer Llama experiment was run.",
    ]
    (RESULTS / "A3_LLAMA31_MIGRATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "status": "PASS", "chosen_batch_size": chosen_batch,
        "model_load_runtime_seconds": model_load_runtime,
        "mini_extraction": extraction_record,
        "fixed_results": fixed_df.to_dict(orient="records"),
        "ridgecv_results": cv_df.to_dict(orient="records"),
        "controls": control, "fixed_cv_spearman": fixed_cv_spearman,
        "total_runtime_seconds": metadata["total_runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
