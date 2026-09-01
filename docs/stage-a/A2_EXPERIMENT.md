# Experiment specification

## Paper

Gurnee & Tegmark, *Language Models Represent Space and Time*, arXiv:2310.02207.

## Reproduction target

The first stage only reproduces the minimal World spatial probing experiment.

```text
entity name
→ tokenizer
→ frozen LLM
→ output_hidden_states=True
→ hidden state at the entity's final token for every layer
→ layer-wise Ridge Regression
→ longitude / latitude
→ held-out evaluation
```

## Important methodological constraints

- Keep the LLM frozen throughout.
- Train a separate probe for each layer.
- Do not concatenate representations from multiple layers.
- Use linear ridge regression for the first version.
- Keep train and test sets strictly separate.
- Do not use the test set for hyperparameter selection.
- Record the exact hidden-state extraction position.
- Save the random seed.
- Save the model name and commit/revision.
- Separate activation extraction from probe training.
- A successful probe only establishes that information is `linearly decodable`; it does not establish that the model uses that information.

## Controls

- Gate 3 executes a train-mean baseline and a layer-15 random-training-label control.
- A linear versus simple nonlinear probe comparison remains deferred to a later stage.

## Stage A2: Pythia-2.8B methodological reproduction

Status: **best-effort methodological reproduction**, not an exact checkpoint-level reproduction.

### Fixed model choice

- Repository: `EleutherAI/pythia-2.8b` (current official non-deduplicated suite)
- Requested checkpoint: `step143000`
- Resolved Hugging Face commit: `dbe7ae300a54abcdc475a33907b3dff81d25709f`
- Local path: `/home/ai/projects/linear-probe-repro/models/pythia-2.8b-step143000`
- Retained weight file: official `pytorch_model.bin`
- Weight SHA-256: `a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf`
- Architecture: 32 blocks, hidden size 2560, 32 attention heads

The same commit's official `model.safetensors` was downloaded and matched its LFS SHA-256 (`462f2b960062159c3779eb1cfe5829783ea72da66b020bcbabf86bc6dd33063c`), but its header has no `format` metadata. Transformers 4.38.2 therefore cannot load it through the standard `from_pretrained` path. The verified official `.bin` was retained instead, and the safetensors copy and task-specific download cache were removed. This leaves one reusable weight copy.

### Irreducible differences from the paper release

1. The paper and public code do not pin a Pythia revision, checkpoint family, or TransformerLens version. The current non-deduplicated `step143000` weights are a documented human-approved best-effort choice and are not claimed to be identical to the Pythia checkpoint used for the paper.
2. The released `main_probe_experiment` defines `RIDGE_ALPHAS` only for Llama models. It does not publish a Pythia RidgeCV grid and would raise a `KeyError` for `pythia-2.8b`.

### Probe policy

- Main methodological reproduction: call `place_probe_experiment(..., probe=None)`, which constructs `Ridge(alpha=activations.shape[1])`. For Pythia-2.8B this is `Ridge(alpha=2560)`.
- Sensitivity analysis: run a separate train-internal `RidgeCV` grid whose range includes 2560. This is our robustness analysis, not a claim about the paper's Pythia hyperparameters.
- The official `is_test` split remains authoritative; test data must not select alpha or fit target normalization.

### Confirmed official-method components

- World `coords` prompt: `What are the lat/lon coordinates of <entity>`
- Official saved World split: 31,668 train / 7,917 test / 39,585 total
- Pythia tokenizer logic: `add_special_tokens=False`, explicit BOS, EOS as pad, right padding
- Entity representation: last valid token selected using `entity_mask`
- Activation definition: `blocks.{layer}.hook_resid_post`, 32 block outputs, embedding excluded
- Layer-wise probes are independent and use `[longitude, latitude]` targets standardized from train statistics only

### Gate 1 compatibility environment

- Python 3.11.15
- PyTorch 2.4.0+cu124
- Transformers 4.38.2
- TransformerLens 1.14.0
- Hugging Face Hub 0.20.3
- Datasets 2.18.0

TransformerLens 1.14.0 resolves `pythia-2.8b` to `EleutherAI/pythia-2.8b`. For fully offline conversion with a preloaded local HF model, its config lookup must be redirected to the pinned local `config.json`; otherwise this version attempts another Hub lookup by model alias. Gate 1 verified the local conversion and all 32 `hook_resid_post` hook names without running a forward pass.

## Stage A2 Gate 2: 16-sample activation sanity check

Status: **PASS**.

- Seed: 42
- Sample selection: 8 official train + 8 official test; within each split, 3 populated places, 3 structures, and 2 natural places
- Formal extraction path: released `get_layer_activations_tl()` and `process_activation_batch()` called from our compatibility wrapper
- Batch size: 1
- Model and saved activation dtype: float16
- Formal activation: `blocks.{layer}.hook_resid_post`, layers 0–31, entity last-token selected from official `entity_mask`
- All layer shapes: `[16, 2560]`
- NaN / Inf: 0 / 0
- Formal extraction runtime: 1.567 seconds
- Total run runtime including HF/TL controls: 22.234 seconds
- Peak CUDA allocated / reserved: 10.642 / 10.994 GiB
- No probe was trained and no full-dataset extraction was run

HF/TL alignment established that GPT-NeoX `hidden_states[32]` is after final layer norm rather than raw block-31 resid-post. HF block module hooks are used for the authoritative comparison. Across two samples and layers 0, 15, and 31, processed TransformerLens versus HF block-hook minimum cosine was 0.9998767; the unprocessed TransformerLens diagnostic minimum cosine was 0.9999103.

The optional official per-channel quint8 round trip was tested without replacing the retained float16 activations: mean MAE across layers 0.0036160, maximum error 0.1882362, and minimum layer mean cosine 0.9999955.

Detailed outputs are recorded in `results/a2_2_activation_sanity_report.md` and `results/a2_compatibility_notes.md`.

## Stage A2 Gate 3: mini layer-wise linear probe

Status: **PASS**. This is a bounded methodological mini-reproduction, not a quantitative reproduction of the full Figure 2 curve.

### Scope and extraction

- Seed: 42
- Official split sampling: 4,000 rows sampled from `is_test=False` and 1,000 rows sampled separately from `is_test=True`; no row overlap
- Layers: 0, 7, 15, 23, and 31 only
- Hooks: `blocks.{layer}.hook_resid_post`
- Extraction path: released `get_layer_activations_tl()` and released entity-last-token aggregation
- Model mode/dtype: evaluation/inference, float16
- Stored activation shape per layer: combined `[5000, 2560]`, train `[4000, 2560]`, test `[1000, 2560]`
- Batch 8 and batch 16 both passed; batch 16 was selected
- Formal extraction runtime: 27.854 seconds
- Peak CUDA allocated / reserved: 5.346 / 5.590 GiB
- Activation artifact size: 123 MiB; SHA-256 `69fbfecc15afa14b67222a22b5ad0dde6b21574f4e210a89884ffb61f4e10ff9`

Ten seeded manifest-to-prompt/entity-mask checks passed. All five activation matrices were finite and their row order matched the manifest.

### Target processing and evaluation

The target order is the released `[longitude, latitude]`. Coordinate means and population standard deviations (`ddof=0`) were fit using only the 4,000 official-train rows. Predictions were unstandardized with those same values. Overall held-out R² is sklearn's default uniform average over the two target columns; longitude and latitude R² are also recorded separately.

The train-mean baseline overall held-out R² was `-0.002122`.

### Primary Ridge(alpha=2560) results

| Layer | Relative depth | Overall R² | Longitude R² | Latitude R² |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.197255 | 0.240779 | 0.153731 |
| 7 | 0.2258 | 0.432988 | 0.467203 | 0.398773 |
| 15 | 0.4839 | 0.554140 | 0.567704 | 0.540576 |
| 23 | 0.7419 | 0.720311 | 0.746989 | 0.693633 |
| 31 | 1.0000 | 0.723285 | 0.749349 | 0.697221 |

Each layer used an independent two-output Ridge probe. Every coefficient array had shape `[2, 2560]`, every intercept had shape `[2]`, and no coefficient or intercept contained NaN or Inf.

### RidgeCV sensitivity analysis

This is our robustness/sensitivity analysis and is not claimed as the paper's Pythia hyperparameter setting. The train-only grid was the sorted union of `np.logspace(1, 5, 17)` and `{2560}`.

| Layer | Selected alpha | Overall R² | Longitude R² | Latitude R² |
|---:|---:|---:|---:|---:|
| 0 | 1000.0000 | 0.212520 | 0.261614 | 0.163426 |
| 7 | 2560.0000 | 0.432988 | 0.467203 | 0.398773 |
| 15 | 1778.2794 | 0.556586 | 0.570290 | 0.542882 |
| 23 | 5623.4133 | 0.722089 | 0.746499 | 0.697678 |
| 31 | 5623.4133 | 0.729911 | 0.754006 | 0.705816 |

### Negative control and interpretation

At layer 15, shuffling only the training labels with seed 42 changed overall held-out R² from `0.554140` to `-0.072370`, a drop of `0.626510`; test targets were unchanged. This materially degraded result, together with the near-zero train-mean baseline, supports the integrity of the mini pipeline.

The monotonic-looking five-point trend is only a mini diagnostic and is not a substitute for the complete 32-layer curve. No full World activation extraction, additional prompt, other model, quantization, or Gate 4 work was run.

Detailed outputs are recorded in `results/a2_3_mini_probe_report.md`, `results/a2_3_mini_probe_fixed_alpha.csv`, `results/a2_3_mini_probe_ridgecv.csv`, and `results/a2_3_mini_probe.png`.


## Stage A2 Gate 4: full World × 32-layer probe

Status: **PASS — Strong best-effort reproduction**.

- Full released World data: 39,585 rows in original order; 31,668 official train and 7,917 official test
- Activations: all 32 `blocks.{layer}.hook_resid_post` matrices, entity-last-token, shape `[39585, 2560]`
- Storage: 78 resumable shards, official per-channel quint8 function applied per shard/layer; 3.072 GiB
- Extraction: batch 16, 286.770 seconds aggregate forward time, 138.037 samples/s
- Peak CUDA allocated/reserved: 5.351 / 5.588 GiB
- Integrity: zero NaN/Inf; 10 rows × 5 layers re-forwarded; minimum saved-vs-live cosine 0.999870837
- Primary probes: 32 independent `Ridge(alpha=2560)` fits with full train-only target standardization
- Peak held-out overall R²: 0.743157 at layer 30; layer 31: 0.741863
- Train-mean baseline overall R²: -0.000013542
- Random-label overall R²: layer 15 = -0.043735; layer 31 = -0.074465
- Figure 2 comparison: matching early-rise/late-plateau shape and close numerical scale; official Pythia raw numeric curve was not published, so comparison is explicitly visual and approximate
- Assessment: Strong best-effort reproduction, not exact checkpoint/hyperparameter reproduction

Detailed outputs are recorded in `results/A2_FULL_REPRODUCTION_REPORT.md` and `results/a2_4_full_activation_integrity.md`. No post-Gate-4 experiment was started.
