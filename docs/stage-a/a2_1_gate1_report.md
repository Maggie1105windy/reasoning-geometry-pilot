# Stage A2 Gate 1 — Model, Compatibility, and Prompt/Data Verification

Date completed: 2026-08-23  
Reproduction status: **best-effort methodological reproduction**  
Target: Pythia-2.8B × World × `coords`

## Gate decision

**Gate 1 passed. Stop before activation extraction.**

No forward pass or activation extraction was run in this gate.

## Ridge implementation re-check

The released `place_probe_experiment(activations, target, is_test, probe=None, ...)` contains:

```python
if probe is None:
    probe = Ridge(alpha=activations.shape[1])
```

For Pythia-2.8B (`hidden_size=2560`), the main methodological reproduction will therefore use `Ridge(alpha=2560)`.

The separate released `main_probe_experiment()` explicitly constructs `RidgeCV(alphas=RIDGE_ALPHAS[args.model], ...)`, but `RIDGE_ALPHAS` has only Llama entries. It has no `pythia-2.8b` key. A future train-internal RidgeCV analysis will be labelled as our sensitivity analysis and will use a range containing 2560; it will not be presented as the paper's Pythia setting.

## Cache and storage audit

- Before download, no Pythia-2.8B target or matching global Hugging Face cache entry existed.
- Existing `/home/ai/.cache/huggingface` size was approximately 9.6GB and was not modified or deleted as part of model cleanup.
- Initial `/home` free space: approximately 248GB.
- Final `/home` free space: approximately 242GB.
- Final model directory size: approximately 5.2GiB (`du -sh`).
- A project-specific temporary cache was used only during download and removed after verification.
- Final model directory contains exactly one weight file; no `.incomplete` file remains.

## Fixed model revision

| Field | Value |
|---|---|
| Repository | `EleutherAI/pythia-2.8b` |
| Weight family | current official non-deduplicated Pythia |
| Requested revision | `step143000` |
| Resolved commit SHA | `dbe7ae300a54abcdc475a33907b3dff81d25709f` |
| Local path | `/home/ai/projects/linear-probe-repro/models/pythia-2.8b-step143000` |
| Retained weight | `pytorch_model.bin` |
| Weight size | 5,550,545,787 bytes |
| Weight SHA-256 | `a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf` |

The revision was resolved via the Hugging Face-compatible mirror because direct `huggingface.co` requests from the server timed out. Both Git refs and the model API resolved `step143000` to the SHA above. Downloads were pinned to that immutable SHA rather than floating `main`.

## Weight-format compatibility finding

The same commit publishes both safetensors and PyTorch formats. The official safetensors file was downloaded and matched its published LFS SHA-256:

```text
462f2b960062159c3779eb1cfe5829783ea72da66b020bcbabf86bc6dd33063c
```

However, its safetensors header contains no `format` metadata. Transformers 4.38.2 calls `metadata.get("format")` during `from_pretrained`, causing `AttributeError: 'NoneType' object has no attribute 'get'`. This is a source-file/library compatibility issue, not a checksum failure.

The official `pytorch_model.bin` at the same commit was therefore downloaded, hash-verified, and loaded through the standard offline Hugging Face path. After it passed HF and TransformerLens checks, the safetensors copy and the task-specific cache were removed. The final directory retains one official weight copy.

## Offline model verification

Environment variables during final verification:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Results:

| Check | Result |
|---|---|
| Config readable | PASS |
| Tokenizer readable | PASS |
| Standard HF weight load (`use_safetensors=False`) | PASS |
| Model type | `gpt_neox` |
| Transformer blocks | 32 |
| Hidden size | 2560 |
| Attention heads | 32 |
| Vocabulary size | 50,304 |
| Loaded parameter count | 2,775,208,960 |
| Load dtype | float16 |
| `.incomplete` files | none |

## TransformerLens compatibility

Installed and locked:

- TransformerLens 1.14.0
- PyTorch 2.4.0+cu124
- Transformers 4.38.2
- einops 0.8.2
- pytest 8.3.5 (TransformerLens 1.14.0 imports pytest at runtime but does not declare it as a dependency)

The alias check resolved:

```text
pythia-2.8b -> EleutherAI/pythia-2.8b
```

An offline HF model was converted into `HookedTransformer` without a forward pass. Because TransformerLens 1.14.0 performs a second config lookup by remote alias even when `hf_model` is supplied, the validation redirected only that config lookup to the pinned local `config.json`.

Conversion results:

```text
n_layers = 32
d_model = 2560
n_heads = 32
blocks.{0..31}.hook_resid_post present = 32/32
```

This local-config redirect must be included in our independent extraction script or replaced with an equivalently pinned local TransformerLens construction. It does not alter weights or activation definitions.

## Official World data and prompt verification

Official paths:

```text
external/world-models/data/entity_datasets/world_place.csv
external/world-models/data/prompt_datasets/pythia/world_place/coords
```

Counts:

```text
total = 39,585
train (`is_test=False`) = 31,668
test (`is_test=True`) = 7,917
```

The saved prompt dataset has 39,585 rows and the columns `entity`, `input_ids`, and `entity_mask`. Its saved format exposes only `input_ids` by default, so `reset_format()` is required before direct row access to `entity_mask`. This does not modify the saved upstream dataset.

Confirmed tokenizer/prompt behavior:

- explicit BOS; BOS ID 0
- EOS ID 0 and EOS assigned as pad ID 0
- right padding
- decoded prefix exactly `What are the lat/lon coordinates of `
- entity positions supplied by `entity_mask`
- last `True` mask position is the entity last-token index

Examples:

| Entity | Last index | Last token |
|---|---:|---|
| Tancredo Neves Pantheon of the Fatherland and Freedom | 23 | `ĠFreedom` |
| Sembuwatta Lake | 13 | `ĠLake` |
| Ford Rotunda | 11 | `unda` |

## Recorded reproduction differences

1. The paper/public code does not pin the Pythia revision; current non-deduplicated `step143000` is the approved best-effort choice.
2. The public `main_probe_experiment` has no Pythia RidgeCV grid.
3. The retained official format is `.bin`, because the same commit's official safetensors lacks metadata required by Transformers 4.38.2. The two official file hashes and the reason for the choice are recorded.
4. TransformerLens 1.14.0 needs a local-config redirect for completely offline construction from the pinned model directory.

## Next gate (not run)

Gate 2 may build the independent activation sanity-check script and run only 16 fixed-seed samples. Full World extraction remains prohibited until the later gate is explicitly approved.
