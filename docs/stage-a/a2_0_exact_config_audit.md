# Stage A2 Gate 0 — Exact Configuration Audit

Date: 2026-08-22  
Target: Gurnee & Tegmark Figure 2, **Pythia-2.8B × World × `coords`**  
Upstream snapshot: `wesg52/world-models` commit `a572f162948ee185e9c842eb5fa15e23aad3d218`

## Gate decision

**Gate 0 does not pass for an exact reproduction. Stop before model download.**

The public repository identifies the TransformerLens alias `pythia-2.8b`, but it does not pin a TransformerLens version, Hugging Face repository revision, Pythia checkpoint step, or v0/current weight family. It also does not define a Ridge alpha grid for Pythia. These omissions are material because EleutherAI publishes distinct original-v0 and retrained/current Pythia weights.

No model was downloaded and no upstream file was modified during this audit.

## Audited configuration

| Item | Evidence from the released implementation | Status |
|---|---|---|
| Model alias | `MODEL_N_LAYERS['pythia-2.8b'] = 32`; TransformerLens receives the alias `pythia-2.8b` | Confirmed |
| Hugging Face repository | TransformerLens is expected to resolve the alias; the author code does not state an explicit HF repository ID | Not exactly pinned |
| Weight family | No statement selecting original `v0`, `deduped-v0`, or retrained/current Pythia | Unresolved |
| Revision / commit | No `revision=` argument, commit SHA, tag, or lockfile | Unresolved |
| Training checkpoint | `checkpoint_index=None`; the CLI does not expose `checkpoint_index` | Unresolved beyond “TransformerLens default/final for the installed version” |
| TransformerLens version | `transformer-lens` is unpinned in `requirements.txt` | Unresolved |
| Architecture | Pythia-2.8B: 32 Transformer blocks; current official model card reports hidden size 2560 and 32 heads | Architecture family confirmed |
| Tokenizer | Explicitly `EleutherAI/pythia-70m` | Confirmed |
| Special-token logic | `add_special_tokens=False`; explicit BOS; EOS assigned as pad token | Confirmed |
| Padding side | Not set explicitly in the Pythia branch; library/tokenizer default is used | Not explicitly pinned |
| Prompt | `What are the lat/lon coordinates of <entity>` | Confirmed |
| Entity normalization | Parenthetical and comma-form place names are rewritten by official World preprocessing | Confirmed |
| Entity position | `entity_mask`; reverse search selects the last valid entity token | Confirmed |
| Activation site | TransformerLens `blocks.{layer_ix}.hook_resid_post` for all blocks | Confirmed |
| Layer convention | 32 block outputs; embedding is not included as a plotted layer | Confirmed |
| World split | Stored official CSV `is_test`; 31,668 train and 7,917 test (39,585 total) | Confirmed |
| Target order | `[longitude, latitude]` | Confirmed |
| Target normalization | Each coordinate uses train-only mean and standard deviation; test does not fit preprocessing | Confirmed |
| Metric | `sklearn.metrics.r2_score` with its default multi-output uniform average | Confirmed |
| Ridge alpha grid | Released dictionary contains Llama entries only; it has no Pythia key | Unresolved |
| Published raw Pythia curve | No Pythia numeric result artifact was found in `paper_results` or repository data | Not available |

## Code-path findings

### Model loading

`external/world-models/load.py` defaults `checkpoint_index=None`. For non-Llama models it calls:

```python
HookedTransformer.from_pretrained(
    model_name,
    device="cpu",
    checkpoint_index=checkpoint_index,
)
```

No Hugging Face `revision`, Pythia version family, or TransformerLens version is recorded. `save_activations.py` does not expose `checkpoint_index` on its command line, so the public activation path cannot select or document a particular Pythia training step.

### Tokenization and prompt construction

`make_prompt_datasets.py` selects `AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")`, assigns EOS as the pad token, tokenizes with `add_special_tokens=False`, and prepends BOS explicitly. The Pythia branch does not explicitly set `padding_side`; exact reproduction would therefore also require the tokenizer/library behavior of the original environment.

For World `coords`, `space_world.py` uses the prefix:

```text
What are the lat/lon coordinates of 
```

The official prompt dataset is already present at:

```text
external/world-models/data/prompt_datasets/pythia/world_place/coords
```

Its recorded dataset fingerprint is `9926af87b8f0017b`.

### Activation definition

The TransformerLens extraction path hooks `blocks.{layer_ix}.hook_resid_post` for every Transformer block. It uses `entity_mask` and reverse argmax to take the last entity token. Thus the intended 32 plotted Pythia layers are residual-stream outputs after blocks 0–31, excluding the embedding state.

### Probe definition

The official World split is the stored `is_test` column. Coordinates are standardized using training data only. The code predicts `[longitude, latitude]` and reports the default multi-output `r2_score`.

However, `RIDGE_ALPHAS` contains only Llama-2 model keys. Consequently, the released `probe_experiment.py` cannot run `pythia-2.8b` unchanged: indexing the alpha dictionary for this model would raise a `KeyError`. The Llama-2-7B grid `np.logspace(0.8, 4.1, 12)` is present, but the repository does not establish that it was the paper's Pythia grid.

## Why current `main` cannot be silently assumed

The official EleutherAI model card states that the current Pythia suite is a retraining and distinguishes it from the original v0 models. It also states that `step143000` corresponds to the current final/main weights. Separate v0 repositories exist. Because the paper code neither pins a repository revision nor names the weight family, choosing current `EleutherAI/pythia-2.8b` would be a documented best-effort reproduction, not an exact recovery of the paper environment.

## Gate 0 checklist

- [x] Model alias identified: `pythia-2.8b`
- [ ] Exact Pythia weight family identified
- [ ] Exact Hugging Face revision / commit identified
- [ ] Exact TransformerLens version identified
- [x] Architecture convention identified: 32 blocks, expected hidden size 2560
- [x] Tokenizer and special-token logic identified
- [x] `coords` prompt identified
- [x] Official `is_test` split identified
- [x] Activation hook and entity-token rule identified
- [ ] Official Pythia Ridge alpha grid identified

## Required human decision before Gate 1

To continue as a clearly labelled best-effort reproduction, approve both assumptions below:

1. Use the **current official non-deduplicated** `EleutherAI/pythia-2.8b` final weights (`step143000` / current `main`), pinning the exact Hugging Face commit SHA observed at download time.
2. Use `alphas=np.logspace(0.8, 4.1, 12)` as an explicit surrogate Ridge grid, while recording that the public code only associates this grid with Llama-2-7B and does not reveal the paper's Pythia grid.

After approval, the experiment log must mark the result as **best-effort current-Pythia**, not an exact weight-level reproduction.

## Primary references

- Paper: <https://arxiv.org/pdf/2310.02207>
- Released code: <https://github.com/wesg52/world-models>
- Current official model card: <https://huggingface.co/EleutherAI/pythia-2.8b>
- Current official model tree: <https://huggingface.co/EleutherAI/pythia-2.8b/tree/main>
- Example old v0-family repository: <https://huggingface.co/EleutherAI/pythia-2.8b-deduped-v0>
