# Stage A2 Compatibility Notes

## Gate 1

- Direct `huggingface.co` access timed out from the server. The Hugging Face-compatible mirror was used, while pinning the immutable resolved commit SHA.
- The official `model.safetensors` at commit `dbe7ae300a54abcdc475a33907b3dff81d25709f` matched its LFS SHA-256 but has no `format` metadata. Transformers 4.38.2 cannot load that file through its standard `from_pretrained` path. The official `pytorch_model.bin` from the same commit was retained and hash-verified instead.
- TransformerLens 1.14.0 imports `pytest` at runtime without declaring it as an install dependency. `pytest==8.3.5` was added to the project environment.

## Gate 2

The upstream repository remains unchanged. Gate 2 uses a wrapper in our own `scripts/` directory and reuses the released `get_layer_activations_tl()` and `process_activation_batch()` functions without editing them.

Two compatibility adaptations are necessary:

1. The official saved prompt Dataset has format state `type='torch', columns=['input_ids'], output_all_columns=False`. Consequently, direct access to `entity_mask` is hidden even though the Arrow data and feature schema contain it. The wrapper loads the dataset, selects the fixed rows, and in memory calls:

   ```python
   dataset.set_format(
       type="torch",
       columns=["input_ids", "entity_mask"],
       output_all_columns=True,
   )
   ```

   No upstream data is rewritten.

2. TransformerLens 1.14.0 calls `AutoConfig.from_pretrained()` using the remote model alias even when a fully loaded local `hf_model` and tokenizer are supplied. In offline mode this fails. During model construction only, the wrapper redirects this config lookup to the pinned local `config.json`, then restores the original method. Weight conversion and hook behavior remain TransformerLens 1.14.0 implementations.

Formal Gate 2 activations retain the released loading semantics: `HookedTransformer.from_pretrained()` defaults followed by `blocks.{layer}.hook_resid_post`. A separate `from_pretrained_no_processing()` model is used only as a diagnostic in the HF/TL numerical comparison and is not used for saved formal activations.

### HF hidden-state alignment

GPT-NeoX returned 33 `hidden_states`. Empirical module-hook comparison showed:

- `hidden_states[0]` is the embedding output;
- `hidden_states[1]` through `[31]` match raw outputs of blocks 0 through 30;
- `hidden_states[32]` is after `gpt_neox.final_layer_norm`, so it is not raw block-31 residual-post.

Therefore the HF/TL check uses forward hooks on `hf_model.gpt_neox.layers[0]`, `[15]`, and `[31]` as the authoritative HF block outputs. The hidden-state tuple is additionally compared to these hooks to document the final-layer exception. No activation definition was changed to force agreement.

### Upstream bytecode cleanup

Importing released modules initially generated untracked Python 3.11 `__pycache__` files under the upstream checkout. During cleanup it was discovered that the repository itself tracks older Python 3.9 `.pyc` files in `feature_datasets/__pycache__`. Those tracked files were restored exactly from the pinned upstream commit, the untracked Python 3.11 cache was removed, and subsequent execution used `PYTHONDONTWRITEBYTECODE=1`. Final `git status --porcelain` is empty.
