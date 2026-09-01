# Stage A3 — Llama-3.1-8B Model Audit

Experiment: **Llama-3.1-8B linear-probe migration sanity experiment**  
Purpose: cross-architecture probing migration validation; not a Figure 2 reproduction.

## Read-only model source

- Path: `/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B`
- Access mode for Stage A3: read-only
- Model directory revision marker: `master` (`.mv`)
- ModelScope manifest revision for config, HF weight shards, and most model files: `24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea`
- Tokenizer JSON manifest revision: `c52792af35fe806a76530f3028f037d1af4e3490`
- No model was downloaded or copied into `linear-probe-repro`.

## Actual config

```text
architecture                 LlamaForCausalLM
model_type                   llama
transformer decoder layers   32
hidden size                  4096
intermediate size            14336
attention heads              32
key/value heads              8
vocabulary size              128256
maximum positions            131072
RMSNorm epsilon              1e-5
RoPE type                    llama3
RoPE factor                  8.0
config dtype                 bfloat16
tie word embeddings          false
```

The four HF safetensors shards contain exactly **8,030,261,248 parameters** across 291 tensors. Every stored parameter is BF16.

## Tokenizer

- Loaded class: `PreTrainedTokenizerFast`
- Backend vocabulary size: 128000
- Tokenizer length including added/special tokens: 128256
- BOS token ID: 128000
- EOS token ID: 128001
- Native pad token: absent; Stage A3 explicitly assigns EOS as pad and uses right padding
- Entity spans are constructed explicitly from the token-ID boundary between the fixed prompt prefix and the complete prompt. A full 5,000-row validation required the decoded token suffix to equal the tokenizer-normalized entity. Fast-tokenizer offsets are retained only as diagnostics: 12/5,000 rows have no usable entity offset and only 3,048/5,000 exactly agree with the verified token boundary.

## Weight files

```text
model-00001-of-00004.safetensors   4,976,698,672 bytes
model-00002-of-00004.safetensors   4,999,802,720 bytes
model-00003-of-00004.safetensors   4,915,916,176 bytes
model-00004-of-00004.safetensors   1,168,138,808 bytes
model.safetensors.index.json              23,950 bytes
```

The pre-existing model directory also contains Meta-format `original/consolidated.00.pth` and `original/tokenizer.model`. Stage A3 does not modify, remove, or copy either representation.

## Runtime compatibility

The Stage A2 environment's Transformers 4.38.2 cannot load this config because it does not understand Llama-3.1 `rope_type=llama3`. Stage A3 therefore uses an isolated package overlay inside `linear-probe-repro`:

```text
transformers       4.44.2
tokenizers         0.19.1
huggingface-hub    0.24.7
safetensors        0.4.5
torch              2.4.0+cu124 (shared from the unchanged A2 venv)
```

## Pre-forward hidden-state source audit

In Transformers 4.44.2 `LlamaModel.forward`:

1. `hidden_states` is initialized to `inputs_embeds`.
2. The current state is appended to `all_hidden_states` **before** each decoder layer.
3. After all decoder layers, `self.norm(hidden_states)` applies the final RMSNorm.
4. The post-final-RMSNorm state is appended once more.

This predicts 33 returned states:

```text
hidden_states[0]      embedding output
hidden_states[1]      raw decoder block 0 output
hidden_states[16]     raw decoder block 15 output
hidden_states[31]     raw decoder block 30 output
hidden_states[32]     block 31 output after final RMSNorm
```

Therefore the tuple is predicted not to expose raw block-31 output. Stage A3 must validate this with direct decoder hooks before formal extraction. If confirmed, formal activations for **all** representative layers will come from identical decoder-layer forward hooks, guaranteeing a uniform raw post-block/pre-final-RMSNorm definition.
