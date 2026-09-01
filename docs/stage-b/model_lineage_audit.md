# Stage B2.0 Model Lineage Audit

Status: **PASS**

> Revisions below are local ModelScope file-level revisions recovered from `.msc`; they are not represented as Hugging Face repository commit SHAs.

| Model | Path | Resolved path | Params | Layers | Hidden | Config vocab | Tokenizer vocab | Model file revision |
|---|---|---|---:|---:|---:|---:|---:|---|
| Base | `/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B` | `/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B` | 8,030,261,248 | 32 | 4096 | 128256 | 128256 | `24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea` |
| SFT | `/home/ai/projects/reasoning-geometry-pilot/models/Llama-3.1-Tulu-3-8B-SFT` | `/home/ai/projects/reasoning-geometry-pilot/models/Llama-3.1-Tulu-3-8B-SFT` | 8,030,326,784 | 32 | 4096 | 128264 | 128257 | `aa9a78450ea2c6916824501e276cb97b7caf47e6` |
| DPO | `/home/ai/projects/reasoning-geometry-pilot/models/Llama-3.1-Tulu-3-8B-DPO` | `/home/ai/projects/reasoning-geometry-pilot/models/allenai/Llama-3___1-Tulu-3-8B-DPO` | 8,030,326,784 | 32 | 4096 | 128264 | 128257 | `2301bf3b5e8733aad822a59b93381369da1692eb` |
| Final | `/home/ai/projects/reasoning-geometry-pilot/models/Llama-3.1-Tulu-3-8B` | `/home/ai/projects/reasoning-geometry-pilot/models/Llama-3.1-Tulu-3-8B` | 8,030,326,784 | 32 | 4096 | 128264 | 128257 | `c5d2996f1e01989a69bd740d8d034edf672cc1e0` |

## Comparability

- four_models_present: **True**
- all_32_decoder_blocks: **True**
- all_hidden_size_4096: **True**
- architecture_dimensions_compatible: **True**
- parameter_counts_match_base_and_tulu_vocab_extension: **True**
- all_safetensor_manifests_complete: **True**
- all_bos_eos_match: **True**

## Final-stage local evidence

Local Final README explicitly identifies Final Models (RLVR), states finetuned from DPO, and documents PPO settings for RLVR. Formal tables retain label Final; report may parenthetically state locally documented RLVR.
