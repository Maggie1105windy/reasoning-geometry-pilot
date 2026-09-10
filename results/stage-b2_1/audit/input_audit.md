# Stage B2.1 input audit

Status: **PASS**

Resolved project root: `/home/ai/chenchen/projects/reasoning-geometry-pilot`

Primary test rows: **2063**; row-ID SHA-256: `dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950`

Hop counts: `{'1': 712, '2': 505, '3': 422, '4': 282, '5': 142}`

Raw prompt exact on all frozen rows: **True**

Protected file snapshot: **2098 files**, `f657fe06d92ba4edffdee769ac392945ce6101fc2df9a3029df7aa9ced404579`

## Checkpoints

| Model | Revision kind | Revision | Vocab tokenizer/config | Layers | Hidden | Chat template |
|---|---|---|---:|---:|---:|---|
| base | ModelScope .msc local revision | `24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea` | 128256/128256 | 32 | 4096 | False |
| sft | ModelScope .msc local revision | `aa9a78450ea2c6916824501e276cb97b7caf47e6` | 128257/128264 | 32 | 4096 | True |
| dpo | ModelScope .msc local revision | `2301bf3b5e8733aad822a59b93381369da1692eb` | 128257/128264 | 32 | 4096 | True |
| final | ModelScope .msc local revision | `c5d2996f1e01989a69bd740d8d034edf672cc1e0` | 128257/128264 | 32 | 4096 | True |

## Frozen natural candidates

```json
{
  "ABOVE": " above",
  "BELOW": " below",
  "LEFT": " left",
  "RIGHT": " right",
  "UPPER_LEFT": " upper-left",
  "UPPER_RIGHT": " upper-right",
  "LOWER_LEFT": " lower-left",
  "LOWER_RIGHT": " lower-right"
}
```

All formal protocols were serialized and SHA-256 frozen before full evaluation.
