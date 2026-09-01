# Stage A3 — 16-sample Activation Sanity Report

Status: **PASS**

- 8 official train and 8 official test rows; Gate 2 row IDs reused
- Llama-3.1 tokenizer with explicit BOS, EOS pad assignment, right padding
- Explicit fixed-prefix token-boundary entity mask; extraction uses entity last-token index
- Fast-tokenizer offsets are retained only as diagnostics because 12/5,000 rows expose zero-length entity offsets
- Formal activation: raw output of every decoder block before final RMSNorm
- All 32 shapes: `[16, 4096]`; float16 CPU storage
- NaN/Inf: 0/0

## Indexing validation

```text
 sample_index  row_id           entity  layer                              comparison  hf_hidden_states_index  max_absolute_difference  mean_absolute_difference  cosine_similarity
            0   26573          Barjora      0 raw_decoder_hook_vs_hidden_states_tuple                       1                   0.0000                  0.000000           1.000001
            0   26573          Barjora     15 raw_decoder_hook_vs_hidden_states_tuple                      16                   0.0000                  0.000000           1.000001
            0   26573          Barjora     31 raw_decoder_hook_vs_hidden_states_tuple                      32                   7.6875                  0.878773           0.989172
            0   26573          Barjora     31  raw_decoder_hook_vs_post_final_RMSNorm                      32                   7.6875                  0.878773           0.989172
            0   26573          Barjora     31    RMSNorm_raw_hook_vs_hidden_states_32                      32                   0.0000                  0.000000           1.000001
            8   30990 Rocha Department      0 raw_decoder_hook_vs_hidden_states_tuple                       1                   0.0000                  0.000000           1.000001
            8   30990 Rocha Department     15 raw_decoder_hook_vs_hidden_states_tuple                      16                   0.0000                  0.000000           1.000001
            8   30990 Rocha Department     31 raw_decoder_hook_vs_hidden_states_tuple                      32                  11.1250                  1.062073           0.992039
            8   30990 Rocha Department     31  raw_decoder_hook_vs_post_final_RMSNorm                      32                  11.1250                  1.062073           0.992039
            8   30990 Rocha Department     31    RMSNorm_raw_hook_vs_hidden_states_32                      32                   0.0000                  0.000000           1.000001
```

HF `hidden_states[0]` is embedding output; indices 1-31 are raw outputs of blocks 0-30; index 32 is raw block-31 after final RMSNorm. Raw block 31 is obtained by its decoder-layer hook. Formal activations use hooks for every layer, so no normalized/raw mixture occurs.

## Ten multi-token entity examples

### Barjora (row 26573)

- Prompt: `What are the lat/lon coordinates of Barjora`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠBar', 'j', 'ora']`
- Entity span: `9..11`
- Entity last token: `ora`

### Georgia's Heard County (row 9184)

- Prompt: `What are the lat/lon coordinates of Georgia's Heard County`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠGeorgia', "'s", 'ĠHeard', 'ĠCounty']`
- Entity span: `9..12`
- Entity last token: `ĠCounty`

### Sangaste (row 30195)

- Prompt: `What are the lat/lon coordinates of Sangaste`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠSang', 'aste']`
- Entity span: `9..10`
- Entity last token: `aste`

### Brisbane's Inner City Bypass (row 536)

- Prompt: `What are the lat/lon coordinates of Brisbane's Inner City Bypass`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠBrisbane', "'s", 'ĠInner', 'ĠCity', 'ĠB', 'ypass']`
- Entity span: `9..14`
- Entity last token: `ypass`

### Tenth Presbyterian Church (row 4489)

- Prompt: `What are the lat/lon coordinates of Tenth Presbyterian Church`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠT', 'enth', 'ĠPresbyterian', 'ĠChurch']`
- Entity span: `9..12`
- Entity last token: `ĠChurch`

### Carlsberg Museum (row 5512)

- Prompt: `What are the lat/lon coordinates of Carlsberg Museum`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠCar', 'ls', 'berg', 'ĠMuseum']`
- Entity span: `9..12`
- Entity last token: `ĠMuseum`

### Lake Ram (row 38447)

- Prompt: `What are the lat/lon coordinates of Lake Ram`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠLake', 'ĠRam']`
- Entity span: `9..10`
- Entity last token: `ĠRam`

### Salim Ali Lake (row 39527)

- Prompt: `What are the lat/lon coordinates of Salim Ali Lake`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠSal', 'im', 'ĠAli', 'ĠLake']`
- Entity span: `9..12`
- Entity last token: `ĠLake`

### Rocha Department (row 30990)

- Prompt: `What are the lat/lon coordinates of Rocha Department`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠRoch', 'a', 'ĠDepartment']`
- Entity span: `9..11`
- Entity last token: `ĠDepartment`

### Sandy Bay First Nation (row 30203)

- Prompt: `What are the lat/lon coordinates of Sandy Bay First Nation`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠSandy', 'ĠBay', 'ĠFirst', 'ĠNation']`
- Entity span: `9..12`
- Entity last token: `ĠNation`

## Representative activation statistics

```text
 layer                                      activation      shape  nan_count  inf_count      mean      std  frobenius_l2_norm  mean_sample_l2_norm
     0  model.layers.0 raw output before final RMSNorm [16, 4096]          0          0 -0.000004 0.016750           4.288041             1.059619
    15 model.layers.15 raw output before final RMSNorm [16, 4096]          0          0 -0.001996 0.184116          47.136471            11.707176
    31 model.layers.31 raw output before final RMSNorm [16, 4096]          0          0 -0.000267 1.060689         271.535736            67.104919
```

## GPU trials

```text
                  phase  batch_size status  runtime_seconds  peak_allocated_bytes  peak_reserved_bytes  peak_allocated_gib  peak_reserved_gib
          batch_trial_1           1   PASS         0.874470           16078394368          16219373568           14.974172          15.105469
          batch_trial_2           2   PASS         0.109766           16089282560          16236150784           14.984312          15.121094
          batch_trial_4           4   PASS         0.059544           16116401152          16269705216           15.009568          15.152344
sanity_16_all_32_layers           1   PASS         0.686863           16086171648          16231956480           14.981415          15.117188
```
