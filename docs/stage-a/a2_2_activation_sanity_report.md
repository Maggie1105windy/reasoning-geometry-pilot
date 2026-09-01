# Stage A2 Gate 2 — 16-sample Activation Sanity Report

Status: **PASS**  
No probe was trained; no full-dataset extraction was run.

## Sample manifest summary

- Seed: 42
- 8 official train rows and 8 official test rows
- Per split: 3 populated places, 3 structures, 2 natural places

```text
 sample_order  row_id                                               entity  latitude  longitude  is_test     entity_type   entity_subtype        country                                          raw_entity
            0   26573                                              Barjora   23.4300  87.280000    False populated_place   PopulatedPlace          India                                             Barjora
            1    9184                               Georgia's Heard County   33.3000 -85.130000    False populated_place           Region  United_States                               Heard County, Georgia
            2   30195                                             Sangaste   57.9261  26.331900    False populated_place   PopulatedPlace        Estonia                                            Sangaste
            3     536                         Brisbane's Inner City Bypass  -27.4531 153.024000    False       structure             Road      Australia                         Inner City Bypass, Brisbane
            4    4489                            Tenth Presbyterian Church   39.9470 -75.169900    False       structure HistoricBuilding  United_States                           Tenth Presbyterian Church
            5    5512                                     Carlsberg Museum   55.6674  12.530700    False       structure         Building        Denmark                                    Carlsberg Museum
            6   38447                                             Lake Ram   33.2325  35.766300    False   natural_place             Lake  Golan_Heights                                            Lake Ram
            7   39527                                       Salim Ali Lake   19.8992  75.342300    False   natural_place      BodyOfWater          India                                      Salim Ali Lake
            8   30990                                     Rocha Department  -34.4833 -54.350000     True populated_place   PopulatedPlace        Uruguay                                    Rocha Department
            9   30203                               Sandy Bay First Nation   50.5500 -98.647200     True populated_place   PopulatedPlace         Canada                              Sandy Bay First Nation
           10   28883                                           Puncheston   51.9300  -4.900000     True populated_place   PopulatedPlace United_Kingdom                                          Puncheston
           11    3307 Lewiston, Maine's Basilica of Saints Peter and Paul    44.0982 -70.212300     True       structure HistoricBuilding  United_States Basilica of Saints Peter and Paul (Lewiston, Maine)
           12    5510                                        Ormeley Lodge   51.4353  -0.302222     True       structure         Building        England                                       Ormeley Lodge
           13    2932                                         Khindsi Lake   21.3965  79.372100     True       structure              Dam          India                                        Khindsi Lake
           14   39395                                          Lake Saroma   44.1667 143.717000     True   natural_place      BodyOfWater          Japan                                         Lake Saroma
           15   39142                                Chamonix's Lac Blanc    45.9822   6.889440     True   natural_place             Lake         France                                Lac Blanc (Chamonix)
```

## Tokenization examples

### Barjora (row 26573)

- Prompt: `What are the lat/lon coordinates of Barjora`
- Token IDs: `[0, 1276, 403, 253, 4329, 16, 18640, 11627, 273, 4033, 75, 6464]`
- Tokens: `['<|endoftext|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠBar', 'j', 'ora']`
- Entity mask: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]`
- Entity first/last: `9` / `11`
- Entity last token: `ora`

### Georgia's Heard County (row 9184)

- Prompt: `What are the lat/lon coordinates of Georgia's Heard County`
- Token IDs: `[0, 1276, 403, 253, 4329, 16, 18640, 11627, 273, 12096, 434, 754, 472, 3928]`
- Tokens: `['<|endoftext|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠGeorgia', "'s", 'ĠHe', 'ard', 'ĠCounty']`
- Entity mask: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`
- Entity first/last: `9` / `13`
- Entity last token: `ĠCounty`

### Sangaste (row 30195)

- Prompt: `What are the lat/lon coordinates of Sangaste`
- Token IDs: `[0, 1276, 403, 253, 4329, 16, 18640, 11627, 273, 38236, 4740]`
- Tokens: `['<|endoftext|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠSang', 'aste']`
- Entity mask: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]`
- Entity first/last: `9` / `10`
- Entity last token: `aste`

### Brisbane's Inner City Bypass (row 536)

- Prompt: `What are the lat/lon coordinates of Brisbane's Inner City Bypass`
- Token IDs: `[0, 1276, 403, 253, 4329, 16, 18640, 11627, 273, 40767, 434, 38598, 3228, 2896, 5858]`
- Tokens: `['<|endoftext|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠBrisbane', "'s", 'ĠInner', 'ĠCity', 'ĠBy', 'pass']`
- Entity mask: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]`
- Entity first/last: `9` / `14`
- Entity last token: `pass`

### Tenth Presbyterian Church (row 4489)

- Prompt: `What are the lat/lon coordinates of Tenth Presbyterian Church`
- Token IDs: `[0, 1276, 403, 253, 4329, 16, 18640, 11627, 273, 49211, 3327, 49945, 6412]`
- Tokens: `['<|endoftext|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠTenth', 'ĠPres', 'byterian', 'ĠChurch']`
- Entity mask: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]`
- Entity first/last: `9` / `12`
- Entity last token: `ĠChurch`

## Activation definition and shapes

- Formal hook: `blocks.{layer}.hook_resid_post`, layers 0–31
- Layer 0 is explicitly `blocks.0.hook_resid_post`, not embedding
- Model dtype: float16; activation storage dtype: float16
- Batch size: 1
- Number of hook layers: 32
- Layer 0 shape: `[16, 2560]`
- Layer 15 shape: `[16, 2560]`
- Layer 31 shape: `[16, 2560]`
- All 32 layer shapes: `[16, 2560]`

## Numeric examples

```json
[
  {
    "sample_index": 0,
    "row_id": 26573,
    "split": "train",
    "entity": "Barjora",
    "prompt": "What are the lat/lon coordinates of Barjora",
    "latitude": 23.43,
    "longitude": 87.28,
    "entity_last_token_index": 11,
    "entity_last_token_text": "ora",
    "layer_0_first_8": [
      -0.921875,
      -0.454345703125,
      -0.0677490234375,
      0.5810546875,
      -0.268798828125,
      -0.2205810546875,
      0.055908203125,
      -0.1103515625
    ],
    "layer_15_first_8": [
      1.7001953125,
      -0.693359375,
      -0.33056640625,
      0.0546875,
      -1.287109375,
      0.85888671875,
      -0.0439453125,
      -2.859375
    ],
    "layer_31_first_8": [
      2.03515625,
      -1.5205078125,
      0.75,
      1.73828125,
      1.7861328125,
      -0.447509765625,
      1.828125,
      -2.1171875
    ]
  },
  {
    "sample_index": 8,
    "row_id": 30990,
    "split": "test",
    "entity": "Rocha Department",
    "prompt": "What are the lat/lon coordinates of Rocha Department",
    "latitude": -34.4833,
    "longitude": -54.35,
    "entity_last_token_index": 11,
    "entity_last_token_text": "ĠDepartment",
    "layer_0_first_8": [
      -1.041015625,
      0.7158203125,
      -0.185546875,
      -0.693359375,
      -0.0723876953125,
      0.0045166015625,
      -0.224853515625,
      0.035003662109375
    ],
    "layer_15_first_8": [
      -1.87890625,
      -0.80078125,
      0.18017578125,
      -0.42578125,
      0.0654296875,
      0.1572265625,
      -0.1160888671875,
      -0.84619140625
    ],
    "layer_31_first_8": [
      0.71044921875,
      -1.150390625,
      2.046875,
      1.5126953125,
      -0.2119140625,
      1.921875,
      -2.470703125,
      0.5859375
    ]
  }
]
```

## Layer statistics

- Total NaN: 0
- Total Inf: 0
- Full per-layer table: `results/a2_2_activation_stats.csv`

```text
 layer                      hook      shape  nan_count  inf_count  zero_count     mean      std  frobenius_l2_norm  mean_sample_l2_norm
     0  blocks.0.hook_resid_post [16, 2560]          0          0           3 0.000007 0.699169         141.502075            35.286762
    15 blocks.15.hook_resid_post [16, 2560]          0          0           5 0.000124 1.835088         371.395966            92.570290
    31 blocks.31.hook_resid_post [16, 2560]          0          0           1 0.000177 3.075592         622.456848           154.028229
```

## Token alignment control

The official coords prompt ends with the entity, and batch-size-1 clipping stops at the entity last token. Therefore the post-clipping sequence-final index is intentionally identical to the entity-last index. The entity-first and entity-last vectors were separately checked and were not identical.

```text
 sample_index  row_id                 entity  layer  first_index  last_index  sequence_final_index  last_equals_sequence_final_index  first_vs_last_identical  last_vs_final_identical  first_last_max_absolute_difference  first_last_mean_absolute_difference  first_last_cosine_similarity
            0   26573                Barjora      0            9          11                    11                              True                    False                     True                            5.890625                             0.440095                      0.702648
            0   26573                Barjora     15            9          11                    11                              True                    False                     True                            5.937500                             0.990510                      0.736524
            0   26573                Barjora     31            9          11                    11                              True                    False                     True                           77.875000                             2.103938                      0.456872
            1    9184 Georgia's Heard County      0            9          13                    13                              True                    False                     True                            2.108398                             0.454200                      0.683389
            1    9184 Georgia's Heard County     15            9          13                    13                              True                    False                     True                           13.125000                             1.059522                      0.754622
            1    9184 Georgia's Heard County     31            9          13                    13                              True                    False                     True                           29.250000                             1.544656                      0.764805
```

## HF versus TransformerLens

HF returned 33 hidden states. Index 0 is the embedding output; indices 1–31 match raw outputs of blocks 0–30, but index 32 is after GPT-NeoX `final_layer_norm`, not raw block-31 resid-post. HF module forward hooks are therefore the authoritative block-output comparison for layers 0, 15, and 31.

The released TransformerLens path uses `from_pretrained` defaults, including weight processing. A second unprocessed-TL diagnostic isolates differences introduced by this processing; formal saved activations remain the released processed-TL definition.

```text
                            comparison  sample_index  layer  hf_hidden_state_index  max_absolute_difference  mean_absolute_difference  cosine_similarity
official_processed_TL_vs_HF_block_hook             0      0                      1                 0.031250                  0.006696           0.999943
  HF_block_hook_vs_hidden_states_tuple             0      0                      1                 0.000000                  0.000000           1.000000
official_processed_TL_vs_HF_block_hook             0     15                     16                 0.062500                  0.006657           0.999990
  HF_block_hook_vs_hidden_states_tuple             0     15                     16                 0.000000                  0.000000           1.000000
official_processed_TL_vs_HF_block_hook             0     31                     32                 0.312500                  0.029578           0.999944
  HF_block_hook_vs_hidden_states_tuple             0     31                     32                53.203125                  0.747746           0.952919
official_processed_TL_vs_HF_block_hook             8      0                      1                 0.015625                  0.009353           0.999903
  HF_block_hook_vs_hidden_states_tuple             8      0                      1                 0.000000                  0.000000           1.000000
official_processed_TL_vs_HF_block_hook             8     15                     16                 0.062500                  0.012558           0.999976
  HF_block_hook_vs_hidden_states_tuple             8     15                     16                 0.000000                  0.000000           1.000000
official_processed_TL_vs_HF_block_hook             8     31                     32                 0.187500                  0.035450           0.999877
  HF_block_hook_vs_hidden_states_tuple             8     31                     32                24.195312                  0.328572           0.975619
       unprocessed_TL_vs_HF_diagnostic             0      0                      1                 0.015625                  0.000185           0.999999
       unprocessed_TL_vs_HF_diagnostic             0     15                     16                 0.062500                  0.001890           0.999999
       unprocessed_TL_vs_HF_diagnostic             0     31                     32                 0.312500                  0.028052           0.999948
       unprocessed_TL_vs_HF_diagnostic             8      0                      1                 0.015625                  0.000198           1.000000
       unprocessed_TL_vs_HF_diagnostic             8     15                     16                 0.125000                  0.001942           0.999999
       unprocessed_TL_vs_HF_diagnostic             8     31                     32                 0.375000                  0.029437           0.999910
```

- Processed TL vs HF block-hook maximum absolute difference: 0.3125
- Processed TL vs HF block-hook minimum cosine similarity: 0.99987674
- Unprocessed TL vs HF block-hook maximum absolute difference: 0.375
- Unprocessed TL vs HF block-hook minimum cosine similarity: 0.9999103
- HF block-hook vs hidden_states minimum cosine for layers 0/15: 0.9999997
- HF block-hook vs hidden_states maximum cosine for layer 31 (post-final-LN mismatch expected): 0.97561896

## Runtime and GPU memory

- Formal 16-sample extraction runtime: 1.567 s
- Total script runtime: 22.234 s
- Peak allocated: 10.642 GiB
- Peak reserved: 10.994 GiB
- GPU OOM: no

## Official quantization round trip (optional check)

- Completed without error: True
- Mean MAE across layers: 0.0036160092
- Maximum error across layers: 0.18823624
- Minimum mean cosine similarity across layers: 0.99999547
- Float16 formal activations were retained; quantization did not replace them.

## Gate checklist

- [x] 16_official_samples_8_train_8_test
- [x] coords_prompt
- [x] entity_mask_and_last_token
- [x] 32_resid_post_layers
- [x] hidden_dim_2560
- [x] all_layer_shapes_16x2560
- [x] no_nan_inf
- [x] different_layers_not_identical
- [x] different_entities_not_identical
- [x] layer_0_is_blocks_0_resid_post
- [x] hf_tl_indexing_checked
- [x] gpu_no_oom
- [x] upstream_clean

Final Gate 2 decision: **PASS**

## Output files

- `results/a2_2_sample_manifest.csv`
- `results/a2_2_activation_sanity_report.md`
- `results/a2_2_activation_stats.csv`
- `results/a2_2_hf_tl_comparison.csv`
- `results/a2_2_token_alignment_control.csv`
- `results/a2_2_quantization_roundtrip.csv`
- `activations/sanity/pythia-2.8b/world/coords/activations.pt`
- `activations/sanity/pythia-2.8b/world/coords/metadata.json`
- `activations/sanity/pythia-2.8b/world/coords/tokenization_examples.json`
- `activations/sanity/pythia-2.8b/world/coords/numeric_examples.json`
