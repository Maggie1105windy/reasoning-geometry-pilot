# Stage A3 — Llama-3.1-8B Migration Report

Status: **PASS**

Experiment: **Llama-3.1-8B linear-probe migration sanity experiment**. This is a cross-architecture pipeline validation, not a Figure 2 reproduction.

## Model

- 32 decoder layers; hidden size 4096; 8,030,261,248 parameters
- All stored weights BF16; runtime BF16; no weight quantization
- Read-only path: `/home/ai/projects/reasoning-geometry-pilot/models/Meta-Llama-3.1-8B`

## Hidden-state indexing and formal activation

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

`hidden_states[0]` is embeddings. Indices 1-31 are raw outputs of blocks 0-30. Index 32 is block-31 output after final RMSNorm, so it cannot serve as raw block 31. Formal A3 activations use identical decoder forward hooks for all layers, yielding raw post-block/pre-final-RMSNorm representations.

## Token/entity alignment examples

The entity span is derived from the token-ID boundary between the fixed prompt prefix and the complete prompt. The decoded suffix was required to equal the entity on all 5,000 rows. Offset mappings are diagnostic only because a small set of single-token names receive zero-length offsets from this local tokenizer artifact.

### Barjora

- Prompt: `What are the lat/lon coordinates of Barjora`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠBar', 'j', 'ora']`
- Entity token span: `9..11`
- Entity last token: `ora`

### Georgia's Heard County

- Prompt: `What are the lat/lon coordinates of Georgia's Heard County`
- Tokens: `['<|begin_of_text|>', 'What', 'Ġare', 'Ġthe', 'Ġlat', '/', 'lon', 'Ġcoordinates', 'Ġof', 'ĠGeorgia', "'s", 'ĠHeard', 'ĠCounty']`
- Entity token span: `9..12`
- Entity last token: `ĠCounty`

## 16-sample sanity

All 32 raw-block activation matrices have shape `[16, 4096]`, with zero NaN/Inf.

```text
 layer                                      activation      shape  nan_count  inf_count      mean      std  frobenius_l2_norm  mean_sample_l2_norm
     0  model.layers.0 raw output before final RMSNorm [16, 4096]          0          0 -0.000004 0.016750           4.288041             1.059619
    15 model.layers.15 raw output before final RMSNorm [16, 4096]          0          0 -0.001996 0.184116          47.136471            11.707176
    31 model.layers.31 raw output before final RMSNorm [16, 4096]          0          0 -0.000267 1.060689         271.535736            67.104919
```

## GPU and extraction

```text
                           phase  batch_size status  runtime_seconds  peak_allocated_bytes  peak_reserved_bytes  peak_allocated_gib  peak_reserved_gib
                   batch_trial_1           1   PASS         0.874470           16078394368          16219373568           14.974172          15.105469
                   batch_trial_2           2   PASS         0.109766           16089282560          16236150784           14.984312          15.121094
                   batch_trial_4           4   PASS         0.059544           16116401152          16269705216           15.009568          15.152344
         sanity_16_all_32_layers           1   PASS         0.686863           16086171648          16231956480           14.981415          15.117188
mini_extraction_5000_five_layers           4   PASS        60.158712           16149946368          16483614720           15.040810          15.351562
```

- Chosen mini batch size: 4
- 5,000-row five-layer extraction runtime: 60.159 seconds
- Extraction peak allocated/reserved: 15.041 / 15.352 GiB

## Primary Ridge(alpha=4096)

```text
 layer  relative_depth  n_train  n_test  hidden_dim  alpha  r2_overall  r2_longitude  r2_latitude  baseline_r2_overall  runtime_seconds
     0        0.000000     4000    1000        4096 4096.0    0.002016      0.002087     0.001946            -0.002122         0.435196
     7        0.225806     4000    1000        4096 4096.0    0.211993      0.215577     0.208409            -0.002122         0.428593
    15        0.483871     4000    1000        4096 4096.0    0.363631      0.374096     0.353165            -0.002122         0.432061
    23        0.741935     4000    1000        4096 4096.0    0.801687      0.791689     0.811685            -0.002122         0.425335
    31        1.000000     4000    1000        4096 4096.0    0.864350      0.857341     0.871358            -0.002122         0.419532
```

Alpha equals activation dimension as a unified migration rule; it is not a Gurnee & Tegmark Llama-3.1 setting.

## RidgeCV — OUR SENSITIVITY ANALYSIS

```text
 layer  selected_alpha  r2_overall  r2_longitude  r2_latitude  runtime_seconds           analysis_label
     0       10.000000    0.172972      0.205222     0.140722         8.227594 OUR SENSITIVITY ANALYSIS
     7       17.782794    0.764424      0.757097     0.771752         8.071069 OUR SENSITIVITY ANALYSIS
    15       17.782794    0.832193      0.826790     0.837595         8.049378 OUR SENSITIVITY ANALYSIS
    23      177.827941    0.895599      0.884859     0.906339         8.356783 OUR SENSITIVITY ANALYSIS
    31      562.341325    0.887657      0.881651     0.893664         8.374253 OUR SENSITIVITY ANALYSIS
```

Fixed-vs-CV five-point Spearman: 0.900000

## Controls

```json
{
  "baseline": {
    "r2_overall": -0.0021222925908113144,
    "r2_longitude": -0.0040426576113412604,
    "r2_latitude": -0.00020192757027959196
  },
  "random_label": {
    "layer": 15,
    "seed": 42,
    "alpha": 4096.0,
    "shuffle_scope": "training labels only",
    "test_labels_unchanged": true,
    "r2_overall": -0.0023866960305241602,
    "r2_longitude": -0.005493266990107548,
    "r2_latitude": 0.0007198749290626694,
    "normal_r2_overall": 0.3636306675128678,
    "drop_vs_normal": 0.366017363543392,
    "runtime_seconds": 0.44540834054350853
  }
}
```

## Pythia Gate 3 sanity reference

```text
 layer  llama31_a3_r2  pythia_gate3_r2  difference_llama_minus_pythia
     0       0.002016         0.197255                      -0.195238
     7       0.211993         0.432988                      -0.220995
    15       0.363631         0.554140                      -0.190510
    23       0.801687         0.720311                       0.081376
    31       0.864350         0.723285                       0.141065
```

The two architectures both support the same probing methodology and show depth-dependent decodability. Differences are not interpreted as a controlled model-capability or reasoning comparison because architecture, model scale, tokenizer, hidden size, and pretraining data differ.

## PASS checklist

- [x] model_config_audited
- [x] layers_32_hidden_4096_bf16_parameters_exact
- [x] hidden_state_indexing_explained
- [x] raw_block_vs_final_RMSNorm_distinguished
- [x] uniform_raw_hook_activation_definition
- [x] entity_last_token_explicit_mask
- [x] entity_span_prefix_boundary_validated
- [x] ten_multi_token_examples
- [x] sanity_16_all_shapes_16x4096
- [x] sanity_no_nan_inf
- [x] gpu_no_oom_bf16
- [x] mini_exact_4000_1000_official_split
- [x] mini_row_ids_equal_a2_gate3
- [x] five_independent_fixed_ridges
- [x] heldout_r2_finite
- [x] fixed_vs_ridgecv_trend_not_conflicting
- [x] baseline_near_no_prediction
- [x] random_label_materially_degraded
- [x] coefficient_shapes_and_values_valid
- [x] layers_not_all_identical
- [x] external_world_models_clean

## Outputs

- `results/a3_llama31_model_audit.md`
- `results/a3_activation_sanity_report.md`
- `results/a3_hidden_state_indexing.csv`
- `results/a3_sample_manifest.csv`
- `results/a3_target_scaler.json`
- `results/a3_llama31_mini_probe_fixed_alpha.csv`
- `results/a3_llama31_mini_probe_ridgecv.csv`
- `results/a3_llama31_layer_r2.png`
- `activations/a3_migration/llama-3.1-8b/world/coords/activations.pt`
- `activations/a3_migration/llama-3.1-8b/world/coords/metadata.json`

No Tulu, reasoning, grounding, or full 39,585 × 32-layer Llama experiment was run.
