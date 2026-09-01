# Stage A2 Gate 3 — Mini Layer-wise Linear Probe

Status: **PASS**  
This is a bounded methodological mini-reproduction, not a quantitative reproduction of the paper's full Figure 2 curve.

## Scope and provenance

- Model: `EleutherAI/pythia-2.8b`, revision `step143000`
- Hugging Face commit: `dbe7ae300a54abcdc475a33907b3dff81d25709f`
- Released world-models commit: `a572f162948ee185e9c842eb5fa15e23aad3d218`
- Prompt: `What are the lat/lon coordinates of <entity>`
- Split: released `is_test`; 4,000 train and 1,000 test sampled independently with seed 42
- Layers/hooks: 0, 7, 15, 23, 31 at `blocks.<layer>.hook_resid_post`
- No full World extraction and no other layers/prompts/models were run

## Activation extraction

- Chosen batch size: 16
- Saved dtype: float16; sklearn conversion: float32
- Every layer: combined `[5000, 2560]`, train `[4000, 2560]`, test `[1000, 2560]`

```text
                 phase  batch_size status  runtime_seconds  peak_allocated_bytes  peak_reserved_bytes  peak_allocated_gib  peak_reserved_gib
         batch_trial_8           8   PASS         0.366644            5720151040           6002049024            5.327306           5.589844
        batch_trial_16          16   PASS         0.128297            5732556800           5997854720            5.338860           5.585938
formal_extraction_5000          16   PASS        27.853777            5740608512           6002049024            5.346358           5.589844
```

## Ten row/token alignment checks

```text
 sample_index  row_id split                        entity                                                            prompt  entity_first_token_index  entity_last_token_index  entity_last_token_id entity_last_token  entity_mask_true_count  row_alignment_ok
         2522   14857 train                 Bacchus Marsh                 What are the lat/lon coordinates of Bacchus Marsh                         9                       11                 15463            ĠMarsh                       3              True
         4194    1480  test     Preston Road tube station     What are the lat/lon coordinates of Preston Road tube station                         9                       12                  4660          Ġstation                       4              True
          218    7979 train             Champhai district             What are the lat/lon coordinates of Champhai district                         9                       13                  3286         Ġdistrict                       5              True
         2882   22359 train          Switzerland's Trient          What are the lat/lon coordinates of Switzerland's Trient                         9                       12                  5692             rient                       4              True
         3256   10462 train                   Kuala Penyu                   What are the lat/lon coordinates of Kuala Penyu                         9                       13                 30838                yu                       5              True
         2321   34601 train               Peover Superior               What are the lat/lon coordinates of Peover Superior                         9                       11                 21112         ĠSuperior                       3              True
          100   32040 train   New South Wales's Rydalmere   What are the lat/lon coordinates of New South Wales's Rydalmere                         9                       16                 37166              mere                       8              True
         1382    1910 train Leicester Square tube station What are the lat/lon coordinates of Leicester Square tube station                         9                       12                  4660          Ġstation                       4              True
         2003   20481 train                    Monkey Bay                    What are the lat/lon coordinates of Monkey Bay                         9                       11                  6912              ĠBay                       3              True
         2935   28728 train                  Borg El Arab                  What are the lat/lon coordinates of Borg El Arab                         9                       11                  8365             ĠArab                       3              True
```

## Target preprocessing and metric

- Released target order: `[longitude, latitude]`
- Mean and population standard deviation (`ddof=0`) computed from official train rows only
- Predictions unstandardized using the same train-only parameters
- Overall R² uses sklearn's default uniform average over longitude and latitude

```json
{
  "target_order": [
    "longitude",
    "latitude"
  ],
  "ddof": 0,
  "fit_rows": 4000,
  "fit_split": "official is_test=False only",
  "longitude_mean": 12.379921279025,
  "longitude_std": 77.44045491300216,
  "latitude_mean": 27.0578063215,
  "latitude_std": 27.952529089660754
}
```

## Primary probe: Ridge(alpha=2560)

```text
 layer  relative_depth  n_train  n_test  hidden_dim  alpha  r2_overall  r2_longitude  r2_latitude  baseline_r2_overall  baseline_r2_longitude  baseline_r2_latitude  fit_predict_runtime_seconds
     0        0.000000     4000    1000        2560 2560.0    0.197255      0.240779     0.153731            -0.002122              -0.004043             -0.000202                     0.349397
     7        0.225806     4000    1000        2560 2560.0    0.432988      0.467203     0.398773            -0.002122              -0.004043             -0.000202                     0.247641
    15        0.483871     4000    1000        2560 2560.0    0.554140      0.567704     0.540576            -0.002122              -0.004043             -0.000202                     0.231165
    23        0.741935     4000    1000        2560 2560.0    0.720311      0.746989     0.693633            -0.002122              -0.004043             -0.000202                     0.228702
    31        1.000000     4000    1000        2560 2560.0    0.723285      0.749349     0.697221            -0.002122              -0.004043             -0.000202                     0.238806
```

## RidgeCV sensitivity analysis (ours, not claimed as the paper's Pythia setting)

Grid: `[10.0, 17.78279410038923, 31.622776601683796, 56.23413251903491, 100.0, 177.82794100389228, 316.2277660168379, 562.341325190349, 1000.0, 1778.2794100389226, 2560.0, 3162.277660168379, 5623.41325190349, 10000.0, 17782.794100389227, 31622.776601683796, 56234.1325190349, 100000.0]`

```text
 layer  relative_depth  n_train  n_test  hidden_dim  selected_alpha  r2_overall  r2_longitude  r2_latitude  baseline_r2_overall  baseline_r2_longitude  baseline_r2_latitude  fit_predict_runtime_seconds                      analysis_label
     0        0.000000     4000    1000        2560     1000.000000    0.212520      0.261614     0.163426            -0.002122              -0.004043             -0.000202                     7.128151 our robustness/sensitivity analysis
     7        0.225806     4000    1000        2560     2560.000000    0.432988      0.467203     0.398773            -0.002122              -0.004043             -0.000202                     6.428342 our robustness/sensitivity analysis
    15        0.483871     4000    1000        2560     1778.279410    0.556586      0.570290     0.542882            -0.002122              -0.004043             -0.000202                     6.534371 our robustness/sensitivity analysis
    23        0.741935     4000    1000        2560     5623.413252    0.722089      0.746499     0.697678            -0.002122              -0.004043             -0.000202                     6.558927 our robustness/sensitivity analysis
    31        1.000000     4000    1000        2560     5623.413252    0.729911      0.754006     0.705816            -0.002122              -0.004043             -0.000202                     6.427865 our robustness/sensitivity analysis
```

## Baseline and negative control

```json
{
  "baseline": {
    "r2_overall": -0.002122292590810426,
    "r2_longitude": -0.0040426576113412604,
    "r2_latitude": -0.00020192757027959196
  },
  "random_label_control": {
    "layer": 15,
    "seed": 42,
    "shuffle_scope": "training targets only",
    "test_targets_unchanged": true,
    "alpha": 2560.0,
    "r2_overall": -0.07236984520621437,
    "r2_longitude": -0.08085061694217854,
    "r2_latitude": -0.0638890734702513,
    "normal_layer15_r2_overall": 0.5541402437995666,
    "r2_drop_vs_normal": 0.626510089005781,
    "fit_predict_runtime_seconds": 0.2516332305967808
  }
}
```

## Coefficient sanity checks

```text
                     analysis  layer coef_shape intercept_shape  coef_l2_longitude  coef_l2_latitude  intercept_l2  coef_nan_count  coef_inf_count
                  fixed_alpha      0  [2, 2560]             [2]           0.357977          0.344744      0.218755               0               0
          ridgecv_sensitivity      0  [2, 2560]             [2]           0.581061          0.581023      0.299853               0               0
                  fixed_alpha      7  [2, 2560]             [2]           0.383959          0.411382      0.176808               0               0
          ridgecv_sensitivity      7  [2, 2560]             [2]           0.383959          0.411382      0.176807               0               0
                  fixed_alpha     15  [2, 2560]             [2]           0.353103          0.357004      0.167544               0               0
          ridgecv_sensitivity     15  [2, 2560]             [2]           0.407807          0.415629      0.237813               0               0
                  fixed_alpha     23  [2, 2560]             [2]           0.247021          0.264028      0.788835               0               0
          ridgecv_sensitivity     23  [2, 2560]             [2]           0.182253          0.189698      0.545964               0               0
                  fixed_alpha     31  [2, 2560]             [2]           0.236872          0.255374      0.281446               0               0
          ridgecv_sensitivity     31  [2, 2560]             [2]           0.172260          0.184942      0.197795               0               0
layer15_shuffled_train_labels     15  [2, 2560]             [2]           0.345083          0.340159      0.354593               0               0
```

## Gate checklist

- [x] official_split_sampled_separately
- [x] no_train_test_row_overlap
- [x] row_alignment_10_of_10
- [x] only_requested_five_layers
- [x] all_shapes_5000x2560_float16
- [x] no_activation_nan_inf
- [x] train_only_target_scaling
- [x] target_order_longitude_latitude
- [x] fixed_alpha_2560_all_layers
- [x] ridgecv_grid_contains_2560
- [x] all_probe_outputs_finite
- [x] coefficient_shapes_and_values_valid
- [x] layers_not_numerically_identical
- [x] random_labels_materially_degrade_layer15
- [x] upstream_clean

## Known irreducible reproduction differences

- The paper/released code does not pin a Pythia revision; this run pins current `step143000` and records its commit.
- Current `main_probe_experiment` publishes RidgeCV grids only for Llama models, not Pythia. Therefore Ridge(alpha=2560) is the released `probe=None` rule; the RidgeCV result above is explicitly our sensitivity analysis.
- `place_probe_experiment` also computes an O(n²) proximity matrix. Gate 3 requests R² only, so this bounded run reproduces its train-only scaling, fitting, unscaling, and exact R² calls without allocating that unrelated matrix.

Final Gate 3 decision: **PASS**
