# Stage A2 Full Reproduction Report

Status: **PASS — Strong best-effort reproduction**

## 1. Goal

Complete the full World × 32-layer Pythia-2.8B spatial linear probing experiment and compare its layer-wise held-out R² curve with Gurnee & Tegmark Figure 2.

## 2. Paper setup

The [official paper](https://arxiv.org/html/2310.02207) reports out-of-sample R² for every layer/model/dataset in Figure 2 and describes spatial/temporal decodability as rising through early layers before reaching a plateau.

## 3. Reproduction scope

- Best-effort methodological reproduction; not an exact checkpoint/hyperparameter reproduction
- Full 39,585-row released World dataset: 31,668 official train and 7,917 official test
- All 32 Pythia-2.8B transformer blocks
- `coords` prompt and original row ordering
- One independent fixed-alpha Ridge probe per layer

## 4. Known unavoidable deviations

- The paper/released code does not pin the Pythia revision. This run pins current non-deduplicated `step143000` at commit `dbe7ae300a54abcdc475a33907b3dff81d25709f`.
- The paper states that lambda is tuned by efficient leave-one-out cross-validation, but current released `RIDGE_ALPHAS` publishes no Pythia grid. Following the pre-registered Gate 4 policy, this run uses the released `probe=None` fallback `Ridge(alpha=activation_dimension)=Ridge(alpha=2560)`.
- TransformerLens 1.14.0 and the current model conversion may differ from the authors' unpinned environment.
- Official per-channel quint8 quantization is applied per shard/layer to support recovery, rather than after assembling an entire layer.

## 5. Model

- `EleutherAI/pythia-2.8b`, revision `step143000`
- Hugging Face commit: `dbe7ae300a54abcdc475a33907b3dff81d25709f`
- Weight SHA-256: `a294bfcf6ff2c091091d04489a3f1c233b4106d74201e152a01f492d24e6fbaf`
- 32 blocks; hidden size 2560; frozen float16 inference

## 6. Dataset

The original 39,585 row order and row IDs are preserved. The released `is_test` mask yields exactly 31,668 training rows and 7,917 testing rows; no resampling, filtering, or repartitioning was performed.

## 7. Prompt/tokenization

`What are the lat/lon coordinates of <entity>` with released Pythia tokenization, explicit BOS, EOS padding, right padding, and released `entity_mask`.

## 8. Activation definition

Entity-last-token activation from `blocks.{layer}.hook_resid_post`, layers 0-31; embedding excluded. Extraction imports the released `get_layer_activations_tl()` path already validated in Gates 2 and 3.

- 78 resumable atomic shards; every layer reconstructs to `[39585, 2560]`
- Official per-channel quint8 storage; dequantized to float32 before sklearn
- Aggregate forward runtime: 286.770 seconds (138.037 samples/s)
- Quantize/save runtime: 57.554 seconds
- Peak allocated/reserved VRAM: 5.351 / 5.588 GiB
- Stored layer files: 3.072 GiB
- Ten-row × five-layer live check: min cosine 0.999870837, mean MAE 0.00984151762, max absolute error 0.480514526

## 9. Probe definition

Targets are `[longitude, latitude]`. Mean/std are computed using all and only the 31,668 official training rows. Each layer creates a new `Ridge(alpha=2560)`; test labels are used only for final scoring. Overall R² is sklearn's default uniform average across the two outputs.

```json
{
  "target_order": [
    "longitude",
    "latitude"
  ],
  "ddof": 0,
  "fit_split": "all official is_test=False rows",
  "fit_rows": 31668,
  "longitude_mean": 14.906220305833743,
  "longitude_std": 78.03314672628153,
  "latitude_mean": 26.505868671428864,
  "latitude_std": 28.377352306499102
}
```

## 10. Full-data results

```text
 layer  relative_depth  n_train  n_test  hidden_dim  alpha  r2_overall  r2_longitude  r2_latitude  fit_runtime_seconds  coef_l2_norm
     0        0.000000    31668    7917        2560 2560.0    0.295067      0.341014     0.249121             0.879512      1.117695
     1        0.032258    31668    7917        2560 2560.0    0.356391      0.405338     0.307443             0.701402      1.116726
     2        0.064516    31668    7917        2560 2560.0    0.359690      0.411390     0.307990             0.744371      1.058182
     3        0.096774    31668    7917        2560 2560.0    0.362861      0.413833     0.311889             0.675314      0.992430
     4        0.129032    31668    7917        2560 2560.0    0.401761      0.450024     0.353497             0.683593      0.963379
     5        0.161290    31668    7917        2560 2560.0    0.475850      0.520206     0.431494             0.671271      0.869300
     6        0.193548    31668    7917        2560 2560.0    0.491619      0.531676     0.451562             0.690540      0.849627
     7        0.225806    31668    7917        2560 2560.0    0.512341      0.550735     0.473948             0.663715      0.838024
     8        0.258065    31668    7917        2560 2560.0    0.538504      0.576551     0.500458             0.713942      0.850053
     9        0.290323    31668    7917        2560 2560.0    0.535602      0.572713     0.498491             0.669142      0.878545
    10        0.322581    31668    7917        2560 2560.0    0.536019      0.572742     0.499295             0.702596      0.883663
    11        0.354839    31668    7917        2560 2560.0    0.548374      0.583882     0.512866             0.671346      0.903099
    12        0.387097    31668    7917        2560 2560.0    0.547847      0.581711     0.513983             0.663348      0.908194
    13        0.419355    31668    7917        2560 2560.0    0.557140      0.586040     0.528240             0.648688      0.904774
    14        0.451613    31668    7917        2560 2560.0    0.570075      0.602090     0.538059             0.656354      0.896996
    15        0.483871    31668    7917        2560 2560.0    0.624598      0.642329     0.606866             0.645972      0.811050
    16        0.516129    31668    7917        2560 2560.0    0.632485      0.644942     0.620027             0.714259      0.785801
    17        0.548387    31668    7917        2560 2560.0    0.646297      0.658681     0.633914             0.679496      0.761500
    18        0.580645    31668    7917        2560 2560.0    0.690809      0.707529     0.674089             0.681743      0.631549
    19        0.612903    31668    7917        2560 2560.0    0.710215      0.725814     0.694616             0.682812      0.591175
    20        0.645161    31668    7917        2560 2560.0    0.715095      0.729602     0.700587             0.851285      0.576653
    21        0.677419    31668    7917        2560 2560.0    0.722879      0.738432     0.707326             0.693194      0.525842
    22        0.709677    31668    7917        2560 2560.0    0.728647      0.747191     0.710103             0.748973      0.510998
    23        0.741935    31668    7917        2560 2560.0    0.734186      0.753753     0.714619             0.754142      0.484644
    24        0.774194    31668    7917        2560 2560.0    0.735619      0.755980     0.715259             0.670657      0.472800
    25        0.806452    31668    7917        2560 2560.0    0.737433      0.757572     0.717293             0.750605      0.466623
    26        0.838710    31668    7917        2560 2560.0    0.738304      0.758531     0.718078             0.763898      0.457770
    27        0.870968    31668    7917        2560 2560.0    0.740618      0.760357     0.720879             0.763676      0.447977
    28        0.903226    31668    7917        2560 2560.0    0.742174      0.761767     0.722581             0.828422      0.437154
    29        0.935484    31668    7917        2560 2560.0    0.742287      0.761492     0.723083             0.792843      0.434651
    30        0.967742    31668    7917        2560 2560.0    0.743157      0.761674     0.724641             0.685009      0.430251
    31        1.000000    31668    7917        2560 2560.0    0.741863      0.761020     0.722707             0.769952      0.429012
```

Peak overall R² is **0.743157 at layer 30**. Layer 31 is 0.741863.

## 11. Layer-wise curve

The curve starts at 0.295, rises in several early/middle steps, reaches 0.710 at layer 19, and then gains slow substantially. The descriptive plateau onset is therefore around layers 19-21 (relative depth 0.61-0.68); this is not presented as a formal paper threshold.

See `results/a2_4_full_layer_r2.png`. No smoothing, interpolation, or layer deletion was applied.

## 12. Longitude vs latitude

At the peak-overall layer 30, longitude R² is 0.761674 and latitude R² is 0.724641. Across the late plateau longitude remains modestly more decodable than latitude. See `results/a2_4_lat_lon_layer_r2.png`.

## 13. Controls

Train-mean baseline overall/longitude/latitude R²: `-1.35417943e-05` / `-1.63536279e-06` / `-2.54482259e-05`.

```text
 layer  seed  alpha        shuffle_scope  test_labels_unchanged  normal_r2_overall  random_label_r2_overall  r2_difference_normal_minus_random  random_label_r2_longitude  random_label_r2_latitude  fit_runtime_seconds
    15    42 2560.0 training labels only                   True           0.624598                -0.043735                           0.668333                  -0.044549                 -0.042921             0.649264
    31    42 2560.0 training labels only                   True           0.741863                -0.074465                           0.816328                  -0.081297                 -0.067632             1.086329
```

Both random-label controls are negative and materially below their normal probes; the test labels were not shuffled.

## 14. Comparison with Gate 3

```text
 layer  full_r2_overall  full_r2_longitude  full_r2_latitude  mini_r2_overall  mini_r2_longitude  mini_r2_latitude  delta_full_minus_mini_overall  delta_full_minus_mini_longitude  delta_full_minus_mini_latitude
     0         0.295067           0.341014          0.249121         0.197255           0.240779          0.153731                       0.097813                         0.100236                        0.095390
     7         0.512341           0.550735          0.473948         0.432988           0.467203          0.398773                       0.079353                         0.083532                        0.075175
    15         0.624598           0.642329          0.606866         0.554140           0.567704          0.540576                       0.070457                         0.074625                        0.066290
    23         0.734186           0.753753          0.714619         0.720311           0.746989          0.693633                       0.013875                         0.006764                        0.020986
    31         0.741863           0.761020          0.722707         0.723285           0.749349          0.697221                       0.018578                         0.011671                        0.025486
```

The five shared points retain the same rank order (Spearman 1.000) and have Pearson correlation 0.997442. Full-data R² is higher at all five points by 0.014-0.098 (mean absolute difference 0.056015), while the early-to-late trend is stable.

## 15. Comparison with Figure 2

The [official repository](https://github.com/wesg52/world-models) does not publish a Pythia raw layer-wise numeric series: all 17 released `paper_results` notebooks and numeric artifact types were audited, with no Pythia result found. The comparison is therefore explicitly **approximate visual comparison**.

In Figure 2's World Map panel, the dashed purple Pythia-2.8B curve begins around 0.30-0.35, rises to roughly 0.60-0.65 near half depth, and reaches a late plateau around 0.70-0.73. Our corresponding values are 0.295 at layer 0, 0.632 at layer 16, and a 0.743 peak at layer 30. Thus the curve shape, numerical scale, peak region, and plateau timing are broadly consistent; our late plateau appears slightly higher by visual inspection.

No paper-curve MAE/Pearson/Spearman is computed because visual estimates are not official numbers and would create false precision.

## 16. Reproduction assessment

**Strong best-effort reproduction.** The qualitative layer-wise trend agrees with Figure 2, the R² scale and late plateau are close, the full integrity and negative-control gates pass, and every known non-identical or unpinned setting is recorded. This classification does not mean exact checkpoint- or hyperparameter-level reproduction.

## 17. Limitations

These results support only that spatial information is linearly decodable from Pythia hidden representations and that decodability changes with depth. They do not establish that the model uses this representation, grounding, symbolic reasoning, or a causal mechanism.

## 18. Exact artifact paths

- `/home/ai/projects/linear-probe-repro/results/A2_FULL_REPRODUCTION_REPORT.md`
- `/home/ai/projects/linear-probe-repro/results/a2_4_full_sample_manifest.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_full_layer_r2.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_full_layer_r2.png`
- `/home/ai/projects/linear-probe-repro/results/a2_4_lat_lon_layer_r2.png`
- `/home/ai/projects/linear-probe-repro/results/a2_4_random_label_control.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_mean_baseline.json`
- `/home/ai/projects/linear-probe-repro/results/a2_4_target_scaler.json`
- `/home/ai/projects/linear-probe-repro/results/a2_4_activation_stats.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_live_forward_comparison.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_full_activation_integrity.md`
- `/home/ai/projects/linear-probe-repro/results/a2_4_gate3_comparison.csv`
- `/home/ai/projects/linear-probe-repro/results/a2_4_official_comparison_audit.md`
- `/home/ai/projects/linear-probe-repro/results/a2_4_figure2_visual_reference.json`
- `/home/ai/projects/linear-probe-repro/activations/full/pythia-2.8b/world/coords/metadata.json`
- `/home/ai/projects/linear-probe-repro/activations/full/pythia-2.8b/world/coords/integrity_report.json`
- `/home/ai/projects/linear-probe-repro/scripts/gate4_a2_full_repro.py`

`external/world-models` clean at finalization: **True**.
