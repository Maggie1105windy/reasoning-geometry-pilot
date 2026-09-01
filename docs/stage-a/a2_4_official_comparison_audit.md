# Gate 4 Official-result Source Audit

## Sources checked

- [Official paper / arXiv HTML](https://arxiv.org/html/2310.02207), especially Figure 2 and Section 3.1.
- [Official world-models repository](https://github.com/wesg52/world-models) at local commit `a572f162948ee185e9c842eb5fa15e23aad3d218`.
- All 17 notebooks under `external/world-models/paper_results`.
- Repository branches/tags and numeric artifacts under `paper_results`.

## Finding

No official Pythia layer-wise numeric series was found. The 17 released notebooks contain no `Pythia` string, and `paper_results` contains no CSV, JSON, pickle, NumPy, or similar numeric result artifact. The shallow clone exposes only the official `main` branch at the pinned commit and no tag carrying result data.

Figure 2 is therefore the highest-priority available source for this comparison. The World Map panel visibly identifies Pythia-2.8B as the dashed purple curve. All values mentioned from that curve are explicitly approximate visual ranges, not reconstructed official measurements.

## Paper-level comparison

The paper describes representations as increasing smoothly through the first half of layers before reaching a plateau. The visually estimated Pythia-2.8B World curve begins around 0.30-0.35, reaches roughly 0.60-0.65 near half depth, and plateaus around 0.70-0.73 late in the model. Our curve begins at `0.295067`, reaches `0.632485` at layer 16, and peaks at `0.743157` at layer 30.

Because the comparison source is visual, no layer-wise MAE or Pearson/Spearman statistic against the paper curve is reported. Computing such metrics would imply false precision.
