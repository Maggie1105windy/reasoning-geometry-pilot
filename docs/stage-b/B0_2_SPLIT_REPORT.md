# StepGame Stage B0.2 Frozen Split Report

## Technical summary

The B0.2 manifests are frozen with seed **20260823** and status **PASS_WITH_LIMITATION**. Primary, balanced, and composition splits satisfy their coverage and zero-leakage gates. Template held-out is frozen as a query-path broad-family diagnostic with zero family/story leakage, but it cannot support a complete 40-cell comparison.

## Frozen Primary split preserves the natural candidate distribution

- Candidates: **13,758**
- Train/val/test: **9,630/2,065/2,063**
- Minimum hop×relation cells: **39/8/9**
- Story-group and exact-group overlaps: `{'train_val': 0, 'train_test': 0, 'val_test': 0}` / `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

Primary keeps every eligible row after the candidate filter because no exact-duplicate group survives twice within this cohort. Evaluation cells below 20 remain unchanged and must be reflected in uncertainty estimates downstream.

## Balanced control reaches the theoretical ceiling

- Feasible N/class/hop: **56**
- Exact per-cell train/val/test: **34/11/11**
- Total: **2,240**
- Leakage: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

## Template held-out is restricted-support, not a complete factorial control

- Definition: **query_path_template_family**
- Families: `{'train': ['generic_directional'], 'val': ['clock_face', 'compass', 'diagonal', 'vertical'], 'test': ['horizontal', 'relative_vertical']}`
- Rows: **6,321**; split sizes `{'train': 2756, 'val': 1644, 'test': 1921}`
- Covered cells: `{'train': 38, 'val': 30, 'test': 30}`; minimum cells `{'train': 0, 'val': 0, 'test': 0}`
- Family overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

The seven B0.1 broad families cannot be partitioned three ways while preserving all 40 hop×relation cells. Results from this manifest must therefore be compared only on shared supported cells, and must be labeled query-path template-family held-out; distractor families may repeat.

## Composition held-out preserves all derived cells

- Unordered signatures: **932** in **932** leakage-safe components
- Split signatures: `{'train': 649, 'val': 142, 'test': 141}`
- Split rows: `{'train': 5901, 'val': 1501, 'test': 1600}`
- Minimum hop×relation cells: `{'train': 39, 'val': 8, 'test': 9}`
- Signature overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

## Target-Absent remains a Primary-membership subset

- Hop totals: `{'2': 1686, '3': 1060, '4': 887, '5': 390}`
- Split rows: `{'train': 2816, 'val': 603, 'test': 604}`
- Minimum overall relation cell: **22**
- Per-split minimum cells: `{'train': 15, 'val': 3, 'test': 4}`

## Parsing, grouping, and reproducibility contract

- FULL_OK/PATH_ONLY/FAILED: **190,237/10,574/9,189**
- PATH_ONLY is excluded because complete explicit-edge and template/path metadata are required by held-out controls.
- Every manifest uses `story_group_id` as the minimum grouping unit; exact duplicates cannot cross a split.
- Checksums are stored in `split_checksums.json`; Stage B1 must verify them before use.

## Limitations and next decision

Primary test cells have a minimum of **9**, and the Target-Absent test minimum is **4**. The Template-held-out manifest has unsupported cells. These limitations do not invalidate Primary/Balanced/Composition, but they require an explicit human decision before Stage B1.
