# Stage B0.2c — Test-only Template Holdout

## Technical summary

Stage B0.2c **PASS**. The Mid-level test-only split passes the strict adoption Gate and is frozen as the formal Template robustness control. It uses 4 reserved mid-level groups and partitions the complete 13,758-row Primary candidate cohort into 10,217 train, 1,802 validation, and 1,739 test rows. Test covers 40/40 hop×relation cells with minimum cell 14; held-out-group leakage and story leakage are both zero. Surface was not run as a formal attempt because the stopping rule requires immediate adoption after Mid-level passes.

## The frozen control tests unseen reserved templates

`T_test` is the reserved Mid-level group set. Every candidate row containing at least one member of `T_test` is assigned to test, and no such group occurs in train or validation. The remaining rows are split 85/15 within each hop×relation cell by deterministic `story_group_id` hashing (seed 20260823). Train and validation may share non-held-out groups, as explicitly allowed.

The MILP first enforces at least 10 test rows in every cell and caps test at 25% of each source cell to preserve a larger, balanced development pool. Within those constraints it maximizes test rows, then held-out group diversity. This balance guardrail is an implementation choice, not a model-derived criterion.

Solver optimality proven: `False`; feasible incumbent used: `True`; reported MIP gap: `0.9312499140157988`. A feasible incumbent is adoption-eligible only after direct bound checks and the independent Gate validator pass; global objective optimality is not part of the adoption Gate.

## Mid-level passes every adoption Gate

| Split | Rows | Supported cells | Min cell | Max cell |
|---|---:|---:|---:|---:|
| Train | 10,217 | 40/40 | 36 | 530 |
| Validation | 1,802 | 40/40 | 6 | 93 |
| Test | 1,739 | 40/40 | 14 | 81 |

- `T_test ∩ T_train = 0`
- `T_test ∩ T_val = 0`
- story-group overlap across all split pairs = 0
- exact-duplicate-group overlap across all split pairs = 0
- suspicious templates = 0; parsing status is `FULL_OK`; hops are 1–5; relations are exactly the eight non-overlap classes

Validation minimum cell is 6, below the preferred 10 but still covers all 40 cells without oversampling. This follows from the rare source cells and the requested 15% validation share of the post-holdout development pool.

## Scope and definitions

The cohort is unchanged from B0.2: `FULL_OK`, five-sentence stories, computed hop 1–5, exactly eight non-overlap relations, and no suspicious templates, grouped by `story_group_id`. Mid-level families are exactly the previously audited B0.2b definitions; no fourth taxonomy and no model outcome were used.

The formal leakage metric follows the request's set definition: reserved `T_test` groups must be absent from development data. A test row may also contain a non-reserved group elsewhere on its multi-edge query path. Consequently, non-held-out observed-group overlaps with train and validation are 89 and 89, respectively. These are expected co-occurring context, not leakage of `T_test`; they mean the control tests paths containing at least one unseen mid-level template, not paths whose every edge template is unseen. Requiring every observed test-path group to be unseen is infeasible because the 93 Mid-level groups form one connected co-occurrence component.

## Broad restricted support remains unchanged

The archived Broad Template-Family Restricted Support Control is unchanged. Its common support remains 28/40 cells and usable hops remain 1–4; missing cells must not be scored as zero.

## Validation and limitations

An independent validator reconstructs the candidate cohort and group membership from source metadata, confirms exact row/source-field agreement, verifies the `T_test` assignment rule, all leakage and domain gates, manifest checksums, and immutability of the protected B0.2 and broad-control files. The protocol does not claim that all templates occurring anywhere in a test story are unseen; it claims the reserved query-path groups are unseen throughout probe development.

## Recommended next step

Stage B0 is complete and Stage B1 may begin using the now-frozen manifests. Base, SFT, DPO, and Final/RLVR must all reuse these exact row IDs and checksums; no future model result may alter the split.

## Required completion record

### 1. Mid-level Test-only

- template groups total: 93
- held-out test groups: 4
- train rows: 10217
- val rows: 1802
- test rows: 1739
- test supported cells: 40/40
- min train cell: 36
- min val cell: 6
- min test cell: 14
- test template leakage: 0
- story leakage: 0
- status: PASS

### 2. Surface Test-only

- 是否运行: NO
- held-out test groups: N/A
- train rows: N/A
- val rows: N/A
- test rows: N/A
- test supported cells: N/A
- min test cell: N/A
- template leakage: N/A
- story leakage: N/A
- status: NOT_RUN_AFTER_MID_PASS

### 3. Final adopted Template control

- grouping: `mid_level_template_family`
- test-only holdout rule: `T_test ∩ (T_train ∪ T_val) = ∅`; every row containing `T_test` is test
- train: 10217
- val: 1802
- test: 1739
- 40/40: YES
- min test cell: 14
- checksum: `896907306c440116682211ff7077ea833c8be9a3eb3a32fb2613bb6baff8845c`

### 4. Broad restricted control

- unchanged: YES
- common cells: 28/40
- usable hops: 1–4

### 5. Final B0 status

- Primary: PASS / frozen unchanged
- Balanced: PASS / frozen unchanged
- Target-Absent: PASS / frozen unchanged
- Composition: PASS / frozen unchanged
- Template: PASS / Mid-level Test-only frozen
- overall: PASS

### 6. 是否进入 Stage B1

- YES
- reason: all frozen B0 controls pass; Mid-level test-only meets 40/40 coverage, minimum test cell ≥5, and zero held-out-template/story leakage

### 7. 输出路径

- report: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b0_2c/B0_2C_TEMPLATE_TEST_REPORT.md`
- protocol: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b0_2c/template_test_protocol.json`
- manifests: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b0_2c/mid_template_test_train.csv`, `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b0_2c/mid_template_test_val.csv`, `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b0_2c/mid_template_test_test.csv`
