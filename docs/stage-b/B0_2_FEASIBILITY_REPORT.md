# StepGame Stage B0.2 Split Feasibility Report

## Technical summary

The frozen-data design is **PASS_WITH_LIMITATION** under FULL_OK parsing, five-sentence stories, eight non-overlap relations, independently computed hop, template-clean filtering, story-level grouping, and exact-duplicate representative retention. This report is generated before final manifests are frozen.

- Primary retains **13,758** candidates after excluding **0** repeated exact rows.
- Balanced control reaches **56 samples/class/hop**, with per-cell train/val/test quotas of **34/11/11**.
- Template held-out adopts **query_path_template_family**; strict story-level feasibility at the >=10 evaluation-cell gate is **False**.
- Composition held-out contains **932** signatures in **932** leakage-safe components.

## Scope and parsing policy

`FULL_OK` requires a complete story/question parse, query entities, canonical explicit edges, template metadata, a reconstructable graph path, and path metadata. `PATH_ONLY` has a query path but lacks at least one complete Primary requirement. Primary uses FULL_OK only.

| status | rows |
| --- | ---: |
| FULL_OK | 190,237 |
| PATH_ONLY | 10,574 |
| FAILED | 9,189 |

## Primary natural-distribution split

- Candidate rows: **13,758**
- Sizes: train **9,630**, val **2,065**, test **2,063**
- Minimum cells: train **39**, val **8**, test **9**
- Maximum absolute per-cell share deviation from 70/15/15: **0.0121**
- Story-group overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

| split | computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1 | 91 | 99 | 90 | 87 | 87 | 86 | 83 | 89 |
| test | 2 | 64 | 59 | 63 | 65 | 58 | 68 | 60 | 68 |
| test | 3 | 46 | 43 | 39 | 63 | 65 | 38 | 65 | 63 |
| test | 4 | 26 | 26 | 28 | 41 | 43 | 28 | 45 | 45 |
| test | 5 | 12 | 14 | 11 | 27 | 24 | 9 | 20 | 25 |
| train | 1 | 424 | 463 | 422 | 409 | 410 | 401 | 385 | 416 |
| train | 2 | 300 | 270 | 293 | 309 | 271 | 317 | 277 | 311 |
| train | 3 | 215 | 197 | 182 | 292 | 305 | 184 | 308 | 288 |
| train | 4 | 117 | 117 | 135 | 195 | 204 | 127 | 209 | 212 |
| train | 5 | 61 | 65 | 51 | 123 | 113 | 39 | 96 | 117 |
| val | 1 | 91 | 99 | 91 | 88 | 88 | 86 | 82 | 89 |
| val | 2 | 64 | 58 | 63 | 67 | 58 | 68 | 59 | 66 |
| val | 3 | 46 | 42 | 39 | 63 | 65 | 40 | 66 | 61 |
| val | 4 | 25 | 25 | 29 | 42 | 44 | 27 | 45 | 46 |
| val | 5 | 14 | 14 | 11 | 26 | 24 | 8 | 21 | 25 |

## Strict balanced control

- B0.1 theoretical ceiling after current gates: **56 samples/class/hop**
- Feasible ceiling under group and split constraints: **56 samples/class/hop**
- Exact per-cell train/val/test: **34/11/11**
- Total: **2,240**

| split | computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| test | 2 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| test | 3 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| test | 4 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| test | 5 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| train | 1 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| train | 2 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| train | 3 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| train | 4 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| train | 5 | 34 | 34 | 34 | 34 | 34 | 34 | 34 | 34 |
| val | 1 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| val | 2 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| val | 3 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| val | 4 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| val | 5 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |

## Template-family held-out feasibility

### Strict story-level definition

- Included rows: **222**
- Split sizes: `{'train': 164, 'val': 23, 'test': 35}`
- Covered cells: `{'train': 36, 'val': 15, 'test': 21}`
- Minimum cells: `{'train': 0, 'val': 0, 'test': 0}`
- Meets >=10 val/test cell gate: **False**

### Query-path definition

- Included rows: **6,321**
- Split sizes: `{'train': 2756, 'val': 1644, 'test': 1921}`
- Families: `{'train': ['generic_directional'], 'val': ['clock_face', 'compass', 'diagonal', 'vertical'], 'test': ['horizontal', 'relative_vertical']}`
- Covered cells: `{'train': 38, 'val': 30, 'test': 30}`
- Minimum cells: `{'train': 0, 'val': 0, 'test': 0}`
- Family overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

Adopted definition: **query_path_template_family**. Distractor templates may recur only under the query-path definition; query-path family sets remain disjoint.

| split | computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1 | 162 | 191 | 134 | 238 | 218 | 110 | 201 | 220 |
| test | 2 | 44 | 38 | 66 | 38 | 39 | 50 | 26 | 55 |
| test | 3 | 8 | 10 | 3 | 16 | 14 | 3 | 10 | 13 |
| test | 4 | 0 | 1 | 1 | 4 | 2 | 3 | 3 | 0 |
| test | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 1 | 263 | 283 | 387 | 144 | 175 | 379 | 172 | 156 |
| train | 2 | 71 | 59 | 66 | 68 | 74 | 78 | 71 | 71 |
| train | 3 | 21 | 18 | 16 | 27 | 34 | 26 | 20 | 19 |
| train | 4 | 2 | 6 | 6 | 4 | 6 | 1 | 10 | 9 |
| train | 5 | 1 | 1 | 0 | 3 | 3 | 0 | 5 | 1 |
| val | 1 | 181 | 187 | 82 | 202 | 192 | 84 | 177 | 218 |
| val | 2 | 28 | 27 | 43 | 30 | 24 | 40 | 21 | 34 |
| val | 3 | 6 | 5 | 5 | 8 | 10 | 3 | 13 | 14 |
| val | 4 | 1 | 2 | 0 | 0 | 1 | 1 | 2 | 3 |
| val | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Composition-held-out feasibility

- Split sizes: `{'train': 5901, 'val': 1501, 'test': 1600}`
- Signature counts: `{'train': 649, 'val': 142, 'test': 141}`
- Signature overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Covered hop×relation cells: `{'train': 32, 'val': 32, 'test': 32}`
- Minimum cells: `{'train': 39, 'val': 8, 'test': 9}`
- Maximum relation share: `{'train': 0.14980511777664804, 'val': 0.1665556295802798, 'test': 0.148125}`

| split | computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2 | 115 | 93 | 118 | 104 | 63 | 125 | 60 | 66 |
| test | 3 | 42 | 38 | 41 | 64 | 69 | 40 | 67 | 66 |
| test | 4 | 25 | 26 | 29 | 42 | 44 | 28 | 45 | 46 |
| test | 5 | 13 | 14 | 11 | 27 | 24 | 9 | 21 | 25 |
| train | 2 | 235 | 220 | 247 | 276 | 219 | 260 | 228 | 264 |
| train | 3 | 224 | 203 | 188 | 291 | 299 | 185 | 305 | 282 |
| train | 4 | 118 | 116 | 134 | 194 | 203 | 126 | 209 | 211 |
| train | 5 | 61 | 65 | 51 | 123 | 113 | 39 | 95 | 117 |
| val | 2 | 78 | 74 | 54 | 61 | 105 | 68 | 108 | 115 |
| val | 3 | 41 | 41 | 31 | 63 | 67 | 37 | 67 | 64 |
| val | 4 | 25 | 26 | 29 | 42 | 44 | 28 | 45 | 46 |
| val | 5 | 13 | 14 | 11 | 26 | 24 | 8 | 21 | 25 |

## Derived Target-Absent subset preserves Primary membership

- Total: **4,023**
- Hop totals: `{'2': 1686, '3': 1060, '4': 887, '5': 390}`
- Minimum overall hop×relation cell: **22**
- Per-split minimum cells: `{'train': 15, 'val': 3, 'test': 4}`
- h=5 split sizes: `{'train': 273, 'val': 58, 'test': 59}`

| split | computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 2 | 37 | 34 | 37 | 26 | 21 | 43 | 27 | 29 |
| test | 3 | 17 | 16 | 13 | 23 | 26 | 16 | 26 | 23 |
| test | 4 | 13 | 15 | 16 | 14 | 18 | 17 | 19 | 19 |
| test | 5 | 5 | 5 | 4 | 12 | 10 | 4 | 9 | 10 |
| train | 2 | 173 | 155 | 175 | 124 | 95 | 202 | 123 | 133 |
| train | 3 | 77 | 71 | 58 | 106 | 122 | 78 | 125 | 105 |
| train | 4 | 60 | 69 | 78 | 68 | 87 | 76 | 91 | 92 |
| train | 5 | 26 | 24 | 18 | 52 | 44 | 15 | 46 | 48 |
| val | 2 | 37 | 33 | 38 | 27 | 20 | 43 | 26 | 28 |
| val | 3 | 16 | 15 | 12 | 23 | 26 | 17 | 27 | 22 |
| val | 4 | 13 | 15 | 17 | 15 | 19 | 16 | 20 | 20 |
| val | 5 | 6 | 5 | 4 | 11 | 9 | 3 | 10 | 10 |

## Gate and limitations

Gate status: **PASS_WITH_LIMITATION**. Hard checks: `{'primary_all_40_cells': True, 'primary_zero_story_leakage': True, 'balanced_eval_cells_at_least_10': True, 'balanced_zero_story_leakage': True, 'template_all_splits_nonempty': True, 'template_zero_family_overlap': True, 'template_zero_story_leakage': True, 'composition_all_32_cells': True, 'composition_zero_signature_overlap': True, 'composition_zero_story_leakage': True}`. Limitation checks: `{'template_all_40_cells': False, 'template_eval_min_cell_at_least_10': False, 'primary_eval_min_cell_at_least_20': False}`.

- Template families are the seven broad B0.1 semantic/surface families, not individual template IDs.
- Query-path held-out does not claim that distractor sentence families are unseen.
- Composition components merge signatures linked by repeated-story groups; this is stricter than row-level signature assignment.
- Primary validation/test bottleneck cells below 20 are reported, not oversampled.

## Recommended next step

If the feasibility gate is accepted, freeze exactly these provisional row IDs, calculate SHA-256 checksums, and run an independent validation pass. Do not use model outputs to revise any assignment.
