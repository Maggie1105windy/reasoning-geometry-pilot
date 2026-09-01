# StepGame Stage B0.1 Data Audit

## Technical summary

CompleteVersion clean contains **210,000** records, not the 610,000 implied by the current README's 30k/1k/30k-per-K prose. The actual checked-in files contain 10k train, 1k valid, and 10k test examples per K for K=1..10.

The parser matches every one of the 1,155,000 clean sentences to official template source. **190,237 / 210,000 (90.59%)** records have every sentence mapped to a defensible two-entity semantic edge; **200,811** still provide a query path.

`qaK` is not the actual query hop: only **36,693 / 210,000 (17.47%)** records have K equal to the independently computed query distance. This is explained by the official generator constructing a K-edge chain and then sampling any entity pair. Treating the mismatch as corruption would be incorrect.

A fixed five-sentence comparison is data-feasible only after redefining K as source story length and using `computed_hop` as the experimental hop. Under the literal requested `hop_match=True` rule, h=1..4 cells are zero; under the audited semantics, the template-clean eight-class balanced ceiling is **56 per relation per hop**.

## The official source is frozen, but README size prose disagrees with files

- Repository: `https://github.com/ShiZhengyan/StepGame.git`
- Commit: `5e6aff1563a6d7f46ee1f1aeff98b94e68c29005`
- Primary: `Dataset/CompleteVersion/clean`
- Actual CompleteVersion clean: train 100,000, valid 10,000, test 100,000; total 210,000
- Actual CompleteVersion noise: train 100,000, valid 10,000, test 100,000; total 210,000
- The 2024 repository is preferred over third-party repacks because it includes the corrected official templates and exact source provenance. Noise is inventoried but excluded from Primary metadata.

## Schema and metric definitions

The record grain is one story–question–label example. `story_len` is the number of sentences. `recorded_hop` is K recovered from `qaK_*`; no hop field exists inside repository records. `computed_hop` is the BFS shortest-path edge count between query entities in the independently parsed semantic graph. A target-absent record has no explicit sentence whose stated canonical relation equals the target label.

| raw_label | canonical_label | inverse_relation | count |
| --- | --- | --- | --- |
| above | ABOVE | BELOW | 22340 |
| below | BELOW | ABOVE | 22319 |
| left | LEFT | RIGHT | 22503 |
| lower-left | LOWER_LEFT | UPPER_RIGHT | 27641 |
| lower-right | LOWER_RIGHT | UPPER_LEFT | 27977 |
| overlap | OVERLAP | OVERLAP | 9130 |
| right | RIGHT | LEFT | 22083 |
| upper-left | UPPER_LEFT | LOWER_RIGHT | 27839 |
| upper-right | UPPER_RIGHT | LOWER_LEFT | 28168 |

## K and actual query distance are different variables

Only 36,693 records match. Of the 173,307 mismatches, 164,118 have a valid shorter query path and 9,189 lack a complete path because of non-defensible templates. `recorded_hop > computed_hop` occurs 164,118 times; `recorded_hop < computed_hop` occurs 0 times.

For `story_len=5`, the actual query-distance distribution is:

| computed_hop | samples | share |
| --- | --- | --- |
| 1 | 6793 | 33.72% |
| 2 | 5326 | 26.44% |
| 3 | 4055 | 20.13% |
| 4 | 2659 | 13.20% |
| 5 | 1314 | 6.52% |

This supports controlling story length while varying actual hop, but requires dropping `hop_match=True` as an exclusion because that predicate encodes the wrong interpretation of K.

## Template source contains ten errors and one unresolved meaning

| semantic_status | template_definitions |
| --- | --- |
| OK | 203 |
| ERROR | 10 |
| UNKNOWN | 1 |

All clean and noise sentences match official template skeletons. Equivalent/duplicate template ambiguity affects 46,586 clean and 75,030 noise sentences; unmatched count is zero in both. Eleven non-OK definitions affect 59,531 clean samples. Ten have a direction or argument error; one uses `front right`, which does not uniquely define upper-right in the canonical 2D system. Original source remains unchanged.

Template-clean, actually used definitions by relation:

| relation | clean_OK_templates |
| --- | --- |
| ABOVE | 28 |
| BELOW | 27 |
| LEFT | 24 |
| RIGHT | 20 |
| UPPER_LEFT | 26 |
| UPPER_RIGHT | 26 |
| LOWER_LEFT | 25 |
| LOWER_RIGHT | 27 |

The available clean pattern diversity is sufficient to attempt template-held-out splits, but ambiguous duplicate surface forms must be grouped by normalized surface skeleton rather than source template ID.

## Five-sentence eight-class feasibility

Counts below use parsed five-sentence records, actual query distance, and eight non-overlap targets. They intentionally do not enforce the misnamed K field.

| computed_hop | ABOVE | BELOW | LEFT | LOWER_LEFT | LOWER_RIGHT | RIGHT | UPPER_LEFT | UPPER_RIGHT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 792 | 875 | 807 | 787 | 767 | 758 | 736 | 797 |
| 2 | 562 | 520 | 549 | 557 | 528 | 572 | 530 | 589 |
| 3 | 379 | 359 | 355 | 558 | 565 | 351 | 587 | 562 |
| 4 | 231 | 226 | 257 | 375 | 383 | 240 | 373 | 394 |
| 5 | 113 | 121 | 87 | 233 | 213 | 78 | 198 | 227 |

- All parsed balanced ceiling: **78** per relation/hop.
- Template-clean balanced ceiling: **56** per relation/hop.
- Literal strict-hop-match ceiling: **0** because h=1..4 are empty.

## Target-absent reasoning controls remain sizeable

| computed_hop | all_parsed_target_absent | all_parsed_eligible_total | all_parsed_rate | template_clean_target_absent | template_clean_eligible_total | template_clean_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2219 | 4407 | 50.35% | 1686 | 3356 | 50.24% |
| 3 | 1426 | 3716 | 38.37% | 1060 | 2815 | 37.66% |
| 4 | 1190 | 2479 | 48.00% | 887 | 1881 | 47.16% |
| 5 | 537 | 1270 | 42.28% | 390 | 950 | 41.05% |

The template-clean Derived-Target-Absent ceiling across h=2..5 and eight relations is **22** per cell. h=1 is reported separately and is not treated as a target-absent derived condition.

## Composition-held-out is plausible but must be group-optimized in B0.2

Within the five-sentence parsed derived scope there are **3,642 ordered** and **1,016 unordered** signatures. 24 unordered signatures have at least 100 samples; 243 are singletons. This is enough diversity to attempt an unseen-composition split, but B0.2 must jointly enforce signature grouping, target balance, hop balance, and story grouping before declaring a final split feasible.

## Overlap is derived, not directly stated

`computed_hop=1` has **0** OVERLAP targets. The earliest observed OVERLAP target is hop **2**. Counts by hop are `{'2': 5548, '3': 1308, '4': 951, '5': 442, '6': 263, '7': 114, '8': 71, '9': 28, '10': 4}`. The generator has eight explicit edge relations and no OVERLAP edge template; overlap targets arise from vector composition.

## Label/path agreement exposes the practical impact of template errors

Among 200,811 records with a unique, composition-computable path, 184,930 labels agree and 15,881 disagree with the meaning of the actual story. Disagreement is segmented below; this is why non-OK templates must remain flagged rather than silently interpreted from their function names.

| contains_suspicious_template | samples | computable | label_match | label_mismatch |
| --- | --- | --- | --- | --- |
| False | 150469 | 147184 | 147184 | 0 |
| True | 59531 | 53627 | 37746 | 15881 |

## Duplicate and leakage audit

There are 806 exact-duplicate groups affecting 1,627 rows (821 excess rows). There are 1,583 repeated-story groups affecting 3,313 rows; 827 story groups contain multiple questions. Future splits must group by `story_group_id`; inverse/query variants must additionally remain in the same story group. `normalized_duplicate_group` and `graph_template_group` are useful secondary leakage controls.

## Robustness, limitations, and exclusions

- Full record parse failures: 19,763; no sentence is unmatched, but some matched templates lack a defensible two-entity canonical meaning.
- Query path unavailable: 9,189 records.
- Multiple/ambiguous source-template matches: 121,616 sentences; equivalent matches are retained and flagged.
- `graph_template_group` is a deterministic heuristic, not a complete graph-isomorphism canonicalizer.
- No raw data or official template source was modified; exclusions are flags only.

## Recommended B0 protocol adjustment

1. Rename `recorded_hop` conceptually to `source_K` or `source_story_len` for clean data.
2. Use independently computed BFS distance as the only experimental hop.
3. Make the eight-class Primary subset `story_len=5`, `computed_hop in 1..5`, parsed path available, and template-clean; do not require `source_K==computed_hop`.
4. Keep OVERLAP outside Primary; retain it as a derived diagnostic class.
5. In B0.2, group by `story_group_id`, normalized template skeleton, and composition signature before selecting train/validation/test rows.

## Further questions for B0.2

- Should all ten ERROR templates be excluded, or should semantically corrected edges be retained only for a sensitivity set?
- Should target-absent mean lexical/stated-relation absence (current definition) or absence after adding inverse entailments?
- Which minimum cell size should the composition-held-out optimizer enforce?

