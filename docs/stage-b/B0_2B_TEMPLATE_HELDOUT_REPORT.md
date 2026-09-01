# StepGame Stage B0.2b Template-Held-Out Audit

## Technical summary

No new fine/mid template control passes the requested minimum **5 validation and 5 test samples in every hop×relation cell**. Surface skeleton and mid-level syntactic-frame partitions both achieve nonzero 40-cell support with zero template/story leakage, but their observed minimum cells are only **2/2** and **1/2** respectively. The formal B0.2b status is therefore **FAIL**, and no new official template manifest is frozen.

## The existing broad control remains restricted-support

- Seven broad families; split sizes `{'train': 2756, 'val': 1644, 'test': 1921}`.
- Covered cells by split: `{'train': 38, 'val': 30, 'test': 30}`.
- Common train/val/test support: **28 / 40 cells**; usable hops `[1, 2, 3, 4]`.
- The original B0.2 template files are unchanged; exact copies are archived under `broad_template_heldout_*.csv`.

Broad results may only be evaluated on `broad_template_common_support.csv`; missing cells must never be encoded as zero performance.

## Surface skeleton is the strongest available nonzero-support candidate

- Query-path groups: **198**.
- Split group counts: `{'train': 40, 'val': 76, 'test': 82}`; rows `{'train': 1130, 'val': 2666, 'test': 2808}`.
- Covered cells: `{'train': 40, 'val': 40, 'test': 40}`; minimum cells `{'train': 1, 'val': 2, 'test': 2}`.
- Template overlap `{'train_val': 0, 'train_test': 0, 'val_test': 0}`; story overlap `{'train_val': 0, 'train_test': 0, 'val_test': 0}`.

This grouping genuinely holds out normalized official surface skeletons, including duplicate template IDs with the same skeleton. It covers every cell but misses the requested minimum count, so its manifests remain provisional.

## Mid-level syntactic frames are stronger but no more feasible

- Query-path groups: **93**.
- Definition: broad semantic family plus a direction-lexeme-normalized syntactic frame. Direction words, clock values, compass phrases, and degree adverbs are normalized; grammatical construction words remain.
- Split group counts: `{'train': 19, 'val': 39, 'test': 35}`; rows `{'train': 1487, 'val': 2185, 'test': 2794}`.
- Covered cells: `{'train': 40, 'val': 40, 'test': 40}`; minimum cells `{'train': 1, 'val': 1, 'test': 2}`.
- Template overlap `{'train_val': 0, 'train_test': 0, 'val_test': 0}`; story overlap `{'train_val': 0, 'train_test': 0, 'val_test': 0}`.

The rule is based only on language structure and was fixed before evaluating split support. Mid-level is a stronger paraphrase control than exact skeleton, but it also fails the minimum-count Gate.

## Scope, methods, and robustness

The candidate cohort is the unchanged B0.2 Primary definition: FULL_OK, five-sentence stories, computed hop 1–5, eight non-overlap relations, and no suspicious templates. Query paths are unique in this cohort. Every template group receives one global split owner; rows whose path spans owners are excluded. No model, tokenizer, activation, or probe result was accessed.

Two optimization routes were tested: deterministic multi-start group assignment and sparse fixed-quota MILP feasibility. The latter timed out without a feasible >=5 seed and without proving mathematical infeasibility. Therefore the report states an empirical maximum found, not a proof of the global ceiling.

## Gate decision and next step

- Structural checks pass for the provisional surface and mid candidates: zero story/template leakage, legal relations/hops, and no suspicious templates.
- The quantitative minimum-count Gate fails.
- Keep Primary, Balanced, Composition, and Target-Absent frozen exactly as before.
- Do not enter Stage B1 with a claimed full-support template robustness control until the protocol either accepts a lower minimum, restricts the evaluated cells, or authorizes a different linguistic grouping definition.
