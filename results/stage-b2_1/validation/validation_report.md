# Stage B2.1 independent validation

Final status: **PASS_WITH_LIMITATION**

## Checks

- PASS: `primary_rows_2063`
- PASS: `hop_counts_exact`
- PASS: `row_ids_exact_b0_vs_b1`
- PASS: `row_ids_unique`
- PASS: `relations_exact`
- PASS: `story_groups_present`
- PASS: `raw_prompts_exact`
- PASS: `protected_B0_to_B2_unchanged`
- PASS: `protocol_hashes_and_seed_valid`
- PASS: `native_boundary_audit_at_least_120_each`
- PASS: `native_boundary_matches_standalone`
- PASS: `equal_length_gate`
- PASS: `candidate_selection_score_independent`
- PASS: `independent_chat_render_boundary`
- PASS: `all_expected_outputs_present`
- PASS: `independent_argmax_and_metric_reconstruction`
- PASS: `all_model_runs_pass`
- PASS: `no_truncation`
- PASS: `eight_nonempty_figures`
- PASS: `gate_A_pass`

## Limitations

- New diagnostic protocols ran with transformers 4.57.6; frozen B2 metadata used 4.44.2. Gate A exact reproduction mitigates but does not eliminate interface-version risk.
