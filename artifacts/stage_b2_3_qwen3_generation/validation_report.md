# Stage B2.3 Independent Validation

Status: **PASS_WITH_LIMITATION**

Checks passed: 60/60.

Validation independently read the two full `row_generations.jsonl` files, reconstructed parser outputs, metrics, confusion matrices, relation recalls, prediction frequencies, derived h2–h5 means, truncation, prompt hashes, pilot coverage, generation configuration, protected artifacts, and figure readability. It did not trust `metrics.csv` as a source.

## Limitations

- FG-T think-boundary parse success is 0.1929.
- At least one full-run generation reached max_new_tokens.

No validation errors were found.
