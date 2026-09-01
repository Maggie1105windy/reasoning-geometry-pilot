# Stage B0 outputs

The subdirectories preserve the four completed B0 validation checkpoints:

- `b0_1/`: core summary metrics.
- `b0_2/`: train/test split protocol and validation.
- `b0_2b/`: template-level split checks and common-support table.
- `b0_2c/`: held-out template test protocol and validation.

Each `validation_summary.json` belongs to its own checkpoint; keeping them in
separate directories avoids filename collisions and preserves provenance.
