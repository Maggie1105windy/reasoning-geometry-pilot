# Synchronization audit

Audit date: 2026-09-10

The cloud SSH endpoint was reachable again during the repository audit. The completed formal artifact directories were copied into a local recovery staging area and checked by file count and total byte size:

| Artifact directory | Files | Bytes | Recovery status |
|---|---:|---:|---|
| Stage B0.2c | 8 | 4,305,206 | complete |
| Stage B1 | 321 | 224,061,894 | complete |
| Stage B2 | 1,500 | 442,733,674 | complete |
| MATH-500 remote reports | 18 | 407,254 | complete |

The Git repository is a curated research layer rather than a byte-for-byte copy of the recovery staging area. It includes formal reports, figures, protocols, audits, validation JSON, aggregated CSV results, selected prediction tables, scripts, and artifact checksums. It excludes model weights, activations, `.npz` probe parameters, redundant checkpoint-level predictions, caches, environments, and logs.

Stage B2 supersedes the earlier local staging note that described it as planned. The recovered `B2_final_validation.json` and `b2_summary.json` both report `PASS`.

Stage B2.1 was completed on the reconnected server on 2026-09-10. The full remote artifact is approximately 59 MB. The Git-friendly sync contains 170 research-layer files (about 1.66 MB before figures), eight figures (about 1.5 MB), five reproducibility scripts, and the formal report. Redundant row-level candidate-score exports and logs remain on the server and are enumerated in `results/stage-b2_1/PUBLIC_SYNC_MANIFEST.md`. Independent validation reports `PASS_WITH_LIMITATION`; all substantive checks pass, with a documented transformers version difference between the historical B2 environment metadata and the currently available diagnostic environment.
