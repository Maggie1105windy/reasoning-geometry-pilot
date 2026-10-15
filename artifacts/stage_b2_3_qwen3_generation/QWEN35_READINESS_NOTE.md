# Qwen3.5-9B Readiness Note

Stage B2.3 selected Case C. No Qwen3.5 model was downloaded or run.

- Candidate model: `Qwen/Qwen3.5-9B`
- Candidate base: `Qwen/Qwen3.5-9B-Base`
- Re-audit tokenizer and chat template.
- Re-confirm reasoning controls and generation defaults.
- Re-audit model architecture; do not assume the Qwen3-8B 36-layer standard Transformer layout.
- Run a 24 GB RTX 3090 memory smoke test before any full experiment.
- Apply a behavior gate before activation extraction.
