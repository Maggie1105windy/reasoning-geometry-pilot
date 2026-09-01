# Tülu-3-8B MATH-500 Pilot-20 Result Summary

## Setting

- Model: Tülu-3-8B
- Dataset: MATH-500 pilot subset
- Number of problems: 20
- Samples per problem: 1
- Decoding: greedy decoding, temperature = 0
- Max new tokens: 1024
- Hardware: RTX 3090 24GB
- Evaluation: boxed-answer extraction + manual review

## Result

| Metric | Value |
|---|---:|
| Number of problems | 20 |
| Number of generations | 20 |
| Auto-correct | 5 / 20 |
| Auto accuracy | 25% |
| Manual-corrected correct | 7 / 20 |
| Manual-corrected accuracy | 35% |
| Final pass@1 | 35% |
| Manual reviewed count | 20 |
| Unresolved manual-needed count | 0 |
| Average generated tokens | 819.65 |
| Median generated tokens | 820.5 |

## Interpretation

This pilot verifies that the project pipeline is executable on the RTX 3090 server:
data preparation, prompt construction, Tülu-3-8B loading, MATH-500 generation,
answer extraction, manual review, and behavior-level summary all ran successfully.

The result is a small-scale feasibility pilot and should not be interpreted as a final benchmark result.
