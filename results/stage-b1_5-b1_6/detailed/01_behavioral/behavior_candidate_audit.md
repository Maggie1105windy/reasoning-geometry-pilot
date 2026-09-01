# Stage B1.5 candidate scoring audit

Status: **PASS**

## Frozen scoring

- Primary: sum of conditional log-probabilities over all candidate tokens.
- Sensitivity: mean conditional log-probability per candidate token.
- Candidate EOS and padding tokens are excluded.
- Coverage: one Primary-test sample for each of 5 hops x 8 relations.

## Candidate tokenization

| Relation | Exact continuation | Token IDs | Tokens | n |
|---|---|---|---|---:|
| ABOVE | ` above` | `[3485]` | `['Ġabove']` | 1 |
| BELOW | ` below` | `[3770]` | `['Ġbelow']` | 1 |
| LEFT | ` left` | `[2163]` | `['Ġleft']` | 1 |
| RIGHT | ` right` | `[1314]` | `['Ġright']` | 1 |
| UPPER_LEFT | ` upper-left` | `[8582, 8109]` | `['Ġupper', '-left']` | 2 |
| UPPER_RIGHT | ` upper-right` | `[8582, 6840]` | `['Ġupper', '-right']` | 2 |
| LOWER_LEFT | ` lower-left` | `[4827, 8109]` | `['Ġlower', '-left']` | 2 |
| LOWER_RIGHT | ` lower-right` | `[4827, 6840]` | `['Ġlower', '-right']` | 2 |

## 40-cell scoring results

| Hop | Gold | Row ID | Sum prediction | Mean prediction | Prompt SHA-256 |
|---:|---|---:|---|---|---|
| 1 | ABOVE | 104161 | LEFT | LOWER_LEFT | `87dd8cfaa1692fec...` |
| 1 | BELOW | 98552 | ABOVE | UPPER_RIGHT | `138e2db409c8f7de...` |
| 1 | LEFT | 90997 | LEFT | UPPER_LEFT | `1c2192a4d9660d4c...` |
| 1 | LOWER_LEFT | 99445 | LEFT | UPPER_RIGHT | `8c0fc774d44e9536...` |
| 1 | LOWER_RIGHT | 94109 | ABOVE | UPPER_LEFT | `9cf5ee47d171a82f...` |
| 1 | RIGHT | 93784 | LEFT | UPPER_LEFT | `40be7eb08494c7b6...` |
| 1 | UPPER_LEFT | 95433 | LEFT | UPPER_LEFT | `b1448cd39016ef02...` |
| 1 | UPPER_RIGHT | 93511 | LEFT | LOWER_RIGHT | `46722ad5b925157d...` |
| 2 | ABOVE | 91318 | LEFT | LOWER_LEFT | `ecbc00c14656ee82...` |
| 2 | BELOW | 85029 | LEFT | UPPER_RIGHT | `6c07f382b458e599...` |
| 2 | LEFT | 93948 | LEFT | UPPER_LEFT | `536f451643bd488a...` |
| 2 | LOWER_LEFT | 100247 | LEFT | UPPER_LEFT | `cb05c41d3bdd53b8...` |
| 2 | LOWER_RIGHT | 102033 | LEFT | UPPER_LEFT | `8a5fdcbcb815b445...` |
| 2 | RIGHT | 85308 | LEFT | UPPER_LEFT | `6655d90d50545fe0...` |
| 2 | UPPER_LEFT | 92590 | LEFT | LOWER_LEFT | `63a2974988d41da9...` |
| 2 | UPPER_RIGHT | 99203 | ABOVE | UPPER_RIGHT | `44fd8bc261435c65...` |
| 3 | ABOVE | 104720 | LEFT | UPPER_LEFT | `400122b7f4012f4f...` |
| 3 | BELOW | 102944 | ABOVE | UPPER_LEFT | `216086c0c8fe93ae...` |
| 3 | LEFT | 98018 | LEFT | UPPER_LEFT | `cbe11117714f5108...` |
| 3 | LOWER_LEFT | 87246 | LEFT | UPPER_LEFT | `2193bad86ebb580b...` |
| 3 | LOWER_RIGHT | 85176 | LEFT | UPPER_LEFT | `1f448c771329402b...` |
| 3 | RIGHT | 87828 | LEFT | UPPER_LEFT | `0c08fd722aa67eb7...` |
| 3 | UPPER_LEFT | 92567 | LEFT | UPPER_RIGHT | `2d99745745cdc943...` |
| 3 | UPPER_RIGHT | 104100 | ABOVE | UPPER_LEFT | `5989470ca9b7c512...` |
| 4 | ABOVE | 94110 | LEFT | UPPER_LEFT | `145758bea065c6c8...` |
| 4 | BELOW | 97425 | LEFT | LOWER_LEFT | `511811d63402b31b...` |
| 4 | LEFT | 98237 | LEFT | UPPER_LEFT | `09d1036cd3306d7b...` |
| 4 | LOWER_LEFT | 100592 | LEFT | UPPER_LEFT | `dbc6146187738c77...` |
| 4 | LOWER_RIGHT | 89541 | ABOVE | UPPER_LEFT | `d7c18c01bd2e8e72...` |
| 4 | RIGHT | 95279 | LEFT | UPPER_LEFT | `e27b63ea252cd881...` |
| 4 | UPPER_LEFT | 102645 | LEFT | UPPER_LEFT | `5c1fa90d50db4539...` |
| 4 | UPPER_RIGHT | 95526 | LEFT | UPPER_LEFT | `e28c363d8aa02f2f...` |
| 5 | ABOVE | 85026 | LEFT | UPPER_LEFT | `7bb582c0c9177133...` |
| 5 | BELOW | 95725 | LEFT | UPPER_LEFT | `07b91bc3680c6f67...` |
| 5 | LEFT | 91455 | ABOVE | UPPER_LEFT | `d892e768e9c7e5d0...` |
| 5 | LOWER_LEFT | 94401 | LEFT | UPPER_LEFT | `7fa46f6a2ce4f63e...` |
| 5 | LOWER_RIGHT | 91422 | LEFT | UPPER_LEFT | `3c58ded86a869fe7...` |
| 5 | RIGHT | 98671 | LEFT | UPPER_LEFT | `f7823c7507079169...` |
| 5 | UPPER_LEFT | 95682 | ABOVE | UPPER_LEFT | `cfb517cebdf5df2b...` |
| 5 | UPPER_RIGHT | 101545 | LEFT | UPPER_RIGHT | `32fa43ce611b61e8...` |

## Prompt examples

### row_id=104161; hop=1; gold=ABOVE

```text
Story:
K is on top of S.
Q is at the 3 o'clock position relative to U.
Q is at the lower side of J.
C is positioned right to J.
U is at the 12 o'clock position relative to K.

Question:
What is the relation of the agent J to the agent Q?
```

### row_id=98552; hop=1; gold=BELOW

```text
Story:
R is diagonally to the bottom right of M.
The object M is positioned directly above the object P.
J and N are next to each other with J on the left and N on the right.
The object B and R are there. The object B is below and slightly to the right of the object R.
The objects P and J are over there. The object P is lower and slightly to the left of the object J.

Question:
What is the relation of the agent P to the agent M?
```

### row_id=90997; hop=1; gold=LEFT

```text
Story:
F is on the same vertical plane directly above U.
The object labeled E is positioned to the left of the object labeled F.
E is placed at the upper left of O.
O is placed in the left direction of A.
A is positioned in the lower right corner of X.

Question:
What is the relation of the agent E to the agent F?
```

