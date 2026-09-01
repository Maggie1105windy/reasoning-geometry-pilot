# Stage B1 Tokenization Examples

Status: **PASS**

Audited 120 rows: three deterministic samples per hop×relation cell.

## row 85277 — h=1, ABOVE

```text
Story:
T is positioned above and to the left of X.
T is below and to the left of Z.
P is sitting at the lower position to J.
X is positioned in the lower left corner of P.
V and J are parallel, and J is below V.

Question:
What is the relation of the agent V to the agent J?
```

- sequence length: 72
- readout index: 71
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 51, 374, 35328, 3485, 323, 311, 279, 2163, 315, 1630, 627, 51, 374, 3770, 323, 311, 279, 2163, 315, 1901, 627, 47, 374, 11961, 520, 279, 4827, 2361, 311, 622, 627, 55, 374, 35328, 304, 279, 4827, 2163, 9309, 315, 393, 627, 53, 323, 622, 527, 15638, 11, 323, 622, 374, 3770, 650, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 650, 311, 279, 8479, 622, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'T', 'Ġis', 'Ġpositioned', 'Ġabove', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠX', '.Ċ', 'T', 'Ġis', 'Ġbelow', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠZ', '.Ċ', 'P', 'Ġis', 'Ġsitting', 'Ġat', 'Ġthe', 'Ġlower', 'Ġposition', 'Ġto', 'ĠJ', '.Ċ', 'X', 'Ġis', 'Ġpositioned', 'Ġin', 'Ġthe', 'Ġlower', 'Ġleft', 'Ġcorner', 'Ġof', 'ĠP', '.Ċ', 'V', 'Ġand', 'ĠJ', 'Ġare', 'Ġparallel', ',', 'Ġand', 'ĠJ', 'Ġis', 'Ġbelow', 'ĠV', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠV', 'Ġto', 'Ġthe', 'Ġagent', 'ĠJ', '?']`

## row 88422 — h=1, ABOVE

```text
Story:
G and S are side by side with S at the bottom and G on the top.
Y is over there and K is on the left.
W is over there and K is on the top of it.
Y and R are parallel, and Y is under R.
R and G are side by side with G at the bottom and R on the top.

Question:
What is the relation of the agent G to the agent S?
```

- sequence length: 90
- readout index: 89
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 38, 323, 328, 527, 3185, 555, 3185, 449, 328, 520, 279, 5740, 323, 480, 389, 279, 1948, 627, 56, 374, 927, 1070, 323, 735, 374, 389, 279, 2163, 627, 54, 374, 927, 1070, 323, 735, 374, 389, 279, 1948, 315, 433, 627, 56, 323, 432, 527, 15638, 11, 323, 816, 374, 1234, 432, 627, 49, 323, 480, 527, 3185, 555, 3185, 449, 480, 520, 279, 5740, 323, 432, 389, 279, 1948, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 480, 311, 279, 8479, 328, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'G', 'Ġand', 'ĠS', 'Ġare', 'Ġside', 'Ġby', 'Ġside', 'Ġwith', 'ĠS', 'Ġat', 'Ġthe', 'Ġbottom', 'Ġand', 'ĠG', 'Ġon', 'Ġthe', 'Ġtop', '.Ċ', 'Y', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠK', 'Ġis', 'Ġon', 'Ġthe', 'Ġleft', '.Ċ', 'W', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠK', 'Ġis', 'Ġon', 'Ġthe', 'Ġtop', 'Ġof', 'Ġit', '.Ċ', 'Y', 'Ġand', 'ĠR', 'Ġare', 'Ġparallel', ',', 'Ġand', 'ĠY', 'Ġis', 'Ġunder', 'ĠR', '.Ċ', 'R', 'Ġand', 'ĠG', 'Ġare', 'Ġside', 'Ġby', 'Ġside', 'Ġwith', 'ĠG', 'Ġat', 'Ġthe', 'Ġbottom', 'Ġand', 'ĠR', 'Ġon', 'Ġthe', 'Ġtop', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠG', 'Ġto', 'Ġthe', 'Ġagent', 'ĠS', '?']`

## row 96866 — h=1, ABOVE

```text
Story:
C is below S with a small gap between them.
R is over there and V is on the right of it.
C is placed at the lower left of V.
U is below D at 4 o'clock.
D is directly below R.

Question:
What is the relation of the agent R to the agent D?
```

- sequence length: 68
- readout index: 67
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 34, 374, 3770, 328, 449, 264, 2678, 13225, 1990, 1124, 627, 49, 374, 927, 1070, 323, 650, 374, 389, 279, 1314, 315, 433, 627, 34, 374, 9277, 520, 279, 4827, 2163, 315, 650, 627, 52, 374, 3770, 423, 520, 220, 19, 297, 63510, 627, 35, 374, 6089, 3770, 432, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 432, 311, 279, 8479, 423, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'C', 'Ġis', 'Ġbelow', 'ĠS', 'Ġwith', 'Ġa', 'Ġsmall', 'Ġgap', 'Ġbetween', 'Ġthem', '.Ċ', 'R', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠV', 'Ġis', 'Ġon', 'Ġthe', 'Ġright', 'Ġof', 'Ġit', '.Ċ', 'C', 'Ġis', 'Ġplaced', 'Ġat', 'Ġthe', 'Ġlower', 'Ġleft', 'Ġof', 'ĠV', '.Ċ', 'U', 'Ġis', 'Ġbelow', 'ĠD', 'Ġat', 'Ġ', '4', 'Ġo', "'clock", '.Ċ', 'D', 'Ġis', 'Ġdirectly', 'Ġbelow', 'ĠR', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠR', 'Ġto', 'Ġthe', 'Ġagent', 'ĠD', '?']`

## row 96074 — h=1, BELOW

```text
Story:
L is sitting at the upper right position to K.
L is at Z's 12 o'clock.
B is at the lower side of K.
H and Y are parallel, and H is under Y.
B is below Y.

Question:
What is the relation of the agent B to the agent Y?
```

- sequence length: 65
- readout index: 64
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 43, 374, 11961, 520, 279, 8582, 1314, 2361, 311, 735, 627, 43, 374, 520, 1901, 596, 220, 717, 297, 63510, 627, 33, 374, 520, 279, 4827, 3185, 315, 735, 627, 39, 323, 816, 527, 15638, 11, 323, 473, 374, 1234, 816, 627, 33, 374, 3770, 816, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 426, 311, 279, 8479, 816, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'L', 'Ġis', 'Ġsitting', 'Ġat', 'Ġthe', 'Ġupper', 'Ġright', 'Ġposition', 'Ġto', 'ĠK', '.Ċ', 'L', 'Ġis', 'Ġat', 'ĠZ', "'s", 'Ġ', '12', 'Ġo', "'clock", '.Ċ', 'B', 'Ġis', 'Ġat', 'Ġthe', 'Ġlower', 'Ġside', 'Ġof', 'ĠK', '.Ċ', 'H', 'Ġand', 'ĠY', 'Ġare', 'Ġparallel', ',', 'Ġand', 'ĠH', 'Ġis', 'Ġunder', 'ĠY', '.Ċ', 'B', 'Ġis', 'Ġbelow', 'ĠY', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠB', 'Ġto', 'Ġthe', 'Ġagent', 'ĠY', '?']`

## row 92072 — h=2, ABOVE

```text
Story:
M is at the lower side of K.
A is at the bottom of V vertically.
A is on the right and J is on the left.
K is directly below S.
M and V are next to each other with V at the bottom M on the top.

Question:
What is the relation of the agent K to the agent V?
```

- sequence length: 72
- readout index: 71
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 44, 374, 520, 279, 4827, 3185, 315, 735, 627, 32, 374, 520, 279, 5740, 315, 650, 52035, 627, 32, 374, 389, 279, 1314, 323, 622, 374, 389, 279, 2163, 627, 42, 374, 6089, 3770, 328, 627, 44, 323, 650, 527, 1828, 311, 1855, 1023, 449, 650, 520, 279, 5740, 386, 389, 279, 1948, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 735, 311, 279, 8479, 650, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'M', 'Ġis', 'Ġat', 'Ġthe', 'Ġlower', 'Ġside', 'Ġof', 'ĠK', '.Ċ', 'A', 'Ġis', 'Ġat', 'Ġthe', 'Ġbottom', 'Ġof', 'ĠV', 'Ġvertically', '.Ċ', 'A', 'Ġis', 'Ġon', 'Ġthe', 'Ġright', 'Ġand', 'ĠJ', 'Ġis', 'Ġon', 'Ġthe', 'Ġleft', '.Ċ', 'K', 'Ġis', 'Ġdirectly', 'Ġbelow', 'ĠS', '.Ċ', 'M', 'Ġand', 'ĠV', 'Ġare', 'Ġnext', 'Ġto', 'Ġeach', 'Ġother', 'Ġwith', 'ĠV', 'Ġat', 'Ġthe', 'Ġbottom', 'ĠM', 'Ġon', 'Ġthe', 'Ġtop', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠK', 'Ġto', 'Ġthe', 'Ġagent', 'ĠV', '?']`

## row 92624 — h=2, ABOVE

```text
Story:
P is placed at the upper right of S.
D presents upper right to S.
G is diagonally above D to the left at a 45 degree angle.
A is to the bottom-right of V.
A is at G’s 9 o'clock.

Question:
What is the relation of the agent G to the agent S?
```

- sequence length: 70
- readout index: 69
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 47, 374, 9277, 520, 279, 8582, 1314, 315, 328, 627, 35, 18911, 8582, 1314, 311, 328, 627, 38, 374, 85118, 750, 3485, 423, 311, 279, 2163, 520, 264, 220, 1774, 8547, 9392, 627, 32, 374, 311, 279, 5740, 6840, 315, 650, 627, 32, 374, 520, 480, 753, 220, 24, 297, 63510, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 480, 311, 279, 8479, 328, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'P', 'Ġis', 'Ġplaced', 'Ġat', 'Ġthe', 'Ġupper', 'Ġright', 'Ġof', 'ĠS', '.Ċ', 'D', 'Ġpresents', 'Ġupper', 'Ġright', 'Ġto', 'ĠS', '.Ċ', 'G', 'Ġis', 'Ġdiagon', 'ally', 'Ġabove', 'ĠD', 'Ġto', 'Ġthe', 'Ġleft', 'Ġat', 'Ġa', 'Ġ', '45', 'Ġdegree', 'Ġangle', '.Ċ', 'A', 'Ġis', 'Ġto', 'Ġthe', 'Ġbottom', '-right', 'Ġof', 'ĠV', '.Ċ', 'A', 'Ġis', 'Ġat', 'ĠG', 'âĢĻs', 'Ġ', '9', 'Ġo', "'clock", '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠG', 'Ġto', 'Ġthe', 'Ġagent', 'ĠS', '?']`

## row 99707 — h=2, ABOVE

```text
Story:
O is to the bottom-right of N.
O is over there and K is at the bottom of it.
P presents right to N.
X is under V.
X is over there and K is on the top of it.

Question:
What is the relation of the agent O to the agent X?
```

- sequence length: 64
- readout index: 63
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 46, 374, 311, 279, 5740, 6840, 315, 452, 627, 46, 374, 927, 1070, 323, 735, 374, 520, 279, 5740, 315, 433, 627, 47, 18911, 1314, 311, 452, 627, 55, 374, 1234, 650, 627, 55, 374, 927, 1070, 323, 735, 374, 389, 279, 1948, 315, 433, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 507, 311, 279, 8479, 1630, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'O', 'Ġis', 'Ġto', 'Ġthe', 'Ġbottom', '-right', 'Ġof', 'ĠN', '.Ċ', 'O', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠK', 'Ġis', 'Ġat', 'Ġthe', 'Ġbottom', 'Ġof', 'Ġit', '.Ċ', 'P', 'Ġpresents', 'Ġright', 'Ġto', 'ĠN', '.Ċ', 'X', 'Ġis', 'Ġunder', 'ĠV', '.Ċ', 'X', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠK', 'Ġis', 'Ġon', 'Ġthe', 'Ġtop', 'Ġof', 'Ġit', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠO', 'Ġto', 'Ġthe', 'Ġagent', 'ĠX', '?']`

## row 84538 — h=2, BELOW

```text
Story:
F is placed at the lower right of W.
N is below S at 7 o'clock.
S is on the right side and below Q.
The object F is positioned directly below the object Z.
Q is sitting at the top position to Z.

Question:
What is the relation of the agent N to the agent Q?
```

- sequence length: 69
- readout index: 68
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 37, 374, 9277, 520, 279, 4827, 1314, 315, 468, 627, 45, 374, 3770, 328, 520, 220, 22, 297, 63510, 627, 50, 374, 389, 279, 1314, 3185, 323, 3770, 1229, 627, 791, 1665, 435, 374, 35328, 6089, 3770, 279, 1665, 1901, 627, 48, 374, 11961, 520, 279, 1948, 2361, 311, 1901, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 452, 311, 279, 8479, 1229, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'F', 'Ġis', 'Ġplaced', 'Ġat', 'Ġthe', 'Ġlower', 'Ġright', 'Ġof', 'ĠW', '.Ċ', 'N', 'Ġis', 'Ġbelow', 'ĠS', 'Ġat', 'Ġ', '7', 'Ġo', "'clock", '.Ċ', 'S', 'Ġis', 'Ġon', 'Ġthe', 'Ġright', 'Ġside', 'Ġand', 'Ġbelow', 'ĠQ', '.Ċ', 'The', 'Ġobject', 'ĠF', 'Ġis', 'Ġpositioned', 'Ġdirectly', 'Ġbelow', 'Ġthe', 'Ġobject', 'ĠZ', '.Ċ', 'Q', 'Ġis', 'Ġsitting', 'Ġat', 'Ġthe', 'Ġtop', 'Ġposition', 'Ġto', 'ĠZ', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠN', 'Ġto', 'Ġthe', 'Ġagent', 'ĠQ', '?']`

## row 87783 — h=3, ABOVE

```text
Story:
F is at the 6 o'clock position relative to P.
C and T are side by side with C to the right and T to the left.
Object I is above object T and to the left of it, too.
I is positioned in the top left corner of F.
P is directly north east of K.

Question:
What is the relation of the agent I to the agent K?
```

- sequence length: 83
- readout index: 82
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 37, 374, 520, 279, 220, 21, 297, 63510, 2361, 8844, 311, 393, 627, 34, 323, 350, 527, 3185, 555, 3185, 449, 356, 311, 279, 1314, 323, 350, 311, 279, 2163, 627, 1211, 358, 374, 3485, 1665, 350, 323, 311, 279, 2163, 315, 433, 11, 2288, 627, 40, 374, 35328, 304, 279, 1948, 2163, 9309, 315, 435, 627, 47, 374, 6089, 10411, 11226, 315, 735, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 358, 311, 279, 8479, 735, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'F', 'Ġis', 'Ġat', 'Ġthe', 'Ġ', '6', 'Ġo', "'clock", 'Ġposition', 'Ġrelative', 'Ġto', 'ĠP', '.Ċ', 'C', 'Ġand', 'ĠT', 'Ġare', 'Ġside', 'Ġby', 'Ġside', 'Ġwith', 'ĠC', 'Ġto', 'Ġthe', 'Ġright', 'Ġand', 'ĠT', 'Ġto', 'Ġthe', 'Ġleft', '.Ċ', 'Object', 'ĠI', 'Ġis', 'Ġabove', 'Ġobject', 'ĠT', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'Ġit', ',', 'Ġtoo', '.Ċ', 'I', 'Ġis', 'Ġpositioned', 'Ġin', 'Ġthe', 'Ġtop', 'Ġleft', 'Ġcorner', 'Ġof', 'ĠF', '.Ċ', 'P', 'Ġis', 'Ġdirectly', 'Ġnorth', 'Ġeast', 'Ġof', 'ĠK', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠI', 'Ġto', 'Ġthe', 'Ġagent', 'ĠK', '?']`

## row 91562 — h=3, ABOVE

```text
Story:
H is over there and I is directly below it.
X is placed at the upper left of Z.
A is to the bottom-left of H.
L and Z are in a horizontal line with L on the left.
I is to the top-right of L.

Question:
What is the relation of the agent H to the agent Z?
```

- sequence length: 71
- readout index: 70
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 39, 374, 927, 1070, 323, 358, 374, 6089, 3770, 433, 627, 55, 374, 9277, 520, 279, 8582, 2163, 315, 1901, 627, 32, 374, 311, 279, 5740, 8109, 315, 473, 627, 43, 323, 1901, 527, 304, 264, 16600, 1584, 449, 445, 389, 279, 2163, 627, 40, 374, 311, 279, 1948, 6840, 315, 445, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 473, 311, 279, 8479, 1901, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'H', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠI', 'Ġis', 'Ġdirectly', 'Ġbelow', 'Ġit', '.Ċ', 'X', 'Ġis', 'Ġplaced', 'Ġat', 'Ġthe', 'Ġupper', 'Ġleft', 'Ġof', 'ĠZ', '.Ċ', 'A', 'Ġis', 'Ġto', 'Ġthe', 'Ġbottom', '-left', 'Ġof', 'ĠH', '.Ċ', 'L', 'Ġand', 'ĠZ', 'Ġare', 'Ġin', 'Ġa', 'Ġhorizontal', 'Ġline', 'Ġwith', 'ĠL', 'Ġon', 'Ġthe', 'Ġleft', '.Ċ', 'I', 'Ġis', 'Ġto', 'Ġthe', 'Ġtop', '-right', 'Ġof', 'ĠL', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠH', 'Ġto', 'Ġthe', 'Ġagent', 'ĠZ', '?']`

## row 91963 — h=3, ABOVE

```text
Story:
F and Y are parallel, and F is under Y.
S is to the bottom left of Q.
Object N is above object Y and to the left of it, too.
N is above and to the right of Q.
F is to the right of M.

Question:
What is the relation of the agent N to the agent M?
```

- sequence length: 72
- readout index: 71
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 37, 323, 816, 527, 15638, 11, 323, 435, 374, 1234, 816, 627, 50, 374, 311, 279, 5740, 2163, 315, 1229, 627, 1211, 452, 374, 3485, 1665, 816, 323, 311, 279, 2163, 315, 433, 11, 2288, 627, 45, 374, 3485, 323, 311, 279, 1314, 315, 1229, 627, 37, 374, 311, 279, 1314, 315, 386, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 452, 311, 279, 8479, 386, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'F', 'Ġand', 'ĠY', 'Ġare', 'Ġparallel', ',', 'Ġand', 'ĠF', 'Ġis', 'Ġunder', 'ĠY', '.Ċ', 'S', 'Ġis', 'Ġto', 'Ġthe', 'Ġbottom', 'Ġleft', 'Ġof', 'ĠQ', '.Ċ', 'Object', 'ĠN', 'Ġis', 'Ġabove', 'Ġobject', 'ĠY', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'Ġit', ',', 'Ġtoo', '.Ċ', 'N', 'Ġis', 'Ġabove', 'Ġand', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'ĠQ', '.Ċ', 'F', 'Ġis', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'ĠM', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠN', 'Ġto', 'Ġthe', 'Ġagent', 'ĠM', '?']`

## row 87550 — h=3, BELOW

```text
Story:
B is diagonally to the upper left of Q.
Q is on the same horizontal plane directly right to W.
E is sitting at the 6:00 position to B.
I and M are next to each other with M on the right and I on the left.
W is over there and I is directly above it.

Question:
What is the relation of the agent Q to the agent M?
```

- sequence length: 84
- readout index: 83
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 33, 374, 85118, 750, 311, 279, 8582, 2163, 315, 1229, 627, 48, 374, 389, 279, 1890, 16600, 11277, 6089, 1314, 311, 468, 627, 36, 374, 11961, 520, 279, 220, 21, 25, 410, 2361, 311, 426, 627, 40, 323, 386, 527, 1828, 311, 1855, 1023, 449, 386, 389, 279, 1314, 323, 358, 389, 279, 2163, 627, 54, 374, 927, 1070, 323, 358, 374, 6089, 3485, 433, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 1229, 311, 279, 8479, 386, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'B', 'Ġis', 'Ġdiagon', 'ally', 'Ġto', 'Ġthe', 'Ġupper', 'Ġleft', 'Ġof', 'ĠQ', '.Ċ', 'Q', 'Ġis', 'Ġon', 'Ġthe', 'Ġsame', 'Ġhorizontal', 'Ġplane', 'Ġdirectly', 'Ġright', 'Ġto', 'ĠW', '.Ċ', 'E', 'Ġis', 'Ġsitting', 'Ġat', 'Ġthe', 'Ġ', '6', ':', '00', 'Ġposition', 'Ġto', 'ĠB', '.Ċ', 'I', 'Ġand', 'ĠM', 'Ġare', 'Ġnext', 'Ġto', 'Ġeach', 'Ġother', 'Ġwith', 'ĠM', 'Ġon', 'Ġthe', 'Ġright', 'Ġand', 'ĠI', 'Ġon', 'Ġthe', 'Ġleft', '.Ċ', 'W', 'Ġis', 'Ġover', 'Ġthere', 'Ġand', 'ĠI', 'Ġis', 'Ġdirectly', 'Ġabove', 'Ġit', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠQ', 'Ġto', 'Ġthe', 'Ġagent', 'ĠM', '?']`

## row 85823 — h=4, ABOVE

```text
Story:
X is to the left of A.
A is to the bottom right of L.
U is to the upper left of K.
V is to the upper left of L.
K is above and to the right of V.

Question:
What is the relation of the agent K to the agent X?
```

- sequence length: 63
- readout index: 62
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 55, 374, 311, 279, 2163, 315, 362, 627, 32, 374, 311, 279, 5740, 1314, 315, 445, 627, 52, 374, 311, 279, 8582, 2163, 315, 735, 627, 53, 374, 311, 279, 8582, 2163, 315, 445, 627, 42, 374, 3485, 323, 311, 279, 1314, 315, 650, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 735, 311, 279, 8479, 1630, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'X', 'Ġis', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠA', '.Ċ', 'A', 'Ġis', 'Ġto', 'Ġthe', 'Ġbottom', 'Ġright', 'Ġof', 'ĠL', '.Ċ', 'U', 'Ġis', 'Ġto', 'Ġthe', 'Ġupper', 'Ġleft', 'Ġof', 'ĠK', '.Ċ', 'V', 'Ġis', 'Ġto', 'Ġthe', 'Ġupper', 'Ġleft', 'Ġof', 'ĠL', '.Ċ', 'K', 'Ġis', 'Ġabove', 'Ġand', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'ĠV', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠK', 'Ġto', 'Ġthe', 'Ġagent', 'ĠX', '?']`

## row 88524 — h=4, ABOVE

```text
Story:
A is at Z's 6 o'clock.
J is below K.
The object N is positioned below and to the right of the object J.
K is diagonally to the bottom left of W.
A is to the top-left of W.

Question:
What is the relation of the agent Z to the agent J?
```

- sequence length: 68
- readout index: 67
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 32, 374, 520, 1901, 596, 220, 21, 297, 63510, 627, 41, 374, 3770, 735, 627, 791, 1665, 452, 374, 35328, 3770, 323, 311, 279, 1314, 315, 279, 1665, 622, 627, 42, 374, 85118, 750, 311, 279, 5740, 2163, 315, 468, 627, 32, 374, 311, 279, 1948, 8109, 315, 468, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 1901, 311, 279, 8479, 622, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'A', 'Ġis', 'Ġat', 'ĠZ', "'s", 'Ġ', '6', 'Ġo', "'clock", '.Ċ', 'J', 'Ġis', 'Ġbelow', 'ĠK', '.Ċ', 'The', 'Ġobject', 'ĠN', 'Ġis', 'Ġpositioned', 'Ġbelow', 'Ġand', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'Ġthe', 'Ġobject', 'ĠJ', '.Ċ', 'K', 'Ġis', 'Ġdiagon', 'ally', 'Ġto', 'Ġthe', 'Ġbottom', 'Ġleft', 'Ġof', 'ĠW', '.Ċ', 'A', 'Ġis', 'Ġto', 'Ġthe', 'Ġtop', '-left', 'Ġof', 'ĠW', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠZ', 'Ġto', 'Ġthe', 'Ġagent', 'ĠJ', '?']`

## row 98305 — h=4, ABOVE

```text
Story:
B and D are side by side with D at the bottom and B on the top.
D is diagonally right and above P.
P and V are parallel, and P is over V.
F is on the top side to H.
V is on the left side and above H.

Question:
What is the relation of the agent D to the agent F?
```

- sequence length: 76
- readout index: 75
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 33, 323, 423, 527, 3185, 555, 3185, 449, 423, 520, 279, 5740, 323, 426, 389, 279, 1948, 627, 35, 374, 85118, 750, 1314, 323, 3485, 393, 627, 47, 323, 650, 527, 15638, 11, 323, 393, 374, 927, 650, 627, 37, 374, 389, 279, 1948, 3185, 311, 473, 627, 53, 374, 389, 279, 2163, 3185, 323, 3485, 473, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 423, 311, 279, 8479, 435, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'B', 'Ġand', 'ĠD', 'Ġare', 'Ġside', 'Ġby', 'Ġside', 'Ġwith', 'ĠD', 'Ġat', 'Ġthe', 'Ġbottom', 'Ġand', 'ĠB', 'Ġon', 'Ġthe', 'Ġtop', '.Ċ', 'D', 'Ġis', 'Ġdiagon', 'ally', 'Ġright', 'Ġand', 'Ġabove', 'ĠP', '.Ċ', 'P', 'Ġand', 'ĠV', 'Ġare', 'Ġparallel', ',', 'Ġand', 'ĠP', 'Ġis', 'Ġover', 'ĠV', '.Ċ', 'F', 'Ġis', 'Ġon', 'Ġthe', 'Ġtop', 'Ġside', 'Ġto', 'ĠH', '.Ċ', 'V', 'Ġis', 'Ġon', 'Ġthe', 'Ġleft', 'Ġside', 'Ġand', 'Ġabove', 'ĠH', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠD', 'Ġto', 'Ġthe', 'Ġagent', 'ĠF', '?']`

## row 85040 — h=4, BELOW

```text
Story:
D is over there with R above.
R is below S.
T is to the right of S horizontally.
D is diagonally to the bottom left of X.
Y is placed in the left direction of T.

Question:
What is the relation of the agent D to the agent Y?
```

- sequence length: 61
- readout index: 60
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 35, 374, 927, 1070, 449, 432, 3485, 627, 49, 374, 3770, 328, 627, 51, 374, 311, 279, 1314, 315, 328, 59988, 627, 35, 374, 85118, 750, 311, 279, 5740, 2163, 315, 1630, 627, 56, 374, 9277, 304, 279, 2163, 5216, 315, 350, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 423, 311, 279, 8479, 816, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'D', 'Ġis', 'Ġover', 'Ġthere', 'Ġwith', 'ĠR', 'Ġabove', '.Ċ', 'R', 'Ġis', 'Ġbelow', 'ĠS', '.Ċ', 'T', 'Ġis', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'ĠS', 'Ġhorizontally', '.Ċ', 'D', 'Ġis', 'Ġdiagon', 'ally', 'Ġto', 'Ġthe', 'Ġbottom', 'Ġleft', 'Ġof', 'ĠX', '.Ċ', 'Y', 'Ġis', 'Ġplaced', 'Ġin', 'Ġthe', 'Ġleft', 'Ġdirection', 'Ġof', 'ĠT', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠD', 'Ġto', 'Ġthe', 'Ġagent', 'ĠY', '?']`

## row 85877 — h=5, ABOVE

```text
Story:
E is sitting at the upper right position to O.
H is over there with R below.
D is positioned above and to the left of H.
D is diagonally to the upper right of O.
F is to the left of R horizontally.

Question:
What is the relation of the agent E to the agent F?
```

- sequence length: 68
- readout index: 67
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 36, 374, 11961, 520, 279, 8582, 1314, 2361, 311, 507, 627, 39, 374, 927, 1070, 449, 432, 3770, 627, 35, 374, 35328, 3485, 323, 311, 279, 2163, 315, 473, 627, 35, 374, 85118, 750, 311, 279, 8582, 1314, 315, 507, 627, 37, 374, 311, 279, 2163, 315, 432, 59988, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 469, 311, 279, 8479, 435, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'E', 'Ġis', 'Ġsitting', 'Ġat', 'Ġthe', 'Ġupper', 'Ġright', 'Ġposition', 'Ġto', 'ĠO', '.Ċ', 'H', 'Ġis', 'Ġover', 'Ġthere', 'Ġwith', 'ĠR', 'Ġbelow', '.Ċ', 'D', 'Ġis', 'Ġpositioned', 'Ġabove', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠH', '.Ċ', 'D', 'Ġis', 'Ġdiagon', 'ally', 'Ġto', 'Ġthe', 'Ġupper', 'Ġright', 'Ġof', 'ĠO', '.Ċ', 'F', 'Ġis', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠR', 'Ġhorizontally', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠE', 'Ġto', 'Ġthe', 'Ġagent', 'ĠF', '?']`

## row 91245 — h=5, ABOVE

```text
Story:
W is directly below G.
B is positioned right to U.
The objects O and Z are over there. The object O is above and slightly to the right of the object Z.
G is positioned down and to the left of Z.
U is on the right and W is on the left.

Question:
What is the relation of the agent O to the agent B?
```

- sequence length: 78
- readout index: 77
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 54, 374, 6089, 3770, 480, 627, 33, 374, 35328, 1314, 311, 549, 627, 791, 6302, 507, 323, 1901, 527, 927, 1070, 13, 578, 1665, 507, 374, 3485, 323, 10284, 311, 279, 1314, 315, 279, 1665, 1901, 627, 38, 374, 35328, 1523, 323, 311, 279, 2163, 315, 1901, 627, 52, 374, 389, 279, 1314, 323, 468, 374, 389, 279, 2163, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 507, 311, 279, 8479, 426, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'W', 'Ġis', 'Ġdirectly', 'Ġbelow', 'ĠG', '.Ċ', 'B', 'Ġis', 'Ġpositioned', 'Ġright', 'Ġto', 'ĠU', '.Ċ', 'The', 'Ġobjects', 'ĠO', 'Ġand', 'ĠZ', 'Ġare', 'Ġover', 'Ġthere', '.', 'ĠThe', 'Ġobject', 'ĠO', 'Ġis', 'Ġabove', 'Ġand', 'Ġslightly', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'Ġthe', 'Ġobject', 'ĠZ', '.Ċ', 'G', 'Ġis', 'Ġpositioned', 'Ġdown', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠZ', '.Ċ', 'U', 'Ġis', 'Ġon', 'Ġthe', 'Ġright', 'Ġand', 'ĠW', 'Ġis', 'Ġon', 'Ġthe', 'Ġleft', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠO', 'Ġto', 'Ġthe', 'Ġagent', 'ĠB', '?']`

## row 97156 — h=5, ABOVE

```text
Story:
The object P and V are there. The object V is above and slightly to the left of the object P.
The object labeled F is positioned to the right of the object labeled D.
S is below W with a small gap between them.
D is placed at the upper left of P.
S is positioned above and to the left of F.

Question:
What is the relation of the agent W to the agent V?
```

- sequence length: 88
- readout index: 87
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 791, 1665, 393, 323, 650, 527, 1070, 13, 578, 1665, 650, 374, 3485, 323, 10284, 311, 279, 2163, 315, 279, 1665, 393, 627, 791, 1665, 30929, 435, 374, 35328, 311, 279, 1314, 315, 279, 1665, 30929, 423, 627, 50, 374, 3770, 468, 449, 264, 2678, 13225, 1990, 1124, 627, 35, 374, 9277, 520, 279, 8582, 2163, 315, 393, 627, 50, 374, 35328, 3485, 323, 311, 279, 2163, 315, 435, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 468, 311, 279, 8479, 650, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'The', 'Ġobject', 'ĠP', 'Ġand', 'ĠV', 'Ġare', 'Ġthere', '.', 'ĠThe', 'Ġobject', 'ĠV', 'Ġis', 'Ġabove', 'Ġand', 'Ġslightly', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'Ġthe', 'Ġobject', 'ĠP', '.Ċ', 'The', 'Ġobject', 'Ġlabeled', 'ĠF', 'Ġis', 'Ġpositioned', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'Ġthe', 'Ġobject', 'Ġlabeled', 'ĠD', '.Ċ', 'S', 'Ġis', 'Ġbelow', 'ĠW', 'Ġwith', 'Ġa', 'Ġsmall', 'Ġgap', 'Ġbetween', 'Ġthem', '.Ċ', 'D', 'Ġis', 'Ġplaced', 'Ġat', 'Ġthe', 'Ġupper', 'Ġleft', 'Ġof', 'ĠP', '.Ċ', 'S', 'Ġis', 'Ġpositioned', 'Ġabove', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠF', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠW', 'Ġto', 'Ġthe', 'Ġagent', 'ĠV', '?']`

## row 90117 — h=5, BELOW

```text
Story:
O is sitting in the right direction of Y.
E is on the same vertical plane directly below Q.
X is at the 3 o'clock position relative to G.
Y is above X and to the left of X.
O is positioned below and to the right of Q.

Question:
What is the relation of the agent G to the agent E?
```

- sequence length: 74
- readout index: 73
- readout token: `?` (id=30)
- token IDs: `[128000, 18422, 512, 46, 374, 11961, 304, 279, 1314, 5216, 315, 816, 627, 36, 374, 389, 279, 1890, 12414, 11277, 6089, 3770, 1229, 627, 55, 374, 520, 279, 220, 18, 297, 63510, 2361, 8844, 311, 480, 627, 56, 374, 3485, 1630, 323, 311, 279, 2163, 315, 1630, 627, 46, 374, 35328, 3770, 323, 311, 279, 1314, 315, 1229, 382, 14924, 512, 3923, 374, 279, 12976, 315, 279, 8479, 480, 311, 279, 8479, 469, 30]`
- tokens: `['<|begin_of_text|>', 'Story', ':Ċ', 'O', 'Ġis', 'Ġsitting', 'Ġin', 'Ġthe', 'Ġright', 'Ġdirection', 'Ġof', 'ĠY', '.Ċ', 'E', 'Ġis', 'Ġon', 'Ġthe', 'Ġsame', 'Ġvertical', 'Ġplane', 'Ġdirectly', 'Ġbelow', 'ĠQ', '.Ċ', 'X', 'Ġis', 'Ġat', 'Ġthe', 'Ġ', '3', 'Ġo', "'clock", 'Ġposition', 'Ġrelative', 'Ġto', 'ĠG', '.Ċ', 'Y', 'Ġis', 'Ġabove', 'ĠX', 'Ġand', 'Ġto', 'Ġthe', 'Ġleft', 'Ġof', 'ĠX', '.Ċ', 'O', 'Ġis', 'Ġpositioned', 'Ġbelow', 'Ġand', 'Ġto', 'Ġthe', 'Ġright', 'Ġof', 'ĠQ', '.ĊĊ', 'Question', ':Ċ', 'What', 'Ġis', 'Ġthe', 'Ġrelation', 'Ġof', 'Ġthe', 'Ġagent', 'ĠG', 'Ġto', 'Ġthe', 'Ġagent', 'ĠE', '?']`

