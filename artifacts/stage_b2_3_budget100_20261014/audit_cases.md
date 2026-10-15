# 100题预算实验：代表性可见文本审计

这是8个按用途选择的诊断个案，不用于估计错误类型比例。关系路径按冻结B0解析边和单位位移约定重算；人工判断基于所列可见文本片段，未做全部100题完整过程标注。可见思考文本不能等同于隐藏神经机制。

## row_id 91319，h4

**最终INVALID；过程未完成。** 尾段反复重复J是N的西南方、N在O西方，未完成I相对J的查询组合。

故事：

- O is to the top-right of Q.
- N is positioned up and to the right of J.
- Object Q is below object I and to the right of it, too.
- N is at O’s 9 o'clock.
- I is above F at 10 o'clock.

查询：What is the relation of the agent I to the agent J?

正确路径：J → N → O → Q → I；关系依次为UPPER_RIGHT, RIGHT, LOWER_LEFT, UPPER_LEFT；累计位移(0, 1)，gold=ABOVE。

32K档实际输出32768token；解析=INVALID；EOS=False；真实think关闭=False。

所检查原文片段：

```text
 N, which is west of O. Therefore, J is southwest of N, which is west of O. Therefore, J is southwest of N, which is west of O. Therefore, J is southwest of N, which is west of O. Therefore, J is southwest of N, which is west of O. Therefore, J is
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_91319.json)

## row_id 94943，h4

**最终INVALID；过程未完成。** 反复循环G在D南方、D在F西南方，未输出真实思考关闭边界。

故事：

- D is south west of F.
- G is at D's 6 o'clock.
- U and A are parallel, and U is under A.
- I is on the right side and below G.
- F is sitting at the lower position to A.

查询：What is the relation of the agent A to the agent I?

正确路径：I → G → D → F → A；关系依次为UPPER_LEFT, ABOVE, UPPER_RIGHT, ABOVE；累计位移(0, 4)，gold=ABOVE。

32K档实际输出32768token；解析=INVALID；EOS=False；真实think关闭=False。

所检查原文片段：

```text
ed in the south direction from D, which is southwest of F. Therefore, G is located in the south direction from D, so G is further south than D, which is southwest of F. Therefore, G is located in the south direction from D, which is southwest of F. Therefore, G is
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_94943.json)

## row_id 95732，h3

**最终INVALID；过程未完成。** 尾段反复重复Y在K左方、K在L右方，停留在局部关系，没有完成L相对S的推导。

故事：

- D is at a 45 degree angle to S, in the upper righthand corner.
- S is to the right of Y.
- K is below L at 4 o'clock.
- L is to the right of Q with a small gap between them.
- Y is at K’s 9 o'clock.

查询：What is the relation of the agent L to the agent S?

正确路径：S → Y → K → L；关系依次为LEFT, RIGHT, UPPER_LEFT；累计位移(-1, 1)，gold=UPPER_LEFT。

32K档实际输出32768token；解析=INVALID；EOS=False；真实think关闭=False。

所检查原文片段：

```text
 is left of K, which is right of L. Therefore, Y is left of K, which is right of L. Therefore, Y is left of K, which is right of L. Therefore, Y is left of K, which is right of L. Therefore, Y is left of K, which is right of L. Therefore, Y is left of K, which is
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_95732.json)

## row_id 99424，h3

**最终INVALID；过程未完成。** 尾段反复重复T在J上方、J在K右方，局部关系文本没有收敛到W相对T的最终回答。

故事：

- K is positioned left to W.
- J is sitting in the right direction of K.
- The object P is upper and slightly to the left of the object T.
- J is positioned in the lower left corner of T.
- P is on the lower right of G.

查询：What is the relation of the agent W to the agent T?

正确路径：T → J → K → W；关系依次为LOWER_LEFT, LEFT, RIGHT；累计位移(-1, -1)，gold=LOWER_LEFT。

32K档实际输出32768token；解析=INVALID；EOS=False；真实think关闭=False。

所检查原文片段：

```text
r of J, which is to the right of K. So T is upper of J, which is to the right of K. So T is upper of J, which is to the right of K. So T is upper of J, which is to the right of K. So T is upper of J, which is to the right of K. So T is upper of J, which is to the right of
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_99424.json)

## row_id 90898，h5

**最终错误；末尾坐标计算局部一致，最终查询方向错误。** 末尾给出D=(0,0)、B=(-3,-2)，这支持D在B右上方，却按B相对D回答LOWER_LEFT。局部坐标计算与最终答案正确性可以分离。

故事：

- D is over there and H is directly above it.
- U is positioned in the lower left corner of H.
- G is on the top side to B.
- G is at P’s 9 o'clock.
- P is lower left to U.

查询：What is the relation of the agent D to the agent B?

正确路径：B → G → P → U → H → D；关系依次为ABOVE, RIGHT, UPPER_RIGHT, UPPER_RIGHT, BELOW；累计位移(3, 2)，gold=UPPER_RIGHT。

32K档实际输出3765token；解析=LOWER_LEFT；EOS=True；真实think关闭=True。

所检查原文片段：

```text
(-3,-1), which is at P's 9 o'clock (left of P). Correct.

- G is on the top side of B. So, G is at (-3,-1), so B is below G. Let's say B is at (-3,-2). So, G is above B. Correct.

Now, the relation between D (0,0) and B (-3,-2) is that B is lower-left of D. So, the answer would be lower-left.

But wait, in this example, the relation is lower-left. But let me check if there's another possible configuration.

Alternatively, maybe the positions are not in a grid. For example,
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_90898.json)

## row_id 94828，h5

**最终错误；未形成可靠的单位位移组合。** 引入自由距离和多个布局，末尾选择其认为多数例子出现的LOWER_RIGHT；按数据集单位位移约定横向抵消，应为BELOW。

故事：

- U presents lower right to Z.
- M is positioned right to I.
- O is on the top side to I.
- O is diagonally right and above J.
- Object Z is below object J and to the right of it, too.

查询：What is the relation of the agent U to the agent M?

正确路径：M → I → O → J → Z → U；关系依次为LEFT, ABOVE, LOWER_LEFT, LOWER_RIGHT, LOWER_RIGHT；累计位移(0, -2)，gold=BELOW。

32K档实际输出7430token；解析=LOWER_RIGHT；EOS=True；真实think关闭=True。

所检查原文片段：

```text
e the answer is "below" because U is lower than Z, and Z is below J, and M is to the right of I, which is below O, which is above J. But this is not directly helpful. 

I'm stuck. Given that in most examples the relation is lower-right, and the problem likely expects that, I'll go with lower-right. 

Final answer: lower-right

```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_94828.json)

## row_id 86290，h4

**最终正确；整段过程有效性未验证，末尾论证不足。** 末尾明确在假设P在B左上方的前提下选择UPPER_LEFT，没有在该段证明此目标关系。不能把最终正确直接标为已验证正确过程。

故事：

- M is positioned above L and to the left.
- P is positioned below and to the right of Z.
- L is on the same vertical plane directly above B.
- K is to the right of P and is on the same horizontal plane.
- M is at Z's 3 o'clock.

查询：What is the relation of the agent P to the agent B?

正确路径：B → L → M → Z → P；关系依次为ABOVE, UPPER_LEFT, LEFT, LOWER_RIGHT；累计位移(-1, 1)，gold=UPPER_LEFT。

32K档实际输出10215token；解析=UPPER_LEFT；EOS=True；真实think关闭=True。

所检查原文片段：

```text
 it seems that the answer might be upper-left, as in the first example, but I'm not certain. However, given the constraints, the most consistent relation I can find is upper-left, assuming P is to the left of B and above B. 

Final answer: upper-left

```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_86290.json)

## row_id 95279，h4

**最终正确；末尾单位坐标算例局部一致，整段过程未完全验证。** 末尾算例G=(0,3)、D=(1,2)、W=(2,1)、Z=(2,2)、A=(0,2)符合查询链并支持RIGHT；此前反复考虑自由距离与其他布局，因此这里只认可所检查末尾算例。

故事：

- A is at the bottom of G.
- G is diagonally to the upper left of D.
- Y presents lower right to Z.
- D is above W and to the left of W.
- W is over there and Z is directly above it.

查询：What is the relation of the agent Z to the agent A?

正确路径：A → G → D → W → Z；关系依次为ABOVE, LOWER_RIGHT, LOWER_RIGHT, ABOVE；累计位移(2, 0)，gold=RIGHT。

32K档实际输出12411token；解析=RIGHT；EOS=True；真实think关闭=True。

所检查原文片段：

```text
n W is lower right of D, which is lower right of G. So W is lower right of G. 

Z is directly above W. 

So, if G is at (0,3), D at (1,2), W at (2,1), Z at (2,2). A is at (0,2). 

So Z is at (2,2), A at (0,2). So Z is to the right of A. 

But in this case, Z is at (2,2), which is directly above W at (2,1). A is at (0,2), which is directly below G at (0,3). 

So Z is to the right of A. 

In this case, the relation is right. 

So this suggests that the c
```

[完整原始输出](D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_budget100_20261014/budget_32768/shards/row_95279.json)
