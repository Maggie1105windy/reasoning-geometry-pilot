# 8K / 16K / 32K 诊断实验：四个个案审计

日期：2026-10-13。仅分析原始 manifest 与已完成的 shards，未追加 GPU 运行，未修改原始输出。

## 方法与边界

先独立逐句转写 StepGame 八方向的单位二维向量，再建双向图，以查询参照实体为起点做 BFS 并求和；之后核对冻结 gold。四题的路径长度与标签全部吻合。x 正向为右，y 正向为上。以下的图坐标是基准数据集的单位关系语义，不以不同实体必须占据不同坐标为前提。

再检查可见 thinking，提取可逐字定位的证据。这里定位的是生成文本中的具体错误，不能据此证明隐藏状态或神经计算机制。个案选取用于解释已观察到的错误，不估计错误类型总体频率。完整向量、各预算运行字段、原文件 SHA256 与引文字符偏移保存在 audit_cases.json。

## 输出对照

| row_id | hop | 独立重算 gold | 8K 答案 / token | 16K 答案 / token | 32K 答案 / token |
|---|---:|---|---|---|---|
| 85687 | 2 | LOWER_RIGHT | RIGHT / 3808 | RIGHT / 3808 | RIGHT / 3808 |
| 90023 | 3 | LOWER_LEFT | UPPER_RIGHT / 3580 | UPPER_RIGHT / 3580 | UPPER_RIGHT / 3580 |
| 101041 | 4 | BELOW | INVALID，截断 / 8192 | LOWER_LEFT / 8253 | LOWER_LEFT / 8253 |
| 92078 | 4 | UPPER_RIGHT | INVALID，截断 / 8192 | UPPER_RIGHT / 10580 | UPPER_RIGHT / 10580 |

四题的 16K 与 32K generated token IDs 全部逐题完全相同。85687、90023 在 8K 已正常结束，增加上限没有延长输出；101041 多生成 61 token 后给出错误答案；92078 多生成 2388 token 后给出正确答案。

## 85687：水平关系未继承纵向状态

原 story：

1. T is above S and to the left of S.
2. Y is sitting at the 6:00 position to B.
3. M is to the left of E.
4. The object labeled E is positioned to the right of the object labeled S.
5. T is at the bottom of Y and is on the same vertical plane.

问题：What is the relation of the agent E to the agent T?

正确路径为 T → S → E：

`S−T=(1,−1); E−S=(1,0); E−T=(2,−1) → LOWER_RIGHT`。

thinking 已正确识别 S 在 T 右下，却把 E 在 S 右方理解为仅确定 x，反复称 E 的 y 不知道。例如：

> E is to the right of S, but could be anywhere vertically.

因此只保留右方分量，最终输出 RIGHT。这个个案直接显示数据集关系语义与生成文本采用的约束语义不一致；8K 已提前正常结束，16K/32K 没有修复。

## 90023：最终回答反转了查询方向

原 story：

1. C and K are side by side with C on the top and K at the bottom.
2. O is upper right to A.
3. O is below and to the right of N.
4. K is at the lower side of A.
5. U is upper right to N.

问题：What is the relation of the agent A to the agent U?

正确路径为 U → N → O → A：

`N−U=(−1,−1); O−N=(1,−1); A−O=(−1,−1); A−U=(−1,−3) → LOWER_LEFT`。

thinking 后段给出 A=(0,2)、U=(1,5)，但明确回答 U 相对 A 的方向：

> A is at (0,2), U is at (1,5). So relative to A, U is to the upper right.

因此输出 UPPER_RIGHT，恰好是 gold 的逆方向。前面曾出现 N 的纵向符号错误，后面又纠正；最后的查询方向反转仍明确可定位。增加预算不改变这个已提前结束的结果。

## 101041：自由距离与禁止重合假设破坏方向抵消

原 story：

1. T and O are in a horizontal line with T on the left.
2. D is sitting at the 9:00 position of J.
3. Y is on the right side to T.
4. D is positioned above and to the left of O.
5. J is to the upper left of A.

问题：What is the relation of the agent Y to the agent J?

正确路径为 J → D → O → T → Y：

`D−J=(−1,0); O−D=(1,−1); T−O=(−1,0); Y−T=(1,0); Y−J=(0,−1) → BELOW`。

在基准单位关系下，Y 与 O 位于相同坐标，两者都是 T 右方一个单位。不同实体可以重合；主评测仅排除查询答案为 OVERLAP，不排除其他实体对重合。

thinking 将 T、O 距离任取为 2，同时将其他边任取不同距离，并明确拒绝 Y 与 O 重合：

> Let me say T is at (0, 0), O is at (2, 0) for simplicity.

> So Y can't be at O's position. So Y is somewhere else.

后面在 BELOW、LOWER_LEFT、LOWER_RIGHT 多种布局之间反复比较，最终以“intended answer”选 LOWER_LEFT。可见错误同时涉及距离语义及排他占位假设，不能归为单一神经机制。16K 解决了输出未结束，但没有解决这个可见推导中的语义问题。

## 92078：答案正确，但推导仍有明确方向错误

原 story：

1. Y and F are next to each other with F on the right and Y on the left.
2. C is sitting at the upper right position to E.
3. The object W is positioned above and to the right of the object C.
4. The object D and E are there. The object E is above and slightly to the left of the object D.
5. F is sitting at the 6:00 position to D.

问题：What is the relation of the agent C to the agent Y?

正确路径为 Y → F → D → E → C：

`F−Y=(1,0); D−F=(0,1); E−D=(−1,1); C−E=(1,1); C−Y=(1,3) → UPPER_RIGHT`。

然而 thinking 开头把 6:00 当作右方，并在后段继续沿用这个解释：

> The 6:00 position is directly to the right, right? Because 6:00 on a clock is pointing to the right.

这个解释应为正下方。thinking 后面大量尝试布局，尾段承认约束没有全满足，再选 UPPER_RIGHT：

> At this point, I think the correct relation is upper-right, based on the earlier setup where C is above and to the right of Y, even if some conditions are not perfectly satisfied.

延长预算确实使答案从无法解析变为正确，但它没有在可见文本中形成符合所有关系的可靠推导。这题不能仅因最终正确就纳入“过程正确轨迹”。

## 对本轮实验的限定结论

增加预算能改善输出完成，不能保证修复已经正常结束的错误，也不能保证最终正确答案对应正确过程。这里三类证据应分别记录：最终答案正确性、输出是否完成、过程是否满足基准关系。只有10题且单个生成 seed，不能估计各错误类型比例，也不能宣布16K为总体最优预算。
