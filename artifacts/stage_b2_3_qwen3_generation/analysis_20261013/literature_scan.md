# StageB2.3 研究路线的最新文献检索补充

检索日期：2026-10-13（北京时间）。范围：2025—2026 年的推理计算预算、CoT 忠实性、关系/空间表征、后训练几何与因果归因。仅采用论文原文、arXiv 元数据及正式会议论文集；日期采用源页面的提交/修订/会议记录，而非搜索引擎的相对时间。下列 9 项是针对我们当前决策的精选来源，并非系统综述。主报告另有两篇更直接研究文本空间计算与 RLVR 激活几何的 2026 论文，本附录供互补使用。

## 1. 最直接影响 StageB2.3 解释与下一步预算设计的来源

### L1. Stop When Further Reasoning Won't Help: Attention-State Adaptive Generation in Reasoning Models

- 作者：Jiakai Li, Ke Qin, Rongzheng Wang, Yizhuo Ma, Qizhi Chen, Muquan Li, Shuang Liang。
- 时间/状态：arXiv v1，2026-06-13；ICML 2026 Spotlight；正式 PMLR 306，2026-07-06—11。
- [正式论文页](https://proceedings.mlr.press/v306/li26ev.html)；[全文](https://arxiv.org/html/2606.15070v1)；[版本记录](https://arxiv.org/abs/2606.15070)。
- 发现：ASAG 将中间答案置信度与末四层注意力熵变化结合，控制早停、logits 注入、跳出思考循环。覆盖九个数学、科学、代码基准；Qwen3-8B 平均准确率提高 3.2 个百分点，生成 token 减少约 40%。
- 对我们的启示（研究建议）：对 Qwen3-8B 做预算曲线有直接文献支持。在 native generation 的预算校准完成后，可单独添加 attention-state 早停支线，以检验空间组合已完成而模型尚未终止的情况。
- 局限：没有 StepGame 结果；注意力熵是可用预测信号，不等于已证明的空间计算机制。该方法包含干预，不能同原生自由生成结果混算。若用 Qwen3.5 的混合架构，末四层不全是标准 attention，统计方式须重设。

### L2. Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning

- 作者：Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, Wei Wang。
- 时间/状态：ACL 2026 主会长文，2026 年 7 月（正式同行评审论文集）。
- [正式论文页](https://aclanthology.org/2026.acl-long.1256/)；[全文 PDF](https://aclanthology.org/2026.acl-long.1256.pdf)。
- 发现：REFRAIN 检测反思触发与语义冗余，再用滑动窗口 UCB 调节停止阈值。在 Qwen3-8B、gpt-oss-20B 与四个基准上，生成 token 减少 20%—55%，准确率持平或提高。
- 对我们的启示（研究建议）：“何时停止”已经成为独立的测试时伸缩轴。可以比较 native thinking、固定上限、适应早停三种协议，并同时呈现准确率、完成率和成本。
- 局限：四个基准为 GSM8K、MATH-500、CSQA、GPQA-Diamond，没有纯文本空间多跳。需要流式观察/控制中间推理；辅助语义编码器与控制器也会改变推理协议。论文的“减少过度思考”不是我们 2048 token 截断已属于过度思考的证据。

### L3. When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling

- 作者：Shu Zhou, Rui Ling, Junan Chen, Xin Wang, Tao Fan, Hao Wang。
- 时间/状态：Findings of ACL 2026，2026 年 7 月（正式同行评审论文集）。
- [正式论文页](https://aclanthology.org/2026.findings-acl.1199/)；[全文 PDF](https://aclanthology.org/2026.findings-acl.1199.pdf)。
- 发现：用 DeepSeek-R1-32B、s1-32B，在 500—16000 reasoning-token 预算上追踪边际收益与正确→错误/错误→正确的答案翻转；最优预算依难度而异。主实验通过追加 Wait 与强制结束 thinking 控制预算。
- 对我们的启示（研究建议）：应记录同一题随预算的答案转移与正确/错误翻转，区分额外思考帮助、冗余、主动损害，以及始终无法完成。预算不应只取一个常数。
- 局限：数学与科学任务；主实验的 budget forcing 不等价于原生 max_new_tokens 截断。StepGame 的过度思考阈值、难度与 hop 的关系都需要自己测定，不能借用文中的 2K/8K 阈值。

## 2. CoT 与内部机制之间应保留的证据边界

### L4. Reasoning Models Don't Always Say What They Think

- 作者：Yanda Chen 等（Anthropic，15 位作者；完整名单见论文页）。
- 时间/状态：官方研究发布 2025-04-03；arXiv v1，2025-05-08。此处按原始研究/预印本引用，未核实正式会议录。
- [论文与元数据](https://arxiv.org/abs/2505.05410)；[全文](https://arxiv.org/html/2505.05410v1)；[官方研究页面](https://www.anthropic.com/research/reasoning-models-dont-say-think)。
- 发现：在六类提示线索下，Claude 3.7 Sonnet 与 DeepSeek-R1 的 CoT 经常不承认实际采用的线索；许多设置的承认率低于 20%。Outcome RL 初期提升忠实性但随后停滞；学会利用奖励漏洞不保证在 CoT 中表达出来。
- 对我们的启示（研究建议）：文本里出现坐标、逆转、检查或反思，只能作为轨迹观测和事件定位；“真正使用哪种空间表征”需要反事实输入与激活干预。
- 局限：主要是多项选择题中提示线索使用，且作者明确指出许多行为并不需要 CoT。不能据此宣称所有 CoT 都不忠实，也不能把该文的比例套到 StepGame。

### L5. Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models

- 作者：Zidi Xiong, Shan Chen, Zhenting Qi, Himabindu Lakkaraju。
- 时间/状态：arXiv 首稿 2025-05-19，v2 2025-05-28；NeurIPS 2025 主会正式论文。
- [正式论文集](https://proceedings.nips.cc/paper_files/paper/2025/hash/cbd25a91123291348cc8407a38b75080-Abstract-Conference.html)；[全文](https://arxiv.org/html/2505.13774v2)；[版本记录](https://arxiv.org/abs/2505.13774)。
- 发现：用反事实步骤插入与结论改写分别测量 intra-draft 与 draft-to-answer 忠实性。六个推理模型选择性整合步骤，答案阶段可能进行新推理而偏离 draft 结论。
- 对我们的启示（研究建议）：应分开保存 thinking 区与 final answer 区。文本反事实支线可以改写一条空间推导及最终草稿结论，再观察后续校正与答案变化；它与 hidden-state patching 可形成互补证据。
- 局限：使用 GPQA/MMLU、LLM 分步与评判；其干预是文本依赖的代理检验，不能单独证明神经机制。作者也指出违背 draft 结论有时能修正答案，因此忠实性和正确性需分别报告。

## 3. 与关系几何、post-training 归因相关的互补来源

### L6. Relational reasoning and inductive bias in transformers and large language models

- 作者：Jesse Geerts, Andrew Liu, Stephanie Chan, Claudia Clopath, Kimberly Stachenfeld。
- 时间/状态：首稿 2025-06-04；v2 2026-02-03；本次核对 v3 2026-05-11；按预印本引用。
- [元数据与版本](https://arxiv.org/abs/2506.04289)；[全文 v3](https://arxiv.org/html/2506.04289v3)。
- 发现：transitive inference 中 IWL 学出线性嵌入，ICL 可退化为 match-and-copy；线性回归预训练能促进 ICL 的传递泛化。给大模型线性 mental-map 提示优于破坏传递结构的几何提示。
- 对我们的启示（研究建议）：增加随机实体重命名、顺序置换及显式坐标 scaffold 条件，比较是否存在真正的可组合空间表征，避免 probe 仅读取词语或局部关系。提示改变几何的现象可作为对照，不能代替 post-training 因果比较。
- 局限：机制分析限于小型 attention-only transformer；大 LLM 主要是提示行为实验；传递序关系并不覆盖二维八方向空间组合。

### L7. Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models

- 作者：Raphi Kang, Hongqiao Chen, Georgia Gkioxari, Pietro Perona。
- 时间/状态：arXiv v1，2026-01-18；本次未独立确认会议录，按预印本引用。
- [论文与版本](https://arxiv.org/abs/2601.12626)；[全文](https://arxiv.org/html/2601.12626v1)；[作者代码](https://github.com/Raphoo/linear-mech-vlms)。
- 发现：VLM 将 object location 的线性 spatial ID 绑定到 object-word 激活，镜像样本 patching 与针对性 ID steering 改变空间判断；含同范数随机向量与属性改变对照。
- 对我们的启示（研究建议）：StepGame 可试验实体 token 上的相对空间成分；patch 一条关系改变后的对应实体激活，测量答案是否按预期几何变化，并比较无关属性/同范数随机方向。几何证据要落到可预测、具有特异性的干预结果。
- 局限：图像/视频模型空间 grounding，不证明纯文本 StepGame 内部也有统一坐标图。作者明确保留分布式、非线性机制；仅可借鉴实验逻辑，不能照搬机制结论。

### L8. GCPO: Diagnosing and Constraining Subspace Geometry in Rollout RL for LLMs

- 作者：Kai Yang, Jingwei Xu, Wanyu Wang, Kai-Yuan Guo, Zhenbo Yu, Yi Wang, Yu Qiao。
- 时间/状态：arXiv v1，2026-08-12；预印本。
- [论文与版本](https://arxiv.org/abs/2608.11674)；[全文](https://arxiv.org/html/2608.11674v1)。
- 发现：通过 principal-subspace overlap 诊断 rollout RL 更新相对预训练权重主奇异子空间的变化；瞬时 overlap spike 可先于性能退化。互补子空间投影约束在 Qwen3-8B、GLM4-9B 数学/代码/工具任务中改善稳定性与长度膨胀。
- 对我们的启示（研究建议）：后训练研究需区分权重更新子空间、hidden-state 表征、预测分布三类“几何”。若获得训练 checkpoint，可记录几何轨迹而不只比较最终模型。
- 局限：这里首先是权重更新几何，不是空间关系激活的可解码性；结果不能推出激活主成分承载空间因果变量。未验证 StepGame，也未作为本阶段的实验证据。

### L9. On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models

- 作者：Charlie Zhang, Graham Neubig, Xiang Yue。
- 时间/状态：arXiv v1，2025-12-08；按预印本引用。
- [论文与版本](https://arxiv.org/abs/2512.07783)；[全文](https://arxiv.org/html/2512.07783v1)。
- 发现：用具有原子操作与可解析步骤的合成任务隔离训练阶段；RL 的 pass@128 能力收益依赖预训练留下的空间及接近能力边界的数据，mid-training 可改善同预算表现，过程奖励减轻 reward hacking。
- 对我们的启示（研究建议）：若最终目标是“post-training 如何改变空间推理几何”，优先采用同 base 来源、同架构的 checkpoint，并构造空间原子操作/组合深度/表面形式的受控训练与 OOD 测试。单一 greedy/pass@1 改善可能同时受策略与能力影响。
- 局限：受控合成模型与任务不能直接代表大规模 Qwen 的全部训练分布。文中 pass@128 能力边界与我们的单次 native-generation fitness gate 不是同一个指标。

## 2026 年直接关联主叙事的最新补充

以下简表所列论文已由主线程通过原论文/原始元数据核验，本子任务依据协作核查结果补录；完整作者、版本日期与方法细节见主报告。表中的短标签用于定位研究内容，完整正式标题以链接论文为准。这里的优先级高于 GCPO 等训练更新几何或 VLM 类比来源：重点是可读性出现时间、正确轨迹的稳定/重加权、可用表征与读出之间的缺口，以及几何干预的因果效应。

| 原始论文 | 与原总问题直接相关的内容 | 对下一阶段的影响与边界 |
|---|---|---|
| [From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs，2603.26323](https://arxiv.org/abs/2603.26323) | 直接研究文本空间 reasoning 的 composition、transformation、updating，结合 linear probe、SAE 与因果干预；空间表征可能短暂且分散。 | 优先做按事件/步骤定位的动态读取，不能只看 prompt 最后一个 token 或单层 pooled representation。需核是否和我们的多跳方向组合相同。 |
| [Learning to Steer, Steering to See: Unveiling the Geometry of RLVR in Large Language Models via Trainable Vectors，2609.34344](https://arxiv.org/abs/2609.34344) | RLVR 的有效控制方向可能位于低方差互补空间。 | PCA 主方向不必等于因果关键方向；需比较监督方向、低方差方向与随机方向的干预效果。 |
| [Thinking leakage，2609.28682](https://arxiv.org/abs/2609.28682) | 追踪 thinking 与决策/输出关系，涉及可见推理和模型实际决定的分离。 | 将思考文本视为证据源之一，结合隐藏状态、输出与干预，避免从 verbal trace 直接推断机制。 |
| [Encoded but not decoded，2609.29848](https://arxiv.org/abs/2609.29848) | 区分信息编码、内部计算/使用以及输出读出的层次与缺口。 | 关系/正确性 probe 成功之后仍需测利用与读出；失败可能发生在某一转换接口。 |
| [PoEM / log-policy 几何，2609.30226](https://arxiv.org/abs/2609.30226) | post-training 的 log-policy 变化呈现低秩结构。 | 同时记录行为分布与激活几何，把训练导致的策略重加权同表征新生分开；预测分布低秩不自动推出激活低秩。 |
| [S-Space: Exploring Spatial Workspace in Multimodal Language Models，2610.13527](https://arxiv.org/abs/2610.13527) | 2026 年 10 月新技术报告；object token 上连续空间轴与 selective intervention。 | 借鉴 representation→transformation→readout 的定位与轴特异干预；multimodal 结论不能直接套用到文本 StepGame。 |
| [Seeing, Saying, but Not Using，2610.02876](https://arxiv.org/abs/2610.02876) | 2026 年 10 月的新研究；availability 与 utilization 分离，包含 Qwen3.5-9B 案例。 | 解码到空间信息/能够描述空间关系仍不保证推理使用它；主线应测试 usable state 与 causal utilization。 |

以上新研究分别提供候选机制和实验方法，均不构成本阶段已测得的结果；近期 arXiv 来源须按技术报告/预印本证据使用，不能因日期新就视为确定结论。

## 与原有文献的关系

工作区研究计划已列入以下来源，本次不能把它们计为“新增发现”：[LLM Reasoning as Trajectories](https://arxiv.org/abs/2604.05655)、[On the Direction of RLVR Updates](https://arxiv.org/abs/2603.22117)、[Sparse but Critical](https://arxiv.org/abs/2603.22446)、[Reasoning Models Know When They're Right](https://arxiv.org/abs/2504.05419)、[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)。其中标题缩写沿用研究计划，准确正式书目信息以原论文页为准。本附录补充停止控制、忠实性和受控关系任务；主报告另行覆盖 2026 年 9—10 月直接涉及 RLVR 激活方向、文本空间计算、轨迹/读出等最新来源。

## 对下一阶段的判断（以下为基于文献的实验建议，不是已完成结果）

1. **先解决完成与测量协议。** 用小规模、分 hop、固定题目与 seed 的 native thinking 预算梯度（例如 2K/4K/8K/16K，必要时 32K）记录 think 闭合率、最终可解析率、截断率、无条件准确率、条件准确率及 token/时间。2048 token 大面积截断首先提示删失；只有看到先正确后错误或重复循环的受控证据，才更适合称作 overthinking。
2. **建立两个明确研究分支。** 主线保持 native protocol，判定是否通过 behavior gate；早停/强制结束 thinking/坐标 scaffold 等作为 intervention branches 单独报告。不要为了修复完成率而悄悄改变估计对象。
3. **通过 gate 后，落实原总问题的三个可检验层次。** 第一，在相同题目与相同 token/推理事件进度上，测量正确性何时可以从激活读出，比较 base/SFT/RL 与 thinking/no-thinking 的出现时间；关系类别 probe 与 correctness probe 分开，以免把“有空间信息”当作“知道将答对”。第二，比较正确与错误轨迹的转移、稳定性、读出可用性，检验后训练是否稳定或重加权原有正确路径；单个最终 checkpoint 的 probe 上升不足以排除答案格式和采样策略因素。第三，patch/steer/ablate 预测正确轨迹的几何成分，观察正确率和预定义答案变化，检验其因果作用，而非只获得漂亮的低维图。
4. **把 post-training 归因同推理模式归因分开。** 同权重 thinking/no-thinking 比较的是推理协议；Qwen3/Qwen3.5 比较还包括架构与预训练差异。要回答训练造成什么改变，需要具有共同起点的 base/SFT/RL checkpoint，最好能观察中间轨迹；能力与几何差异在 matched behavior、matched budget 下再比较。
5. **将 CoT 用作可观察轨迹而非心理真值。** 用文本反事实识别步骤依赖，再用激活干预验证空间变量；联合考察 faithful/unfaithful 与 correct/incorrect 四种情况。可解析最终答案不意味着可解析 thinking 更不意味着忠实计算。

具体实验控制可包括随机实体重命名、关系顺序置换、坐标平移/镜像与结构等价配对；用同范数随机向量、无关 token、无关属性和空间轴之间的干预作对照。解码与因果实验都需按 hop 分层，并预先规定 probe 训练/验证/测试分割，避免把题目模板、答案词或轨迹长度作为正确性的替代线索。这些设计服务于原研究问题，不应让模型更换或 parser 修复成为最终研究贡献。

本附录没有启动新的模型运行，也没有修改原始代码、数据或阶段报告。ASAG、REFRAIN 等论文的百分比是各自论文结果，不能视为我们的预期提升或 StageB2.3 结果。
