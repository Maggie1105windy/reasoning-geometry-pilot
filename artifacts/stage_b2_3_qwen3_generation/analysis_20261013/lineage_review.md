# Stage B2.3 研究思路与前序证据复盘

整理日期：2026-10-13。本文依据本地研究方案、仓库报告、冻结协议和可读聊天摘要梳理研究路线；不修改既有实验，不把本地旧文献笔记当作已独立验证的新论文。研究方案中的段落编号 P 为 OOXML 提取顺序，供定位使用，不是页码。

## 1 总项目与空间分支应分开说明

总项目的较干净原始来源是 [研究方案附件 初稿.docx](<D:/2026本基/立项申请书/研究方案附件 初稿.docx>)。项目名称为“可验证推理任务中大语言模型后训练的轨迹几何机制与因果验证研究”。P25 的目标是：在数学、代码等可验证任务上，比较同一模型家族的 Base、SFT、DPO、RLVR，检验后训练是否主要重加权 Base 中已有的正确轨迹，使正确性信号更早可读、轨迹更稳定、关键推理 token 更可能，并验证这些改变对最终正确性的因果作用。

三个原始研究问题（该附件 P33–P35；与 [7月文献调研与研究思路计划.docx](<D:/2026本基/7月文献调研与研究思路计划.docx>) P2 一致）为：

1. RQ1：后训练是否让模型更早表现出正确性信号的可读性？
2. RQ2：后训练是否稳定化并重加权正确推理轨迹？
3. RQ3：这些几何变化对模型推理最终正确性是否具有因果作用？

方法链为：同题多轨迹采样 → 可验证正确性标签 → 多层、多位置二元正确性 probe → 正确/错误轨迹的曲率、方差、分离和终止区域 → 同一轨迹 teacher-forcing 的 signed Δlogp → 同模型 patching / 双向 steering。附件明确要求优先同模型干预；跨阶段干预先处理表示对齐。

空间路线则是一个受控 pilot 分支。其前期目标见 [grounding_reasoning_first_round_reading.md](<D:/2026本基/grounding_reasoning_first_round_reading.md>)：研究文本内实体—关系绑定与多步关系组合能否区分，以及后训练影响哪个部分。StepGame 提供方向图、实际路径和中间位移等可自动验证标签，适于训练研究流程并建立受控对照。它测量的是任务内关系绑定，不能直接等同现实世界符号接地。

尤其要区分：B1/B2 的八类 relation probe 预测的是方向标签；RQ1 的 correctness probe 预测的是整条轨迹最终正确/错误。前者不能替代后者。B2.3 的 Think/NoThink 比较使用同一套后训练 Qwen3-8B 权重，不能替代 Base→SFT→DPO→RLVR 比较。

## 2 从前序实验到 B2.3 的决策因果链

| 阶段 | 当时要解决的问题 | 得到的证据 | 为什么需要下一步 |
|---|---|---|---|
| A2/A3 | 空间表征 probe 是否能复现、迁移 | Pythia-2.8B World 坐标复现；Llama-3.1-8B 迁移审计与 mini probe | 由静态坐标转向可组合的文本关系 |
| B0 | StepGame hop/模板/标签/泄漏是否可信 | 原 `qaK` 与实际 BFS hop 只有 17.47% 相同；冻结自然分布、平衡、目标未陈述、组合/模板未见等划分 | 先避免把数据错误解释为机制 |
| B1 | Base 是否有显式/派生关系的线性结构 | h1 峰值 F1=0.3573；h2–h5=0.1485–0.1702；无 hop 越大峰值越晚的规律 | probe 可读信息还需行为对照 |
| B1.5/B1.6 | 模型是否实际用到信息、h1结构能否跨hop | candidate 行为弱；transfer 只有弱信号，gap 接近0不能证明共享强代码 | 研究后训练是否改变这两部分 |
| B2 | 同谱系后训练阶段是否改善多跳机制 | h1 晚层 F1 从 Base 0.3218 升至 SFT 0.3907/DPO 0.4045/Final 0.3948；派生行为约0.10–0.11，变化非单调 | 不先判定“后训练没有机制”，先审计测量接口 |
| B2.1 | 低行为是否由 raw/native 接口与 verbalizer 长度产生 | Final Native-Sum derived由0.1112升至0.1829；Sum/Mean出现相反方向偏好；等长映射也会改任务 | Tülu/Llama保留为弱推理控制，引入强模型但先gate |
| B2.2 | Qwen3-8B 在同一冻结评测上是否明显更强 | Native-Sum derived=0.1853，Macro-F1=0.1236，BA=0.1574；相对Tülu增量0.0024，CI跨0；protocol sensitivity HIGH | candidate评测未测真正自由生成和thinking，模型资格仍UNCERTAIN |
| B2.3 | 候选似然是否低估自由生成，thinking是否改善 | FG-N derived=0.2565；FG-T在2048预算下=0.0352，高截断/低解析；门控仍UNCERTAIN | 暂缓大规模activations，先区分预算完成失败和能力不足 |

前序数字与解释来自 [Stage B整合报告](<D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/STAGE_B_RESEARCH_REPORT_ZH.md>)、[B2.1报告](<D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md>)、[B2.2报告](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_2_qwen3/STAGE_B2_2_QWEN3_REPORT.md>)。

B2.3 不是偏離研究问题的无目的换模型。它在修复关键前提：当研究对象还没有稳健成功的多跳行为时，probe 读不出派生信息，不能区分“成功推理采用另一种机制”与“该模型没有成功完成任务”。同时，正确轨迹几何研究需要足够多的真正完成的正确/错误轨迹，而不是大量预算截断序列。

## 3 B2.3 各步骤为什么这样安排

完整执行顺序见 [脚本说明](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/README.md>)。各阶段解决的问题为：

| 步骤 | 设计理由 | 本阶段结果或边界 |
|---|---|---|
| 恢复B2.2并审计 | 保持同2063行、文本/gold/row顺序与模型权重，排除换数据解释 | h1–h5=712/505/422/282/142；row hash一致 |
| 冻结prompt/parser/pilot/config | 阻止看结果后改提示、输出规范、样本或门槛 | 只接受最后一个合法Final answer行；INVALID计错 |
| 1024-token smoke | 仅凭截断调整预算，不凭准确率选预算 | 10/10 thinking smoke截断，因此统一升至2048；仅允许两档 |
| batch/memory smoke | RTX3090 24GB上控制BF16内存与吞吐 | 只基于内存/吞吐选batch，不按正确率选 |
| 250-row × 3-seed pilot | 评估采样随机性，不凭某一seed做决定 | 每hop50行；派生准确率range FG-N .015/FG-T .030，分类LOW |
| matched decoding | 官方FG-N与FG-T温度/top-p不同，需排除采样配置差别 | 同配置的250-row对照仍FG-N .295 vs FG-T .035 |
| 全量FG-N/FG-T | 在原测试分布上回答两个阶段问题 | 每个协议2063行，seed20260924 |
| 配对bootstrap/错误分析/图表 | 同story-group配对，分hop，避免样本不平衡掩盖结果 | 2000重采样；derived是h2–h5等权均值 |
| 独立验证/同步 | 区分产物计算正确与科学解释充分 | 60/60检查通过；PASS_WITH_LIMITATION仍有高截断 |

FG-N 为 `enable_thinking=False`, temperature .7, top-p .8；FG-T 为 `True`, temperature .6, top-p .95；双方 top-k20、min-p0。matched双方用 .6/.95/20/0。它们是同一后训练模型的运行模式差异，并非不同权重阶段。

重要设计限制：matched控制排除了温度/top-p差别，仍没有排除共同2048-token预算造成的差别；它估计的是“固定预算下两模式端到端答题表现”，不是未受约束的内部推理能力差。

## 4 这一阶段的成果和科学解释

正式成果见 [B2.3报告](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/STAGE_B2_3_QWEN3_GENERATION_REPORT.md>) 与 [机器决策](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/fitness_decision.json>)。

### 非thinking完成了更接近日常生成接口的行为校准

FG-N h1=70.22%；h2=33.07%；h3=23.93%；h4=23.05%；h5=22.54%。派生均值25.65%，Macro-F1=25.65%、BA=25.13%，解析率92.48%，无预定义预测塌缩。较B2.2 Native-Sum提升7.12个百分点，95%CI=[3.85,10.55]个百分点。

这支持“评测协议会改变观测行为”。不能说已严格证明candidate likelihood系统性低估了真实推理能力：新自由生成还加入了输出格式说明，协议整体变化；预注册明确低估门槛要求增量≥10个百分点及解析率≥98%，两个条件均不满足。因此既有 `candidate_likelihood_underestimates_behavior=UNCERTAIN` 是符合阈值的，显著非零增量与UNCERTAIN不矛盾。

h1远强于h2–h5是可靠行为现象，但不是内部grounding与reasoning已经严格分离的证据。实体绑定、句子模板、查询反向、组合运算和最终输出均可能影响这种差距。

### thinking结果显示预算下的完成失败

FG-T派生准确率3.52%，解析率4.37%；FG-T−FG-N为−22.13个百分点，95%CI=[−24.87,−19.51]。这是固定协议的端到端结果；官方报告的 `thinking benefit=NO` 可按这一限定理解。

但FG-T h2–h5截断率约89%、97%、98%、99%，thinking长度中位数均2047。许多样本尚未产生final answer。科学上不能由此推断“thinking本身破坏空间推理”或“Qwen3完整思考也无法成功”；也不能只保留已完成样本的高条件正确率代表全体，因为完成与题目难度/正确性相关，存在选择偏差。

本阶段可靠的新发现是：在受控StepGame任务和该预算下，长thinking难以转化为及时、规范的答案。它把停止/完成动力学暴露为后训练轨迹研究的一个必须测量的维度。

## 5 Qwen3.5 的决定存在到什么程度

已记录的决定：`recommended_next_step=C`；[QWEN35_READINESS_NOTE.md](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/QWEN35_READINESS_NOTE.md>) 将 `Qwen/Qwen3.5-9B`、对应Base列为下一候选，要求重新审计tokenizer、模板、reasoning控制、架构与RTX3090内存；明确“未下载、未运行”。

代码 [finalize_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/finalize_b23.py>) 中 A/B/C 的逻辑是：FG-N强则保留非thinking机制路线；只有FG-T强则转向动态reasoning轨迹；两者未达强阈值则Case C。它是预注册工程分支，不是证据已经证明Qwen3.5是最佳或证明Qwen3不具潜在能力。

决定的充分部分：两协议都没有达到strong门槛，所以现在不抽Qwen3全部activations是合理的；下一候选先behavior gate也合理。

尚不充分部分：预算截断导致Qwen3强资格仍 `UNCERTAIN`，而不是 `NO`；当前没有Qwen3.5的同任务证据；架构改动会减少跨模型/阶段几何可比性；若仅有Base与最终模型，无法自动恢复原项目的四阶段归因。

因此建议把“预算审计后的Qwen3复核”与“Qwen3.5小规模资格比较”并行准备，避免仅因一次预算失配就无限换模型。两者均应由新的冻结pilot判断是否扩展，不在本复盘中实际启动。

## 6 已有参考文献如何定位研究贡献

仓库 [CITATIONS.md](<D:/2026本基/参考文献1/reasoning-geometry-pilot/CITATIONS.md>) 主要覆盖Alain/Bengio线性probe、Gurnee/Tegmark空间坐标、StepGame及模板误差、Llama3与Tülu3。总方案还引用DeepSeek-R1、Qwen2.5-Math、RLVR beyond Base争论、Reasoning Models Know When They're Right、LLM Reasoning as Trajectories、On the Direction of RLVR Updates、Sparse but Critical。网页内容和最新日期应在主报告另行联网核实，本文只确认它们在项目中的位置。

| 文献路线 | 已有工作在本项目中的作用 | 值得保留的研究区别 |
|---|---|---|
| 空间表征/线性probe | 说明潜在空间或关系特征可被读出 | 数据/模板控制下，relation读出与实际组合行为分开测 |
| extraction–reasoning外部模块分离 | 可把抽取和规则推理作独立系统 | 单Transformer内是否分离、后训练作用在哪里、状态是否因果使用 |
| RLVR creates vs reweights能力 | 由pass@k讨论能力覆盖与采样概率 | 同题、同轨迹、跨阶段signedΔlogp连接内部几何与外部概率 |
| correctness probing | 中间状态能预测正确性 | 信号出现时间是否随阶段变化，是否跨题泛化及与输出真正使用一致 |
| trajectory geometry/流形 | 正确/错误轨迹存在几何结构 | 稳定正确盆地概率质量与稳定错误盆地、多样性塌缩分开 |
| steering/patching | 可测试内部状态功能 | 同前缀/同模型、双向、随机方向、等幅度对照，串联state→next-token→final accuracy |

当前最有价值的贡献定位不是“首次发现LLM有流形”，也不是“证明RLVR创造符号机制”。更具体的候选贡献是：在可验证任务中，用可复核的阶段比较把接口适配、输出完成、关系/正确性可解码、轨迹概率重加权和因果作用串联起来，解释表征增强何时转化为成功推理、何时仅改变答题接口或停止行为。

B2.3本身主要产生行为校准和复现基础设施成果，没有测hidden-state轨迹、非线性流形、正确性信号出现时间、Δlogp或因果干预；它不能宣称总RQ已得到回答。

## 7 建议的下一阶段路线

1. **先完成预算与终止审计。** 在冻结pilot上另预注册2048/4096/8192等预算，固定模式/采样参数/seed，报告准确率、parse率、thinking闭合率、EOS率、重复率与计算成本；必要时对同一未完成前缀继续生成，避免把换seed与增加预算混淆。预算选择仅由完成率/资源上限决定，不能挑正确率最好的预算。所有新结果独立归档。
2. **建立错误类型层次。** 先区分预算耗尽、think未关闭、格式未遵守、已提交但关系错误，再对完成答案作实体/边抽取、方向反转、路径与位移组合审查。严格parser保留为主；宽松提取或人工盲审只作预注册敏感性验证，防止事后“救分”。
3. **用oracle符号输入区分语言绑定与关系组合。** 在新开发/验证集做自然语言、规范化关系边、显式中间累计位移的对照；配实体重命名、句序、查询反转、模板重述。当前Primary test不可用来调提示再报告泛化。
4. **给模型选择设置停止条件。** 若提高预算后Qwen3仍没有足够完成和稳定成功轨迹，比较一个新候选；若一个路线达到了合理的行为资格，先做小规模机制闭环，避免连续gate挤占总项目。
5. **回到总RQ。** 同题多条correct/wrong轨迹，按题目分组划分训练/验证/测试；先少数层与chunk位置的correctness AUC/校准/长度控制，继而正确/错误盆地方差及策略多样性、signedΔlogp和终止相关变化。StepGame用于过程标签，MATH-500用于原项目对照，代码域后扩展。
6. **最后开展因果闭环。** 在相同输入前缀上测试关键位置对下一token分布和最终正确率的贡献；先同模型correct↔wrong patch，再正/反方向steering；加入随机位置/方向、等范数、任务外损害、长度和格式控制。仅有probe高分或PCA漂亮分群不能作为因果证据。

“流形”应有可检验对象：表示流形、动态轨迹、训练checkpoint路径和log-policy子空间分别测量，不用同一词替代全部。低维PCA或维数下降本身不证明真实流形形成；跨模型同hidden size也不证明坐标已对齐。

## 8 本地资料的更新和质量边界

- README/STATUS/CITATIONS最近更新为2026-09-10，尚未列B2.2/B2.3；当前最新阶段应以独立artifacts为准。
- 所给附件 [已粘贴的文本.txt](<C:/Users/Maggie/.codex/attachments/77dbd572-60a6-4695-aeab-eefd709f214e/已粘贴的文本.txt>) 标题实际是Stage B2.2 Model Selection Gate，解释B2.1→B2.2，不能称为B2.3原方案。
- [0508后训练本基申报书.docx](<D:/2026本基/立项申请书/0508后训练本基申报书.docx>) 新目标与附件初稿一致，但仍混有旧“对齐税/空间挤压”段落；引用当前目标宜优先干净附件。
- `Tracing the Representation Geometry of Language Models from Pretraining to Post-training.pdf` 与 `Direct Preference Optimization Your Language Model is Secretly a Reward Model.pdf` 是逐字节同一文件，SHA256均为 `92CB3A2B71362ACDA98A789B03D88688FD33CF5FCF13F81D2B1DE30EE7D3B67A`。前者名称不能证明已拥有对应论文。
- 3.24 idea里的某些“首次从几何上证明”等强陈述为早期笔记，不能直接转成综述事实或独特性主张。
- 研究跟踪聊天与StageB2.3聊天可辅助追溯意图，但旧agent解释不是独立科学验证。执行聊天截至2026-10-12说明B2.3完成/6290文件SHA一致，当前不需要把早期网络收尾阻塞当作实验未完成。

只读抽取结果保留于 [sources_text](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/analysis_20261013/sources_text>)，以便核对原始研究计划；它们是辅助原文，不作为新增实验结果。
