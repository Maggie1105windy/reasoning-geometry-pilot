# Stage B2.3 代码、数据及统计证据审计

审计日期：2026-10-13。原始文件保持不变；只在本目录新增审计脚本、JSON、CSV及本文。未加载模型，未运行GPU实验。

## 结论与证据等级

**现有正式结论**：`PASS_WITH_LIMITATION`；候选似然是否低估行为为 `UNCERTAIN`；thinking是否提升为 `NO`；正常使用适配度 `WEAK`；强推理机制研究对象 `UNCERTAIN`；分支 C，先对另一模型做行为门控。

**本次数据实际支持**：在当前原生模板、固定采样配置及2048新token预算下，FG-N多跳表现高于B2.2 Native-Sum；FG-T的交付失败率极高，主要原因是思考尚未闭合时耗尽预算。不能据此推断thinking的一般能力差、完整推理质量差或所有预算下无益。FG-N与FG-T都未达到冻结的强行为门槛，因此暂缓Qwen3-8B大规模机制实验是合理的；但优先进行预算/终止机制诊断，比直接把当前低分归结于模型能力更符合这些数据。

原始数据及bootstrap数值完整复核通过。另发现真实的legacy think边界标志兼容问题、报告与执行RNG说明不一致、日志保留旧FAIL状态，以及本地研究档案不能原样重跑完整远程验证的问题。

## 读取范围与可检查文件

Stage B2.3正式产物共 **6,282文件**，代码目录共 **13文件**（5个Python源码、2个shell入口、README、5个pyc）；总清单6,295文件、270,708,187字节。清单逐文件记录绝对路径、大小及SHA-256，见 [audit_file_inventory.csv](./audit_file_inventory.csv)。清单未计入本审计目录。

全量读取 **15个row_generations.jsonl、6,166条生成记录及6,166个对应JSON shard**；核验每条冻结gold/hop/story_group、重复与遗漏、parser/correct、token数量及EOS/长度终止、batch seed、CSV字段、JSONL/CSV/shard manifest哈希。未发现数据不一致。所有主指标、pilot/matched汇总、derived/overall、逐行错误迁移、迁移汇总以及8×8混淆矩阵复核无差异。108条公开单组/配对bootstrap区间独立重算，最大绝对误差 **9.8879×10⁻¹⁷**。

机器结果见 [audit_code_data.json](./audit_code_data.json)，区间逐项对照见 [bootstrap_recomputed.csv](./bootstrap_recomputed.csv)。复核程序为 [audit_code_data.py](./audit_code_data.py) 与 [audit_supplement.py](./audit_supplement.py)；按这个次序运行，仅需CPU、numpy、pandas。主脚本若发现同目录已有清单会复用第一次全量哈希清单，原始文件变更后应删除该清单或改脚本强制重新哈希。

## 研究协议与每一步结果

1. 恢复B2.2参照、冻结数据与协议：2,063条test，h1–h5=712/505/422/282/142；manifest SHA-256 `83e6db8374b70ca8a86b466626475a174129a633f3b9a29db2d29672a7eaa50c`；row-ID SHA-256 `dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950`。原始Story/Question逐条重建完全一致，全部user message哈希与冻结audit一致。B2.2 Native-Sum h2–h5等权准确率=0.1852896569。
2. 模型/环境审计（历史远程记录）：Qwen/Qwen3-8B，本地BF16，36层、hidden size4096，eval+inference mode，SDPA；Linux、Python3.10.20、torch2.4.0+cu124、CUDA12.4、transformers4.57.6、RTX3090。config/tokenizer/chat-template/5个weights shard均记录哈希并与B2.2比较PASS。本地档案不含models目录，因此此次不能重新核验实际模型权重或tokenizer文本解码。
3. 长度smoke：两模式各10条，初始1024；FG-T10/10耗尽预算。按冻结规则（≥20%截断）统一切换2048；依据截断而非准确率。
4. 批次smoke：FG-N batch8共8条；FG-T保存batch4的4条及batch8的8条，最终两模式batch8；FG-T batch8最大reserved23.3418GiB。最终记录FG-N wall0.0252秒反映该次调用恢复已有shards，不可作为8条真实生成用时。
5. 稳定性pilot：每hop随机50条，共250；FG-N/FG-T各3个seed（20260924/25/26），共1,500生成。报告表展示的是 **h2–h5等权均值，对应每次200条**，不是250条总体准确率；FG-N=0.300/0.285/0.295，FG-T=0.035/0.055/0.065，准确率range=0.015/0.030，冻结分类LOW。FG-T接近零的稳定性很大程度反映截断地板，不能外推高预算稳定性。
6. matched decoding：同250条，seed20260924，两模式均temperature0.6/top_p0.95/top_k20/min_p0，各250生成。h2–h5等权FG-N-Matched=0.295；FG-T-Matched=0.035；差=-0.260，CI[-0.325,-0.195]。仅此子集控制采样参数；主实验官方使用配置同时改变模式及temperature/top_p。
7. full：两模式各2,063条，seed20260924，共4,126生成。FG-N 865正确、1905有效解析、158无效；FG-T372正确、398有效解析、1665无效。完整运行manifest记录FG-N约10203秒、FG-T约31823秒；这些是实际full调用的wall数，而非所有审计/重试/加载的项目总工时。
8. 统计、图及门控：15个hop主指标行、3个derived/overall行、6张图、配对迁移及108条bootstrap区间，冻结门控产生分支C。
9. 独立验证：最终文件60/60检查通过、PASS_WITH_LIMITATION。原始phase_8日志仍保存修复legacy边界检查前的FAIL；应保留历史同时明确最终状态。

## 主结果：不同分母必须分开

主研究指标为每hop计算后对h2/h3/h4/h5等权平均。其n=1351表示四层样本总数，不表示准确率简单等于正确数/1351。

| 协议 | h2–h5等权Accuracy | Macro-F1 | Balanced accuracy | 等权有效解析率 | h2–h5合并Accuracy（次要描述） |
|---|---:|---:|---:|---:|---:|
| B2.2 Native-Sum | 0.185290 | 0.123613 | 0.157406 | 1.000000 | 238/1351=0.176166 |
| FG-N | 0.256470 | 0.256518 | 0.251349 | 0.924790 | 365/1351=0.270170 |
| FG-T | 0.035167 | 0.061740 | 0.033957 | 0.043672 | 60/1351=0.044412 |

FG-N−B2.2 Native-Sum等权Accuracy=+0.0711798753，CI[+0.0385437464,+0.1055491815]；FG-T−FG-N=-0.2213024769，CI[-0.2486846573,-0.1951412852]。这些是在给定协议及预算下的行为差异。候选似然对比同时改变答案生产方式及附加格式指令，不能宣称free generation数学上修正了似然测量。

| 模式/hop | n | 正确 | 有效解析 | 耗尽预算且无效 | EOS完成但格式无效 | EOS有效但错 | 有效输出内正确率（选择样本） |
|---|---:|---:|---:|---:|---:|---:|---:|
| FG-N h1 | 712 | 500 | 660 | 0 | 52 | 160 | 75.76% |
| FG-N h2 | 505 | 167 | 473 | 3 | 29 | 306 | 35.31% |
| FG-N h3 | 422 | 101 | 380 | 2 | 40 | 279 | 26.58% |
| FG-N h4 | 282 | 65 | 257 | 2 | 23 | 192 | 25.29% |
| FG-N h5 | 142 | 32 | 135 | 1 | 6 | 103 | 23.70% |
| FG-T h1 | 712 | 312 | 322 | 390 | 0 | 10 | 96.89% |
| FG-T h2 | 505 | 46 | 56 | 449 | 0 | 10 | 82.14% |
| FG-T h3 | 422 | 6 | 12 | 410 | 0 | 6 | 50.00% |
| FG-T h4 | 282 | 6 | 6 | 276 | 0 | 0 | 100.00%（6例） |
| FG-T h5 | 142 | 2 | 2 | 140 | 0 | 0 | 100.00%（2例） |

表中“耗尽预算”严格按`generation_token_count>=max_new_tokens`，“完成”严格按最后一个生成token属于记录的EOS集合；两模式本次所有行中hit-max和EOS互斥、覆盖全部样本。本次未出现“截断且有效”，因此上表也覆盖全部正确/错误/格式失败/截断情况。

FG-T无效输出恰好全部为无EOS、未闭合思考、2048截断；EOS完成398例恰好全部有效。高条件正确率只说明完成子样本的行为，**难题更容易截断，完成条件是强选择变量**；不能据其100%（6/2例）推断h4/h5总体推理正确率。FG-N的158无效只有8例截断，主要另有150例格式问题；1905例有效中1040例答案错，显示其主要瓶颈也包含真实任务错误。

全体错误迁移：Candidate错误→FG-N正确655例，Candidate正确→FG-N错误192例；FG-N错误→FG-T正确101例，FG-N正确→FG-T错误594例；三者全错932例、全对80例。按hop、gold关系、cardinal/diagonal的完整迁移表已逐项校验。

## 四项解释与复现风险

**1. think边界旧flag与修正定义不一致（高置信，中等数据文档风险）。** `run_b23.py:117`把成功定义成prompt有open且生成有close；本模板open在续文生成中。因此正式FG-T原始`think_parse_success`全为False，generation_stats全0。`validate_b23.py:171–178`用(prompt_open OR generated_open) AND generated_close正确重建；总体398/2063=0.192923，其h1–h5为0.452247/0.110891/0.028436/0.021277/0.014085。`finalize_b23.py:125`仍用旧flag。本次提供 [corrected_generation_stats.csv](./corrected_generation_stats.csv)，仅校正边界率并保留legacy列。最终答案解析与主Accuracy不依赖旧flag，数值无需改写。无close时`run_b23.py:119–121`将全部续文作为final text的fallback，但本次截断例没有有效Final answer，未造成额外正确数。

**2. RNG说明有旧字段残留（高置信，可复现性风险）。** 原报告说每row child_seed；generation_config也保留`row_rng_policy`及`batch_size=1`。正式full/pilot/matched实际使用batch8及`sha256(seed:comma-separated-row-ids)`，见`run_b23.py:243–245,258–271,485–488`，其manifest及每行batch_row_ids/batch_rng_seed均一致且已全量复核。恢复会重放完整canonical批次，并比对已完成行的token/config/batch_seed（:543–565）；只补缺失行，已有shard不覆盖。**固定批次成员及顺序才成立**；改变order、batch size或独立重跑某一缺失row会改变随机数流。部分batch缺行按完整batch重放可安全续跑，改变批次定义不能沿用原seed承诺。GPU内核跨版本/硬件确定性未由此证明。

**3. 8×8混淆矩阵遗漏INVALID（高置信，展示风险）。** `common_b23.py:403–405`只用8关系labels，无效输出被排除，再按剩余row归一化。FG-N all_count实际总1905，FG-T仅398；不是2063。FG-T高准确的对角线是完成子样本表现，必须与解析率/截断并排展示；后续最好提供8×9含INVALID列矩阵。预设“prediction collapse”gate也只针对8关系频率，所以False不能掩盖FG-T derived95.63%无效这一交付失败。

**4. 当前档案与远程实验环境不等同（高置信，档案边界）。** 本地`models/Qwen3-8B`缺失，无法重解码token或重建native chat哈希；此次能够重建冻结raw user prompt、parser及统计。`validate_b23.py:21`仍固定`artifacts/stage_b1`，本地目录已迁至`results/stage-b1`；common_b23已提供fallback，但独立validator未同步。唯一环境lock的transformers=4.38.2/tokenizers=0.15.2，与B2.3audit4.57.6不同，不能当成B2.3完整锁定环境。旧阶段protected_snapshot2678文件经过映射仍1763缺失、26与旧hash不同；主要是curated档案省略checkpoint行分数、probe参数及后续文档/路径元数据编辑（SYNC_AUDIT解释该原则）。它不能证明原远程验证失效，也不能声称本地又重复通过了旧阶段全部哈希检查。B1核心union manifest及本阶段全量正式生成产物哈希保持一致。

## 统计设计实际边界

区间按story_group_id within hop有放回重采样，模式间配对，derived再等权平均；2000次、base seed20260910，各metric/scope添加哈希offset。**本测试集2063行就是2063个不同story_group_id，无跨hop组重叠**，所以在此集cluster bootstrap等价于行级paired bootstrap。区间反映固定测试集、固定一次full seed的样本不确定性，不包含模型选择、多种预算/提示、全量不同seed的随机生成方差。pilot三seed只证明250条这个子集在当前预算下的range。

不存在多重比较调整；应把冻结derived Accuracy看作主endpoint，多个hop、类别recall、F1、区间看作描述性/次要分析。没有训练/干预或activation/probe/patching，因此不能把输出改善解释成内部图形成、符号推理或忠实CoT。

## 下阶段由证据驱动的最小补充

先冻结“交付完成率、含INVALID总体正确率、完成条件正确率、token预算/耗时”四类endpoint，使用同一pilot集、匹配采样、多seed，测试2048及可承受的更高token预算；补入合适的thinking预算/强制结束思考接口时应视为新的受控协议。先做memory smoke，batch变更要记录新RNG协议，建议采用可重现的per-row随机流，避免把memory约束与模式差异混在一起。条件完成指标仅做机制诊断，总体主指标仍保留无效为错。

完成预算诊断后，再做新模型行为门控；需重新核验架构、tokenizer及thinking开关，不能复用Qwen3-8B层数/提取代码假设。只有在冻结的强门槛（derivedAcc≥0.35、Macro-F1≥0.30、BalancedAcc≥0.30、parse≥0.98、无有效关系collapse、每个多跳层高于majority、至少5类recall≥0.05）通过后，才能合理投入下一阶段几何/图表征及因果实验。门槛是研究中自行定义的准入标准，不是该领域通用能力认证。

## 绝对路径代码证据索引

- 数据、seed、prompt、parser、阈值：[common_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/common_b23.py>)：55–74、86–145、263–320、330–363、366–406。
- BF16原生template、think切分、batch生成、恢复：[run_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/run_b23.py>)：43–62、72–134、243–338、478–598、601–675、749–803。
- 冻结prompt/parser/pilot：[audit_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/audit_b23.py>)：234–288。
- Derived、bootstrap、decision及报告：[finalize_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/finalize_b23.py>)：80–129、137–227、238–338、444–549、715–775。
- 独立validator的修正边界：[validate_b23.py](<D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/validate_b23.py>)：145–224；当前硬编码B1/model路径21–22、372。
- 正式JSONL：[FG-N](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_nonthinking/row_generations.jsonl>)；[FG-T](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_thinking/row_generations.jsonl>)。
- 研究档案同步边界：[SYNC_AUDIT.md](<D:/2026本基/参考文献1/reasoning-geometry-pilot/SYNC_AUDIT.md>)；正式报告：[STAGE_B2_3_QWEN3_GENERATION_REPORT.md](<D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/STAGE_B2_3_QWEN3_GENERATION_REPORT.md>)。

