# B1–B2.3 阶段证据核查（2026-10-15）

本次仅只读核查正式报告、CSV、JSON、脚本和全量 FG-N/FG-T JSONL；未运行模型，未改动原始实验。下列百分数与百分点分开使用。

## 1. B2.2–B2.3 核心指标

主指标 `h2–h5-derived` 是先计算每个 hop，再对四个 hop 等权平均；标记的 n=1351 是样本总数，**不是简单正确数/1351**。

| 协议 | h1 Accuracy | h2 | h3 | h4 | h5 | h2–h5等权 Accuracy | Macro-F1 | Balanced Accuracy | 等权答案解析率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2.2 Native-Sum | 23.0337% | 17.6238% | 15.8768% | 17.3759% | 23.2394% | 18.5290% | 12.3613% | 15.7406% | 100%（封闭候选） |
| FG-N，2K | 70.2247% | 33.0693% | 23.9336% | 23.0496% | 22.5352% | 25.6470% | 25.6518% | 25.1349% | 92.4790% |
| FG-T，2K | 43.8202% | 9.1089% | 1.4218% | 2.1277% | 1.4085% | 3.5167% | 6.1740% | 3.3957% | 4.3672% |

源（绝对路径）：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/statistics/full_metrics_by_hop.csv`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/statistics/derived_summary.csv`

| Accuracy 配对差异 | 点估计（pp） | 95% bootstrap CI（pp） |
|---|---:|---:|
| FG-N − B2.2 Native-Sum | +7.11799 | [+3.85437,+10.55492] |
| FG-T − FG-N | −22.13025 | [−24.86847,−19.51413] |
| Matched FG-T − Matched FG-N | −26.0 | [−32.5,−19.5] |

Matched 子集为 250 题，五 hop 各 50；该 derived 对比实际覆盖 h2–h5 的 200 题。FG-N=29.5%，FG-T=3.5%。两者 T=0.6、top-p=0.95、top-k=20、min-p=0；主实验 FG-N T=0.7/top-p=0.8，FG-T T=0.6/top-p=0.95。

Bootstrap：2000 次，seed=20260910（各 metric/scope 加哈希 offset），hop 内配对 story_group 重采样，derived 再等权。当前 2063 行为 2063 个不同 story_group_id，因此此处 cluster bootstrap 等价于按行配对重采样，未增加模板层保护或生成 seed 方差覆盖。

源：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/statistics/paired_bootstrap.csv`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/statistics/matched_decoding_metrics.csv`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/fitness_decision.json`

## 2. 从原始 JSONL 直接重算的交付结果

每模式同一 2063 题，h1–h5=712/505/422/282/142。

| 模式 | 正确 | EOS | 达上限 | INVALID | EOS但格式无效 | 输出token总数 |
|---|---:|---:|---:|---:|---:|---:|
| FG-N | 865 | 2055 | 8 | 158 | 150 | 719187 |
| FG-T | 372 | 398 | 1665 | 1665 | 0 | 3967696 |

FG-T 达上限率=80.7077%；全部 INVALID 都是无 EOS 的截断，398 条 EOS 都能解析。FG-N主要无效原因包含格式遵循，不能把两模式 INVALID 全归于同一原因。

| hop | FG-N 正确/解析/截断 | FG-T 正确/解析/截断 |
|---|---:|---:|
| h1 | 500/660/0 | 312/322/390 |
| h2 | 167/473/3 | 46/56/449 |
| h3 | 101/380/2 | 6/12/410 |
| h4 | 65/257/2 | 6/6/276 |
| h5 | 32/135/1 | 2/2/140 |

所有 FG-T hop 的生成 token 中位数 2048，thinking token 中位数约2047。正式 `generation_stats.csv` 的 legacy think_parse_success 全0有字段兼容错误；用 `(prompt_open OR generated_open) AND generated_close` 重算为398/2063=19.2923%。最终答案 accuracy 不依赖旧flag。对应修正表：

`D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/analysis_20261013/corrected_generation_stats.csv`

原始 JSONL：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_nonthinking/row_generations.jsonl`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_thinking/row_generations.jsonl`

## 3. 冻结门槛与原判定

强对象门槛：derived Accuracy≥35%、Macro-F1≥30%、Balanced Accuracy≥30%、答案解析率≥98%，无预测塌缩，每个多跳层高于多数类，至少5类 recall≥5%。这些是项目自定义准入标准。

候选评分低估行为的 YES 条件包含至少+10pp accuracy、解析率≥98%等。FG-N的+7.12pp虽然CI排除0，仍未达实际意义门槛；不可将“显著”写成“门控通过”。

原冻结判定：候选评分低估 `UNCERTAIN`；thinking改善 `NO`；任务适配性 `WEAK`；强推理对象 `UNCERTAIN`；分支C（改模型前先行为门控）。数值判定应作为2K条件历史结果保留。报告当前科学解释必须结合100题预算校准，不应继续把严重2K删失当thinking一般能力结论。

## 4. 实际 batch/RNG 与解析协议

正式两模式 batch_size=8，末批7；每模式2056条属于满批，7条属于末批。RNG为固定排序 canonical batches，batch_seed=SHA256(`generation_seed:comma-separated-row-ids`)首8位hex取模2^31−1。主生成seed20260924。

旧 generation_config/report 的 batch1 / row child seed 是陈旧说明；复现应以实际 run_manifest、行级 batch_row_ids/batch_rng_seed 和脚本批运行分支为准。改变 batch 后不承诺逐题轨迹相同。新预算实验 batch1 的同题 row seed 协议与旧2K不同，因此两轮不能作严格逐题因果配对。

源：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_thinking/run_manifest.json`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_3_qwen3_generation/full/fg_nonthinking/run_manifest.json`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/scripts/stage-b2_3/run_b23.py`，243–271、478–598行附近。

原解析器取最后一条有效 `Final answer:` 行，不要求物理最后一行匹配；8类规范关系，无有效行INVALID且计错。原8×8混淆矩阵排除INVALID，FG-T仅398样本入矩阵。新报告应重做8×9含INVALID，或清楚标为完成子集并并列交付率。

## 5. B2.2 协议敏感性与 Tülu 比较

Qwen3 h2–h5：Raw-Sum10.6067%、Native-Sum18.5290%、Native-Mean19.8704%、EqualLength-Native10.0598%。Tülu Final Raw-Sum11.1197%、Native-Sum18.2879%。Qwen−Tülu Native-Sum仅+0.2411pp，CI[−2.5025,+2.9617]pp，不能说Qwen在候选协议中明显更强。Qwen Raw-Sum预测100%cardinal且82.55%为BELOW，Native-Sum无gate定义的塌缩；Native-Mean存在塌缩。不得只用最高accuracy抹平偏置。

源：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_2_qwen3/statistics/derived_summary.csv`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/artifacts/stage_b2_2_qwen3/fitness_decision.json`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/results/stage-b2_1/statistics/core_summary.csv`

## 6. 前置B1/B2应保留的成果与边界

B1：Llama3.1-8B Base question-end relation probes，h1 peak Macro-F1=.3573@L17；h2=.1670@L19、h3=.1702@L31、h4=.1563@L30、h5=.1485@L1。无单调hop→更晚层顺序。h1 reserved-template peak降至.1794；balanced peak .2163，提示读出受表面模板与分布影响。随机标签accuracy=.1285，Macro-F1=.1247，正常matched=.1790。峰值层为描述性test曲线摘要，不能作能力诞生时间或推理层定位。

B2：Base/SFT/DPO/Final来自同Llama/Tülu谱系；Final的本地README标RLVR，但正式结果仍用保守Final。Raw-Sum derived行为=11.3368/10.0770/11.1980/11.1197%，无单调改善；后续B2.1 Native-Sum改善后仍弱，不能只引用Raw-Sum断言整谱系无能力。

晚层L16–31 derived within-hop Macro-F1=13.1859/13.9535/14.3535/13.9953%；Final−Base +.0081，各h2–h5 CI均含0。h1 late Mean=32.1781/39.0728/40.4509/39.4752%，SFT后h1改善更明确。h1→derived transfer mean=13.2485/13.7769/14.2130/14.5669%（从下列b2_summary各hop求均值）；整体是弱、混合证据，不能宣布共享几何增强或神经符号机制。

源：

- `D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/results/stage-b1/stage_b1_gate.json`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/B2_BASE_TULU_STAGE_COMPARISON_REPORT.md`
- `D:/2026本基/参考文献1/reasoning-geometry-pilot/results/stage-b2/b2_summary.json`

## 7. 本次综合报告需要更新的旧表述

1. “下一步预算校准”已完成10题和100题，本报告应将其作为阶段成果；下一步转向过程标注、可用状态与少量因果试验。
2. “16K足够”仅旧10题成立；新100题16K仍4例截断，32K同4例截断且无准确/解析增益。
3. 旧10题32K均提前EOS，不能验证满32K显存；新100题确实有4条32768输出，能给batch1短输入条件的实测边界。
4. 2K FG-T低分是固定预算交付表现；新的高预算说明部分预算删失可解除，但仍有错误推理和循环，不可把更新写成充分预算行为已全面可靠。
5. “可读出关系”与“最终正确性信号”“生成中关系状态”“实际使用”“后训练创造能力”五层证据须拆开。
6. 当前Qwen thinking/nonthinking为同后训练checkpoint，不能作为Base/SFT/DPO/RLVR对比。
7. 原2K统计显著改善、100题预算诊断、代表性过程审计、机制假设使用不同证据层级，不应混成一个确认性结论。

