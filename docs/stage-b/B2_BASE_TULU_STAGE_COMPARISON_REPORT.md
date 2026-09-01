# Stage B2 完成

## 1. Model lineage

- Base: revision `24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea`, 8,030,261,248 parameters, 32×4096, vocab(config/tokenizer)=128256/128256
- SFT: revision `aa9a78450ea2c6916824501e276cb97b7caf47e6`, 8,030,326,784 parameters, 32×4096, vocab(config/tokenizer)=128264/128257
- DPO: revision `2301bf3b5e8733aad822a59b93381369da1692eb`, 8,030,326,784 parameters, 32×4096, vocab(config/tokenizer)=128264/128257
- Final: revision `c5d2996f1e01989a69bd740d8d034edf672cc1e0`, 8,030,326,784 parameters, 32×4096, vocab(config/tokenizer)=128264/128257
- architecture/tokenizer comparability: 四个 checkpoint 均为 32 个 Llama decoder blocks、hidden size 4096；120 条冻结 raw prompt 在 5 hops×8 relations 上的 token IDs、BOS、序列长度、readout index 和末尾 `?` 完全一致。Tokenizer 文件/revision 不同，但实际冻结 prompt tokenization 一致。
- status: PASS。Final 在本地 README 表格中被明确标为 Final Models (RLVR)，正式结果仍使用保守名称 Final。Revision 来自 ModelScope `.msc` 文件级本地记录，不宣称为 Hugging Face commit。

## 2. Behavioral accuracy

| Checkpoint | h1 | h2 | h3 | h4 | h5 | Derived mean |
|---|---|---|---|---|---|---|
| Base | 0.1517 | 0.1465 | 0.1090 | 0.1064 | 0.0915 | 0.1134 |
| SFT | 0.1770 | 0.1267 | 0.1280 | 0.0709 | 0.0775 | 0.1008 |
| DPO | 0.1854 | 0.1267 | 0.1374 | 0.0922 | 0.0915 | 0.1120 |
| Final | 0.1826 | 0.1188 | 0.1422 | 0.0993 | 0.0845 | 0.1112 |

- Base→SFT: derived mean -0.0126
- SFT→DPO: derived mean +0.0112
- DPO→Final: derived mean -0.0008
- Final→Base: derived mean -0.0022
- 哪个 stage 对 derived behavior 改变最大: Base→SFT 的绝对变化最大，但方向是降低；SFT→DPO 大致恢复该降幅。各 hop 的 paired CI 不支持统一的 derived 行为提升。

## 3. Within-hop decodability

Late-layer mean (L16–31):

| Checkpoint | h1 | h2 | h3 | h4 | h5 | Derived mean |
|---|---|---|---|---|---|---|
| Base | 0.3218 | 0.1507 | 0.1437 | 0.1168 | 0.1162 | 0.1319 |
| SFT | 0.3907 | 0.1524 | 0.1416 | 0.1440 | 0.1201 | 0.1395 |
| DPO | 0.4045 | 0.1565 | 0.1355 | 0.1574 | 0.1247 | 0.1435 |
| Final | 0.3948 | 0.1611 | 0.1308 | 0.1490 | 0.1190 | 0.1400 |

- h1: Base→SFT 增加 +0.0689，Final→Base 增加 +0.0730；两者 paired 95% CI 排除 0。
- h2: Final→Base +0.0104
- h3: Final→Base -0.0129
- h4: Final→Base +0.0321
- h5: Final→Base +0.0028
- late-layer mean by checkpoint: derived mean Base=0.1319, SFT=0.1395, DPO=0.1435, Final=0.1400
- stagewise delta: Base→SFT=+0.0077, SFT→DPO=+0.0040, DPO→Final=-0.0036, Final→Base=+0.0081
- 是否出现稳定 derived enhancement: NO；四个 derived hop 的 late-layer paired CI 均包含 0，h3 与 h4/h2 方向不一致。Peak/all-layer/late-layer 均已保留在 `within_hop/summaries.csv`，不以单层 peak 作为主证据。

## 4. Cross-hop transfer

Late-layer h1-trained transfer:

| Checkpoint | h1→h2 | h1→h3 | h1→h4 | h1→h5 | Derived mean |
|---|---|---|---|---|---|
| Base | 0.1253 | 0.1345 | 0.1441 | 0.1260 | 0.1325 |
| SFT | 0.1344 | 0.1509 | 0.1444 | 0.1214 | 0.1378 |
| DPO | 0.1398 | 0.1454 | 0.1459 | 0.1374 | 0.1421 |
| Final | 0.1423 | 0.1494 | 0.1498 | 0.1412 | 0.1457 |

Late-layer TransferGap = within-hop − h1-transfer:

| Checkpoint | h2 | h3 | h4 | h5 |
|---|---|---|---|---|
| Base | 0.0253 | 0.0092 | -0.0272 | -0.0098 |
| SFT | 0.0180 | -0.0093 | -0.0004 | -0.0012 |
| DPO | 0.0167 | -0.0099 | 0.0115 | -0.0127 |
| Final | 0.0188 | -0.0186 | -0.0008 | -0.0222 |

- Base: derived transfer mean 0.1325
- SFT: derived transfer mean 0.1378
- DPO: derived transfer mean 0.1421
- Final: derived transfer mean 0.1457
- h1→h2: Base=0.1253, Final=0.1423
- h1→h3: Base=0.1345, Final=0.1494
- h1→h4: Base=0.1441, Final=0.1498
- h1→h5: Base=0.1260, Final=0.1412
- TransferGap: 对 hop 和 checkpoint 呈混合符号，且多个 within-hop derived 分数仅略高于 chance，retention ratio 不作强解释。
- 是否有 evidence for stronger explicit-derived shared linear geometry: MIXED/弱证据。Final 相对 Base 的 derived transfer 平均增加 +0.0132，但 gap 未在各 hop 统一缩小，且 within-hop 提升不稳定。

## 5. Balanced control

- 是否复现主要 stagewise trend: 部分复现但更噪声化。Derived late mean: Base=0.1519, SFT=0.1286, DPO=0.1280, Final=0.1250。Balanced 只作小样本 robustness control，不以 peak 排序作主结论。

## 6. Target-Absent

- 是否复现 derived improvement: 未复现一致的阶段性提升。Primary probe 直接评价 Target-Absent，未重训练；derived late mean: Base=0.1288, SFT=0.1397, DPO=0.1404, Final=0.1369。

## 7. Composition-heldout

- 是否复现 derived improvement: 未出现跨 h2–h5 稳定的提升。Frozen unseen unordered composition signatures 的 derived late mean: Base=0.1175, SFT=0.1357, DPO=0.1389, Final=0.1398。

## 8. Template Test-only

- h1: late mean Base=0.1609, SFT=0.1850, DPO=0.1870, Final=0.1923
- derived: h2–h5 late mean Base=0.1282, SFT=0.1330, DPO=0.1394, Final=0.1411
- post-training 是否降低 template sensitivity: MIXED。这里只能解释为 reserved Mid-level template families 在 probe train/validation 中未见，不能扩展为 all reasoning-path templates unseen。

## 9. Random-label

- Base/SFT/DPO/Final: Base accuracy=0.1285, Macro-F1=0.1247, matched-normal drop=0.0543; SFT accuracy=0.1254, Macro-F1=0.1209, matched-normal drop=0.0689; DPO accuracy=0.1274, Macro-F1=0.1218, matched-normal drop=0.0788; Final accuracy=0.1290, Macro-F1=0.1253, matched-normal drop=0.0720
- status: PASS。所有 mean accuracy 均在 0.125±0.04，且 Macro-F1 均比 matched normal-label probe 低 >0.02。

## 10. 三个 Primary answers

### A. Post-training 是否提高 derived behavior

- MIXED
- evidence: derived mean accuracy Base=0.1134, SFT=0.1008, DPO=0.1120, Final=0.1112；无单调改善，而且 paired hop-level CI 不支持稳定提升。

### B. Post-training 是否提高 derived linear decodability

- MIXED
- evidence: derived late mean 有小幅变化，Final−Base=+0.0081，但 hop 方向不一致，所有 h2–h5 late-summary paired 95% CI 包含 0。

### C. Post-training 是否提高 explicit→derived linear geometry sharing

- MIXED
- evidence: derived transfer mean Base=0.1325 到 Final=0.1457，但 within-hop 没有相应稳定增强，TransferGap 也未各 hop 同向缩小。

## 11. 哪个 post-training stage 变化最大

- behavior: Base→SFT 绝对变化最大（derived mean -0.0126，为降低）。
- decodability: Base→SFT 对 h1 的增幅最明确；derived 平均的最大连续阶段变化是 Base→SFT=+0.0077，但无稳定 hop-level CI 支持。
- transfer: 连续阶段 derived mean 变化为 Base→SFT=+0.0053, SFT→DPO=+0.0044, DPO→Final=+0.0035, Final→Base=+0.0132；变化小且分散。

## 12. 当前能支持的结论

- Post-training 与 question-end 线性 relation geometry 的改变相关，其中 h1 explicit relation 的 late-layer decodability 在 SFT 后明显增强。
- Derived-hop 行为、within-hop decodability 和 h1→derived transfer 的变化较小且不统一；不存在一条简单的 Base→SFT→DPO→Final 单调提升曲线。
- Full-layer curves、late mean、all-layer mean 和 descriptive peak 均已保留；主解释不依赖孤立 peak。

## 13. 当前不能支持的结论

- 不能声称 post-training proves grounding、Final/RLVR creates symbolic reasoning、某层是 reasoning layer、probe 改善导致行为改善，或 transfer 证明 symbolic code。
- 不能将弱而混合的 derived 曲线解释为已建立的 explicit-derived 共享机制。
- 不能把 ModelScope `.msc` revision 当作 Hugging Face commit SHA，也不能忽略 Tulu 的 8-entry config vocab padding 差异。

## 14. Stage B2 status

- PASS

## 15. 是否建议进入因果验证

- NO（暂不自动进入）
- reason: 三个 Primary 对 derived relation 都呈混合/弱效应，应先人工审阅 B2 曲线、paired CI 和 controls，再决定是否对 h1 的稳定变化或特定 derived hop 定义因果问题。

## 16. 输出

- report: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/B2_BASE_TULU_STAGE_COMPARISON_REPORT.md`
- behavior: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/behavior`
- probes: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/within_hop`
- transfer: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/cross_hop`
- controls: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/controls`
- figures: `/home/ai/projects/reasoning-geometry-pilot/artifacts/stage_b2/figures`
- activations: `/home/ai/projects/reasoning-geometry-pilot/activations/stage_b2/{tulu_sft,tulu_dpo,tulu_final}`
