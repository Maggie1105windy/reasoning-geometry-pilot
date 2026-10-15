# Results Index

## Stage A

### A2 — Pythia-2.8B World spatial probing

- 完整正式报告：[`docs/stage-a/A2_FULL_REPRODUCTION_REPORT.md`](docs/stage-a/A2_FULL_REPRODUCTION_REPORT.md)
- 激活 sanity：[`docs/stage-a/a2_2_activation_sanity_report.md`](docs/stage-a/a2_2_activation_sanity_report.md)
- Mini probe 报告：[`docs/stage-a/a2_3_mini_probe_report.md`](docs/stage-a/a2_3_mini_probe_report.md)
- 配置与精确复现边界：[`docs/stage-a/a2_0_exact_config_audit.md`](docs/stage-a/a2_0_exact_config_audit.md)
- Gate 1 模型和环境验证：[`docs/stage-a/a2_1_gate1_report.md`](docs/stage-a/a2_1_gate1_report.md)
- 兼容性说明：[`docs/stage-a/a2_compatibility_notes.md`](docs/stage-a/a2_compatibility_notes.md)
- 32 层总 R² 图：[`figures/stage-a/a2_4_full_layer_r2.png`](figures/stage-a/a2_4_full_layer_r2.png)
- 经纬度分项图：[`figures/stage-a/a2_4_lat_lon_layer_r2.png`](figures/stage-a/a2_4_lat_lon_layer_r2.png)
- 机器可读结果：[`results/stage-a2/`](results/stage-a2/)

### A3 — Llama-3.1-8B World migration

- 正式报告：[`docs/stage-a/A3_LLAMA31_MIGRATION_REPORT.md`](docs/stage-a/A3_LLAMA31_MIGRATION_REPORT.md)
- 模型审计：[`docs/stage-a/a3_llama31_model_audit.md`](docs/stage-a/a3_llama31_model_audit.md)
- 激活 sanity：[`docs/stage-a/a3_activation_sanity_report.md`](docs/stage-a/a3_activation_sanity_report.md)
- 代表层 R²：[`figures/stage-a/a3_llama31_layer_r2.png`](figures/stage-a/a3_llama31_layer_r2.png)
- 机器可读结果：[`results/stage-a3/`](results/stage-a3/)

## Stage B

完整研究方法、实验步骤、图像数据分析与综合结论：[`docs/stage-b/STAGE_B_RESEARCH_REPORT_ZH.md`](docs/stage-b/STAGE_B_RESEARCH_REPORT_ZH.md)

适合下载、打印和提交的排版版本：[`output/pdf/STAGE_B_RESEARCH_REPORT_ZH.pdf`](output/pdf/STAGE_B_RESEARCH_REPORT_ZH.pdf)。

配套的可筛选图表、逐图证据说明与数据来源面板：[`reports/stage-b/index.html`](reports/stage-b/index.html)。下载后可在浏览器离线打开。

### B0 — StepGame data and splits

- 数据审计：[`docs/stage-b/B0_1_DATA_AUDIT_REPORT.md`](docs/stage-b/B0_1_DATA_AUDIT_REPORT.md)
- 划分可行性：[`docs/stage-b/B0_2_FEASIBILITY_REPORT.md`](docs/stage-b/B0_2_FEASIBILITY_REPORT.md)
- 冻结划分：[`docs/stage-b/B0_2_SPLIT_REPORT.md`](docs/stage-b/B0_2_SPLIT_REPORT.md)
- 模板留出：[`docs/stage-b/B0_2B_TEMPLATE_HELDOUT_REPORT.md`](docs/stage-b/B0_2B_TEMPLATE_HELDOUT_REPORT.md)
- Test-only 模板控制：[`docs/stage-b/B0_2C_TEMPLATE_TEST_REPORT.md`](docs/stage-b/B0_2C_TEMPLATE_TEST_REPORT.md)

### B1 — Base relation probing

- 正式英文技术报告：[`docs/stage-b/B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md`](docs/stage-b/B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md)
- 独立验证：[`results/stage-b1/validation_pre_report.json`](results/stage-b1/validation_pre_report.json)
- 完整聚合结果与逐层检查点表：[`results/stage-b1/`](results/stage-b1/)
- 主要峰值：h1 0.3573@L17；h2 0.1670@L19；h3 0.1702@L31；h4 0.1563@L30；h5 0.1485@L1。
- 结果图：[`figures/stage-b1/`](figures/stage-b1/)

### B1.5–B1.6 — Behavior and cross-hop transfer

- 正式英文技术报告：[`docs/stage-b/B1_5_B1_6_BEHAVIOR_TRANSFER_REPORT.md`](docs/stage-b/B1_5_B1_6_BEHAVIOR_TRANSFER_REPORT.md)
- 中文完整整合报告：[`EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md)
- 机器可读摘要：[`results/stage-b1_5-b1_6/`](results/stage-b1_5-b1_6/)
- 结果图：[`figures/stage-b1_5-b1_6/`](figures/stage-b1_5-b1_6/)

### B2 — Base/SFT/DPO/Final comparison

- 正式中文报告：[`docs/stage-b/B2_BASE_TULU_STAGE_COMPARISON_REPORT.md`](docs/stage-b/B2_BASE_TULU_STAGE_COMPARISON_REPORT.md)
- 最终验证：[`results/stage-b2/B2_final_validation.json`](results/stage-b2/B2_final_validation.json)
- 结果摘要：[`results/stage-b2/b2_summary.json`](results/stage-b2/b2_summary.json)
- 模型谱系审计：[`docs/stage-b/model_lineage_audit.md`](docs/stage-b/model_lineage_audit.md)
- 聚合结果：[`results/stage-b2/`](results/stage-b2/)
- 10 张正式图：[`figures/stage-b2/`](figures/stage-b2/)
- 逐图中文解释：[`docs/stage-b/B2_FIGURE_GUIDE_ZH.md`](docs/stage-b/B2_FIGURE_GUIDE_ZH.md)

### B2.1 — Model-task Fitness Gate

- 正式报告：[`docs/stage-b/STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md`](docs/stage-b/STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md)
- 独立验证：[`results/stage-b2_1/validation/validation_report.md`](results/stage-b2_1/validation/validation_report.md)
- 核心总表：[`results/stage-b2_1/statistics/core_summary.csv`](results/stage-b2_1/statistics/core_summary.csv)
- 配对 bootstrap：[`results/stage-b2_1/statistics/paired_bootstrap.csv`](results/stage-b2_1/statistics/paired_bootstrap.csv)
- 协议与审计：[`results/stage-b2_1/audit/`](results/stage-b2_1/audit/)
- 8 张正式图：[`figures/stage-b2_1/`](figures/stage-b2_1/)
- 关键结果：Final Raw-Sum derived mean 0.1112，Native-Sum 0.1829（增量 95% CI [0.0396, 0.1035]），Raw-Mean 0.1599，Native-Mean 0.1917，EqualLength-Native 0.1522。
- Fitness decision：接口与 token-length scoring 都明显影响原 Raw 结果，但 Final 在最公平的等长 native 协议下 h2–h5 仍未稳定高于 0.125。建议保留当前模型作 weak-reasoning control，新增 strong-behavior model，不自动进入因果干预。

### B2.2—B2.3 — Qwen行为、思考预算与可见轨迹

- 最新图文综合报告：[`docs/stage-b/STAGE_B2_3_RESEARCH_SUMMARY_ZH.md`](docs/stage-b/STAGE_B2_3_RESEARCH_SUMMARY_ZH.md)。
- 同步范围、逐文件SHA与压缩还原：[`results/stage-b2_3_sync/README.md`](results/stage-b2_3_sync/README.md)。
- B2.2参照：[`artifacts/stage_b2_2_qwen3/`](artifacts/stage_b2_2_qwen3/)。
- 原2063题自由生成：[`artifacts/stage_b2_3_qwen3_generation/`](artifacts/stage_b2_3_qwen3_generation/)。
- 10题预算：[`artifacts/stage_b2_3_budget_pilot_20261013/`](artifacts/stage_b2_3_budget_pilot_20261013/)。
- 100题预算：[`artifacts/stage_b2_3_budget100_20261014/`](artifacts/stage_b2_3_budget100_20261014/)。
- 轨迹高频：[`analysis_high_frequency_20261015/`](artifacts/stage_b2_3_budget100_20261014/analysis_high_frequency_20261015/)。
- 八张PNG／SVG及图表数据：[`artifacts/stage_b2_3_synthesis_20261015/`](artifacts/stage_b2_3_synthesis_20261015/)。
- 非思考多步准确率25.65%，原生候选评分18.53%；原2K思考多步3.52%受到严重截断，不能解释为充分预算的thinking纯效应。
- 新100题三档正确59／64／64、解析77／96／96。单seed等hop预算诊断，非自然总体估计，也不是选预算后的独立能力确认。
- 本阶段没有完成生成轨迹隐藏状态提取、token级Δlog p或因果激活机制验证。

### Preliminary — MATH-500 pilot-20

- 四阶段结果与人工复核材料：[`preliminary/math500/`](preliminary/math500/)
- 人工复核准确率：Base 25%、SFT 20%、DPO 20%、Final/RLVR 35%。由于每阶段只有 20 题，不用于模型能力排名或正式 benchmark 推断。

## Interpretation boundary

线性 probe 结果只说明信息可以从冻结隐藏状态中线性解码，不证明模型在生成答案时实际使用该信息。峰值层、L50 和 L90 也不能被解释为推理操作发生的时间或因果层。
