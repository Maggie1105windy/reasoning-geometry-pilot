# Results Index

## Stage A

### A2 — Pythia-2.8B World spatial probing

- 配置与精确复现边界：[`docs/stage-a/a2_0_exact_config_audit.md`](docs/stage-a/a2_0_exact_config_audit.md)
- Gate 1 模型和环境验证：[`docs/stage-a/a2_1_gate1_report.md`](docs/stage-a/a2_1_gate1_report.md)
- 兼容性说明：[`docs/stage-a/a2_compatibility_notes.md`](docs/stage-a/a2_compatibility_notes.md)
- 32 层总 R² 图：[`figures/stage-a/a2_4_full_layer_r2.png`](figures/stage-a/a2_4_full_layer_r2.png)
- 经纬度分项图：[`figures/stage-a/a2_4_lat_lon_layer_r2.png`](figures/stage-a/a2_4_lat_lon_layer_r2.png)

### A3 — Llama-3.1-8B World migration

- 正式报告：[`docs/stage-a/A3_LLAMA31_MIGRATION_REPORT.md`](docs/stage-a/A3_LLAMA31_MIGRATION_REPORT.md)
- 模型审计：[`docs/stage-a/a3_llama31_model_audit.md`](docs/stage-a/a3_llama31_model_audit.md)
- 代表层 R²：[`figures/stage-a/a3_llama31_layer_r2.png`](figures/stage-a/a3_llama31_layer_r2.png)

## Stage B

### B0 — StepGame data and splits

- 数据审计：[`docs/stage-b/B0_1_DATA_AUDIT_REPORT.md`](docs/stage-b/B0_1_DATA_AUDIT_REPORT.md)
- 划分可行性：[`docs/stage-b/B0_2_FEASIBILITY_REPORT.md`](docs/stage-b/B0_2_FEASIBILITY_REPORT.md)
- 冻结划分：[`docs/stage-b/B0_2_SPLIT_REPORT.md`](docs/stage-b/B0_2_SPLIT_REPORT.md)
- 模板留出：[`docs/stage-b/B0_2B_TEMPLATE_HELDOUT_REPORT.md`](docs/stage-b/B0_2B_TEMPLATE_HELDOUT_REPORT.md)
- Test-only 模板控制：[`docs/stage-b/B0_2C_TEMPLATE_TEST_REPORT.md`](docs/stage-b/B0_2C_TEMPLATE_TEST_REPORT.md)

### B1 — Base relation probing

- 正式英文技术报告：[`docs/stage-b/B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md`](docs/stage-b/B1_LLAMA31_BASE_RELATION_PROBE_REPORT.md)
- 主要峰值：h1 0.3573@L17；h2 0.1670@L19；h3 0.1702@L31；h4 0.1563@L30；h5 0.1485@L1。
- 结果图：[`figures/stage-b1/`](figures/stage-b1/)

### B1.5–B1.6 — Behavior and cross-hop transfer

- 正式英文技术报告：[`docs/stage-b/B1_5_B1_6_BEHAVIOR_TRANSFER_REPORT.md`](docs/stage-b/B1_5_B1_6_BEHAVIOR_TRANSFER_REPORT.md)
- 中文完整整合报告：[`EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md)
- 机器可读摘要：[`results/stage-b1_5-b1_6/`](results/stage-b1_5-b1_6/)
- 结果图：[`figures/stage-b1_5-b1_6/`](figures/stage-b1_5-b1_6/)

## Interpretation boundary

线性 probe 结果只说明信息可以从冻结隐藏状态中线性解码，不证明模型在生成答案时实际使用该信息。峰值层、L50 和 L90 也不能被解释为推理操作发生的时间或因果层。

