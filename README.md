# Reasoning Geometry Pilot

一个用于研究大型语言模型内部空间表征、显式关系编码、多步推理和后训练影响的可复核实验仓库。

本仓库保存项目的公开研究层：实验方法、冻结协议、结果报告、图表、CSV/JSON 摘要、验证文件和复现脚本。模型权重、完整激活矩阵、缓存、虚拟环境、原始论文 PDF 和凭据不进入仓库。

## 研究问题

1. 语言模型的中间层是否形成可线性读取的空间表示？
2. 显式一步关系与多步派生关系在层间如何分布？
3. hop 增加是否对应更晚的可解码层？
4. 线性可解码信息是否与模型自身的答案行为一致？
5. 显式关系解码器能否跨 hop 迁移，以及后训练是否改变这种结构？

## 当前阶段

| 阶段 | 模型与任务 | 范围 | 状态 |
|---|---|---|---|
| A2 | Pythia-2.8B × World coordinates | 32 层空间坐标线性探针复现 | 完成 |
| A3 | Llama-3.1-8B Base × World coordinates | 模型迁移审计、激活 sanity、5 层 mini probe | 完成（既定范围） |
| B0 | StepGame | 数据、模板、hop、泄漏与冻结划分审计 | 完成 |
| B1 | Llama-3.1-8B Base × StepGame | 32 层显式/派生关系探针与控制实验 | PASS，38/38 checks |
| B1.5 | Llama-3.1-8B Base × StepGame | 候选答案条件似然行为基线 | PASS |
| B1.6 | Llama-3.1-8B Base × StepGame | 跨 hop 线性迁移 | PASS |
| B2 | Base 与后训练模型比较 | 脚本准备阶段；本仓库不提供已完成结果声明 | PLANNED |

详细完成边界见 [STATUS.md](STATUS.md)。

## 核心结果

- B1 中，显式 h1 的峰值 Macro-F1 为 **0.3573@L17**；h2–h5 峰值仅为 **0.1485–0.1702**。
- h2–h5 没有呈现“hop 越大、可解码层越晚”的单调顺序。
- 平衡控制使 h1 峰值降至 **0.2163**；未见模板测试使其降至 **0.1794**，说明强显式读出对数据分布和表面模板敏感。
- B1.5 主候选评分下，行为准确率从 h1 的 **0.1517** 下降到 h5 的 **0.0915**；派生 hop 未建立稳健的高于随机能力。
- B1.6 支持有限、较弱的跨 hop 线性迁移，但不支持统一符号关系代码或可定位的“推理层”。

完整结果索引见 [RESULTS.md](RESULTS.md)，中文技术报告见 [EXPERIMENT_REPORT_ZH.md](EXPERIMENT_REPORT_ZH.md)。

## 代表性图表

### B1：逐层关系可解码性

![Primary relation probe](figures/stage-b1/figure1_primary_macro_f1.png)

### B1.5：行为与探针回答不同问题

![Behavior versus probe](figures/stage-b1_5-b1_6/figure_2_behavior_vs_probe.png)

### B1.6：跨 hop 迁移差值

![Transfer gap heatmap](figures/stage-b1_5-b1_6/figure_5_transfer_gap_heatmap.png)

## 仓库结构

```text
.
├── README.md
├── STATUS.md
├── RESULTS.md
├── EXPERIMENT_REPORT_ZH.md
├── CITATIONS.md
├── docs/
│   ├── stage-a/
│   └── stage-b/
├── figures/
│   ├── stage-a/
│   ├── stage-b1/
│   └── stage-b1_5-b1_6/
├── results/
│   ├── stage-b0/
│   └── stage-b1_5-b1_6/
├── scripts/
│   ├── stage-a/
│   ├── stage-b0/
│   ├── stage-b1/
│   ├── stage-b1_5-b1_6/
│   └── stage-b2-planned/
└── environment/
```

## 复现边界

- Stage A2 的 Pythia revision、权重 SHA-256、运行环境和 Gate 记录位于 `docs/stage-a/`。
- Stage B 使用 StepGame 官方仓库冻结 commit `5e6aff1563a6d7f46ee1f1aeff98b94e68c29005`。
- Llama-3.1-8B Base 模型 revision 为 `24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea`，tokenizer revision 为 `c52792af35fe806a76530f3028f037d1af4e3490`。
- 历史执行脚本保留了服务器项目布局，用于审计已执行流程；在新环境运行前应把项目根目录改为可配置路径。
- 大型二进制资产不在 Git 中。报告通过 shape、dtype、row order、checksum 和重前向验证记录其完整性。

## 引用与数据来源

论文、数据集和方法来源见 [CITATIONS.md](CITATIONS.md)。仓库不重新分发论文 PDF。

## 许可

当前尚未选择开源许可证。在添加明确许可证前，仓库内容默认不授予再分发或衍生使用许可。准备公开发布前应根据项目归属选择合适许可证。

