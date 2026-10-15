# Stage B2.3 文件与结果归档

日期：2026-10-15。范围覆盖已完成的B2.3主实验、10题／100题预算实验、可见思考轨迹分析和综合报告，并补齐B2.2对照依赖。正在运行的B3没有纳入。

## 阅读入口

- [最新中文综合报告与8张重点图](../../docs/stage-b/STAGE_B2_3_RESEARCH_SUMMARY_ZH.md)
- [原始2063题两模式生成](../../artifacts/stage_b2_3_qwen3_generation/)
- [100题三预算结果](../../artifacts/stage_b2_3_budget100_20261014/)
- [可见轨迹高频与循环分析](../../artifacts/stage_b2_3_budget100_20261014/analysis_high_frequency_20261015/)
- [综合图表及其数据](../../artifacts/stage_b2_3_synthesis_20261015/)
- [原生生成代码](../../scripts/stage-b2_3/)、[100题实验与独立审计代码](../../scripts/stage-b2_3_budget100_20261014/)
- [自包含交互式报告](../../reports/stage-b/index.html)：下载后打开，构建已核验，网页响应式视觉检查未完成。

## 保存了什么

保留完整逐题生成shards、token IDs、提示、模型／样本／配置身份hash、合并结果、各hop指标、配对区间、混淆矩阵、预算变化、遥测、审计、图表及相关运行日志。CPU审计目录的tokenizer和配置文件用于不加载模型的复核，不含模型权重。统计结果未因同步而重算或改写。

`SYNC_MANIFEST.csv`逐文件列出原始与发布路径、字节数、SHA-256及存储方式。`EXCLUDED_FILES.csv`记录排除的Python缓存、临时渲染页、本地自动化／进程辅助信息及不属于实验结果的内部申报书／研究草稿源文。正式报告图与最终PDF保留；不上传权重、完整激活、凭据或新B3文件。

## 还原62MB合并原始结果

`full/fg_thinking/row_generations.jsonl`以同目录的`.jsonl.gz`无损保存；2063个对应原始shards仍可直接查看。完整解压后SHA已与本地原文件核对一致。

```sh
python scripts/stage-b2_3/restore_archived_results.py --verify-only
python scripts/stage-b2_3/restore_archived_results.py
```

还原脚本只创建缺失的原路径，遇到内容不同的已有文件即停止，不覆盖数据。精确档案信息见`RAW_ARCHIVES.json`。作用域限定的`.gitattributes`保留实验文件的原始换行字节，避免Windows自动转换破坏审计hash。

## 如何解释这些结果

- 原2K思考实验80.71%截断，不能据其低分断言思考有害。
- 新100题三档正确59／64／64、解析77／96／96；单seed、等hop诊断样本，不代表自然测试总体准确率或普遍最佳预算。
- 最终正确、可见过程有效与隐藏状态的因果使用是不同证据。本次没有完成后两者的完整机制确认，也不能用同权重模式比较归因后训练。
- 原2K实际batch8、尾批7，以实际运行manifest和batch RNG为准；旧配置中batch1／per-row说明是历史不一致。本轮预算实验为batch1、row派生seed，不能与旧2K作严格因果配对。

原始报告保留历史绝对来源路径；建议从本页和新版综合报告使用GitHub相对链接阅读。服务器启动笔记为历史运行快照，不是可直接复用的当前连接配置。
