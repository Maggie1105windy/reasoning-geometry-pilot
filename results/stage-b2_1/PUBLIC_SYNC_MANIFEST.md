# Stage B2.1 public sync manifest

同步日期：2026-09-10。

本目录保存适合 Git 的完整研究层：冻结协议、输入与 tokenizer/chat-template 审计、运行元数据、Gate A、聚合 hop 指标、bootstrap CI、逐类/基数–对角诊断、混淆矩阵、统计比较和独立验证。

以下大体积或冗余运行层没有放入 Git：

- `raw_reproduction/row_scores.csv`
- `native_chat/per_model/`
- `native_chat/row_scores_sum.csv`
- `native_chat/row_scores_mean.csv`
- `equal_length_verbalizer/per_model/`
- `equal_length_verbalizer/raw_row_scores.csv`
- `equal_length_verbalizer/native_row_scores.csv`
- `logs/`

这些文件仍保留在云端 `/home/ai/chenchen/projects/reasoning-geometry-pilot/artifacts/stage_b2_1/`。Git 研究层可以通过 `audit/run_metadata_*.json`、原始来源 SHA-256、协议 SHA-256 和独立验证报告追溯完整运行。
