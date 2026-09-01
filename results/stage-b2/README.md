# Stage B2 machine-readable results

Stage B2 比较 Base、SFT、DPO 和 Final 四个 checkpoint。`B2_final_validation.json` 的状态为 PASS；本目录包含模型谱系与冻结输入审计、行为结果、within-hop/cross-hop 聚合表、bootstrap、控制实验和完整 artifact checksum。

为控制 Git 体积，本目录不包含 `.npz` probe 参数、逐层 checkpoint 目录和大体积逐样本 prediction/candidate-score 表。其完整副本保留在服务器和本地恢复暂存区，`artifact_checksums.json` 可用于核验。
