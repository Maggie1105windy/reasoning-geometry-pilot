# Project Status

更新时间：2026-09-01

## 已完成并可作为正式结果引用

- **Stage A2**：Pythia-2.8B × World 空间坐标探针。云端正式范围包含 Gate 0–4 和 32 层全量复现；当前仓库保留配置审计、Gate、兼容性说明、执行脚本和正式图表。
- **Stage A3**：Llama-3.1-8B Base 的 World 任务迁移审计、激活 sanity check 和 5,000 样本×5 层 mini probe。该阶段按批准范围完成，但不是全数据×32 层实验。
- **Stage B0**：StepGame 数据、模板、实际 hop、重复/泄漏、Primary/Balanced/Target-Absent/Composition/Template 划分完成并冻结。
- **Stage B1**：Llama-3.1-8B Base 的 32 层显式/派生关系探针，38/38 独立检查通过。
- **Stage B1.5–B1.6**：候选行为基线与跨 hop transfer，联合验证通过。

## 尚未作为正式结果发布

- **Stage B2**：目录中仅保存计划和实现脚本。除非后续加入正式报告、验证文件和冻结结果，否则不得表述为已完成 Base–post-training 比较。

## 本仓库有意不包含

- Hugging Face/ModelScope 模型权重与缓存；
- Stage A/B 完整 activation tensors；
- Python/Conda 虚拟环境与安装缓存；
- 原始 StepGame 数据副本；
- 论文 PDF；
- SSH key、API token、凭据和个人目录文件；
- 服务器临时日志和中间 checkpoint。

## 待补强项

1. 从云服务器恢复并核对 Stage A2 的最终独立 Markdown 报告及机器可读 Gate 4 结果包。
2. 在所有阶段完成同步后生成统一 `FILE_INVENTORY.csv` 和 SHA-256 发布清单。
3. 为脚本增加统一的可配置项目根目录，替代历史服务器绝对路径。
4. 选择开源许可证后再将仓库设为公开。

