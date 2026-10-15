# Project Status

更新时间：2026-10-15

## 已完成并可作为正式结果引用

- **Stage A2**：Pythia-2.8B × World 空间坐标探针。Gate 0–4、32 层结果、独立正式报告、机器可读表格、控制实验和完整性记录均已恢复入库。
- **Stage A3**：Llama-3.1-8B Base 的 World 任务迁移审计、激活 sanity check 和 5,000 样本×5 层 mini probe。该阶段按批准范围完成，但不是全数据×32 层实验。
- **Stage B0**：StepGame 数据、模板、实际 hop、重复/泄漏、Primary/Balanced/Target-Absent/Composition/Template 划分完成并冻结。
- **Stage B1**：Llama-3.1-8B Base 的 32 层显式/派生关系探针，38/38 独立检查通过。
- **Stage B1.5–B1.6**：候选行为基线与跨 hop transfer，联合验证通过；详细逐层 CSV/JSON 已入库，`.npz` probe 参数除外。
- **Stage B2**：Base、SFT、DPO、Final 四阶段的行为、within-hop、cross-hop、bootstrap 和冻结控制比较完成，最终验证状态为 PASS，10 张正式图已入库。
- **Stage B2.1**：Model-task Fitness Gate 完成。Raw 行为精确复现；native chat、mean-token 和等长 verbalizer 预注册对照、2,000 次 story-group paired bootstrap、8 张图和独立 validator 已完成。状态为 PASS_WITH_LIMITATION，限制为新诊断使用 transformers 4.57.6，而旧 B2 元数据为 4.44.2。
- **Stage B2.2**：Qwen3-8B候选接口及评分敏感性完成，Native-Sum多步准确率18.53%。作为B2.3参照一并保存。
- **Stage B2.3**：两种自由生成模式各2063题完成；三seed pilot、匹配解码、bootstrap与独立复核已归档。非思考多步25.65%；原2K思考输出80.71%截断，不能据此判断充分预算thinking纯效应。
- **Stage B2.3预算校准**：旧10题30次与新100题300次8K／16K／32K独立生成均完成。新100题正确59／64／64、解析77／96／96，无OOM；独立审计5075项通过。
- **Stage B2.3轨迹与综合报告**：100条唯一32K轨迹高频／精确重复分析、诊断个案、14节图文总结与8张重点图完成。可见过程、最终正确与隐藏因果机制分开解释。
- **MATH-500 preliminary**：同一 pilot-20 上的 Base/SFT/DPO/Final 行为层流程完成；这是可行性实验，不作为正式 benchmark 结论。

## 已恢复但不进入 Git 的资产

本次B2.3为原始生成记录增加了归档例外：逐题shards、合并结果、冻结协议、审计、遥测与相关日志进入仓库；62MB合并JSONL使用无损gzip，附原始SHA及还原脚本。模型权重、完整激活、缓存与临时渲染页仍排除。精确清单见 `results/stage-b2_3_sync/`。

- 云服务器正式 artifact 已于 2026-09-01 重新连接并续传到本地暂存区：B0.2c 8 个文件、B1 321 个文件、B2 1500 个文件。
- Stage B2.1 云端 artifact 约 59 MB，完整保留于服务器；Git 仅纳入约 1.6 MB 的协议、审计、聚合数据、统计与验证，以及约 1.5 MB 的 8 张图。排除范围见 `results/stage-b2_1/PUBLIC_SYNC_MANIFEST.md`。
- `.npz` probe 参数、逐层冗余 checkpoint 文件和大体积逐样本预测只保留在本地暂存/服务器，不进入 Git 主仓库；摘要、协议、验证、聚合表格和 checksum 已入库。
- 文件数和容量核验见 [`SYNC_AUDIT.md`](SYNC_AUDIT.md)。

## 本仓库有意不包含

- Hugging Face/ModelScope 模型权重与缓存；
- Stage A/B 完整 activation tensors；
- Python/Conda 虚拟环境与安装缓存；
- 原始 StepGame 数据副本；
- 论文 PDF；
- SSH key、API token、凭据和个人目录文件；
- 无关服务器临时日志和中间 checkpoint；B2.3已完成实验的相关运行日志为本轮明确归档例外。

## 后续实验边界

B3状态形成／续写／受控干预实验正在服务器上独立运行，未包含在本次B2.3同步中。B2.3的结果不能写成B3已完成，也不能据单一Qwen检查点归因后训练。

## 发布前待办

1. 为脚本增加统一的可配置项目根目录，替代历史服务器绝对路径。
2. 选择开源许可证后再将仓库设为公开。
