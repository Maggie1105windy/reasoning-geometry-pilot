# StageB2.3 thinking 预算与运行时间：本地证据和估算模型

日期：2026-10-13。此文件仅分析已有档案，不运行 GPU，不覆盖既有实验文件。更高预算未实测；下表是明确假设下的资源规划场景，不是统计置信区间或保证完成时间。

## 1. 已有实测

运行环境档案：RTX 3090、BF16、Transformers 4.57.6、PyTorch 2.4.0+cu124。当前保存代码 `scripts/stage-b2_3/run_b23.py` 的模型加载使用 SDPA，生成启用 KV cache。

| 运行 | 样本数 | batch | 上限 | 生成时长 | 有效输出 token 吞吐 |
|---|---:|---:|---:|---:|---:|
| 单条 smoke | 10 | 1 | 1024 | 1178.75 秒 = 19.65 分钟 | 8.69 token/s |
| batch smoke | 4 | 4 | 2048 | 321.80 秒 = 5.36 分钟 | 24.71 token/s |
| batch smoke | 8 | 8 | 2048 | 319.50 秒 = 5.33 分钟 | 48.59 token/s |
| pilot，seed 20260924 | 250 | 8 | 2048 | 2.41 小时 | 57.00 token/s |
| pilot，seed 20260925 | 250 | 8 | 2048 | 2.61 小时 | 52.42 token/s |
| pilot，seed 20260926 | 250 | 8 | 2048 | 4.17 小时 | 32.54 token/s |
| matched pilot | 250 | 8 | 2048 | 2.24 小时 | 61.40 token/s |
| full | 2063 | 8 | 2048 | 8.84 小时 | 124.74 token/s |

full 生成 3,967,696 个有效输出 token。上述吞吐均为整个 batch 的有效输出吞吐；每条记录的 `latency_seconds` 是 `batch_wall_latency_seconds / batch_size`，并非每条样本真实独立推理耗时。将逐条归因 latency 求和可以恢复批次总时长。

full 的 258 个 batch 全部至少有一题生成至 2048；3 个 pilot 各 32 个 batch 也全部至少有一题生成至上限。因此 full 的实际 decoder 步数为 258 × 2048 = 528,384，除以批次 latency 总和 31,807.81 秒得到 16.612 decoder steps/s。三种子 pilot 分别为 7.556、6.991、4.371 steps/s。batch4、batch8 smoke 分别为 6.366、6.413 steps/s。

已有档案不足以解释各运行的明显速度差异：没有逐运行 GPU 并发占用、功耗、温度、频率或内核配置追踪。不能将 full 的最好速度视为恒定硬件性能，也不能把 batch4 smoke 的低吞吐归因于 batch 缩小本身。当前代码路径没有给出 smoke 和 full 使用不同 attention 实现的证据。

## 2. 显存约束

Qwen3-8B 官方配置为 36 层、8 个 KV heads、head_dim 128、BF16。普通完整 KV cache 每条序列每个 token 的理论占用：

`2(K,V) × 36 × 8 × 128 × 2 bytes = 147,456 bytes = 144 KiB`。

官方配置参考：<https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json>。此公式不含模型权重、输入 prompt、attention 临时工作区、缓存拼接峰值及 CUDA allocator 保留空间。

| 上限与 batch 规划 | 仅输出 KV cache |
|---|---:|
| 2048 × batch8 | 2.25 GiB |
| 4096 × batch8 | 4.50 GiB |
| 8192 × batch4 | 4.50 GiB |
| 16384 × batch2 | 4.50 GiB |
| 32768 × batch1 | 4.50 GiB |

full 在 2048 × batch8 的实际 allocated 峰值 17.98 GiB，reserved 峰值 23.34 GiB；reserved 不能简单当作持久活跃张量占用，也不能作为高预算一定可运行的保证。4096 × batch8 的理论额外 KV 为 2.25 GiB，需要新的显存 smoke 确认。若需退至 batch4，4K 同样样本的满预算时间大约再翻倍。8K × batch8 理论输出 KV 9 GiB，加约 15 GiB 权重即已接近 24 GiB 显存上限，不应原样沿用 batch8。

## 3. 运行时间应按 batch 最长生成长度估算

当前 `generate_batch` 使用普通静态 batch：达到 EOS 的样本不会被移出，也没有动态补入下一题的调度。批次一直运行到所有样本结束或上限。应使用：

`T ≈ ceil(N / batch_size) × E[batch 内最大生成长度] / decoder_step_rate`。

不能仅使用“平均每题有效 token 数 / 全量124.74 token/s”外推：缩小 batch 会降低聚合吞吐，而提前结束的样本仍会随同批次最长题付出计算。

下表是假设每批至少一题跑满预算的规划场景。左端复用 full 最快平均 16.612 steps/s；右端复用 batch4 smoke 的 6.366 steps/s。batch2、batch1 及所有 4K 以上预算均未实测，所用速率只是比较场景。长上下文的额外 attention 成本未建模。

| token 上限 | 暂定 batch | 100 题 | 250 题 | 2063 题 |
|---|---:|---:|---:|---:|
| 4096 | 8（须 smoke） | 0.89–2.32 小时 | 2.19–5.72 小时 | 17.67–46.11 小时 |
| 8192 | 4 | 3.42–8.94 小时 | 8.63–22.52 小时 | 70.68–184.46 小时（2.95–7.69 天） |
| 16384 | 2 | 13.70–35.75 小时 | 34.25–89.37 小时 | 282.74–737.83 小时（11.78–30.74 天） |
| 32768 | 1 | 54.79–142.99 小时 | 136.98–357.48 小时 | 1130.40–2949.90 小时（47.10–122.91 天） |

这些不是“合理最低–最高完成时间”：如果 batch 的最长题提前完成，实际时间会缩短；例如 16K 实验若平均 batch max 仅 6K，表中时间约乘 6/16。若遇到旧 pilot 最慢的 4.371 steps/s，表格右端还应乘 1.456；长上下文 attention、GPU 并发或 batch 回退也可能进一步增加耗时。单题 smoke 的 8.69 steps/s与 batch4的6.366差异，不能直接说明长上下文单条速度。模型加载曾实测约29秒，统计与文件合并开销很小；排队时间未计。

## 4. 建议的实际执行顺序

先用 8–20 个覆盖各 hop 的样本进行 4K/8K 显存与时长 smoke，记录每批 max length、真实 wall time、allocated/reserved、EOS、think close、final parse 和 GPU 状态。将新的实测 decoder step rate 和 batch max length 带入公式后，再承诺 pilot 或全量时长。

行为校准优先 100 题固定分层集合跑 4K→8K，按完成率、正确率和成本决定是否 16K；32K 先用于 10–20 个长尾失败样本。如果采用只扩展未完成题的预算阶梯，应冻结和记录采样策略、原始提示、batch/seed 变动与统计规则，避免把不同条件生成的完成子集当成无偏总体。

读取证据：`audit/model_manifest.json`、`audit/environment.json`、`audit/model_load_smoke.json`、thinking 的 6 个 pilot/full/smoke `run_manifest.json` 和 `row_generations.jsonl`；代码重点为 `run_b23.py` 的加载函数、`generate_batch` 与 `run_one_batched`。原始文件保持不变。

## 5. token上限的参考与选择标准

Qwen3-8B官方Best Practices推荐多数查询设置32768个输出token，复杂数学/代码竞赛可使用38912。这是官方通用使用参考，不是StepGame专门验证的最优值。`max_new_tokens`包含思考和最终答案的全部生成；不会强制每题生成满该长度，也不是只限制thinking区。官方默认config.max_position_embeddings=40960，使用32K输出时还应检查prompt与output之和是否符合实际加载配置。[官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[官方配置](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)

本项目建议先以8192为探索上限；如果输出仍明显受截断影响，扩展到16384。全量确认预算应选择在校准集上达到低截断率（例如各hop≤5%）与高答案解析率（例如98%）的最小档位，参考原门控标准，并在独立确认集复核。不能预先承诺16K一定足够。32768先用于少量预先定义的长尾诊断样本，不直接用来淘汰/选择模型；若要估计完整32K能力，仍需代表性样本和独立确认。

ICML2026的ASAG研究直接包含Qwen3-8B，显示停止控制可减少生成量并提高平均准确率；Findings of ACL2026的预算研究发现边际收益递减和难度相关的最佳预算。它们支持测量完成、正确率与成本的预算曲线，并不证明本项目的8K或16K是通用最佳值。早停/强制关闭思考均改变推理协议，应与native cap实验分开。[ASAG](https://proceedings.mlr.press/v306/li26ev.html)、[When More Thinking Hurts](https://aclanthology.org/2026.findings-acl.1199/)

所有时间按一个生成seed计算。三个seed运行约增加至三倍；模式/预算组合也需分别计费。高预算mem-smoke、模型加载、独立验证、文件传输、任务等待、失败恢复和统计耗时应另计，原阶段经历的日历跨度不是单次GPU生成时间。
