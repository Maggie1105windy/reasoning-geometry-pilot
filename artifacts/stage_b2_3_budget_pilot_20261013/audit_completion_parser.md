# 独立完成率与答案边界复核

状态：PASS_WITH_DOCUMENTED_LIMITATIONS。30个原始shards逐条复核；317/317检查通过。未导入runner，未以既有summary替代计算。

| Token上限 | 正常结束 | think闭合/可解析 | 正确 | 截断 | 总生成token | 实测生成分钟 |
|---|---:|---:|---:|---:|---:|---:|
| 8192 | 8/10 | 8/10 | 6/10 | 2/10 | 46040 | 32.40 |
| 16384 | 10/10 | 10/10 | 7/10 | 0/10 | 48489 | 34.30 |
| 32768 | 10/10 | 10/10 | 7/10 | 0/10 | 48489 | 34.29 |

三档两两比较的30对输出，重叠部分token前缀全部一致：30/30。16K与32K十题的完整输出均相同。

8K截断的是h4的row 101041和92078。16K将它们分别完成为LOWER_LEFT（错误）和UPPER_RIGHT（正确）；32K结果与16K相同。其余八题三档输出完全一致。

严格边界复核：根据generated_ids中的151667/151668独立定位think开闭，检查final_response_ids等于闭合token后的完整切片；未闭合的两条禁止解析其思考中可能出现的Final answer。由完整generated_text的闭合文本边界投影最终回答，再重新匹配冻结regex和gold，30条结果全部一致。

模型、计划、样本hash和seed、原提示正文及chat wrapper、token计数、EOS/上限关系、完整原始token哈希均通过。启动和逐题前后快照仅有本实验PID，日志30个ROW_COMPLETE，未发现OOM、Traceback或ERROR文件。

限制与档案注记：

- 本次直接重读30条输出，不以runner或既有summary代替独立计算。
- 严格think闭合位置和闭合后token切片由生成token IDs及运行记录的特殊token IDs（151667/151668）独立核对。
- 最终回答文本由完整generated_text在`</think>`之后投影、移除特殊EOS文本，再应用冻结正则。
- 本地没有tokenizer资产，未独立重新解码全部普通token；已核对完整token列表hash、特殊边界索引和最终切片。
- EOS ID 151645来自历史模型manifest；所有正常完成的输出均以该ID结束。
- 本地没有GPU模型权重；已将本次运行重新计算的五个权重SHA与历史档案逐一核对，未在本地再次读取权重计算hash。
- 十题、每hop两题、每题一次随机生成不能支持总体准确率或最小安全预算的结论；三个预算是同题重复实验，不能视作30个独立问题。
- GPU快照记录每次生成前后状态，不能排除记录间隔中的瞬时资源争用；档案和日志未发现OOM或错误证据。
- status.json中的pending_runs=30是初始化遗留值；COMPLETE、completed_runs=30与30条已复核shards一致。
