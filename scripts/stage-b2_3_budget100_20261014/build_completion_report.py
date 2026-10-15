"""Build the requested Chinese Markdown handoff from audited raw results."""
from __future__ import annotations
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/stage_b2_3_budget100_20261014"
CAPS = (8192,16384,32768)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def duration(seconds):
    seconds = round(seconds)
    return f"{seconds//3600}小时{seconds%3600//60:02d}分{seconds%60:02d}秒"


def link(name, label=None):
    return f"[{label or name}]({(OUT/name).resolve().as_posix()})"


def main():
    audit, status = read(OUT/"audit_completion.json"), read(OUT/"status.json")
    assert audit["status"] == "PASS" and not audit["checks_failed"]
    samples = {s["row_id"]:s for s in read(OUT/"sample_manifest.json")["samples"]}
    graph = read(OUT/"audit_completion_graphs.json")
    rows = {cap:{rid:read(OUT/f"budget_{cap}/shards/row_{rid}.json") for rid in samples} for cap in CAPS}
    metrics = {(m["budget"],str(m["hop"])):m for m in audit["metrics"]}
    allm = [metrics[(cap,"all")] for cap in CAPS]
    zone = timezone(timedelta(hours=8))
    start,end = [datetime.fromisoformat(status[k]).astimezone(zone).strftime("%Y-%m-%d %H:%M:%S") for k in ("started_utc","ended_utc")]
    case_notes = {
        91319:("最终INVALID；过程未完成", "尾段反复重复J是N的西南方、N在O西方，未完成I相对J的查询组合。", "Therefore, J is southwest of N, which is west of O."),
        94943:("最终INVALID；过程未完成", "反复循环G在D南方、D在F西南方，未输出真实思考关闭边界。", "G is located in the south direction from D"),
        95732:("最终INVALID；过程未完成", "尾段反复重复Y在K左方、K在L右方，停留在局部关系，没有完成L相对S的推导。", "Therefore, Y is left of K, which is right of L."),
        99424:("最终INVALID；过程未完成", "尾段反复重复T在J上方、J在K右方，局部关系文本没有收敛到W相对T的最终回答。", "So T is upper of J, which is to the right of K."),
        90898:("最终错误；末尾坐标计算局部一致，最终查询方向错误", "末尾给出D=(0,0)、B=(-3,-2)，这支持D在B右上方，却按B相对D回答LOWER_LEFT。局部坐标计算与最终答案正确性可以分离。", "Now, the relation between D (0,0) and B (-3,-2) is that B is lower-left of D."),
        94828:("最终错误；未形成可靠的单位位移组合", "引入自由距离和多个布局，末尾选择其认为多数例子出现的LOWER_RIGHT；按数据集单位位移约定横向抵消，应为BELOW。", "Given that in most examples the relation is lower-right"),
        86290:("最终正确；整段过程有效性未验证，末尾论证不足", "末尾明确在假设P在B左上方的前提下选择UPPER_LEFT，没有在该段证明此目标关系。不能把最终正确直接标为已验证正确过程。", "assuming P is to the left of B and above B"),
        95279:("最终正确；末尾单位坐标算例局部一致，整段过程未完全验证", "末尾算例G=(0,3)、D=(1,2)、W=(2,1)、Z=(2,2)、A=(0,2)符合查询链并支持RIGHT；此前反复考虑自由距离与其他布局，因此这里只认可所检查末尾算例。", "So Z is at (2,2), A at (0,2). So Z is to the right of A."),
    }
    case_records, case_text = [], ["# 100题预算实验：代表性可见文本审计", "", "这是8个按用途选择的诊断个案，不用于估计错误类型比例。关系路径按冻结B0解析边和单位位移约定重算；人工判断基于所列可见文本片段，未做全部100题完整过程标注。可见思考文本不能等同于隐藏神经机制。", ""]
    for rid,(label,note,phrase) in case_notes.items():
        s,r,g = samples[rid],rows[32768][rid],graph[str(rid)]
        pos = r["thinking_text"].rfind(phrase)
        excerpt = r["thinking_text"][max(0,pos-180):pos+len(phrase)+220] if pos>=0 else r["thinking_text"][-800:]
        record = dict(row_id=rid,hop=r["computed_hop"],gold=r["gold_relation"],parsed=r["parsed_relation"],final_correct=r["correct"],process_assessment=label,note=note,checked_excerpt=excerpt,phrase=phrase,phrase_occurrences=r["thinking_text"].count(phrase),gold_path=g["path"],gold_path_relations=g["path_relations"],dx=g["dx"],dy=g["dy"],raw_shard=f"budget_32768/shards/row_{rid}.json",raw_shard_sha256=hashlib.sha256((OUT/f"budget_32768/shards/row_{rid}.json").read_bytes()).hexdigest())
        case_records.append(record)
        case_text += [f"## row_id {rid}，h{r['computed_hop']}", "",f"**{label}。** {note}", "", "故事：", ""]+[f"- {t}" for t in s["story"]]+["",f"查询：{s['question']}", "", f"正确路径：{' → '.join(g['path'])}；关系依次为{', '.join(g['path_relations'])}；累计位移({g['dx']}, {g['dy']})，gold={g['gold']}。", "", f"32K档实际输出{r['generation_token_count']}token；解析={r['parsed_relation']}；EOS={r['eos_terminated']}；真实think关闭={r['think_boundary_ok']}。", "", "所检查原文片段：", "", "```text",excerpt,"```", "",link(record["raw_shard"],"完整原始输出"), ""]
    (OUT/"audit_cases.md").write_text("\n".join(case_text),encoding="utf-8")
    (OUT/"audit_cases.json").write_text(json.dumps(case_records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    telemetry = []
    for cap in CAPS:
        for rid,r in rows[cap].items():
            for phase in ("before","after"):
                snap = r[f"gpu_{phase}"]
                telemetry.append(dict(row_id=rid,budget=cap,phase=phase,**{k:snap[k] for k in ("captured_utc","uuid","name","memory.total","memory.used","memory.free","temperature.gpu","power.draw","pstate","clocks.current.sm","clocks.current.memory","utilization.gpu")}))
    with (OUT/"audit_completion_telemetry.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(telemetry[0]));w.writeheader();w.writerows(telemetry)

    diff8_16 = allm[1]["seconds"]-allm[0]["seconds"]
    diff16_32 = allm[2]["seconds"]-allm[1]["seconds"]
    text = ["# Qwen3-8B思考预算100题实验：16K改善完成率，32K没有新增有效答案", "",
        f"实验于北京时间{start}启动，{end}完成；总进程时长{duration(audit['wall_seconds'])}。同一100题独立运行8K、16K、32K，共300次生成全部执行完成，无运行异常或OOM。", "",
        "**主要结果：8K有23题截断，16K剩4题；32K仍是相同4题截断。最终正确题数为59、64、64。16K到32K增加约1小时42分钟生成时间，没有新增可解析答案或正确答案。** 本轮支持把16K作为后续成本与完成率权衡的候选，但不能宣布行为基础已经可靠或16K普遍最优。", "",
        "## 三档结果", "",
        "下表分母均为本轮固定的100题。执行完成表示生成调用正常返回并保存了结果；EOS自然结束表示模型在达到上限前或上限处发出结束token。截断调用也属于执行完成。", "",
        "| 输出上限 | 执行完成 | EOS／真实think关闭／可解析 | 截断 | 最终正确 | 平均／中位输出token | 100题纯生成耗时 | 平均每题耗时 | 峰值allocated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in allm:
        text.append(f"| {m['budget']//1024}K | 100/100 | {m['eos']}/100，三项一致 | {m['truncated']}/100 | {m['correct']}/100 | {m['mean_tokens']:.2f}／{m['median_tokens']:.1f} | {duration(m['seconds'])} | {m['mean_seconds']/60:.2f}分钟 | {m['peak_allocated_gib']:.2f}GiB |")
    text += ["", f"三档纯生成合计{duration(audit['generation_seconds'])}，共生成{sum(m['total_tokens'] for m in allm):,}token。其他进程时间合计{duration(audit['wall_seconds']-audit['generation_seconds'])}，包含准备、遥测、解码、保存和汇总，不能全部归因于权重hash。", "",
        "## 按hop的完成与正确性", "",
        "每个hop固定20题；表格每格表示“正确题数／EOS且可解析题数”，分母均为20。", "",
        "| hop | 8K正确／结束 | 16K正确／结束 | 32K正确／结束 | 16K／32K剩余截断 |", "|---|---:|---:|---:|---:|"]
    for h in range(1,6):
        ms=[metrics[(c,str(h))] for c in CAPS]
        text.append(f"| h{h} | {ms[0]['correct']}/20；{ms[0]['eos']}/20 | {ms[1]['correct']}/20；{ms[1]['eos']}/20 | {ms[2]['correct']}/20；{ms[2]['eos']}/20 | {ms[1]['truncated']}/20 |")
    text += ["", "h1在20题上全部答对；h2—h5在16K／32K上合计44/80正确，等hop平均为55%。h5仅7/20正确，而且所有20题都正常结束，说明其主要剩余问题已经不能用输出上限不足来解释。h3和h4各有2/20截断，即各10%。每hop只有20题且仅一个生成seed，不据此宣称总体能力或hop难度排名。", "",
        "## 同题答案如何变化", "",
        f"8K到16K：77题完整输出逐token不变；原23个截断中19题得到可解析最终答案，其中5题正确、14题错误，另4题继续截断。正确题数增加5，未出现正确转错误。生成时间增加{duration(diff8_16)}，约{diff8_16/allm[0]['seconds']*100:.1f}%。", "",
        "新增正确的5题：", "", "| row_id | hop | gold／16K答案 | 16K实际输出token |", "|---|---:|---|---:|"]
    for rid in samples:
        a,b=rows[8192][rid],rows[16384][rid]
        if not a["correct"] and b["correct"]:
            text.append(f"| {rid} | h{b['computed_hop']} | {b['gold_relation']} | {b['generation_token_count']} |")
    text += ["", f"16K到32K：96题完整输出逐token相同；另外4题的独立32K输出与16K共享前缀，但延长至32768token后仍再次被截断。没有任何答案或正确性变化。新增输出{allm[2]['total_tokens']-allm[1]['total_tokens']:,}token，生成时间增加{duration(diff16_32)}，约{diff16_32/allm[1]['seconds']*100:.1f}%。预算不强制模型延迟EOS；已经结束的错误答案不会因提高上限而继续修正。", "",
        "三个预算组合各100个配对，共300个配对的重叠token前缀全部一致。每档确实从原始提示独立运行；这是实际测得的一致性，没有用32K输出裁剪来代替8K／16K实验。", "",
        "## 四个长尾样本：延长到32K仍未结束", "",
        "| row_id | hop | gold | 32K实际输出 | 32K单题耗时 | 结果 |", "|---|---:|---|---:|---:|---|"]
    for rid in audit["full_cap_32k_rows"]:
        r=rows[32768][rid]
        text.append(f"| {rid} | h{r['computed_hop']} | {r['gold_relation']} | 32768 | {r['wall_seconds']/60:.2f}分钟 | 无EOS、无真实think关闭、INVALID |")
    text += ["", "四题尾段都出现大量局部关系重复。被检查片段没有提供继续延长到64K就能得到答案的证据；它们更适合作为重复退化与任务约定理解的诊断样本。这里的现象来自可见输出文本，不是对隐藏神经机制的直接观测。", "",
        "## 最终正确与可见过程需要分开", "",
        "本轮人工审阅了8个诊断个案的代表性片段，完整模型输出保留在逐题shard中。未将所有正确答案自动标为过程有效，也未完成100题全轨迹过程标注。", "",
        "| 个案 | 最终结果 | 可见过程判断 |", "|---|---|---|"]
    for rid in (90898,94828,86290,95279):
        label,note,_=case_notes[rid]
        text.append(f"| {rid} | {'正确' if rows[32768][rid]['correct'] else '错误'} | {note} |")
    text += ["", "90898尤其适合提醒下一阶段：文本中的坐标计算可以支持正确空间关系，最终答案却因查询方向反转而错误；应分别研究关系状态和最终正确性信号。86290则说明最终答对并不自动提供已验证的有效推理链。以上均是对所检查可见文本的判断，不能等同于隐藏表征的因果机制。", "",
        link("audit_cases.md","个案原文片段、完整故事、正确路径与过程标注"), "",
        "## 实际耗时、显存与遥测", "",
        "平均每题生成耗时从8K的3.87分钟升至16K的4.57分钟、32K的5.59分钟；三档中位数均约3.9分钟。少数长尾拉高平均值，不能将最长39分钟当作每题平均耗时。", "",
        "| 档位 | 总输出token | 有效输出吞吐 | 最长单题 | 峰值reserved |", "|---|---:|---:|---:|---:|"]
    for m in allm:
        text.append(f"| {m['budget']//1024}K | {m['total_tokens']:,} | {m['total_tokens']/m['seconds']:.2f}token/s | {m['max_seconds']/60:.2f}分钟 | {m['peak_reserved_gib']:.2f}GiB |")
    text += ["", "本轮有4次32K档真实生成到32768token，因而已经实际验证当前单卡RTX3090、BF16、batch1及这批短prompt的满32K输出配置。最长输入157token，上下文上限40960。该结论不外推到更长输入、更大batch或不同实现。allocated峰值为20.29GiB；全运行reserved峰值为23.30GiB，reserved包含allocator保留空间，不能简单当作活跃张量大小。", "",
        "每次生成前后各保存一份GPU快照，共600份：记录同一GPU身份和本进程，没有发现其他计算进程。快照观察到显存占用最高22745MiB，温度34—74°C，功耗99.45—349.42W，SM频率1695—1950MHz。它们是调用前后的点状采样，不能证明调用内部的持续利用率、绝对最高温度或具体性能瓶颈。", "",
        "## 固定实验条件与解析定义", "",
        "- 从冻结2063题Primary test排除旧10题，按抽样seed20261014和确定性SHA排序，各hop抽20题。未按旧正确性、生成长度、截断或解析结果筛选。", "- 同题分别真实从原提示独立生成8192、16384、32768输出上限；batch1，generation seed20260924，同题使用相同row派生seed，预算不参与seed派生。", "- Qwen3-8B冻结权重；BF16，无量化，SDPA，KV cache，eval／inference_mode；temperature=0.6，top_p=0.95，top_k=20，min_p=0，do_sample=True。", "- PyTorch2.4.0+cu124，Transformers4.57.6，Python3.10.20；同一RTX3090。", "- 输出上限同时包含思考和最终回答。EOS为冻结generation_config中的结束token，必须检查实际生成token。", "- 严格think边界：生成侧必须有唯一实际`</think>`token（151668），并具有有效的思考打开状态、没有歧义或关闭后重新打开。缺失／歧义边界计INVALID，即便思考文本中出现“Final answer”也不能作为最终答案。", "- 仅对该边界之后的文本做整行Final answer解析；若存在多个合法最终答案行，冻结实现取最后一个。本轮所有可解析输出都只有1个合法行，未触发多行选择。", "- 这是预算校准，不是选择预算之后的独立能力确认。100题等hop，hop×关系覆盖39/40，h3 ABOVE为0；不能直接估计自然2063题分布的总体准确率或单seed之外的稳定性。", "- 旧2K实验batch与RNG分组不同，不作严格因果配对。旧10题与本轮不重叠，但同属已经评估过的Primary test源池。", "",
        "## 审计、身份与可复现文件", "",
        f"独立审计通过{audit['checks_total']}项检查，无失败：复核100题盲抽样、源文件与内容hash、300个唯一shard、输出token hash、冻结tokenizer重解码、prompt重建、EOS、实际think边界、答案解析、gold正确性、汇总、执行顺序及300个配对前缀。汇总与原始JSONL逐项一致，未修改原始shard。", "",
        "关系路径审计使用远端冻结B0解析边元数据的100题子集，独立重算BFS和单位位移，100题hop与gold全部一致；模板匹配也复核了对应句子。没有在本轮重新进行全数据集模板语义审计。模型全权重SHA由启动runner在远端重算，本地审计将5分片hash与历史清单比较，并重新核对下载的配置和tokenizer文件；未在本地重复下载、重算全部模型权重。", "",
        f"- plan SHA256：`{audit['plan_sha256']}`", f"- model audit SHA256：`{audit['model_audit_sha256']}`", f"- 样本manifest SHA256：`{read(OUT/'experiment_plan.json')['sample_manifest_sha256']}`", f"- 冻结B0关系元数据SHA256：`{audit['graph_source_sha256']}`", "",
        link("audit_completion.json","独立审计详细记录")+"；"+link("audit_completion_rows.csv","300次逐题审计指标")+"；"+link("audit_completion_metrics.csv","三档及各hop独立汇总")+"；"+link("audit_completion_pairs.csv","300个跨预算配对")+"；"+link("audit_completion_telemetry.csv","600份遥测快照")+"；"+link("audit_completion_inventory.json","本地文件hash清单")+"。", "",
        "原始文件："+link("status.json")+"；"+link("runtime_identity.json")+"；"+link("summary.json")+"；"+link("budget_summary.csv")+"；"+link("paired_budget_changes.csv")+"；"+link("logs/run_20261013T162339Z.log","运行日志")+"。三档各100个完整shard及合并JSONL均已同步本地。", "",
        "复现审计脚本："+f"[audit_completed.py]({(ROOT/'scripts/stage-b2_3_budget100_20261014/audit_completed.py').as_posix()})"+"。脚本只读取冻结原始输出，写入派生审计文件，不重新生成答案。", "",
        "## 对下一阶段的直接建议", "",
        "1. 把16K作为下一轮候选原生预算，先在未用于预算选择的独立确认样本上复核完成与正确性。本轮解析率96%，h3／h4截断各10%，仍不足以宣布行为基础可靠。", "2. 对四个循环个案诊断重复退化，对错误与正确但论证不足的个案诊断单位位移／同坐标约定和查询方向；可另设明确单位网格规则与结构化关系组合的提示对照，作为新实验条件保存，不能覆盖原生结果。", "3. 再开展关系状态与最终正确性两条分析：把可见关系组合、最终答案和过程标注分别记录，在共享轨迹上计算token级Δlog p与几何变化。可见过程有效性需要独立验证；不将正确答案集合自动视为正确推理过程集合。", "",
        "本轮校准的实用结论是：从8K到16K主要恢复最终答案可读出，32K的额外成本没有换来本样本中的新增有效结果；剩余研究重点应转向组合规则、查询读出与重复退化，而不是继续无条件延长思考。", ""]
    path = OUT/"8K_16K_32K_100题实验总结_20261015.md"
    path.write_text("\n".join(text),encoding="utf-8")
    print(json.dumps(dict(report=str(path),cases=len(case_records),telemetry_snapshots=len(telemetry),bytes=path.stat().st_size),ensure_ascii=True))


if __name__ == "__main__":
    main()
