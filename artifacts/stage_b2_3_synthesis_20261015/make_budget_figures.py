"""Reproduce reviewed static figures for the Stage B2.3 synthesis.

Reads frozen experiment/audit outputs only; never rewrites experiment data.
Run with Python 3 + pandas, numpy, matplotlib.
"""
from pathlib import Path
import hashlib
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "stage_b2_3_budget100_20261014"
FREQ = SOURCE / "analysis_high_frequency_20261015"
OUT = HERE / "figures"
DATA = HERE / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 11, "figure.dpi": 140,
    "savefig.dpi": 210, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#B9C0C7",
    "axes.labelcolor": "#28333F", "text.color": "#28333F",
    "xtick.color": "#4C5865", "ytick.color": "#4C5865",
})
BLUE = "#246C9B"
GOLD = "#D6A548"
ORANGE = "#C96340"
GREY = "#DBE0E5"
DARK = "#213B51"
PALETTE = {8192: GOLD, 16384: BLUE, 32768: DARK}
LABEL = {8192: "8K", 16384: "16K", 32768: "32K"}
checks = []
used_sources = set()


def check(name, condition, detail=None):
    checks.append({"check": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def read_csv(path):
    used_sources.add(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path):
    used_sources.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def export_csv(frame, name):
    frame.to_csv(DATA / name, index=False, encoding="utf-8-sig")


def tidy(ax, grid="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#EBEEF1", linewidth=0.8)
    ax.tick_params(length=0)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    plt.close(fig)


rows = read_csv(SOURCE / "audit_completion_rows.csv")
metrics = read_csv(SOURCE / "audit_completion_metrics.csv")
budget_summary = read_csv(SOURCE / "budget_summary.csv")
pairs = read_csv(SOURCE / "audit_completion_pairs.csv")
rep = read_csv(FREQ / "per_trajectory_repetition.csv")
phrases = read_csv(FREQ / "phrase_frequencies.csv")
sentences = read_csv(FREQ / "repeated_sentences_32k.csv")
frequency_summary = read_json(FREQ / "frequency_summary.json")
frequency_validation = read_json(FREQ / "frequency_validation.json")
check("frequency_validation_pass", frequency_validation["status"] == "PASS" and not frequency_validation["failed"])
check("300_unique_generation_rows", len(rows) == 300 and not rows.duplicated(["row_id", "budget"]).any())
check("100_unique_rows", rows.row_id.nunique() == 100)

# Reconcile all figure metrics to the frozen single-generation files.
raw = {}
for row in rows.itertuples(index=False):
    path = SOURCE / f"budget_{row.budget}" / "shards" / f"row_{row.row_id}.json"
    a = read_json(path)
    raw[(row.row_id, row.budget)] = a
    expected = {
        "row_id": row.row_id, "computed_hop": row.hop,
        "generation_token_count": row.tokens, "thinking_token_count": row.thinking_tokens,
        "correct": row.correct, "parse_success": row.parse_success,
        "eos_terminated": row.eos, "hit_max_new_tokens": row.truncated,
    }
    check(f"raw_metrics:{row.row_id}:{row.budget}", all(a[k] == v for k, v in expected.items()))
    check(f"raw_time:{row.row_id}:{row.budget}", math.isclose(a["wall_seconds"], row.seconds, abs_tol=1e-8))
    check(f"raw_token_count:{row.row_id}:{row.budget}", len(a["generated_ids"]) == row.tokens)

group_rows = []
for budget in [8192, 16384, 32768]:
    for hop in ["all", 1, 2, 3, 4, 5]:
        s = rows[rows.budget.eq(budget)]
        if hop != "all":
            s = s[s.hop.eq(hop)]
        rec = {"budget": budget, "budget_label": LABEL[budget], "hop": str(hop), "n": len(s),
               "correct": int(s.correct.sum()), "parsed": int(s.parse_success.sum()),
               "eos": int(s.eos.sum()), "truncated": int(s.truncated.sum()),
               "wrong_parsed": int((s.parse_success & ~s.correct).sum()),
               "generation_tokens": int(s.tokens.sum()), "thinking_tokens": int(s.thinking_tokens.sum()),
               "seconds": float(s.seconds.sum()), "hours": float(s.seconds.sum()/3600),
               "mean_minutes": float(s.seconds.mean()/60),
               "accuracy": float(s.correct.mean()), "completion_rate": float(s.eos.mean())}
        m = metrics[metrics.budget.eq(budget) & metrics.hop.astype(str).eq(str(hop))].iloc[0]
        b = budget_summary[budget_summary.max_new_tokens.eq(budget) & budget_summary.hop.astype(str).eq(str(hop))].iloc[0]
        check(f"summary_counts:{budget}:{hop}", all(int(m[k]) == rec[k] for k in ["n", "correct", "parsed", "eos", "truncated"]))
        check(f"summary_rates:{budget}:{hop}", math.isclose(rec["accuracy"], b.accuracy) and math.isclose(rec["completion_rate"], b.eos_completion_rate))
        check(f"summary_time:{budget}:{hop}", math.isclose(rec["seconds"], b.total_generation_wall_seconds, abs_tol=1e-7))
        check(f"partition:{budget}:{hop}", rec["correct"]+rec["wrong_parsed"]+rec["truncated"] == len(s))
        check(f"strict_completion_equalities:{budget}:{hop}", s.eos.equals(s.parse_success) and s.eos.equals(s.think_close))
        group_rows.append(rec)
groups = pd.DataFrame(group_rows)
overall = groups[groups.hop.eq("all")].copy()
hops = groups[~groups.hop.eq("all")].copy()
export_csv(overall, "budget_outcomes_and_cost.csv")
export_csv(hops, "budget_hop_results.csv")

for pair in pairs.itertuples(index=False):
    a = raw[(pair.row_id, pair.budget_a)]["generated_ids"]
    b = raw[(pair.row_id, pair.budget_b)]["generated_ids"]
    check(f"shared_prefix:{pair.row_id}:{pair.budget_a}:{pair.budget_b}", a[:min(len(a),len(b))] == b[:min(len(a),len(b))] and pair.prefix_equal)
pair16 = pairs[pairs.budget_a.eq(16384) & pairs.budget_b.eq(32768)]
check("100_paired_16k_32k", len(pair16) == 100)
check("96_identical_full_outputs_16k_32k", int(pair16.entire_equal.sum()) == 96)
check("no_new_valid_or_correct_answers_16k_32k", pair16.parsed_a.equals(pair16.parsed_b) and pair16.correct_a.equals(pair16.correct_b))
pair_delta_rows = []
for (budget_a,budget_b), s in pairs.groupby(["budget_a","budget_b"]):
    a = overall[overall.budget.eq(budget_a)].iloc[0]
    b = overall[overall.budget.eq(budget_b)].iloc[0]
    pair_delta_rows.append({
        "budget_a":int(budget_a), "budget_b":int(budget_b), "n_pairs":len(s),
        "same_full_output":int(s.entire_equal.sum()),
        "new_parsed":int((s.parsed_a.eq("INVALID") & s.parsed_b.ne("INVALID")).sum()),
        "new_correct":int((~s.correct_a & s.correct_b).sum()),
        "lost_correct":int((s.correct_a & ~s.correct_b).sum()),
        "extra_generation_tokens":int((s.tokens_b-s.tokens_a).sum()),
        "extra_generation_seconds":float(b.seconds-a.seconds),
        "relative_generation_time_increase":float(b.seconds/a.seconds-1),
    })
export_csv(pd.DataFrame(pair_delta_rows), "paired_budget_delta_summary.csv")

# Repetition figures use one 32K-cap output per question, not all 300 prefixes.
r32 = rep[rep.budget.eq(32768)].copy()
check("repetition_row_count", len(r32) == 100 and not r32.row_id.duplicated().any())
check("repetition_token_match", int(r32.thinking_tokens.sum()) == int(overall[overall.budget.eq(32768)].thinking_tokens.iloc[0]))
check("primary_frequency_summary", all(int(r32[col].sum()) == frequency_summary["primary"][key] for col,key in [("thinking_tokens","thinking_tokens"),("word_count","words"),("extra_repeated_sentence_words","extra_repeated_sentence_words")]))
loop = r32[r32.truncated]
nonloop = r32[~r32.truncated]
loop_ids = sorted(loop.row_id.astype(int).tolist())
check("four_truncated_32k_traces", len(loop) == 4)
shares = pd.DataFrame([
    {"measure":"题数", "loop_value":len(loop), "total_value":len(r32)},
    {"measure":"可见思考 token", "loop_value":int(loop.thinking_tokens.sum()), "total_value":int(r32.thinking_tokens.sum())},
    {"measure":"重复句的额外词数", "loop_value":int(loop.extra_repeated_sentence_words.sum()), "total_value":int(r32.extra_repeated_sentence_words.sum())},
])
shares["loop_share"] = shares.loop_value / shares.total_value
export_csv(shares, "long_tail_concentration.csv")
tail_tokens = []
for budget in [16384,32768]:
    s = rep[rep.budget.eq(budget)]
    tail_tokens.append({"budget":budget, "budget_label":LABEL[budget],
                       "other96_thinking_tokens":int(s[~s.row_id.isin(loop_ids)].thinking_tokens.sum()),
                       "four_tail_thinking_tokens":int(s[s.row_id.isin(loop_ids)].thinking_tokens.sum())})
tail_tokens = pd.DataFrame(tail_tokens)
extra = int(tail_tokens.iloc[1].four_tail_thinking_tokens-tail_tokens.iloc[0].four_tail_thinking_tokens)
check("extra_65536_tokens_only_in_four_tails", extra == 65536 and tail_tokens.other96_thinking_tokens.nunique() == 1 and int((pair16.tokens_b-pair16.tokens_a).sum()) == extra)
export_csv(tail_tokens, "long_tail_budget_extension.csv")
export_csv(loop, "four_full_32k_truncated_traces.csv")

# Independently recount selected literal phrase coverage from the 100 source texts.
# These lexical matches are descriptions of text, not process-validity labels.
selected = ["to the right of", "coordinates", "to the left of", "clock", "unit steps",
            "wait", "maybe", "let me try", "alternatively", "I'm not sure"]
p32 = phrases[phrases.budget.eq(32768)].set_index("phrase")
chosen = []
for phrase in selected:
    pattern = re.compile(r"\b" + re.escape(phrase.casefold()) + r"\b")
    counts = []
    for row_id in r32.row_id:
        text = raw[(int(row_id),32768)]["thinking_text"].casefold().replace("’", "'")
        counts.append(len(pattern.findall(text)))
    p = p32.loc[phrase]
    check(f"phrase_recount:{phrase}", sum(c>0 for c in counts) == int(p.trajectories_with_phrase) and sum(counts) == int(p.total_occurrences))
    chosen.append({"phrase":phrase, "group":"主题表达" if phrase in selected[:5] else "检查与尝试表达",
                   "covered_questions":int(p.trajectories_with_phrase), "n_questions":100,
                   "coverage":float(p.trajectories_with_phrase/100), "total_occurrences":int(p.total_occurrences)})
chosen = pd.DataFrame(chosen)
export_csv(chosen, "selected_phrase_question_coverage.csv")
top_loops = sentences[sentences.row_of_max.isin(loop_ids)].sort_values("total_occurrences", ascending=False).drop_duplicates("row_of_max").head(4)
check("four_top_loop_sentences_each_one_question", len(top_loops) == 4 and top_loops.trajectories.eq(1).all())
export_csv(top_loops, "loop_sentence_examples.csv")

# 01: Outcomes and cost. Both denominators and independent generation are visible.
fig, axes = plt.subplots(1,2, figsize=(13.4,5.9), gridspec_kw={"width_ratios":[1.35,1]})
fig.subplots_adjust(left=.075,right=.96,top=.76,bottom=.22,wspace=.27)
fig.suptitle("扩大预算主要修复完成率；16K→32K未增加有效答案", x=.075,y=.96,ha="left",fontsize=18,fontweight="bold")
fig.text(.075,.88,"同一100题 · h1–h5各20题 · batch 1 · 每档从原提示独立生成，同题使用同一派生 seed",fontsize=10.7,color="#596774")
ys=np.arange(3)
ax=axes[0]
left=np.zeros(3)
for col,label,color in [("correct","正确",BLUE),("wrong_parsed","已解析但错误",GOLD),("truncated","截断，无正式答案",ORANGE)]:
    vals=overall[col].to_numpy()
    ax.barh(ys, vals, left=left, height=.54,color=color,label=label)
    for y,v,l in zip(ys,vals,left):
        ax.text(l+v/2,y,str(v),ha="center",va="center",fontsize=12,fontweight="bold",color="white" if color!=GOLD else DARK)
    left+=vals
ax.set(yticks=ys,yticklabels=["8K","16K","32K"],xlim=(0,100),xlabel="题数（每档 n=100）",title="最终结果分解")
ax.invert_yaxis(); tidy(ax)
ax.legend(loc="upper left",bbox_to_anchor=(-.02,1.20),frameon=False,ncol=3,fontsize=10,columnspacing=1.2,handlelength=1.1)
ax=axes[1]
ax.barh(ys,overall.hours,height=.54,color=[PALETTE[b] for b in overall.budget])
for y,rec in zip(ys,overall.itertuples(index=False)):
    ax.text(rec.hours+.18,y,f"{rec.hours:.2f} h",va="center",fontsize=12,fontweight="bold")
ax.set(yticks=ys,yticklabels=["8K","16K","32K"],xlim=(0,11.4),xlabel="生成总耗时（小时）",title="串行生成成本")
ax.invert_yaxis(); tidy(ax)
delta_pct=(overall.iloc[2].seconds/overall.iloc[1].seconds-1)*100
fig.text(.075,.10,"8K→16K：新增19个正式答案，其中5个正确。",fontsize=11)
fig.text(.55,.10,f"16K→32K：生成耗时 +{delta_pct:.1f}%，正确题数 +0。",fontsize=11)
fig.text(.075,.045,"注：300次调用均执行成功；此处“完成”指EOS终止、实际关闭think且能解析。耗时不含启动与汇总开销。",fontsize=9.5,color="#65717D")
save(fig,"fig_01_budget_outcomes_cost")

# 02: Correctness and completion by hop, paired denominators.
fig=plt.figure(figsize=(12.7,7.5))
gs=fig.add_gridspec(2,1,height_ratios=[2.8,1],left=.075,right=.96,top=.78,bottom=.15,hspace=.40)
fig.suptitle("完成不等于答对：h5在16K已全部完成，仍仅答对7/20",x=.075,y=.96,ha="left",fontsize=18,fontweight="bold")
fig.text(.075,.885,"同一100题 · 每个 hop、每档 n=20；h1–h5反映查询路径长度，故事均包含5条关系陈述",fontsize=10.8,color="#596774")
ax=fig.add_subplot(gs[0]); x=np.arange(5); width=.23
for k,budget in enumerate([8192,16384,32768]):
    s=hops[hops.budget.eq(budget)]
    vals=s.accuracy.to_numpy()
    ax.bar(x+(k-1)*width,vals,width=width,label=LABEL[budget],color=PALETTE[budget],hatch="//" if budget==32768 else None,edgecolor="white",linewidth=.6)
    for xx,v in zip(x+(k-1)*width,vals):
        ax.text(xx,v+.018,f"{v:.0%}",ha="center",va="bottom",fontsize=10)
ax.set(xticks=x,xticklabels=[f"h{i}" for i in range(1,6)],ylim=(0,1.16),ylabel="正确率（所有题作分母）")
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0)); ax.set_yticks(np.arange(0,1.01,.2)); tidy(ax,"y")
ax.legend(loc="upper right",frameon=False,ncol=3)
ax=fig.add_subplot(gs[1]); ax.axis("off")
table_vals=[]
for budget in [8192,16384,32768]:
    s=hops[hops.budget.eq(budget)]
    table_vals.append([f"{int(v)}/20  ({v/20:.0%})" for v in s.eos])
table=ax.table(cellText=table_vals,rowLabels=["8K完成","16K完成","32K完成"],colLabels=[f"h{i}" for i in range(1,6)],cellLoc="center",rowLoc="center",bbox=[.02,0,.98,1])
table.auto_set_font_size(False);table.set_fontsize(10.5)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor("white");cell.set_facecolor("#F1F4F6" if r==0 else "#FAFBFC")
    if c == -1: cell.set_text_props(color=PALETTE[[8192,16384,32768][r-1]],fontweight="bold")
fig.text(.075,.055,"注：完成=EOS、实际关闭think、严格边界后解析成功（本轮三者一致）。等hop诊断样本，不用于估计自然test分布准确率。",fontsize=9.5,color="#65717D")
save(fig,"fig_02_budget_hop_accuracy_completion")

# 03: Four tails dominate repetition; the marginal budget extends only them.
fig,axes=plt.subplots(1,2,figsize=(13.4,6.4),gridspec_kw={"width_ratios":[1.02,1.15]})
fig.subplots_adjust(left=.19,right=.97,top=.76,bottom=.25,wspace=.38)
fig.suptitle("4条长循环轨迹集中了77.3%的重复词，32K新增计算未带来答案",x=.065,y=.96,ha="left",fontsize=17,fontweight="bold")
fig.text(.065,.88,"以100条32K上限轨迹为主语料；三档均从原提示独立生成，共同前缀全部一致",fontsize=10.8,color="#596774")
ax=axes[0];y=np.arange(3);vals=shares.loop_share.to_numpy()*100
ax.barh(y,vals,height=.51,color=ORANGE,label="4条截断轨迹")
ax.barh(y,100-vals,left=vals,height=.51,color=GREY,label="其余96条")
for yy,v in zip(y,vals):
    ax.text(v+2,yy,f"{v:.1f}%",va="center",fontsize=12,fontweight="bold",color=ORANGE if v<15 else DARK)
ax.set(yticks=y,yticklabels=shares.measure.tolist(),xlim=(0,100),xlabel="这4条轨迹占各项总量的比例",title="长尾集中度")
ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.invert_yaxis();tidy(ax)
ax.legend(loc="upper left",bbox_to_anchor=(-.03,1.24),frameon=False,ncol=2,fontsize=10,handlelength=1)
ax=axes[1]
base=tail_tokens.other96_thinking_tokens.to_numpy()/1000
tail=tail_tokens.four_tail_thinking_tokens.to_numpy()/1000
ax.bar([0,1],base,width=.52,color=GREY,label="相同的其余96条")
ax.bar([0,1],tail,bottom=base,width=.52,color=ORANGE,label="4条截断轨迹")
for xx,b,t in zip([0,1],base,tail):
    ax.text(xx,b+t+12,f"{b+t:,.1f}",ha="center",fontweight="bold",fontsize=11)
    ax.text(xx,b+t/2,f"{t:,.1f}",ha="center",va="center",fontsize=11,color="white")
ax.set(xticks=[0,1],xticklabels=["16K","32K"],ylim=(0,735),ylabel="可见思考 token（千）",title="16K→32K只延长这4条轨迹")
tidy(ax,"y")
fig.text(.065,.14,f"新增 {extra:,} 个思考token；正式答案 +0，正确答案 +0。",fontsize=12,fontweight="bold")
fig.text(.065,.085,"重复词口径：规范化后，同一句保留首次，其后出现次数×句子词数；这不是“无效推理率”。",fontsize=9.8,color="#65717D")
fig.text(.065,.045,"这4题在32K均无EOS、无实际think关闭。其内部重复词占比92.0%；其余96题为7.2%（均按词数加权）。",fontsize=9.8,color="#65717D")
save(fig,"fig_03_long_tail_repetition")

# 04: Breadth of language versus concentrated loops. Different right-axis measure is explicit.
fig,axes=plt.subplots(1,3,figsize=(16.5,6.0),gridspec_kw={"width_ratios":[1.12,1.12,.9]})
fig.subplots_adjust(left=.12,right=.97,top=.75,bottom=.24,wspace=.65)
fig.suptitle("常见表达覆盖很多题；极高重复次数主要来自少数关系句循环",x=.055,y=.96,ha="left",fontsize=18,fontweight="bold")
fig.text(.055,.88,"100题各取一条32K上限的可见思考文本；统计不包含题干和正式答案",fontsize=10.8,color="#596774")
for ax,group,title,color in [(axes[0],"主题表达","任务与空间表达",BLUE),(axes[1],"检查与尝试表达","检查、尝试与不确定表达",GOLD)]:
    s=chosen[chosen.group.eq(group)];y=np.arange(len(s))
    ax.barh(y,s.coverage*100,height=.57,color=color)
    for yy,v in zip(y,s.coverage*100):
        ax.text(v+2,yy,f"{v:.0f}%",va="center",fontsize=10.5)
    ax.set(yticks=y,yticklabels=s.phrase.tolist(),xlim=(0,119),xlabel="出现过该表达的题占比",title=title)
    ax.set_xticks([0,50,100]);ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.invert_yaxis();tidy(ax)
ax=axes[2];y=np.arange(len(top_loops))
ax.barh(y,top_loops.max_in_one_trace,height=.54,color=ORANGE)
for yy,v in zip(y,top_loops.max_in_one_trace):
    ax.text(v+45,yy,f"{int(v):,}",va="center",fontsize=10.5)
ax.set(yticks=y,yticklabels=[f"题 {int(r)}" for r in top_loops.row_of_max],xlim=(0,2700),xlabel="同一句在单题中的出现次数",title="4条循环轨迹的最高频句")
ax.invert_yaxis();tidy(ax);ax.set_xticks([0,1000,2000])
fig.text(.055,.145,"左、中：覆盖率可避免长文本主导总词频。右：每句均只覆盖1题（1%），却可在该题中重复上千次。",fontsize=10.5)
fig.text(.055,.09,"wait / maybe / let me try 是可见文本行为标记，不能直接证明模型完成了检查或有效纠错。",fontsize=10,color="#65717D")
fig.text(.055,.045,"字面短语按预定义规则匹配，类别不互斥；“unit steps”仅4%不代表仅4题理解了单位步长；clock也匹配o’clock中的clock。",fontsize=9.6,color="#65717D")
save(fig,"fig_04_trace_frequency_coverage")

notes = [
    {"id":"fig_01_budget_outcomes_cost", "title":"预算、结果与生成成本", "caption":"同一100题在8K/16K/32K下分别得到59/64/64个正确答案；截断23/4/4题。生成耗时分别为6.44/7.61/9.31小时。8K到16K释放19个正式答案（其中5个正确），16K到32K没有新增正式答案。", "risk":"这是等hop、单seed的预算诊断；不要与旧2K batch8实验作严格因果配对，也不能据此宣称16K为总体最优预算。"},
    {"id":"fig_02_budget_hop_accuracy_completion", "title":"按hop区分完成率与正确性", "caption":"每个hop每档20题。16K与32K的正确题数均为20/12/12/13/7；完成题数均为20/20/18/18/20。h5全部完成却只答对7题，说明扩大预算未消除组合或读出错误。", "risk":"h5样本仅20题；hop与相关边数量、干扰边数量共变，不是独立控制的纯复杂度实验。"},
    {"id":"fig_03_long_tail_repetition", "title":"长尾计算与重复集中", "caption":"4条32K截断轨迹占题数4%、可见思考token19.9%、重复句额外词数77.3%。16K到32K新增65,536个思考token全部来自这4题，未增加正式答案或正确答案。", "risk":"重复率是规范化后完全相同句子的词数加权描述，不等同语义无效率；可见文本也不等同隐藏神经机制。"},
    {"id":"fig_04_trace_frequency_coverage", "title":"高频表达的广度与集中度", "caption":"在100条独立题目轨迹中，wait、maybe覆盖全部题，coordinates覆盖95题；右侧4个具体关系句各只出现于1题，却各重复710–2,208次。", "risk":"短语分类只是展示分组，不互斥；高覆盖的检查措辞不证明有效检查，低unit steps字面覆盖不证明未理解单位步长。"},
]
(DATA / "figure_captions_and_risks.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf-8")
provenance=[]
for path in sorted(used_sources):
    provenance.append({"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
audit={"status":"PASS", "n_checks":len(checks), "failed":[c for c in checks if not c["passed"]], "checks":checks,
       "sources":provenance,"generated_figures":[n["id"] for n in notes],
       "scope":"100 equal-hop questions; three independently generated budgets; one generation seed; visible text only",
       "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(DATA / "budget_figure_validation.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
summary_md=["# 预算与轨迹图：图注、口径与审计", "",f"独立核对 {len(checks)} 项，全部通过。图表数据源及SHA-256见 `budget_figure_validation.json`。", ""]
for n in notes:
    summary_md += [f"## {n['title']}","",f"![{n['title']}](figures/{n['id']}.png)","",n["caption"],"",f"误读风险：{n['risk']}",""]
(HERE / "预算与轨迹图说明.md").write_text("\n".join(summary_md),encoding="utf-8")
print(json.dumps({"status":"PASS","checks":len(checks),"figures":[str(OUT / f"{n['id']}.png") for n in notes]},ensure_ascii=False))
