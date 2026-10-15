"""Update the existing report, preserving artifact ID and historical content."""
from pathlib import Path
import csv, json, re, shutil, hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
APP=ROOT.parent/'stage-b-research-report-app'
MD=HERE/'StageB2_2-B2_3_阶段综合报告_20261015.md'
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def json_read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
text=MD.read_text(encoding='utf-8')
for a,b in {
 'fig03_budget_outcomes_cost':'fig_01_budget_outcomes_cost',
 'fig04_budget_by_hop':'fig_02_budget_hop_accuracy_completion',
 'fig05_loop_concentration':'fig_03_long_tail_repetition',
 'fig06_high_frequency_coverage':'fig_04_trace_frequency_coverage',
}.items():text=text.replace(a,b)
MD.write_text(text,encoding='utf-8')

snapshot_path=APP/'src/data.json'
snapshot=json_read(snapshot_path)
if not (HERE/'report_snapshot_before_update.json').exists():
    shutil.copy2(snapshot_path,HERE/'report_snapshot_before_update.json')
snapshot.update({'title':'StepGame：预算完成问题已定位，研究转向关系状态与答案使用',
 'generatedAt':'2026-10-15T00:00:00+08:00','buildStatus':'updating'})
snapshot['report']['asOf']='2026-10-15'

def query(key,rows,files,definition,components):
    snapshot['queries'][key]={'rows':rows,'source':{
      'label':definition,'files':files,'filters':['冻结原始实验；本次整理不重新生成'],
      'metricDefinitions':[{'label':'范围与定义','definition':definition,'componentIds':components,
      'sourceLineage':[{'files':files}]}]},
      'methods':[{'language':'text','code':'由本地冻结指标/逐题审计读取；复现路径：build_stage_figures.py、make_budget_figures.py、build_report_app.py。保留分母与等hop定义。'}]}
stage=json_read(HERE/'figure_data/stage_derived.json')
pretty={'B2.2 Native-Sum':'候选评分','FG-N':'非思考生成','FG-T':'思考生成（2K）'}
full=[{**r,'protocolLabel':pretty[r['protocol']],'hopLabel':'h'+str(r['hop'])} for r in stage['full_behavior']]
query('qwen_full',full,['artifacts/stage_b2_3_qwen3_generation/statistics/full_metrics_by_hop.csv',
 'artifacts/stage_b2_3_qwen3_generation/analysis_20261013/corrected_generation_stats.csv'],
 '同一2063题，h1—h5 n=712/505/422/282/142。accuracy含INVALID计错；derived为h2—h5等权平均，非全体样本加权。',
 ['current-sec-1','current-sec-2','current-sec-4','current-sec-5','current-chart-full','current-chart-completion','current-sec-14'])
query('qwen_confusion',stage['confusion_8x9'],[
 'artifacts/stage_b2_3_qwen3_generation/full/fg_nonthinking/row_generations.jsonl',
 'artifacts/stage_b2_3_qwen3_generation/full/fg_thinking/row_generations.jsonl'],
 '两模式全部2063题构建8×9矩阵，INVALID保留为第九列；rowRate为每个gold类中所占比例。',
 ['current-sec-5','current-fig02_confusion_with_invalid'])
derived=read_csv(ROOT/'artifacts/stage_b2_3_qwen3_generation/statistics/derived_summary.csv')
query('qwen_derived',derived,['artifacts/stage_b2_3_qwen3_generation/statistics/derived_summary.csv',
 'artifacts/stage_b2_3_qwen3_generation/statistics/paired_bootstrap.csv',
 'artifacts/stage_b2_3_qwen3_generation/statistics/matched_decoding_metrics.csv'],
 'h2—h5每hop等权；95%区间为hop内story_group_id配对bootstrap，2000次。matched250题中多步200题。',
 ['current-sec-1','current-sec-5'])
B100=ROOT/'artifacts/stage_b2_3_budget100_20261014'
m=read_csv(B100/'audit_completion_metrics.csv')
budget=[];hop=[];cost=[]
for r in m:
    b=f'{int(r["budget"])//1024}K';n=int(r['n'])
    row={'budget':b,'hop':'全部' if r['hop']=='all' else 'h'+r['hop'],'n':n,
      'accuracyRate':int(r['correct'])/n,'parseRate':int(r['parsed'])/n,
      '正确':int(r['correct']),'已解析但错误':int(r['parsed'])-int(r['correct']),
      '截断无正式答案':int(r['truncated']),'meanTokens':float(r['mean_tokens']),
      'generationHours':float(r['seconds'])/3600,'meanMinutes':float(r['mean_seconds'])/60,
      'peakAllocatedGiB':float(r['peak_allocated_gib'])}
    if r['hop']=='all':budget.append(row)
    else:hop.append(row)
query('budget100',budget,['artifacts/stage_b2_3_budget100_20261014/audit_completion_metrics.csv',
 'artifacts/stage_b2_3_budget100_20261014/experiment_plan.json',
 'artifacts/stage_b2_3_budget100_20261014/paired_budget_changes.csv'],
 '同一100题，各hop20；排除旧10题，抽样seed20261014。三档从原提示独立运行，batch1，同题同派生seed；正确/解析错误/截断互斥。诊断样本不估计自然总体。',
 ['current-sec-1','current-sec-6','current-chart-budget','current-chart-cost','current-sec-11','current-sec-13','current-sec-14'])
query('budget_hop',hop,['artifacts/stage_b2_3_budget100_20261014/audit_completion_metrics.csv'],
 '每预算每hop20题；accuracyRate含截断计错，parseRate是交付可解析最终答案的比例。单seed小样本，不排序总体难度。',
 ['current-sec-6','current-chart-hop'])
freq=read_csv(B100/'analysis_high_frequency_20261015/phrase_frequencies.csv')
chosen=['maybe','wait','let me try','coordinates','alternatively','let me think','perhaps',"let's think"]
phrases=[{'phrase':r['phrase'],'coverageRate':int(r['trajectories_with_phrase'])/100,
 'occurrences':int(r['total_occurrences']),'meaning':r['meaning']} for r in freq if r['budget']=='32768' and r['phrase'] in chosen]
phrases.sort(key=lambda r:-r['coverageRate'])
query('trace_frequency',phrases,[
 'artifacts/stage_b2_3_budget100_20261014/analysis_high_frequency_20261015/phrase_frequencies.csv',
 'artifacts/stage_b2_3_budget100_20261014/analysis_high_frequency_20261015/frequency_summary.json',
 'artifacts/stage_b2_3_budget100_20261014/analysis_high_frequency_20261015/per_trajectory_repetition.csv'],
 '100条唯一32K可见thinking_text，排除prompt/final；coverageRate为出现该精确短语的题数/100。精确重复为同一规范化句子额外出现词数/总词数，不代表过程无效。',
 ['current-sec-7','current-chart-frequency','current-fig_03_long_tail_repetition'])
cases=json_read(B100/'audit_cases.json')
case_rows=[{k:r.get(k) for k in ['row_id','hop','gold','parsed','final_correct','process_assessment','note']} for r in cases]
query('trajectory_cases',case_rows,['artifacts/stage_b2_3_budget100_20261014/audit_cases.json'],
 '8个诊断个案代表性片段人工审阅，不是100条全轨迹过程审计；最终正确与过程有效分开。90898末尾坐标支持右上，最终答左下。',
 ['current-sec-8','current-fig07_state_vs_answer_case'])
snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')

candidate=read_csv(ROOT/'artifacts/stage_b2_2_qwen3/statistics/derived_summary.csv')
query('candidate_sensitivity',candidate,['artifacts/stage_b2_2_qwen3/statistics/derived_summary.csv'],
 'B2.2八种候选接口结果。Qwen六条件为h2—h5等hop平均；等长候选同时改变表述，不是纯长度干预。',
 ['current-sec-2','current-sec-4'])
snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')

mapping={
 1:['qwen_derived','budget100'],2:['qwen_full','b2_within','candidate_sensitivity'],3:['b0_splits'],4:['qwen_full','candidate_sensitivity'],
 5:['qwen_full','qwen_confusion','qwen_derived'],6:['budget100','budget_hop'],
 7:['trace_frequency'],8:['trajectory_cases'],11:['budget100','qwen_full'],13:['budget100'],14:['qwen_full','budget100']}
img_re=re.compile(r'!\[([^\]]*)\]\(figures/([^)]*)\)')
sections=[]
for block in re.split(r'\n(?=## \d+\.)',text)[1:]:
    num=int(re.match(r'## (\d+)',block).group(1));parts=[];pos=0
    for match in img_re.finditer(block):
        if match.start()>pos:parts.append({'type':'text','value':block[pos:match.start()].strip()})
        parts.append({'type':'figure','alt':match.group(1),'file':match.group(2)})
        pos=match.end()
    if pos<len(block):parts.append({'type':'text','value':block[pos:].strip()})
    # Local evidence files are available through the source inspector / companion Markdown.
    for part in parts:
        if part['type']=='text':part['value']=re.sub(r'\[([^\]]+)\]\((?!https://)[^)]*\)',r'\1',part['value'])
    sections.append({'id':f'current-sec-{num}','number':num,'queryIds':mapping.get(num,[]),'parts':parts})
(APP/'src/content/report/current-sections.json').write_text(json.dumps(sections,ensure_ascii=False,indent=2),encoding='utf-8')
assets=APP/'src/content/assets/stage-current';assets.mkdir(exist_ok=True)
for p in (HERE/'figures').glob('*.png'):shutil.copy2(p,assets/p.name)
(HERE/'figure_data/report_build_metadata.json').write_text(json.dumps({
 'artifactId':snapshot['id'],'sections':len(sections),'queryIds':list(snapshot['queries']),
 'companionMarkdownSha256':hashlib.sha256(MD.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'id':snapshot['id'],'sections':len(sections),'assets':len(list(assets.glob('*.png'))),'queries':len(snapshot['queries'])},ensure_ascii=False))
