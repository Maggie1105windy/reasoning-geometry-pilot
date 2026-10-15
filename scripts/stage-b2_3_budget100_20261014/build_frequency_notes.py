"""Validate frequency findings and write inspectable supporting notes."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/stage_b2_3_budget100_20261014'
OUT=SOURCE/'analysis_high_frequency_20261015'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    result=read(OUT/'frequency_summary.json')
    inventory={r['path']:r for r in read(SOURCE/'audit_completion_inventory.json')}
    checks=[]
    def check(name,passed):
        checks.append(dict(check=name,passed=bool(passed)))
    for source in result['source_provenance']:
        check('frozen_shard_hash:'+source['path'],source['sha256']==inventory[source['path']]['sha256'])
    rows=[read(p) for p in sorted((SOURCE/'budget_32768/shards').glob('row_*.json'))]
    texts=[r['thinking_text'].replace('\u2019',"'").replace('\u2018',"'").lower() for r in rows]
    phrase_by={r['phrase']:r for r in result['phrases']}
    for phrase in ('maybe','wait','alternatively','coordinates'):
        counts=[len(re.findall(r'(?<![a-z])'+re.escape(phrase)+r'(?![a-z])',text)) for text in texts]
        check('independent_word_count:'+phrase,sum(counts)==phrase_by[phrase]['total_occurrences'] and sum(c>0 for c in counts)==phrase_by[phrase]['trajectories_with_phrase'])
    for record in result['top_repeated_sentences'][:5]:
        counts=[text.count(record['sentence'].lower()) for text in texts]
        check('literal_repeated_sentence:'+str(record['row_of_max']),sum(counts)==record['total_occurrences'] and max(counts)==record['max_in_one_trace'])
    for budget in (8192,16384,32768):
        groups={r['group']:r for r in result['groups'] if r['budget']==budget}
        for key in ('n','thinking_tokens','words','extra_repeated_sentence_words'):
            check(f'group_reconciliation:{budget}:{key}',sum(groups[g][key] for g in ('correct','wrong_parsed','truncated'))==groups['all'][key])
    failed=[c for c in checks if not c['passed']]
    validation=dict(status='PASS' if not failed else 'FAIL',checked_utc=datetime.now(timezone.utc).isoformat(),checks=checks,failed=failed)
    (OUT/'frequency_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert not failed,failed
    main,tail=result['primary'],result['truncated']
    normal_repeat=(main['extra_repeated_sentence_words']-tail['extra_repeated_sentence_words'])/(main['words']-tail['words'])
    preferred=('maybe','wait','alternatively','let me try','coordinates','let me think','for example','to the right of','to the left of','clock')
    lines=['# 思考轨迹高频内容：100条32K档可见思考文本','',
        '高频内容主要包括空间关系重述、反复试探与自我检查、坐标／时钟构造，以及少数样本中的整句循环。',
        '',f"主统计读取32K档每题一条，共100条、{main['thinking_tokens']:,}个思考token、{main['words']:,}个词元。只读取thinking_text，不统计输入提示和思考关闭后的最终回答。三档共享大量前缀，因此没有将300条记录混作300个独立样本。",'',
        '## 经常出现的表达','',
        '| 原词／短语 | 含义 | 总出现次数 | 覆盖轨迹 | 每千词次数 |','|---|---|---:|---:|---:|']
    for name in preferred:
        p=phrase_by[name];display="clock（含o'clock中的clock）" if name=='clock' else name
        lines.append(f"| {display} | {p['meaning']} | {p['total_occurrences']:,} | {p['trajectories_with_phrase']}/100 | {p['occurrences_per_1000_words']:.2f} |")
    lines += ['', 'maybe和wait在全部100条轨迹中都出现；坐标构造、换一种方式、再尝试推导的表达也覆盖大部分轨迹。它们同时存在于正确和错误输出中，不能单独据此判断推理无效或确定错误原因。时钟表达频率受题目包含的描述模板影响，不能直接与坐标表达作能力优劣比较。','',
        '原词计数中，right出现12,205次、left出现10,916次，均覆盖100条；therefore出现6,336次，覆盖96条。方位词属于任务内容，出现频率高本身并非异常。','',
        '跨样本常见的完整句还包括：Let me think again.（173次，73条）；Let me try to assign coordinates.（56次，45条）；Let me check again.（48次，38条）；But I\'m not sure.（36次，23条）。','',
        '## 最高频的完整句集中在4条截断轨迹','',
        '| row_id | 原句 | 单条轨迹出现次数 |','|---|---|---:|']
    for record in result['top_repeated_sentences'][:5]:
        lines.append(f"| {record['row_of_max']} | {record['sentence']} | {record['max_in_one_trace']:,} |")
    lines += ['', '91319和95732分别反复陈述一个固定的两段空间关系；99424反复陈述T、J、K的局部位置；94943在两个相近句子之间交替。四条最终均未关闭思考或给出可解析最终答案。这里的最高频整句只覆盖各自1条轨迹，不能称作100题共同的思考内容。','',
        '## 循环集中度','',
        f"4条截断轨迹占32K档思考token的{result['truncated_token_share']*100:.1f}%，却贡献了本统计中{result['truncated_repeat_word_share']*100:.1f}%的“完全相同规范化句的额外重复词数”。",'',
        f"全部100条中，额外重复句词数占全部词元的{main['weighted_repeated_sentence_word_fraction']*100:.1f}%；去除4条截断轨迹后为{normal_repeat*100:.1f}%。4条内部合计为{tail['weighted_repeated_sentence_word_fraction']*100:.1f}%。这些是文本精确重复指标，不是经人工判断的无效推理比例。",'',
        '| 档位 | 全部轨迹的重复句词数占比 | 正确输出 | 错误但已解析输出 | 截断输出 |','|---|---:|---:|---:|---:|']
    for cap in (8192,16384,32768):
        groups={r['group']:r for r in result['groups'] if r['budget']==cap}
        values=[f"{groups[g]['weighted_repeated_sentence_word_fraction']*100:.1f}%（{groups[g]['n']}条）" for g in ('all','correct','wrong_parsed','truncated')]
        lines.append(f"| {cap//1024}K | "+' | '.join(values)+' |')
    lines += ['', '三档各组成员并不完全相同，组间差异还受轨迹长度、题目与hop影响，不能把差异当作重复导致错误的因果证据。16K到32K的96条完整轨迹逐token不变；新增65,536个思考token全部来自4条持续截断轨迹。全体的重复指标上升主要集中在这4条，而非其余96题发生普遍变化。','',
        '## 统计口径与可复核文件','',
        '- 词／词组计数不区分大小写，统一直／弯撇号；按列出的英文原词／词组匹配，不合并同义表达。总次数与覆盖轨迹数是不同指标。',
        '- 整句用标点后空白或换行作启发式分段，统一大小写和空白、去掉句首列表符号及句尾标点；保留实体字母与坐标，不把不同题目的实体替换为同一名称。',
        '- 重复词数先保留每个规范化句的第一次出现，再把额外出现的该句词数相加，分母为全部思考词元。词元定义包括英文字词、单字母实体和数字，与模型token不是同一单位。',
        '- 高频句、词与短语会重叠，不能把各行次数相加解释为互斥的内容类型比例。对思考有效性与隐藏机制的判断需要额外证据。',
        '- 与完成时的文件hash清单比较，300个输入shard均未变化；独立复核部分原词计数、最高频5句的原文直接计数，以及三档分组加总。', '',
        '| 文件 | 内容 |','|---|---|',
        '| phrase_frequencies.csv | 三档各24个预定义表达的总频次、覆盖、长度归一化频次 |',
        '| repeated_sentences_32k.csv | 最高频200个规范化句及来源row_id |',
        '| word_frequencies_32k.csv | 去除已列停用词后的高频词150项 |',
        '| ngram_frequencies_32k.csv | 各3—6词长度、覆盖至少5条轨迹的高覆盖短语 |',
        '| per_trajectory_repetition.csv | 300次生成各自的精确句子重复指标 |',
        '| frequency_summary.json | 统计定义、汇总与300个源文件hash |',
        '| frequency_validation.json | 本轮频次结果的验证记录 |','',
        '原始输出仍保留在上一层budget_*/shards目录，本分析没有修改或重新生成模型输出。','']
    path=OUT/'思考轨迹高频内容_20261015.md';path.write_text('\n'.join(lines),encoding='utf-8')
    preview=[{k:phrase_by[name][k] for k in ('phrase','total_occurrences','trajectories_with_phrase')} for name in preferred[:7]]
    common_source=dict(label='Qwen3-8B预算实验可见思考文本',files=[{'label':'32K档100个原始shard'}],filters=['输出预算：32768','每题保留一条轨迹','字段：thinking_text','不含prompt和最终回答'],caveats=['100题为等hop诊断集合，只有一个生成seed。','高频表达不等于无效推理或隐藏神经机制。'])
    receipt={'schemaVersion':1,'items':[
        {'id':'frequent-expressions','title':'跨轨迹高频表达','queries':[{'id':'expression-counts','source':dict(common_source,metricDefinitions=[{'label':'出现次数和覆盖','definition':'出现次数统计原词或词组在100条可见思考中的匹配次数；覆盖统计至少出现一次的轨迹数。'}]),'columns':['phrase','total_occurrences','trajectories_with_phrase'],'rows':preview,'methods':[{'language':'calculation','code':'每条轨迹分别匹配英文原词／词组；总次数求和，覆盖数计入至少匹配一次的轨迹。'}]}]},
        {'id':'repeated-sentences','title':'单条轨迹中的整句循环','queries':[{'id':'literal-sentence-counts','source':dict(common_source,metricDefinitions=[{'label':'整句重复次数','definition':'整句计数统一大小写、空白和句尾标点，保留实体字母与坐标。'}]),'columns':['row_id','sentence','count'],'rows':[{'row_id':s['row_of_max'],'sentence':s['sentence'],'count':s['max_in_one_trace']} for s in result['top_repeated_sentences'][:5]]}]},
        {'id':'loop-concentration','title':'4条截断轨迹集中贡献重复文本','queries':[{'id':'repeat-concentration','source':dict(common_source,metricDefinitions=[{'label':'额外重复句词数','definition':'每个规范化句保留第一次出现，其余出现次数乘以该句词数，再对所有句求和。'}],caveats=common_source['caveats']+['精确句子重复只描述文本，不能直接估计无效推理比例。']),'columns':['group','n','thinking_tokens','words','extra_repeated_sentence_words'],'rows':[{k:result[kind][k] for k in ('group','n','thinking_tokens','words','extra_repeated_sentence_words')} for kind in ('primary','truncated')],'methods':[{'language':'calculation','code':f"4条截断轨迹思考token占比：131068 / 659460 = {result['truncated_token_share']*100:.2f}%\n额外重复句词数贡献：98441 / 127357 = {result['truncated_repeat_word_share']*100:.2f}%"}]}]}
    ]}
    (OUT/'inline_sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'notes':str(path),'validation':validation['status'],'checks':len(checks),'normal_repeat_fraction':normal_repeat},ensure_ascii=True))


if __name__=='__main__':
    main()
