from pathlib import Path
import csv
import json
import re
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'results/stage-b1/union_manifest.csv'
MEMBERSHIP = ROOT / 'results/stage-b0/b0_2/primary_test.csv'
with SOURCE.open(encoding='utf-8-sig', newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['primary_split'] == 'test']
with MEMBERSHIP.open(encoding='utf-8-sig', newline='') as f:
    primary_ids = {int(r['row_id']) for r in csv.DictReader(f)}
assert len(rows) == 2063
assert {int(r['row_id']) for r in rows} == primary_ids
assert all(len(json.loads(r['story'])) == 5 for r in rows)
assert all(len(set(re.findall(r'\b[A-Z]\b', ' '.join(json.loads(r['story']))))) == 6 for r in rows)
by_id = {int(r['row_id']): r for r in rows}
annotations = [
    (84006, 1, 'RIGHT', 'R相对于J在右侧。第2条直接给出，其余4条不在查询路径。', ['J','R'], [(1,0)]),
    (84039, 3, 'LOWER_RIGHT', '从B出发，O在右下，E在O下方，R在E下方；R相对B为(1,-3)，右下。', ['B','O','E','R'], [(1,-1),(0,-1),(0,-1)]),
    (84042, 5, 'RIGHT', 'F→Z→T→J→O→C，位移和为(2,0)，上下分量抵消，答案为正右方。', ['F','Z','T','J','O','C'], [(-1,1),(1,-1),(1,-1),(0,1),(1,0)]),
]
examples = []
for row_id, hop, label, explanation, path, displacements in annotations:
    r = by_id[row_id]
    assert int(r['computed_hop']) == hop and r['canonical_relation'] == label
    dx = sum(d[0] for d in displacements)
    dy = sum(d[1] for d in displacements)
    direction = {(1,0):'RIGHT',(1,-1):'LOWER_RIGHT'}[((dx>0)-(dx<0),(dy>0)-(dy<0))]
    assert direction == label and len(displacements) == hop
    examples.append({
        'row_id': row_id, 'computed_hop': hop, 'story': json.loads(r['story']),
        'question': r['question'], 'label': label.lower().replace('_','-'),
        'annotation': {'explanation_zh': explanation, 'path':path,
                       'unit_grid_displacements':displacements, 'sum_displacement':[dx,dy]}
    })
payload = {'source':str(SOURCE), 'primary_membership':str(MEMBERSHIP),
           'note':'Story/question/label/hop are from the frozen test. Path and explanation are added annotations using the dataset unit-grid convention.',
           'examples':examples}
(HERE/'StepGame_真实样例_20261013.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
parts = ['# StepGame当前数据复杂度与真实样例',
         '\n本文件整理于2026-10-13。三个样例均属于本项目冻结Primary test，英文story/question保持原文；路径与中文解释为补充标注。',
         '\n当前2063题全部有5条关系陈述、6个字母实体；h1–h5为实际查询最短路径的边数，数量依次为712/505/422/282/142。我们保留八方向、排除overlap，使用clean且通过语义审计的样本。',
         '\n原数据文件qaK的K不能直接当查询hop：生成K边关系链后可以询问任意两实体。我们的源故事长度固定为5，独立计算真实hop。',
         '\n难度变量包括实体绑定、查询方向反转、对角关系、关系乱序、路径选择、分量抵消和表面模板。h增加时相关边从1增至5，非路径边从4降至0；因此没有将干扰数量与hop完全独立控制。',
         '\n推导采用数据集离散单位网格规则：右=(1,0)，上=(0,1)，右下=(1,-1)，左上=(-1,1)，反向关系取负；最终按总位移的符号确定方向。这是数据生成约定，不能把一般自然语言中的定性方向都默认成相同距离。']
for e in examples:
    parts += [f"\n## h{e['computed_hop']}：row_id={e['row_id']}",
              '\n```text\n'+'\n'.join(e['story'])+'\n\n'+e['question']+'\n```',
              f"\n正确标签：`{e['label']}`。",
              '\n'+e['annotation']['explanation_zh'],
              '\n路径：'+' → '.join(e['annotation']['path'])+'。']
parts += ['\n## 算法复杂度与适用范围',
          '\n把文本正确转换成关系图之后，用BFS找查询路径为O(V+E)，沿路径累加位移为O(h)。当前V=6、E=5、h≤5，符号计算量很小；模型难点在语言解析、实体绑定和组合执行。这个数据适合受控关系机制研究，但不能单独代表一般数学/代码推理或真实世界空间智能。',
          '\n本地来源：[冻结输入](D:/2026本基/参考文献1/reasoning-geometry-pilot/results/stage-b1/union_manifest.csv)、[数据审计](D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/B0_1_DATA_AUDIT_REPORT.md)、[划分报告](D:/2026本基/参考文献1/reasoning-geometry-pilot/docs/stage-b/B0_2_SPLIT_REPORT.md)。',
          '\n公开来源：[StepGame原论文](https://ojs.aaai.org/index.php/AAAI/article/view/21383)、[官方仓库](https://github.com/ShiZhengyan/StepGame)。']
(HERE/'StepGame_复杂度与真实样例_20261013.md').write_text('\n'.join(parts)+'\n',encoding='utf-8')
print(json.dumps({'rows':len(rows),'hop_counts':dict(Counter(int(r['computed_hop']) for r in rows)), 'example_ids':[e['row_id'] for e in examples]},ensure_ascii=False))
