"""Focused artifact/evidence verification. Does not rerun GPU experiments."""
from pathlib import Path
import csv, json, re, hashlib
from PIL import Image
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
APP=ROOT.parent/'stage-b-research-report-app'
md=HERE/'StageB2_2-B2_3_阶段综合报告_20261015.md'
text=md.read_text(encoding='utf-8')
checks=[]
def check(name,ok,detail):checks.append({'name':name,'pass':bool(ok),'detail':detail})
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
for link in links:
    if not link.startswith('https://'):check('local_link', (md.parent/link).exists(),link)
figures=re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)',text)
check('eight_report_figures',len(figures)==8,len(figures))
dimensions={}
for link in figures:
    p=HERE/link
    with Image.open(p) as im:
        dimensions[p.name]=list(im.size)
        check('figure_dimensions',im.width>=1800 and im.height>=700,{'file':p.name,'pixels':im.size})
    check('matching_svg',p.with_suffix('.svg').exists(),p.name)
    check('app_asset_matches',hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((APP/'src/content/assets/stage-current'/p.name).read_bytes()).hexdigest(),p.name)
data=json.loads((APP/'src/data.json').read_text(encoding='utf-8'))
before=json.loads((HERE/'report_snapshot_before_update.json').read_text(encoding='utf-8'))
check('artifact_identity_preserved',data['id']==before['id'],data['id'])
for key,q in before['queries'].items():check('historical_query_preserved',data['queries'][key]==q,key)
sections=json.loads((APP/'src/content/report/current-sections.json').read_text(encoding='utf-8'))
check('14_sections',len(sections)==14,len(sections))
for s in sections:
    for q in s['queryIds']:check('section_binding_exists',q in data['queries'],[s['id'],q])
budget=data['queries']['budget100']['rows']
for b in budget:check('outcome_denominator',b['正确']+b['已解析但错误']+b['截断无正式答案']==b['n']==100,b)
check('correct_by_budget',[b['正确'] for b in budget]==[59,64,64],[b['正确'] for b in budget])
check('truncated_by_budget',[b['截断无正式答案'] for b in budget]==[23,4,4],[b['截断无正式答案'] for b in budget])
for protocol,expected in [('FG-N',158),('FG-T',1665)]:
    rows=[r for r in data['queries']['qwen_confusion']['rows'] if r['protocol']==protocol]
    check('confusion_all_rows',sum(r['count'] for r in rows)==2063,protocol)
    check('confusion_includes_invalid',sum(r['count'] for r in rows if r['prediction']=='INVALID')==expected,protocol)
check('strict_boundary_copy_fixed','关闭后不得重新打开' in text,'边界定义')
check('no_replacement_character','\ufffd' not in text,'Unicode')
record={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
 'checks':checks,'passed':sum(c['pass'] for c in checks),'total':len(checks),
 'figures':dimensions,'visualReview':'All 8 standalone PNGs inspected; one legend repositioned to avoid overlapping labels.',
 'webPreview':'NOT VISUALLY VERIFIED: browser policy rejected page access. No alternate browser or indirect screenshot attempted.',
 'scope':'Focused report/source/binding/link/figure checks. Reuses prior 5075-run audit and 1312-figure evidence checks; does not replace them.',
 'markdown_sha256':hashlib.sha256(md.read_bytes()).hexdigest()}
(HERE/'figure_data/synthesis_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':record['status'],'passed':record['passed'],'total':record['total'],'figures':len(figures)},ensure_ascii=False))
if record['status']!='PASS':raise SystemExit(1)
