"""Prepare a reviewable Stage B2.3 sync; never touches GPU or Git index."""
from pathlib import Path
import csv,gzip,hashlib,json,os,re,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/stage-b2_3_sync';OUT.mkdir(exist_ok=True)
SOURCE_DIRS=[ROOT/'artifacts'/n for n in ['stage_b2_2_qwen3','stage_b2_3_qwen3_generation',
 'stage_b2_3_budget_pilot_20261013','stage_b2_3_budget100_20261014','stage_b2_3_synthesis_20261015']]
SCRIPT_DIRS=[ROOT/'scripts'/n for n in ['stage-b2_2','stage-b2_3','stage-b2_3_budget_pilot_20261013','stage-b2_3_budget100_20261014']]
def hfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows,fields):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
excluded=[];selected=[]
if not (OUT/'PREPARATION.json').exists():write_json(OUT/'PREPARATION.json',{'status':'PREPARING'})
for directory in SOURCE_DIRS+SCRIPT_DIRS:
    for p in sorted(directory.rglob('*')):
        if not p.is_file():continue
        parts=p.relative_to(directory).parts
        reason=None
        if 'sources_text' in parts:reason='Unrelated internal proposal/draft source text; not experimental output'
        elif '__pycache__' in parts or p.suffix in ['.pyc','.pyo']:reason='Python bytecode/cache'
        elif 'tmp' in parts or 'temp' in parts:reason='Transient render/review scratch; final figures/PDF preserved'
        elif p.name in ['automation_info.json','launcher.pid','run.lock','latest_log_path.txt','remote_launch.json']:reason='Local automation/process transport metadata'
        elif p.suffix in ['.safetensors','.bin','.pt','.pth','.npy','.npz']:reason='Model/activation assets excluded by repository policy'
        if reason:excluded.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'reason':reason})
        else:selected.append(p)

archives=[];logical=[]
for p in list(selected):
    if p.stat().st_size<=50*1024*1024:continue
    archive=p.with_name(p.name+'.gz')
    with p.open('rb') as src,archive.open('wb') as dst:
        with gzip.GzipFile(fileobj=dst,mode='wb',filename='',mtime=0,compresslevel=9) as gz:shutil.copyfileobj(src,gz)
    original_hash=hfile(p);decoded=hashlib.sha256();n=0
    with gzip.open(archive,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):decoded.update(b);n+=len(b)
    if decoded.hexdigest()!=original_hash or n!=p.stat().st_size:raise RuntimeError('Archive roundtrip failed')
    item={'original_path':p.relative_to(ROOT).as_posix(),'original_bytes':p.stat().st_size,'original_sha256':original_hash,
      'archive_path':archive.relative_to(ROOT).as_posix(),'archive_bytes':archive.stat().st_size,'archive_sha256':hfile(archive)}
    archives.append(item);selected.remove(p);selected.append(archive)
    logical.append({'source_path':item['original_path'],'source_bytes':item['original_bytes'],'source_sha256':original_hash,
     'published_path':item['archive_path'],'published_bytes':item['archive_bytes'],'published_sha256':item['archive_sha256'],'storage':'lossless_gzip_roundtrip_verified'})
write_json(OUT/'RAW_ARCHIVES.json',{'format_version':1,'archives':archives,'restore_command':'python scripts/stage-b2_3/restore_archived_results.py','integrity':'SHA256 refers to exact bytes, without line-ending normalization'})
for p in selected:
    if any(p.relative_to(ROOT).as_posix()==a['archive_path'] for a in archives):continue
    rel=p.relative_to(ROOT).as_posix();h=hfile(p)
    logical.append({'source_path':rel,'source_bytes':p.stat().st_size,'source_sha256':h,'published_path':rel,
      'published_bytes':p.stat().st_size,'published_sha256':h,'storage':'exact_original_bytes'})
csvwrite(OUT/'SYNC_MANIFEST.csv',logical,['source_path','source_bytes','source_sha256','published_path','published_bytes','published_sha256','storage'])
csvwrite(OUT/'EXCLUDED_FILES.csv',excluded,['path','bytes','reason'])

# Publish an entry point whose links work from GitHub, without changing historical reports.
source=ROOT/'artifacts/stage_b2_3_synthesis_20261015/StageB2_2-B2_3_阶段综合报告_20261015.md'
dest=ROOT/'docs/stage-b/STAGE_B2_3_RESEARCH_SUMMARY_ZH.md'
def link(m):
    label,target=m.groups()
    if target.startswith(('https://','http://','#')):return m.group(0)
    path=(source.parent/target).resolve()
    if not path.exists() or not path.is_relative_to(ROOT):return m.group(0)
    return '['+label+']('+os.path.relpath(path,dest.parent).replace('\\','/')+')'
published=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',link,source.read_text(encoding='utf-8'))
published=published.replace('`figures/`','[图表目录](../../artifacts/stage_b2_3_synthesis_20261015/figures/)')
published=published.replace('`figure_data/`','[图表数据](../../artifacts/stage_b2_3_synthesis_20261015/figure_data/)')
dest.write_text(published,encoding='utf-8')

app=ROOT.parent/'stage-b-research-report-app'
shutil.copy2(app/'dist/index.html',ROOT/'reports/stage-b/index.html')
shutil.copy2(app/'src/data.json',ROOT/'reporting/stage-b-reviewed-snapshot.json')
selected += [dest,ROOT/'reports/stage-b/index.html',ROOT/'reporting/stage-b-reviewed-snapshot.json',
 ROOT/'reporting/audit_notes/stage_evidence_audit_20261015.md']
selected+=list(OUT.glob('*'))
selected+= [ROOT/p for p in ['.gitignore','.gitattributes','README.md','RESULTS.md','STATUS.md','SYNC_AUDIT.md','CITATIONS.md','reports/stage-b/README.md']]

paths=sorted({p.relative_to(ROOT).as_posix() for p in selected if p.is_file()})
for rel in paths:
    if rel.startswith(('scripts/stage-b3','artifacts/stage_b3')):raise RuntimeError('B3 escaped scope')
    if (ROOT/rel).stat().st_size>=50*1024*1024:raise RuntimeError('Publication file still too large: '+rel)
tmp=ROOT/'tmp';tmp.mkdir(exist_ok=True)
(tmp/'b23_sync_paths.nul').write_bytes(b''.join(p.encode()+b'\0' for p in paths))
write_json(OUT/'PREPARATION.json',{'status':'PREPARED_FOR_REVIEW','date':'2026-10-15','repository':'Maggie1105windy/reasoning-geometry-pilot',
 'selected_files':len(paths),'content_bytes_excluding_preparation':sum((ROOT/p).stat().st_size for p in paths if p!='results/stage-b2_3_sync/PREPARATION.json'),'source_records':len(logical),
 'excluded_files':len(excluded),'archives':len(archives),'scope':'B2.2 dependency and completed B2.3 runs/budgets/trajectory analyses; excludes live B3',
 'html_visual_verification':'Build verified and static figures inspected; live browser inspection unavailable due browser policy.',
 'archive_roundtrip_verified':True})
print(json.dumps({'files':len(paths),'bytes':sum((ROOT/p).stat().st_size for p in paths),'excluded':len(excluded),'archives':len(archives)},ensure_ascii=False))
