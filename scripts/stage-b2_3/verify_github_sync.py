"""Verify the actual Git-index blobs and write an exact Git-blob inventory."""
from pathlib import Path
import csv,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/stage-b2_3_sync';TMP=ROOT/'tmp'
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
expected={r['published_path']:r for r in csvrows(OUT/'SYNC_MANIFEST.csv')}
checkpoints=[]
def check(k,b):checkpoints.append({'check':k,'passed':bool(b)})
listing=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT)
entries=[]
for raw in listing.split(b'\0'):
    if not raw:continue
    meta,path=raw.split(b'\t',1);mode,sha,stage=meta.decode().split()
    if stage!='0':raise RuntimeError('Unmerged index entry')
    entries.append((sha,path.decode('utf-8')))
names={p for _,p in entries}
for path in expected:check('manifest_path_staged:'+path,path in names)
check('no_live_b3',not any(p.startswith(('scripts/stage-b3','artifacts/stage_b3')) for p in names))
check('no_internal_sources_text',not any('/sources_text/' in p for p in names))
check('full_budget100_raw_shards',len([p for p in names if p.startswith('artifacts/stage_b2_3_budget100_20261014/budget_') and '/shards/row_' in p and p.endswith('.json')])==300)
for protocol in ['fg_nonthinking','fg_thinking']:
    check('full_2k_shards:'+protocol,len([p for p in names if p.startswith('artifacts/stage_b2_3_qwen3_generation/full/'+protocol+'/shards/') and p.endswith('.json')])==2063)

requests=TMP/'b23_index_blob_requests.txt'
requests.write_text(''.join(sha+'\n' for sha,_ in entries),encoding='ascii')
inventory=[]
with requests.open('rb') as request_file:
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=request_file,stdout=subprocess.PIPE)
    for sha,path in entries:
        header=proc.stdout.readline().decode().strip().split()
        if len(header)!=3 or header[1]!='blob':raise RuntimeError('Unexpected Git object response')
        n=int(header[2]);content=proc.stdout.read(n)
        if proc.stdout.read(1)!=b'\n' or len(content)!=n:raise RuntimeError('Truncated Git blob')
        h=hashlib.sha256(content).hexdigest()
        if path in expected:
            r=expected[path];check('indexed_exact_bytes:'+path,n==int(r['published_bytes']) and h==r['published_sha256'])
        check('git_file_under_50MiB:'+path,n<50*1024*1024)
        if path not in ['FILE_INVENTORY.csv','results/stage-b2_3_sync/INDEX_VERIFICATION.json']:
            inventory.append({'relative_path':path,'size_bytes':n,'sha256':h})
    if proc.wait()!=0:raise RuntimeError('git cat-file failed')
with (ROOT/'FILE_INVENTORY.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']);w.writeheader();w.writerows(inventory)
result={'status':'PASS' if all(c['passed'] for c in checkpoints) else 'FAIL','checks_passed':sum(c['passed'] for c in checkpoints),
 'checks_failed':sum(not c['passed'] for c in checkpoints),'staged_files':len(entries),'manifest_files':len(expected),
 'inventory_scope':'Exact Git-index blob bytes; excludes FILE_INVENTORY.csv and INDEX_VERIFICATION.json to avoid circular self hashes.',
 'index_line_endings':'Audited artifacts and frozen scripts use scoped -text attributes.',
 'checks':checkpoints}
(OUT/'INDEX_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','checks_passed','checks_failed','staged_files','manifest_files']}))
if result['status']!='PASS':raise SystemExit(1)
