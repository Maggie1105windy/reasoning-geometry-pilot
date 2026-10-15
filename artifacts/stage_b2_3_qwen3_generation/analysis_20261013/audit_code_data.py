"""Read-only Stage B2.3 data audit; writes only this analysis directory.

Run: python audit_code_data.py
Reads all generation JSONL/shards and their manifests, recomputes frozen metrics
and all published confidence intervals without requiring GPU/model weights.
"""
from pathlib import Path
import csv, hashlib, json, re, collections
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
REL = ['ABOVE','BELOW','LEFT','RIGHT','UPPER_LEFT','UPPER_RIGHT','LOWER_LEFT','LOWER_RIGHT']
RE = re.compile(r'(?im)^[ \t]*Final[ \t]+answer[ \t]*:[ \t]*(above|below|left|right|upper(?:[ _-]+)left|upper(?:[ _-]+)right|lower(?:[ _-]+)left|lower(?:[ _-]+)right)[ \t]*$')
RMAP = {x.lower().replace('_','-'):x for x in REL}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def parsed(s):
    found=list(RE.finditer(s))
    return RMAP[re.sub(r'[ _-]+','-',found[-1].group(1).strip().lower())] if found else 'INVALID'
def jsave(name,x): (HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v: v.item() if hasattr(v,'item') else str(v))+'\n',encoding='utf-8')
def cmat(rows, pred='parsed_relation'):
    m=np.zeros((8,9),dtype=int)
    for r in rows: m[REL.index(r['gold_relation']), REL.index(r[pred]) if r[pred] in REL else 8]+=1
    return m
def calc(m,metric=None):
    tp=np.diag(m[:,:8]).astype(float); actual=m.sum(axis=1); predicted=m[:,:8].sum(axis=0)
    recall=np.divide(tp,actual,out=np.zeros(8),where=actual!=0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(8),where=(actual+predicted)!=0)
    x={'accuracy':float(tp.sum()/m.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'parse_success_rate':float(m[:,:8].sum()/m.sum())}
    x.update({'recall_'+r:float(v) for r,v in zip(REL,recall)})
    x.update({'prediction_frequency_'+r:float(v/m.sum()) for r,v in zip(REL,predicted)})
    return x[metric] if metric else x
def boundary(r): return bool((r.get('prompt_has_think_open',False) or r.get('generated_has_think_open',False)) and r.get('generated_has_think_close',False))
def summary(rows,protocol,hop):
    m=cmat(rows); x=calc(m); x.update(protocol=protocol,hop=hop,n=len(rows),n_story_groups=len(set(r['story_group_id'] for r in rows)),correct=int(np.diag(m[:,:8]).sum()),parsed=int(m[:,:8].sum()),invalid=int(m[:,8].sum()))
    x['correct_among_parsed']=x['correct']/x['parsed'] if x['parsed'] else None
    x['hit_max']=sum(r.get('hit_max_new_tokens',False) for r in rows)
    x['eos_terminated']=sum(r.get('eos_terminated',False) for r in rows)
    x['hit_max_and_eos']=sum(r.get('hit_max_new_tokens',False) and r.get('eos_terminated',False) for r in rows)
    x['hit_max_and_invalid']=sum(r.get('hit_max_new_tokens',False) and r['parsed_relation']=='INVALID' for r in rows)
    x['non_hitmax_and_invalid']=sum(not r.get('hit_max_new_tokens',False) and r['parsed_relation']=='INVALID' for r in rows)
    x['eos_and_invalid']=sum(r.get('eos_terminated',False) and r['parsed_relation']=='INVALID' for r in rows)
    x['non_hitmax_parsed_wrong']=sum(not r.get('hit_max_new_tokens',False) and r['parsed_relation']!='INVALID' and not r['correct'] for r in rows)
    x['eos_parsed_wrong']=sum(r.get('eos_terminated',False) and r['parsed_relation']!='INVALID' and not r['correct'] for r in rows)
    x['boundary_success']=sum(boundary(r) for r in rows)
    x['legacy_think_success']=sum(r.get('think_parse_success',False) for r in rows)
    x['invalid_prediction_counts']={r:int(m[i,8]) for i,r in enumerate(REL)}
    x['prediction_counts']={r:int(m[:,i].sum()) for i,r in enumerate(REL+['INVALID'])}
    return x

HERE.mkdir(exist_ok=True)
inventory=[]
if (HERE/'audit_file_inventory.csv').is_file():
    inventory=pd.read_csv(HERE/'audit_file_inventory.csv').to_dict('records')
else:
    for base in [OUT,ROOT/'scripts'/'stage-b2_3']:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or HERE in p.parents: continue
            inventory.append(dict(path=str(p),relative=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    pd.DataFrame(inventory).to_csv(HERE/'audit_file_inventory.csv',index=False,encoding='utf-8-sig')
print('Inventory read and hashed:',len(inventory),flush=True)

manifest_path=ROOT/'results'/'stage-b1'/'union_manifest.csv'
primary=pd.read_csv(manifest_path); primary=primary.loc[primary.primary_split.eq('test')].sort_values('row_id')
truth={int(r.row_id):r for r in primary.itertuples(index=False)}
audit={'inventory_files':len(inventory),'inventory_bytes':sum(r['bytes'] for r in inventory),'inventory_by_extension':dict(collections.Counter(Path(r['path']).suffix for r in inventory)), 'scope':'All Stage B2.3 generation JSONL, shards, run manifests, CSV, and metadata; model/GPU not rerun.', 'frozen_primary':{'path':str(manifest_path),'sha256':digest(manifest_path),'n':len(primary),'hop_counts':primary.groupby('computed_hop').size().to_dict(),'story_groups':primary.story_group_id.nunique(),'row_hash':hashlib.sha256(('\n'.join(str(i) for i in primary.row_id)+'\n').encode()).hexdigest()},'runs':[],'data_errors':[],'reported_metric_mismatches':[]}
runs={}
for p in sorted(OUT.rglob('row_generations.jsonl')):
    if HERE in p.parents: continue
    rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    runs[p.parent.relative_to(OUT).as_posix()]=rows
    mf=json.loads((p.parent/'run_manifest.json').read_text(encoding='utf-8'))
    errs=[]; ids=[r['row_id'] for r in rows]
    if ids!=mf['row_ids'] or len(ids)!=len(set(ids)): errs.append('coverage/order/uniqueness')
    if digest(p)!=mf.get('row_generations_sha256'): errs.append('JSONL hash')
    cp=p.parent/'metrics.csv'
    if digest(cp)!=mf.get('metrics_sha256'): errs.append('CSV hash')
    csvrows=pd.read_csv(cp).to_dict('records')
    for r,c in zip(rows,csvrows):
        rid=r['row_id']; t=truth[rid]
        if r['gold_relation']!=t.canonical_relation or r['hop']!=t.computed_hop or r['story_group_id']!=t.story_group_id: errs.append(f'{rid}:frozen truth')
        if parsed(r['final_response_text'])!=r['parsed_relation'] or r['correct']!=(r['parsed_relation']==r['gold_relation']): errs.append(f'{rid}:parser/correct')
        tokenids=r['generated_token_ids']; eos=r['generation_config']['eos_token_id']; eos=[eos] if isinstance(eos,int) else eos
        if len(tokenids)!=r['generation_token_count'] or bool(tokenids and tokenids[-1] in eos)!=r['eos_terminated'] or (len(tokenids)>=r['generation_config']['max_new_tokens'])!=r['hit_max_new_tokens']: errs.append(f'{rid}:token count/termination')
        for flag,tokenid in [('generated_has_think_open',151667),('generated_has_think_close',151668)]:
            if r[flag]!=(tokenid in tokenids): errs.append(f'{rid}:{flag}')
        for key,val in c.items():
            if isinstance(val,float): same=(np.isnan(val) and r.get(key) is None) or np.isclose(val,r[key],rtol=1e-12,atol=1e-12)
            else: same=val==r.get(key)
            if not same: errs.append(f'{rid}:CSV mismatch:{key}')
        sp=p.parent/'shards'/f'row_{rid}.json'
        if json.loads(sp.read_text(encoding='utf-8'))!=r: errs.append(f'{rid}:shard differs')
        if digest(sp)!=mf['shards'][str(rid)]['sha256']: errs.append(f'{rid}:shard hash')
        if 'batch_rng_seed' in r:
            bseed=int(hashlib.sha256((str(r['generation_seed'])+':'+','.join(str(i) for i in r['batch_row_ids'])).encode()).hexdigest()[:8],16)%(2**31-1)
            if bseed!=r['batch_rng_seed'] or bseed!=r['row_rng_seed']: errs.append(f'{rid}:batch seed')
    s=summary(rows,rows[0]['protocol'],'all'); s['path']=str(p); s['generation_seed']=rows[0]['generation_seed']; s['manifest_status']=mf['status']; s['batch_size']=mf.get('batch_size',1); s['errors']=errs
    s['manifest_wall_this_invocation']=mf.get('wall_seconds_this_invocation'); s['sum_attributed_generation_latency']=sum(r['latency_seconds'] for r in rows)
    audit['runs'].append(s)
    audit['data_errors'].extend(str(p)+':'+e for e in errs)
    print('Read and verified',p.parent.relative_to(OUT).as_posix(),len(rows),'errors',len(errs),flush=True)

full={k:runs['full/'+k] for k in ['fg_nonthinking','fg_thinking']}
candidate=pd.read_csv(ROOT/'artifacts'/'stage_b2_2_qwen3'/'native_sum'/'row_scores.csv').drop_duplicates('row_id').sort_values('row_id').to_dict('records')
candidate=[{**r,'parsed_relation':r['prediction_sum'],'correct':r['prediction_sum']==r['gold_relation']} for r in candidate]
audit['full_by_hop']=[summary([r for r in rows if r['hop']==h],rows[0]['protocol'],h) for rows in full.values() for h in range(1,6)]
allframes={'FG-N':full['fg_nonthinking'],'FG-T':full['fg_thinking'],'B2.2 Native-Sum':candidate}
reported=pd.read_csv(OUT/'statistics'/'full_metrics_by_hop.csv')
for label,rows in allframes.items():
    for h in range(1,6):
        metrics=calc(cmat([r for r in rows if r['hop']==h]))
        rr=reported.loc[reported.protocol.eq(label)&reported.hop.eq(h)].iloc[0]
        for key,value in metrics.items():
            if not np.isclose(value,rr[key],rtol=1e-12,atol=1e-12): audit['reported_metric_mismatches'].append(f'{label}:h{h}:{key}')
audit['derived']={label:{metric:float(np.mean([calc(cmat([r for r in rows if r['hop']==h]),metric) for h in [2,3,4,5]])) for metric in ['accuracy','macro_f1','balanced_accuracy','parse_success_rate']} for label,rows in allframes.items()}
stats=pd.read_csv(OUT/'statistics'/'generation_stats.csv')
stats['legacy_think_parse_success_rate']=stats.think_parse_success_rate
for i,s in stats.iterrows():
    rows=[r for r in allframes[s.protocol] if r['hop']==s.hop]
    stats.loc[i,'think_parse_success_rate']=np.mean([boundary(r) if r['enable_thinking'] else r['think_parse_success'] for r in rows])
stats['correction_note']='FG-T uses (prompt_open OR generated_open) AND generated_close; FG-N original flag retained.'
stats.to_csv(HERE/'corrected_generation_stats.csv',index=False,encoding='utf-8-sig')
jsave('audit_code_data.json',audit)
print('Metrics and corrected generation stats saved.',flush=True)

# All published CIs, preserving the original RNG choice calls and ordering.
def groupmat(rows):
    groups=sorted(set(r['story_group_id'] for r in rows)); gmap={g:i for i,g in enumerate(groups)}
    arr=np.zeros((len(groups),8,9),dtype=int)
    for r in rows: arr[gmap[r['story_group_id']],REL.index(r['gold_relation']),REL.index(r['parsed_relation']) if r['parsed_relation'] in REL else 8]+=1
    return arr
def sampled_metric(gmat,rng,metric,indices=None):
    if indices is None: indices=rng.choice(len(gmat),size=len(gmat),replace=True)
    return calc(gmat[indices].sum(axis=0),metric),indices
recomputed=[]
for csvname in ['bootstrap_ci.csv','paired_bootstrap.csv']:
    src=pd.read_csv(OUT/'statistics'/csvname)
    for rr in src.to_dict('records'):
        scope=rr['scope']; hops=[2,3,4,5] if scope=='h2-h5-derived' else [int(scope[1:])]
        metric=rr['metric']; paired=csvname=='paired_bootstrap.csv'
        if paired:
            labels=(rr['left'],rr['right'])
            for label in labels:
                if label.endswith('-Matched') and label not in allframes:
                    allframes[label]=runs['pilot/matched_decoding/'+('fg_nonthinking' if label.startswith('FG-N') else 'fg_thinking')]
            seedname=f"paired:{rr['comparison']}:{'derived' if len(hops)>1 else hops[0]}:{metric}"
        else:
            labels=(rr['protocol'],); seedname=f"single:{rr['protocol']}:{'derived' if len(hops)>1 else hops[0]}:{metric}"
        rng=np.random.default_rng(20260910+int(hashlib.sha256(seedname.encode()).hexdigest()[:8],16))
        matrices={label:{h:groupmat([r for r in allframes[label] if r['hop']==h]) for h in hops} for label in labels}
        values=[]
        for _ in range(2000):
            byhop=[]
            for h in hops:
                lv,ix=sampled_metric(matrices[labels[0]][h],rng,metric)
                byhop.append(sampled_metric(matrices[labels[1]][h],rng,metric,ix)[0]-lv if paired else lv)
            values.append(np.mean(byhop))
        ci=np.quantile(values,[.025,.975]); err=float(max(abs(ci[0]-rr['ci95_low']),abs(ci[1]-rr['ci95_high'])))
        recomputed.append(dict(file=csvname,scope=scope,metric=metric,label=rr.get('comparison',rr.get('protocol')),ci95_low_recomputed=ci[0],ci95_high_recomputed=ci[1],max_absolute_difference=err))
audit['bootstrap_recompute']={'rows':len(recomputed),'max_abs_error':max(r['max_absolute_difference'] for r in recomputed),'matches_within_1e_12':all(r['max_absolute_difference']<1e-12 for r in recomputed)}
pd.DataFrame(recomputed).to_csv(HERE/'bootstrap_recomputed.csv',index=False,encoding='utf-8-sig')
audit['story_group_hop_overlap']=int((primary.groupby('story_group_id').computed_hop.nunique()>1).sum())
snapshot=json.loads((OUT/'audit'/'protected_snapshot.json').read_text(encoding='utf-8'))
missing=[]; changed=[]; relocated=[]
for rel,expected in snapshot['files'].items():
    p=ROOT/rel
    if not p.is_file():
        repl=rel.replace('artifacts/stage_b0_2/','results/stage-b0/b0_2/').replace('artifacts/stage_b1/','results/stage-b1/')
        if repl!=rel and (ROOT/repl).is_file(): p=ROOT/repl; relocated.append({'original':rel,'local':repl})
        else: missing.append(rel); continue
    if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']: changed.append(rel)
audit['protected_current_copy']={'snapshot_files':len(snapshot['files']),'missing_files':missing,'changed_files':changed,'relocated_files':relocated,'note':'Original validator hardcodes original artifacts paths; local archive relocates some folders. This is archive verification, not rerun of original GPU validator.'}
audit['local_model_dir_present']=(ROOT/'models'/'Qwen3-8B').exists()
audit['pilot_derived_denominator']=200
audit['pilot_all_denominator']=250
audit['total_jsonl_rows_read']=sum(len(v) for v in runs.values())
jsave('audit_code_data.json',audit)
print(json.dumps({k:audit[k] for k in ['inventory_files','inventory_bytes','total_jsonl_rows_read','data_errors','reported_metric_mismatches','derived','bootstrap_recompute','story_group_hop_overlap','local_model_dir_present']},ensure_ascii=False,indent=2))
