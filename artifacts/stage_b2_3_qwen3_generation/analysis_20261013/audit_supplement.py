"""Supplementary read-only audit of aggregate CSV and curated archive mapping."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent; OUT=HERE.parent; ROOT=OUT.parents[1]
REL=['ABOVE','BELOW','LEFT','RIGHT','UPPER_LEFT','UPPER_RIGHT','LOWER_LEFT','LOWER_RIGHT']
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(rows):
    m=np.zeros((8,9),dtype=int)
    for r in rows: m[REL.index(r['gold_relation']),REL.index(r['parsed_relation']) if r['parsed_relation'] in REL else 8]+=1
    tp=np.diag(m[:,:8]); actual=m.sum(1); pred=m[:,:8].sum(0)
    recall=np.divide(tp,actual,out=np.zeros(8,dtype=float),where=actual!=0)
    f1=np.divide(2*tp,actual+pred,out=np.zeros(8,dtype=float),where=(actual+pred)!=0)
    return dict(accuracy=float(tp.sum()/m.sum()),macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),parse_success_rate=float(m[:,:8].sum()/m.sum()))
runs={p.parent.relative_to(OUT).as_posix():[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] for p in OUT.rglob('row_generations.jsonl') if HERE not in p.parents}
candidate=pd.read_csv(ROOT/'artifacts/stage_b2_2_qwen3/native_sum/row_scores.csv').drop_duplicates('row_id').to_dict('records')
candidate=[dict(r,parsed_relation=r['prediction_sum']) for r in candidate]
labels={'B2.2 Native-Sum':candidate,'FG-N':runs['full/fg_nonthinking'],'FG-T':runs['full/fg_thinking']}
errors=[]
for name in ['derived_summary.csv','overall_metrics.csv','pilot_metrics.csv','matched_decoding_metrics.csv']:
    for r in pd.read_csv(OUT/'statistics'/name).to_dict('records'):
        lab=r['protocol']
        if name=='pilot_metrics.csv': rows=runs['pilot/'+('fg_nonthinking' if lab=='FG-N' else 'fg_thinking')+'/seed_'+str(r['generation_seed'])]
        elif name=='matched_decoding_metrics.csv': rows=runs['pilot/matched_decoding/'+('fg_nonthinking' if lab=='FG-N-Matched' else 'fg_thinking')]
        else: rows=labels[lab]
        if r['scope']=='all': expected=metric(rows)
        elif r['scope']=='h2-h5-derived': expected={k:float(np.mean([metric([x for x in rows if x['hop']==h])[k] for h in [2,3,4,5]])) for k in metric(rows)}
        else: expected=metric([x for x in rows if x['hop']==int(r['scope'][1:])])
        errors.extend(f"{name}:{lab}:{r['scope']}:{k}" for k,v in expected.items() if not np.isclose(v,r[k],rtol=1e-12,atol=1e-12))

maps={lab:{r['row_id']:r for r in rows} for lab,rows in labels.items()}
expected=[]
for rowid,c in maps['B2.2 Native-Sum'].items():
    n=maps['FG-N'][rowid]; t=maps['FG-T'][rowid]; gold=c['gold_relation']
    cc=c['parsed_relation']==gold; nc=n['parsed_relation']==gold; tc=t['parsed_relation']==gold
    expected.append(dict(row_id=rowid,story_group_id=c['story_group_id'],hop=c['hop'],gold_relation=gold,candidate_prediction=c['parsed_relation'],fgn_prediction=n['parsed_relation'],fgt_prediction=t['parsed_relation'],candidate_correct=cc,fgn_correct=nc,fgt_correct=tc,relation_type='cardinal' if gold in REL[:4] else 'diagonal',cl_wrong_to_fgn_correct=not cc and nc,cl_correct_to_fgn_wrong=cc and not nc,fgn_wrong_to_fgt_correct=not nc and tc,fgn_correct_to_fgt_wrong=nc and not tc,all_wrong=not cc and not nc and not tc,all_correct=cc and nc and tc))
ex={r['row_id']:r for r in expected}
for name in ['candidate_vs_generation.csv','nonthinking_vs_thinking.csv','error_transitions.csv']:
    actual=pd.read_csv(OUT/'comparisons'/name)
    if len(actual)!=2063 or not actual.row_id.is_unique: errors.append(name+':coverage')
    for r in actual.to_dict('records'):
        errors.extend(f"{name}:{r['row_id']}:{k}" for k,v in r.items() if v!=ex[r['row_id']][k])
for r in pd.read_csv(OUT/'comparisons/error_transition_summary.csv').to_dict('records'):
    part=[x for x in expected if x[r['grouping']]==r[r['grouping']]]
    for k in ['n','cl_wrong_to_fgn_correct','cl_correct_to_fgn_wrong','fgn_wrong_to_fgt_correct','fgn_correct_to_fgt_wrong','all_wrong','all_correct']:
        v=len(part) if k=='n' else sum(x[k] for x in part)
        if v!=r[k]: errors.append('error_transition_summary.csv:'+str(r['grouping'])+':'+k)
conf=[]
for proto,lab in [('fg_nonthinking','FG-N'),('fg_thinking','FG-T')]:
    for scope in ['all','h1','h2','h3','h4','h5']:
        rows=labels[lab] if scope=='all' else [r for r in labels[lab] if r['hop']==int(scope[1:])]
        cm=np.zeros((8,8),dtype=int)
        for r in rows:
            if r['parsed_relation'] in REL: cm[REL.index(r['gold_relation']),REL.index(r['parsed_relation'])]+=1
        count=pd.read_csv(OUT/'statistics/confusion_matrices'/proto/(scope+'_count.csv'),index_col=0).to_numpy()
        normalized=pd.read_csv(OUT/'statistics/confusion_matrices'/proto/(scope+'_row_normalized.csv'),index_col=0).to_numpy()
        norm=np.divide(cm,cm.sum(1)[:,None],out=np.zeros((8,8),float),where=cm.sum(1)[:,None]!=0)
        if not np.array_equal(cm,count) or not np.allclose(norm,normalized,rtol=1e-12,atol=1e-12): errors.append(f'confusion:{proto}:{scope}')
        conf.append(dict(protocol=lab,scope=scope,total_rows=len(rows),matrix_rows=int(cm.sum()),omitted_invalid=len(rows)-int(cm.sum())))

snapshot=json.loads((OUT/'audit/protected_snapshot.json').read_text(encoding='utf-8'))
mapping={'artifacts/stage_b0_2/':'results/stage-b0/b0_2/','artifacts/stage_b0_2c/':'results/stage-b0/b0_2c/','artifacts/stage_b1/':'results/stage-b1/','artifacts/stage_b1_5_1_6/':'results/stage-b1_5-b1_6/','artifacts/stage_b2/':'results/stage-b2/','artifacts/stage_b2_1/':'results/stage-b2_1/'}
missing=[]; changed=[]; relocated=[]
for rel,e in snapshot['files'].items():
    p=ROOT/rel
    if not p.is_file():
        repl=next((rel.replace(old,new,1) for old,new in mapping.items() if rel.startswith(old)),rel)
        p=ROOT/repl
        if not p.is_file() and rel.endswith('.md') and (ROOT/'docs/stage-b'/Path(rel).name).is_file(): p=ROOT/'docs/stage-b'/Path(rel).name
        if not p.is_file(): missing.append(rel); continue
        relocated.append(dict(original=rel,local=p.relative_to(ROOT).as_posix()))
    if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']: changed.append(rel)
a=json.loads((HERE/'audit_code_data.json').read_text(encoding='utf-8'))
a['artifact_files']=int(sum('artifacts/stage_b2_3_qwen3_generation/' in r['relative'].replace('\\','/') for r in pd.read_csv(HERE/'audit_file_inventory.csv').to_dict('records')))
a['source_scripts_files']=a['inventory_files']-a['artifact_files']
primary=pd.read_csv(ROOT/'results/stage-b1/union_manifest.csv'); primary=primary.loc[primary.primary_split.eq('test')].sort_values('row_id')
prompt_audit=json.loads((OUT/'audit/prompt_format_audit.json').read_text(encoding='utf-8'))
contents=[]; raw_mismatch=[]
for r in primary.itertuples(index=False):
    prompt='Story:\n'+'\n'.join(json.loads(r.story))+'\n\nQuestion:\n'+str(r.question).strip()
    if prompt!=r.prompt: raw_mismatch.append(int(r.row_id))
    contents.append(json.dumps([int(r.row_id),prompt+'\n\n'+prompt_audit['only_appended_text']],ensure_ascii=False,separators=(',',':')))
userhash=hashlib.sha256(('\n'.join(contents)+'\n').encode()).hexdigest()
a['independent_raw_prompt_audit']={'raw_prompt_mismatch':raw_mismatch,'all_user_messages_hash':userhash,'matches_recorded':userhash==prompt_audit['all_user_messages_sha256'],'native_template_hash_reconstruction':'Unavailable locally because model tokenizer files are omitted; original validation PASS is recorded evidence.'}
a['aggregate_and_comparison_csv_errors']=errors
a['confusion_matrix_coverage']=conf
a['protected_current_copy']={'snapshot_files':len(snapshot['files']),'missing_files':missing,'changed_files':changed,'relocated_files':relocated,'note':'Mapped curated archive paths; omitted historical checkpoint data and edited sync metadata do not imply original run failed.'}
a['full_h2_h5_pooled']={lab:dict(metric([r for r in rows if r['hop']>1]),n=1351,correct=sum(r['parsed_relation']==r['gold_relation'] for r in rows if r['hop']>1)) for lab,rows in labels.items()}
a['error_transitions_all']={k:sum(r[k] for r in expected) for k in ['cl_wrong_to_fgn_correct','cl_correct_to_fgn_wrong','fgn_wrong_to_fgt_correct','fgn_correct_to_fgt_wrong','all_wrong','all_correct']}
(HERE/'audit_code_data.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'aggregate_csv_errors':errors,'protected':{k:len(v) if isinstance(v,list) else v for k,v in a['protected_current_copy'].items()},'pooled':a['full_h2_h5_pooled'],'transitions':a['error_transitions_all']},ensure_ascii=False,indent=2))
