"""Read-only derivation of B2.2/B2.3 report figures; never edits source runs."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
APP = ROOT.parent / 'stage-b-research-report-app'
FIG = OUT / 'figures'; DATA = OUT / 'figure_data'
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
B23 = ROOT / 'artifacts/stage_b2_3_qwen3_generation'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
 'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11,'axes.titlesize':14,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
 'savefig.facecolor':'white','axes.titlepad':15})
C = ['#64748b','#137c86','#c77926']
sources = []
def read_csv(p):
    sources.append(p)
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
def write_csv(name,rows):
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

metrics=read_csv(B23/'statistics/full_metrics_by_hop.csv')
gen=read_csv(B23/'analysis_20261013/corrected_generation_stats.csv')
shown=[]
fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.2,1]})
for protocol,color,label,style in zip(['B2.2 Native-Sum','FG-N','FG-T'],C,
 ['候选评分 Native-Sum','非思考自由生成 FG-N','思考自由生成 FG-T（2K）'],['--','-','-']):
    rows=[r for r in metrics if r['protocol']==protocol]
    rows.sort(key=lambda r:int(r['hop']))
    xs=[int(r['hop']) for r in rows];ys=[float(r['accuracy']) for r in rows]
    axs[0].plot(xs,ys,'o',color=color,linestyle=style,label=label,lw=2)
    for r in rows:
        shown.append({'protocol':protocol,'hop':int(r['hop']),'n':int(r['n']),
        'accuracyRate':float(r['accuracy']),'parseRate':float(r['parse_success_rate'])})
axs[0].axhline(.125,color='#b4bdc9',ls=':',label='八类均匀随机 12.5%')
axs[0].set(xticks=range(1,6),xticklabels=[f'h{i}' for i in range(1,6)],ylim=(0,.8),ylabel='准确率',title='A  同一冻结 test：2063 题')
axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[0].legend(fontsize=9,loc='upper right',frameon=False)
rows=[r for r in gen if r['protocol']=='FG-T']
parse=[float(next(m for m in metrics if m['protocol']=='FG-T' and m['hop']==r['hop'])['parse_success_rate']) for r in rows]
trunc=[float(r['truncation_rate']) for r in rows]
x=np.arange(5)
axs[1].bar(x,parse,color=C[1],label='可解析且自然结束')
axs[1].bar(x,trunc,bottom=parse,color=C[2],label='截断、无最终答案')
for i,p in enumerate(parse):axs[1].text(i,p+.02,f'{p:.1%}',ha='center',fontsize=10)
axs[1].set(xticks=x,xticklabels=[f'h{i}' for i in range(1,6)],ylim=(0,1.08),title='B  FG-T 低分伴随严重完成失败',ylabel='各 hop 题目占比')
axs[1].yaxis.set_major_formatter(PercentFormatter(1));axs[1].legend(loc='upper center',fontsize=9,frameon=True,framealpha=.96)
fig.text(.06,-.015,'各 hop n=712 / 505 / 422 / 282 / 142；FG-T 全体 80.71% 截断。2K 结果不能识别充分预算下的 thinking 纯效应。',fontsize=10,color='#586474')
fig.tight_layout(w_pad=3)
save(fig,'fig01_full_behavior_and_completion')
write_csv('full_behavior.csv',shown)

# Include INVALID: reconstruct all 8x9 counts directly from every frozen output.
labels=['ABOVE','BELOW','LEFT','RIGHT','UPPER_LEFT','UPPER_RIGHT','LOWER_LEFT','LOWER_RIGHT']
cn=['上','下','左','右','左上','右上','左下','右下','无效']
heat=[]; raw_counts=[]
fig,axs=plt.subplots(1,2,figsize=(12,5.3))
for ax,key,title in zip(axs,['fg_nonthinking','fg_thinking'],['FG-N：含格式失败','FG-T（2K）：无效列主导']):
    p=B23/f'full/{key}/row_generations.jsonl';sources.append(p)
    counts=np.zeros((8,9),dtype=int)
    with p.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); gold=labels.index(r['gold_relation']); pred=r['parsed_relation']
            col=labels.index(pred) if pred in labels else 8; counts[gold,col]+=1
    assert counts.sum()==2063
    norm=counts/counts.sum(axis=1,keepdims=True)
    im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(8):
        for j in range(9):
            ax.text(j,i,f'{norm[i,j]*100:.0f}',ha='center',va='center',fontsize=9,color='white' if norm[i,j]>.55 else '#17334e')
            heat.append({'protocol':'FG-N' if key=='fg_nonthinking' else 'FG-T','gold':labels[i],
            'prediction':labels[j] if j<8 else 'INVALID','count':int(counts[i,j]),'rowRate':float(norm[i,j])})
    raw_counts.append({'protocol':key,'total':int(counts.sum()),'invalid':int(counts[:,8].sum())})
    ax.set(xticks=range(9),xticklabels=cn,yticks=range(8),yticklabels=cn[:8],title=title,xlabel='模型预测',ylabel='标准答案')
fig.subplots_adjust(wspace=.25,bottom=.15,right=.90)
cax=fig.add_axes([.93,.19,.015,.65]);fig.colorbar(im,cax=cax,format=PercentFormatter(1))
fig.text(.08,.035,'每一行按该 gold 类全部题目归一化；格中为百分比（四舍五入）。无效预测保留在第 9 列，未只筛选成功解析样本。',fontsize=10,color='#586474')
save(fig,'fig02_confusion_with_invalid')
write_csv('confusion_8x9.csv',heat)

# A real audited example: correct visible coordinates, reversed answer direction.
fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),gridspec_kw={'width_ratios':[1.05,1]})
coords={'D':(0,0),'H':(0,1),'U':(-1,0),'P':(-2,-1),'G':(-3,-1),'B':(-3,-2)}
ax=axs[0]
for a,b in [('D','H'),('H','U'),('U','P'),('P','G'),('G','B')]:
    ax.plot([coords[a][0],coords[b][0]],[coords[a][1],coords[b][1]],c='#bccad0',lw=2)
for entity,(xv,yv) in coords.items():
    ax.scatter([xv],[yv],s=330,c=C[1] if entity in ['D','B'] else '#d8e6e7',zorder=3)
    ax.text(xv,yv,entity,ha='center',va='center',color='white' if entity in ['D','B'] else '#244c50',weight='bold')
ax.add_patch(FancyArrowPatch(coords['B'],coords['D'],arrowstyle='-|>',mutation_scale=20,color='#137c86',lw=2,connectionstyle='arc3,rad=-.2',shrinkA=12,shrinkB=12))
ax.text(-1.5,-1.45,'D 相对 B = (3, 2)\n应答：右上 UPPER_RIGHT',ha='center',color=C[1],fontsize=11,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':3})
ax.set(xticks=range(-3,1),yticks=range(-2,2),xlim=(-3.7,.8),ylim=(-2.6,1.6),title='A  可见末尾坐标支持正确关系',xlabel='x',ylabel='y');ax.grid(alpha=.16);ax.set_aspect('equal')
ax=axs[1];ax.axis('off');ax.set_title('B  最终回答却把查询方向反转')
text=('真实个案 row 90898 · h5\n\n查询：D 相对 B 在什么方向？\n\n轨迹末尾：D=(0,0)，B=(−3,−2)\n模型最终：LOWER_LEFT（左下）\n\n8K / 16K / 32K 完整输出相同，均错误。\n\n这说明可见坐标与最终作答可以分离。\n它尚未证明隐藏状态中同样存在、\n或实际使用了正确关系。')
ax.text(.02,.92,text,va='top',fontsize=12,linespacing=1.65,color='#2b3946')
fig.tight_layout(w_pad=3);save(fig,'fig07_state_vs_answer_case')

# Proposed operational design, explicitly separate the two time axes.
fig,ax=plt.subplots(figsize=(12,4.8));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
ax.text(.1,4.7,'下一阶段研究设计：把“何时形成状态”与“哪个训练阶段改变”分开',fontsize=16,weight='bold',color='#19323d')
def box(x,y,w,h,t,fc='#edf5f5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10,rounding_size=.12',facecolor=fc,edgecolor='#b6cccf'))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
for i,t in enumerate(['提示末状态\n已有何种信息？','t_state\n正确关系首次可读','t_commit\n关系能否稳定保持','最终答案\n是否正确使用？']):box(.3+i*3,2.7,2.35,.9,t)
for i in range(3):ax.annotate('',xy=(3.15+i*3,3.15),xytext=(2.75+i*3,3.15),arrowprops={'arrowstyle':'->','color':C[1],'lw':2})
ax.text(.3,4.0,'生成时间轴：固定权重，观察状态形成、保持与使用',fontsize=12,color=C[1])
for i,t in enumerate(['同谱系 Base','SFT','DPO','RLVR / 中间检查点']):box(.3+i*3,.75,2.35,.75,t,fc='#f4f1e8')
for i in range(3):ax.annotate('',xy=(3.15+i*3,1.12),xytext=(2.75+i*3,1.12),arrowprops={'arrowstyle':'->','color':C[2],'lw':2})
ax.text(.3,1.95,'训练时间轴：用共享轨迹比较 token 级 Δlog p、几何与干预效应',fontsize=12,color='#906022')
ax.text(.3,.15,'证据递进：自然行为 → 结构状态可读 → 稳定性／泛化 → 目标特异的因果干预。此图是待执行方案，不是已有实验结果。',fontsize=10,color='#586474')
save(fig,'fig08_two_time_axes_design')

payload={'full_behavior':shown,'confusion_8x9':heat,'raw_counts':raw_counts,
 'sources':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
(DATA/'stage_derived.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'figures':4,'raw_counts':raw_counts,'source_files':len(sources)},ensure_ascii=False))
