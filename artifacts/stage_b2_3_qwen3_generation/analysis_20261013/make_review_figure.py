from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parent
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'svg.fonttype': 'none'})
reviewed = json.loads((ROOT / 'audit_code_data.json').read_text(encoding='utf-8'))
assert not reviewed['data_errors']
assert not reviewed['reported_metric_mismatches']
data = pd.DataFrame(reviewed['full_by_hop'])
candidate = pd.read_csv(ARTIFACT / 'statistics/full_metrics_by_hop.csv')
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={'width_ratios':[1.05,1]})
ax = axes[0]
for protocol, color, marker in [('B2.2 Native-Sum','#7a818b','s'),('FG-N','#175a92','o'),('FG-T','#d95d39','^')]:
    if protocol.startswith('B2.2'):
        part = candidate[candidate.protocol.eq(protocol)].sort_values('hop')
    else:
        part = data[data.protocol.eq(protocol)].sort_values('hop')
    ax.plot(part.hop.astype(int), part.accuracy*100, color=color, marker=marker, lw=2, label=protocol)
ax.set(title='A  各 hop 的端到端准确率', xlabel='关系组合步数（hop）', ylabel='准确率（%）', xticks=[1,2,3,4,5], ylim=(0,80))
ax.grid(axis='y', color='#dddddd', alpha=.7)
ax.legend(frameon=False, loc='upper right')
ax.text(0,-.28,'INVALID 计错；候选评分与自由生成是不同协议', transform=ax.transAxes, fontsize=9, color='#4b5563')

part = data[data.protocol.eq('FG-T')].sort_values('hop')
ax = axes[1]
x = np.arange(5)
correct = part.correct.to_numpy()/part.n.to_numpy()*100
wrong = (part.parsed.to_numpy()-part.correct.to_numpy())/part.n.to_numpy()*100
invalid = part.invalid.to_numpy()/part.n.to_numpy()*100
assert np.allclose(correct+wrong+invalid,100)
assert np.array_equal(part.invalid.to_numpy(),part.hit_max_and_invalid.to_numpy())
ax.bar(x, correct, color='#3c8c70', width=.65, label='完成且正确')
ax.bar(x, wrong, bottom=correct, color='#e7b14a', width=.65, label='完成但错误')
ax.bar(x, invalid, bottom=correct+wrong, color='#d66556', width=.65, label='达到上限且无效')
for pos, rate, n in zip(x, invalid, part.n):
    ax.text(pos, 102, f'n={n}', ha='center', fontsize=8.5, color='#4b5563')
    ax.text(pos, correct[pos]+wrong[pos]+invalid[pos]/2, f'{rate:.1f}%', ha='center', va='center', color='white', weight='bold', fontsize=9)
ax.set(title='B  Thinking 的失败来源（2048-token 上限）', ylabel='该 hop 的全部样本（%）', xlabel='关系组合步数（hop）', xticks=x, xticklabels=['h1','h2','h3','h4','h5'], ylim=(0,108))
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5,-.28), ncol=3, fontsize=9)
ax.set_yticks([0,20,40,60,80,100])
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Stage B2.3：协议影响与预算下的输出删失', x=.07, ha='left', fontsize=16, weight='bold')
fig.text(.07,.91,'原始生成记录复核；完成子集有选择偏差，不能替代总体能力评估', fontsize=10, color='#4b5563')
fig.text(.07,.015,'来源：audit_code_data.json（逐条重算）及 statistics/full_metrics_by_hop.csv；复核日期 2026-10-13。', fontsize=8.5, color='#4b5563')
fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.22,wspace=.30)
for ext in ['png','svg']:
    fig.savefig(ROOT/f'StageB2_3_复核概览.{ext}', dpi=190, facecolor='white')
plt.close(fig)
print('Verified figure written: PNG and SVG')
