# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Summarize fixed-run results without selecting checkpoints or more runs."""
from pathlib import Path
import json, statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/recall_pilot'
rows=[json.loads((OUT/f'd{d}-s{s}.json').read_text()) for d in [1,2] for s in [0,1,2]]
summary={'paired_test_differences':[],'group_means':[],'training_seconds_total':sum(sum(r['training_seconds'].values()) for r in rows),'total_training_events':sum(2*r['tokens_per_run'] for r in rows)}
for r in rows:
    summary['paired_test_differences'].append({'depth':r['depth'],'seed':r['seed'],
        'complete_minus_released_cross_entropy':r['test']['complete']['cross_entropy']-r['test']['released']['cross_entropy'],
        'complete_minus_released_accuracy_pp':100*(r['test']['complete']['accuracy']-r['test']['released']['accuracy'])})
for d in [1,2]:
    subset=[r for r in rows if r['depth']==d]
    for mode in ['released','complete']:
        summary['group_means'].append({'depth':d,'mode':mode,'cross_entropy':statistics.mean(r['test'][mode]['cross_entropy'] for r in subset),'accuracy':statistics.mean(r['test'][mode]['accuracy'] for r in subset),'no_adaptation_accuracy':statistics.mean(r['no_write_test'][mode]['accuracy'] for r in subset)})
summary['deep_paired_cross_entropy_mean']=statistics.mean(r['complete_minus_released_cross_entropy'] for r in summary['paired_test_differences'] if r['depth']==2)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
colors=['#0072B2','#D55E00','#009E73']
for r in rows:
    ax=axes[r['depth']-1]
    for mode in ['released','complete']:
        ax.plot([h['step'] for h in r['history']],[h['validation'][mode]['cross_entropy'] for h in r['history']],color=colors[r['seed']],linestyle='-' if mode=='released' else '--',lw=1.8 if mode=='released' else 1.3,label=f'Seed {r["seed"]}, {mode}')
for d,ax in enumerate(axes,1):
    ax.set_title(f'{d} fast factor'+('s' if d>1 else ''))
    ax.set_xlabel('Outer updates');ax.set_ylabel('Validation cross-entropy / query')
    ax.set_ylim(-0.04,2.2);ax.grid(alpha=.2)
axes[1].legend(fontsize=7,loc='lower left')
fig.suptitle('Held-out associative recall: fixed 600-step pilot',fontsize=12)
fig.savefig(OUT/'validation_curves.png',dpi=180)
print(json.dumps(summary,indent=2))
