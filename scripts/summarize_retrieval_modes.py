# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2"]
# ///
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/class_collapse'
s=json.loads((OUT/'retrieval_modes.json').read_text());v=json.loads((OUT/'diagnostics.json').read_text())
lookup={(r['label'],r['stream_seed'],r['mode']):r for r in v['rows']}
examples=[next(r for r in s['rows'] if r['value_cluster_fit'] and r['ce']>=.1),next(r for r in s['rows'] if r['key_cluster_fit'] and r['ce']>=.1)]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,r,title in zip(axes,examples,['Value-confusion example','Key-confusion example']):
    value=lookup[r['label'],r['stream_seed'],r['mode']]
    ax.bar([i-.18 for i in range(8)],value['per_class_cross_entropy'],width=.36,label='By true value',color='#0072B2')
    ax.bar([i+.18 for i in range(8)],r['per_key_ce'],width=.36,label='By queried key',color='#D55E00')
    ax.set_xticks(range(8));ax.set_xlabel('Value-class / query-key index');ax.set_ylabel('Mean test cross-entropy')
    ax.set_title(f"{title}\n{r['label']}, stream {r['stream_seed']}, {r['mode']}",fontsize=10)
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Partial retrieval: errors can follow value identity or query-key identity')
fig.savefig(OUT/'retrieval_examples.png',dpi=180)
