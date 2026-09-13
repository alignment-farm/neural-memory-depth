# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2"]
# ///
import hashlib,itertools,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/embedding_swap';SEEDS=list(range(10,20))
GROUPS=['query','key','value'];CONFIGS=[dict(zip(GROUPS,v)) for v in itertools.product([11,16],repeat=3)]
def label(c):return f"q{c['query']}-k{c['key']}-v{c['value']}"
def main():
    records={(label(c),t):json.loads((OUT/f'{label(c)}-t{t}.json').read_text()) for c in CONFIGS for t in SEEDS}
    groups=[];individual=[];contrasts=[]
    for c in CONFIGS:
        for mode in ['released','complete']:
            vals=[]
            for t in SEEDS:
                r=records[label(c),t];h=r['history'];ys=[x['validation'][mode]['cross_entropy'] for x in h];steps=[x['step'] for x in h]
                metric={'configuration':c,'label':label(c),'stream_seed':t,'mode':mode,'origin':r['origin'],**r['test'][mode],
                    'final_validation_low_loss':ys[-1]<.1,'final_validation_ce':ys[-1],
                    'validation_auc':sum((b-a)*(x+y)/2 for a,b,x,y in zip(steps,steps[1:],ys,ys[1:]))/(steps[-1]-steps[0]),
                    'no_adaptation_accuracy':r['no_write_test'][mode]['accuracy']}
                vals.append(metric);individual.append(metric)
            groups.append({'configuration':c,'label':label(c),'mode':mode,'mean_test_ce':statistics.mean(r['cross_entropy'] for r in vals),
                'mean_test_accuracy':statistics.mean(r['accuracy'] for r in vals),'low_loss_streams':sum(r['final_validation_low_loss'] for r in vals),
                'mean_validation_auc':statistics.mean(r['validation_auc'] for r in vals)})
    for mode in ['released','complete']:
        def ce(c,t):return records[label(c),t]['test'][mode]['cross_entropy']
        for group in GROUPS:
            others=[g for g in GROUPS if g!=group];rows=[]
            for t in SEEDS:
                diffs=[]
                for v in itertools.product([11,16],repeat=2):
                    c=dict(zip(others,v));diffs.append(ce(dict(c,**{group:11}),t)-ce(dict(c,**{group:16}),t))
                bad={g:16 for g in GROUPS};good={g:11 for g in GROUPS}
                rows.append({'stream_seed':t,'marginal_11_minus_16':statistics.mean(diffs),
                    'replace_group_in_16_background':ce(dict(bad,**{group:11}),t)-ce(bad,t),
                    'replace_group_in_11_background':ce(dict(good,**{group:16}),t)-ce(good,t)})
            contrasts.append({'mode':mode,'group':group,'per_stream':rows,
                'means':{k:statistics.mean(r[k] for r in rows) for k in rows[0] if k!='stream_seed'},
                'marginal_range':[min(r['marginal_11_minus_16'] for r in rows),max(r['marginal_11_minus_16'] for r in rows)]})
    new=[r for r in records.values() if r['origin']=='new']
    report={'models_total':160,'new_models':120,'reused_models':40,
        'new_training_seconds':sum(sum(r['training_seconds'].values()) for r in new),
        'new_training_events':sum(2*r['tokens_per_run'] for r in new),'group_summaries':groups,'contrasts':contrasts,
        'individual_metrics':individual,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');norm=LogNorm(vmin=.0001,vmax=2.5)
    for ax,mode in zip(axes,['released','complete']):
        mat=[[records[label(c),t]['test'][mode]['cross_entropy'] for t in SEEDS] for c in CONFIGS]
        im=ax.imshow(mat,norm=norm,cmap='viridis',aspect='auto')
        ax.set_xticks(range(10),SEEDS);ax.set_yticks(range(8),[f"{c['query']} / {c['key']} / {c['value']}" for c in CONFIGS])
        ax.set_xlabel('Training-stream seed');ax.set_ylabel('Donor: query / key / value');ax.set_title(mode.title())
        for i,row in enumerate(mat):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.0e}' if v<.001 else f'{v:.3f}' if v<.01 else f'{v:.2f}',ha='center',va='center',fontsize=7,color='black' if norm(v)>.65 else 'white')
    fig.colorbar(im,ax=axes,shrink=.75,label='Final test CE (log color scale)')
    fig.suptitle('Query / key / value swaps: memory and readout fixed at donor 16')
    fig.savefig(OUT/'swap_loss.png',dpi=180)
    print(json.dumps({k:report[k] for k in ['new_training_seconds','group_summaries','contrasts']},indent=2))
if __name__=='__main__':main()
