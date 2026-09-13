# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Prespecified refresh contrasts and descriptive plots; no checkpoint selection."""
import hashlib,json,random,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/refresh_comparison'
CHUNKS=[1,2,4,8];SEEDS=list(range(10,20));MODES=['released','complete']

def interval(values):
    gen=random.Random(20260914)
    samples=sorted(statistics.mean(gen.choices(values,k=len(values))) for _ in range(10000))
    def quantile(p):
        i=(len(samples)-1)*p;lo=int(i);hi=min(lo+1,len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(i-lo)
    return [quantile(.025),quantile(.975)]

def main():
    rows={(c,d,s):json.loads((OUT/f'c{c}-d{d}-s{s}.json').read_text()) for c in CHUNKS for d in [1,2] for s in SEEDS}
    groups=[];contrasts=[];individual=[]
    for (c,d,s),r in rows.items():
        for mode in MODES:
            h=r['history'];steps=[x['step'] for x in h];ys=[x['validation'][mode]['cross_entropy'] for x in h]
            auc=sum((b-a)*(x+y)/2 for a,b,x,y in zip(steps,steps[1:],ys,ys[1:]))/(steps[-1]-steps[0])
            reached=next((step for step,y in zip(steps,ys) if y<.1),None)
            individual.append({'chunk':c,'depth':d,'seed':s,'mode':mode,**r['test'][mode],
                'validation_normalized_auc':auc,'first_validation_ce_below_0_1':reached,
                'threshold_right_censored':reached is None,'training_seconds':r['training_seconds'][mode],
                'no_adaptation_accuracy':r['no_write_test'][mode]['accuracy']})
    for c in CHUNKS:
        for d in [1,2]:
            for mode in MODES:
                selected=[r for r in individual if (r['chunk'],r['depth'],r['mode'])==(c,d,mode)]
                groups.append({'chunk':c,'depth':d,'mode':mode,
                    'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in selected),
                    'sd_cross_entropy':statistics.stdev(r['cross_entropy'] for r in selected),
                    'mean_accuracy':statistics.mean(r['accuracy'] for r in selected),
                    'mean_validation_auc':statistics.mean(r['validation_normalized_auc'] for r in selected),
                    'threshold_reached_seeds':sum(not r['threshold_right_censored'] for r in selected),
                    'mean_training_seconds':statistics.mean(r['training_seconds'] for r in selected),
                    'mean_no_adaptation_accuracy':statistics.mean(r['no_adaptation_accuracy'] for r in selected)})
    for mode in MODES:
        for baseline in [8,4]:
            changes=[]
            for s in SEEDS:
                shallow=rows[1,1,s]['test'][mode]['cross_entropy']-rows[baseline,1,s]['test'][mode]['cross_entropy']
                deep=rows[1,2,s]['test'][mode]['cross_entropy']-rows[baseline,2,s]['test'][mode]['cross_entropy']
                changes.append({'seed':s,'shallow_change':shallow,'deep_change':deep,'interaction':deep-shallow})
            vals=[x['interaction'] for x in changes]
            contrasts.append({'mode':mode,'small_chunk':1,'baseline_chunk':baseline,
                'role':'primary' if mode=='released' and baseline==8 else 'corroborating' if baseline==8 else 'secondary',
                'per_seed':changes,'mean_interaction':statistics.mean(vals),'bootstrap_95_percentile_interval':interval(vals),
                'seeds_negative_interaction':sum(v<0 for v in vals)})
    paired=[]
    for (c,d,s),r in rows.items():
        paired.append({'chunk':c,'depth':d,'seed':s,
            'complete_minus_released_ce':r['test']['complete']['cross_entropy']-r['test']['released']['cross_entropy'],
            'complete_minus_released_accuracy_pp':100*(r['test']['complete']['accuracy']-r['test']['released']['accuracy'])})
    summary={'models':160,'total_training_events':sum(2*r['tokens_per_run'] for r in rows.values()),
        'training_seconds_total':sum(sum(r['training_seconds'].values()) for r in rows.values()),
        'group_summaries':groups,'contrasts':contrasts,'individual_metrics':individual,'derivative_pairs':paired,
        'bootstrap':{'unit':'paired training seed','samples':10000,'seed':20260914,'interval':'descriptive 95% percentile'},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharex=True,sharey=True)
    for d in [1,2]:
        for j,mode in enumerate(MODES):
            ax=axes[d-1,j]
            for c,color in zip(CHUNKS,colors):
                histories=[rows[c,d,s]['history'] for s in SEEDS]
                steps=[x['step'] for x in histories[0]]
                values=[[h[i]['validation'][mode]['cross_entropy'] for h in histories] for i in range(len(steps))]
                ax.plot(steps,[statistics.mean(v) for v in values],color=color,label=f'Chunk {c}')
            ax.set_title(f'Depth {d}, {mode}');ax.grid(alpha=.2)
            ax.set_xlabel('Outer updates');ax.set_ylabel('Mean validation CE / query')
    axes[0,0].legend(fontsize=8);fig.suptitle('Gradient refresh: ten paired seeds, fixed 600 updates')
    fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
    for ax,mode in zip(axes,MODES):
        for d,color in [(1,'#0072B2'),(2,'#D55E00')]:
            for s in SEEDS:
                ax.plot(CHUNKS,[rows[c,d,s]['test'][mode]['cross_entropy'] for c in CHUNKS],color=color,alpha=.18,lw=.8)
            ax.plot(CHUNKS,[statistics.mean(rows[c,d,s]['test'][mode]['cross_entropy'] for s in SEEDS) for c in CHUNKS],color=color,marker='o',lw=2.2,label=f'Depth {d} mean')
        ax.set_xticks(CHUNKS);ax.set_xlabel('Chunk size');ax.set_ylabel('Final test CE / query');ax.set_title(mode.title());ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Final held-out loss: faint lines connect the same seed')
    fig.savefig(OUT/'final_loss.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['models','total_training_events','training_seconds_total','group_summaries','contrasts']},indent=2))
if __name__=='__main__':main()
