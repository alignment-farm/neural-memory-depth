# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Descriptive balanced two-factor decomposition of the frozen seed grid."""
import hashlib,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/seed_separation'
SEEDS=list(range(10,20));MODES=['released','complete']

def decompose(matrix):
    n=len(matrix);m=len(matrix[0]);grand=statistics.mean(v for row in matrix for v in row)
    rows=[statistics.mean(row) for row in matrix]
    cols=[statistics.mean(matrix[i][j] for i in range(n)) for j in range(m)]
    ss_init=m*sum((r-grand)**2 for r in rows)
    ss_stream=n*sum((c-grand)**2 for c in cols)
    ss_interaction=sum((matrix[i][j]-rows[i]-cols[j]+grand)**2 for i in range(n) for j in range(m))
    ss_total=sum((v-grand)**2 for row in matrix for v in row)
    assert math.isclose(ss_init+ss_stream+ss_interaction,ss_total,rel_tol=1e-10,abs_tol=1e-12)
    terms={'initialization':ss_init,'training_stream':ss_stream,'interaction':ss_interaction}
    return {'grand_mean':grand,'row_means':rows,'column_means':cols,'sum_squares_total':ss_total,
        'sum_squares':terms,'fractions':{k:v/ss_total if ss_total else 0 for k,v in terms.items()}}

def main():
    checks=[([[0,0],[1,1]],{'initialization':1.,'training_stream':0.,'interaction':0.}),
        ([[0,1],[0,1]],{'initialization':0.,'training_stream':1.,'interaction':0.}),
        ([[0,1],[1,0]],{'initialization':0.,'training_stream':0.,'interaction':1.}),
        ([[0,1],[2,3]],{'initialization':.8,'training_stream':.2,'interaction':0.})]
    for matrix,expected in checks:
        found=decompose(matrix)['fractions']
        assert all(math.isclose(found[k],v,abs_tol=1e-12) for k,v in expected.items())
    records={(i,t):json.loads((OUT/f'i{i}-t{t}.json').read_text()) for i in SEEDS for t in SEEDS}
    individual=[];analyses={};pairs=[]
    for (i,t),r in records.items():
        for mode in MODES:
            h=r['history'];steps=[x['step'] for x in h];ys=[x['validation'][mode]['cross_entropy'] for x in h]
            area=sum((b-a)*(x+y)/2 for a,b,x,y in zip(steps,steps[1:],ys,ys[1:]))/(steps[-1]-steps[0])
            reached=next((step for step,y in zip(steps,ys) if y<.1),None)
            individual.append({'init_seed':i,'stream_seed':t,'mode':mode,'origin':r['origin'],**r['test'][mode],
                'final_validation_ce':ys[-1],'final_validation_low_loss':ys[-1]<.1,
                'first_validation_ce_below_0_1':reached,'threshold_right_censored':reached is None,
                'validation_normalized_auc':area,'training_seconds':r['training_seconds'][mode],
                'no_adaptation_accuracy':r['no_write_test'][mode]['accuracy']})
        pairs.append({'init_seed':i,'stream_seed':t,
            'complete_minus_released_test_ce':r['test']['complete']['cross_entropy']-r['test']['released']['cross_entropy'],
            'released_low_loss':r['history'][-1]['validation']['released']['cross_entropy']<.1,
            'complete_low_loss':r['history'][-1]['validation']['complete']['cross_entropy']<.1})
    for mode in MODES:
        cells={(r['init_seed'],r['stream_seed']):r for r in individual if r['mode']==mode}
        matrix=[[cells[i,t]['cross_entropy'] for t in SEEDS] for i in SEEDS]
        low=[[cells[i,t]['final_validation_low_loss'] for t in SEEDS] for i in SEEDS]
        row_summaries=[];column_summaries=[]
        for i in SEEDS:
            subset=[cells[i,t] for t in SEEDS];diag=cells[i,i]['final_validation_low_loss']
            row_summaries.append({'init_seed':i,'diagonal_low_loss':diag,
                'low_loss_streams':sum(r['final_validation_low_loss'] for r in subset),
                'mean_test_ce':statistics.mean(r['cross_entropy'] for r in subset),
                'min_test_ce':min(r['cross_entropy'] for r in subset),'max_test_ce':max(r['cross_entropy'] for r in subset),
                'mean_test_accuracy':statistics.mean(r['accuracy'] for r in subset),
                'rescued_off_diagonal':sum(cells[i,t]['final_validation_low_loss'] for t in SEEDS if t!=i) if not diag else 0,
                'disrupted_off_diagonal':sum(not cells[i,t]['final_validation_low_loss'] for t in SEEDS if t!=i) if diag else 0})
        for t in SEEDS:
            subset=[cells[i,t] for i in SEEDS]
            column_summaries.append({'stream_seed':t,'low_loss_initializations':sum(r['final_validation_low_loss'] for r in subset),
                'mean_test_ce':statistics.mean(r['cross_entropy'] for r in subset),
                'mean_test_accuracy':statistics.mean(r['accuracy'] for r in subset)})
        analyses[mode]={'test_ce_matrix':matrix,'final_validation_low_loss_matrix':low,
            'ce_decomposition':decompose(matrix),'low_loss_cells':sum(sum(row) for row in low),
            'mean_accuracy':statistics.mean(r['accuracy'] for r in cells.values()),
            'mean_validation_auc':statistics.mean(r['validation_normalized_auc'] for r in cells.values()),
            'initialization_summaries':row_summaries,'stream_summaries':column_summaries}
    new=[r for r in records.values() if r['origin']=='new']
    summary={'decomposition_known_case_checks_passed':4,'initialization_seeds':SEEDS,'stream_seeds':SEEDS,'models_total':200,'new_models':2*len(new),
        'reused_models':200-2*len(new),'new_training_events':sum(2*r['tokens_per_run'] for r in new),
        'new_training_seconds':sum(sum(r['training_seconds'].values()) for r in new),
        'analyses':analyses,'individual_metrics':individual,'derivative_pairs':pairs,
        'derivative_mean_test_ce_change':statistics.mean(r['complete_minus_released_test_ce'] for r in pairs),
        'complete_rescues':sum(not r['released_low_loss'] and r['complete_low_loss'] for r in pairs),
        'complete_disruptions':sum(r['released_low_loss'] and not r['complete_low_loss'] for r in pairs),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    norm=LogNorm(vmin=0.0001,vmax=2.5)
    for ax,mode in zip(axes,MODES):
        matrix=analyses[mode]['test_ce_matrix'];im=ax.imshow(matrix,norm=norm,cmap='viridis',aspect='equal')
        ax.set_xticks(range(10),SEEDS);ax.set_yticks(range(10),SEEDS)
        ax.set_xlabel('Training-stream seed');ax.set_ylabel('Whole-model initialization seed');ax.set_title(mode.title())
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):
                label=f'{value:.0e}' if value<.001 else f'{value:.3f}' if value<.01 else f'{value:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=7,color='black' if norm(value)>.65 else 'white')
    fig.colorbar(im,ax=axes,shrink=.7,label='Final test CE / query (log color scale)')
    fig.suptitle('Initialization × training stream: fixed 600-update outcomes',fontsize=13)
    fig.savefig(OUT/'crossed_loss.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    left=[0.,0.]
    for factor,color in [('initialization','#0072B2'),('training_stream','#D55E00'),('interaction','#009E73')]:
        values=[100*analyses[m]['ce_decomposition']['fractions'][factor] for m in MODES]
        ax.barh(MODES,values,left=left,color=color,label=factor.replace('_',' ').title())
        for j,value in enumerate(values):
            if value>=4:ax.text(left[j]+value/2,j,f'{value:.1f}%',ha='center',va='center',color='white')
        left=[a+b for a,b in zip(left,values)]
    ax.set_xlim(0,100);ax.set_xlabel('Share of final test CE sum of squares (%)')
    ax.set_title('Descriptive variation across the fixed 10 × 10 grid')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.15),ncol=3,fontsize=8)
    fig.savefig(OUT/'variation_decomposition.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['models_total','new_models','reused_models','new_training_events','new_training_seconds','derivative_mean_test_ce_change','complete_rescues','complete_disruptions']},indent=2))
    for mode in MODES:
        print(mode,json.dumps({k:analyses[mode][k] for k in ['ce_decomposition','low_loss_cells','initialization_summaries','stream_summaries']},indent=2))
if __name__=='__main__':main()
