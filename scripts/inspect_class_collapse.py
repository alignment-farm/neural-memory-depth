# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Post-hoc checkpoint diagnostic of unresolved value classes; no training."""
import hashlib,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
import run_embedding_swap as study
OUT=study.ROOT/'analysis/class_collapse'

def pair_distance(matrix,indices):
    if len(indices)<2:return None
    x=matrix[indices];d=torch.cdist(x,x);return d[~torch.eye(len(x),dtype=torch.bool)].mean().item()

@torch.no_grad()
def inspect(record,mode,data):
    info=record['checkpoints'][mode];path=study.ROOT/info['path'];assert study.sha(path)==info['sha256']
    model=study.RecallModel(2,4);model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
    assert study.state_hash(model)==record['final_hashes'][mode]
    memory=[]
    handle=model.norm.register_forward_pre_hook(lambda module,args:memory.append(args[0].detach()))
    counts=torch.zeros(8,dtype=torch.float64);ce=torch.zeros(8,dtype=torch.float64);correct=torch.zeros(8,dtype=torch.float64)
    probs=torch.zeros(8,8,dtype=torch.float64);conf=torch.zeros(8,8,dtype=torch.int64);means=torch.zeros(8,16,dtype=torch.float64)
    norm2=torch.zeros(8,dtype=torch.float64);gen=torch.Generator().manual_seed(8200)
    for start in range(0,len(data),128):
        batch=study.batch_for(data[start:start+128],gen);memory.clear()
        logits=model(batch,mode);raw=memory[-1].reshape(-1,16).double();target=batch[-1].reshape(-1)
        pred=logits.reshape(-1,8);p=pred.softmax(-1).double();loss=F.cross_entropy(pred,target,reduction='none').double();choice=pred.argmax(-1)
        for value in range(8):
            mask=target==value;n=mask.sum();counts[value]+=n;ce[value]+=loss[mask].sum();correct[value]+=(choice[mask]==value).sum()
            probs[value]+=p[mask].sum(0);means[value]+=raw[mask].sum(0);norm2[value]+=raw[mask].square().sum(-1).sum()
            conf[value]+=torch.bincount(choice[mask],minlength=8)
    handle.remove();ce/=counts;probs/=counts[:,None];means/=counts[:,None]
    accuracy=correct/counts;within=norm2/counts-means.square().sum(-1)
    unresolved=(ce>=.1).nonzero().flatten().tolist();resolved=(ce<.1).nonzero().flatten().tolist();n=len(unresolved)
    ideal=torch.zeros(8,dtype=torch.float64)
    if n:ideal[unresolved]=1/n
    expected=n/8*math.log(n) if n else 0.
    v=model.v.weight.detach().double()
    result={'configuration':record['configuration'],'label':record['label'],'stream_seed':record['stream_seed'],'mode':mode,
        'checkpoint_sha256':info['sha256'],'per_class_queries':counts.tolist(),'per_class_cross_entropy':ce.tolist(),
        'per_class_accuracy':accuracy.tolist(),'mean_probabilities_by_true_class':probs.tolist(),'confusion_counts':conf.tolist(),
        'overall_ce':ce.mean().item(),'recorded_ce':record['test'][mode]['cross_entropy'],
        'unresolved_classes_ce_at_least_0_1':unresolved,'unresolved_count':n,
        'uniform_single_cluster_expected_ce':expected,'observed_minus_cluster_ce':ce.mean().item()-expected,
        'unresolved_probability_l1_to_uniform':(probs[unresolved]-ideal).abs().sum(-1).mean().item() if n else None,
        'unresolved_probability_mass_on_cluster':probs[unresolved][:,unresolved].sum(-1).mean().item() if n else None,
        'memory_class_centroids':means.tolist(),'memory_within_class_mean_square_distance':within.tolist(),
        'memory_unresolved_centroid_mean_distance':pair_distance(means,unresolved),
        'memory_all_centroid_mean_distance':pair_distance(means,list(range(8))),
        'value_embedding_norms':v.norm(dim=-1).tolist(),'value_embedding_cosines':(F.normalize(v,dim=-1)@F.normalize(v,dim=-1).T).tolist(),
        'value_unresolved_mean_distance':pair_distance(v,unresolved),'value_all_mean_distance':pair_distance(v,list(range(8)))}
    # Independent per-query accumulation differs slightly from batch-mean reduction.
    assert abs(result['overall_ce']-result['recorded_ce'])<1e-6
    assert result['per_class_queries']==[4224.]*8
    return result

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    data=study.splits()['test'];OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for config in study.CONFIGS:
        label=study.config_name(config)
        for stream in study.SEEDS:
            record=json.loads((study.OUT/f'{label}-t{stream}.json').read_text())
            for mode in ['released','complete']:rows.append(inspect(record,mode,data))
        print(f'Inspected {label}',flush=True)
    report={'kind':'Post-hoc descriptive checkpoint inspection, no new training',
        'criterion':'Unresolved class: mean test CE >= 0.1; threshold selected for this diagnostic after global outcomes were seen.',
        'hypothesis':'One unresolved set of m equiprobable value classes predicts m/8 * ln(m) total CE if resolved classes are perfect.',
        'limits':'A matching plateau and similar averaged predictions do not prove identical per-episode memory representations or a causal training mechanism.',
        'rows':rows,'script_sha256':study.sha(__file__),'source_definitions':study.prior.prior.refresh.pilot.engine.loaded}
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    failed=[r for r in rows if r['recorded_ce']>=.1]
    print(json.dumps({'checkpoints':len(rows),'failed_ce_at_least_0_1':len(failed),
        'cluster_ce_abs_error_mean':sum(abs(r['observed_minus_cluster_ce']) for r in failed)/len(failed),
        'cluster_ce_within_0_05':sum(abs(r['observed_minus_cluster_ce'])<.05 for r in failed),
        'cluster_probability_l1_mean':sum(r['unresolved_probability_l1_to_uniform'] for r in failed)/len(failed)},indent=2))
if __name__=='__main__':main()
