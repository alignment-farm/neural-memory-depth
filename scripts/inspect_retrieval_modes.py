# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Post-hoc query-key versus value-class retrieval diagnostics; no training."""
import json,math
import torch
from torch.nn import functional as F
import run_embedding_swap as study
OUT=study.ROOT/'analysis/class_collapse'

@torch.no_grad()
def inspect(record,mode,data,value_record):
    info=record['checkpoints'][mode];assert study.sha(study.ROOT/info['path'])==info['sha256']
    model=study.RecallModel(2,4);model.load_state_dict(torch.load(study.ROOT/info['path'],weights_only=True))
    captured=[];hook=model.norm.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    all_probs=[];all_targets=[];all_raw=[];all_ce=[];gen=torch.Generator().manual_seed(8200)
    for start in range(0,len(data),128):
        batch=study.batch_for(data[start:start+128],gen);captured.clear();logits=model(batch,mode)
        # Every query key occurs exactly once; sort query positions into key identity order.
        order=batch[0][:,8:].argsort(-1)
        all_probs.append(logits.softmax(-1).gather(1,order[:,:,None].expand(-1,-1,8)).double())
        all_targets.append(batch[-1].gather(1,order))
        all_raw.append(captured[-1].gather(1,order[:,:,None].expand(-1,-1,16)).double())
        ce=F.cross_entropy(logits.reshape(-1,8),batch[-1].reshape(-1),reduction='none').reshape(-1,8)
        all_ce.append(ce.gather(1,order).double())
    hook.remove();prob=torch.cat(all_probs);target=torch.cat(all_targets);raw=torch.cat(all_raw);ce=torch.cat(all_ce)
    key_ce=ce.mean(0);unknown=(key_ce>=.1).nonzero().flatten().tolist();n=len(unknown)
    expected=n/8*math.log(n) if n else 0.
    # Uniform uncertainty over values assigned to the unresolved keys, per episode.
    l1=None;mass=None;rawdist=None;probpair=None
    if n:
        ideal=torch.zeros(len(data),8,dtype=torch.float64)
        ideal.scatter_(1,target[:,unknown],1/n)
        l1=(prob[:,unknown]-ideal[:,None]).abs().sum(-1).mean().item()
        mass=(prob[:,unknown]*ideal[:,None].gt(0)).sum(-1).mean().item()
        if n>=2:
            off=~torch.eye(n,dtype=torch.bool)
            rawdist=torch.cdist(raw[:,unknown],raw[:,unknown])[:,off].mean().item()
            probpair=torch.cdist(prob[:,unknown],prob[:,unknown],p=1)[:,off].mean().item()
    # Same pair distances for queries whose target values belong to the fixed unresolved value set.
    unresolved_values=value_record['unresolved_classes_ce_at_least_0_1'];nv=len(unresolved_values)
    value_rawdist=None;value_probpair=None
    if nv>=2:
        positions=target.argsort(-1)[:,unresolved_values]
        vr=raw.gather(1,positions[:,:,None].expand(-1,-1,16));vp=prob.gather(1,positions[:,:,None].expand(-1,-1,8))
        off=~torch.eye(nv,dtype=torch.bool)
        value_rawdist=torch.cdist(vr,vr)[:,off].mean().item();value_probpair=torch.cdist(vp,vp,p=1)[:,off].mean().item()
    result={'label':record['label'],'configuration':record['configuration'],'stream_seed':record['stream_seed'],'mode':mode,
        'checkpoint_sha256':info['sha256'],'ce':ce.mean().item(),'per_key_ce':key_ce.tolist(),
        'per_key_accuracy':(prob.argmax(-1)==target).double().mean(0).tolist(),'unresolved_keys':unknown,'unresolved_key_count':n,
        'key_cluster_expected_ce':expected,'observed_minus_key_cluster_ce':ce.mean().item()-expected,
        'key_cluster_probability_l1_to_uniform_assigned_values':l1,'key_cluster_probability_mass':mass,
        'key_cluster_within_episode_raw_distance':rawdist,'key_cluster_within_episode_probability_l1':probpair,
        'value_cluster_count':nv,'value_cluster_ce_error':value_record['observed_minus_cluster_ce'],
        'value_cluster_probability_l1_to_uniform':value_record['unresolved_probability_l1_to_uniform'],
        'value_cluster_within_episode_raw_distance':value_rawdist,'value_cluster_within_episode_probability_l1':value_probpair,
        'all_queries_within_episode_raw_distance':torch.cdist(raw,raw)[:,~torch.eye(8,dtype=torch.bool)].mean().item()}
    assert abs(result['ce']-record['test'][mode]['cross_entropy'])<1e-6
    return result

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    path=OUT/'diagnostics.json';previous=json.loads(path.read_text());lookup={(r['label'],r['stream_seed'],r['mode']):r for r in previous['rows']}
    data=study.splits()['test'];rows=[]
    for config in study.CONFIGS:
        label=study.config_name(config)
        for stream in study.SEEDS:
            record=json.loads((study.OUT/f'{label}-t{stream}.json').read_text())
            for mode in ['released','complete']:rows.append(inspect(record,mode,data,lookup[label,stream,mode]))
        print(f'Inspected key/value retrieval modes for {label}',flush=True)
    failed=[r for r in rows if r['ce']>=.1]
    # These fit tolerances are exploratory diagnostic summaries, chosen after seeing the earlier value-only diagnostic.
    for r in rows:
        r['value_cluster_fit']=r['value_cluster_count']>=2 and abs(r['value_cluster_ce_error'])<.05 and r['value_cluster_probability_l1_to_uniform']<.15
        r['key_cluster_fit']=r['unresolved_key_count']>=2 and abs(r['observed_minus_key_cluster_ce'])<.05 and r['key_cluster_probability_l1_to_uniform_assigned_values']<.15
    summary={'checkpoints':len(rows),'failed':len(failed),'value_fit':sum(r['value_cluster_fit'] for r in failed),
        'key_fit':sum(r['key_cluster_fit'] for r in failed),'either_fit':sum(r['value_cluster_fit'] or r['key_cluster_fit'] for r in failed),
        'both_fit':sum(r['value_cluster_fit'] and r['key_cluster_fit'] for r in failed)}
    report={'kind':'Exploratory post-hoc key and value retrieval-mode inspection; no training',
        'fit_rule':'Unresolved class/key mean CE >= 0.1; single-cluster CE residual < 0.05 and mean prediction L1 to uniform cluster < 0.15; at least two members.',
        'limits':'Fit tolerances selected during post-hoc diagnosis; categories describe checkpoint behavior, not proven causes or population rates.',
        'input_value_diagnostic_sha256':study.sha(path),'script_sha256':study.sha(__file__),'summary':summary,'rows':rows}
    (OUT/'retrieval_modes.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
