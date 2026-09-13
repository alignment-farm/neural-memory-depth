# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Full factorial of query/key/value donors with fixed donor-16 memory and readout."""
import copy,hashlib,itertools,json,time
from pathlib import Path
import torch
from torch import nn
import run_group_swap as prior
from run_group_swap import RecallModel,sha,batch_for,splits,loss,state_hash,evaluate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/embedding_swap'
SEEDS=list(range(10,20));GROUPS=['query','key','value']
CONFIGS=[dict(zip(GROUPS,v)) for v in itertools.product([11,16],repeat=3)]
def config_name(c):return f"q{c['query']}-k{c['key']}-v{c['value']}"
def group(name):
    if name.startswith('q.'):return 'query'
    if name.startswith('k.'):return 'key'
    if name.startswith('v.'):return 'value'
    return None

def donors():
    models={}
    for seed in [11,16]:
        torch.manual_seed(seed);models[seed]=RecallModel(2,4)
    assert set(filter(None,map(group,models[11].state_dict())))==set(GROUPS)
    return models

def make_model(config):
    models=donors();model=copy.deepcopy(models[16]);states={s:m.state_dict() for s,m in models.items()}
    model.load_state_dict({name:states[config[group(name)] if group(name) else 16][name] for name in model.state_dict()})
    return model

def donor_paths():
    return [f'analysis/group_swap/m16-e{e}-r16-t{t}.json' for e in [11,16] for t in SEEDS]+[
        'analysis/group_swap/manifest.json','analysis/group_swap/verification.json','analysis/group_swap/preflight.json']
def check_donors():
    m=json.loads((prior.OUT/'manifest.json').read_text())
    for g in ['files','donor_files']:
        for path,digest in m[g].items():assert sha(ROOT/path)==digest,path
    v=json.loads((prior.OUT/'verification.json').read_text())
    assert v['all_frozen_hashes_verified'] and len(v['checks'])==160
    assert json.loads((prior.OUT/'preflight.json').read_text())['source_definitions']==prior.prior.refresh.pilot.engine.loaded
    for e in [11,16]:
        for stream in SEEDS:
            r=json.loads((prior.OUT/f'm16-e{e}-r16-t{stream}.json').read_text())
            for info in r['checkpoints'].values():assert sha(ROOT/info['path'])==info['sha256']

def train_pair(config,stream_seed,data,steps=600,smoke=False):
    label=config_name(config)
    depth=2;chunk=4
    original=make_model(config)
    models={'released':original,'complete':copy.deepcopy(original)}
    opts={mode:torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01) for mode,model in models.items()}
    initial_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(initial_hashes.values()))==1
    gen=torch.Generator().manual_seed(10000+stream_seed)
    rows=[]; elapsed={mode:0. for mode in models}
    first_grad_delta=None
    batch_digest=hashlib.sha256()
    for step in range(1,steps+1):
        indexes=torch.randint(len(data['train']),(32,),generator=gen)
        batch=batch_for(data['train'][indexes],gen)
        for tensor in batch: batch_digest.update(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
        training={};gradsets={}
        for mode,model in models.items():
            started=time.perf_counter()
            opts[mode].zero_grad(set_to_none=True)
            logits=model(batch,mode); l=loss(logits,batch[-1]); assert torch.isfinite(l)
            l.backward()
            if step==1: gradsets[mode]=[p.grad.detach().clone() for p in model.parameters()]
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opts[mode].step()
            elapsed[mode]+=time.perf_counter()-started
            training[mode]={'loss':l.item(),'gradient_norm_before_clip':norm.item()}
        if step==1:
            first_grad_delta=max((a-b).abs().max().item() for a,b in zip(gradsets['released'],gradsets['complete']))
            assert first_grad_delta==0 if depth==1 else first_grad_delta>0
        if step==1 or step%100==0 or step==steps:
            row={'step':step,'train':training,'training_seconds':dict(elapsed)}
            if not smoke: row['validation']={mode:evaluate(m,data['validation'],mode) for mode,m in models.items()}
            rows.append(row)
            print(json.dumps({'chunk':chunk,'depth':depth,'configuration':config,'label':label,'stream_seed':stream_seed,**row}),flush=True)
    final_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(final_hashes.values()))==1 if depth==1 else len(set(final_hashes.values()))==2
    result={'chunk':chunk,'depth':depth,'configuration':config,'label':label,'stream_seed':stream_seed,'steps':steps,'tokens_per_run':steps*32*16,'origin':'new',
        'training_batches_sha256':batch_digest.hexdigest(),'parameters_per_model':sum(p.numel() for p in original.parameters()),'initial_hashes':initial_hashes,
        'final_hashes':final_hashes,'first_step_max_gradient_difference':first_grad_delta,'training_seconds':elapsed,'history':rows}
    if not smoke:
        result['test']={mode:evaluate(m,data['test'],mode,seed=8200) for mode,m in models.items()}
        result['no_write_test']={mode:evaluate(m,data['test'],mode,no_write=True,seed=8200) for mode,m in models.items()}
        folder=ROOT/'models/embedding_swap';folder.mkdir(parents=True,exist_ok=True)
        result['checkpoints']={}
        for mode,m in models.items():
            path=folder/f'{label}-t{stream_seed}-{mode}.pt';torch.save(m.state_dict(),path)
            result['checkpoints'][mode]={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (OUT/f'{label}-t{stream_seed}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'completed':f'{label}-t{stream_seed}','test':result['test'],'no_write_test':result['no_write_test']}),flush=True)
    return result

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    check_donors();data=splits();OUT.mkdir(parents=True,exist_ok=True)
    paths=['notes/EMBEDDING_SWAP_PROTOCOL.md','scripts/run_embedding_swap.py','scripts/check_embedding_preflight.py',
        'scripts/run_group_swap.py','scripts/run_seed_separation.py','scripts/run_refresh_comparison.py','scripts/run_recall_pilot.py','scripts/check_graph_gradients.py']
    model=make_model(CONFIGS[0])
    metadata={'files':{p:sha(ROOT/p) for p in paths},'donor_files':{p:sha(ROOT/p) for p in donor_paths()},
        'source_revision':prior.prior.refresh.pilot.engine.REV,'torch':torch.__version__,'torch_git':torch.version.git_version,
        'device':'cpu','dtype':'float32','threads':1,'configurations':CONFIGS,'stream_seeds':SEEDS,'steps':600,
        'new_models':120,'reused_models':40,'preflight_sha256':sha(OUT/'preflight.json'),
        'fixed_non_embedding_donor':16,'parameter_groups':{name:group(name) for name in model.state_dict()},
        'initial_hashes':{config_name(c):state_hash(make_model(c)) for c in CONFIGS},
        'splits':{k:{'episodes':len(v),'sha256':hashlib.sha256(bytes(v.to(torch.uint8).flatten().tolist())).hexdigest()} for k,v in data.items()}}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==metadata,'Frozen manifest changed'
    else:path.write_text(json.dumps(metadata,indent=2)+'\n')
    for config in CONFIGS:
        label=config_name(config)
        for stream in SEEDS:
            path=OUT/f'{label}-t{stream}.json'
            if path.exists():
                r=json.loads(path.read_text());assert r['configuration']==config and r['stream_seed']==stream
                for info in r['checkpoints'].values():assert sha(ROOT/info['path'])==info['sha256']
                continue
            if len(set(config.values()))==1:
                seed=config['query'];donor=f'analysis/group_swap/m16-e{seed}-r16-t{stream}.json'
                record=json.loads((ROOT/donor).read_text())
                record.update(configuration=config,label=label,origin='reused_embedding_baseline',donor_path=donor,donor_sha256=sha(ROOT/donor))
                path.write_text(json.dumps(record,indent=2)+'\n')
            else:train_pair(config,stream,data)
if __name__=='__main__':main()
