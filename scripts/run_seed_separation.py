# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Cross whole-model initialization with the sampled training stream."""
import copy,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
import run_refresh_comparison as refresh
from run_refresh_comparison import RecallModel,sha,batch_for,splits,loss,state_hash,evaluate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/seed_separation'
SEEDS=list(range(10,20))

def train_pair(init_seed,stream_seed,data,steps=600,smoke=False):
    depth=2;chunk=4
    torch.manual_seed(init_seed)
    original=RecallModel(depth,chunk)
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
            print(json.dumps({'chunk':chunk,'depth':depth,'init_seed':init_seed,'stream_seed':stream_seed,**row}),flush=True)
    final_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(final_hashes.values()))==1 if depth==1 else len(set(final_hashes.values()))==2
    result={'chunk':chunk,'depth':depth,'init_seed':init_seed,'stream_seed':stream_seed,'steps':steps,'tokens_per_run':steps*32*16,'origin':'new',
        'training_batches_sha256':batch_digest.hexdigest(),'parameters_per_model':sum(p.numel() for p in original.parameters()),'initial_hashes':initial_hashes,
        'final_hashes':final_hashes,'first_step_max_gradient_difference':first_grad_delta,'training_seconds':elapsed,'history':rows}
    if not smoke:
        result['test']={mode:evaluate(m,data['test'],mode,seed=8200) for mode,m in models.items()}
        result['no_write_test']={mode:evaluate(m,data['test'],mode,no_write=True,seed=8200) for mode,m in models.items()}
        folder=ROOT/'models/seed_separation';folder.mkdir(parents=True,exist_ok=True)
        result['checkpoints']={}
        for mode,m in models.items():
            path=folder/f'i{init_seed}-t{stream_seed}-{mode}.pt';torch.save(m.state_dict(),path)
            result['checkpoints'][mode]={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (OUT/f'i{init_seed}-t{stream_seed}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'completed':f'i{init_seed}-t{stream_seed}','test':result['test'],'no_write_test':result['no_write_test']}),flush=True)
    return result

def donor_paths():
    return [f'analysis/refresh_comparison/c4-d2-s{s}.json' for s in SEEDS]+[
        'analysis/refresh_comparison/manifest.json','analysis/refresh_comparison/preflight.json',
        'analysis/refresh_comparison/verification.json']

def check_donors():
    previous=json.loads((refresh.OUT/'manifest.json').read_text())
    for path,digest in previous['files'].items():assert sha(ROOT/path)==digest,path
    assert sha(refresh.OUT/'preflight.json')==previous['preflight_sha256']
    assert json.loads((refresh.OUT/'preflight.json').read_text())['source_definitions']==refresh.pilot.engine.loaded
    verification=json.loads((refresh.OUT/'verification.json').read_text())
    assert verification['protocol_code_and_source_hashes_verified'] and len(verification['checks'])==160
    for s in SEEDS:
        record=json.loads((refresh.OUT/f'c4-d2-s{s}.json').read_text())
        for mode,info in record['checkpoints'].items():
            assert sha(ROOT/info['path'])==info['sha256']
            model=RecallModel(2,4);model.load_state_dict(torch.load(ROOT/info['path'],weights_only=True))
            assert state_hash(model)==record['final_hashes'][mode]

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    check_donors();data=splits();OUT.mkdir(parents=True,exist_ok=True)
    paths=['notes/SEED_SEPARATION_PROTOCOL.md','scripts/run_seed_separation.py','scripts/check_seed_preflight.py',
        'scripts/run_refresh_comparison.py','scripts/run_recall_pilot.py','scripts/check_graph_gradients.py']
    metadata={'files':{p:sha(ROOT/p) for p in paths},'donor_files':{p:sha(ROOT/p) for p in donor_paths()},
        'source_revision':refresh.pilot.engine.REV,'torch':torch.__version__,'torch_git':torch.version.git_version,
        'device':'cpu','dtype':'float32','threads':1,'initialization_seeds':SEEDS,'stream_seeds':SEEDS,
        'depth':2,'chunk':4,'steps':600,'new_models':180,'reused_models':20,
        'splits':{k:{'episodes':len(v),'sha256':hashlib.sha256(bytes(v.to(torch.uint8).flatten().tolist())).hexdigest()} for k,v in data.items()},
        'preflight_sha256':sha(OUT/'preflight.json')}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==metadata,'Frozen manifest changed'
    else:path.write_text(json.dumps(metadata,indent=2)+'\n')
    for init in SEEDS:
        for stream in SEEDS:
            path=OUT/f'i{init}-t{stream}.json'
            if path.exists():
                record=json.loads(path.read_text())
                assert (record['init_seed'],record['stream_seed'],record['steps'])==(init,stream,600)
                for info in record['checkpoints'].values():assert sha(ROOT/info['path'])==info['sha256']
                continue
            if init==stream:
                donor=f'analysis/refresh_comparison/c4-d2-s{init}.json'
                record=json.loads((ROOT/donor).read_text());record.pop('seed')
                record.update(init_seed=init,stream_seed=stream,origin='reused_diagonal',donor_path=donor,donor_sha256=sha(ROOT/donor))
                path.write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps({'reused':f'i{init}-t{stream}'}),flush=True)
            else:train_pair(init,stream,data)
if __name__=='__main__':main()
