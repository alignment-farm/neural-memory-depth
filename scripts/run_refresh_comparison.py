# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Frozen chunk-size comparison; see notes/REFRESH_COMPARISON_PROTOCOL.md."""
import copy, hashlib, json, time
from pathlib import Path
import torch
from torch import nn
import run_recall_pilot as pilot
from run_recall_pilot import batch_for, splits, loss, state_hash, evaluate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/refresh_comparison'
CHUNKS=[1,2,4,8]
SEEDS=list(range(10,20))
class RecallModel(pilot.RecallModel):
    def __init__(self,depth,chunk):
        super().__init__(depth)
        self.memory.chunk_size=chunk

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def train_pair(depth,seed,chunk,data,steps,smoke=False):
    torch.manual_seed(seed)
    original=RecallModel(depth,chunk)
    models={'released':original,'complete':copy.deepcopy(original)}
    opts={mode:torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01) for mode,model in models.items()}
    initial_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(initial_hashes.values()))==1
    gen=torch.Generator().manual_seed(10000+seed)
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
            print(json.dumps({'chunk':chunk,'depth':depth,'seed':seed,**row}),flush=True)
    final_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(final_hashes.values()))==1 if depth==1 else len(set(final_hashes.values()))==2
    result={'chunk':chunk,'depth':depth,'seed':seed,'steps':steps,'tokens_per_run':steps*32*16,
        'training_batches_sha256':batch_digest.hexdigest(),'parameters_per_model':sum(p.numel() for p in original.parameters()),'initial_hashes':initial_hashes,
        'final_hashes':final_hashes,'first_step_max_gradient_difference':first_grad_delta,'training_seconds':elapsed,'history':rows}
    if not smoke:
        result['test']={mode:evaluate(m,data['test'],mode,seed=8200) for mode,m in models.items()}
        result['no_write_test']={mode:evaluate(m,data['test'],mode,no_write=True,seed=8200) for mode,m in models.items()}
        folder=ROOT/'models/refresh_comparison';folder.mkdir(parents=True,exist_ok=True)
        result['checkpoints']={}
        for mode,m in models.items():
            path=folder/f'c{chunk}-d{depth}-s{seed}-{mode}.pt';torch.save(m.state_dict(),path)
            result['checkpoints'][mode]={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (OUT/f'c{chunk}-d{depth}-s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'completed':f'c{chunk}-d{depth}-s{seed}','test':result['test'],'no_write_test':result['no_write_test']}),flush=True)
    return result

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    data=splits();OUT.mkdir(parents=True,exist_ok=True)
    paths=['notes/REFRESH_COMPARISON_PROTOCOL.md','scripts/run_refresh_comparison.py',
           'scripts/run_recall_pilot.py','scripts/check_graph_gradients.py','scripts/check_refresh_preflight.py']
    metadata={'files':{p:sha(ROOT/p) for p in paths},'source_revision':pilot.engine.REV,
        'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','dtype':'float32','threads':1,
        'chunks':CHUNKS,'depths':[1,2],'seeds':SEEDS,'steps':600,
        'splits':{k:{'episodes':len(v),'sha256':hashlib.sha256(bytes(v.to(torch.uint8).flatten().tolist())).hexdigest()} for k,v in data.items()},
        'preflight_sha256':sha(OUT/'preflight.json')}
    manifest=OUT/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==metadata, 'Frozen manifest changed'
    else:manifest.write_text(json.dumps(metadata,indent=2)+'\n')
    for seed in SEEDS:
        for depth in [1,2]:
            for chunk in CHUNKS:
                path=OUT/f'c{chunk}-d{depth}-s{seed}.json'
                if path.exists():
                    record=json.loads(path.read_text())
                    assert (record['chunk'],record['depth'],record['seed'],record['steps'])==(chunk,depth,seed,600)
                    for info in record['checkpoints'].values():assert sha(ROOT/info['path'])==info['sha256']
                    continue
                train_pair(depth,seed,chunk,data,600)
if __name__=='__main__':main()
