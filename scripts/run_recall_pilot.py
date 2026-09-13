# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Paired held-out associative-recall pilot using source-loaded TTT graph.
No natural-language pretraining. See notes/RECALL_PILOT_PROTOCOL.md.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import check_graph_gradients as engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/recall_pilot'

class RecallModel(nn.Module):
    def __init__(self,depth):
        super().__init__()
        d=16
        self.q=nn.Embedding(8,d); self.k=nn.Embedding(8,d); self.v=nn.Embedding(8,d)
        for embed in [self.q,self.k,self.v]: nn.init.normal_(embed.weight,std=0.2)
        self.rates=nn.Linear(2*d,depth,bias=True)
        self.decays=nn.Linear(2*d,depth,bias=True)
        nn.init.zeros_(self.rates.weight); nn.init.constant_(self.rates.bias,math.log(0.0005/0.9995))
        nn.init.zeros_(self.decays.weight); nn.init.constant_(self.decays.bias,math.log(0.99/0.01))
        specs=[];prev='input'
        for j in range(depth):
            specs.extend([{'name':f'w{j}','type':'linear','inputs':[prev],'d_out':'output'},
                          {'name':f'a{j}','type':'act','inputs':[f'w{j}'],'activation':'silu'}])
            prev=f'a{j}'
        self.memory=engine.ns['TTTGraphMemory'](h=1,d=d,e=d,graph_nodes=specs,
            graph_output=prev,chunk_size=4,mean_loss=False,init_std=0.02,init_style='official')
        self.norm=nn.LayerNorm(d,eps=1e-5)
        self.head=nn.Linear(d,8)
        self.depth=depth

    def forward(self,batch,mode,no_write=False):
        engine.ns['linear_grad']=engine.released_linear if mode=='released' else engine.complete_linear
        ids,values,mask,target=batch
        q=self.q(ids).unsqueeze(2)
        key_raw=self.k(ids)
        k=F.normalize(key_raw,dim=-1).unsqueeze(2)
        v_raw=self.v(values)*mask.unsqueeze(-1)
        v=v_raw.unsqueeze(2)
        features=torch.cat([key_raw,v_raw],-1)
        rates=2*torch.sigmoid(self.rates(features))*mask.unsqueeze(-1)
        if no_write: rates=rates*0
        log_decay=F.logsigmoid(self.decays(features))*mask.unsqueeze(-1)
        if no_write: log_decay=log_decay*0
        lr=[rates[:,:,j:j+1].unsqueeze(2) for j in range(self.depth)]
        ld=[log_decay[:,:,j:j+1].unsqueeze(2) for j in range(self.depth)]
        out,_,_=self.memory(q=q,k=k,v=v,lr=lr,lr_type='scalar',log_f=ld)
        return self.head(self.norm(out[:,8:,0]))


def splits():
    permutations=list(itertools.permutations(range(8)))
    random.Random(20260913).shuffle(permutations)
    data=torch.tensor(permutations,dtype=torch.long)
    return {'train':data[:32000],'validation':data[32000:36096],'test':data[36096:]}

def batch_for(mapping,gen):
    n=len(mapping)
    write_order=torch.rand(n,8,generator=gen).argsort(-1)
    query_order=torch.rand(n,8,generator=gen).argsort(-1)
    ids=torch.cat([write_order,query_order],1)
    values=torch.cat([mapping.gather(1,write_order),torch.zeros_like(query_order)],1)
    mask=torch.cat([torch.ones(n,8),torch.zeros(n,8)],1)
    return ids,values,mask,mapping.gather(1,query_order)

def loss(logits,target): return F.cross_entropy(logits.reshape(-1,8),target.reshape(-1))

def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode());digest.update(bytes(tensor.detach().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()

@torch.no_grad()
def evaluate(model,data,mode,no_write=False,seed=8100):
    gen=torch.Generator().manual_seed(seed); total_loss=0.; correct=0; count=0
    for start in range(0,len(data),128):
        batch=batch_for(data[start:start+128],gen)
        logits=model(batch,mode,no_write); n=batch[-1].numel()
        total_loss+=loss(logits,batch[-1]).item()*n
        correct+=(logits.argmax(-1)==batch[-1]).sum().item();count+=n
    return {'cross_entropy':total_loss/count,'accuracy':correct/count,'queries':count}

def train_pair(depth,seed,data,steps,smoke=False):
    torch.manual_seed(seed)
    original=RecallModel(depth)
    models={'released':original,'complete':copy.deepcopy(original)}
    opts={mode:torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01) for mode,model in models.items()}
    initial_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(initial_hashes.values()))==1
    gen=torch.Generator().manual_seed(10000+seed)
    rows=[]; elapsed={mode:0. for mode in models}
    first_grad_delta=None
    for step in range(1,steps+1):
        indexes=torch.randint(len(data['train']),(32,),generator=gen)
        batch=batch_for(data['train'][indexes],gen)
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
            print(json.dumps({'depth':depth,'seed':seed,**row}),flush=True)
    final_hashes={mode:state_hash(m) for mode,m in models.items()}
    assert len(set(final_hashes.values()))==1 if depth==1 else len(set(final_hashes.values()))==2
    result={'depth':depth,'seed':seed,'steps':steps,'tokens_per_run':steps*32*16,
        'parameters_per_model':sum(p.numel() for p in original.parameters()),'initial_hashes':initial_hashes,
        'final_hashes':final_hashes,'first_step_max_gradient_difference':first_grad_delta,'training_seconds':elapsed,'history':rows}
    if not smoke:
        result['test']={mode:evaluate(m,data['test'],mode,seed=8200) for mode,m in models.items()}
        result['no_write_test']={mode:evaluate(m,data['test'],mode,no_write=True,seed=8200) for mode,m in models.items()}
        folder=ROOT/'models/recall_pilot';folder.mkdir(parents=True,exist_ok=True)
        result['checkpoints']={}
        for mode,m in models.items():
            path=folder/f'd{depth}-s{seed}-{mode}.pt';torch.save(m.state_dict(),path)
            result['checkpoints'][mode]={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (OUT/f'd{depth}-s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'completed':f'd{depth}-s{seed}','test':result['test'],'no_write_test':result['no_write_test']}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    data=splits();OUT.mkdir(parents=True,exist_ok=True)
    metadata={'task':'eight-pair associative recall; disjoint permutation splits',
        'data_split_seed':20260913,'splits':{k:{'episodes':len(v),'sha256':hashlib.sha256(bytes(v.to(torch.uint8).flatten().tolist())).hexdigest()} for k,v in data.items()},
        'source_revision':engine.REV,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'engine_script_sha256':hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        'torch':torch.__version__,'torch_git':torch.version.git_version,'device':'cpu','dtype':'float32','threads':1,
        'protocol_sha256':None if args.smoke else hashlib.sha256((ROOT/'notes/RECALL_PILOT_PROTOCOL.md').read_bytes()).hexdigest()}
    if args.smoke:
        rows=[train_pair(d,99,data,10,True) for d in [1,2]]
        (OUT/'smoke.json').write_text(json.dumps({'metadata':metadata,'pairs':rows},indent=2)+'\n')
    else:
        (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        for depth in [1,2]:
            for seed in [0,1,2]:train_pair(depth,seed,data,600)

if __name__=='__main__':main()
