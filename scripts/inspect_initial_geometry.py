# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Exploratory initialization-only geometry; no training or held-out evaluation."""
import hashlib,itertools,json
from pathlib import Path
import torch
from torch.nn import functional as F
import run_group_swap as study
ROOT=study.ROOT

def stats(x):return {'min':x.min().item(),'max':x.max().item(),'mean':x.mean().item()}
def matrix(x):
    x=x.detach().double();sv=torch.linalg.svdvals(x)
    cos=F.normalize(x,dim=-1)@F.normalize(x,dim=-1).T
    return {'shape':list(x.shape),'singular_values':sv.tolist(),'condition_number':(sv.max()/sv.min()).item(),
        'row_norms':x.norm(dim=-1).tolist(),'off_diagonal_row_cosine':stats(cos[~torch.eye(len(x),dtype=torch.bool)])}

def main():
    torch.set_num_threads(1);donors=study.donors();result={'scope':'Exploratory untrained geometry, selected donors; no causal or predictive claim','donors':{},'query_key_crosses':[]}
    for seed,model in donors.items():
        result['donors'][seed]={name:matrix(getattr(model,name).weight) for name in ['q','k','v','head']}
    for qseed,kseed in itertools.product([11,16],repeat=2):
        q=F.normalize(donors[qseed].q.weight.detach().double(),dim=-1);k=F.normalize(donors[kseed].k.weight.detach().double(),dim=-1)
        sim=q@k.T;diag=sim.diag();off=sim[~torch.eye(8,dtype=torch.bool)]
        result['query_key_crosses'].append({'query_donor':qseed,'key_donor':kseed,'cosine_matrix':sim.tolist(),
            'matching_key_cosines':diag.tolist(),'matching_key_stats':stats(diag),'nonmatching_cosine_stats':stats(off),
            'matching_key_is_largest_count':int((sim.argmax(-1)==torch.arange(8)).sum()),'negative_matching_count':int((diag<0).sum())})
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['donor_initial_hashes']={s:study.state_hash(m) for s,m in donors.items()}
    (ROOT/'analysis/initial_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
