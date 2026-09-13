# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Extend the independent graph audit to the four planned chunk sizes."""
import hashlib,json
from pathlib import Path
import torch
import check_graph_gradients as engine
import run_recall_pilot as pilot
import run_refresh_comparison as study

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    original=engine.make_data
    def expanded(depth,seed):
        q,k,v,lr,ld,target=original(depth,seed)
        def repeat(x):return x.detach().repeat(1,4,1,1).requires_grad_(x.requires_grad)
        return repeat(q),repeat(k),repeat(v),list(map(repeat,lr)),list(map(repeat,ld)),repeat(target)
    engine.make_data=expanded
    cases=[engine.validate(d,c,False,99) for d in [1,2] for c in study.CHUNKS]
    engine.make_data=original
    torch.set_default_dtype(torch.float32)
    batch=pilot.batch_for(pilot.splits()['train'][:4],torch.Generator().manual_seed(99))
    controls=[]
    for d in [1,2]:
        torch.manual_seed(99);a=pilot.RecallModel(d)
        torch.manual_seed(99);b=study.RecallModel(d,4)
        assert pilot.state_hash(a)==pilot.state_hash(b)
        with torch.no_grad():assert torch.equal(a(batch,'released'),b(batch,'released'))
        controls.append({'depth':d,'chunk_4_exact_pilot_parity':True})
    report={'source_revision':engine.REV,'tokens':24,'diagnostic_seed':99,'cases':cases,'controls':controls,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_definitions':engine.loaded}
    study.OUT.mkdir(parents=True,exist_ok=True)
    (study.OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed all eight chunk/depth reference audits and exact chunk-4 pilot parity.')
if __name__=='__main__':main()
