# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Verify split isolation, saved checkpoints, paired controls and final metrics."""
import hashlib,json
from pathlib import Path
import torch
import run_recall_pilot as pilot
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/recall_pilot'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
manifest=json.loads((OUT/'manifest.json').read_text())
assert hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()==manifest['script_sha256']
assert hashlib.sha256(Path(pilot.engine.__file__).read_bytes()).hexdigest()==manifest['engine_script_sha256']
assert hashlib.sha256((ROOT/'notes/RECALL_PILOT_PROTOCOL.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
data=pilot.splits();sets={k:set(map(tuple,v.tolist())) for k,v in data.items()}
assert len(set.union(*sets.values()))==40320
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not sets[a]&sets[b]
for name,tensor in data.items():
    assert hashlib.sha256(bytes(tensor.to(torch.uint8).flatten().tolist())).hexdigest()==manifest['splits'][name]['sha256']
checks=[]
for depth in [1,2]:
    for seed in [0,1,2]:
        record=json.loads((OUT/f'd{depth}-s{seed}.json').read_text())
        first=record['history'][0]['train']
        assert first['released']['loss']==first['complete']['loss']
        assert len(set(record['initial_hashes'].values()))==1
        assert len(set(record['final_hashes'].values()))==(1 if depth==1 else 2)
        for mode in ['released','complete']:
            info=record['checkpoints'][mode];path=ROOT/info['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            model=pilot.RecallModel(depth)
            model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
            assert pilot.state_hash(model)==record['final_hashes'][mode]
            metrics=pilot.evaluate(model,data['test'],mode,seed=8200)
            ablation=pilot.evaluate(model,data['test'],mode,no_write=True,seed=8200)
            assert metrics==record['test'][mode] and ablation==record['no_write_test'][mode]
            batch=pilot.batch_for(data['test'][:4],torch.Generator().manual_seed(0))
            assert (batch[1][:,8:]==0).all() and (batch[2][:,8:]==0).all()
            altered_values=batch[1].clone();altered_values[:,:8]=(altered_values[:,:8]+1)%8
            altered=(batch[0],altered_values,batch[2],batch[3])
            with torch.no_grad():
                assert torch.equal(model(batch,mode,True),model(altered,mode,True))
            checks.append({'depth':depth,'seed':seed,'mode':mode,'checkpoint_hash_verified':True,'test_metrics_reproduced_exactly':True,'no_adaptation_value_invariance':True})
report={'split_overlap':0,'unique_mappings':40320,'protocol_and_code_hashes_verified':True,'checks':checks}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified all 12 checkpoints, exact final metrics, split isolation, paired controls, and no-adaptation invariance.')
