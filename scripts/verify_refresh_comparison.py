# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Verify the frozen refresh comparison and reproduce every checkpoint metric."""
import hashlib,json
from pathlib import Path
import torch
import run_refresh_comparison as study

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    out=study.OUT;root=study.ROOT
    manifest=json.loads((out/'manifest.json').read_text())
    for path,digest in manifest['files'].items():assert study.sha(root/path)==digest,path
    assert study.sha(out/'preflight.json')==manifest['preflight_sha256']
    preflight=json.loads((out/'preflight.json').read_text())
    assert preflight['source_definitions']==study.pilot.engine.loaded
    assert manifest['torch']==torch.__version__ and manifest['torch_git']==torch.version.git_version
    data=study.splits();sets={k:set(map(tuple,v.tolist())) for k,v in data.items()}
    assert len(set.union(*sets.values()))==40320
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not sets[a]&sets[b]
    for name,tensor in data.items():
        assert hashlib.sha256(bytes(tensor.to(torch.uint8).flatten().tolist())).hexdigest()==manifest['splits'][name]['sha256']
    checks=[];pairing=[]
    for seed in study.SEEDS:
        gen=torch.Generator().manual_seed(10000+seed);digest=hashlib.sha256()
        for step in range(600):
            indices=torch.randint(len(data['train']),(32,),generator=gen)
            batch=study.batch_for(data['train'][indices],gen)
            for tensor in batch:digest.update(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
        expected_batches=digest.hexdigest()
        for depth in [1,2]:
            initial=[]
            for chunk in study.CHUNKS:
                record=json.loads((out/f'c{chunk}-d{depth}-s{seed}.json').read_text())
                assert (record['chunk'],record['depth'],record['seed'],record['steps'])==(chunk,depth,seed,600)
                assert record['tokens_per_run']==307200
                assert record['training_batches_sha256']==expected_batches
                first=record['history'][0]['train']
                assert first['released']['loss']==first['complete']['loss']
                assert len(set(record['initial_hashes'].values()))==1
                initial.extend(record['initial_hashes'].values())
                assert len(set(record['final_hashes'].values()))==(1 if depth==1 else 2)
                delta=record['first_step_max_gradient_difference']
                assert delta==0 if depth==1 else delta>0
                torch.manual_seed(seed);fresh=study.RecallModel(depth,chunk)
                assert study.state_hash(fresh)==record['initial_hashes']['released']
                for mode in ['released','complete']:
                    info=record['checkpoints'][mode];path=root/info['path']
                    assert study.sha(path)==info['sha256']
                    model=study.RecallModel(depth,chunk)
                    model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                    assert sum(p.numel() for p in model.parameters())==record['parameters_per_model']
                    assert study.state_hash(model)==record['final_hashes'][mode]
                    assert study.evaluate(model,data['test'],mode,seed=8200)==record['test'][mode]
                    assert study.evaluate(model,data['test'],mode,True,seed=8200)==record['no_write_test'][mode]
                    batch=study.batch_for(data['test'][:4],torch.Generator().manual_seed(0))
                    assert (batch[1][:,8:]==0).all() and (batch[2][:,8:]==0).all()
                    values=batch[1].clone();values[:,:8]=(values[:,:8]+1)%8
                    altered=(batch[0],values,batch[2],batch[3])
                    with torch.no_grad():assert torch.equal(model(batch,mode,True),model(altered,mode,True))
                    checks.append({'chunk':chunk,'depth':depth,'seed':seed,'mode':mode,
                        'checkpoint_hash_verified':True,'test_and_ablation_metrics_exact':True,
                        'no_adaptation_value_invariance':True})
            assert len(set(initial))==1
            pairing.append({'depth':depth,'seed':seed,'identical_initial_parameters_across_chunks':True,
                'training_batch_hash_regenerated':expected_batches})
        print(f'Verified seed {seed}: all 16 checkpoints.',flush=True)
    report={'protocol_code_and_source_hashes_verified':True,'unique_mappings':40320,'split_overlap':0,
        'paired_starts_and_batches':pairing,'checks':checks,
        'verifier_sha256':study.sha(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all 160 checkpoints, exact held-out metrics, source hashes, splits and cross-chunk pairing.')
if __name__=='__main__':main()
