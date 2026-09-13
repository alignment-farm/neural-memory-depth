# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Verify initialization/stream isolation and all 200 crossed outcomes."""
import hashlib,json
from pathlib import Path
import torch
import run_seed_separation as study

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    root,out=study.ROOT,study.OUT
    manifest=json.loads((out/'manifest.json').read_text())
    for group in ['files','donor_files']:
        for path,digest in manifest[group].items():assert study.sha(root/path)==digest,path
    assert study.sha(out/'preflight.json')==manifest['preflight_sha256']
    assert json.loads((out/'preflight.json').read_text())['source_definitions']==study.refresh.pilot.engine.loaded
    assert manifest['torch']==torch.__version__ and manifest['torch_git']==torch.version.git_version
    data=study.splits();sets={k:set(map(tuple,v.tolist())) for k,v in data.items()}
    assert len(set.union(*sets.values()))==40320
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not sets[a]&sets[b]
    for name,tensor in data.items():assert hashlib.sha256(bytes(tensor.to(torch.uint8).flatten().tolist())).hexdigest()==manifest['splits'][name]['sha256']
    streams={};initial={}
    for seed in study.SEEDS:
        gen=torch.Generator().manual_seed(10000+seed);digest=hashlib.sha256()
        for step in range(600):
            ids=torch.randint(len(data['train']),(32,),generator=gen)
            batch=study.batch_for(data['train'][ids],gen)
            for tensor in batch:digest.update(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
        streams[seed]=digest.hexdigest()
        torch.manual_seed(seed);initial[seed]=study.state_hash(study.RecallModel(2,4))
    assert len(set(streams.values()))==len(set(initial.values()))==10
    checks=[];origins={'new':0,'reused_diagonal':0}
    for init in study.SEEDS:
        for stream in study.SEEDS:
            record=json.loads((out/f'i{init}-t{stream}.json').read_text())
            assert (record['init_seed'],record['stream_seed'],record['depth'],record['chunk'],record['steps'])==(init,stream,2,4,600)
            assert record['tokens_per_run']==307200 and record['parameters_per_model']==1196
            assert set(record['initial_hashes'].values())=={initial[init]}
            assert record['training_batches_sha256']==streams[stream]
            first=record['history'][0]['train'];assert first['released']['loss']==first['complete']['loss']
            assert record['first_step_max_gradient_difference']>0
            assert len(set(record['final_hashes'].values()))==2
            assert record['origin']==('reused_diagonal' if init==stream else 'new')
            origins[record['origin']]+=2
            if init==stream:
                assert study.sha(root/record['donor_path'])==record['donor_sha256']
                donor=json.loads((root/record['donor_path']).read_text())
                for key,value in donor.items():
                    if key!='seed':assert record[key]==value,key
            for mode in ['released','complete']:
                info=record['checkpoints'][mode];path=root/info['path']
                assert study.sha(path)==info['sha256']
                model=study.RecallModel(2,4)
                model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
                assert study.state_hash(model)==record['final_hashes'][mode]
                assert study.evaluate(model,data['test'],mode,seed=8200)==record['test'][mode]
                assert study.evaluate(model,data['validation'],mode,seed=8100)==record['history'][-1]['validation'][mode]
                assert study.evaluate(model,data['test'],mode,True,seed=8200)==record['no_write_test'][mode]
                batch=study.batch_for(data['test'][:4],torch.Generator().manual_seed(0))
                assert (batch[1][:,8:]==0).all() and (batch[2][:,8:]==0).all()
                values=batch[1].clone();values[:,:8]=(values[:,:8]+1)%8
                altered=(batch[0],values,batch[2],batch[3])
                with torch.no_grad():assert torch.equal(model(batch,mode,True),model(altered,mode,True))
                checks.append({'init_seed':init,'stream_seed':stream,'mode':mode,'origin':record['origin'],
                    'checkpoint_hash_verified':True,'test_validation_and_ablation_metrics_exact':True,'no_adaptation_value_invariance':True})
        print(f'Verified initialization {init}: all ten streams and both derivatives.',flush=True)
    assert origins=={'new':180,'reused_diagonal':20}
    report={'protocol_code_donor_and_source_hashes_verified':True,'unique_mappings':40320,'split_overlap':0,
        'initial_state_hashes_by_init':initial,'regenerated_batch_hashes_by_stream':streams,
        'row_constant_initialization_and_column_constant_stream_verified':True,'models_by_origin':origins,
        'checks':checks,'verifier_sha256':study.sha(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all 200 outcomes, independent factor hashes, exact metrics and donor preservation.')
if __name__=='__main__':main()
