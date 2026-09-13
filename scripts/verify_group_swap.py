# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
import hashlib,json
import torch
import run_group_swap as study

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    root,out=study.ROOT,study.OUT;m=json.loads((out/'manifest.json').read_text())
    for g in ['files','donor_files']:
        for path,digest in m[g].items():assert study.sha(root/path)==digest,path
    assert study.sha(out/'preflight.json')==m['preflight_sha256']
    assert json.loads((out/'preflight.json').read_text())['source_definitions']==study.prior.refresh.pilot.engine.loaded
    data=study.splits();sets={k:set(map(tuple,v.tolist())) for k,v in data.items()}
    assert len(set.union(*sets.values()))==40320
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not sets[a]&sets[b]
    for k,v in data.items():assert hashlib.sha256(bytes(v.to(torch.uint8).flatten().tolist())).hexdigest()==m['splits'][k]['sha256']
    streams={}
    for seed in study.SEEDS:
        gen=torch.Generator().manual_seed(10000+seed);digest=hashlib.sha256()
        for step in range(600):
            ix=torch.randint(len(data['train']),(32,),generator=gen)
            for x in study.batch_for(data['train'][ix],gen):digest.update(bytes(x.contiguous().view(torch.uint8).flatten().tolist()))
        streams[seed]=digest.hexdigest()
    checks=[];origins={'new':0,'reused_pure_donor':0};donors=study.donors()
    for config in study.CONFIGS:
        label=study.config_name(config);model=study.make_model(config)
        assert study.state_hash(model)==m['initial_hashes'][label]
        for name,x in model.state_dict().items():
            assert study.group(name)==m['parameter_groups'][name]
            seed=config[study.group(name)] if study.group(name) else 11
            assert torch.equal(x,donors[seed].state_dict()[name])
        for stream in study.SEEDS:
            r=json.loads((out/f'{label}-t{stream}.json').read_text())
            assert r['configuration']==config and r['stream_seed']==stream and r['steps']==600
            assert set(r['initial_hashes'].values())=={m['initial_hashes'][label]}
            assert r['training_batches_sha256']==streams[stream]
            assert r['parameters_per_model']==1196 and r['tokens_per_run']==307200
            assert r['history'][0]['train']['released']['loss']==r['history'][0]['train']['complete']['loss']
            assert r['first_step_max_gradient_difference']>0 and len(set(r['final_hashes'].values()))==2
            reused=len(set(config.values()))==1
            assert r['origin']==('reused_pure_donor' if reused else 'new');origins[r['origin']]+=2
            if reused:
                assert study.sha(root/r['donor_path'])==r['donor_sha256']
                donor=json.loads((root/r['donor_path']).read_text())
                for key,value in donor.items():
                    if key not in ['init_seed','origin','donor_path','donor_sha256']:assert r[key]==value,key
            for mode in ['released','complete']:
                info=r['checkpoints'][mode];assert study.sha(root/info['path'])==info['sha256']
                model=study.RecallModel(2,4);model.load_state_dict(torch.load(root/info['path'],weights_only=True,map_location='cpu'))
                assert study.state_hash(model)==r['final_hashes'][mode]
                assert study.evaluate(model,data['test'],mode,seed=8200)==r['test'][mode]
                assert study.evaluate(model,data['validation'],mode,seed=8100)==r['history'][-1]['validation'][mode]
                assert study.evaluate(model,data['test'],mode,True,seed=8200)==r['no_write_test'][mode]
                batch=study.batch_for(data['test'][:4],torch.Generator().manual_seed(0))
                assert (batch[1][:,8:]==0).all() and (batch[2][:,8:]==0).all()
                values=batch[1].clone();values[:,:8]=(values[:,:8]+1)%8
                with torch.no_grad():assert torch.equal(model(batch,mode,True),model((batch[0],values,batch[2],batch[3]),mode,True))
                checks.append({'configuration':config,'stream_seed':stream,'mode':mode,'origin':r['origin'],'checkpoint_and_all_final_metrics_verified':True})
        print(f'Verified {label}: all ten streams and both derivatives.',flush=True)
    assert origins=={'new':120,'reused_pure_donor':40}
    (out/'verification.json').write_text(json.dumps({'all_frozen_hashes_verified':True,'all_group_tensor_origins_verified':True,
        'regenerated_stream_hashes':streams,'models_by_origin':origins,'checks':checks,'verifier_sha256':study.sha(__file__)},indent=2)+'\n')
    print('Verified all 160 group-swap outcomes and unchanged baselines.')
if __name__=='__main__':main()
