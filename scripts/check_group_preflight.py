# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
import contextlib,io,json
import torch
import run_group_swap as study

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    study.check_donors();data=study.splits();checks=[];donors=study.donors()
    for config in study.CONFIGS:
        model=study.make_model(config)
        for name,value in model.state_dict().items():
            source=config[study.group(name)] if study.group(name) else 11
            assert torch.equal(value,donors[source].state_dict()[name])
    for seed in [11,16]:
        config={g:seed for g in study.GROUPS}
        with contextlib.redirect_stdout(io.StringIO()):
            old=study.prior.train_pair(seed,99,data,10,True)
            new=study.train_pair(config,99,data,10,True)
        for key in ['initial_hashes','final_hashes','first_step_max_gradient_difference','training_batches_sha256','parameters_per_model']:
            assert old[key]==new[key],key
        for a,b in zip(old['history'],new['history']):assert a['train']==b['train']
        checks.append({'donor':seed,'stream':99,'steps':10,'exact_training_parity':True})
    report={'all_group_origins_verified':True,'pure_donor_training_checks':checks,
        'source_definitions':study.prior.refresh.pilot.engine.loaded,'script_sha256':study.sha(__file__)}
    study.OUT.mkdir(parents=True,exist_ok=True)
    (study.OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified every hybrid tensor origin and exact pure-donor training parity.')
if __name__=='__main__':main()
