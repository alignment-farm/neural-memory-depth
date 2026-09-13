# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Training-only diagonal parity against the frozen refresh runner."""
import contextlib,io,json
import torch
import run_seed_separation as study

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    study.check_donors();data=study.splits()
    with contextlib.redirect_stdout(io.StringIO()):
        old=study.refresh.train_pair(2,99,4,data,10,True)
        new=study.train_pair(99,99,data,10,True)
    keys=['initial_hashes','final_hashes','first_step_max_gradient_difference','training_batches_sha256','parameters_per_model']
    for key in keys:assert old[key]==new[key],key
    for a,b in zip(old['history'],new['history']):assert a['step']==b['step'] and a['train']==b['train']
    study.OUT.mkdir(parents=True,exist_ok=True)
    report={'training_only_seed':99,'steps':10,'diagonal_exact_parity':True,
        'checked_fields':keys+['training_losses_and_gradient_norms'],
        'source_definitions':study.refresh.pilot.engine.loaded,
        'initial_hashes':new['initial_hashes'],'final_hashes':new['final_hashes'],
        'script_sha256':study.sha(__file__)}
    (study.OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified donor provenance and exact ten-step diagonal parity; no held-out evaluation.')
if __name__=='__main__':main()
