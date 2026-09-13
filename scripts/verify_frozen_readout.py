# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
import json
import torch
from torch.nn import functional as F
import probe_frozen_readout as probe

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    root,out=probe.ROOT,probe.OUT;manifest=json.loads((out/'manifest.json').read_text())
    for group in ['files','inputs']:
        for path,digest in manifest[group].items():assert probe.study.sha(root/path)==digest,path
    data=probe.study.splits();checks=[]
    for case,label,stream in probe.CASES:
        for mode in ['released','complete']:
            r=json.loads((out/f'{case}-{mode}.json').read_text());original=probe.study.RecallModel(2,4)
            original.load_state_dict(torch.load(root/r['input_checkpoint']['path'],weights_only=True))
            for key in ['input_checkpoint','checkpoint','float64_head']:assert probe.study.sha(root/r[key]['path'])==r[key]['sha256']
            model=probe.study.RecallModel(2,4);model.load_state_dict(torch.load(root/r['checkpoint']['path'],weights_only=True))
            assert probe.nonhead_hash(original)==probe.nonhead_hash(model)==r['nonhead_hash_before']==r['nonhead_hash_after']
            fitted=torch.load(root/r['float64_head']['path'],weights_only=True)
            for (split,d),seed in zip(data.items(),[8300,8100,8200]):
                x,y,k=probe.features(original,d,mode,seed)
                old=probe.metrics(F.linear(x,original.head.weight.double(),original.head.bias.double()),y,k)
                new=probe.metrics(F.linear(x,fitted['weight'],fitted['bias']),y,k)
                assert old==r['original_metrics'][split] and new==r['probe_metrics'][split]
                if split=='train':
                    assert probe.tensor_hash(x)==r['training_feature_sha256']
                    assert torch.equal(x.mean(0),fitted['mean']) and torch.equal(x.std(0,unbiased=False).clamp_min(1e-6),fitted['scale'])
                    w=fitted['weight']*fitted['scale'];objective=F.cross_entropy(F.linear(x,fitted['weight'],fitted['bias']),y)+.5e-4*w.square().sum()
                    assert abs(objective.item()-r['final_regularized_objective'])<1e-9
            assert torch.equal(model.head.weight,fitted['weight'].float()) and torch.equal(model.head.bias,fitted['bias'].float())
            assert probe.study.evaluate(model,data['test'],mode,seed=8200)==r['float32_folded_graph_test']
            checks.append({'case':case,'mode':mode,'nonhead_state_unchanged':True,'all_original_and_probe_metrics_exact':True,
                'training_feature_and_normalization_verified':True,'folded_graph_metrics_exact':True})
            print(f'Verified frozen readout {case}, {mode}.',flush=True)
    (out/'verification.json').write_text(json.dumps({'frozen_files_and_inputs_verified':True,'checks':checks,
        'verifier_sha256':probe.study.sha(__file__)},indent=2)+'\n')
if __name__=='__main__':main()
