# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Fit a linear readout to detached, frozen normalized memory features."""
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import run_embedding_swap as study
ROOT=study.ROOT;OUT=ROOT/'analysis/frozen_readout'
CASES=[('value_confusion','q11-k11-v16',11),('key_confusion','q11-k16-v11',15),('successful_control','q11-k11-v11',11)]

def tensor_hash(x):return hashlib.sha256(bytes(x.contiguous().view(torch.uint8).flatten().tolist())).hexdigest()
def nonhead_hash(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():
        if not name.startswith('head.'):h.update(name.encode());h.update(bytes(x.contiguous().view(torch.uint8).flatten().tolist()))
    return h.hexdigest()
@torch.no_grad()
def features(model,data,mode,seed):
    captured=[];handle=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    xs=[];ys=[];keys=[];gen=torch.Generator().manual_seed(seed)
    for start in range(0,len(data),128):
        batch=study.batch_for(data[start:start+128],gen);captured.clear();model(batch,mode)
        xs.append(captured[-1].reshape(-1,16));ys.append(batch[-1].reshape(-1));keys.append(batch[0][:,8:].reshape(-1))
    handle.remove();return torch.cat(xs).double(),torch.cat(ys),torch.cat(keys)
@torch.no_grad()
def metrics(logits,labels,keys):
    ce=F.cross_entropy(logits,labels,reduction='none')
    return {'cross_entropy':ce.mean().item(),'accuracy':(logits.argmax(-1)==labels).double().mean().item(),
        'queries':len(labels),'per_value_ce':[ce[labels==i].mean().item() for i in range(8)],
        'per_key_ce':[ce[keys==i].mean().item() for i in range(8)]}

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    prior=json.loads((study.OUT/'manifest.json').read_text())
    for group in ['files','donor_files']:
        for path,digest in prior[group].items():assert study.sha(ROOT/path)==digest
    verification=json.loads((study.OUT/'verification.json').read_text());assert len(verification['checks'])==160
    inputs={}
    for _,label,stream in CASES:
        path=study.OUT/f'{label}-t{stream}.json';inputs[str(path.relative_to(ROOT))]=study.sha(path)
        r=json.loads(path.read_text())
        for info in r['checkpoints'].values():assert study.sha(ROOT/info['path'])==info['sha256'];inputs[info['path']]=info['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    paths=['notes/FROZEN_READOUT_PROTOCOL.md','scripts/probe_frozen_readout.py','scripts/run_embedding_swap.py',
        'scripts/run_group_swap.py','scripts/run_seed_separation.py','scripts/run_refresh_comparison.py','scripts/run_recall_pilot.py','scripts/check_graph_gradients.py',
        'analysis/embedding_swap/manifest.json','analysis/embedding_swap/verification.json']
    manifest={'files':{p:study.sha(ROOT/p) for p in paths},'inputs':inputs,'cases':CASES,
        'torch':torch.__version__,'torch_git':torch.version.git_version,'fitting_dtype':'float64','feature_dtype':'float32','threads':1,
        'training_feature_order_seed':8300,'max_iter':200,'max_eval':250,'regularization':1e-4}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(manifest))
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    data=study.splits()
    for case,label,stream in CASES:
        record=json.loads((study.OUT/f'{label}-t{stream}.json').read_text())
        for mode in ['released','complete']:
            name=f'{case}-{mode}';result_path=OUT/f'{name}.json'
            if result_path.exists():continue
            model=study.RecallModel(2,4);model.load_state_dict(torch.load(ROOT/record['checkpoints'][mode]['path'],weights_only=True))
            before=nonhead_hash(model)
            cached={split:features(model,d,mode,seed) for (split,d),seed in zip(data.items(),[8300,8100,8200])}
            original={split:metrics(F.linear(x,model.head.weight.double(),model.head.bias.double()),y,k) for split,(x,y,k) in cached.items()}
            assert abs(original['test']['cross_entropy']-record['test'][mode]['cross_entropy'])<1e-6
            assert original['test']['accuracy']==record['test'][mode]['accuracy']
            x,y,_=cached['train'];mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6);z=(x-mean)/scale
            head=nn.Linear(16,8,dtype=torch.float64);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=200,max_eval=250,history_size=20,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')
            evaluations=0
            def closure():
                nonlocal evaluations
                opt.zero_grad();objective=F.cross_entropy(head(z),y)+.5e-4*head.weight.square().sum();objective.backward();evaluations+=1
                return objective
            started=time.perf_counter();opt.step(closure);seconds=time.perf_counter()-started
            final_objective=closure().item();gradient=max(p.grad.abs().max().item() for p in head.parameters())
            with torch.no_grad():
                weight=head.weight/scale;bias=head.bias-weight@mean
                after_metrics={split:metrics(F.linear(a,weight,bias),b,k) for split,(a,b,k) in cached.items()}
                maxerr=max((head((a-mean)/scale)-F.linear(a,weight,bias)).abs().max().item() for a,_,_ in cached.values())
                assert maxerr<1e-8
                model.head.weight.copy_(weight.float());model.head.bias.copy_(bias.float())
            assert nonhead_hash(model)==before
            graph_test=study.evaluate(model,data['test'],mode,seed=8200)
            # Folding into a float32 model may slightly change logits; record rather than hide it.
            folder=ROOT/'models/frozen_readout';folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/f'{name}.pt';torch.save(model.state_dict(),checkpoint)
            probe=folder/f'{name}-float64-head.pt';torch.save({'weight':weight.detach(),'bias':bias.detach(),'mean':mean,'scale':scale},probe)
            state=opt.state[next(iter(head.parameters()))]
            result={'case':case,'label':label,'stream_seed':stream,'mode':mode,'input_checkpoint':record['checkpoints'][mode],
                'original_metrics':original,'probe_metrics':after_metrics,'float32_folded_graph_test':graph_test,
                'float64_fold_equivalence_max_abs_error':maxerr,'nonhead_hash_before':before,'nonhead_hash_after':nonhead_hash(model),
                'optimizer_iterations':state['n_iter'],'optimizer_function_evaluations':state['func_evals'],'closure_calls_including_final':evaluations,
                'final_regularized_objective':final_objective,'final_gradient_infinity_norm':gradient,'fitting_seconds':seconds,
                'training_feature_shape':list(x.shape),'training_feature_sha256':tensor_hash(x),
                'checkpoint':{'path':str(checkpoint.relative_to(ROOT)),'sha256':study.sha(checkpoint)},
                'float64_head':{'path':str(probe.relative_to(ROOT)),'sha256':study.sha(probe)}}
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'completed':name,'original_test':original['test']['cross_entropy'],'probe_test':after_metrics['test']['cross_entropy'],
                'probe_accuracy':after_metrics['test']['accuracy'],'iterations':state['n_iter'],'gradient_inf':gradient}),flush=True)
if __name__=='__main__':main()
