# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0", "einops==0.8.1"]
# ///
"""Released TTT graph execution on CPU with explicit scalar-decay kernel stand-in.
Unchanged source AST definitions are loaded; the CUDA kernel and cumsum are
substituted. Tests are diagnostics, not language modeling or reproduction.
Run: uv run --locked scripts/check_graph_gradients.py
"""
import ast
import dataclasses
import functools
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'sources/Modular-TTT'
BASE=REPO/'modular_ttt/modular_ttt'
REV='33afe26100f8590272940e55dbee5067a8040da6'
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==REV
ns=dict(globals(),dataclass=dataclasses.dataclass)
loaded=[]

def load(path,names):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    loaded.append({'path':str(path.relative_to(REPO)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'definitions':names})

# Dtype-preserving CPU scalar-decay implementation. Not a Triton kernel test.
def cpu_lightning(q,k,v,ld=None,initial_state=None,**kwargs):
    assert kwargs.get('decay_type','scalar')=='scalar'
    assert kwargs.get('cu_seqlens') is None
    b,n,h,d=q.shape; e=v.shape[-1]
    assert ld is None or ld.shape == (b,n,h)
    state=q.new_zeros(b,h,d,e) if initial_state is None else initial_state
    out=[]
    for t in range(n):
        decay=1 if ld is None else ld[:,t].exp()[...,None,None]
        state=decay*state+torch.einsum('bhd,bhe->bhde',k[:,t],v[:,t])
        out.append(torch.einsum('bhd,bhde->bhe',q[:,t],state))
    return torch.stack(out,1),state

def cpu_cumsum(x,dim=1,cu_seqlens=None,**kwargs):
    assert cu_seqlens is None
    return torch.cumsum(x,dim=dim)

load(BASE/'utils/utils.py',['contiguous'])
load(BASE/'ttt/grad/layer/linear.py',['LinearGradFn','linear_grad'])
load(BASE/'ttt/grad/layer/act.py',['_get_act_grad','ActGradFn','act_grad'])
load(BASE/'ttt/grad/loss/mse.py',['MSELossGradFn','mse_loss_grad'])
load(BASE/'ttt/loss/__init__.py',['get_loss_fn'])
ns['get_loss_grad_fn']=lambda loss: ns['mse_loss_grad'] if loss=='mse' else (_ for _ in ()).throw(ValueError(loss))
load(BASE/'ttt/optim/base_optim.py',['TTTOptBase'])
load(BASE/'ttt/optim/sgd.py',['TTTSGD'])
load(BASE/'ttt/optim/__init__.py',['get_opt_fn'])
ns.update(lightning_attn_func=cpu_lightning,cumsum_triton=cpu_cumsum)
load(REPO/'xopes/xopes/ops/modular_ttt/__init__.py',['lightning_attn_func_warpper','modular_ttt_func'])
load(BASE/'ttt/modules/ttt_linear.py',['TTTLinear'])
load(BASE/'ttt/modules/ttt_act.py',['get_ttt_activation_fn','TTTAct'])
load(BASE/'ttt/utils/__init__.py',['get_chunk_indices'])
ns.update(_GRAPH_NODE_REGISTRY={},GraphNodeBuilder=Callable)
load(BASE/'ttt/memory/graph_registry.py',['GraphNodeBuildContext','register_graph_node','get_graph_node_builder','build_linear_node','build_act_node'])
ns['register_graph_node']('linear',ns['build_linear_node'])
ns['register_graph_node']('act',ns['build_act_node'])
load(BASE/'ttt/memory/graph_memory.py',['TTTGraphNode','TTTGraphMemory'])
released_linear=ns['linear_grad']

def complete_linear(x,w,dy):
    return torch.einsum('b n h e,b h d e->b n h d',dy,w)

def mm(x,w): return torch.einsum('bnhd,bhde->bnhe',x,w)

def make_graph(depth,chunk,mean,seed):
    torch.manual_seed(seed)
    specs=[]; prev='input'
    for j in range(depth):
        specs.append({'name':f'w{j}','type':'linear','inputs':[prev],'d_out':'output'})
        specs.append({'name':f'a{j}','type':'act','inputs':[f'w{j}'],'activation':'silu'})
        prev=f'a{j}'
    return ns['TTTGraphMemory'](h=1,d=3,e=3,graph_nodes=specs,graph_output=prev,
        chunk_size=chunk,mean_loss=mean,init_std=0.3,init_style='official')

def make_data(depth,seed):
    gen=torch.Generator().manual_seed(100+seed)
    def rand(shape,scale): return (torch.randn(shape,generator=gen)*scale).requires_grad_()
    q,k,v=[rand((1,6,1,3),0.6) for _ in range(3)]
    lr=[(torch.rand(1,6,1,1,generator=gen)*0.1+0.1).requires_grad_() for _ in range(depth)]
    ld=[(-torch.rand(1,6,1,1,generator=gen)*0.1).requires_grad_() for _ in range(depth)]
    target=torch.randn(1,6,1,3,generator=gen)*0.4
    return q,k,v,lr,ld,target

def graph_run(model,data,mode):
    ns['linear_grad']=released_linear if mode=='released' else complete_linear
    q,k,v,lr,ld,_=data
    out,_,state=model(q=q,k=k,v=v,lr=lr,lr_type='scalar',log_f=ld,return_state=True)
    return out,[s[0] for s in state]

# Independent per-token inner autograd + explicit prefix fast matrices.
def reference(model,data):
    q,k,v,lr,ld,_=data
    ws=[p.unsqueeze(0) for p in model.parameters()]
    out=[]
    for start in range(0,q.shape[1],model.chunk_size):
        end=start+model.chunk_size; anchor=ws
        carry=list(anchor); writes=[torch.zeros_like(w) for w in anchor]
        for t in range(start,end):
            pred=k[:,t:t+1]
            for w in anchor: pred=F.silu(mm(pred,w))
            loss=0.5*(pred-v[:,t:t+1]).square().sum()
            grads=torch.autograd.grad(loss,anchor,create_graph=True)
            now=[]
            for j,w in enumerate(anchor):
                decay=ld[j][:,t].squeeze(-1).exp()[...,None,None]
                carry[j]=decay*carry[j]
                writes[j]=decay*writes[j]-lr[j][:,t,...,None]*grads[j]
                denom=t-start+1 if model.mean_loss else 1
                now.append(carry[j]+writes[j]/denom)
            read=q[:,t:t+1]
            for w in now: read=F.silu(mm(read,w))
            out.append(read)
        ws=now
    return torch.cat(out,1),ws

def loss_of(out,data): return 0.5*(out-data[-1]).square().mean()
def maxerr(xs,ys): return max((x-y).abs().max().item() for x,y in zip(xs,ys))
def leaves(model,data): return list(model.parameters())+list(data[:3])+data[3]+data[4]

def validate(depth,chunk,mean,seed):
    model=make_graph(depth,chunk,mean,seed); data=make_data(depth,seed)
    params=leaves(model,data)
    outputs={}; states={}; gradients={}
    for mode in ['released','complete','reference']:
        out,state=reference(model,data) if mode=='reference' else graph_run(model,data,mode)
        outputs[mode]=out; states[mode]=state
        gradients[mode]=torch.autograd.grad(loss_of(out,data),params)
    assert outputs['complete'].shape == outputs['reference'].shape == data[0].shape
    out_error=maxerr([outputs['complete']],[outputs['reference']])
    state_error=maxerr(states['complete'],states['reference'])
    corrected_error=maxerr(gradients['complete'],gradients['reference'])
    omission=maxerr(gradients['released'],gradients['reference'])
    labels=[f'initial_weight_{j}' for j in range(depth)]+['query','key','value']+[f'inner_rate_{j}' for j in range(depth)]+[f'log_decay_{j}' for j in range(depth)]
    by_input={name:(a-b).abs().max().item() for name,a,b in zip(labels,gradients['released'],gradients['reference'])}
    assert maxerr([outputs['released']],[outputs['complete']])==0
    assert maxerr(states['released'],states['complete'])==0
    assert out_error<1e-11 and state_error<1e-11 and corrected_error<1e-10, (depth,chunk,mean,seed,out_error,state_error,corrected_error)
    assert omission<1e-11 if depth==1 else omission>1e-9
    # Independent finite-difference directional derivative across all inputs,
    # rate/decay controls and initial weights, at the same point.
    gen=torch.Generator().manual_seed(1000+seed)
    directions=[torch.randn(p.shape,generator=gen) for p in params]
    originals=[p.detach().clone() for p in params]
    def at(eps):
        with torch.no_grad():
            for p,original,d in zip(params,originals,directions): p.copy_(original+eps*d)
        return loss_of(graph_run(model,data,'complete')[0],data).item()
    eps=1e-5; fd=(at(eps)-at(-eps))/(2*eps)
    at(0)
    directional={mode:sum((g*d).sum().item() for g,d in zip(gradients[mode],directions)) for mode in gradients}
    assert abs(fd-directional['complete'])<1e-8
    # Future-input perturbations cannot change earlier reads.
    altered=tuple([x.detach().clone().requires_grad_() for x in data[:3]]+[data[3],data[4],data[5]])
    with torch.no_grad():
        for x in altered[:3]: x[:,4:]+=2
    future=graph_run(model,altered,'complete')[0]
    causality=(future[:,:4]-outputs['complete'][:,:4]).abs().max().item()
    reset_error=maxerr([graph_run(model,data,'complete')[0]],[outputs['complete']])
    assert causality<1e-12 and reset_error<1e-12
    return {'depth':depth,'chunk':chunk,'mean_scaling':mean,'seed':seed,
        'forward_vs_reference_max_abs_error':out_error,'state_vs_reference_max_abs_error':state_error,
        'complete_gradient_vs_reference_max_abs_error':corrected_error,
        'released_gradient_vs_reference_max_abs_error':omission,
        'released_gradient_error_by_input':by_input,
        'complete_directional_vs_finite_difference_abs_error':abs(fd-directional['complete']),
        'released_directional_vs_finite_difference_abs_error':abs(fd-directional['released']),
        'causality_max_abs_error':causality,'reset_max_abs_error':reset_error}

def optimizer_trace(depth,seed):
    models={mode:make_graph(depth,3,False,seed) for mode in ['released','complete']}
    data=make_data(depth,seed)
    trace=[]
    for step in range(6):
        vals={}
        for mode,model in models.items():
            out,_=graph_run(model,data,mode); loss=loss_of(out,data)
            grads=torch.autograd.grad(loss,tuple(model.parameters()))
            vals[mode]=loss.item()
            if step<5:
                with torch.no_grad():
                    for p,g in zip(model.parameters(),grads): p.add_(g,alpha=-0.1)
        dist=maxerr(list(models['released'].parameters()),list(models['complete'].parameters()))
        trace.append({'outer_step':step,'pre_update_losses':vals,'parameter_difference':dist,'updates_completed':min(step+1,5)})
    if depth==1:
        assert all(r['parameter_difference']==0 for r in trace)
    else:
        assert trace[-1]['parameter_difference']>1e-9
    return {'depth':depth,'seed':seed,'trace':trace}

def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    cases=[validate(d,c,m,s) for d in [1,2] for c in [1,3] for m in [False,True] for s in [0,1]]
    traces=[optimizer_trace(d,s) for d in [1,2] for s in [0,1]]
    report={'kind':'source-loaded CPU graph verification and five-step synthetic optimizer diagnostic',
        'revision':REV,'runtime':{'torch':torch.__version__,'torch_git':torch.version.git_version,'python':platform.python_version(),'device':'cpu','dtype':'float64'},
        'sources':loaded,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'substitutions':['CUDA lightning attention replaced by scalar-decay CPU recurrence.','Triton cumsum replaced by torch.cumsum.','MSE dispatch narrowed to the released MSE gradient.','Graph registry narrowed to released linear and activation builders.','Complete mode replaces only linear error transport with ordinary differentiable einsum.'],
        'diagnostic_config':{'batch':1,'heads':1,'width':3,'tokens':6,'depths':[1,2],'chunks':[1,3],'mean_scaling':[False,True],'seeds':[0,1],'fast_init_std':0.3,'qkv_std':0.6,'inner_rate_range':[0.1,0.2],'log_decay_range':[-0.1,0.0],'outer_target_std':0.4,'optimizer_steps':5,'optimizer':'SGD on initial fast weights only','outer_lr':0.1},
        'cases':cases,'optimizer_diagnostics':traces,
        'limits':['Only Linear-SiLU chains, scalar decay, full chunks, training/prefill path; no decode, packed documents, norm or branch graphs.','Not a validation of fused CUDA/BF16 kernels.','Optimizer trace reuses one tiny random episode and updates only initial fast weights; no quality or benchmark inference.','Synthetic rate/init/width/context differ from paper; no checkpoint mapping.']}
    (ROOT/'analysis/graph_gradient_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'max_forward_error':max(c['forward_vs_reference_max_abs_error'] for c in cases),
        'max_complete_gradient_error':max(c['complete_gradient_vs_reference_max_abs_error'] for c in cases),
        'max_released_gradient_error':max(c['released_gradient_vs_reference_max_abs_error'] for c in cases),
        'max_finite_difference_error':max(c['complete_directional_vs_finite_difference_abs_error'] for c in cases),'final_parameter_differences':[{'depth':t['depth'],'seed':t['seed'],'difference':t['trace'][-1]['parameter_difference']} for t in traces]},indent=2))

if __name__=='__main__': main()
