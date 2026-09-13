# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.0"]
# ///
"""CPU witness for a released custom-gradient omission; no model training.

Loads unchanged AST definitions from the pinned source (including its layout
wrapper), avoiding unrelated CUDA/Triton imports. Compares a small composed
one-step learner to full autodiff and central differences. Not full-model or
fused-kernel validation. Run: uv run scripts/check_outer_gradients.py
"""
import ast
import functools
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT/'sources/Modular-TTT'
REV = '33afe26100f8590272940e55dbee5067a8040da6'
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip() == REV
BASE = REPO/'modular_ttt/modular_ttt'
loaded = []
ns = {'torch':torch, 'functools':functools}

def load_definitions(relative, names):
    path = BASE/relative
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    loaded.append({'path':str(path.relative_to(REPO)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'definitions':names})

load_definitions('utils/utils.py',['contiguous'])
load_definitions('ttt/grad/layer/linear.py',['LinearGradFn','linear_grad'])
load_definitions('ttt/grad/layer/act.py',['_get_act_grad','ActGradFn','act_grad'])
# Only the SiLU branch is used, so GELU constants are not required.
load_definitions('ttt/grad/loss/mse.py',['MSELossGradFn','mse_loss_grad'])
linear_grad = ns['linear_grad']; act_grad = ns['act_grad']; mse_grad = ns['mse_loss_grad']

class CompleteLinearTransport(torch.autograd.Function):
    """Local candidate VJP for these batched shapes; upstream stays unchanged."""
    @staticmethod
    def forward(ctx, x, w, dy):
        ctx.save_for_backward(w,dy)
        return torch.einsum('b n h e,b h d e->b n h d',dy,w)

    @staticmethod
    def backward(ctx, ddx):
        w,dy=ctx.saved_tensors
        dw=torch.einsum('b n h d,b n h e->b h d e',ddx,dy)
        ddy=torch.einsum('b n h d,b h d e->b n h e',ddx,w)
        return None,dw,ddy

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def mm(x,w): return torch.einsum('b n h d,b h d e->b n h e',x,w)
def objective(mode, depth, data):
    k,v,q,w1,w2 = data
    weights = [w1] if depth == 1 else [w1,w2]
    inputs, preacts = [], []
    y = k
    for w in weights:
        inputs.append(y)
        z = mm(y,w); preacts.append(z)
        y = torch.nn.functional.silu(z)
    dy = mse_grad(y,v) if mode in ['released','complete_vjp'] else y-v
    dweights = []
    for j in reversed(range(depth)):
        if mode in ['released','complete_vjp']:
            dz = act_grad(preacts[j],dy,'silu')
        else:
            sig = torch.sigmoid(preacts[j])
            dz = dy*sig*(1+preacts[j]*(1-sig))
        dweights.append(torch.einsum('b n h d,b n h e->b h d e',inputs[j],dz))
        if j:
            if mode == 'released':
                dy = linear_grad(inputs[j],weights[j],dz)
            elif mode == 'complete_vjp':
                dy = CompleteLinearTransport.apply(inputs[j],weights[j],dz)
            else:
                transport_w = weights[j].detach() if mode == 'stop_transport' else weights[j]
                dy = torch.einsum('b n h e,b h d e->b n h d',dz,transport_w)
    dweights.reverse()
    # Single write/read, no momentum/decay/mean scaling: an algebraic witness,
    # not the paper hyperparameters. A larger LR makes the omitted term visible.
    adapted = [w - 0.2*g for w,g in zip(weights,dweights)]
    out = q
    for w in adapted: out = torch.nn.functional.silu(mm(out,w))
    outer_loss = 0.5*(out-0.7).square().sum()
    return outer_loss,out,adapted

def finite_difference(depth,data,index,eps=1e-5):
    target=data[index]; fd=torch.zeros_like(target)
    for i in range(target.numel()):
        plus=[x.detach().clone() for x in data]; minus=[x.detach().clone() for x in data]
        plus[index].view(-1)[i]+=eps; minus[index].view(-1)[i]-=eps
        fd.view(-1)[i]=(objective('released',depth,plus)[0]-objective('released',depth,minus)[0])/(2*eps)
    return fd

# Direct primitive witness: same numerical output, different derivative wrt W.
x=torch.ones(1,1,1,2,requires_grad=True)
w=torch.tensor([[[[1.,2.],[3.,4.]]]],requires_grad=True)
dy=torch.tensor([[[[0.4,-0.2]]]],requires_grad=True)
a=linear_grad(x,w,dy); b=torch.einsum('b n h e,b h d e->b n h d',dy,w)
ga=torch.autograd.grad(a.sum(),w,allow_unused=True)[0]
gb=torch.autograd.grad(b.sum(),w)[0]
assert ga is None and gb.abs().max()>0
primitive_gradcheck = torch.autograd.gradcheck(linear_grad,(x,w,dy),raise_exception=False)
complete_gradcheck = torch.autograd.gradcheck(CompleteLinearTransport.apply,(x,w,dy))
assert not primitive_gradcheck and complete_gradcheck

rows=[]
for seed in [0,1,2]:
    gen=torch.Generator().manual_seed(seed)
    data=[torch.randn(shape,generator=gen).mul(0.6).requires_grad_() for shape in [(1,1,1,2)]*3+[(1,1,2,2)]*2]
    for depth in [1,2]:
        results={mode:objective(mode,depth,data) for mode in ['released','full','stop_transport','complete_vjp']}
        selected=data[3:3+depth]
        gradients={mode:torch.autograd.grad(result[0],selected,retain_graph=True) for mode,result in results.items()}
        output_error=(results['released'][1]-results['full'][1]).abs().max().item()
        state_error=max((a-b).abs().max().item() for a,b in zip(results['released'][2],results['full'][2]))
        fd=[finite_difference(depth,data,3+i) for i in range(depth)]
        fd_full=max((a-b).abs().max().item() for a,b in zip(fd,gradients['full']))
        fd_released=max((a-b).abs().max().item() for a,b in zip(fd,gradients['released']))
        stop_error=max((a-b).abs().max().item() for a,b in zip(gradients['released'],gradients['stop_transport']))
        difference=max((a-b).abs().max().item() for a,b in zip(gradients['released'],gradients['full']))
        complete_error=max((a-b).abs().max().item() for a,b in zip(gradients['complete_vjp'],gradients['full']))
        assert complete_error < 1e-12
        assert output_error<1e-12 and state_error<1e-12 and fd_full<1e-8 and stop_error<1e-12
        assert difference<1e-12 if depth==1 else difference>1e-7
        rows.append({'seed':seed,'depth':depth,'output_max_abs_difference':output_error,'adapted_weight_max_abs_difference':state_error,
            'released_vs_full_outer_gradient_max_abs_difference':difference,
            'full_vs_central_difference_max_abs_error':fd_full,'released_vs_central_difference_max_abs_error':fd_released,
            'released_vs_explicit_stop_transport_gradient_max_abs_error':stop_error,
            'complete_vjp_vs_full_outer_gradient_max_abs_error':complete_error})
report={'kind':'CPU float64 isolated differentiation witness; no trained model or language-model result',
    'source_revision':REV,'sources':loaded,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'runtime':{'torch':torch.__version__,'torch_git':torch.version.git_version,'python':platform.python_version(),'device':'cpu','dtype':'float64'},
    'primitive':{'forward_max_abs_difference':(a-b).abs().max().item(),'released_weight_derivative':None,'full_weight_derivative':gb.tolist(),
        'released_gradcheck_passed':primitive_gradcheck,'complete_vjp_gradcheck_passed':complete_gradcheck},
    'cases':rows,
    'limits':['No CUDA or fused kernels executed.','Only single-token one-step Linear-SiLU and Linear-SiLU-Linear-SiLU witnesses.',
        'No inference about which source revision produced paper tables.','Omission may be deliberate; no language-model effect size measured.']}
p=ROOT/'analysis/outer_gradient_check.json'; p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
