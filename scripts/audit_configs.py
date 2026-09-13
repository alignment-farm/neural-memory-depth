"""Static audit of released graph configs; no model imports or training.
Run: uv run python scripts/audit_configs.py
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'sources/Modular-TTT'
config_source = REPO / 'modular_ttt/modular_ttt/models/configuration_ttt.py'
tree = ast.parse(config_source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TTTConfig')
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
defaults = {a.arg: ast.literal_eval(v) for a, v in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}
patterns = {
    'shallow_silu': 'ttt_linear_graph_mse_no_norm_sd_silu_single_sinlr_official_init/*90m*.json',
    'deep_silu_no_mean': 'ttt_mlp_graph_mse_sd_linear_silu_linear_silu_sinlr_official_init/*90m*.json',
    'deep_silu_mean': 'ttt_mlp_graph_mse_mean_sd_linear_silu_linear_silu_sinlr_official_init/*90m*.json',
    'deep_norm_eps05': 'ttt_mlp_graph_mse_mean_sd_silu_norm_sinlr1e4_official_init_eps05/*90m*.json',
}
rows = []
for label, pattern in patterns.items():
    paths = list((REPO/'flame/configs/modular_ttt').glob(pattern))
    assert len(paths) == 1, paths
    path = paths[0]
    raw = json.loads(path.read_text())
    cfg = defaults | raw
    h = cfg['num_heads']; d = cfg['embed_dim'] // h
    assert cfg['v_head_dim'] == -1 and cfg['lr_type'] == cfg['decay_type'] == 'scalar'
    assert not cfg['bias']
    dims = {'input': d}; shapes = []
    for node in cfg['graph_nodes']:
        width = dims[node['inputs'][0]]
        if node['type'] == 'linear':
            out = node['d_out']
            out = d if out == 'output' else cfg['d_mid_list'][int(out.split(':')[1])]
            shapes.append([h, width, out]); width = out
        dims[node['name']] = width
    matrices = sum(math.prod(shape) for shape in shapes) * cfg['num_hidden_layers']
    # Each scalar predictor projects embed_dim to heads * number_of_fast_linear_nodes.
    predictors = 2 * cfg['embed_dim'] * h * len(shapes) * cfg['num_hidden_layers']
    eta = cfg['s_in_lr_init']; scale = 2 if cfg['lr_range'] == '02' else 1
    logit = math.log((eta/scale)/(1-eta/scale))
    recovered = scale/(1+math.exp(-logit))
    assert math.isclose(recovered, eta, rel_tol=1e-12)
    rows.append({'label': label, 'path': str(path.relative_to(REPO)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'chunk_size': cfg['chunk_size'], 'mean_loss': cfg['mean_loss'],
        'inner_rate_at_zero_predictor_output': recovered,
        'inner_rate_explicit_override': 's_in_lr_init' in raw,
        'fast_matrix_shapes_per_block': shapes, 'fast_matrix_entries_all_blocks': matrices,
        'scalar_lr_and_decay_predictor_weights_all_blocks': predictors,
        'topology': [node['type'] + (':' + node['activation'] if 'activation' in node else '') for node in cfg['graph_nodes']],
        'rmsnorm_epsilon': cfg['ttt_rmsnorm_eps']})
result = {'kind': 'static configuration and shape analysis, not executed-model measurement',
    'revision': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    'rows': rows,
    'gradient_reference_refreshes_in_2048_tokens': {str(c): 2048//c for c in [16,128,256,512]},
    'limits': ['No checkpoint-to-table mapping established.', 'Counts omit unrelated parameters and cache buffers.',
               'Rate is at zero predictor output, not a measured token-average rate.']}
out = ROOT/'analysis/config_audit.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
