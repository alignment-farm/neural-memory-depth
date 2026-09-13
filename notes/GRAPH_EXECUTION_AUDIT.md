# The derivative omission persists through the released graph path

13 September 2026. **Source-loaded CPU verification and a five-step synthetic optimizer diagnostic. No language-model reproduction.** This extends the [isolated primitive audit](OUTER_GRADIENT_AUDIT.md).

The omitted derivative is not just an artifact of the earlier hand-composed witness. It persists through the released graph orchestrator, per-node state handling, causal update wrapper, and multiple chunks when the CUDA attention kernel is replaced by a transparent CPU recurrence. In the tested chain graphs, restoring differentiation through the linear error-transport weight makes the complete graph agree with an independent reference.

## What was executed

[Script](../scripts/check_graph_gradients.py), [lockfile](../scripts/check_graph_gradients.py.lock), [results and source hashes](../analysis/graph_gradient_check.json). Reproduce with:

```sh
uv run --locked scripts/check_graph_gradients.py
```

The loader executes unchanged AST definitions from Modular TTT revision `33afe26100f8590272940e55dbee5067a8040da6`: `TTTGraphMemory`, graph node construction for linear/SiLU nodes, `TTTLinear`, `TTTAct`, state/SGD classes, loss and gradient functions, the chunk-index helper, and the `modular_ttt_func` update wrapper. It also preserves the source's contiguous-layout decorator.

Two accelerated operations are substituted: scalar-decay lightning attention becomes an explicit CPU recurrence, and Triton cumsum becomes `torch.cumsum`. Dispatch is restricted to the tested MSE and linear/activation nodes. Complete-derivative mode changes only linear error transport to ordinary differentiable `einsum`. The upstream checkout is unchanged. This does **not** validate a CUDA kernel, the full language-model backbone, normalization/branching graph families, packed documents, partial chunks, or decoding.

An independent reference computes each token's inner gradient using ordinary autodiff at the chunk-start weights, constructs the causal prefix fast matrices explicitly, and reads through those updated matrices. Mean scaling divides the within-chunk accumulated writes by prefix length; the carried state is decayed separately. Both implementations process every token and carry boundary states between chunks.

The diagnostic has batch size 1, one head, width 3, six tokens, depths 1/2, chunks 1/3, mean scaling off/on, and seeds 0/1: **16 cases**. All use scalar, token-varying decay and rates. Initialization std 0.3, Q/K/V std 0.6, rates in [0.1, 0.2), and log decay in (-0.1, 0] are synthetic diagnostic settings. These are not paper-scale settings. Runtime: CPU float64, PyTorch 2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`), Python 3.13.12, einops 0.8.1.

## Results

| Check across the 16 cases | Maximum absolute error |
| --- | --- |
| Complete-derivative graph forward versus independent reference | 4.78×10⁻¹⁷ |
| Complete-derivative boundary state versus independent reference | 1.67×10⁻¹⁶ |
| Complete outer gradient versus independent reference | 1.05×10⁻¹⁷ |
| Complete directional derivative versus central finite difference | 2.04×10⁻¹⁰ |
| Released outer gradient versus independent reference | 1.97×10⁻⁴ |

The finite-difference direction jointly perturbs initial weights, Q/K/V, rates and log decays, using epsilon 10⁻⁵. All eight shallow cases agree with the reference to tolerance. All eight deep cases retain a gradient discrepancy. Released and complete modes have exactly identical forward outputs and states at the same parameters; the differing operation is the derivative.

Causality checks perturb Q/K/V at positions 5–6 and confirm unchanged outputs at positions 1–4. Reset checks run an altered sequence and then the original sequence, confirming the latter starts from the same meta weights rather than leftover fast state. These pass for all cases. This does not test reset boundaries inside packed documents or the inference cache.

The discrepancy also propagates beyond the downstream initial weight. For the depth-2, chunk-3, no-mean, seed-0 case, gradients differ for keys, values, and the downstream factor's rate and decay controls. Thus it can affect the representations and write controls learned by an outer model. It is not merely a different numerical gradient label attached to an otherwise disconnected parameter.

## A short optimizer diagnostic

For depths 1/2 and seeds 0/1, two graph copies start with identical weights and reuse the same tiny random episode. Five SGD updates at outer rate 0.1 train only the initial fast weights against a random squared-error read target. Initial loss is identical between derivative modes. Shallow copies remain identical throughout. After five updates the deep copies' maximum parameter differences are 7.43×10⁻⁵ and 9.83×10⁻⁵.

This establishes that the omitted path can change an optimizer trajectory through the released graph. The tiny fixed episode, arbitrary target, non-paper rates and absence of held-out data make it unsuitable for judging learning quality, generalization or the sign of the depth effect. The trace is a diagnostic, not a benchmark. No positive language-model effect is claimed.

## What this changes

The evidential status advances from an isolated gradient counterexample to **graph-level confirmation under explicit CPU kernel substitutions**, including recurrent state and a short optimizer trace. The test supports adding differentiation semantics as a required experimental control. It still does not establish that the omission caused Modular TTT's reported negative depth trend, that it was unintended, or that the source revision matches the checkpoints used in the paper.

The next meaningful empirical step would require the complete execution path and a held-out sequence-modeling task: first test depth × derivative at fixed chunk size, then depth × gradient-refresh frequency. Repeating more tiny witnesses is unlikely to answer the language-model question. The present study can already publish a useful bounded result: reported depth comparisons are conditional, and the released implementation contains a depth-dependent outer-gradient approximation that must be accounted for before assigning the difference solely to the inner optimizer or memory capacity.


**Follow-up completed:** the [held-out recall pilot](RECALL_PILOT_RESULTS.md) now tests depth × derivative across three seeds. Correcting the derivative did not consistently improve final recall under that fixed recipe; the implementation finding remains, with a narrower empirical interpretation.
