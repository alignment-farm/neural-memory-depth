# A depth-dependent outer-gradient omission in the released implementation

13 September 2026. **New local numerical verification of isolated source primitives, not a language-model reproduction.** This finding changes the next priority: establish the derivative semantics before using a depth-by-chunk comparison to explain the papers.

**Follow-up:** the [graph execution audit](GRAPH_EXECUTION_AUDIT.md) extends this finding to the released graph path with CPU kernel substitutions, multiple chunks, and a short optimizer diagnostic.

## Finding

In Modular TTT revision `33afe26100f8590272940e55dbee5067a8040da6`, `LinearGradFn` computes the correct input-gradient value for a linear map, but its custom backward omits the derivative of that value with respect to the fast weight. The omission matters when the gradient is used to update an upstream fast factor and the outer objective differentiates through that update.

The relevant source is [`ttt/grad/layer/linear.py`, lines 24–50](https://github.com/ByteDance-Seed/Modular-TTT/blob/33afe26100f8590272940e55dbee5067a8040da6/modular_ttt/modular_ttt/ttt/grad/layer/linear.py#L24). Its forward is, in row-vector notation,

\[
A = D W^\top.
\]

For an incoming outer adjoint H = ∂J/∂A, the complete vector-Jacobian product is

\[
\frac{\partial J}{\partial D}=H W,
\qquad
\frac{\partial J}{\partial W}=H^\top D,
\]

with batch/head axes preserved and the token axis summed. The released backward returns `(None, None, ddy)` for arguments `(x, w, dy)`: it supplies the D derivative, but omits the W derivative. Returning no derivative for x is appropriate because this primitive's value does not depend directly on x. The repository's adjacent `linear_grad_ref` instead uses ordinary autodiff with `create_graph=True`.

This is a **specific stopped derivative**, not a claim that all higher-order gradients are absent. Other routes through the forward computation, loss signal, and read still contribute gradients to the fast weights.

## Why depth exposes it

For f(k) = SiLU(SiLU(kW₁)W₂), computing the first factor's write requires transporting the output error through W₂. That transport depends on W₂ both through its incoming signal and directly as the multiplying matrix. The custom backward drops the latter contribution to the outer gradient. In a one-factor learner, transporting error backward through its only linear map ends at the raw graph input; `graph_memory.py` discards that graph-input gradient, so this particular omission does not enter the one-factor write/read computation.

The static call path is `GraphMemory._compute_module_grads` → `TTTLinear.compute_grad` → `linear_grad` → `LinearGradFn.apply`. The graph caches each factor's incoming write signal and propagates transported signals upstream. This establishes relevance to the inspected implementation, without establishing which exact revision produced the published checkpoints.

## Numerical evidence

Run the [witness script](../scripts/check_outer_gradients.py) with:

```sh
uv run --locked scripts/check_outer_gradients.py
```

The adjacent script lockfile pins the dependency resolution. [Machine-readable results](../analysis/outer_gradient_check.json) include source hashes, script hash, exact PyTorch revision, dtype and device. Execution used PyTorch 2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`), Python 3.13.12, CPU float64. All tensors and fast-weight updates are directly accessible. No Mac Studio endpoint or pretrained model was used.

The script loads unchanged AST definitions for the released linear, SiLU, MSE-gradient and contiguous-layout primitives, avoiding unrelated CUDA imports. It composes a tiny single-token, one-step learner with no decay, momentum or mean scaling. The two-dimensional tensors use three fixed random seeds, scale 0.6 and write rate 0.2. These are deliberately transparent diagnostic values, **not the paper's Gaussian scale or learning rate**. It compares the released custom gradients, full differentiation of the same computation, an explicit stopped-transport variant, and a local candidate that supplies the missing W derivative.

| Check | Result |
| --- | --- |
| Released primitive versus ordinary linear-transport value | Identical |
| Primitive finite-difference `gradcheck` | Released fails; complete local derivative passes |
| One-factor controls, three seeds | Released and full outer gradients identical |
| Two-factor controls, three seeds | Writes and read outputs identical, but outer gradients differ |
| Two-factor max absolute gradient differences | 0.00150519, 0.000281705, 0.00290408 |
| Full differentiation versus central differences of the released numerical forward | Maximum error across all six cases < 1.2×10⁻¹¹, using ε = 10⁻⁵ |
| Released versus explicit stopped-transport differentiation | Identical in all cases |
| Local complete derivative versus full differentiation | Identical in all cases |

The finite differences independently differentiate the numerical forward, which does not execute a custom backward. Agreement of the full gradient and the complete local derivative with this check establishes the missing mathematical term. The gradient differences above are diagnostic magnitudes, not estimates of a validation-loss effect.

The source checkout remains unchanged. The complete local derivative is a candidate for these tested shapes, not a published upstream fix or a validated replacement for the complete fused training implementation.

## Consequence for the study

The previous source comparison treated the outer loop too generically. The map now needs an explicit **outer differentiation rule** row. In the inspected Modular TTT release, the graph uses at least this partial derivative; exact derivative semantics for the original Titans depth runs remain unverified. The reported depth trend therefore cannot yet be assigned solely to capacity, inner-loop factor coupling, momentum or gradient refresh frequency.

Do not infer that the paper's negative result is wrong or that supplying the missing derivative improves quality. A stopped derivative could be intentional, could change stability favorably or unfavorably, and might differ from the code used for the paper. No justification for this specific omission was found in the inspected paper text or local code comments, but that is an access-limited observation, not a claim about author intent.

The smallest useful next empirical control is now one versus two factors crossed with released versus complete transport differentiation, at a fixed chunk size. Match initial tensors, data order, rate/decay controls, forward computation and tuning allowance. The one-factor cells are a negative control for this omission. Validate the full training path's outputs, states and gradients first; supplying this one derivative does not establish that every fused primitive's backward is complete.

Then cross chunk sizes 16 versus 256. With depth d, differentiation rule a and chunk size c, record Δ(a,c) = loss(deep,a,c) − loss(shallow,a,c). Compare Δ(complete,c) − Δ(released,c) for a derivative interaction, and Δ(a,16) − Δ(a,256) for a refresh interaction. A fixed-token result and an equal-measured-compute result answer different questions. Count full state and parameters separately, and preserve the earlier rank warning for equal-fast-matrix designs.

**Scope boundary:** this turn produced an executable differentiation witness. It did not train a language model, verify fused GPU kernels, assess BF16 behavior, or resolve checkpoint provenance. A generic served model would not answer the remaining question; the next resource need would be an auditable training implementation with explicit mutable state and gradients.
