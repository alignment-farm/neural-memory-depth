# Configuration follow-up: gradient refresh is the clearest new lead

13 September 2026. Literature and static-code analysis; no neural experiments. This extends the [v1 comparison](INITIAL_COMPARISON.md) without replacing its sources silently.

**Subsequent finding:** the [outer-gradient audit](OUTER_GRADIENT_AUDIT.md) identifies a derivative omission and makes differentiation semantics the immediate control before the chunk comparison proposed here.

**The apparent conflict remains an unresolved difference between operating conditions. The next comparison should prioritize gradient-reference refresh frequency, while controlling memory topology and update scaling.** A later Titans paper supplies a general 16-token chunk setting, outside Modular TTT's published 128–512 depth robustness sweep. This makes frequency a concrete candidate, not an established explanation.

## Later Titans evidence

The [NeurIPS 2025 proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf) is a separately identified publication, not an assumed arXiv version. The proceedings identify the authors and venue; the cached 38-page PDF has creation metadata dated 24 October 2025. See the source ledger for its hash. Relevant pages were rendered and inspected.

| Fact | Starting v1 | Conference paper | Consequence |
| --- | --- | --- | --- |
| Default inner memory | Simple MLP; complete internal recipe not specified in §4.4 | §3.3, p. 7: default two layers, 4× expansion, GELU, residual plus LayerNorm, written as x + LN(W₁σ(W₂x)) | Modular TTT's two square factors with SiLU and optional output RMSNorm are not this exact learner |
| Gradient-reference chunk | §3.2 gives the scheme; depth-run value unverified | Appendix E.1, p. 32: general chunk size 16; §3.1 distinguishes update chunks from attention segments | A small-chunk regime remains outside the published Modular TTT depth sweep |
| Learned write controls | v1 recurrence describes input dependence | Appendices B.3 and F, pp. 29–30, 33: authors say layer-specific controls matter for deeper memory; Appendix F also describes channel-wise forgetting | A concrete author hypothesis, but no isolated numerical ablation for this claim was located |
| Depth result | §5.5, Figure 7, Pile subset | §4.4 and Appendix G.3, Figure 7, p. 34 retain the Pile experiment and 170M/360M/760M labels | General FineWeb-Edu settings still cannot be asserted as the exact depth recipe |

The new details do **not** fully recover the depth experiment. Appendix E's architecture table has a 340M row, while the depth panel is 360M; its learning-rate column also differs from the adjacent prose's 4e-4 value. There are no per-depth widths or checkpoint-to-curve mappings in the inspected material. The two-layer formula does not determine how the one-, three-, and four-layer variants handle residuals and normalization. Fast initialization, Pile sampling/token budget, exact parameter matching and per-depth compute remain open.

The proceedings page exposes a paper link but no separate code or supplement link. Google Research links to OpenReview, whose forum returned a browser verification page. No author code was recovered through these routes. arXiv metadata was retried once in this follow-up, with a descriptive User-Agent and long separation from the first session; the request timed out after 40 seconds with no response body. Thus later arXiv version enumeration remains unverified.

## What the Modular TTT release actually does

Remote HEAD still matches the pinned revision `33afe26100f8590272940e55dbee5067a8040da6`. [Static audit output](../analysis/config_audit.json) is reproducible with `uv run python scripts/audit_configs.py`; it reads JSON and AST defaults without importing or running a model.

**Per-layer controls are already present.** `TTTAttention` obtains one specification per fast node, allocates scalar rate and decay predictor outputs for each node and head, and splits those outputs for graph execution. Scalar here means within a head's factor update, not a single scalar shared by all factors. Therefore, Titans' assertion that layer-specific controls are needed cannot by itself distinguish it from Modular TTT. Channel-wise versus scalar control and momentum remain differences worth isolating.

**The small-rate override reaches the module.** `flame/train.py` loads the JSON using `AutoConfig`; the token-mixer factory forwards `config.s_in_lr_init`; `TTTAttention` uses it in its sigmoid logit offset. At zero predictor output, the ordinary selected configs yield a rate of 1e-3, while the epsilon-0.5 stabilized deep config yields 1e-4. The shell training LR argument is for outer AdamW and does not erase that inner override. This resolves the code plumbing, but no checkpoint manifest establishes the exact config that produced paper loss 3.1144. Treat the difference as an unresolved mapping, not evidence that the published result is incorrect.

**Mean scaling also changes causal reads.** In `graph_memory.py`, the loss-gradient helper is called without its optional mean flag. Scaling is applied later through each fast node's `use_mean` kernel argument. For training/prefill, `xopes/ops/modular_ttt/__init__.py` divides the accumulated within-chunk write contribution by the prefix length for each query, and by full chunk length for the boundary state. The carried initial-state contribution is handled separately with decay. Thus a replacement that divides every gradient by the full chunk size and calls an ordinary forward pass will not reproduce early-prefix reads. This is a static trace of the wrapper, not a numerical audit of the underlying fused kernels.

**Capacity is not exactly matched.** The inspected 12-block configs have 1,179,648 fast-matrix entries for the shallow learner and 2,359,296 for two-factor learners. Rate-plus-decay predictor weights likewise rise from 110,592 to 221,184. These are shape-derived component counts, not full-model counts. The negative depth result is not obviously caused by a smaller deep fast state; nevertheless width, rank, and total resource matching remain distinct issues.

## Revised smallest useful comparison

For 2,048 tokens, chunk lengths 16 and 256 imply 128 and 8 gradient-reference refreshes respectively. This is **16× more refreshes**, not 16× more gradient FLOPs: both process all tokens, while sequential dependencies and kernel efficiency differ. Smaller chunks evaluate later gradients at more recently updated factors. Factor coupling makes that difference potentially relevant for a deep learner. This is our mechanistic inference, not a reproduced result.

Replace the first note's momentum-first suggestion with a staged comparison:

1. In one implementation, compare one versus two fast factors at chunk sizes 16 versus 256, with the same data, backbone, views, decay, reset policy and equal tuning allowance. Start with the already trainable no-mean activation-only deep graph to avoid conflating prefix averaging with refresh frequency. Record validation loss, inner-loss change, update norms and wall time. The key quantity is the change in the deep-minus-shallow loss gap between chunk sizes.
2. If this interaction is absent, test momentum in the same controlled setup, then the residual/GELU/LayerNorm recipe and channel-wise controls. Do not combine all changes and attribute a gain to one mechanism.
3. Report fixed-token, parameter/capacity-matched and compute-matched comparisons separately. Smaller chunks alter runtime; equal matrix counts can impose a bottleneck rank constraint; momentum adds state. None of these matches substitutes for the others.

This plan goes beyond Modular TTT's existing 128–512 sweep and zero/Gaussian initialization checks. It is a mechanism test unless the original run recipes are recovered. A model pull is unnecessary for the written synthesis. Any later experiment needs an implementation exposing fast weights, gradients and outer training; a chat-completions endpoint alone does not establish those capabilities. No model or server configuration is requested at this stage.
