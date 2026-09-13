# Neural-memory depth: first source alignment

13 September 2026. **Initial literature and static-code analysis; no reproduction or training runs.** [Source ledger](SOURCES.md).

**Follow-up:** the [conference-version and code audit](CONFIGURATION_FOLLOWUP.md) adds a 16-token general Titans chunk setting and a fuller memory recipe. It updates the suggested next comparison to prioritize gradient-reference refresh frequency. Original v1-specific gaps below remain identified as such.

The reported depth trends differ, but the available evidence does not identify the cause of that difference. Two tempting explanations already fail: Titans' actual depth sweep excludes attention, and Modular TTT's deeper learners remain worse after nonzero initialization and stabilization. The productive question is now: **which update-rule or configuration interaction changes the depth effect, once memory capacity and training resources are accounted for?** This sharpens the README question without changing its scope.

## What the depth experiments actually establish

**Titans, 2501.00663v1, §5.5 and Figures 7–8:** the authors compare memories with one through four layers using a common training procedure on a subset of the Pile. The footnote explicitly restricts this experiment to neural memory without attention. Figure 7 labels its panels 170M, **360M**, and 760M parameters; this is distinct from the 340M scale in the main language-model table. The 170M panel plots sequence lengths 2,000–32,000. Visual inspection confirms that deeper memories outperform the one-layer memory, but does **not** support strictly monotonic improvement at every added layer: the three- and four-layer curves cross. Figure 8 and its discussion report a throughput penalty for depth. [Paper](https://arxiv.org/html/2501.00663v1#S5.SS5).

The nominal parameter labels are evidence of comparisons at stated model scales. They do not reveal exact per-depth counts, how widths or backbone sizes were adjusted, or fast-state sizes. It would be premature either to call the experiment exactly parameter-matched or to explain its result simply as extra parameters. The §5.1 FineWeb-Edu recipe (15B/30B tokens) must not be substituted for the explicitly different Pile experiment. The latter's token budget and exact configuration mapping remain unverified.

**Modular TTT, 2608.07110v1, §4.1 and Appendix 9.4:** at the 160M scale, the shallow Linear–SiLU reference has validation loss 3.0205. The best completed deeper row, Linear–SiLU–Linear–Norm with RMSNorm epsilon 0.5, has 3.1144, a gap of 0.0939. A two-linear-map learner without activation has 3.1265; an activation-only two-map learner reaches 3.1156. These are reported losses, not measurements made here. Tables 22–27 examine activation placement, mean scaling, normalization epsilon, residual/gated graphs, initialization, and chunk sizes. The selected deep learner remains behind shallow learners at chunk sizes 128, 256, and 512 at both 160M and 410M. [Paper](https://arxiv.org/html/2608.07110v1#S9.SS4).

The ablation recipe uses 2,048-token sequences and 20,000 outer steps, approximately 10.5B tokens (rounded to 10B), with GPT-2 BPE. The 100B-token, 4K-context results are a later scale-up of selected variants, not the primary depth sweep. The inspected text describes a large English pretraining corpus; exact dataset composition and validation split are not established here. Absolute perplexity/loss across the two papers is therefore not a common outcome measure.

**Subsequent implementation check:** the [outer-gradient audit](OUTER_GRADIENT_AUDIT.md) shows that the released graph implementation omits one derivative through linear error transport. The generic outer-loop description below must not be read as a claim of complete differentiation.

## Common mechanism and configuration map

Let slow parameters be θ, sequence-local fast weights be W, and learned input views be k, v, q. A write uses an associative inner loss ℓ(f_W(k), v); a read evaluates f_W(q) with the causally appropriate fast state. The outer language-model objective trains θ through the inner computation. This is the formulation in [TTT layers v4, §§2.1–2.3](https://arxiv.org/html/2407.04620v4#S2), where the inner target itself is learned through outer-trained projections. Inner loss and next-token loss are different objectives.

| Dimension | Titans v1 | Modular TTT v1 / inspected release |
| --- | --- | --- |
| Mutable memory | MLP weights; additional momentum state | Fast weights of graph nodes; chain depth counts trainable linear factors, not activation nodes |
| Inner objective | Squared key–value reconstruction error, Eq. 12 | MSE in the depth sweep; mean scaling is an explicit ablation |
| Write dynamics | S_t = η_t S_(t−1) − θ_t g_t; W_t = (1−α_t)W_(t−1) + S_t (paper's symbols, Eqs. 13–14) | SGD-style graph updates with scalar decay and learned scalar rates; the tested path omits Titans-style momentum |
| Gradient schedule | §3.2 evaluates gradients at a shared chunk reference and accumulates updates; token-dependent rate/decay/momentum coefficients | One train-view forward/backward per chunk, followed by causal query-view evaluation and state transition |
| Update count | Do not infer multiple optimization steps per chunk from momentum or depth; exact §5.5 chunk size is unverified | One-step means no repeated optimization of the same chunk; weights still evolve over successive chunks, and reads include causal within-chunk writes |
| Initialization | Exact §5.5 fast-weight initialization not established | Gaussian std 0.02 initializes learned meta weights; new uncached sequences start from those meta weights, not newly sampled Gaussian weights |
| Normalization / local processing | §4.4 describes Q/K L2 normalization, SiLU Q/K/V processing, depthwise convolution, residuals and output normalization/gating; full depth-run mapping unavailable | Inspected graph configs use key L2, no query normalization, RMSNorm backbone, GLU channel mixer, gated output; learner normalization is varied separately |
| Persistence and reset | Sequence recurrence is specified; exact training packing/reset implementation unverified; separately named persistent memory is not online fast state | `reset_ttt_state` initializes from meta weights or supplied cache; graph state carries across chunks. Document boundaries within packed training data are not verified |
| Outer training | Autoregressive language modeling; exact Pile depth recipe incomplete | Autoregressive cross-entropy; fixed backbone/data/token budget in reported ablations |

Sources: Titans §§3.1–3.3, 4.4; Modular TTT §§3, 7.1–7.2, 9.4–9.5; code locations in the ledger. This table separates paper-level specifications from implementation observations. Neither serving access nor permanent cross-session learning is implied.

## What the code adds—and leaves unresolved

The author-linked Modular TTT checkout is pinned to `33afe26100f8590272940e55dbee5067a8040da6`. Its README explains that `90m`, `310m`, and `1_2b` filenames correspond to paper scales 160M, 410M, and 1.45B because the filename convention excludes vocabulary parameters approximately. Filenames are not exact parameter accounting.

At the 160M scale, inspected shallow and two-factor graph configs retain a 768-wide, 12-block backbone with six 128-dimensional heads. The two-factor hidden width is also 128. Thus each block's fast matrices have 6×128×128 = 98,304 entries for one factor versus 196,608 for two. Across 12 blocks this is 1,179,648 versus 2,359,296 matrix entries. **This is static shape arithmetic, not an instantiated total parameter count or full cache-memory measurement.** Rate and decay predictors can also change with the number of fast nodes. Matching backbone and approximate total scale does not match fast-state capacity exactly.

There is a configuration provenance issue worth resolving before a run: the released `ttt_mlp_graph_mse_mean_sd_silu_norm_sinlr1e4_official_init_eps05` config spells out `s_in_lr_init: 1e-4`, while Appendix 7.1 gives approximately `1e-3` as the general small-rate initialization. The token-mixer code consumes that override. The filename and graph make it a candidate for the stabilized deep row, but no run manifest was inspected that proves which checkpoint produced 3.1144. We should preserve both facts rather than silently treat the appendix default as the executed setting.

A pinned third-party Titans implementation was also inspected. It explicitly calls itself unofficial and contains additional experiments. Its `MemoryMLP` uses Xavier initialization, hidden expansion factor 2, and GELU between weight matrices. These choices demonstrate an available implementation route; they do not recover Titans Figure 7's recipe. No author-provided Figure 7 run configuration was identified in this first pass.

## Mechanistic interpretation and next discriminating comparison

Modular TTT Appendix 8.4 supplies a useful mechanism, not a cross-paper causal proof. For f(K)=KW₁W₂ and output gradient D, the two factor gradients are KᵀDW₂ᵀ and (KW₁)ᵀD. Each write depends on the other factor. All-zero factors can block both writes; nonzero initialization removes that particular trap but does not remove factor coupling. This does not show that coupling must hurt, or that momentum will fix it. Even the convex-quadratic argument for a single linear factor does not apply unchanged to the nonlinear Linear–SiLU reference.

The best-supported present conclusion is **a conditional depth effect with unresolved attribution**. Hybrid attention and zero initialization are insufficient explanations. Momentum, gradient-reference frequency, feature normalization, data, capacity allocation, and training cost remain candidates. Both papers already have chunked updates and forgetting, so “Titans learns repeatedly while Modular TTT learns only once” and “only Titans forgets” would both misdescribe the evidence.

The next priority is the exact Titans depth recipe and the Modular TTT deep-row run mapping. If those cannot be recovered, a new comparison must be labeled a mechanism test rather than a reproduction. A small useful design would cross one versus two fast factors with momentum off versus on, in one shared implementation, keeping data, views, loss scaling, reset policy, and chunk size fixed. Use nonzero initialization and equal tuning allowance, and measure both inner-loss improvement and held-out next-token loss. A depth-by-momentum interaction, rather than a deeper model's isolated gain, would test this candidate explanation.

Run two resource comparisons separately: (1) equal fast-matrix capacity, with total parameters also recorded—e.g. a two-factor 128→64→128 learner has the same matrix-entry count as a 128→128 learner, but introduces a rank constraint that must be acknowledged; (2) equal measured training compute, including inner backward and outer differentiation, allowing different token exposure if necessary. Equal tokens alone do not imply equal FLOPs or wall time. Momentum state must also be counted. Use repeated seeds before interpreting small differences. No such experiment has been launched or budgeted.
