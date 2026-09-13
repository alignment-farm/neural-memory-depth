# Source and implementation ledger

Checked 13 September 2026. Files already existed at session start; their original retrieval provenance is unknown. All three cached HTML files were independently fetched from their exact versioned arXiv URLs and were byte-identical. Hashes and verification timestamp are in [provenance.json](../sources/provenance.json). Text extractions are reading aids; the HTML and figure are the source of record.

| Source | Reading scope | Local evidence |
| --- | --- | --- |
| [Titans 2501.00663v1](https://arxiv.org/html/2501.00663v1) | §§3.1–3.3, 4.4, 5.1, 5.5; Figure 7 captions and 170M image; Figure 8 discussion | `sources/2501.00663v1.html`; `.cache/titans-depth-170m.png` |
| [Modular TTT 2608.07110v1](https://arxiv.org/html/2608.07110v1) | §§3–4, 7.1–7.2, 8.4, 9.4–9.5, 10 | `sources/2608.07110v1.html` |
| [TTT layers 2407.04620v4](https://arxiv.org/html/2407.04620v4) | §§2.1, 2.3, 2.6–2.7, as formulation/initialization background | `sources/2407.04620v4.html` |

Metadata request: one batch GET to `https://export.arxiv.org/api/query?id_list=2501.00663v1,2608.07110v1,2407.04620v4&max_results=3`, descriptive User-Agent `NeuralMemoryDepthStudy/0.1 (local academic literature comparison)`. Received HTTP 429, body “Rate exceeded.” No Retry-After header was supplied. Headers and body are cached under `.cache/arxiv/metadata.*`; no immediate retry was attempted. **API metadata verification and later-version review remain open.** Versioned content retrieval succeeded separately. Only one API request was made in this session.

## Code revisions and inspection

[ByteDance-Seed/Modular-TTT](https://github.com/ByteDance-Seed/Modular-TTT/tree/33afe26100f8590272940e55dbee5067a8040da6), author-linked from the paper:

- Revision `33afe26100f8590272940e55dbee5067a8040da6`, commit date 2026-08-10T13:19:08+08:00. Existing local checkout `sources/Modular-TTT`; working tree clean when inspected.
- `README.md` lines 165 onward: config-to-paper scale mapping.
- `modular_ttt/modular_ttt/ttt/modules/ttt_linear.py` lines 51–113: matrix shape, Gaussian initialization and reset from learned meta weights; lines 144–158: kernel dispatch and state update.
- `modular_ttt/modular_ttt/ttt/memory/graph_memory.py` lines 292–314, 359–490: state reset/cache, chunk traversal, train/backward/query passes, per-node learning rates.
- `modular_ttt/modular_ttt/modules/token_mixers/ttt_attention.py` lines 218–224, 309–312: small-rate logit offset and its use; lines 354–358: Q/K normalization.
- Config paths below are relative to `flame/configs/modular_ttt/`; filenames in each directory include the same prefix and `90m` scale:
  - `ttt_linear_graph_mse_no_norm_sd_silu_single_sinlr_official_init/`: shallow reference, mean loss false, one factor.
  - `ttt_mlp_graph_mse_mean_sd_linear_silu_linear_silu_sinlr_official_init/`: two factors, hidden width 128, mean loss true.
  - `ttt_mlp_graph_mse_mean_sd_silu_norm_sinlr1e4_official_init_eps05/`: two factors and non-fast RMSNorm, epsilon 0.5, explicit inner-rate init 1e-4.
- `debug_ttt.sh` lines 169–177 and 306–315: example deep-memory command mappings. These are commands, not authenticated experiment logs.
- Inspection did not execute the model, count all instantiated parameters, audit fused-kernel numerical equivalence, or inspect training checkpoints.

[lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch/tree/1d40c445fa1794fb28582c721786482b5b683e47), **unofficial**:

- Revision `1d40c445fa1794fb28582c721786482b5b683e47`, commit date 2026-07-13T07:26:23-07:00; cloned read-only for inspection to `.cache/titans-pytorch`.
- `README.md` line 7 declares unofficial status and architectural explorations.
- `titans_pytorch/memory_models.py` lines 54–81: `MemoryMLP` shape, initialization and activation.
- `titans_pytorch/neural_memory.py`: located configurable momentum, chunk handling, functional memory evaluation, learned initial parameters and state initialization; not a full correctness audit.
- Not used as evidence of the original Titans experiment configuration. The versioned Titans HTML has no implementation link. A first-pass web search and Google Research publication lead did not identify an author-provided Figure 7 configuration; this is not a claim that none exists.

No model endpoints were contacted, dependencies for training installed, or neural interventions attempted. Python reading utilities were run with `uv`.

## Follow-up publication and audit

- [Titans, NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html), DOI `10.52202/085713-3786`. Retrieved the linked 38-page paper as `sources/titans-neurips2025.pdf`; SHA-256 and URL recorded in `sources/provenance.json`. Its mapping to an arXiv revision is unverified. Extracted with Poppler `pdftotext -layout`; visually inspected pages 7, 29–30, 32–34. Reading scope: §3.3, §4.4, Appendices B.3, E.1, F, G.3, and relevant code-access checklist text. The later publication is kept separate from v1.
- Follow-up query API request used unversioned `id_list=2501.00663,2608.07110,2407.04620`, User-Agent `NeuralMemoryDepthStudy/0.2 (literature and configuration audit)`, one connection, with more than three seconds since any previous API call. It timed out after 40 seconds without a response body. Partial header output is `.cache/arxiv/metadata-followup.headers`. There was no subsequent retry. The earlier session's 429 cache remains preserved.
- Google Research's publication link led to OpenReview forum `8GjSf9Rh7Z`, which returned browser verification. The proceedings landing page supplied a paper but no separate supplementary/code link. These access results do not prove code is unavailable elsewhere.
- `git ls-remote origin HEAD` independently confirmed Modular TTT remote HEAD remains `33afe26100f8590272940e55dbee5067a8040da6`.
- Extended static code trace at that revision: `flame/train.py` lines 289–300 (config loading and model dispatch); `modular_ttt/modular_ttt/models/configuration_ttt.py` lines 67, 109–120 (default and assignment); `modular_ttt/modular_ttt/modules/token_mixers/__init__.py` line 72 (forwarding); `ttt_attention.py` lines 166–224 and 309–335 (per-node controls and rate offset); `ttt/memory/graph_memory.py` lines 76–79, 423–425 (unscaled loss-gradient dispatch); `xopes/xopes/ops/modular_ttt/__init__.py` lines 78–135 (training/prefill prefix scaling); `ttt/optim/base_optim.py` and `sgd.py` (state ownership, no momentum in this path).
- [Audit script](../scripts/audit_configs.py) and [JSON output](../analysis/config_audit.json) record config hashes, defaults, four graph configurations, fast matrix shapes, scalar control predictor counts, rate-offset arithmetic and refresh counts. Run succeeded with stdlib Python through `uv`. This is static arithmetic, not training, numerical kernel validation or an instantiated model parameter count.

## Outer-gradient numerical audit

13 September 2026 follow-on work:

- Source revision remains `33afe26100f8590272940e55dbee5067a8040da6`, with a clean upstream working tree after the check.
- Critical definition: `modular_ttt/modular_ttt/ttt/grad/layer/linear.py`, lines 24–50. Call-path inspection: `ttt/modules/ttt_linear.py`, lines 92–95; `ttt/memory/graph_memory.py`, lines 326–337 and 430–451. The custom transport returns no weight derivative; its ordinary autodiff reference requests `create_graph=True`.
- [Witness script](../scripts/check_outer_gradients.py), [dependency lock](../scripts/check_outer_gradients.py.lock), and [results](../analysis/outer_gradient_check.json). Four unchanged source-definition groups were loaded by AST, including the actual contiguous-layout wrapper; unrelated package imports were not executed. Source and script hashes are stored with results.
- Exact execution resource: local CPU, float64, PyTorch 2.14.0 revision `08187d9e0fba026dc8217405802ab5381dc88d90`, Python 3.13.12. Fast tensors, state updates and gradients were directly accessible. Mac Studio availability and training support remain untested. `uv` managed the isolated script environment.
- Verification: a direct primitive gradcheck, three single-factor controls, three two-factor witnesses, central differences of every initial-weight entry, an explicit stopped-transport comparison and a local complete-VJP candidate. No full model, pretrained checkpoint, outer training run, CUDA kernel or BF16 execution was used.
- [PyTorch autograd API documentation](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html) is supporting implementation documentation; the mathematical finding is established by the pinned source and numerical check, not by a documentation claim.

The initial statements above about static-only inspection describe earlier passes. This last pass adds isolated executable evidence, with its limits detailed in [the audit](OUTER_GRADIENT_AUDIT.md).

## Released graph execution audit

13 September 2026 extension:

- [Graph witness script](../scripts/check_graph_gradients.py), [lockfile](../scripts/check_graph_gradients.py.lock), [machine-readable results](../analysis/graph_gradient_check.json), and [interpretation](GRAPH_EXECUTION_AUDIT.md).
- Unchanged source definitions loaded from revision `33afe26100f8590272940e55dbee5067a8040da6`: graph orchestrator/builders, linear and SiLU modules, MSE loss and gradient, linear/activation gradients, state and SGD classes, chunk-index helper and the `xopes.ops.modular_ttt` wrapper. The JSON enumerates all definitions and hashes. No upstream file was edited.
- Explicit CPU substitutions replace the CUDA lightning-attention kernel and Triton cumsum; dispatch is limited to MSE and linear/SiLU chains. The complete-gradient mode changes the linear transport to an ordinary differentiable einsum. These substitutions prevent claiming full CUDA execution parity.
- Verified 16 cases across depth 1/2, chunk 1/3, mean scaling off/on and seeds 0/1, using six-token sequences with scalar decay. An independent per-token autograd reference checks outputs, states and gradients; directional central differences, causal future-input perturbations and sequence reset checks also pass. The result file includes all diagnostic hyperparameters and per-input gradient differences.
- Four paired five-step synthetic SGD traces (depth × seed) update only initial fast weights on a repeated tiny random episode. These are optimizer diagnostics, not held-out evaluations or paper reproductions. Language-model quality and generalization remain unmeasured.
- Locked run succeeded with local CPU float64, PyTorch 2.14.0 revision `08187d9e0fba026dc8217405802ab5381dc88d90`, Python 3.13.12 and einops 0.8.1. No remote model endpoint was called.


## Held-out synthetic recall pilot

13 September 2026 extension, authorized after the graph audit:

- [Frozen protocol](RECALL_PILOT_PROTOCOL.md), [results](RECALL_PILOT_RESULTS.md), [manifest](../analysis/recall_pilot/manifest.json), [summary](../analysis/recall_pilot/summary.json), and [verification](../analysis/recall_pilot/verification.json). This is a new local experiment, not a result reported by either paper.
- [Training script](../scripts/run_recall_pilot.py), [summary script](../scripts/summarize_recall_pilot.py), and [verifier](../scripts/verify_recall_pilot.py), each with an adjacent uv lockfile. The training script imports the previously audited graph engine at the same upstream revision, with the same explicit CPU kernel substitutions. No upstream source was changed.
- Before validation/test runs, the manifest froze SHA-256 hashes of the protocol (`7ccacaf4d88e97d8f7be647a46496fc8b73b73ad18949da2e8c66cac5c712182`), training script (`de9150ce911cfc0d53065c3edf5f6dc1891c8037f404b351657a9248c0161ba4`) and graph engine (`9432ac4c142501480f83aec8b3ec899c6f5338d62c77d946196a3db3c23b2030`). A separate training-only smoke run used seed 99; no held-out outcome was used to tune the recipe.
- Synthetic data enumerate all 40,320 permutations of eight values, split into 32,000 training, 4,096 validation and 4,224 test mappings with no overlap. Split hashes and order seeds are recorded in the manifest; no external dataset was downloaded.
- Twelve CPU float32 models: three seeds × two depths × two derivative rules, 600 updates each. PyTorch 2.14.0 revision `08187d9e0fba026dc8217405802ab5381dc88d90`, einops 0.8.1, one CPU thread and deterministic algorithms. Mutable states and gradients were directly accessible on the local machine; Mac Studio capabilities remain untested.
- The verifier reloaded all 12 checkpoints, checked their hashes and exact test/no-adaptation metrics, regenerated isolated splits, checked paired initial losses and shallow final equality, and confirmed that changing written values cannot affect logits when both writes and decay are disabled. Checkpoints remain local under `models/recall_pilot/`; per-pair JSON files retain their hashes and full training histories.
- Results are fixed-token comparisons. Derivative pairs share initial weights, batches and parameter counts; depth comparisons are neither parameter-matched nor compute-matched. Natural-language quality, fused CUDA parity and original-paper checkpoint provenance remain unverified. No remote model endpoint was used.


## Four-chunk gradient-refresh comparison

13 September 2026 follow-up, authorized after the proposed ten-seed experiment:

- [Protocol](REFRESH_COMPARISON_PROTOCOL.md), [results](REFRESH_COMPARISON_RESULTS.md), [manifest](../analysis/refresh_comparison/manifest.json), [summary](../analysis/refresh_comparison/summary.json) and [verification](../analysis/refresh_comparison/verification.json). This new local experiment follows the initial pilot; the existing synthetic data split is reused and ten fresh training seeds 10–19 are introduced.
- [Runner](../scripts/run_refresh_comparison.py) imports the frozen pilot helpers and changes only the model's `memory.chunk_size` to 1/2/4/8. Two depths × two derivatives × four chunks × ten seeds gives 160 models, each trained for 600 updates. Per-token rate/decay formulas and mean-loss false remain unchanged; chunks divide both write and query phases. All starts and minibatches match across chunks at fixed depth/seed.
- [Preflight](../scripts/check_refresh_preflight.py) extends the independent graph reference checks to all four chunks, using 24-token float64 inputs and diagnostic seed 99. Its [record](../analysis/refresh_comparison/preflight.json) stores every loaded upstream definition and source-file hash; the manifest freezes that record, protocol, runner and imported helper hashes before training.
- [Verifier](../scripts/verify_refresh_comparison.py) regenerated data splits and all 600 minibatches per seed, checked cross-chunk initial equality, and reloaded all 160 checkpoints to reproduce exact final test/no-adaptation metrics. Source definitions match the preflight hashes. The pinned upstream revision remains `33afe26100f8590272940e55dbee5067a8040da6`.
- [Summary script](../scripts/summarize_refresh_comparison.py) records individual results, prespecified primary/corroborating contrasts, secondary learning-speed summaries and descriptive paired-seed bootstrap intervals. All four new scripts have adjacent uv lockfiles. Training used CPU float32, PyTorch 2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`), einops 0.8.1, one thread and deterministic algorithms. Matplotlib 3.11.2 produced the plots, which were visually checked.
- Checkpoints remain local under `models/refresh_comparison/`. JSON records retain checkpoint and batch hashes, validation histories and training-loop timing. Total training events: 49,152,000; measured training-loop CPU wall time: 428.14 seconds, excluding evaluation and batch construction. Timing is descriptive, not a compute-matched benchmark.
- No original-paper checkpoint, natural-language training, remote model endpoint or fused CUDA/BF16 execution was used. The result constrains the proposed refresh mechanism under one small-task recipe; initialization and training-order contributions to persistent seed failures remain unseparated.


## Initialization × sampled training-stream separation

13 September 2026 extension, explicitly authorized after the refresh comparison:

- [Protocol](SEED_SEPARATION_PROTOCOL.md), [results](SEED_SEPARATION_RESULTS.md), [manifest](../analysis/seed_separation/manifest.json), [summary](../analysis/seed_separation/summary.json) and [verification](../analysis/seed_separation/verification.json). All initialization seeds 10–19 are crossed with all stream seeds 10–19 at depth 2/chunk 4 under both derivative rules. There are 180 new model runs plus 20 explicitly marked reused diagonal outcomes from the refresh comparison.
- [Runner](../scripts/run_seed_separation.py) imports the frozen model and helpers unchanged and separates the global initialization seed from the independent batch-generator seed. “Stream” includes sampling of mappings and both minibatch and episode order. “Initialization” includes all initially random parameters, not only fast matrices. No other hyperparameter is changed.
- [Training-only preflight](../scripts/check_seed_preflight.py) verifies donor provenance and exact ten-step diagonal parity with the old runner at excluded diagnostic seed 99. Its [record](../analysis/seed_separation/preflight.json) retains source hashes and matched state hashes. No held-out metric was used in preflight. Protocol/runner/helper/preflight hashes, donor JSON hashes and prior manifest/verification hashes were frozen before off-diagonal training.
- [Verifier](../scripts/verify_seed_separation.py) checks all 200 checkpoint hashes, reproduces final test, final validation and no-adaptation metrics exactly, regenerates every 600-step stream, verifies row-constant initial state and column-constant batch hashes, and confirms exact donor preservation. It also verifies source hashes, split isolation, masked query targets and no-adaptation value invariance.
- [Summary/plot script](../scripts/summarize_seed_separation.py) records the full CE matrix, row/column summaries, paired derivative changes, threshold counts and descriptive two-factor sums-of-squares decomposition. The decomposition passes four known-case checks, and all shares sum to total variation. Interaction is nonadditive dependence over the fixed grid, not independent noise or a population causal-variance estimate. Both plots were visually inspected.
- All four new scripts have adjacent uv lockfiles. Training and verification use CPU float32, one thread, deterministic algorithms, PyTorch 2.14.0 revision `08187d9e0fba026dc8217405802ab5381dc88d90` and einops 0.8.1; plotting uses Matplotlib 3.11.2. New training-loop time totals 530.43 seconds and 55,296,000 events. Reused models contribute no new training time or events. These are fixed-event, not compute-matched comparisons.
- New checkpoints are local under `models/seed_separation/`; donor checkpoint paths retain their original location. Source revision stays `33afe26100f8590272940e55dbee5067a8040da6`, with explicit CPU kernel substitutions. No upstream modification, remote model call or natural-language training was performed. Initial parameter-group attribution remains unresolved.

## Reciprocal initial parameter-group interventions

13 September 2026, performed under explicit authorization to continue autonomously:

- [Full group protocol](GROUP_SWAP_PROTOCOL.md), [results](GROUP_SWAP_RESULTS.md), [manifest](../analysis/group_swap/manifest.json) and [verification](../analysis/group_swap/verification.json). Selected untrained donors 11 and 16 supply fast matrices, all embeddings and readout in a complete 2×2×2 factorial over ten streams and both derivative modes. There are 120 new models and 40 reused baseline outcomes. Every tensor origin and all 160 final test/validation/ablation metrics were verified; pure baselines retain earlier checkpoint paths.
- [Embedding protocol](EMBEDDING_SWAP_PROTOCOL.md), [results](EMBEDDING_SWAP_RESULTS.md), [manifest](../analysis/embedding_swap/manifest.json) and [verification](../analysis/embedding_swap/verification.json). Query, key and value origins are crossed independently with every non-embedding tensor fixed at untrained donor 16. There are 120 new models and 40 reused embedding endpoints. The complete 160-outcome verification reproduces all final metrics and exact tensor origins/stream hashes.
- Runners: [group](../scripts/run_group_swap.py) and [embedding](../scripts/run_embedding_swap.py); separate preflight, summary and verification scripts are named alongside each runner and have adjacent uv lockfiles. Preflights use ten training-only updates at excluded stream 99 and require exact parity with both relevant baseline runners. No held-out tuning is performed. The manifests freeze every new protocol/helper/runner hash and all donor result/provenance hashes before new training.
- The two stages each add 36,864,000 training events. Measured training-loop CPU times are 353.85 and 359.72 seconds respectively, excluding evaluation and batch construction. Reused outcomes contribute no new training cost. New checkpoints are under `models/group_swap/` and `models/embedding_swap/`.
- Source stays `33afe26100f8590272940e55dbee5067a8040da6` with explicit CPU kernel substitutions. Runtime is deterministic CPU float32, one thread, torch 2.14.0/einops 0.8.1 via uv locks; Matplotlib 3.11.2 plots were visually checked at original resolution. These adaptively selected donor experiments do not establish general donor/task behavior or original-paper causality.

## Exploratory geometry and retrieval diagnostics

- [Initial geometry script](../scripts/inspect_initial_geometry.py) and [data](../analysis/initial_geometry.json) inspect the reconstructed untrained donors only. They record embedding/readout singular values, row norms, cosine matrices and all four query/key donor pairings. No training or held-out evaluation is involved. These descriptive statistics were inspected while the embedding factorial ran, without changing its protocol.
- [Value-class checkpoint inspection](../scripts/inspect_class_collapse.py) produces [diagnostics](../analysis/class_collapse/diagnostics.json); [query-key extension](../scripts/inspect_retrieval_modes.py) produces [retrieval modes](../analysis/class_collapse/retrieval_modes.json). Both inspect every one of the 160 embedding-swap checkpoints, verify checkpoint hashes and independently reproduce aggregate CE within 1e-6. The extension hashes its input value-diagnostic JSON.
- [Interpretation](RETRIEVAL_MODES.md) explicitly labels the unresolved-class threshold and fit tolerances post-hoc. The simple descriptions fit 29 fixed-value and 7 fixed-key cases out of 52 failed outcomes, with 16 unclassified; no population or training-cause claim is made. Per-episode probability and memory-vector comparisons complement conditional means. The class-inspection harness initially failed at final metadata assembly because of an incorrect module reference; the reference was corrected and the read-only inspection repeated. No training changed.
- [Diagnostic plot script](../scripts/summarize_retrieval_modes.py) uses the first enumerated value/key fits as examples. All diagnostic scripts have uv lockfiles. Results retain source/script hashes; there are no extra recurrent training runs or remote calls.

## Frozen-memory linear readout probes

- [Protocol](FROZEN_READOUT_PROTOCOL.md), [results](FROZEN_READOUT_RESULTS.md), [manifest](../analysis/frozen_readout/manifest.json) and [verification](../analysis/frozen_readout/verification.json). Three adaptively chosen cases (value-confusion example, key-confusion example, successful control) under both derivative rules give six additional head fits. They preserve all trained non-head tensors exactly.
- [Probe runner](../scripts/probe_frozen_readout.py) caches detached post-LayerNorm features from 32,000 training mappings (256,000 queries, order seed 8300), using the existing validation/test orders 8100/8200. Training-only feature standardization precedes zero-initialized full-batch float64 LBFGS, with fixed limits and L2 weight penalty. This probe has a distinct data/compute budget and cannot be presented as matched recurrent training.
- [Verifier](../scripts/verify_frozen_readout.py) reproduces all original and fitted train/validation/test metrics exactly, regenerates training feature hashes and normalization statistics, checks the final regularized objective, verifies unchanged non-head state, and reproduces graph-level float32 folded-head metrics. The six fits stop in 21–137 iterations; some satisfy objective-change stopping before the stricter gradient tolerance. No tuning or iteration extension occurs.
- Standardization is folded into ordinary linear-head weights/biases, saved both as float64 head tensors and separate float32 model checkpoints under `models/frozen_readout/`. Input/output checkpoints and protocol/helper hashes are frozen or verified. The fitting-only time sums to approximately 8.53 CPU seconds; feature extraction and verification are excluded. Runtime and source are otherwise unchanged.
- These probes do not rescue the selected ambiguities, but a penalized linear probe cannot prove that no faint or nonlinear information remains. Their interpretation is limited to accessible information in selected fixed representations. No longer recurrent training, model endpoint call or natural-language task is involved.
