# Gradient-refresh comparison protocol

Fixed before this comparison's validation or test results, 13 September 2026. The user authorized the proposed four-chunk, two-depth, two-derivative, ten-seed comparison. This is a follow-up informed by the earlier recall pilot, not an independently selected task or a reproduction of either paper.

## Question and contrasts

Does more frequent recomputation of inner gradients improve deep memory more than shallow memory at a fixed outer-training budget?

Run chunks 1, 2, 4 and 8; depths 1 and 2; released and complete linear-transport derivatives; ten fresh paired seeds 10–19 (excluding the previous pilot's 0–2 and smoke seed 99). Total: 160 models, 80 derivative pairs. All configurations at a given depth/seed start with identical parameters and receive identical minibatches. Changing chunk size changes the forward update algorithm, so cross-chunk forward equality is not expected. Derivative pairs at a fixed chunk must have equal initial outputs; shallow pairs must retain exact equality.

Primary outcome is final-checkpoint test cross-entropy after 600 updates. Primary contrast under the released rule is the paired depth interaction:

    (CE(depth 2, chunk 1) − CE(depth 2, chunk 8))
      − (CE(depth 1, chunk 1) − CE(depth 1, chunk 8)).

A negative value means that the smaller chunk favors deep memory more. Report its ten per-seed values, mean and a descriptive paired-seed percentile bootstrap interval (10,000 resamples, RNG seed 20260914). The complete-rule interaction is the prespecified corroborating comparison. Neither a favorable interaction alone nor a small mean difference establishes a general depth advantage; report absolute losses and accuracies as well. Other chunks, derivative differences and comparisons against chunk 4 are secondary, without selecting the most favorable chunk as the primary result. No formal multiple-comparison significance claims.

Secondary learning-speed outcomes: validation cross-entropy curves, normalized trapezoidal area from updates 1 through 600, and first scheduled validation measurement below 0.1 nats (unreached thresholds are right-censored at 600; no interpolation). Final accuracy and no-adaptation accuracy are also reported. No tuning, early stopping, checkpoint selection or outcome-driven extension.

## Fixed recipe and scaling

Reuse the prior pilot's exact [task, split and training recipe](RECALL_PILOT_PROTOCOL.md): eight writes followed by eight queries, disjoint full mappings, width 16, one head, Gaussian fast initialization std 0.02, Linear–SiLU factors, learned per-factor rates/decay, AdamW LR 0.003 and weight decay 0.01, clipping 1.0, batch 32, 600 updates. Train/validation/test mappings and evaluation order seeds remain unchanged. The held-out mappings were already used in the earlier pilot; this follow-up adds fresh training seeds, not a new untouched test set.

All chunk sizes divide the eight-event write phase and the complete sixteen-event episode. The inner-gradient reference is recomputed at chunk boundaries, giving 8, 4, 2 or 1 reference evaluations over the write phase. Every write event retains the same scalar rate rule (initially 1e-3) and decay rule (initially 0.99). Mean scaling remains false: neither rates nor accumulated writes are divided or multiplied by chunk length. Query writes and decay are disabled. Thus this manipulates reference refresh without an explicit chunk-dependent rate multiplier; actual gradients, learned rates and trajectories can still differ. No adaptation crosses episode boundaries.

Depths have 874/1,196 parameters and 256/512 fast matrix entries. Comparisons across chunks or derivatives at fixed depth are parameter-matched; depth comparisons are not. Each model sees 307,200 events, totaling 49,152,000 events. These are fixed-token, not compute-matched runs. Report training-loop time per configuration and total; timings exclude evaluation and include CPU scheduling/warm-up/order effects. Smaller chunks execute more graph passes and may cost more. No throughput or equal-compute claim.

## Execution and verification

Import the frozen pilot's model/data helpers and the same pinned source-loaded graph engine; change only the memory's chunk_size attribute. Preserve upstream revision `33afe26100f8590272940e55dbee5067a8040da6` and explicit CPU kernel substitutions. Use uv, torch 2.14.0, einops 0.8.1, CPU float32, one thread, deterministic algorithms. No remote model is needed.

Before measured training, check all four chunks and both depths against the independent float64 per-token reference, including state/outer-gradient agreement, directional finite differences and causal/reset checks. Use diagnostic seed 99 with 24-token synthetic inputs so all chunks divide sequence length. Confirm chunk-4 model equality with the frozen pilot. The diagnostic is not a held-out outcome or hyperparameter search.

Freeze protocol, runner, pilot/engine helper and preflight script hashes in a manifest before measured training. Save all checkpoints and per-pair histories. Verification checks manifest/source hashes, regenerated split isolation, identical starts/batches across chunks, derivative controls, checkpoint integrity, exact final test and no-adaptation metrics, and no-adaptation invariance to changed written values. Resume only completed records with matching frozen metadata; no silent overwrites of an earlier run.

CPU kernels, synthetic task, fixed initialization/optimizer and small state limit interpretation. This does not directly test the papers' 16-versus-256 setting or isolate gradient refresh from every consequence of changing chunked updates.
