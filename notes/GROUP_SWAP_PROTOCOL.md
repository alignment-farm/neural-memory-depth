# Initial parameter-group swaps: fixed protocol

13 September 2026, fixed before hybrid-model training or held-out evaluation. The user authorized continuing autonomously after initialization/stream separation.

## Intervention

Use initialization 11 (the lowest-numbered start successful on all ten streams under both derivatives) and 16 (the only start unsuccessful on all ten). These are deliberately selected contrasting donors, not randomly sampled representatives. Reconstruct their original **untrained** parameters; no trained weights are transplanted.

Partition randomly initialized parameters into three groups: fast memory matrices (`memory.*`), key/value/query embeddings (`q.*`, `k.*`, `v.*`), and readout (`head.*`). Assert all remaining state (rate/decay predictors and LayerNorm) is identical between donors. Cross the two donor origins independently for all three groups: a complete 2×2×2 factorial, eight configurations. This includes reciprocal single-group swaps and their complements. Use all ten training streams 10–19 and both derivative rules, for 160 outcomes. Reuse the 40 pure-donor outcomes from the seed crossing and train 120 hybrid models.

Depth 2, chunk 4, width 16, task/data/evaluation order, optimizer and 600-update budget remain exactly as in the [seed-separation protocol](SEED_SEPARATION_PROTOCOL.md). All initial state entries in a hybrid must match the indicated donor group exactly; every unswapped entry must remain identical. Parameter count is fixed at 1,196. Each hybrid sees 307,200 events; new total is 36,864,000 events. Streams remain independent of parameter selection. No hyperparameter tuning, checkpoint selection, early stopping or extension after results.

## Outcomes and contrasts

Primary endpoint: final test cross-entropy, released derivative primary and complete corroborating. Report all configurations across all streams, with final validation CE < 0.1 as the prespecified secondary low-loss criterion. Display each pure baseline and each hybrid explicitly.

For each group, report the paired per-stream marginal contrast of donor-11 versus donor-16 initialization, averaged equally over the four combinations of the other two groups. Negative CE change favors donor 11. Also report both direct reciprocal contrasts: replacing just that group in the all-16 background, and replacing it with donor 16 in the all-11 background. These reveal whether a group transfers robustness or depends on its background. Report mean, range and per-stream differences without a population significance claim. Retain all background combinations rather than selecting a favorable hybrid as the sole result.

Report low-loss counts, final accuracy, validation curves/area, derivative-pair changes and no-adaptation accuracy. The experiment identifies causal effects of swapping these donor tensors in this fixed algorithm/recipe; it does not identify a universal initialization rule or the internal mechanism. A follow-up within a implicated group must be specified separately before further training.

## Provenance and verification

Import frozen helpers and graph source without edits. Source revision `33afe26100f8590272940e55dbee5067a8040da6`, explicit CPU kernel substitutions, torch 2.14.0/einops 0.8.1, one CPU thread, deterministic float32, uv locks. Reuse prior checkpoints only after verifying seed-separation manifests, source hashes and its 200-outcome verification record.

Before measured runs, verify the partition is exhaustive over all differing donor parameters and that pure-donor configurations have exact original initial hashes. A training-only ten-step preflight compares each pure donor on stream 99 with the frozen seed runner; require exact initial/final hashes, batch hash, losses and gradient diagnostics. No held-out evaluation in preflight.

Freeze protocol, runner, preflight/helper hashes, group membership, reconstructed initial state hashes and donor result/provenance hashes in a manifest before hybrid training. Mark pure-donor outcomes as reused and exclude their training events/timing from new resource totals. Save new checkpoints under `models/group_swap/` and results under `analysis/group_swap/`.

Verify all 160 final test, final validation and no-adaptation metrics by reloading checkpoints. Regenerate stream hashes, reconstruct exact hybrid initial state, verify group membership/tensor origin, source/protocol/donor integrity and unchanged pure baselines. Query-target isolation and no-adaptation value invariance remain required. No remote model is needed.
