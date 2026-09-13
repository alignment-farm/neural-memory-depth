# Query / key / value initialization swaps

13 September 2026. Fixed before partial-embedding training or held-out evaluation. Authorized by the user's instruction to continue autonomously. This follow-up is selected because the full parameter-group experiment transferred robustness with donor-11 embeddings in the donor-16 memory/readout background.

## Fixed intervention

Hold fast memory matrices, readout, rate/decay predictors and LayerNorm at their original **untrained donor-16** values. Independently choose query (`q.*`), key (`k.*`) and value (`v.*`) embeddings from original untrained donors 11 or 16. Run the full 2×2×2 factorial over all ten previously used streams 10–19 and both derivative rules: 160 outcomes, including 40 reused all-11/all-16 embedding baselines and 120 newly trained partial swaps.

The all-11 embedding baseline is the previously evaluated `m16-e11-r16` model; it is not the pure whole-model donor 11. The all-16 baseline is `m16-e16-r16`. All non-embedding tensors must stay exactly at donor 16 in every cell. Query and key embeddings remain separate learned tensors, with only keys L2-normalized as before. Value embeddings are zero-masked at queries. No tying, rescaling, rotations, warm-started trained weights or altered update rule.

Keep the same depth-2/chunk-4 model, data split, evaluation seeds, 600 updates, batch 32, AdamW LR 0.003/weight decay 0.01, clipping 1.0 and all other recipe settings. The model has 1,196 parameters throughout. New training events: 36,864,000. No tuning, stopping, checkpoint selection or extension based on outcomes. This is the same reused synthetic test distribution and selected donor pair, not an independent replication or a universal initialization test.

## Prespecified analysis

Primary endpoint is final test CE, released derivative primary and complete corroborating. For each of query/key/value, report the per-stream marginal donor-11-minus-donor-16 contrast averaged over all four backgrounds of the other two embeddings. Report mean/range and each stream's difference. Negative favors donor 11.

Also report direct replacement of each single embedding in the all-16 embedding background and the reciprocal replacement in the all-11 embedding background. Preserve the complete configuration table. Final validation CE < 0.1, accuracy, validation area, no-adaptation accuracy and derivative comparisons are secondary. Interpret single-group sufficiency and background dependence only for the tested donors, streams and budget. If multiple groups interact, retain that result rather than forcing one-group attribution.

## Verification and provenance

Verify the previous group's complete 160-outcome record and frozen provenance before reuse. A training-only ten-step preflight at excluded stream 99 must match each endpoint's prior runner exactly; no held-out evaluation in preflight. Check tensor origins for every hybrid, including that every non-embedding entry comes from donor 16. Freeze this protocol, runner, preflight and imported helper hashes, reconstructed initial hashes, donor records and prior provenance before new training.

Store results under `analysis/embedding_swap/` and new checkpoints under `models/embedding_swap/`, marking reused baselines separately and excluding their costs. Reload every final checkpoint to reproduce final validation, test and no-adaptation metrics. Regenerate streams and initial states, preserve baseline records, and check source/protocol integrity, query-target isolation and no-adaptation value invariance.

Use uv locks, torch 2.14.0/einops 0.8.1, deterministic CPU float32 and one thread. Keep upstream `33afe26100f8590272940e55dbee5067a8040da6` and explicit CPU kernel substitutions unchanged. This experiment localizes an intervention within embeddings; it does not yet identify a geometric mechanism or explain the papers' depth trends. No remote model is required.
