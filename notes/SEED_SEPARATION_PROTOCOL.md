# Initialization × training-stream separation protocol

Fixed before off-diagonal training or held-out evaluation, 13 September 2026. The user authorized separating initialization from training-data-order effects after the refresh comparison. This is a diagnostic follow-up on an already used task and held-out split.

## Factors and scope

Use the complete Cartesian product of initialization seeds 10–19 and training-stream seeds 10–19, at depth 2 and chunk 4, retaining released and complete derivatives. This is 100 seed combinations / 200 model outcomes. Reuse the 20 previously verified diagonal outcomes (initialization seed equals stream seed) from the refresh comparison; train the remaining 180 models. All ten seeds are included, not a subset chosen for especially good or bad results. The diagonal was already observed and informed this study; it is not fresh confirmatory evidence.

Initialization seed controls all randomly initialized model parameters, including embeddings, fast initial matrices and readout. Deterministic rate/decay initialization remains unchanged. The training-stream seed controls the independent torch generator at `10000 + stream_seed`: sampled training mappings, their order across minibatches, and within-episode write/query event order. Thus this separates initialization from the complete sampled training stream, not data ordering alone while holding the exact multiset of examples fixed. It also does not isolate fast matrices from other initial parameters.

At a fixed initialization, changing the stream must preserve the initial state hash. At a fixed stream, changing initialization must preserve the full 600-step batch hash. At each cell both derivative modes start identically and receive identical batches. The model's forward algorithm and every other hyperparameter remain fixed.

## Recipe and endpoints

Reuse the frozen pilot model and data helpers, with the source-loaded graph and CPU kernel substitutions at revision `33afe26100f8590272940e55dbee5067a8040da6`. Fixed width 16, one head, two Linear–SiLU factors, chunk 4, mean scaling false, fast init std 0.02, learned per-factor rates/decay, initial rate 1e-3 and decay 0.99. Preserve AdamW LR 0.003, weight decay 0.01, clipping 1.0, batch 32, 600 outer updates, query-only outer cross-entropy and no adaptation at query events.

Data splits and evaluation order seeds remain exactly as in the [pilot](RECALL_PILOT_PROTOCOL.md). No new held-out task or untouched test set is claimed. Validation at step 1 and every 100; final checkpoint only, with no tuning, stopping or extension based on outcomes. Each newly trained model processes 307,200 events; 180 new models total 55,296,000 events. All compared models have 1,196 parameters, including 512 fast matrix entries. Fixed training events do not imply equal compute across derivative modes.

Primary outcome: final test CE per query, analyzed separately for the released derivative (primary) and complete derivative (corroborating). Report the full 10×10 matrix, mean CE by initialization (row) and by stream (column), and a descriptive balanced two-factor decomposition:

- `SS_init = 10 * sum((row_mean - grand_mean)^2)`.
- `SS_stream = 10 * sum((column_mean - grand_mean)^2)`.
- `SS_interaction = sum((cell - row_mean - column_mean + grand_mean)^2)`.
- `SS_total = sum((cell - grand_mean)^2)`; report each term's fraction and verify their sum.

This is a finite-grid descriptive decomposition, not an F-test or a population causal variance estimate. There is one deterministic run per cell per derivative. The interaction term contains nonadditive initialization–stream dependence; no independent residual-noise estimate is available. Compare init and stream shares, but do not force a single-cause conclusion when interaction is substantial.

Secondary outcomes: final accuracy, final validation CE < 0.1 (low-loss criterion), row/column low-loss counts, first scheduled validation CE < 0.1 with unreached outcomes censored, and normalized validation CE area over steps 1–600. Count how often changing streams rescues or disrupts a given initialization relative to its previously observed diagonal outcome. Also report released-versus-complete paired CE differences and low-loss transitions. These are descriptive secondary analyses, not selected primary endpoints. Record no-adaptation test accuracy and training-loop times.

## Provenance and verification

Import the existing model and helper files without editing them. The new runner separates `torch.manual_seed(init_seed)` from `torch.Generator().manual_seed(10000 + stream_seed)`. Before new training, verify the prior refresh manifest, source-definition hashes, checkpoint hashes and complete verification record. A training-only 10-step preflight compares the new runner against the frozen refresh runner at seed 99 on the diagonal (no validation or test); initial/final hashes, training losses and first gradient discrepancy must match exactly.

Freeze this protocol, new runner, preflight script and existing helper hashes in a manifest before any off-diagonal run. Hash all 10 donor result files and the prior manifest and verification record. Mark reused diagonal records explicitly and preserve their original checkpoint paths; do not count their timings or training events as new compute. Save new checkpoints separately under `models/seed_separation/`.

Verification must regenerate each stream's full batch hash independently, check row-constant initialization hashes and column-constant batch hashes, check equal initial loss within derivative pairs, verify source and manifest integrity, and reload every grid checkpoint to reproduce exact final test/no-adaptation metrics. Reuse no-adaptation value-invariance checks and query-target isolation. Confirm donor records retain their exact original results. Run with uv, torch 2.14.0, einops 0.8.1, CPU float32, one thread, deterministic algorithms. No remote model is required.

This diagnostic identifies whether failures follow whole-model initialization, the sampled training stream or their interaction under one fixed small-task recipe. It cannot by itself identify a conditioning mechanism, establish fast-weight initialization as the cause, or explain the published language-model depth trends. Any parameter-group intervention or longer training horizon would require a separately specified follow-up.
