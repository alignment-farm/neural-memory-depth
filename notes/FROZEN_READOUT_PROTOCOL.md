# Frozen-memory readout probe

13 September 2026. Fixed before fitting the probes. This authorized autonomous follow-up is diagnostic and adaptively selected after checkpoint inspection, not a new independent model-quality benchmark.

## Question and selected checkpoints

Can a newly fitted linear classifier recover value information from the existing normalized memory features, without changing memory, embeddings, rates, decay or LayerNorm?

Use three specific embedding-swap configurations/streams under both derivative rules (six probes):

- Value-confusion example: `q11-k11-v16`, stream 11, the first configuration/stream in enumeration that fits the post-hoc value-cluster diagnostic.
- Key-confusion example: `q11-k16-v11`, stream 15, the first configuration/stream fitting the key-cluster diagnostic.
- Successful control: `q11-k11-v11`, stream 11, the matching fully swapped embedding baseline for the first example.

Memory and readout donor 16 are the background for these previously trained checkpoints. Freeze every trained parameter except the 16→8 linear head; no memory training, extended recurrent training, parameter reset, or altered inner update is permitted. This tests recoverable information under an alternative readout fit, not why the original joint optimizer failed.

## Fixed probe fit

Cache the features immediately before the existing linear head (after the trained LayerNorm) and their value labels. Use all 32,000 existing training mappings with a single fixed write/query-order draw per mapping (generator seed 8300): 256,000 training queries. Validation/test use the existing fixed orders 8100/8200. These maps remain disjoint, but the held-out sets were used in earlier diagnostic selection. The probe sees more distinct training mappings than an individual original 600-step run; do not compare its compute or data budget as equal.

Compute feature mean and population standard deviation on training features only, with standard deviation clamped to at least 1e-6. Standardization is an invertible affine transformation and does not change linear-head expressivity. Fit a zero-initialized float64 eight-class linear head with full-batch LBFGS, learning rate 1, strong-Wolfe line search, max_iter 200, max_eval 250, history_size 20, tolerance_grad 1e-8 and tolerance_change 1e-10. Objective is mean training cross-entropy plus `1e-4 * 0.5 * sum(weight^2)`; biases are unpenalized. No hyperparameter search, validation selection, extra iterations after outcomes or nonlinear probe.

Evaluate final train/validation/test CE and accuracy. Report optimizer iteration/evaluation counts, final gradient infinity norm and objective; a weak or unconverged linear probe cannot establish absence of information. For comparison retain original checkpoint metrics. Also report the post-fit per-value and per-key CE distributions, using the same <0.1 descriptive criterion for resolved classes/keys. Positive held-out recovery establishes that some relevant information was accessible to this linear probe; it does not imply that the original head alone was the cause of training dynamics.

## Verification and provenance

Verify all input checkpoint hashes and the embedding-swap manifest/160-outcome verification record. Freeze this protocol, runner and input checkpoint/result hashes in a probe manifest before fitting. Reproduce original held-out metrics from cached features to tolerance. Fit only detached features; preserve all non-head state hashes. Fold standardization into the fitted weights and biases to produce an ordinary 16→8 head on the original feature scale. Verify equivalence of standardized and folded predictions, then evaluate the resulting frozen-memory model through the usual graph and save its state separately under `models/frozen_readout/`.

Use uv, pinned torch 2.14.0/einops 0.8.1, one CPU thread and deterministic execution. Feature extraction remains float32 in the existing CPU graph; probe fitting and feature-metric calculations are float64. No remote resource is needed. This is a bounded information-access diagnostic on selected examples, not a reproduction, a matched-budget training intervention, or proof of a full failure mechanism.
