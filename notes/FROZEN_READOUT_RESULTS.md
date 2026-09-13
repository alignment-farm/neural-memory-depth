# Refitting the linear readout does not resolve the selected partial-retrieval failures

13 September 2026. **Six frozen-feature readout probes; no recurrent-memory retraining.** [Protocol](FROZEN_READOUT_PROTOCOL.md), [manifest](../analysis/frozen_readout/manifest.json), [verification](../analysis/frozen_readout/verification.json).

A newly fitted linear head does not resolve either selected failure: the value-confusion example remains near the seven-value uncertainty limit, and the key-confusion example retains its two unresolved query keys. The successful control remains at 100% accuracy. This narrows a readout-only account of the final checkpoints, while leaving open faint, nonlinear or otherwise inaccessible information in the features.

## Results

The head is fitted to detached features after the trained LayerNorm. Memory, embeddings, rates, decay and LayerNorm remain bit-identical to the original checkpoint. The probe uses all 32,000 existing training mappings with fixed event orders, a different data/compute budget from the original joint training; these are information-access diagnostics, not matched-budget improvements.

| Example | Derivative | Original test CE | Refit test CE | Refit accuracy | LBFGS iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| Value Confusion | released | 1.703114 | 1.702787 | 24.896% | 131 |
| Value Confusion | complete | 1.703026 | 1.702798 | 24.571% | 137 |
| Key Confusion | released | 0.178157 | 0.175474 | 87.379% | 73 |
| Key Confusion | complete | 0.178170 | 0.175466 | 87.382% | 94 |
| Successful Control | released | 0.000594 | 0.000448 | 100.000% | 21 |
| Successful Control | complete | 0.000841 | 0.000455 | 100.000% | 26 |

The value-confusion checkpoint is `q11-k11-v16`, stream 11. After refitting, value 4 has very low CE and the other seven classes remain near ln(7). The mean is approximately 1.70279, close to 7/8·ln(7)=1.70267. Accuracy near 25% is consistent with recovering one class and guessing among seven others.

The key-confusion checkpoint is `q11-k16-v11`, stream 15. Query keys 0 and 2 retain CE close to ln(2); other keys are below approximately 0.002. The resulting mean near 0.17547 and accuracy near 87.4% remain close to the two-key ambiguity prediction of 0.17329 CE and 87.5% accuracy. The small calibration improvement does not resolve the missing key distinction.

The successful control is `q11-k11-v11`, stream 11. It stays at 100% accuracy and slightly improves CE, confirming that the fitting/evaluation path can preserve accessible distinctions.

## Fit and verification

Training features are standardized using training-only mean and population standard deviation, clamped at 1e-6. A zero-initialized float64 16→8 linear head is fitted with full-batch LBFGS and an L2 weight penalty of 1e-4; biases are unpenalized. The fixed limit is 200 iterations / 250 evaluations, with strong-Wolfe search and the tolerances recorded in the protocol. There is no tuning, nonlinear probe or extension after outcomes.

The six fits stop after 21–137 iterations. Final regularized-objective gradient infinity norms range from roughly 4.2×10⁻⁸ to 2.3×10⁻⁶; some stop on the objective-change criterion before meeting the stricter gradient tolerance. These are approximately stationary penalized linear fits, not a proof of the unpenalized global optimum or absence of information. Total measured fitting time is about 8.53 CPU seconds, excluding feature extraction, hashing and verification.

Standardization is folded into ordinary weights and biases on the original feature scale. The standardized and folded float64 logits agree within the recorded numerical tolerance. Casting the folded head to float32 and evaluating through the full graph reproduces the same practical outcomes. New model/head files are separate from the input checkpoints.

The verifier reloads all inputs and outputs, confirms exact non-head state equality, regenerates the training feature hashes and normalization statistics, reproduces original and probe train/validation/test metrics, checks the final penalized objective, and reproduces the float32 folded-model test metrics. All six probes pass. Individual JSON records retain per-value/per-key losses, optimizer diagnostics, timing and checkpoint hashes.

## Implication and limits

The earlier readout-initialization swap sometimes rescues joint training, but refitting the readout after a failed feature-learning trajectory does not rescue these examples. Those observations are compatible: the initial head can affect how representations learn, while a later linear refit cannot recover distinctions that the frozen features do not make accessible to this probe.

This supports describing the failures as **partial-retrieval solutions reached by the joint learning process at the fixed 600-update horizon**, rather than simply a stale final classifier. It does not establish irreversible information loss, a stable attractor, or impossibility of recovery through longer training, a nonlinear probe or another regularization choice. The original architecture supports full recall under other initial conditions, so the observed failure is not an unavoidable capacity limit on this synthetic task.

The three examples were selected after the checkpoint diagnostic and use the same already inspected test distribution. Both derivative modes are retained, but this is not an independent donor/task replication or evidence about original-paper checkpoints. The probe changes only a trained readout while holding memory fixed; no Mac Studio model was installed or used.

[Runner](../scripts/probe_frozen_readout.py) and [verifier](../scripts/verify_frozen_readout.py) have adjacent uv lockfiles. Records are under `analysis/frozen_readout/`, and folded models plus float64 heads under `models/frozen_readout/`. Runtime remains pinned to torch 2.14.0/einops 0.8.1 and deterministic CPU execution.
