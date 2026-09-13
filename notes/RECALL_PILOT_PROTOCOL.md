# Pilot protocol: derivative rule and held-out associative recall

Frozen 13 September 2026 **before validation or test evaluation**. Authorized by the user's “Good find, proceed” after the proposal for a held-out sequence-modeling comparison. This extends the original written scope with a small local experiment, not a paper-scale training campaign.

## Question and primary comparison

Does supplying the omitted linear-transport derivative change held-out recall quality after training a fast-memory system? Compare released versus complete derivative within each of depths 1 and 2. A shallow pair is a negative control: this omission should not affect its training trajectory. Depth is a secondary comparison because deeper models contain additional parameters and fast state.

Primary outcome: final-checkpoint test cross-entropy per query, after exactly 600 outer updates. Also report accuracy, validation learning curves, paired differences by seed and their mean/range. No checkpoint selection, early stopping, tuning on validation, or test-driven extension is planned. Three paired training seeds: 0, 1, 2. There are 12 trained models (six derivative pairs). No formal significance claim from three seeds.

## Data and target isolation

Eight keys and eight values form an episode-specific bijection. Each episode has eight write events, one per key, followed by eight query events, one per key. Write and query orders are independently randomized. A query receives its key but **no target value** as input. It must recover the earlier assigned value. The objective is cross-entropy over query events only; labels never enter the inner update at query time. Query write rates are zero and decay is one.

Enumerate all 8! = 40,320 permutations, shuffle with Python `random.Random(20260913)`, then allocate 32,000 training mappings, 4,096 validation mappings, and 4,224 test mappings. Entire mappings are disjoint across splits; key and value symbols remain shared. Training samples mappings with replacement and redraws event orders. Validation and test use fixed event-order seeds 8100 and 8200. Split hashes are recorded. This is generalization to new mappings in the same synthetic distribution, not natural-language or out-of-distribution evaluation.

Chance accuracy is 12.5%; uniform-predictor cross-entropy is ln(8). After training, evaluate the test split with both associative writes and decay disabled. This no-adaptation control checks whether query predictions depend on mutable memory rather than a fixed key-to-class bias. It is an inference ablation, not a separately trained baseline.

## Fixed model and training recipe

- Source-loaded Modular TTT graph and CPU scalar-decay substitutions from the [graph audit](GRAPH_EXECUTION_AUDIT.md), revision `33afe26100f8590272940e55dbee5067a8040da6`.
- One head, width 16, one Linear–SiLU factor versus two Linear–SiLU factors. Chunk size 4, no mean scaling, scalar per-factor rates and decay, fast initialization Gaussian std 0.02.
- Separate learned key, value and query embeddings, each width 16 and Gaussian std 0.2. Keys are L2-normalized; queries and values are not. The value embedding at query events is masked to zero.
- Per-factor learned rate and decay predictors take concatenated key/value features. Predictor weights start at zero; initial rate is 1e-3 and decay 0.99. Queries force rate zero and decay one. These slow parameters are trained along with embeddings, initial fast weights, and readout.
- Readout is LayerNorm (epsilon 1e-5) and a linear eight-class head, applied to the memory read. No attention or direct value-to-output path.
- AdamW, outer LR 0.003, weight decay 0.01, gradient clipping at norm 1.0, batch 32 episodes, 600 steps, no scheduler. Each model processes 19,200 episodes / 307,200 events, with 153,600 supervised queries.
- CPU float32, one thread, deterministic algorithms. PyTorch 2.14.0, einops 0.8.1, locked `uv` script environment. Checkpoints and code/data hashes retained.

Each derivative pair starts with bit-identical state and receives the identical training batches. Changing derivative mode changes only differentiation through the linear error-transport weight; forward computations and all other mechanisms are the same. Initial forward loss equality, a nonzero first-step deep gradient difference and exact shallow final-state equality are checked.

A training-only ten-step smoke test used seed 99, excluded from the three measured seeds, to verify gradients and estimate runtime. It performed no validation or test evaluation and led to no hyperparameter search. A final implementation check made the no-adaptation ablation disable decay as well as gradient writes.

## Resource accounting and boundaries

This is a fixed-token pilot. Record parameter counts, runtime per run, and final checkpoint hashes. Neither depth nor derivative modes are asserted to be compute-matched; changing differentiation changes backward work, and CPU timings include order/warm-up effects. Fast matrix capacity doubles with depth at fixed width. Compare the derivative effect within depth before interpreting cross-depth rankings.

This small associative task deliberately isolates mutable memory use. It cannot establish the effect on Modular TTT's language-model loss, recover Titans' depth curves, or justify a general claim about deeper memory. Original paper checkpoint provenance, fused CUDA/BF16 behavior and the 16-versus-256 chunk comparison remain separate work. No pretrained model or Mac Studio installation is needed for this bounded CPU pilot.
