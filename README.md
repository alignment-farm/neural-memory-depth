# Neural memory depth

**Status: local synthesis and bounded mechanism investigation published, 13 September 2026.**
The study independently checked versioned source content, audited a
source-loaded graph derivative, and completed controlled recall comparisons.
The latest tensor swaps show strong interactions between initial embeddings;
checkpoint diagnostics distinguish value-confusion and key-confusion patterns.
Refitting a linear readout does not resolve selected examples with memory
frozen. These are synthetic-task findings, not a reproduction or causal
explanation of the published language-model depth curves. No model endpoint
calls or language-model pretraining runs have been performed.

This study asks which operating conditions explain the differing neural-memory
depth results described in Titans and Modular TTT. It arises from Construct-2's
[S3 question](../../construct-2/studies/README.md#s3-when-does-deeper-neural-memory-repay-its-harder-update-problem).
The first intended outcome is a written comparison of their mechanisms and
experimental configurations.

## Question

**When does additional neural-memory depth help, and which differences in the
write dynamics or surrounding system explain the reported depth trends?**

The parent's paper review reports a favorable depth trend in Titans and an
unfavorable trend in Modular TTT's tested setting. This motivates an apparent
tension, but does not establish that the experiments are directly comparable.
Additional depth might provide useful capacity only when the update dynamics
can exploit it. Parameter count, training compute, task distribution, or the
surrounding architecture might instead explain the difference. These are
starting explanations to examine, not findings of this study.

## Initial expectation

Begin with the papers and their available implementations. Develop a common
description of the mutable memory, inner loss and update rule, update schedule,
initialization, normalization, persistence/reset boundary, and outer training
objective. Relate the reported depth comparisons to their tasks, context and
chunk lengths, parameter counts, surrounding architectures, and training
compute where reported. Identify missing configuration facts explicitly.

The intended first publication is a concise local research note explaining
which comparisons are meaningful and whether the apparent disagreement
survives alignment. An explanation that resolves it is a complete useful
outcome. If uncertainty remains, propose the smallest comparison that could
distinguish its causes, with separate attention to parameter and compute
matching. A general advantage over explicit context is not required to answer
this narrower mechanism question.

The ancillary investigator owns the methods, level of detail, and any revision
of the question warranted by the reading. The initial scope is written
synthesis; training runs and a larger empirical campaign are possible later
work, not commissioned by this preparation. No numerical time, compute, or
spending budget has been specified.

The user subsequently authorized gradient audits, bounded experiments and
autonomous continuation. The completed empirical work includes 592 distinct
recurrent-model training runs and six additional frozen-readout fits. Reused
baselines are explicitly marked and excluded from new training counts.
Protocols, results and verification records are linked below.

## Starting sources

These versioned reading leads come from the parent's existing review. The
first pass checked their content; exact reading scope and provenance are in
the [source ledger](notes/SOURCES.md).

- **Titans — P2:** [2501.00663v1](https://arxiv.org/html/2501.00663v1).
  Start with the memory mechanism in §§3–4 and the depth comparison in §5.5,
  including the configurations behind that comparison.
- **Modular TTT — P12:** [2608.07110v1](https://arxiv.org/html/2608.07110v1).
  Start with the depth analysis, Appendices 8.4 and 9.4–9.5, and limitations.
  Account for the initialization and stabilization comparisons already made.
- **TTT layers — P1:** [2407.04620v4](https://arxiv.org/html/2407.04620v4).
  §§2.1–2.3 and 2.6–2.7 supply a starting formulation for learned recurrent
  memory and its inner/outer learning processes.

The parent's [paper map](../../construct-2/studies/README.md#3-paper-map-what-we-can-build-on)
records its reading scope. Its [worked examples](../../construct-2/studies/README.md#b0-a-worked-explanation-of-neural-writes-reads-and-resets)
are optional mechanism background. Check the exact papers, code revisions,
and configurations used in this study; distinguish source-reported results
from new analysis or reproduction. If later versions matter, identify what
changed rather than silently replacing the starting references.

## Resources

[AGENTS.md](AGENTS.md#model-resources) lists the available model routes,
including the preferred Mac Studio serving resource. Written alignment can
begin from source material. Any later neural intervention needs access to the
relevant parameters or mutable state in the chosen implementation; serving
access alone does not establish that capability. Use `uv` for Python and
Docker/Compose where supporting resources are needed.

The user offered on 13 September 2026 to install models on the Mac Studio:
provide the exact model ID and configuration when needed. No installation
is currently requested; a neural intervention would first require verified
gradient and mutable-state access.

## Findings and publication

Read [current findings](FINDINGS.md) for the consolidated comparison and
mechanism synthesis. The empirical evidence is organized as follows:

| Stage | Results | Frozen protocol | Verification |
| --- | --- | --- | --- |
| Derivative pilot | [12 models](notes/RECALL_PILOT_RESULTS.md) | [Protocol](notes/RECALL_PILOT_PROTOCOL.md) | [Checkpoints](analysis/recall_pilot/verification.json) |
| Gradient refresh | [160 models](notes/REFRESH_COMPARISON_RESULTS.md) | [Protocol](notes/REFRESH_COMPARISON_PROTOCOL.md) | [Checkpoints](analysis/refresh_comparison/verification.json) |
| Initialization × stream | [200 outcomes; 180 new](notes/SEED_SEPARATION_RESULTS.md) | [Protocol](notes/SEED_SEPARATION_PROTOCOL.md) | [Factor isolation/checkpoints](analysis/seed_separation/verification.json) |
| Memory / embeddings / readout swaps | [160 outcomes; 120 new](notes/GROUP_SWAP_RESULTS.md) | [Protocol](notes/GROUP_SWAP_PROTOCOL.md) | [Tensor origins/checkpoints](analysis/group_swap/verification.json) |
| Query / key / value swaps | [160 outcomes; 120 new](notes/EMBEDDING_SWAP_RESULTS.md) | [Protocol](notes/EMBEDDING_SWAP_PROTOCOL.md) | [Tensor origins/checkpoints](analysis/embedding_swap/verification.json) |
| Frozen-memory readout | [Six fits](notes/FROZEN_READOUT_RESULTS.md) | [Protocol](notes/FROZEN_READOUT_PROTOCOL.md) | [Frozen state/metrics](analysis/frozen_readout/verification.json) |

The [retrieval-mode inspection](notes/RETRIEVAL_MODES.md) is an explicitly
post-hoc diagnostic on saved checkpoints. It does not add recurrent training
or replace the prespecified experiment outcomes.

The [graph execution audit](notes/GRAPH_EXECUTION_AUDIT.md) establishes the
depth-dependent derivative omission through the released graph with CPU kernel
substitutions, extending the [primitive audit](notes/OUTER_GRADIENT_AUDIT.md).
Its effect on the original paper's results remains unknown.

Supporting evidence includes the [initial mechanism comparison](notes/INITIAL_COMPARISON.md),
[configuration follow-up](notes/CONFIGURATION_FOLLOWUP.md),
[source ledger](notes/SOURCES.md), [static config audit](analysis/config_audit.json),
[primitive results](analysis/outer_gradient_check.json), and
[graph results](analysis/graph_gradient_check.json).
The root can assess this publication under the
[ancillary-study approach](../../construct-2/notes/ANCILLARY_STUDY.md).

## Versioned publication

The Git checkpoint preserves the study documentation, scripts and uv
lockfiles, supporting JSON records, figures and versioned reference material.
Model files, caches, environments and the external implementation checkout
remain excluded. See [source restoration and artifact scope](sources/README.md)
for the pinned upstream revision and the limits of checkpoint-only verification.
The bounded contribution is complete; resolving the original cross-paper
question is not a prerequisite for closure.
