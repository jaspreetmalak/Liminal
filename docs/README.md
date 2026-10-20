# Liminal documentation

Liminal is an original spiking neural network (SNN) for natural language generation,
built from scratch in Rust, including training. The binding project rules are in
[AGENTS.md](../AGENTS.md).

The initial recurrent character language model is specified in
[architecture.md](architecture.md), including neuron dynamics, exact output-loss
derivatives, causal spike/memory feedback, Burn surrogate autodiff, masked batches,
optimization, and state boundaries. Runnable commands
are in the [project README](../README.md). Measured results belong in
`experiments/`; equations alone are not evidence of learning or language quality.
FineWeb is the primary corpus; [data.md](data.md) documents bounded Rust
preparation, whole-document splits, provenance, and source terms.

## Design and validation

Add documentation as work is completed:

- Design: model purpose, equations, component responsibilities, Rust module
  ownership, state/reset semantics, and the reasons for architectural decisions.
- Reproduction protocol: model configuration, data sources and splits,
  preprocessing, initialization, seeds, precision, optimizer/schedule, training
  budget, evaluation metrics, and acceptance criteria chosen before validation.
- Validation evidence: exact commands and working directories, environment,
  component checks, state/reset checks, gradient and optimizer checks where
  applicable, training/evaluation measurements, failures, and remaining gaps.

Keep documentation organized by its purpose and distinguish design intentions
from behavior supported by measurements. Record artifact locations within
Liminal so results can be reproduced.

## Experiments

Log experiments in `docs/experiments/` as they are performed. Use descriptive dated
Markdown filenames, with one record per experiment containing:

- Question or hypothesis and intended change.
- Model configuration, seed, data/splits, hardware, and compute/token budget.
- Reproduction commands and artifact locations within Liminal.
- Metrics and comparison with a stated baseline when one exists.
- Observations, outcome, and the decision to keep, revise, or remove the change.

Experiment records may accumulate; source code remains one current implementation.
Do not retain alternate code versions alongside it.

The [frozen understanding diagnostic](experiments/2026-10-18-understanding-diagnostic.md)
specifies balanced binding, negation, order, and semantic-role continuations plus
ordered-context likelihood controls on genuine FineWeb. It compares fixed
trained checkpoints with fresh initialization through the same model, without
training updates. CE control, voltage penalty, and fresh initialization score
65/128, 62/128, and 63/128 correct cases, respectively; no family meets the
predefined fact-consistency criterion. Genuine text likelihood improves with
training, and ordered context affects some predictions, without establishing
consistent interpretation of changed facts. The chosen fixtures do not form a
general comprehension benchmark or establish emotional reasoning. See the
project README for the `understanding_probe` command and source-module
responsibilities.
