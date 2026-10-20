# Initial spiking language baseline — 2026-10-18

Historical run before FineWeb became the primary corpus. Its toy input file is
no longer present and its checkpoint has been replaced by the current FineWeb
model. The commands and hashes below describe that original run; current defaults
and current model results are recorded in [the FineWeb experiment](2026-10-18-fineweb-data.md).

## Question and acceptance criteria

Can the original Rust SNN train end to end on causal next-character prediction,
preserve its predictions through a checkpoint, and generate a continuation?

Before running the text experiment, the engineering acceptance criteria are:

- Neuron/state, causality, surrogate-gradient, optimizer, persistence, and tokenizer
  tests pass; formatting and Clippy pass with warnings denied.
- The deterministic alternating-character test reaches less than 70% of its
  initial training loss and generates its exact expected greedy continuation.
- In the bundled text run, all metrics are finite, training loss falls at least
  10%, and held-out loss decreases from the untrained model.
- Compare held-out loss to a train-target-frequency unigram control with
  additive-one smoothing. Report the outcome without changing the budget or
  acceptance criteria based on results.
- Save and reload the model, then exercise generation and evaluation.

These are engineering checks. The original tiny corpus is repetitive and is not
evidence of natural language generalization, useful long context, or efficiency.

## Fixed configuration

- Corpus: `data/tiny.txt`, original project text; no external model/data copied.
- Split: first 90% of Unicode scalars for training, final 10% for validation,
  rounding validation size up; no pair crosses the boundary. Vocabulary is fitted
  to training only, with unknown token 0.
- Model: one recurrent LIF layer, 48 hidden neurons, leak 0.9, threshold 1,
  triangular surrogate width 1; full attached subtractive-reset gradient.
- Precision/device: `f64`, dense single-process CPU Rust, no third-party crates.
- Initialization, sampled windows, and generation seed: 42; SplitMix64.
- Training: 500 updates, one random zero-state window of 32 targets per update
  (16,000 sampled target positions, including repetitions).
- Optimizer: Adam, learning rate 0.003, betas 0.9/0.999, epsilon 1e-8, no weight
  decay, global gradient norm clip 1.
- Evaluation: each adjacent pair scored once; reset every 32 predictions and
  target-weight the final shorter window. Cross-entropy uses natural logs;
  perplexity is `exp(loss)` per character.
- Output checkpoint: `artifacts/liminal.bin`; model/vocabulary only, no optimizer
  resume state. Generation uses prompt `The `, temperature 0.8, 160 new tokens.

## Reproduction

Run from the Liminal project root:

```text
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo build --release
target/release/liminal.exe train --data data/tiny.txt --checkpoint artifacts/liminal.bin --steps 500 --hidden 48 --sequence-length 32 --learning-rate 0.003 --gradient-clip 1 --seed 42
target/release/liminal.exe generate --checkpoint artifacts/liminal.bin --prompt "The " --tokens 160 --temperature 0.8 --seed 42
target/release/liminal.exe evaluate --checkpoint artifacts/liminal.bin --data data/tiny.txt --sequence-length 32
```

On other platforms use `target/release/liminal` instead of the `.exe` path.
The final evaluation command checks the loaded checkpoint on the entire bundled
file, including training text; label that result as mixed/in-sample. The separate
held-out metric comes from `train`'s validation split.

## Execution evidence

Executed on 2026-10-18 with `rustc 1.98.1 (48a229cea 2026-09-01)` and
`cargo 1.98.1 (797e8a9bc 2026-08-05)`, Windows build `10.0.26300`, AMD Ryzen 7
7800X3D (8 cores/16 logical processors). The implementation is single-threaded;
the machine's core count is not a scaling result.

- Data SHA-256: `61ED71FA64A31D963F4F557E35604CD4B1958D925B477CC0639C81D906BE1203`.
- Corpus: 1,766 Unicode scalars; 1,589 train and 177 validation; 36 vocabulary
  entries including unknown. Validation had zero unknown characters.
- Model: 5,844 trainable parameters, matching `2VH + H² + H + V`.
- Initial/final losses were evaluated with the same 32-prediction reset policy.

| Evaluation set/control | Cross-entropy (nats/character) | Character perplexity |
| --- | ---: | ---: |
| Training, untrained model | 3.603284 | 36.718632 |
| Training, after 500 updates | 2.104471 | 8.202760 |
| Validation, untrained model | 3.611613 | 37.025720 |
| Validation, train-frequency unigram | 3.000642 | 20.098438 |
| Validation, after 500 updates | 2.500466 | 12.188169 |
| Entire file, loaded checkpoint (mixed/in-sample) | 2.137272 | 8.476280 |

Training loss fell 41.6%; validation loss fell 30.8%. Final validation loss was
0.500176 nats/character below the smoothed unigram control. This is evidence of
learning beyond that frequency control on this particular tiny split, not broad
natural language generalization. Training plus reporting and checkpoint I/O took
approximately 0.269 seconds in one optimized run; this was not a benchmark.

The saved checkpoint SHA-256 was
`6B65C1A9107CCEB805EB4DF68E53CCE67FBB1E05C5D35447B56769CF9C4B1B67`.
Checkpoints and captured command output are in `artifacts/liminal.bin`,
`artifacts/baseline-train.log`, `artifacts/baseline-generation.txt`, and
`artifacts/baseline-evaluate.log`. These are generated files, not additional
source versions.

Generation from the reloaded checkpoint, fixed prompt/seed/temperature above:

```text
The sail the sane sean stene hers trene aln sags ea ruld iastes dtra suetsen the the thee.enenen eran. The.d.
Mheodl..
The wore baWh rint witien wton sirnd. othe ir
```

The sample contains local word-like fragments and substantial errors. The model
does not yet produce reliable fluent text.

## Verification and decision

- Rust tests cover hand-computed threshold/reset/leak/recurrence, state boundaries
  and causality, extreme-logit loss, complete surrogate BPTT, optimizer moments,
  global clipping, deterministic learning, tokenizer/CLI, and persistence.
- All-parameter gradient checks use frozen local surrogate slopes and centered
  differences with step `1e-6`, absolute tolerance `5e-10`, active and inactive
  slope cases, and a nonzero fixed initial state. True readout-gradient finite
  differences use absolute tolerance `3e-10`.
- The alternating-character learning test trains twice with seed 7, checks exact
  parameter reproducibility and at least 30% loss reduction, and requires the
  exact 12-character greedy continuation. It is a synthetic algorithm check,
  separate from the natural-text experiment.
- Formatting passed, all 23 Rust tests passed (20 library and 3 integration),
  and Clippy passed with warnings denied.
- Training, held-out improvement, control comparison, model save/load, generation,
  and evaluation satisfy the fixed engineering acceptance criteria.

Keep this single implementation as the initial reference. Next work should use
a larger, separately sourced and deduplicated corpus with genuinely disjoint
validation text, measure neuron firing and temporal gradients, and test memory
over longer contexts before increasing model complexity. There is no claim of
language fluency, neuromorphic efficiency, or a solved language model.
