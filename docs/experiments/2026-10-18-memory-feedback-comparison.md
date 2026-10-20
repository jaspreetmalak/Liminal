# Matched feedback comparison on one-bit retention

## Protocol before the comparison run

The component curriculum tests whether each path can learn simple retention.
This separate fixed-budget comparison asks whether either feedback direction
helps learning the same delay-16 task directly from a fresh initialization.
It uses a cold start, not transferred curriculum weights, so initialization and
update schedules can be matched exactly without vocabulary or optimizer changes.

Use the existing memory diagnostic's data seed 42, delay-16 stage: eight training
noise histories paired with payload bits (16 examples), 32 distinct held-out
histories (64 examples). The seven-token vocabulary is fitted on the union of
training-stage prompts only. Input is two irrelevant prefix characters, one bit,
16 distractors, and `?`; predict the remembered bit at the final position.
All payload-ignorant binary controls score 50%, and reading the payload is a
100% deterministic oracle. Do not change data or hyperparameters after results.

For each seed 42, 43, 44, compare the four existing FeedbackMode settings using
ModelPath::Full. Match initial parameter bytes, parameter count, full-batch case
order, Adam settings, and exactly 300 updates. Use H32, batch 16, f32 WGPU on
RX 9060 XT, lr .003, global norm clip 1, Adam .9/.999/epsilon1e-8, no weight decay.
Use fresh optimizer state per condition, no early stopping, no curriculum, no
held-out parameter updates, and zero all recurrent states at each independent
prompt. Every mode retains both output projections and continuous memory.

Report initial/final training and held-out full-vocabulary accuracy/NLL, actual
updates, targets, input tokens, data hashes, initial parameter hashes, and logs.
Each condition uses 4,800 answer-target presentations and 96,000 input tokens.

The numerical feedback-benefit criterion is the same as earlier comparisons:
Both mean held-out accuracy must exceed the best other retrained mode by at least
5 percentage points, win on at least two of three paired seeds, and have lower
mean held-out answer NLL. Otherwise report no demonstrated benefit in this
setting. Separately report whether held-out accuracy reaches 95%; a small gain
near chance would not meet that practical retention milestone.

This comparison matches initialization, parameters, update counts, and input/target
budgets on a single simple task. It does not measure equal hardware work, runtime,
or energy; zero-gated paths may execute differently. The
curriculum's earlier training-based stopping makes its update budgets different;
its results must not be substituted for these controls. Neither experiment
establishes general semantic, emotional, or language capability.

```powershell
cargo run --release --bin memory_probe -- --compare-feedback --device gpu --steps 300 --seeds 42,43,44 --output artifacts/memory-feedback-comparison.json
```

The comparison writes its report and log without a language-checkpoint write.
The completed run used the protocol above without changing its acceptance rule.
The actual invocation used the executable after a shared release build:

```powershell
cargo build --release --bin memory_probe --bin language_probe
./target/release/memory_probe.exe --compare-feedback --device gpu --steps 300 --seeds 42,43,44 --output artifacts/memory-feedback-comparison.json 2>&1 | Tee-Object -FilePath artifacts/memory-feedback-comparison.log
```

Build and experiment commands completed with exit status zero. The shared
release build took 3m 53s; that is compilation time, not measured training time.

## Measured result

All **12 conditions** finished exactly 300 updates and reached **100% training
and 100% held-out full-vocabulary answer accuracy**. Every condition exceeded the
separate 95% held-out retention milestone. Both had the lowest mean answer NLL,
but its accuracy advantage was **0 percentage points**, with **0/3 positive paired
accuracy wins**. The predefined feedback-benefit criterion therefore **failed**.
Lower NLL alone does not satisfy its three required conditions.

Every model has H=32, V=7, and **5,863 parameters**. The initial parameter bytes
match across modes within each seed; different gates can nevertheless produce
different initial predictions. Initial accuracy below the 50% binary control is
possible because the untrained argmax ranges over all seven vocabulary tokens.
All final accuracies in the following table are 100%; NLL is nats per supervised
answer, not per input token.

| Seed | Mode | Initial train accuracy % | Initial held-out accuracy % | Initial train NLL | Initial held-out NLL | Final train NLL | Final held-out NLL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | both | 12.5000 | 12.5000 | 1.920702 | 1.895839 | 0.000387353 | 0.000387408 |
| 42 | top-down | 25.0000 | 15.6250 | 1.900994 | 1.913971 | 0.000498676 | 0.000500679 |
| 42 | bottom-up | 12.5000 | 10.9375 | 1.879554 | 1.943289 | 0.000667006 | 0.000651360 |
| 42 | neither | 18.7500 | 14.0625 | 1.877968 | 1.938869 | 0.000425299 | 0.000459274 |
| 43 | both | 6.2500 | 7.8125 | 2.003636 | 1.928233 | 0.000449720 | 0.000451060 |
| 43 | top-down | 18.7500 | 29.6875 | 1.964563 | 1.920674 | 0.000487559 | 0.000488256 |
| 43 | bottom-up | 12.5000 | 18.7500 | 1.957643 | 1.940877 | 0.000506733 | 0.000534810 |
| 43 | neither | 18.7500 | 23.4375 | 1.951571 | 1.934990 | 0.000787688 | 0.000773575 |
| 44 | both | 31.2500 | 21.8750 | 1.935826 | 1.968273 | 0.000389426 | 0.000389178 |
| 44 | top-down | 18.7500 | 18.7500 | 1.889155 | 1.935442 | 0.000629069 | 0.000628301 |
| 44 | bottom-up | 12.5000 | 21.8750 | 1.987605 | 1.939367 | 0.000399300 | 0.000402256 |
| 44 | neither | 12.5000 | 28.1250 | 1.946092 | 1.897374 | 0.000708772 | 0.000718832 |

| Mode | Mean final held-out accuracy | Mean final held-out NLL |
| --- | ---: | ---: |
| both | 100% | 0.0004092155043634721 |
| top-down | 100% | 0.0005390787192246690 |
| bottom-up | 100% | 0.0005294752575198033 |
| neither | 100% | 0.0006505602428561139 |

All comparator accuracies tie, so the predefined accuracy-then-NLL ordering
selects **bottom-up** as the best other retrained mode. Both minus bottom-up:

| Seed | Held-out accuracy difference | Held-out NLL difference |
| ---: | ---: | ---: |
| 42 | 0 percentage points | -0.00026395216110464137 |
| 43 | 0 percentage points | -0.00008374995019949649 |
| 44 | 0 percentage points | -0.00001307714816485561 |
| Mean | 0 percentage points | -0.0001202597531563312 |

The NLL component passes descriptively in all three paired seeds, while the
required >=5-point mean accuracy advantage and >=2 positive accuracy wins do
not. The report records `uses_preregistered_steps_and_seeds=true`,
`meets_numerical_criterion=false`, and
`decision=not_supported_by_preregistered_test`. The accuracy ceiling is a limit
of this task/budget, not grounds to replace the acceptance rule after seeing it.

## Matched budgets and retained evidence

Each condition used **300 updates, 4,800 answer-target presentations, and 96,000
input tokens**. Across twelve conditions this is **3,600 updates, 57,600 answer
targets, and 1,152,000 input tokens**. No early stopping, curriculum weight
transfer, optimizer-state transfer, or held-out updates occurred. Each condition
repeated the same ordered full batch of 16 cases and reset all four recurrent
states independently for every prompt. Initial/final evaluations did not select
weights. There were no checkpoint writes.

Execution used f32 Burn/WGPU on the **AMD Radeon RX 9060 XT**, with recorded device
fingerprint `spirv_4098_30096`. Matching parameter bytes, update count, and
input/target budgets does not establish equal hardware work: zero gates may
change backend execution. This report does not measure training speed, runtime
equivalence, latency, energy, or an efficiency advantage.

The delay-16 data is unchanged from the
[memory curriculum](2026-10-18-memory-diagnostic.md): eight training histories
paired into 16 prompts, and 32 distinct held-out histories paired into 64 prompts.
For example, training prompt `xz0xyyxxzyxxzyzxxxx?` and its bit-1 counterpart
share the same noise history; held-out prompt `zx0xyzxyyxzzyyzxzzx?` uses a
different history. The sole payload determines the answer. Complete case lists,
predictions, training logs, and parameter digests remain in the report.

- Raw report: [memory-feedback-comparison.json](../../artifacts/memory-feedback-comparison.json),
  SHA256 `e93b3dcc593308f7f07211768d56e1389ecb8a8c83668260c409bc437de5548a`.
- Command log: [memory-feedback-comparison.log](../../artifacts/memory-feedback-comparison.log),
  SHA256 `694176c0f072eaca037196c158610ed1d50ca065b7a55176801988cb6e0c5b40`.
- Ordered delay-16 training cases SHA256:
  `18e5b434d5a6d205ce101d9840373cfa66a973efbd11f7662b8c5f3c74df6912`.
- Ordered delay-16 held-out cases SHA256:
  `d9821d3b4f00db412bfa9403ca381d3fac117c0dd168cf3c51e248d50a4890f2`.
- All curriculum-stage data manifest SHA256, used to verify the unchanged
  training-only vocabulary source:
  `a2a57ba4660a9f749bd37b6a266d8e46b71e7f32c88d807e0609992e848be571`.
- Repeated full-batch index schedule SHA256:
  `f23d672bb9b341f9afa8498423b75deb80e726145969391d4b9392464c2298ee`.

| Seed | Shared initial parameter SHA256, row-major f32 little-endian |
| ---: | --- |
| 42 | `e350c108b6bdc70dbb6c615dee407a64f8ba1586f6a1c4ba0faa70f73039bc20` |
| 43 | `1b589837f014248f8fd00a26a48893ecd495dae6723255241d41a5386b96bae1` |
| 44 | `50112de1d229967dfa23cfffbb85d1572be9a7c5bee86030073d4aa2b5ad8e95` |

Validation checks confirmed the 52-test CPU suite, `cargo fmt --check`, and
`cargo clippy --all-targets -- -D warnings`. The explicit discrete-GPU parity
test previously passed for Full, memory-only, and spikes-only paths: all
45 parameter gradients agree with CPU within 3e-5, binary spikes agree exactly,
and disconnected groups have exact zero gradients. These numerical checks do
not substitute for the task measurements above.

## Interpretation and decision

Simple delay-16 recall is learnable directly from these fresh initializations,
without the earlier curriculum, at this fixed 300-update budget. That extends
the curriculum's capability result while removing its unequal-update
comparison. In particular, the Neither condition also reaches 100%; the two
feedback interfaces are not necessary for this particular retention task and
budget. Neither still has both output projections and continuous memory; it is
not the spikes-only or memory-only component control.

Both has lower observed NLL than the alternatives, but this is a descriptive
confidence/loss difference at an accuracy ceiling. The predefined evidence for
an accuracy benefit is absent. Do not report the feedback hypothesis as
confirmed, and do not report feedback as universally useless. Three model seeds
on one 64-case held-out split provide narrow evidence, and the dataset is shared
across seeds rather than independently resampled. No significance test or broad
generalization claim is made. No shuffled-feedback or post-training direction
intervention was measured in this separate comparison.

The earlier correction-task failure remains unresolved: it cannot be attributed
simply to an inability to retain any single bit for 16 distractors, but this
experiment does not identify its cause or demonstrate entity binding, emotional
reasoning, grammar, or conversational ability. No neuron equations or learned
parameterization changed for these diagnostics. The independent
[FineWeb passage diagnostic](2026-10-18-fineweb-overfit.md) tests the remaining
basic text-fitting question with genuine training text and separate validation
documents. Its result must be judged separately; one-bit recall does not
establish useful natural-language generation.
