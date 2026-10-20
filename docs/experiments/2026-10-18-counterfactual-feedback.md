# Paired counterfactual feedback test

## Question and protocol, recorded before this run

Does bidirectional spike/memory feedback improve delayed query-conditioned correction when
ignoring the queried entity cannot outperform 50% accuracy?

The [first pilot](2026-10-18-semantic-feedback.md) failed its predefined feedback
criterion. Subsequent review found that predicting the latest assigned bit while
ignoring the query already scores 77.73% on that test set. This second experiment
addresses that task weakness. It does not erase or replace the first outcome.
The first JSON report preserves all original examples and predictions.

Keep the same single model, equations, initialization, optimizer, hyperparameters,
and four feedback modes. Replace the fixture generator in place with paired
counterfactual questions, without retaining a second model or generator tree.

### Data and leakage controls

- Data seed 42; 512 training histories and 128 held-out histories. Each history
  generates two 32-character prompts: identical assignments/correction/distractors
  followed by either `?a=` or `?b=`. This gives 1,024 training and 256 test cases.
- Require the final values for a and b to differ. Thus the paired prompts have
  opposite correct answers. A predictor ignoring the query must answer one of
  each pair incorrectly; constants and the last-assigned-bit heuristic score
  exactly 50%. The intended latest-value-per-entity oracle scores 100%.
  Binary-valued randomized query-blind predictors have expected accuracy 50%; their finite
  sampled accuracy can vary.
- Each history assigns both entities, includes a later opposite-value correction
  to one entity, and contains distractors before and after that correction.
  Balance corrected entity and final a value exactly; include both queries for
  each history. Randomize initial assignment order and distractors independently.
- Split on entire histories/prefixes before creating queries. No history or full
  prompt crosses train/test. Fit vocabulary on training prompts only, and hash
  all retained examples. Do not feed the answer into the prompt.
- Apply loss only to the final answer prediction, with all context computation
  attached. No test examples enter optimization.

Review before training identified a remaining scope limit: opposite final binary
values and one flipped assignment imply that the two initial values are equal.
A solver can use only the corrected entity, its new bit, and the query: return
the new bit for that entity and its complement for the other. This rule scores
100% without remembering two independent bindings. Record that diagnostic too.
The task therefore tests delayed query-conditioned correction; it cannot establish
integration of two independent entity histories.

### Budget and decision

- Model seeds 42, 43, 44; H=32 for both spiking and continuous states; batch 32;
  300 Adam updates per seed/mode; learning rate .003; global clip 1; f32 WGPU on
  the RX 9060 XT. β=.9, θ=1, triangular width 1, trace decay .8, memory decay .9.
- Four retrained conditions: both, top-down, bottom-up, neither. Match initial
  parameters, parameter count, nominal graph, minibatch schedule, and budget
  within each seed. Every mode retains direct memory input/readout and recurrence.
- Report full-vocabulary argmax accuracy, answer NLL, train performance, exact
  heuristic baselines, and posttraining direction interventions on the Both runs.
- Acceptance remains: Both mean test accuracy >= best other retrained mode +
  **5 percentage points**, positive paired advantage for **at least two of three
  seeds**, and lower mean answer NLL. Otherwise this setting does not support the
  proposed benefit. Report exact numbers regardless of outcome.

This is a focused synthetic mechanism test, not natural-language, emotional, or
general symbolic-reasoning validation. Three seeds and one generated grammar
provide limited evidence. No extra runs or hyperparameter tuning are part of
this comparison. FineWeb remains the primary language-training corpus.

## Commands

```powershell
cargo test --lib probe::tests -- --nocapture
cargo clippy --all-targets -- -D warnings
./target/release/feedback_probe.exe --device gpu --steps 300 --seeds 42,43,44 --output artifacts/feedback-counterfactual.json
```

The direct executable command above was used for the fixed run after the release
build. The equivalent Cargo command, which builds when needed, is:

```powershell
cargo run --release --bin feedback_probe -- --device gpu --steps 300 --seeds 42,43,44 --output artifacts/feedback-counterfactual.json
```

## Fixture checks before model training

Five focused Rust tests pass, including prefix-group isolation, opposite-answer
pairing, target-only masking, independently parsed oracle labels, and controls.
Clippy passes across all targets with warnings denied. The model/optimizer did
not change from the first pilot's gradient and GPU checks.

The Rust-only baseline command completed without initializing a model or GPU:

```powershell
./target/release/feedback_probe.exe --baselines-only --output artifacts/feedback-counterfactual-baselines.json
```

Its [report](../../artifacts/feedback-counterfactual-baselines.json) confirms:

| Deterministic predictor | Training accuracy | Held-out accuracy |
| --- | ---: | ---: |
| Constant 0 / constant 1 | 50% | 50% |
| Latest assigned bit, ignoring query | 50% | 50% |
| First assigned bit, ignoring query | 50% | 50% |
| Latest queried entity value oracle | 100% | 100% |
| Correction-only plus query rule | 100% | 100% |

Training hash: `1c0a5a4d8fe5265abc1f1ae441419a6c2b6d290ee9d62008cdd5454ce713b8ba`.
Held-out hash: `08c8e821cd68ffc3029c1eb5d1b3653a77fb9c6938f3b112c905b7315fee54fc`.
Combined: `8ff72bbb39c307ff9c78185bddea482de8553a9ecddc034832c0e896e5751039`.
Hash encoding is ordered ASCII prompt + TAB + answer + LF; combined hash is
train hash + LF + test hash + LF.

## Fixed-run results

The predeclared benefit criterion **failed**. Every independently trained
seed/mode finished at **50% training and 50% held-out accuracy**. The mean
bidirectional accuracy advantage was **0 percentage points**, positive in
**0/3** seeds. Among the tied accuracy controls, `bottom-up` had the lowest mean
answer NLL. `Both` was worse by **0.001583882 nats** on that comparison.

All 12 held-out prediction lists contain only `0`: a constant-answer collapse.
Each model also scores exactly 50% on training, so the measured failure is
underfitting, not a train/test generalization gap. All conditions failed to learn
the training task beyond the constant-answer baseline under this budget. This
common performance floor makes the primary
comparison non-discriminating about the value of feedback: there is no measured
benefit here, but the outcome does not disprove the broader feedback idea or
identify the cause of the learning failure. No further runs, tuning, or model
changes were selected after seeing these results.

The table gives full-vocabulary argmax accuracy and answer NLL in nats. Initial
accuracy can be below 50% because an untrained model may choose any of the 11
vocabulary symbols, including symbols other than the binary answers. Every
final metric uses the final 300-update model.

| Seed | Mode | Initial test accuracy % | Initial test NLL | Final train NLL | Final test NLL | Final train/test accuracy % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | both | 5.078125 | 2.482315 | 0.702094 | 0.702111 | 50 / 50 |
| 42 | top-down | 6.250000 | 2.460717 | 0.700995 | 0.701017 | 50 / 50 |
| 42 | bottom-up | 10.937500 | 2.378133 | 0.699719 | 0.699723 | 50 / 50 |
| 42 | neither | 10.156250 | 2.404726 | 0.700107 | 0.699507 | 50 / 50 |
| 43 | both | 3.515625 | 2.595592 | 0.702222 | 0.702208 | 50 / 50 |
| 43 | top-down | 0.781250 | 2.632378 | 0.701681 | 0.701679 | 50 / 50 |
| 43 | bottom-up | 2.343750 | 2.620031 | 0.702148 | 0.702364 | 50 / 50 |
| 43 | neither | 1.562500 | 2.616996 | 0.701150 | 0.701597 | 50 / 50 |
| 44 | both | 12.109375 | 2.425954 | 0.702265 | 0.702403 | 50 / 50 |
| 44 | top-down | 7.812500 | 2.380511 | 0.702861 | 0.702918 | 50 / 50 |
| 44 | bottom-up | 14.453125 | 2.371301 | 0.700407 | 0.699884 | 50 / 50 |
| 44 | neither | 8.203125 | 2.383528 | 0.701160 | 0.701184 | 50 / 50 |

| Mode | Mean final test accuracy % | Mean final test NLL |
| --- | ---: | ---: |
| both | 50 | 0.702241 |
| top-down | 50 | 0.701871 |
| bottom-up | 50 | 0.700657 |
| neither | 50 | 0.700763 |

The completed Rust diagnostics confirm the stated baseline scores, also embedded
in the training report: constants, first-bit, and latest-bit rules each get
**512/1,024 training and 128/256 held-out cases correct (50%)**. Both the
latest-queried-value oracle and the restricted correction-plus-query rule get
**1,024/1,024 and 256/256 (100%)**. The latter remains a known task limitation;
these results cannot establish independent binding of two entity histories.

### Posttraining interventions

These change gates on each trained `both` model without retraining. Cells show
held-out accuracy percent / answer NLL nats; these interventions are not the
independently retrained primary conditions.

| Seed | Both retained | Only top-down retained | Only bottom-up retained | Neither retained |
| --- | ---: | ---: | ---: | ---: |
| 42 | 50.000000 / 0.702111 | 51.953125 / 0.694361 | 50.390625 / 0.704468 | 55.859375 / 0.693163 |
| 43 | 50.000000 / 0.702208 | 50.000000 / 0.700240 | 49.609375 / 0.698297 | 49.218750 / 0.697674 |
| 44 | 50.000000 / 0.702403 | 50.000000 / 0.704666 | 50.390625 / 0.701951 | 50.390625 / 0.704569 |

Gate changes alter some predictions, but this is not a consistent advantage for
the bidirectional model. No natural-language improvement, semantic or emotional
reasoning, statistical significance, or real-time interaction is demonstrated.

## Budget, timing, and integrity evidence

All 12 runs used the same **6,251-parameter** model. Within each seed, all four
conditions cloned identical initial f32 parameter bytes; the SHA256 hashes over
little-endian bytes were:

- Seed 42: `88004682593f194590e5a49d7a505c61911649e3d013c5c39c5d50293d3bd267`.
- Seed 43: `655fc84026651495f5efb593c70c26cde265f101fbfdf269010ce7e555db9651`.
- Seed 44: `09b033ee82b39c688fc780dacc021cc84b2e8d9df138a6e826e5beef69630965`.

Each run used **300 updates, 9,600 answer targets, and 307,200 context positions**,
with training-case indices sampled with replacement. Across 12 runs this totals
**3,600 updates, 115,200 answer targets, and 3,686,400 context positions**. The
shared sampled-index schedule was identical for every mode and seed. No test
cases entered updates, and no FineWeb or existing language-checkpoint weights
were changed by this diagnostic.

Execution used Burn 0.22, f32, and WGPU discrete GPU 0: **AMD Radeon RX 9060 XT**,
fingerprint `spirv_4098_30096`. Host: Windows `10.0.26300`, Ryzen 7 7800X3D,
driver `32.0.31041.1004`, Rust/Cargo 1.98.1. Training wall time including the
final parameter readback was **13.8639487–14.1142317 seconds per run**, totaling
**167.7140582 seconds**. Final 256-case held-out evaluation in batches of 32 took
**0.1251383–0.1481047 seconds**, including input host transfers and synchronized
final-logit readbacks. These are batched evaluation timings, not measurements of
individual interaction latency. Build time is excluded from these training
measurements.

Data seed 42 produced the train/test hashes recorded above, matching the
standalone baseline report. The schedule seed was `8314604121742442562`; its
SHA256 over flattened little-endian u64 indices was
`3433b7ea5b4c720f227ea7eb1843a003bfe068160cccdb57764e1d504eb4a0b9`.
The complete cases, labels, model predictions, logged updates, and measurements
are retained in these artifacts, with SHA256:

- [Training report](../../artifacts/feedback-counterfactual.json): `9623b89adcf7fa5984a2317496f57c3cdf3acc42de8fdd4175d7e4d274935b00`.
- [Training log](../../artifacts/feedback-counterfactual.log): `c20877cf70a93bb2df55b4f4589a5c83ea312bdaff09f1d9b52e0a827f7ae78d`.
- [Standalone baseline report](../../artifacts/feedback-counterfactual-baselines.json): `368bf2a17775839ec20e810b53f4efce2575671893186eee9192502e100c8ad9`.

The existing language checkpoint remains unchanged, SHA256
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`.
This experiment adds mechanism evidence and identifies an unresolved training
failure; it does not improve that checkpoint's conversational ability.

Final full-tree checks after the paired-fixture changes passed:

- `cargo test`: **34 library + 4 integration = 38 CPU tests passed**. The GPU
  hardware test is ignored by this default command.
- `cargo fmt --check`: passed.
- `cargo clippy --all-targets -- -D warnings`: passed.

The separately completed GPU parity check covers the unchanged model/optimizer
core; its measured loss and gradient agreement are recorded in the
[first pilot](2026-10-18-semantic-feedback.md).
