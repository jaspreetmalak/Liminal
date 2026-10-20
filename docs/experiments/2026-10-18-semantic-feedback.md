# Causal semantic-memory feedback pilot

## Question and decision rule, recorded before training

Does a learned, persistent continuous memory coupled in both directions to
Liminal's spiking neurons improve delayed contextual binding over the same model
with either connection or both connections disabled?

This is a focused synthetic engineering experiment, permitted by the project
rules. It tests contextual memory on generated symbolic assignments and later
corrections. It does not test natural-language fluency, emotional reasoning, or
general symbolic reasoning. FineWeb remains the primary language-training corpus.

Use one maintained Rust/Burn model for all conditions. Keep parameter count,
initial parameters, minibatch schedule, data, and update budget identical within
each model seed. The four modes are `both`, `top-down`, `bottom-up`, and `neither`.
All retain the same continuous memory and direct memory-to-output projection;
the two coupling terms are multiplied by zero or one. Consequently this compares
the usefulness of the connections, rather than comparing a larger model against
the previous smaller model.

Predeclared protocol:

- Data seed 42; 1,024 training and 256 held-out prompts, with disjoint complete
  prompt hashes and balanced queried entities and binary answers.
- Sequences assign values to two entities, introduce a later assignment, include
  distractors, and ask for the queried entity's latest value. The expected answer
  is computed from the assignment history. Prompt length is 32 characters.
- Loss is next-token cross-entropy over the full vocabulary at the single answer
  position. The answer is a target and is never fed before its prediction.
  Prompt positions have zero loss mask but remain attached in the recurrent
  graph. No validation examples enter updates.
- Model seeds 42, 43, 44; hidden/memory width 32, batch 32, 300 Adam updates,
  learning rate 0.003, global gradient clip 1, f32, WGPU on the RX 9060 XT.
- Membrane decay 0.9, threshold 1, triangular surrogate width 1; spike-trace decay
  0.8 and semantic-memory mixing decay 0.9. Attached subtractive reset is retained.
- Report initial/final held-out answer NLL and exact full-vocabulary argmax
  accuracy, training accuracy, and synchronized evaluation wall time. Balanced
  binary answer chance is 50%; the untrained full-vocabulary model can be lower.
- After each `both` run, separately disable either direction or both without
  retraining. These are interventions on a trained model, distinct from the
  independently trained four conditions.

The pilot supports the proposed benefit only if mean `both` held-out accuracy
exceeds the best other retrained mode's mean by at least **5 percentage points**,
the paired advantage over that same comparator is positive for at least **two
of three seeds**, and mean held-out answer NLL is lower. Otherwise report the
hypothesis as unsupported or inconclusive for this setting. Three seeds and one
task are not a general capability or statistical-significance claim.

## Implementation and validation

The single model now has filtered spike traces and a continuous tanh memory.
Previous memory modulates the current spike input; completed current spike traces
update memory; both spikes and memory contribute to next-character logits.
Memory width equals hidden width. The full equations and reverse graph are in
[architecture.md](../architecture.md). This is a minimal memory-core prototype,
not an independently capable LLM or a demonstrated semantic reasoning system.

All 11 parameter groups share one enumeration for clipping and persistence, and
the language trainer and probe share `training::optimize_batch`. The checkpoint
loader preserves old artifact predictions by importing zero memory weights into
the current model; fresh training uses nonzero seeded initialization.

Checks before the fixed-budget training run:

- `cargo fmt --check`: passed.
- `cargo test`: **32 library + 4 integration tests passed**. The explicit hardware
  test is ignored by this default command and was run separately.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test --test gpu -- --ignored --nocapture`: passed on the RX 9060 XT.
  CPU/GPU fixture loss 1.12197852 / 1.12197840; maximum gradient difference across
  all 45 parameters **1.4901161e-8**, below 3e-5. All six additional groups have
  nonzero gradients in that fixture. Binary spikes agree exactly; voltage, trace,
  and memory states agree within 2e-5.
- The independent frozen-tangent finite-difference oracle covers all 45 fixture
  parameters, with epsilon 1e-5 and absolute gradient tolerance 3e-6. It includes
  nonzero initial trace/memory and both attached feedback directions.
- Probe tests check globally disjoint prompts, exact joint balance, latest-value
  labels, prompt-only inputs, final-answer masking, and bounded CLI output paths.
- Mode tests check causal influence of previous memory and present zero gradients
  for disabled coupling terms. Checkpoint tests include new-mode round trips and
  old-artifact prediction preservation.

## First pilot results

The predeclared feedback-benefit criterion **failed**. Across the three seeds,
`both` averaged **66.406250%** held-out accuracy versus **76.692708%** for
`top-down`, an advantage of **-10.286458 percentage points**. Its mean answer NLL
was also worse: **0.601468 versus 0.451528 nats**. The paired accuracy advantage
was positive in **0/3** seeds. `Top-down` and `neither` tied on mean accuracy;
`top-down` was the comparator because its mean NLL was lower, as specified by
the report's accuracy-then-NLL selection rule.

All accuracies below use full-vocabulary argmax. NLL is in nats at the single
answer target. Every final metric uses the final 300-update model; no model was
selected by its test score.

| Seed | Mode | Initial test accuracy % | Initial test NLL | Final train accuracy % | Final train NLL | Final test accuracy % | Final test NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | both | 6.250000 | 2.441125 | 75.390625 | 0.488851 | 78.125000 | 0.475391 |
| 42 | top-down | 11.718750 | 2.415774 | 77.343750 | 0.399542 | 79.687500 | 0.388697 |
| 42 | bottom-up | 9.765625 | 2.381503 | 67.675781 | 0.646664 | 71.093750 | 0.642736 |
| 42 | neither | 8.984375 | 2.405650 | 77.343750 | 0.431023 | 78.515625 | 0.413682 |
| 43 | both | 2.734375 | 2.610643 | 50.000000 | 0.694892 | 50.000000 | 0.691523 |
| 43 | top-down | 1.171875 | 2.634975 | 75.781250 | 0.565492 | 75.390625 | 0.557984 |
| 43 | bottom-up | 2.734375 | 2.613382 | 50.000000 | 0.693264 | 50.000000 | 0.687807 |
| 43 | neither | 1.562500 | 2.609677 | 75.683594 | 0.586803 | 75.781250 | 0.586997 |
| 44 | both | 9.375000 | 2.420293 | 68.164063 | 0.640795 | 71.093750 | 0.637489 |
| 44 | top-down | 3.125000 | 2.393260 | 76.074219 | 0.420916 | 75.000000 | 0.407903 |
| 44 | bottom-up | 12.500000 | 2.364094 | 52.539063 | 0.670043 | 50.781250 | 0.665612 |
| 44 | neither | 7.812500 | 2.377951 | 75.585938 | 0.452882 | 75.781250 | 0.458929 |

| Mode | Mean final test accuracy % | Mean final test NLL |
| --- | ---: | ---: |
| both | 66.406250 | 0.601468 |
| top-down | 76.692708 | 0.451528 |
| bottom-up | 57.291667 | 0.665385 |
| neither | 76.692708 | 0.486536 |

### Interventions on the trained bidirectional model

These are the same `both` weights with gates changed **without retraining**;
they are separate from the independently trained modes above. Cells show held-out
accuracy percent / answer NLL nats.

| Seed | Both retained | Only top-down retained | Only bottom-up retained | Neither retained |
| --- | ---: | ---: | ---: | ---: |
| 42 | 78.125000 / 0.475391 | 50.000000 / 0.861891 | 75.390625 / 0.567006 | 50.000000 / 0.898113 |
| 43 | 50.000000 / 0.691523 | 67.968750 / 0.676718 | 53.906250 / 0.683326 | 60.546875 / 0.679308 |
| 44 | 71.093750 / 0.637489 | 67.578125 / 0.636786 | 69.140625 / 0.637861 | 64.843750 / 0.638718 |

Disabling bottom-up feedback reduced seed 42 from 78.125% to 50%, but improved
seed 43 from 50% to 67.968750%. The interfaces can influence behavior; their
influence is not a consistent benefit under this budget.

### Post hoc shortcut audit and interpretation

After inspecting the results, a query-independent rule was checked: predict the
last assigned binary value anywhere in the prompt, ignoring which entity is
queried. It scores **760/1,024 = 74.218750%** on training and
**199/256 = 77.734375%** on test. This is a **post hoc diagnostic**, not part of
the original numerical acceptance criterion. A separate
[Rust audit report](../../artifacts/feedback-shortcut-baselines.json) of the
saved cases confirms both counts. Its ordered train/test hashes match the first
run exactly, and the original full report hash is unchanged. These counts remain
distinguished from the predeclared comparison.

Every mode's mean test accuracy is below this simple test-set heuristic. The
first dataset therefore provides weak evidence for entity binding even for the
better individual runs. Its balance across query/answer/corrected-entity strata
does not remove the last-value shortcut. No semantic, emotional, or general
symbolic reasoning capability is established.

The measured conclusion is that this minimal feedback design did not show the
predeclared benefit on the first task and budget. This does not disprove feedback
architectures in general. The next decision is a separately predeclared
[paired counterfactual task](2026-10-18-counterfactual-feedback.md), which targets
the identified shortcut. Its generator replaces the first generator in place;
the original prompts, labels, predictions, and metrics remain in the historical
JSON report. No second model or obsolete generator code tree is retained.

### Budget, environment, and reproducibility evidence

All runs used **6,251 parameters**. Within each seed, all four modes started from
identical f32 parameter bytes. SHA256 over those little-endian bytes was:

- Seed 42: `88004682593f194590e5a49d7a505c61911649e3d013c5c39c5d50293d3bd267`.
- Seed 43: `655fc84026651495f5efb593c70c26cde265f101fbfdf269010ce7e555db9651`.
- Seed 44: `09b033ee82b39c688fc780dacc021cc84b2e8d9df138a6e826e5beef69630965`.

Each run used **9,600 answer targets** and **307,200 context positions** sampled
with replacement. The 12 runs total **3,600 updates, 115,200 answer targets, and
3,686,400 context positions**. This is a focused synthetic test; no FineWeb data
or language-checkpoint weights entered its updates.

Execution used Burn 0.22, f32, WGPU discrete GPU 0: **AMD Radeon RX 9060 XT**,
fingerprint `spirv_4098_30096`. The host was Windows `10.0.26300`, Ryzen 7 7800X3D,
driver `32.0.31041.1004`, Rust/Cargo 1.98.1. Recorded training wall times, including
the final parameter readback, ranged **13.8044934–15.9046963 seconds per run**;
their sum was **169.9657345 seconds**. Final 256-case held-out evaluation in
batches of 32 ranged **0.1203370–0.1516738 seconds**, including host input
transfers and synchronized final-logit readbacks. These are batched diagnostic
timings, not individual interaction latency or a demonstrated real-time system.
The release build in the log took 1m 53s and is separate from the training times.

The report records data seed 42 and the exact shared sampled-index schedule,
whose seed was `8314604121742442562`. Dataset hashes use ordered ASCII
`prompt + TAB + answer + LF`; the combined hash uses the train hash plus LF,
then the test hash plus LF. The schedule hash uses flattened little-endian u64
training indices. SHA256 values:

- Training cases: `257f5dfa54a2c7276b0e54fe57e3b11a8d42909f455be65ad3e7900bf3e6e5df`.
- Test cases: `be71a7c8f9f774a5905515dac35b4ed3f9515a1c1b73f9367817b255f579f50a`.
- Combined data: `e916ba7d0251b83e0bc497088164235624b632c1fe518123474ee9955e0aa865`.
- Shared minibatch schedule: `3433b7ea5b4c720f227ea7eb1843a003bfe068160cccdb57764e1d504eb4a0b9`.
- [Full report](../../artifacts/feedback-probe.json): `d7083dff57a7cd2c95e80b8df25cc7c3b71e538fa886d509d9251617d602881b`.
- [Build/training log](../../artifacts/feedback-probe.log): `a4d42427351b45d5ea06dcf0a030cc8b77c76dc38fbf5122fab227f2858ea1b3`.

The existing language checkpoint was not overwritten: its SHA256 remains
`4eb56bc9fff8709931313d12e49c1e59d784eccdc1487ff1a21e52a6023f6fca`.
The new debug executable's 120-character generation matched the saved prior
output exactly. That confirms preserved artifact behavior, not improved language
quality. Earlier language results remain historical evidence for their recorded
architecture.

## Reproduction

This records the command actually executed for the **first** pilot. The current
`feedback_probe` uses the separately documented paired counterfactual task; this
command now runs that replacement task rather than regenerating the first one.
Audit the saved first-run JSON for its exact cases and outputs.

```text
cargo test
cargo test --test gpu -- --ignored --nocapture
cargo run --release --bin feedback_probe -- --device gpu --steps 300 --seeds 42,43,44 --output artifacts/feedback-probe.json
```

The current tool protects that historical output against replacement. Reproduce
the post hoc audit of its saved cases with:

```powershell
cargo run --release --bin feedback_probe -- --baselines-only --dataset-report artifacts/feedback-probe.json --output artifacts/feedback-shortcut-baselines.json
```

The probe uses the production model and shared Burn optimization step. Its
synthetic cases and masks are task fixtures, not a second model or trainer.
